#!/usr/bin/env python
"""
filter_pdbbind.py
=================
Build the ``pdbbind_cleaned`` docking-input set from a raw PDBbind v2020
download (``pdbbind_2020plus``).

This is the extracted, documented version of the ``notebooks/pdbbind_filter``
notebook that originally produced ``data/pdbbind_cleaned``. The filters and
thresholds are reproduced exactly; the only changes are structural (functions,
CLI, summary table) plus a control-flow fix so each removed complex is counted
under a single reason. See ``docs/dataset_preparation.md`` for the rationale
behind every filter.

Pipeline
--------
1. Index filtering   — annotation flags + resolution (no file I/O).
2. Ligand filtering  — RDKit parse / QED / heavy-atom count / element whitelist.
3. Structure cleaning— 7 A contact analysis: chain selection, co-ligand and
   modified-residue rejection, size cap, metal/cofactor retention.
4. Write             — stripped protein PDB + copied ligand SDF/MOL2 + index.txt.

Usage
-----
    uv run scripts/filter_pdbbind.py --dry-run          # report counts only
    uv run scripts/filter_pdbbind.py --out data/pdbbind_cleaned_rebuilt
    uv run scripts/filter_pdbbind.py --force            # overwrite existing --out

By default the script refuses to write into an existing, non-empty output
directory (the canonical ``data/pdbbind_cleaned`` was already used for docking).
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from scipy.spatial import cKDTree
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# Filter parameters (see docs/dataset_preparation.md)
# --------------------------------------------------------------------------- #
# Stage 1 — PDBbind index annotations that disqualify a complex.
REMOVE_FLAGS = ["peptide (xmer)", "covalent", "incomplete ligand", "redundant"]
RESOLUTION_MAX = 2.5  # Angstrom; complexes worse than this (or NMR) are dropped.

# Stage 2 — ligand filters.
QED_MIN = 0.1
HEAVY_ATOM_MIN = 8
HEAVY_ATOM_MAX = 500
# Elements kept (Leak-Proof PDBBind keeps elements present in >0.1 % of entries;
# B and Fe are additionally dropped because RosettaLigand handles metals poorly).
NON_RARE_ELEMENTS = {"C", "S", "O", "N", "Cl", "P", "F", "Br", "I", "H"}

# Stage 3 — structure filters.
CONTACT_RADIUS = 7.0   # Angstrom ligand contact shell.
MAX_RESIDUES = 1000    # protein size cap (residues across contacting chains).
# Metal ions / cofactors present at >0.1 % of PDBbind entries that are retained
# when in contact with the ligand. (HOH is listed but always removed below.)
KEEP_LIGANDS = [" ZN", " MG", " MN", " CA", " NA", "  K",
                " FE", " NI", " CO", "FE2", "HOH"]
REMOVE_FROM_FILE = ["HOH"]  # waters are stripped from the written PDB.


# --------------------------------------------------------------------------- #
# Stage 1: index-annotation filter
# --------------------------------------------------------------------------- #
def filter_index(index_path: Path) -> tuple[list[str], dict[str, int]]:
    """Return (kept pdb ids, removal counts) from the PDBbind index file.

    The index line layout is::

        <pdb>  <resolution>  <year>  <binding>   // <ref>.pdf (<ligand>) <flags>

    A complex is dropped if any annotation flag matches ``REMOVE_FLAGS``, if its
    ligand name ends in ``mer`` (a peptide), if its resolution is worse than
    ``RESOLUTION_MAX``, or if it has no numeric resolution (NMR).
    """
    kept: list[str] = []
    counts = {f: 0 for f in REMOVE_FLAGS}
    counts["resolution"] = 0
    counts["NMR"] = 0

    with open(index_path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            head, _, tail = line.partition("//")
            pdb_id = head.split()[0]
            resolution = head.split()[1]
            ligand_name = tail.split("(")[1].split(")")[0]
            flags = "".join(tail.split(")")[1:]).strip().split(";")
            if flags == [""]:
                flags = []
            if ligand_name.endswith("mer"):
                flags.append("peptide (xmer)")

            reason = None
            for flag in flags:
                flag = flag.strip().lower()
                for rf in REMOVE_FLAGS:
                    if rf in flag:
                        reason = rf
                        break
                if reason:
                    break

            if reason is None:
                try:
                    if float(resolution) > RESOLUTION_MAX:
                        reason = "resolution"
                except ValueError:
                    reason = "NMR"

            if reason is None:
                kept.append(pdb_id)
            else:
                counts[reason] += 1

    return kept, counts


# --------------------------------------------------------------------------- #
# Stage 2: ligand filter
# --------------------------------------------------------------------------- #
def ligand_reject_reason(sdf_path: Path):
    """Return a removal reason for the ligand, or ``None`` if it passes.

    Uses the SDF (the MOL2 files are unreliable for these structures). The mol
    is read without sanitisation; QED is the first call that forces perception,
    so kekulisation/valence problems surface there.
    """
    from rdkit import Chem
    from rdkit.Chem import QED, Lipinski
    from rdkit.Chem import (AtomKekulizeException, KekulizeException,
                            AtomValenceException)

    mol = Chem.MolFromMolFile(str(sdf_path), sanitize=False, removeHs=False)
    if mol is None:
        return "unparsable"

    try:
        if QED.qed(mol) <= QED_MIN:
            return "qed"
    except (AtomKekulizeException, KekulizeException):
        return "kekule"
    except AtomValenceException:
        return "valence"

    n_heavy = Lipinski.HeavyAtomCount(mol)
    if n_heavy < HEAVY_ATOM_MIN or n_heavy > HEAVY_ATOM_MAX:
        return "heavy_atom_count"

    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in NON_RARE_ELEMENTS:
            return "rare_element"

    return None


# --------------------------------------------------------------------------- #
# Stage 3: structure analysis + filter
# --------------------------------------------------------------------------- #
def analyze_structure(pdb_path: Path, sdf_path: Path, radius: float = CONTACT_RADIUS):
    """Analyse one complex's contact shell.

    Returns a dict with ``chains`` (protein chains contacting the ligand),
    ``cofactors`` (kept metal/ion atoms, as ``"<atom name>:<atom num>"``),
    ``size`` (residue count over contacting chains), ``has_co_ligand`` and
    ``modified_residues`` (non-canonical residues in contact), or ``None`` if
    the PDB has no parsable atoms.
    """
    pos, chain_of, name_of, res_of, num_of, resnum_of, ter_of = (
        [], [], [], [], [], [], [])
    ter = 0
    with open(pdb_path) as fh:
        for line in fh:
            if line[0:3] == "TER":
                ter += 1
            is_atom = line[0:4] == "ATOM"
            is_het = line[0:6] == "HETATM"
            if not (is_atom or is_het):
                continue
            pos.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            chain_of.append(line[21])
            num_of.append(int(line[6:11]))
            resnum_of.append(int(line[22:26]))
            ter_of.append(ter)
            if is_het:
                name_of.append(line[12:16])
                res_of.append(line[17:20])
            else:
                name_of.append("Protein")
                res_of.append("Protein")

    if not pos:
        return None

    tree = cKDTree(pos)

    # Collect atom indices within `radius` of any ligand atom.
    contacts: set[int] = set()
    with open(sdf_path) as fh:
        for line in fh:
            tok = line.strip().split()
            # SDF V2000 atom-block lines split into 10 or 16 whitespace tokens.
            if len(tok) == 10 or len(tok) == 16:
                xyz = [float(tok[0]), float(tok[1]), float(tok[2])]
                contacts.update(tree.query_ball_point(xyz, radius))

    # The ligand/cofactor block follows the last TER; HETATMs inside chain
    # blocks (lower TER count) are modified residues embedded in the protein.
    last_block = max(ter_of)
    chains: set[str] = set()
    modified_residues: set[str] = set()
    contacting_ligands: set[str] = set()
    cofactors: list[int] = []
    for idx in contacts:
        if name_of[idx] == "Protein":
            chains.add(chain_of[idx])
        elif ter_of[idx] == last_block:
            contacting_ligands.add(res_of[idx])
            if res_of[idx] in KEEP_LIGANDS and res_of[idx] not in REMOVE_FROM_FILE:
                cofactors.append(idx)
        else:
            modified_residues.add(res_of[idx])

    size = 0
    for chain in chains:
        size += len({resnum_of[i] for i in range(len(chain_of)) if chain_of[i] == chain})

    has_co_ligand = any(res not in KEEP_LIGANDS for res in contacting_ligands)

    return {
        "chains": sorted(chains),
        "cofactors": [f"{name_of[i]}:{num_of[i]}" for i in cofactors],
        "size": size,
        "has_co_ligand": has_co_ligand,
        "modified_residues": modified_residues,
    }


# --------------------------------------------------------------------------- #
# Stage 4: write the cleaned complex
# --------------------------------------------------------------------------- #
def write_cleaned(pdb_id: str, src_dir: Path, out_dir: Path,
                  chains: list[str], cofactors: list[str], copy_mol2: bool) -> None:
    """Write the cleaned protein PDB and copy the ligand file(s)."""
    src_pdb = src_dir / f"{pdb_id}_protein.pdb"
    src_sdf = src_dir / f"{pdb_id}_ligand.sdf"
    dst = out_dir / pdb_id
    dst.mkdir(parents=True, exist_ok=True)

    cofactor_keys = {tuple(cf.split(":")) for cf in cofactors}
    kept_lines: list[str] = []
    with open(src_pdb) as fh:
        for line in fh:
            if line[0:3] == "TER":
                kept_lines.append(line)
            elif line[0:4] == "ATOM" and line[21] in chains and line[76:78] != " H":
                kept_lines.append(line)  # protein heavy atoms of contacting chains
            elif line[0:6] == "HETATM":
                key = (line[12:16], str(int(line[6:11])))
                if key in cofactor_keys:
                    kept_lines.append(line)  # retained metal / cofactor atom

    (dst / f"{pdb_id}_protein.pdb").write_text("".join(kept_lines))
    shutil.copyfile(src_sdf, dst / f"{pdb_id}_ligand.sdf")
    if copy_mol2:
        src_mol2 = src_dir / f"{pdb_id}_ligand.mol2"
        if src_mol2.exists():
            shutil.copyfile(src_mol2, dst / f"{pdb_id}_ligand.mol2")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=Path("data/pdbbind_2020plus"),
                    help="Raw PDBbind v2020 directory (default: data/pdbbind_2020plus).")
    ap.add_argument("--index", type=Path, default=None,
                    help="Index file (default: <source>/INDEX_general_PL.2020R1.lst).")
    ap.add_argument("--out", type=Path, default=Path("data/pdbbind_cleaned"),
                    help="Output directory (default: data/pdbbind_cleaned).")
    ap.add_argument("--contact-radius", type=float, default=CONTACT_RADIUS,
                    help=f"Ligand contact radius in A (default: {CONTACT_RADIUS}).")
    ap.add_argument("--copy-mol2", action="store_true",
                    help="Also copy the ligand MOL2 alongside the SDF.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report removal counts without writing any files.")
    ap.add_argument("--force", action="store_true",
                    help="Allow writing into an existing, non-empty --out.")
    args = ap.parse_args()

    index_path = args.index or (args.source / "INDEX_general_PL.2020R1.lst")
    if not args.dry_run and args.out.exists() and any(args.out.iterdir()) and not args.force:
        ap.error(f"{args.out} exists and is not empty; pass --force or a different --out "
                 f"(or use --dry-run).")

    removed: dict[str, int] = {}

    def drop(reason: str) -> None:
        removed[reason] = removed.get(reason, 0) + 1

    # --- Stage 1 -----------------------------------------------------------
    print(f"[1/4] Index filtering: {index_path}")
    kept, index_counts = filter_index(index_path)
    removed.update(index_counts)
    total = len(kept) + sum(index_counts.values())
    print(f"      {total} complexes in index -> {len(kept)} pass annotation/resolution")

    # --- Stage 2 -----------------------------------------------------------
    print("[2/4] Ligand filtering (RDKit QED / heavy atoms / elements)")
    ligand_ok: list[str] = []
    for pdb_id in tqdm(kept):
        sdf = args.source / pdb_id / f"{pdb_id}_ligand.sdf"
        reason = ligand_reject_reason(sdf)
        if reason is None:
            ligand_ok.append(pdb_id)
        else:
            drop(reason)
    print(f"      -> {len(ligand_ok)} pass ligand filters")

    # --- Stage 3 -----------------------------------------------------------
    print(f"[3/4] Structure cleaning ({args.contact_radius} A contact shell)")
    final: list[tuple[str, list[str], list[str]]] = []
    for pdb_id in tqdm(ligand_ok):
        src_dir = args.source / pdb_id
        info = analyze_structure(src_dir / f"{pdb_id}_protein.pdb",
                                 src_dir / f"{pdb_id}_ligand.sdf",
                                 radius=args.contact_radius)
        if info is None:
            drop("no_atoms")
        elif info["size"] > MAX_RESIDUES:
            drop("size")
        elif info["has_co_ligand"]:
            drop("co_ligand")
        elif info["modified_residues"]:
            drop("modified_residue")
        else:
            final.append((pdb_id, info["chains"], info["cofactors"]))
    print(f"      -> {len(final)} pass structure filters")

    # --- Stage 4 -----------------------------------------------------------
    if args.dry_run:
        print("[4/4] (dry run: skipping writes)")
    else:
        print(f"[4/4] Writing cleaned complexes to {args.out}")
        args.out.mkdir(parents=True, exist_ok=True)
        for pdb_id, chains, cofactors in tqdm(final):
            write_cleaned(pdb_id, args.source / pdb_id, args.out,
                          chains, cofactors, copy_mol2=args.copy_mol2)
        with open(args.out / "index.txt", "w") as fh:
            for pdb_id, chains, cofactors in final:
                fh.write(f"{pdb_id};{','.join(chains)};{','.join(cofactors)}\n")

    # --- Summary -----------------------------------------------------------
    print("\nRemoval summary")
    print("-" * 36)
    for reason in sorted(removed):
        print(f"  {reason:<20} {removed[reason]:>6}")
    print("-" * 36)
    print(f"  {'removed (total)':<20} {sum(removed.values()):>6}")
    print(f"  {'KEPT':<20} {len(final):>6}")


if __name__ == "__main__":
    main()
