"""
docking.ligand_rmsd
-------------------
Symmetry-corrected, **in-place** heavy-atom ligand RMSD.

* In place: no superposition. The ligand is compared in the receptor frame, so
  translation and rotation errors count as docking errors.
* Symmetry-corrected: the RMSD is minimised over the automorphisms of the
  heavy-atom molecular graph (element labels only; bond orders, formal charges
  and hydrogens are ignored, as in spyrmsd / DockRMSD). This makes a flipped
  phenyl ring, a rotated carboxylate or a swapped tert-butyl methyl group cost
  nothing, which plain atom-name matching cannot do.

The implementation was validated against Rosetta's ``core::scoring::automorphic_rmsd``
(superimpose=False) on docked poses: identical to 1e-3 Å, and ~1000x faster because
the automorphisms are enumerated once per ligand instead of once per call.

Used by ``docking.result.Result`` to recompute the stored RMSDs from the PDB
strings in the HDF5 files (and therefore by ``docking_analysis.extract``). The
docking runner itself is unchanged and still stores Rosetta's uncorrected values;
``heavy_xyz_from_residue`` is provided for live Rosetta poses.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem

DEFAULT_MAX_AUTOMORPHISMS = 10_000


class LigandSymmetry:
    """Heavy-atom automorphisms of a ligand plus RMSD helpers.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        The ligand, typically ``Chem.MolFromMolBlock(ligand_sdf, sanitize=False,
        removeHs=False)``. Atom indices of this molecule are the reference frame
        for ``atmname_to_idx`` (Rosetta atom name -> RDKit index).
    max_automorphisms : int
        Cap on the number of graph automorphisms enumerated. If the cap is hit
        (``capped`` is True), the RMSD is an upper bound (still <= plain RMSD).

    Attributes
    ----------
    heavy_idx : np.ndarray
        RDKit atom indices of the heavy atoms, in the order used for coordinates.
    perms : np.ndarray, shape (M, n_heavy)
        ``perms[m, k]`` is the heavy-atom position (0..n_heavy-1) mapped onto
        reference position ``k`` by automorphism ``m``. Row 0 is the identity.
    """

    def __init__(self, mol: Chem.Mol, max_automorphisms: int = DEFAULT_MAX_AUTOMORPHISMS):
        self.n_atoms = mol.GetNumAtoms()
        self.heavy_idx = np.array(
            [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1], dtype=np.int64
        )
        pos = {int(h): i for i, h in enumerate(self.heavy_idx)}

        graph = Chem.RWMol()
        for h in self.heavy_idx:
            atom = Chem.Atom(mol.GetAtomWithIdx(int(h)).GetAtomicNum())
            atom.SetNoImplicit(True)
            graph.AddAtom(atom)
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if i in pos and j in pos:
                graph.AddBond(pos[i], pos[j], Chem.BondType.SINGLE)
        graph = graph.GetMol()
        graph.UpdatePropertyCache(strict=False)

        matches = graph.GetSubstructMatches(
            graph, uniquify=False, useChirality=False, maxMatches=max_automorphisms
        )
        perms = np.array(matches, dtype=np.int64)
        if perms.ndim != 2 or perms.shape[1] != len(self.heavy_idx):
            # Should not happen (a graph always matches itself); fall back to identity.
            perms = np.arange(len(self.heavy_idx), dtype=np.int64)[None, :]
        # identity first (stable: makes 'capped' upper bounds still >= true value trivially)
        ident = np.arange(len(self.heavy_idx))
        is_ident = (perms == ident).all(axis=1)
        if is_ident.any():
            perms = np.concatenate([perms[is_ident][:1], perms[~is_ident]], axis=0)
        else:
            perms = np.concatenate([ident[None, :], perms], axis=0)
        self.perms = perms
        self.capped = len(matches) >= max_automorphisms

    # ------------------------------------------------------------------ #
    @property
    def n_heavy(self) -> int:
        return len(self.heavy_idx)

    @property
    def n_automorphisms(self) -> int:
        return int(self.perms.shape[0])

    # ------------------------------------------------------------------ #
    def rmsd(self, xyz: np.ndarray, ref: np.ndarray) -> float:
        """Symmetry-corrected in-place RMSD of one pose.

        ``xyz`` and ``ref`` have shape (n_heavy, 3), heavy atoms in
        ``heavy_idx`` order.
        """
        d = xyz[self.perms] - ref[None, :, :]            # (M, n, 3)
        return float(np.sqrt((d * d).sum(axis=2).mean(axis=1)).min())

    def rmsd_many(self, xyz: np.ndarray, ref: np.ndarray, max_elements: int = 5_000_000) -> np.ndarray:
        """Symmetry-corrected in-place RMSD for a batch of poses, shape (N, n_heavy, 3)."""
        n = xyz.shape[0]
        if n == 0:
            return np.zeros(0)
        chunk = max(1, max_elements // max(1, self.perms.shape[0] * self.n_heavy * 3))
        out = np.empty(n)
        for s in range(0, n, chunk):
            block = xyz[s:s + chunk]                      # (b, n, 3)
            d = block[:, self.perms] - ref[None, None, :, :]   # (b, M, n, 3)
            out[s:s + chunk] = np.sqrt((d * d).sum(axis=3).mean(axis=2)).min(axis=1)
        return out

    @staticmethod
    def plain_rmsd(xyz: np.ndarray, ref: np.ndarray) -> float:
        """In-place RMSD with atom-name matching only (what Rosetta's RMSDMetric reports)."""
        d = xyz - ref
        return float(np.sqrt((d * d).sum(axis=1).mean()))


# ---------------------------------------------------------------------- #
# Coordinate helpers
# ---------------------------------------------------------------------- #

def heavy_xyz_from_pdb_lines(lines, atmname_to_idx: dict, sym: LigandSymmetry) -> np.ndarray:
    """Heavy-atom coordinates (n_heavy, 3) from the ligand's HETATM lines of a PDB string.

    ``atmname_to_idx`` maps the 4-character PDB atom name (``line[12:16]``) to
    the RDKit atom index, as stored in the HDF5 ``atmname_to_idx`` dataset.
    """
    xyz = np.full((sym.n_atoms, 3), np.nan)
    for line in lines:
        idx = atmname_to_idx.get(line[12:16])
        if idx is not None:
            xyz[idx] = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
    out = xyz[sym.heavy_idx]
    if np.isnan(out).any():
        raise ValueError("ligand heavy atom missing from PDB lines")
    return out


def heavy_xyz_from_residue(residue, atmname_to_idx: dict, sym: LigandSymmetry) -> np.ndarray:
    """Heavy-atom coordinates (n_heavy, 3) from a Rosetta ``Residue`` (the ligand)."""
    xyz = np.full((sym.n_atoms, 3), np.nan)
    for i in range(1, residue.natoms() + 1):
        idx = atmname_to_idx.get(residue.atom_name(i))
        if idx is not None:
            v = residue.xyz(i)
            xyz[idx] = (v.x, v.y, v.z)
    out = xyz[sym.heavy_idx]
    if np.isnan(out).any():
        raise ValueError("ligand heavy atom missing from Rosetta residue")
    return out


def ligand_lines(pdb_lines) -> list[int]:
    """Indices of the ligand HETATM lines (residue name UNK) in a list of PDB lines."""
    return [i for i, l in enumerate(pdb_lines) if l.startswith("HETATM") and l[17:20] == "UNK"]


def docked_ligand_lines(ref_lines, ref_lig_idx, diff: dict) -> list[str]:
    """Ligand HETATM lines of a docked pose from its line diff against the reference PDB.

    Docking can change the line order of the written PDB (residue variants may add or drop
    atoms), so the ligand is collected **by atom name**: reference ligand lines that the diff
    left untouched, overridden by every ligand line found in the diff.
    """
    by_name: dict[str, str] = {}
    for i in ref_lig_idx:
        if str(i) not in diff:
            line = ref_lines[i]
            by_name[line[12:16]] = line
    for line in diff.values():
        if line.startswith("HETATM") and line[17:20] == "UNK":
            by_name[line[12:16]] = line
    return list(by_name.values())
