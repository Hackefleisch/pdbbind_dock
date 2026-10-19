# Dataset preparation: PDBbind → `pdbbind_cleaned`

How the docking input set (`data/pdbbind_cleaned`) is derived from a raw
PDBbind v2020 download (`data/pdbbind_2020plus`).

- **Reference implementation:** [`scripts/filter_pdbbind.py`](../scripts/filter_pdbbind.py)
  (extracted from the original exploratory notebook, which has been removed from
  the repository; the script is the authoritative version).
- **Result:** **19,037** complexes in the PDBbind general set → **11,460** cleaned
  docking inputs.

The same scoring/docking machinery is also applied to the PoseBusters and Astex
Diverse sets; for those the complexes are *flagged* rather than removed, so this
document covers only the PDBbind filtering that actually subsets the data.

## Source data

PDBbind v2020 general set (`INDEX_general_PL.2020R1.lst`, 19,037 protein–ligand
complexes). Each complex directory provides `<pdb>_protein.pdb`,
`<pdb>_ligand.sdf` and `<pdb>_ligand.mol2`. Index lines look like:

```
<pdb>  <resolution>  <year>  <binding>   // <ref>.pdf (<ligand name>) <annotations>
```

## Filters

The pipeline runs four stages. Stages 1–3 reject complexes; stage 4 writes the
survivors. Every threshold below is defined once at the top of
`scripts/filter_pdbbind.py`.

### Stage 1 — index annotations & resolution

Parsed straight from the index file (no structure I/O).

| Filter | Rule | Rationale |
|--------|------|-----------|
| Peptide ligand | ligand name ends in `mer` (e.g. `2-mer`) **or** flagged `peptide (xmer)` | peptides are out of scope for small-molecule docking |
| Covalent | annotation contains `covalent` | covalent binders are not modelled |
| Incomplete ligand | annotation contains `incomplete ligand` | partial coordinates are unusable |
| Redundant | annotation contains `redundant` | de-duplicate PDBbind's flagged redundancies |
| Resolution | crystal resolution > **2.5 Å** | low-resolution coordinates are unreliable |
| NMR / no resolution | resolution field is non-numeric | NMR ensembles lack a single defined geometry |

> The original notebook additionally surveyed *unrecognised* annotation strings
> to make sure nothing filter-worthy was missed; that survey was diagnostic only
> and is not part of `filter_pdbbind.py`.

### Stage 2 — ligand filters (RDKit, from the SDF)

The ligand is read from the **SDF** without sanitisation (`sanitize=False,
removeHs=False`). The MOL2 files are not used here — they raise frequent
kekulisation errors. `QED.qed()` is the first call that forces RDKit to perceive
the molecule, so kekulisation/valence failures surface at that point.

| Filter | Rule | Rationale |
|--------|------|-----------|
| Parse / kekulé / valence | molecule fails to load, or QED raises `(Atom)KekulizeException` / `AtomValenceException` | structure cannot be processed reliably |
| Drug-likeness | `QED ≤ 0.1` | remove non-drug-like / pathological ligands |
| Heavy-atom count | `< 8` or `> 500` heavy atoms | exclude tiny fragments and oversized ligands |
| Element whitelist | any atom outside `{C, S, O, N, Cl, P, F, Br, I, H}` | see below |

**Element whitelist.** Following the Leak-Proof PDBBind protocol, elements that
appear in fewer than 0.1 % of entries (≈19 complexes) are treated as rare and
removed. Boron and iron are dropped in addition, because RosettaLigand is not
well set up to handle those metals in the ligand.

### Stage 3 — structure cleaning & filters

For each remaining complex the protein PDB atoms are loaded into a `cKDTree` and
every atom within **7.0 Å** of any ligand atom defines the "contact shell".
`TER` records are counted while parsing so that the ligand/cofactor block (after
the **last** `TER`) can be distinguished from HETATM groups embedded inside
protein chains (lower `TER` count → modified residues).

| Filter | Rule | Rationale |
|--------|------|-----------|
| Protein size | > **1000** residues across the contacting chains | cap docking cost / memory |
| Co-ligand | a non-water, non-retained ligand contacts the binding ligand | avoid ambiguous multi-ligand pockets |
| Modified residue | a non-canonical residue (e.g. `MSE`, `LLP`, `KCX`, `OCS`, `CAS`) contacts the ligand | conservatively reject rather than repair |

**Chain selection.** Only protein chains with an atom in the contact shell are
retained; all other chains are dropped from the written structure.

**Retained cofactors.** Metal ions / small cofactors that occur in > 0.1 % of
PDBbind entries are kept *when in the contact shell*:
`ZN, MG, MN, CA, NA, K, FE, NI, CO, FE2`. Waters (`HOH`) are always removed.

### Stage 4 — write cleaned inputs

For each surviving complex:

- **Protein PDB** — keep `TER` records, `ATOM` records of the selected chains
  **excluding hydrogens** (`element != " H"`), and the retained-cofactor
  `HETATM` records (matched by atom name + atom number). Written to
  `data/pdbbind_cleaned/<pdb>/<pdb>_protein.pdb`.
- **Ligand** — `<pdb>_ligand.sdf` (and optionally `.mol2`) copied verbatim.

## Output layout

```
data/pdbbind_cleaned/
  index.txt
  <pdb>/
    <pdb>_protein.pdb     # chain-trimmed, heavy-atoms only, + kept cofactors
    <pdb>_ligand.sdf      # copied unchanged
    <pdb>_ligand.mol2     # copied unchanged
```

**`index.txt`** has one line per complex:

```
<pdb_id>;<chain1,chain2,...>;<cofactor1,cofactor2,...>
```

where each cofactor is `"<4-char atom name>:<atom number>"`, e.g.
`4tln;A;ZN  :4706` or `1rnt;A;` (no cofactor). The chain/cofactor fields are
metadata; the docking [`Runner`](../docking/runner.py) consumes only the written
`_protein.pdb` and `_ligand.sdf` (the chain/cofactor selection is already baked
into the PDB). [`docking/main.py`](../docking/main.py) reads each line and uses
the leading 4-character PDB id.

## Why heavy-atoms only (key design decision)

Hydrogen positions are not resolved in these experimental structures, and the
available bond-order/protonation sources (PDBbind, RCSB, SDF, MOL2, the CCD)
disagree with one another. Rather than trust any one of them, the pipeline:

- uses **only heavy-atom positions** and lets the docking pipeline (re)place
  hydrogens (hence hydrogens are stripped from the protein PDB on write);
- works with **kekulé structures only** (SDFs carry no aromaticity flags, and
  kekulé forms are the easiest to obtain consistently);

and the method is therefore assessed end-to-end, purely on its ability to
reproduce heavy-atom positions.

## Results

Authoritative counts from the current data:

| | Complexes |
|---|---|
| PDBbind v2020 general set (index) | 19,037 |
| **Cleaned docking inputs** (`pdbbind_cleaned`) | **11,460** |

Indicative per-stage removals (Stage 1 from a fresh run of
`filter_pdbbind.py`; Stages 2–3 as recorded during the original notebook run):

| Stage | Reason | Removed |
|-------|--------|---------|
| 1 | peptide (xmer) | 1,995 |
| 1 | resolution > 2.5 Å | 2,881 |
| 1 | covalent | 633 |
| 1 | incomplete ligand | 515 |
| 1 | redundant | 179 |
| 1 | NMR / no resolution | 98 |
| 2 | kekulé / valence / QED / heavy-atoms / rare element | ≈600 |
| 3 | size / co-ligand / modified residue | ≈620 |

> The PDBbind index file has been updated since the original run (19,037 vs
> 19,535 entries then), so per-stage numbers shift slightly on re-run; the
> development run recorded a final 11,518 vs the 11,460 in the directory today.
> Run `uv run scripts/filter_pdbbind.py --dry-run` for an exact current
> breakdown.

## Caveats / implementation notes

- **SDF atom-line detection** uses a token-count heuristic (a V2000 atom line
  splits into 10 or 16 whitespace tokens); robust for this dataset but not a
  general SDF parser.
- **Modified residues are rejected, not repaired.** Rosetta's `clean_pdb` would
  convert e.g. `MSE→MET`, but such complexes are dropped here for simplicity.
- **Single removal reason.** A complex can match several filters; the script
  attributes it to the first one hit (the notebook occasionally double-counted),
  so totals here may differ slightly from the notebook's printouts. Membership
  of the kept set is unaffected.

## Reproduce

```bash
uv run scripts/filter_pdbbind.py --dry-run                  # counts only
uv run scripts/filter_pdbbind.py --out data/pdbbind_cleaned_rebuilt --copy-mol2
```

By default the script refuses to overwrite a non-empty `--out`; pass `--force`
to rebuild in place.

---

> **Citation needed:** add the Leak-Proof PDBBind reference for the rare-element
> rule (and the PDBbind v2020 citation) when writing this up.
