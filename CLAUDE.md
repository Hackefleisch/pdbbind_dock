# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this project does

Re-docks protein–ligand complexes with **PyRosetta**, stores every pose +
its decomposed Rosetta energy terms, then evaluates RosettaLigand's sampling and
ranking (success-rate tables) and trains a **linear reweighting model** that maps
the energy terms to binding affinity (log Kd). Benchmarks used: PDBbind v2020,
the PoseBusters benchmark set, and the Astex Diverse set.

Two distinct phases:

1. **Docking** (compute-heavy, runs on an HPC/SLURM cluster) — `docking/`
   package; produces one HDF5 file per complex.
2. **Analysis** (runs locally) — `docking_analysis/` package: extracts the HDF5
   files into parquet tables, computes the benchmark success-rate tables, and (via
   the `extract_docking_stats.py` entry point) trains the affinity reweighting
   models and writes figures.

## Repository layout

```
docking/                 Phase-1 docking engine (Python package)
  main.py                CLI entry point — one complex per invocation (SLURM arrays)
  runner.py              docking engine: relax + dock, writes one <pdb>.h5
  load_ligand.py         RDKit <-> Rosetta ligand handling
  ligand_rmsd.py         symmetry-corrected in-place ligand RMSD (used by result.py)
  result.py              reader for the docking HDF5 files (Result / PDBRinterface)
  xml_protocols/         RosettaScripts protocols (docking_std.xml, docking_perturb.xml)
docking_analysis/        Phase-2 analysis package (see "Analysis" below)
extract_docking_stats.py Phase-2 entry point of the affinity grid (repo root — imports docking_analysis)
scripts/                 run_*.slurm, filter_pdbbind.py, h5_size.py
docs/                    methods-level documentation and the paper planning documents
data/                    ALL inputs, docking outputs, caches, figures — git-ignored
```

Everything large or generated lives under **`data/`** (a single git-ignored
parent — see "Data layout"). Code, scripts and docs are the only things at the repo
root; the small tracked reference data (`scfx_weights.json`, `failed_pdb_ids/`,
`splits.csv`) ships inside `docking_analysis/`.

**Run entry points as modules from the repo root** so the `docking` package
resolves, e.g. `uv run python -m docking.main --help`. The `docking_analysis` package
and `extract_docking_stats.py` rely on being run from the repo root too
(cwd = repo root puts both packages on `sys.path`). There are no notebooks in the
repository; the former exploratory notebooks were folded into `scripts/filter_pdbbind.py`,
`docking_analysis.extract` and `docking_analysis/failed_pdb_ids/`.

## Phase 1 — Docking

- [docking/main.py](docking/main.py) — picks one PDB id by line index from a
  list file and docks it. Designed for SLURM array jobs (one task per complex).
- [docking/runner.py](docking/runner.py) — the engine. Relaxes the holo and apo
  poses, runs N docking trajectories per protocol, writes results incrementally
  to `<pdb>.h5`. Re-running **resumes** an existing file.
- [docking/load_ligand.py](docking/load_ligand.py) — conformer generation,
  mutable-residue creation, pose assembly. The input (crystal) conformer is kept and
  loaded into the Rosetta rotamer library together with the generated ones.
- [docking/ligand_rmsd.py](docking/ligand_rmsd.py) — symmetry-corrected, **in-place**
  heavy-atom ligand RMSD (graph automorphisms; validated against Rosetta's
  `automorphic_rmsd`). Used by the auto-correction in `docking/result.py` (and thus by
  the extraction); the runner itself is untouched.
- [docking/xml_protocols/](docking/xml_protocols/) — each protocol XML must
  define a `hard_rep` score function and a `ParsedProtocol` mover.
- Launchers in [scripts/](scripts/): `run_all.slurm` (PDBbind, array 0–11459),
  `run_astex.slurm`, `run_posebuster.slurm`. They `cd` into the repo and run
  the cluster venv's Python directly as `$VENV_PYTHON -m docking.main` (no `uv`).
- The docking **inputs** under `data/pdbbind_cleaned/` are produced from the raw
  PDBbind download by [scripts/filter_pdbbind.py](scripts/filter_pdbbind.py);
  the filters are documented in [docs/dataset_preparation.md](docs/dataset_preparation.md).

## Phase 2 — Analysis

Two generations of code live here. **Current** (used for Paper 1): `extract.py` and
`benchmark.py`, which work on the full-set parquet tables. **Legacy, to be rewritten**:
the affinity reweighting grid (`data.py`, `experiment.py`, `reweighting.py`,
`figures.py`, `clustering.py`, driven by `extract_docking_stats.py`). It was written
against zip bundles (exact-Kd subset, uncorrected RMSDs, constraint-laden totals) that
were deleted on 2026-10-19, so it cannot run; it stays as the reference for its term
selections, aggregations and figures until it is rewritten on the full extraction.

- [extract_docking_stats.py](extract_docking_stats.py) — orchestrator of the affinity
  grid: loads the filtered subset + Kd index, runs the experiment grid (aggregation ×
  term selection), optional pose clustering, writes figures + result CSVs.
  Run: `uv run extract_docking_stats.py [--n_workers N] [--rmsd_cutoff 2.0]`.
- [docking_analysis/](docking_analysis/):
  - `constants.py` — **all paths, regexes, unit conversions live here. Start here.**
    Paths are computed as `REPO_ROOT / "data" / …` (via the `DATA_DIR` constant),
    so the package must stay at the repo root and the data lives under `data/`.
    `FAILED_PDB_IDS_DIR`, `SPLITS_PATH`, `SCFX_WEIGHTS_PATH` point at the package data below.
  - `data.py` — load/filter dataframes, per-PDB aggregation (legacy zips).
  - `clustering.py` — RMSD-based pose clustering (cached under `docking_stats/`);
    lazily imports `docking.result.Result` to read the H5 files; plain (non-symmetric)
    RMSD between poses.
  - `experiment.py` — experiment configs (`DEFAULT_EXPERIMENTS`), results matrix.
  - `reweighting.py` — PyTorch linear reweighting of `raw_delta_*` terms.
  - `figures.py` — matplotlib figure generation (output → `figures/`).
  - `scfx_weights.json` — default Rosetta score-function weights as dumped by the runner.
  - `failed_pdb_ids/<set>.txt` — ids of the complexes whose docking failed or was removed
    after the first run (pdbbind 74, posebuster 40, astex 1); derived from `data/failed_logs/`.
  - `splits.csv` — PDBbind split annotations for 19,443 ids: `plindersplit`, `hacsplit`
    (its `test` = the 290-id CASF-2016 core set, `val` = the 300-id CASF validation set),
    `cleansplit`, `lppdbbindsplit`. Input for the leakage-aware split decision.
  - `extract.py` — **full-set extraction**: every H5 file of a set → parquet tables
    (`docked`, `seeds`, `complexes`, `manifest.json`) under `data/docking_stats/<set>_full/`,
    read through `Result(autocorrect=True)`, so `total_score`, `rmsd_to_crystal` and
    `rmsd_to_input` are corrected and the stored values sit next to them as `*_legacy`.
    All complexes are included; filter on `complexes.complete`.
    Run `uv run python -m docking_analysis.extract --set pdbbind` (`--no-autocorrect` for raw).
  - `benchmark.py` — six success-rate tables, one per receptor state × protocol, with the
    benchmark sets as rows, produced once per ranking score (interface delta and corrected
    total score) and preceded by a column legend. Output: `data/figures/benchmark/`.
    Run `uv run python -m docking_analysis.benchmark`.
- [docking/result.py](docking/result.py) — `Result` / `PDBRinterface`: the H5
  reader used by `clustering.py` and by the extraction. **Auto-corrects
  `total_score`, `rmsd_to_crystal`, `rmsd_to_input` by default** (see "Legacy semantics").

## HDF5 schema (one file per complex, written by runner.py)

- `ligand_sdf`, `atmname_to_idx`, `scfx_weights` — scalar JSON/SDF metadata.
- `poses/` group: `pdb_strings` (zlib-compressed full PDB), `results`
  (float matrix; `column_names` stored as an attr), `row_names`
  (`crystal` / `relax` / `apo_relax`).
- Top level: `pdb_strings` (per-pose **diffs** vs. the relaxed reference, JSON+zlib),
  `results` (per-docking-pose float matrix), `protocol` (`<pose>_<protocol>`).

`results` columns: `total_score`, `idelta_score`, `rmsd_to_crystal`,
`rmsd_to_input`, `prepare_time`, then `raw_<term>` and `raw_delta_<term>` per
active Rosetta energy term. Inspect any file with `uv run scripts/h5_size.py <file.h5>`.

**Legacy semantics and auto-correction** (decided 2026-10-15: the stored data is final, no
re-docking, the docking code stays as it is):

- Stored `total_score` **includes** the coordinate-constraint energy
  (0.5 × `raw_coordinate_constraint`) for seed rows *and* docked poses: the runner scores the
  relaxed seeds before removing the FastRelax restraints, and the FinalMinimizer leaves its Cα
  restraints in the pose. `idelta_score` is unaffected.
- Stored `rmsd_to_crystal` / `rmsd_to_input` are Rosetta `RMSDMetric` values: in place, all
  heavy atoms, matched by atom name, **no symmetry correction**.
- `Result(..., autocorrect=True)` (the default) rewrites exactly these three columns on access
  (constraint-free total; symmetry-corrected in-place RMSDs from the stored structures) and
  keeps the stored values as `total_score_legacy`, `rmsd_to_crystal_legacy`,
  `rmsd_to_input_legacy`. Everything else stays as stored. `autocorrect=False` gives the raw
  tables. The docking reference of a receptor state is always the lowest *stored* total
  (`Result.reference_index`), as in the docking run.

## Data layout

Everything below lives under **`data/`** (git-ignored). On the HPC cluster the
SLURM scripts expect the same names under `$WORKSPACE/data/`.

| Path (under `data/`) | Size | Role |
|------|------|------|
| `pdbbind_2020plus/` | 16 G | Raw PDBbind v2020 download; source of `INDEX_general_PL.2020R1.lst` (Kd values). |
| `pdbbind_cleaned/` | 2.5 G | Cleaned docking **inputs** (`<pdb>_protein.pdb` + `<pdb>_ligand.sdf`) + `index.txt`. What docking reads. |
| `posebusters_paper_data/` | 211 M | Astex + PoseBusters benchmark inputs (with id-list files). |
| `pdbbind_h5/` | **374 G** | Docking output, 11 460 `.h5` files. |
| `posebuster_h5/` `astex_h5/` | 14 G / 3.2 G | Docking output for the other two sets. |
| `docking_stats/<set>_full/` | 3.2 G | **Full-set parquet tables** from `docking_analysis.extract` — the data the analyses must use. |
| `failed_logs/` | 472 K | SLURM `.out`/`.err` of the 57 docking tasks that failed on the cluster; source of `docking_analysis/failed_pdb_ids/`. |
| `figures/` | — | Generated figures and benchmark tables (regenerated; safe to clear). |
| `h5_test/` | — | Output of the local smoke test (`python -m docking.runner`); absent unless it was run. |

The legacy zip bundles, the cluster cache and the Nov-2025 summary CSV were deleted on
2026-10-19 (the CSV's split columns survive as `docking_analysis/splits.csv`). The small
tracked reference data lives inside `docking_analysis/` (see "Phase 2").

## Environment

- Python ≥ 3.11, managed with **uv** ([pyproject.toml](pyproject.toml), `uv.lock`).
  PyRosetta 2025.47 (Rosetta commit `8bb54e8a6d`, GitHub tag `v2025.47-dev62202`) is
  installed from a **vendored wheel** in `wheels/` (git-ignored): the JHU wheel archive keeps
  only the latest ~8 releases and its TLS chain is incomplete, so the wheel (sha256 in
  `pyproject.toml`; a copy is in `~/Downloads`) must be placed there once before `uv sync`.
  The same build is served permanently as a conda package
  (`conda install -c https://conda.graylab.jhu.edu pyrosetta=2025.47`), and the Rosetta
  source at that tag can be built. Obtaining PyRosetta implies accepting its license.
- Analysis extras: scikit-learn, umap-learn, seaborn, statsmodels, pyarrow.
- Run anything with `uv run <script>` (or `.venv/bin/python`).
- PyRosetta init flags used throughout:
  `-in:auto_setup_metals -ex1 -ex2 -restore_pre_talaris_2013_behavior true`
- Local docking smoke test: `uv run python -m docking.runner` (its `__main__` docks
  `5k00` once into `data/h5_test/`).

## Conventions & gotchas

- **`docking_analysis/constants.py` is the single source of truth for paths.**
  When inputs move, edit there, not in scattered scripts.
- **The docking code and the stored H5 files keep their legacy semantics; corrections happen
  on access.** `Result(autocorrect=True)` (default) and therefore the extraction tables report
  constraint-free totals and symmetry-corrected in-place RMSDs, with the stored values kept as
  `*_legacy`. Do not "fix" `runner.py` or the XMLs for this: the data will not be re-docked.
- **Analyses run on the full extraction** (`data/docking_stats/<set>_full/`). The older
  partial zips and the cluster cache no longer exist; the affinity grid that read them
  (`extract_docking_stats.py`, `docking_analysis/data.py`, `clustering.py`, …) is to be
  rewritten on the parquet tables, not patched.
- Large data dirs and outputs are git-ignored (see [.gitignore](.gitignore)).
  **Do not commit `.h5`, parquet, large `.csv`, or figures.** Only code, docs and the
  small package data in `docking_analysis/` are tracked.
