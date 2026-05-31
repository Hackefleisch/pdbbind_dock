# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this project does

Re-docks protein–ligand complexes with **PyRosetta**, stores every pose +
its decomposed Rosetta energy terms, then trains a **linear reweighting model**
that maps those energy terms to binding affinity (log Kd). Benchmarks used:
PDBbind v2020, the PoseBusters benchmark set, and the Astex Diverse set.

Two distinct phases:

1. **Docking** (compute-heavy, runs on an HPC/SLURM cluster) — `docking/`
   package; produces one HDF5 file per complex.
2. **Analysis** (runs locally) — `docking_analysis/` package + the
   `extract_docking_stats.py` entry point; reads the HDF5 files, clusters
   poses, trains reweighting models, and writes figures.

## Repository layout

```
docking/                 Phase-1 docking engine (Python package)
  main.py                CLI entry point — one complex per invocation (SLURM arrays)
  runner.py              docking engine: relax + dock, writes one <pdb>.h5
  load_ligand.py         RDKit <-> Rosetta ligand handling
  result.py              reader for the docking HDF5 files (Result / PDBRinterface)
  xml_protocols/         RosettaScripts protocols (docking_std.xml, docking_perturb.xml)
docking_analysis/        Phase-2 analysis package (see "Analysis" below)
extract_docking_stats.py Phase-2 entry point (stays at repo root — imports docking_analysis)
notebooks/               exploratory / extraction notebooks
scripts/                 run_*.slurm, run_local.sh, h5_size.py
reference/               tracked reference data (2016_core_set/, failed_pdb_ids/)
data/                    ALL inputs, docking outputs, caches, figures — git-ignored
```

Everything large or generated lives under **`data/`** (a single git-ignored
parent — see "Data layout"). Code, scripts, and small tracked reference lists
are the only things at the repo root.

**Run entry points as modules from the repo root** so the `docking` package
resolves, e.g. `python -m docking.main --help`. The `docking_analysis` package
and `extract_docking_stats.py` rely on being run from the repo root too
(cwd = repo root puts both packages on `sys.path`).

## Phase 1 — Docking

- [docking/main.py](docking/main.py) — picks one PDB id by line index from a
  list file and docks it. Designed for SLURM array jobs (one task per complex).
- [docking/runner.py](docking/runner.py) — the engine. Relaxes the holo and apo
  poses, runs N docking trajectories per protocol, writes results incrementally
  to `<pdb>.h5`. Re-running **resumes** an existing file.
- [docking/load_ligand.py](docking/load_ligand.py) — conformer generation,
  mutable-residue creation, pose assembly.
- [docking/xml_protocols/](docking/xml_protocols/) — each protocol XML must
  define a `hard_rep` score function and a `ParsedProtocol` mover.
- Launchers in [scripts/](scripts/): `run_all.slurm` (PDBbind, array 0–11459),
  `run_astex.slurm`, `run_posebuster.slurm`, and `run_local.sh`
  (GNU parallel, local smoke test). The SLURM scripts `cd` into the repo and
  invoke `python -m docking.main`.

## Phase 2 — Analysis

- [extract_docking_stats.py](extract_docking_stats.py) — orchestrator: loads the
  filtered subset + Kd index, runs the experiment grid (aggregation × term
  selection), optional pose clustering, writes figures + result CSVs.
  Run: `uv run extract_docking_stats.py [--n_workers N] [--rmsd_cutoff 2.0]`.
- [docking_analysis/](docking_analysis/):
  - `constants.py` — **all paths, regexes, unit conversions live here. Start here.**
    Paths are computed as `REPO_ROOT / "data" / …` (via the `DATA_DIR` constant),
    so the package must stay at the repo root and the data lives under `data/`.
  - `data.py` — load/filter dataframes, per-PDB aggregation.
  - `clustering.py` — RMSD-based pose clustering (cached under `docking_stats/`);
    lazily imports `docking.result.Result` to read the H5 files.
  - `experiment.py` — experiment configs (`DEFAULT_EXPERIMENTS`), results matrix.
  - `reweighting.py` — PyTorch linear reweighting of `raw_delta_*` terms.
  - `figures.py` — matplotlib figure generation (output → `figures/`).
  - `scfx_weights.json` — default Rosetta score-function weights.
- [docking/result.py](docking/result.py) — `Result` / `PDBRinterface`: the H5
  reader used by the notebooks and by `clustering.py`.
- [notebooks/](notebooks/) — `extract_docking_dataframe.ipynb` (current; builds
  the `docking_stats/*.zip` bundles from the H5 files), plus older exploratory
  notebooks (`h5_analysis`, `failed_log_analysis`, `pdbbind_filter`). They import
  `from docking.result import …`; launch Jupyter/VS Code with the workspace root
  as cwd so the `docking` package resolves.

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
| `docking_stats/` | 3.3 G | Extracted dataframe zips + `cluster_cache_filtered_v2/` consumed by analysis. |
| `figures/` | — | Generated figures (regenerated; safe to clear). |
| `h5_test/` | — | Scratch output of the local smoke test. |
| `pdbbind_rosetta.csv` | 24 M | Legacy summary table; kept on disk but untracked. |

Tracked reference lists (CASF-2016 splits, failed-id lists) live outside `data/`
in **`reference/`**.

## Environment

- Python ≥ 3.11, managed with **uv** ([pyproject.toml](pyproject.toml), `uv.lock`).
  PyRosetta is pinned to a JHU wheel URL; install requires accepting its license.
- Run anything with `uv run <script>` (or `.venv/bin/python`).
- PyRosetta init flags used throughout:
  `-in:auto_setup_metals -ex1 -ex2 -restore_pre_talaris_2013_behavior true`
- Local docking smoke test: `bash scripts/run_local.sh`, or
  `python -m docking.runner` (its `__main__` docks `5k00` into `data/h5_test/`).

## Conventions & gotchas

- **`docking_analysis/constants.py` is the single source of truth for paths.**
  When inputs move, edit there, not in scattered scripts.
- Data/cache artifacts are versioned by filename suffix (`_v2`, `rmsd_2.0`). The
  current analysis targets `pdbbind_relax_perturb_filtered_kd_v2.zip` and
  `cluster_cache_filtered_v2/`.
- Large data dirs and outputs are git-ignored (see [.gitignore](.gitignore)).
  **Do not commit `.h5`, `.zarr`, large `.csv`, or rendered notebooks.**
- A retired zarr-based pipeline (`prepare_dataset.py`, `full_test.zarr/`, and two
  legacy notebooks) was removed during the 2026-05-31 cleanup. The HDF5 pipeline
  documented above is the only live workflow.
