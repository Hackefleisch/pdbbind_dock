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
scripts/                 run_*.slurm, run_local.sh, filter_pdbbind.py, h5_size.py
docs/                    methods-level documentation (feeds the paper)
reference/               tracked reference data (2016_core_set/, failed_pdb_ids/)
data/                    ALL inputs, docking outputs, caches, figures — git-ignored
```

Everything large or generated lives under **`data/`** (a single git-ignored
parent — see "Data layout"). Code, scripts, and small tracked reference lists
are the only things at the repo root.

**Run entry points as modules from the repo root** so the `docking` package
resolves, e.g. `uv run python -m docking.main --help`. The `docking_analysis` package
and `extract_docking_stats.py` rely on being run from the repo root too
(cwd = repo root puts both packages on `sys.path`).

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
  `run_astex.slurm`, `run_posebuster.slurm`, and `run_local.sh`
  (GNU parallel, local smoke test). The SLURM scripts `cd` into the repo and run
  the cluster venv's Python directly as `$VENV_PYTHON -m docking.main` (no `uv`).
- The docking **inputs** under `data/pdbbind_cleaned/` are produced from the raw
  PDBbind download by [scripts/filter_pdbbind.py](scripts/filter_pdbbind.py);
  the filters are documented in [docs/dataset_preparation.md](docs/dataset_preparation.md).

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
  reader used by the notebooks, by `clustering.py` and by the extraction. **Auto-corrects
  `total_score`, `rmsd_to_crystal`, `rmsd_to_input` by default** (see "Legacy semantics").
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
| `docking_stats/` | 3.3 G | Legacy extracted dataframe zips + `cluster_cache_filtered_v2/` (partial: complete complexes / exact-Kd subset, uncorrected RMSD, totals with constraint energy). |
| `docking_stats/<set>_full/` | ~6 G | **Full-set parquet tables** from `docking_analysis.extract` — the data the analyses must use. |
| `h5_test_v2/` | — | Scratch output of a 2026-10-15 smoke test with a since-reverted runner (`184l`); safe to delete. |
| `figures/` | — | Generated figures (regenerated; safe to clear). |
| `h5_test/` | — | Scratch output of the local smoke test. |
| `pdbbind_rosetta.csv` | 24 M | Legacy summary table; kept on disk but untracked. |

Tracked reference lists (CASF-2016 splits, failed-id lists) live outside `data/`
in **`reference/`**.

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
- Local docking smoke test: `bash scripts/run_local.sh`, or
  `uv run python -m docking.runner` (its `__main__` docks `5k00` into `data/h5_test/`).

## Conventions & gotchas

- **`docking_analysis/constants.py` is the single source of truth for paths.**
  When inputs move, edit there, not in scattered scripts.
- **The docking code and the stored H5 files keep their legacy semantics; corrections happen
  on access.** `Result(autocorrect=True)` (default) and therefore the extraction tables report
  constraint-free totals and symmetry-corrected in-place RMSDs, with the stored values kept as
  `*_legacy`. Do not "fix" `runner.py` or the XMLs for this: the data will not be re-docked.
- **Analyses run on the full extraction** (`data/docking_stats/<set>_full/`), never on the
  older partial zips, which cover only complete complexes / the exact-Kd subset and carry
  uncorrected RMSDs and constraint-laden totals. `extract_docking_stats.py` and
  `docking_analysis/data.py` still read the legacy zips and are due for porting.
- Data/cache artifacts are versioned by filename suffix (`_v2`, `rmsd_2.0`). The
  current analysis targets `pdbbind_relax_perturb_filtered_kd_v2.zip` and
  `cluster_cache_filtered_v2/`.
- Large data dirs and outputs are git-ignored (see [.gitignore](.gitignore)).
  **Do not commit `.h5`, `.zarr`, large `.csv`, or rendered notebooks.**
- A retired zarr-based pipeline (`prepare_dataset.py`, `full_test.zarr/`, and two
  legacy notebooks) was removed during the 2026-05-31 cleanup. The HDF5 pipeline
  documented above is the only live workflow.
