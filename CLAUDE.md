# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this project does

Re-docks protein–ligand complexes with **PyRosetta**, stores every pose +
its decomposed Rosetta energy terms, then evaluates RosettaLigand's sampling and
ranking (success-rate tables) and trains a **linear reweighting model** that maps
the energy terms to binding affinity (log Kd). Benchmarks used: PDBbind v2020,
the PoseBusters benchmark set, and the Astex Diverse set.

Two distinct phases, and the analysis phase follows the three tracks of the paper
(see [docs/PAPER1_STORY_INVENTORY.md](docs/PAPER1_STORY_INVENTORY.md)):

1. **Docking** (compute-heavy, runs on an HPC/SLURM cluster) — `docking/`
   package; produces one HDF5 file per complex.
2. **Analysis** (runs locally), one package per track:
   - `docking_analysis/` — **Track 1, benchmark**: extraction of the HDF5 files into
     parquet tables and the success-rate tables (sampling vs. ranking).
   - `ranking/` — **Track 2, fix the ranking**: the pose-ranking models (linear reweighting
     and an MLP ranker, ported from Luca Friebe's repository) with the Success@k evaluation;
     the energy-degeneracy analysis and the per-complex oracle are still planned.
   - `affinity/` — **Track 3, affinity**: the legacy reweighting grid, kept as
     documentation only (not runnable, to be rewritten).

## Repository layout

```
docking/                 Phase-1 docking engine (Python package)
  main.py                CLI entry point — one complex per invocation (SLURM arrays)
  runner.py              docking engine: relax + dock, writes one <pdb>.h5
  load_ligand.py         RDKit <-> Rosetta ligand handling
  ligand_rmsd.py         symmetry-corrected in-place ligand RMSD (used by result.py)
  result.py              reader for the docking HDF5 files (Result / PDBRinterface)
  xml_protocols/         RosettaScripts protocols (docking_std.xml, docking_perturb.xml)
docking_analysis/        Track 1 — extraction + benchmark tables (see "Analysis" below)
  constants.py           ALL paths of the project; package data paths
  extract.py             H5 -> parquet (data/docking_stats/<set>_full/)
  benchmark.py           success-rate tables per receptor state x protocol
  scfx_weights.json, failed_pdb_ids/, splits.csv   tracked package data
ranking/                 Track 2 — pose-ranking models (port of Luca's reweighting / ranker)
  data.py                pose tables from docked.parquet: features, hit labels, runs, PDBbind split
  models.py              Reweight (linear) and Ranker (MLP), BCE + InfoNCE loss, training loop
  evaluate.py            Success@k (as benchmark.py), bootstrap CIs, paired deltas, AUCs
  train.py, report.py    CLIs: one run -> data/ranking/<run>/; comparison tables -> data/figures/ranking/
affinity/                Track 3 — legacy reweighting grid, documentation only
  constants.py           paths of its (deleted) inputs, Kd regex, zero-variance terms
  data.py, experiment.py, reweighting.py, figures.py, clustering.py, extract_docking_stats.py
scripts/                 run_*.slurm, filter_pdbbind.py, h5_size.py
docs/                    methods-level documentation and the paper planning documents
data/                    ALL inputs, docking outputs, caches, figures — git-ignored
```

Everything large or generated lives under **`data/`** (a single git-ignored
parent — see "Data layout"). Code, scripts and docs are the only things at the repo
root; the small tracked reference data (`scfx_weights.json`, `failed_pdb_ids/`,
`splits.csv`) ships inside `docking_analysis/`.

**Run entry points as modules**, e.g. `uv run python -m docking.main --help`,
`uv run python -m docking_analysis.extract --set pdbbind`. `uv sync` installs the four
packages in editable mode (hatchling build backend, see `pyproject.toml`), so they resolve
from any working directory; the SLURM scripts still `cd` into the repo. There are no
notebooks in the repository; the former exploratory notebooks were folded into
`scripts/filter_pdbbind.py`, `docking_analysis.extract` and `docking_analysis/failed_pdb_ids/`.

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

### Track 1 — `docking_analysis/` (current)

- `constants.py` — **all paths, package-data paths live here. Start here.** Paths are
  computed as `REPO_ROOT / "data" / …` (via the `DATA_DIR` constant), so the package must
  stay at the repo root and the data lives under `data/`. `FAILED_PDB_IDS_DIR`,
  `SPLITS_PATH`, `SCFX_WEIGHTS_PATH` point at the package data; the other tracks import
  their shared paths from here.
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
- `scfx_weights.json` — default Rosetta score-function weights as dumped by the runner.
- `failed_pdb_ids/<set>.txt` — ids of the complexes whose docking failed or was removed
  after the first run (pdbbind 74, posebuster 40, astex 1); derived from `data/failed_logs/`.
- `splits.csv` — PDBbind split annotations for 19,443 ids: `plindersplit`, `hacsplit`
  (its `test` = the 290-id CASF-2016 core set, `val` = the 300-id CASF validation set),
  `cleansplit`, `lppdbbindsplit`. Input for the leakage-aware split decision.

### Track 2 — `ranking/` (pose-ranking models: S5 implemented, S6 planned)

Port of Luca Friebe's `rosettaligand-vs-posebuster` (branch `test_new_architecture`) onto the
full extraction, with the fixes listed in the inventory §6.1 (labels, features, split, Astex
overlap). Credited in the package docstring; his repository stays the reference for the
thesis-era numbers. Methods and results: [docs/ranking.md](docs/ranking.md).

- `data.py` — `load_poses(set, protocols, feature_cols)` reads one or more protocols of
  `docked.parquet` (complete complexes only, sorted so that every docking run = 300 poses is a
  contiguous block). Features: `delta` (default) = the seven non-zero interface terms
  `raw_delta_{fa_atr, fa_rep, fa_sol, fa_elec, fa_pair, hbond_bb_sc, hbond_sc}` — Luca's 25
  requested terms reduce to exactly these once the all-zero ones are dropped; `total` = the 20
  constraint-free `raw_*` terms; `both`. Luca's preprocessing is kept (delta `fa_rep` capped at
  50, robust scaling, scaled values clipped to ±10) but the scaler is fitted on the training poses
  only (`RobustScale`, stored as JSON). Hit = `rmsd_to_crystal` ≤ 2 Å (symmetry-corrected).
  `assign_split` maps the PDBbind complexes to train/val/test: `lppdbbind` (default, LP-PDBBind
  from `splits.csv`), `hac` (CASF-2016 core as test), `cleansplit`, `random` (Luca's
  StratifiedGroupKFold, fold 0 = val = "test"). `train.py` removes PDBbind complexes whose PDB
  entry is an Astex complex from train/val (43 of the 84 Astex entries are in the PDBbind set, no
  PoseBusters entry is).
- `models.py` — `Reweight` (bias-free linear; `--init rosetta` starts it at the Rosetta weights,
  i.e. at the interface-delta ranking) and `Ranker` (64-32-16-1 MLP); `BCEPlusInfoNCE` (Luca's
  loss, τ = 0.5, β = 1); `fit`: Adam, one docking run per batch, best epoch by validation
  Success@1 then Success@5 (the untrained model counts as epoch 0 and can be selected).
- `evaluate.py` — `per_group_success` uses the benchmark's definition (stable sort by score,
  Success@k ⇔ rank of the first hit < k); percentile-bootstrap CIs over runs, paired differences
  to `idelta_score` on the same runs, pooled AUC and mean per-run AUC (the pooled AUC of
  `total_score` is meaningless, totals are not comparable across complexes).
- `train.py` — one run: `uv run python -m ranking.train --model {reweight,ranker}
  [--protocol relax_docking_perturb] [--features delta] [--split lppdbbind] [--init rosetta]
  [--eval-protocols ...]`; outputs under `data/ranking/<run>/` (`results.md/json`,
  `per_run.parquet`, `model.pt`, `scaler.json`, `history.csv`, `weights.csv`). It prints a
  sanity line with the full-set baseline Success@k, which must equal the benchmark tables.
  Luca's original setup: `--split random --rmsd-col rmsd_to_input_legacy --no-exclude-overlap`.
- `report.py` — `uv run python -m ranking.report` collects all runs into
  `data/figures/ranking/ranking_tables.md` + `ranking_results.csv`.

### Track 3 — `affinity/` (legacy, documentation only)

The first implementation of the affinity track: `extract_docking_stats.py` drove a grid of
per-complex aggregations × energy-term selections (`experiment.py`), trained a non-negative
linear reweighting against log Kd (`reweighting.py`), clustered poses by RMSD
(`clustering.py`) and drew the figures (`figures.py`); `data.py` read the zip bundles and
parsed the Kd index; `constants.py` holds its paths, the Kd regex and the zero-variance
terms. **It is not runnable**: its inputs (exact-Kd zip, cluster cache) were deleted on
2026-10-19. The files stay unchanged as the record of what was done and of the limitations
the rewrite must lift (inventory S7); the rewrite will work on the parquet tables.

- [docking/result.py](docking/result.py) — `Result` / `PDBRinterface`: the H5
  reader used by the extraction (and by the legacy `affinity.clustering`). **Auto-corrects
  `total_score`, `rmsd_to_crystal`, `rmsd_to_input` by default** (see "Legacy semantics").
  It does **not** import PyRosetta (it carries its own copy of `mol_result`), so the whole
  analysis runs without PyRosetta.

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
| `ranking/<run>/` | small | Track 2 runs written by `ranking.train`: model, scaler, per-run tables, `results.json/.md`, `train.log`. |
| `failed_logs/` | 472 K | SLURM `.out`/`.err` of the 57 docking tasks that failed on the cluster; source of `docking_analysis/failed_pdb_ids/`. |
| `figures/` | — | Generated figures and tables: `benchmark/` (Track 1), `ranking/` (Track 2 report); regenerated, safe to clear. |
| `h5_test/` | — | Output of the local smoke test (`python -m docking.runner`); absent unless it was run. |

The legacy zip bundles, the cluster cache and the Nov-2025 summary CSV were deleted on
2026-10-19 (the CSV's split columns survive as `docking_analysis/splits.csv`). The small
tracked reference data lives inside `docking_analysis/` (see "Phase 2").

## Environment

- Python ≥ 3.11, managed with **uv** ([pyproject.toml](pyproject.toml), `uv.lock`;
  hatchling build backend, the four packages are installed editable by `uv sync`).
- **PyRosetta is needed only by the docking engine** (Phase 1). It is the `docking`
  dependency group: `uv sync --group docking`. Analysis-only environments without the
  wheel file must use `uv sync --frozen` (a plain `uv sync` re-validates the lock and
  fails when the path source is missing).
  The build used for all docking runs is PyRosetta 2025.47 (Rosetta commit `8bb54e8a6d`,
  GitHub tag `v2025.47-dev62202`), installed from a **vendored wheel** in `wheels/`
  (git-ignored; sha256 in `pyproject.toml`; a copy is in `~/Downloads`). The JHU wheel
  archive keeps only the latest ~8 weekly releases (2025.47 is gone) and sends an
  incomplete TLS chain; the same build is served permanently as a conda package
  (`conda install -c https://conda.graylab.jhu.edu pyrosetta=2025.47`). How third parties
  obtain it is an open decision (see README). Obtaining PyRosetta implies accepting its
  license.
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
  partial zips and the cluster cache no longer exist; the `affinity/` package that read
  them is documentation only and is to be rewritten on the parquet tables, not patched.
- **One package per track.** New Track 2 code goes to `ranking/`, the rewritten Track 3
  to `affinity/` (replace, do not extend, the legacy modules); shared paths stay in
  `docking_analysis/constants.py`.
- Large data dirs and outputs are git-ignored (see [.gitignore](.gitignore)).
  **Do not commit `.h5`, parquet, large `.csv`, or figures.** Only code, docs and the
  small package data in `docking_analysis/` are tracked.
