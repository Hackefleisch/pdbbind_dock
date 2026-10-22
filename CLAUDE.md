# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this project does

Re-docks protein–ligand complexes with **PyRosetta**, stores every pose +
its decomposed Rosetta energy terms, then evaluates RosettaLigand's sampling and
ranking (success-rate tables) and trains a **linear reweighting model** that maps
the energy terms to binding affinity (pK from the exact Kd / Ki values). Benchmarks used: PDBbind v2020,
the PoseBusters benchmark set, and the Astex Diverse set.

Two distinct phases, and the analysis phase follows the three tracks of the paper
(see [docs/PAPER1_STORY_INVENTORY.md](docs/PAPER1_STORY_INVENTORY.md)):

1. **Docking** (compute-heavy, runs on an HPC/SLURM cluster) — `docking/`
   package; produces one HDF5 file per complex.
2. **Analysis** (runs locally), one package per track:
   - `docking_analysis/` — **Track 1, benchmark**: extraction of the HDF5 files into
     parquet tables and the success-rate tables (sampling vs. ranking).
   - `ranking/` — **Track 2, fix the ranking**: four approaches (non-negative reweighting
     from a random or the Rosetta start, LightGBM LambdaRank tree, Luca Friebe's MLP ranker)
     against the unmodified interface delta, with the Success@k evaluation (S5); the
     energy-degeneracy analysis and the per-complex oracle with its figures, example
     complexes and UMAP/t-SNE illustration (S6).
   - `affinity/` — **Track 3, affinity**: the interface delta and its linear reweighting against
     the experimental affinity (S7): label sets, pose pools, per-complex aggregations, bootstrap
     evaluation on the three splits of S5.

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
ranking/                 Track 2 — ranking approaches vs. the unmodified interface delta
  data.py                pose tables from docked.parquet: the 7 interface terms, hit labels, buckets, 3 splits
  models.py              Reweight (non-negative, raw terms), Ranker (Luca's MLP), TreeRanker (LightGBM LambdaRank)
  evaluate.py            Success@k (as benchmark.py), bootstrap CIs, paired deltas, rank shift, approach tables
  train.py, report.py    CLIs: one approach x 3 splits -> data/ranking/<approach>_<split>/; tables -> data/figures/ranking/
  degeneracy.py          S6: per-run energy degeneracy (A.1) + per-run oracle (A.2) -> data/ranking/degeneracy[_total]/
  degeneracy_report.py   S6 figures, oracle-vs-approaches gap, example complexes, UMAP/t-SNE -> data/figures/degeneracy/
affinity/                Track 3 — score vs. affinity (S7): the interface delta and its reweighting against pK
  data.py                labels from the PDBbind index, pose pools, per-complex aggregations, feature cache; split from ranking.data
  models.py              calibrated interface delta, ligand size, free-sign / non-negative reweighting (+ size), all least squares
  evaluate.py            Pearson r, Spearman ρ, RMSE with bootstrap CIs and the paired Δ to the calibrated interface delta
  train.py, report.py    CLIs: the grid label set x split x pool x aggregation -> data/affinity/; tables + figures -> data/figures/affinity/
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

### Track 2 — `ranking/` (ranking approaches S5; degeneracy + oracle S6)

Setup decided 2026-10-20, documented with the results in [docs/ranking.md](docs/ranking.md):
the **seven raw interface-delta terms** are the only features anywhere in this track; hit =
symmetry-corrected `rmsd_to_crystal` ≤ 2 Å; training pool = `relax_docking_perturb` +
`apo_relax_docking_perturb` of the PDBbind training complexes (not crystal, not std), one
bucket per complex × receptor state (a pooled std run would join the perturb run of its
state; relax and apo_relax stay separate; poses are never compared across complexes);
evaluation per docking run on the PDBbind test complexes, PoseBusters and Astex, **headline =
Δ to the unmodified interface delta on `apo_relax_docking_perturb`**; all three non-random
splits (`lppdbbind`, `hac`, `cleansplit`) are reported; the 43 Astex entries that are PDBbind
complexes are removed from train/val. Thesis-era numbers and the random split are not used.

- `data.py` — `load_poses`, `assign_split`, `Preprocess` (`none` = raw energies for the
  reweighters and the tree; `robust` = Luca's scaling for the MLP, fitted on the training
  poses), `Grouped.build(..., group_by="state" | "protocol")`.
- `models.py` — `Reweight`: score = Σ w_t x_t on the raw terms, **w_t ≥ 0** (clamped after
  every step), all seven weights free (`--gauge fa_atr` would pin the scale; not used),
  `init="random"` or `"rosetta"` (two baselines; the learned vector is a Rosetta weight set,
  written normalised to `fa_atr` = 0.8 as `weights.wts`), InfoNCE loss only; `Ranker`: Luca's `7-64-32-16-1` MLP with his BCE + InfoNCE loss; `TreeRanker`:
  LightGBM LambdaRank, groups = buckets, early stopping on validation NDCG@1; `fit`: Adam, one
  bucket per step, best epoch by validation Success@1 then Success@5.
- `evaluate.py` — `per_group_success` (the benchmark's definition), bootstrap CIs, paired
  differences, `rank_shift` (MRR, earlier/same/later, log-rank shift),
  `approach_table_markdown` (one row per approach, one column per k).
- `train.py` — `uv run python -m ranking.train --model {reweight,ranker,tree} [--init random|rosetta]
  [--split all|lppdbbind|hac|cleansplit]`; loads the data once and trains one run per split
  into `data/ranking/<approach>_<split>/` (`results.md/json`, `per_run.parquet`, model,
  `preprocess.json`, `weights.wts` / `importance.csv`). Prints a sanity line with the unmodified
  Success@k on all complete PDBbind runs, which must equal the benchmark tables.
- `report.py` — `uv run python -m ranking.report`: the paper's tables (per protocol, headline
  first; per split for the PDBbind test complexes; per external set) and `ranking_results.csv`
  under `data/figures/ranking/`; reads only runs with `"schema": 2` (the first-pass runs of
  2026-10-19 are ignored and can be deleted).
- Dependency: `lightgbm` (added 2026-10-20) for the tree.
- `degeneracy.py` — **S6 / Appendix A.1 + A.2** (implemented 2026-10-21, documented in
  [docs/degeneracy.md](docs/degeneracy.md)): per docking run (complex × perturb protocol, all
  three receptor states, all three sets) the seven terms are z-scored over the run's poses;
  nearest-neighbour distances (Euclidean and Mahalanobis) of every pose to the nearest other
  correct / incorrect pose give the three populations of A.1; the run's degeneracy threshold is
  its median correct–correct distance; the per-run **oracle** is a logistic regression on the
  z-scores (in-sample and stratified-CV AUC, first-hit rank). Skipped by decision: the
  cross-complex permutation control. `uv run python -m ranking.degeneracy [--terms total]`
  → `data/ranking/degeneracy[_total]/{runs,poses}.parquet`, `summary.{csv,md}` (~1.5 min, 20 cores).
- `degeneracy_report.py` — `uv run python -m ranking.degeneracy_report [--no-embedding]`: the
  A.1 / A.2 figures per set, the oracle vs the schema-2 approaches on the same runs
  (`oracle_gap.csv`), three example runs (hard / median / easy by CV oracle AUC on the headline
  protocol, ≥ 10 hits; `examples.{png,csv}`), pooled UMAP + t-SNE of 100 runs, the three-panel
  summary `headline.png` (`--headline-only` writes just that), and `degeneracy_tables.md` under
  `data/figures/degeneracy/`.

### Track 3 — `affinity/` (score vs. affinity, S7 = Appendix A.3 step 1)

Rewritten 2026-10-21 on the full extraction (the legacy grid of the zip era is in the git history
before that date), documented with the results in [docs/affinity.md](docs/affinity.md). Question:
how well does the interface delta of the docked poses track the experimental affinity, and how
much does a linear reweighting of the seven interface terms add? Setup: labels = pK of the
**exact** Kd / Ki values of the PDBbind index (headline `kdki`, 6,708 complete complexes; IC50
separately, 4,579); the **same three splits as S5** (LP-PDBBind headline, HAC = CASF-2016 core as
test, CleanSplit), point fits on the training complexes and the conformal slack calibrated on
the validation complexes, plus the CASF core complexes outside train + val as external test;
**pools** = which poses are aggregated (one docking run, the perturb + std runs of a state, the
relaxed holo complex `relax_ref`, the unrelaxed `crystal` complex); **aggregations** `best`
(lowest interface delta, the headline), `mean10`, `mean` (over poses with interface delta ≤ 0),
`best_hit` (oracle); **approaches** = calibrated interface delta (reference), ligand heavy-atom
count, free-sign reweighting, non-negative reweighting (a Rosetta weight set), reweighting + size;
**metrics** (revised 2026-10-22: absolute prediction, not correlation) **MAE in pK against the
constant predictor**, paired ΔMAE with bootstrap intervals, **95 % lower bounds per complex**
(split conformal calibrated on the validation complexes, and a τ = 0.05 quantile model; coverage,
slack, share certified above pK 5 / 6), Pearson r secondary; point fits with `l1` (median
regression, headline) and `l2`; the reweightings refitted on three **term sets** (all / without
fa_pair / enthalpic). Headline cell: Kd + Ki, `apo_relax_docking_perturb`, best pose, LP-PDBBind
test complexes, l1 fits.

- `data.py` — `load_labels` (index parser; relation `=` only is exact), `RUN_POOLS` / `SEED_POOLS`,
  `aggregate`, `build_features` / `load_features` (cache `data/affinity/features.parquet`, ~40 s),
  `complex_info`, `casf_core_ids`; the split comes from `ranking.data.assign_split`. For the seed
  pools the interface delta is recomputed from the terms (the stored seed value is corrupt for
  five relax seeds and one crystal row).
- `models.py` — `Linear` (pK = X·coef + b), `fit_ols`, `fit_nnls` (non-negative on centred data),
  `fit_quantile` (median regression `l1` and the τ = 0.05 lower-bound model; IRLS, or a linear
  programme for the non-negative model), `TERM_SETS` (all / no_pair / enthalpic), `predictors()`;
  energy weights = −coef, normalised to fa_atr = 0.8 for the `.wts` file.
- `evaluate.py` — `evaluate(y, preds, bounds)`: MAE / RMSE / r with bootstrap intervals, paired
  ΔMAE to the reference and to the constant null, coverage / slack / certified share of the
  lower bounds; `conformal_slack` (one-sided split conformal); `mae_table_markdown`.
- `train.py` — `uv run python -m affinity.train [--labels kdki ic50] [--split all] [--pool ...|all]
  [--aggregation ...]`: builds the feature cache if missing, writes `baseline_all.csv` (the unmodified
  score on all labelled complexes, no split), then the grid → `data/affinity/{results.csv, weights.csv,
  weights_nonneg_<split>.wts, predictions.parquet, results.json, results.md, train.log}` (~10 min).
- `report.py` — `uv run python -m affinity.report`: `data/figures/affinity/affinity_tables.md`
  (eight tables), `affinity_results.csv`, `headline_scatter.png`, `pools.png`, `weights.png`.
- Not implemented by decision: Appendix A.3 step 2 (sampled-density free energy) and the RMSD
  clustering of the legacy grid.
- Result (2026-10-22, headline cell, MAE in pK): constant 1.56, unmodified 1.49, ligand size
  1.46, reweighted 1.45–1.46; the 95 % lower bound sits 2.4 pK below the prediction and certifies
  a 10 µM binder for ~10 % of the complexes; the best hit (1.54) and the relaxed holo complex
  (1.60) are worse than the best pose; dropping fa_pair costs nothing, the enthalpic set ≤ 0.01
  pK; the non-negative weights switch fa_rep and fa_pair off on every split. Pearson r (l2):
  0.31 / 0.42 / 0.37.

- [docking/result.py](docking/result.py) — `Result` / `PDBRinterface`: the H5
  reader used by the extraction. **Auto-corrects
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
| `ranking/degeneracy/`, `ranking/degeneracy_total/` | 530 M each | S6 tables of `ranking.degeneracy` (interface-delta terms / raw totals): `runs.parquet`, `poses.parquet` (one row per pose of every perturb run with a hit), `summary.{csv,md}`. |
| `affinity/` | 27 M | Track 3 (`affinity.train`): `features.parquet` (per complex × pool × aggregation: the aggregated terms, interface delta, RMSD), `labels.csv`, `baseline_all.csv`, `results.csv`, `weights.csv` + `.wts`, `predictions.parquet`, `results.{json,md}`, `train.log`. |
| `failed_logs/` | 472 K | SLURM `.out`/`.err` of the 57 docking tasks that failed on the cluster; source of `docking_analysis/failed_pdb_ids/`. |
| `figures/` | — | Generated figures and tables: `benchmark/` (Track 1), `ranking/` (Track 2 report), `degeneracy/` (S6 figures, examples, embeddings, `degeneracy_tables.md`), `affinity/` (Track 3 tables + figures); regenerated, safe to clear. |
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
  partial zips and the cluster cache no longer exist; every track reads the parquet tables.
- **One package per track.** New Track 2 code goes to `ranking/`, Track 3 code to `affinity/`;
  shared paths stay in `docking_analysis/constants.py`; `affinity` imports the split and the
  pose loader from `ranking.data`.
- Large data dirs and outputs are git-ignored (see [.gitignore](.gitignore)).
  **Do not commit `.h5`, parquet, large `.csv`, or figures.** Only code, docs and the
  small package data in `docking_analysis/` are tracked.
