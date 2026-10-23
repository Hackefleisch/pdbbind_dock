# pdbbind-dock

Re-dock protein–ligand complexes with **PyRosetta** (RosettaLigand), record every pose
together with its decomposed Rosetta energy terms, and evaluate RosettaLigand at scale:
how well it *samples* near-native poses, how well its score *ranks* them, and how the
energy terms relate to binding affinity (pK from the exact *K*d / *K*i values).

Benchmarks: PDBbind v2020 (11,460 docked complexes), the PoseBusters benchmark set
(428) and the Astex Diverse set (85). Per complex: 1 crystal pose, 10 FastRelax seeds
with the ligand in place (`relax`), 10 FastRelax seeds with the ligand moved 500 Å away
and placed back afterwards (`apo_relax`), then 300 docking trajectories for each of the
three receptor states × two protocols (`std` starts at the input ligand placement,
`perturb` adds an initial random perturbation of up to 5 Å), i.e. 1,800 docked poses.

> **Semantics of the stored docking results (no re-docking planned).** Every HDF5 file
> under `data/*_h5/` was produced by the docking code as it is in this repository, and that
> code is kept unchanged so the data stays reproducible. Two things about the stored
> numbers must be known: (1) `total_score` **includes the coordinate-constraint energy**
> (`0.5 × raw_coordinate_constraint`) for the relaxed seeds and for every docked pose;
> `idelta_score` is not affected. (2) `rmsd_to_crystal` / `rmsd_to_input` are in-place
> heavy-atom RMSDs **without symmetry correction**. The reader in
> [`docking/result.py`](docking/result.py) therefore **auto-corrects these three columns
> by default** (constraint-free total, symmetry-corrected RMSDs recomputed from the stored
> structures; the stored values are kept as `*_legacy`). The extraction goes through this
> reader, so its tables carry the corrected columns. Details: [CLAUDE.md](CLAUDE.md).

## Repository layout

Four code directories: the docking engine and one package per analysis track of the
paper ([docs/PAPER1_STORY_INVENTORY.md](docs/PAPER1_STORY_INVENTORY.md)).

```
docking/             Phase 1 — docking engine (needs PyRosetta)
  main.py            CLI entry point: one complex per invocation (SLURM arrays)
  runner.py          relax + dock; writes one <pdb>.h5, resumes an existing file
  load_ligand.py     RDKit <-> Rosetta ligand handling, conformers, reprotonation
  ligand_rmsd.py     symmetry-corrected in-place ligand RMSD (PyRosetta-free)
  result.py          reader for the HDF5 files with the auto-correction (PyRosetta-free)
  xml_protocols/     RosettaScripts protocols docking_std.xml / docking_perturb.xml
docking_analysis/    Phase 2, Track 1 — extraction + benchmark success tables
  constants.py       all paths of the project (everything resolves under data/)
  extract.py         H5 files -> parquet tables (data/docking_stats/<set>_full/)
  benchmark.py       success-rate tables per receptor state x protocol
  scfx_weights.json  score-function weights as dumped by the runner
  failed_pdb_ids/    ids of the complexes whose docking failed, one file per set
  splits.csv         PDBbind split annotations (PLINDER, HAC, CleanSplit, LP-PDBBind)
ranking/             Phase 2, Track 2 — ranking approaches (non-negative reweighting, tree,
                     Luca Friebe's MLP ranker) vs. the unmodified interface delta
affinity/            Phase 2, Track 3 — interface delta vs. affinity: pose pools, per-complex
                     aggregations, linear reweighting, bootstrap intervals on the three splits
scripts/             run_*.slurm (cluster launchers), filter_pdbbind.py, h5_size.py
docs/                dataset preparation, paper planning documents
data/                all inputs, docking outputs, extractions, figures (git-ignored)
```

## 1. Preparing the repository

### 1.1 Python environment

Requirements: Linux x86_64, Python 3.11 (pinned in `.python-version`; the PyRosetta
wheel is `cp311`), [uv](https://docs.astral.sh/uv/). The four packages are installed in
editable mode, so every entry point runs as `python -m <package>.<module>` from any
directory.

```bash
git clone <repo>  && cd pdbbind_dock        # TODO: final repository URL
uv sync --frozen                            # analysis only: everything except PyRosetta
uv sync --group docking                     # docking as well: needs the PyRosetta wheel, see 1.2
```

`--frozen` is required for the analysis-only install because PyRosetta is declared as a
local wheel; without the file present a plain `uv sync` fails while re-validating the
lock. The analysis tracks (reading the HDF5 files, extraction, benchmark) do not import
PyRosetta.

### 1.2 PyRosetta (docking only)

All docking runs used **PyRosetta 2025.47** (`pyrosetta-2025.47+release.8bb54e8a6d`,
Rosetta commit `8bb54e8a6d`, GitHub tag `v2025.47-dev62202`), with the init flags
`-in:auto_setup_metals -ex1 -ex2 -restore_pre_talaris_2013_behavior true`. Obtaining
PyRosetta implies accepting its license (free for academic use, see pyrosetta.org).

`pyproject.toml` expects the wheel at
`wheels/pyrosetta-2025.47+release.8bb54e8a6d-cp311-cp311-linux_x86_64.whl` (git-ignored;
sha256 `724b440b…1c00f6` recorded in `uv.lock`). Where to get it:

- **TODO (open decision):** the PyRosetta wheel archives (`graylab.jhu.edu` and
  `west.rosettacommons.org`) keep only the latest ~8 weekly releases, so 2025.47 can no
  longer be downloaded as a wheel. The identical build is served permanently by the
  PyRosetta conda channel (`conda install -c https://conda.graylab.jhu.edu pyrosetta=2025.47`),
  which uv cannot consume. How the exact build is distributed to readers (conda
  environment file, re-hosted wheel, or "any current release") is not decided yet.
- A **current** PyRosetta release (not 2025.47) installs through uv from the
  Rosetta Commons flat index, e.g.
  `uv pip install --find-links https://west.rosettacommons.org/pyrosetta/release/release/PyRosetta4.Release.python311.ubuntu.wheel/ pyrosetta`.
  Use it only for new runs or smoke tests, not to reproduce the stored data.

### 1.3 Data

Everything lives under `data/` (git-ignored). Sizes are those of the complete project.

| `data/…` | Content | Source | Size |
|---|---|---|---|
| `pdbbind_2020plus/` | PDBbind v2020 "special data package": the 19,037 protein–ligand complexes of v2020 that remain in v2024, with structures re-processed by the v2024 workflow; index `INDEX_general_PL.2020R1.lst` (binding data, resolution, annotations) | PDBbind+ website, free for registered users. **TODO:** exact package name, download date and citation | 16 GB |
| `posebusters_paper_data/` | PoseBusters benchmark set (428) and Astex Diverse set (85) as distributed with the PoseBusters paper, incl. the two `*_ids.txt` lists | Zenodo [10.5281/zenodo.8278563](https://zenodo.org/records/8278563), `posebusters_paper_data.zip` (CC BY 4.0) | 211 MB |
| `pdbbind_cleaned/` | the PDBbind docking inputs (`<pdb>_protein.pdb` + `<pdb>_ligand.sdf`, `index.txt`), derived by `scripts/filter_pdbbind.py` (section 2.1) | derived. **TODO:** ship the exact `index.txt` of the docked set (11,460 ids) | 2.5 GB |
| `pdbbind_h5/`, `posebuster_h5/`, `astex_h5/` | docking outputs, one `<pdb>.h5` per complex (section 2.3) | **TODO:** deposit (e.g. Zenodo) and DOI; ~390 GB in total | 374 + 14 + 3.2 GB |
| `docking_stats/<set>_full/` | parquet extraction of the H5 files (section 3.1) | derived; **TODO:** deposit the compact per-pose tables (~3.2 GB) for readers who do not want the H5 files | 3.2 GB |
| `ranking/<approach>_<split>/` | Track 2 runs (section 3.2): model, preprocessing, per-run tables, results | derived | small |
| `failed_logs/` | SLURM logs of the docking tasks that failed on the cluster | optional; source of `docking_analysis/failed_pdb_ids/` | 0.5 MB |
| `figures/` | benchmark tables and figures | derived | small |

## 2. Phase 1 — docking

### 2.1 Prepare the inputs

PDBbind: filter the raw download into the docking inputs
([docs/dataset_preparation.md](docs/dataset_preparation.md) documents every filter):

```bash
uv run scripts/filter_pdbbind.py --dry-run                 # counts only
uv run scripts/filter_pdbbind.py --out data/pdbbind_cleaned
```

The index file has been updated by PDBbind since the original run, so a fresh run yields
slightly different counts (the docked set has 11,460 complexes; 58 further ids were
removed after a first failed run, see `docking_analysis/failed_pdb_ids/pdbbind.txt`).

PoseBusters and Astex: the complexes are docked as distributed; `main.py` reads the ids
from `posebusters_benchmark_set_ids.txt` / `astex_diverse_set_ids.txt` and the files
`<id>/<id>_protein.pdb` + `<id>/<id>_ligand.sdf`.

### 2.2 Run

One complex per invocation, selected by its line index in an id list:

```bash
uv run python -m docking.main \
    --pdbbind data/pdbbind_cleaned --pdb_file data/pdbbind_cleaned/index.txt --pdb_index 0 \
    --n_relax 10 --n_relax_ligaway 10 --n_dock 300 \
    --outdir data/pdbbind_h5 \
    --protocols docking/xml_protocols/docking_perturb.xml docking/xml_protocols/docking_std.xml
```

On the cluster, the SLURM array scripts in `scripts/` do exactly this for every index
(`run_all.slurm`: array 0–11459, `run_posebuster.slurm`: 0–427, `run_astex.slurm`: 0–84).
Edit `WORKSPACE` and `VENV_PYTHON` at the top of each script; they expect the data tree
mirrored under `$WORKSPACE/data/` and the repository checked out at
`$WORKSPACE/pdbbind_dock` with its venv. Resource request: 1 CPU, 10 GB, 3 days per task.
Re-submitting a task **resumes** its `<pdb>.h5` (finished seeds and poses are kept).
Cost: ≈ 1–17 min per relax seed and ≈ 1–4 s per docked pose, i.e. **TODO:** report the
distribution of the wall time per complex from `prepare_time`.

Smoke test without a cluster: `uv run python -m docking.runner` docks `5k00` once into
`data/h5_test/`.

### 2.3 Output: one HDF5 file per complex

- `ligand_sdf`, `atmname_to_idx`, `scfx_weights` — scalar JSON/SDF metadata.
- `poses/` — `pdb_strings` (zlib-compressed full PDB), `results` (float matrix; column
  names in the `column_names` attr), `row_names` (`crystal` / `relax` / `apo_relax`).
- top level — `pdb_strings` (per-pose **diffs** vs. the relaxed reference), `results`
  (per-docking-pose float matrix), `protocol` (`<state>_docking_<std|perturb>`).

`results` columns: `total_score`, `idelta_score`, `rmsd_to_crystal`, `rmsd_to_input`,
`prepare_time`, then `raw_<term>` / `raw_delta_<term>` per active Rosetta energy term.
Inspect a file with `uv run scripts/h5_size.py <file.h5>`; read it with
`docking.result.Result` (auto-corrected) or `Result(..., autocorrect=False)` (as stored).
Complexes that failed are listed per set in `docking_analysis/failed_pdb_ids/`.

## 3. Phase 2 — analysis

### 3.1 Track 1 — sampling vs. ranking (`docking_analysis/`)

```bash
uv run python -m docking_analysis.extract --set pdbbind      # ~30 min with 20 workers
uv run python -m docking_analysis.extract --set posebuster
uv run python -m docking_analysis.extract --set astex
uv run python -m docking_analysis.benchmark                  # all three sets
```

The extraction writes `docked.parquet` (one row per docked pose), `seeds.parquet`,
`complexes.parquet` (completeness, ligand symmetry, references) and `manifest.json` under
`data/docking_stats/<set>_full/`; every complex is included and `complexes.complete`
flags the ones with all 6 × 300 poses. The benchmark writes one success-rate table per
receptor state × protocol (sampling success, Top-1/5/10 by score, analytic
success-vs-N), once ranked by interface delta and once by the corrected total score, with
a legend for every column, to `data/figures/benchmark/success_tables_*.md` (+ CSV and
per-complex parquet). **TODO:** the figures (success-vs-N, any-hit vs. Top-1,
rank-of-first-hit) are not generated yet.

### 3.2 Track 2 — can the ranking be fixed? (`ranking/`)

Four approaches are compared with the unmodified Rosetta interface delta on the seven raw
interface energy terms: a **non-negative reweighting** of the terms (from a random start and
from the Rosetta weights; the learned vector is a Rosetta weight set), a **LightGBM LambdaRank
tree**, and the **MLP ranker** of Luca Friebe's thesis repository
([lucafrb/rosettaligand-vs-posebuster](https://github.com/lucafrb/rosettaligand-vs-posebuster),
branch `test_new_architecture`, ported here). They are trained on the pooled relax · perturb
and apo_relax · perturb runs of the PDBbind training complexes (one bucket per complex and
receptor state, so poses are only ranked against poses of the same system) and evaluated by
Success@k on the PDBbind test complexes of the three non-random splits (LP-PDBBind, HAC,
CleanSplit), PoseBusters and Astex; the headline number is the difference to the unmodified
ranking on apo_relax · perturb. Methods and results: [docs/ranking.md](docs/ranking.md).

```bash
uv run python -m ranking.train --model reweight --init random     # all three splits
uv run python -m ranking.train --model reweight --init rosetta
uv run python -m ranking.train --model ranker
uv run python -m ranking.train --model tree
uv run python -m ranking.report                                   # tables of all runs
```

Each run writes `results.md/json`, the per-run table, the model and the preprocessing to
`data/ranking/<approach>_<split>/`; the report collects every run into
`data/figures/ranking/ranking_tables.md` (one row per approach, one column per k). Why the
ranking resists repair — the energy-degeneracy analysis and the per-complex oracle (inventory
Appendix A.1/A.2) — is `ranking.degeneracy` + `ranking.degeneracy_report`, documented in
[docs/degeneracy.md](docs/degeneracy.md).

### 3.3 Track 3 — interface delta vs. affinity (`affinity/`)

How well does the Rosetta interface delta of the docked poses track the experimental affinity,
and how much does a linear reweighting of its seven interface terms add? Labels are the pK of
the exact Kd and Ki values of the PDBbind index (IC50 reported separately); the poses of a
complex are aggregated per pool (one docking run, the perturb + std runs of a receptor state,
the relaxed holo complex, the crystal complex) and aggregation (best pose, mean of the 10 best,
filtered mean, best hit as oracle); the constant predictor, the calibrated interface delta, the
ligand heavy-atom count and three reweightings of the interface terms (median regression and
least squares, three term sets) are compared by the MAE in pK with bootstrap intervals and by
95 % lower bounds per complex (split conformal and quantile regression) on the test complexes
of the three splits of Track 2 and on the CASF-2016 core set; Pearson r is kept for comparison
with the CASF scoring power. Methods and results: [docs/affinity.md](docs/affinity.md).

```bash
uv run python -m affinity.train                 # feature cache + the grid, ~15 min
uv run python -m affinity.report                # tables and figures
uv run python -m affinity.diagnostics           # regression-to-the-mean and pose-energy-distribution diagnostics
uv run python -m affinity.basins                # basins of every perturb run (pairwise ligand RMSD + leader clustering), ~30 min
```

Outputs: `data/affinity/` (feature cache, `results.csv`, weights, per-complex predictions) and
`data/figures/affinity/affinity_tables.md` with the figures; `data/affinity/basins/` (the basin
tables and the pairwise RMSD matrices, [docs/basins.md](docs/basins.md)).

## Open TODOs

- Final repository URL and data deposit (H5 files and/or the parquet extraction) with DOI.
- Distribution of the exact PyRosetta build 2025.47 (section 1.2).
- Exact PDBbind package name, download date and citation; ship the docked-set `index.txt`.
- Wall-time distribution per complex (section 2.2).
- Track 1 figures (section 3); Appendix A.3 step 2 of the affinity track (part of this paper
  by decision; the basins of step 2a exist, the test of their statistics and step 2b are open).

See [CLAUDE.md](CLAUDE.md) for the developer-level documentation and
[docs/](docs/README.md) for the paper planning documents.
