# pdbbind-dock

Re-dock protein–ligand complexes with **PyRosetta**, record every pose together
with its decomposed Rosetta energy terms, and evaluate RosettaLigand at scale:
how well it *samples* near-native poses, how well its score *ranks* them, and how
the energy terms relate to binding affinity (log *K*d).

Benchmarks: PDBbind v2020 (11,460 docked complexes), the PoseBusters benchmark set
(428), and the Astex Diverse set (85).

> **Important — semantics of the stored docking results (no re-docking planned).**
> Every HDF5 file under `data/*_h5/` was produced by the docking code as it is in this
> repository, and that code is kept unchanged so the data stays reproducible. Two things
> about the stored numbers must be known:
>
> 1. `total_score` **includes the coordinate-constraint energy**
>    (`0.5 × raw_coordinate_constraint`), for the relaxed seeds (scored before the FastRelax
>    restraints were removed) and for every docked pose (the FinalMinimizer keeps its Cα
>    restraints). `idelta_score` is not affected.
> 2. `rmsd_to_crystal` / `rmsd_to_input` are in-place heavy-atom RMSDs **without symmetry
>    correction** (Rosetta `RMSDMetric`, atom-name matching).
>
> The reader in [`docking/result.py`](docking/result.py) therefore **auto-corrects these
> three columns by default** (`Result(..., autocorrect=True)`): constraint-free
> `total_score`, symmetry-corrected `rmsd_to_crystal` / `rmsd_to_input` recomputed from the
> stored structures, with the stored values kept as `*_legacy`. All other columns stay as
> stored. Pass `autocorrect=False` for the raw tables. The extraction
> (`docking_analysis.extract`) goes through this reader, so its parquet tables carry the
> corrected columns.

The project runs in two phases:

1. **Docking** (compute-heavy, built for an HPC/SLURM cluster) — for each complex,
   relax the holo structure with and without the ligand, run 300 docking trajectories
   per protocol from each of the three receptor states, and write one self-contained
   HDF5 file per complex.
2. **Analysis** (local) — extract the HDF5 files into parquet tables, compute the
   success-rate tables per receptor state × protocol, and train the energy-term
   reweighting against binding affinity.

## Layout

```
docking/             Phase-1 docking engine (Python package)
  main.py            CLI entry point — one complex per invocation (SLURM arrays)
  runner.py          relax + dock; writes one <pdb>.h5
  load_ligand.py     RDKit <-> Rosetta ligand handling
  ligand_rmsd.py     symmetry-corrected in-place ligand RMSD
  result.py          reader for the docking HDF5 files (Result / PDBRinterface)
  xml_protocols/     RosettaScripts docking protocols (std, perturb)
docking_analysis/    Phase-2 analysis package
  constants.py       all paths (everything resolves under data/)
  extract.py         H5 files -> parquet tables (data/docking_stats/<set>_full/)
  benchmark.py       success-rate tables per receptor state x protocol
  data.py, experiment.py, reweighting.py, figures.py, clustering.py
                     affinity reweighting grid (legacy inputs, see below)
extract_docking_stats.py   entry point of the affinity reweighting grid
scripts/             run_*.slurm, run_local.sh, filter_pdbbind.py, h5_size.py
docs/                methods documentation and the paper planning documents
reference/           tracked reference lists (CASF-2016 splits, failed ids)
data/                all inputs, docking outputs, caches, figures (git-ignored)
```

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/). PyRosetta 2025.47
(Rosetta commit `8bb54e8a6d`, GitHub tag `v2025.47-dev62202`) is installed from a
vendored wheel because the JHU wheel archive only keeps recent releases: place
`pyrosetta-2025.47+release.8bb54e8a6d-cp311-cp311-linux_x86_64.whl` (sha256 in
`pyproject.toml`; obtaining it implies accepting the PyRosetta license) into
`wheels/` (git-ignored), then

```bash
uv sync
```

The same build is available permanently from the PyRosetta conda channel
(`conda install -c https://conda.graylab.jhu.edu pyrosetta=2025.47`).

Run entry points as modules **from the repo root** so the packages resolve:

```bash
uv run python -m docking.main --help
uv run python -m docking_analysis.extract --help
uv run python -m docking_analysis.benchmark --help
```

## Phase 1 — docking

Inputs live under `data/` (`data/pdbbind_cleaned/<pdb>/<pdb>_protein.pdb` +
`_ligand.sdf`, plus an `index.txt` list); outputs go to `data/<set>_h5/`. The
PDBbind inputs are filtered from the raw download by
[`scripts/filter_pdbbind.py`](scripts/filter_pdbbind.py) — the filters are
documented in [docs/dataset_preparation.md](docs/dataset_preparation.md). The
PoseBusters and Astex inputs are the files distributed with the PoseBusters paper.

- **Cluster:** submit the SLURM array jobs in `scripts/`
  (`run_all.slurm`, `run_astex.slurm`, `run_posebuster.slurm`). They expect the
  data tree mirrored under `$WORKSPACE/data/`.
- **Local smoke test:** `uv run python -m docking.runner` (docks `5k00` once into
  `data/h5_test/`), or `bash scripts/run_local.sh` (GNU parallel over the index).

Per complex: 1 crystal pose, 10 FastRelax seeds with the ligand in place (`relax`),
10 FastRelax seeds with the ligand moved 500 Å away and placed back afterwards
(`apo_relax`), then 300 docking trajectories for each receptor state × protocol
(`docking_std.xml` starts at the input ligand placement, `docking_perturb.xml` adds
an initial random perturbation of up to 5 Å), i.e. 1,800 docked poses. Score function:
`ligand` weights with the RosettaLigand reweights, `-restore_pre_talaris_2013_behavior`.
Failed complexes are listed in [`reference/failed_pdb_ids/`](reference/failed_pdb_ids/).

### HDF5 schema (one file per complex)

- `ligand_sdf`, `atmname_to_idx`, `scfx_weights` — scalar JSON/SDF metadata.
- `poses/` — `pdb_strings` (zlib-compressed full PDB), `results`
  (float matrix; column names in the `column_names` attr), `row_names`
  (`crystal` / `relax` / `apo_relax`).
- top level — `pdb_strings` (per-pose **diffs** vs. the relaxed reference),
  `results` (per-docking-pose float matrix), `protocol` (`<state>_docking_<std|perturb>`).

`results` columns: `total_score`, `idelta_score`, `rmsd_to_crystal`,
`rmsd_to_input`, `prepare_time`, then `raw_<term>` / `raw_delta_<term>` per active
Rosetta energy term. Inspect a file with `uv run scripts/h5_size.py <file.h5>`.

## Phase 2 — analysis

All analyses work on the **full extraction** of each set:

```bash
uv run python -m docking_analysis.extract --set pdbbind      # H5 -> data/docking_stats/pdbbind_full/*.parquet
uv run python -m docking_analysis.extract --set posebuster
uv run python -m docking_analysis.extract --set astex
uv run python -m docking_analysis.benchmark                  # success tables -> data/figures/benchmark/
```

The extraction goes through `Result(autocorrect=True)` (see the note at the top), so
its tables carry constraint-free total scores and symmetry-corrected, in-place ligand
RMSDs next to the stored legacy values; every complex is included and
`complexes.parquet` flags the complete ones. The benchmark writes one success-rate
table per receptor state × protocol (sampling success, Top-1/5/10, analytic
success-vs-N), once ranked by interface delta and once by total score, with a legend
explaining every column (`data/figures/benchmark/success_tables_*.md`).

The **affinity reweighting grid** (aggregation × energy-term selection, default vs.
learned weights, optional pose clustering) is run with

```bash
uv run extract_docking_stats.py [--n_workers N] [--rmsd_cutoff 2.0]
```

It still reads the older zip bundles under `data/docking_stats/` (exact-Kd subset,
uncorrected RMSDs) and is due to be ported to the full extraction; see
[CLAUDE.md](CLAUDE.md).

All paths are defined in one place: **`docking_analysis/constants.py`**
(everything resolves under `data/`).

## Notes

- `data/` is git-ignored; it is large (the PDBbind outputs alone are ~370 GB) and
  not version-controlled.
- The paper planning documents are in [docs/](docs/README.md).
- See [CLAUDE.md](CLAUDE.md) for the developer-level documentation (data layout,
  legacy semantics, conventions).
