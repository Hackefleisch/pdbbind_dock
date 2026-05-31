# pdbbind-dock

Re-dock protein–ligand complexes with **PyRosetta**, record every pose together
with its decomposed Rosetta energy terms, and train a **linear reweighting model**
that maps those terms to binding affinity (log *K*d).

The project runs in two phases:

1. **Docking** (compute-heavy, built for an HPC/SLURM cluster) — for each complex,
   relax the holo and apo structures, run many docking trajectories per protocol,
   and write one self-contained HDF5 file per complex.
2. **Analysis** (local) — read the HDF5 files, optionally cluster poses by RMSD,
   train the energy-term reweighting, and generate figures and result tables.

Benchmarks: PDBbind v2020 (~11,460 complexes), the PoseBusters benchmark set
(428), and the Astex Diverse set (85).

## Layout

```
docking/             Phase-1 docking engine (Python package)
  main.py            CLI entry point — one complex per invocation (SLURM arrays)
  runner.py          relax + dock; writes one <pdb>.h5
  load_ligand.py     RDKit <-> Rosetta ligand handling
  result.py          reader for the docking HDF5 files (Result / PDBRinterface)
  xml_protocols/     RosettaScripts docking protocols
docking_analysis/    Phase-2 analysis package (paths, data, clustering,
                     experiments, reweighting, figures)
extract_docking_stats.py   Phase-2 entry point
notebooks/           extraction + exploratory notebooks
scripts/             run_*.slurm, run_local.sh, h5_size.py
reference/           tracked reference lists (CASF-2016 splits, failed ids)
data/                all inputs, docking outputs, caches, figures (git-ignored)
```

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/). PyRosetta is
pinned to a JHU wheel (installing it implies accepting the PyRosetta license).

```bash
uv sync
```

Run entry points as modules **from the repo root** so the packages resolve:

```bash
uv run python -m docking.main --help
uv run extract_docking_stats.py --help
```

## Phase 1 — docking

Inputs live under `data/` (`data/pdbbind_cleaned/<pdb>/<pdb>_protein.pdb` +
`_ligand.sdf`, plus an `index.txt` list); outputs go to `data/<set>_h5/`.

- **Cluster:** submit the SLURM array jobs in `scripts/`
  (`run_all.slurm`, `run_astex.slurm`, `run_posebuster.slurm`). They expect the
  data tree mirrored under `$WORKSPACE/data/`.
- **Local smoke test:** `bash scripts/run_local.sh`, or `uv run python -m docking.runner`
  (docks `5k00` into `data/h5_test/`).

Docking defaults: 10 holo relaxes, 10 apo relaxes, 300 docking trajectories per
protocol, using `docking_perturb.xml` and `docking_std.xml`.

### HDF5 schema (one file per complex)

- `ligand_sdf`, `atmname_to_idx`, `scfx_weights` — scalar JSON/SDF metadata.
- `poses/` — `pdb_strings` (zlib-compressed full PDB), `results`
  (float matrix; column names in the `column_names` attr), `row_names`
  (`crystal` / `relax` / `apo_relax`).
- top level — `pdb_strings` (per-pose **diffs** vs. the relaxed reference),
  `results` (per-docking-pose float matrix), `protocol`.

`results` columns: `total_score`, `idelta_score`, `rmsd_to_crystal`,
`rmsd_to_input`, `prepare_time`, then `raw_<term>` / `raw_delta_<term>` per active
Rosetta energy term. Inspect a file with `uv run scripts/h5_size.py <file.h5>`.

## Phase 2 — analysis

```bash
uv run extract_docking_stats.py [--n_workers N] [--rmsd_cutoff 2.0]
```

This loads the pre-filtered docking subset and the *K*d index, runs the
experiment grid (aggregation × energy-term selection), optionally clusters poses,
trains the reweighting model, and writes figures and result CSVs to `data/figures/`.
The notebooks in `notebooks/` build the extracted-dataframe bundles
(`extract_docking_dataframe.ipynb`) and support exploratory analysis; launch them
with the repo root as the working directory.

All paths are defined in one place: **`docking_analysis/constants.py`**
(everything resolves under `data/`).

## Notes

- `data/` is git-ignored; it is large (the PDBbind outputs alone are ~370 GB) and
  not version-controlled.
- See [CLAUDE.md](CLAUDE.md) for deeper developer/agent-oriented documentation.
