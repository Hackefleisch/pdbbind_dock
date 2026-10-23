# docs

Methods-level documentation and the planning documents of the paper. Keep the
methods pages in sync with the code; each page states the authoritative source
file(s) it documents.

## Methods

- [dataset_preparation.md](dataset_preparation.md) — how `pdbbind_cleaned` is
  filtered and built from the raw PDBbind v2020 download
  (source: [`scripts/filter_pdbbind.py`](../scripts/filter_pdbbind.py)).
- [ranking.md](ranking.md) — Track 2: the pose-ranking models ported from Luca Friebe's
  repository (data, labels, split, models, loss, evaluation), what changed in the port, and
  their results on the full extraction (source: [`ranking/`](../ranking/)).
- [degeneracy.md](degeneracy.md) — Track 2, S6: why the ranking resists repair — the
  energy-degeneracy analysis (Appendix A.1), the per-complex oracle (A.2), the example
  complexes and the UMAP/t-SNE illustration (source: [`ranking/degeneracy.py`](../ranking/degeneracy.py),
  [`ranking/degeneracy_report.py`](../ranking/degeneracy_report.py)).
- [affinity.md](affinity.md) — Track 3, S7: the interface delta and its linear reweighting
  against the experimental affinity — labels, pose pools, aggregations, splits, the bootstrap
  evaluation and the results (source: [`affinity/`](../affinity/)).
- [basins.md](basins.md) — Track 3, Appendix A.3 step 2a: the basins of every docking run —
  pairwise symmetry-corrected ligand RMSD, greedy leader clustering, the basin and run tables
  and their descriptive statistics (source: [`affinity/basins.py`](../affinity/basins.py)).

## Paper 1 (characterisation of RosettaLigand)

- [PAPER1_STORY_INVENTORY.md](PAPER1_STORY_INVENTORY.md) — the planning document of
  Paper 1: the three-track story, per story what the data supports today (with the
  full-set success tables), the caveats, what is still missing, the specifications of
  the new analyses (Appendix A) and the additional compute (Appendix B). The earlier
  section outline was retired on 2026-10-19 (see the git history for it).

## Paper 2 (deferred)

- [PAPER2_CONCEPT.md](PAPER2_CONCEPT.md) — concept of the follow-up paper on ensemble
  thermodynamics. Nothing in the code implements it yet.

## Pages still to write

- `docking_protocol.md` — the relax + dock procedure, the RosettaScripts protocols,
  score function, and per-complex sampling settings
  (source: `docking/runner.py`, `docking/xml_protocols/`; the verified facts are
  collected in PAPER1_STORY_INVENTORY.md §0.2).
- `benchmarks.md` — the PoseBusters / Astex Diverse evaluation sets and how they
  differ from the PDBbind set; the success-rate metrics of `docking_analysis/benchmark.py`.

See [../CLAUDE.md](../CLAUDE.md) for the code-level overview and
[../README.md](../README.md) for the project summary.
