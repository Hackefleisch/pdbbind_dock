"""
affinity — Track 3: energy terms vs. binding affinity (LEGACY, documentation only)
==================================================================================

This package holds the first implementation of the affinity track: a linear reweighting of
the per-pose Rosetta interface terms (``raw_delta_*``) against log Kd, evaluated over a grid of
per-complex aggregations (lowest pose, mean of the 10 lowest, mean of all poses, RMSD-cluster
representatives) x energy-term selections (all terms, combined h-bond terms, without
``fa_sol``/``fa_pair``, combined van der Waals), with default and learned weights.

**It is not runnable any more and is kept only as documentation of what has been done.**
Its inputs were the legacy zip bundles (exact-Kd subset of the relax/apo_relax Perturb poses,
uncorrected RMSDs, totals including the constraint energy) and the cluster cache, all deleted on
2026-10-19. The track will be rewritten on the full extraction
(``data/docking_stats/<set>_full/``, see ``docking_analysis.extract``), with the limitations listed
in ``docs/PAPER1_STORY_INVENTORY.md`` S7 lifted (Ki labels, leakage-aware split, bootstrap CIs,
symmetry-corrected RMSD in the clustering, Standard poses in/out).

Modules
-------
- constants             : paths of the deleted inputs, the Kd regex, unit conversions, zero-variance terms
- data                  : Kd index parsing, zip loading, per-PDB aggregation (min / mean_n / filtered_mean / clustered)
- experiment            : aggregation x term-selection configs, default-weight baseline, results matrix
- reweighting           : bias-free, non-negative linear layer (PyTorch, L1 loss) on the aggregated terms
- figures               : density scatters, predicted-vs-actual, learned-vs-default weight bars
- clustering            : greedy RMSD clustering of poses per complex with a pickle cache
- extract_docking_stats : the orchestrator that ran the whole grid
"""
