"""
affinity — Track 3: Rosetta energy terms vs. binding affinity
==============================================================

The affinity question of Paper 1 (inventory S7 = Appendix A.3 step 1): how well does the Rosetta
interface delta of the docked poses track the experimental affinity (pK = −log10 of the exact
Kd, Ki or IC50 of the PDBbind index), and how much does a linear reweighting of its seven
interface terms add? Everything runs on the full extraction (``data/docking_stats/pdbbind_full/``).

Modules
-------
- data      : labels from the PDBbind index, pose pools (docking runs, perturb + std, the relaxed
              holo and the crystal complex), per-complex aggregations (best pose, mean of 10,
              filtered mean, best-hit oracle), the cached feature table, the split (shared with S5)
- models    : the linear approaches — calibrated interface delta, ligand size, free-sign and
              non-negative reweighting of the raw terms, reweighting + size — all least squares
- evaluate  : Pearson r / Spearman ρ / RMSE with bootstrap intervals and the paired Δ to the reference
- train     : the grid (label set × split × pool × aggregation) -> ``data/affinity/``
- report    : the tables and figures -> ``data/figures/affinity/``

The first implementation (an aggregation × term-selection grid on the legacy zip bundles,
exact Kd only, random split, MAE loss) was removed on 2026-10-21; it is in the git history
before that date. This rewrite lifts its limitations listed in ``docs/PAPER1_STORY_INVENTORY.md``
S7 (Ki labels, leakage-aware splits, bootstrap intervals, corrected RMSDs and totals, Standard
poses in or out). Appendix A.3 step 2 (the sampled-density free energy) is not implemented.
"""
