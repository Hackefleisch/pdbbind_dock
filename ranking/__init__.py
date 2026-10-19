"""
ranking — Track 2: can the pose ranking be repaired, and why does it resist?
============================================================================

Planned package; nothing is implemented yet. The analyses it will hold are specified in
``docs/PAPER1_STORY_INVENTORY.md`` (stories S5 and S6, Appendix A.1 and A.2) and all run on the
full extraction (``data/docking_stats/<set>_full/docked.parquet``):

- energy-degeneracy analysis: per-complex z-scored interface-term vectors, nearest-correct-pose
  distances of incorrect poses against the correct-correct and cross-complex reference scales,
  density plot, hexbin, headline percentages;
- per-complex oracle probe: a cross-validated linear classifier per complex as the upper bound on
  what any reweighting can achieve, AUC distribution;
- pose-ranking reweighting and a LambdaRank-style head (port of Luca Friebe's models) on
  constraint-free scores and symmetry-corrected RMSD labels, with a leakage-aware split.
"""
