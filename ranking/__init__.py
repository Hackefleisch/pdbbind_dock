"""
ranking — Track 2: can the pose ranking be repaired, and why does it resist?
=============================================================================

Implemented (2026-10-19): the pose-ranking models of Luca Friebe's thesis repository
(``github.com/lucafrb/rosettaligand-vs-posebuster``, branch ``test_new_architecture``) ported
to the full extraction (``data/docking_stats/<set>_full/docked.parquet``) — inventory story S5.
His repository stays the reference for the thesis-era numbers; the port fixes the labels
(symmetry-corrected ``rmsd_to_crystal`` instead of the stored ``rmsd_to_input``), the features
(constraint-free), the split (leakage-aware instead of random by complex) and removes the
Astex entries from the PDBbind training set.

- ``data.py``      pose tables: the seven interface terms (or the twenty constraint-free total
                   terms) as features, hits from ``rmsd_to_crystal`` ≤ 2 Å, one docking run
                   (complex × protocol) as the ranking unit, the PDBbind split (LP-PDBBind by
                   default; HAC/CASF, CleanSplit, or Luca's random group split);
- ``models.py``    ``Reweight`` (bias-free linear reweighting) and ``Ranker`` (64-32-16-1 MLP),
                   the BCE + InfoNCE loss, one run per batch, best epoch by validation Success@1/5;
- ``evaluate.py``  Success@k as in ``docking_analysis.benchmark``, bootstrap intervals, paired
                   differences to the interface delta, pooled and per-run AUC;
- ``train.py``     CLI: one run = train + evaluate on PDBbind test, PoseBusters and Astex,
                   outputs under ``data/ranking/<run>/``;
- ``report.py``    CLI: comparison tables of all runs under ``data/figures/ranking/``.

Still planned (inventory S6, Appendix A.1/A.2): the energy-degeneracy analysis and the
per-complex oracle probe; a tree model as a second reference head (S5 b).
"""
