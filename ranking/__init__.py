"""
ranking — Track 2: can the pose ranking be repaired, and why does it resist?
============================================================================

Four approaches are compared with the **unmodified Rosetta interface delta** on the seven raw
interface-delta terms, trained on the pooled ``relax`` · ``perturb`` + ``apo_relax`` · ``perturb``
runs of the PDBbind training complexes (one bucket per complex × receptor state) and evaluated
per docking run on the PDBbind test complexes of each of the three non-random splits
(LP-PDBBind, HAC, CleanSplit), on PoseBusters and on Astex; the headline is the Δ Success@k on
``apo_relax`` · ``perturb``. Methods and results: ``docs/ranking.md`` (setup decided 2026-10-20).

- ``data.py``      pose tables from ``docked.parquet``: the seven features, hit labels, buckets,
                   the three splits, preprocessing (raw, or Luca's robust scaling for the MLP);
- ``models.py``    ``Reweight`` (non-negative reweighting of the raw terms, random or Rosetta
                   start — the physics baseline, its weights are a Rosetta weight set),
                   ``Ranker`` (Luca Friebe's MLP with his BCE + InfoNCE loss), ``TreeRanker``
                   (LightGBM LambdaRank), the training loop;
- ``evaluate.py``  Success@k as in ``docking_analysis.benchmark``, bootstrap intervals, paired
                   differences, AUCs, first-hit-rank shift, the one-row-per-approach table;
- ``train.py``     CLI: one approach, all three splits, outputs under ``data/ranking/<approach>_<split>/``;
- ``report.py``    CLI: the paper's tables over all runs under ``data/figures/ranking/``.

The MLP ranker and its loss are a port of Luca Friebe's ``rosettaligand-vs-posebuster``
(branch ``test_new_architecture``). Still planned (inventory S6, Appendix A.1/A.2): the
energy-degeneracy analysis and the per-complex oracle probe.
"""
