"""
docking_analysis — Track 1: how well does RosettaLigand sample and rank?
========================================================================
Extraction of the docking HDF5 files into parquet tables and the benchmark success-rate
tables (sampling success, Top-k by score, analytic success-vs-N) per receptor state x protocol.

Submodules
----------
- constants : all paths of the project (under ``data/``) and the package data below
- extract   : full-set extraction, H5 -> parquet (``docked``, ``seeds``, ``complexes``, ``manifest.json``)
- benchmark : success-rate tables on the extraction, one per receptor state x protocol

Package data
------------
- ``scfx_weights.json``  : score-function weights as dumped by the docking runner
- ``failed_pdb_ids/``    : ids of the complexes whose docking failed, one file per benchmark set
- ``splits.csv``         : PDBbind split annotations (PLINDER, HAC, CleanSplit, LP-PDBBind)

The other analysis tracks are separate packages: ``ranking`` (Track 2, planned) and ``affinity``
(Track 3, legacy implementation kept as documentation).
"""
