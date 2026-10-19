"""
docking_analysis
================
Phase-2 analysis package: extraction of the docking HDF5 files, benchmark
success-rate tables, and the energy-term reweighting against binding affinity.

Submodules
----------
- constants   : all paths (under ``data/``), regexes, unit conversions
- extract     : full-set extraction, H5 -> parquet (``docked``, ``seeds``, ``complexes``)
- benchmark   : success-rate tables per receptor state x protocol on the extraction
- data        : loading / filtering / per-PDB aggregation (affinity grid; reads the legacy zips)
- experiment  : experiment configs and results matrix of the affinity grid
- reweighting : PyTorch linear reweighting of ``raw_delta_*`` terms against log Kd
- figures     : matplotlib figures of the affinity grid
- clustering  : RMSD-based pose clustering with a pickle cache (affinity grid)

``extract`` and ``benchmark`` work on ``data/docking_stats/<set>_full/``. The affinity
grid (``data``, ``experiment``, ``reweighting``, ``figures``, ``clustering`` and the
``extract_docking_stats.py`` entry point) still reads the older zip bundles and is
due to be ported to the full extraction.
"""
