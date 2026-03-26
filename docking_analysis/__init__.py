"""
docking_analysis
================
Modular toolkit for loading PDBbind docking data, generating figures,
and training energy-term reweighting models.

Submodules
----------
- constants : shared paths, regex, unit conversion, zero-variance terms
- data      : data loading, filtering, per-PDB aggregation helpers
- figures   : all matplotlib figure-generation functions
- reweighting : PyTorch-based linear reweighting of raw_delta energy terms
"""
