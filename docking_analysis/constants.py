"""
docking_analysis.constants
--------------------------
Shared configuration: every path of the project, computed from the repository root
(``REPO_ROOT / "data" / ...``), plus the package data shipped with ``docking_analysis``.
The other tracks (``ranking``, ``affinity``) import their shared paths from here.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT  = Path(__file__).resolve().parent.parent
PACKAGE_DIR         = Path(__file__).resolve().parent
DATA_DIR            = REPO_ROOT / "data"
INDEX_PATH          = DATA_DIR / "pdbbind_2020plus" / "INDEX_general_PL.2020R1.lst"
OUTPUT_DIR          = DATA_DIR / "figures"

# small tracked reference data shipped with the package
SCFX_WEIGHTS_PATH   = PACKAGE_DIR / "scfx_weights.json"      # score-function weights as dumped by the runner
FAILED_PDB_IDS_DIR  = PACKAGE_DIR / "failed_pdb_ids"         # <set>.txt: complexes whose docking failed
SPLITS_PATH         = PACKAGE_DIR / "splits.csv"             # pdb, plindersplit, hacsplit, cleansplit, lppdbbindsplit

# docking outputs and their extraction
H5_DIRS             = {                      # one directory of <pdb>.h5 files per benchmark set
    "pdbbind":    DATA_DIR / "pdbbind_h5",
    "posebuster": DATA_DIR / "posebuster_h5",
    "astex":      DATA_DIR / "astex_h5",
}
EXTRACT_DIR         = DATA_DIR / "docking_stats"   # full extractions live in <EXTRACT_DIR>/<set>_full/
RANKING_DIR         = DATA_DIR / "ranking"         # Track 2 runs: <RANKING_DIR>/<run>/ (models, per-run tables, results)
DEGENERACY_DIR      = RANKING_DIR / "degeneracy"   # Track 2, S6: per-run energy degeneracy + oracle tables (ranking.degeneracy)
