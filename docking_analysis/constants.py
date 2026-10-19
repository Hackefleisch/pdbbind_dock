"""
docking_analysis.constants
--------------------------
Shared configuration: paths, regex patterns, unit conversion factors,
and the set of raw_delta energy terms known to have zero variance.
"""

import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT  = Path(__file__).resolve().parent.parent
DATA_DIR            = REPO_ROOT / "data"
ZIP_PATH            = DATA_DIR / "docking_stats" / "pdbbind_docking_data.zip"
FILTERED_ZIP        = DATA_DIR / "docking_stats" / "pdbbind_relax_perturb_filtered_kd_v2.zip"
INDEX_PATH          = DATA_DIR / "pdbbind_2020plus" / "INDEX_general_PL.2020R1.lst"
OUTPUT_DIR          = DATA_DIR / "figures"
SCFX_WEIGHTS_PATH   = Path(__file__).resolve().parent / "scfx_weights.json"
H5_DIR              = DATA_DIR / "pdbbind_h5"
H5_DIRS             = {                      # per benchmark set (docking_analysis.extract)
    "pdbbind":    DATA_DIR / "pdbbind_h5",
    "posebuster": DATA_DIR / "posebuster_h5",
    "astex":      DATA_DIR / "astex_h5",
}
EXTRACT_DIR         = DATA_DIR / "docking_stats"   # full extractions live in <EXTRACT_DIR>/<set>_full/
CLUSTER_CACHE_DIR   = DATA_DIR / "docking_stats" / "cluster_cache_filtered_v2"

# ---------------------------------------------------------------------------
# Unit → Molar conversion factors
# ---------------------------------------------------------------------------
UNIT_TO_MOLAR: dict[str, float] = {
    "mM": 1e-3,
    "uM": 1e-6,
    "nM": 1e-9,
    "pM": 1e-12,
    "fM": 1e-15,
    "M":  1.0,
}

# ---------------------------------------------------------------------------
# Regex: matches ONLY exact equalities  Kd=<value><unit>
# Skips inequalities (<, <=, >, >=, ~) and non-Kd measurements (Ki, IC50).
# ---------------------------------------------------------------------------
KD_RE = re.compile(
    r"^Kd=(?P<value>[0-9]+(?:\.[0-9]+)?)(?P<unit>mM|uM|nM|pM|fM|M)\b"
)

# ---------------------------------------------------------------------------
# raw_delta_* energy terms that are universally zero (constant across all
# poses in the dataset) and therefore carry no information for ML training.
# ---------------------------------------------------------------------------
ZERO_VARIANCE_TERMS: set[str] = {
    "raw_delta_fa_intra_rep",
    "raw_delta_pro_close",
    "raw_delta_hbond_sr_bb",
    "raw_delta_hbond_lr_bb",
    "raw_delta_dslf_ss_dst",
    "raw_delta_dslf_cs_ang",
    "raw_delta_dslf_ss_dih",
    "raw_delta_dslf_ca_dih",
    "raw_delta_atom_pair_constraint",
    "raw_delta_coordinate_constraint",
    "raw_delta_angle_constraint",
    "raw_delta_dihedral_constraint",
    "raw_delta_rama",
    "raw_delta_omega",
    "raw_delta_fa_dun",
    "raw_delta_p_aa_pp",
    "raw_delta_ref",
    "raw_delta_chainbreak",
}
