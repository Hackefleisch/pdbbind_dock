"""
affinity.constants
------------------
Configuration of the legacy affinity grid. Shared paths come from
:mod:`docking_analysis.constants`; everything else here belongs to this track only.

The zip bundles and the cluster cache named below were deleted on 2026-10-19; the paths are kept
so that the code documents what it read.
"""

import re

from docking_analysis.constants import (  # noqa: F401  (re-exported for the legacy modules)
    DATA_DIR,
    INDEX_PATH,
    OUTPUT_DIR,
    SCFX_WEIGHTS_PATH,
)

# ---------------------------------------------------------------------------
# Inputs of the legacy grid (deleted)
# ---------------------------------------------------------------------------
ZIP_PATH            = DATA_DIR / "docking_stats" / "pdbbind_docking_data.zip"                 # all complete PDBbind poses (notebook export)
FILTERED_ZIP        = DATA_DIR / "docking_stats" / "pdbbind_relax_perturb_filtered_kd_v2.zip"  # relax/apo_relax Perturb, scores <= 0, exact Kd
CLUSTER_CACHE_DIR   = DATA_DIR / "docking_stats" / "cluster_cache_filtered_v2"                # <rmsd_2.0|rmsd_4.0>/<pdb>.pkl
H5_DIR              = DATA_DIR / "pdbbind_h5"                                                 # read by clustering.py for the structures

# ---------------------------------------------------------------------------
# Unit -> Molar conversion factors
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
