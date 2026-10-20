"""
ranking.data
------------
Pose tables for the pose-ranking models (Track 2): features, labels, groups and the PDBbind
split, read from the full extraction (``<EXTRACT_DIR>/<set>_full/docked.parquet``).

Port of the data handling of Luca Friebe's ``rosettaligand-vs-posebuster`` (branch
``test_new_architecture``, ``ranking/data_loader.py``), rewritten against the parquet tables.
What is kept: the feature preprocessing (``fa_rep`` clipped at 50, robust scaling, scaled values
clipped to ±10), the relevance bins used to stratify the random split, one docking run as the
unit of ranking. What differs: labels come from the symmetry-corrected ``rmsd_to_crystal``
instead of the stored ``rmsd_to_input``; the scaler is fitted on the training poses only; the
split is leakage-aware by default; PDBbind entries that are also Astex complexes are removed
from training.

Features (``FEATURE_SETS``)
    ``delta``  the seven non-zero interface-delta terms (``raw_delta_*``: fa_atr, fa_rep, fa_sol,
               fa_elec, fa_pair, hbond_bb_sc, hbond_sc) — the terms behind ``idelta_score``.
               Luca's 25 requested terms reduce to exactly these seven once the all-zero ones
               are dropped.
    ``total``  the twenty constraint-free total-energy terms (``raw_*`` without the four
               constraint terms and ``chainbreak``); the terms behind the corrected ``total_score``.
    ``both``   the union.
Labels
    a pose is a hit when ``rmsd_col`` (default ``rmsd_to_crystal``) is <= ``cutoff`` (2 Å).
Groups
    one docking run = (pdb, protocol) = 300 poses; the losses and the Success@k metrics are
    computed within a run.
PDBbind split (:func:`assign_split`)
    ``lppdbbind``  LP-PDBBind train/val/test (``docking_analysis/splits.csv``), unassigned dropped;
    ``hac``        HAC split: test = CASF-2016 core set, val = CASF validation set, train = rest;
    ``cleansplit`` CleanSplit train/test, 10 % of the training complexes held out as val;
    ``random``     StratifiedGroupKFold(5) by complex on Luca's relevance bins, fold 0 = val;
                   there is no separate test set (Luca reported PDBbind on this fold).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from docking_analysis.constants import EXTRACT_DIR, SCFX_WEIGHTS_PATH, SPLITS_PATH

STATES = ("crystal", "relax", "apo_relax")
START_MODES = ("std", "perturb")
PROTOCOLS = [f"{s}_docking_{m}" for s in STATES for m in START_MODES]

DELTA_TERMS = ("fa_atr", "fa_rep", "fa_sol", "fa_elec", "fa_pair", "hbond_bb_sc", "hbond_sc")
TOTAL_TERMS = ("fa_atr", "fa_rep", "fa_sol", "fa_intra_rep", "fa_elec", "pro_close", "fa_pair",
               "hbond_sr_bb", "hbond_lr_bb", "hbond_bb_sc", "hbond_sc",
               "dslf_ss_dst", "dslf_cs_ang", "dslf_ss_dih", "dslf_ca_dih",
               "rama", "omega", "fa_dun", "p_aa_pp", "ref")
FEATURE_SETS: dict[str, list[str]] = {
    "delta": [f"raw_delta_{t}" for t in DELTA_TERMS],
    "total": [f"raw_{t}" for t in TOTAL_TERMS],
}
FEATURE_SETS["both"] = FEATURE_SETS["delta"] + FEATURE_SETS["total"]

BASELINE_SCORES = ("idelta_score", "total_score")          # Rosetta scores, lower is better
META_COLS = ["pdb", "protocol", "seq_id", "total_score", "idelta_score",
             "rmsd_to_crystal", "rmsd_to_input", "rmsd_to_crystal_legacy", "rmsd_to_input_legacy"]
SPLIT_SCHEMES = ("lppdbbind", "hac", "cleansplit", "random")
SPLIT_COLUMNS = {"lppdbbind": "lppdbbindsplit", "hac": "hacsplit", "cleansplit": "cleansplit"}
RELEVANCE_BINS = (-np.inf, 0.5, 1.0, 1.5, 2.0, np.inf)    # Luca's RMSD bins (relevance 4..0)
FA_REP_CLIP = 50.0                                        # Luca: delta fa_rep clipped at 50 before scaling
SCALE_CLIP = 10.0                                         # Luca: scaled features clipped to ±10


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def extraction_dir(set_name: str, extract_dir: Path | str | None = None) -> Path:
    return Path(extract_dir) if extract_dir else EXTRACT_DIR / f"{set_name}_full"


def load_poses(set_name: str, protocols: str | list[str], feature_cols: list[str],
               extract_dir: Path | str | None = None, complete_only: bool = True) -> pd.DataFrame:
    """Docked poses of one set for the given protocol(s): meta columns + features, sorted by
    (pdb, protocol, seq_id) so that every docking run is a contiguous block."""
    protocols = [protocols] if isinstance(protocols, str) else list(protocols)
    unknown = sorted(set(protocols) - set(PROTOCOLS))
    if unknown:
        raise ValueError(f"unknown protocol(s) {unknown}; choose from {PROTOCOLS}")
    d = extraction_dir(set_name, extract_dir)
    cols = META_COLS + [c for c in feature_cols if c not in META_COLS]
    df = pq.read_table(d / "docked.parquet", columns=cols,
                       filters=[("protocol", "in", protocols)]).to_pandas()
    if complete_only:
        cx = pd.read_parquet(d / "complexes.parquet", columns=["pdb", "complete"])
        df = df[df["pdb"].isin(cx.loc[cx["complete"].eq(True), "pdb"])]
    df = df.sort_values(["pdb", "protocol", "seq_id"], kind="mergesort").reset_index(drop=True)
    df.insert(0, "set", set_name)
    return df


def external_pdb_ids(sets: tuple[str, ...] = ("astex", "posebuster"),
                     extract_dir: Path | str | None = None) -> dict[str, set[str]]:
    """PDB entries (lower-case 4-character ids) of the external benchmark sets, per set."""
    out = {}
    for s in sets:
        cx = pd.read_parquet(extraction_dir(s, extract_dir) / "complexes.parquet", columns=["pdb"])
        out[s] = set(cx["pdb"].str.split("_").str[0].str.lower())
    return out


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------
def assign_split(poses: pd.DataFrame, scheme: str, seed: int = 0, rmsd_col: str = "rmsd_to_crystal",
                 val_frac: float = 0.1) -> pd.Series:
    """Map every complex of *poses* to ``train`` / ``val`` / ``test`` / ``drop`` (see module doc)."""
    if scheme not in SPLIT_SCHEMES:
        raise ValueError(f"unknown split scheme {scheme!r}; choose from {SPLIT_SCHEMES}")
    pdbs = pd.Index(pd.unique(poses["pdb"]))
    if scheme == "random":
        from sklearn.model_selection import StratifiedGroupKFold
        relevance = pd.cut(poses[rmsd_col], RELEVANCE_BINS, labels=[4, 3, 2, 1, 0]).astype(int).to_numpy()
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        _, val_idx = next(iter(sgkf.split(np.zeros((len(poses), 1)), relevance, groups=poses["pdb"].to_numpy())))
        val = set(poses["pdb"].iloc[val_idx])
        return pd.Series(["val" if p in val else "train" for p in pdbs], index=pdbs, name="split")

    ann = pd.read_csv(SPLITS_PATH, dtype=str).set_index("pdb")
    s = ann[SPLIT_COLUMNS[scheme]].reindex(pdbs).fillna("drop")
    s = s.where(s.isin(["train", "val", "test"]), "drop")
    if scheme == "cleansplit":                 # CleanSplit has no validation set: hold out val_frac of train
        rng = np.random.default_rng(seed)
        train_ids = s.index[s == "train"].to_numpy()
        held = rng.choice(train_ids, size=int(round(val_frac * len(train_ids))), replace=False)
        s.loc[held] = "val"
    s.name = "split"
    return s


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------
def clip_features(df: pd.DataFrame, feature_cols: list[str], fa_rep_clip: float | None = FA_REP_CLIP) -> pd.DataFrame:
    """Luca's pre-scaling clip: the interface-delta ``fa_rep`` is capped at *fa_rep_clip*
    (a clash of 60 and a clash of 5,000 are equally bad). Total-energy terms are not clipped."""
    if fa_rep_clip is not None and "raw_delta_fa_rep" in feature_cols:
        df = df.copy()
        df["raw_delta_fa_rep"] = df["raw_delta_fa_rep"].clip(upper=fa_rep_clip)
    return df


@dataclass
class RobustScale:
    """sklearn's RobustScaler (median / IQR) with the scaled values clipped to ±clip; stored as JSON."""
    names: list[str]
    center: np.ndarray
    scale: np.ndarray
    clip: float = SCALE_CLIP

    @classmethod
    def fit(cls, X: np.ndarray, names: list[str], clip: float = SCALE_CLIP) -> "RobustScale":
        q25, q50, q75 = np.nanpercentile(X, [25, 50, 75], axis=0)
        scale = (q75 - q25).astype(np.float64)
        scale[scale == 0] = 1.0                       # constant feature: keep it centred, unscaled
        return cls(list(names), q50.astype(np.float64), scale, clip)

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = (np.asarray(X, dtype=np.float64) - self.center) / self.scale
        return np.clip(Z, -self.clip, self.clip).astype(np.float32)

    def to_dict(self) -> dict:
        return {"names": self.names, "center": self.center.tolist(), "scale": self.scale.tolist(), "clip": self.clip}

    @classmethod
    def from_dict(cls, d: dict) -> "RobustScale":
        return cls(list(d["names"]), np.asarray(d["center"], float), np.asarray(d["scale"], float), float(d["clip"]))


def rosetta_weights(feature_cols: list[str]) -> np.ndarray:
    """Default score-function weight of every feature column (``scfx_weights.json``)."""
    w = json.loads(Path(SCFX_WEIGHTS_PATH).read_text())
    return np.array([w[c.replace("raw_delta_", "").replace("raw_", "")] for c in feature_cols], dtype=np.float64)


# ---------------------------------------------------------------------------
# grouped arrays
# ---------------------------------------------------------------------------
@dataclass
class Grouped:
    """Poses of one or more docking runs as arrays: ``frame`` (meta + raw features, sorted by
    (pdb, protocol, seq_id)), ``X`` (scaled features, float32), ``hit`` (label), ``offsets``
    (run boundaries: run g is rows ``offsets[g]:offsets[g+1]``)."""
    frame: pd.DataFrame
    X: np.ndarray
    hit: np.ndarray
    offsets: np.ndarray

    @classmethod
    def build(cls, frame: pd.DataFrame, X: np.ndarray, rmsd_col: str, cutoff: float) -> "Grouped":
        frame = frame.reset_index(drop=True)
        change = frame["pdb"].ne(frame["pdb"].shift()) | frame["protocol"].ne(frame["protocol"].shift())
        offsets = np.append(np.flatnonzero(change.to_numpy()), len(frame)).astype(np.int64)
        hit = frame[rmsd_col].to_numpy() <= cutoff
        return cls(frame, np.ascontiguousarray(X, dtype=np.float32), hit, offsets)

    @property
    def n_groups(self) -> int:
        return len(self.offsets) - 1

    @property
    def n_complexes(self) -> int:
        return int(self.frame["pdb"].nunique())

    def keys(self) -> pd.DataFrame:
        """One row per docking run: set, pdb, protocol."""
        return self.frame.iloc[self.offsets[:-1]][["set", "pdb", "protocol"]].reset_index(drop=True)

    def column(self, name: str) -> np.ndarray:
        return self.frame[name].to_numpy()

    def subset(self, mask: np.ndarray, rmsd_col: str, cutoff: float) -> "Grouped":
        """Rows selected by a boolean mask over the frame (whole runs, or the runs break)."""
        return Grouped.build(self.frame[mask], self.X[mask], rmsd_col, cutoff)
