"""
ranking.data
------------
Pose tables for the ranking question (Track 2): features, labels, groups and the PDBbind
split, read from the full extraction (``<EXTRACT_DIR>/<set>_full/docked.parquet``).

Fixed by the decisions of 2026-10-20 (see ``docs/ranking.md``):

Features (``FEATURE_COLS``)
    the seven non-zero interface-delta terms ``raw_delta_{fa_atr, fa_rep, fa_sol, fa_elec,
    fa_pair, hbond_bb_sc, hbond_sc}`` — raw Rosetta energies; ``idelta_score`` is their
    weighted sum with the default weights. Nothing else is used anywhere in this track.
Training pool (``TRAIN_PROTOCOLS``)
    ``relax_docking_perturb`` + ``apo_relax_docking_perturb`` (not the crystal state, not the
    std runs). The headline evaluation is ``HEADLINE_PROTOCOL`` = ``apo_relax_docking_perturb``.
Labels
    a pose is a hit when ``rmsd_col`` (default the symmetry-corrected ``rmsd_to_crystal``) is
    <= ``cutoff`` (2 Å).
Groups
    for **training**, one bucket = (complex, receptor state): every pose of a complex that was
    docked from the same starting structure, i.e. the perturb run alone with the default pool,
    the perturb + std runs (600 poses) if the std protocol is pooled in; relax and apo_relax
    buckets of a complex stay separate. For **evaluation**, one group = one docking run
    (complex, protocol), as in the benchmark. Poses are never compared across complexes.
PDBbind splits (:func:`assign_split`; ``docking_analysis/splits.csv``)
    ``lppdbbind``  LP-PDBBind train / val / test, unassigned dropped;
    ``hac``        test = CASF-2016 core set, val = CASF validation set, train = rest;
    ``cleansplit`` CleanSplit train / test, 10 % of the training complexes held out as val.
    PDBbind entries that are also Astex complexes (43 of 84) are removed from train and val.
Preprocessing (:class:`Preprocess`)
    ``none``   raw energies (the reweighter and the tree);
    ``robust`` Luca's preprocessing for the MLP ranker: delta ``fa_rep`` capped at 50, median /
               IQR scaling fitted on the training poses, scaled values clipped to ±10.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from docking_analysis.constants import EXTRACT_DIR, SCFX_WEIGHTS_PATH, SPLITS_PATH

STATES = ("crystal", "relax", "apo_relax")
START_MODES = ("std", "perturb")
PROTOCOLS = [f"{s}_docking_{m}" for s in STATES for m in START_MODES]
TRAIN_PROTOCOLS = ("relax_docking_perturb", "apo_relax_docking_perturb")
HEADLINE_PROTOCOL = "apo_relax_docking_perturb"

DELTA_TERMS = ("fa_atr", "fa_rep", "fa_sol", "fa_elec", "fa_pair", "hbond_bb_sc", "hbond_sc")
FEATURE_COLS = [f"raw_delta_{t}" for t in DELTA_TERMS]
BASELINE_SCORE = "idelta_score"                            # the unmodified Rosetta ranking, lower is better
META_COLS = ["pdb", "protocol", "receptor_state", "seq_id", "total_score", "idelta_score",
             "rmsd_to_crystal", "rmsd_to_input", "rmsd_to_crystal_legacy", "rmsd_to_input_legacy"]
SPLIT_SCHEMES = ("lppdbbind", "hac", "cleansplit")
SPLIT_COLUMNS = {"lppdbbind": "lppdbbindsplit", "hac": "hacsplit", "cleansplit": "cleansplit"}
FA_REP_CLIP = 50.0                                         # Luca: delta fa_rep capped at 50 before scaling
SCALE_CLIP = 10.0                                          # Luca: scaled features clipped to ±10


def term_name(col: str) -> str:
    return col.replace("raw_delta_", "").replace("raw_", "")


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def extraction_dir(set_name: str, extract_dir: Path | str | None = None) -> Path:
    return Path(extract_dir) / f"{set_name}_full" if extract_dir else EXTRACT_DIR / f"{set_name}_full"


def load_poses(set_name: str, protocols: str | list[str] | tuple[str, ...], extract_dir: Path | str | None = None,
               complete_only: bool = True) -> pd.DataFrame:
    """Docked poses of one set for the given protocol(s): meta columns + the seven features,
    sorted by (pdb, protocol, seq_id) so that every docking run is a contiguous block."""
    protocols = [protocols] if isinstance(protocols, str) else list(protocols)
    unknown = sorted(set(protocols) - set(PROTOCOLS))
    if unknown:
        raise ValueError(f"unknown protocol(s) {unknown}; choose from {PROTOCOLS}")
    d = extraction_dir(set_name, extract_dir)
    df = pq.read_table(d / "docked.parquet", columns=META_COLS + FEATURE_COLS,
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
def assign_split(pdbs, scheme: str, seed: int = 0, val_frac: float = 0.1) -> pd.Series:
    """Map every complex id to ``train`` / ``val`` / ``test`` / ``drop`` (see module doc)."""
    if scheme not in SPLIT_SCHEMES:
        raise ValueError(f"unknown split scheme {scheme!r}; choose from {SPLIT_SCHEMES}")
    pdbs = pd.Index(pd.unique(pd.Series(pdbs)))
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
def rosetta_weights(feature_cols: list[str] = FEATURE_COLS) -> np.ndarray:
    """Default score-function weight of every feature column (``scfx_weights.json``)."""
    w = json.loads(Path(SCFX_WEIGHTS_PATH).read_text())
    return np.array([w[term_name(c)] for c in feature_cols], dtype=np.float64)


@dataclass
class RobustScale:
    """sklearn's RobustScaler (median / IQR) with the scaled values clipped to ±clip."""
    names: list[str]
    center: np.ndarray
    scale: np.ndarray
    clip: float = SCALE_CLIP

    @classmethod
    def fit(cls, X: np.ndarray, names: list[str], clip: float = SCALE_CLIP) -> "RobustScale":
        q25, q50, q75 = np.nanpercentile(X, [25, 50, 75], axis=0)
        scale = (q75 - q25).astype(np.float64)
        scale[scale == 0] = 1.0
        return cls(list(names), q50.astype(np.float64), scale, float(clip))

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = (np.asarray(X, dtype=np.float64) - self.center) / self.scale
        return np.clip(Z, -self.clip, self.clip).astype(np.float32)

    def to_dict(self) -> dict:
        return {"names": self.names, "center": self.center.tolist(), "scale": self.scale.tolist(), "clip": self.clip}

    @classmethod
    def from_dict(cls, d: dict) -> "RobustScale":
        return cls(list(d["names"]), np.asarray(d["center"], float), np.asarray(d["scale"], float), float(d["clip"]))


@dataclass
class Preprocess:
    """``none``: raw energies as float32. ``robust``: Luca's preprocessing (fa_rep cap, median/IQR
    scaling fitted on the training poses, ±clip)."""
    scaling: str = "none"
    fa_rep_clip: float | None = None
    scale_clip: float = SCALE_CLIP
    names: list[str] = field(default_factory=lambda: list(FEATURE_COLS))
    scaler: RobustScale | None = None

    def _clip(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64).copy()
        if self.fa_rep_clip is not None and "raw_delta_fa_rep" in self.names:
            j = self.names.index("raw_delta_fa_rep")
            X[:, j] = np.minimum(X[:, j], self.fa_rep_clip)
        return X

    def fit(self, X: np.ndarray) -> "Preprocess":
        if self.scaling == "robust":
            self.scaler = RobustScale.fit(self._clip(X), self.names, self.scale_clip)
        elif self.scaling != "none":
            raise ValueError(f"unknown scaling {self.scaling!r}")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self._clip(X)
        return self.scaler.transform(X) if self.scaler is not None else X.astype(np.float32)

    def to_dict(self) -> dict:
        return {"scaling": self.scaling, "fa_rep_clip": self.fa_rep_clip, "scale_clip": self.scale_clip,
                "names": self.names, "scaler": self.scaler.to_dict() if self.scaler else None}

    @classmethod
    def from_dict(cls, d: dict) -> "Preprocess":
        p = cls(d["scaling"], d["fa_rep_clip"], float(d["scale_clip"]), list(d["names"]))
        p.scaler = RobustScale.from_dict(d["scaler"]) if d.get("scaler") else None
        return p


# ---------------------------------------------------------------------------
# grouped arrays
# ---------------------------------------------------------------------------
@dataclass
class Grouped:
    """Poses of one or more docking runs as arrays: ``frame`` (meta + raw features, sorted by
    (pdb, protocol, seq_id)), ``X`` (model inputs, float32), ``hit`` (label), ``offsets``
    (run boundaries: run g is rows ``offsets[g]:offsets[g+1]``)."""
    frame: pd.DataFrame
    X: np.ndarray
    hit: np.ndarray
    offsets: np.ndarray

    @classmethod
    def build(cls, frame: pd.DataFrame, X: np.ndarray, rmsd_col: str, cutoff: float, group_by: str = "protocol") -> "Grouped":
        """``group_by="protocol"``: one group per docking run (evaluation); ``"state"``: one group
        per complex × receptor state (training buckets). The frame is sorted by (pdb, protocol),
        and the protocols of one state are adjacent, so both groupings are contiguous blocks."""
        frame = frame.reset_index(drop=True)
        key = {"protocol": "protocol", "state": "receptor_state"}[group_by]
        change = frame["pdb"].ne(frame["pdb"].shift()) | frame[key].ne(frame[key].shift())
        offsets = np.append(np.flatnonzero(change.to_numpy()), len(frame)).astype(np.int64)
        hit = frame[rmsd_col].to_numpy() <= cutoff
        return cls(frame, np.ascontiguousarray(X, dtype=np.float32), hit, offsets)

    @property
    def n_groups(self) -> int:
        return len(self.offsets) - 1

    @property
    def n_complexes(self) -> int:
        return int(self.frame["pdb"].nunique())

    @property
    def group_sizes(self) -> np.ndarray:
        return np.diff(self.offsets)

    def keys(self) -> pd.DataFrame:
        """One row per group: set, pdb, receptor_state, protocol (of the group's first pose)."""
        return self.frame.iloc[self.offsets[:-1]][["set", "pdb", "receptor_state", "protocol"]].reset_index(drop=True)

    def column(self, name: str) -> np.ndarray:
        return self.frame[name].to_numpy()
