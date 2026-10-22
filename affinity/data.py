"""
affinity.data
-------------
Labels, per-complex pose aggregation and the split of the affinity question (Track 3,
inventory S7), read from the full extraction (``<EXTRACT_DIR>/pdbbind_full/``).

Labels (:func:`load_labels`)
    the binding-data column of ``INDEX_general_PL.2020R1.lst`` parsed into the measure
    (``Kd``, ``Ki``, ``IC50``), the relation (``=``, ``<``, ``<=``, ``>``, ``>=``, ``~``), the
    value in mol/l and pK = −log10(value / M). Only **exact** values (relation ``=``) are used.
    Label sets (``LABEL_SETS``): ``kdki`` — Kd and Ki pooled, the headline; ``ic50`` — reported
    separately (assay-dependent); ``kd`` / ``ki`` — the two halves of the headline test set.
Pools (``RUN_POOLS``, ``SEED_POOLS``)
    which poses of a complex are aggregated into its feature vector: one docking run (complex ×
    protocol, 300 poses), the perturb + std runs of one receptor state (600 poses: the "with
    Standard poses" ablation), or one reference structure from ``seeds.parquet`` — ``crystal``
    (the input complex as prepared, unrelaxed) and ``relax_ref`` (the relaxed holo complex the
    docking started from, i.e. the native pose after FastRelax).
Aggregations (``AGGREGATIONS``)
    ``best``      the pose with the lowest interface delta — the pose a user would take;
    ``mean10``    the mean of the terms over the 10 lowest-interface-delta poses;
    ``mean``      the mean over the poses of the pool with a non-positive interface delta (the
                  legacy grid's "filtered mean"; the plain mean is dominated by clashing poses);
    ``best_hit``  oracle: the lowest-interface-delta pose among the hits (``rmsd_to_crystal`` ≤
                  2 Å) — what the score would give if the pose were always right; complexes
                  without a hit are absent.
Features
    the seven raw interface-delta terms of the aggregated pose (``FEATURE_COLS`` of
    ``ranking.data``; ``idelta_score`` is their weighted sum with the default weights) and
    ``ligand_n_heavy`` from ``complexes.parquet`` as the size reference. ``n_agg`` is the number
    of poses the aggregation used. For the seed pools the interface delta is **recomputed** from
    the seven terms with the default weights: the stored ``idelta_score`` of a few relax seeds is
    corrupt (values down to −4.6e8 with sane terms, e.g. 2f5t, 3cr5), whereas for the docked
    poses the stored value equals the weighted sum to 1e-4 (``ranking.md`` §1).
Split
    ``ranking.data.assign_split``: the three non-random PDBbind splits of S5 (LP-PDBBind, HAC with
    the CASF-2016 core set as test, CleanSplit). The models have no hyper-parameters, so the
    validation complexes join the training complexes. The CASF-2016 core complexes outside a
    split's training set are the external test of that split (:func:`casf_core_ids`).
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pandas as pd

from docking_analysis.constants import AFFINITY_DIR, INDEX_PATH, SPLITS_PATH
from ranking.data import (FEATURE_COLS, PROTOCOLS, SPLIT_SCHEMES, assign_split, extraction_dir, load_poses,  # noqa: F401
                          rosetta_weights, term_name)

# ---------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------
KINDS = ("Kd", "Ki", "IC50")
LABEL_SETS: dict[str, tuple[str, ...]] = {"kdki": ("Kd", "Ki"), "kd": ("Kd",), "ki": ("Ki",), "ic50": ("IC50",)}
TRAIN_LABEL_SETS = ("kdki", "ic50")
HEADLINE_LABELS = "kdki"
LABEL_RE = re.compile(r"^(?P<kind>Kd|Ki|IC50)(?P<relation><=|>=|=|<|>|~)(?P<value>[0-9]*\.?[0-9]+)(?P<unit>[munpf]?M)$")
UNIT_TO_MOLAR = {"M": 1.0, "mM": 1e-3, "uM": 1e-6, "nM": 1e-9, "pM": 1e-12, "fM": 1e-15}


def load_labels(index_path: Path | str = INDEX_PATH) -> pd.DataFrame:
    """Every entry of the PDBbind index: ``pdb``, ``kind``, ``relation``, ``value_M``, ``pk``,
    ``exact`` (relation ``=``), the raw ``binding`` field, ``resolution``, ``year``."""
    rows = []
    with open(index_path) as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            m = LABEL_RE.match(parts[3])
            if m is None:
                raise ValueError(f"unparsed binding field {parts[3]!r} for {parts[0]}")
            value = float(m["value"]) * UNIT_TO_MOLAR[m["unit"]]
            rows.append({"pdb": parts[0].lower(), "kind": m["kind"], "relation": m["relation"], "value_M": value,
                         "pk": -math.log10(value), "exact": m["relation"] == "=", "binding": parts[3],
                         "resolution": parts[1], "year": int(parts[2])})
    df = pd.DataFrame(rows)
    if not df["pdb"].is_unique:
        raise ValueError("duplicate PDB ids in the index")
    return df


def casf_core_ids() -> set[str]:
    """The CASF-2016 core set (= the HAC test set of ``splits.csv``)."""
    ann = pd.read_csv(SPLITS_PATH, dtype=str)
    return set(ann.loc[ann["hacsplit"].eq("test"), "pdb"])


# ---------------------------------------------------------------------------
# pools and aggregations
# ---------------------------------------------------------------------------
RUN_POOLS: dict[str, list[str]] = {p: [p] for p in PROTOCOLS}
for _s in ("relax", "apo_relax"):
    RUN_POOLS[f"{_s}_docking_perturb+std"] = [f"{_s}_docking_perturb", f"{_s}_docking_std"]
SEED_POOLS: dict[str, str] = {"crystal": "crystal", "relax_ref": "relax"}       # pool -> receptor state in seeds.parquet
POOLS: tuple[str, ...] = tuple(RUN_POOLS) + tuple(SEED_POOLS)
TRAIN_POOLS = ("apo_relax_docking_perturb", "relax_docking_perturb",
               "apo_relax_docking_perturb+std", "relax_docking_perturb+std", "relax_ref", "crystal")
HEADLINE_POOL = "apo_relax_docking_perturb"
AGGREGATIONS = ("best", "mean10", "mean", "best_hit")
HEADLINE_AGG = "best"
HIT_CUTOFF = 2.0
VALUE_COLS = ["idelta_score", "rmsd_to_crystal"] + FEATURE_COLS
FEATURES_PATH = AFFINITY_DIR / "features.parquet"


def pool_label(pool: str) -> str:
    if pool == "crystal":
        return "crystal complex (input, unrelaxed)"
    if pool == "relax_ref":
        return "relaxed holo complex (docking reference)"
    state, rest = pool.split("_docking_")
    return f"{state} · {rest.replace('+std', ' + std')}"


def aggregate(poses: pd.DataFrame, aggregation: str, cutoff: float = HIT_CUTOFF) -> pd.DataFrame:
    """One row per complex of *poses* (one pool, possibly several protocols of one receptor
    state): the aggregated ``VALUE_COLS``, ``n_poses`` and ``n_hits`` of the pool."""
    hit = poses["rmsd_to_crystal"] <= cutoff
    g = poses.groupby("pdb", sort=False)
    counts = pd.DataFrame({"n_poses": g.size(), "n_hits": hit.groupby(poses["pdb"], sort=False).sum().astype(int)})
    if aggregation == "best":
        out = poses.loc[g["idelta_score"].idxmin(), ["pdb"] + VALUE_COLS].set_index("pdb")
        out["n_agg"] = 1
    elif aggregation == "best_hit":
        h = poses[hit]
        out = h.loc[h.groupby("pdb", sort=False)["idelta_score"].idxmin(), ["pdb"] + VALUE_COLS].set_index("pdb")
        out["n_agg"] = 1
    elif aggregation == "mean10":
        s = poses.sort_values(["pdb", "idelta_score"], kind="mergesort")
        top = s.groupby("pdb", sort=False).head(10).groupby("pdb", sort=False)
        out = top[VALUE_COLS].mean()
        out["n_agg"] = top.size()
    elif aggregation == "mean":
        neg = poses[poses["idelta_score"] <= 0].groupby("pdb", sort=False)
        out = neg[VALUE_COLS].mean()
        out["n_agg"] = neg.size()
    else:
        raise ValueError(f"unknown aggregation {aggregation!r}; choose from {AGGREGATIONS}")
    out = out.join(counts, how="left")
    out.insert(0, "aggregation", aggregation)
    return out.reset_index()


def complex_info(set_name: str = "pdbbind", extract_dir: Path | str | None = None) -> pd.DataFrame:
    """The complete complexes of a set: ``pdb``, ``ligand_n_heavy``, ``ligand_n_automorphisms``."""
    cx = pd.read_parquet(extraction_dir(set_name, extract_dir) / "complexes.parquet",
                         columns=["pdb", "complete", "ligand_n_heavy", "ligand_n_automorphisms"])
    return cx[cx["complete"].eq(True)].drop(columns="complete").reset_index(drop=True)


def seed_rows(set_name: str, pool: str, extract_dir: Path | str | None = None, cutoff: float = HIT_CUTOFF) -> pd.DataFrame:
    """The reference structure of a ``SEED_POOLS`` pool as a one-pose aggregation (``best``)."""
    d = extraction_dir(set_name, extract_dir)
    state = SEED_POOLS[pool]
    df = pd.read_parquet(d / "seeds.parquet", columns=["pdb", "state", "is_docking_reference"] + VALUE_COLS)
    df = df[df["state"].eq(state)]
    if state != "crystal":
        df = df[df["is_docking_reference"].eq(True)]
    df = df[df["pdb"].isin(complex_info(set_name, extract_dir)["pdb"])]
    if not df["pdb"].is_unique:
        raise ValueError(f"several {state} reference rows per complex")
    out = df[["pdb"] + VALUE_COLS].reset_index(drop=True)
    recomputed = out[FEATURE_COLS].to_numpy() @ rosetta_weights()          # the stored seed idelta is corrupt for a few rows
    out["idelta_score_stored"] = out["idelta_score"]
    out["idelta_score"] = recomputed
    out["n_poses"] = 1
    out["n_agg"] = 1
    out["n_hits"] = (out["rmsd_to_crystal"] <= cutoff).astype(int)
    out.insert(1, "aggregation", "best")
    return out


def build_features(set_name: str = "pdbbind", extract_dir: Path | str | None = None, cutoff: float = HIT_CUTOFF,
                   pools: tuple[str, ...] = POOLS, log=print) -> pd.DataFrame:
    """The long feature table: one row per complete complex × pool × aggregation with the
    aggregated ``VALUE_COLS``, ``n_poses`` and ``n_hits``."""
    parts = []
    for pool in pools:
        if pool in SEED_POOLS:
            t = seed_rows(set_name, pool, extract_dir, cutoff)
            bad = int((abs(t["idelta_score_stored"] - t["idelta_score"]) > 1e-2).sum())
            log(f"features: pool {pool}: stored interface delta recomputed from the terms; {bad} rows differed by > 0.01")
            t = t.drop(columns="idelta_score_stored")
        else:
            poses = load_poses(set_name, RUN_POOLS[pool], extract_dir)
            t = pd.concat([aggregate(poses, a, cutoff) for a in AGGREGATIONS], ignore_index=True)
            del poses
        t.insert(1, "pool", pool)
        parts.append(t)
        log(f"features: pool {pool}: {t['pdb'].nunique():,} complexes, {len(t):,} rows")
    return pd.concat(parts, ignore_index=True)


def load_features(path: Path | str = FEATURES_PATH, rebuild: bool = False, log=print, **kwargs) -> pd.DataFrame:
    """The cached feature table (built with :func:`build_features` when missing or ``rebuild``)."""
    path = Path(path)
    if path.exists() and not rebuild:
        return pd.read_parquet(path)
    feats = build_features(log=log, **kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(path, index=False)
    log(f"features written -> {path} ({len(feats):,} rows)")
    return feats
