"""
affinity.diagnostics
--------------------
Two diagnostics of the affinity result (added 2026-10-22 after the discussion of the headline
figures), written to ``<OUTPUT_DIR>/affinity/diagnostics.md``:

1. **Regression to the mean** on the headline cell: the spread of the predictions against the
   spread of the labels, the slope of the prediction on the label, and the mean prediction and
   MAE per label decile — why no complex is predicted below pK 4 and why the tails carry the MAE.
2. **The pose-energy distribution as a signal** (the hypothesis that strong binders populate
   their top basin more densely than weak ones): per docking run of the headline pool,
   statistics of the interface-delta distribution relative to its minimum — the gap from the best
   pose to the 10th and to the median pose, the number of poses within 2 and 5 REU of the
   minimum, the hit fraction, the log-sum-exp free energy F = −kT log Σ exp(−E/kT) at kT = 1 and
   3 REU and its entropy-like part F − E_min — with their Spearman correlation with pK and with
   the heavy-atom count, and the partial correlations after conditioning on the heavy-atom count
   and on the minimum energy; plus the medians of the same statistics for weak (pK < 5) and strong
   (pK > 8) binders within quintiles of the minimum energy (a matched comparison). Per-run
   statistics are saved as ``<AFFINITY_DIR>/shape_statistics.parquet``.

Usage::

    uv run python -m affinity.diagnostics [--pool apo_relax_docking_perturb] [--split lppdbbind]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import pearsonr, spearmanr

from docking_analysis.constants import AFFINITY_DIR, OUTPUT_DIR

from .data import HEADLINE_AGG, HEADLINE_POOL, HIT_CUTOFF, LABEL_SETS, RUN_POOLS, complex_info, extraction_dir, load_labels
from .evaluate import frame_markdown

KEYS = ("constant", "unmodified", "reweight", "size", "reweight_size")
SHAPE_COLS = ["e_min", "e_10", "e_med", "gap10", "gap_med", "n_within2", "n_within5", "hit_frac", "F_kT1", "F_kT3", "S_kT1", "S_kT3", "n_heavy"]


# ---------------------------------------------------------------------------
# 1. regression to the mean
# ---------------------------------------------------------------------------
def decile_table(pred: pd.DataFrame, label_set: str = "kdki", split: str = "lppdbbind", pool: str = HEADLINE_POOL,
                 agg: str = HEADLINE_AGG, loss: str = "l1", keys=KEYS) -> pd.DataFrame:
    t = pred[(pred["label_set"] == label_set) & (pred["split"] == split) & (pred["pool"] == pool) & (pred["aggregation"] == agg)
             & (pred["eval_set"] == "test") & (pred["loss"] == loss)]
    y = t["pk"].to_numpy()
    dec = pd.qcut(y, 10, labels=False)
    q10, q90 = np.percentile(y, 10), np.percentile(y, 90)
    rows = []
    for k in keys:
        yh = t[f"pred_{k}"].to_numpy()
        err = np.abs(yh - y)
        const = np.ptp(yh) == 0
        rows.append({"predictor": k, "sd_label": y.std(), "sd_pred": yh.std(), "sd_ratio": yh.std() / y.std(),
                     "r": np.nan if const else pearsonr(yh, y).statistic, "slope_pred_on_label": np.nan if const else np.polyfit(y, yh, 1)[0],
                     "mean_pred_lowest_decile": yh[dec == 0].mean(), "mean_pred_highest_decile": yh[dec == 9].mean(),
                     "mae_lowest_decile": err[dec == 0].mean(), "mae_middle_deciles": err[(dec >= 3) & (dec <= 6)].mean(),
                     "mae_highest_decile": err[dec == 9].mean(), "n": len(y), "label_q10": q10, "label_q90": q90})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. the pose-energy distribution
# ---------------------------------------------------------------------------
def shape_statistics(pool: str = HEADLINE_POOL, label_set: str = "kdki", extract_dir=None, cutoff: float = HIT_CUTOFF) -> pd.DataFrame:
    """One row per labelled complete complex: the distribution statistics of the pool's
    interface deltas relative to the run's minimum, the label and the heavy-atom count."""
    labels = load_labels()
    pk = labels[labels["exact"] & labels["kind"].isin(LABEL_SETS[label_set])].set_index("pdb")["pk"]
    info = complex_info("pdbbind", extract_dir).set_index("pdb")["ligand_n_heavy"]
    d = extraction_dir("pdbbind", extract_dir)
    df = pq.read_table(d / "docked.parquet", columns=["pdb", "idelta_score", "rmsd_to_crystal"],
                       filters=[("protocol", "in", RUN_POOLS[pool])]).to_pandas()
    df = df[df["pdb"].isin(pk.index) & df["pdb"].isin(info.index)].sort_values(["pdb", "idelta_score"], kind="mergesort")
    e = df["idelta_score"]
    rel = e - df.groupby("pdb")["idelta_score"].transform("min")
    df = df.assign(w2=rel <= 2, w5=rel <= 5, hit=df["rmsd_to_crystal"] <= cutoff, b1=np.exp(-rel), b3=np.exp(-rel / 3),
                   rank=df.groupby("pdb").cumcount())
    g = df.groupby("pdb")
    s = pd.DataFrame({"e_min": g["idelta_score"].min(), "e_med": g["idelta_score"].median(), "n_poses": g.size(),
                      "n_within2": g["w2"].sum(), "n_within5": g["w5"].sum(), "hit_frac": g["hit"].mean(),
                      "sum_b1": g["b1"].sum(), "sum_b3": g["b3"].sum()})
    s["e_10"] = df[df["rank"] == 9].set_index("pdb")["idelta_score"].reindex(s.index)
    s["gap10"] = s["e_10"] - s["e_min"]
    s["gap_med"] = s["e_med"] - s["e_min"]
    s["F_kT1"] = s["e_min"] - np.log(s["sum_b1"])
    s["F_kT3"] = s["e_min"] - 3 * np.log(s["sum_b3"])
    s["S_kT1"] = s["F_kT1"] - s["e_min"]
    s["S_kT3"] = s["F_kT3"] - s["e_min"]
    s["pk"] = pk.reindex(s.index)
    s["n_heavy"] = info.reindex(s.index)
    return s.drop(columns=["sum_b1", "sum_b3"]).dropna().reset_index()


def _partial_spearman(a: np.ndarray, b: np.ndarray, controls: np.ndarray) -> float:
    X = np.column_stack([np.ones(len(a)), controls])
    ra = a - X @ np.linalg.lstsq(X, a, rcond=None)[0]
    rb = b - X @ np.linalg.lstsq(X, b, rcond=None)[0]
    return float(spearmanr(ra, rb).statistic)


def shape_correlations(s: pd.DataFrame) -> pd.DataFrame:
    y, h, e = s["pk"].to_numpy(float), s["n_heavy"].to_numpy(float), s["e_min"].to_numpy(float)
    rows = []
    for c in SHAPE_COLS:
        a = s[c].to_numpy(float)
        rows.append({"statistic": c, "rho_pk": spearmanr(a, y).statistic, "rho_heavy_atoms": spearmanr(a, h).statistic,
                     "partial_rho_pk_given_heavy": _partial_spearman(a, y, h[:, None]),
                     "partial_rho_pk_given_heavy_and_e_min": _partial_spearman(a, y, np.column_stack([h, e]))})
    return pd.DataFrame(rows)


def matched_table(s: pd.DataFrame, weak: float = 5.0, strong: float = 8.0, n_bins: int = 5) -> pd.DataFrame:
    """Medians of the shape statistics for weak (pK < weak) and strong (pK > strong) binders
    within quantile bins of the minimum energy."""
    q = pd.qcut(s["e_min"], n_bins, labels=False)
    rows = []
    for i in range(n_bins):
        b = s[q == i]
        w, st = b[b["pk"] < weak], b[b["pk"] > strong]
        rows.append({"e_min_bin": f"{b['e_min'].min():.0f} … {b['e_min'].max():.0f} REU", "n_weak": len(w), "n_strong": len(st),
                     "gap10_weak": w["gap10"].median(), "gap10_strong": st["gap10"].median(),
                     "n_within2_weak": w["n_within2"].median(), "n_within2_strong": st["n_within2"].median(),
                     "hit_frac_weak": w["hit_frac"].median(), "hit_frac_strong": st["hit_frac"].median(),
                     "S_kT3_weak": w["S_kT3"].median(), "S_kT3_strong": st["S_kT3"].median()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", type=Path, default=AFFINITY_DIR)
    ap.add_argument("--out", type=Path, default=OUTPUT_DIR / "affinity")
    ap.add_argument("--pool", default=HEADLINE_POOL, choices=list(RUN_POOLS))
    ap.add_argument("--split", default="lppdbbind")
    ap.add_argument("--label-set", default="kdki")
    ap.add_argument("--loss", default="l1")
    ap.add_argument("--extract-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    pred = pd.read_parquet(args.in_dir / "predictions.parquet")
    dt = decile_table(pred, args.label_set, args.split, args.pool, HEADLINE_AGG, args.loss)
    s = shape_statistics(args.pool, args.label_set, args.extract_dir)
    s.to_parquet(args.in_dir / "shape_statistics.parquet", index=False)
    corr = shape_correlations(s)
    matched = matched_table(s)
    md = ["# Affinity diagnostics (`affinity.diagnostics`)", "",
          f"## 1. Regression to the mean — {args.label_set}, `{args.pool}`, best pose, {args.split} test complexes, {args.loss} fits", "",
          "`sd_ratio` = sd(prediction) / sd(label); `slope_pred_on_label` the least-squares slope of the prediction on the label "
          "(≈ r² for a linear predictor); deciles of the experimental pK (lowest decile below `label_q10`, highest above `label_q90`).", "",
          frame_markdown(dt, ".2f"), "",
          f"## 2. The pose-energy distribution as a signal — `{args.pool}`, {len(s):,} labelled complete complexes, no split", "",
          "Per run: `e_min` the best interface delta, `e_10` / `e_med` the 10th-best / median pose, `gap10` / `gap_med` their distance to the "
          "minimum, `n_within2` / `n_within5` poses within 2 / 5 REU of the minimum, `hit_frac` the share of poses within 2 Å of the "
          "crystal ligand, `F_kT1` / `F_kT3` the log-sum-exp free energy at kT = 1 / 3 REU, `S_kT1` / `S_kT3` = F − e_min (≤ 0; closer "
          "to 0 = fewer poses near the minimum). Spearman ρ with pK, with the heavy-atom count, and partial ρ with pK after regressing "
          "both on the heavy-atom count, and on the heavy-atom count and `e_min`.", "",
          frame_markdown(corr, ".3f"), "",
          "Medians for weak (pK < 5) and strong (pK > 8) binders within quintiles of `e_min`:", "",
          frame_markdown(matched, ".2f"), ""]
    (args.out / "diagnostics.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"-> {args.out / 'diagnostics.md'}; per-run statistics -> {args.in_dir / 'shape_statistics.parquet'}")


if __name__ == "__main__":
    main()
