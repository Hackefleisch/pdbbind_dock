"""
ranking.evaluate
----------------
Success@k evaluation of pose scorers on grouped pose tables (:class:`ranking.data.Grouped`).

The definition is the one of ``docking_analysis.benchmark``: within a docking run the poses are
sorted by the score (lower is better, stable sort), and the run is a Success@k when a hit
(``rmsd_col`` <= cutoff) is among the k best-ranked poses, i.e. when the rank of the first hit
is < k. Rates are percentages of docking runs. Luca's ``Success@1`` / ``Success@5`` are exactly
this (his scores are higher-is-better; :func:`evaluate_scorers` takes care of the sign).

Added for the paper: percentile-bootstrap confidence intervals over runs, the paired bootstrap
of the difference to the Rosetta interface delta on the same runs, the pooled ROC AUC (Luca's
``roc_auc``), the mean per-run AUC (the quantity the per-complex oracle of inventory
Appendix A.2 bounds) and the whole-distribution comparison of the first-hit rank
(:func:`rank_shift`: MRR, share of runs whose first hit moves earlier / later, log-rank shift).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from .data import Grouped

KS = (1, 5, 10)


# ---------------------------------------------------------------------------
# per docking run
# ---------------------------------------------------------------------------
def per_group_success(score_lower_better: np.ndarray, hit: np.ndarray, offsets: np.ndarray,
                      ks: tuple[int, ...] = KS) -> pd.DataFrame:
    """One row per run: n, n_hit, first_hit_rank (0-based, NaN without hit), top{k} flags, auc."""
    score = np.asarray(score_lower_better, dtype=np.float64)
    hit = np.asarray(hit, dtype=bool)
    n_groups = len(offsets) - 1
    n_hit = np.zeros(n_groups, dtype=np.int32)
    first = np.full(n_groups, np.nan)
    auc = np.full(n_groups, np.nan)
    for g in range(n_groups):
        a, b = offsets[g], offsets[g + 1]
        h = hit[a:b]
        n_hit[g] = h.sum()
        if n_hit[g] == 0:
            continue
        order = np.argsort(score[a:b], kind="stable")
        first[g] = int(np.argmax(h[order]))
        n_neg = (b - a) - n_hit[g]
        if n_neg > 0:                                   # AUC of "higher = better" within the run
            r = rankdata(-score[a:b])
            auc[g] = (r[h].sum() - n_hit[g] * (n_hit[g] + 1) / 2) / (n_hit[g] * n_neg)
    out = pd.DataFrame({"n": np.diff(offsets), "n_hit": n_hit, "first_hit_rank": first, "auc": auc})
    for k in ks:
        out[f"top{k}"] = first < k
    return out


# ---------------------------------------------------------------------------
# rates and intervals
# ---------------------------------------------------------------------------
def bootstrap_mean(flags: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Percent of True with a percentile-bootstrap 95 % interval over runs."""
    f = np.asarray(flags, dtype=np.float64)
    if len(f) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(f), size=(n_boot, len(f)))
    means = f[idx].mean(axis=1)
    return (100 * f.mean(), 100 * np.percentile(means, 2.5), 100 * np.percentile(means, 97.5))


def paired_bootstrap(flags: np.ndarray, reference: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Difference (points) between two scorers on the same runs, with its bootstrap 95 % interval."""
    d = np.asarray(flags, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    if len(d) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return (100 * d.mean(), 100 * np.percentile(means, 2.5), 100 * np.percentile(means, 97.5))


def quick_rates(score_higher_better: np.ndarray, grouped: Grouped, ks: tuple[int, ...] = (1, 5)) -> dict[str, float]:
    """Success@k in percent plus the pooled AUC — the validation metrics of the training loop."""
    pc = per_group_success(-np.asarray(score_higher_better), grouped.hit, grouped.offsets, ks)
    out = {f"top{k}": 100 * pc[f"top{k}"].mean() for k in ks}
    out["auc_pooled"] = pooled_auc(grouped.hit, score_higher_better)
    return out


def pooled_auc(hit: np.ndarray, score_higher_better: np.ndarray) -> float:
    hit = np.asarray(hit, dtype=bool)
    if hit.all() or not hit.any():
        return float("nan")
    return float(roc_auc_score(hit, score_higher_better))


# ---------------------------------------------------------------------------
# several scorers on one table
# ---------------------------------------------------------------------------
def evaluate_scorers(grouped: Grouped, scorers: dict[str, np.ndarray], reference: str = "idelta_score",
                     ks: tuple[int, ...] = KS, n_boot: int = 2000, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate ``scorers`` (name -> per-pose score, **higher is better**) on *grouped*.

    Returns ``(summary, per_run)``: one summary row per scorer (rates in percent with bootstrap
    intervals, paired difference of every Success@k to *reference*, AUCs) and the long per-run
    table (set, pdb, protocol, scorer, n_hit, first_hit_rank, top{k}, auc)."""
    keys = grouped.keys()
    per = {}
    for name, s in scorers.items():
        pc = per_group_success(-np.asarray(s, dtype=np.float64), grouped.hit, grouped.offsets, ks)
        per[name] = pc
    ref = per.get(reference)
    rows = []
    for name, pc in per.items():
        row = {"scorer": name, "runs": len(pc), "complexes": int(keys["pdb"].nunique()),
               "any_hit": 100 * (pc["n_hit"] > 0).mean()}
        for k in ks:
            m, lo, hi = bootstrap_mean(pc[f"top{k}"].to_numpy(), n_boot, seed)
            row[f"top{k}"], row[f"top{k}_lo"], row[f"top{k}_hi"] = m, lo, hi
            if ref is not None:
                d, dlo, dhi = paired_bootstrap(pc[f"top{k}"].to_numpy(), ref[f"top{k}"].to_numpy(), n_boot, seed)
                row[f"d_top{k}"], row[f"d_top{k}_lo"], row[f"d_top{k}_hi"] = d, dlo, dhi
        row["auc_pooled"] = pooled_auc(grouped.hit, scorers[name])
        row["auc_run_mean"] = float(np.nanmean(pc["auc"])) if pc["auc"].notna().any() else float("nan")
        rows.append(row)
    summary = pd.DataFrame(rows)
    per_run = pd.concat([pd.concat([keys, pc], axis=1).assign(scorer=name) for name, pc in per.items()],
                        ignore_index=True)
    return summary, per_run


def frame_markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """Plain markdown table of a frame (no tabulate dependency)."""
    cols = list(df.columns)
    fmt = lambda v: (f"{v:{floatfmt}}" if isinstance(v, (float, np.floating)) else str(v))
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def summary_markdown(summary: pd.DataFrame, ks: tuple[int, ...] = KS, reference: str = "idelta_score") -> str:
    """Markdown table of a summary frame (one row per scorer, or per set × scorer if ``set`` exists)."""
    cols = ["scorer"] if "set" not in summary.columns else ["set", "scorer"]
    head = cols + ["runs"] + [f"Success@{k} [95 % CI]" for k in ks] + [f"Δ@{k} vs {reference} [95 % CI]" for k in ks] + ["AUC pooled", "AUC per run"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for _, r in summary.iterrows():
        cells = [str(r[c]) for c in cols] + [str(int(r["runs"]))]
        cells += [f"{r[f'top{k}']:.1f} [{r[f'top{k}_lo']:.1f}, {r[f'top{k}_hi']:.1f}]" for k in ks]
        for k in ks:
            if r["scorer"] == reference:
                cells.append("—")
            elif f"d_top{k}" in r and pd.notna(r[f"d_top{k}"]):
                cells.append(f"{r[f'd_top{k}']:+.1f} [{r[f'd_top{k}_lo']:+.1f}, {r[f'd_top{k}_hi']:+.1f}]")
            else:
                cells.append("—")
        cells += [f"{r['auc_pooled']:.3f}", f"{r['auc_run_mean']:.3f}"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# whole-distribution comparison of the first-hit rank
# ---------------------------------------------------------------------------
def rank_shift(per_run: pd.DataFrame, reference: str = "idelta_score") -> pd.DataFrame:
    """Compare the rank of the first hit of every scorer with *reference* on the same runs
    (``per_run`` as returned by :func:`evaluate_scorers`, one set × protocol at a time).

    Columns: ``mrr`` mean reciprocal rank of the first hit (runs without a hit count as 0),
    ``d_mrr`` its difference to the reference, ``earlier`` / ``same`` / ``later`` percent of the
    runs with a hit under both scorers whose first hit moves to a better / the same / a worse rank,
    ``log2_rank_shift`` the mean of log2(rank+1) − log2(rank_ref+1) over those runs (negative =
    earlier)."""
    wide = per_run.set_index(["set", "pdb", "protocol", "scorer"])["first_hit_rank"].unstack("scorer")
    ref = wide[reference]
    mrr_ref = float(np.where(ref.notna(), 1 / (ref + 1), 0.0).mean())
    rows = []
    for sc in wide.columns:
        f = wide[sc]
        both = (f.notna() & ref.notna()).to_numpy()
        fv, rv = f.to_numpy(dtype=float)[both], ref.to_numpy(dtype=float)[both]
        d = fv - rv
        mrr = float(np.where(f.notna(), 1 / (f + 1), 0.0).mean())
        rows.append({"scorer": sc, "runs": int(len(f)), "runs_hit_both": int(both.sum()), "mrr": mrr, "d_mrr": mrr - mrr_ref,
                     "earlier": 100 * (d < 0).mean() if both.any() else np.nan,
                     "same": 100 * (d == 0).mean() if both.any() else np.nan,
                     "later": 100 * (d > 0).mean() if both.any() else np.nan,
                     "log2_rank_shift": float(np.mean(np.log2(fv + 1) - np.log2(rv + 1))) if both.any() else np.nan})
    return pd.DataFrame(rows)


def rank_shift_markdown(df: pd.DataFrame, reference: str = "idelta_score") -> str:
    cols = ["scorer"] if "set" not in df.columns else ["set", "scorer"]
    head = cols + ["runs", "MRR", f"Δ MRR vs {reference}", "first hit earlier / same / later (% of runs with a hit under both)", "mean Δ log2(rank+1)"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for _, r in df.iterrows():
        cells = [str(r[c]) for c in cols] + [str(int(r["runs"])), f"{r['mrr']:.3f}"]
        if r["scorer"] == reference:
            cells += ["—", "—", "—"]
        else:
            cells += [f"{r['d_mrr']:+.3f}", f"{r['earlier']:.1f} / {r['same']:.1f} / {r['later']:.1f}", f"{r['log2_rank_shift']:+.3f}"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# the paper's table: one row per approach, one column per k
# ---------------------------------------------------------------------------
APPROACH_LABELS = {"idelta_score": "unmodified (interface delta)", "tree": "tree (LambdaRank)",
                   "reweight_random": "reweighted, random start (non-negative, raw terms)",
                   "reweight_rosetta": "reweighted, Rosetta start (non-negative, raw terms)",
                   "ranker": "ranker (MLP)"}
APPROACH_ORDER = ["idelta_score", "tree", "reweight_random", "reweight_rosetta", "ranker"]


def approach_table_markdown(summary: pd.DataFrame, ks: tuple[int, ...] = KS, reference: str = "idelta_score",
                            label_col: str | None = None) -> str:
    """Rows = approaches in ``APPROACH_ORDER`` (the reference row shows the absolute Success@k, the
    others the paired Δ [95 % CI] to it), columns = k. ``label_col`` adds its value to the row
    label (e.g. the split when several runs share one table)."""
    head = ["approach", "runs"] + [f"Δ Success@{k} [95 % CI]" for k in ks]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    order = {name: i for i, name in enumerate(APPROACH_ORDER)}
    rows = summary.assign(_o=summary["scorer"].map(lambda n: order.get(n, len(order)))).sort_values(["_o"], kind="stable")
    for _, r in rows.iterrows():
        label = APPROACH_LABELS.get(r["scorer"], r["scorer"])
        if label_col and label_col in r and pd.notna(r[label_col]):
            label += f" — {r[label_col]}"
        cells = [label, str(int(r["runs"]))]
        for k in ks:
            if r["scorer"] == reference or f"d_top{k}" not in r or pd.isna(r[f"d_top{k}"]):
                cells.append(f"{r[f'top{k}']:.1f} (absolute)")
            else:
                cells.append(f"{r[f'd_top{k}']:+.1f} [{r[f'd_top{k}_lo']:+.1f}, {r[f'd_top{k}_hi']:+.1f}]")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
