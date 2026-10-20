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
``roc_auc``) and the mean per-run AUC (the quantity the per-complex oracle of inventory
Appendix A.2 bounds).
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
