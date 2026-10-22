"""
affinity.evaluate
-----------------
Scoring-power metrics of predicted against experimental pK over complexes, with
percentile-bootstrap intervals over the complexes (2,000 resamples) and the paired bootstrap of
the difference to the reference approach (the calibrated interface delta) on the same complexes:
Pearson r (the CASF "scoring power"), Spearman ρ, RMSE and MAE in pK units.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr

from .models import APPROACH_LABELS, APPROACHES, REFERENCE

MIN_N = 20                                   # evaluation sets smaller than this are skipped


def _pearson_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A - A.mean(axis=1, keepdims=True)
    B = B - B.mean(axis=1, keepdims=True)
    den = np.sqrt((A * A).sum(axis=1) * (B * B).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (A * B).sum(axis=1) / den


def metrics(y, yhat) -> dict:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return {"n": int(len(y)), "r": float(pearsonr(yhat, y).statistic), "rho": float(spearmanr(yhat, y).statistic),
            "rmse": float(np.sqrt(np.mean((yhat - y) ** 2))), "mae": float(np.mean(np.abs(yhat - y)))}


def evaluate(y, preds: dict[str, np.ndarray], reference: str = REFERENCE, n_boot: int = 2000, seed: int = 0,
             with_rmse: bool = True) -> pd.DataFrame:
    """One row per approach: the metrics, their bootstrap 95 % intervals and the paired
    difference (Δr, ΔRMSE) to *reference* with its interval, all on the same resamples."""
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    Y = y[idx]
    Yr = rankdata(Y, axis=1)
    boots, rows = {}, {}
    for name, p in preds.items():
        p = np.asarray(p, dtype=np.float64)
        P = p[idx]
        r = _pearson_rows(P, Y)
        rho = _pearson_rows(rankdata(P, axis=1), Yr)
        rmse = np.sqrt(np.mean((P - Y) ** 2, axis=1))
        boots[name] = (r, rho, rmse)
        m = metrics(y, p)
        row = {"approach": name, "n": n, "r": m["r"], "r_lo": np.nanpercentile(r, 2.5), "r_hi": np.nanpercentile(r, 97.5),
               "rho": m["rho"], "rho_lo": np.nanpercentile(rho, 2.5), "rho_hi": np.nanpercentile(rho, 97.5)}
        if with_rmse:
            row.update({"rmse": m["rmse"], "rmse_lo": np.percentile(rmse, 2.5), "rmse_hi": np.percentile(rmse, 97.5), "mae": m["mae"]})
        rows[name] = row
    if reference in boots:
        r_ref, _, rmse_ref = boots[reference]
        for name, row in rows.items():
            if name == reference:
                continue
            d = boots[name][0] - r_ref
            row.update({"d_r": row["r"] - rows[reference]["r"], "d_r_lo": np.nanpercentile(d, 2.5), "d_r_hi": np.nanpercentile(d, 97.5)})
            if with_rmse:
                d = boots[name][2] - rmse_ref
                row.update({"d_rmse": row["rmse"] - rows[reference]["rmse"], "d_rmse_lo": np.percentile(d, 2.5), "d_rmse_hi": np.percentile(d, 97.5)})
    return pd.DataFrame(list(rows.values()))


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------
def frame_markdown(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    cols = list(df.columns)
    fmt = lambda v: (f"{v:{floatfmt}}" if isinstance(v, (float, np.floating)) else str(v))
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def _ci(r, k: str, fmt: str = ".3f", sign: bool = False) -> str:
    if k not in r or pd.isna(r[k]):
        return "—"
    f = f"+{fmt}" if sign else fmt
    return f"{r[k]:{f}} [{r[f'{k}_lo']:{f}}, {r[f'{k}_hi']:{f}}]"


def approach_table_markdown(summary: pd.DataFrame, reference: str = REFERENCE, label_col: str | None = None,
                            with_rmse: bool = True) -> str:
    """Rows = approaches (reference first), columns = n, Pearson r [CI], Δr to the reference
    [CI], Spearman ρ, RMSE [CI]; ``label_col`` adds its value to the row label."""
    head = ["approach", "n", "Pearson r [95 % CI]", f"Δr vs {APPROACH_LABELS.get(reference, reference).split(' (')[0]} [95 % CI]", "Spearman ρ"]
    if with_rmse:
        head.append("RMSE (pK) [95 % CI]")
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    order = {a: i for i, a in enumerate(APPROACHES)}
    rows = summary.assign(_o=summary["approach"].map(lambda a: order.get(a, len(order)))).sort_values("_o", kind="stable")
    for _, r in rows.iterrows():
        label = APPROACH_LABELS.get(r["approach"], r["approach"])
        if label_col and label_col in r and pd.notna(r[label_col]):
            label += f" — {r[label_col]}"
        cells = [label, str(int(r["n"])), _ci(r, "r"), "—" if r["approach"] == reference else _ci(r, "d_r", sign=True), f"{r['rho']:.3f}"]
        if with_rmse:
            cells.append(_ci(r, "rmse", ".2f"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
