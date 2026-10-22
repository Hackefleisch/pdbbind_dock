"""
affinity.evaluate
-----------------
Absolute-prediction metrics of predicted against experimental pK over complexes, with
percentile-bootstrap intervals over the complexes (2,000 resamples) and paired differences on the
same resamples: **MAE** (the headline), RMSE, Pearson r (the CASF "scoring power", kept for
comparability), Spearman ρ (point estimate only); ΔMAE to the reference (the calibrated interface
delta) and to the null (the constant predictor).

95 % lower bounds per complex (``bounds``): a bound L_i is evaluated by its **coverage** (share of
complexes with pK ≥ L_i, nominal 95 %), its mean **slack** (point prediction − L_i: how much has to
be given away to be 95 % safe) and the share of complexes **certified** above a threshold
(L_i ≥ pK 5 / 6, i.e. a 10 µM / 1 µM binder certified). Two bounds are produced by ``train``:
``cp`` the one-sided split-conformal bound (:func:`conformal_slack`: the point prediction minus
the ⌈(m + 1)(1 − α)⌉-th smallest signed residual on the calibration complexes; finite-sample
coverage ≥ 1 − α for exchangeable complexes, no model assumption) and ``qr`` the τ = 0.05
quantile-regression model of ``affinity.models``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from .models import NULL, REFERENCE

MIN_N = 20                                   # evaluation / calibration sets smaller than this are skipped
ALPHA = 0.05                                 # lower bounds at 95 %
CERT_THRESHOLDS = (5.0, 6.0)                 # certified binder: lower bound >= pK 5 (10 µM), 6 (1 µM)
BOUND_KINDS = ("cp", "qr")


def conformal_slack(pred_cal, y_cal, alpha: float = ALPHA) -> float:
    """One-sided split-conformal slack s: L = pred − s satisfies P(pK ≥ L) ≥ 1 − α."""
    r = np.sort(np.asarray(pred_cal, dtype=np.float64) - np.asarray(y_cal, dtype=np.float64))
    m = len(r)
    k = min(int(np.ceil((m + 1) * (1 - alpha))), m)
    return float(r[k - 1])


def _pearson_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A - A.mean(axis=1, keepdims=True)
    B = B - B.mean(axis=1, keepdims=True)
    den = np.sqrt((A * A).sum(axis=1) * (B * B).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (A * B).sum(axis=1) / den


def _pct(a: np.ndarray) -> tuple[float, float]:
    a = np.asarray(a, dtype=np.float64)
    if not np.isfinite(a).any():
        return (float("nan"), float("nan"))
    return float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))


def metrics(y, yhat) -> dict:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    out = {"n": int(len(y)), "mae": float(np.mean(np.abs(yhat - y))), "rmse": float(np.sqrt(np.mean((yhat - y) ** 2))),
           "r": float("nan"), "rho": float("nan")}
    if np.ptp(yhat) > 0:                       # a constant predictor has no correlation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out["r"], out["rho"] = float(pearsonr(yhat, y).statistic), float(spearmanr(yhat, y).statistic)
    return out


def evaluate(y, preds: dict[str, np.ndarray], bounds: dict[str, dict[str, np.ndarray]] | None = None,
             reference: str = REFERENCE, null: str = NULL, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """One row per predictor: MAE / RMSE / r with bootstrap intervals, Spearman ρ, the paired ΔMAE
    and Δr to *reference* and the paired ΔMAE to *null* on the same resamples, and for every
    lower bound in ``bounds[name]`` its coverage (with interval), mean slack, mean bound and the
    certified shares."""
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    Y = y[idx]
    boots, rows = {}, {}
    for name, p in preds.items():
        p = np.asarray(p, dtype=np.float64)
        P = p[idx]
        mae = np.mean(np.abs(P - Y), axis=1)
        rmse = np.sqrt(np.mean((P - Y) ** 2, axis=1))
        r = _pearson_rows(P, Y) if np.ptp(p) > 0 else np.full(n_boot, np.nan)
        boots[name] = (mae, rmse, r)
        row = {"predictor": name, **metrics(y, p)}
        row["mae_lo"], row["mae_hi"] = _pct(mae)
        row["rmse_lo"], row["rmse_hi"] = _pct(rmse)
        row["r_lo"], row["r_hi"] = _pct(r)
        for kind, L in (bounds or {}).get(name, {}).items():
            L = np.asarray(L, dtype=np.float64)
            cov = (y >= L).astype(np.float64)
            row[f"{kind}_coverage"] = 100 * float(cov.mean())
            row[f"{kind}_coverage_lo"], row[f"{kind}_coverage_hi"] = (100 * v for v in _pct(cov[idx].mean(axis=1)))
            row[f"{kind}_slack"] = float(np.mean(p - L))
            row[f"{kind}_mean"] = float(np.mean(L))
            for t in CERT_THRESHOLDS:
                row[f"{kind}_cert{t:g}"] = 100 * float(np.mean(L >= t))
        rows[name] = row
    for other, tag in ((reference, ""), (null, "_null")):
        if other not in boots:
            continue
        for name, row in rows.items():
            if name == other:
                continue
            for metric, j in (("mae", 0), ("r", 2)):
                d = boots[name][j] - boots[other][j]
                row[f"d_{metric}{tag}"] = row[metric] - rows[other][metric]
                row[f"d_{metric}{tag}_lo"], row[f"d_{metric}{tag}_hi"] = _pct(d)
    return pd.DataFrame(list(rows.values()))


def pearson_ci(y, score, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Pearson r of *score* with *y* and its percentile-bootstrap interval (no fit)."""
    y, s = np.asarray(y, float), np.asarray(score, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    r = _pearson_rows(s[idx], y[idx])
    return float(pearsonr(s, y).statistic), *_pct(r)


# ---------------------------------------------------------------------------
# markdown helpers
# ---------------------------------------------------------------------------
def frame_markdown(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    cols = list(df.columns)
    fmt = lambda v: (f"{v:{floatfmt}}" if isinstance(v, (float, np.floating)) else str(v))
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def _ci(r, k: str, fmt: str = ".2f", sign: bool = False) -> str:
    if k not in r or pd.isna(r[k]):
        return "—"
    f = f"+{fmt}" if sign else fmt
    return f"{r[k]:{f}} [{r[f'{k}_lo']:{f}}, {r[f'{k}_hi']:{f}}]"


def predictor_label(key: str) -> str:
    from .models import APPROACH_LABELS, TERM_LABELS, split_key
    a, t = split_key(key)
    return APPROACH_LABELS.get(a, a) + (f", {TERM_LABELS[t]}" if t != "all" else "")


def mae_table_markdown(summary: pd.DataFrame, order: list[str] | None = None, reference: str = REFERENCE, null: str = NULL) -> str:
    """Rows = predictors, columns = n, MAE [CI], ΔMAE to the reference and to the null [CI], RMSE, r,
    and the conformal / quantile-regression lower bounds (slack, coverage, certified ≥ pK 5)."""
    head = ["predictor", "n", "MAE [95 % CI]", "ΔMAE vs unmodified [95 % CI]", "ΔMAE vs constant [95 % CI]", "RMSE", "r",
            "conformal LB: slack", "coverage %", "≥ pK 5 %", "≥ pK 6 %", "QR LB: slack", "coverage %", "≥ pK 5 %"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    order = order or list(summary["predictor"])
    s = summary.set_index("predictor")
    for k in order:
        if k not in s.index:
            continue
        r = s.loc[k]
        cells = [predictor_label(k), str(int(r["n"])), _ci(r, "mae"),
                 "—" if k == reference else _ci(r, "d_mae", sign=True), "—" if k == null else _ci(r, "d_mae_null", sign=True),
                 f"{r['rmse']:.2f}", f"{r['r']:.3f}"]
        for kind in BOUND_KINDS:
            if f"{kind}_slack" in r and pd.notna(r[f"{kind}_slack"]):
                cells += [f"{r[f'{kind}_slack']:.2f}", f"{r[f'{kind}_coverage']:.1f}", f"{r[f'{kind}_cert5']:.1f}"]
                if kind == "cp":
                    cells.append(f"{r[f'{kind}_cert6']:.1f}")
            else:
                cells += ["—", "—", "—"] + (["—"] if kind == "cp" else [])
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
