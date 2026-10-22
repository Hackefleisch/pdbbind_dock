"""
affinity.report
---------------
The paper's tables and figures of the affinity track from the outputs of :mod:`affinity.train`
(``<AFFINITY_DIR>/results.csv``, ``baseline_all.csv``, ``weights.csv``, ``predictions.parquet``).
Headline metric: the MAE in pK of the ``l1`` (median-regression) fits against the constant
predictor, with the 95 % lower bounds; Pearson r of the ``l2`` fits is kept in a later table for
comparability with the CASF scoring power.

Tables (``<OUTPUT_DIR>/affinity/affinity_tables.md``; every row of ``results.csv`` is copied to
``affinity_results.csv``):

1. headline cell on LP-PDBBind — every predictor with all metrics and both lower bounds;
2. headline cell across the three splits — MAE / ΔMAE, and the conformal bound per split;
3. term ablation (all / without fa_pair / enthalpic) per split;
4. pools × aggregations on the headline split;
5. the Kd and the Ki half of the headline test set;
6. IC50 (trained and tested on IC50 complexes);
7. the CASF-2016 core complexes outside each split's training and calibration sets;
8. the unmodified interface delta on all labelled complexes (no split, no fit; Pearson r);
9. the learned energy weights of the headline cell (l1 fits, normalised to fa_atr = 0.8) and the raw coefficients;
10. Pearson r of the l2 fits per split.

Figures: ``headline_scatter.png`` (predicted vs experimental pK), ``lower_bound.png`` (share of
test complexes certified above a pK threshold by the conformal bound), ``pools.png`` (MAE with
95 % intervals per pool × aggregation), ``weights.png`` (learned non-negative weights vs the
defaults).

Usage::

    uv run python -m affinity.report [--in-dir ...] [--out ...] [--no-figures]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from docking_analysis.constants import AFFINITY_DIR, OUTPUT_DIR

from .data import AGGREGATIONS, HEADLINE_AGG, HEADLINE_POOL, HIT_CUTOFF, SPLIT_SCHEMES, TRAIN_POOLS, pool_label, term_name
from .evaluate import BOUND_KINDS, _ci, mae_table_markdown, predictor_label
from .models import APPROACH_LABELS, NULL, REFERENCE, TERM_APPROACHES, TERM_LABELS, TERM_SETS, predictor_key

SPLIT_LABELS = {"lppdbbind": "LP-PDBBind", "hac": "HAC (CASF-2016 core as test)", "cleansplit": "CleanSplit"}
SPLIT_SHORT = {"lppdbbind": "LP-PDBBind", "hac": "HAC", "cleansplit": "CleanSplit"}
LABEL_SET_LABELS = {"kdki": "Kd + Ki", "kd": "Kd", "ki": "Ki", "ic50": "IC50"}
AGG_LABELS = {"best": "best pose", "mean10": "mean of 10 best", "mean": "mean (interface delta ≤ 0)", "best_hit": "best hit (oracle)"}
HEADLINE_SPLIT = "lppdbbind"
HEADLINE_LOSS = "l1"
MAIN_KEYS = ["constant", "unmodified", "size", "reweight", "reweight_nonneg", "reweight_size"]
ABLATION_KEYS = [predictor_key(a, t) for a in TERM_APPROACHES for t in TERM_SETS]

# the reference palette of the dataviz method (slots 1-3 validate all-pairs; chrome from the same file)
C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, MUTED, GRID, BASELINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SERIES = [("unmodified", "unmodified interface delta", C_BLUE, "o"), ("reweight", "reweighted (free sign)", C_ORANGE, "s"),
          ("size", "ligand size", C_AQUA, "D")]


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASELINE)
    ax.tick_params(colors=INK2, labelsize=8.5, length=3, color=BASELINE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def _sel(df: pd.DataFrame, **kw) -> pd.DataFrame:
    m = np.ones(len(df), dtype=bool)
    for k, v in kw.items():
        m &= df[k].eq(v).to_numpy()
    return df[m]


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def _metric_cells(r, key: str, metric: str) -> list[str]:
    if metric == "mae":
        return [_ci(r, "mae"), "—" if key == REFERENCE else _ci(r, "d_mae", sign=True)]
    return [_ci(r, "r", ".3f"), "—" if key == REFERENCE else _ci(r, "d_r", ".3f", sign=True)]


def wide_split_table(res: pd.DataFrame, label_set: str, pool: str, agg: str, keys: list[str], eval_set: str = "test",
                     loss: str = HEADLINE_LOSS, splits: tuple[str, ...] = SPLIT_SCHEMES, metric: str = "mae") -> str:
    """Rows = predictors, columns = metric [CI] and its paired Δ to the calibrated interface delta per split."""
    name = "MAE" if metric == "mae" else "r"
    cols, subs = [], {}
    for s in splits:
        sub = _sel(res, label_set=label_set, pool=pool, aggregation=agg, eval_set=eval_set, split=s, loss=loss).set_index("predictor")
        if len(sub) == 0:
            continue
        subs[s] = sub
        cols += [f"{SPLIT_SHORT[s]} (n = {int(sub['n'].iloc[0]):,}) {name} [95 % CI]", f"{SPLIT_SHORT[s]} Δ{name} vs unmodified [95 % CI]"]
    lines = ["| predictor | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for k in keys:
        cells = [predictor_label(k)]
        for s, sub in subs.items():
            cells += _metric_cells(sub.loc[k], k, metric) if k in sub.index else ["—", "—"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def bounds_split_table(res: pd.DataFrame, label_set: str, pool: str, agg: str, keys: list[str], eval_set: str = "test",
                       loss: str = HEADLINE_LOSS, kind: str = "cp") -> str:
    """Rows = predictors, columns = slack / coverage / certified ≥ pK 5 of one lower bound per split."""
    cols, subs = [], {}
    for s in SPLIT_SCHEMES:
        sub = _sel(res, label_set=label_set, pool=pool, aggregation=agg, eval_set=eval_set, split=s, loss=loss).set_index("predictor")
        if len(sub) == 0 or f"{kind}_slack" not in sub.columns or sub[f"{kind}_slack"].isna().all():
            continue
        subs[s] = sub
        cols += [f"{SPLIT_SHORT[s]} slack (pK)", f"{SPLIT_SHORT[s]} coverage % [95 % CI]", f"{SPLIT_SHORT[s]} ≥ pK 5 %"]
    lines = ["| predictor | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for k in keys:
        cells = [predictor_label(k)]
        for s, sub in subs.items():
            if k not in sub.index:
                cells += ["—"] * 3
                continue
            r = sub.loc[k]
            cells += [f"{r[f'{kind}_slack']:.2f}", f"{r[f'{kind}_coverage']:.1f} [{r[f'{kind}_coverage_lo']:.1f}, {r[f'{kind}_coverage_hi']:.1f}]",
                      f"{r[f'{kind}_cert5']:.1f}"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def pools_table(res: pd.DataFrame, pred: pd.DataFrame, label_set: str, split: str, eval_set: str = "test", loss: str = HEADLINE_LOSS,
                pools: tuple[str, ...] = TRAIN_POOLS, aggs: tuple[str, ...] = AGGREGATIONS) -> str:
    head = ["pool", "aggregation", "n", "pose RMSD median (Å)", "pose ≤ 2 Å (%)", "constant MAE", "unmodified MAE [95 % CI]",
            "ligand size MAE [95 % CI]", "reweighted ΔMAE [95 % CI]", "non-negative ΔMAE [95 % CI]", "reweighted + size ΔMAE [95 % CI]",
            "conformal slack unmodified / reweighted", "≥ pK 5 % unmodified / reweighted"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for pool in pools:
        for agg in aggs:
            sub = _sel(res, label_set=label_set, split=split, pool=pool, aggregation=agg, eval_set=eval_set, loss=loss).set_index("predictor")
            if len(sub) == 0:
                continue
            p = _sel(pred, label_set=label_set, split=split, pool=pool, aggregation=agg, eval_set=eval_set, loss=loss)
            rmsd, hit = (f"{p['rmsd_to_crystal'].median():.2f}", f"{100 * (p['rmsd_to_crystal'] <= HIT_CUTOFF).mean():.0f}") if len(p) else ("—", "—")
            u, w = sub.loc[REFERENCE], sub.loc["reweight"]
            sl = f"{u['cp_slack']:.2f} / {w['cp_slack']:.2f}" if "cp_slack" in sub.columns and pd.notna(u["cp_slack"]) else "—"
            ce = f"{u['cp_cert5']:.1f} / {w['cp_cert5']:.1f}" if "cp_cert5" in sub.columns and pd.notna(u["cp_cert5"]) else "—"
            cells = [pool_label(pool), AGG_LABELS[agg], f"{int(sub['n'].iloc[0]):,}", rmsd, hit, f"{sub.loc[NULL, 'mae']:.2f}",
                     _ci(u, "mae"), _ci(sub.loc["size"], "mae"), _ci(w, "d_mae", sign=True), _ci(sub.loc["reweight_nonneg"], "d_mae", sign=True),
                     _ci(sub.loc["reweight_size"], "d_mae", sign=True), sl, ce]
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def eval_sets_table(res: pd.DataFrame, label_set: str, split: str, pool: str, agg: str, eval_sets: dict[str, str], keys: list[str],
                    loss: str = HEADLINE_LOSS) -> str:
    cols, subs = [], {}
    for ev, name in eval_sets.items():
        sub = _sel(res, label_set=label_set, split=split, pool=pool, aggregation=agg, eval_set=ev, loss=loss).set_index("predictor")
        if len(sub) == 0:
            continue
        subs[ev] = sub
        cols += [f"{name} (n = {int(sub['n'].iloc[0]):,}) MAE [95 % CI]", f"{name} ΔMAE vs unmodified [95 % CI]"]
    lines = ["| predictor | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for k in keys:
        cells = [predictor_label(k)]
        for ev, sub in subs.items():
            cells += _metric_cells(sub.loc[k], k, "mae") if k in sub.index else ["—", "—"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def casf_table(res: pd.DataFrame, label_set: str, pool: str, agg: str, keys: list[str], loss: str = HEADLINE_LOSS) -> str:
    cols, subs = [], {}
    for s in SPLIT_SCHEMES:
        sub = _sel(res, label_set=label_set, split=s, pool=pool, aggregation=agg, eval_set="casf_core", loss=loss).set_index("predictor")
        if len(sub) == 0:
            continue
        subs[s] = sub
        cols += [f"trained on {SPLIT_SHORT[s]} (n = {int(sub['n'].iloc[0]):,}) MAE [95 % CI]", f"{SPLIT_SHORT[s]} ΔMAE vs unmodified [95 % CI]"]
    lines = ["| predictor | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for k in keys:
        cells = [predictor_label(k)]
        for s, sub in subs.items():
            cells += _metric_cells(sub.loc[k], k, "mae") if k in sub.index else ["—", "—"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def baseline_table(base: pd.DataFrame, pools: tuple[str, ...], label_sets=("kdki", "kd", "ki", "ic50")) -> str:
    head = ["pool", "aggregation", "pose ≤ 2 Å (%)"] + [f"{LABEL_SET_LABELS[ls]} r [95 % CI] (n)" for ls in label_sets]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for pool in pools:
        for agg in AGGREGATIONS:
            sub = base[base["pool"].eq(pool) & base["aggregation"].eq(agg)].set_index("label_set")
            if len(sub) == 0:
                continue
            cells = [pool_label(pool), AGG_LABELS[agg], f"{sub['pose_hit_rate'].iloc[0]:.0f}"]
            for ls in label_sets:
                cells.append(f"{_ci(sub.loc[ls], 'r', '.3f')} ({int(sub.loc[ls, 'n']):,})" if ls in sub.index else "—")
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def weights_table(wts: pd.DataFrame, label_set: str, pool: str, agg: str, loss: str = HEADLINE_LOSS) -> str:
    """Energy weights (normalised to fa_atr = 0.8) of the reweightings per term set and split, the
    defaults, and the τ = 0.05 lower-bound model of the non-negative reweighting."""
    sub = _sel(wts, label_set=label_set, pool=pool, aggregation=agg)
    sub = sub[sub["approach"].isin(TERM_APPROACHES) & sub["energy_weight"].notna()]
    terms = [term_name(c) for c in TERM_SETS["all"]]
    head = ["model · term set · split"] + terms + ["intercept (pK)", "n train"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    d = sub.drop_duplicates("feature").set_index("feature")["rosetta_default"]
    lines.append("| Rosetta default | " + " | ".join(f"{d.get(t, np.nan):.2f}" for t in terms) + " | — | — |")
    def rows(frame, label):
        for s in SPLIT_SCHEMES:
            g = frame[frame["split"].eq(s)].set_index("feature")
            if len(g) == 0:
                continue
            lines.append(f"| {label} · {SPLIT_SHORT[s]} | " + " | ".join(f"{g.loc[t, 'weight_fa_atr_0.8']:.2f}" if t in g.index else "—" for t in terms)
                         + f" | {g['intercept'].iloc[0]:.2f} | {int(g['n_train'].iloc[0]):,} |")
    for a in ("reweight_nonneg", "reweight", "reweight_size"):
        for ts in TERM_SETS:
            rows(sub[sub["approach"].eq(a) & sub["terms"].eq(ts) & sub["loss"].eq(loss)], f"{APPROACH_LABELS[a]}, {TERM_LABELS[ts]}")
    rows(sub[sub["approach"].eq("reweight_nonneg") & sub["terms"].eq("all") & sub["loss"].eq("quantile")],
         "lower-bound model (τ = 0.05), non-negative, all terms")
    return "\n".join(lines)


def coef_table(wts: pd.DataFrame, label_set: str, split: str, pool: str, agg: str, loss: str = HEADLINE_LOSS) -> str:
    """The raw coefficients (pK per unit of the feature) of the main predictors of one cell."""
    sub = _sel(wts, label_set=label_set, split=split, pool=pool, aggregation=agg, loss=loss)
    sub = sub[sub["terms"].eq("all")]
    feats = ["idelta_score", "ligand_n_heavy"] + [term_name(c) for c in TERM_SETS["all"]]
    head = ["predictor"] + feats + ["intercept"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for k in MAIN_KEYS:
        g = sub[sub["approach"].eq(k)].set_index("feature")
        if len(g) == 0:
            continue
        lines.append(f"| {predictor_label(k)} | " + " | ".join(f"{g.loc[f, 'coef_pk_per_unit']:.4f}" if f in g.index else "—" for f in feats)
                     + f" | {g['intercept'].iloc[0]:.2f} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def fig_headline_scatter(pred: pd.DataFrame, res: pd.DataFrame, out: Path, label_set: str = "kdki", split: str = HEADLINE_SPLIT,
                         pool: str = HEADLINE_POOL, agg: str = HEADLINE_AGG, loss: str = HEADLINE_LOSS,
                         keys=("unmodified", "reweight", "size")) -> Path | None:
    p = _sel(pred, label_set=label_set, split=split, pool=pool, aggregation=agg, eval_set="test", loss=loss)
    s = _sel(res, label_set=label_set, split=split, pool=pool, aggregation=agg, eval_set="test", loss=loss).set_index("predictor")
    if len(p) == 0:
        return None
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    fig, axes = plt.subplots(1, len(keys), figsize=(3.4 * len(keys), 3.7), sharex=True, sharey=True, facecolor=SURFACE)
    lo, hi = float(p["pk"].min()) - 0.5, float(p["pk"].max()) + 0.5
    for ax, k in zip(np.atleast_1d(axes), keys):
        _style(ax)
        ax.hexbin(p["pk"], p[f"pred_{k}"], gridsize=28, cmap=cmap, mincnt=1, linewidths=0.2, edgecolors=SURFACE, extent=(lo, hi, lo, hi))
        ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        r = s.loc[k]
        ax.set_title(f"{predictor_label(k)}\nMAE {r['mae']:.2f} [{r['mae_lo']:.2f}, {r['mae_hi']:.2f}] pK, r = {r['r']:.2f}", fontsize=9)
        ax.set_xlabel("experimental pK")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
    np.atleast_1d(axes)[0].set_ylabel("predicted pK")
    c = s.loc[NULL]
    fig.suptitle(f"{LABEL_SET_LABELS[label_set]}, {SPLIT_LABELS[split]} test complexes (n = {len(p):,}), {pool_label(pool)}, {AGG_LABELS[agg]}; "
                 f"constant predictor MAE {c['mae']:.2f} pK", fontsize=9, color=INK2)
    fig.tight_layout()
    path = out / "headline_scatter.png"
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_lower_bound(pred: pd.DataFrame, out: Path, label_set: str = "kdki", split: str = HEADLINE_SPLIT, pool: str = HEADLINE_POOL,
                    agg: str = HEADLINE_AGG, loss: str = HEADLINE_LOSS, kind: str = "cp") -> Path | None:
    p = _sel(pred, label_set=label_set, split=split, pool=pool, aggregation=agg, eval_set="test", loss=loss)
    if len(p) == 0 or f"{kind}_unmodified" not in p.columns or p[f"{kind}_unmodified"].isna().all():
        return None
    thr = np.linspace(2, 10, 161)
    fig, ax = plt.subplots(figsize=(6.0, 3.8), facecolor=SURFACE)
    _style(ax)
    y = p["pk"].to_numpy()
    ax.plot(thr, [100 * np.mean(y >= t) for t in thr], color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), label="complexes with experimental pK ≥ threshold")
    ax.plot(thr, [100 * np.mean(p[f"{kind}_constant"].to_numpy() >= t) for t in thr], color=BASELINE, linewidth=2, label="constant predictor")
    for k, label, color, _ in SERIES:
        L = p[f"{kind}_{k}"].to_numpy()
        ax.plot(thr, [100 * np.mean(L >= t) for t in thr], color=color, linewidth=2, label=label)
    ax.set_xlabel("pK threshold")
    ax.set_ylabel("complexes certified above threshold (%)")
    ax.set_xlim(2, 10)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    bound = "95 % conformal lower bound" if kind == "cp" else "τ = 0.05 quantile-regression lower bound"
    ax.set_title(f"{bound}\n{LABEL_SET_LABELS[label_set]}, {SPLIT_SHORT[split]} test complexes (n = {len(p):,}), {pool_label(pool)}, {AGG_LABELS[agg]}, {loss} fits",
                 fontsize=9)
    fig.tight_layout()
    path = out / "lower_bound.png"
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_pools(res: pd.DataFrame, out: Path, label_set: str = "kdki", split: str = HEADLINE_SPLIT, eval_set: str = "test", loss: str = HEADLINE_LOSS,
              pools: tuple[str, ...] = TRAIN_POOLS, aggs: tuple[str, ...] = ("best", "mean10", "best_hit")) -> Path | None:
    sub = _sel(res, label_set=label_set, split=split, eval_set=eval_set, loss=loss)
    if len(sub) == 0:
        return None
    fig, axes = plt.subplots(1, len(aggs), figsize=(3.2 * len(aggs) + 1.5, 0.42 * len(pools) + 1.8), sharey=True, sharex=True, facecolor=SURFACE)
    ys = np.arange(len(pools))[::-1]
    for ax, agg in zip(np.atleast_1d(axes), aggs):
        _style(ax)
        for y, pool in zip(ys, pools):
            c = _sel(sub, pool=pool, aggregation=agg, predictor=NULL)
            if len(c):
                ax.plot([c["mae_lo"].iloc[0], c["mae_hi"].iloc[0]], [y + 0.33, y + 0.33], color=BASELINE, linewidth=2, solid_capstyle="round")
                ax.plot(c["mae"].iloc[0], y + 0.33, marker="v", color=BASELINE, markersize=5.5, markeredgecolor=SURFACE, markeredgewidth=1,
                        label="constant predictor" if y == ys[0] else None)
        for j, (k, label, color, marker) in enumerate(SERIES):
            off = (j - 1) * 0.22
            for y, pool in zip(ys, pools):
                r = _sel(sub, pool=pool, aggregation=agg, predictor=k)
                if len(r) == 0:
                    continue
                r = r.iloc[0]
                ax.plot([r["mae_lo"], r["mae_hi"]], [y + off, y + off], color=color, linewidth=2, solid_capstyle="round")
                ax.plot(r["mae"], y + off, marker=marker, color=color, markersize=5.5, markeredgecolor=SURFACE, markeredgewidth=1,
                        label=label if y == ys[0] else None)
        ax.set_title(AGG_LABELS[agg], fontsize=9.5)
    keys = [NULL] + [s[0] for s in SERIES]
    v = sub[sub["pool"].isin(pools) & sub["predictor"].isin(keys)]
    np.atleast_1d(axes)[0].set_xlim(float(v["mae_lo"].min()) - 0.03, float(v["mae_hi"].max()) + 0.03)
    fig.supxlabel("MAE of the predicted pK (lower is better)", fontsize=9, color=INK2, y=0.085)
    ax0 = np.atleast_1d(axes)[0]
    ax0.set_yticks(ys)
    ax0.set_yticklabels([pool_label(p) for p in pools], fontsize=8.5)
    handles, labels = ax0.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"{LABEL_SET_LABELS[label_set]}, {SPLIT_LABELS[split]} {'test' if eval_set == 'test' else eval_set} complexes, {loss} fits; "
                 f"markers = MAE, bars = 95 % bootstrap interval", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    path = out / "pools.png"
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_weights(wts: pd.DataFrame, out: Path, label_set: str = "kdki", pool: str = HEADLINE_POOL, agg: str = HEADLINE_AGG,
                loss: str = HEADLINE_LOSS) -> Path | None:
    sub = _sel(wts, label_set=label_set, pool=pool, aggregation=agg, approach="reweight_nonneg", terms="all", loss=loss)
    if len(sub) == 0:
        return None
    terms = list(dict.fromkeys(sub["feature"]))
    d = sub.drop_duplicates("feature").set_index("feature")["rosetta_default"]
    fig, ax = plt.subplots(figsize=(6.2, 3.9), facecolor=SURFACE)
    _style(ax)
    ys = np.arange(len(terms))[::-1]
    ax.barh(ys + 0.2, [d[t] for t in terms], height=0.36, color=BASELINE, label="Rosetta default")
    colors = {"lppdbbind": C_BLUE, "hac": C_ORANGE, "cleansplit": C_AQUA}
    markers = {"lppdbbind": "o", "hac": "s", "cleansplit": "D"}
    first = True
    for s in SPLIT_SCHEMES:
        g = sub[sub["split"].eq(s)].set_index("feature")
        if len(g) == 0:
            continue
        vals = [g.loc[t, "weight_fa_atr_0.8"] for t in terms]
        if first:
            ax.barh(ys - 0.2, vals, height=0.36, color=C_BLUE, label=f"learned, non-negative ({SPLIT_SHORT[s]})")
            first = False
        else:
            ax.plot(vals, ys - 0.2, linestyle="none", marker=markers[s], color=colors[s], markersize=6, markeredgecolor=SURFACE,
                    markeredgewidth=1, label=f"learned, non-negative ({SPLIT_SHORT[s]})")
    ax.set_yticks(ys)
    ax.set_yticklabels(terms)
    ax.set_xlabel("weight (normalised to fa_atr = 0.8)")
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: (0 if labels[i].startswith("Rosetta") else 1 + list(SPLIT_SCHEMES).index(
        next(s for s in SPLIT_SCHEMES if SPLIT_SHORT[s] in labels[i]))))
    ax.legend([handles[i] for i in order], [labels[i] for i in order], frameon=False, fontsize=8.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), ncol=2)
    ax.set_title(f"{LABEL_SET_LABELS[label_set]}, {pool_label(pool)}, {AGG_LABELS[agg]}, {loss} fit", fontsize=9.5)
    fig.tight_layout()
    path = out / "weights.png"
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", type=Path, default=AFFINITY_DIR)
    ap.add_argument("--out", type=Path, default=OUTPUT_DIR / "affinity")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    res = pd.read_csv(args.in_dir / "results.csv")
    base = pd.read_csv(args.in_dir / "baseline_all.csv")
    wts = pd.read_csv(args.in_dir / "weights.csv")
    pred = pd.read_parquet(args.in_dir / "predictions.parquet")
    meta = json.loads((args.in_dir / "results.json").read_text())
    shutil.copy(args.in_dir / "results.csv", out / "affinity_results.csv")
    have = set(res["pool"])
    pools_present = tuple([p for p in TRAIN_POOLS if p in have] + [p for p in res["pool"].unique() if p not in TRAIN_POOLS])
    H = (HEADLINE_POOL, HEADLINE_AGG)
    sub_lp = _sel(res, label_set="kdki", split=HEADLINE_SPLIT, pool=H[0], aggregation=H[1], eval_set="test", loss=HEADLINE_LOSS)

    md = ["# Affinity track — tables of `affinity.report`", "",
          f"Source: `{args.in_dir}` (`affinity.train`, {meta['n_results']} result rows, {meta['config']['n_boot']} bootstrap resamples, "
          f"seed {meta['config']['seed']}). Labelled complete PDBbind complexes: {meta['label_counts']}. Point fits on the training "
          "complexes of the split, conformal slack calibrated on its validation complexes, evaluation on the test complexes. Cells: "
          "`MAE` in pK with its percentile-bootstrap 95 % interval; `ΔMAE` the paired difference to the calibrated interface delta "
          "(`unmodified`) or to the constant predictor on the same complexes, negative = better. Lower bounds at 95 %: `conformal` = "
          "prediction minus the slack calibrated on the validation complexes, `QR` = the τ = 0.05 quantile-regression model; `slack` "
          "the mean distance between the point prediction and the bound, `coverage` the share of test complexes whose experimental pK "
          "lies above the bound (nominal 95), `≥ pK 5 / 6` the share of complexes certified above 10 µM / 1 µM. Predictors: the "
          "constant (training median), the calibrated interface delta, the ligand heavy-atom count, and three least-squares / "
          "median-regression reweightings of the raw interface terms (free sign; non-negative = a Rosetta weight set; free sign + "
          "heavy-atom count), each fitted with all seven terms, without `fa_pair`, and enthalpic only (without `fa_pair` and "
          f"`fa_sol`). Headline rows: the `{HEADLINE_LOSS}` fits (median regression, the loss matching the MAE).", ""]
    md += [f"## 1. Headline cell — {LABEL_SET_LABELS['kdki']}, `{H[0]}`, {AGG_LABELS[H[1]]}, {SPLIT_LABELS[HEADLINE_SPLIT]} test complexes, {HEADLINE_LOSS} fits", "",
           mae_table_markdown(sub_lp, MAIN_KEYS + [k for k in ABLATION_KEYS if k not in MAIN_KEYS]), ""]
    md += [f"## 2. Headline cell across the splits — MAE and the conformal lower bound", "",
           wide_split_table(res, "kdki", H[0], H[1], MAIN_KEYS), "", "Conformal 95 % lower bound per split:", "",
           bounds_split_table(res, "kdki", H[0], H[1], MAIN_KEYS), "", "Quantile-regression (τ = 0.05) lower bound per split:", "",
           bounds_split_table(res, "kdki", H[0], H[1], MAIN_KEYS, kind="qr"), ""]
    md += ["## 3. Term ablation — the three reweightings with all terms, without fa_pair, enthalpic only (full refit)", "",
           wide_split_table(res, "kdki", H[0], H[1], ["constant", "unmodified"] + ABLATION_KEYS), ""]
    for pool in ("relax_docking_perturb", "relax_ref"):
        if pool in have:
            md += [f"**{pool_label(pool)}** (`{pool}`), best pose", "", wide_split_table(res, "kdki", pool, "best", ["constant", "unmodified"] + ABLATION_KEYS), ""]
    md += [f"## 4. Pools × aggregations — {LABEL_SET_LABELS['kdki']}, {SPLIT_LABELS[HEADLINE_SPLIT]} test complexes, {HEADLINE_LOSS} fits", "",
           "`pose RMSD` / `pose ≤ 2 Å`: the symmetry-corrected RMSD to the crystal ligand of the aggregated pose on the test complexes "
           "(for the means: the mean RMSD of the averaged poses).", "",
           pools_table(res, pred, "kdki", HEADLINE_SPLIT, pools=pools_present), ""]
    md += [f"## 5. The Kd and the Ki half of the headline test set — {SPLIT_LABELS[HEADLINE_SPLIT]}, `{H[0]}`, {AGG_LABELS[H[1]]}", "",
           eval_sets_table(res, "kdki", HEADLINE_SPLIT, H[0], H[1], {"test": "Kd + Ki", "test_kd": "Kd only", "test_ki": "Ki only"}, MAIN_KEYS), ""]
    if "ic50" in set(res["label_set"]):
        md += [f"## 6. IC50 (trained and tested on IC50 complexes) — `{H[0]}`, {AGG_LABELS[H[1]]}", "",
               wide_split_table(res, "ic50", H[0], H[1], MAIN_KEYS), ""]
        if "relax_ref" in have:
            md += [f"**{pool_label('relax_ref')}**, IC50", "", wide_split_table(res, "ic50", "relax_ref", "best", MAIN_KEYS), ""]
    md += [f"## 7. CASF-2016 core complexes outside the training and calibration sets — {LABEL_SET_LABELS['kdki']}, `{H[0]}`, {AGG_LABELS[H[1]]}", "",
           "For HAC the core set *is* the test set (table 2); for the other splits the core complexes that fell into their training or "
           "validation set are removed.", "", casf_table(res, "kdki", H[0], H[1], MAIN_KEYS), ""]
    if "relax_ref" in have:
        md += [f"**{pool_label('relax_ref')}**, CASF-2016 core", "", casf_table(res, "kdki", "relax_ref", "best", MAIN_KEYS), ""]
    md += ["## 8. The unmodified interface delta on all labelled complete complexes (no split, no fit)", "",
           "Pearson r of −`idelta_score` with pK, percentile-bootstrap interval over complexes; `pose ≤ 2 Å` the share of the "
           "aggregated poses within 2 Å of the crystal ligand (for `mean`: of the averaged poses).", "",
           baseline_table(base, tuple(list(TRAIN_POOLS) + [p for p in base["pool"].unique() if p not in TRAIN_POOLS])), ""]
    md += [f"## 9. Learned energy weights — {LABEL_SET_LABELS['kdki']}, `{H[0]}`, {AGG_LABELS[H[1]]}, {HEADLINE_LOSS} fits", "",
           "Energy weights w (pK = b − Σ w · term) normalised to fa_atr = 0.8 like the ranking weights of S5; the free-sign fits can "
           "give negative weights; `—` = term not in the set.", "", weights_table(wts, "kdki", H[0], H[1]), "",
           f"Raw coefficients (pK per raw unit) of the {SPLIT_SHORT[HEADLINE_SPLIT]} {HEADLINE_LOSS} models of the headline cell:", "",
           coef_table(wts, "kdki", HEADLINE_SPLIT, H[0], H[1]), ""]
    if "relax_ref" in have:
        md += [f"Energy weights on the relaxed holo complex (`relax_ref`):", "", weights_table(wts, "kdki", "relax_ref", "best"), ""]
    if "l2" in set(res["loss"]):
        md += [f"## 10. Pearson r of the least-squares (l2) fits — {LABEL_SET_LABELS['kdki']}, `{H[0]}`, {AGG_LABELS[H[1]]}, test complexes of each split", "",
               wide_split_table(res, "kdki", H[0], H[1], MAIN_KEYS, loss="l2", metric="r"), ""]
        for pool in ("relax_docking_perturb", "relax_ref", "crystal"):
            if pool in have:
                md += [f"**{pool_label(pool)}** (`{pool}`), best pose, l2 fits, Pearson r", "", wide_split_table(res, "kdki", pool, "best", MAIN_KEYS, loss="l2", metric="r"), ""]
    if not args.no_figures:
        figs = [fig_headline_scatter(pred, res, out), fig_lower_bound(pred, out), fig_pools(res, out), fig_weights(wts, out)]
        md += ["## Figures", ""] + [f"- `{f.name}`" for f in figs if f is not None] + [""]
    (out / "affinity_tables.md").write_text("\n".join(md))
    print(f"tables -> {out / 'affinity_tables.md'}")


if __name__ == "__main__":
    main()
