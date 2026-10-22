"""
affinity.report
---------------
The paper's tables and figures of the affinity track from the outputs of :mod:`affinity.train`
(``<AFFINITY_DIR>/results.csv``, ``baseline_all.csv``, ``weights.csv``, ``predictions.parquet``).

Tables (``<OUTPUT_DIR>/affinity/affinity_tables.md``; every row of ``results.csv`` is copied to
``affinity_results.csv``):

1. headline — Kd + Ki, ``apo_relax`` · ``perturb``, best pose, test complexes of the three splits:
   one row per approach, Pearson r and the paired Δr to the calibrated interface delta;
2. the same for ``relax`` · ``perturb``, the relaxed holo complex and the crystal complex;
3. pools × aggregations on the headline split (LP-PDBBind);
4. the Kd and the Ki half of the headline test set;
5. IC50 (trained and tested on IC50 complexes);
6. the CASF-2016 core complexes outside each split's training set;
7. the unmodified interface delta on all labelled complexes (no split): pools × aggregations × label sets;
8. the learned energy weights of the headline cell (normalised to fa_atr = 0.8) next to the Rosetta defaults.

Figures: ``headline_scatter.png`` (predicted vs experimental pK on the headline cell),
``pools.png`` (Pearson r with 95 % intervals per pool × aggregation for the unmodified score,
the reweighting and ligand size), ``weights.png`` (learned non-negative weights vs the defaults).

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
from .evaluate import _ci, frame_markdown
from .models import APPROACH_LABELS, APPROACHES, REFERENCE

SPLIT_LABELS = {"lppdbbind": "LP-PDBBind", "hac": "HAC (CASF-2016 core as test)", "cleansplit": "CleanSplit"}
SPLIT_SHORT = {"lppdbbind": "LP-PDBBind", "hac": "HAC", "cleansplit": "CleanSplit"}
LABEL_SET_LABELS = {"kdki": "Kd + Ki", "kd": "Kd", "ki": "Ki", "ic50": "IC50"}
AGG_LABELS = {"best": "best pose", "mean10": "mean of 10 best", "mean": "mean (interface delta ≤ 0)", "best_hit": "best hit (oracle)"}
HEADLINE_SPLIT = "lppdbbind"

# the reference palette of the dataviz method (slots 1-3 validate all-pairs; chrome from the same file)
C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, MUTED, GRID, BASELINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


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


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def _sel(res: pd.DataFrame, **kw) -> pd.DataFrame:
    m = np.ones(len(res), dtype=bool)
    for k, v in kw.items():
        m &= res[k].eq(v).to_numpy()
    return res[m]


def wide_split_table(res: pd.DataFrame, label_set: str, pool: str, agg: str, eval_set: str = "test",
                     splits: tuple[str, ...] = SPLIT_SCHEMES) -> str:
    """Rows = approaches, columns = r [CI] and Δr [CI] per split."""
    cols, cells_by_split = [], {}
    for s in splits:
        sub = _sel(res, label_set=label_set, pool=pool, aggregation=agg, eval_set=eval_set, split=s).set_index("approach")
        if len(sub) == 0:
            continue
        cells_by_split[s] = sub
        cols += [f"{SPLIT_SHORT[s]} (n = {int(sub['n'].iloc[0]):,}) r [95 % CI]", f"{SPLIT_SHORT[s]} Δr [95 % CI]"]
    lines = ["| approach | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for a in APPROACHES:
        cells = [APPROACH_LABELS[a]]
        for s, sub in cells_by_split.items():
            if a not in sub.index:
                cells += ["—", "—"]
                continue
            r = sub.loc[a]
            cells += [_ci(r, "r"), "—" if a == REFERENCE else _ci(r, "d_r", sign=True)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def pools_table(res: pd.DataFrame, pred: pd.DataFrame, label_set: str, split: str, eval_set: str = "test",
                pools: tuple[str, ...] = TRAIN_POOLS, aggs: tuple[str, ...] = AGGREGATIONS) -> str:
    """Rows = pool × aggregation; columns = n, median RMSD and hit rate of the aggregated pose,
    unmodified r, size r, Δr of the three reweightings."""
    head = ["pool", "aggregation", "n", "pose RMSD median (Å)", "pose ≤ 2 Å (%)", "unmodified r [95 % CI]", "ligand size r",
            "reweighted Δr [95 % CI]", "non-negative Δr [95 % CI]", "reweighted + size Δr [95 % CI]"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for pool in pools:
        for agg in aggs:
            sub = _sel(res, label_set=label_set, split=split, pool=pool, aggregation=agg, eval_set=eval_set).set_index("approach")
            if len(sub) == 0:
                continue
            p = _sel(pred, label_set=label_set, split=split, pool=pool, aggregation=agg, eval_set=eval_set)
            rmsd, hit = (f"{p['rmsd_to_crystal'].median():.2f}", f"{100 * (p['rmsd_to_crystal'] <= HIT_CUTOFF).mean():.0f}") if len(p) else ("—", "—")
            cells = [pool_label(pool), AGG_LABELS[agg], f"{int(sub['n'].iloc[0]):,}", rmsd, hit, _ci(sub.loc[REFERENCE], "r"),
                     f"{sub.loc['size', 'r']:.3f}", _ci(sub.loc["reweight"], "d_r", sign=True),
                     _ci(sub.loc["reweight_nonneg"], "d_r", sign=True), _ci(sub.loc["reweight_size"], "d_r", sign=True)]
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def eval_sets_table(res: pd.DataFrame, label_set: str, split: str, pool: str, agg: str, eval_sets: dict[str, str]) -> str:
    """Rows = approaches, columns = r [CI] and Δr [CI] per evaluation set of one split."""
    cols, subs = [], {}
    for ev, name in eval_sets.items():
        sub = _sel(res, label_set=label_set, split=split, pool=pool, aggregation=agg, eval_set=ev).set_index("approach")
        if len(sub) == 0:
            continue
        subs[ev] = sub
        cols += [f"{name} (n = {int(sub['n'].iloc[0]):,}) r [95 % CI]", f"{name} Δr [95 % CI]"]
    lines = ["| approach | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for a in APPROACHES:
        cells = [APPROACH_LABELS[a]]
        for ev, sub in subs.items():
            r = sub.loc[a]
            cells += [_ci(r, "r"), "—" if a == REFERENCE else _ci(r, "d_r", sign=True)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def casf_table(res: pd.DataFrame, label_set: str, pool: str, agg: str) -> str:
    cols, subs = [], {}
    for s in SPLIT_SCHEMES:
        sub = _sel(res, label_set=label_set, split=s, pool=pool, aggregation=agg, eval_set="casf_core").set_index("approach")
        if len(sub) == 0:
            continue
        subs[s] = sub
        cols += [f"trained on {SPLIT_SHORT[s]} (n = {int(sub['n'].iloc[0]):,}) r [95 % CI]", f"{SPLIT_SHORT[s]} Δr [95 % CI]"]
    lines = ["| approach | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for a in APPROACHES:
        cells = [APPROACH_LABELS[a]]
        for s, sub in subs.items():
            r = sub.loc[a]
            cells += [_ci(r, "r"), "—" if a == REFERENCE else _ci(r, "d_r", sign=True)]
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
                cells.append(f"{_ci(sub.loc[ls], 'r')} ({int(sub.loc[ls, 'n']):,})" if ls in sub.index else "—")
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def weights_table(wts: pd.DataFrame, label_set: str, pool: str, agg: str) -> str:
    """Energy weights normalised to fa_atr = 0.8 of the reweightings per split, with the defaults."""
    sub = _sel(wts, label_set=label_set, pool=pool, aggregation=agg)
    sub = sub[sub["approach"].isin(["reweight", "reweight_nonneg", "reweight_size"]) & sub["energy_weight"].notna()]
    terms = [term_name(c) for c in sub["feature"].unique()] if len(sub) else []
    terms = list(dict.fromkeys(sub["feature"]))
    head = ["approach · split"] + terms + ["intercept (pK)", "n train"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    d = sub.drop_duplicates("feature").set_index("feature")["rosetta_default"]
    lines.append("| Rosetta default | " + " | ".join(f"{d[t]:.2f}" for t in terms) + " | — | — |")
    for a in ("reweight_nonneg", "reweight", "reweight_size"):
        for s in SPLIT_SCHEMES:
            g = sub[sub["approach"].eq(a) & sub["split"].eq(s)].set_index("feature")
            if len(g) == 0:
                continue
            lines.append(f"| {APPROACH_LABELS[a]} · {SPLIT_SHORT[s]} | " + " | ".join(f"{g.loc[t, 'weight_fa_atr_0.8']:.2f}" for t in terms)
                         + f" | {g['intercept'].iloc[0]:.2f} | {int(g['n_train'].iloc[0]):,} |")
    return "\n".join(lines)


def coef_table(wts: pd.DataFrame, label_set: str, split: str, pool: str, agg: str) -> str:
    """The raw coefficients (pK per unit of the feature) of every approach of one cell."""
    sub = _sel(wts, label_set=label_set, split=split, pool=pool, aggregation=agg)
    feats = list(dict.fromkeys(sub["feature"]))
    head = ["approach"] + feats + ["intercept"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for a in APPROACHES:
        g = sub[sub["approach"].eq(a)].set_index("feature")
        if len(g) == 0:
            continue
        lines.append(f"| {APPROACH_LABELS[a]} | " + " | ".join(f"{g.loc[f, 'coef_pk_per_unit']:.4f}" if f in g.index else "—" for f in feats)
                     + f" | {g['intercept'].iloc[0]:.2f} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def fig_headline_scatter(pred: pd.DataFrame, res: pd.DataFrame, out: Path, label_set: str = "kdki", split: str = HEADLINE_SPLIT,
                         pool: str = HEADLINE_POOL, agg: str = HEADLINE_AGG, approaches=("unmodified", "reweight", "size")) -> Path | None:
    p = _sel(pred, label_set=label_set, split=split, pool=pool, aggregation=agg, eval_set="test")
    s = _sel(res, label_set=label_set, split=split, pool=pool, aggregation=agg, eval_set="test").set_index("approach")
    if len(p) == 0:
        return None
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    fig, axes = plt.subplots(1, len(approaches), figsize=(3.4 * len(approaches), 3.6), sharex=True, sharey=True, facecolor=SURFACE)
    lo, hi = float(p["pk"].min()) - 0.5, float(p["pk"].max()) + 0.5
    for ax, a in zip(np.atleast_1d(axes), approaches):
        _style(ax)
        ax.hexbin(p["pk"], p[f"pred_{a}"], gridsize=28, cmap=cmap, mincnt=1, linewidths=0.2, edgecolors=SURFACE, extent=(lo, hi, lo, hi))
        ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        r = s.loc[a]
        ax.set_title(f"{APPROACH_LABELS[a]}\nr = {r['r']:.2f} [{r['r_lo']:.2f}, {r['r_hi']:.2f}], RMSE {r['rmse']:.2f}", fontsize=9)
        ax.set_xlabel("experimental pK")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
    np.atleast_1d(axes)[0].set_ylabel("predicted pK")
    fig.suptitle(f"{LABEL_SET_LABELS[label_set]}, {SPLIT_LABELS[split]} test complexes (n = {len(p):,}), {pool_label(pool)}, {AGG_LABELS[agg]}",
                 fontsize=9, color=INK2)
    fig.tight_layout()
    path = out / "headline_scatter.png"
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_pools(res: pd.DataFrame, out: Path, label_set: str = "kdki", split: str = HEADLINE_SPLIT, eval_set: str = "test",
              pools: tuple[str, ...] = TRAIN_POOLS, aggs: tuple[str, ...] = ("best", "mean10", "best_hit")) -> Path | None:
    series = [("unmodified", "unmodified interface delta", C_BLUE, "o"), ("reweight", "reweighted (free sign)", C_ORANGE, "s"),
              ("size", "ligand size", C_AQUA, "D")]
    sub = _sel(res, label_set=label_set, split=split, eval_set=eval_set)
    if len(sub) == 0:
        return None
    fig, axes = plt.subplots(1, len(aggs), figsize=(3.2 * len(aggs) + 1.5, 0.42 * len(pools) + 1.6), sharey=True, sharex=True, facecolor=SURFACE)
    ys = np.arange(len(pools))[::-1]
    for ax, agg in zip(np.atleast_1d(axes), aggs):
        _style(ax)
        for j, (a, label, color, marker) in enumerate(series):
            off = (j - 1) * 0.22
            for y, pool in zip(ys, pools):
                r = _sel(sub, pool=pool, aggregation=agg, approach=a)
                if len(r) == 0:
                    continue
                r = r.iloc[0]
                ax.plot([r["r_lo"], r["r_hi"]], [y + off, y + off], color=color, linewidth=2, solid_capstyle="round")
                ax.plot(r["r"], y + off, marker=marker, color=color, markersize=5.5, markeredgecolor=SURFACE, markeredgewidth=1,
                        label=label if (y == ys[0]) else None)
        ax.set_title(AGG_LABELS[agg], fontsize=9.5)
        ax.axvline(0, color=BASELINE, linewidth=0.8)
    lo = min(-0.05, float(sub[sub["pool"].isin(pools) & sub["approach"].isin([s[0] for s in series])]["r_lo"].min()) - 0.03)
    hi = float(sub[sub["pool"].isin(pools) & sub["approach"].isin([s[0] for s in series])]["r_hi"].max()) + 0.03
    np.atleast_1d(axes)[0].set_xlim(lo, hi)
    fig.supxlabel("Pearson r (predicted vs experimental pK)", fontsize=9, color=INK2, y=0.085)
    ax0 = np.atleast_1d(axes)[0]
    ax0.set_yticks(ys)
    ax0.set_yticklabels([pool_label(p) for p in pools], fontsize=8.5)
    handles, labels = ax0.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"{LABEL_SET_LABELS[label_set]}, {SPLIT_LABELS[split]} {'test' if eval_set == 'test' else eval_set} complexes; "
                 f"markers = r, bars = 95 % bootstrap interval", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    path = out / "pools.png"
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_weights(wts: pd.DataFrame, out: Path, label_set: str = "kdki", pool: str = HEADLINE_POOL, agg: str = HEADLINE_AGG) -> Path | None:
    sub = _sel(wts, label_set=label_set, pool=pool, aggregation=agg)
    sub = sub[sub["approach"].eq("reweight_nonneg")]
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
    ax.set_title(f"{LABEL_SET_LABELS[label_set]}, {pool_label(pool)}, {AGG_LABELS[agg]}", fontsize=9.5)
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
    pools_present = [p for p in TRAIN_POOLS if p in set(res["pool"])] + [p for p in res["pool"].unique() if p not in TRAIN_POOLS]

    md = ["# Affinity track — tables of `affinity.report`", "",
          f"Source: `{args.in_dir}` (`affinity.train`, {meta['n_results']} result rows, {meta['config']['n_boot']} bootstrap resamples, "
          f"seed {meta['config']['seed']}). Labelled complete PDBbind complexes: {meta['label_counts']}. "
          "Cells: Pearson r between the predicted and the experimental pK on the evaluation complexes with its percentile-bootstrap "
          "95 % interval; `Δr` the paired difference to the calibrated interface delta on the same complexes. Training complexes = "
          "train + val of the split (no hyper-parameters, nothing to select). Approaches: the interface delta linearly calibrated "
          "(`unmodified`), the ligand heavy-atom count (`size`), a free-sign least-squares reweighting of the seven raw interface "
          "terms (`reweighted`), the same with non-negative weights (`non-negative`, a Rosetta weight set), and the free-sign "
          "reweighting plus the heavy-atom count (`reweighted + size`).", ""]
    md += [f"## 1. Headline — {LABEL_SET_LABELS['kdki']}, `{HEADLINE_POOL}`, {AGG_LABELS[HEADLINE_AGG]}, test complexes of each split", "",
           wide_split_table(res, "kdki", HEADLINE_POOL, HEADLINE_AGG), ""]
    md += ["## 2. The other pools, best pose, test complexes of each split (Kd + Ki)", ""]
    for pool in ("relax_docking_perturb", "apo_relax_docking_perturb+std", "relax_docking_perturb+std", "relax_ref", "crystal"):
        if pool in set(res["pool"]):
            md += [f"**{pool_label(pool)}** (`{pool}`)", "", wide_split_table(res, "kdki", pool, "best"), ""]
    md += [f"## 3. Pools × aggregations — {LABEL_SET_LABELS['kdki']}, {SPLIT_LABELS[HEADLINE_SPLIT]} test complexes", "",
           "`pose RMSD` / `pose ≤ 2 Å`: the symmetry-corrected RMSD to the crystal ligand of the aggregated pose on the test complexes "
           "(for the means: the mean RMSD of the averaged poses).", "",
           pools_table(res, pred, "kdki", HEADLINE_SPLIT, pools=tuple(pools_present)), ""]
    md += [f"## 4. The Kd and the Ki half of the headline test set — {SPLIT_LABELS[HEADLINE_SPLIT]}, `{HEADLINE_POOL}`, {AGG_LABELS[HEADLINE_AGG]}", "",
           eval_sets_table(res, "kdki", HEADLINE_SPLIT, HEADLINE_POOL, HEADLINE_AGG,
                           {"test": "Kd + Ki", "test_kd": "Kd only", "test_ki": "Ki only"}), ""]
    if "ic50" in set(res["label_set"]):
        md += [f"## 5. IC50 (trained and tested on IC50 complexes) — `{HEADLINE_POOL}`, {AGG_LABELS[HEADLINE_AGG]}", "",
               wide_split_table(res, "ic50", HEADLINE_POOL, HEADLINE_AGG), ""]
        if "relax_ref" in set(res["pool"]):
            md += [f"**{pool_label('relax_ref')}**, IC50", "", wide_split_table(res, "ic50", "relax_ref", "best"), ""]
    md += [f"## 6. CASF-2016 core complexes outside the training set — {LABEL_SET_LABELS['kdki']}, `{HEADLINE_POOL}`, {AGG_LABELS[HEADLINE_AGG]}", "",
           "For HAC the core set *is* the test set (table 1); for the other splits the core complexes that fell into their training "
           "set are removed.", "",
           casf_table(res, "kdki", HEADLINE_POOL, HEADLINE_AGG), ""]
    if "relax_ref" in set(res["pool"]):
        md += [f"**{pool_label('relax_ref')}**, CASF-2016 core", "", casf_table(res, "kdki", "relax_ref", "best"), ""]
    md += ["## 7. The unmodified interface delta on all labelled complete complexes (no split, no fit)", "",
           "Pearson r of −`idelta_score` with pK, percentile-bootstrap interval over complexes; `pose ≤ 2 Å` the share of the "
           "aggregated poses within 2 Å of the crystal ligand (for `mean`: of the averaged poses).", "",
           baseline_table(base, tuple(p for p in list(TRAIN_POOLS) + [p for p in base["pool"].unique() if p not in TRAIN_POOLS])), ""]
    md += [f"## 8. Learned energy weights — {LABEL_SET_LABELS['kdki']}, `{HEADLINE_POOL}`, {AGG_LABELS[HEADLINE_AGG]}", "",
           "Energy weights w (pK = b − Σ w·term), normalised to fa_atr = 0.8 like the ranking weights of S5; the free-sign fit can "
           "give negative weights.", "",
           weights_table(wts, "kdki", HEADLINE_POOL, HEADLINE_AGG), "",
           f"Raw coefficients (pK per raw unit) of the {SPLIT_SHORT[HEADLINE_SPLIT]} models of the headline cell:", "",
           coef_table(wts, "kdki", HEADLINE_SPLIT, HEADLINE_POOL, HEADLINE_AGG), ""]
    if "relax_ref" in set(res["pool"]):
        md += [f"Energy weights on the relaxed holo complex (`relax_ref`):", "", weights_table(wts, "kdki", "relax_ref", "best"), ""]

    if not args.no_figures:
        figs = [fig_headline_scatter(pred, res, out), fig_pools(res, out), fig_weights(wts, out)]
        md += ["## Figures", ""] + [f"- `{f.name}`" for f in figs if f is not None] + [""]
    (out / "affinity_tables.md").write_text("\n".join(md))
    print(f"tables -> {out / 'affinity_tables.md'}")


if __name__ == "__main__":
    main()
