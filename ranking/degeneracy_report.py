"""
ranking.degeneracy_report
-------------------------
Figures and tables of inventory S6 (Appendix A.1 energy degeneracy, A.2 per-complex oracle)
from the outputs of :mod:`ranking.degeneracy` (``<DEGENERACY_DIR>/runs.parquet``,
``poses.parquet``, ``summary.csv``; ``<DEGENERACY_DIR>_total/`` — the ``raw_*`` totals — as
the robustness check when present), plus the per-run AUCs of the ranking approaches
(``<RANKING_DIR>/<approach>_<split>/per_run.parquet``, schema 2) for the gap between the
oracle and what the global models reach on the same runs, three example complexes (hard /
median / easy by the cross-validated oracle AUC on the headline protocol) and the UMAP / t-SNE
illustrations of the per-run z-scored energy vectors.

Output (``<OUTPUT_DIR>/degeneracy/``):

``degeneracy_tables.md``            every table: the headline statistics (both distance metrics,
                                    both term sets), the oracle vs. the approaches, the examples
``nn_distance_density_<set>[_ma].png``  A.1: the three nearest-neighbour populations pooled over
                                    runs, per protocol; raw distance and distance / run threshold
``hexbin_<set>_<protocol>.png``     A.1: d_i / threshold (x) vs RMSD (y): counts, mean Δ interface delta
``degenerate_fraction_<set>.png``   A.1 per-run summary: fraction of degenerate incorrect poses
``oracle_auc_<set>.png``            A.2: ECDF of the per-run AUC (interface delta, total, oracle CV / in-sample)
``oracle_vs_degeneracy_<set>.png``  A.2: oracle CV AUC vs per-run degenerate fraction
``oracle_gap.csv``                  per run: the AUC / Success@1 of every approach and of the oracle
``examples.png``, ``examples.csv``  the three example runs (energy funnel, d_i vs RMSD, ROC, UMAP)
``embedding_pooled.png``            UMAP + t-SNE of a sample of runs (hits on top; coloured by RMSD)

Usage::

    uv run python -m ranking.degeneracy_report [--no-embedding] [--embedding-runs 100]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm, TwoSlopeNorm  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from docking_analysis.constants import DEGENERACY_DIR, OUTPUT_DIR, RANKING_DIR  # noqa: E402

from .data import HEADLINE_PROTOCOL, SPLIT_SCHEMES, extraction_dir, load_poses  # noqa: E402
from .degeneracy import METRIC_LABELS, METRICS, PERTURB_PROTOCOLS, TERM_SETS, out_dir_for, summary_markdown, zscore  # noqa: E402
from .evaluate import APPROACH_LABELS, APPROACH_ORDER, bootstrap_mean  # noqa: E402

SETS = ("pdbbind", "posebuster", "astex")
SET_LABELS = {"pdbbind": "PDBbind", "posebuster": "PoseBusters", "astex": "Astex Diverse"}
PROTO_LABELS = {"crystal_docking_perturb": "crystal · perturb", "relax_docking_perturb": "relax · perturb",
                "apo_relax_docking_perturb": "apo_relax · perturb", "crystal_docking_std": "crystal · std",
                "relax_docking_std": "relax · std", "apo_relax_docking_std": "apo_relax · std"}
SCORE_LABELS = {"idelta": "interface delta (unmodified)", "total": "total score (unmodified)",
                "oracle_cv": "per-run oracle, cross-validated", "oracle_in": "per-run oracle, in-sample"}

# palette (dataviz reference instance): categorical slots in fixed order, one sequential hue, diverging blue–grey–red
BLUE, ORANGE, AQUA, YELLOW, RED, GREY_MID, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948", "#f0efec", "#0b0b0b", "#9a9892"
POP = {"ic": ("incorrect → nearest correct (d_i)", BLUE), "cc": ("correct → nearest correct", ORANGE),
       "ii": ("incorrect → nearest incorrect", AQUA)}
SCORE_STYLE = {"idelta": (BLUE, "-"), "total": (YELLOW, "-"), "oracle_cv": (ORANGE, "-"), "oracle_in": (AQUA, "--")}
DIVERGING = LinearSegmentedColormap.from_list("delta", [RED, GREY_MID, BLUE])    # negative = decoy scores better = red
SEQ = "Blues"


def style():
    plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                         "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "axes.axisbelow": True,
                         "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight", "legend.frameon": False,
                         "xtick.color": "#52514e", "ytick.color": "#52514e", "text.color": INK, "axes.labelcolor": INK})


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
POSE_COLS = ["set", "pdb", "protocol", "seq_id", "hit", "rmsd", "idelta", "d_correct_eu", "d_correct_ma", "d_incorrect_eu",
             "d_incorrect_ma", "degenerate_eu", "degenerate_ma", "d_idelta_nn", "oracle_cv", "oracle_in"]
RUN_KEY = ["set", "pdb", "protocol"]


def load(deg_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    runs = pd.read_parquet(deg_dir / "runs.parquet")
    poses = pd.read_parquet(deg_dir / "poses.parquet", columns=POSE_COLS)
    for c in RUN_KEY:
        poses[c] = poses[c].astype(str)
    summary = pd.read_csv(deg_dir / "summary.csv")
    config = json.loads((deg_dir / "config.json").read_text())
    # the run's threshold next to every pose: distance / threshold is comparable across runs
    thr = runs.set_index(RUN_KEY)[["median_cc_eu", "median_cc_ma", "n_hit"]]
    idx = pd.MultiIndex.from_frame(poses[RUN_KEY])
    for m in METRICS:
        t = thr[f"median_cc_{m}"].reindex(idx).to_numpy()
        poses[f"rel_correct_{m}"] = (poses[f"d_correct_{m}"] / t).astype(np.float32)
        poses[f"rel_incorrect_{m}"] = (poses[f"d_incorrect_{m}"] / t).astype(np.float32)
    poses["n_hit"] = thr["n_hit"].reindex(idx).to_numpy().astype(np.int16)
    return runs, poses, summary, config


# ---------------------------------------------------------------------------
# A.1 figures
# ---------------------------------------------------------------------------
def _populations(p: pd.DataFrame, m: str, rel: bool) -> dict[str, np.ndarray]:
    pre = "rel" if rel else "d"
    a = p[p["n_hit"] >= 2]
    ic = a.loc[~a["hit"], f"{pre}_correct_{m}"].to_numpy()
    cc = a.loc[a["hit"], f"{pre}_correct_{m}"].to_numpy()
    ii = a.loc[~a["hit"], f"{pre}_incorrect_{m}"].to_numpy()
    return {"ic": ic[np.isfinite(ic)], "cc": cc[np.isfinite(cc)], "ii": ii[np.isfinite(ii)]}


def fig_density(poses: pd.DataFrame, set_name: str, protocols: list[str], metric: str, out: Path):
    fig, axes = plt.subplots(2, len(protocols), figsize=(3.4 * len(protocols), 5.4), squeeze=False)
    handles, labels = [], []
    for j, proto in enumerate(protocols):
        p = poses[(poses["set"] == set_name) & (poses["protocol"] == proto)]
        for i, rel in enumerate((False, True)):
            ax = axes[i, j]
            pops = _populations(p, metric, rel)
            hi = np.nanpercentile(np.concatenate(list(pops.values())), 99) if len(p) else 1.0
            bins = np.linspace(0, hi, 80)
            for key, (label, color) in POP.items():
                x = pops[key]
                if len(x) == 0:
                    continue
                h, e = np.histogram(x, bins=bins, range=(0, hi), density=True)       # values beyond the 99th percentile are not shown
                (line,) = ax.plot(0.5 * (e[1:] + e[:-1]), h, color=color, lw=1.6)
                if i == 0 and j == 0:
                    handles.append(line)
                    labels.append(f"{label}  (n = {len(x):,} in this panel)")
            if rel:
                ax.axvline(1.0, color=MUTED, lw=1, ls="--")
                ax.set_xlabel("distance / run threshold")
            else:
                ax.set_xlabel("nearest-neighbour distance")
            ax.set_ylabel("density" if j == 0 else "")
            if i == 0:
                ax.set_title(PROTO_LABELS[proto])
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"{SET_LABELS[set_name]}: A.1 nearest-neighbour distances in energy space ({METRIC_LABELS[metric]}), runs with ≥ 2 correct poses; "
                 "top: distance, bottom: distance / the run's median correct–correct distance (dashed = threshold)", y=1.0)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out)
    plt.close(fig)


def fig_hexbin(poses: pd.DataFrame, set_name: str, proto: str, metric: str, out: Path, far: float, cutoff: float,
               xmax: float = 4.0, ymax: float = 12.0, dclip: float = 5.0):
    p = poses[(poses["set"] == set_name) & (poses["protocol"] == proto) & (poses["n_hit"] >= 2) & ~poses["hit"]]
    x = np.clip(p[f"rel_correct_{metric}"].to_numpy(), 0, xmax)
    y = np.clip(p["rmsd"].to_numpy(), 0, ymax)
    d = np.clip(p["d_idelta_nn"].to_numpy(), -dclip, dclip)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(d)
    x, y, d = x[ok], y[ok], d[ok]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    hb = axes[0].hexbin(x, y, gridsize=60, extent=(0, xmax, 0, ymax), mincnt=1, cmap=SEQ, norm=LogNorm(), linewidths=0.2)
    fig.colorbar(hb, ax=axes[0], label="incorrect poses per cell")
    hb2 = axes[1].hexbin(x, y, C=d, reduce_C_function=np.mean, gridsize=60, extent=(0, xmax, 0, ymax), mincnt=3, cmap=DIVERGING,
                         norm=TwoSlopeNorm(vcenter=0.0, vmin=-dclip, vmax=dclip), linewidths=0.2)
    fig.colorbar(hb2, ax=axes[1], label="Δ interface delta (REU; < 0: decoy better)")
    for ax in axes:
        ax.axvline(1.0, color=INK, lw=0.8, ls="--")
        ax.axhline(far, color=INK, lw=0.8, ls=":")
        ax.axhline(cutoff, color=INK, lw=0.8, ls=":")
        ax.set_xlabel(f"d_i / run threshold ({METRIC_LABELS[metric]}; clipped at {xmax:g})")
        ax.set_ylabel("RMSD to crystal (Å)")
        ax.grid(False)
    deg_far = ((x < 1) & (y > far)).mean() * 100 if len(x) else np.nan
    axes[0].set_title("incorrect poses per cell", loc="left")
    axes[1].set_title("mean score difference to the nearest correct pose", loc="left")
    fig.suptitle(f"{SET_LABELS[set_name]} — {PROTO_LABELS[proto]}: {len(x):,} incorrect poses of the runs with ≥ 2 hits; "
                 f"{deg_far:.1f} % left of the dashed line and above {far:g} Å (energy-degenerate decoys)")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_degenerate_fraction(runs: pd.DataFrame, set_name: str, protocols: list[str], out: Path):
    fig, axes = plt.subplots(1, len(protocols), figsize=(3.2 * len(protocols), 2.9), squeeze=False)
    bins = np.linspace(0, 1, 26)
    for j, proto in enumerate(protocols):
        ax = axes[0, j]
        r = runs[(runs["set"] == set_name) & (runs["protocol"] == proto) & (runs["n_hit"] >= 2)]
        for m, color, kind in (("eu", BLUE, "bar"), ("ma", ORANGE, "step")):
            x = r[f"frac_degenerate_{m}"].dropna().to_numpy()
            if kind == "bar":
                ax.hist(x, bins=bins, color=color, alpha=0.85, label=f"{METRIC_LABELS[m]} (median {np.median(x):.2f})")
            else:
                ax.hist(x, bins=bins, histtype="step", color=color, lw=1.4, label=f"{METRIC_LABELS[m]} (median {np.median(x):.2f})")
        ax.set_title(f"{PROTO_LABELS[proto]} (n = {len(r):,} runs)")
        ax.set_ylabel("runs" if j == 0 else "")
        ax.legend(loc="upper right")
    fig.supxlabel("per run: fraction of incorrect poses nearer to a correct pose than the run's median correct–correct distance")
    fig.suptitle(f"{SET_LABELS[set_name]}: A.1 per-run summary (runs with ≥ 2 correct poses)")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# A.2 figures
# ---------------------------------------------------------------------------
def fig_oracle_auc(runs: pd.DataFrame, set_name: str, protocols: list[str], out: Path):
    fig, axes = plt.subplots(1, len(protocols), figsize=(3.2 * len(protocols), 2.9), squeeze=False)
    for j, proto in enumerate(protocols):
        ax = axes[0, j]
        r = runs[(runs["set"] == set_name) & (runs["protocol"] == proto) & (runs["oracle_folds"] > 0)]
        for key, (color, ls) in SCORE_STYLE.items():
            x = r[f"auc_{key}"].dropna().to_numpy()
            if len(x):
                ax.ecdf(x, color=color, ls=ls, lw=1.6, label=f"{SCORE_LABELS[key]} (median {np.median(x):.3f})")
        ax.set_xlim(0.4, 1.0)
        ax.set_xlabel("per-run ROC AUC (hit vs non-hit)")
        ax.set_ylabel("fraction of runs" if j == 0 else "")
        ax.set_title(f"{PROTO_LABELS[proto]} (n = {len(r):,} runs)")
        ax.legend(loc="upper left")
    fig.suptitle(f"{SET_LABELS[set_name]}: A.2 per-run AUC of the unmodified scores and of the per-run oracle (runs with ≥ 2 correct poses)")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_oracle_vs_degeneracy(runs: pd.DataFrame, set_name: str, protocols: list[str], metric: str, out: Path):
    fig, axes = plt.subplots(1, len(protocols), figsize=(3.3 * len(protocols), 3.0), squeeze=False)
    for j, proto in enumerate(protocols):
        ax = axes[0, j]
        r = runs[(runs["set"] == set_name) & (runs["protocol"] == proto) & (runs["oracle_folds"] > 0)]
        x, y = r[f"frac_degenerate_{metric}"].to_numpy(), r["auc_oracle_cv"].to_numpy()
        ok = np.isfinite(x) & np.isfinite(y)
        rho = spearmanr(x[ok], y[ok])[0] if ok.sum() > 2 else np.nan
        hb = ax.hexbin(x[ok], y[ok], gridsize=35, extent=(0, 1, 0.3, 1), mincnt=1, cmap=SEQ, norm=LogNorm(), linewidths=0.2)
        fig.colorbar(hb, ax=ax, label="runs")
        ax.set_ylabel("oracle AUC, cross-validated" if j == 0 else "")
        ax.set_title(f"{PROTO_LABELS[proto]}: Spearman ρ = {rho:+.2f} (n = {int(ok.sum()):,})")
        ax.grid(False)
    fig.supxlabel(f"per run: fraction of incorrect poses that are degenerate ({METRIC_LABELS[metric]})")
    fig.suptitle(f"{SET_LABELS[set_name]}: A.2 oracle AUC against the A.1 degenerate fraction")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# oracle vs the ranking approaches on the same runs
# ---------------------------------------------------------------------------
def collect_approaches(runs_dir: Path) -> pd.DataFrame:
    """Per (split, set, pdb, protocol): the per-run AUC and Top-1 flag of every schema-2 approach."""
    frames = []
    for p in sorted(runs_dir.glob("*/results.json")):
        r = json.loads(p.read_text())
        if r.get("schema") != 2 or not (p.parent / "per_run.parquet").exists():
            continue
        per = pd.read_parquet(p.parent / "per_run.parquet", columns=["set", "pdb", "protocol", "scorer", "auc", "top1", "n_hit"])
        per["split"] = r["split"]
        frames.append(per)
    if not frames:
        return pd.DataFrame()
    per = pd.concat(frames, ignore_index=True).drop_duplicates(["split", "set", "pdb", "protocol", "scorer"])
    wide = per.pivot_table(index=["split", "set", "pdb", "protocol"], columns="scorer", values=["auc", "top1"], aggfunc="first")
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    return wide.reset_index()


def oracle_gap(runs: pd.DataFrame, approaches: pd.DataFrame) -> pd.DataFrame:
    """The oracle columns joined to the approaches' per-run metrics (runs with ≥ 2 hits only)."""
    o = runs[runs["oracle_folds"] > 0][RUN_KEY + ["n_hit", "auc_idelta", "auc_oracle_cv", "auc_oracle_in",
                                                   "first_hit_rank_idelta", "first_hit_rank_oracle_cv", "first_hit_rank_oracle_in",
                                                   "frac_degenerate_eu"]].copy()
    o["top1_oracle_cv"] = o["first_hit_rank_oracle_cv"] < 1
    o["top1_oracle_in"] = o["first_hit_rank_oracle_in"] < 1
    if approaches.empty:
        return o
    return approaches.merge(o, on=RUN_KEY, how="inner")


def gap_markdown(gap: pd.DataFrame, split: str) -> str:
    """One table per set × protocol: mean per-run AUC and Success@1 of every scorer on the runs
    with ≥ 2 hits that the approaches were evaluated on (the test complexes of the split)."""
    if gap.empty or "split" not in gap.columns:
        return "_no schema-2 ranking runs found — the gap to the approaches is not computed_"
    g = gap[gap["split"] == split]
    scorers = [a for a in APPROACH_ORDER if f"auc_{a}" in g.columns]
    head = ["set", "protocol", "runs"] + [APPROACH_LABELS.get(a, a) for a in scorers] + ["oracle, cross-validated", "oracle, in-sample"]
    lines = [f"Split `{split}` (the approaches were trained on its training complexes). Cells: mean per-run AUC / Success@1 (%).", "",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for (s, proto), r in g.groupby(["set", "protocol"]):
        cells = [SET_LABELS.get(s, s), f"`{proto}`", f"{len(r):,}"]
        for a in scorers:
            cells.append(f"{r[f'auc_{a}'].mean():.3f} / {100 * r[f'top1_{a}'].mean():.1f}")
        cells.append(f"{r['auc_oracle_cv'].mean():.3f} / {100 * r['top1_oracle_cv'].mean():.1f}")
        cells.append(f"{r['auc_oracle_in'].mean():.3f} / {100 * r['top1_oracle_in'].mean():.1f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# example complexes
# ---------------------------------------------------------------------------
def choose_examples(runs: pd.DataFrame, set_name: str, proto: str, min_hits: int,
                    quantiles: tuple[float, float, float] = (0.0, 0.5, 1.0)) -> pd.DataFrame:
    """hard / median / easy = the runs at the given quantiles of the cross-validated oracle AUC
    among the runs with ≥ min_hits hits (default: the lowest, the middle, the highest)."""
    c = runs[(runs["set"] == set_name) & (runs["protocol"] == proto) & (runs["n_hit"] >= min_hits) & (runs["oracle_folds"] > 0)]
    c = c.sort_values(["auc_oracle_cv", "auc_idelta", "pdb"], kind="stable").reset_index(drop=True)
    picks = {name: int(round(q * (len(c) - 1))) for name, q in zip(("hard", "median", "easy"), quantiles)}
    ex = c.iloc[list(picks.values())].copy()
    ex.insert(0, "example", list(picks.keys()))
    ex["candidates"] = len(c)
    return ex


def example_table(ex: pd.DataFrame, poses: pd.DataFrame, gap: pd.DataFrame, extract_dir: Path | None, cutoff: float) -> pd.DataFrame:
    cx = pd.read_parquet(extraction_dir(ex["set"].iloc[0], extract_dir) / "complexes.parquet",
                         columns=["pdb", "ligand_n_heavy", "ligand_smiles"]).set_index("pdb")
    rows = []
    for _, r in ex.iterrows():
        p = poses[(poses["set"] == r["set"]) & (poses["pdb"] == r["pdb"]) & (poses["protocol"] == r["protocol"])]
        miss = p[~p["hit"]]
        best_hit_idelta = p.loc[p["hit"], "idelta"].min()
        row = {"example": r["example"], "pdb": r["pdb"], "protocol": r["protocol"], "ligand_heavy_atoms": int(cx.loc[r["pdb"], "ligand_n_heavy"]),
               "ligand_smiles": cx.loc[r["pdb"], "ligand_smiles"], "n_hit": int(r["n_hit"]),
               "auc_idelta": r["auc_idelta"], "auc_oracle_cv": r["auc_oracle_cv"], "auc_oracle_in": r["auc_oracle_in"],
               "first_hit_rank_idelta": r["first_hit_rank_idelta"], "first_hit_rank_oracle_cv": r["first_hit_rank_oracle_cv"],
               "first_hit_rank_oracle_in": r["first_hit_rank_oracle_in"],
               "best_rmsd": r["best_rmsd"], "best_idelta": r["best_idelta"], "best_minus_best_hit_idelta": r["best_idelta"] - best_hit_idelta,
               "best_rel_d_eu": r["best_d_eu"] / r["median_cc_eu"] if pd.notna(r["best_d_eu"]) else np.nan,
               "frac_degenerate_eu": r["frac_degenerate_eu"], "frac_degenerate_far_eu": r["frac_degenerate_far_eu"],
               "n_decoys_better_than_best_hit": int((miss["idelta"] < best_hit_idelta).sum()),
               "median_rmsd_decoys_better_than_best_hit": float(miss.loc[miss["idelta"] < best_hit_idelta, "rmsd"].median()) if (miss["idelta"] < best_hit_idelta).any() else np.nan,
               "candidates": int(r["candidates"])}
        if not gap.empty and "split" in gap.columns:
            for _, gr in gap[(gap["set"] == r["set"]) & (gap["pdb"] == r["pdb"]) & (gap["protocol"] == r["protocol"])].iterrows():
                for a in APPROACH_ORDER[1:]:
                    if f"auc_{a}" in gr and pd.notna(gr[f"auc_{a}"]):
                        row[f"auc_{a}_{gr['split']}"] = gr[f"auc_{a}"]
        rows.append(row)
    return pd.DataFrame(rows)


def examples_markdown(tab: pd.DataFrame, proto: str, min_hits: int) -> str:
    out = [f"Chosen on `{proto}` (PDBbind) among the {int(tab['candidates'].iloc[0]):,} runs with ≥ {min_hits} hits, by the cross-validated oracle AUC "
           "(ties by the interface-delta AUC): hard / median / easy = the runs at the quantiles `--example-quantiles` (default the lowest, the middle, the highest). "
           "Ranks are 0-based; `best pose` = the best-scoring pose by interface delta; `Δ to best hit` = its interface delta minus "
           "that of the best-scoring hit (negative: a decoy wins); `d / threshold` = its distance to the nearest correct pose over the "
           "run's median correct–correct distance (< 1: energy-degenerate).", "",
           "| example | pdb | ligand (heavy atoms) | hits / 300 | AUC interface delta | AUC oracle CV / in-sample | first hit rank: idelta / oracle CV / in-sample | "
           "best pose RMSD (Å) | Δ to best hit (REU) | d / threshold | decoys scoring better than the best hit (median RMSD) | fraction degenerate |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in tab.iterrows():
        nb = f"{r['n_decoys_better_than_best_hit']}" + (f" ({r['median_rmsd_decoys_better_than_best_hit']:.1f} Å)" if r["n_decoys_better_than_best_hit"] else "")
        out.append(f"| {r['example']} | `{r['pdb']}` | {r['ligand_heavy_atoms']} | {r['n_hit']} | {r['auc_idelta']:.3f} | "
                   f"{r['auc_oracle_cv']:.3f} / {r['auc_oracle_in']:.3f} | {r['first_hit_rank_idelta']:.0f} / {r['first_hit_rank_oracle_cv']:.0f} / "
                   f"{r['first_hit_rank_oracle_in']:.0f} | {r['best_rmsd']:.1f} | {r['best_minus_best_hit_idelta']:+.2f} | "
                   f"{r['best_rel_d_eu']:.2f} | {nb} | {r['frac_degenerate_eu']:.2f} |")
    extra = [c for c in tab.columns if c.startswith("auc_") and c.split("_")[-1] in SPLIT_SCHEMES]
    if extra:
        out += ["", "Per-run AUC of the global approaches on these runs (only where the complex is a test complex of the split):", "",
                "| example | pdb | " + " | ".join(extra) + " |", "|---|---|" + "---|" * len(extra)]
        for _, r in tab.iterrows():
            out.append(f"| {r['example']} | `{r['pdb']}` | " + " | ".join(f"{r[c]:.3f}" if pd.notna(r.get(c)) else "—" for c in extra) + " |")
    return "\n".join(out)


def _umap(Z: np.ndarray, seed: int, n_neighbors: int = 15) -> np.ndarray:
    import umap
    return umap.UMAP(n_neighbors=min(n_neighbors, len(Z) - 1), min_dist=0.1, random_state=seed).fit_transform(Z)


def _tsne(Z: np.ndarray, seed: int, perplexity: float = 30.0, n_jobs: int = 8) -> np.ndarray:
    from sklearn.manifold import TSNE
    return TSNE(perplexity=min(perplexity, (len(Z) - 1) / 3), init="pca", random_state=seed, n_jobs=n_jobs).fit_transform(Z)


def fig_examples(tab: pd.DataFrame, poses: pd.DataFrame, feature_cols: list[str], extract_dir: Path | None, out: Path,
                 cutoff: float, far: float, seed: int, embedding: bool = True):
    raw = load_poses("pdbbind", tab["protocol"].unique().tolist(), extract_dir, feature_cols=feature_cols)
    ncol = 4 if embedding else 3
    fig, axes = plt.subplots(len(tab), ncol, figsize=(3.3 * ncol, 3.0 * len(tab)), squeeze=False)
    for i, (_, r) in enumerate(tab.iterrows()):
        p = poses[(poses["set"] == "pdbbind") & (poses["pdb"] == r["pdb"]) & (poses["protocol"] == r["protocol"])].sort_values("seq_id")
        hit = p["hit"].to_numpy()
        rmsd, idelta = p["rmsd"].to_numpy(), p["idelta"].to_numpy()
        best = int(np.argsort(idelta, kind="stable")[0])
        title = f"{r['example']}: {r['pdb']}, {int(r['n_hit'])} hits / 300"
        # (a) energy funnel
        ax = axes[i, 0]
        ax.scatter(rmsd[~hit], idelta[~hit], s=9, color=MUTED, alpha=0.7, lw=0, label="incorrect")
        ax.scatter(rmsd[hit], idelta[hit], s=14, color=ORANGE, edgecolor=INK, lw=0.4, label="correct (≤ 2 Å)")
        ax.scatter([rmsd[best]], [idelta[best]], s=70, facecolor="none", edgecolor=RED, lw=1.4, label="best-scoring pose")
        ax.axvline(cutoff, color=INK, lw=0.8, ls=":")
        ax.set_xlabel("RMSD to crystal (Å)")
        ax.set_ylabel("interface delta (REU)")
        ax.set_title(title, loc="left")
        if i == 0:
            ax.legend(loc="upper right")
        # (b) d_i / threshold vs RMSD, coloured by the score difference
        ax = axes[i, 1]
        rel = p["rel_correct_eu"].to_numpy()
        d = np.clip(p["d_idelta_nn"].to_numpy(), -5, 5)
        sc = ax.scatter(np.clip(rel[~hit], 0, 4), rmsd[~hit], c=d[~hit], cmap=DIVERGING, norm=TwoSlopeNorm(0, -5, 5), s=10, lw=0)
        ax.scatter(np.clip(rel[hit], 0, 4), rmsd[hit], s=14, color=ORANGE, edgecolor=INK, lw=0.4)
        ax.scatter([min(rel[best], 4)], [rmsd[best]], s=70, facecolor="none", edgecolor=RED, lw=1.4)
        ax.axvline(1.0, color=INK, lw=0.8, ls="--")
        ax.axhline(far, color=INK, lw=0.8, ls=":")
        ax.set_xlabel("d / run threshold (Euclidean z-scores; clipped at 4)")
        ax.set_ylabel("RMSD to crystal (Å)")
        ax.set_title(f"{100 * r['frac_degenerate_eu']:.0f} % of incorrect poses degenerate", loc="left")
        if i == 0:
            fig.colorbar(sc, ax=ax, label="Δ interface delta to nearest correct (REU)")
        # (c) ROC within the run
        ax = axes[i, 2]
        for key, col in (("idelta", -idelta), ("oracle_cv", p["oracle_cv"].to_numpy()), ("oracle_in", p["oracle_in"].to_numpy())):
            if np.isfinite(col).all():
                fpr, tpr, _ = roc_curve(hit, col)
                color, ls = SCORE_STYLE[key]
                ax.plot(fpr, tpr, color=color, ls=ls, lw=1.6, label=f"{SCORE_LABELS[key]} ({r[f'auc_{key}']:.2f})")
        ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls=":")
        ax.set_xlabel("false-positive rate (incorrect poses ranked above)")
        ax.set_ylabel("true-positive rate (correct poses)")
        ax.set_title("ROC within the run", loc="left")
        ax.legend(loc="lower right")
        # (d) UMAP of the run's z-scored energy vectors
        if embedding:
            ax = axes[i, 3]
            rp = raw[(raw["pdb"] == r["pdb"]) & (raw["protocol"] == r["protocol"])].sort_values("seq_id")
            Z, _ = zscore(rp[feature_cols].to_numpy(np.float64))
            E = _umap(Z, seed)
            sc = ax.scatter(E[~hit, 0], E[~hit, 1], c=np.clip(rmsd[~hit], 0, 10), cmap=SEQ, vmin=0, vmax=10, s=12, lw=0)
            ax.scatter(E[hit, 0], E[hit, 1], s=16, color=ORANGE, edgecolor=INK, lw=0.4)
            ax.scatter([E[best, 0]], [E[best, 1]], s=70, facecolor="none", edgecolor=RED, lw=1.4)
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.set_title("UMAP of the 7 z-scored terms", loc="left")
            ax.grid(False)
            if i == 0:
                fig.colorbar(sc, ax=ax, label="RMSD of incorrect poses (Å)")
    fig.suptitle(f"Example runs ({PROTO_LABELS[tab['protocol'].iloc[0]]}, PDBbind): hard / median / easy by the cross-validated oracle AUC "
                 f"({', '.join(f'{e}: {a:.2f}' for e, a in zip(tab['example'], tab['auc_oracle_cv']))})")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# pooled embedding
# ---------------------------------------------------------------------------
def fig_embedding(runs: pd.DataFrame, set_name: str, proto: str, feature_cols: list[str], extract_dir: Path | None, out: Path,
                  n_runs: int, seed: int, cutoff: float, log=print):
    cand = runs[(runs["set"] == set_name) & (runs["protocol"] == proto) & (runs["n_hit"] >= 2)]["pdb"].to_numpy()
    pick = np.random.default_rng(seed).choice(cand, size=min(n_runs, len(cand)), replace=False)
    raw = load_poses(set_name, proto, extract_dir, feature_cols=feature_cols)
    raw = raw[raw["pdb"].isin(pick)].sort_values(["pdb", "seq_id"]).reset_index(drop=True)
    Zs = []
    for _, g in raw.groupby("pdb", sort=False):
        Zs.append(zscore(g[feature_cols].to_numpy(np.float64))[0])
    Z = np.vstack(Zs)
    rmsd = raw["rmsd_to_crystal"].to_numpy()
    hit = rmsd <= cutoff
    t0 = time.time()
    U = _umap(Z, seed, n_neighbors=30)
    log(f"UMAP of {len(Z):,} poses in {time.time() - t0:.0f}s")
    t0 = time.time()
    T = _tsne(Z, seed)
    log(f"t-SNE of {len(Z):,} poses in {time.time() - t0:.0f}s")
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 8.0))
    for i, (E, name) in enumerate(((U, "UMAP"), (T, "t-SNE"))):
        ax = axes[i, 0]
        ax.scatter(E[~hit, 0], E[~hit, 1], s=2, color=MUTED, alpha=0.35, lw=0, label=f"incorrect (n = {(~hit).sum():,})")
        ax.scatter(E[hit, 0], E[hit, 1], s=4, color=ORANGE, alpha=0.9, lw=0, label=f"correct ≤ {cutoff:g} Å (n = {hit.sum():,})")
        ax.set_title(f"{name}: correct poses drawn on top", loc="left")
        ax.legend(loc="upper right", markerscale=4)
        ax = axes[i, 1]
        order = np.argsort(-rmsd)                        # low RMSD drawn last
        sc = ax.scatter(E[order, 0], E[order, 1], c=np.clip(rmsd[order], 0, 10), cmap=f"{SEQ}_r", vmin=0, vmax=10, s=2, alpha=0.6, lw=0)
        fig.colorbar(sc, ax=ax, label="RMSD to crystal (Å; dark = near-native)")
        ax.set_title(f"{name}: coloured by RMSD", loc="left")
        for a in axes[i]:
            a.set_xlabel(f"{name} 1")
            a.set_ylabel(f"{name} 2")
            a.grid(False)
    fig.suptitle(f"{len(Z):,} poses of {len(pick)} random {SET_LABELS[set_name]} runs ({PROTO_LABELS[proto]}, ≥ 2 hits), "
                 "the seven interface terms z-scored per run")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return len(Z)


# ---------------------------------------------------------------------------
# the headline figure of S6
# ---------------------------------------------------------------------------
HEADLINE_SCORERS = [("idelta_score", "unmodified interface delta", BLUE, "o"),
                    ("tree", "tree (LambdaRank)", AQUA, "s"), ("reweight_random", "reweighted, random start", AQUA, "s"),
                    ("reweight_rosetta", "reweighted, Rosetta start", AQUA, "s"), ("ranker", "ranker (MLP)", AQUA, "s"),
                    ("oracle_cv", "per-complex oracle, cross-validated", ORANGE, "D"),
                    ("oracle_in", "per-complex oracle, in-sample", ORANGE, "D")]


def fig_headline(runs: pd.DataFrame, gap: pd.DataFrame, summary: pd.DataFrame, out: Path, proto: str = HEADLINE_PROTOCOL,
                 split: str = "lppdbbind", far: float = 5.0, n_boot: int = 2000, seed: int = 0, protocols: list[str] | None = None):
    """Three panels for the one sentence of S6: (a) Success@1 of the S5 approaches and of the
    per-complex oracle on the same runs — the ceiling of per-complex linear weights; (b) the
    best-scoring pose of every Top-1 failure: its distance to the nearest correct pose against
    its score difference to that pose; (c) the share of energy-degenerate decoys per protocol."""
    protocols = protocols or [p for p in PERTURB_PROTOCOLS if p in set(summary["protocol"])]
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.9), gridspec_kw={"width_ratios": [1.25, 1, 1]})

    # (a) the ceiling
    ax = axes[0]
    g = gap[(gap["split"] == split) & (gap["set"] == "pdbbind") & (gap["protocol"] == proto)] if "split" in gap.columns else gap.iloc[0:0]
    cols = [f"top1_{k}" for k, *_ in HEADLINE_SCORERS if f"top1_{k}" in g.columns]
    g = g.dropna(subset=cols)
    rows = [(k, lab, c, m) for k, lab, c, m in HEADLINE_SCORERS if f"top1_{k}" in g.columns]
    ref = g["top1_idelta_score"].to_numpy(float) if "top1_idelta_score" in g.columns else None
    for i, (k, lab, c, m) in enumerate(rows):
        flags = g[f"top1_{k}"].to_numpy(float)
        mean, lo, hi = bootstrap_mean(flags, n_boot, seed)
        y = len(rows) - 1 - i
        ax.errorbar(mean, y, xerr=[[mean - lo], [hi - mean]], fmt=m, color=c, ms=6, lw=1.2, capsize=2)
        d = "" if (ref is None or k == "idelta_score") else f"  (Δ {100 * (flags - ref).mean():+.1f})"
        ax.text(hi + 0.4, y, f"{mean:.1f}{d}", va="center", fontsize=7.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([lab for _, lab, *_ in rows][::-1])
    ax.set_xlabel("Success@1 (% of runs, 95 % bootstrap interval)")
    ax.set_title("(a) Success@1: the S5 models vs the per-complex oracle", loc="left")
    ax.grid(axis="y", visible=False)
    if len(g):
        ax.set_xlim(left=min(55, ax.get_xlim()[0]), right=ax.get_xlim()[1] + 6)

    # (b) the anatomy of the Top-1 failures
    ax = axes[1]
    f = runs[(runs["set"] == "pdbbind") & (runs["protocol"] == proto) & (runs["n_hit"] >= 2) & ~runs["best_hit"]]
    x = (f["best_d_eu"] / f["median_cc_eu"]).to_numpy()
    y = f["best_d_idelta_nn"].to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    xmax, ymin, ymax = 6.0, -10.0, 2.0
    hb = ax.hexbin(np.clip(x, 0, xmax), np.clip(y, ymin, ymax), gridsize=38, extent=(0, xmax, ymin, ymax), mincnt=1, cmap=SEQ,
                   norm=LogNorm(), linewidths=0.2)
    fig.colorbar(hb, ax=ax, label="runs per cell", pad=0.02)
    ax.axvline(1.0, color=INK, lw=0.9, ls="--")
    ax.axhline(0.0, color=INK, lw=0.9, ls=":")
    pct_deg = 100 * (x < 1).mean() if len(x) else np.nan
    ax.text(0.98, 0.04, f"{pct_deg:.0f} % degenerate (left of the line)\nmedian {np.median(x):.1f} thresholds, {np.median(y):+.1f} REU",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=MUTED, lw=0.6))
    ax.set_xlabel("distance to the nearest correct pose / run threshold")
    ax.set_ylabel("Δ interface delta to that pose (REU; < 0: decoy better)")
    ax.set_title(f"(b) the best-scoring pose of the {len(x):,} Top-1 failures", loc="left")
    ax.grid(False)

    # (c) energy-degenerate decoys are a minority
    ax = axes[2]
    s = summary[(summary["set"] == "pdbbind") & (summary["metric"] == "eu")].set_index("protocol").reindex(protocols)
    bars = [("pct_degenerate", "nearer to a correct pose than the run threshold", BLUE),
            ("pct_degenerate_far", f"… and > {far:g} Å (energy-degenerate decoy)", ORANGE),
            ("pct_degenerate_far_better", "… and scoring ≤ its nearest correct pose", AQUA)]
    w = 0.26
    xs = np.arange(len(protocols))
    for j, (col, lab, c) in enumerate(bars):
        v = s[col].to_numpy(float)
        ax.bar(xs + (j - 1) * w, v, width=w - 0.03, color=c, label=lab)
        for xi, vi in zip(xs + (j - 1) * w, v):
            ax.text(xi, vi + 0.3, f"{vi:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(xs)
    ax.set_xticklabels([p.replace("_docking_perturb", "").replace("_docking_std", " · std") for p in protocols])
    ax.set_xlabel("receptor state (perturb runs)")
    ax.set_ylabel("% of incorrect poses")
    ax.set_ylim(0, max(15, float(np.nanmax(s[[b[0] for b in bars]].to_numpy())) * 1.35))
    ax.set_title("(c) energy-degenerate decoys are a minority", loc="left")
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(axis="x", visible=False)

    fig.suptitle("S6 — the information is not in the seven interface terms: per-complex optimal weights barely move Success@1, "
                 "and the decoys that win are distinct, better-scoring states, not degenerate copies of correct poses\n"
                 f"PDBbind, runs with ≥ 2 hits. (a) the {len(g):,} {split} test runs of {PROTO_LABELS[proto]}, 95 % bootstrap intervals, Δ to the unmodified; "
                 f"(b) {PROTO_LABELS[proto]}, dashed = the run's median correct–correct distance; (c) per receptor state", y=1.06, fontsize=9)
    fig.tight_layout(w_pad=2.5)
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# robustness table (term sets × metrics)
# ---------------------------------------------------------------------------
def robustness_markdown(summaries: dict[str, pd.DataFrame]) -> str:
    head = ["terms", "distance", "set", "protocol", "% incorrect poses degenerate", "% energy-degenerate decoys (degenerate, > 5 Å)",
            "% runs: best pose a degenerate decoy", "oracle AUC CV, median", "oracle AUC in-sample, median", "Spearman ρ (oracle CV, fraction degenerate)"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for terms, s in summaries.items():
        for _, r in s.sort_values(["metric", "set", "protocol"]).iterrows():
            lines.append(f"| {terms} | {METRIC_LABELS[r['metric']]} | {r['set']} | `{r['protocol']}` | {r['pct_degenerate']:.1f} | {r['pct_degenerate_far']:.1f} | "
                         f"{r['pct_runs_best_degenerate']:.1f} | {r['auc_oracle_cv_median']:.3f} | {r['auc_oracle_in_median']:.3f} | "
                         f"{r['spearman_oracle_cv_vs_degenerate']:+.2f} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deg-dir", type=Path, default=DEGENERACY_DIR, help="output of ranking.degeneracy (delta terms)")
    ap.add_argument("--runs-dir", type=Path, default=RANKING_DIR, help="the ranking runs (per_run.parquet of the approaches)")
    ap.add_argument("--out", type=Path, default=OUTPUT_DIR / "degeneracy")
    ap.add_argument("--sets", nargs="+", default=list(SETS), choices=SETS)
    ap.add_argument("--headline-protocol", default=HEADLINE_PROTOCOL)
    ap.add_argument("--gap-split", default="lppdbbind", choices=SPLIT_SCHEMES, help="split whose approaches are tabulated against the oracle")
    ap.add_argument("--example-min-hits", type=int, default=10)
    ap.add_argument("--example-quantiles", type=float, nargs=3, default=(0.0, 0.5, 1.0),
                    help="quantiles of the CV oracle AUC that pick the hard / median / easy example")
    ap.add_argument("--embedding-runs", type=int, default=100, help="runs in the pooled UMAP / t-SNE")
    ap.add_argument("--no-embedding", action="store_true", help="skip UMAP / t-SNE (pooled and per example)")
    ap.add_argument("--headline-only", action="store_true", help="write only headline.png (the three-panel summary figure)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extract-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    style()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    runs, poses, summary, config = load(args.deg_dir)
    cutoff, far = float(config["cutoff"]), float(config["far"])
    feature_cols = config["feature_cols"]
    protocols = [p for p in config["protocols"] if p in set(runs["protocol"])]
    print(f"loaded {len(runs):,} runs, {len(poses):,} poses ({time.time() - t0:.0f}s)", flush=True)
    summaries = {config["terms"]: summary}
    for terms in TERM_SETS:                      # the other term set's summary, if it was computed
        d = out_dir_for(terms, args.deg_dir)
        if terms != config["terms"] and (d / "summary.csv").exists():
            summaries[terms] = pd.read_csv(d / "summary.csv")

    # ---- the headline figure ----------------------------------------------------------------
    gap = oracle_gap(runs, collect_approaches(args.runs_dir))
    fig_headline(runs, gap, summary, args.out / "headline.png", args.headline_protocol, args.gap_split, far, protocols=protocols)
    print(f"headline figure written ({time.time() - t0:.0f}s)", flush=True)
    if args.headline_only:
        return

    # ---- A.1 / A.2 figures ------------------------------------------------------------------
    for s in args.sets:
        if s not in set(runs["set"]):
            continue
        for m in METRICS:
            fig_density(poses, s, protocols, m, args.out / f"nn_distance_density_{s}{'' if m == 'eu' else '_ma'}.png")
        for proto in protocols:
            fig_hexbin(poses, s, proto, "eu", args.out / f"hexbin_{s}_{proto}.png", far, cutoff)
        fig_degenerate_fraction(runs, s, protocols, args.out / f"degenerate_fraction_{s}.png")
        fig_oracle_auc(runs, s, protocols, args.out / f"oracle_auc_{s}.png")
        fig_oracle_vs_degeneracy(runs, s, protocols, "eu", args.out / f"oracle_vs_degeneracy_{s}.png")
        print(f"[{s}] figures written ({time.time() - t0:.0f}s)", flush=True)

    # ---- oracle vs approaches ---------------------------------------------------------------
    gap.to_csv(args.out / "oracle_gap.csv", index=False)
    gap_md = gap_markdown(gap, args.gap_split)

    # ---- examples ---------------------------------------------------------------------------
    ex = choose_examples(runs, "pdbbind", args.headline_protocol, args.example_min_hits, tuple(args.example_quantiles))
    tab = example_table(ex, poses, gap, args.extract_dir, cutoff)
    tab.to_csv(args.out / "examples.csv", index=False)
    fig_examples(tab, poses, feature_cols, args.extract_dir, args.out / "examples.png", cutoff, far, args.seed, embedding=not args.no_embedding)
    print(f"examples: {tab[['example', 'pdb', 'auc_oracle_cv']].to_dict('records')} ({time.time() - t0:.0f}s)", flush=True)

    # ---- pooled embedding -------------------------------------------------------------------
    n_embedded = 0
    if not args.no_embedding:
        n_embedded = fig_embedding(runs, "pdbbind", args.headline_protocol, feature_cols, args.extract_dir, args.out / "embedding_pooled.png",
                                   args.embedding_runs, args.seed, cutoff)

    # ---- tables -----------------------------------------------------------------------------
    counts = config.get("counts", {})
    md = ["# Why the ranking resists repair — energy degeneracy (A.1) and the per-complex oracle (A.2)", "",
          f"Source: `ranking.degeneracy` on the {', '.join(f'`{p}`' for p in protocols)} runs of {', '.join(SET_LABELS[s] for s in args.sets)} "
          f"(terms `{config['terms']}`: {', '.join(c.replace('raw_delta_', '').replace('raw_', '') for c in feature_cols)}; correct ≤ {cutoff:g} Å, "
          f"far > {far:g} Å; z-scores per run; oracle = logistic regression per run, {config['max_folds']}-fold stratified cross-validation). "
          "Unit of analysis: one docking run (complex × protocol, 300 poses). Runs without a correct pose are excluded; runs with one correct "
          "pose have no correct–correct reference and no oracle and are excluded from the percentages. Counts per set: "
          + "; ".join(f"{SET_LABELS[s]} {c['complexes']:,} complexes, {c['runs']:,} runs, {c['runs_no_hit']:,} without a hit, {c['runs_analysed']:,} analysed"
                      for s, c in counts.items()) + ". Score differences (Δ) refer to the nearest correct pose in the Euclidean metric.", "",
          "## A.1 — headline statistics", ""]
    for m in METRICS:
        md += [summary_markdown(summary[summary["metric"] == m], far), ""]
    md += ["### Robustness: distance metric × term set", "",
           "The same statistics with the Mahalanobis distance (whitened by the run's own covariance) and, where computed, on the whole-complex "
           "`raw_*` energies of the same seven terms instead of the interface deltas.", "", robustness_markdown(summaries), "",
           "## A.2 — oracle vs. the global approaches on the same runs", "",
           "Per-run AUC (hit vs non-hit within the run) and Success@1 of the unmodified interface delta, the trained approaches of "
           "`ranking.train` and the per-run oracle, on the runs with ≥ 2 hits among the complexes the approaches were evaluated on. The "
           "cross-validated oracle is the honest upper bound of a per-complex linear scorer; the in-sample oracle the optimistic one.", "",
           gap_md, "",
           "## Example complexes", "", examples_markdown(tab, args.headline_protocol, args.example_min_hits), "",
           "## Figures", "",
           "- `headline.png`: the three-panel summary — (a) Success@1 of the S5 approaches and of the per-complex oracle on the same "
           "runs, (b) the best-scoring pose of every Top-1 failure (distance to the nearest correct pose vs score difference), "
           "(c) the share of energy-degenerate decoys per protocol.",
           "- `nn_distance_density_<set>.png` (`_ma`: Mahalanobis): density of the three nearest-neighbour populations, raw distance (top) and "
           "distance / run threshold (bottom; the dashed line is the degeneracy threshold).",
           "- `hexbin_<set>_<protocol>.png`: incorrect poses, d_i / threshold vs RMSD; left counts, right mean Δ interface delta to the nearest "
           "correct pose (red: the decoy scores better). Energy-degenerate decoys are left of the dashed line and above the 5 Å line.",
           "- `degenerate_fraction_<set>.png`: per-run fraction of degenerate incorrect poses (A.1 per-complex summary).",
           "- `oracle_auc_<set>.png`: ECDF over runs of the per-run AUC of the unmodified scores and the oracle.",
           "- `oracle_vs_degeneracy_<set>.png`: oracle CV AUC against the run's degenerate fraction.",
           "- `examples.png`: the three example runs — energy funnel, d_i vs RMSD, ROC curves, UMAP of the run's z-scored terms.",
           f"- `embedding_pooled.png`: UMAP and t-SNE of {n_embedded:,} poses ({args.embedding_runs} random `{args.headline_protocol}` runs with ≥ 2 hits), "
           "z-scored per run; correct poses on top, and coloured by RMSD." if n_embedded else "- (pooled embedding skipped)", ""]
    (args.out / "degeneracy_tables.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"\nwritten: {args.out} ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
