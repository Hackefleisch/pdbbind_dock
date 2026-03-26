"""
docking_analysis.figures
------------------------
All matplotlib figure-generation functions for docking analysis.

Provides:
- A shared ``density_scatter`` helper (KDE-coloured scatter + linear fit)
- Per-PDB docking-score vs. binding-affinity scatter plots
- Predicted-vs-actual scatter for the reweighting model
- Weight bar chart comparing learned vs. original energy-term weights
- Illustrative energy-distribution plot (synthetic data)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — no display required
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .constants import OUTPUT_DIR
from .data import aggregate_per_pdb, join_kd_columns


# ---------------------------------------------------------------------------
# Shared density-scatter helper
# ---------------------------------------------------------------------------
def density_scatter(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    x_label: str = r"$\log_{10}(K_d / \mathrm{M})$",
    y_label: str = "",
    title: str = "",
    equal_axes: bool = True,
    cmap: str = "viridis",
    point_size: int = 18,
) -> plt.cm.ScalarMappable:
    """
    KDE-coloured scatter plot with linear fit, y=x reference, and stats.

    Parameters
    ----------
    ax : Axes
        Target matplotlib axes.
    x, y : ndarray
        Data vectors (same length).
    x_label, y_label, title : str
        Axis / title labels.
    equal_axes : bool
        If *True*, set xlim == ylim spanning all data.
    cmap : str
        Colourmap name for the density colouring.
    point_size : int
        Marker size.

    Returns
    -------
    ScalarMappable returned by ``ax.scatter`` (useful for colorbars).
    """
    # --- KDE density --------------------------------------------------------
    xy = np.vstack([x, y])
    density = stats.gaussian_kde(xy)(xy)
    order = density.argsort()
    x_s, y_s, d_s = x[order], y[order], density[order]

    # --- Pearson r & linear fit ---------------------------------------------
    r, pval = stats.pearsonr(x, y)
    slope, intercept, *_ = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = slope * x_line + intercept

    # --- Axis limits --------------------------------------------------------
    if equal_axes:
        min_val = min(x.min(), y.min())
        max_val = max(x.max(), y.max())
        pad = (max_val - min_val) * 0.05
        axis_lim = (min_val - pad, max_val + pad)
        ax.set_xlim(axis_lim)
        ax.set_ylim(axis_lim)
        ax.plot(axis_lim, axis_lim, color="grey", lw=1.0, ls="--",
                zorder=1, label="y = x")

    # --- Draw ---------------------------------------------------------------
    sc = ax.scatter(
        x_s, y_s, c=d_s, cmap=cmap, s=point_size,
        linewidths=0, alpha=0.85, zorder=2,
    )
    ax.plot(x_line, y_line, color="crimson", lw=1.5, zorder=3,
            label="Linear fit")

    # --- Annotation ---------------------------------------------------------
    p_str = f"p-value = {pval:.2e}" if pval >= 1e-6 else "p-value ≈ 0"
    ax.text(
        0.05, 0.95,
        f"Pearson r = {r:.3f}\n{p_str}\nn = {len(x):,}",
        transform=ax.transAxes, va="top", ha="left", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8),
    )

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    return sc


# ---------------------------------------------------------------------------
# Helper: create a standard single-panel density-scatter figure
# ---------------------------------------------------------------------------
def _save_density_figure(
    x: np.ndarray,
    y: np.ndarray,
    y_label: str,
    title: str,
    out_path: Path,
    **scatter_kw,
) -> Path:
    """Create, save, and close a standard density-scatter figure."""
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = density_scatter(ax, x, y, y_label=y_label, title=title, **scatter_kw)
    fig.colorbar(sc, ax=ax, pad=0.02).set_label("Point density (KDE)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Figure saved → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Figure 1 — min idelta vs log Kd
# ---------------------------------------------------------------------------
def plot_min_idelta_vs_logkd(
    df: pd.DataFrame,
    out_path: Path = OUTPUT_DIR / "density_scatter_idelta_vs_logkd.png",
) -> Path:
    """
    Density-coloured scatter of log₁₀(Kd) vs. per-PDB **minimum**
    idelta_score.
    """
    per_pdb = aggregate_per_pdb(df, score_col="idelta_score", strategy="min")
    per_pdb = per_pdb.rename(columns={"idelta_score": "min_idelta_score"})
    plot_df = join_kd_columns(per_pdb, df)

    return _save_density_figure(
        x=plot_df["log_kd"].to_numpy(),
        y=plot_df["min_idelta_score"].to_numpy(),
        y_label="Min. $\\Delta\\Delta E$ score (idelta_score)",
        title=(
            "Docking score vs. binding affinity\n"
            "(relax + perturb protocols, lowest idelta per PDB)"
        ),
        out_path=out_path,
    )


# ---------------------------------------------------------------------------
# Figure 2 — mean of 10 lowest idelta vs log Kd
# ---------------------------------------------------------------------------
def plot_mean10_idelta_vs_logkd(
    df: pd.DataFrame,
    out_path: Path = OUTPUT_DIR / "density_scatter_mean10idelta_vs_logkd.png",
) -> Path:
    """
    Density-coloured scatter of log₁₀(Kd) vs. per-PDB **mean of the
    10 lowest** idelta_score values.
    """
    per_pdb = aggregate_per_pdb(
        df, score_col="idelta_score", strategy="mean_n", n=10,
    )
    per_pdb = per_pdb.rename(columns={"idelta_score": "mean10_idelta_score"})
    plot_df = join_kd_columns(per_pdb, df)

    return _save_density_figure(
        x=plot_df["log_kd"].to_numpy(),
        y=plot_df["mean10_idelta_score"].to_numpy(),
        y_label="Mean of 10 lowest idelta_score",
        title=(
            "Docking score vs. binding affinity\n"
            "(relax + perturb protocols, mean of 10 lowest idelta per PDB)"
        ),
        out_path=out_path,
    )


# ---------------------------------------------------------------------------
# Figure 3 — filtered mean idelta vs log Kd
# ---------------------------------------------------------------------------
def plot_filtered_mean_idelta_vs_logkd(
    df: pd.DataFrame,
    out_path: Path = OUTPUT_DIR / "density_scatter_filtered_mean_idelta_vs_logkd.png",
) -> Path:
    """
    Density-coloured scatter of log₁₀(Kd) vs. per-PDB **mean**
    idelta_score, considering only poses where both ``total_score ≤ 0``
    and ``idelta_score ≤ 0``.
    """
    per_pdb = aggregate_per_pdb(
        df, score_col="idelta_score", strategy="filtered_mean",
    )
    per_pdb = per_pdb.rename(columns={"idelta_score": "mean_idelta_filtered"})

    n_valid = len(
        df.loc[(df["total_score"] <= 0) & (df["idelta_score"] <= 0)]
    )
    print(f"  Poses used after score filter : {n_valid:,} / {len(df):,}")

    plot_df = join_kd_columns(per_pdb, df)
    n_before = df["pdb"].nunique()
    print(f"  PDB entries with surviving poses : {len(plot_df):,} / {n_before:,}")

    return _save_density_figure(
        x=plot_df["log_kd"].to_numpy(),
        y=plot_df["mean_idelta_filtered"].to_numpy(),
        y_label="Mean idelta_score\n(total_score ≤ 0 & idelta_score ≤ 0 poses only)",
        title=(
            "Docking score vs. binding affinity\n"
            "(relax + perturb, non-positive score filter)"
        ),
        out_path=out_path,
    )


# ---------------------------------------------------------------------------
# Reweighting model figures
# ---------------------------------------------------------------------------
def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    strategy: str,
    out_dir: Path = OUTPUT_DIR,
) -> Path:
    """
    Density-coloured scatter of actual vs. predicted log₁₀(Kd) from a
    reweighting model.
    """
    out_path = out_dir / f"reweighting_{strategy}_predicted_vs_actual.png"
    return _save_density_figure(
        x=y_true,
        y=y_pred,
        x_label=r"Actual $\log_{10}(K_d / \mathrm{M})$",
        y_label=r"Predicted $\log_{10}(K_d)$",
        title=(
            "Learned non-negative reweighting vs. binding affinity\n"
            f"(linear model on {strategy} features)"
        ),
        out_path=out_path,
    )


def plot_weight_bar_chart(
    weights_df: pd.DataFrame,
    strategy: str,
    out_dir: Path = OUTPUT_DIR,
    scfx_json_path: Path | None = None,
) -> Path:
    """
    Horizontal bar chart comparing the top 15 learned energy-term weights
    against their original values from a score-function JSON.
    """
    import json

    if scfx_json_path is None:
        scfx_json_path = out_dir / "scfx_weights.json"

    # Load original weights for ALL terms so we can sort by them
    all_terms = weights_df.copy()
    original_weights: dict[str, float] = {t: 0.0 for t in all_terms["energy_term"]}
    if scfx_json_path.exists():
        with open(scfx_json_path, "r") as f:
            scfx_data = json.load(f)
            for raw_term in all_terms["energy_term"]:
                clean_term = raw_term.replace("raw_delta_", "")
                original_weights[raw_term] = scfx_data.get(clean_term, 0.0)

    all_terms["original_weight"] = all_terms["energy_term"].map(original_weights)

    # Sort by original weight (descending) and take top 15
    top = (
        all_terms
        .sort_values("original_weight", key=abs, ascending=False)
        .head(15)
        .copy()
    )
    clean_labels = top["energy_term"].str.replace("^raw_delta_", "", regex=True)

    fig, ax = plt.subplots(figsize=(9, 6))

    y_pos = np.arange(len(top))
    height = 0.35

    ax.barh(y_pos + height / 2, top["weight"], height,
            label="Learned (raw features)", color="#e05252")
    ax.barh(y_pos - height / 2, top["original_weight"], height,
            label="Original (scfx.json)", color="#5282e0")

    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(clean_labels)
    ax.set_xlabel("Weight value", fontsize=11)
    ax.set_title("Top Learned vs. Original Energy Terms", fontsize=11)
    ax.invert_yaxis()
    ax.legend()
    fig.tight_layout()

    bar_path = out_dir / f"reweighting_{strategy}_weights_bar.png"
    fig.savefig(bar_path, dpi=150)
    plt.close(fig)
    print(f"  Figure saved → {bar_path}")
    return bar_path


# ---------------------------------------------------------------------------
# Illustrative energy-distribution plot (synthetic / explanatory)
# ---------------------------------------------------------------------------
def plot_energy_distributions(
    out_path: Path = OUTPUT_DIR / "energy_distributions.png",
) -> Path:
    """
    Generate a **synthetic** illustrative plot showing two skew-normal
    energy distributions for two hypothetical protein–ligand complexes.

    Purpose
    -------
    This figure demonstrates *why* the full distribution of enthalpic
    (docking) energies matters, not just the single best pose.  Two
    complexes can share the same lowest-energy docking pose yet differ
    dramatically in the spread and shape of their remaining poses.  A
    complex whose energy landscape is tightly concentrated around a deep
    minimum is more likely to be a genuine strong binder than one whose
    low-energy pose is an outlier in a broad, shallow distribution.

    The plot is purely illustrative — the distributions are made up — and
    is intended for didactic use (e.g. in a presentation or paper figure
    motivating the reweighting approach).
    """
    from scipy.stats import skewnorm

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.linspace(-12, 0, 2000)

    # Distribution 1 — broad, moderately skewed
    a1, scale1 = 3.0, 2.0
    y1_dummy = skewnorm.pdf(x, a1, loc=0, scale=scale1)
    peak1 = x[np.argmax(y1_dummy)]
    y1 = skewnorm.pdf(x, a1, loc=-6 - peak1, scale=scale1)
    y1 = (y1 / y1.max()) * 0.7

    # Distribution 2 — narrow, strongly skewed
    a2, scale2 = 6.0, 1.2
    y2_dummy = skewnorm.pdf(x, a2, loc=0, scale=scale2)
    peak2 = x[np.argmax(y2_dummy)]
    y2 = skewnorm.pdf(x, a2, loc=-8 - peak2, scale=scale2)
    y2 = (y2 / y2.max()) * 0.9

    ax.plot(x, y1, label="Distribution 1", color="#1f77b4", linewidth=2.5)
    ax.plot(x, y2, label="Distribution 2", color="#d62728", linewidth=2.5)
    ax.fill_between(x, y1, alpha=0.15, color="#1f77b4")
    ax.fill_between(x, y2, alpha=0.15, color="#d62728")

    ax.set_xlim(0, -12)
    ax.set_ylim(0, 1)

    ax.set_xlabel("Binding Free Energy", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)

    ax.axvline(-10, color="black", linestyle="--", linewidth=1.5)
    ax.text(
        -10.2, 0.5,
        "Lowest energy docking pose",
        rotation=90,
        verticalalignment="center",
        horizontalalignment="right",
        fontsize=12,
    )

    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  Figure saved → {out_path}")
    return out_path
