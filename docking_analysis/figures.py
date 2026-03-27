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

from .constants import OUTPUT_DIR, SCFX_WEIGHTS_PATH
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

    # --- Pearson r, MAE & linear fit ----------------------------------------
    r, pval = stats.pearsonr(x, y)
    mae = float(np.mean(np.abs(x - y)))
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
        f"Pearson r = {r:.3f}\nMAE = {mae:.3f}\n{p_str}\nn = {len(x):,}",
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
# Aggregated idelta vs log Kd  (generic — driven by ExperimentConfig)
# ---------------------------------------------------------------------------

# Human-readable labels for each aggregation strategy
_AGG_Y_LABELS: dict[str, str] = {
    "min":           "Min. $\\Delta\\Delta E$ score (idelta_score)",
    "mean_n":        "Mean of N lowest idelta_score",
    "filtered_mean": (
        "Mean idelta_score\n"
        "(total_score ≤ 0 & idelta_score ≤ 0 poses only)"
    ),
}

_AGG_TITLES: dict[str, str] = {
    "min":           "lowest idelta per PDB",
    "mean_n":        "mean of N lowest idelta per PDB",
    "filtered_mean": "non-positive score filter",
}


def plot_idelta_vs_logkd(
    df: pd.DataFrame,
    config: "ExperimentConfig",
    out_dir: Path = OUTPUT_DIR,
    scfx_json_path: Path = SCFX_WEIGHTS_PATH,
) -> Path:
    """
    Density-coloured scatter of log₁₀(Kd) vs. per-PDB aggregated
    docking score, driven by *config*.

    When the term selection is **not** ``all_terms``, the score is
    recomputed as the original-weight-weighted sum of only the selected
    ``raw_delta_*`` terms.  This ensures the scatter actually reflects
    the effect of combining / excluding terms.

    For ``all_terms`` the pre-computed ``idelta_score`` column is used
    directly (identical to the recomputed sum).
    """
    import json

    from .data import get_raw_delta_columns
    from .experiment import (
        ExperimentConfig,
        apply_term_selection,
        resolve_original_weights,
    )

    agg = config.aggregation
    tag = config.tag
    ts = config.term_selection
    use_original = not ts.exclude and not ts.combine  # all_terms → True

    if use_original:
        # Fast path: use the pre-computed idelta_score directly
        score_col = "idelta_score"
    else:
        # Recompute score from selected terms × original weights
        raw_delta_cols = get_raw_delta_columns(df)
        df_sel, feature_cols = apply_term_selection(df, raw_delta_cols, ts)

        # Load original scfx weights (with correct combined-term handling)
        scfx_data = {}
        if scfx_json_path.exists():
            with open(scfx_json_path, "r") as f:
                scfx_data = json.load(f)
        original_w = resolve_original_weights(feature_cols, scfx_data, ts)

        # Weighted sum → new score column
        score_col = "_recomputed_score"
        df_sel[score_col] = sum(
            df_sel[col] * original_w[col] for col in feature_cols
        )
        df = df_sel

    # Aggregate per PDB
    per_pdb = aggregate_per_pdb(
        df,
        score_col=score_col,
        strategy=agg.strategy,
        **agg.params,
    )
    per_pdb = per_pdb.rename(columns={score_col: "agg_score"})
    plot_df = join_kd_columns(per_pdb, df)

    # Log stats for filtered_mean
    if agg.strategy == "filtered_mean":
        n_valid = len(
            df.loc[(df["total_score"] <= 0) & (df[score_col] <= 0)]
        )
        print(f"  Poses used after score filter : {n_valid:,} / {len(df):,}")
        print(
            f"  PDB entries with surviving poses : "
            f"{len(plot_df):,} / {df['pdb'].nunique():,}"
        )

    y_label = _AGG_Y_LABELS.get(agg.strategy, "Aggregated score")
    title_detail = _AGG_TITLES.get(agg.strategy, agg.name)

    out_path = out_dir / f"density_scatter_{tag}.png"
    return _save_density_figure(
        x=plot_df["log_kd"].to_numpy(),
        y=plot_df["agg_score"].to_numpy(),
        y_label=y_label,
        title=(
            "Docking score vs. binding affinity\n"
            f"({title_detail}, {config.term_selection.name})"
        ),
        out_path=out_path,
    )


# ---------------------------------------------------------------------------
# Reweighting model figures
# ---------------------------------------------------------------------------
def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    config: "ExperimentConfig",
    out_dir: Path = OUTPUT_DIR,
) -> Path:
    """
    Density-coloured scatter of actual vs. predicted log₁₀(Kd) from a
    reweighting model.
    """
    from .experiment import ExperimentConfig  # deferred to avoid circular import

    tag = config.tag
    out_path = out_dir / f"scatter_predicted_{tag}.png"
    return _save_density_figure(
        x=y_true,
        y=y_pred,
        x_label=r"Actual $\log_{10}(K_d / \mathrm{M})$",
        y_label=r"Predicted $\log_{10}(K_d)$",
        title=(
            "Learned non-negative reweighting vs. binding affinity\n"
            f"({config.aggregation.name}, {config.term_selection.name})"
        ),
        out_path=out_path,
    )


def plot_weight_bar_chart(
    weights_df: pd.DataFrame,
    config: "ExperimentConfig",
    out_dir: Path = OUTPUT_DIR,
    scfx_json_path: Path = SCFX_WEIGHTS_PATH,
) -> Path:
    """
    Horizontal bar chart comparing the top 15 learned energy-term weights
    against their original values from a score-function JSON.
    """
    import json

    from .experiment import (  # deferred to avoid circular import
        ExperimentConfig,
        resolve_original_weights,
    )

    tag = config.tag

    # Load original weights for ALL terms so we can sort by them
    all_terms = weights_df.copy()
    scfx_data = {}
    if scfx_json_path.exists():
        with open(scfx_json_path, "r") as f:
            scfx_data = json.load(f)
    original_weights = resolve_original_weights(
        list(all_terms["energy_term"]), scfx_data, config.term_selection,
    )

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
    ax.set_title(
        f"Top Learned vs. Original Energy Terms\n"
        f"({config.aggregation.name}, {config.term_selection.name})",
        fontsize=11,
    )
    ax.invert_yaxis()
    ax.legend()
    fig.tight_layout()

    bar_path = out_dir / f"bar_weights_{tag}.png"
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
