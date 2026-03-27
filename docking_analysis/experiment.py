"""
docking_analysis.experiment
---------------------------
Configuration dataclasses, presets, and results collection for the
reweighting experiment grid.

An experiment is defined by two orthogonal axes:

1. **Aggregation** — how per-PDB feature vectors are constructed from
   the raw pose data (e.g. min, mean-of-N, filtered mean, clustered).
2. **Term selection** — which energy terms to include and how to
   combine them (e.g. all terms, exclude certain terms, sum groups).

The :class:`ResultsMatrix` collects :class:`ExperimentResult` objects
across the grid and can export them as a summary table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════
# Configuration dataclasses
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class AggregationConfig:
    """How to aggregate raw poses into per-PDB feature vectors.

    Parameters
    ----------
    name : str
        Human-readable label used in filenames and the results table
        (e.g. ``"min"``, ``"mean10"``).
    strategy : str
        Key recognised by :func:`~docking_analysis.data.aggregate_per_pdb`
        (``"min"``, ``"mean_n"``, ``"filtered_mean"``; more to come).
    params : dict
        Extra keyword arguments forwarded to the aggregation function,
        e.g. ``{"n": 10}`` for the ``"mean_n"`` strategy.
    """

    name: str
    strategy: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TermSelectionConfig:
    """Which energy terms to include and how to combine them.

    Parameters
    ----------
    name : str
        Human-readable label (e.g. ``"all_terms"``).
    exclude : list[str]
        ``raw_delta_*`` column names to drop entirely.
    combine : dict[str, list[str]]
        Mapping from a new combined column name to the constituent
        ``raw_delta_*`` columns whose values will be summed.
        The original columns are dropped after combining.
    """

    name: str
    exclude: list[str] = field(default_factory=list)
    combine: dict[str, list[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentConfig:
    """A single experiment: aggregation × term selection."""

    aggregation: AggregationConfig
    term_selection: TermSelectionConfig

    @property
    def tag(self) -> str:
        """Short string used in filenames and log messages."""
        return f"{self.aggregation.name}__{self.term_selection.name}"


# ═══════════════════════════════════════════════════════════════════════
# Pre-built presets
# ═══════════════════════════════════════════════════════════════════════

# Aggregation presets
AGG_MIN           = AggregationConfig("min",           "min",           {})
AGG_MEAN10        = AggregationConfig("mean10",        "mean_n",        {"n": 10})
AGG_FILTERED_MEAN = AggregationConfig("filtered_mean", "filtered_mean", {})

# Term-selection presets
#
# Each preset builds cumulatively on the previous one:
#   1. all_terms       — baseline, no changes
#   2. combined_hbond  — sum hbond_bb_sc + hbond_sc → hbond_combined
#   3. no_sol_pair     — (2) + drop fa_sol and fa_pair
#   4. combined_vdw    — (3) + sum fa_atr + fa_rep → fa_vdw

TERMS_ALL = TermSelectionConfig("all_terms")

TERMS_COMBINED_HBOND = TermSelectionConfig(
    "combined_hbond",
    combine={
        "raw_delta_hbond_combined": [
            "raw_delta_hbond_bb_sc",
            "raw_delta_hbond_sc",
        ],
    },
)

TERMS_NO_SOL_PAIR = TermSelectionConfig(
    "no_sol_pair",
    exclude=[
        "raw_delta_fa_sol",
        "raw_delta_fa_pair",
    ],
    combine={
        "raw_delta_hbond_combined": [
            "raw_delta_hbond_bb_sc",
            "raw_delta_hbond_sc",
        ],
    },
)

TERMS_COMBINED_VDW = TermSelectionConfig(
    "combined_vdw",
    exclude=[
        "raw_delta_fa_sol",
        "raw_delta_fa_pair",
    ],
    combine={
        "raw_delta_hbond_combined": [
            "raw_delta_hbond_bb_sc",
            "raw_delta_hbond_sc",
        ],
        "raw_delta_fa_vdw": [
            "raw_delta_fa_atr",
            "raw_delta_fa_rep",
        ],
    },
)

_ALL_AGGREGATIONS = (AGG_MIN, AGG_MEAN10, AGG_FILTERED_MEAN)
_ALL_TERM_SELECTIONS = (TERMS_ALL, TERMS_COMBINED_HBOND, TERMS_NO_SOL_PAIR, TERMS_COMBINED_VDW)

# Default experiment grid: full cross-product (3 agg × 4 terms = 12 experiments)
DEFAULT_EXPERIMENTS: list[ExperimentConfig] = [
    ExperimentConfig(agg, terms)
    for agg in _ALL_AGGREGATIONS
    for terms in _ALL_TERM_SELECTIONS
]

# --- Clustered aggregation (requires pre-computed cluster map) ----------
AGG_CLUSTERED = AggregationConfig("clustered", "clustered", {"rmsd_cutoff": 2.0})

CLUSTERED_EXPERIMENTS: list[ExperimentConfig] = [
    ExperimentConfig(AGG_CLUSTERED, terms)
    for terms in _ALL_TERM_SELECTIONS
]


# ═══════════════════════════════════════════════════════════════════════
# Term-selection application
# ═══════════════════════════════════════════════════════════════════════

def apply_term_selection(
    df: pd.DataFrame,
    feature_cols: list[str],
    config: TermSelectionConfig,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Apply a :class:`TermSelectionConfig` to a DataFrame.

    Parameters
    ----------
    df : DataFrame
        Must contain all columns listed in *feature_cols*.
    feature_cols : list[str]
        Current set of feature column names.
    config : TermSelectionConfig
        Specifies which columns to drop and/or combine.

    Returns
    -------
    (df_modified, new_feature_cols)
        A **copy** of *df* with transformations applied, and the updated
        list of feature column names.
    """
    df = df.copy()
    cols = list(feature_cols)

    # --- exclusions ---------------------------------------------------------
    if config.exclude:
        cols = [c for c in cols if c not in config.exclude]

    # --- combinations (sum constituent columns → new column) ----------------
    for new_col, constituents in config.combine.items():
        present = [c for c in constituents if c in cols]
        if not present:
            continue
        df[new_col] = df[present].sum(axis=1)
        cols = [c for c in cols if c not in present]
        cols.append(new_col)

    return df, cols


def resolve_original_weights(
    feature_cols: list[str],
    scfx_data: dict[str, float],
    config: TermSelectionConfig,
) -> dict[str, float]:
    """
    Build a ``{feature_col: original_weight}`` mapping that correctly
    handles combined columns.

    For simple (non-combined) columns the weight is looked up directly
    in *scfx_data*.  For combined columns the weight is the **mean** of
    each constituent's original weight.  This is exact when the
    constituent weights are equal and a reasonable approximation when
    they differ.

    Parameters
    ----------
    feature_cols : list[str]
        Feature columns **after** :func:`apply_term_selection`.
    scfx_data : dict[str, float]
        Raw score-function weights keyed by clean term name
        (e.g. ``"fa_atr"``).
    config : TermSelectionConfig
        The same config used with ``apply_term_selection`` (needed to
        know which columns are combinations and their constituents).
    """
    weights: dict[str, float] = {}

    for col in feature_cols:
        if col in config.combine:
            # Combined column → mean of constituent weights
            constituents = config.combine[col]
            const_weights = [
                scfx_data.get(c.replace("raw_delta_", ""), 0.0)
                for c in constituents
            ]
            weights[col] = (
                sum(const_weights) / len(const_weights)
                if const_weights else 0.0
            )
        else:
            clean = col.replace("raw_delta_", "")
            weights[col] = scfx_data.get(clean, 0.0)

    return weights


# ═══════════════════════════════════════════════════════════════════════
# Baseline (default-weight) metrics
# ═══════════════════════════════════════════════════════════════════════

def compute_baseline_metrics(
    df: pd.DataFrame,
    config: ExperimentConfig,
    scfx_json_path: Path | None = None,
    agg_kwargs: dict | None = None,
) -> dict:
    """
    Compute Pearson r and MAE for the **original** (scfx.json) weights.

    For each experiment config, the score is computed as the
    original-weight-weighted sum of the selected ``raw_delta_*`` terms
    (or ``idelta_score`` directly for ``all_terms``), aggregated per PDB,
    and compared against ``log_kd``.

    Returns a dict with keys ``aggregation``, ``term_selection``,
    ``r``, ``mae``, ``n``.
    """
    import json

    from scipy import stats as sp_stats

    from .constants import SCFX_WEIGHTS_PATH
    from .data import aggregate_per_pdb, get_raw_delta_columns, join_kd_columns
    from .reweighting import _collapse_weighted_features

    if scfx_json_path is None:
        scfx_json_path = SCFX_WEIGHTS_PATH

    agg = config.aggregation
    ts = config.term_selection
    use_original = not ts.exclude and not ts.combine

    # ALWAYS aggregate first using idelta_score (like train_reweighting_model)
    raw_delta_cols = get_raw_delta_columns(df)
    df_sel, feature_cols = apply_term_selection(df, raw_delta_cols, ts)

    merged_kwargs = {**agg.params, **(agg_kwargs or {})}
    per_pdb = aggregate_per_pdb(
        df_sel,
        score_col="idelta_score",
        strategy=agg.strategy,
        extra_cols=feature_cols,
        **merged_kwargs,
    )

    plot_df = join_kd_columns(per_pdb, df_sel)
    plot_df = plot_df.dropna(subset=feature_cols)
    # Collapse both features and the score itself!
    plot_df = _collapse_weighted_features(plot_df, feature_cols + ["idelta_score"])

    if use_original:
        plot_df["agg_score"] = plot_df["idelta_score"]
    else:
        scfx_data = {}
        if scfx_json_path.exists():
            with open(scfx_json_path, "r") as f:
                scfx_data = json.load(f)

        original_w = resolve_original_weights(feature_cols, scfx_data, ts)
        plot_df["agg_score"] = sum(
            plot_df[col] * original_w[col] for col in feature_cols
        )

    x = plot_df["log_kd"].to_numpy()
    y = plot_df["agg_score"].to_numpy()

    r = sp_stats.pearsonr(x, y).statistic
    mae = float(np.mean(np.abs(x - y)))

    return {
        "aggregation": agg.name,
        "term_selection": ts.name,
        "r": r,
        "mae": mae,
        "n": len(x),
    }


# ═══════════════════════════════════════════════════════════════════════
# Experiment result
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ExperimentResult:
    """Outcome of a single reweighting experiment."""

    config: ExperimentConfig
    feature_cols: list[str]
    weights_df: pd.DataFrame
    r_train: float
    r_test: float
    mae_train: float
    mae_test: float
    y_train: np.ndarray
    y_test: np.ndarray
    y_pred_train: np.ndarray
    y_pred_test: np.ndarray
    model: object  # torch.nn.Module — kept generic to avoid hard import


# ═══════════════════════════════════════════════════════════════════════
# Results matrix
# ═══════════════════════════════════════════════════════════════════════

class ResultsMatrix:
    """Collects :class:`ExperimentResult` objects and produces summary tables.

    Usage::

        matrix = ResultsMatrix()
        for cfg in experiments:
            result = train_reweighting_model(df, cfg)
            matrix.add(result)
        print(matrix)
        matrix.save_csv("results.csv")
    """

    def __init__(self) -> None:
        self._results: list[ExperimentResult] = []

    def add(self, result: ExperimentResult) -> None:
        """Append an experiment result."""
        self._results.append(result)

    @property
    def results(self) -> list[ExperimentResult]:
        """All stored results (read-only view)."""
        return list(self._results)

    def to_dataframe(self) -> pd.DataFrame:
        """Return a summary DataFrame with one row per experiment."""
        rows = []
        for r in self._results:
            rows.append(
                {
                    "aggregation": r.config.aggregation.name,
                    "term_selection": r.config.term_selection.name,
                    "r_train": r.r_train,
                    "r_test": r.r_test,
                    "mae_train": r.mae_train,
                    "mae_test": r.mae_test,
                    "n_features": len(r.feature_cols),
                }
            )
        return pd.DataFrame(rows)

    def save_csv(self, path: Path) -> Path:
        """Write the summary table to *path*."""
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        print(f"  Experiment results saved → {path}")
        return path

    def __repr__(self) -> str:
        if not self._results:
            return "ResultsMatrix(empty)"
        return (
            f"ResultsMatrix ({len(self._results)} experiments)\n"
            f"{self.to_dataframe().to_string(index=False)}"
        )
