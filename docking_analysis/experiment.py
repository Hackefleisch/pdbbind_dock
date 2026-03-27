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
TERMS_ALL = TermSelectionConfig("all_terms")

# Default experiment grid (reproduces current behaviour)
DEFAULT_EXPERIMENTS: list[ExperimentConfig] = [
    ExperimentConfig(agg, TERMS_ALL)
    for agg in (AGG_MIN, AGG_MEAN10, AGG_FILTERED_MEAN)
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
