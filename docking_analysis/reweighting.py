"""
docking_analysis.reweighting
----------------------------
PyTorch-based linear reweighting of raw_delta energy terms.

Learns per-energy-term weights  w_i  such that

    ŷ = Σ_i  w_i · mean_raw_delta_i

approximates  log₁₀(Kd)  for each PDB entry.

This module is **pure ML** — it does not generate any figures.  Callers
should use :mod:`docking_analysis.figures` to visualise the results.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .constants import OUTPUT_DIR
from .data import aggregate_per_pdb, get_raw_delta_columns, join_kd_columns
from .experiment import (
    ExperimentConfig,
    ExperimentResult,
    apply_term_selection,
)


def _collapse_weighted_features(
    ml_df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Collapse multi-row PDB entries into single rows using
    ``sample_weight`` as the weighting factor.

    For current strategies (one row per PDB, weight=1.0) this is a
    no-op.  For future clustered strategies it computes the weighted
    average of feature vectors.
    """
    if "sample_weight" not in ml_df.columns:
        return ml_df

    def _weighted_mean(grp: pd.DataFrame) -> pd.Series:
        w = grp["sample_weight"].to_numpy()
        w_sum = w.sum()
        if w_sum == 0:
            return grp[feature_cols].iloc[0]
        result = {}
        for col in feature_cols:
            result[col] = np.average(grp[col].to_numpy(), weights=w)
        # Preserve log_kd (same for all rows of a PDB)
        result["log_kd"] = grp["log_kd"].iloc[0]
        return pd.Series(result)

    # Short-circuit: if every PDB has exactly one row, skip the groupby
    counts = ml_df.groupby("pdb", sort=False).size()
    if (counts == 1).all():
        return ml_df

    collapsed = (
        ml_df.groupby("pdb", sort=False, group_keys=False)
        .apply(_weighted_mean, include_groups=False)
        .reset_index()
    )
    return collapsed


def train_reweighting_model(
    df: pd.DataFrame,
    config: ExperimentConfig,
    n_epochs: int = 2000,
    lr: float = 1e-3,
    train_frac: float = 0.8,
    seed: int = 42,
    out_dir: Path = OUTPUT_DIR,
    agg_kwargs: dict | None = None,
) -> ExperimentResult:
    """
    Train a bias-free linear layer to predict log₁₀(Kd) from per-PDB
    aggregated ``raw_delta_*`` energy terms.

    Parameters
    ----------
    df : DataFrame
        Full docking frame (already joined with Kd columns).
    config : ExperimentConfig
        Specifies the aggregation strategy and term selection.
    n_epochs, lr : int, float
        Training hyper-parameters.
    train_frac : float
        Fraction of PDB entries used for training (rest is test).
    seed : int
        Random seed for reproducibility.
    out_dir : Path
        Directory for the weights CSV.
    agg_kwargs : dict | None
        Extra keyword arguments forwarded to
        :func:`~docking_analysis.data.aggregate_per_pdb` (e.g.
        ``{"cluster_map": ...}`` for the clustered strategy).

    Returns
    -------
    ExperimentResult
    """
    import torch
    import torch.nn as nn

    tag = config.tag

    # ------------------------------------------------------------------
    # 1. Identify informative raw_delta columns
    # ------------------------------------------------------------------
    raw_delta_cols = get_raw_delta_columns(df)

    # ------------------------------------------------------------------
    # 2. Apply term selection (exclude / combine)
    # ------------------------------------------------------------------
    df_sel, feature_cols = apply_term_selection(
        df, raw_delta_cols, config.term_selection,
    )

    print(f"\n  [{tag}] Active features : {len(feature_cols)}")
    print(f"  -> {', '.join(feature_cols)}")

    # ------------------------------------------------------------------
    # 3. Aggregate features per PDB
    # ------------------------------------------------------------------
    agg = config.aggregation
    merged_kwargs = {**agg.params, **(agg_kwargs or {})}
    print(f"  [{tag}] Aggregating per PDB (strategy: {agg.name}) …")
    per_pdb_feats = aggregate_per_pdb(
        df_sel,
        score_col="idelta_score",
        strategy=agg.strategy,
        extra_cols=feature_cols,
        **merged_kwargs,
    )

    # ------------------------------------------------------------------
    # 4. Join with log_kd and collapse weighted vectors
    # ------------------------------------------------------------------
    ml_df = join_kd_columns(per_pdb_feats, df_sel)
    ml_df = ml_df.dropna(subset=feature_cols)
    ml_df = _collapse_weighted_features(ml_df, feature_cols)
    print(f"  [{tag}] PDB entries for ML  : {len(ml_df):,}")

    X_all = ml_df[feature_cols].to_numpy(dtype=np.float32)
    y_all = ml_df["log_kd"].to_numpy(dtype=np.float32)

    # ------------------------------------------------------------------
    # 5. Train / test split (deterministic shuffle)
    # ------------------------------------------------------------------
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ml_df))
    n_train = int(len(idx) * train_frac)
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    X_train, X_test = X_all[train_idx], X_all[test_idx]
    y_train, y_test = y_all[train_idx], y_all[test_idx]

    # ------------------------------------------------------------------
    # 6. PyTorch linear reweighting  (no bias → pure dot product)
    # ------------------------------------------------------------------
    torch.manual_seed(seed)
    n_features = len(feature_cols)

    model = nn.Linear(n_features, 1, bias=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()  # Mean Absolute Error

    X_t = torch.from_numpy(X_train)
    y_t = torch.from_numpy(y_train).unsqueeze(1)

    print(f"  [{tag}] Training for {n_epochs} epochs …")
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        optimizer.step()

        # Enforce non-negative weights constraint
        model.weight.data.clamp_(min=0.0)

        if (epoch + 1) % 500 == 0:
            print(f"    epoch {epoch+1:>5d}  MAE={loss.item():.4f}")

    # ------------------------------------------------------------------
    # 7. Evaluate
    # ------------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        y_pred_train = model(X_t).squeeze().numpy()
        y_pred_test  = model(torch.from_numpy(X_test)).squeeze().numpy()

    r_train = stats.pearsonr(y_train, y_pred_train).statistic
    r_test  = stats.pearsonr(y_test,  y_pred_test).statistic
    mae_train = float(np.mean(np.abs(y_train - y_pred_train)))
    mae_test  = float(np.mean(np.abs(y_test  - y_pred_test)))
    print(f"  [{tag}] Pearson r  train={r_train:.3f}  test={r_test:.3f}")
    print(f"  [{tag}] MAE        train={mae_train:.4f}  test={mae_test:.4f}")

    # ------------------------------------------------------------------
    # 8. Learned weights → CSV
    # ------------------------------------------------------------------
    w_orig = model.weight.detach().numpy().flatten()

    weights_df = pd.DataFrame({
        "energy_term": feature_cols,
        "weight": w_orig,
    }).sort_values("weight", key=abs, ascending=False)

    weights_path = out_dir / f"weights_{tag}.csv"
    weights_df.to_csv(weights_path, index=False)
    print(f"  [{tag}] Weights saved → {weights_path}")

    return ExperimentResult(
        config=config,
        feature_cols=feature_cols,
        weights_df=weights_df,
        r_train=r_train,
        r_test=r_test,
        mae_train=mae_train,
        mae_test=mae_test,
        y_train=y_train,
        y_test=y_test,
        y_pred_train=y_pred_train,
        y_pred_test=y_pred_test,
        model=model,
    )
