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


def train_reweighting_model(
    df: pd.DataFrame,
    strategy: str = "mean10",
    n_epochs: int = 2000,
    lr: float = 1e-3,
    train_frac: float = 0.8,
    seed: int = 42,
    out_dir: Path = OUTPUT_DIR,
) -> dict:
    """
    Train a bias-free linear layer to predict log₁₀(Kd) from per-PDB
    averaged ``raw_delta_*`` energy terms.

    Parameters
    ----------
    df : DataFrame
        Full docking frame (already joined with Kd columns).
    strategy : str
        ``"min"``            — features from the single best pose.
        ``"mean10"``         — mean features from the 10 lowest-idelta poses.
        ``"filtered_mean"``  — mean over poses with non-positive total &
                               idelta scores.
    n_epochs, lr : int, float
        Training hyper-parameters.
    train_frac : float
        Fraction of PDB entries used for training (rest is test).
    seed : int
        Random seed for reproducibility.
    out_dir : Path
        Directory for the weights CSV.

    Returns
    -------
    dict with keys:
        ``feature_cols``, ``weights_df``, ``r_train``, ``r_test``,
        ``model``, ``y_train``, ``y_test``, ``y_pred_train``,
        ``y_pred_test``
    """
    import torch
    import torch.nn as nn

    # ------------------------------------------------------------------
    # 1. Identify informative raw_delta columns
    # ------------------------------------------------------------------
    raw_delta_cols = get_raw_delta_columns(df)

    print(f"\n  Active raw_delta features : {len(raw_delta_cols)}")
    print(f"  -> {', '.join(raw_delta_cols)}")

    # ------------------------------------------------------------------
    # 2. Aggregate features per PDB
    # ------------------------------------------------------------------
    strategy_map = {
        "min":           "min",
        "mean10":        "mean_n",
        "filtered_mean": "filtered_mean",
    }
    agg_strategy = strategy_map.get(strategy)
    if agg_strategy is None:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    print(f"  Aggregating features per PDB (strategy: {strategy}) …")
    per_pdb_feats = aggregate_per_pdb(
        df,
        score_col="idelta_score",
        strategy=agg_strategy,
        n=10,
        extra_cols=raw_delta_cols,
    )

    # ------------------------------------------------------------------
    # 3. Join with log_kd
    # ------------------------------------------------------------------
    ml_df = join_kd_columns(per_pdb_feats, df)
    ml_df = ml_df.dropna(subset=raw_delta_cols)
    print(f"  PDB entries for ML  : {len(ml_df):,}")

    X_all = ml_df[raw_delta_cols].to_numpy(dtype=np.float32)
    y_all = ml_df["log_kd"].to_numpy(dtype=np.float32)

    # ------------------------------------------------------------------
    # 4. Train / test split (deterministic shuffle)
    # ------------------------------------------------------------------
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ml_df))
    n_train = int(len(idx) * train_frac)
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    X_train, X_test = X_all[train_idx], X_all[test_idx]
    y_train, y_test = y_all[train_idx], y_all[test_idx]

    # ------------------------------------------------------------------
    # 5. PyTorch linear reweighting  (no bias → pure dot product)
    # ------------------------------------------------------------------
    torch.manual_seed(seed)
    n_features = len(raw_delta_cols)

    model = nn.Linear(n_features, 1, bias=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()  # Mean Absolute Error

    X_t = torch.from_numpy(X_train)
    y_t = torch.from_numpy(y_train).unsqueeze(1)

    print(f"  Training for {n_epochs} epochs …")
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
    # 6. Evaluate
    # ------------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        y_pred_train = model(X_t).squeeze().numpy()
        y_pred_test  = model(torch.from_numpy(X_test)).squeeze().numpy()

    r_train = stats.pearsonr(y_train, y_pred_train).statistic
    r_test  = stats.pearsonr(y_test,  y_pred_test).statistic
    print(f"  Pearson r  train={r_train:.3f}  test={r_test:.3f}")

    # ------------------------------------------------------------------
    # 7. Learned weights → CSV
    # ------------------------------------------------------------------
    w_orig = model.weight.detach().numpy().flatten()

    weights_df = pd.DataFrame({
        "energy_term": raw_delta_cols,
        "weight": w_orig,
    }).sort_values("weight", key=abs, ascending=False)

    weights_path = out_dir / f"reweighting_{strategy}_weights.csv"
    weights_df.to_csv(weights_path, index=False)
    print(f"  Weights saved → {weights_path}")

    return dict(
        feature_cols=raw_delta_cols,
        weights_df=weights_df,
        r_train=r_train,
        r_test=r_test,
        model=model,
        y_train=y_train,
        y_test=y_test,
        y_pred_train=y_pred_train,
        y_pred_test=y_pred_test,
    )
