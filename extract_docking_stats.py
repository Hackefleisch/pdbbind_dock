"""
extract_docking_stats.py
------------------------
Reads pdbbind_docking_data.zip and extracts rows whose `type` column
contains BOTH the buzzwords "relax" AND "perturb".

Additionally loads Kd values from the PDBbind index file, converts them
to Molar, computes log10(Kd), and joins them onto the filtered table via
the PDB ID.

Matching types (from the full dataset):
  - relax_docking_perturb
  - apo_relax_docking_perturb
"""

import math
import re
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # headless — no display required
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT  = Path(__file__).parent
ZIP_PATH   = REPO_ROOT / "docking_stats" / "pdbbind_docking_data.zip"
INDEX_PATH = REPO_ROOT / "pdbbind_2020plus" / "INDEX_general_PL.2020R1.lst"
OUTPUT_DIR = REPO_ROOT / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Unit → Molar conversion factors
# ---------------------------------------------------------------------------
_UNIT_TO_MOLAR: dict[str, float] = {
    "mM": 1e-3,
    "uM": 1e-6,
    "nM": 1e-9,
    "pM": 1e-12,
    "fM": 1e-15,
    "M":  1.0,
}

# Regex that matches ONLY exact equalities: Kd=<value><unit>
# Skips inequalities (<, <=, >, >=, ~) and non-Kd measurements (Ki, IC50).
_KD_RE = re.compile(
    r"^Kd=(?P<value>[0-9]+(?:\.[0-9]+)?)(?P<unit>mM|uM|nM|pM|fM|M)\b"
)


# ---------------------------------------------------------------------------
# Kd index loader
# ---------------------------------------------------------------------------
def load_kd_index(index_path: Path = INDEX_PATH) -> pd.DataFrame:
    """
    Parse INDEX_general_PL.2020R1.lst and return a DataFrame with columns:
      - pdb      : 4-char PDB ID (lower-case)
      - kd_M     : Kd in Molar (float)
      - log_kd   : log10(kd_M)

    Only rows with an exact Kd= measurement are included; Ki, IC50, and
    inequality values (Kd<, Kd<=, Kd~, …) are skipped.
    """
    records: list[dict] = []

    with open(index_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Fields are whitespace-separated; column 0 = PDB, column 3 = binding data
            parts = line.split()
            if len(parts) < 4:
                continue

            pdb_id = parts[0].lower()
            binding_field = parts[3]

            m = _KD_RE.match(binding_field)
            if m is None:
                continue  # Ki, IC50, or inequality — skip

            value = float(m.group("value"))
            unit  = m.group("unit")
            kd_mol = value * _UNIT_TO_MOLAR[unit]

            records.append(
                {
                    "pdb": pdb_id,
                    "kd_M": kd_mol,
                    "log_kd": math.log10(kd_mol),
                }
            )

    df_kd = pd.DataFrame(records)
    print(f"  Kd entries parsed : {len(df_kd):,}")
    return df_kd


# ---------------------------------------------------------------------------
# Docking data loader
# ---------------------------------------------------------------------------
def load_dataframe(zip_path: Path = ZIP_PATH) -> pd.DataFrame:
    """Read the CSV from inside the zip and return a DataFrame."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_name = zf.namelist()[0]          # only one file inside
        with zf.open(csv_name) as fh:
            df = pd.read_csv(fh)
    return df


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------
def filter_relax_and_perturb(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only rows whose `type` contains BOTH 'relax' AND 'perturb'
    as individual underscore-separated buzzwords.

    Example matching types:
        relax_docking_perturb
        apo_relax_docking_perturb
    """
    buzzwords = df["type"].str.split("_")
    has_relax   = buzzwords.apply(lambda bw: "relax"   in bw)
    has_perturb = buzzwords.apply(lambda bw: "perturb" in bw)
    mask = has_relax & has_perturb
    return df.loc[mask].copy()


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_density_scatter_min_idelta_vs_logkd(
    df: pd.DataFrame,
    out_path: Path = OUTPUT_DIR / "density_scatter_idelta_vs_logkd.png",
) -> Path:
    """
    Density-coloured scatter plot of
        x = log10(Kd)  [log_kd]
        y = per-PDB minimum idelta_score

    Each point represents one PDB entry.  Point colour encodes local
    point density estimated via a 2-D Gaussian KDE.
    A line of best fit and Pearson r are annotated.
    """
    # --- aggregate: lowest idelta_score per PDB entry -----------------------
    per_pdb = (
        df.groupby("pdb", sort=False)["idelta_score"]
        .min()
        .reset_index()
        .rename(columns={"idelta_score": "min_idelta_score"})
    )

    # bring in Kd columns (one unique Kd per PDB after the earlier join)
    kd_cols = df[["pdb", "log_kd"]].drop_duplicates("pdb")
    plot_df = per_pdb.merge(kd_cols, on="pdb", how="inner").dropna(
        subset=["min_idelta_score", "log_kd"]
    )

    x = plot_df["log_kd"].to_numpy()
    y = plot_df["min_idelta_score"].to_numpy()

    # --- KDE density colouring ----------------------------------------------
    xy = np.vstack([x, y])
    density = stats.gaussian_kde(xy)(xy)
    # sort so densest points are drawn on top
    order = density.argsort()
    x_s, y_s, d_s = x[order], y[order], density[order]

    # --- Pearson r & line of best fit ---------------------------------------
    r, pval = stats.pearsonr(x, y)
    slope, intercept, *_ = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = slope * x_line + intercept

    # --- plot ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))

    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())
    pad = (max_val - min_val) * 0.05
    AXIS_LIM = (min_val - pad, max_val + pad)

    sc = ax.scatter(
        x_s, y_s,
        c=d_s,
        cmap="viridis",
        s=18,
        linewidths=0,
        alpha=0.85,
        zorder=2,
    )
    ax.plot(x_line, y_line, color="crimson", lw=1.5, zorder=3, label="Linear fit")
    # diagonal y = x reference
    ax.plot(AXIS_LIM, AXIS_LIM, color="grey", lw=1.0, ls="--", zorder=1, label="y = x")

    ax.set_xlim(AXIS_LIM)
    ax.set_ylim(AXIS_LIM)

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Point density (KDE)", fontsize=10)

    # annotation
    p_str = f"p-value = {pval:.2e}" if pval >= 1e-6 else "p-value ≈ 0"
    ax.text(
        0.05, 0.95,
        f"Pearson r = {r:.3f}\n{p_str}\nn = {len(x):,}",
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8),
    )

    ax.set_xlabel(r"$\log_{10}(K_d / \mathrm{M})$", fontsize=12)
    ax.set_ylabel("Min. $\Delta\Delta E$ score (idelta_score)", fontsize=12)
    ax.set_title(
        "Docking score vs. binding affinity\n"
        "(relax + perturb protocols, lowest idelta per PDB)",
        fontsize=12,
    )
    ax.legend(fontsize=9, loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Figure saved → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# (helper shared by both figure functions)
# ---------------------------------------------------------------------------
def _density_scatter(
    ax,
    x,
    y,
    y_label: str,
    title: str,
):
    """Internal: KDE-coloured scatter + linear fit drawn onto *ax*."""
    # Determine limits spanning all data (plus/minus a little slack)
    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())
    pad = (max_val - min_val) * 0.05
    AXIS_LIM = (min_val - pad, max_val + pad)

    xy      = np.vstack([x, y])
    density = stats.gaussian_kde(xy)(xy)
    order   = density.argsort()
    x_s, y_s, d_s = x[order], y[order], density[order]

    r, pval        = stats.pearsonr(x, y)
    slope, intercept, *_ = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = slope * x_line + intercept

    sc = ax.scatter(x_s, y_s, c=d_s, cmap="viridis", s=18,
                    linewidths=0, alpha=0.85, zorder=2)
    ax.plot(x_line, y_line, color="crimson", lw=1.5, zorder=3, label="Linear fit")
    ax.plot(AXIS_LIM, AXIS_LIM, color="grey",  lw=1.0, ls="--", zorder=1, label="y = x")

    p_str = f"p-value = {pval:.2e}" if pval >= 1e-6 else "p-value ≈ 0"
    ax.text(0.05, 0.95,
            f"Pearson r = {r:.3f}\n{p_str}\nn = {len(x):,}",
            transform=ax.transAxes, va="top", ha="left", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8))

    ax.set_xlabel(r"$\log_{10}(K_d / \mathrm{M})$", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    return sc


def plot_density_scatter_mean10_idelta_vs_logkd(
    df: pd.DataFrame,
    out_path: Path = OUTPUT_DIR / "density_scatter_mean10idelta_vs_logkd.png",
) -> Path:
    """
    Density-coloured scatter plot of log10(Kd) vs the mean of the
    10 lowest idelta_score values per PDB entry.
    """
    per_pdb = (
        df.groupby("pdb", sort=False)["idelta_score"]
        .apply(lambda s: s.nsmallest(10).mean())
        .reset_index()
        .rename(columns={"idelta_score": "mean10_idelta_score"})
    )
    kd_cols  = df[["pdb", "log_kd"]].drop_duplicates("pdb")
    plot_df  = per_pdb.merge(kd_cols, on="pdb", how="inner").dropna(
        subset=["mean10_idelta_score", "log_kd"]
    )

    x = plot_df["log_kd"].to_numpy()
    y = plot_df["mean10_idelta_score"].to_numpy()

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = _density_scatter(
        ax, x, y,
        y_label="Mean of 10 lowest idelta_score",
        title=(
            "Docking score vs. binding affinity\n"
            "(relax + perturb protocols, mean of 10 lowest idelta per PDB)"
        ),
    )
    fig.colorbar(sc, ax=ax, pad=0.02).set_label("Point density (KDE)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Figure saved → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Figure 3 — mean idelta (non-positive total & idelta only) vs log Kd
# ---------------------------------------------------------------------------
def plot_density_scatter_filtered_mean_idelta_vs_logkd(
    df: pd.DataFrame,
    out_path: Path = OUTPUT_DIR / "density_scatter_filtered_mean_idelta_vs_logkd.png",
) -> Path:
    """
    Density-coloured scatter plot of log10(Kd) vs the mean idelta_score
    per PDB entry, considering only poses where BOTH:
      - total_score <= 0
      - idelta_score <= 0

    PDB entries where no poses survive the filter are excluded.
    """
    # --- filter poses --------------------------------------------------------
    valid = df.loc[(df["total_score"] <= 0) & (df["idelta_score"] <= 0)].copy()

    # --- aggregate: mean idelta per PDB over all surviving poses -------------
    per_pdb = (
        valid.groupby("pdb", sort=False)["idelta_score"]
        .mean()
        .reset_index()
        .rename(columns={"idelta_score": "mean_idelta_filtered"})
    )

    kd_cols = df[["pdb", "log_kd"]].drop_duplicates("pdb")
    plot_df = per_pdb.merge(kd_cols, on="pdb", how="inner").dropna(
        subset=["mean_idelta_filtered", "log_kd"]
    )

    n_before = df["pdb"].nunique()
    n_after  = len(plot_df)
    print(f"  Poses used after score filter : {len(valid):,} / {len(df):,}")
    print(f"  PDB entries with surviving poses : {n_after:,} / {n_before:,}")

    x = plot_df["log_kd"].to_numpy()
    y = plot_df["mean_idelta_filtered"].to_numpy()

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = _density_scatter(
        ax, x, y,
        y_label="Mean idelta_score\n(total_score ≤ 0 & idelta_score ≤ 0 poses only)",
        title=(
            "Docking score vs. binding affinity\n"
            "(relax + perturb, non-positive score filter)"
        ),
    )
    fig.colorbar(sc, ax=ax, pad=0.02).set_label("Point density (KDE)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Figure saved \u2192 {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# ML reweighting of raw_delta_ energy terms
# ---------------------------------------------------------------------------
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
    Learn per-energy-term weights w_i such that

        ŷ = Σ_i  w_i · mean_raw_delta_i

    approaches  log10(Kd)  for each PDB entry.

    Strategy
    --------
    1. For each PDB keep the 10 poses with the lowest idelta_score.
    2. Average all raw_delta_* columns over those 10 poses → feature
       matrix X  (shape: n_pdb × n_terms).
    3. Join with log_kd → target vector y.
    4. Standardise X (zero mean, unit variance) so the gradient scales
       are comparable across terms.
    5. Fit a bias-free linear layer (PyTorch) with Adam, MSE loss.
    6. Save a weights table and a predicted-vs-actual scatter.

    Returns a dict with keys:
      feature_cols, weights_df, r_train, r_test, model
    """
    import torch
    import torch.nn as nn

    # ------------------------------------------------------------------
    # 1. Identify raw_delta columns
    # ------------------------------------------------------------------
    # Discard 'raw_delta_???' and 18 terms known to be universally zero
    # (e.g. internal constraints, disulfide, rama) based on the first 10k rows.
    zero_variance_terms = {
        "raw_delta_fa_intra_rep", "raw_delta_pro_close", "raw_delta_hbond_sr_bb",
        "raw_delta_hbond_lr_bb", "raw_delta_dslf_ss_dst", "raw_delta_dslf_cs_ang",
        "raw_delta_dslf_ss_dih", "raw_delta_dslf_ca_dih", "raw_delta_atom_pair_constraint",
        "raw_delta_coordinate_constraint", "raw_delta_angle_constraint",
        "raw_delta_dihedral_constraint", "raw_delta_rama", "raw_delta_omega",
        "raw_delta_fa_dun", "raw_delta_p_aa_pp", "raw_delta_ref", "raw_delta_chainbreak"
    }

    raw_delta_cols = [
        c for c in df.columns
        if c.startswith("raw_delta_") and "???" not in c and c not in zero_variance_terms
    ]

    print(f"\n  Removed {len(zero_variance_terms)} zero-variance raw_delta terms.")
    print(f"  Active raw_delta features : {len(raw_delta_cols)}")
    print(f"  -> {', '.join(raw_delta_cols)}")

    # ------------------------------------------------------------------
    # 2. Aggregate raw_delta_* terms per PDB based on strategy
    # ------------------------------------------------------------------
    print(f"  Aggregating features per PDB (strategy: {strategy}) …")

    if strategy == "min":
        # Features from the single pose with the lowest idelta_score
        per_pdb_feats = (
            df.loc[df.groupby("pdb", sort=False)["idelta_score"].idxmin()]
            [["pdb"] + raw_delta_cols]
            .copy()
        )
    elif strategy == "mean10":
        # Mean of the 10 lowest-idelta poses
        def mean_top10(grp: pd.DataFrame) -> pd.Series:
            best10 = grp.nsmallest(10, "idelta_score")
            return best10[raw_delta_cols].mean()

        per_pdb_feats = (
            df.groupby("pdb", sort=False, group_keys=False)
            .apply(mean_top10, include_groups=False)
            .reset_index()
        )
    elif strategy == "filtered_mean":
        # Mean over all poses where total_score <= 0 and idelta_score <= 0
        valid = df.loc[(df["total_score"] <= 0) & (df["idelta_score"] <= 0)].copy()
        per_pdb_feats = (
            valid.groupby("pdb", sort=False)[raw_delta_cols]
            .mean()
            .reset_index()
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # ------------------------------------------------------------------
    # 3. Join with log_kd
    # ------------------------------------------------------------------
    kd_cols = df[["pdb", "log_kd"]].drop_duplicates("pdb")
    ml_df = per_pdb_feats.merge(kd_cols, on="pdb", how="inner").dropna(
        subset=["log_kd"] + raw_delta_cols
    )
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
    # 6. Evaluate on train and test sets
    # ------------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        y_pred_train = model(X_t).squeeze().numpy()
        y_pred_test  = model(torch.from_numpy(X_test)).squeeze().numpy()

    r_train = stats.pearsonr(y_train, y_pred_train).statistic
    r_test  = stats.pearsonr(y_test,  y_pred_test).statistic
    print(f"  Pearson r  train={r_train:.3f}  test={r_test:.3f}")

    # ------------------------------------------------------------------
    # 7. Learned weights
    # ------------------------------------------------------------------
    w_orig = model.weight.detach().numpy().flatten()

    weights_df = pd.DataFrame({
        "energy_term": raw_delta_cols,
        "weight": w_orig,
    }).sort_values("weight", key=abs, ascending=False)

    weights_path = out_dir / f"reweighting_{strategy}_weights.csv"
    weights_df.to_csv(weights_path, index=False)
    print(f"  Weights saved → {weights_path}")

    # ------------------------------------------------------------------
    # 8. KDE Scatter: predicted vs actual log_kd  (using the full data to see the density)
    # ------------------------------------------------------------------
    # Using the whole set (train+test) since it is just an evaluation visualization
    y_pred_all = np.concatenate([y_pred_train, y_pred_test])
    y_true_all = np.concatenate([y_train, y_test])

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = _density_scatter(
        ax, y_true_all, y_pred_all,
        y_label=r"Predicted $\log_{10}(K_d)$",
        title=(
            "Learned non-negative reweighting vs. binding affinity\n"
            f"(linear model on {strategy} features)"
        ),
    )
    # Adjust x label which is overwritten by _density_scatter by default
    ax.set_xlabel(r"Actual $\log_{10}(K_d / \mathrm{M})$", fontsize=12)
    fig.colorbar(sc, ax=ax, pad=0.02).set_label("Point density (KDE)", fontsize=10)
    
    fig.tight_layout()
    scatter_path = out_dir / f"reweighting_{strategy}_predicted_vs_actual.png"
    fig.savefig(scatter_path, dpi=150)
    plt.close(fig)
    print(f"  Figure saved → {scatter_path}")

    # ------------------------------------------------------------------
    # 9. Bar chart of the top weights
    # ------------------------------------------------------------------
    top = weights_df.head(15).copy()
    # Remove the "raw_delta_" prefix for cleaner labels
    clean_labels = top["energy_term"].str.replace("^raw_delta_", "", regex=True)

    import json
    scfx_json_path = out_dir / "scfx_weights.json"
    
    # Defaults in case the JSON is missing
    original_weights = {term: 0.0 for term in top["energy_term"]}
    if scfx_json_path.exists():
        with open(scfx_json_path, "r") as f:
            scfx_data = json.load(f)
            for raw_term in top["energy_term"]:
                # strip prefix to match json keys
                clean_term = raw_term.replace("raw_delta_", "")
                original_weights[raw_term] = scfx_data.get(clean_term, 0.0)

    top["original_weight"] = top["energy_term"].map(original_weights)

    fig2, ax2 = plt.subplots(figsize=(9, 6))
    
    y_pos = np.arange(len(top))
    height = 0.35

    ax2.barh(y_pos + height/2, top["weight"], height, 
             label="Learned (raw features)", color="#e05252")
    ax2.barh(y_pos - height/2, top["original_weight"], height, 
             label="Original (scfx.json)", color="#5282e0")

    ax2.axvline(0, color="black", lw=0.8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(clean_labels)
    ax2.set_xlabel("Weight value", fontsize=11)
    ax2.set_title("Top Learned vs. Original Energy Terms", fontsize=11)
    ax2.invert_yaxis()
    ax2.legend()
    fig2.tight_layout()
    bar_path = out_dir / f"reweighting_{strategy}_weights_bar.png"
    fig2.savefig(bar_path, dpi=150)
    plt.close(fig2)
    print(f"  Figure saved → {bar_path}")

    return dict(
        feature_cols=raw_delta_cols,
        weights_df=weights_df,
        r_train=r_train,
        r_test=r_test,
        model=model,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> pd.DataFrame:
    # --- docking data --------------------------------------------------------
    print(f"Loading docking data from {ZIP_PATH} …")
    df_full = load_dataframe()
    print(f"  Total rows loaded : {len(df_full):,}")
    print(f"  All type values   : {sorted(df_full['type'].unique())}")

    print("\nFiltering for types containing both 'relax' and 'perturb' …")
    df = filter_relax_and_perturb(df_full)
    print(f"  Rows after filter : {len(df):,}")
    print(f"  Matching types    : {sorted(df['type'].unique())}")

    # --- Kd index ------------------------------------------------------------
    print(f"\nLoading Kd values from {INDEX_PATH} …")
    df_kd = load_kd_index()

    # --- join ----------------------------------------------------------------
    df = df.merge(df_kd, on="pdb", how="left")
    n_matched = df["kd_M"].notna().sum()
    print(
        f"\n  PDB IDs in filtered docking data : {df['pdb'].nunique():,}\n"
        f"  Rows with a Kd value after join  : {n_matched:,} / {len(df):,}"
    )

    # --- figures -------------------------------------------------------------
    print("\nGenerating figures \u2026")
    plot_density_scatter_min_idelta_vs_logkd(df)
    plot_density_scatter_mean10_idelta_vs_logkd(df)
    plot_density_scatter_filtered_mean_idelta_vs_logkd(df)

    print("\nTraining ML reweighting model (min idelta) ...")
    train_reweighting_model(df, strategy="min")

    print("\nTraining ML reweighting model (mean of 10 lowest idelta) ...")
    train_reweighting_model(df, strategy="mean10")

    print("\nTraining ML reweighting model (mean of filtered poses) ...")
    train_reweighting_model(df, strategy="filtered_mean")

    return df


if __name__ == "__main__":
    df = main()
    print(f"\nResulting DataFrame shape: {df.shape}")
    print(df[["pdb", "type", "total_score", "kd_M", "log_kd"]].head(10))
