"""
extract_docking_stats.py
------------------------
Orchestrator script that ties together the ``docking_analysis`` package:

1. Loads the PDBbind docking data and Kd index.
2. Filters for relax+perturb protocols.
3. Generates density-scatter figures (docking score vs. binding affinity).
4. Trains linear reweighting models and plots the results.
5. Generates an illustrative energy-distribution figure.
"""

import numpy as np

from docking_analysis.constants import INDEX_PATH, OUTPUT_DIR, ZIP_PATH
from docking_analysis.data import (
    filter_relax_and_perturb,
    load_docking_dataframe,
    load_kd_index,
)
from docking_analysis.figures import (
    plot_energy_distributions,
    plot_filtered_mean_idelta_vs_logkd,
    plot_mean10_idelta_vs_logkd,
    plot_min_idelta_vs_logkd,
    plot_predicted_vs_actual,
    plot_weight_bar_chart,
)
from docking_analysis.reweighting import train_reweighting_model

OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    # --- docking data --------------------------------------------------------
    print(f"Loading docking data from {ZIP_PATH} …")
    df_full = load_docking_dataframe()
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

    # --- docking-score figures -----------------------------------------------
    print("\nGenerating figures …")
    plot_min_idelta_vs_logkd(df)
    plot_mean10_idelta_vs_logkd(df)
    plot_filtered_mean_idelta_vs_logkd(df)

    # --- reweighting models + their figures ----------------------------------
    for strategy in ("min", "mean10", "filtered_mean"):
        print(f"\nTraining ML reweighting model (strategy={strategy}) ...")
        result = train_reweighting_model(df, strategy=strategy)

        y_all_true = np.concatenate([result["y_train"], result["y_test"]])
        y_all_pred = np.concatenate([result["y_pred_train"], result["y_pred_test"]])

        plot_predicted_vs_actual(y_all_true, y_all_pred, strategy=strategy)
        plot_weight_bar_chart(result["weights_df"], strategy=strategy)

    # --- illustrative energy-distribution figure -----------------------------
    print("\nGenerating illustrative energy-distribution figure …")
    plot_energy_distributions()

    return df


if __name__ == "__main__":
    df = main()
    print(f"\nResulting DataFrame shape: {df.shape}")
    print(df[["pdb", "type", "total_score", "kd_M", "log_kd"]].head(10))
