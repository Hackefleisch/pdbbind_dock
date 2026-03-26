"""
extract_docking_stats.py
------------------------
Orchestrator script that ties together the ``docking_analysis`` package:

1. Loads the pre-filtered PDBbind docking data (relax+perturb subset)
   and Kd index.  On the first run the filtered CSV is created
   automatically from the full zip (~20 M → ~2 M rows).
2. Generates density-scatter figures (docking score vs. binding affinity).
3. Trains linear reweighting models and plots the results.
4. Generates an illustrative energy-distribution figure.
"""

import numpy as np

from docking_analysis.constants import INDEX_PATH, OUTPUT_DIR
from docking_analysis.data import (
    load_filtered_dataframe,
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
    # --- Kd index (needed to build the filtered subset on first run) ---------
    print(f"Loading Kd values from {INDEX_PATH} …")
    df_kd = load_kd_index()

    # --- docking data (pre-filtered subset with Kd values) -------------------
    df = load_filtered_dataframe(df_kd)
    print(
        f"\n  PDB IDs : {df['pdb'].nunique():,}"
        f"  |  Rows : {len(df):,}"
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
    print(df[["pdb", "type", "total_score", "idelta_score", "kd_M", "log_kd"]].head(10))
