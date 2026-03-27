"""
extract_docking_stats.py
------------------------
Orchestrator script that ties together the ``docking_analysis`` package:

1. Loads the pre-filtered PDBbind docking data (relax+perturb subset)
   and Kd index.  On the first run the filtered zip is created
   automatically from the full data (~20 M → ~2 M rows).
2. Runs the experiment grid (aggregation × term selection):
   - Density-scatter of aggregated idelta vs. log Kd
   - Linear reweighting model training
   - Predicted-vs-actual scatter and weight bar chart
3. Generates an illustrative energy-distribution figure.
"""

import numpy as np
import pandas as pd

from docking_analysis.constants import INDEX_PATH, OUTPUT_DIR
from docking_analysis.data import (
    load_filtered_dataframe,
    load_kd_index,
)
from docking_analysis.experiment import (
    DEFAULT_EXPERIMENTS,
    ResultsMatrix,
    compute_baseline_metrics,
)
from docking_analysis.figures import (
    plot_energy_distributions,
    plot_idelta_vs_logkd,
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

    # --- experiment grid -----------------------------------------------------
    matrix = ResultsMatrix()

    for config in DEFAULT_EXPERIMENTS:
        print(f"\n{'='*60}")
        print(f"  Experiment: {config.tag}")
        print(f"{'='*60}")

        # Density scatter (no retraining — raw aggregated idelta vs Kd)
        plot_idelta_vs_logkd(df, config=config)

        # Train reweighting model
        result = train_reweighting_model(df, config)
        matrix.add(result)

        # Reweighting figures
        y_all_true = np.concatenate([result.y_train, result.y_test])
        y_all_pred = np.concatenate([result.y_pred_train, result.y_pred_test])

        plot_predicted_vs_actual(y_all_true, y_all_pred, config=config)
        plot_weight_bar_chart(result.weights_df, config=config)

    # --- baseline (default weights) metrics -----------------------------------
    print(f"\n{'='*60}")
    print("  Baseline (Default Weights)")
    print(f"{'='*60}")
    baseline_rows = []
    for config in DEFAULT_EXPERIMENTS:
        baseline_rows.append(compute_baseline_metrics(df, config))
    baseline_df = pd.DataFrame(baseline_rows)
    print(baseline_df.to_string(index=False))
    baseline_df.to_csv(OUTPUT_DIR / "baseline_results.csv", index=False)
    print(f"  Baseline results saved → {OUTPUT_DIR / 'baseline_results.csv'}")

    # --- learned-weights results summary -------------------------------------
    print(f"\n{'='*60}")
    print("  Learned Weights Results")
    print(f"{'='*60}")
    print(matrix)
    matrix.save_csv(OUTPUT_DIR / "experiment_results.csv")

    # --- illustrative energy-distribution figure -----------------------------
    print("\nGenerating illustrative energy-distribution figure …")
    plot_energy_distributions()

    return df


if __name__ == "__main__":
    df = main()
    print(f"\nResulting DataFrame shape: {df.shape}")
    print(df[["pdb", "type", "total_score", "idelta_score", "kd_M", "log_kd"]].head(10))

