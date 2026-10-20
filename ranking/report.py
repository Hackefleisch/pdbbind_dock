"""
ranking.report
--------------
Comparison tables of all ranking runs (``<RANKING_DIR>/*/results.json``, written by
:mod:`ranking.train`): one table per evaluation protocol × PDBbind split × label definition
and benchmark set, with the Rosetta baselines (once) and every run's model as rows.

Output: ``<OUTPUT_DIR>/ranking/ranking_results.csv`` (long table, one row per run × set ×
protocol × scorer) and ``ranking_tables.md``.

Usage::

    uv run python -m ranking.report [--runs-dir ...] [--out ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from docking_analysis.constants import OUTPUT_DIR, RANKING_DIR

from .data import BASELINE_SCORES
from .evaluate import KS, summary_markdown

SET_LABELS = {"pdbbind": "PDBbind (test complexes of the split)", "posebuster": "PoseBusters", "astex": "Astex Diverse"}


def collect(runs_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(runs_dir.glob("*/results.json")):
        r = json.loads(p.read_text())
        c = r["config"]
        for e in r["evaluation"]:
            rows.append({"run": r["run"], "model": c["model"], "features": c["features"], "split": c["split"],
                         "train_protocols": "+".join(c["protocol"]), "rmsd_col": c["rmsd_col"], "cutoff": c["cutoff"],
                         "init": c["init"], "overlap_excluded": not c["no_exclude_overlap"], "best_epoch": r["best_epoch"],
                         "n_train_complexes": r["n_train_complexes"], **e})
    return pd.DataFrame(rows)


def tables_markdown(df: pd.DataFrame) -> str:
    out = ["# Pose-ranking models — comparison of runs", "",
           "Rates are percent of docking runs (300 poses of one complex and protocol) with a hit among the k "
           "best-ranked poses; `[95 % CI]` percentile bootstrap over runs; `Δ` paired difference to the Rosetta "
           "interface delta (`idelta_score`) on the same runs; `AUC pooled` over all poses of the set, "
           "`AUC per run` mean within-run AUC. Baselines: `idelta_score` (interface delta) and `total_score` "
           "(constraint-free total), lower is better. Model rows are labelled by their run; a run's configuration is "
           "listed in `ranking_results.csv`.", ""]
    group_cols = ["protocol", "split", "rmsd_col", "cutoff"]
    for (proto, split, rmsd_col, cutoff), g in df.groupby(group_cols, sort=True):
        out.append(f"## Protocol `{proto}` — PDBbind split `{split}`, hit: `{rmsd_col}` ≤ {cutoff:g} Å")
        out.append("")
        for set_name, gs in g.groupby("set", sort=False):
            base = gs[gs["scorer"].isin(BASELINE_SCORES)].drop_duplicates("scorer")
            models = gs[~gs["scorer"].isin(BASELINE_SCORES)].copy()
            models["scorer"] = models["scorer"] + " (" + models["run"] + ")"
            tab = pd.concat([base, models], ignore_index=True)
            out.append(f"**{SET_LABELS.get(set_name, set_name)}**\n")
            out.append(summary_markdown(tab.drop(columns=["set"]), KS))
            out.append("")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", type=Path, default=RANKING_DIR)
    ap.add_argument("--out", type=Path, default=OUTPUT_DIR / "ranking")
    args = ap.parse_args(argv)
    df = collect(args.runs_dir)
    if df.empty:
        raise SystemExit(f"no results.json under {args.runs_dir}")
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "ranking_results.csv", index=False)
    md = tables_markdown(df)
    (args.out / "ranking_tables.md").write_text(md)
    print(md)
    print(f"\nwritten: {args.out}/ranking_results.csv and ranking_tables.md ({df['run'].nunique()} runs)")


if __name__ == "__main__":
    main()
