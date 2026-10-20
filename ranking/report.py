"""
ranking.report
--------------
The paper's tables over all runs of :mod:`ranking.train` (``<RANKING_DIR>/*/results.json`` with
``schema`` 2): per evaluation protocol — the headline ``apo_relax_docking_perturb`` first — and
per benchmark set, **one row per approach** (unmodified interface delta, tree, reweighted from a
random start, reweighted from the Rosetta start, ranker) and **one column per k** (1, 5, 10)
with the paired Δ Success@k to the unmodified ranking, for each of the three splits.

Output: ``<OUTPUT_DIR>/ranking/ranking_tables.md`` and ``ranking_results.csv`` (one row per run ×
protocol × set × scorer with every metric, including the first-hit-rank statistics).

Usage::

    uv run python -m ranking.report [--runs-dir ...] [--out ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from docking_analysis.constants import OUTPUT_DIR, RANKING_DIR

from .data import BASELINE_SCORE, HEADLINE_PROTOCOL, SPLIT_SCHEMES
from .evaluate import APPROACH_LABELS, APPROACH_ORDER, KS, approach_table_markdown, rank_shift, rank_shift_markdown

SCHEMA = 2
SET_LABELS = {"pdbbind": "PDBbind test complexes of the split", "posebuster": "PoseBusters", "astex": "Astex Diverse"}
SPLIT_LABELS = {"lppdbbind": "LP-PDBBind", "hac": "HAC (CASF-2016 core as test)", "cleansplit": "CleanSplit"}


def collect(runs_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(runs_dir.glob("*/results.json")):
        r = json.loads(p.read_text())
        if r.get("schema") != SCHEMA:
            continue
        c = r["config"]
        shifts = None
        if (p.parent / "per_run.parquet").exists():
            per_run = pd.read_parquet(p.parent / "per_run.parquet")
            shifts = pd.concat([rank_shift(g).assign(set=s_, protocol=pr) for (s_, pr), g in per_run.groupby(["set", "protocol"])],
                               ignore_index=True).set_index(["set", "protocol", "scorer"])
        for e in r["evaluation"]:
            row = {"run": r["run"], "approach": r["approach"], "model": r["model"], "split": r["split"],
                   "train_protocols": "+".join(c["protocol"]), "group_by": c.get("group_by"), "init": c.get("init"),
                   "rmsd_col": c["rmsd_col"], "cutoff": c["cutoff"], "best": r["best"],
                   "n_train_complexes": r["n_train_complexes"], **e}
            if shifts is not None and (e["set"], e["protocol"], e["scorer"]) in shifts.index:
                sh = shifts.loc[(e["set"], e["protocol"], e["scorer"])]
                row.update({k: sh[k] for k in ("runs_hit_both", "mrr", "d_mrr", "earlier", "same", "later", "log2_rank_shift")})
            rows.append(row)
    return pd.DataFrame(rows)


def _order(df: pd.DataFrame) -> pd.DataFrame:
    o = {a: i for i, a in enumerate(APPROACH_ORDER)}
    sp = {s: i for i, s in enumerate(SPLIT_SCHEMES)}
    return df.assign(_a=df["scorer"].map(lambda a: o.get(a, 99)), _s=df["split"].map(lambda s: sp.get(s, 99))).sort_values(["_s", "_a"], kind="stable")


def headline_markdown(df: pd.DataFrame, protocol: str) -> str:
    """One row per approach, one block of Δ@1 / Δ@5 / Δ@10 columns per split (PDBbind test complexes)."""
    sub = df[(df["protocol"] == protocol) & (df["set"] == "pdbbind")]
    splits = [s for s in SPLIT_SCHEMES if s in set(sub["split"])]
    head = ["approach"] + [f"{SPLIT_LABELS[s]} (n = {int(sub[sub.split == s]['runs'].iloc[0])}) Δ@{k}" for s in splits for k in KS]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    approaches = [a for a in APPROACH_ORDER if a in set(sub["scorer"])]
    for a in approaches:
        cells = [APPROACH_LABELS.get(a, a)]
        for s in splits:
            r = sub[(sub.split == s) & (sub.scorer == a)]
            for k in KS:
                if r.empty:
                    cells.append("—")
                elif a == BASELINE_SCORE:
                    cells.append(f"{r[f'top{k}'].iloc[0]:.1f} (absolute)")
                else:
                    cells.append(f"{r[f'd_top{k}'].iloc[0]:+.1f} [{r[f'd_top{k}_lo'].iloc[0]:+.1f}, {r[f'd_top{k}_hi'].iloc[0]:+.1f}]")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def tables_markdown(df: pd.DataFrame) -> str:
    out = ["# Pose ranking — approaches vs. the unmodified interface delta", "",
           "One row per approach, one column per k. Rates are percent of docking runs (300 poses of one complex and protocol) "
           "with a hit (`rmsd_to_crystal` ≤ 2 Å, symmetry-corrected) among the k best-ranked poses. The unmodified row is the "
           "absolute Success@k of the Rosetta interface delta; every other row is the paired difference to it on the same runs "
           "with a percentile-bootstrap 95 % interval. Approaches: `tree` LightGBM LambdaRank on the seven raw interface terms; "
           "`reweighted` non-negative reweighting of the seven raw interface terms from a random start or from the Rosetta weights "
           "(`fa_atr` fixed at 0.8 as the gauge; the learned weights are Rosetta weights, see `weights.wts` of the run); "
           "`ranker` Luca Friebe's MLP on robust-scaled features with his BCE + InfoNCE loss. All are trained on the pooled "
           "relax · perturb + apo_relax · perturb runs of the PDBbind training complexes of the split (buckets per complex and "
           "receptor state; the Astex-overlapping entries removed) and selected on the validation complexes. "
           "`MRR` mean reciprocal rank of the first hit (runs without a hit count 0); `earlier / same / later` percent of the runs "
           "with a hit under both scorers whose first hit moves to a better / the same / a worse rank; `Δ log2(rank+1)` the mean "
           "log-rank shift (negative = earlier). Every number is in `ranking_results.csv`.", ""]
    protocols = [HEADLINE_PROTOCOL] + sorted(p for p in df["protocol"].unique() if p != HEADLINE_PROTOCOL)
    for proto in protocols:
        tag = " — headline" if proto == HEADLINE_PROTOCOL else ""
        out.append(f"## Evaluation protocol `{proto}`{tag}")
        out.append("")
        if (df["protocol"] == proto).any() and ((df["protocol"] == proto) & (df["set"] == "pdbbind")).any():
            out.append("**PDBbind test complexes, per split** (the headline table for the apo_relax protocol)\n")
            out.append(headline_markdown(df, proto))
            out.append("")
        for set_name in [s for s in ("pdbbind", "posebuster", "astex") if s in set(df["set"])]:
            sub = _order(df[(df["protocol"] == proto) & (df["set"] == set_name)])
            if sub.empty:
                continue
            if set_name == "pdbbind":
                out.append(f"**{SET_LABELS[set_name]}**, per split\n")
                for s in [s for s in SPLIT_SCHEMES if s in set(sub["split"])]:
                    block = sub[sub.split == s]
                    block = pd.concat([block[block.scorer == BASELINE_SCORE].head(1), block[block.scorer != BASELINE_SCORE]], ignore_index=True)
                    out.append(f"*{SPLIT_LABELS[s]}* — {int(block['runs'].iloc[0])} test complexes\n")
                    out.append(approach_table_markdown(block, KS))
                    out.append("")
                    if "mrr" in block.columns and block["mrr"].notna().any():
                        rs = block[block["mrr"].notna()].assign(scorer=block["scorer"].map(lambda a: APPROACH_LABELS.get(a, a)))
                        out.append(rank_shift_markdown(rs, reference=APPROACH_LABELS[BASELINE_SCORE]))
                        out.append("")
            else:
                base = sub[sub.scorer == BASELINE_SCORE].drop_duplicates("scorer")
                models = sub[sub.scorer != BASELINE_SCORE].copy()
                models["label"] = models["split"].map(SPLIT_LABELS)
                tab = pd.concat([base, models], ignore_index=True)
                out.append(f"**{SET_LABELS[set_name]}** — {int(base['runs'].iloc[0])} complexes; the split only changes the training set\n")
                out.append(approach_table_markdown(tab, KS, label_col="label"))
                out.append("")
                if "mrr" in models.columns and models["mrr"].notna().any():
                    rs = pd.concat([base, models], ignore_index=True)
                    rs = rs.assign(scorer=[APPROACH_LABELS.get(a, a) + (f" — {SPLIT_LABELS[s]}" if a != BASELINE_SCORE else "") for a, s in zip(rs["scorer"], rs["split"])])
                    out.append(rank_shift_markdown(rs[rs["mrr"].notna()], reference=APPROACH_LABELS[BASELINE_SCORE]))
                    out.append("")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", type=Path, default=RANKING_DIR)
    ap.add_argument("--out", type=Path, default=OUTPUT_DIR / "ranking")
    args = ap.parse_args(argv)
    df = collect(args.runs_dir)
    if df.empty:
        raise SystemExit(f"no schema-{SCHEMA} results.json under {args.runs_dir}")
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "ranking_results.csv", index=False)
    md = tables_markdown(df)
    (args.out / "ranking_tables.md").write_text(md)
    print(md)
    print(f"\nwritten: {args.out}/ranking_results.csv and ranking_tables.md ({df['run'].nunique()} runs)")


if __name__ == "__main__":
    main()
