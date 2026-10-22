"""
affinity.train
--------------
Score vs. affinity (Track 3, inventory S7 = Appendix A.3 step 1): how well does the Rosetta
interface delta of the docked poses track the experimental binding affinity, and how much does
a linear reweighting of its seven terms add?

The grid: label set (``kdki`` = exact Kd + Ki, the headline; ``ic50`` separately) × split
(LP-PDBBind, HAC with the CASF-2016 core set as test, CleanSplit) × pool (which poses of a
complex are aggregated: one docking run, the perturb + std runs of one receptor state, the
relaxed holo complex, the crystal complex) × aggregation (``best``, ``mean10``, ``mean``,
``best_hit`` oracle). For every cell the five approaches of :mod:`affinity.models` are fitted on
the training complexes (train + val of the split: nothing is selected, so no validation set is
needed) and evaluated on the test complexes (``test``), on the two halves of the Kd + Ki test
set (``test_kd``, ``test_ki``) and on the CASF-2016 core complexes outside the split's training
set (``casf_core``; for HAC that is its test set). Before the grid, the unmodified interface
delta is correlated with pK on **all** labelled complete complexes per pool × aggregation ×
label set — no split, no fit (``baseline_all.csv``) — the numbers comparable with the legacy
grid.

Outputs, all under ``<AFFINITY_DIR>/``:

``features.parquet``    the per-complex feature table (cache, ``affinity.data.build_features``)
``labels.csv``          the parsed PDBbind index
``baseline_all.csv``    the unmodified score on all labelled complexes per pool × aggregation × label set
``results.csv``         one row per label set × split × pool × aggregation × evaluation set × approach
``weights.csv``         the fitted coefficients and the energy weights of every model; ``weights_nonneg_<split>.wts``
                        the non-negative weights of the headline cell as a Rosetta weight set
``predictions.parquet`` per complex of the ``test`` / ``casf_core`` sets: label, features, every approach's prediction
``results.json`` / ``results.md`` / ``train.log``

Usage::

    uv run python -m affinity.train                                   # the default grid (6 pools)
    uv run python -m affinity.train --pool all                        # all ten pools
    uv run python -m affinity.train --labels kdki --split lppdbbind --pool apo_relax_docking_perturb --aggregation best
    uv run python -m affinity.report                                  # the tables and figures
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from docking_analysis.constants import AFFINITY_DIR

from .data import (AGGREGATIONS, FEATURE_COLS, FEATURES_PATH, HEADLINE_AGG, HEADLINE_POOL, HIT_CUTOFF, LABEL_SETS, POOLS,
                   SEED_POOLS, SPLIT_SCHEMES, TRAIN_LABEL_SETS, TRAIN_POOLS, assign_split, casf_core_ids, complex_info,
                   load_features, load_labels)
from .evaluate import MIN_N, approach_table_markdown, evaluate, frame_markdown
from .models import APPROACH_FEATURES, APPROACHES, REFERENCE, fit

SCHEMA = 1
EVAL_SETS = ("test", "test_kd", "test_ki", "casf_core")
KEYS = ["label_set", "split", "pool", "aggregation"]


class Log:
    def __init__(self, path: Path):
        self.f = open(path, "w")

    def __call__(self, msg: str = ""):
        print(msg, flush=True)
        self.f.write(msg + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def _jsonable(o):
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pd.DataFrame):
        return o.to_dict("records")
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def parse(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", nargs="+", default=list(TRAIN_LABEL_SETS), choices=list(LABEL_SETS),
                    help="label sets to train on (default: kdki and ic50)")
    ap.add_argument("--split", nargs="+", default=["all"], choices=SPLIT_SCHEMES + ("all",))
    ap.add_argument("--pool", nargs="+", default=list(TRAIN_POOLS), choices=POOLS + ("all",),
                    help="pose pools (default: the two perturb runs, with and without the std run, the relaxed holo and the crystal complex)")
    ap.add_argument("--aggregation", nargs="+", default=list(AGGREGATIONS), choices=AGGREGATIONS)
    ap.add_argument("--cutoff", type=float, default=HIT_CUTOFF, help="hit: rmsd_to_crystal <= cutoff (Å); only used when the features are (re)built")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=AFFINITY_DIR)
    ap.add_argument("--extract-dir", type=Path, default=None, help="directory holding <set>_full/ (default EXTRACT_DIR)")
    ap.add_argument("--rebuild-features", action="store_true")
    args = ap.parse_args(argv)
    args.splits = list(SPLIT_SCHEMES) if "all" in args.split else args.split
    args.pools = list(POOLS) if "all" in args.pool else args.pool
    return args


def baseline_all(table: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    """The unmodified interface delta (−idelta_score, no fit) against pK on all labelled
    complete complexes, per label set × pool × aggregation; plus the hit rate of the pose."""
    rows = []
    for (pool, agg), t in table.groupby(["pool", "aggregation"], sort=False):
        for label_set, kinds in LABEL_SETS.items():
            u = t[t["kind"].isin(kinds)]
            if len(u) < MIN_N:
                continue
            s = evaluate(u["pk"], {"unmodified": -u["idelta_score"].to_numpy()}, n_boot=n_boot, seed=seed, with_rmse=False).iloc[0]
            rows.append({"label_set": label_set, "pool": pool, "aggregation": agg, "n": int(len(u)),
                         "pose_hit_rate": 100 * float((u["rmsd_to_crystal"] <= HIT_CUTOFF).mean()),
                         "r": s["r"], "r_lo": s["r_lo"], "r_hi": s["r_hi"], "rho": s["rho"], "rho_lo": s["rho_lo"], "rho_hi": s["rho_hi"]})
    return pd.DataFrame(rows)


def main(argv=None):
    args = parse(argv)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    log = Log(out / "train.log")
    t0 = time.time()
    log(f"affinity.train: labels={args.labels} splits={args.splits} pools={args.pools} aggregations={args.aggregation} "
        f"n_boot={args.n_boot} seed={args.seed}")

    # ---- data ---------------------------------------------------------------------------------------
    feats = load_features(FEATURES_PATH if args.out_dir == AFFINITY_DIR else out / "features.parquet", rebuild=args.rebuild_features,
                          log=log, extract_dir=args.extract_dir, cutoff=args.cutoff)
    info = complex_info("pdbbind", args.extract_dir)
    labels = load_labels()
    labels.to_csv(out / "labels.csv", index=False)
    exact = labels[labels["exact"]]
    table = (feats.merge(info[["pdb", "ligand_n_heavy"]], on="pdb", how="inner")
                  .merge(exact[["pdb", "kind", "pk"]], on="pdb", how="inner"))
    counts = {k: int(exact[exact["kind"].isin(v) & exact["pdb"].isin(info["pdb"])]["pdb"].nunique()) for k, v in LABEL_SETS.items()}
    log(f"complete complexes {len(info):,}; with an exact label: {counts}")

    # ---- the unmodified score on everything ---------------------------------------------------------
    base = baseline_all(table, args.n_boot, args.seed)
    base.to_csv(out / "baseline_all.csv", index=False)
    log("unmodified interface delta vs pK on all labelled complexes (Pearson r [95 % CI]; pool × aggregation; kdki / ic50):")
    for (pool, agg), g in base.groupby(["pool", "aggregation"], sort=False):
        cells = {ls: f"{r['r']:.3f} [{r['r_lo']:.3f}, {r['r_hi']:.3f}] (n={int(r['n'])})" for ls, r in g.set_index("label_set").iterrows()}
        log(f"  {pool:34s} {agg:9s} kdki {cells.get('kdki', '—'):38s} ic50 {cells.get('ic50', '—')}")

    # ---- the grid -----------------------------------------------------------------------------------
    casf = casf_core_ids()
    results, weights, predictions, split_counts = [], [], [], {}
    for label_set in args.labels:
        kinds = LABEL_SETS[label_set]
        for split in args.splits:
            assignment = assign_split(info["pdb"], split, args.seed)
            train_ids = set(assignment.index[assignment.isin(["train", "val"])])
            test_ids = set(assignment.index[assignment.eq("test")])
            casf_ids = (casf & set(info["pdb"])) - train_ids
            lab_ids = set(table.loc[table["kind"].isin(kinds), "pdb"])
            split_counts[f"{label_set}/{split}"] = {"train": len(train_ids & lab_ids), "test": len(test_ids & lab_ids), "casf_core": len(casf_ids & lab_ids)}
            log(f"[{label_set} / {split}] labelled complexes: train {len(train_ids & lab_ids):,} (train + val), test {len(test_ids & lab_ids):,}, "
                f"CASF core outside the training set {len(casf_ids & lab_ids):,}")
            for pool in args.pools:
                for agg in (["best"] if pool in SEED_POOLS else args.aggregation):
                    t = table[table["pool"].eq(pool) & table["aggregation"].eq(agg) & table["kind"].isin(kinds)]
                    tr, te = t[t["pdb"].isin(train_ids)], t[t["pdb"].isin(test_ids)]
                    if len(tr) < MIN_N:
                        log(f"  {pool} {agg}: {len(tr)} training complexes — skipped")
                        continue
                    models = {a: fit(a, tr, tr["pk"]) for a in APPROACHES}
                    for a, m in models.items():
                        wt = m.weights_table()
                        wt.insert(0, "approach", a)
                        for k, v in zip(KEYS, (label_set, split, pool, agg)):
                            wt.insert(0, k, v)
                        wt["n_train"] = len(tr)
                        weights.append(wt)
                    if label_set == "kdki" and pool == HEADLINE_POOL and agg == HEADLINE_AGG:
                        (out / f"weights_nonneg_{split}.wts").write_text(models["reweight_nonneg"].rosetta_wts())
                    evals = {"test": te, "casf_core": t[t["pdb"].isin(casf_ids)]}
                    if label_set == "kdki":
                        evals["test_kd"], evals["test_ki"] = te[te["kind"].eq("Kd")], te[te["kind"].eq("Ki")]
                    test_summary = None
                    for ev_name, frame in evals.items():
                        if len(frame) < MIN_N:
                            continue
                        preds = {a: m.predict(frame[APPROACH_FEATURES[a]]) for a, m in models.items()}
                        summary = evaluate(frame["pk"], preds, reference=REFERENCE, n_boot=args.n_boot, seed=args.seed)
                        summary.insert(0, "eval_set", ev_name)
                        for k, v in zip(reversed(KEYS), reversed((label_set, split, pool, agg))):
                            summary.insert(0, k, v)
                        summary["n_train"] = len(tr)
                        results.append(summary)
                        if ev_name == "test":
                            test_summary = summary
                        if ev_name in ("test", "casf_core"):
                            p = frame[["pdb", "kind", "pk", "ligand_n_heavy", "idelta_score", "rmsd_to_crystal", "n_hits"] + FEATURE_COLS].copy()
                            for a, yhat in preds.items():
                                p[f"pred_{a}"] = yhat
                            p.insert(0, "eval_set", ev_name)
                            for k, v in zip(reversed(KEYS), reversed((label_set, split, pool, agg))):
                                p.insert(0, k, v)
                            predictions.append(p)
                    if test_summary is not None:
                        line = " | ".join(f"{r['approach']} {r['r']:.3f}" + ("" if r["approach"] == REFERENCE else f" ({r['d_r']:+.3f} [{r['d_r_lo']:+.3f}, {r['d_r_hi']:+.3f}])")
                                          for _, r in test_summary.iterrows())
                        log(f"  {pool:34s} {agg:9s} train {len(tr):,} test {len(te):,}: {line}")
                    else:
                        log(f"  {pool:34s} {agg:9s} train {len(tr):,} test {len(te):,}: test set too small")

    results = pd.concat(results, ignore_index=True)
    weights = pd.concat(weights, ignore_index=True)
    predictions = pd.concat(predictions, ignore_index=True)
    results.to_csv(out / "results.csv", index=False)
    weights.to_csv(out / "weights.csv", index=False)
    predictions.to_parquet(out / "predictions.parquet", index=False)

    # ---- results.md / results.json ------------------------------------------------------------------
    md = [f"# affinity.train — results of {time.strftime('%Y-%m-%d')}", "",
          f"Label sets {args.labels}, splits {args.splits}, pools {args.pools}, aggregations {args.aggregation}; "
          f"{args.n_boot} bootstrap resamples over complexes (seed {args.seed}). Training complexes = train + val of the split. "
          f"Labelled complete complexes: {counts}.", "",
          "`Pearson r` between the predicted and the experimental pK on the test complexes with its percentile-bootstrap 95 % interval; "
          "`Δr` the paired difference to the calibrated interface delta on the same complexes; `RMSE` in pK units after the linear fit.", ""]
    for label_set in args.labels:
        for split in args.splits:
            s = results[(results["label_set"] == label_set) & (results["split"] == split) & (results["pool"] == HEADLINE_POOL)
                        & (results["aggregation"] == HEADLINE_AGG) & (results["eval_set"] == "test")]
            if len(s):
                md += [f"## `{label_set}` · split `{split}` · pool `{HEADLINE_POOL}` · aggregation `{HEADLINE_AGG}` — test complexes", "",
                       approach_table_markdown(s), ""]
    b = base[base["label_set"] == "kdki"][["pool", "aggregation", "n", "pose_hit_rate", "r", "r_lo", "r_hi", "rho"]]
    md += ["## Unmodified interface delta vs pK on all labelled complete complexes (Kd + Ki), no split", "", frame_markdown(b), ""]
    (out / "results.md").write_text("\n".join(md))
    config = {k: v for k, v in vars(args).items()}
    (out / "results.json").write_text(json.dumps({"schema": SCHEMA, "config": config, "label_counts": counts, "split_counts": split_counts,
                                                   "feature_cols": FEATURE_COLS, "approaches": list(APPROACHES), "eval_sets": list(EVAL_SETS),
                                                   "n_results": len(results), "elapsed_s": time.time() - t0}, indent=1, default=_jsonable))
    log(f"done in {(time.time() - t0) / 60:.1f} min -> {out}")
    log.close()


if __name__ == "__main__":
    main()
