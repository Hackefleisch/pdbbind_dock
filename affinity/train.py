"""
affinity.train
--------------
Score vs. affinity (Track 3, inventory S7 = Appendix A.3 step 1): can the binding affinity be
predicted from the Rosetta interface delta of the docked poses, and how much does a linear
reweighting of its interface terms add? The headline metric is the **MAE in pK** against the
constant predictor (the null), with a **95 % lower bound** per complex; Pearson r is kept for
comparability with the CASF scoring power.

The grid: label set (``kdki`` = exact Kd + Ki, the headline; ``ic50`` separately) × split
(LP-PDBBind, HAC with the CASF-2016 core set as test, CleanSplit) × pool (which poses of a
complex are aggregated: one docking run, the perturb + std runs of one receptor state, the
relaxed holo complex, the crystal complex) × aggregation (``best``, ``mean10``, ``mean``,
``best_hit`` oracle). Per cell every predictor of :mod:`affinity.models` (constant, calibrated
interface delta, ligand size, and the three reweightings × three term sets) is fitted on the
**training** complexes with both losses (``l1`` median regression, ``l2`` least squares), the
τ = 0.05 quantile model is fitted as the model-side lower bound, the conformal slack is
calibrated on the **validation** complexes, and everything is evaluated on the test complexes
(``test``), the two halves of the Kd + Ki test set (``test_kd``, ``test_ki``) and the CASF-2016
core complexes outside the training and calibration sets (``casf_core``; for HAC that is its
test set). Before the grid, the unmodified interface delta is correlated with pK on **all**
labelled complete complexes per pool × aggregation × label set (no split, no fit;
``baseline_all.csv``).

Outputs, all under ``<AFFINITY_DIR>/``:

``features.parquet``    the per-complex feature table (cache, ``affinity.data.build_features``)
``labels.csv``          the parsed PDBbind index
``baseline_all.csv``    Pearson r of the unmodified score on all labelled complexes per pool × aggregation × label set
``results.csv``         one row per label set × split × pool × aggregation × evaluation set × loss × predictor
``weights.csv``         the fitted coefficients and energy weights of every model (point fits and τ = 0.05 models);
                        ``weights_nonneg_<split>.wts`` the non-negative L1 weights of the headline cell as a Rosetta weight set
``predictions.parquet`` per complex of the ``test`` / ``casf_core`` sets and loss: label, features, every
                        predictor's prediction (``pred_<key>``) and lower bounds (``cp_<key>``, ``qr_<key>``)
``results.json`` / ``results.md`` / ``train.log``

Usage::

    uv run python -m affinity.train                                   # the default grid (6 pools, ~15 min)
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
from .evaluate import MIN_N, conformal_slack, evaluate, frame_markdown, mae_table_markdown, pearson_ci
from .models import LOSSES, LOWER_TAU, NULL, REFERENCE, TERM_SETS, fit, predictor_key, predictors, split_key

SCHEMA = 2
EVAL_SETS = ("test", "test_kd", "test_ki", "casf_core")
KEYS = ["label_set", "split", "pool", "aggregation"]
HEADLINE_KEYS = ["constant", "unmodified", "size", "reweight", "reweight_nonneg", "reweight_size"]


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
    ap.add_argument("--loss", nargs="+", default=list(LOSSES), choices=LOSSES, help="point-predictor losses (default: both)")
    ap.add_argument("--terms", nargs="+", default=list(TERM_SETS), choices=list(TERM_SETS), help="term sets of the reweightings")
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
    """Pearson r of −idelta_score (no fit) with pK on all labelled complete complexes, per label
    set × pool × aggregation, plus the hit rate of the aggregated pose."""
    rows = []
    for (pool, agg), t in table.groupby(["pool", "aggregation"], sort=False):
        for label_set, kinds in LABEL_SETS.items():
            u = t[t["kind"].isin(kinds)]
            if len(u) < MIN_N:
                continue
            r, lo, hi = pearson_ci(u["pk"], -u["idelta_score"], n_boot, seed)
            rows.append({"label_set": label_set, "pool": pool, "aggregation": agg, "n": int(len(u)),
                         "pose_hit_rate": 100 * float((u["rmsd_to_crystal"] <= HIT_CUTOFF).mean()), "r": r, "r_lo": lo, "r_hi": hi})
    return pd.DataFrame(rows)


def _cell_line(summary: pd.DataFrame) -> str:
    s = summary.set_index("predictor")
    parts = []
    for k in HEADLINE_KEYS:
        if k not in s.index:
            continue
        r = s.loc[k]
        d = "" if k in (REFERENCE,) or pd.isna(r.get("d_mae", np.nan)) else f" ({r['d_mae']:+.3f} [{r['d_mae_lo']:+.3f}, {r['d_mae_hi']:+.3f}])"
        parts.append(f"{k} {r['mae']:.3f}{d}")
    lb = ""
    if "cp_slack" in s.columns and pd.notna(s.loc["unmodified", "cp_slack"]):
        lb = " | LB slack/cov/cert5: " + ", ".join(f"{k} {s.loc[k, 'cp_slack']:.2f}/{s.loc[k, 'cp_coverage']:.0f}/{s.loc[k, 'cp_cert5']:.0f}"
                                                 for k in ("constant", "unmodified", "reweight", "size") if k in s.index)
    return " | ".join(parts) + lb


def main(argv=None):
    args = parse(argv)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    log = Log(out / "train.log")
    t0 = time.time()
    log(f"affinity.train: labels={args.labels} splits={args.splits} pools={args.pools} aggregations={args.aggregation} "
        f"losses={args.loss} terms={args.terms} n_boot={args.n_boot} seed={args.seed}")

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
    log("unmodified interface delta vs pK on all labelled complexes (Pearson r [95 % CI], no fit; kdki / ic50):")
    for (pool, agg), g in base.groupby(["pool", "aggregation"], sort=False):
        cells = {ls: f"{r['r']:.3f} [{r['r_lo']:.3f}, {r['r_hi']:.3f}] (n={int(r['n'])})" for ls, r in g.set_index("label_set").iterrows()}
        log(f"  {pool:34s} {agg:9s} kdki {cells.get('kdki', '—'):38s} ic50 {cells.get('ic50', '—')}")

    # ---- the grid -----------------------------------------------------------------------------------
    casf = casf_core_ids()
    plist = predictors(term_sets=tuple(args.terms))
    results, weights, predictions, split_counts = [], [], [], {}
    for label_set in args.labels:
        kinds = LABEL_SETS[label_set]
        for split in args.splits:
            assignment = assign_split(info["pdb"], split, args.seed)
            train_ids = set(assignment.index[assignment.eq("train")])
            cal_ids = set(assignment.index[assignment.eq("val")])
            test_ids = set(assignment.index[assignment.eq("test")])
            casf_ids = (casf & set(info["pdb"])) - train_ids - cal_ids
            lab_ids = set(table.loc[table["kind"].isin(kinds), "pdb"])
            sc = {"train": len(train_ids & lab_ids), "val": len(cal_ids & lab_ids), "test": len(test_ids & lab_ids), "casf_core": len(casf_ids & lab_ids)}
            split_counts[f"{label_set}/{split}"] = sc
            log(f"[{label_set} / {split}] labelled complexes: train {sc['train']:,}, calibration (val) {sc['val']:,}, test {sc['test']:,}, "
                f"CASF core outside train + val {sc['casf_core']:,}")
            for pool in args.pools:
                for agg in (["best"] if pool in SEED_POOLS else args.aggregation):
                    t = table[table["pool"].eq(pool) & table["aggregation"].eq(agg) & table["kind"].isin(kinds)]
                    tr, ca, te = t[t["pdb"].isin(train_ids)], t[t["pdb"].isin(cal_ids)], t[t["pdb"].isin(test_ids)]
                    if len(tr) < MIN_N:
                        log(f"  {pool} {agg}: {len(tr)} training complexes — skipped")
                        continue
                    models, qr = {}, {}
                    for a, terms in plist:
                        k = predictor_key(a, terms)
                        for loss in args.loss:
                            models[(k, loss)] = fit(a, tr, tr["pk"], loss=loss, terms=terms)
                        qr[k] = fit(a, tr, tr["pk"], terms=terms, tau=LOWER_TAU)
                    slack = {}
                    if len(ca) >= MIN_N:
                        slack = {(k, loss): conformal_slack(m.predict(ca), ca["pk"]) for (k, loss), m in models.items()}
                    for m in list(models.values()) + list(qr.values()):
                        if not m.names:
                            continue
                        wt = m.weights_table()
                        wt.insert(0, "tau", m.tau if m.tau is not None else np.nan)
                        wt.insert(0, "loss", m.loss if m.tau is None else "quantile")
                        wt.insert(0, "terms", m.terms)
                        wt.insert(0, "approach", m.approach)
                        for kk, v in zip(KEYS, (label_set, split, pool, agg)):
                            wt.insert(0, kk, v)
                        wt["n_train"] = len(tr)
                        weights.append(wt)
                    if label_set == "kdki" and pool == HEADLINE_POOL and agg == HEADLINE_AGG and ("reweight_nonneg", "l1") in models:
                        (out / f"weights_nonneg_{split}.wts").write_text(models[("reweight_nonneg", "l1")].rosetta_wts())
                    evals = {"test": te, "casf_core": t[t["pdb"].isin(casf_ids)]}
                    if label_set == "kdki":
                        evals["test_kd"], evals["test_ki"] = te[te["kind"].eq("Kd")], te[te["kind"].eq("Ki")]
                    test_summary = None
                    for ev_name, frame in evals.items():
                        if len(frame) < MIN_N:
                            continue
                        for loss in args.loss:
                            preds = {k: models[(k, loss)].predict(frame) for k in qr}
                            bounds = {k: {"qr": qr[k].predict(frame)} for k in qr}
                            if slack:
                                for k in qr:
                                    bounds[k]["cp"] = preds[k] - slack[(k, loss)]
                            summary = evaluate(frame["pk"], preds, bounds, reference=REFERENCE, null=NULL, n_boot=args.n_boot, seed=args.seed)
                            summary.insert(1, "terms", [split_key(k)[1] for k in summary["predictor"]])
                            summary.insert(1, "approach", [split_key(k)[0] for k in summary["predictor"]])
                            summary.insert(0, "loss", loss)
                            summary.insert(0, "eval_set", ev_name)
                            for kk, v in zip(reversed(KEYS), reversed((label_set, split, pool, agg))):
                                summary.insert(0, kk, v)
                            summary["n_train"], summary["n_cal"] = len(tr), len(ca)
                            results.append(summary)
                            if ev_name == "test" and loss == "l1":
                                test_summary = summary
                            if ev_name in ("test", "casf_core"):
                                p = frame[["pdb", "kind", "pk", "ligand_n_heavy", "idelta_score", "rmsd_to_crystal", "n_hits"] + FEATURE_COLS].copy()
                                for k in qr:
                                    p[f"pred_{k}"] = preds[k]
                                    for kind, L in bounds[k].items():
                                        p[f"{kind}_{k}"] = L
                                p.insert(0, "loss", loss)
                                p.insert(0, "eval_set", ev_name)
                                for kk, v in zip(reversed(KEYS), reversed((label_set, split, pool, agg))):
                                    p.insert(0, kk, v)
                                predictions.append(p)
                    if test_summary is not None:
                        log(f"  {pool:34s} {agg:9s} train {len(tr):,} cal {len(ca):,} test {len(te):,} (l1, MAE): {_cell_line(test_summary)}")
                    else:
                        log(f"  {pool:34s} {agg:9s} train {len(tr):,} cal {len(ca):,} test {len(te):,}: test set too small")

    results = pd.concat(results, ignore_index=True)
    weights = pd.concat(weights, ignore_index=True)
    predictions = pd.concat(predictions, ignore_index=True)
    results.to_csv(out / "results.csv", index=False)
    weights.to_csv(out / "weights.csv", index=False)
    predictions.to_parquet(out / "predictions.parquet", index=False)

    # ---- results.md / results.json ------------------------------------------------------------------
    md = [f"# affinity.train — results of {time.strftime('%Y-%m-%d')}", "",
          f"Label sets {args.labels}, splits {args.splits}, pools {args.pools}, aggregations {args.aggregation}, losses {args.loss}, "
          f"term sets {args.terms}; {args.n_boot} bootstrap resamples over complexes (seed {args.seed}). Point fits on the training "
          f"complexes, conformal slack on the validation complexes. Labelled complete complexes: {counts}.", "",
          "`MAE` in pK on the test complexes with its percentile-bootstrap 95 % interval; `ΔMAE` the paired difference to the "
          "calibrated interface delta / to the constant predictor on the same complexes; `LB`: 95 % lower bounds per complex "
          "(conformal: prediction minus the calibrated slack; QR: the τ = 0.05 quantile model) with their coverage and the share "
          "of complexes certified above pK 5 / 6. Headline rows: the `l1` (median-regression) fits with all seven terms.", ""]
    for label_set in args.labels:
        for split in args.splits:
            s = results[(results["label_set"] == label_set) & (results["split"] == split) & (results["pool"] == HEADLINE_POOL)
                        & (results["aggregation"] == HEADLINE_AGG) & (results["eval_set"] == "test") & (results["loss"] == "l1")]
            if len(s):
                md += [f"## `{label_set}` · split `{split}` · pool `{HEADLINE_POOL}` · aggregation `{HEADLINE_AGG}` — test complexes, l1 fits", "",
                       mae_table_markdown(s, HEADLINE_KEYS + [k for k in s["predictor"] if k not in HEADLINE_KEYS]), ""]
    b = base[base["label_set"] == "kdki"][["pool", "aggregation", "n", "pose_hit_rate", "r", "r_lo", "r_hi"]]
    md += ["## Unmodified interface delta vs pK on all labelled complete complexes (Kd + Ki), no split, no fit (Pearson r)", "", frame_markdown(b), ""]
    (out / "results.md").write_text("\n".join(md))
    config = {k: v for k, v in vars(args).items()}
    (out / "results.json").write_text(json.dumps({"schema": SCHEMA, "config": config, "label_counts": counts, "split_counts": split_counts,
                                                   "feature_cols": FEATURE_COLS, "term_sets": TERM_SETS, "predictors": [predictor_key(a, t) for a, t in plist],
                                                   "losses": list(args.loss), "lower_tau": LOWER_TAU, "eval_sets": list(EVAL_SETS),
                                                   "n_results": len(results), "elapsed_s": time.time() - t0}, indent=1, default=_jsonable))
    log(f"done in {(time.time() - t0) / 60:.1f} min -> {out}")
    log.close()


if __name__ == "__main__":
    main()
