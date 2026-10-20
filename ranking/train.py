"""
ranking.train
-------------
Train and evaluate **one** pose-ranking model on the full extraction (Track 2, inventory S5):
the port of Luca Friebe's reweighting / ranking pipelines, with corrected labels and features
and a leakage-aware split.

One run = (model, features, training protocol(s), split, label definition). The model is trained
on the PDBbind training complexes, selected on the validation complexes (best Success@1, then
Success@5) and evaluated — next to the Rosetta interface delta and the corrected total score on
the same docking runs — on the PDBbind test complexes, PoseBusters and Astex, for the training
protocol(s) (``--eval-protocols`` adds others, e.g. the apo_relax setting for a model trained
on relax poses). Everything is written to ``<RANKING_DIR>/<run>/``:

``config.json``       all arguments, the feature columns, the split counts, the excluded entries
``train.log``         the epoch log
``history.csv``       per-epoch loss terms and validation metrics, the selected epoch flagged
``model.pt``          state dict of the selected epoch;  ``scaler.json``  the feature scaler
``weights.csv``       (reweight only) the learned term weights, in scaled, raw and Rosetta units
``per_run.parquet``   per docking run and scorer: n_hit, first_hit_rank, top1/5/10, AUC
``results.json``      summary of the evaluation (rates, intervals, AUCs) + provenance
``results.md``        the same as tables

Usage::

    uv run python -m ranking.train --model reweight                      # defaults: relax_docking_perturb,
    uv run python -m ranking.train --model ranker                        #   delta features, LP-PDBBind split
    uv run python -m ranking.train --model reweight --protocol apo_relax_docking_perturb --init rosetta
    uv run python -m ranking.train --model reweight --split random --rmsd-col rmsd_to_input_legacy \\
        --no-exclude-overlap --run-name luca_setup_reweight             # Luca's original setup
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from docking_analysis.constants import RANKING_DIR

from .data import (BASELINE_SCORES, FA_REP_CLIP, FEATURE_SETS, PROTOCOLS, SCALE_CLIP, SPLIT_SCHEMES, Grouped,
                   RobustScale, assign_split, clip_features, external_pdb_ids, load_poses)
from .evaluate import KS, evaluate_scorers, frame_markdown, per_group_success, summary_markdown
from .models import DEFAULT_LR, MODELS, Reweight, fit, init_rosetta, make_model, predict, select_device, weights_in_raw_units

SETS = ("pdbbind", "posebuster", "astex")


class Log:
    def __init__(self, path: Path):
        self.f = open(path, "a")

    def __call__(self, msg: str = ""):
        print(msg, flush=True)
        self.f.write(msg + "\n")
        self.f.flush()


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
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--protocol", nargs="+", default=["relax_docking_perturb"], choices=PROTOCOLS,
                    help="training protocol(s) = receptor state x start mode (default relax_docking_perturb, Luca's setting)")
    ap.add_argument("--features", default="delta", choices=sorted(FEATURE_SETS))
    ap.add_argument("--split", default="lppdbbind", choices=SPLIT_SCHEMES)
    ap.add_argument("--rmsd-col", default="rmsd_to_crystal", help="label column (rmsd_to_input_legacy reproduces Luca's labels)")
    ap.add_argument("--cutoff", type=float, default=2.0, help="hit: rmsd <= cutoff (Å)")
    ap.add_argument("--no-exclude-overlap", action="store_true",
                    help="keep PDBbind complexes whose PDB entry is also an Astex/PoseBusters complex in train/val")
    ap.add_argument("--init", default="random", choices=("random", "rosetta"), help="reweight only: start at the Rosetta weights")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=None, help="default: 1e-4 (reweight), 1e-5 (ranker)")
    ap.add_argument("--beta", type=float, default=1.0, help="weight of the InfoNCE term")
    ap.add_argument("--tau", type=float, default=0.5, help="InfoNCE temperature")
    ap.add_argument("--batch-runs", type=int, default=1, help="docking runs per optimiser step")
    ap.add_argument("--patience", type=int, default=None, help="stop after N epochs without validation improvement")
    ap.add_argument("--fa-rep-clip", default=str(FA_REP_CLIP), help="cap of the delta fa_rep before scaling, or 'none'")
    ap.add_argument("--scale-clip", type=float, default=SCALE_CLIP, help="scaled features are clipped to ±this")
    ap.add_argument("--eval-protocols", nargs="*", default=None, choices=PROTOCOLS, help="default: the training protocol(s)")
    ap.add_argument("--eval-sets", nargs="+", default=list(SETS), choices=SETS)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--subsample", type=int, default=None, help="use only N random training complexes (smoke tests)")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--out", type=Path, default=None, help="run directory (default <RANKING_DIR>/<run-name>)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--extract-dir", type=Path, default=None, help="directory holding <set>_full/ (default EXTRACT_DIR)")
    args = ap.parse_args(argv)
    args.fa_rep_clip = None if str(args.fa_rep_clip).lower() == "none" else float(args.fa_rep_clip)
    if args.init == "rosetta" and args.model != "reweight":
        ap.error("--init rosetta applies to --model reweight only")
    return args


def default_run_name(args) -> str:
    name = f"{'+'.join(args.protocol)}_{args.features}_{args.split}_{args.model}"
    if args.init == "rosetta":
        name += "_rosetta"
    if args.rmsd_col != "rmsd_to_crystal" or args.cutoff != 2.0:
        name += f"_{args.rmsd_col}_{args.cutoff:g}A"
    if args.no_exclude_overlap:
        name += "_overlap"
    return name


def main(argv=None):
    args = parse(argv)
    t_start = time.time()
    feature_cols = FEATURE_SETS[args.features]
    device = select_device(args.device)
    run_name = args.run_name or default_run_name(args)
    out = args.out or RANKING_DIR / run_name
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        sys.exit(f"run directory exists: {out} (use --overwrite)")
    out.mkdir(parents=True, exist_ok=True)
    log = Log(out / "train.log")
    log(f"[{run_name}] model={args.model} features={args.features} ({len(feature_cols)}) protocols={args.protocol} "
        f"split={args.split} label={args.rmsd_col}<={args.cutoff:g} device={device}")

    # ---- PDBbind poses, split --------------------------------------------------------------
    extract_dir = args.extract_dir
    pdbbind = load_poses("pdbbind", args.protocol, feature_cols, extract_dir and extract_dir / "pdbbind_full")
    pdbbind = clip_features(pdbbind, feature_cols, args.fa_rep_clip)
    split = assign_split(pdbbind, args.split, args.seed, args.rmsd_col)
    overlap_excluded: dict[str, list[str]] = {}
    if not args.no_exclude_overlap:
        ext = external_pdb_ids(extract_dir=None) if extract_dir is None else \
            {s: external_pdb_ids((s,), extract_dir / f"{s}_full")[s] for s in ("astex", "posebuster")}
        lower = split.index.str.lower()
        in_train = split.isin(["train", "val"]).to_numpy()
        for s, ids in ext.items():
            hit = lower.isin(ids) & in_train
            overlap_excluded[s] = sorted(split.index[hit])
            split[hit] = "drop"
        log("excluded from train/val (same PDB entry in an external set): " +
            ", ".join(f"{s}: {len(v)}" for s, v in overlap_excluded.items()))
    if args.subsample:
        rng = np.random.default_rng(args.seed)
        train_ids = split.index[split == "train"].to_numpy()
        keep = set(rng.choice(train_ids, size=min(args.subsample, len(train_ids)), replace=False))
        split[[p for p in train_ids if p not in keep]] = "drop"
    test_label = "val" if args.split == "random" else "test"
    split_counts = split.value_counts().to_dict()
    log(f"split ({args.split}): {split_counts}" + ("  [random split: PDBbind is evaluated on the validation fold, as Luca did]" if args.split == "random" else ""))

    # sanity: Rosetta baselines on all complete PDBbind runs, to compare with the benchmark tables
    full = Grouped.build(pdbbind, np.zeros((len(pdbbind), 0), np.float32), args.rmsd_col, args.cutoff)
    sanity = {}
    keys = full.keys()
    for sc in BASELINE_SCORES:
        pc = per_group_success(full.column(sc), full.hit, full.offsets, KS)
        for proto, g in pc.groupby(keys["protocol"].to_numpy()):
            sanity.setdefault(proto, {})[sc] = {f"top{k}": round(100 * g[f"top{k}"].mean(), 2) for k in KS}
    log(f"sanity, all {full.n_complexes:,} complete PDBbind complexes: {json.dumps(sanity)}")
    del full

    # ---- features, scaler, grouped arrays -------------------------------------------------
    pose_split = pdbbind["pdb"].map(split).to_numpy()
    frames = {k: pdbbind[pose_split == k] for k in ("train", "val")}
    frames["test"] = pdbbind[pose_split == test_label]
    scaler = RobustScale.fit(frames["train"][feature_cols].to_numpy(), feature_cols, args.scale_clip)

    def grouped(frame: pd.DataFrame) -> Grouped:
        return Grouped.build(frame, scaler.transform(frame[feature_cols].to_numpy()), args.rmsd_col, args.cutoff)

    train_g, val_g = grouped(frames["train"]), grouped(frames["val"])
    log(f"train: {train_g.n_groups:,} runs / {train_g.n_complexes:,} complexes, {len(train_g.X):,} poses, "
        f"{100 * train_g.hit.mean():.2f} % hits, {(per_group_success(np.zeros(len(train_g.X)), train_g.hit, train_g.offsets)['n_hit'] == 0).sum():,} runs without a hit; "
        f"val: {val_g.n_groups:,} runs; test: {frames['test']['pdb'].nunique():,} complexes")

    # ---- model, training --------------------------------------------------------------------
    model = make_model(args.model, len(feature_cols))
    if args.init == "rosetta":
        init_rosetta(model, feature_cols, scaler)
    lr = args.lr if args.lr is not None else DEFAULT_LR[args.model]
    res = fit(model, train_g, val_g, epochs=args.epochs, lr=lr, beta=args.beta, tau=args.tau, batch_runs=args.batch_runs,
              patience=args.patience, seed=args.seed, device=device, log=log)
    log(f"selected epoch {res.best_epoch}: val Success@1 {res.history.loc[res.history['selected'], 'val_top1'].iloc[0]:.2f}")
    res.history.to_csv(out / "history.csv", index=False)
    torch.save(res.best_state, out / "model.pt")
    (out / "scaler.json").write_text(json.dumps(scaler.to_dict(), indent=1))
    weights = None
    if isinstance(model, Reweight):
        weights = weights_in_raw_units(model, scaler, feature_cols)
        weights.to_csv(out / "weights.csv", index=False)
        log("learned weights (Rosetta sign, normalised to fa_atr = 0.8):\n" +
            weights[["feature", "w_normalised_fa_atr_0.8", "rosetta_default"]].to_string(index=False))

    # ---- evaluation --------------------------------------------------------------------------
    eval_protocols = args.eval_protocols or list(args.protocol)
    test_pdbs = set(split.index[split == test_label])
    summaries, per_runs, md = [], [], []
    for proto in eval_protocols:
        for set_name in args.eval_sets:
            if set_name == "pdbbind":
                frame = frames["test"] if proto in args.protocol else \
                    clip_features(load_poses("pdbbind", proto, feature_cols, extract_dir and extract_dir / "pdbbind_full"), feature_cols, args.fa_rep_clip)
                frame = frame[frame["protocol"].eq(proto) & frame["pdb"].isin(test_pdbs)]
            else:
                frame = clip_features(load_poses(set_name, proto, feature_cols, extract_dir and extract_dir / f"{set_name}_full"),
                                      feature_cols, args.fa_rep_clip)
            if len(frame) == 0:
                log(f"[{proto}] {set_name}: no poses — skipped")
                continue
            g = grouped(frame)
            scorers = {sc: -g.column(sc).astype(np.float64) for sc in BASELINE_SCORES}
            scorers[args.model] = predict(model, g.X, device)
            summary, per_run = evaluate_scorers(g, scorers, reference="idelta_score", ks=KS, n_boot=args.n_boot, seed=args.seed)
            summary.insert(0, "set", set_name)
            summary.insert(1, "protocol", proto)
            per_run.insert(0, "run", run_name)
            summaries.append(summary)
            per_runs.append(per_run)
            note = " (PDBbind test complexes of the split)" if set_name == "pdbbind" else ""
            md.append(f"**{set_name}{note}, protocol `{proto}`** — hit: `{args.rmsd_col}` ≤ {args.cutoff:g} Å\n\n"
                      + summary_markdown(summary.drop(columns=["set", "protocol"]), KS) + "\n")
            log(md[-1])
    evaluation = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    if per_runs:
        pd.concat(per_runs, ignore_index=True).to_parquet(out / "per_run.parquet", index=False)

    # ---- outputs -----------------------------------------------------------------------------
    config = {**vars(args), "lr": lr, "run": run_name, "out": out, "feature_cols": feature_cols, "device": str(device)}
    results = {
        "run": run_name, "config": config, "split_counts": split_counts, "test_label": test_label,
        "overlap_excluded": overlap_excluded, "n_train_runs": train_g.n_groups, "n_train_complexes": train_g.n_complexes,
        "n_val_runs": val_g.n_groups, "sanity_full_pdbbind": sanity, "best_epoch": res.best_epoch,
        "history": res.history, "scaler": scaler.to_dict(), "weights": weights, "evaluation": evaluation,
        "elapsed_s": time.time() - t_start,
    }
    (out / "config.json").write_text(json.dumps(config, indent=1, default=_jsonable))
    (out / "results.json").write_text(json.dumps(results, indent=1, default=_jsonable))
    head = [f"# Run `{run_name}`", "",
            f"model `{args.model}`, features `{args.features}` ({', '.join(feature_cols)}), training protocol(s) "
            f"{', '.join(f'`{p}`' for p in args.protocol)}, split `{args.split}` ({split_counts}), label `{args.rmsd_col}` ≤ {args.cutoff:g} Å, "
            f"init `{args.init}`, lr {lr:g}, beta {args.beta:g}, tau {args.tau:g}, epochs {args.epochs}, selected epoch {res.best_epoch}.",
            "", "Rates are percent of docking runs (one run = 300 poses of one complex and protocol); "
            "`[95 % CI]` percentile bootstrap over runs; `Δ` paired difference to the interface delta on the same runs; "
            "`AUC pooled` over all poses, `AUC per run` mean of the within-run AUCs (runs with hits and misses).", ""]
    if weights is not None:
        head += ["Learned weights:", "", frame_markdown(weights), ""]
    (out / "results.md").write_text("\n".join(head + md))
    log(f"[{run_name}] done in {(time.time() - t_start) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
