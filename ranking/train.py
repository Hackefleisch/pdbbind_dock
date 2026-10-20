"""
ranking.train
-------------
Train and evaluate **one ranking approach** against the unmodified Rosetta interface delta
(Track 2, inventory S5):

``reweight``  non-negative reweighting of the raw interface terms — the baseline whose weights
              can be copied back into Rosetta, not ML — from a random start (``--init random``)
              or from the Rosetta weights (``--init rosetta``), reported as two baselines;
``ranker``    Luca Friebe's MLP on his robust-scaled features with his BCE + InfoNCE loss;
``tree``      LightGBM LambdaRank on the raw interface terms.

Training pool: the ``relax_docking_perturb`` + ``apo_relax_docking_perturb`` runs of the PDBbind
training complexes, one training bucket per complex × receptor state (``--group-by state``: a
pooled std run joins the perturb run of the same state); model selection on the validation
complexes (Success@1, then Success@5);
evaluation on the PDBbind test complexes, PoseBusters and Astex for both training protocols —
the headline is the Δ to the interface delta on ``apo_relax_docking_perturb``. ``--split all``
(default) trains and evaluates once per split (LP-PDBBind, HAC, CleanSplit); the PDBbind
poses are loaded once. One run = one (approach, split); outputs under
``<RANKING_DIR>/<approach>_<split>/`` (approach = ``reweight_random``, ``reweight_rosetta``, ``ranker``, ``tree``):

``config.json``     arguments, split counts, excluded entries;  ``train.log``  the epoch log
``history.csv``     per-epoch (or per-tree) validation metrics, the selected epoch flagged
``model.pt`` / ``model.txt``  the selected torch state dict / LightGBM booster
``preprocess.json`` the feature preprocessing (identity for reweight and tree)
``weights.csv``, ``weights.wts``  (reweight) the learned Rosetta weights;  ``importance.csv`` (tree)
``per_run.parquet`` per docking run and scorer: n_hit, first_hit_rank, top1/5/10, AUC
``results.json`` / ``results.md``  the evaluation (rates, intervals, rank-shift statistics)

Usage::

    uv run python -m ranking.train --model reweight --init random   # all three splits
    uv run python -m ranking.train --model reweight --init rosetta
    uv run python -m ranking.train --model ranker --split hac
    uv run python -m ranking.train --model tree
    uv run python -m ranking.report                                 # tables over all runs
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

from .data import (BASELINE_SCORE, FA_REP_CLIP, FEATURE_COLS, HEADLINE_PROTOCOL, PROTOCOLS, SCALE_CLIP, SPLIT_SCHEMES,
                   TRAIN_PROTOCOLS, Grouped, Preprocess, assign_split, external_pdb_ids, load_poses)
from .evaluate import KS, approach_table_markdown, evaluate_scorers, frame_markdown, per_group_success, rank_shift, rank_shift_markdown
from .models import (DEFAULT_BCE_WEIGHT, DEFAULT_LR, DEFAULT_SCALING, INITS, MODELS, Ranker, Reweight, TreeRanker, fit, predict,
                     select_device)

SCHEMA = 2
SETS = ("pdbbind", "posebuster", "astex")


class Log:
    def __init__(self, path: Path):
        self.f = open(path, "w")                 # a rerun with --overwrite starts a fresh log

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
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--split", default="all", choices=SPLIT_SCHEMES + ("all",))
    ap.add_argument("--protocol", nargs="+", default=list(TRAIN_PROTOCOLS), choices=PROTOCOLS,
                    help="training protocols (default: relax and apo_relax perturb)")
    ap.add_argument("--eval-protocols", nargs="*", default=None, choices=PROTOCOLS, help="default: the training protocols")
    ap.add_argument("--eval-sets", nargs="+", default=list(SETS), choices=SETS)
    ap.add_argument("--rmsd-col", default="rmsd_to_crystal")
    ap.add_argument("--cutoff", type=float, default=2.0, help="hit: rmsd <= cutoff (Å)")
    ap.add_argument("--no-exclude-overlap", action="store_true", help="keep the Astex-overlapping PDBbind entries in train/val")
    ap.add_argument("--scaling", default="auto", choices=("auto", "none", "robust"),
                    help="feature preprocessing (auto: raw energies for reweight and tree, Luca's robust scaling for the ranker)")
    ap.add_argument("--fa-rep-clip", default="auto", help="cap of the delta fa_rep before scaling (auto: 50 with robust scaling, none otherwise)")
    ap.add_argument("--scale-clip", type=float, default=SCALE_CLIP)
    ap.add_argument("--init", default="random", choices=INITS, help="reweight: random start or the Rosetta weights")
    ap.add_argument("--gauge", default="none", help="reweight: term whose weight stays fixed (default none: all seven weights free)")
    ap.add_argument("--group-by", default="state", choices=("state", "protocol"),
                    help="training bucket: complex x receptor state (default) or complex x protocol")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=None, help="default: 1e-4 (reweight), 1e-5 (ranker)")
    ap.add_argument("--beta", type=float, default=1.0, help="weight of the InfoNCE term")
    ap.add_argument("--tau", type=float, default=0.5, help="InfoNCE temperature")
    ap.add_argument("--bce-weight", type=float, default=None, help="weight of the BCE term (default: 0 reweight, 1 ranker)")
    ap.add_argument("--batch-runs", type=int, default=1, help="docking runs per optimiser step")
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--tree-estimators", type=int, default=2000)
    ap.add_argument("--tree-lr", type=float, default=0.05)
    ap.add_argument("--tree-leaves", type=int, default=31)
    ap.add_argument("--tree-min-child", type=int, default=100)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--subsample", type=int, default=None, help="use only N random training complexes (smoke tests)")
    ap.add_argument("--tag", default=None, help="suffix of the run directory name (<model>_<split>_<tag>)")
    ap.add_argument("--out-dir", type=Path, default=RANKING_DIR, help="parent of the run directories")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--extract-dir", type=Path, default=None, help="directory holding <set>_full/ (default EXTRACT_DIR)")
    args = ap.parse_args(argv)
    if args.scaling == "auto":
        args.scaling = DEFAULT_SCALING[args.model]
    if str(args.fa_rep_clip).lower() == "auto":
        args.fa_rep_clip = FA_REP_CLIP if args.scaling == "robust" else None
    elif str(args.fa_rep_clip).lower() == "none":
        args.fa_rep_clip = None
    else:
        args.fa_rep_clip = float(args.fa_rep_clip)
    if args.lr is None:
        args.lr = DEFAULT_LR.get(args.model)
    if args.bce_weight is None:
        args.bce_weight = DEFAULT_BCE_WEIGHT.get(args.model)
    if args.eval_protocols is None:
        args.eval_protocols = list(args.protocol)
    args.splits = list(SPLIT_SCHEMES) if args.split == "all" else [args.split]
    args.approach = f"reweight_{args.init}" if args.model == "reweight" else args.model
    return args


# ---------------------------------------------------------------------------
def baseline_sanity(pdbbind: pd.DataFrame, rmsd_col: str, cutoff: float) -> dict:
    """Unmodified Success@k on all complete PDBbind runs per protocol — must equal the benchmark tables."""
    full = Grouped.build(pdbbind, np.zeros((len(pdbbind), 0), np.float32), rmsd_col, cutoff)
    pc = per_group_success(full.column(BASELINE_SCORE), full.hit, full.offsets, KS)
    out = {}
    for proto, g in pc.groupby(full.keys()["protocol"].to_numpy()):
        out[proto] = {f"top{k}": round(100 * g[f"top{k}"].mean(), 2) for k in KS}
    return out


def train_one(args, split: str, pdbbind: pd.DataFrame, overlap_ids: dict[str, set[str]], external: dict, sanity: dict,
              device: torch.device) -> Path:
    run_name = f"{args.approach}_{split}" + (f"_{args.tag}" if args.tag else "")
    out = args.out_dir / run_name
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        sys.exit(f"run directory exists: {out} (use --overwrite)")
    out.mkdir(parents=True, exist_ok=True)
    log = Log(out / "train.log")
    t0 = time.time()
    log(f"[{run_name}] approach={args.approach} split={split} protocols={args.protocol} group_by={args.group_by} "
        f"scaling={args.scaling} fa_rep_clip={args.fa_rep_clip} label={args.rmsd_col}<={args.cutoff:g} device={device}")

    # ---- split ----------------------------------------------------------------------------------
    assignment = assign_split(pdbbind["pdb"], split, args.seed)
    overlap_excluded: dict[str, list[str]] = {}
    if not args.no_exclude_overlap:
        lower = assignment.index.str.lower()
        in_train = assignment.isin(["train", "val"]).to_numpy()
        for s_name, ids in overlap_ids.items():
            hit = lower.isin(ids) & in_train
            overlap_excluded[s_name] = sorted(assignment.index[hit])
            assignment[hit] = "drop"
        log("excluded from train/val (same PDB entry in an external set): " + ", ".join(f"{k}: {len(v)}" for k, v in overlap_excluded.items()))
    if args.subsample:
        rng = np.random.default_rng(args.seed)
        train_ids = assignment.index[assignment == "train"].to_numpy()
        keep = set(rng.choice(train_ids, size=min(args.subsample, len(train_ids)), replace=False))
        assignment[[p for p in train_ids if p not in keep]] = "drop"
    split_counts = assignment.value_counts().to_dict()
    log(f"split ({split}): {split_counts}")
    pose_split = pdbbind["pdb"].map(assignment).to_numpy()
    frames = {k: pdbbind[pose_split == k] for k in ("train", "val", "test")}

    # ---- preprocessing, grouped arrays ----------------------------------------------------------
    pre = Preprocess(args.scaling, args.fa_rep_clip, args.scale_clip).fit(frames["train"][FEATURE_COLS].to_numpy())

    def grouped(frame: pd.DataFrame, group_by: str = "protocol") -> Grouped:
        return Grouped.build(frame, pre.transform(frame[FEATURE_COLS].to_numpy()), args.rmsd_col, args.cutoff, group_by)

    train_g, val_g = grouped(frames["train"], args.group_by), grouped(frames["val"], args.group_by)
    n_no_hit = int((per_group_success(np.zeros(len(train_g.X)), train_g.hit, train_g.offsets)["n_hit"] == 0).sum())
    sizes = np.unique(train_g.group_sizes)
    log(f"train: {train_g.n_groups:,} buckets (complex x {args.group_by}; sizes {sizes.tolist()}) / {train_g.n_complexes:,} complexes, "
        f"{len(train_g.X):,} poses, {100 * train_g.hit.mean():.2f} % hits, {n_no_hit:,} buckets without a hit; "
        f"val: {val_g.n_groups:,} buckets; test: {frames['test']['pdb'].nunique():,} complexes")

    # ---- model ------------------------------------------------------------------------------------
    history, best, extras = None, None, {}
    if args.model == "tree":
        tree = TreeRanker(args.tree_estimators, args.tree_lr, args.tree_leaves, args.tree_min_child, seed=args.seed).fit(train_g, val_g, log=log)
        history, best = tree.history, tree.best_iteration
        tree.save(out / "model.txt")
        imp = tree.importance()
        imp.to_csv(out / "importance.csv", index=False)
        extras["importance"] = imp
        scorer = tree.predict
        log("feature importance (gain):\n" + imp.to_string(index=False))
    else:
        model = Reweight(FEATURE_COLS, init=args.init, gauge=args.gauge, seed=args.seed) if args.model == "reweight" else Ranker(len(FEATURE_COLS))
        res = fit(model, train_g, val_g, epochs=args.epochs, lr=args.lr, beta=args.beta, tau=args.tau, bce_weight=args.bce_weight,
                  batch_runs=args.batch_runs, patience=args.patience, seed=args.seed, device=device, log=log)
        history, best = res.history, res.best_epoch
        torch.save(res.best_state, out / "model.pt")
        log(f"selected epoch {best}: val Success@1 {history.loc[history['selected'], 'val_top1'].iloc[0]:.2f}")
        if isinstance(model, Reweight):
            wt = model.weights_table()
            wt.to_csv(out / "weights.csv", index=False)
            (out / "weights.wts").write_text(model.rosetta_wts())
            extras["weights"] = wt
            log("learned Rosetta weights:\n" + wt.to_string(index=False))
        scorer = lambda X: predict(model, X, device)
    history.to_csv(out / "history.csv", index=False)
    (out / "preprocess.json").write_text(json.dumps(pre.to_dict(), indent=1))

    # ---- evaluation --------------------------------------------------------------------------------
    test_pdbs = set(assignment.index[assignment == "test"])
    summaries, per_runs, shifts, md = [], [], [], []
    for proto in args.eval_protocols:
        for set_name in args.eval_sets:
            if set_name == "pdbbind":
                frame = frames["test"] if proto in args.protocol else external["pdbbind"][proto]
                frame = frame[frame["protocol"].eq(proto) & frame["pdb"].isin(test_pdbs)]
            else:
                frame = external[set_name][proto]
            if len(frame) == 0:
                log(f"[{proto}] {set_name}: no poses — skipped")
                continue
            g = grouped(frame)
            scorers = {BASELINE_SCORE: -g.column(BASELINE_SCORE).astype(np.float64), args.approach: scorer(g.X)}
            summary, per_run = evaluate_scorers(g, scorers, reference=BASELINE_SCORE, ks=KS, n_boot=args.n_boot, seed=args.seed)
            shift = rank_shift(per_run, reference=BASELINE_SCORE)
            for df_ in (summary, shift):
                df_.insert(0, "set", set_name)
                df_.insert(1, "protocol", proto)
            per_run.insert(0, "run", run_name)
            summaries.append(summary)
            per_runs.append(per_run)
            shifts.append(shift)
            note = " (PDBbind test complexes of the split)" if set_name == "pdbbind" else ""
            tag = " — **headline**" if proto == HEADLINE_PROTOCOL else ""
            md.append(f"**{set_name}{note}, protocol `{proto}`**{tag} — hit: `{args.rmsd_col}` ≤ {args.cutoff:g} Å\n\n"
                      + approach_table_markdown(summary.drop(columns=["set", "protocol"]), KS) + "\n\n"
                      + "First-hit rank distribution vs. the interface delta:\n\n"
                      + rank_shift_markdown(shift.drop(columns=["set", "protocol"])) + "\n")
            log(md[-1])
    evaluation = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    rank_shifts = pd.concat(shifts, ignore_index=True) if shifts else pd.DataFrame()
    if per_runs:
        pd.concat(per_runs, ignore_index=True).to_parquet(out / "per_run.parquet", index=False)

    # ---- outputs -----------------------------------------------------------------------------------
    config = {**{k: v for k, v in vars(args).items() if k not in ("splits",)}, "split": split, "run": run_name, "out": out,
              "feature_cols": FEATURE_COLS, "device": str(device)}
    results = {"schema": SCHEMA, "run": run_name, "model": args.model, "approach": args.approach, "split": split, "config": config,
               "split_counts": split_counts, "overlap_excluded": overlap_excluded,
               "n_train_runs": train_g.n_groups, "n_train_complexes": train_g.n_complexes, "n_val_runs": val_g.n_groups,
               "sanity_full_pdbbind": sanity, "best": best, "history": history, "preprocess": pre.to_dict(),
               **extras, "evaluation": evaluation, "rank_shift": rank_shifts, "elapsed_s": time.time() - t0}
    (out / "config.json").write_text(json.dumps(config, indent=1, default=_jsonable))
    (out / "results.json").write_text(json.dumps(results, indent=1, default=_jsonable))
    head = [f"# Run `{run_name}`", "",
            f"approach `{args.approach}`, split `{split}` ({split_counts}), training protocols {', '.join(f'`{p}`' for p in args.protocol)} "
            f"(buckets per complex x {args.group_by}), "
            f"features: the seven raw interface terms ({', '.join(FEATURE_COLS)}), preprocessing `{args.scaling}`, "
            f"label `{args.rmsd_col}` ≤ {args.cutoff:g} Å, selected {'trees' if args.model == 'tree' else 'epoch'} {best}.", "",
            "Rates are percent of docking runs (one run = 300 poses of one complex and protocol). The unmodified row shows the "
            "absolute Success@k of the interface delta; the approach row the paired difference to it on the same runs with its "
            "percentile-bootstrap 95 % interval. `MRR` mean reciprocal rank of the first hit (runs without a hit count 0); "
            "`earlier / same / later` percent of the runs with a hit under both scorers whose first hit moves to a better / the "
            "same / a worse rank; `Δ log2(rank+1)` the mean log-rank shift (negative = earlier).", ""]
    if "weights" in extras:
        head += ["Learned Rosetta weights (`weights.wts`):", "", frame_markdown(extras["weights"]), ""]
    if "importance" in extras:
        head += ["Feature importance of the tree (gain / split counts):", "", frame_markdown(extras["importance"], ".0f"), ""]
    (out / "results.md").write_text("\n".join(head + md))
    log(f"[{run_name}] done in {(time.time() - t0) / 60:.1f} min -> {out}")
    log.close()
    return out


def main(argv=None):
    args = parse(argv)
    device = select_device(args.device)
    t0 = time.time()
    print(f"loading PDBbind poses for {args.protocol} ...", flush=True)
    pdbbind = load_poses("pdbbind", args.protocol, args.extract_dir)
    sanity = baseline_sanity(pdbbind, args.rmsd_col, args.cutoff)
    print(f"sanity, all {pdbbind['pdb'].nunique():,} complete PDBbind complexes: {json.dumps(sanity)}", flush=True)
    overlap_ids = {} if args.no_exclude_overlap else external_pdb_ids(("astex", "posebuster"), args.extract_dir)
    external: dict = {"pdbbind": {}}
    for proto in args.eval_protocols:
        if proto not in args.protocol:
            external["pdbbind"][proto] = load_poses("pdbbind", proto, args.extract_dir)
        for set_name in args.eval_sets:
            if set_name != "pdbbind":
                external.setdefault(set_name, {})[proto] = load_poses(set_name, proto, args.extract_dir)
    print(f"data loaded in {time.time() - t0:.0f}s; splits: {args.splits}", flush=True)
    for split in args.splits:
        train_one(args, split, pdbbind, overlap_ids, external, sanity, device)


if __name__ == "__main__":
    main()
