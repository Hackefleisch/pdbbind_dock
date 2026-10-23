"""
affinity.basins
---------------
Inventory Appendix A.3 step 2a — item 1 of step 2, decided 2026-10-22: the pose representation
of the ensemble question made discrete. Every docking run (one complex × one protocol; the
perturb run of each receptor state by default, all three sets) is reduced to its **basins** by a
greedy leader clustering of the run's poses on their pairwise symmetry-corrected in-place ligand
RMSD, the basin's minimum being its representative.

Per docking run:

pairwise RMSD
    the heavy-atom coordinates of the ligand of every pose, reconstructed from the line diffs of
    the HDF5 file (``Result.docked_ligand_xyz``), and the symmetry-corrected in-place RMSD of
    every pose pair (``LigandSymmetry.pairwise_rmsd``: 44,850 pairs for 300 poses). The condensed
    upper triangles are stored as float16 in ``rmsd/<set>.h5`` (dataset ``<pdb>/<protocol>``,
    pose order = ``seq_id``; :func:`load_rmsd_matrix`), so other cutoffs or clusterings need no
    recomputation.
leader clustering (:func:`leader_clusters`)
    the poses are sorted by the score (``idelta_score``, the ranking score of the track, lower is
    better; ``--score total`` for the constraint-free total); the best pose founds the first
    basin; every following pose joins the **first** founder (in score order) within the cutoff,
    or founds a new basin. The founder is the basin's minimum; basins are numbered in founder
    order (0 = the basin of the best pose). Two cutoffs from the same matrix: 1 Å (the first test
    of the decision) and 2 Å (the legacy grid's value, the reference).
per basin (``basins.parquet``)
    population (members including the founder) and its share of the run; founder score, total,
    RMSD to the crystal ligand and score rank; the gap of the founder score to the run minimum;
    the width = mean / max RMSD of the members to the founder (NaN for a singleton) and the mean
    pairwise RMSD within the basin; the number of hits among the members (symmetry-corrected
    ``rmsd_to_crystal`` ≤ 2 Å) and ``native`` = the founder is a hit.
per run (``runs.parquet``)
    number of basins and of singletons, basins whose founder lies within 2 / 5 REU of the run
    minimum, the best basin (population, width, native, hits), the native basin = the lowest
    basin whose founder is a hit (index, gap, population, width, founder rank, share of the run's
    hits), the number of native basins and of basins containing a hit, the entropy of the
    population distribution and its effective number of basins, the largest basin.
per pose (``poses.parquet``)
    score, total, RMSD to the crystal ligand, score rank; per cutoff the basin id, the founder
    flag and the RMSD to the own founder; the RMSD to the best pose and to the nearest other pose.

``summary.{csv,md}`` give the medians per set × protocol × cutoff; ``config.json`` the call.
Only complete complexes (``complexes.parquet`` of the extraction) are processed.

Usage::

    uv run python -m affinity.basins                                   # three perturb protocols, three sets, 1 and 2 Å
    uv run python -m affinity.basins --limit 20 --out-dir /tmp/basins  # smoke test
    uv run python -m affinity.basins --score total                     # founders by the constraint-free total score
"""

from __future__ import annotations

import os

# One BLAS thread per worker: the Gram products of pairwise_rmsd would otherwise spawn a thread
# pool per worker process and oversubscribe the machine (set before numpy is imported).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
from threadpoolctl import threadpool_limits
import pandas as pd
from scipy.spatial.distance import squareform

from docking.result import Result
from docking_analysis.constants import BASINS_DIR, H5_DIRS
from docking_analysis.extract import _ParquetSink

from .data import HIT_CUTOFF, PROTOCOLS, complex_info

SCHEMA = 1
SETS = ("pdbbind", "posebuster", "astex")
PERTURB_PROTOCOLS = tuple(p for p in PROTOCOLS if p.endswith("_perturb"))
DEFAULT_CUTOFFS = (1.0, 2.0)
SCORES = {"idelta": "idelta_score", "total": "total_score"}
GAPS = (2.0, 5.0)                   # REU: basins whose founder lies within these of the run minimum are counted per run
KEYS = ["set", "pdb", "protocol", "receptor_state"]


def cutoff_tag(cutoff: float) -> str:
    """Column suffix of a cutoff: 1 -> ``1``, 2.5 -> ``2p5``."""
    return f"{cutoff:g}".replace(".", "p")


# ---------------------------------------------------------------------------
# clustering
# ---------------------------------------------------------------------------
def leader_clusters(D: np.ndarray, order: np.ndarray, cutoff: float) -> tuple[np.ndarray, np.ndarray]:
    """Greedy leader clustering on a pairwise distance matrix.

    ``order`` lists the poses to cluster from the best to the worst score. The first pose founds
    basin 0; every following pose joins the first founder (in founder order) whose distance is
    ≤ ``cutoff``, otherwise it founds the next basin. Returns ``labels`` (basin id per pose, −1
    for poses not in ``order``) and ``founders`` (the founder pose of every basin, in basin order).
    """
    labels = np.full(D.shape[0], -1, dtype=np.int32)
    founders = np.empty(len(order), dtype=np.int64)
    k = 0
    for p in order:
        if k:
            near = np.flatnonzero(D[p, founders[:k]] <= cutoff)
            if len(near):
                labels[p] = near[0]
                continue
        founders[k] = p
        labels[p] = k
        k += 1
    return labels, founders[:k]


def analyse_run(D: np.ndarray, valid: np.ndarray, score: np.ndarray, total: np.ndarray, rmsd_c: np.ndarray,
                cutoffs=DEFAULT_CUTOFFS, hit_cutoff: float = HIT_CUTOFF, gaps=GAPS) -> tuple[dict, list[dict], list[dict]]:
    """Basins of one docking run at every cutoff.

    ``D`` the pairwise RMSD matrix of the run's poses, ``valid`` which poses have coordinates,
    ``score`` the score that orders the poses (lower is better), ``total`` the constraint-free
    total, ``rmsd_c`` the symmetry-corrected RMSD to the crystal ligand. Returns the per-pose
    columns (dict of arrays), the basin rows and the run rows (one per cutoff).
    """
    n = len(score)
    order = np.flatnonzero(valid)[np.argsort(score[valid], kind="stable")]
    rank = np.full(n, -1, dtype=np.int32)
    rank[order] = np.arange(len(order))
    hit = rmsd_c <= hit_cutoff
    best = int(order[0])
    e_min = float(score[best])
    n_hits = int(hit[valid].sum())
    Dv = D.copy()
    np.fill_diagonal(Dv, np.nan)
    with np.errstate(all="ignore"):
        d_nearest = np.nanmin(Dv, axis=1) if len(order) > 1 else np.full(n, np.nan)
    per_pose = {"score": score, "total": total, "rmsd_to_crystal": rmsd_c, "rank": rank, "hit": hit, "valid": valid,
                "d_best": D[:, best], "d_nearest": d_nearest}
    basin_rows, run_rows = [], []
    for c in cutoffs:
        tag = cutoff_tag(c)
        labels, founders = leader_clusters(D, order, c)
        is_founder = np.zeros(n, dtype=bool)
        is_founder[founders] = True
        d_founder = np.full(n, np.nan)
        m = labels >= 0
        d_founder[m] = D[m, founders[labels[m]]]
        per_pose[f"basin_{tag}"] = labels.astype(np.int16)
        per_pose[f"founder_{tag}"] = is_founder
        per_pose[f"d_founder_{tag}"] = d_founder
        nb = len(founders)
        pop = np.bincount(labels[m], minlength=nb)
        rows = []
        for b, f in enumerate(founders):
            members = np.flatnonzero(labels == b)
            others = members[members != f]
            dm = D[others, f]
            pair = D[np.ix_(members, members)][np.triu_indices(len(members), 1)]
            rows.append({"cutoff": c, "basin": b, "founder_seq_id": int(f), "founder_rank": int(rank[f]),
                         "population": int(len(members)), "population_frac": len(members) / len(order),
                         "founder_score": float(score[f]), "founder_total": float(total[f]), "gap": float(score[f] - e_min),
                         "founder_rmsd_to_crystal": float(rmsd_c[f]), "native": bool(hit[f]),
                         "n_hits": int(hit[members].sum()), "min_rmsd_to_crystal": float(rmsd_c[members].min()),
                         "width_mean": float(dm.mean()) if len(dm) else np.nan, "width_max": float(dm.max()) if len(dm) else np.nan,
                         "width_pair_mean": float(pair.mean()) if len(pair) else np.nan,
                         "mean_score": float(score[members].mean())})
        basin_rows += rows
        gap = np.array([r["gap"] for r in rows])
        native = np.array([r["native"] for r in rows], dtype=bool)
        n_hit_b = np.array([r["n_hits"] for r in rows])
        p = pop / pop.sum()
        entropy = float(-(p * np.log(p)).sum())
        nat = np.flatnonzero(native)
        nb_nat = int(nat[0]) if len(nat) else -1
        largest = int(np.argmax(pop))
        run = {"cutoff": c, "n_poses": n, "n_clustered": int(len(order)), "n_basins": nb, "n_singletons": int((pop == 1).sum()),
               **{f"n_basins_within{g:g}": int((gap <= g).sum()) for g in gaps},
               "n_native_basins": int(native.sum()), "n_basins_with_hit": int((n_hit_b > 0).sum()), "n_hits": n_hits,
               "e_min": e_min, "best_rmsd_to_crystal": float(rmsd_c[best]), "best_hit": bool(hit[best]),
               "best_population": int(pop[0]), "best_population_frac": float(p[0]),
               "best_width_mean": rows[0]["width_mean"], "best_width_max": rows[0]["width_max"], "best_n_hits": int(n_hit_b[0]),
               "native_basin": nb_nat,
               "native_gap": float(gap[nb_nat]) if nb_nat >= 0 else np.nan,
               "native_population": int(pop[nb_nat]) if nb_nat >= 0 else -1,
               "native_population_frac": float(p[nb_nat]) if nb_nat >= 0 else np.nan,
               "native_width_mean": rows[nb_nat]["width_mean"] if nb_nat >= 0 else np.nan,
               "native_founder_rank": rows[nb_nat]["founder_rank"] if nb_nat >= 0 else -1,
               "native_hits_frac": float(n_hit_b[nb_nat] / n_hits) if (nb_nat >= 0 and n_hits) else np.nan,
               "pop_entropy": entropy, "n_eff": float(np.exp(entropy)),
               "largest_basin": largest, "largest_population": int(pop[largest]), "largest_is_best": bool(largest == 0),
               "largest_native": bool(native[largest])}
        run_rows.append(run)
    return per_pose, basin_rows, run_rows


# ---------------------------------------------------------------------------
# per-file worker
# ---------------------------------------------------------------------------
def basins_one(h5_path: str, set_name: str, protocols=PERTURB_PROTOCOLS, cutoffs=DEFAULT_CUTOFFS, score: str = "idelta",
               hit_cutoff: float = HIT_CUTOFF, keep_matrices: bool = True) -> dict:
    """Basins of every requested protocol of one HDF5 file: ``{"pdb", "set", "error", "poses",
    "basins", "runs", "matrices"}`` (frames; ``matrices`` = protocol -> condensed float16 RMSDs)."""
    pdb = Path(h5_path).stem
    out = {"pdb": pdb, "set": set_name, "error": None, "poses": None, "basins": None, "runs": None, "matrices": {}}
    try:
        threadpool_limits(limits=1)
        res = Result(h5_path, pdb, load_structure=True, autocorrect=True)
        xyz = res.docked_ligand_xyz()
        sym = res.symmetry
        df = res.docking_df
        prot = df["type"].to_numpy()
        score_col = SCORES[score]
        poses, basins, runs = [], [], []
        for p in protocols:
            rows = np.flatnonzero(prot == p)
            if len(rows) == 0:
                continue
            X = xyz[rows]
            valid = ~np.isnan(X).any(axis=(1, 2))
            D = sym.pairwise_rmsd(X)
            per_pose, b_rows, r_rows = analyse_run(
                D, valid, df[score_col].to_numpy(np.float64)[rows], df["total_score"].to_numpy(np.float64)[rows],
                df["rmsd_to_crystal"].to_numpy(np.float64)[rows], cutoffs, hit_cutoff)
            meta = {"set": set_name, "pdb": pdb, "protocol": p, "receptor_state": "_".join(p.split("_")[:-2])}
            pp = pd.DataFrame(per_pose)
            pp.insert(0, "seq_id", np.arange(len(rows), dtype=np.int16))
            for k, v in reversed(meta.items()):
                pp.insert(0, k, v)
            poses.append(pp)
            bb = pd.DataFrame(b_rows)
            for k, v in reversed(meta.items()):
                bb.insert(0, k, v)
            basins.append(bb)
            rr = pd.DataFrame(r_rows)
            for k, v in reversed(meta.items()):
                rr.insert(0, k, v)
            rr["n_automorphisms"] = sym.n_automorphisms
            rr["automorphisms_capped"] = sym.capped
            rr["n_heavy"] = sym.n_heavy
            runs.append(rr)
            if keep_matrices:
                out["matrices"][p] = D[np.triu_indices(len(rows), 1)].astype(np.float16)
        if poses:
            out["poses"], out["basins"], out["runs"] = (pd.concat(x, ignore_index=True) for x in (poses, basins, runs))
    except Exception as e:  # unreadable file, broken structure
        out["error"] = f"{type(e).__name__}: {e}"
    return out


# ---------------------------------------------------------------------------
# stored matrices
# ---------------------------------------------------------------------------
def load_rmsd_matrix(set_name: str, pdb: str, protocol: str, base: Path | str = BASINS_DIR) -> np.ndarray:
    """The stored pairwise RMSD matrix of one run as a square float32 array (pose order = ``seq_id``)."""
    with h5py.File(Path(base) / "rmsd" / f"{set_name}.h5", "r") as f:
        v = f[pdb][protocol][()].astype(np.float32)
    return squareform(v)


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
def _q(x: pd.Series) -> tuple[float, float, float]:
    x = x.dropna()
    return (np.nan, np.nan, np.nan) if x.empty else (float(x.median()), float(x.quantile(0.25)), float(x.quantile(0.75)))


def summarise(runs: pd.DataFrame, basins: pd.DataFrame) -> pd.DataFrame:
    """Medians [IQR] and percentages per set × protocol × cutoff over the runs; basin-level
    shares pooled over the basins of those runs."""
    rows = []
    for (s, proto, c), r in runs.groupby(["set", "protocol", "cutoff"], observed=True, sort=False):
        b = basins[(basins["set"] == s) & (basins["protocol"] == proto) & (basins["cutoff"] == c)]
        has_nat = r["native_basin"] >= 0
        nat_not_best = has_nat & (r["native_basin"] > 0)
        nb, ns, w2, w5 = _q(r["n_basins"]), _q(r["n_singletons"]), _q(r["n_basins_within2"]), _q(r["n_basins_within5"])
        bp, bf, bw, bwx = _q(r["best_population"]), _q(r["best_population_frac"]), _q(r["best_width_mean"]), _q(r["best_width_max"])
        ne, lp = _q(r["n_eff"]), _q(r["largest_population"])
        ng, nr, npop = _q(r.loc[nat_not_best, "native_gap"]), _q(r.loc[nat_not_best, "native_basin"]), _q(r.loc[has_nat, "native_population"])
        nw, nhf = _q(r.loc[has_nat, "native_width_mean"]), _q(r.loc[has_nat, "native_hits_frac"])
        multi = b[b["population"] > 1]
        rows.append({
            "set": s, "protocol": proto, "cutoff": c, "runs": len(r), "runs_with_hit": int((r["n_hits"] > 0).sum()),
            "n_basins_median": nb[0], "n_basins_q25": nb[1], "n_basins_q75": nb[2],
            "n_singletons_median": ns[0], "pct_basins_singleton": 100 * (b["population"] == 1).mean(),
            "n_basins_within2_median": w2[0], "n_basins_within2_q75": w2[2],
            "n_basins_within5_median": w5[0], "n_basins_within5_q75": w5[2],
            "n_eff_median": ne[0], "n_eff_q25": ne[1], "n_eff_q75": ne[2],
            "best_population_median": bp[0], "best_population_q25": bp[1], "best_population_q75": bp[2],
            "best_population_frac_median": bf[0],
            "best_width_mean_median": bw[0], "best_width_mean_q25": bw[1], "best_width_mean_q75": bw[2], "best_width_max_median": bwx[0],
            "width_mean_median_all": float(multi["width_mean"].median()) if len(multi) else np.nan,
            "pct_best_native": 100 * r["best_hit"].mean(),
            "pct_native_exists": 100 * has_nat.mean(),
            "pct_native_exists_of_runs_with_hit": 100 * has_nat[r["n_hits"] > 0].mean() if (r["n_hits"] > 0).any() else np.nan,
            "pct_native_not_best_given_exists": 100 * (nat_not_best.sum() / has_nat.sum()) if has_nat.any() else np.nan,
            "native_basin_index_median": nr[0], "native_basin_index_q75": nr[2],
            "native_gap_median": ng[0], "native_gap_q25": ng[1], "native_gap_q75": ng[2],
            "native_population_median": npop[0], "native_population_q75": npop[2],
            "native_width_mean_median": nw[0], "native_hits_frac_median": nhf[0],
            "pct_largest_is_best": 100 * r["largest_is_best"].mean(),
            "pct_largest_native": 100 * r["largest_native"].mean(),
            "largest_population_median": lp[0], "largest_population_q75": lp[2],
            "n_native_basins_median": float(r["n_native_basins"].median()),
        })
    return pd.DataFrame(rows)


def summary_markdown(summary: pd.DataFrame) -> str:
    out = []
    head = ["set", "protocol", "cutoff (Å)", "runs", "basins per run, median [IQR]", "singletons per run, median", "% of basins singletons",
            "basins within 2 REU of the minimum, median (q75)", "within 5 REU", "effective number of basins, median [IQR]"]
    out += ["Number of basins per docking run:", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for _, r in summary.iterrows():
        out.append(f"| {r.set} | `{r.protocol}` | {r.cutoff:g} | {r.runs:,} | {r.n_basins_median:.0f} [{r.n_basins_q25:.0f}, {r.n_basins_q75:.0f}] | "
                   f"{r.n_singletons_median:.0f} | {r.pct_basins_singleton:.0f} | {r.n_basins_within2_median:.0f} ({r.n_basins_within2_q75:.0f}) | "
                   f"{r.n_basins_within5_median:.0f} ({r.n_basins_within5_q75:.0f}) | {r.n_eff_median:.1f} [{r.n_eff_q25:.1f}, {r.n_eff_q75:.1f}] |")
    head = ["set", "protocol", "cutoff (Å)", "best basin: population, median [IQR]", "share of the run, median", "width (mean RMSD to the founder), median [IQR]",
            "width (max), median", "% runs best basin native", "% runs largest basin = best", "% runs largest basin native", "largest basin population, median (q75)"]
    out += ["", "The basin of the best-scoring pose (`native` = its founder is within 2 Å of the crystal ligand; the share of runs whose best basin is "
            "native equals the benchmark's Success@1 of the score):", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for _, r in summary.iterrows():
        out.append(f"| {r.set} | `{r.protocol}` | {r.cutoff:g} | {r.best_population_median:.0f} [{r.best_population_q25:.0f}, {r.best_population_q75:.0f}] | "
                   f"{100 * r.best_population_frac_median:.0f} % | {r.best_width_mean_median:.2f} [{r.best_width_mean_q25:.2f}, {r.best_width_mean_q75:.2f}] | "
                   f"{r.best_width_max_median:.2f} | {r.pct_best_native:.1f} | {r.pct_largest_is_best:.1f} | {r.pct_largest_native:.1f} | "
                   f"{r.largest_population_median:.0f} ({r.largest_population_q75:.0f}) |")
    head = ["set", "protocol", "cutoff (Å)", "% runs with a native basin", "% of the runs with a hit", "native basins per run, median",
            "% native basin is not the best (of runs with one)", "its index, median (q75)", "its gap to the minimum (REU), median [IQR]",
            "native basin population, median (q75)", "its width, median", "share of the run's hits in it, median"]
    out += ["", "The native basin (the lowest basin whose founder is a hit):", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for _, r in summary.iterrows():
        out.append(f"| {r.set} | `{r.protocol}` | {r.cutoff:g} | {r.pct_native_exists:.1f} | {r.pct_native_exists_of_runs_with_hit:.1f} | "
                   f"{r.n_native_basins_median:.0f} | {r.pct_native_not_best_given_exists:.1f} | {r.native_basin_index_median:.0f} ({r.native_basin_index_q75:.0f}) | "
                   f"{r.native_gap_median:.2f} [{r.native_gap_q25:.2f}, {r.native_gap_q75:.2f}] | {r.native_population_median:.0f} ({r.native_population_q75:.0f}) | "
                   f"{r.native_width_mean_median:.2f} | {r.native_hits_frac_median:.2f} |")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def _git_hash() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def out_dir_for(score: str, base: Path = BASINS_DIR) -> Path:
    return base if score == "idelta" else Path(f"{base}_{score}")


def run(sets, protocols, cutoffs, score: str, out: Path, workers: int, limit: int | None, keep_matrices: bool,
        hit_cutoff: float = HIT_CUTOFF, extract_dir=None, overwrite: bool = False) -> None:
    out.mkdir(parents=True, exist_ok=True)
    targets = {n: out / f"{n}.parquet" for n in ("poses", "basins", "runs")}
    existing = [p.name for p in targets.values() if p.exists()]
    if existing and not overwrite:
        sys.exit(f"outputs exist in {out} (use --overwrite): {existing}")
    files: list[tuple[str, str]] = []
    counts = {}
    for s in sets:
        complete = set(complex_info(s, extract_dir)["pdb"])
        fs = sorted((str(p), s) for p in H5_DIRS[s].glob("*.h5") if p.stem in complete)
        if limit:
            fs = fs[:limit]
        counts[s] = {"complete": len(complete), "files": len(fs)}
        files += fs
    per_set = ", ".join(f"{s}: {c['files']:,}" for s, c in counts.items())
    print(f"{len(files):,} files ({per_set}) -> {out} with {workers} workers; protocols {list(protocols)}, "
          f"cutoffs {list(cutoffs)}, score {score}", flush=True)
    sinks = {n: _ParquetSink(p, flush_rows={"poses": 400_000, "basins": 200_000, "runs": 20_000}[n]) for n, p in targets.items()}
    stores: dict[str, h5py.File] = {}
    if keep_matrices:
        (out / "rmsd").mkdir(exist_ok=True)
        for s in sets:
            stores[s] = h5py.File(out / "rmsd" / f"{s}.h5", "w")
            stores[s].attrs["description"] = ("condensed upper triangle (scipy squareform order) of the pairwise symmetry-corrected "
                                              "in-place heavy-atom ligand RMSD of the poses of one docking run, float16, pose order = seq_id")
    errors = []
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(basins_one, fp, s, protocols, cutoffs, score, hit_cutoff, keep_matrices) for fp, s in files]
        for fut in as_completed(futures):
            r = fut.result()
            done += 1
            if r["error"] is not None:
                errors.append({"set": r["set"], "pdb": r["pdb"], "error": r["error"]})
            else:
                for n in sinks:
                    sinks[n].add(r[n])
                if keep_matrices:
                    g = stores[r["set"]].require_group(r["pdb"])
                    for p, v in r["matrices"].items():
                        d = g.create_dataset(p, data=v)
                        d.attrs["n"] = int(round((1 + np.sqrt(1 + 8 * len(v))) / 2))
            if done % 200 == 0 or done == len(files):
                el = time.time() - t0
                print(f"  {done:,}/{len(files):,} files  {el / 60:.1f} min  ({done / el:.1f} files/s)  {len(errors)} errors", flush=True)
    for sk in sinks.values():
        sk.close()
    for st in stores.values():
        st.close()
    runs = pd.read_parquet(targets["runs"])
    basins = pd.read_parquet(targets["basins"], columns=["set", "protocol", "cutoff", "population", "width_mean"])
    summary = summarise(runs, basins)
    summary.to_csv(out / "summary.csv", index=False)
    md = [f"# Basins per docking run — leader clustering on the pairwise symmetry-corrected ligand RMSD, founders by `{SCORES[score]}`, "
          f"hit ≤ {hit_cutoff:g} Å", "", summary_markdown(summary), ""]
    (out / "summary.md").write_text("\n".join(md))
    config = {"schema": SCHEMA, "sets": list(sets), "protocols": list(protocols), "cutoffs": list(cutoffs), "score": score,
              "score_column": SCORES[score], "hit_cutoff": hit_cutoff, "keep_matrices": keep_matrices, "limit": limit,
              "workers": workers, "counts": counts, "n_runs": int(len(runs) / len(cutoffs)), "n_errors": len(errors), "errors": errors,
              "git_hash": _git_hash(), "created": time.strftime("%Y-%m-%d %H:%M:%S"), "elapsed_s": time.time() - t0}
    (out / "config.json").write_text(json.dumps(config, indent=1))
    print("\n".join(md))
    print(f"written: {out} ({config['n_runs']:,} runs, {sinks['poses'].n_rows:,} poses, {sinks['basins'].n_rows:,} basin rows, "
          f"{len(errors)} errors) in {(time.time() - t0) / 60:.1f} min", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="+", default=list(SETS), choices=SETS)
    ap.add_argument("--protocols", nargs="+", default=list(PERTURB_PROTOCOLS), choices=PROTOCOLS)
    ap.add_argument("--cutoffs", nargs="+", type=float, default=list(DEFAULT_CUTOFFS), help="RMSD cutoffs of the leader clustering (Å)")
    ap.add_argument("--score", default="idelta", choices=tuple(SCORES), help="the score that orders the poses (founder = minimum)")
    ap.add_argument("--hit-cutoff", type=float, default=HIT_CUTOFF, help="hit / native: rmsd_to_crystal <= this (Å)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--limit", type=int, default=None, help="only the first N complete complexes per set (smoke tests)")
    ap.add_argument("--no-matrices", action="store_true", help="do not store the pairwise RMSD matrices (rmsd/<set>.h5)")
    ap.add_argument("--out-dir", type=Path, default=None, help=f"default {BASINS_DIR} (suffixed _<score> for --score total)")
    ap.add_argument("--extract-dir", type=Path, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    out = args.out_dir or out_dir_for(args.score)
    run(args.sets, tuple(args.protocols), tuple(args.cutoffs), args.score, out, args.workers, args.limit, not args.no_matrices,
        args.hit_cutoff, args.extract_dir, args.overwrite)


if __name__ == "__main__":
    main()
