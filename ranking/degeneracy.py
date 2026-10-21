"""
ranking.degeneracy
------------------
Inventory S6 — *why* the ranking resists repair: the energy-degeneracy analysis (Appendix
A.1) and the per-complex oracle probe (Appendix A.2), computed per **docking run** (one
complex × one protocol — the Perturb run of each receptor state by default) on the same seven
raw interface-delta terms as the rest of the track (``--terms total``: the whole-complex
``raw_*`` energies of the same seven terms, the robustness check of A.1).

Per run (:func:`analyse_run`):

correct / incorrect
    a pose is *correct* when the symmetry-corrected ``rmsd_to_crystal`` ≤ ``cutoff`` (2 Å),
    *incorrect* otherwise; *far* when the RMSD > ``far`` (5 Å, not a near-miss). Runs without
    a correct pose are excluded (counted); runs with exactly one correct pose give
    incorrect→correct distances but no correct–correct reference and no oracle.
energy vector
    the seven terms z-scored **over the run's own poses** (a term without variance in the run
    drops out). Distances: Euclidean on the z-scores (``eu``, the default of A.1) and
    Mahalanobis (``ma``: Euclidean after whitening with the run's own covariance of the
    z-scores; eigenvalues below 1e-6 of the largest are dropped), the variant for the strongly
    correlated terms.
nearest neighbours
    for every pose the distance to the nearest *other* correct pose (``d_correct``) and to the
    nearest other incorrect pose (``d_incorrect``). The three populations of A.1 are then
    incorrect→correct = ``d_correct`` of the incorrect poses (the quantity d_i), correct→correct
    = ``d_correct`` of the correct poses (the within-basin spread), incorrect→incorrect =
    ``d_incorrect`` of the incorrect poses (the density of the decoy cloud). The run's
    **degeneracy threshold** is its median correct–correct distance; an incorrect pose is
    *degenerate* when its d_i lies below it; the headline definition of an **energy-degenerate
    decoy** adds RMSD > ``far``. Δ idelta / Δ total to the nearest correct pose are kept per pose
    (negative = the decoy scores better than its nearest correct pose).
oracle (A.2)
    a logistic regression (L2, C = 1, balanced classes) on the z-scores of that run alone —
    the best linear separation of correct from incorrect poses *for this complex*, strictly
    more freedom than any global reweighting; reported in-sample (the optimistic bound) and
    cross-validated (stratified k-fold, k = min(5, n_hit), pooled out-of-fold scores). The
    unmodified AUC of the interface delta and of the total score on the same poses are the
    reference; the first-hit rank under every score gives Success@k as in the benchmark.

Outputs of the CLI (``<RANKING_DIR>/degeneracy[_total]/``): ``runs.parquet`` (one row per
analysed run), ``poses.parquet`` (one row per pose of the analysed runs), ``summary.csv`` /
``summary.md`` (the headline statistics per set × protocol, both metrics), ``config.json``.
The figures, the oracle-vs-approaches gap, the example complexes and the embeddings are made
by :mod:`ranking.degeneracy_report`.

Usage::

    uv run python -m ranking.degeneracy                       # three perturb protocols, three sets
    uv run python -m ranking.degeneracy --terms total         # robustness check (raw_* totals)
    uv run python -m ranking.degeneracy_report                # figures + tables
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from docking_analysis.constants import DEGENERACY_DIR

from .data import DELTA_TERMS, FEATURE_COLS, PROTOCOLS, Grouped, load_poses, term_name

SCHEMA = 1
SETS = ("pdbbind", "posebuster", "astex")
PERTURB_PROTOCOLS = tuple(p for p in PROTOCOLS if p.endswith("_perturb"))
TERM_SETS = {"delta": list(FEATURE_COLS), "total": [f"raw_{t}" for t in DELTA_TERMS]}
METRICS = ("eu", "ma")
METRIC_LABELS = {"eu": "Euclidean (z-scores)", "ma": "Mahalanobis"}
EIG_FLOOR = 1e-6


# ---------------------------------------------------------------------------
# per run
# ---------------------------------------------------------------------------
def zscore(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """z-score every column over the rows; columns without variance become 0 (``used`` False)."""
    X = np.asarray(X, dtype=np.float64)
    mu, sd = X.mean(0), X.std(0)
    used = sd > 0
    Z = np.zeros_like(X)
    Z[:, used] = (X[:, used] - mu[used]) / sd[used]
    return Z, used


def whiten(Z: np.ndarray, floor: float = EIG_FLOOR) -> np.ndarray:
    """Coordinates in which the Euclidean distance is the Mahalanobis distance of the run
    (eigen-decomposition of the covariance; directions with eigenvalue < floor × max dropped)."""
    if len(Z) < 2:
        return Z[:, :0]
    C = np.cov(Z, rowvar=False)
    w, V = np.linalg.eigh(np.atleast_2d(C))
    good = w > floor * max(w.max(), np.finfo(float).tiny)
    if not good.any():
        return Z[:, :0]
    return Z @ (V[:, good] / np.sqrt(w[good]))


def _nearest_other(A: np.ndarray, B: np.ndarray, same: bool) -> tuple[np.ndarray, np.ndarray]:
    """For every row of A the distance to, and index of, the nearest row of B (B ≠ the row
    itself when ``same``). NaN / -1 when there is no candidate."""
    n, m = len(A), len(B)
    if n == 0 or m == 0 or (same and m < 2):
        return np.full(n, np.nan), np.full(n, -1, dtype=np.int64)
    D = cdist(A, B) if A.shape[1] else np.zeros((n, m))       # no varying term: every distance is 0
    if same:
        np.fill_diagonal(D, np.inf)
    j = D.argmin(1)
    return D[np.arange(n), j], j


def first_hit_rank(score_lower_better: np.ndarray, hit: np.ndarray) -> float:
    """0-based rank of the first hit when the poses are sorted by the score (stable); NaN without a hit."""
    if not hit.any():
        return np.nan
    order = np.argsort(score_lower_better, kind="stable")
    return float(np.argmax(hit[order]))


def run_auc(score_higher_better: np.ndarray, hit: np.ndarray) -> float:
    """ROC AUC of hit vs non-hit within one run (rank statistic; NaN without both classes)."""
    n_pos, n_neg = int(hit.sum()), int((~hit).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    r = rankdata(score_higher_better)
    return float((r[hit].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def oracle_scores(Z: np.ndarray, hit: np.ndarray, max_folds: int = 5, seed: int = 0, C: float = 1.0
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """In-sample and out-of-fold decision values of the per-run logistic regression, its
    in-sample coefficients, and the number of folds (0 when no cross-validation was possible)."""
    n_pos, n_neg = int(hit.sum()), int((~hit).sum())
    y = hit.astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=1000).fit(Z, y)
        s_in = clf.decision_function(Z)
        k = min(max_folds, n_pos, n_neg)
        s_cv = np.full(len(Z), np.nan)
        if k >= 2:
            for tr, te in StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(Z, y):
                f = LogisticRegression(C=C, class_weight="balanced", max_iter=1000).fit(Z[tr], y[tr])
                s_cv[te] = f.decision_function(Z[te])
        else:
            k = 0
    return s_in, s_cv, clf.coef_[0], k


def analyse_run(X: np.ndarray, rmsd: np.ndarray, idelta: np.ndarray, total: np.ndarray, *, cutoff: float = 2.0,
                far: float = 5.0, max_folds: int = 5, seed: int = 0, term_names: list[str] | None = None,
                oracle: bool = True) -> tuple[dict, dict] | None:
    """A.1 distances and the A.2 oracle of one docking run; ``None`` when the run has no
    correct pose. Returns ``(per_pose, per_run)``: dicts of arrays / scalars (see module doc)."""
    rmsd = np.asarray(rmsd, dtype=np.float64)
    idelta = np.asarray(idelta, dtype=np.float64)
    total = np.asarray(total, dtype=np.float64)
    hit = rmsd <= cutoff
    n, n_hit = len(rmsd), int(hit.sum())
    if n_hit == 0:
        return None
    miss = ~hit
    n_miss = n - n_hit
    is_far = rmsd > far
    Z, used = zscore(X)
    coords = {"eu": Z, "ma": whiten(Z)}
    idx_hit, idx_miss = np.flatnonzero(hit), np.flatnonzero(miss)
    rank_idelta = rankdata(idelta, method="ordinal") - 1
    best = int(np.argsort(idelta, kind="stable")[0])

    per_pose = {"hit": hit, "rmsd": rmsd, "idelta": idelta, "total": total, "rank_idelta": rank_idelta.astype(np.int16)}
    run = {"n": n, "n_hit": n_hit, "n_miss": n_miss, "n_far": int(is_far.sum()), "n_terms_used": int(used.sum()),
           "first_hit_rank_idelta": first_hit_rank(idelta, hit), "first_hit_rank_total": first_hit_rank(total, hit),
           "auc_idelta": run_auc(-idelta, hit), "auc_total": run_auc(-total, hit),
           "best_rmsd": rmsd[best], "best_hit": bool(hit[best]), "best_far": bool(is_far[best]),
           "best_idelta": idelta[best]}
    for m, Q in coords.items():
        d_c = np.full(n, np.nan)
        nn_c = np.full(n, -1, dtype=np.int64)
        d_c[idx_miss], j = _nearest_other(Q[idx_miss], Q[idx_hit], same=False)
        nn_c[idx_miss] = idx_hit[j] if len(j) else -1
        d_c[idx_hit], j = _nearest_other(Q[idx_hit], Q[idx_hit], same=True)
        nn_c[idx_hit] = np.where(j >= 0, idx_hit[np.clip(j, 0, None)], -1) if n_hit >= 2 else -1
        d_i = np.full(n, np.nan)
        d_i[idx_miss], _ = _nearest_other(Q[idx_miss], Q[idx_miss], same=True)
        d_i[idx_hit], _ = _nearest_other(Q[idx_hit], Q[idx_miss], same=False)
        per_pose[f"d_correct_{m}"] = d_c
        per_pose[f"d_incorrect_{m}"] = d_i
        if m == "eu":                          # the nearest correct pose and the score differences to it
            has = nn_c >= 0
            per_pose["nn_correct_idx"] = nn_c
            per_pose["d_idelta_nn"] = np.where(has, idelta - idelta[np.clip(nn_c, 0, None)], np.nan)
            per_pose["d_total_nn"] = np.where(has, total - total[np.clip(nn_c, 0, None)], np.nan)
        cc = d_c[idx_hit]
        median_cc = float(np.nanmedian(cc)) if n_hit >= 2 else np.nan
        degenerate = miss & (d_c < median_cc) if n_hit >= 2 else np.zeros(n, dtype=bool)
        per_pose[f"degenerate_{m}"] = degenerate
        ic, ii = d_c[idx_miss], d_i[idx_miss]
        run.update({
            f"median_cc_{m}": median_cc,
            f"median_ic_{m}": float(np.nanmedian(ic)) if n_miss else np.nan,
            f"median_ii_{m}": float(np.nanmedian(ii)) if n_miss >= 2 else np.nan,
            f"frac_degenerate_{m}": float(degenerate[idx_miss].mean()) if (n_hit >= 2 and n_miss) else np.nan,
            f"frac_degenerate_far_{m}": float((degenerate & is_far)[idx_miss].mean()) if (n_hit >= 2 and n_miss) else np.nan,
            f"frac_degenerate_far_better_{m}": float((degenerate & is_far & (per_pose["d_idelta_nn"] <= 0))[idx_miss].mean())
            if (n_hit >= 2 and n_miss) else np.nan,
            f"frac_far_degenerate_{m}": float(degenerate[miss & is_far].mean()) if (n_hit >= 2 and (miss & is_far).any()) else np.nan,
            f"best_d_{m}": d_c[best] if miss[best] else np.nan,
            f"best_degenerate_{m}": bool(degenerate[best]),
            f"best_degenerate_far_{m}": bool(degenerate[best] and is_far[best]),
        })
    run["best_d_idelta_nn"] = per_pose["d_idelta_nn"][best] if miss[best] else np.nan

    # ---- oracle (A.2) ----
    names = term_names or [f"t{i}" for i in range(X.shape[1])]
    if oracle and n_hit >= 2 and n_miss >= 2:
        s_in, s_cv, coef, k = oracle_scores(Z, hit, max_folds, seed)
        per_pose["oracle_in"] = s_in
        per_pose["oracle_cv"] = s_cv
        run.update({"oracle_folds": k, "auc_oracle_in": run_auc(s_in, hit),
                    "auc_oracle_cv": run_auc(s_cv, hit) if k else np.nan,
                    "first_hit_rank_oracle_in": first_hit_rank(-s_in, hit),
                    "first_hit_rank_oracle_cv": first_hit_rank(-s_cv, hit) if k else np.nan,
                    **{f"coef_{t}": c for t, c in zip(names, coef)}})
    else:
        per_pose["oracle_in"] = np.full(n, np.nan)
        per_pose["oracle_cv"] = np.full(n, np.nan)
        run.update({"oracle_folds": 0, "auc_oracle_in": np.nan, "auc_oracle_cv": np.nan,
                    "first_hit_rank_oracle_in": np.nan, "first_hit_rank_oracle_cv": np.nan,
                    **{f"coef_{t}": np.nan for t in names}})
    return per_pose, run


# ---------------------------------------------------------------------------
# all runs of a table
# ---------------------------------------------------------------------------
POSE_FLOAT32 = ("rmsd", "idelta", "total", "d_correct_eu", "d_correct_ma", "d_incorrect_eu", "d_incorrect_ma",
                "d_idelta_nn", "d_total_nn", "oracle_in", "oracle_cv")


def _batch(groups: list[tuple[int, int, int]], X, rmsd, idelta, total, kw) -> tuple[list, list]:
    poses, runs = [], []
    for g, a, b in groups:
        res = analyse_run(X[a:b], rmsd[a:b], idelta[a:b], total[a:b], **kw)
        if res is None:
            continue
        pp, rr = res
        pp["group"] = np.full(b - a, g, dtype=np.int64)
        rr["group"] = g
        poses.append(pp)
        runs.append(rr)
    return poses, runs


def analyse(frame: pd.DataFrame, feature_cols: list[str], *, rmsd_col: str = "rmsd_to_crystal", cutoff: float = 2.0,
            far: float = 5.0, max_folds: int = 5, seed: int = 0, n_jobs: int = 16, batch: int = 64,
            log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    """:func:`analyse_run` over every docking run of *frame* (as loaded by ``load_poses``).
    Returns ``(poses, runs)``: the per-pose table of the analysed runs (keys: set, pdb,
    protocol, receptor_state, seq_id) and the per-run table; the runs without a hit are
    counted in ``runs.attrs["n_runs_no_hit"]``."""
    g = Grouped.build(frame, frame[feature_cols].to_numpy(np.float64), rmsd_col, cutoff)
    X, rmsd = g.X.astype(np.float64), g.column(rmsd_col).astype(np.float64)
    idelta, total = g.column("idelta_score").astype(np.float64), g.column("total_score").astype(np.float64)
    kw = dict(cutoff=cutoff, far=far, max_folds=max_folds, seed=seed, term_names=[term_name(c) for c in feature_cols])
    groups = [(i, int(g.offsets[i]), int(g.offsets[i + 1])) for i in range(g.n_groups)]
    batches = [groups[i:i + batch] for i in range(0, len(groups), batch)]
    t0 = time.time()
    results = Parallel(n_jobs=n_jobs)(delayed(_batch)(bt, X, rmsd, idelta, total, kw) for bt in batches)
    log(f"analysed {g.n_groups:,} runs ({len(X):,} poses) in {time.time() - t0:.0f}s")
    pose_parts = [pp for poses, _ in results for pp in poses]
    run_rows = [rr for _, runs in results for rr in runs]
    keys = g.keys()
    runs = pd.DataFrame(run_rows)
    runs = pd.concat([keys.iloc[runs["group"].to_numpy()].reset_index(drop=True), runs.drop(columns="group")], axis=1)
    runs.attrs["n_runs_no_hit"] = int(g.n_groups - len(runs))
    if pose_parts:
        cols = {k: np.concatenate([pp[k] for pp in pose_parts]) for k in pose_parts[0]}
    else:
        cols = {}
    grp = cols.pop("group")
    rows = np.concatenate([np.arange(g.offsets[i], g.offsets[i + 1]) for i in np.unique(grp)]) if len(grp) else np.zeros(0, int)
    meta = g.frame.iloc[rows][["set", "pdb", "protocol", "receptor_state", "seq_id"]].reset_index(drop=True)
    poses = pd.concat([meta, pd.DataFrame(cols)], axis=1)
    for c in POSE_FLOAT32:
        poses[c] = poses[c].astype(np.float32)
    poses["nn_correct_idx"] = poses["nn_correct_idx"].astype(np.int16)
    for c in ("set", "pdb", "protocol", "receptor_state"):
        poses[c] = poses[c].astype("category")
    return poses, runs


# ---------------------------------------------------------------------------
# headline statistics
# ---------------------------------------------------------------------------
def _iqr(x: pd.Series) -> tuple[float, float, float]:
    x = x.dropna()
    return (np.nan, np.nan, np.nan) if x.empty else (float(x.median()), float(x.quantile(0.25)), float(x.quantile(0.75)))


def summarise(runs: pd.DataFrame, poses: pd.DataFrame, metric: str = "eu", far: float = 5.0) -> pd.DataFrame:
    """The headline statistics of A.1 / A.2 per (set, protocol) for one distance metric.

    Pose-level percentages are pooled over the incorrect poses of the *analysed* runs (≥ 2
    correct poses, so that the correct–correct reference exists); run-level percentages are
    over the same runs unless stated. ``top1_fail`` connects to the benchmark: the share of
    runs whose best-scoring pose (interface delta) is not a hit."""
    rows = []
    p_all = poses.merge(runs[["set", "pdb", "protocol", "n_hit"]], on=["set", "pdb", "protocol"], how="left")
    for (s, proto), r in runs.groupby(["set", "protocol"], observed=True):
        a = r[r["n_hit"] >= 2]
        p = p_all[(p_all["set"] == s) & (p_all["protocol"] == proto) & (p_all["n_hit"] >= 2) & ~p_all["hit"]]
        deg = p[f"degenerate_{metric}"].to_numpy(bool)
        isfar = p["rmsd"].to_numpy() > far
        better = p["d_idelta_nn"].to_numpy() <= 0
        fail = ~a["best_hit"]
        fd = _iqr(a[f"frac_degenerate_{metric}"])
        o_cv, o_in, o_id, o_tot = _iqr(a["auc_oracle_cv"]), _iqr(a["auc_oracle_in"]), _iqr(a["auc_idelta"]), _iqr(a["auc_total"])
        both = a[["auc_oracle_cv", f"frac_degenerate_{metric}"]].dropna()
        rho = spearmanr(both["auc_oracle_cv"], both[f"frac_degenerate_{metric}"])[0] if len(both) > 2 else np.nan
        cv = a[a["oracle_folds"] > 0]
        rows.append({
            "set": s, "protocol": proto, "metric": metric,
            "runs_with_hit": len(r), "runs_one_hit": int((r["n_hit"] == 1).sum()), "runs_analysed": len(a),
            "incorrect_poses": len(p), "far_poses": int(isfar.sum()),
            "pct_degenerate": 100 * deg.mean() if len(p) else np.nan,
            "pct_degenerate_far": 100 * (deg & isfar).mean() if len(p) else np.nan,
            "pct_degenerate_far_better": 100 * (deg & isfar & better).mean() if len(p) else np.nan,
            "pct_far_that_are_degenerate": 100 * deg[isfar].mean() if isfar.any() else np.nan,
            "pct_degenerate_that_score_better": 100 * better[deg].mean() if deg.any() else np.nan,
            "run_frac_degenerate_median": fd[0], "run_frac_degenerate_q25": fd[1], "run_frac_degenerate_q75": fd[2],
            "pct_runs_top1_fail": 100 * fail.mean(),
            "pct_runs_best_degenerate": 100 * a[f"best_degenerate_{metric}"].mean(),
            "pct_runs_best_degenerate_far": 100 * a[f"best_degenerate_far_{metric}"].mean(),
            "pct_top1_fail_degenerate": 100 * a.loc[fail, f"best_degenerate_{metric}"].mean() if fail.any() else np.nan,
            "pct_top1_fail_far": 100 * a.loc[fail, "best_far"].mean() if fail.any() else np.nan,
            "pct_top1_fail_degenerate_far": 100 * a.loc[fail, f"best_degenerate_far_{metric}"].mean() if fail.any() else np.nan,
            "auc_idelta_median": o_id[0], "auc_idelta_q25": o_id[1], "auc_idelta_q75": o_id[2],
            "auc_total_median": o_tot[0],
            "auc_oracle_cv_median": o_cv[0], "auc_oracle_cv_q25": o_cv[1], "auc_oracle_cv_q75": o_cv[2],
            "auc_oracle_in_median": o_in[0], "auc_oracle_in_q25": o_in[1], "auc_oracle_in_q75": o_in[2],
            "pct_runs_oracle_cv_below_0.9": 100 * (cv["auc_oracle_cv"] < 0.9).mean() if len(cv) else np.nan,
            "pct_runs_oracle_cv_below_0.8": 100 * (cv["auc_oracle_cv"] < 0.8).mean() if len(cv) else np.nan,
            "top1_idelta": 100 * (a["first_hit_rank_idelta"] < 1).mean(),
            "top1_oracle_cv": 100 * (cv["first_hit_rank_oracle_cv"] < 1).mean() if len(cv) else np.nan,
            "top1_oracle_in": 100 * (a["first_hit_rank_oracle_in"] < 1).mean(),
            "top5_idelta": 100 * (a["first_hit_rank_idelta"] < 5).mean(),
            "top5_oracle_cv": 100 * (cv["first_hit_rank_oracle_cv"] < 5).mean() if len(cv) else np.nan,
            "spearman_oracle_cv_vs_degenerate": rho,
        })
    return pd.DataFrame(rows)


def summary_markdown(summary: pd.DataFrame, far: float = 5.0) -> str:
    """The headline tables of one metric: counts, pose-level percentages, run-level percentages, oracle."""
    m = summary["metric"].iloc[0]
    out = [f"**Distance: {METRIC_LABELS[m]}.**", ""]
    head = ["set", "protocol", "runs with a hit", "runs with one hit (no reference)", "runs analysed (≥ 2 hits)",
            "incorrect poses", f"of which > {far:g} Å"]
    out += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for _, r in summary.iterrows():
        out.append(f"| {r.set} | `{r.protocol}` | {r.runs_with_hit:,} | {r.runs_one_hit:,} | {r.runs_analysed:,} | "
                   f"{r.incorrect_poses:,} | {r.far_poses:,} |")
    out += ["", "Incorrect poses (pooled over the analysed runs; *degenerate* = nearer to a correct pose than the run's "
            "median correct–correct distance):", ""]
    head = ["set", "protocol", "% degenerate", f"% degenerate and > {far:g} Å (**energy-degenerate decoys**)",
            f"% degenerate, > {far:g} Å and scoring ≤ nearest correct pose", f"% of the > {far:g} Å poses that are degenerate",
            "% of degenerate poses scoring ≤ nearest correct pose", "per-run fraction degenerate, median [IQR]"]
    out += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for _, r in summary.iterrows():
        out.append(f"| {r.set} | `{r.protocol}` | {r.pct_degenerate:.1f} | {r.pct_degenerate_far:.1f} | "
                   f"{r.pct_degenerate_far_better:.1f} | {r.pct_far_that_are_degenerate:.1f} | {r.pct_degenerate_that_score_better:.1f} | "
                   f"{r.run_frac_degenerate_median:.2f} [{r.run_frac_degenerate_q25:.2f}, {r.run_frac_degenerate_q75:.2f}] |")
    out += ["", "Runs (the best-scoring pose by interface delta; `Top-1 failure` = it is not a hit):", ""]
    head = ["set", "protocol", "% Top-1 failures", "% best pose a degenerate decoy", f"% best pose degenerate and > {far:g} Å",
            "of the Top-1 failures: % degenerate", f"% > {far:g} Å", f"% degenerate and > {far:g} Å"]
    out += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for _, r in summary.iterrows():
        out.append(f"| {r.set} | `{r.protocol}` | {r.pct_runs_top1_fail:.1f} | {r.pct_runs_best_degenerate:.1f} | "
                   f"{r.pct_runs_best_degenerate_far:.1f} | {r.pct_top1_fail_degenerate:.1f} | {r.pct_top1_fail_far:.1f} | "
                   f"{r.pct_top1_fail_degenerate_far:.1f} |")
    out += ["", "Per-run oracle (A.2; logistic regression on the run's z-scored terms; AUC median [IQR] over the analysed runs; "
            "Success@1 in % of those runs):", ""]
    head = ["set", "protocol", "AUC interface delta", "AUC total score", "AUC oracle, cross-validated", "AUC oracle, in-sample",
            "% runs oracle CV AUC < 0.9", "< 0.8", "Success@1 idelta / oracle CV / oracle in-sample",
            "Success@5 idelta / oracle CV", "Spearman ρ (oracle CV AUC, fraction degenerate)"]
    out += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for _, r in summary.iterrows():
        out.append(f"| {r.set} | `{r.protocol}` | {r.auc_idelta_median:.3f} [{r.auc_idelta_q25:.3f}, {r.auc_idelta_q75:.3f}] | "
                   f"{r.auc_total_median:.3f} | {r.auc_oracle_cv_median:.3f} [{r.auc_oracle_cv_q25:.3f}, {r.auc_oracle_cv_q75:.3f}] | "
                   f"{r.auc_oracle_in_median:.3f} [{r.auc_oracle_in_q25:.3f}, {r.auc_oracle_in_q75:.3f}] | "
                   f"{r['pct_runs_oracle_cv_below_0.9']:.1f} | {r['pct_runs_oracle_cv_below_0.8']:.1f} | "
                   f"{r.top1_idelta:.1f} / {r.top1_oracle_cv:.1f} / {r.top1_oracle_in:.1f} | {r.top5_idelta:.1f} / {r.top5_oracle_cv:.1f} | "
                   f"{r.spearman_oracle_cv_vs_degenerate:+.2f} |")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def out_dir_for(terms: str, base: Path = DEGENERACY_DIR) -> Path:
    return base if terms == "delta" else Path(f"{base}_{terms}")


def parse(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="+", default=list(SETS), choices=SETS)
    ap.add_argument("--protocols", nargs="+", default=list(PERTURB_PROTOCOLS), choices=PROTOCOLS)
    ap.add_argument("--terms", default="delta", choices=tuple(TERM_SETS), help="energy vector: raw_delta_* (default) or raw_* of the same seven terms")
    ap.add_argument("--rmsd-col", default="rmsd_to_crystal")
    ap.add_argument("--cutoff", type=float, default=2.0, help="correct: rmsd <= cutoff (Å)")
    ap.add_argument("--far", type=float, default=5.0, help="far decoy: rmsd > far (Å)")
    ap.add_argument("--max-folds", type=int, default=5, help="oracle cross-validation folds (min(max_folds, n_hit))")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=None, help=f"default {DEGENERACY_DIR} (suffixed _<terms> for --terms total)")
    ap.add_argument("--extract-dir", type=Path, default=None)
    ap.add_argument("--subsample", type=int, default=None, help="use only N random complexes per set (smoke tests)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse(argv)
    out = args.out_dir or out_dir_for(args.terms)
    out.mkdir(parents=True, exist_ok=True)
    feature_cols = TERM_SETS[args.terms]
    t0 = time.time()
    poses_all, runs_all, counts = [], [], {}
    for s in args.sets:
        frame = load_poses(s, args.protocols, args.extract_dir, feature_cols=feature_cols)
        if args.subsample:
            keep = np.random.default_rng(args.seed).choice(frame["pdb"].unique(), size=min(args.subsample, frame["pdb"].nunique()), replace=False)
            frame = frame[frame["pdb"].isin(keep)].reset_index(drop=True)
        print(f"[{s}] {frame['pdb'].nunique():,} complexes, {len(frame):,} poses loaded ({time.time() - t0:.0f}s)", flush=True)
        poses, runs = analyse(frame, feature_cols, rmsd_col=args.rmsd_col, cutoff=args.cutoff, far=args.far,
                              max_folds=args.max_folds, seed=args.seed, n_jobs=args.jobs)
        counts[s] = {"complexes": int(frame["pdb"].nunique()), "runs": int(len(runs) + runs.attrs["n_runs_no_hit"]),
                     "runs_no_hit": runs.attrs["n_runs_no_hit"], "runs_analysed": int((runs["n_hit"] >= 2).sum())}
        poses_all.append(poses)
        runs_all.append(runs)
        del frame
    poses = pd.concat(poses_all, ignore_index=True)
    runs = pd.concat(runs_all, ignore_index=True)
    for c in ("set", "pdb", "protocol", "receptor_state"):
        poses[c] = poses[c].astype(str).astype("category")
    poses.to_parquet(out / "poses.parquet", index=False)
    runs.to_parquet(out / "runs.parquet", index=False)
    summaries = pd.concat([summarise(runs, poses, m, args.far) for m in METRICS], ignore_index=True)
    summaries.to_csv(out / "summary.csv", index=False)
    md = [f"# Energy degeneracy (A.1) and per-run oracle (A.2) — terms `{args.terms}` "
          f"({', '.join(term_name(c) for c in feature_cols)}), correct ≤ {args.cutoff:g} Å, far > {args.far:g} Å", ""]
    for m in METRICS:
        md += [summary_markdown(summaries[summaries["metric"] == m], args.far), ""]
    (out / "summary.md").write_text("\n".join(md))
    config = {"schema": SCHEMA, **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              "feature_cols": feature_cols, "counts": counts, "elapsed_s": time.time() - t0}
    (out / "config.json").write_text(json.dumps(config, indent=1))
    print("\n".join(md))
    print(f"\nwritten: {out} ({len(runs):,} runs, {len(poses):,} poses) in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
