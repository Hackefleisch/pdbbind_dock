"""
docking_analysis.benchmark
--------------------------
Success-rate tables on the **full** extraction tables written by :mod:`docking_analysis.extract`
(``<EXTRACT_DIR>/<set>_full/``).

The evaluation is reported **per combination of starting complex (receptor state) and
protocol** — six tables, one for each of ``{crystal, relax, apo_relax} x {std, perturb}`` —
with the benchmark sets (PDBbind, PoseBusters, Astex) as rows. The six tables are produced
once per ranking score (by default for the interface delta *and* for the constraint-free
total score, to see which score ranks better). The markdown report starts with a legend that
defines every column (see :data:`COLUMN_LEGEND`).

Usage::

    uv run python -m docking_analysis.benchmark [--sets pdbbind posebuster astex]
        [--score-cols idelta_score total_score] [--rmsd-col rmsd_to_crystal] [--cutoff 2.0]
        [--include-incomplete]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import EXTRACT_DIR, OUTPUT_DIR

STATES = ("crystal", "relax", "apo_relax")
START_MODES = ("std", "perturb")
PROTOCOL_ORDER = [f"{s}_docking_{m}" for m in START_MODES for s in STATES]
SET_LABELS = {"pdbbind": "PDBbind", "posebuster": "PoseBusters", "astex": "Astex Diverse"}
DEFAULT_SCORE_COLS = ("idelta_score", "total_score")
DEFAULT_COLUMNS = ["pdb", "protocol", "total_score", "idelta_score", "rmsd_to_crystal", "rmsd_to_crystal_legacy", "rmsd_to_input"]
N_GRID = (10, 50, 150, 300)

STATE_LEGEND = {
    "crystal": "the input complex as prepared (no relaxation)",
    "relax": "lowest-scoring of 10 FastRelax runs of the complex with the ligand in place (backbone restrained)",
    "apo_relax": "lowest-scoring of 10 FastRelax runs with the ligand moved 500 Å away, ligand placed back afterwards "
                 "(a holo receptor relaxed without its ligand, not an apo structure; the placed-back ligand can be displaced)",
}
MODE_LEGEND = {
    "std": "Transform/HighResDocker/FinalMinimizer starting at the input ligand placement (no initial perturbation)",
    "perturb": "same protocol with an initial random perturbation of up to 5 Å (`initial_perturb=\"5.0\"`)",
}
SCORE_LEGEND = {
    "idelta_score": "Rosetta interface delta (energy of the complex minus energy of the separated partners; "
                    "lower is better, not affected by the constraint issue)",
    "total_score": "Rosetta total score with the coordinate-constraint energy removed "
                   "(auto-corrected; the stored value is `total_score_legacy`); lower is better",
}
COLUMN_LEGEND = [
    ("complexes", "number of complexes of the set that contribute (complete runs: all 300 poses of this combination present)"),
    ("poses_hit", "pose-level hit rate: percent of all poses of the combination (300 x complexes, pooled) that are hits"),
    ("any_hit", "percent of complexes with at least one hit among their 300 poses — the sampling success, independent of any score"),
    ("median_hit_frac", "median over complexes of the per-complex fraction of hits (percent) — how dense hits are in a typical run"),
    ("top1", "percent of complexes whose best-ranked pose (by the ranking score) is a hit"),
    ("top5", "percent of complexes with a hit among the 5 best-ranked poses"),
    ("top10", "percent of complexes with a hit among the 10 best-ranked poses"),
    ("N=k", "analytic sampling success: percent of complexes expected to have at least one hit among k samples, "
            "1 - (1 - p)^k averaged over complexes, p = per-complex hit fraction (treats samples as independent)"),
]


# ---------------------------------------------------------------------------
def load_docked(set_name: str, extraction_dir: Path | None = None, complete_only: bool = True,
                columns: list[str] | None = None):
    """Load ``docked.parquet`` (+ ``complexes.parquet``) of one extraction."""
    d = Path(extraction_dir) if extraction_dir else EXTRACT_DIR / f"{set_name}_full"
    docked = pd.read_parquet(d / "docked.parquet", columns=columns or DEFAULT_COLUMNS)
    complexes = pd.read_parquet(d / "complexes.parquet")
    if complete_only:
        ok = set(complexes.loc[complexes["complete"].eq(True), "pdb"])
        docked = docked[docked["pdb"].isin(ok)]
    return docked, complexes


def per_complex_protocol(docked: pd.DataFrame, rmsd_col: str = "rmsd_to_crystal",
                         score_col: str = "idelta_score", cutoff: float = 2.0,
                         ks: tuple[int, ...] = (1, 5, 10)) -> pd.DataFrame:
    """One row per (pdb, protocol): hit counts, best RMSD, Top-k RMSDs by *score_col*, rank of the first hit."""
    cols = ["pdb", "protocol", rmsd_col, score_col]
    df = docked[cols].dropna(subset=[rmsd_col]).sort_values(["pdb", "protocol", score_col], kind="mergesort")
    keys = ["pdb", "protocol"]
    df["_rank"] = df.groupby(keys, sort=False).cumcount()
    df["_hit"] = df[rmsd_col] <= cutoff
    g = df.groupby(keys, sort=False)
    out = g.agg(n=(rmsd_col, "size"), n_hit=("_hit", "sum"), min_rmsd=(rmsd_col, "min"),
                median_rmsd=(rmsd_col, "median"), min_score=(score_col, "min"))
    for k in ks:
        out[f"top{k}_rmsd"] = df[df["_rank"] < k].groupby(keys, sort=False)[rmsd_col].min()
    out["first_hit_rank"] = df[df["_hit"]].groupby(keys, sort=False)["_rank"].min()  # 0-based, NaN = no hit
    out["hit_frac"] = out["n_hit"] / out["n"]
    return out.reset_index()


def success_row(g: pd.DataFrame, cutoff: float, n_grid: tuple[int, ...] = N_GRID) -> dict:
    """Success rates of one (set, protocol) group of per-complex rows."""
    q = g["hit_frac"].to_numpy()
    row = {
        "complexes": len(g),
        "poses_hit": 100 * g["n_hit"].sum() / g["n"].sum(),
        "any_hit": 100 * (g["n_hit"] > 0).mean(),
        "median_hit_frac": 100 * g["hit_frac"].median(),
        "top1": 100 * (g["top1_rmsd"] <= cutoff).mean(),
        "top5": 100 * (g["top5_rmsd"] <= cutoff).mean(),
        "top10": 100 * (g["top10_rmsd"] <= cutoff).mean(),
    }
    row.update({f"N={N}": 100 * np.mean(1 - (1 - q) ** N) for N in n_grid})
    return row


def combination_tables(pc: pd.DataFrame, cutoff: float = 2.0) -> dict[str, pd.DataFrame]:
    """Six tables (one per receptor state x protocol); rows = benchmark sets."""
    tables = {}
    for protocol in PROTOCOL_ORDER:
        sub = pc[pc["protocol"] == protocol]
        rows = {}
        for set_name in [s for s in SET_LABELS if s in set(sub["set"])]:
            rows[SET_LABELS[set_name]] = success_row(sub[sub["set"] == set_name], cutoff)
        if rows:
            tables[protocol] = pd.DataFrame(rows).T
    return tables


def legend_markdown(rmsd_col: str, cutoff: float, score_cols: tuple[str, ...]) -> str:
    out = ["## Legend", ""]
    out.append(f"**Hit.** A pose is a hit when `{rmsd_col}` ≤ {cutoff:g} Å. `rmsd_to_crystal` is the symmetry-corrected, "
               "in-place (no superposition) heavy-atom RMSD of the docked ligand to the crystal ligand, as auto-corrected by "
               "`docking.result.Result`; the stored Rosetta value without symmetry correction is `rmsd_to_crystal_legacy`.")
    out.append("")
    out.append("**Ranking score** (one family of tables per score; lower is better):")
    for sc in score_cols:
        out.append(f"- `{sc}` — {SCORE_LEGEND.get(sc, 'score column of the docked table')}")
    out.append("")
    out.append("**Starting complex** (receptor state the docking started from):")
    for st in STATES:
        out.append(f"- `{st}` — {STATE_LEGEND[st]}")
    out.append("")
    out.append("**Protocol:**")
    for m in START_MODES:
        out.append(f"- `{m}` — {MODE_LEGEND[m]}")
    out.append("")
    out.append("**Columns** (percent of complexes unless stated; 300 poses per complex and combination):")
    for name, desc in COLUMN_LEGEND:
        out.append(f"- `{name}` — {desc}")
    out.append("")
    return "\n".join(out)


def tables_to_markdown(tables: dict[str, pd.DataFrame], cutoff: float, rmsd_col: str, score_col: str) -> str:
    out = [f"## Ranking by `{score_col}`", ""]
    for protocol, tab in tables.items():
        state, mode = "_".join(protocol.split("_")[:-2]), protocol.split("_")[-1]
        out.append(f"**Starting complex `{state}`, protocol `{mode}`** — hit: `{rmsd_col}` ≤ {cutoff:g} Å, ranking by `{score_col}`\n")
        cols = list(tab.columns)
        out.append("| set | " + " | ".join(cols) + " |")
        out.append("|---|" + "---|" * len(cols))
        for name, r in tab.iterrows():
            out.append(f"| {name} | " + " | ".join(f"{int(r[c])}" if c == "complexes" else f"{r[c]:.1f}" for c in cols) + " |")
        out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="+", default=list(SET_LABELS), choices=list(SET_LABELS))
    ap.add_argument("--extract-dir", type=Path, default=None, help="directory holding <set>_full/ (default EXTRACT_DIR)")
    ap.add_argument("--rmsd-col", default="rmsd_to_crystal")
    ap.add_argument("--score-cols", nargs="+", default=list(DEFAULT_SCORE_COLS))
    ap.add_argument("--cutoff", type=float, default=2.0)
    ap.add_argument("--include-incomplete", action="store_true")
    ap.add_argument("--out", type=Path, default=OUTPUT_DIR / "benchmark")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    score_cols = tuple(args.score_cols)
    base_tag = f"{args.rmsd_col}_{args.cutoff:g}A"

    docked_by_set = {}
    for set_name in args.sets:
        d = (args.extract_dir / f"{set_name}_full") if args.extract_dir else None
        docked, _ = load_docked(set_name, d, complete_only=not args.include_incomplete)
        docked_by_set[set_name] = docked
        print(f"[{set_name}] {len(docked):,} poses, {docked['pdb'].nunique():,} complexes", flush=True)

    report = [f"# Success rates per starting complex × protocol (hit: `{args.rmsd_col}` ≤ {args.cutoff:g} Å)", "",
              legend_markdown(args.rmsd_col, args.cutoff, score_cols)]
    for score_col in score_cols:
        tag = f"{args.rmsd_col}_{score_col}_{args.cutoff:g}A"
        frames = []
        for set_name, docked in docked_by_set.items():
            pc = per_complex_protocol(docked, args.rmsd_col, score_col, args.cutoff)
            pc.insert(0, "set", set_name)
            pc.to_parquet(args.out / f"per_complex_protocol_{set_name}_{tag}.parquet", index=False)
            frames.append(pc)
        tables = combination_tables(pd.concat(frames, ignore_index=True), args.cutoff)
        for protocol, tab in tables.items():
            tab.round(2).to_csv(args.out / f"success_{protocol}_{tag}.csv")
        report.append(tables_to_markdown(tables, args.cutoff, args.rmsd_col, score_col))
        print(f"[{score_col}] six tables written", flush=True)

    md = "\n".join(report)
    (args.out / f"success_tables_{base_tag}.md").write_text(md)
    print("\n" + md)
    print(f"written: {args.out}/success_<combination>_{args.rmsd_col}_<score>_{args.cutoff:g}A.csv and success_tables_{base_tag}.md")


if __name__ == "__main__":
    main()
