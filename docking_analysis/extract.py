"""
docking_analysis.extract
------------------------
Extract the **complete** docking results of one benchmark set from the per-complex HDF5
files into three parquet tables (``<EXTRACT_DIR>/<set>_full/``), going through
:class:`docking.result.Result` so that the auto-correction of the access code is applied:

``docked.parquet``     one row per docked pose (every complex, complete or not)
``seeds.parquet``      one row per reference row (crystal / relax / apo_relax seeds)
``complexes.parquet``  one row per HDF5 file: completeness, ligand symmetry, references
``manifest.json``      provenance and column semantics

With ``autocorrect`` (default) exactly three columns are corrected and the stored values are
kept next to them (see ``docking/result.py``):

* ``total_score``      Rosetta total **without** the coordinate-constraint energy
                       (``total_score_legacy`` = as stored, which includes it);
* ``rmsd_to_crystal`` / ``rmsd_to_input``  symmetry-corrected, in-place heavy-atom ligand
                       RMSDs (``*_legacy`` = Rosetta's atom-name RMSDMetric values).
                       ``rmsd_to_input`` refers to the ligand of the seed the docking started
                       from (lowest *stored* total score of that receptor state).

Every other column is exactly as stored in the HDF5 files. ``--no-autocorrect`` writes the
raw legacy tables instead.

Usage::

    uv run python -m docking_analysis.extract --set pdbbind [--workers 20] [--limit 50] [--no-autocorrect]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem

from docking.result import Result

from .constants import EXTRACT_DIR, H5_DIRS

STATES = ("crystal", "relax", "apo_relax")
EXPECTED_SEEDS = {"crystal": 1, "relax": 10, "apo_relax": 10}
START_MODES = ("perturb", "std")
PROTOCOLS = [f"{s}_docking_{m}" for s in STATES for m in START_MODES]
N_DOCK = 300
DROP_COLS = ("raw_???", "raw_delta_???")   # a bogus zero score type dumped by PyRosetta


# ---------------------------------------------------------------------------
# per-file worker
# ---------------------------------------------------------------------------
def extract_one(h5_path: str, set_name: str, autocorrect: bool = True):
    """Return ``(complex_row: dict, seeds: DataFrame | None, docked: DataFrame | None)``."""
    pdb = Path(h5_path).stem
    row: dict = {"pdb": pdb, "set": set_name, "error": None}
    try:
        res = Result(h5_path, pdb, load_structure=False, autocorrect=autocorrect)
    except Exception as e:  # unreadable / corrupted file
        row["error"] = f"{type(e).__name__}: {e}"
        return row, None, None

    seeds = res.relax_df.copy()
    docked = res.docking_df.copy()
    for df in (seeds, docked):
        df.drop(columns=[c for c in df.columns if c in DROP_COLS], inplace=True)

    # ---- seeds ------------------------------------------------------------------
    rn = seeds["type"].to_numpy()
    seeds = seeds.rename(columns={"type": "state"})
    seeds.insert(1, "seed_idx", pd.Series(rn).groupby(rn).cumcount().to_numpy().astype(np.int16))
    seeds.insert(2, "h5_row", np.arange(len(rn), dtype=np.int32))
    seeds.insert(0, "pdb", pdb)
    seeds["is_docking_reference"] = False
    ref_row: dict[str, int] = {}
    for st in STATES:
        if (rn == st).any():
            ref_row[st] = res.reference_index(st)
            seeds.loc[ref_row[st], "is_docking_reference"] = True
    n_seed = {s: int((rn == s).sum()) for s in STATES}

    # ---- docked poses -------------------------------------------------------------
    prot = docked["type"].to_numpy()
    docked = docked.rename(columns={"type": "protocol"})
    state = np.array(["_".join(p.split("_")[:-2]) for p in prot])
    mode = np.array([p.split("_")[-1] for p in prot])
    docked.insert(1, "receptor_state", state)
    docked.insert(2, "start_mode", mode)
    docked.insert(3, "seq_id", pd.Series(prot).groupby(prot).cumcount().to_numpy().astype(np.int16))
    docked.insert(4, "h5_row", np.arange(len(prot), dtype=np.int32))
    docked.insert(0, "pdb", pdb)
    n_prot = {p: int((prot == p).sum()) for p in PROTOCOLS}

    # ---- complex summary --------------------------------------------------------------
    sym = res.ligand_symmetry
    try:
        smiles = Chem.MolToSmiles(Chem.RemoveHs(res.mol, sanitize=False))
    except Exception:
        smiles = None
    complete = all(n_seed[s] >= EXPECTED_SEEDS[s] for s in STATES) and all(n_prot[p] >= N_DOCK for p in PROTOCOLS)
    row.update({f"n_seed_{s}": n_seed[s] for s in STATES})
    row.update({f"n_{p}": n_prot[p] for p in PROTOCOLS})
    row.update({
        "complete": bool(complete),
        "autocorrected": bool(autocorrect),
        "ligand_n_heavy": None if sym is None else sym.n_heavy,
        "ligand_n_automorphisms": None if sym is None else sym.n_automorphisms,
        "ligand_automorphisms_capped": None if sym is None else bool(sym.capped),
        "ligand_smiles": smiles,
        "n_rmsd_uncorrected": int(res.n_rmsd_uncorrected),
        "ref_row_relax": ref_row.get("relax"),
        "ref_row_apo_relax": ref_row.get("apo_relax"),
        "relax_ref_ligand_rmsd": float(seeds.loc[ref_row["relax"], "rmsd_to_crystal"]) if "relax" in ref_row else None,
        "apo_relax_ref_ligand_rmsd": float(seeds.loc[ref_row["apo_relax"], "rmsd_to_crystal"]) if "apo_relax" in ref_row else None,
        "seed_prepare_time_total_s": float(seeds["prepare_time"].sum()),
        "dock_time_total_s": float(docked["prepare_time"].sum()) if len(docked) else 0.0,
    })
    return row, seeds, docked


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
class _ParquetSink:
    """Append pandas frames to one parquet file with a schema fixed by the first frame."""

    def __init__(self, path: Path, flush_rows: int = 400_000):
        self.path, self.flush_rows = path, flush_rows
        self.writer = None
        self.columns = None
        self.buffer: list[pd.DataFrame] = []
        self.n_buf = 0
        self.n_rows = 0

    def add(self, df: pd.DataFrame):
        if df is None or len(df) == 0:
            return
        if self.columns is None:
            self.columns = list(df.columns)
        self.buffer.append(df.reindex(columns=self.columns))
        self.n_buf += len(df)
        if self.n_buf >= self.flush_rows:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        table = pa.Table.from_pandas(pd.concat(self.buffer, ignore_index=True), preserve_index=False)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path, table.schema, compression="zstd")
        self.writer.write_table(table.cast(self.writer.schema))
        self.n_rows += table.num_rows
        self.buffer, self.n_buf = [], 0

    def close(self):
        self.flush()
        if self.writer is not None:
            self.writer.close()


def _git_hash() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def run(set_name: str, out_dir: Path, workers: int, limit: int | None, overwrite: bool,
        autocorrect: bool = True) -> None:
    h5_dir = H5_DIRS[set_name]
    files = sorted(str(p) for p in h5_dir.glob("*.h5"))
    if limit:
        files = files[:limit]
    if not files:
        sys.exit(f"no .h5 files in {h5_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    targets = {n: out_dir / f"{n}.parquet" for n in ("docked", "seeds", "complexes")}
    existing = [p for p in targets.values() if p.exists()]
    if existing and not overwrite:
        sys.exit(f"outputs exist in {out_dir} (use --overwrite): {[p.name for p in existing]}")

    print(f"[{set_name}] {len(files):,} HDF5 files from {h5_dir} -> {out_dir} with {workers} workers "
          f"(autocorrect={'on' if autocorrect else 'off'})", flush=True)
    docked_sink, seeds_sink = _ParquetSink(targets["docked"]), _ParquetSink(targets["seeds"], flush_rows=50_000)
    complex_rows: list[dict] = []
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(extract_one, fp, set_name, autocorrect): fp for fp in files}
        for fut in as_completed(futures):
            row, seeds, docked = fut.result()
            complex_rows.append(row)
            seeds_sink.add(seeds)
            docked_sink.add(docked)
            done += 1
            if done % 200 == 0 or done == len(files):
                el = time.time() - t0
                print(f"  {done:,}/{len(files):,} files  {el/60:.1f} min  ({done/el:.1f} files/s)", flush=True)
    docked_sink.close()
    seeds_sink.close()

    complexes = pd.DataFrame(complex_rows).sort_values("pdb").reset_index(drop=True)
    complexes.to_parquet(targets["complexes"], index=False)

    manifest = {
        "set": set_name,
        "source_dir": str(h5_dir),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_hash": _git_hash(),
        "autocorrect": autocorrect,
        "n_files": len(files),
        "n_complete": int(complexes["complete"].eq(True).sum()),
        "n_errors": int(complexes["error"].notna().sum()),
        "n_docked_rows": docked_sink.n_rows,
        "n_seed_rows": seeds_sink.n_rows,
        "semantics": {
            "legacy": "all HDF5 files: stored total_score includes the coordinate-constraint energy; stored RMSDs are "
                      "Rosetta RMSDMetric values without symmetry correction (see docking/result.py)",
            "total_score": ("stored total minus scfx weight x raw_coordinate_constraint" if autocorrect else "as stored"),
            "total_score_legacy": "as stored",
            "rmsd_to_crystal": ("symmetry-corrected in-place heavy-atom ligand RMSD to the crystal ligand" if autocorrect else "as stored"),
            "rmsd_to_input": ("same, to the ligand of the seed the docking started from (lowest stored total score of that "
                              "receptor state); seed rows: RMSD to the crystal ligand" if autocorrect else "as stored"),
            "rmsd_*_legacy": "as stored (atom-name matching, no symmetry correction)",
            "all other columns": "as stored",
        },
        "elapsed_s": time.time() - t0,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[{set_name}] done: {manifest['n_docked_rows']:,} docked rows, {manifest['n_seed_rows']:,} seed rows, "
          f"{manifest['n_complete']:,} complete complexes, {manifest['n_errors']} unreadable files, "
          f"{manifest['elapsed_s']/60:.1f} min", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True, choices=sorted(H5_DIRS))
    ap.add_argument("--out", type=Path, default=None, help="output directory (default <EXTRACT_DIR>/<set>_full)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--limit", type=int, default=None, help="only the first N files (testing)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-autocorrect", action="store_true", help="write the raw legacy tables")
    args = ap.parse_args(argv)
    out = args.out or (EXTRACT_DIR / f"{args.set}_full")
    run(args.set, out, args.workers, args.limit, args.overwrite, autocorrect=not args.no_autocorrect)


if __name__ == "__main__":
    main()
