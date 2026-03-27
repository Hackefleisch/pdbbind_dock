"""
docking_analysis.clustering
----------------------------
RMSD-based greedy clustering of docked poses per PDB.

The algorithm sorts poses by interface-delta score (ascending) and
greedily assigns each pose to the nearest existing cluster if the
unaligned heavy-atom RMSD to its representative is below a cutoff;
otherwise a new cluster is formed.

Cluster results are cached as pickle files (one per PDB) so that
expensive H5 I/O + RDKit molecule reconstruction only happens once.
"""

from __future__ import annotations

import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════════
# Cluster data structure
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Cluster:
    """One structural cluster of docked poses for a single PDB.

    Attributes
    ----------
    representative_idx : int
        Index of the cluster representative within the PDB's pose list
        (i.e. ``pose_indices[representative_idx]`` gives the H5 row).
    representative_h5_idx : int
        Absolute H5 row index of the representative.
    member_indices : list[int]
        Indices into ``pose_indices`` for all cluster members (including
        the representative).
    member_h5_indices : list[int]
        Absolute H5 row indices for all cluster members.
    member_rmsds : list[float]
        Unaligned heavy-atom RMSD of each member to the representative.
        The representative itself has RMSD 0.0.
    size : int
        Number of poses in the cluster.
    """

    representative_idx: int
    representative_h5_idx: int
    member_indices: list[int] = field(default_factory=list)
    member_h5_indices: list[int] = field(default_factory=list)
    member_rmsds: list[float] = field(default_factory=list)
    size: int = 0


# ═══════════════════════════════════════════════════════════════════════
# RMSD computation
# ═══════════════════════════════════════════════════════════════════════

def compute_heavy_atom_rmsd(mol_a, mol_b) -> float:
    """Unaligned RMSD over heavy atoms (no superposition).

    Both molecules must have the same atom ordering and at least one
    heavy atom.  Hydrogen atoms are excluded from the calculation.
    """
    conf_a = mol_a.GetConformer(0)
    conf_b = mol_b.GetConformer(0)

    heavy = [a.GetIdx() for a in mol_a.GetAtoms() if a.GetAtomicNum() > 1]
    if not heavy:
        return 0.0

    coords_a = np.array([[conf_a.GetAtomPosition(i).x,
                           conf_a.GetAtomPosition(i).y,
                           conf_a.GetAtomPosition(i).z] for i in heavy])
    coords_b = np.array([[conf_b.GetAtomPosition(i).x,
                           conf_b.GetAtomPosition(i).y,
                           conf_b.GetAtomPosition(i).z] for i in heavy])

    return float(np.sqrt(np.mean(np.sum((coords_a - coords_b) ** 2, axis=1))))


# ═══════════════════════════════════════════════════════════════════════
# Per-PDB clustering
# ═══════════════════════════════════════════════════════════════════════

def cluster_poses_for_pdb(
    pdb_id: str,
    h5_dir: Path,
    pose_h5_indices: list[int],
    pose_scores: np.ndarray,
    rmsd_cutoff: float = 2.0,
) -> list[Cluster]:
    """Cluster docked poses for a single PDB by structural similarity.

    Parameters
    ----------
    pdb_id : str
        PDB identifier (e.g. ``"4eo6"``).
    h5_dir : Path
        Directory containing ``<pdb_id>.h5`` files.
    pose_h5_indices : list[int]
        Absolute H5 row indices for the poses to cluster.
    pose_scores : ndarray
        Interface-delta scores corresponding to *pose_h5_indices*,
        used to determine processing order (lowest first).
    rmsd_cutoff : float
        Maximum unaligned heavy-atom RMSD (Å) for a pose to join an
        existing cluster.

    Returns
    -------
    list[Cluster]
        One :class:`Cluster` per structural cluster found.
    """
    # Lazy import to avoid loading RDKit / h5py at module level
    sys.path.insert(0, str(h5_dir.parent))
    from data import Result

    h5_path = h5_dir / f"{pdb_id}.h5"
    if not h5_path.exists():
        return []

    result = Result(str(h5_path), pdb_id, load_structure=True)

    # Sort poses by score (ascending = best first)
    order = np.argsort(pose_scores)
    sorted_h5_idxs = [pose_h5_indices[i] for i in order]
    sorted_local_idxs = list(order)

    # Pre-compute molecules for all poses
    mols: dict[int, object] = {}
    for h5_idx in sorted_h5_idxs:
        try:
            mols[h5_idx] = result.docked_mol(h5_idx)
        except Exception:
            pass  # skip poses that can't be reconstructed

    # Greedy clustering
    clusters: list[Cluster] = []

    for rank, (local_idx, h5_idx) in enumerate(
        zip(sorted_local_idxs, sorted_h5_idxs)
    ):
        if h5_idx not in mols:
            continue

        mol = mols[h5_idx]
        best_cluster_idx = -1
        best_rmsd = float("inf")

        for ci, cluster in enumerate(clusters):
            rep_mol = mols.get(cluster.representative_h5_idx)
            if rep_mol is None:
                continue
            rmsd = compute_heavy_atom_rmsd(rep_mol, mol)
            if rmsd < best_rmsd:
                best_rmsd = rmsd
                best_cluster_idx = ci

        if best_cluster_idx >= 0 and best_rmsd < rmsd_cutoff:
            # Add to existing cluster
            c = clusters[best_cluster_idx]
            c.member_indices.append(local_idx)
            c.member_h5_indices.append(h5_idx)
            c.member_rmsds.append(best_rmsd)
            c.size += 1
        else:
            # Start a new cluster with this pose as representative
            clusters.append(Cluster(
                representative_idx=local_idx,
                representative_h5_idx=h5_idx,
                member_indices=[local_idx],
                member_h5_indices=[h5_idx],
                member_rmsds=[0.0],
                size=1,
            ))

    return clusters


# ═══════════════════════════════════════════════════════════════════════
# Batch clustering with caching
# ═══════════════════════════════════════════════════════════════════════

def _cache_path(cache_dir: Path, pdb_id: str, rmsd_cutoff: float) -> Path:
    """Return the pickle cache path for a single PDB."""
    subdir = cache_dir / f"rmsd_{rmsd_cutoff:.1f}"
    return subdir / f"{pdb_id}.pkl"


def _build_h5_index_mapping(
    pose_types: np.ndarray,
    protocols: np.ndarray,
    type_seq_ids: np.ndarray,
) -> tuple[list[int], list[bool]]:
    """Map CSV rows to H5 indices via type + explicit sequence ID."""
    type_to_h5: dict[str, list[int]] = {}
    for idx, proto in enumerate(protocols):
        type_to_h5.setdefault(proto, []).append(idx)

    pose_h5_indices: list[int] = []
    valid_mask: list[bool] = []

    for t, seq_id in zip(pose_types, type_seq_ids):
        h5_list = type_to_h5.get(t, [])
        if seq_id < len(h5_list):
            pose_h5_indices.append(h5_list[seq_id])
            valid_mask.append(True)
        else:
            pose_h5_indices.append(-1)
            valid_mask.append(False)

    return pose_h5_indices, valid_mask


def _cluster_single_pdb(args: tuple) -> tuple[str, list[Cluster] | None]:
    """Worker function for parallel clustering (must be top-level for pickling).

    Parameters
    ----------
    args : tuple
        ``(pdb_id, h5_dir, pose_types, pose_seq_ids, pose_scores, rmsd_cutoff,
          cache_dir)``

    Returns
    -------
    ``(pdb_id, clusters)`` or ``(pdb_id, None)`` on failure.
    """
    import h5py

    pdb_id, h5_dir, pose_types, pose_seq_ids, pose_scores, rmsd_cutoff, cache_dir = args
    h5_dir = Path(h5_dir)

    h5_path = h5_dir / f"{pdb_id}.h5"
    if not h5_path.exists():
        return pdb_id, None

    # Read protocols from H5
    try:
        with h5py.File(str(h5_path), "r") as f:
            protocols = f["protocol"].asstr()[()]
    except Exception:
        return pdb_id, None

    # Map CSV rows → H5 indices
    pose_h5_indices, valid_mask = _build_h5_index_mapping(pose_types, protocols, pose_seq_ids)

    valid_h5 = [h for h, v in zip(pose_h5_indices, valid_mask) if v]
    valid_scores = np.array([s for s, v in zip(pose_scores, valid_mask) if v])

    if len(valid_h5) == 0:
        return pdb_id, None

    clusters = cluster_poses_for_pdb(
        pdb_id, h5_dir, valid_h5, valid_scores, rmsd_cutoff,
    )

    # Save to cache
    if cache_dir is not None:
        cp = _cache_path(Path(cache_dir), pdb_id, rmsd_cutoff)
        with open(cp, "wb") as f:
            pickle.dump(clusters, f)

    return pdb_id, clusters


def cluster_all_pdbs(
    df: pd.DataFrame,
    h5_dir: Path,
    rmsd_cutoff: float = 2.0,
    cache_dir: Path | None = None,
    score_col: str = "idelta_score",
    n_workers: int | None = None,
) -> dict[str, list[Cluster]]:
    """Cluster all PDBs present in *df*.

    Parameters
    ----------
    df : DataFrame
        Filtered docking data with columns ``pdb``, ``type``, and
        *score_col*.  Both ``relax_docking_perturb`` and
        ``apo_relax_docking_perturb`` rows are included.
    h5_dir : Path
        Directory containing the ``*.h5`` files.
    rmsd_cutoff : float
        RMSD threshold in Å for cluster membership.
    cache_dir : Path | None
        If given, cluster results are cached as pickle files under
        ``<cache_dir>/rmsd_<cutoff>/<pdb>.pkl``.
    score_col : str
        Score column used for ordering poses within each PDB.
    n_workers : int | None
        Number of parallel worker processes.  ``None`` (default) uses
        all available CPUs.  Set to ``1`` to disable parallelism.

    Returns
    -------
    dict mapping PDB ID → list of :class:`Cluster`.
    """
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    pdb_ids = df["pdb"].unique()
    cluster_map: dict[str, list[Cluster]] = {}

    # Prepare cache directory
    if cache_dir is not None:
        subdir = cache_dir / f"rmsd_{rmsd_cutoff:.1f}"
        subdir.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: load cached results ────────────────────────────────
    to_compute: list[str] = []
    for pdb_id in pdb_ids:
        if cache_dir is not None:
            cp = _cache_path(cache_dir, pdb_id, rmsd_cutoff)
            if cp.exists():
                with open(cp, "rb") as f:
                    cluster_map[pdb_id] = pickle.load(f)
                continue
        to_compute.append(pdb_id)

    if cluster_map:
        print(f"  Loaded {len(cluster_map):,} PDBs from cache")

    if not to_compute:
        return cluster_map

    # ── Phase 2: parallel clustering of uncached PDBs ───────────────
    workers = n_workers if n_workers is not None else os.cpu_count()
    print(f"  Clustering {len(to_compute):,} PDBs with {workers} workers …")

    # Pre-extract per-PDB data to avoid pickling the whole DataFrame
    pdb_groups = df.groupby("pdb", sort=False)
    work_items = []
    for pdb_id in to_compute:
        grp = pdb_groups.get_group(pdb_id)
        work_items.append((
            pdb_id,
            str(h5_dir),
            grp["type"].values,
            grp["type_seq_id"].values,
            grp[score_col].values,
            rmsd_cutoff,
            str(cache_dir) if cache_dir is not None else None,
        ))

    skipped = 0

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_cluster_single_pdb, item): item[0]
            for item in work_items
        }
        with tqdm(total=len(futures), desc="Clustering PDBs", unit="pdb") as pbar:
            for future in as_completed(futures):
                pdb_id, clusters = future.result()
                if clusters is not None:
                    cluster_map[pdb_id] = clusters
                else:
                    skipped += 1
                pbar.update(1)

    if skipped:
        print(f"  Clustering: skipped {skipped} PDBs (missing H5 or mapping)")

    return cluster_map


# ═══════════════════════════════════════════════════════════════════════
# Statistics reporting
# ═══════════════════════════════════════════════════════════════════════

def print_cluster_statistics(cluster_map: dict[str, list[Cluster]]) -> None:
    """Print a summary of clustering results."""
    if not cluster_map:
        print("  No clusters to report.")
        return

    n_pdbs = len(cluster_map)
    all_sizes = []
    n_clusters_per_pdb = []

    for pdb_id, clusters in cluster_map.items():
        n_clusters_per_pdb.append(len(clusters))
        for c in clusters:
            all_sizes.append(c.size)

    total_clusters = sum(n_clusters_per_pdb)
    total_poses = sum(all_sizes)

    print(f"  Cluster Statistics")
    print(f"  {'─' * 40}")
    print(f"  PDBs clustered        : {n_pdbs:,}")
    print(f"  Total clusters        : {total_clusters:,}")
    print(f"  Total poses clustered : {total_poses:,}")
    print(f"  Avg clusters / PDB    : {np.mean(n_clusters_per_pdb):.1f}"
          f"  (median {np.median(n_clusters_per_pdb):.0f},"
          f" max {max(n_clusters_per_pdb)})")
    print(f"  Avg cluster size      : {np.mean(all_sizes):.1f}"
          f"  (median {np.median(all_sizes):.0f},"
          f" max {max(all_sizes)})")
    print(f"  Singletons            : {sum(1 for s in all_sizes if s == 1):,}"
          f"  ({100 * sum(1 for s in all_sizes if s == 1) / len(all_sizes):.1f}%)")
