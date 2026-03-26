"""
docking_analysis.data
---------------------
Data loading, filtering, and per-PDB aggregation utilities.

This module provides:
- Kd index parsing from the PDBbind INDEX file
- Docking data loading from the zipped CSV
- Buzzword-based row filtering
- Unified per-PDB aggregation (min, mean-of-N, filtered mean)
- Kd column joining helper
- raw_delta column discovery
"""

import math
import zipfile
from pathlib import Path

import pandas as pd

from .constants import INDEX_PATH, KD_RE, UNIT_TO_MOLAR, ZERO_VARIANCE_TERMS, ZIP_PATH


# ---------------------------------------------------------------------------
# Kd index loader
# ---------------------------------------------------------------------------
def load_kd_index(index_path: Path = INDEX_PATH) -> pd.DataFrame:
    """
    Parse INDEX_general_PL.2020R1.lst and return a DataFrame with columns:
      - pdb      : 4-char PDB ID (lower-case)
      - kd_M     : Kd in Molar (float)
      - log_kd   : log10(kd_M)

    Only rows with an exact Kd= measurement are included; Ki, IC50, and
    inequality values (Kd<, Kd<=, Kd~, …) are skipped.
    """
    records: list[dict] = []

    with open(index_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Fields are whitespace-separated; column 0 = PDB, column 3 = binding data
            parts = line.split()
            if len(parts) < 4:
                continue

            pdb_id = parts[0].lower()
            binding_field = parts[3]

            m = KD_RE.match(binding_field)
            if m is None:
                continue  # Ki, IC50, or inequality — skip

            value = float(m.group("value"))
            unit  = m.group("unit")
            kd_mol = value * UNIT_TO_MOLAR[unit]

            records.append(
                {
                    "pdb": pdb_id,
                    "kd_M": kd_mol,
                    "log_kd": math.log10(kd_mol),
                }
            )

    df_kd = pd.DataFrame(records)
    print(f"  Kd entries parsed : {len(df_kd):,}")
    return df_kd


# ---------------------------------------------------------------------------
# Docking data loader
# ---------------------------------------------------------------------------
def load_docking_dataframe(zip_path: Path = ZIP_PATH) -> pd.DataFrame:
    """Read the CSV from inside the zip and return a DataFrame."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_name = zf.namelist()[0]  # only one file inside
        with zf.open(csv_name) as fh:
            df = pd.read_csv(fh)
    return df


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------
def filter_relax_and_perturb(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only rows whose ``type`` contains BOTH 'relax' AND 'perturb'
    as individual underscore-separated buzzwords.

    Example matching types::

        relax_docking_perturb
        apo_relax_docking_perturb
    """
    buzzwords = df["type"].str.split("_")
    has_relax   = buzzwords.apply(lambda bw: "relax"   in bw)
    has_perturb = buzzwords.apply(lambda bw: "perturb" in bw)
    mask = has_relax & has_perturb
    return df.loc[mask].copy()


# ---------------------------------------------------------------------------
# Unified per-PDB aggregation
# ---------------------------------------------------------------------------
def aggregate_per_pdb(
    df: pd.DataFrame,
    score_col: str = "idelta_score",
    strategy: str = "min",
    n: int = 10,
    extra_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Aggregate *score_col* (and optionally *extra_cols*) per PDB entry.

    Parameters
    ----------
    df : DataFrame
        Must contain a ``"pdb"`` column plus *score_col* (and *extra_cols*).
    score_col : str
        Column to aggregate.
    strategy : str
        ``"min"``            — single row with the lowest *score_col*.
        ``"mean_n"``         — mean of the *n* lowest *score_col* rows.
        ``"filtered_mean"``  — mean over rows where both ``total_score <= 0``
                               and ``score_col <= 0``.
    n : int
        Used only when *strategy* is ``"mean_n"``.
    extra_cols : list[str] | None
        Additional columns to aggregate alongside *score_col*.  When
        *strategy* is ``"min"`` these are taken from the same row; for the
        mean strategies they are averaged.

    Returns
    -------
    DataFrame with ``"pdb"`` plus the aggregated columns.
    """
    cols = [score_col] + (extra_cols or [])

    if strategy == "min":
        result = (
            df.loc[df.groupby("pdb", sort=False)[score_col].idxmin()]
            [["pdb"] + cols]
            .copy()
        )

    elif strategy == "mean_n":
        def _mean_n(grp: pd.DataFrame) -> pd.Series:
            return grp.nsmallest(n, score_col)[cols].mean()

        result = (
            df.groupby("pdb", sort=False, group_keys=False)
            .apply(_mean_n, include_groups=False)
            .reset_index()
        )

    elif strategy == "filtered_mean":
        valid = df.loc[(df["total_score"] <= 0) & (df[score_col] <= 0)].copy()
        result = (
            valid.groupby("pdb", sort=False)[cols]
            .mean()
            .reset_index()
        )

    else:
        raise ValueError(f"Unknown aggregation strategy: {strategy!r}")

    return result


# ---------------------------------------------------------------------------
# Kd column join helper
# ---------------------------------------------------------------------------
def join_kd_columns(
    per_pdb_df: pd.DataFrame,
    source_df: pd.DataFrame,
    merge_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Merge Kd-related columns from *source_df* onto *per_pdb_df* via the
    ``"pdb"`` key.

    Parameters
    ----------
    per_pdb_df : DataFrame
        One row per PDB (e.g. output of :func:`aggregate_per_pdb`).
    source_df : DataFrame
        Full docking frame that contains ``"pdb"`` and the Kd columns.
    merge_cols : list[str] | None
        Columns to bring over.  Defaults to ``["log_kd"]``.

    Returns
    -------
    Merged DataFrame (inner join, NaN rows dropped on *merge_cols*).
    """
    if merge_cols is None:
        merge_cols = ["log_kd"]

    kd = source_df[["pdb"] + merge_cols].drop_duplicates("pdb")
    merged = per_pdb_df.merge(kd, on="pdb", how="inner")
    return merged.dropna(subset=merge_cols)


# ---------------------------------------------------------------------------
# raw_delta column discovery
# ---------------------------------------------------------------------------
def get_raw_delta_columns(df: pd.DataFrame) -> list[str]:
    """
    Return the list of informative ``raw_delta_*`` column names.

    Excludes:
    - Columns containing ``"???"`` (unknown terms)
    - Columns in :data:`~docking_analysis.constants.ZERO_VARIANCE_TERMS`
    """
    return [
        c for c in df.columns
        if c.startswith("raw_delta_")
        and "???" not in c
        and c not in ZERO_VARIANCE_TERMS
    ]
