"""Deterministic, identity-aware train/val/test splits.

Splits group by `identity` (or `clip_id` when identity is unknown) so the same
person never appears in both train and val/test — a common deepfake-leakage trap.
"""
from __future__ import annotations
import hashlib
from typing import Iterable, Tuple
import pandas as pd


def _bucket(key: str, salt: str) -> float:
    h = hashlib.md5(f"{salt}:{key}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def assign_splits(
    df: pd.DataFrame,
    group_col: str = "identity",
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    salt: str = "ddet",
) -> pd.DataFrame:
    """Add a `split` column with values in {train, val, test}.

    Groups by `group_col` so all rows for one identity end up in the same split.
    """
    train_r, val_r, _ = ratios
    df = df.copy()
    buckets = df[group_col].astype(str).map(lambda k: _bucket(k, salt))
    df["split"] = "test"
    df.loc[buckets < train_r, "split"] = "train"
    df.loc[(buckets >= train_r) & (buckets < train_r + val_r), "split"] = "val"
    return df


def write_split_manifests(df: pd.DataFrame, out_dir: str) -> dict:
    """Write train/val/test CSVs and return paths."""
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for s in ("train", "val", "test"):
        p = out / f"{s}.csv"
        df[df["split"] == s].drop(columns=["split"]).to_csv(p, index=False)
        paths[s] = str(p)
    return paths
