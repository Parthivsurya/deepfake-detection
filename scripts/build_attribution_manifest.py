"""Build a generator-labeled manifest for attribution training.

Takes an existing extracted manifest (one with `frames_dir` already populated)
and assigns a `generator_id` per row according to attribution/generators.py:

    real clips                       -> 0
    FF++ rows with `manipulation`    -> FF_METHOD_TO_ID[manipulation]
    Celeb-DF synthesis               -> 6
    DFDC fakes                       -> 7
    FakeAVCeleb fakes                -> 8

If a row's dataset/manipulation pair can't be mapped, it is dropped (with a
warning) — attribution requires a clean label per sample.

Usage:
    python scripts/build_attribution_manifest.py \
        --in  manifests/train.extracted.csv \
        --out manifests/attribution_train.csv
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attribution.generators import FF_METHOD_TO_ID, GENERATOR_REGISTRY  # noqa: E402


def assign_generator_id(row) -> int | None:
    label = int(row["label"])
    if label == 0:
        return 0  # real
    ds = str(row.get("dataset", "")).lower()
    if ds == "faceforensics":
        m = row.get("manipulation")
        if isinstance(m, str) and m in FF_METHOD_TO_ID:
            return FF_METHOD_TO_ID[m]
        # FF++ row marked fake but no manipulation column — try split_hint
        sh = str(row.get("split_hint", ""))
        for method, gid in FF_METHOD_TO_ID.items():
            if method.lower() in sh.lower():
                return gid
        return None
    if ds == "celebdf":
        return 6
    if ds == "dfdc":
        return 7
    if ds == "fakeavceleb":
        return 8
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True, help="input extracted manifest CSV")
    p.add_argument("--out", required=True, help="output attribution manifest CSV")
    p.add_argument("--keep-real", action="store_true",
                   help="keep real (label=0) rows (generator_id=0). Default: drop them.")
    args = p.parse_args()

    df = pd.read_csv(args.inp)
    df["generator_id"] = df.apply(assign_generator_id, axis=1)
    n_before = len(df)
    df_unmapped = df[df["generator_id"].isna()]
    if len(df_unmapped):
        print(f"WARN: dropping {len(df_unmapped)} rows with no generator mapping")
        df = df[~df["generator_id"].isna()].copy()
    df["generator_id"] = df["generator_id"].astype(int)

    if not args.keep_real:
        df = df[df["generator_id"] != 0].copy()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"\nWrote {len(df)}/{n_before} rows -> {args.out}")
    print("\nGenerator class distribution:")
    for gid, cnt in df["generator_id"].value_counts().sort_index().items():
        info = GENERATOR_REGISTRY.get(int(gid))
        name = info.name if info else f"id{gid}"
        family = info.family if info else "?"
        print(f"  {gid:>2}  {name:<18}  family={family:<16}  n={cnt}")


if __name__ == "__main__":
    main()
