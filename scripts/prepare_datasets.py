"""Build a unified manifest (CSV) for one or more datasets and write splits.

Usage:
    python scripts/prepare_datasets.py \
        --dataset faceforensics --root /data/FF++ \
        --out manifests/

    python scripts/prepare_datasets.py \
        --dataset faceforensics celebdf \
        --root /data/FF++ /data/Celeb-DF-v2 \
        --out manifests/

The script writes raw manifests per dataset, a combined manifest, and
train/val/test splits grouped by identity to prevent leakage.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

# enable `python scripts/prepare_datasets.py` execution from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.datasets import BUILDERS, VideoManifest  # noqa: E402
from data.splits import assign_splits, write_split_manifests  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", nargs="+", required=True, choices=list(BUILDERS),
                   help="dataset name(s)")
    p.add_argument("--root", nargs="+", required=True,
                   help="dataset root directory; one per --dataset")
    p.add_argument("--out", required=True, help="output directory for manifests")
    p.add_argument("--ratios", nargs=3, type=float, default=(0.8, 0.1, 0.1),
                   metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--salt", default="ddet", help="hash salt for reproducible splits")
    args = p.parse_args()

    if len(args.dataset) != len(args.root):
        p.error("--dataset and --root must have equal lengths")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    combined: VideoManifest | None = None
    for name, root in zip(args.dataset, args.root):
        print(f"[build] {name} <- {root}")
        manifest = BUILDERS[name](root).build()
        manifest.save(out / f"{name}.csv")
        print(f"  rows={len(manifest)}")
        combined = manifest if combined is None else combined.concat(manifest)

    assert combined is not None
    df = assign_splits(combined.df, group_col="identity", ratios=tuple(args.ratios),
                       salt=args.salt)
    combined_path = out / "combined.csv"
    df.to_csv(combined_path, index=False)
    paths = write_split_manifests(df, args.out)
    print("[splits]")
    for k, v in paths.items():
        n = (df["split"] == k).sum()
        print(f"  {k}: {n} rows -> {v}")


if __name__ == "__main__":
    main()
