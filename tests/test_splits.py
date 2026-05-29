"""Verify splits are deterministic and identity-disjoint."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
from data.splits import assign_splits  # noqa: E402


def main() -> int:
    df = pd.DataFrame({
        "clip_id": [f"c{i}" for i in range(200)],
        "identity": [f"id{i // 4}" for i in range(200)],     # 4 clips per identity
        "video_path": [""] * 200,
        "label": [i % 2 for i in range(200)],
        "dataset": ["ff"] * 200,
    })
    a = assign_splits(df)
    b = assign_splits(df)
    assert a["split"].equals(b["split"]), "splits are not deterministic"

    overlap = set()
    for s in ("train", "val", "test"):
        ids = set(a.loc[a["split"] == s, "identity"])
        if overlap & ids:
            raise AssertionError(f"identity leak across splits: {overlap & ids}")
        overlap |= ids

    counts = a["split"].value_counts().to_dict()
    print(f"OK  counts={counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
