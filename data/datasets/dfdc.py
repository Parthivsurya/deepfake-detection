"""DFDC (full + preview) manifest builder.

Each part folder contains a `metadata.json` mapping `<file>.mp4 -> {label, original}`,
e.g. {"label": "FAKE", "original": "abcde.mp4"} or {"label": "REAL"}.

Identity = the original real video the fake was derived from (so all fakes of the
same source are grouped with that source).
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import List
import pandas as pd

from .base import VideoManifest


class DFDCBuilder:
    name = "dfdc"

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def build(self) -> VideoManifest:
        rows: List[dict] = []
        for meta_path in self.root.rglob("metadata.json"):
            part_dir = meta_path.parent
            with open(meta_path) as f:
                meta = json.load(f)
            for fname, info in meta.items():
                video = part_dir / fname
                if not video.exists():
                    continue
                label = 1 if str(info.get("label", "")).upper() == "FAKE" else 0
                identity = Path(info.get("original") or fname).stem
                rows.append({
                    "clip_id": f"dfdc_{video.stem}",
                    "video_path": str(video),
                    "label": label,
                    "identity": f"dfdc_{identity}",
                    "dataset": self.name,
                    "part": part_dir.name,
                })
        return VideoManifest(pd.DataFrame(rows))
