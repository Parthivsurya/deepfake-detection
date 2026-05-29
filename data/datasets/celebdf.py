"""Celeb-DF (v2) manifest builder.

Expected layout:
    <root>/Celeb-real/*.mp4         (label 0)
    <root>/Celeb-synthesis/*.mp4    (label 1)
    <root>/YouTube-real/*.mp4       (label 0)
    <root>/List_of_testing_videos.txt  (optional eval list)

Identity = the celebrity id encoded in the filename ("id0_0000.mp4" -> "id0").
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import List
import pandas as pd

from .base import VideoManifest


_ID_RE = re.compile(r"(id\d+)")


def _identity(stem: str) -> str:
    m = _ID_RE.search(stem)
    return m.group(1) if m else stem


class CelebDFBuilder:
    name = "celebdf"

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def build(self) -> VideoManifest:
        rows: List[dict] = []
        for sub, label in [("Celeb-real", 0), ("YouTube-real", 0), ("Celeb-synthesis", 1)]:
            d = self.root / sub
            if not d.exists():
                continue
            for v in sorted(d.glob("*.mp4")):
                rows.append({
                    "clip_id": f"cdf_{sub.lower()}_{v.stem}",
                    "video_path": str(v),
                    "label": label,
                    "identity": f"cdf_{_identity(v.stem)}",
                    "dataset": self.name,
                    "split_hint": sub,
                })
        return VideoManifest(pd.DataFrame(rows))
