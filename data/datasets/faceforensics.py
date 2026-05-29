"""FaceForensics++ manifest builder.

Expected layout (per official release):
    <root>/original_sequences/youtube/c23/videos/*.mp4
    <root>/manipulated_sequences/<method>/c23/videos/*.mp4
where method ∈ {Deepfakes, Face2Face, FaceSwap, NeuralTextures, FaceShifter}.

Clip identity = the source video number ("000" in "000_003.mp4") so the same
actor never spans splits.
"""
from __future__ import annotations
from pathlib import Path
from typing import List
import pandas as pd

from .base import VideoManifest


METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter"]


class FaceForensicsBuilder:
    name = "faceforensics"

    def __init__(self, root: str | Path, compression: str = "c23"):
        self.root = Path(root)
        self.compression = compression

    def build(self) -> VideoManifest:
        rows: List[dict] = []
        real_dir = self.root / "original_sequences" / "youtube" / self.compression / "videos"
        for v in sorted(real_dir.glob("*.mp4")):
            rows.append({
                "clip_id": f"ff_real_{v.stem}",
                "video_path": str(v),
                "label": 0,
                "identity": f"ff_{v.stem}",
                "dataset": self.name,
            })
        for method in METHODS:
            d = self.root / "manipulated_sequences" / method / self.compression / "videos"
            if not d.exists():
                continue
            for v in sorted(d.glob("*.mp4")):
                src = v.stem.split("_")[0]  # "000_003" -> "000"
                rows.append({
                    "clip_id": f"ff_{method.lower()}_{v.stem}",
                    "video_path": str(v),
                    "label": 1,
                    "identity": f"ff_{src}",
                    "dataset": self.name,
                    "manipulation": method,
                })
        return VideoManifest(pd.DataFrame(rows))
