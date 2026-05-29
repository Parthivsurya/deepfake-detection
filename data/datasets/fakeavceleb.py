"""FakeAVCeleb manifest builder — audio-visual deepfake benchmark.

Layout (per the release CSV):
    <root>/FakeAVCeleb_v1.2/<category>/<id>/<video>.mp4
    <root>/meta_data.csv  with columns including: path, category, type,
                                                  source, race, gender

`type` ∈ {"real", "FakeVideo-RealAudio", "RealVideo-FakeAudio", "FakeVideo-FakeAudio"}.
We map any "Fake*" category to label=1, and additionally expose
`audio_fake` / `video_fake` flags for downstream multimodal losses.
"""
from __future__ import annotations
from pathlib import Path
from typing import List
import pandas as pd

from .base import VideoManifest


class FakeAVCelebBuilder:
    name = "fakeavceleb"

    def __init__(self, root: str | Path, meta_csv: str | Path | None = None):
        self.root = Path(root)
        self.meta_csv = Path(meta_csv) if meta_csv else self.root / "meta_data.csv"

    def build(self) -> VideoManifest:
        if not self.meta_csv.exists():
            raise FileNotFoundError(f"missing FakeAVCeleb metadata: {self.meta_csv}")
        meta = pd.read_csv(self.meta_csv)
        rows: List[dict] = []
        for _, r in meta.iterrows():
            video = self.root / str(r.get("path", "")).lstrip("/")
            if not video.exists():
                continue
            t = str(r.get("type", "real")).lower()
            video_fake = int("fakevideo" in t)
            audio_fake = int("fakeaudio" in t)
            label = int(video_fake or audio_fake)
            rows.append({
                "clip_id": f"avceleb_{video.stem}",
                "video_path": str(video),
                "label": label,
                "identity": f"avceleb_{r.get('source', video.stem)}",
                "dataset": self.name,
                "video_fake": video_fake,
                "audio_fake": audio_fake,
            })
        return VideoManifest(pd.DataFrame(rows))
