"""Dataset adapter for attribution training.

Reads a manifest with `generator_id` column (built by
`scripts/build_attribution_manifest.py`) and yields {frames, generator_id, ...}.

Only the visual stream is used — generator fingerprints live in the pixels and
their high-frequency content, not in waveform features. Audio is kept around
in the row for downstream consumers but not loaded here.
"""
from __future__ import annotations
import math
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from data.datasets.base import VideoManifest, _IMG_MEAN, _IMG_STD


class AttributionDataset(Dataset):
    def __init__(
        self,
        manifest: VideoManifest,
        num_frames: int = 16,
        frame_size: int = 224,
        training: bool = False,
    ):
        if "generator_id" not in manifest.df.columns:
            raise ValueError(
                "manifest missing `generator_id` column — "
                "run scripts/build_attribution_manifest.py first"
            )
        self.df = manifest.df.reset_index(drop=True)
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.training = training

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        frames = self._load_frames(row.get("frames_dir"))
        return {
            "frames": torch.from_numpy(frames),
            "generator_id": torch.tensor(int(row["generator_id"]), dtype=torch.long),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "clip_id": str(row["clip_id"]),
        }

    def _load_frames(self, frames_dir) -> np.ndarray:
        if not frames_dir or not Path(str(frames_dir)).exists():
            return np.zeros((self.num_frames, 3, self.frame_size, self.frame_size), dtype=np.float32)
        paths = sorted(Path(frames_dir).glob("frame_*.jpg"))
        if not paths:
            return np.zeros((self.num_frames, 3, self.frame_size, self.frame_size), dtype=np.float32)
        idx = self._sample_indices(len(paths))
        out = np.empty((self.num_frames, 3, self.frame_size, self.frame_size), dtype=np.float32)
        for i, j in enumerate(idx):
            img = Image.open(paths[j]).convert("RGB").resize(
                (self.frame_size, self.frame_size), Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32) / 255.0
            arr = (arr - _IMG_MEAN) / _IMG_STD
            out[i] = arr.transpose(2, 0, 1)
        return out

    def _sample_indices(self, n_available: int) -> List[int]:
        T = self.num_frames
        if n_available >= T:
            if self.training:
                start = np.random.randint(0, n_available - T + 1)
                return list(range(start, start + T))
            step = n_available / T
            return [int(math.floor(i * step)) for i in range(T)]
        return list(range(n_available)) + [n_available - 1] * (T - n_available)
