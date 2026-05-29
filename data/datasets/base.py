"""Manifest abstraction + torch Dataset for extracted clips.

A *manifest* is a CSV with at least these columns:
    clip_id, video_path, label, identity, dataset

After frame extraction it also carries:
    frames_dir, audio_path
"""
from __future__ import annotations
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


REQUIRED_COLS = ["clip_id", "video_path", "label", "identity", "dataset"]


@dataclass
class VideoManifest:
    df: pd.DataFrame

    @classmethod
    def load(cls, path: str | Path) -> "VideoManifest":
        df = pd.read_csv(path)
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"manifest {path} is missing columns: {missing}")
        return cls(df)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(path, index=False)

    def __len__(self) -> int:
        return len(self.df)

    def concat(self, other: "VideoManifest") -> "VideoManifest":
        return VideoManifest(pd.concat([self.df, other.df], ignore_index=True))


_IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class VideoClipDataset(Dataset):
    """Loads T frames + an audio clip per video.

    The manifest is expected to have `frames_dir` (output of frame extraction).
    If `audio_path` is missing or the file doesn't exist, returns a zero waveform
    and `has_audio=0` — the model handles missing audio gracefully.
    """

    def __init__(
        self,
        manifest: VideoManifest,
        num_frames: int = 16,
        frame_size: int = 224,
        audio_sample_rate: int = 16000,
        audio_seconds: float = 4.0,
        training: bool = False,
    ):
        self.df = manifest.df.reset_index(drop=True)
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.sr = audio_sample_rate
        self.audio_seconds = audio_seconds
        self.training = training

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        frames = self._load_frames(row.get("frames_dir"))
        waveform, has_audio = self._load_audio(row.get("audio_path"))
        return {
            "frames": torch.from_numpy(frames),          # (T, 3, H, W) float32
            "audio": torch.from_numpy(waveform),         # (samples,) float32
            "has_audio": torch.tensor(has_audio, dtype=torch.float32),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "clip_id": str(row["clip_id"]),
        }

    # ---------- frames ----------
    def _load_frames(self, frames_dir: Optional[str]) -> np.ndarray:
        if not frames_dir or not Path(frames_dir).exists():
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

    # ---------- audio ----------
    def _load_audio(self, audio_path: Optional[str]) -> Tuple[np.ndarray, int]:
        target = int(self.sr * self.audio_seconds)
        if not audio_path or not Path(str(audio_path)).exists():
            return np.zeros(target, dtype=np.float32), 0
        try:
            from data.preprocessing.audio_extraction import load_audio_clip
            return load_audio_clip(audio_path, self.sr, self.audio_seconds), 1
        except Exception:
            return np.zeros(target, dtype=np.float32), 0
