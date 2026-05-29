"""Real-time stream simulator.

Replays a video file (or webcam) as if frames arrive live, optionally with
network jitter and frame drops. Used by Task 5 (real-time benchmarking) and by
the integration pipeline at inference time.
"""
from __future__ import annotations
import time
from collections import deque
from pathlib import Path
from typing import Iterator, Optional, Tuple
import numpy as np

from utils.video_utils import iter_frames


class StreamSimulator:
    """Yields (timestamp_s, frame_rgb) pairs in real time."""

    def __init__(
        self,
        source: str | Path | int,
        fps: float = 25.0,
        drop_prob: float = 0.0,
        jitter_ms: float = 0.0,
        seed: int = 0,
    ):
        self.source = source
        self.fps = fps
        self.dt = 1.0 / fps
        self.drop_prob = drop_prob
        self.jitter_ms = jitter_ms
        self.rng = np.random.default_rng(seed)

    def __iter__(self) -> Iterator[Tuple[float, np.ndarray]]:
        start = time.time()
        for i, frame in enumerate(iter_frames(self.source, sample_fps=self.fps)):
            if self.drop_prob and self.rng.random() < self.drop_prob:
                continue
            target = start + i * self.dt
            if self.jitter_ms:
                target += self.rng.normal(0, self.jitter_ms / 1000)
            sleep_for = target - time.time()
            if sleep_for > 0:
                time.sleep(sleep_for)
            yield time.time() - start, frame


class SlidingWindowBuffer:
    """Fixed-length frame buffer for windowed inference on a stream."""

    def __init__(self, window: int, stride: int = 1):
        self.window = window
        self.stride = stride
        self._buf: deque = deque(maxlen=window)
        self._since_emit = 0

    def push(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Append frame; return stacked window (T, H, W, C) when ready, else None."""
        self._buf.append(frame)
        self._since_emit += 1
        if len(self._buf) < self.window:
            return None
        if self._since_emit < self.stride:
            return None
        self._since_emit = 0
        return np.stack(list(self._buf), axis=0)
