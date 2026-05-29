"""Lightweight video I/O helpers built on PyAV with OpenCV fallback."""
from __future__ import annotations
from pathlib import Path
from typing import Iterator, Optional
import numpy as np


def iter_frames(path: str | Path, sample_fps: Optional[float] = None) -> Iterator[np.ndarray]:
    """Yield RGB uint8 frames. If `sample_fps` is set, decimate to that rate."""
    path = str(path)
    try:
        import av
        container = av.open(path)
        stream = container.streams.video[0]
        src_fps = float(stream.average_rate) if stream.average_rate else 25.0
        step = max(int(round(src_fps / sample_fps)), 1) if sample_fps else 1
        for i, frame in enumerate(container.decode(stream)):
            if i % step != 0:
                continue
            yield frame.to_ndarray(format="rgb24")
        container.close()
        return
    except Exception:
        pass

    import cv2
    cap = cv2.VideoCapture(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(int(round(src_fps / sample_fps)), 1) if sample_fps else 1
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        idx += 1
    cap.release()


def probe_duration(path: str | Path) -> float:
    """Return video duration in seconds (best effort)."""
    try:
        import av
        with av.open(str(path)) as c:
            if c.duration:
                return c.duration / 1_000_000
    except Exception:
        pass
    import cv2
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return float(n / fps) if fps else 0.0
