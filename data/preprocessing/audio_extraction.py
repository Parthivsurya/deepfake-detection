"""Audio extraction from video files plus clip loading."""
from __future__ import annotations
import subprocess
from pathlib import Path
from typing import Optional
import numpy as np


def extract_audio(video_path: str | Path, out_path: str | Path, sample_rate: int = 16000) -> Optional[str]:
    """Demux audio to mono WAV at `sample_rate` using ffmpeg.

    Returns the output path or None if extraction failed (e.g. silent video).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import av
        with av.open(str(video_path)) as container:
            if len(container.streams.audio) == 0:
                return None
    except Exception:
        pass
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video_path),
        "-ac", "1", "-ar", str(sample_rate),
        "-vn", str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return str(out_path) if out_path.exists() else None


def load_audio_clip(
    path: str | Path,
    sample_rate: int = 16000,
    seconds: float = 4.0,
    offset: float = 0.0,
) -> np.ndarray:
    """Load `seconds` of audio starting at `offset`. Pads with zeros if too short."""
    from utils.audio_utils import load_waveform
    target_len = int(sample_rate * seconds)
    y = load_waveform(path, sample_rate)
    start = int(offset * sample_rate)
    clip = y[start:start + target_len]
    if len(clip) < target_len:
        clip = np.pad(clip, (0, target_len - len(clip)))
    return clip
