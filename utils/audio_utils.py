"""Audio loading and feature extraction helpers."""
from __future__ import annotations
from pathlib import Path
import numpy as np


def load_waveform(path: str | Path, sample_rate: int = 16000) -> np.ndarray:
    """Load mono waveform resampled to `sample_rate`. Returns float32 [-1, 1]."""
    import librosa
    y, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    return y.astype(np.float32)


def log_mel_spectrogram(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    n_mels: int = 80,
    n_fft: int = 400,
    hop_length: int = 160,
) -> np.ndarray:
    """Return log-mel spectrogram of shape (n_mels, T)."""
    import librosa
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )
    return np.log1p(mel).astype(np.float32)
