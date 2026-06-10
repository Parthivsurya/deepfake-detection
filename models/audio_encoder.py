"""Lightweight 1D CNN audio encoder over log-mel spectrograms.

Input is the raw waveform; the encoder computes log-mel on the fly with
torchaudio so the dataset side stays free of audio-DSP dependencies at train
time. Output is a sequence of T_a embeddings (one per time-frame) plus a
mean-pooled clip embedding.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torchaudio


class AudioEncoder(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 400,
        hop_length: int = 160,
        embed_dim: int = 256,
    ):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )
        self.log = torchaudio.transforms.AmplitudeToDB(stype="power")

        ch = 64
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, ch, 5, padding=2), nn.BatchNorm1d(ch), nn.GELU(),
            nn.Conv1d(ch, ch * 2, 5, stride=2, padding=2), nn.BatchNorm1d(ch * 2), nn.GELU(),
            nn.Conv1d(ch * 2, ch * 2, 3, padding=1), nn.BatchNorm1d(ch * 2), nn.GELU(),
            nn.Conv1d(ch * 2, ch * 4, 3, stride=2, padding=1), nn.BatchNorm1d(ch * 4), nn.GELU(),
            nn.Conv1d(ch * 4, embed_dim, 3, padding=1), nn.BatchNorm1d(embed_dim), nn.GELU(),
        )
        self.embed_dim = embed_dim

    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # waveform: (B, samples)
        mel = self.log(self.mel(waveform))            # (B, n_mels, T_a)
        h = self.conv(mel)                            # (B, D, T')
        seq = h.transpose(1, 2).contiguous()          # (B, T', D)
        clip = seq.mean(dim=1)                        # (B, D)
        return clip, seq


class Wav2VecAudioEncoder(nn.Module):
    """Drop-in replacement using HuggingFace wav2vec2 features.

    Returns the same `(clip, seq)` interface as `AudioEncoder` so the rest of
    the pipeline doesn't change. The wav2vec2 backbone is loaded lazily and
    optionally frozen (default: frozen — feature extractor mode).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        embed_dim: int = 256,
        pretrained: str = "facebook/wav2vec2-base",
        freeze: bool = True,
    ):
        super().__init__()
        try:
            from transformers import Wav2Vec2Model
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "Wav2VecAudioEncoder requires `pip install transformers`"
            ) from e
        if sample_rate != 16000:
            raise ValueError("wav2vec2-base expects 16 kHz audio")
        self.backbone = Wav2Vec2Model.from_pretrained(pretrained)
        self.proj = nn.Linear(self.backbone.config.hidden_size, embed_dim)
        self.embed_dim = embed_dim
        self.freeze = freeze
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.backbone.eval()
        return self

    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # waveform: (B, samples)  float32 in [-1, 1]
        if self.freeze:
            with torch.no_grad():
                h = self.backbone(waveform).last_hidden_state    # (B, T', hidden)
        else:
            h = self.backbone(waveform).last_hidden_state
        seq = self.proj(h)                                       # (B, T', D)
        clip = seq.mean(dim=1)                                   # (B, D)
        return clip, seq


def build_audio_encoder(
    kind: str = "cnn",
    sample_rate: int = 16000,
    embed_dim: int = 256,
    **kwargs,
) -> nn.Module:
    """Factory: kind in {"cnn", "wav2vec"}."""
    if kind == "cnn":
        return AudioEncoder(sample_rate=sample_rate, embed_dim=embed_dim, **kwargs)
    if kind == "wav2vec":
        return Wav2VecAudioEncoder(sample_rate=sample_rate, embed_dim=embed_dim, **kwargs)
    raise ValueError(f"unknown audio encoder kind: {kind!r}")
