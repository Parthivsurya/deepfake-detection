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
