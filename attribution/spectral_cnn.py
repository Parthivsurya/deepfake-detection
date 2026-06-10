"""Small CNNs that consume the spectral and residual fingerprint maps.

Both branches share the same architecture but have separate weights — they
look at different signals so a single shared trunk would conflate features.
"""
from __future__ import annotations
import torch
import torch.nn as nn


def _conv_bn(c_in: int, c_out: int, k: int = 3, s: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, k, stride=s, padding=k // 2, bias=False),
        nn.BatchNorm2d(c_out),
        nn.GELU(),
    )


class _FingerprintCNN(nn.Module):
    """Lightweight CNN -> per-frame embedding -> temporal-mean pooling."""

    def __init__(self, in_channels: int = 3, embed_dim: int = 256):
        super().__init__()
        self.stem = nn.Sequential(
            _conv_bn(in_channels, 32, k=3, s=2),  # 224 -> 112
            _conv_bn(32, 64, k=3, s=2),           # 112 -> 56
            _conv_bn(64, 128, k=3, s=2),          # 56  -> 28
            _conv_bn(128, 192, k=3, s=2),         # 28  -> 14
            _conv_bn(192, 256, k=3, s=2),         # 14  -> 7
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, embed_dim),
        )
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C, H, W) -> (B, embed_dim) by temporal mean."""
        B, T, C, H, W = x.shape
        flat = x.reshape(B * T, C, H, W)
        feat = self.head(self.stem(flat))                 # (B*T, D)
        return feat.reshape(B, T, -1).mean(dim=1)         # (B, D)


class SpectralCNN(_FingerprintCNN):
    pass


class ResidualCNN(_FingerprintCNN):
    pass


# ------------------------------------------------------------------ audio
def _conv1d_bn(c_in: int, c_out: int, k: int = 3, s: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(c_in, c_out, k, stride=s, padding=k // 2, bias=False),
        nn.BatchNorm1d(c_out),
        nn.GELU(),
    )


class AudioFingerprintCNN(nn.Module):
    """1D CNN over the mel-residual spectrogram -> clip embedding.

    Input:  (B, n_mels, T_a) mel residual
    Output: (B, embed_dim)
    """

    def __init__(self, n_mels: int = 80, embed_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            _conv1d_bn(n_mels, 64, k=5, s=2),
            _conv1d_bn(64, 128, k=5, s=2),
            _conv1d_bn(128, 192, k=3, s=2),
            _conv1d_bn(192, 256, k=3, s=2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, embed_dim),
        )
        self.embed_dim = embed_dim

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(mel))
