"""Perturbation detectors.

Two flavours, both producing a per-clip score in [0, 1] (1 = likely
adversarially perturbed):

* `HeuristicPerturbationDetector` — Laplacian high-pass energy. Adversarial
  perturbations concentrate in high-frequency bands, so the mean |∇²x| is a
  cheap, training-free signal. Calibrated by a global threshold/temperature.

* `LearnablePerturbationDetector` — a small frame-wise CNN binary classifier
  pooled over the temporal axis. Trained separately on (clean, adversarial)
  pairs; the architecture is provided here, training is left to the user.

Both consume normalized frames in (B, T, 3, H, W) shape.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


_LAPLACIAN = torch.tensor([[0.0, -1.0, 0.0],
                           [-1.0, 4.0, -1.0],
                           [0.0, -1.0, 0.0]])


def high_frequency_energy(frames: torch.Tensor) -> torch.Tensor:
    """Mean absolute Laplacian per clip. Returns (B,) tensor."""
    if frames.dim() != 5:
        raise ValueError("expected (B, T, 3, H, W)")
    B, T, C, H, W = frames.shape
    kernel = _LAPLACIAN.to(frames).view(1, 1, 3, 3).expand(C, 1, 3, 3)
    x = frames.reshape(B * T, C, H, W)
    lap = F.conv2d(x, kernel, padding=1, groups=C)
    energy = lap.abs().mean(dim=(1, 2, 3))     # (B*T,)
    return energy.view(B, T).mean(dim=1)       # (B,)


class HeuristicPerturbationDetector(nn.Module):
    """Thresholded Laplacian energy with a sigmoid soft margin.

    Score = σ((energy − threshold) / temperature)

    Calibrate `threshold` on a held-out batch of clean clips
    (e.g. `threshold = clean_energy.mean() + 2 * clean_energy.std()`).
    """

    def __init__(self, threshold: float = 0.1, temperature: float = 0.02):
        super().__init__()
        self.register_buffer("threshold", torch.tensor(float(threshold)))
        self.register_buffer("temperature", torch.tensor(float(temperature)))

    @torch.inference_mode()
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        e = high_frequency_energy(frames)
        return torch.sigmoid((e - self.threshold) / self.temperature)

    def calibrate(self, clean_frames: torch.Tensor, k: float = 2.0) -> None:
        """Set `threshold = mean + k * std` of clean-clip energies."""
        e = high_frequency_energy(clean_frames)
        self.threshold = (e.mean() + k * e.std()).detach()


class LearnablePerturbationDetector(nn.Module):
    """Tiny per-frame CNN with mean-pool over the temporal axis."""

    def __init__(self, in_channels: int = 3, base_ch: int = 32):
        super().__init__()
        c1, c2, c3 = base_ch, base_ch * 2, base_ch * 4
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, padding=1), nn.GroupNorm(8, c1), nn.SiLU(),
            nn.Conv2d(c1, c1, 3, stride=2, padding=1), nn.GroupNorm(8, c1), nn.SiLU(),
            nn.Conv2d(c1, c2, 3, padding=1), nn.GroupNorm(8, c2), nn.SiLU(),
            nn.Conv2d(c2, c2, 3, stride=2, padding=1), nn.GroupNorm(8, c2), nn.SiLU(),
            nn.Conv2d(c2, c3, 3, padding=1), nn.GroupNorm(8, c3), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(c3, 1)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: (B, T, 3, H, W)
        B, T, C, H, W = frames.shape
        x = frames.reshape(B * T, C, H, W)
        f = self.cnn(x).flatten(1)            # (B*T, c3)
        logits = self.head(f).squeeze(-1)     # (B*T,)
        logits = logits.view(B, T).mean(dim=1)
        return torch.sigmoid(logits)
