"""rPPG signal extraction + 1D temporal CNN encoder.

Pipeline (all parameter-free until the CNN):
    (B, T, 3, H, W) normalized frames
        ─► de-normalize to [0,1]
        ─► center face crop ROI mean per channel  -> (B, T, 3)
        ─► temporal AC normalization              -> (B, T, 3)
        ─► POS chrominance projection             -> (B, T, 2)
        ─► band-pass filter via 1D conv           -> (B, T, 2)
        ─► PhysioEncoder (1D CNN)                 -> (B, embed_dim)

POS reference: Wang et al. 2017, "Algorithmic Principles of Remote PPG".

Design notes:
  * We use the *center 50% bbox* of each frame as a face ROI; the upstream
    frame extractor already gives us face-cropped frames, so center-crop is a
    good proxy for the cheek region (where PPG SNR is highest).
  * The bandpass-via-conv keeps everything differentiable and on-GPU.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.datasets.base import _IMG_MEAN, _IMG_STD


# Cached as buffers on-the-fly per dtype/device
_MEAN = torch.tensor(_IMG_MEAN)
_STD = torch.tensor(_IMG_STD)


def _denormalize(frames: torch.Tensor) -> torch.Tensor:
    """(B, T, 3, H, W) normalized -> (B, T, 3, H, W) in [0, 1]."""
    mean = _MEAN.to(frames.device, frames.dtype).view(1, 1, 3, 1, 1)
    std = _STD.to(frames.device, frames.dtype).view(1, 1, 3, 1, 1)
    return (frames * std + mean).clamp(0.0, 1.0)


def extract_face_rgb_means(
    frames: torch.Tensor, roi_fraction: float = 0.5
) -> torch.Tensor:
    """Mean RGB of the central `roi_fraction` x `roi_fraction` square per frame.

    Input  : (B, T, 3, H, W) normalized
    Output : (B, T, 3)
    """
    rgb = _denormalize(frames)
    _, _, _, H, W = rgb.shape
    h = int(H * roi_fraction); w = int(W * roi_fraction)
    y0 = (H - h) // 2; x0 = (W - w) // 2
    roi = rgb[:, :, :, y0:y0 + h, x0:x0 + w]                # (B, T, 3, h, w)
    return roi.mean(dim=(-1, -2))                            # (B, T, 3)


def pos_chrominance(rgb_seq: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """POS (Plane-Orthogonal-to-Skin) chrominance signal.

    Input  : (B, T, 3) per-frame RGB means
    Output : (B, T, 2) two chrominance channels (X, Y in POS notation)
    """
    # Temporal AC: divide by per-channel temporal mean and subtract 1.
    mu = rgb_seq.mean(dim=1, keepdim=True).clamp_min(eps)
    ac = rgb_seq / mu - 1.0                                 # (B, T, 3)

    R, G, B = ac.unbind(dim=-1)
    X = 3.0 * R - 2.0 * G                                   # POS X
    Y = 1.5 * R + G - 1.5 * B                               # POS Y
    return torch.stack([X, Y], dim=-1)                      # (B, T, 2)


def _bandpass_kernel(length: int, low_hz: float, high_hz: float, fps: float) -> torch.Tensor:
    """Simple FIR band-pass via DFT magnitude shaping; returns (1, 1, L) kernel."""
    n = torch.arange(length, dtype=torch.float32) - length // 2
    # Difference of two sincs = bandpass
    def _sinc_lp(cutoff_norm):
        x = 2.0 * cutoff_norm * n
        return torch.where(n == 0, torch.tensor(2.0 * cutoff_norm), torch.sin(torch.pi * x) / (torch.pi * n))
    low_n = low_hz / fps
    high_n = high_hz / fps
    bp = _sinc_lp(high_n) - _sinc_lp(low_n)
    # Hann window
    w = 0.5 - 0.5 * torch.cos(2 * torch.pi * torch.arange(length).float() / max(length - 1, 1))
    bp = bp * w
    return bp.view(1, 1, length)


def rppg_temporal_signal(
    frames: torch.Tensor,
    fps: float = 4.0,
    low_hz: float = 0.7,
    high_hz: float = 3.0,
    bp_taps: int = 9,
) -> torch.Tensor:
    """End-to-end rPPG signal up to (but not including) the encoder CNN.

    Input  : (B, T, 3, H, W) normalized
    Output : (B, T, 2) band-pass-filtered POS chrominance
    """
    rgb = extract_face_rgb_means(frames)                    # (B, T, 3)
    pos = pos_chrominance(rgb)                              # (B, T, 2)

    # Make sure the FIR window isn't longer than T.
    T = pos.size(1)
    taps = min(bp_taps, max(T // 2 * 2 + 1, 3))
    kern = _bandpass_kernel(taps, low_hz, high_hz, fps).to(pos.device, pos.dtype)
    # Same kernel applied to each of the 2 channels.
    kern = kern.expand(2, 1, taps)
    pad = taps // 2
    x = pos.transpose(1, 2)                                 # (B, 2, T)
    y = F.conv1d(x, kern, padding=pad, groups=2)
    return y.transpose(1, 2)                                # (B, T, 2)


# ---------------------------------------------------------------------- encoder
class PhysioEncoder(nn.Module):
    """Tiny 1D CNN over the rPPG signal -> clip embedding F_P."""

    def __init__(
        self,
        embed_dim: int = 128,
        in_channels: int = 2,
        fps: float = 4.0,
    ):
        super().__init__()
        self.fps = fps
        ch = 32
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, ch, 5, padding=2), nn.BatchNorm1d(ch), nn.GELU(),
            nn.Conv1d(ch, ch * 2, 5, padding=2),       nn.BatchNorm1d(ch * 2), nn.GELU(),
            nn.Conv1d(ch * 2, ch * 4, 3, padding=1),   nn.BatchNorm1d(ch * 4), nn.GELU(),
            nn.Conv1d(ch * 4, embed_dim, 3, padding=1), nn.BatchNorm1d(embed_dim), nn.GELU(),
        )
        self.embed_dim = embed_dim

    def forward(self, frames: torch.Tensor) -> dict:
        """frames: (B, T, 3, H, W) -> {clip: (B, D), seq: (B, T, D)}."""
        sig = rppg_temporal_signal(frames, fps=self.fps)    # (B, T, 2)
        x = sig.transpose(1, 2)                             # (B, 2, T)
        h = self.net(x)                                     # (B, D, T)
        seq = h.transpose(1, 2).contiguous()                # (B, T, D)
        clip = seq.mean(dim=1)                              # (B, D)
        return {"clip": clip, "seq": seq, "signal": sig}
