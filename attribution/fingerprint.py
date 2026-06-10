"""Forensic fingerprint feature extractors.

Two complementary signals:
  1. `dct_highpass`     — block-DCT high-frequency magnitude map; captures
                          upsampling-checkerboard artifacts of GAN architectures
                          and the over-smoothness of diffusion sampling.
  2. `denoise_residual` — `x - blur(x)`; a cheap proxy for the PRNU-style
                          generator-noise pattern (Marra et al. 2019).

Both extractors are *parameter-free* tensor ops so they run on GPU inside the
training loop and do not need pre-computation.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ DCT
def _dct_matrix(n: int, device, dtype) -> torch.Tensor:
    """Orthonormal DCT-II basis of size (n, n)."""
    k = torch.arange(n, device=device, dtype=dtype).unsqueeze(0)
    i = torch.arange(n, device=device, dtype=dtype).unsqueeze(1)
    m = torch.cos((2 * i + 1) * k * torch.pi / (2 * n))
    m[0] *= 1.0 / (n ** 0.5)
    m[1:] *= (2.0 / n) ** 0.5
    return m  # rows are basis vectors


def block_dct2(x: torch.Tensor, block: int = 8) -> torch.Tensor:
    """2D block DCT on a (B, C, H, W) image. Returns the same shape, per-block DCT."""
    B, C, H, W = x.shape
    assert H % block == 0 and W % block == 0, f"img {H}x{W} not divisible by block {block}"
    m = _dct_matrix(block, x.device, x.dtype)              # (b, b)
    # Unfold into blocks: (B, C, H/b, b, W/b, b) -> (B, C, H/b, W/b, b, b)
    xb = x.unfold(2, block, block).unfold(3, block, block)
    xb = xb.contiguous()
    coef = m @ xb @ m.t()                                  # 2D DCT per block
    # Re-fold to (B, C, H, W)
    coef = coef.permute(0, 1, 2, 4, 3, 5).reshape(B, C, H, W)
    return coef


def dct_highpass(x: torch.Tensor, block: int = 8, keep_low: int = 2) -> torch.Tensor:
    """Block-DCT magnitude with the top-left `keep_low x keep_low` low-freq coefs zeroed.

    The result is log1p-magnitude — large where generator upsampling leaks
    high-frequency energy.
    """
    coef = block_dct2(x, block=block)
    B, C, H, W = coef.shape
    mask = torch.ones(block, block, device=x.device, dtype=x.dtype)
    mask[:keep_low, :keep_low] = 0.0
    mask = mask.repeat(H // block, W // block)             # (H, W)
    return torch.log1p(coef.abs() * mask)


# ------------------------------------------------------------------ residual
_GAUSS_K = 5


def _gaussian_kernel(k: int, sigma: float, device, dtype) -> torch.Tensor:
    ax = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return torch.outer(g, g)                               # (k, k)


def denoise_residual(x: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """`x - gaussian_blur(x)` per channel.  Cheap PRNU-style residual."""
    B, C, H, W = x.shape
    k = _gaussian_kernel(_GAUSS_K, sigma, x.device, x.dtype)
    k = k.expand(C, 1, _GAUSS_K, _GAUSS_K).contiguous()
    pad = _GAUSS_K // 2
    blur = F.conv2d(x, k, padding=pad, groups=C)
    return x - blur


# ------------------------------------------------------------------ wrapper
class FingerprintExtractor(nn.Module):
    """Stateless module that returns both spectral and residual feature maps.

    Input:  (B, T, 3, H, W)  — already normalized by the detector pipeline.
    Output: dict with `spectral` and `residual`, each (B, T, 3, H, W).

    The temporal axis is preserved; downstream CNNs treat it as an extra batch
    dimension to keep them lightweight.
    """

    def __init__(self, dct_block: int = 8, keep_low: int = 2, blur_sigma: float = 1.0):
        super().__init__()
        self.dct_block = dct_block
        self.keep_low = keep_low
        self.blur_sigma = blur_sigma

    def forward(self, frames: torch.Tensor) -> dict:
        B, T, C, H, W = frames.shape
        flat = frames.reshape(B * T, C, H, W)
        spec = dct_highpass(flat, block=self.dct_block, keep_low=self.keep_low)
        resid = denoise_residual(flat, sigma=self.blur_sigma)
        return {
            "spectral": spec.reshape(B, T, C, H, W),
            "residual": resid.reshape(B, T, C, H, W),
        }
