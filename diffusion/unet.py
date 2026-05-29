"""Compact UNet noise predictor ε_θ(x_t, t) for video frames.

A standard three-level UNet with sinusoidal time embeddings, GroupNorm + SiLU
activations, and additive time conditioning at every block. Operates on
(B, 3, H, W) image tensors — frames are flattened across the temporal axis
before being passed in.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device) / max(half - 1, 1)
    )
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(groups, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(min(groups, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SmallUNet(nn.Module):
    """Three-level UNet. ~1.5 M parameters at base_ch=32."""

    def __init__(self, in_channels: int = 3, base_ch: int = 32, time_dim: int = 128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        self.time_dim = time_dim

        c1, c2, c3 = base_ch, base_ch * 2, base_ch * 4
        self.in_conv = nn.Conv2d(in_channels, c1, 3, padding=1)

        self.down1 = ResBlock(c1, c1, time_dim)
        self.down2 = ResBlock(c1, c2, time_dim)
        self.down3 = ResBlock(c2, c3, time_dim)
        self.pool = nn.AvgPool2d(2)

        self.mid = ResBlock(c3, c3, time_dim)

        self.up3 = ResBlock(c3 + c3, c2, time_dim)
        self.up2 = ResBlock(c2 + c2, c1, time_dim)
        self.up1 = ResBlock(c1 + c1, c1, time_dim)

        self.out_norm = nn.GroupNorm(min(8, c1), c1)
        self.out_conv = nn.Conv2d(c1, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(sinusoidal_time_embedding(t, self.time_dim))
        h0 = self.in_conv(x)               # (B, c1, H, W)
        h1 = self.down1(h0, t_emb)
        h2 = self.down2(self.pool(h1), t_emb)
        h3 = self.down3(self.pool(h2), t_emb)
        m = self.mid(h3, t_emb)
        u3 = self.up3(torch.cat([m, h3], dim=1), t_emb)
        u3 = F.interpolate(u3, scale_factor=2, mode="nearest")
        u2 = self.up2(torch.cat([u3, h2], dim=1), t_emb)
        u2 = F.interpolate(u2, scale_factor=2, mode="nearest")
        u1 = self.up1(torch.cat([u2, h1], dim=1), t_emb)
        return self.out_conv(F.silu(self.out_norm(u1)))
