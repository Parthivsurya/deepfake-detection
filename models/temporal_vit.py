"""Temporal Vision Transformer (factorized space-time attention).

A spatial ViT extracts a [CLS] embedding per frame, then a temporal Transformer
attends across frames. This is the TimeSformer "divided attention" pattern
without the extra patch-time attention, which is much cheaper and tends to be
sufficient for short clips (T ≤ 32).

Returns:
    cls_token    : (B, D) clip-level summary
    frame_tokens : (B, T, D) per-frame summaries (used by AV-sync)
"""
from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
from einops import rearrange


class PatchEmbed(nn.Module):
    def __init__(self, image_size: int = 224, patch_size: int = 16, in_chans: int = 3,
                 embed_dim: int = 384):
        super().__init__()
        assert image_size % patch_size == 0
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, N, D)
        x = self.proj(x)
        return rearrange(x, "b d h w -> b (h w) d")


class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout=dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False, attn_mask=attn_mask)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x


class SpatialViT(nn.Module):
    def __init__(self, image_size: int = 224, patch_size: int = 16, embed_dim: int = 384,
                 depth: int = 6, num_heads: int = 6, mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(image_size, patch_size, 3, embed_dim)
        n_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) -> (B, D)
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)[:, 0]   # (B, D)


class TemporalTransformer(nn.Module):
    """Attention over T per-frame embeddings, with learned temporal position."""

    def __init__(self, embed_dim: int, depth: int = 4, num_heads: int = 6,
                 mlp_ratio: float = 4.0, dropout: float = 0.1, max_frames: int = 64):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_frames + 1, embed_dim))
        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, frame_feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # frame_feats: (B, T, D)
        B, T, _ = frame_feats.shape
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, frame_feats], dim=1) + self.pos_embed[:, : T + 1]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0], x[:, 1:]    # cls, per-frame tokens


class TemporalViT(nn.Module):
    """Per-frame ViT + temporal Transformer."""

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 384,
        spatial_depth: int = 6,
        temporal_depth: int = 4,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_frames: int = 64,
    ):
        super().__init__()
        self.spatial = SpatialViT(image_size, patch_size, embed_dim, spatial_depth,
                                  num_heads, mlp_ratio, dropout)
        self.temporal = TemporalTransformer(embed_dim, temporal_depth, num_heads,
                                            mlp_ratio, dropout, max_frames)
        self.embed_dim = embed_dim

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # frames: (B, T, 3, H, W)
        B, T = frames.shape[:2]
        x = rearrange(frames, "b t c h w -> (b t) c h w")
        per_frame = self.spatial(x)                       # (B*T, D)
        per_frame = rearrange(per_frame, "(b t) d -> b t d", b=B, t=T)
        cls, frame_tokens = self.temporal(per_frame)      # (B, D), (B, T, D)
        return cls, frame_tokens
