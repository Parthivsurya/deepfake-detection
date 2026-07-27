"""ImageNet-pretrained visual backbones with the same interface as TemporalViT.

Returns:
    cls_token    : (B, embed_dim) clip-level summary
    frame_tokens : (B, T, embed_dim) per-frame summaries

This drops in cleanly wherever the detector previously used `TemporalViT`.

Available kinds:
    "resnet50"        — 25.6M params, 4.1 GFLOPs, the prior default.
    "efficientnet_b0" — 5.3M params, ~0.39 GFLOPs, lightweight-edge default.

For small-data deepfake training, fine-tuning a from-scratch ViT requires far
more clips than we have (~750). A frozen ImageNet-pretrained CNN acting as a
feature extractor + a small temporal head is the standard, well-trodden setup
(Xception was the original FaceForensics++ baseline). EfficientNet-B0 is the
lightweight choice for real-time / edge deployment in line with the RDD spec.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from einops import rearrange

from .temporal_vit import TemporalTransformer


class ResNet50Backbone(nn.Module):
    """Frozen ImageNet-pretrained ResNet50 + small temporal transformer."""

    def __init__(
        self,
        embed_dim: int = 384,
        temporal_depth: int = 4,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_frames: int = 64,
        freeze: bool = True,
    ):
        super().__init__()
        from torchvision.models import resnet50, ResNet50_Weights
        weights = ResNet50_Weights.IMAGENET1K_V2  # newer, slightly better weights
        m = resnet50(weights=weights)
        # Keep everything up to (but not including) the final FC layer.
        # output of avgpool is (B, 2048, 1, 1).
        self.backbone = nn.Sequential(*list(m.children())[:-1])
        self.proj = nn.Linear(2048, embed_dim)
        self.temporal = TemporalTransformer(
            embed_dim, depth=temporal_depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, dropout=dropout, max_frames=max_frames,
        )
        self.embed_dim = embed_dim
        self.freeze = freeze
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            # Keep BN running-stats fixed throughout training.
            self.backbone.eval()
        return self

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # frames: (B, T, 3, H, W) — already normalized with ImageNet mean/std
        B, T = frames.shape[:2]
        x = rearrange(frames, "b t c h w -> (b t) c h w")

        if self.freeze:
            with torch.no_grad():
                feats = self.backbone(x).flatten(1)        # (B*T, 2048)
        else:
            feats = self.backbone(x).flatten(1)

        per_frame = self.proj(feats)                       # (B*T, D)
        per_frame = rearrange(per_frame, "(b t) d -> b t d", b=B, t=T)
        cls, frame_tokens = self.temporal(per_frame)       # (B, D), (B, T, D)
        return cls, frame_tokens


class EfficientNetB0Backbone(nn.Module):
    """Frozen ImageNet-pretrained EfficientNet-B0 + small temporal transformer.

    ~5.3M params, ~0.39 GFLOPs at 224 — roughly 5x lighter than ResNet50 with
    comparable ImageNet top-1 (~77%). Output of avgpool is (B, 1280, 1, 1).
    """

    def __init__(
        self,
        embed_dim: int = 384,
        temporal_depth: int = 4,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_frames: int = 64,
        freeze: bool = True,
    ):
        super().__init__()
        import os as _os
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        # For eval the ImageNet weights are overwritten by the checkpoint, so we
        # can skip the download with DDET_BACKBONE_NO_PRETRAINED=1 (avoids the
        # torchvision hub fetch that stalls on flaky networks).
        weights = None if _os.environ.get("DDET_BACKBONE_NO_PRETRAINED") == "1" \
            else EfficientNet_B0_Weights.IMAGENET1K_V1
        m = efficientnet_b0(weights=weights)
        # features (conv stack) + avgpool; drop classifier (Dropout + Linear)
        self.backbone = nn.Sequential(m.features, m.avgpool)
        self.proj = nn.Linear(1280, embed_dim)
        self.temporal = TemporalTransformer(
            embed_dim, depth=temporal_depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, dropout=dropout, max_frames=max_frames,
        )
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

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T = frames.shape[:2]
        x = rearrange(frames, "b t c h w -> (b t) c h w")

        if self.freeze:
            with torch.no_grad():
                feats = self.backbone(x).flatten(1)        # (B*T, 1280)
        else:
            feats = self.backbone(x).flatten(1)

        per_frame = self.proj(feats)                       # (B*T, D)
        per_frame = rearrange(per_frame, "(b t) d -> b t d", b=B, t=T)
        cls, frame_tokens = self.temporal(per_frame)       # (B, D), (B, T, D)
        return cls, frame_tokens


class ViTB16Backbone(nn.Module):
    """Frozen ImageNet-pretrained ViT-B/16 + small temporal transformer.

    86M backbone params. Strips the classification head; CLS token output
    is (B*T, 768) per frame, projected to embed_dim for the temporal head.
    """

    def __init__(
        self,
        embed_dim: int = 384,
        temporal_depth: int = 4,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_frames: int = 64,
        freeze: bool = True,
    ):
        super().__init__()
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        m = vit_b_16(weights=weights)
        m.heads = nn.Identity()  # forward now returns the 768-d CLS token
        self.backbone = m
        self.proj = nn.Linear(768, embed_dim)
        self.temporal = TemporalTransformer(
            embed_dim, depth=temporal_depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, dropout=dropout, max_frames=max_frames,
        )
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

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T = frames.shape[:2]
        x = rearrange(frames, "b t c h w -> (b t) c h w")

        if self.freeze:
            with torch.no_grad():
                feats = self.backbone(x)                   # (B*T, 768)
        else:
            feats = self.backbone(x)

        per_frame = self.proj(feats)                       # (B*T, D)
        per_frame = rearrange(per_frame, "(b t) d -> b t d", b=B, t=T)
        cls, frame_tokens = self.temporal(per_frame)       # (B, D), (B, T, D)
        return cls, frame_tokens


class ConvNeXtTinyBackbone(nn.Module):
    """Frozen ImageNet-pretrained ConvNeXt-Tiny + small temporal transformer.

    ~28M backbone params, modern-CNN baseline. Output of features+avgpool+LN
    is (B*T, 768) per frame, projected to embed_dim for the temporal head.
    """

    def __init__(
        self,
        embed_dim: int = 384,
        temporal_depth: int = 4,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_frames: int = 64,
        freeze: bool = True,
    ):
        super().__init__()
        from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        m = convnext_tiny(weights=weights)
        # features + avgpool + classifier[0] (LayerNorm2d) + classifier[1] (Flatten);
        # drop classifier[2] (Linear -> 1000 classes).
        self.backbone = nn.Sequential(
            m.features, m.avgpool, m.classifier[0], m.classifier[1],
        )
        self.proj = nn.Linear(768, embed_dim)
        self.temporal = TemporalTransformer(
            embed_dim, depth=temporal_depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, dropout=dropout, max_frames=max_frames,
        )
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

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T = frames.shape[:2]
        x = rearrange(frames, "b t c h w -> (b t) c h w")

        if self.freeze:
            with torch.no_grad():
                feats = self.backbone(x)                   # (B*T, 768)
        else:
            feats = self.backbone(x)

        per_frame = self.proj(feats)                       # (B*T, D)
        per_frame = rearrange(per_frame, "(b t) d -> b t d", b=B, t=T)
        cls, frame_tokens = self.temporal(per_frame)       # (B, D), (B, T, D)
        return cls, frame_tokens


BACKBONES = {
    "temporal_vit": None,           # sentinel; the detector keeps its existing branch
    "resnet50": ResNet50Backbone,
    "efficientnet_b0": EfficientNetB0Backbone,
    "vit_b_16": ViTB16Backbone,
    "convnext_tiny": ConvNeXtTinyBackbone,
}


def build_visual_backbone(
    kind: str,
    image_size: int,
    embed_dim: int,
    temporal_depth: int,
    num_heads: int,
    mlp_ratio: float,
    dropout: float,
    max_frames: int,
    # passed through to TemporalViT only
    patch_size: int = 16,
    spatial_depth: int = 6,
    # passed through to pretrained backbones only
    freeze: bool = True,
) -> nn.Module:
    """Factory: kind in BACKBONES keys."""
    common = dict(
        embed_dim=embed_dim,
        temporal_depth=temporal_depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        max_frames=max_frames,
        freeze=freeze,
    )
    if kind == "resnet50":
        return ResNet50Backbone(**common)
    if kind == "efficientnet_b0":
        return EfficientNetB0Backbone(**common)
    if kind == "vit_b_16":
        return ViTB16Backbone(**common)
    if kind == "convnext_tiny":
        return ConvNeXtTinyBackbone(**common)
    if kind == "temporal_vit":
        from .temporal_vit import TemporalViT
        return TemporalViT(
            image_size, patch_size, embed_dim, spatial_depth,
            temporal_depth, num_heads, mlp_ratio, dropout, max_frames,
        )
    raise ValueError(f"unknown visual backbone: {kind!r}")
