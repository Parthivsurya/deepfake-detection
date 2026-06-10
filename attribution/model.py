"""End-to-end source attribution model.

Wires together:
  * FingerprintExtractor  (parameter-free DCT + residual)
  * SpectralCNN, ResidualCNN
  * Frozen TemporalViT from the existing detector (semantic branch)
  * AttributionHead

The TemporalViT is loaded from a detector checkpoint and run in eval mode with
gradients disabled — we only train the fingerprint branches + head.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from models.temporal_vit import TemporalViT

from .attribution_head import AttributionHead
from .fingerprint import FingerprintExtractor
from .generators import num_known_classes
from .spectral_cnn import ResidualCNN, SpectralCNN


class SourceAttributionModel(nn.Module):
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
        spectral_dim: int = 256,
        residual_dim: int = 256,
        head_hidden: int = 384,
        num_classes: Optional[int] = None,
    ):
        super().__init__()
        if num_classes is None:
            num_classes = num_known_classes()

        self.backbone = TemporalViT(
            image_size, patch_size, embed_dim, spatial_depth,
            temporal_depth, num_heads, mlp_ratio, dropout, max_frames,
        )
        self.fingerprint = FingerprintExtractor()
        self.spectral_cnn = SpectralCNN(in_channels=3, embed_dim=spectral_dim)
        self.residual_cnn = ResidualCNN(in_channels=3, embed_dim=residual_dim)
        self.head = AttributionHead(
            semantic_dim=embed_dim,
            spectral_dim=spectral_dim,
            residual_dim=residual_dim,
            hidden_dim=head_hidden,
            num_classes=num_classes,
            dropout=dropout,
        )

        self._backbone_frozen = False

    # ------------------------------------------------------------------
    def load_backbone(self, ckpt_path: str | Path, strict: bool = False) -> None:
        """Load TemporalViT weights from a detector checkpoint and freeze them.

        Detector checkpoint stores the full MultimodalDeepfakeDetector
        state-dict under the "model" key with `visual.*` prefix on the ViT.
        """
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt.get("model", ckpt)
        visual_sd = {
            k[len("visual."):]: v
            for k, v in sd.items() if k.startswith("visual.")
        }
        if not visual_sd:
            raise RuntimeError(
                f"checkpoint {ckpt_path} has no `visual.*` keys "
                "— expected a MultimodalDeepfakeDetector checkpoint"
            )
        missing, unexpected = self.backbone.load_state_dict(visual_sd, strict=strict)
        if strict and (missing or unexpected):
            raise RuntimeError(f"backbone load mismatch: missing={missing} unexpected={unexpected}")
        self.freeze_backbone()

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        self._backbone_frozen = True

    def train(self, mode: bool = True):  # keep backbone in eval even when training
        super().train(mode)
        if self._backbone_frozen:
            self.backbone.eval()
        return self

    # ------------------------------------------------------------------
    def forward(self, frames: torch.Tensor) -> dict:
        """frames: (B, T, 3, H, W) normalized RGB."""
        # Semantic branch (frozen).
        if self._backbone_frozen:
            with torch.no_grad():
                _, v_tokens = self.backbone(frames)
        else:
            _, v_tokens = self.backbone(frames)
        semantic = v_tokens.mean(dim=1)                  # (B, embed_dim)

        # Fingerprint branches.
        fp = self.fingerprint(frames)
        spectral = self.spectral_cnn(fp["spectral"])      # (B, spectral_dim)
        residual = self.residual_cnn(fp["residual"])      # (B, residual_dim)

        out = self.head(semantic, spectral, residual)
        return {
            "logits": out["logits"],
            "embed": out["embed"],
            "semantic": semantic,
            "spectral": spectral,
            "residual": residual,
        }
