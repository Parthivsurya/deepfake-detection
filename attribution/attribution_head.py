"""Fusion + classification head.

Concatenates the three branch embeddings and projects to `num_classes` logits.
Returns the pre-logit embedding too so the open-set scorers can fit on it.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class AttributionHead(nn.Module):
    def __init__(
        self,
        semantic_dim: int,
        spectral_dim: int,
        residual_dim: int,
        hidden_dim: int = 384,
        num_classes: int = 9,
        dropout: float = 0.1,
    ):
        super().__init__()
        in_dim = semantic_dim + spectral_dim + residual_dim
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.num_classes = num_classes
        self.embed_dim = hidden_dim

    def forward(
        self,
        semantic: torch.Tensor,
        spectral: torch.Tensor,
        residual: torch.Tensor,
    ) -> dict:
        fused = torch.cat([semantic, spectral, residual], dim=-1)
        embed = self.proj(fused)
        logits = self.classifier(embed)
        return {"logits": logits, "embed": embed}
