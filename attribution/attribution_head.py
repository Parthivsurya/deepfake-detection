"""Fusion + classification head.

Accepts a variable list of branch embeddings (video/audio/spectral/residual)
and projects to `num_classes` logits.  Returns the pre-logit embedding too so
the open-set scorers can fit on it.
"""
from __future__ import annotations
from typing import Sequence

import torch
import torch.nn as nn


class AttributionHead(nn.Module):
    def __init__(
        self,
        branch_dims: Sequence[int],
        hidden_dim: int = 384,
        num_classes: int = 9,
        dropout: float = 0.1,
    ):
        super().__init__()
        in_dim = sum(branch_dims)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.num_classes = num_classes
        self.embed_dim = hidden_dim

    def forward(self, *branches: torch.Tensor) -> dict:
        fused = torch.cat(branches, dim=-1)
        embed = self.proj(fused)
        logits = self.classifier(embed)
        return {"logits": logits, "embed": embed}
