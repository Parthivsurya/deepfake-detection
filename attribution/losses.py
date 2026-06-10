"""Auxiliary losses for attribution.

Supervised contrastive loss (Khosla et al. 2020) — the L_CPL term from the
TRINETRA diagram. Pulls same-generator embeddings together, pushes different-
generator embeddings apart. Used as an auxiliary term alongside the standard
cross-entropy.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised contrastive loss on L2-normalized embeddings.

    Args:
        temperature: lower -> sharper contrast (default 0.1).
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # embeds: (B, D), labels: (B,)
        if embeds.size(0) < 2:
            return embeds.new_zeros(())
        z = F.normalize(embeds, dim=-1)
        sim = z @ z.t() / self.temperature                    # (B, B)
        # numerical stability: subtract per-row max before exp
        sim = sim - sim.max(dim=-1, keepdim=True).values.detach()

        B = embeds.size(0)
        eye = torch.eye(B, dtype=torch.bool, device=embeds.device)
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~eye
        # If a sample has no other positive in the batch, skip it.
        valid = pos_mask.any(dim=-1)
        if not valid.any():
            return embeds.new_zeros(())

        exp_sim = torch.exp(sim) * (~eye).float()             # zero out self
        denom = exp_sim.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        log_prob = sim - torch.log(denom)

        # Mean of log-prob over positive pairs, then negate.
        pos_count = pos_mask.sum(dim=-1).clamp_min(1).float()
        mean_log_prob_pos = (log_prob * pos_mask.float()).sum(dim=-1) / pos_count
        loss = -mean_log_prob_pos[valid].mean()
        return loss
