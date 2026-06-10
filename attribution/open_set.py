"""Open-set scoring for unknown generators.

Two cheap, complementary scorers fit on the validation embeddings:
  * EnergyScorer       — -logsumexp(logits); low energy => in-distribution.
                         Liu et al. 2020, "Energy-based OOD Detection".
  * MahalanobisScorer  — distance to per-class Gaussians in the head's embedding
                         space; classic Lee et al. 2018 OOD detector.

Both are decoupled from the model — fit on stored val features, then score new
inputs.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


class EnergyScorer:
    """Computes free-energy OOD score.  No fitting required."""

    @staticmethod
    def score(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        # Higher score => more likely OOD (unknown generator).
        return -temperature * torch.logsumexp(logits / temperature, dim=-1)


class MahalanobisScorer:
    """Per-class Gaussian on the AttributionHead embedding space.

    Use:
        s = MahalanobisScorer().fit(val_embeds, val_labels).score(test_embeds)
    """

    def __init__(self) -> None:
        self.means: torch.Tensor | None = None       # (K, D)
        self.precision: torch.Tensor | None = None   # (D, D)
        self.classes: list[int] = []

    def fit(self, embeds: torch.Tensor, labels: torch.Tensor) -> "MahalanobisScorer":
        embeds = embeds.detach().cpu().float()
        labels = labels.detach().cpu().long()
        classes = sorted(labels.unique().tolist())
        means, centered = [], []
        for c in classes:
            mask = labels == c
            xc = embeds[mask]
            mu = xc.mean(dim=0)
            means.append(mu)
            centered.append(xc - mu)
        self.means = torch.stack(means, dim=0)              # (K, D)
        all_c = torch.cat(centered, dim=0)                  # (N, D)
        cov = (all_c.t() @ all_c) / max(all_c.size(0) - 1, 1)
        # ridge for numerical stability
        cov += 1e-4 * torch.eye(cov.size(0))
        self.precision = torch.linalg.inv(cov)
        self.classes = classes
        return self

    def score(self, embeds: torch.Tensor) -> torch.Tensor:
        """Min Mahalanobis distance to any class centroid.  Higher => more OOD."""
        assert self.means is not None and self.precision is not None, "fit() first"
        x = embeds.detach().cpu().float()
        diffs = x.unsqueeze(1) - self.means.unsqueeze(0)            # (N, K, D)
        m = torch.einsum("nkd,de,nke->nk", diffs, self.precision, diffs)
        return m.min(dim=1).values                                  # (N,)


def softmax_confidence(logits: torch.Tensor) -> torch.Tensor:
    """Top-1 softmax probability — handy as a simple confidence reading."""
    return F.softmax(logits, dim=-1).max(dim=-1).values
