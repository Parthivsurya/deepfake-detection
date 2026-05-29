"""Elastic Weight Consolidation (Kirkpatrick et al., 2017).

Mitigates catastrophic forgetting by penalising movement of parameters that
were important to a previous task. "Importance" is the diagonal of the
Fisher Information Matrix, estimated empirically as

    F_i = E_x[(∂ log p_θ(y|x) / ∂ θ_i)^2]

We snapshot θ* (post-training values) and add

    L_EWC = (λ / 2) · Σ_i F_i (θ_i − θ*_i)^2

to subsequent training losses. Anchors and Fisher entries accumulate across
tasks so each task's important weights stay protected.
"""
from __future__ import annotations
from typing import Callable, Dict, Iterable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def _ce_loss_fn(model: nn.Module, batch: Dict[str, torch.Tensor],
                device: str) -> torch.Tensor:
    out = model(
        batch["frames"].to(device),
        batch["audio"].to(device),
        has_audio=batch.get("has_audio").to(device) if batch.get("has_audio") is not None else None,
    )
    return F.cross_entropy(out["logits"], batch["label"].to(device))


class ElasticWeightConsolidation:
    """Accumulates Fisher importances and parameter anchors across tasks."""

    def __init__(self, lam: float = 1.0):
        self.lam = float(lam)
        self.fisher: Dict[str, torch.Tensor] = {}
        self.anchors: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def _snapshot_params(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    def consolidate(
        self,
        model: nn.Module,
        loader: Iterable,
        device: str = "cpu",
        max_batches: Optional[int] = None,
        loss_fn: Optional[Callable] = None,
    ) -> None:
        """Estimate diagonal Fisher on `loader` and merge with prior task data.

        Adds the new Fisher entries to any pre-existing ones (Fisher is
        additive across tasks under independent-task assumptions) and
        replaces anchors with the current parameter values.
        """
        model.eval()
        loss_fn = loss_fn or _ce_loss_fn

        fisher_new: Dict[str, torch.Tensor] = {
            n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad
        }
        n_batches = 0
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            model.zero_grad(set_to_none=True)
            loss = loss_fn(model, batch, device)
            loss.backward()
            for n, p in model.named_parameters():
                if p.grad is not None and n in fisher_new:
                    fisher_new[n] += p.grad.detach() ** 2
            n_batches += 1
        if n_batches > 0:
            for n in fisher_new:
                fisher_new[n] /= n_batches

        # accumulate across tasks
        for n, f in fisher_new.items():
            if n in self.fisher:
                self.fisher[n] = self.fisher[n] + f
            else:
                self.fisher[n] = f

        # anchor at current θ
        self.anchors = self._snapshot_params(model)
        model.zero_grad(set_to_none=True)

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """Quadratic penalty around the most recent anchor."""
        if not self.fisher:
            # no consolidated task yet — return a zero tensor on the model's device
            p_any = next(model.parameters())
            return torch.zeros((), device=p_any.device, dtype=p_any.dtype)
        loss = None
        for n, p in model.named_parameters():
            if n not in self.fisher:
                continue
            f = self.fisher[n].to(p.device)
            anchor = self.anchors[n].to(p.device)
            term = (f * (p - anchor).pow(2)).sum()
            loss = term if loss is None else loss + term
        if loss is None:
            p_any = next(model.parameters())
            return torch.zeros((), device=p_any.device, dtype=p_any.dtype)
        return 0.5 * self.lam * loss

    def state_dict(self) -> dict:
        return {"lam": self.lam, "fisher": self.fisher, "anchors": self.anchors}

    def load_state_dict(self, state: dict) -> None:
        self.lam = float(state["lam"])
        self.fisher = dict(state["fisher"])
        self.anchors = dict(state["anchors"])
