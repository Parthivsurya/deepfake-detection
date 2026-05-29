"""Adaptive fine-tuning trainer that combines new-task data, replay, and EWC.

Loss assembled per step:

    L = CE(new_batch) + λ_replay · CE(replay_batch) + L_EWC

`λ_replay` controls how strongly past distributions are preserved through
exemplar replay; `EWC.lam` controls how strongly important parameters from
prior tasks are anchored. Both are zero by default — turning either off
gives plain fine-tuning, and turning both on gives the standard
"replay + EWC" continual-learning recipe.

Lifecycle:

    trainer.train_on(task_loader, n_epochs=...)
    trainer.finish_task(consolidation_loader)   # snapshot Fisher + anchors
                                                 # and seed replay buffer

`finish_task` is what turns the model's current state into the next "old
task" that EWC will protect on subsequent calls.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .memory_buffer import ReplayBuffer, ClassBalancedReplayBuffer
from .ewc import ElasticWeightConsolidation


@dataclass
class ContinualConfig:
    replay_lambda: float = 1.0     # weight on replay-batch CE
    replay_batch_size: int = 4
    ewc_lambda: float = 0.0        # weight on EWC quadratic penalty
    consolidation_batches: int = 8 # batches used to estimate Fisher at task end
    grad_clip: Optional[float] = 1.0


def _forward_ce(model: nn.Module, batch: Dict[str, torch.Tensor],
                device: str) -> torch.Tensor:
    has_audio = batch.get("has_audio")
    if has_audio is not None:
        has_audio = has_audio.to(device)
    out = model(
        batch["frames"].to(device),
        batch["audio"].to(device),
        has_audio=has_audio,
    )
    return F.cross_entropy(out["logits"], batch["label"].to(device))


class ContinualTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        buffer: ReplayBuffer | ClassBalancedReplayBuffer,
        ewc: Optional[ElasticWeightConsolidation] = None,
        config: Optional[ContinualConfig] = None,
        device: str = "cpu",
        loss_fn: Optional[Callable] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.buffer = buffer
        self.ewc = ewc or ElasticWeightConsolidation(lam=0.0)
        self.cfg = config or ContinualConfig()
        self.device = device
        self.loss_fn = loss_fn or _forward_ce

    # ------------------------------------------------------------------
    # Per-step / per-epoch training
    # ------------------------------------------------------------------
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        new_loss = self.loss_fn(self.model, batch, self.device)
        total = new_loss

        replay_loss_val = 0.0
        if self.cfg.replay_lambda > 0 and len(self.buffer) > 0:
            rb = self.buffer.sample(self.cfg.replay_batch_size)
            if rb is not None:
                replay_loss = self.loss_fn(self.model, rb, self.device)
                total = total + self.cfg.replay_lambda * replay_loss
                replay_loss_val = float(replay_loss.item())

        ewc_loss_val = 0.0
        if self.cfg.ewc_lambda > 0 and self.ewc.fisher:
            # EWC.lam handles its own scaling; cfg.ewc_lambda is an outer multiplier
            # so users can keep ewc.lam at 1.0 and tune from the trainer config
            ewc_loss = self.ewc.penalty(self.model)
            total = total + self.cfg.ewc_lambda * ewc_loss
            ewc_loss_val = float(ewc_loss.item())

        total.backward()
        if self.cfg.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.optimizer.step()

        return {
            "loss_total": float(total.item()),
            "loss_new": float(new_loss.item()),
            "loss_replay": replay_loss_val,
            "loss_ewc": ewc_loss_val,
        }

    def train_on(self, loader: Iterable, n_epochs: int = 1,
                 add_to_buffer: bool = True) -> Dict[str, float]:
        n = 0
        agg = {"loss_total": 0.0, "loss_new": 0.0, "loss_replay": 0.0, "loss_ewc": 0.0}
        for _ in range(int(n_epochs)):
            for batch in loader:
                stats = self.train_step(batch)
                for k in agg:
                    agg[k] += stats[k]
                n += 1
                if add_to_buffer:
                    self.buffer.add(batch)
        if n > 0:
            for k in agg:
                agg[k] /= n
        return agg

    # ------------------------------------------------------------------
    # Task boundary
    # ------------------------------------------------------------------
    def finish_task(self, consolidation_loader: Optional[Iterable] = None) -> None:
        """Snapshot Fisher + anchors for EWC. Call after each task."""
        if self.cfg.ewc_lambda > 0 and consolidation_loader is not None:
            self.ewc.consolidate(
                self.model, consolidation_loader,
                device=self.device,
                max_batches=self.cfg.consolidation_batches,
            )
