"""Exemplar replay buffers for continual learning.

Two variants:

* `ReplayBuffer` — capacity-bounded reservoir-sampled buffer. Each new sample
  is admitted with probability `capacity / seen`, so the buffer stays a
  uniform random subset of the full stream regardless of how many samples
  have flowed through it. This is the standard "experience replay" recipe.

* `ClassBalancedReplayBuffer` — wraps one `ReplayBuffer` per label, so the
  number of stored fake / real exemplars stays balanced even when the new
  task is heavily skewed. Important for deepfake detection, where new
  generator families can flood the stream with one class.

Both buffers store the full batch dict (`frames`, `audio`, `has_audio`,
`label`) on CPU and return torch.utils.data-compatible batches at sample time.
"""
from __future__ import annotations
from typing import Dict, List, Optional
import random
import torch


_BATCH_KEYS = ("frames", "audio", "has_audio", "label")


def _to_cpu(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in batch.items() if k in _BATCH_KEYS}


def _split_batch(batch: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
    B = batch["frames"].size(0)
    out = []
    for i in range(B):
        sample = {}
        for k in _BATCH_KEYS:
            if k in batch:
                sample[k] = batch[k][i]
        out.append(sample)
    return out


def _collate(samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    keys = samples[0].keys()
    return {k: torch.stack([s[k] for s in samples], dim=0) for k in keys}


class ReplayBuffer:
    """Reservoir-sampled buffer of (frames, audio, has_audio, label) tuples.

    Capacity is in *clips*, not bytes. The buffer keeps a uniform random
    sample of all clips ever added (Vitter's Algorithm R).
    """

    def __init__(self, capacity: int = 512, seed: int = 0):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self._items: List[Dict[str, torch.Tensor]] = []
        self._seen = 0
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._items)

    @property
    def seen(self) -> int:
        return self._seen

    def add(self, batch: Dict[str, torch.Tensor]) -> None:
        batch = _to_cpu(batch)
        for sample in _split_batch(batch):
            self._seen += 1
            if len(self._items) < self.capacity:
                self._items.append(sample)
            else:
                j = self._rng.randint(0, self._seen - 1)
                if j < self.capacity:
                    self._items[j] = sample

    def sample(self, n: int) -> Optional[Dict[str, torch.Tensor]]:
        if not self._items:
            return None
        k = min(n, len(self._items))
        chosen = self._rng.sample(self._items, k)
        return _collate(chosen)

    def state_dict(self) -> dict:
        return {"items": self._items, "seen": self._seen, "capacity": self.capacity}

    def load_state_dict(self, state: dict) -> None:
        self._items = list(state["items"])
        self._seen = int(state["seen"])
        self.capacity = int(state["capacity"])


class ClassBalancedReplayBuffer:
    """Per-class `ReplayBuffer` with even sampling across classes.

    Useful when a new task introduces a heavy class imbalance — naive
    reservoir sampling would let the new majority class dominate replays
    and cause forgetting in the minority class.
    """

    def __init__(self, capacity_per_class: int = 256, num_classes: int = 2,
                 seed: int = 0):
        self.num_classes = int(num_classes)
        self._buffers = [
            ReplayBuffer(capacity=capacity_per_class, seed=seed + c)
            for c in range(num_classes)
        ]

    def __len__(self) -> int:
        return sum(len(b) for b in self._buffers)

    def add(self, batch: Dict[str, torch.Tensor]) -> None:
        labels = batch["label"]
        for c in range(self.num_classes):
            mask = labels == c
            if not mask.any():
                continue
            sub = {k: v[mask] for k, v in batch.items() if k in _BATCH_KEYS}
            self._buffers[c].add(sub)

    def sample(self, n: int) -> Optional[Dict[str, torch.Tensor]]:
        non_empty = [b for b in self._buffers if len(b) > 0]
        if not non_empty:
            return None
        per_class = max(1, n // len(non_empty))
        chunks = [b.sample(per_class) for b in non_empty]
        chunks = [c for c in chunks if c is not None]
        if not chunks:
            return None
        return {k: torch.cat([c[k] for c in chunks], dim=0) for k in chunks[0]}

    def state_dict(self) -> dict:
        return {"buffers": [b.state_dict() for b in self._buffers],
                "num_classes": self.num_classes}

    def load_state_dict(self, state: dict) -> None:
        self.num_classes = int(state["num_classes"])
        for b, s in zip(self._buffers, state["buffers"]):
            b.load_state_dict(s)
