"""Online drift detector for the deepfake stream.

Monitors a sliding window of detector outputs and flags when the
distribution of fake-probabilities looks different from a clean reference.
Two signals are fused:

* **Mean shift** — `|mean(window) − ref_mean|` exceeds `k · ref_std`. A fast
  unreliable detector — picks up obvious shifts.
* **PSI (Population Stability Index)** — bucketed distributional distance
  between window and reference. Slower but catches shape changes the mean
  misses.

A drift is reported when *either* signal fires (logical OR). The trigger is
sticky for `cooldown` updates after firing so we don't ping-pong.
"""
from __future__ import annotations
from collections import deque
from typing import Deque, Optional
import math
import torch


def population_stability_index(window: torch.Tensor, reference: torch.Tensor,
                                n_bins: int = 10) -> float:
    """PSI between two 1-D score distributions in [0, 1].

    PSI < 0.1 = stable, 0.1–0.25 = some drift, >0.25 = significant drift.
    """
    edges = torch.linspace(0.0, 1.0, n_bins + 1)
    # tiny epsilon so log(0) is finite
    eps = 1e-6
    w_hist = torch.histc(window.clamp(0, 1), bins=n_bins, min=0.0, max=1.0)
    r_hist = torch.histc(reference.clamp(0, 1), bins=n_bins, min=0.0, max=1.0)
    w = (w_hist / max(w_hist.sum().item(), 1.0)).clamp(min=eps)
    r = (r_hist / max(r_hist.sum().item(), 1.0)).clamp(min=eps)
    psi = ((w - r) * (w / r).log()).sum().item()
    _ = edges  # silence linter — edges are implicit in histc
    return float(psi)


class DriftDetector:
    def __init__(
        self,
        window_size: int = 256,
        ref_mean: Optional[float] = None,
        ref_std: Optional[float] = None,
        reference: Optional[torch.Tensor] = None,
        k_sigma: float = 3.0,
        psi_threshold: float = 0.25,
        cooldown: int = 50,
    ):
        self.window: Deque[float] = deque(maxlen=int(window_size))
        self.reference = reference.detach().clone() if reference is not None else None
        self.ref_mean = float(ref_mean) if ref_mean is not None else (
            float(reference.mean()) if reference is not None else 0.5
        )
        self.ref_std = float(ref_std) if ref_std is not None else (
            float(reference.std()) if reference is not None else 0.25
        )
        self.ref_std = max(self.ref_std, 1e-6)
        self.k_sigma = float(k_sigma)
        self.psi_threshold = float(psi_threshold)
        self.cooldown = int(cooldown)
        self._cooldown_left = 0
        self._last_triggered = False

    def fit_reference(self, scores: torch.Tensor) -> None:
        """Set reference stats from a batch of clean-stream scores."""
        scores = scores.detach().flatten().float().cpu()
        self.reference = scores.clone()
        self.ref_mean = float(scores.mean())
        self.ref_std = max(float(scores.std()), 1e-6)

    def update(self, scores: torch.Tensor) -> None:
        for v in scores.detach().flatten().float().cpu().tolist():
            self.window.append(v)
        if self._cooldown_left > 0:
            self._cooldown_left -= 1

    def stats(self) -> dict:
        if not self.window:
            return {"n": 0, "mean": 0.0, "std": 0.0, "mean_shift_sigma": 0.0,
                    "psi": 0.0}
        w = torch.tensor(list(self.window))
        mean = float(w.mean())
        std = float(w.std(unbiased=False)) if len(w) > 1 else 0.0
        mean_shift = abs(mean - self.ref_mean) / self.ref_std
        psi = (population_stability_index(w, self.reference)
               if self.reference is not None and len(w) >= 4 else 0.0)
        return {"n": len(w), "mean": mean, "std": std,
                "mean_shift_sigma": mean_shift, "psi": psi}

    def is_drifting(self) -> bool:
        if self._cooldown_left > 0:
            return self._last_triggered
        s = self.stats()
        # PSI is undefined / unreliable for very small windows
        psi_fires = s["psi"] > self.psi_threshold and s["n"] >= max(16, self.window.maxlen // 4)
        mean_fires = s["mean_shift_sigma"] > self.k_sigma and s["n"] >= 8
        triggered = bool(psi_fires or mean_fires)
        if triggered:
            self._cooldown_left = self.cooldown
            self._last_triggered = True
        else:
            self._last_triggered = False
        return triggered
