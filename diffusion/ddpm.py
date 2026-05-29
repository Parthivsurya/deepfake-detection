"""DDPM training loss and `purify()` — partial forward + full reverse.

`purify()` is the DiffPure recipe (Nie et al., 2022): add noise up to a small
timestep `t_star`, then run the reverse process back to t=0. This removes
high-frequency adversarial perturbations while preserving semantic content.
The smaller `t_star`, the closer the output is to the input; larger values
erase more (including some legitimate detail).

The class is agnostic to whether the noise predictor was trained — calling
`purify()` on an untrained UNet will simply return a noisy reconstruction.
The pipeline still runs end-to-end so the rest of Module 2 can be exercised.
"""
from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .schedule import DiffusionSchedule


class DDPM:
    def __init__(self, eps_net: nn.Module, schedule: DiffusionSchedule):
        self.eps_net = eps_net
        self.schedule = schedule

    def to(self, device) -> "DDPM":
        self.eps_net = self.eps_net.to(device)
        self.schedule = self.schedule.to(device)
        return self

    # ------------------------------------------------------------------
    # Training loss (simple ε-prediction MSE — for when the user trains
    # the denoiser; not used by the pipeline at inference time).
    # ------------------------------------------------------------------
    def training_loss(self, x0: torch.Tensor) -> torch.Tensor:
        B = x0.size(0)
        t = torch.randint(0, self.schedule.T, (B,), device=x0.device)
        xt, noise = self.schedule.q_sample(x0, t)
        pred = self.eps_net(xt, t)
        return F.mse_loss(pred, noise)

    # ------------------------------------------------------------------
    # DDPM reverse step
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _reverse_step(self, xt: torch.Tensor, t: int) -> torch.Tensor:
        sch = self.schedule
        ts = torch.full((xt.size(0),), t, device=xt.device, dtype=torch.long)
        eps = self.eps_net(xt, ts)
        beta_t = sch.betas[t]
        alpha_t = sch.alphas[t]
        sqrt_one_minus_abar = sch.sqrt_one_minus_alpha_bars[t]
        mean = (1.0 / torch.sqrt(alpha_t)) * (
            xt - (beta_t / sqrt_one_minus_abar) * eps
        )
        if t == 0:
            return mean
        noise = torch.randn_like(xt)
        sigma = torch.sqrt(beta_t)
        return mean + sigma * noise

    # ------------------------------------------------------------------
    # Forensic purification (DiffPure)
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def purify(self, x: torch.Tensor, t_star: int = 50,
               n_chunks: Optional[int] = None) -> torch.Tensor:
        """Add noise up to timestep `t_star`, then denoise back to t=0.

        For (B, T, 3, H, W) videos we reshape to (B*T, 3, H, W) and purify
        each frame independently. Pass `n_chunks` to split very large batches
        across multiple forward passes (memory control).
        """
        if t_star <= 0 or t_star >= self.schedule.T:
            raise ValueError(f"t_star must be in (0, {self.schedule.T}), got {t_star}")
        orig_shape = x.shape
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.reshape(B * T, C, H, W)
        elif x.dim() != 4:
            raise ValueError(f"expected 4D or 5D input, got {x.dim()}")

        chunks = [x] if n_chunks is None else x.chunk(n_chunks)
        out_chunks = []
        for chunk in chunks:
            t_idx = torch.full((chunk.size(0),), t_star, device=chunk.device, dtype=torch.long)
            xt, _ = self.schedule.q_sample(chunk, t_idx)
            for t in range(t_star, -1, -1):
                xt = self._reverse_step(xt, t)
            out_chunks.append(xt)

        out = torch.cat(out_chunks, dim=0)
        return out.reshape(orig_shape)
