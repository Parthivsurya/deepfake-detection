"""DDPM noise schedule.

Forward process:
    q(x_t | x_0) = N(x_t; sqrt(ᾱ_t) x_0, (1 − ᾱ_t) I)
with ᾱ_t = Π_{s≤t} (1 − β_s).

Reverse posterior mean (used by DDPM sampling):
    μ_t(x_t, ε̂) = (1 / √α_t) (x_t − (β_t / √(1 − ᾱ_t)) · ε̂)

Reverse posterior variance is the simple `β_t` choice from the original paper.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch


def linear_beta_schedule(T: int = 1000, beta_start: float = 1e-4,
                         beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, T)


def cosine_beta_schedule(T: int = 1000, s: float = 0.008) -> torch.Tensor:
    """Nichol & Dhariwal cosine schedule — gentler at high t."""
    steps = T + 1
    t = torch.linspace(0, T, steps) / T
    a_bar = torch.cos((t + s) / (1 + s) * torch.pi / 2) ** 2
    a_bar = a_bar / a_bar[0]
    betas = 1 - (a_bar[1:] / a_bar[:-1])
    return betas.clamp(min=1e-6, max=0.999)


@dataclass
class DiffusionSchedule:
    T: int = 1000
    betas: torch.Tensor = None  # type: ignore
    schedule: str = "linear"

    def __post_init__(self):
        if self.betas is None:
            self.betas = (cosine_beta_schedule(self.T)
                          if self.schedule == "cosine"
                          else linear_beta_schedule(self.T))
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

    def to(self, device) -> "DiffusionSchedule":
        for k in ("betas", "alphas", "alpha_bars",
                  "sqrt_alpha_bars", "sqrt_one_minus_alpha_bars"):
            setattr(self, k, getattr(self, k).to(device))
        return self

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample x_t given x_0. Returns (x_t, noise)."""
        if noise is None:
            noise = torch.randn_like(x0)
        # broadcast t-dependent scalars to image dims
        sab = self.sqrt_alpha_bars[t].view(-1, *([1] * (x0.dim() - 1)))
        somab = self.sqrt_one_minus_alpha_bars[t].view(-1, *([1] * (x0.dim() - 1)))
        return sab * x0 + somab * noise, noise
