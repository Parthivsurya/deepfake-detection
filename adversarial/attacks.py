"""Gradient-based adversarial attacks against the multimodal deepfake detector.

All attacks perturb only the *visual* input (the `frames` tensor). Audio passes
through untouched, which mirrors the standard threat model in the video
deepfake literature: an attacker controls pixels at upload time but cannot
synchronously craft a matching audio perturbation. The attack design here is
modular enough that adding an audio attack later is a matter of subclassing
`BaseAttack` and overriding `perturb`.

Inputs are assumed to be already-normalized frames in the model's input space
(ImageNet-style standardization). Therefore `epsilon` is in *normalized* units
and corresponds approximately to `pixel_eps / 0.225` for ImageNet std.

The four attacks implemented (FGSM, PGD, CW-L2, DeepFool) cover the canonical
white-box landscape used in the deepfake-robustness literature:

    FGSM       — single-step L∞ baseline (Goodfellow et al., 2015)
    PGD        — iterative L∞, the de-facto strong attack (Madry et al., 2018)
    CW-L2      — optimization-based, finds minimal-L2 adversarial example
                 (Carlini & Wagner, 2017)
    DeepFool   — iterative linearization, geometric L2 minimal perturbation
                 (Moosavi-Dezfooli et al., 2016)
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AttackResult:
    """Output bundle from a single attack call."""
    frames_adv: torch.Tensor          # (B, T, 3, H, W) perturbed frames
    perturbation: torch.Tensor        # frames_adv - frames
    success: torch.Tensor             # (B,) bool: prediction flipped
    l2_norm: torch.Tensor             # (B,) per-clip L2 of perturbation
    linf_norm: torch.Tensor           # (B,) per-clip L∞ of perturbation


def _model_eval_no_grad(model: nn.Module):
    """Force eval mode (BatchNorm uses running stats) without disabling grad
    on inputs."""
    model.eval()


class BaseAttack:
    """Common scaffolding: forward in eval mode, classify, summarize.

    Subclasses must implement `perturb`.
    """

    name: str = "base"

    def __init__(self, model: nn.Module, epsilon: float = 0.03, targeted: bool = False):
        self.model = model
        self.epsilon = float(epsilon)
        self.targeted = targeted

    # ----- helpers --------------------------------------------------------
    def _forward(self, frames: torch.Tensor, audio: torch.Tensor,
                 has_audio: Optional[torch.Tensor]) -> torch.Tensor:
        return self.model(frames, audio, has_audio=has_audio)["logits"]

    @staticmethod
    def _summarise(frames_clean: torch.Tensor, frames_adv: torch.Tensor,
                   labels: torch.Tensor, logits_adv: torch.Tensor) -> AttackResult:
        delta = frames_adv - frames_clean
        flat = delta.flatten(1)
        pred_adv = logits_adv.argmax(dim=-1)
        return AttackResult(
            frames_adv=frames_adv.detach(),
            perturbation=delta.detach(),
            success=(pred_adv != labels).detach(),
            l2_norm=flat.norm(p=2, dim=1).detach(),
            linf_norm=flat.abs().amax(dim=1).detach(),
        )

    # ----- API ------------------------------------------------------------
    def perturb(
        self,
        frames: torch.Tensor,
        audio: torch.Tensor,
        has_audio: Optional[torch.Tensor],
        labels: torch.Tensor,
    ) -> AttackResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# FGSM
# ---------------------------------------------------------------------------

class FGSM(BaseAttack):
    """Fast Gradient Sign Method (Goodfellow et al., 2015).

    Untargeted update:    x_adv = x + ε · sign(∇_x CE(f(x), y))
    Targeted update:      x_adv = x - ε · sign(∇_x CE(f(x), y_target))
    """
    name = "fgsm"

    def perturb(self, frames, audio, has_audio, labels) -> AttackResult:
        _model_eval_no_grad(self.model)
        x = frames.clone().detach().requires_grad_(True)
        logits = self._forward(x, audio, has_audio)
        loss = F.cross_entropy(logits, labels)
        grad = torch.autograd.grad(loss, x, only_inputs=True)[0]
        sign = grad.sign()
        x_adv = (x + self.epsilon * sign) if not self.targeted else (x - self.epsilon * sign)
        with torch.inference_mode():
            logits_adv = self._forward(x_adv, audio, has_audio)
        return self._summarise(frames, x_adv, labels, logits_adv)


# ---------------------------------------------------------------------------
# PGD
# ---------------------------------------------------------------------------

class PGD(BaseAttack):
    """Projected Gradient Descent (Madry et al., 2018) with L∞ ball.

    Iterative FGSM with step size α and an ε-projection at every step. Starts
    from a uniform-random point inside the ε-ball (true PGD; pass
    `random_start=False` for vanilla iterative FGSM).
    """
    name = "pgd"

    def __init__(self, model: nn.Module, epsilon: float = 0.03, alpha: float = 0.005,
                 steps: int = 10, random_start: bool = True, targeted: bool = False):
        super().__init__(model, epsilon=epsilon, targeted=targeted)
        self.alpha = float(alpha)
        self.steps = int(steps)
        self.random_start = bool(random_start)

    def perturb(self, frames, audio, has_audio, labels) -> AttackResult:
        _model_eval_no_grad(self.model)
        x_clean = frames.clone().detach()
        if self.random_start:
            delta = torch.empty_like(x_clean).uniform_(-self.epsilon, self.epsilon)
        else:
            delta = torch.zeros_like(x_clean)

        for _ in range(self.steps):
            x = (x_clean + delta).detach().requires_grad_(True)
            logits = self._forward(x, audio, has_audio)
            loss = F.cross_entropy(logits, labels)
            grad = torch.autograd.grad(loss, x, only_inputs=True)[0]
            sign = grad.sign() if not self.targeted else -grad.sign()
            delta = (delta + self.alpha * sign).clamp_(-self.epsilon, self.epsilon).detach()

        x_adv = (x_clean + delta).detach()
        with torch.inference_mode():
            logits_adv = self._forward(x_adv, audio, has_audio)
        return self._summarise(frames, x_adv, labels, logits_adv)


# ---------------------------------------------------------------------------
# Carlini & Wagner — L2
# ---------------------------------------------------------------------------

class CarliniWagnerL2(BaseAttack):
    """Optimization-based L2 attack (Carlini & Wagner, 2017).

    For untargeted attacks, minimises
        ||δ||_2^2  +  c · max(z_true − max_{i≠true} z_i + κ, 0)

    where z are logits and κ is the confidence margin. Optimised with Adam on
    the raw perturbation tensor δ; we do not use the tanh change of variables
    because our inputs are already-standardised (no [0,1] box constraint).

    `c` defaults to 1.0; the original paper does a binary search over c but
    that is expensive — a fixed c is fine for benchmarking robustness here.
    """
    name = "cw"

    def __init__(self, model: nn.Module, epsilon: float = 0.03, steps: int = 50,
                 lr: float = 0.01, c: float = 1.0, kappa: float = 0.0,
                 targeted: bool = False):
        super().__init__(model, epsilon=epsilon, targeted=targeted)
        self.steps = int(steps)
        self.lr = float(lr)
        self.c = float(c)
        self.kappa = float(kappa)

    @staticmethod
    def _cw_loss(logits: torch.Tensor, labels: torch.Tensor, kappa: float,
                 targeted: bool) -> torch.Tensor:
        B, C = logits.shape
        true = logits.gather(1, labels.view(-1, 1)).squeeze(1)
        # mask the true class with -inf so the argmax picks the runner-up
        mask = torch.full_like(logits, float("-inf")).scatter_(
            1, labels.view(-1, 1), 0.0)
        other = (logits + mask).amax(dim=1)
        # rest of the row plus -inf at the true class -> still the true class
        # if the true class isn't the argmax. Recover the proper "other-class"
        # logit by zero-ing only the true position:
        masked = logits.clone()
        masked.scatter_(1, labels.view(-1, 1), float("-inf"))
        other = masked.amax(dim=1)
        if targeted:
            # we want logits[target] - max_{i!=target} logits[i] > kappa
            return torch.clamp(other - true + kappa, min=0.0)
        # untargeted: push true class below the runner-up by kappa
        return torch.clamp(true - other + kappa, min=0.0)

    def perturb(self, frames, audio, has_audio, labels) -> AttackResult:
        _model_eval_no_grad(self.model)
        x_clean = frames.clone().detach()
        delta = torch.zeros_like(x_clean, requires_grad=True)
        opt = torch.optim.Adam([delta], lr=self.lr)

        for _ in range(self.steps):
            x = x_clean + delta
            logits = self._forward(x, audio, has_audio)
            l2 = delta.flatten(1).pow(2).sum(dim=1)
            adv = self._cw_loss(logits, labels, self.kappa, self.targeted)
            loss = (l2 + self.c * adv).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            # soft project to the ε-L∞ ball so reported norms stay bounded
            with torch.no_grad():
                delta.clamp_(-self.epsilon, self.epsilon)

        x_adv = (x_clean + delta).detach()
        with torch.inference_mode():
            logits_adv = self._forward(x_adv, audio, has_audio)
        return self._summarise(frames, x_adv, labels, logits_adv)


# ---------------------------------------------------------------------------
# DeepFool (binary)
# ---------------------------------------------------------------------------

class DeepFool(BaseAttack):
    """Iterative linearization attack (Moosavi-Dezfooli et al., 2016).

    Specialised to binary classification: at every iteration we linearize the
    logit difference around the current point and step toward the decision
    boundary along that direction. After a `1+overshoot` factor is applied so
    the perturbation lands just past the boundary.

    Bounded by `epsilon` (L∞) so the attack is comparable to FGSM/PGD even
    though DeepFool is natively L2-minimal.
    """
    name = "deepfool"

    def __init__(self, model: nn.Module, epsilon: float = 0.03, steps: int = 20,
                 overshoot: float = 0.02):
        super().__init__(model, epsilon=epsilon, targeted=False)
        self.steps = int(steps)
        self.overshoot = float(overshoot)

    def perturb(self, frames, audio, has_audio, labels) -> AttackResult:
        _model_eval_no_grad(self.model)
        x_clean = frames.clone().detach()
        x_adv = x_clean.clone().detach()

        for _ in range(self.steps):
            x = x_adv.detach().requires_grad_(True)
            logits = self._forward(x, audio, has_audio)
            pred = logits.argmax(dim=-1)
            # stop early when every sample has already flipped
            still_correct = (pred == labels)
            if not still_correct.any():
                break
            # we treat the problem as binary: score = z[1] - z[0]; we want sign flip
            # gradient w.r.t. input of (z_other - z_true)
            other = 1 - labels
            score = logits.gather(1, other.view(-1, 1)).squeeze(1) - \
                    logits.gather(1, labels.view(-1, 1)).squeeze(1)
            grad = torch.autograd.grad(score.sum(), x, only_inputs=True)[0]
            flat_g = grad.flatten(1)
            denom = flat_g.pow(2).sum(dim=1).clamp_min(1e-12)
            # closed-form step for a linearized binary classifier:
            #   r = ((z_true - z_other) / ||grad||²) * grad
            r = ((-score) / denom).view(-1, *([1] * (grad.dim() - 1))) * grad
            r = r * (1.0 + self.overshoot)
            # only update samples that haven't flipped yet
            update_mask = still_correct.view(-1, *([1] * (grad.dim() - 1))).float()
            x_adv = (x_adv + update_mask * r).detach()
            # project to the ε-L∞ ball around the clean input
            delta = (x_adv - x_clean).clamp_(-self.epsilon, self.epsilon)
            x_adv = (x_clean + delta).detach()

        with torch.inference_mode():
            logits_adv = self._forward(x_adv, audio, has_audio)
        return self._summarise(frames, x_adv, labels, logits_adv)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_REGISTRY = {
    "fgsm": FGSM,
    "pgd": PGD,
    "cw": CarliniWagnerL2,
    "deepfool": DeepFool,
}


def build_attack(name: str, model: nn.Module, **kwargs) -> BaseAttack:
    """Construct an attack by short name (`fgsm`, `pgd`, `cw`, `deepfool`)."""
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"unknown attack {name!r}; available: {list(_REGISTRY)}")
    return _REGISTRY[key](model, **kwargs)
