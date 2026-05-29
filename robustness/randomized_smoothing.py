"""Cohen-Rosenfeld-Kolter randomized smoothing (ICML 2019).

Given a base classifier $f: \\mathcal{X} \\to \\mathcal{Y}$, the *smoothed*
classifier is
$$g(x) = \\arg\\max_{c \\in \\mathcal{Y}} \\; \\mathbb{P}_{\\eta \\sim \\mathcal{N}(0, \\sigma^2 I)}\\big[f(x + \\eta) = c\\big].$$

**Theorem (Cohen et al., 2019).** If $\\mathbb{P}[f(x + \\eta) = c_A] \\ge \\underline{p_A}
\\ge \\overline{p_B} \\ge \\mathbb{P}[f(x + \\eta) = c_B]$ for the runner-up class
$c_B$, then $g$ is robust at $x$ for any L2 perturbation of magnitude up to
$$R(x) = \\tfrac{\\sigma}{2}\\big(\\Phi^{-1}(\\underline{p_A}) - \\Phi^{-1}(\\overline{p_B})\\big).$$
With the standard simplification $\\overline{p_B} = 1 - \\underline{p_A}$ this
becomes $R(x) = \\sigma \\,\\Phi^{-1}(\\underline{p_A})$.

We estimate $\\underline{p_A}$ as a one-sided Clopper-Pearson lower bound on
$n_A / n$ at confidence $1 - \\alpha$. The procedure abstains if the lower
bound falls below $1/2$ (no certificate possible).

Smoothing in our pipeline acts on the *frame* tensor — audio passes through
unmodified. This matches the threat model of Module 2 attacks, which perturb
the visual stream only.
"""
from __future__ import annotations
from typing import Optional
import math
import torch
import torch.nn as nn

from scipy.stats import binomtest, norm  # type: ignore


ABSTAIN = -1


def _clopper_pearson_lower(k: int, n: int, alpha: float) -> float:
    """One-sided Clopper-Pearson lower bound on a binomial proportion."""
    if n == 0:
        return 0.0
    if k == 0:
        return 0.0
    return float(binomtest(k, n, p=0.5).proportion_ci(
        confidence_level=1 - 2 * alpha, method="exact").low)


class SmoothedClassifier:
    """Randomized-smoothing wrapper around `MultimodalDeepfakeDetector`-style models.

    The base model is expected to accept `(frames, audio, has_audio=...)` and
    return a dict with key `"logits"`.
    """

    def __init__(
        self,
        base: nn.Module,
        sigma: float,
        num_classes: int = 2,
    ):
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self.base = base
        self.sigma = float(sigma)
        self.num_classes = int(num_classes)

    @torch.inference_mode()
    def _sample_counts(
        self,
        frames: torch.Tensor,
        audio: torch.Tensor,
        has_audio: Optional[torch.Tensor],
        n: int,
        batch_size: int,
    ) -> torch.Tensor:
        device = frames.device
        counts = torch.zeros(self.num_classes, dtype=torch.long, device=device)
        remaining = n
        while remaining > 0:
            b = min(batch_size, remaining)
            f = frames.unsqueeze(0).expand(b, *frames.shape).contiguous()
            a = audio.unsqueeze(0).expand(b, *audio.shape).contiguous()
            ha = None
            if has_audio is not None:
                ha = has_audio.expand(b).contiguous()
            f = f + torch.randn_like(f) * self.sigma
            out = self.base(f, a, has_audio=ha)
            preds = out["logits"].argmax(dim=-1)
            counts.scatter_add_(
                0, preds, torch.ones_like(preds, dtype=torch.long)
            )
            remaining -= b
        return counts

    @torch.inference_mode()
    def predict(
        self,
        frames: torch.Tensor,
        audio: torch.Tensor,
        has_audio: Optional[torch.Tensor] = None,
        n: int = 100,
        batch_size: int = 16,
        alpha: float = 0.001,
    ) -> int:
        """Smoothed prediction with abstention.

        Returns the most-counted class if the top-two-test (Cohen Algorithm 1)
        rejects at level $\\alpha$, else `ABSTAIN`.
        """
        counts = self._sample_counts(frames, audio, has_audio, n, batch_size)
        top2 = torch.topk(counts, 2)
        nA = int(top2.values[0].item())
        nB = int(top2.values[1].item())
        # binomial test: H0 (counts are tied) ~ Binomial(nA+nB, 0.5)
        try:
            p = binomtest(nA, nA + nB, p=0.5).pvalue
        except ValueError:
            p = 1.0
        if p > alpha:
            return ABSTAIN
        return int(top2.indices[0].item())

    @torch.inference_mode()
    def certify(
        self,
        frames: torch.Tensor,
        audio: torch.Tensor,
        has_audio: Optional[torch.Tensor] = None,
        n0: int = 50,
        n: int = 500,
        batch_size: int = 16,
        alpha: float = 0.001,
    ) -> tuple[int, float]:
        """Returns `(predicted_class, certified_radius)` with `(ABSTAIN, 0.0)`
        when no certificate is possible.

        `n0` samples are used to *select* the candidate top class; `n` further
        samples are used to *certify* it (Cohen et al. Algorithm 2). This
        sample-splitting prevents inflated confidence from selection bias.
        """
        counts0 = self._sample_counts(frames, audio, has_audio, n0, batch_size)
        c_hat = int(counts0.argmax().item())
        counts = self._sample_counts(frames, audio, has_audio, n, batch_size)
        nA = int(counts[c_hat].item())
        p_lower = _clopper_pearson_lower(nA, n, alpha)
        if p_lower < 0.5:
            return ABSTAIN, 0.0
        radius = self.sigma * float(norm.ppf(p_lower))
        return c_hat, radius
