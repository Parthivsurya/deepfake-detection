"""Logit-margin analysis and certified-accuracy curves.

For a $K$-class classifier with logits $z(x)$, the *prediction margin* is
$$m(x) = z_{\\hat y}(x) - \\max_{j \\ne \\hat y} z_j(x),$$
where $\\hat y = \\arg\\max_c z_c(x)$. Combined with a Lipschitz bound (see
`lipschitz.py`), the margin yields a per-sample certified L2 radius
$r(x) = m(x) / (\\sqrt 2 L)$. The empirical certified-accuracy curve
$$\\mathrm{CA}(r) = \\frac{1}{N} \\sum_i \\mathbb{1}\\{r(x_i) \\ge r \\wedge \\hat y_i = y_i\\}$$
is the gold-standard summary of deterministic robustness guarantees.
"""
from __future__ import annotations
from typing import Iterable, List, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.inference_mode()
def compute_margins(
    model: nn.Module,
    loader: Iterable,
    device: str = "cpu",
    max_batches: Optional[int] = None,
) -> dict:
    """Returns `{margins, predictions, labels, correct}` as 1-D CPU tensors."""
    model.eval()
    margins, preds_all, labels_all = [], [], []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        has_audio = batch.get("has_audio")
        if has_audio is not None:
            has_audio = has_audio.to(device)
        logits = model(batch["frames"].to(device),
                        batch["audio"].to(device),
                        has_audio=has_audio)["logits"]
        top2 = torch.topk(logits, k=2, dim=-1)
        # margin is top1 - top2 (always non-negative)
        m = (top2.values[:, 0] - top2.values[:, 1]).detach().cpu()
        margins.append(m)
        preds_all.append(top2.indices[:, 0].detach().cpu())
        labels_all.append(batch["label"].detach().cpu())
    margins = torch.cat(margins) if margins else torch.zeros(0)
    preds = torch.cat(preds_all) if preds_all else torch.zeros(0, dtype=torch.long)
    labels = torch.cat(labels_all) if labels_all else torch.zeros(0, dtype=torch.long)
    return {
        "margins": margins,
        "predictions": preds,
        "labels": labels,
        "correct": (preds == labels),
    }


def certified_accuracy_curve(
    margins: torch.Tensor,
    lipschitz_constant: float,
    correct: torch.Tensor,
    radii: List[float],
) -> List[dict]:
    """Empirical certified accuracy via the Lipschitz–margin bound.

    Returns one row per radius:
        {radius, certified_accuracy, n_certified, n_total}
    `certified_accuracy = #{correct and r(x_i) ≥ radius} / N`.
    """
    if lipschitz_constant <= 0 or not math.isfinite(lipschitz_constant):
        per_sample = torch.zeros_like(margins)
    else:
        per_sample = margins.clamp(min=0.0) / (math.sqrt(2.0) * lipschitz_constant)
    N = max(margins.numel(), 1)
    out = []
    for r in radii:
        certified = (per_sample >= r) & correct
        out.append({
            "radius": float(r),
            "certified_accuracy": float(certified.sum().item()) / N,
            "n_certified": int(certified.sum().item()),
            "n_total": int(margins.numel()),
        })
    return out


def margin_summary(margins: torch.Tensor) -> dict:
    if margins.numel() == 0:
        return {"n": 0, "mean": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0,
                "min": 0.0, "max": 0.0}
    return {
        "n": int(margins.numel()),
        "mean": float(margins.mean()),
        "median": float(margins.median()),
        "p10": float(torch.quantile(margins, 0.10)),
        "p90": float(torch.quantile(margins, 0.90)),
        "min": float(margins.min()),
        "max": float(margins.max()),
    }
