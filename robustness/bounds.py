"""Adversarial risk decomposition and accuracy–robustness trade-off.

Following Tsipras et al. (ICLR 2019) and Madry et al. (ICLR 2018):

  * **Natural risk**         $R_\\mathrm{nat}(f) = \\mathbb{E}[\\,\\mathbb{1}\\{f(x) \\ne y\\}\\,]$
  * **Adversarial risk**     $R_\\mathrm{adv}(f, \\varepsilon) = \\mathbb{E}\\!\\left[\\,\\sup_{\\|\\delta\\| \\le \\varepsilon} \\mathbb{1}\\{f(x+\\delta) \\ne y\\}\\,\\right]$
  * **Boundary risk**        $R_\\mathrm{bd}(f, \\varepsilon) = R_\\mathrm{adv}(f, \\varepsilon) - R_\\mathrm{nat}(f) \\ge 0$

The boundary risk isolates the *robustness gap* — failures that arise only
because the adversary moves a clean-correct sample across the decision
boundary. Tsipras et al. prove that under separated class manifolds it is
possible for $R_\\mathrm{nat}$ to be arbitrarily small while
$R_\\mathrm{adv}$ remains large; minimising one does not guarantee the
other. This module provides Monte-Carlo estimators for these quantities and
a small helper for the accuracy–robustness trade-off curve.
"""
from __future__ import annotations
from typing import Iterable, List, Optional
import torch
import torch.nn as nn


@torch.inference_mode()
def natural_risk(
    model: nn.Module,
    loader: Iterable,
    device: str = "cpu",
    max_batches: Optional[int] = None,
) -> dict:
    model.eval()
    correct, total = 0, 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        has_audio = batch.get("has_audio")
        if has_audio is not None:
            has_audio = has_audio.to(device)
        logits = model(batch["frames"].to(device), batch["audio"].to(device),
                        has_audio=has_audio)["logits"]
        pred = logits.argmax(dim=-1)
        correct += int((pred == batch["label"].to(device)).sum().item())
        total += int(batch["label"].size(0))
    acc = correct / max(total, 1)
    return {"accuracy": acc, "risk": 1.0 - acc, "n_samples": total}


def adversarial_risk(
    model: nn.Module,
    loader: Iterable,
    attack,
    device: str = "cpu",
    max_batches: Optional[int] = None,
) -> dict:
    """Empirical adversarial risk against a concrete `attack` (`BaseAttack`).

    Note that this is a *lower* bound on the true $R_\\mathrm{adv}$ because the
    sup over $\\delta$ is replaced by a single attack's output.
    """
    model.eval()
    correct, total = 0, 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        has_audio = batch.get("has_audio")
        if has_audio is not None:
            has_audio = has_audio.to(device)
        frames = batch["frames"].to(device)
        audio = batch["audio"].to(device)
        labels = batch["label"].to(device)
        res = attack.perturb(frames, audio, has_audio, labels)
        with torch.inference_mode():
            logits = model(res.frames_adv, audio, has_audio=has_audio)["logits"]
        pred = logits.argmax(dim=-1)
        correct += int((pred == labels).sum().item())
        total += int(labels.size(0))
    acc = correct / max(total, 1)
    return {"accuracy": acc, "risk": 1.0 - acc, "n_samples": total}


def risk_decomposition(natural: dict, adversarial: dict) -> dict:
    """`boundary_risk = adv_risk - natural_risk` (clamped to ≥ 0)."""
    nat_r = float(natural["risk"])
    adv_r = float(adversarial["risk"])
    bd_r = max(0.0, adv_r - nat_r)
    return {
        "natural_risk": nat_r,
        "adversarial_risk": adv_r,
        "boundary_risk": bd_r,
        "robustness_gap": adv_r - nat_r,
        "n_samples": int(natural.get("n_samples", 0)),
    }


def accuracy_robustness_tradeoff(
    model: nn.Module,
    loader: Iterable,
    attack_factory,
    epsilons: List[float],
    device: str = "cpu",
    max_batches: Optional[int] = None,
) -> List[dict]:
    """Sweeps $\\varepsilon$ for a *family* of attacks built by `attack_factory(eps)`.

    Returns one row per $\\varepsilon$ with natural risk, adversarial risk, and
    the implied boundary risk. The natural row is included as
    $\\varepsilon = 0$.
    """
    rows = []
    nat = natural_risk(model, loader, device=device, max_batches=max_batches)
    rows.append({"epsilon": 0.0, "natural_risk": nat["risk"],
                  "adversarial_risk": nat["risk"], "boundary_risk": 0.0,
                  "n_samples": nat["n_samples"]})
    for eps in epsilons:
        atk = attack_factory(eps)
        adv = adversarial_risk(model, loader, atk, device=device,
                                max_batches=max_batches)
        rows.append({
            "epsilon": float(eps),
            "natural_risk": nat["risk"],
            "adversarial_risk": adv["risk"],
            "boundary_risk": max(0.0, adv["risk"] - nat["risk"]),
            "n_samples": adv["n_samples"],
        })
    return rows
