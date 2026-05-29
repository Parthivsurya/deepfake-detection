"""Evaluation harness for adversarial attacks.

Runs an attack over a dataloader and reports clean accuracy, adversarial
accuracy, attack success rate (ASR), and L2 / L∞ statistics of the
perturbations. ASR is computed only on samples that were *correctly classified
on the clean input* — i.e. it measures the attacker's success against samples
the detector already gets right, which is the standard convention.
"""
from __future__ import annotations
from typing import Dict, Iterable, Optional
import numpy as np
import torch
import torch.nn.functional as F

from .attacks import BaseAttack, build_attack


def _clean_predict(model, frames, audio, has_audio):
    with torch.inference_mode():
        out = model(frames, audio, has_audio=has_audio)
    return out["logits"]


def evaluate_attack(
    model,
    attack: BaseAttack,
    loader: Iterable,
    device: str = "cpu",
    max_batches: Optional[int] = None,
) -> dict:
    model.eval()
    n_total = 0
    n_clean_correct = 0
    n_adv_correct = 0
    n_attacked_succeeded = 0
    n_attacked_total = 0
    l2_all, linf_all = [], []

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        frames = batch["frames"].to(device)
        audio = batch["audio"].to(device)
        has_audio = batch["has_audio"].to(device)
        labels = batch["label"].to(device)

        clean_logits = _clean_predict(model, frames, audio, has_audio)
        clean_pred = clean_logits.argmax(dim=-1)
        clean_correct = (clean_pred == labels)

        result = attack.perturb(frames, audio, has_audio, labels)
        adv_pred = (_clean_predict(model, result.frames_adv, audio, has_audio)
                    .argmax(dim=-1))
        adv_correct = (adv_pred == labels)

        n_total += labels.numel()
        n_clean_correct += int(clean_correct.sum().item())
        n_adv_correct += int(adv_correct.sum().item())
        # ASR conditioned on clean-correct samples
        n_attacked_total += int(clean_correct.sum().item())
        n_attacked_succeeded += int((clean_correct & ~adv_correct).sum().item())

        l2_all.append(result.l2_norm.cpu().numpy())
        linf_all.append(result.linf_norm.cpu().numpy())

    l2 = np.concatenate(l2_all) if l2_all else np.array([])
    linf = np.concatenate(linf_all) if linf_all else np.array([])
    return {
        "attack": attack.name,
        "epsilon": attack.epsilon,
        "n_samples": n_total,
        "clean_accuracy": n_clean_correct / max(n_total, 1),
        "adv_accuracy": n_adv_correct / max(n_total, 1),
        "attack_success_rate": n_attacked_succeeded / max(n_attacked_total, 1),
        "l2_mean": float(l2.mean()) if l2.size else None,
        "l2_max": float(l2.max()) if l2.size else None,
        "linf_mean": float(linf.mean()) if linf.size else None,
        "linf_max": float(linf.max()) if linf.size else None,
    }


def evaluate_all_attacks(
    model,
    loader: Iterable,
    device: str = "cpu",
    epsilon: float = 0.03,
    pgd_steps: int = 10,
    cw_steps: int = 30,
    deepfool_steps: int = 20,
    max_batches: Optional[int] = None,
) -> Dict[str, dict]:
    specs = [
        ("fgsm", {"epsilon": epsilon}),
        ("pgd", {"epsilon": epsilon, "alpha": epsilon / 4, "steps": pgd_steps}),
        ("cw", {"epsilon": epsilon, "steps": cw_steps}),
        ("deepfool", {"epsilon": epsilon, "steps": deepfool_steps}),
    ]
    out: Dict[str, dict] = {}
    for name, kw in specs:
        atk = build_attack(name, model, **kw)
        out[name] = evaluate_attack(model, atk, loader, device=device,
                                    max_batches=max_batches)
    return out
