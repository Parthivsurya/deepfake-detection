"""Adversarial failure analysis (Module 2 Task 2).

Three orthogonal axes of failure are studied here:

1. **Perturbation budget sweep** — at what ε does the detector start to fail?
   This is the canonical "robustness curve" used in robustness benchmarks.

2. **Perturbation-norm sensitivity** — for a fixed ε, what is the distribution
   of L2 norms of *successful* adversarial perturbations? Sorting samples into
   norm buckets reveals whether the detector is fooled by tiny perturbations
   (a serious vulnerability) or only by large, near-budget ones.

3. **Compression robustness** — accuracy as JPEG quality drops, both with and
   without an adversarial perturbation applied first. Compression destroys
   high-frequency adversarial noise; this is a classic cheap baseline defence.

A "per-class" vulnerability breakdown (real vs fake ASR) is included as a
fourth diagnostic because real vs fake samples often have asymmetric attack
surfaces.
"""
from __future__ import annotations
from typing import Iterable, List, Optional, Sequence
import numpy as np
import torch
import torch.nn.functional as F

from data.preprocessing.compression import jpeg_roundtrip
from .attacks import BaseAttack, build_attack
from .evaluation import evaluate_attack


def _predict(model, frames, audio, has_audio) -> torch.Tensor:
    with torch.inference_mode():
        return model(frames, audio, has_audio=has_audio)["logits"].argmax(dim=-1)


# ---------------------------------------------------------------------------
# 1. Robustness curve: ASR / accuracy as a function of ε
# ---------------------------------------------------------------------------

def sweep_epsilon(
    model,
    loader: Iterable,
    attack_name: str,
    epsilons: Sequence[float],
    device: str = "cpu",
    attack_kwargs: Optional[dict] = None,
    max_batches: Optional[int] = None,
) -> List[dict]:
    """Evaluate `attack_name` at every ε in `epsilons`.

    For PGD we auto-scale `alpha = epsilon / 4` (typical recipe) unless
    overridden via `attack_kwargs`.
    """
    rows: List[dict] = []
    for eps in epsilons:
        kw = dict(attack_kwargs or {})
        kw["epsilon"] = eps
        if attack_name == "pgd" and "alpha" not in kw:
            kw["alpha"] = eps / 4.0
        atk = build_attack(attack_name, model, **kw)
        metrics = evaluate_attack(model, atk, loader, device=device,
                                  max_batches=max_batches)
        rows.append(metrics)
    return rows


# ---------------------------------------------------------------------------
# 2. Perturbation-norm sensitivity
# ---------------------------------------------------------------------------

def perturbation_norm_buckets(
    model,
    loader: Iterable,
    attack: BaseAttack,
    device: str = "cpu",
    n_buckets: int = 4,
    max_batches: Optional[int] = None,
) -> dict:
    """Partition adversarial L2 norms into `n_buckets` equal-population bins and
    report success rate in each. Reveals whether failures cluster at low- or
    high-norm perturbations.
    """
    l2_list, success_list = [], []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        frames = batch["frames"].to(device)
        audio = batch["audio"].to(device)
        has_audio = batch["has_audio"].to(device)
        labels = batch["label"].to(device)
        # only attack samples that are clean-correct (standard convention)
        clean_pred = _predict(model, frames, audio, has_audio)
        keep = (clean_pred == labels)
        if not keep.any():
            continue
        result = attack.perturb(frames[keep], audio[keep], has_audio[keep],
                                labels[keep])
        l2_list.append(result.l2_norm.cpu().numpy())
        success_list.append(result.success.cpu().numpy())

    if not l2_list:
        return {"n_samples": 0, "buckets": []}
    l2 = np.concatenate(l2_list)
    succ = np.concatenate(success_list).astype(bool)
    order = np.argsort(l2)
    l2 = l2[order]; succ = succ[order]
    splits = np.array_split(np.arange(len(l2)), n_buckets)
    buckets = []
    for s in splits:
        if s.size == 0:
            continue
        buckets.append({
            "l2_min": float(l2[s[0]]),
            "l2_max": float(l2[s[-1]]),
            "l2_mean": float(l2[s].mean()),
            "asr": float(succ[s].mean()),
            "n": int(s.size),
        })
    return {
        "n_samples": int(len(l2)),
        "overall_asr": float(succ.mean()),
        "buckets": buckets,
    }


# ---------------------------------------------------------------------------
# 3. JPEG compression robustness (natural corruption + adversarial defence)
# ---------------------------------------------------------------------------

def compression_robustness(
    model,
    loader: Iterable,
    qualities: Sequence[int],
    device: str = "cpu",
    attack: Optional[BaseAttack] = None,
    max_batches: Optional[int] = None,
) -> List[dict]:
    """Accuracy at each JPEG `quality`.

    If `attack` is supplied, the perturbation is applied *before* compression —
    this measures whether JPEG can wash out an adversarial example. If `attack`
    is None, we measure plain natural-corruption robustness.
    """
    rows: List[dict] = []
    for q in qualities:
        n_total = 0
        n_correct = 0
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            frames = batch["frames"].to(device)
            audio = batch["audio"].to(device)
            has_audio = batch["has_audio"].to(device)
            labels = batch["label"].to(device)
            if attack is not None:
                result = attack.perturb(frames, audio, has_audio, labels)
                frames_in = result.frames_adv
            else:
                frames_in = frames
            frames_comp = jpeg_roundtrip(frames_in, quality=q)
            pred = _predict(model, frames_comp, audio, has_audio)
            n_total += labels.numel()
            n_correct += int((pred == labels).sum().item())
        rows.append({
            "quality": int(q),
            "with_attack": attack is not None,
            "attack_name": attack.name if attack is not None else None,
            "epsilon": attack.epsilon if attack is not None else None,
            "n_samples": n_total,
            "accuracy": n_correct / max(n_total, 1),
        })
    return rows


# ---------------------------------------------------------------------------
# 4. Per-class vulnerability breakdown
# ---------------------------------------------------------------------------

def vulnerability_breakdown(
    model,
    loader: Iterable,
    attack: BaseAttack,
    device: str = "cpu",
    max_batches: Optional[int] = None,
) -> dict:
    """Per-class clean accuracy and ASR (real vs fake samples)."""
    stats = {0: {"clean_correct": 0, "n": 0, "attacked_succ": 0, "attacked_n": 0},
             1: {"clean_correct": 0, "n": 0, "attacked_succ": 0, "attacked_n": 0}}
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        frames = batch["frames"].to(device)
        audio = batch["audio"].to(device)
        has_audio = batch["has_audio"].to(device)
        labels = batch["label"].to(device)
        clean_pred = _predict(model, frames, audio, has_audio)
        clean_correct = (clean_pred == labels)
        result = attack.perturb(frames, audio, has_audio, labels)
        adv_pred = _predict(model, result.frames_adv, audio, has_audio)
        adv_correct = (adv_pred == labels)
        for c in (0, 1):
            mask = (labels == c)
            stats[c]["n"] += int(mask.sum().item())
            stats[c]["clean_correct"] += int((mask & clean_correct).sum().item())
            # ASR conditioned on clean-correct
            attackable = mask & clean_correct
            stats[c]["attacked_n"] += int(attackable.sum().item())
            stats[c]["attacked_succ"] += int(
                (attackable & ~adv_correct).sum().item())
    return {
        "attack": attack.name,
        "epsilon": attack.epsilon,
        "real": {
            "n": stats[0]["n"],
            "clean_accuracy": stats[0]["clean_correct"] / max(stats[0]["n"], 1),
            "attack_success_rate": stats[0]["attacked_succ"] /
                                   max(stats[0]["attacked_n"], 1),
        },
        "fake": {
            "n": stats[1]["n"],
            "clean_accuracy": stats[1]["clean_correct"] / max(stats[1]["n"], 1),
            "attack_success_rate": stats[1]["attacked_succ"] /
                                   max(stats[1]["attacked_n"], 1),
        },
    }
