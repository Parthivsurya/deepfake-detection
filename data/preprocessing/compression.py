"""JPEG compression round-trip for already-normalized frame tensors.

Used by the adversarial failure analysis to (a) measure detector robustness to
natural compression artefacts and (b) test JPEG as a cheap defence against
adversarial perturbations. Both directions are standard in the literature.

The detector consumes ImageNet-normalized frames, so this module first
un-normalizes to [0, 1] pixel space, encodes/decodes as JPEG at the requested
quality, then re-normalizes. The round-trip is exact when `quality=100` for
practical purposes (small floating-point drift only).
"""
from __future__ import annotations
import io
import numpy as np
import torch
from PIL import Image


_MEAN = torch.tensor([0.485, 0.456, 0.406])
_STD = torch.tensor([0.229, 0.224, 0.225])


def _channels_to_xx(t: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    return mean.to(t).view(3, 1, 1), std.to(t).view(3, 1, 1)


def jpeg_roundtrip(frames: torch.Tensor, quality: int) -> torch.Tensor:
    """Apply JPEG compression at the given quality factor.

    Accepts either (B, T, 3, H, W) or (T, 3, H, W) or (3, H, W) tensors
    and returns the same shape. `quality` is in the standard 1–100 range; lower
    means more compression and larger artefacts.
    """
    if quality < 1 or quality > 100:
        raise ValueError(f"quality must be in [1, 100], got {quality}")
    orig_shape = frames.shape
    if frames.dim() == 3:
        frames = frames.unsqueeze(0).unsqueeze(0)
    elif frames.dim() == 4:
        frames = frames.unsqueeze(0)
    elif frames.dim() != 5:
        raise ValueError(f"expected 3/4/5 dims, got {frames.dim()}")

    B, T, C, H, W = frames.shape
    if C != 3:
        raise ValueError(f"expected 3 channels, got {C}")
    mean, std = _channels_to_xx(frames, _MEAN, _STD)
    out = torch.empty_like(frames)
    for b in range(B):
        for t in range(T):
            x01 = (frames[b, t] * std + mean).clamp(0.0, 1.0)
            arr = (x01 * 255.0).byte().permute(1, 2, 0).cpu().numpy()
            pil = Image.fromarray(arr)
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=int(quality))
            buf.seek(0)
            decoded = np.array(Image.open(buf).convert("RGB"))
            t01 = torch.from_numpy(decoded).to(frames).permute(2, 0, 1).float() / 255.0
            out[b, t] = (t01 - mean) / std
    return out.reshape(orig_shape)
