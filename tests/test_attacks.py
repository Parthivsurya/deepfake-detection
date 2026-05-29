"""Smoke test for the four adversarial attacks.

We use a tiny detector and synthetic inputs. We don't expect *all* attacks to
succeed on a randomly-initialised model (it's too easy to fool, or sometimes
gradients are degenerate); we only verify the API works end-to-end:

  * each attack returns the expected tensor shapes
  * perturbation L∞ is within the configured epsilon
  * at least one attack flips at least one prediction (basic sanity)
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from models import MultimodalDeepfakeDetector  # noqa: E402
from adversarial import build_attack  # noqa: E402


def _make_model() -> MultimodalDeepfakeDetector:
    return MultimodalDeepfakeDetector(
        image_size=64, patch_size=16, embed_dim=96,
        spatial_depth=2, temporal_depth=2, num_heads=4,
        mlp_ratio=2.0, dropout=0.0, max_frames=8,
        audio_sample_rate=16000, audio_embed_dim=64,
        fusion_dim=128, fusion_depth=1, fusion_heads=4, max_audio_tokens=64,
    )


def main() -> int:
    torch.manual_seed(0)
    model = _make_model().eval()
    B, T = 2, 8
    frames = torch.randn(B, T, 3, 64, 64)
    audio = torch.randn(B, 16000)
    has_audio = torch.tensor([1.0, 0.0])
    labels = torch.tensor([1, 0])

    eps = 0.05
    configs = [
        ("fgsm", {"epsilon": eps}),
        ("pgd", {"epsilon": eps, "alpha": eps / 4, "steps": 4, "random_start": False}),
        ("cw", {"epsilon": eps, "steps": 8, "lr": 0.02, "c": 5.0}),
        ("deepfool", {"epsilon": eps, "steps": 6, "overshoot": 0.02}),
    ]

    any_flip = False
    for name, kw in configs:
        atk = build_attack(name, model, **kw)
        result = atk.perturb(frames, audio, has_audio, labels)
        assert result.frames_adv.shape == frames.shape, f"{name}: bad shape"
        # L∞ budget respected (with a tiny floating-point slack)
        linf = result.linf_norm.max().item()
        assert linf <= eps + 1e-5, f"{name}: linf={linf} exceeds eps={eps}"
        flips = int(result.success.sum().item())
        any_flip = any_flip or flips > 0
        print(f"  {name:<9} linf={linf:.4f} l2_mean={result.l2_norm.mean():.4f} "
              f"flips={flips}/{B}")
    assert any_flip, "no attack flipped any prediction — pipeline likely broken"
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
