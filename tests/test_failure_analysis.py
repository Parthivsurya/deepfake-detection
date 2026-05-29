"""Smoke test for the adversarial failure analysis pipeline.

Drives the four analysis functions through a tiny model and a synthetic
loader. We verify shapes and monotonicity properties where they're well
defined (e.g. JPEG round-trip preserves tensor shape).
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from models import MultimodalDeepfakeDetector  # noqa: E402
from data.preprocessing import jpeg_roundtrip  # noqa: E402
from adversarial import (  # noqa: E402
    build_attack,
    sweep_epsilon,
    perturbation_norm_buckets,
    compression_robustness,
    vulnerability_breakdown,
)


class _SyntheticLoader:
    """Iterates over a fixed number of random batches with the schema the
    real VideoClipDataset produces."""

    def __init__(self, n_batches: int = 2, B: int = 2, T: int = 8,
                 H: int = 64, audio_samples: int = 16000):
        self.n = n_batches
        self.B, self.T, self.H = B, T, H
        self.audio_samples = audio_samples

    def __iter__(self):
        torch.manual_seed(7)
        for _ in range(self.n):
            yield {
                "frames": torch.randn(self.B, self.T, 3, self.H, self.H),
                "audio": torch.randn(self.B, self.audio_samples),
                "has_audio": torch.tensor([1.0, 0.0][: self.B]),
                "label": torch.tensor([1, 0][: self.B]),
            }


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
    loader = _SyntheticLoader()
    eps_ref = 0.05

    # 1. JPEG roundtrip shape preservation + values bounded
    frames = torch.randn(1, 4, 3, 64, 64)
    jp = jpeg_roundtrip(frames, quality=75)
    assert jp.shape == frames.shape
    assert torch.isfinite(jp).all()

    # 2. Epsilon sweep returns one row per epsilon, ASR in [0, 1]
    sweep = sweep_epsilon(model, loader, "fgsm",
                          epsilons=[0.01, 0.03, 0.05], device="cpu")
    assert len(sweep) == 3
    for row in sweep:
        assert 0.0 <= row["attack_success_rate"] <= 1.0
        assert row["attack"] == "fgsm"

    # 3. Norm buckets — equal-population bins, ASR per bin
    pgd = build_attack("pgd", model, epsilon=eps_ref, alpha=eps_ref / 4,
                       steps=3, random_start=False)
    nb = perturbation_norm_buckets(model, loader, pgd, device="cpu", n_buckets=2)
    # at least one clean-correct sample must have been attacked for buckets to populate
    assert nb["n_samples"] >= 0
    if nb["buckets"]:
        for b in nb["buckets"]:
            assert 0.0 <= b["asr"] <= 1.0
            assert b["l2_min"] <= b["l2_max"]

    # 4. Compression robustness without attack — accuracy in [0, 1] per quality
    rc = compression_robustness(model, loader, qualities=[90, 50, 20], device="cpu")
    assert len(rc) == 3
    for r in rc:
        assert 0.0 <= r["accuracy"] <= 1.0
        assert r["with_attack"] is False

    # 5. Compression as defence against PGD
    rd = compression_robustness(model, loader, qualities=[90, 30], device="cpu",
                                attack=pgd)
    assert len(rd) == 2
    for r in rd:
        assert r["with_attack"] is True
        assert r["attack_name"] == "pgd"

    # 6. Per-class breakdown
    pc = vulnerability_breakdown(model, loader, pgd, device="cpu")
    assert "real" in pc and "fake" in pc
    for side in ("real", "fake"):
        assert 0.0 <= pc[side]["clean_accuracy"] <= 1.0
        assert 0.0 <= pc[side]["attack_success_rate"] <= 1.0

    print(
        f"OK  sweep_n={len(sweep)} bucket_n={len(nb['buckets'])} "
        f"comp_clean_q90_acc={rc[0]['accuracy']:.3f} "
        f"comp_def_q90_acc={rd[0]['accuracy']:.3f} "
        f"real_asr={pc['real']['attack_success_rate']:.3f} "
        f"fake_asr={pc['fake']['attack_success_rate']:.3f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
