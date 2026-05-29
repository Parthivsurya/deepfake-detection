"""Synthetic-data smoke test.

Runs one forward+backward pass through the full detector with random tensors,
verifies output shapes, and exits non-zero on failure. Use this before/after
any model change — `python -m tests.smoke_test`.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from models import MultimodalDeepfakeDetector  # noqa: E402


def main() -> int:
    torch.manual_seed(0)
    B, T = 2, 8
    H = W = 64
    sr = 16000
    audio_seconds = 1.0
    model = MultimodalDeepfakeDetector(
        image_size=H, patch_size=16, embed_dim=96,
        spatial_depth=2, temporal_depth=2, num_heads=4,
        mlp_ratio=2.0, dropout=0.0, max_frames=T,
        audio_sample_rate=sr, audio_embed_dim=64, fusion_dim=128,
    )
    frames = torch.randn(B, T, 3, H, W)
    waveform = torch.randn(B, int(sr * audio_seconds))
    has_audio = torch.tensor([1.0, 0.0])
    labels = torch.tensor([1, 0])

    out = model(frames, waveform, has_audio=has_audio)
    assert out["logits"].shape == (B, 2), out["logits"].shape
    assert out["sync_score"].shape == (B,), out["sync_score"].shape

    loss = F.cross_entropy(out["logits"], labels) + 0.2 * out["sync_loss"]
    loss.backward()

    grad_ok = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert grad_ok, "no gradients produced"
    print(f"OK  logits={out['logits'].shape}  loss={loss.item():.4f}  "
          f"sync_score={out['sync_score'].tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
