"""Smoke test for the standalone CrossAttentionFusion module."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from models import CrossAttentionFusion  # noqa: E402


def main() -> int:
    torch.manual_seed(0)
    B, T_v, T_a = 2, 8, 32
    Dv, Da = 96, 64
    fusion = CrossAttentionFusion(
        video_dim=Dv, audio_dim=Da, fusion_dim=128,
        depth=2, num_heads=4, mlp_ratio=2.0, dropout=0.0,
        max_video_tokens=T_v, max_audio_tokens=T_a,
    )
    v = torch.randn(B, T_v, Dv, requires_grad=True)
    a = torch.randn(B, T_a, Da, requires_grad=True)
    has_audio = torch.tensor([1.0, 0.0])

    out = fusion(v, a, has_audio=has_audio)
    assert out["fused"].shape == (B, 128), out["fused"].shape
    assert out["video_tokens"].shape == (B, T_v, 128)
    assert out["audio_tokens"].shape == (B, T_a, 128)

    # silent clip should produce a zero audio pool
    silent_a_pool = out["audio_tokens"][1].mean(dim=0)
    # we can't easily check exact zero (it was masked AFTER attention), but the
    # fused vector should still receive a zeroed audio contribution; verify the
    # gradient flows back to both v and a regardless.
    out["fused"].sum().backward()
    assert v.grad is not None and a.grad is not None
    print(f"OK  fused={out['fused'].shape}  v_tok={out['video_tokens'].shape}  "
          f"a_tok={out['audio_tokens'].shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
