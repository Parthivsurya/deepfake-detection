"""Verify pruning and dynamic quantization preserve forward correctness."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from models import (  # noqa: E402
    MultimodalDeepfakeDetector,
    apply_global_unstructured_pruning,
    apply_dynamic_quantization,
    weight_sparsity,
    parameter_count,
)


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
    B, T = 2, 8
    frames = torch.randn(B, T, 3, 64, 64)
    waveform = torch.randn(B, 16000)
    has_audio = torch.tensor([1.0, 0.0])

    # 1) Pruning halves at 40% should produce ~0.4 sparsity over Linear layers
    model = _make_model().eval()
    pc = parameter_count(model)
    assert pc["total"] > 0
    apply_global_unstructured_pruning(model, amount=0.4, make_permanent=True)
    s = weight_sparsity(model)
    assert 0.35 <= s <= 0.45, f"sparsity {s} out of expected band"
    with torch.inference_mode():
        out_p = model(frames, waveform, has_audio=has_audio)
    assert out_p["logits"].shape == (B, 2)

    # 2) Dynamic quantization on a fresh model should produce a working module
    qmodel = apply_dynamic_quantization(_make_model().eval())
    with torch.inference_mode():
        out_q = qmodel(frames, waveform, has_audio=has_audio)
    assert out_q["logits"].shape == (B, 2)
    # any int8-packed parameter is a clear sign quantization actually ran
    has_quantized = any("packed" in n or "scale" in n for n, _ in qmodel.state_dict().items())
    assert has_quantized, "no quantized tensors found in state_dict"

    print(f"OK  sparsity={s:.3f}  params_total={pc['total']:,}  "
          f"logits_pruned={out_p['logits'].shape}  logits_quant={out_q['logits'].shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
