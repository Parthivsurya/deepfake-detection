"""Smoke test for mathematical robustness modeling (Module 2 Task 5).

Verifies:
  - linear_spectral_norm matches torch.linalg.matrix_norm
  - conv_spectral_norm gives a positive scalar for both Conv1d and Conv2d
  - LipschitzEstimator walks the detector, flags MultiheadAttention as
    unbounded, returns a finite product bound for the bounded subset
  - certified_radius_from_margin clips negatives and scales as 1/L
  - SmoothedClassifier.certify returns (class, radius) or (ABSTAIN, 0)
  - SmoothedClassifier.predict returns a class or ABSTAIN
  - compute_margins returns non-negative margins with correct shapes
  - certified_accuracy_curve is monotonically non-increasing in radius
  - natural_risk / adversarial_risk / risk_decomposition agree and bd ≥ 0
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from models import MultimodalDeepfakeDetector  # noqa: E402
from adversarial import build_attack  # noqa: E402
from robustness import (  # noqa: E402
    linear_spectral_norm,
    conv_spectral_norm,
    LipschitzEstimator,
    certified_radius_from_margin,
    infer_conv_input_shapes,
    SmoothedClassifier,
    ABSTAIN,
    compute_margins,
    certified_accuracy_curve,
    margin_summary,
    natural_risk,
    adversarial_risk,
    risk_decomposition,
)


def _make_model() -> MultimodalDeepfakeDetector:
    return MultimodalDeepfakeDetector(
        image_size=32, patch_size=8, embed_dim=64,
        spatial_depth=1, temporal_depth=1, num_heads=4,
        mlp_ratio=2.0, dropout=0.0, max_frames=4,
        audio_sample_rate=16000, audio_embed_dim=32,
        fusion_dim=64, fusion_depth=1, fusion_heads=4, max_audio_tokens=32,
    )


def _loader(n_batches: int = 2, B: int = 2, T: int = 4, H: int = 32):
    torch.manual_seed(7)
    for _ in range(n_batches):
        yield {
            "frames": torch.randn(B, T, 3, H, H),
            "audio": torch.randn(B, 16000),
            "has_audio": torch.ones(B),
            "label": torch.randint(0, 2, (B,)).long(),
        }


def main() -> int:
    torch.manual_seed(0)

    # 1. linear_spectral_norm == torch top SVD
    W = torch.randn(8, 16)
    ref = float(torch.linalg.matrix_norm(W, ord=2).item())
    got = linear_spectral_norm(W)
    assert math.isclose(got, ref, rel_tol=1e-5), (got, ref)

    # 2. conv_spectral_norm — Conv2d and Conv1d
    conv2d = nn.Conv2d(3, 8, kernel_size=3, padding=1)
    s2 = conv_spectral_norm(conv2d.weight, input_shape=(3, 16, 16),
                             stride=1, padding=1, n_iter=20)
    assert s2 > 0

    conv1d = nn.Conv1d(4, 6, kernel_size=3, padding=1)
    s1 = conv_spectral_norm(conv1d.weight, input_shape=(4, 32),
                             stride=1, padding=1, n_iter=20)
    assert s1 > 0

    # 3. LipschitzEstimator on the real (small) detector
    model = _make_model().eval()
    # capture exact per-conv input shapes by running a forward pass with hooks
    dummy_frames = torch.randn(1, 4, 3, 32, 32)
    dummy_audio = torch.randn(1, 16000)
    dummy_ha = torch.ones(1)
    conv_shapes = infer_conv_input_shapes(
        model, lambda m: m(dummy_frames, dummy_audio, has_audio=dummy_ha)
    )
    assert len(conv_shapes) >= 1
    lip = LipschitzEstimator().estimate(
        model,
        input_shape_by_conv=conv_shapes,
        n_iter=10,
    )
    summary = lip.summary()
    assert summary["n_unbounded"] >= 1, "MHA should be flagged unbounded"
    assert any("attn" in n for n in lip.unbounded_names)
    assert math.isfinite(summary["product_bound_bounded_only"])
    assert summary["product_bound_bounded_only"] > 0

    # 4. certified_radius_from_margin: clamps negatives + scales 1/L
    margins = torch.tensor([-0.5, 0.0, 1.0, 2.0])
    r = certified_radius_from_margin(margins, lipschitz_constant=2.0)
    assert torch.all(r >= 0)
    assert math.isclose(r[2].item(), 1.0 / (math.sqrt(2.0) * 2.0), rel_tol=1e-6)
    assert math.isclose(r[3].item(), 2.0 / (math.sqrt(2.0) * 2.0), rel_tol=1e-6)
    # zero L → zero radii (safe fallback)
    assert torch.all(certified_radius_from_margin(margins, 0.0) == 0)

    # 5. SmoothedClassifier: predict + certify
    smoothed = SmoothedClassifier(base=model, sigma=0.25, num_classes=2)
    frames = torch.randn(4, 3, 32, 32)
    audio = torch.randn(16000)
    has_audio = torch.ones(1)
    pred = smoothed.predict(frames, audio, has_audio=has_audio,
                             n=20, batch_size=4, alpha=0.05)
    assert pred in (0, 1, ABSTAIN)
    c_hat, radius = smoothed.certify(frames, audio, has_audio=has_audio,
                                       n0=10, n=40, batch_size=4, alpha=0.05)
    assert c_hat in (0, 1, ABSTAIN)
    assert radius >= 0.0
    if c_hat != ABSTAIN:
        # sigma * Phi^{-1}(p_lower) — p_lower ≥ 0.5 → radius ≥ 0
        assert radius < 10.0  # sanity bound

    # 6. compute_margins + monotone CA curve
    loader = list(_loader(n_batches=2, B=2, T=4, H=32))
    m_out = compute_margins(model, loader, device="cpu")
    assert m_out["margins"].numel() == 4
    assert torch.all(m_out["margins"] >= 0.0)
    summary_m = margin_summary(m_out["margins"])
    assert summary_m["n"] == 4
    radii = [0.0, 0.05, 0.1, 0.5, 1.0, 10.0]
    curve = certified_accuracy_curve(
        margins=m_out["margins"],
        lipschitz_constant=summary["product_bound_bounded_only"],
        correct=m_out["correct"],
        radii=radii,
    )
    cas = [row["certified_accuracy"] for row in curve]
    for a, b in zip(cas, cas[1:]):
        assert a >= b - 1e-9, f"CA must be non-increasing: {cas}"

    # 7. risk decomposition: bd ≥ 0
    nat = natural_risk(model, _loader(2), device="cpu")
    atk = build_attack("fgsm", model, epsilon=0.05)
    adv = adversarial_risk(model, _loader(2), atk, device="cpu")
    decomp = risk_decomposition(nat, adv)
    assert decomp["boundary_risk"] >= 0
    assert decomp["adversarial_risk"] >= decomp["natural_risk"] - 1e-9

    print(
        f"OK  lin_sn={got:.3f} ref={ref:.3f} "
        f"conv2d_sn={s2:.3f} conv1d_sn={s1:.3f} "
        f"L_bounded={summary['product_bound_bounded_only']:.2e} "
        f"unbounded={summary['n_unbounded']} "
        f"m_mean={summary_m['mean']:.3f} "
        f"smooth=({c_hat},{radius:.3f}) "
        f"nat_R={decomp['natural_risk']:.3f} adv_R={decomp['adversarial_risk']:.3f} "
        f"bd_R={decomp['boundary_risk']:.3f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
