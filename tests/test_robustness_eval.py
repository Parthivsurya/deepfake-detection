"""Smoke test for the unified Module 2 Task 6 evaluation orchestrator.

Builds a tiny detector + a synthetic in-memory loader and exercises every
block of `scripts/robustness_eval.py` directly (no subprocess):

  - _run_detection           — detection metrics + per-clip latencies
  - _run_recovery            — diffusion-pipeline pass on clean and adversarial
  - _run_certified           — Lipschitz product bound, margin CA curve,
                               randomized smoothing on a 2-clip subset
  - risk_decomposition        — natural / adversarial / boundary
  - _headlines               — pulls top-level numbers out of the assembled report
  - render_markdown          — produces a non-empty Markdown summary that
                               mentions every block that ran

Mirrors the style of tests/test_robustness.py and tests/test_attacks.py.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse  # noqa: E402
import torch  # noqa: E402

from models import MultimodalDeepfakeDetector  # noqa: E402
from adversarial import build_attack  # noqa: E402
from diffusion import (  # noqa: E402
    DiffusionSchedule,
    SmallUNet,
    DDPM,
    HeuristicPerturbationDetector,
    ForensicRecoveryPipeline,
)
from robustness import (  # noqa: E402
    natural_risk,
    adversarial_risk,
    risk_decomposition,
)
from utils import render_markdown  # noqa: E402

# Block functions live in scripts/robustness_eval.py — import as a module
import importlib.util  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "robustness_eval",
    Path(__file__).resolve().parents[1] / "scripts" / "robustness_eval.py",
)
_re = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_re)


def _tiny_model() -> MultimodalDeepfakeDetector:
    return MultimodalDeepfakeDetector(
        image_size=32, patch_size=8, embed_dim=64,
        spatial_depth=1, temporal_depth=1, num_heads=4,
        mlp_ratio=2.0, dropout=0.0, max_frames=4,
        audio_sample_rate=16000, audio_embed_dim=32,
        fusion_dim=64, fusion_depth=1, fusion_heads=4, max_audio_tokens=32,
    )


def _synthetic_loader(n_batches: int = 2, B: int = 2, T: int = 4, H: int = 32):
    """List-of-dicts that quacks like a DataLoader (re-iterable)."""
    torch.manual_seed(0)
    return [
        {
            "frames": torch.randn(B, T, 3, H, H),
            "audio": torch.randn(B, 16000),
            "has_audio": torch.ones(B),
            # alternate labels so we have both classes in the eval
            "label": torch.tensor([i % 2 for i in range(B)]).long(),
        }
        for _ in range(n_batches)
    ]


def main() -> int:
    torch.manual_seed(0)
    device = "cpu"
    model = _tiny_model().to(device).eval()
    loader = _synthetic_loader(n_batches=2, B=2)

    # ----- Block 1: detection + latencies ------------------------------
    det = _re._run_detection(model, loader, device, threshold=0.5)
    assert "detection" in det
    assert det["detection"]["n"] == 4
    assert det["latencies_ms"]
    assert all(L > 0 for L in det["latencies_ms"])

    # ----- Block 3: forensic recovery (clean + adversarial) ------------
    schedule = DiffusionSchedule(T=50, schedule="cosine")
    eps_net = SmallUNet(in_channels=3, base_ch=16)
    diffusion = DDPM(eps_net=eps_net, schedule=schedule).to(device)
    pipeline = ForensicRecoveryPipeline(
        detector=model,
        perturbation_detector=HeuristicPerturbationDetector().to(device),
        diffusion=diffusion,
        recon_threshold=0.5,
        t_star=5,
    )
    # calibrate the heuristic so its scores are in a sensible range
    clean_frames = torch.cat([b["frames"] for b in loader], dim=0)
    pipeline.perturbation_detector.calibrate(clean_frames, k=2.0)

    rec_clean = _re._run_recovery(pipeline, loader, device, attack=None,
                                    max_batches=None)
    assert rec_clean["n_samples"] == 4
    assert 0.0 <= rec_clean["accuracy_raw"] <= 1.0
    assert 0.0 <= rec_clean["accuracy_recovered"] <= 1.0
    assert "trust_score" in rec_clean

    attack = build_attack("fgsm", model, epsilon=0.05)
    rec_adv = _re._run_recovery(pipeline, loader, device, attack=attack,
                                  max_batches=None)
    assert rec_adv["n_samples"] == 4
    assert "trust_score" in rec_adv

    # ----- Block 4: certified radius -----------------------------------
    cert_args = argparse.Namespace(
        lipschitz_power_iters=5,
        radii=[0.0, 0.05, 0.5, 5.0],
        max_margin_batches=None,
        sigma=0.25,
        n0_smoothing=5,
        n_smoothing_samples=10,
        smoothing_batch_size=4,
        smoothing_alpha=0.05,
        max_smoothing_clips=2,
    )
    cert = _re._run_certified(model, loader, cert_args, device)
    assert "lipschitz" in cert and "summary" in cert["lipschitz"]
    assert cert["lipschitz"]["summary"]["n_unbounded"] >= 1  # MHA flagged
    assert len(cert["certified_accuracy_lipschitz"]) == 4
    # CA must be monotonically non-increasing in radius
    cas = [row["certified_accuracy"] for row in cert["certified_accuracy_lipschitz"]]
    for a, b in zip(cas, cas[1:]):
        assert a >= b - 1e-9
    assert cert["randomized_smoothing"]["n_clips"] == 2

    # ----- Block 5: risk decomposition --------------------------------
    nat = natural_risk(model, loader, device=device)
    adv = adversarial_risk(model, loader, attack, device=device)
    decomp = risk_decomposition(nat, adv)
    assert decomp["boundary_risk"] >= 0
    assert decomp["adversarial_risk"] >= decomp["natural_risk"] - 1e-9

    # ----- Assemble a report dict + headline numbers -------------------
    report = {
        "config": "configs/default.yaml",
        "manifest": "synthetic",
        "ckpt": "synthetic",
        "device": device,
        "reference_epsilon": 0.05,
        "target_certified_radius": 0.05,
        "detection": {"overall": det["detection"]},
        "realtime": {"n": len(det["latencies_ms"]),
                      "mean_ms": sum(det["latencies_ms"]) / len(det["latencies_ms"]),
                      "fps_effective": 100.0, "realtime_factor": 2.0},
        "recovery": {
            "t_star": 5, "recon_threshold": 0.5,
            "clean": rec_clean, "adversarial": rec_adv,
            "adversarial_attack": "fgsm", "adversarial_epsilon": 0.05,
        },
        "certified": cert,
        "risk": {
            "reference_attack": "fgsm",
            "reference_epsilon": 0.05,
            "decomposition_at_reference": decomp,
            "tradeoff_curve": [
                {"epsilon": 0.0, "natural_risk": nat["risk"],
                 "adversarial_risk": nat["risk"], "boundary_risk": 0.0},
                {"epsilon": 0.05, "natural_risk": nat["risk"],
                 "adversarial_risk": adv["risk"],
                 "boundary_risk": max(0.0, adv["risk"] - nat["risk"])},
            ],
        },
    }
    h = _re._headlines(report, ref_eps=0.05, target_r=0.05)
    assert "clean_accuracy" in h
    assert "mean_trust_score_clean" in h
    assert "recovered_accuracy_adv_ref" in h
    assert "natural_risk" in h
    assert h["reference_epsilon"] == 0.05
    report["headlines"] = h

    # ----- Markdown rendering ------------------------------------------
    md = render_markdown(report)
    for section in [
        "# Robustness evaluation ledger",
        "## Headline metrics",
        "## Detection",
        "## Adversarial",         # comes from "## Adversarial attacks" or fallback
        "## Forensic recovery",
        "## Certified robustness",
        "## Risk decomposition",
    ]:
        # the "attacks" block was skipped (we didn't run evaluate_all_attacks)
        if "Adversarial" in section:
            continue
        assert section in md, f"missing section: {section}\n---\n{md}"
    # _continual was not provided; section should be absent
    assert "Continual learning" not in md

    print(
        f"OK  clean_acc={h.get('clean_accuracy'):.3f} "
        f"acc_raw={rec_clean['accuracy_raw']:.3f} "
        f"acc_rec={rec_clean['accuracy_recovered']:.3f} "
        f"trust_mean_clean={h.get('mean_trust_score_clean'):.3f} "
        f"L_bounded={cert['lipschitz']['summary']['product_bound_bounded_only']:.2e} "
        f"R_nat={decomp['natural_risk']:.3f} R_adv={decomp['adversarial_risk']:.3f} "
        f"md_chars={len(md)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
