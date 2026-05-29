"""Smoke test for the diffusion forensic recovery pipeline (Module 2 Task 3).

Exercises every layer of the stack with synthetic data:
  - DiffusionSchedule.q_sample shape + finite values
  - SmallUNet forward shape preservation with time conditioning
  - DDPM.purify on (B, T, 3, H, W) clips (small t_star to stay fast)
  - HeuristicPerturbationDetector returns (B,) sigmoid scores + calibrate()
  - LearnablePerturbationDetector returns (B,) sigmoid scores
  - ForensicRecoveryPipeline.run returns a RecoveryResult with bounded probs
    and a usable trust score; check both branches (recon triggered vs not).
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from models import MultimodalDeepfakeDetector  # noqa: E402
from diffusion import (  # noqa: E402
    DiffusionSchedule,
    SmallUNet,
    DDPM,
    high_frequency_energy,
    HeuristicPerturbationDetector,
    LearnablePerturbationDetector,
    ForensicRecoveryPipeline,
)


def _make_model() -> MultimodalDeepfakeDetector:
    return MultimodalDeepfakeDetector(
        image_size=32, patch_size=8, embed_dim=64,
        spatial_depth=1, temporal_depth=1, num_heads=4,
        mlp_ratio=2.0, dropout=0.0, max_frames=4,
        audio_sample_rate=16000, audio_embed_dim=32,
        fusion_dim=64, fusion_depth=1, fusion_heads=4, max_audio_tokens=32,
    )


def main() -> int:
    torch.manual_seed(0)
    B, T, H = 2, 4, 32

    # 1. Schedule
    sch = DiffusionSchedule(T=50, schedule="cosine")
    x0 = torch.randn(B, 3, H, H)
    t = torch.randint(0, 50, (B,))
    xt, noise = sch.q_sample(x0, t)
    assert xt.shape == x0.shape and noise.shape == x0.shape
    assert torch.isfinite(xt).all()

    # 2. UNet shape preservation
    unet = SmallUNet(in_channels=3, base_ch=16)
    pred = unet(xt, t)
    assert pred.shape == x0.shape

    # 3. DDPM purify (small t_star — keeps the test fast)
    ddpm = DDPM(eps_net=unet, schedule=sch)
    clip = torch.randn(B, T, 3, H, H)
    purified = ddpm.purify(clip, t_star=3)
    assert purified.shape == clip.shape
    assert torch.isfinite(purified).all()

    # 3b. training_loss returns a scalar
    loss = ddpm.training_loss(x0)
    assert loss.dim() == 0 and torch.isfinite(loss)

    # 4. Heuristic perturbation detector
    hpd = HeuristicPerturbationDetector(threshold=0.0, temperature=0.02)
    score = hpd(clip)
    assert score.shape == (B,)
    assert ((score >= 0) & (score <= 1)).all()

    # 4b. calibrate moves threshold toward mean + k*std
    hpd.calibrate(clip, k=2.0)
    assert torch.isfinite(hpd.threshold)

    # 4c. Laplacian energy sanity — higher-freq noise has higher energy
    smooth = torch.zeros(1, 2, 3, H, H)
    noisy = torch.randn(1, 2, 3, H, H)
    assert high_frequency_energy(noisy).item() > high_frequency_energy(smooth).item()

    # 5. Learnable perturbation detector
    lpd = LearnablePerturbationDetector(in_channels=3, base_ch=8)
    score_l = lpd(clip)
    assert score_l.shape == (B,)
    assert ((score_l >= 0) & (score_l <= 1)).all()

    # 6. ForensicRecoveryPipeline — recon NOT triggered (threshold > 1)
    detector = _make_model().eval()
    pipe_no_recon = ForensicRecoveryPipeline(
        detector=detector,
        perturbation_detector=hpd,
        diffusion=ddpm,
        recon_threshold=1.5,  # impossible to exceed
        t_star=3,
    )
    audio = torch.randn(B, 16000)
    has_audio = torch.tensor([1.0, 1.0])
    out = pipe_no_recon.run(clip, audio, has_audio=has_audio)
    assert out.p_perturbation.shape == (B,)
    assert out.p_orig_fake.shape == (B,)
    assert out.p_recon_fake.shape == (B,)
    assert out.p_final_fake.shape == (B,)
    assert out.trust_score.shape == (B,)
    assert ((out.p_final_fake >= 0) & (out.p_final_fake <= 1)).all()
    assert ((out.trust_score >= 0) & (out.trust_score <= 1)).all()
    assert out.reconstructed is False
    # when nothing is reconstructed, recon should equal orig
    assert torch.allclose(out.p_recon_fake, out.p_orig_fake)

    # 7. Pipeline with recon triggered (threshold = 0)
    pipe_recon = ForensicRecoveryPipeline(
        detector=detector,
        perturbation_detector=hpd,
        diffusion=ddpm,
        recon_threshold=0.0,
        t_star=3,
    )
    out2 = pipe_recon.run(clip, audio, has_audio=has_audio,
                          keep_reconstructed=True)
    assert out2.reconstructed is True
    assert out2.frames_reconstructed is not None
    assert out2.frames_reconstructed.shape == clip.shape

    print(
        f"OK  trust_mean={out.trust_score.mean():.3f} "
        f"p_pert_mean={out.p_perturbation.mean():.3f} "
        f"p_final_mean={out.p_final_fake.mean():.3f} "
        f"recon_ok={out2.reconstructed}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
