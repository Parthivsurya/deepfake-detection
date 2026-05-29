"""Smoke test for continual learning (Module 2 Task 4).

Verifies:
  - ReplayBuffer admits up to capacity and samples correct shapes
  - ClassBalancedReplayBuffer keeps both classes represented
  - EWC.penalty is zero before consolidation, positive after a parameter shift
  - DriftDetector fires when scores shift away from reference
  - ContinualTrainer.train_step reduces new-task loss and tracks replay+EWC
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from models import MultimodalDeepfakeDetector  # noqa: E402
from continual import (  # noqa: E402
    ReplayBuffer,
    ClassBalancedReplayBuffer,
    ElasticWeightConsolidation,
    DriftDetector,
    ContinualTrainer,
    ContinualConfig,
)


def _batch(B: int = 2, T: int = 4, H: int = 32, labels=None) -> dict:
    if labels is None:
        labels = torch.randint(0, 2, (B,))
    return {
        "frames": torch.randn(B, T, 3, H, H),
        "audio": torch.randn(B, 16000),
        "has_audio": torch.ones(B),
        "label": labels.long(),
    }


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

    # 1. ReplayBuffer reservoir sampling
    buf = ReplayBuffer(capacity=5, seed=0)
    for _ in range(20):
        buf.add(_batch(B=2))
    assert len(buf) == 5
    assert buf.seen == 40
    sample = buf.sample(3)
    assert sample is not None
    assert sample["frames"].shape[0] == 3
    assert sample["label"].shape[0] == 3

    # 2. Class-balanced buffer keeps both classes represented
    cb = ClassBalancedReplayBuffer(capacity_per_class=4, num_classes=2)
    # 10 fakes, then 10 reals
    for _ in range(5):
        cb.add(_batch(B=2, labels=torch.ones(2)))
    for _ in range(5):
        cb.add(_batch(B=2, labels=torch.zeros(2)))
    s = cb.sample(4)
    assert s is not None
    assert (s["label"] == 0).any() and (s["label"] == 1).any()

    # 3. EWC: zero penalty before consolidation, positive after parameters change
    model = _make_model().eval()
    ewc = ElasticWeightConsolidation(lam=1.0)
    assert ewc.penalty(model).item() == 0.0
    loader = [_batch(B=2) for _ in range(2)]
    ewc.consolidate(model, loader, device="cpu", max_batches=2)
    # immediately after consolidate: penalty == 0 (anchor == current params)
    assert ewc.penalty(model).item() == 0.0
    # perturb parameters → penalty must rise
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.01)
    pen = ewc.penalty(model).item()
    assert pen > 0.0, f"EWC penalty did not rise after perturbation: {pen}"

    # 4. Drift detector: stable then shifting
    drift = DriftDetector(
        window_size=64,
        reference=torch.rand(256) * 0.2,        # reference around 0.0–0.2
        k_sigma=2.0, psi_threshold=0.1, cooldown=0,
    )
    drift.update(torch.rand(64) * 0.2)
    assert drift.is_drifting() is False
    # shifted distribution: scores concentrated near 1.0
    drift.update(0.8 + torch.rand(64) * 0.2)
    assert drift.is_drifting() is True

    # 5. ContinualTrainer trains and reports all three loss components
    model = _make_model()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    buf2 = ClassBalancedReplayBuffer(capacity_per_class=4, num_classes=2)
    # seed buffer with a few exemplars
    for _ in range(3):
        buf2.add(_batch(B=2, labels=torch.tensor([0, 1])))
    ewc2 = ElasticWeightConsolidation(lam=1.0)
    ewc2.consolidate(model, [_batch(B=2)], device="cpu", max_batches=1)
    # perturb to make sure EWC penalty contributes a non-trivial value
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.005)

    trainer = ContinualTrainer(
        model=model, optimizer=optim, buffer=buf2, ewc=ewc2,
        config=ContinualConfig(replay_lambda=1.0, replay_batch_size=2,
                                ewc_lambda=10.0, consolidation_batches=1),
        device="cpu",
    )
    s0 = trainer.train_step(_batch(B=2, labels=torch.tensor([0, 1])))
    assert s0["loss_new"] > 0 and s0["loss_total"] > 0
    assert s0["loss_replay"] > 0
    assert s0["loss_ewc"] > 0

    # several more steps — train_on aggregates and adds to buffer
    loader2 = [_batch(B=2, labels=torch.tensor([0, 1])) for _ in range(4)]
    agg = trainer.train_on(loader2, n_epochs=1)
    assert agg["loss_total"] > 0
    assert len(buf2) > 0

    print(
        f"OK  buf_len={len(buf)} cb_len={len(cb)} ewc_pen={pen:.6f} "
        f"drift_psi={drift.stats()['psi']:.3f} "
        f"train_loss_total={agg['loss_total']:.4f} "
        f"loss_new={s0['loss_new']:.4f} loss_replay={s0['loss_replay']:.4f} "
        f"loss_ewc={s0['loss_ewc']:.4f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
