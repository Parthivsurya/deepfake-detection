"""Adversarial failure analysis CLI (Module 2 Task 2).

Combines four diagnostics into one JSON report:

  * `epsilon_sweep`         — ASR vs perturbation budget for PGD (default)
  * `norm_buckets`          — ASR conditioned on perturbation L2 magnitude
  * `compression_clean`     — accuracy as JPEG quality drops (no attack)
  * `compression_defence`   — accuracy when JPEG is applied to PGD adversarials
                              (cheap-defence baseline)
  * `per_class`             — real-vs-fake clean accuracy and ASR breakdown

Example:
    python scripts/failure_analysis.py \\
        --config configs/default.yaml \\
        --manifest manifests/test.extracted.csv \\
        --ckpt checkpoints/best.pt \\
        --epsilons 0.005 0.01 0.02 0.03 0.05 \\
        --jpeg_qualities 95 75 50 25 10 \\
        --out results/failure_analysis.json
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from data.datasets import VideoManifest, VideoClipDataset  # noqa: E402
from models import MultimodalDeepfakeDetector  # noqa: E402
from adversarial import (  # noqa: E402
    build_attack,
    sweep_epsilon,
    perturbation_norm_buckets,
    compression_robustness,
    vulnerability_breakdown,
)


def build_model(cfg: dict) -> MultimodalDeepfakeDetector:
    m, d = cfg["model"], cfg["data"]
    return MultimodalDeepfakeDetector(
        image_size=d["frame_size"],
        patch_size=m["patch_size"],
        embed_dim=m["embed_dim"],
        spatial_depth=m["spatial_depth"],
        temporal_depth=m["temporal_depth"],
        num_heads=m["num_heads"],
        mlp_ratio=m["mlp_ratio"],
        dropout=0.0,
        max_frames=max(d["num_frames"], 64),
        audio_sample_rate=d["audio_sample_rate"],
        audio_embed_dim=m["audio_embed_dim"],
        fusion_dim=m["fusion_dim"],
        fusion_depth=m.get("fusion_depth", 2),
        fusion_heads=m.get("fusion_heads", 8),
        max_audio_tokens=m.get("max_audio_tokens", 256),
    )


def _collate(batch):
    return torch.utils.data.default_collate(
        [{k: v for k, v in b.items() if k != "clip_id"} for b in batch])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--ckpt", required=True)
    default_device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--device", default=default_device)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--attack", default="pgd", choices=["fgsm", "pgd", "cw", "deepfool"],
                   help="attack used for sweeps and breakdowns")
    p.add_argument("--epsilons", nargs="+", type=float,
                   default=[0.005, 0.01, 0.02, 0.03, 0.05])
    p.add_argument("--reference_epsilon", type=float, default=0.03,
                   help="epsilon used for norm-bucket, compression-defence, and per-class analyses")
    p.add_argument("--jpeg_qualities", nargs="+", type=int,
                   default=[95, 75, 50, 25, 10])
    p.add_argument("--pgd_steps", type=int, default=10)
    p.add_argument("--max_batches", type=int, default=None)
    p.add_argument("--n_buckets", type=int, default=4)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    manifest = VideoManifest.load(args.manifest)
    ds = VideoClipDataset(
        manifest,
        num_frames=cfg["data"]["num_frames"],
        frame_size=cfg["data"]["frame_size"],
        audio_sample_rate=cfg["data"]["audio_sample_rate"],
        audio_seconds=cfg["data"]["audio_seconds"],
        training=False,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True,
                        collate_fn=_collate)

    model = build_model(cfg).to(args.device)
    state = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()

    attack_kwargs = {}
    if args.attack == "pgd":
        attack_kwargs = {"steps": args.pgd_steps}

    ref = build_attack(args.attack, model, epsilon=args.reference_epsilon,
                       **({"alpha": args.reference_epsilon / 4, **attack_kwargs}
                          if args.attack == "pgd" else attack_kwargs))

    report = {
        "config": args.config,
        "manifest": args.manifest,
        "ckpt": args.ckpt,
        "device": args.device,
        "attack": args.attack,
        "reference_epsilon": args.reference_epsilon,
        "max_batches": args.max_batches,
        "epsilon_sweep": sweep_epsilon(
            model, loader, args.attack, args.epsilons, device=args.device,
            attack_kwargs=attack_kwargs, max_batches=args.max_batches),
        "norm_buckets": perturbation_norm_buckets(
            model, loader, ref, device=args.device,
            n_buckets=args.n_buckets, max_batches=args.max_batches),
        "compression_clean": compression_robustness(
            model, loader, args.jpeg_qualities, device=args.device,
            attack=None, max_batches=args.max_batches),
        "compression_defence": compression_robustness(
            model, loader, args.jpeg_qualities, device=args.device,
            attack=ref, max_batches=args.max_batches),
        "per_class": vulnerability_breakdown(
            model, loader, ref, device=args.device,
            max_batches=args.max_batches),
    }
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
