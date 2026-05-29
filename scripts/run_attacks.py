"""Run all four adversarial attacks against a trained detector (Module 2 Task 1).

Loads a checkpoint, builds the test loader, runs FGSM / PGD / CW-L2 / DeepFool
in sequence, and writes a JSON report with clean accuracy, adversarial
accuracy, attack success rate, and perturbation-norm stats per attack.

Example:
    python scripts/run_attacks.py \
        --config configs/default.yaml \
        --manifest manifests/test.extracted.csv \
        --ckpt checkpoints/best.pt \
        --epsilon 0.03 \
        --out results/attacks.json
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
from adversarial import evaluate_all_attacks  # noqa: E402


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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=4,
                   help="attacks need gradients on inputs, so use a smaller batch than eval")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--epsilon", type=float, default=0.03,
                   help="L∞ perturbation budget in normalized input units")
    p.add_argument("--pgd_steps", type=int, default=10)
    p.add_argument("--cw_steps", type=int, default=30)
    p.add_argument("--deepfool_steps", type=int, default=20)
    p.add_argument("--max_batches", type=int, default=None,
                   help="cap number of batches per attack (useful for quick ablations)")
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

    def _collate(batch):
        return torch.utils.data.default_collate(
            [{k: v for k, v in b.items() if k != "clip_id"} for b in batch])

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True,
                        collate_fn=_collate)

    model = build_model(cfg).to(args.device)
    state = torch.load(args.ckpt, map_location=args.device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()

    report = {
        "config": args.config,
        "manifest": args.manifest,
        "ckpt": args.ckpt,
        "device": args.device,
        "epsilon": args.epsilon,
        "attacks": evaluate_all_attacks(
            model, loader, device=args.device, epsilon=args.epsilon,
            pgd_steps=args.pgd_steps, cw_steps=args.cw_steps,
            deepfool_steps=args.deepfool_steps, max_batches=args.max_batches,
        ),
    }
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
