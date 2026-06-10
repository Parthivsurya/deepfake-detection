"""Mathematical robustness analysis CLI (Module 2 Task 5).

Produces a JSON report combining the four robustness quantities from
`docs/robustness.md`:

  * `lipschitz`         — per-layer constants + product bound (bounded subset)
  * `margins`           — empirical margin distribution + summary stats
  * `certified_accuracy_lipschitz`
                        — Tsuzuku-style $r = m / (\\sqrt{2} L)$ curve
  * `randomized_smoothing`
                        — Cohen et al. per-sample $(\\hat c, R)$ certificates
                          on a small subset (smoothing is expensive)
  * `risk_decomposition`
                        — natural / adversarial / boundary risk under one attack

Example:
    python scripts/robustness_analysis.py \\
        --config configs/default.yaml \\
        --manifest manifests/test.extracted.csv \\
        --ckpt checkpoints/best.pt \\
        --attack pgd --epsilon 0.03 \\
        --sigma 0.25 --n_smoothing_samples 500 --max_smoothing_clips 8 \\
        --radii 0.0 0.01 0.05 0.1 0.25 0.5 1.0 \\
        --out results/robustness.json
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
from adversarial import build_attack  # noqa: E402
from robustness import (  # noqa: E402
    LipschitzEstimator,
    SmoothedClassifier,
    ABSTAIN,
    compute_margins,
    certified_accuracy_curve,
    margin_summary,
    natural_risk,
    adversarial_risk,
    risk_decomposition,
    infer_conv_input_shapes,
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
    # Lipschitz
    p.add_argument("--lipschitz_power_iters", type=int, default=30)
    # Margin / certified-accuracy curve
    p.add_argument("--radii", nargs="+", type=float,
                   default=[0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0])
    p.add_argument("--max_margin_batches", type=int, default=None)
    # Smoothing
    p.add_argument("--sigma", type=float, default=0.25)
    p.add_argument("--n0_smoothing", type=int, default=50)
    p.add_argument("--n_smoothing_samples", type=int, default=500)
    p.add_argument("--smoothing_batch_size", type=int, default=16)
    p.add_argument("--smoothing_alpha", type=float, default=0.001)
    p.add_argument("--max_smoothing_clips", type=int, default=8)
    # Risk decomposition
    p.add_argument("--attack", default="pgd",
                   choices=["fgsm", "pgd", "cw", "deepfool"])
    p.add_argument("--epsilon", type=float, default=0.03)
    p.add_argument("--pgd_steps", type=int, default=10)
    p.add_argument("--max_risk_batches", type=int, default=None)
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

    # --- Lipschitz product bound over the bounded subset of the network ---
    # capture each conv's actual input shape with hooks (audio encoder has a
    # Conv1d stack with varying in-channels, so a single default won't work)
    sample_batch = next(iter(loader))
    sample_frames = sample_batch["frames"][:1].to(args.device)
    sample_audio = sample_batch["audio"][:1].to(args.device)
    sample_ha = (sample_batch["has_audio"][:1].to(args.device)
                 if sample_batch.get("has_audio") is not None else None)
    conv_shapes = infer_conv_input_shapes(
        model,
        lambda m: m(sample_frames, sample_audio, has_audio=sample_ha),
    )
    lip = LipschitzEstimator().estimate(
        model,
        input_shape_by_conv=conv_shapes,
        n_iter=args.lipschitz_power_iters,
    )

    # --- Margin distribution + Lipschitz-margin certified-accuracy curve ---
    margins_out = compute_margins(model, loader, device=args.device,
                                    max_batches=args.max_margin_batches)
    L = lip.product_bound
    ca_curve = certified_accuracy_curve(
        margins=margins_out["margins"],
        lipschitz_constant=L,
        correct=margins_out["correct"],
        radii=args.radii,
    )

    # --- Randomized smoothing on a small subset (it's expensive) ---
    smoothed = SmoothedClassifier(base=model, sigma=args.sigma, num_classes=2)
    smoothing_results = []
    n_done = 0
    for batch in loader:
        if n_done >= args.max_smoothing_clips:
            break
        for i in range(batch["frames"].size(0)):
            if n_done >= args.max_smoothing_clips:
                break
            f = batch["frames"][i].to(args.device)
            a = batch["audio"][i].to(args.device)
            ha = (batch["has_audio"][i].to(args.device)
                  if batch.get("has_audio") is not None else None)
            y = int(batch["label"][i].item())
            c_hat, R = smoothed.certify(
                f, a, has_audio=ha.unsqueeze(0) if ha is not None else None,
                n0=args.n0_smoothing, n=args.n_smoothing_samples,
                batch_size=args.smoothing_batch_size, alpha=args.smoothing_alpha,
            )
            smoothing_results.append({
                "label": y,
                "predicted": c_hat,
                "certified_radius": R,
                "abstain": c_hat == ABSTAIN,
                "correct_and_certified": (c_hat == y),
            })
            n_done += 1

    n_smoothed = len(smoothing_results)
    if n_smoothed > 0:
        n_cert = sum(1 for r in smoothing_results
                     if not r["abstain"] and r["correct_and_certified"])
        n_abstain = sum(1 for r in smoothing_results if r["abstain"])
        avg_radius = (sum(r["certified_radius"] for r in smoothing_results
                          if not r["abstain"]) / max(n_smoothed - n_abstain, 1))
    else:
        n_cert = n_abstain = 0
        avg_radius = 0.0

    # --- Risk decomposition under one concrete attack ---
    nat = natural_risk(model, loader, device=args.device,
                       max_batches=args.max_risk_batches)
    atk_kw = {"alpha": args.epsilon / 4, "steps": args.pgd_steps} \
        if args.attack == "pgd" else {}
    attack = build_attack(args.attack, model, epsilon=args.epsilon, **atk_kw)
    adv = adversarial_risk(model, loader, attack, device=args.device,
                            max_batches=args.max_risk_batches)
    decomp = risk_decomposition(nat, adv)

    report = {
        "config": args.config,
        "manifest": args.manifest,
        "ckpt": args.ckpt,
        "device": args.device,
        "lipschitz": {
            "summary": lip.summary(),
            "per_layer": lip.per_layer,
        },
        "margins": margin_summary(margins_out["margins"]),
        "certified_accuracy_lipschitz": ca_curve,
        "randomized_smoothing": {
            "sigma": args.sigma,
            "n0": args.n0_smoothing,
            "n": args.n_smoothing_samples,
            "alpha": args.smoothing_alpha,
            "n_clips": n_smoothed,
            "n_certified_correct": n_cert,
            "n_abstain": n_abstain,
            "mean_certified_radius": avg_radius,
            "samples": smoothing_results,
        },
        "risk_decomposition": {
            "attack": args.attack,
            "epsilon": args.epsilon,
            **decomp,
        },
    }
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
