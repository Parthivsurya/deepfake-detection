"""Forensic recovery pipeline CLI (Module 2 Task 3).

Runs the full detect → reconstruct → re-verify pipeline over a manifest and
dumps a JSON report with:

  * `trust_scores`         — per-clip 1 − p_perturbation (system confidence)
  * `p_perturbation`       — per-clip adversarial-likelihood score
  * `p_orig_fake`          — detector probability on raw input
  * `p_recon_fake`         — detector probability after diffusion purification
  * `p_final_fake`         — blended probability used for final decision
  * `accuracy_raw`         — accuracy using p_orig
  * `accuracy_recovered`   — accuracy using p_final (the pipeline output)
  * `recon_rate`           — fraction of clips that triggered diffusion

If `--adversarial_attack` is set, the pipeline is exercised on attacked clips
so the recovery numbers reflect real adversarial conditions.

Example:
    python scripts/run_recovery.py \\
        --config configs/default.yaml \\
        --manifest manifests/test.extracted.csv \\
        --ckpt checkpoints/best.pt \\
        --t_star 50 \\
        --recon_threshold 0.5 \\
        --adversarial_attack pgd --epsilon 0.03 \\
        --out results/recovery.json
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
from diffusion import (  # noqa: E402
    DiffusionSchedule,
    SmallUNet,
    DDPM,
    HeuristicPerturbationDetector,
    ForensicRecoveryPipeline,
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


def _calibrate_threshold(detector: HeuristicPerturbationDetector,
                          loader, device: str, max_batches: int = 2,
                          k: float = 2.0) -> None:
    clean = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        clean.append(batch["frames"].to(device))
    if not clean:
        return
    detector.calibrate(torch.cat(clean, dim=0), k=k)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--diffusion_ckpt", default=None,
                   help="optional UNet checkpoint; without one purify() uses an "
                        "untrained denoiser (pipeline still runs end-to-end)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--t_star", type=int, default=50,
                   help="DiffPure noising timestep — larger = more aggressive purification")
    p.add_argument("--recon_threshold", type=float, default=0.5,
                   help="p_perturbation above this triggers reconstruction")
    p.add_argument("--diffusion_T", type=int, default=1000)
    p.add_argument("--diffusion_schedule", default="cosine", choices=["linear", "cosine"])
    p.add_argument("--unet_base_ch", type=int, default=32)
    p.add_argument("--calibrate_k", type=float, default=2.0)
    p.add_argument("--calibrate_batches", type=int, default=2)
    p.add_argument("--adversarial_attack", default=None,
                   choices=[None, "fgsm", "pgd", "cw", "deepfool"],
                   help="optionally attack inputs before recovery to measure defence")
    p.add_argument("--epsilon", type=float, default=0.03)
    p.add_argument("--pgd_steps", type=int, default=10)
    p.add_argument("--max_batches", type=int, default=None)
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

    schedule = DiffusionSchedule(T=args.diffusion_T, schedule=args.diffusion_schedule)
    eps_net = SmallUNet(in_channels=3, base_ch=args.unet_base_ch)
    if args.diffusion_ckpt:
        eps_state = torch.load(args.diffusion_ckpt, map_location=args.device, weights_only=False)
        eps_net.load_state_dict(eps_state["model"] if "model" in eps_state else eps_state)
    diffusion = DDPM(eps_net=eps_net, schedule=schedule).to(args.device)

    pert_detector = HeuristicPerturbationDetector().to(args.device)
    _calibrate_threshold(pert_detector, loader, args.device,
                          max_batches=args.calibrate_batches, k=args.calibrate_k)

    pipeline = ForensicRecoveryPipeline(
        detector=model,
        perturbation_detector=pert_detector,
        diffusion=diffusion,
        recon_threshold=args.recon_threshold,
        t_star=args.t_star,
    )

    attack = None
    if args.adversarial_attack:
        kw = {"alpha": args.epsilon / 4, "steps": args.pgd_steps} \
            if args.adversarial_attack == "pgd" else {}
        attack = build_attack(args.adversarial_attack, model,
                              epsilon=args.epsilon, **kw)

    trust_all, p_pert_all, p_orig_all, p_recon_all, p_final_all = [], [], [], [], []
    labels_all = []
    n_reconstructed = 0
    n_total = 0

    for i, batch in enumerate(loader):
        if args.max_batches is not None and i >= args.max_batches:
            break
        frames = batch["frames"].to(args.device)
        audio = batch["audio"].to(args.device)
        has_audio = batch.get("has_audio")
        if has_audio is not None:
            has_audio = has_audio.to(args.device)
        labels = batch["label"].to(args.device)

        if attack is not None:
            res = attack.perturb(frames, audio, labels, has_audio=has_audio)
            frames = res.frames_adv

        out = pipeline.run(frames, audio, has_audio=has_audio)
        trust_all.append(out.trust_score.cpu())
        p_pert_all.append(out.p_perturbation.cpu())
        p_orig_all.append(out.p_orig_fake.cpu())
        p_recon_all.append(out.p_recon_fake.cpu())
        p_final_all.append(out.p_final_fake.cpu())
        labels_all.append(labels.cpu())
        n_reconstructed += int((out.p_perturbation > args.recon_threshold).sum().item())
        n_total += int(frames.size(0))

    trust = torch.cat(trust_all)
    p_pert = torch.cat(p_pert_all)
    p_orig = torch.cat(p_orig_all)
    p_recon = torch.cat(p_recon_all)
    p_final = torch.cat(p_final_all)
    labels = torch.cat(labels_all)

    pred_raw = (p_orig > 0.5).long()
    pred_rec = (p_final > 0.5).long()
    acc_raw = (pred_raw == labels).float().mean().item()
    acc_rec = (pred_rec == labels).float().mean().item()

    report = {
        "config": args.config,
        "manifest": args.manifest,
        "ckpt": args.ckpt,
        "device": args.device,
        "t_star": args.t_star,
        "recon_threshold": args.recon_threshold,
        "diffusion_schedule": args.diffusion_schedule,
        "diffusion_T": args.diffusion_T,
        "adversarial_attack": args.adversarial_attack,
        "epsilon": args.epsilon if args.adversarial_attack else None,
        "n_samples": n_total,
        "n_reconstructed": n_reconstructed,
        "recon_rate": n_reconstructed / max(n_total, 1),
        "accuracy_raw": acc_raw,
        "accuracy_recovered": acc_rec,
        "trust_score": {
            "mean": float(trust.mean()),
            "min": float(trust.min()),
            "max": float(trust.max()),
        },
        "p_perturbation": {
            "mean": float(p_pert.mean()),
            "min": float(p_pert.min()),
            "max": float(p_pert.max()),
        },
        "p_orig_fake_mean": float(p_orig.mean()),
        "p_recon_fake_mean": float(p_recon.mean()),
        "p_final_fake_mean": float(p_final.mean()),
    }
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
