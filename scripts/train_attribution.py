"""Train the source attribution head on top of a frozen detector backbone.

Pipeline:
    1. Load detector checkpoint -> freeze TemporalViT (+ AudioEncoder, +Physio).
    2. Train SpectralCNN + ResidualCNN + AttributionHead on generator_id labels.
    3. Save best checkpoint by macro-F1 on val.
    4. Fit Mahalanobis OOD scorer on val embeddings and save alongside the model.

Supports:
    * `--resume <ckpt>` — continue from a previous attribution_best.pt.
    * DataParallel across all visible CUDA devices when count > 1.

Usage:
    python scripts/train_attribution.py --config configs/attribution.yaml
"""
from __future__ import annotations
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attribution.dataset import AttributionDataset                       # noqa: E402
from attribution.generators import GENERATOR_REGISTRY, num_known_classes  # noqa: E402
from attribution.losses import SupConLoss                                # noqa: E402
from attribution.model import SourceAttributionModel                     # noqa: E402
from attribution.open_set import MahalanobisScorer                       # noqa: E402
from data.datasets.base import VideoManifest                             # noqa: E402


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


# ----------------------------------------------------------- data
def class_balanced_sampler(df) -> WeightedRandomSampler:
    y = df["generator_id"].values.astype(int)
    counts = np.bincount(y, minlength=num_known_classes())
    w = 1.0 / np.maximum(counts[y], 1)
    return WeightedRandomSampler(weights=w.tolist(), num_samples=len(y), replacement=True)


def build_loaders(cfg: dict):
    train_m = VideoManifest.load(cfg["data"]["manifest_train"])
    val_m = VideoManifest.load(cfg["data"]["manifest_val"])
    common = dict(
        num_frames=cfg["data"]["num_frames"],
        frame_size=cfg["data"]["frame_size"],
        audio_sample_rate=cfg["data"].get("audio_sample_rate", 16000),
        audio_seconds=cfg["data"].get("audio_seconds", 4.0),
        load_audio=cfg["model"].get("use_audio", True),
    )
    train_ds = AttributionDataset(train_m, training=True, **common)
    val_ds = AttributionDataset(val_m, training=False, **common)

    sampler = class_balanced_sampler(train_m.df) if cfg["train"].get("balanced", True) else None
    train_loader = DataLoader(
        train_ds, batch_size=cfg["data"]["batch_size"],
        sampler=sampler, shuffle=(sampler is None),
        num_workers=cfg["data"]["num_workers"], pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["data"]["batch_size"], shuffle=False,
        num_workers=cfg["data"]["num_workers"], pin_memory=True,
    )
    return train_loader, val_loader


# ----------------------------------------------------------- model
def build_model(cfg: dict) -> SourceAttributionModel:
    m = cfg["model"]
    return SourceAttributionModel(
        image_size=cfg["data"]["frame_size"],
        patch_size=m["patch_size"],
        embed_dim=m["embed_dim"],
        spatial_depth=m["spatial_depth"],
        temporal_depth=m["temporal_depth"],
        num_heads=m["num_heads"],
        mlp_ratio=m["mlp_ratio"],
        dropout=m["dropout"],
        max_frames=max(cfg["data"]["num_frames"], 64),
        spectral_dim=m.get("spectral_dim", 256),
        residual_dim=m.get("residual_dim", 256),
        head_hidden=m.get("head_hidden", 384),
        num_classes=num_known_classes(),
        use_audio=m.get("use_audio", True),
        audio_sample_rate=cfg["data"].get("audio_sample_rate", 16000),
        audio_embed_dim=m.get("audio_embed_dim", 256),
        audio_fp_dim=m.get("audio_fp_dim", 256),
        audio_n_mels=m.get("audio_n_mels", 80),
        audio_encoder_kind=m.get("audio_encoder", "cnn"),
        wav2vec_pretrained=m.get("wav2vec_pretrained", "facebook/wav2vec2-base"),
        wav2vec_freeze=m.get("wav2vec_freeze", True),
        use_physio=m.get("use_physio", False),
        physio_embed_dim=m.get("physio_embed_dim", 128),
        physio_fps=m.get("physio_fps", 4.0),
        backbone=m.get("backbone", "temporal_vit"),
    )


def cosine_lr(step: int, total: int, warmup: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return 0.5 * base_lr * (1 + math.cos(math.pi * p))


# ----------------------------------------------------------- eval
def _batch_to_device(model, batch, device):
    inner = _unwrap(model)
    frames = batch["frames"].to(device, non_blocking=True)
    audio = batch["audio"].to(device, non_blocking=True) if inner.use_audio else None
    has_audio = batch["has_audio"].to(device, non_blocking=True) if inner.use_audio else None
    return frames, audio, has_audio


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[dict, torch.Tensor, torch.Tensor]:
    model.eval()
    preds, labels, embeds = [], [], []
    for batch in loader:
        frames, audio, has_audio = _batch_to_device(model, batch, device)
        y = batch["generator_id"]
        out = model(frames, waveform=audio, has_audio=has_audio)
        preds.append(out["logits"].argmax(dim=-1).cpu().numpy())
        labels.append(y.numpy())
        embeds.append(out["embed"].cpu())
    if not preds:
        return {}, torch.empty(0), torch.empty(0)
    p = np.concatenate(preds); y = np.concatenate(labels)
    metrics = {
        "acc": accuracy_score(y, p),
        "f1_macro": f1_score(y, p, average="macro"),
        "per_class_f1": {int(c): float(s) for c, s in
                         zip(sorted(set(y)),
                             f1_score(y, p, labels=sorted(set(y)), average=None))},
    }
    return metrics, torch.cat(embeds, dim=0), torch.from_numpy(y)


# ----------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    default_device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--device", default=default_device)
    p.add_argument("--resume", type=str, default=None,
                   help="path to a previous attribution_best.pt; restores model, "
                        "optimizer, scaler, step counter, and best_f1.")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])

    train_loader, val_loader = build_loaders(cfg)
    model = build_model(cfg).to(args.device)

    # Load and freeze detector backbone BEFORE DataParallel wrapping
    # (load_backbone reads .visual, .audio, .physio attrs).
    backbone_ckpt = cfg["model"]["detector_ckpt"]
    print(f"loading detector backbone: {backbone_ckpt}")
    model.load_backbone(backbone_ckpt, strict=False)

    use_dp = args.device == "cuda" and torch.cuda.device_count() > 1
    if use_dp:
        print(f"DataParallel across {torch.cuda.device_count()} CUDA devices")
        model = nn.DataParallel(model)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in trainable):,}")

    opt = torch.optim.AdamW(trainable, lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])
    supcon_w = float(cfg["train"].get("supcon_weight", 0.0))
    supcon = SupConLoss(temperature=float(cfg["train"].get("supcon_temperature", 0.1)))
    if supcon_w > 0:
        print(f"contrastive provenance loss enabled (weight={supcon_w}, T={supcon.temperature})")
    amp_enabled = cfg["train"]["amp"] and args.device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    total_steps = cfg["train"]["epochs"] * max(len(train_loader), 1)
    warmup = cfg["train"]["warmup_epochs"] * max(len(train_loader), 1)

    ckpt_dir = Path(cfg["train"]["ckpt_dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    step = 0
    start_epoch = 0

    # ---------------- resume ----------------
    if args.resume:
        ckpt_path = Path(args.resume)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"--resume {ckpt_path} does not exist")
        print(f"resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
        _unwrap(model).load_state_dict(ckpt["model"], strict=False)
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt and amp_enabled:
            try:
                scaler.load_state_dict(ckpt["scaler"])
            except Exception as e:
                print(f"WARN: scaler state mismatch ({e}); starting scaler fresh")
        step = int(ckpt.get("step", 0))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_f1 = float(ckpt.get("metrics", {}).get("f1_macro", best_f1))
        print(f"  resumed at epoch {start_epoch}, step {step}, best_f1 {best_f1:.4f}")

    # Gradient accumulation: simulate a larger effective batch using N micro-
    # batches, keeping peak memory low. Set cfg.train.grad_accum_steps > 1 to
    # enable. effective_batch_size = batch_size * grad_accum_steps.
    accum = int(cfg["train"].get("grad_accum_steps", 1))
    if accum > 1:
        print(f"gradient accumulation enabled: {accum} micro-steps per optimizer step "
              f"(effective batch = {cfg['data']['batch_size'] * accum})")

    # ---------------- train ----------------
    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        model.train()
        opt.zero_grad(set_to_none=True)
        micro = 0
        for batch in train_loader:
            frames, audio, has_audio = _batch_to_device(model, batch, args.device)
            y = batch["generator_id"].to(args.device, non_blocking=True)
            lr = cosine_lr(step, total_steps, warmup, cfg["train"]["lr"])
            for g in opt.param_groups:
                g["lr"] = lr
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                out = model(frames, waveform=audio, has_audio=has_audio)
                ce = F.cross_entropy(out["logits"], y,
                                     label_smoothing=cfg["train"].get("label_smoothing", 0.0))
                if supcon_w > 0:
                    cpl = supcon(out["embed"], y)
                    loss = ce + supcon_w * cpl
                else:
                    cpl = ce.new_zeros(())
                    loss = ce
                if accum > 1:
                    loss = loss / accum
            scaler.scale(loss).backward()
            micro += 1
            if micro >= accum:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(trainable, cfg["train"]["grad_clip"])
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)
                micro = 0
            if step % 50 == 0:
                print(f"[e{epoch} s{step}] loss={loss.item() * accum:.4f} ce={ce.item():.4f} "
                      f"cpl={cpl.item():.4f} lr={lr:.2e}")
            step += 1

        metrics, embeds, labels = evaluate(model, val_loader, args.device)
        print(f"[val e{epoch}] acc={metrics['acc']:.4f} macro-f1={metrics['f1_macro']:.4f}")
        for c, s in metrics["per_class_f1"].items():
            info = GENERATOR_REGISTRY.get(c)
            print(f"    cls {c} ({info.name if info else '?'}): f1={s:.3f}")

        f1 = metrics["f1_macro"]
        if f1 > best_f1:
            best_f1 = f1
            scorer = MahalanobisScorer().fit(embeds, labels)
            torch.save({
                "model": _unwrap(model).state_dict(),
                "optimizer": opt.state_dict(),
                "scaler": scaler.state_dict() if amp_enabled else None,
                "cfg": cfg,
                "epoch": epoch,
                "step": step,
                "metrics": metrics,
                "mahalanobis": {
                    "means": scorer.means,
                    "precision": scorer.precision,
                    "classes": scorer.classes,
                },
            }, ckpt_dir / "attribution_best.pt")
            print(f"  saved -> {ckpt_dir/'attribution_best.pt'} (macro-f1={f1:.4f})")


if __name__ == "__main__":
    main()
