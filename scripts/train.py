"""Training entry point.

Loads YAML config, builds dataloaders + model, trains with AMP, evaluates on
the val split, and writes checkpoints. Metrics: accuracy, F1, AUC (Task 6).

Supports:
  * `--resume <ckpt>` — continue training from a previous best.pt with
    optimizer + scaler + step counter restored.
  * DataParallel across all visible CUDA devices when count > 1
    (useful on Kaggle's T4 x2).
"""
from __future__ import annotations
import argparse
import math
import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score  # noqa: E402

from data.datasets import VideoManifest, VideoClipDataset  # noqa: E402
from models import MultimodalDeepfakeDetector  # noqa: E402


def build_loaders(cfg: dict) -> tuple[DataLoader, DataLoader]:
    train_m = VideoManifest.load(cfg["data"]["manifest_train"])
    val_m = VideoManifest.load(cfg["data"]["manifest_val"])
    common = dict(
        num_frames=cfg["data"]["num_frames"],
        frame_size=cfg["data"]["frame_size"],
        audio_sample_rate=cfg["data"]["audio_sample_rate"],
        audio_seconds=cfg["data"]["audio_seconds"],
    )
    train_ds = VideoClipDataset(train_m, training=True, **common)
    val_ds = VideoClipDataset(val_m, training=False, **common)
    dl = lambda ds, shuf: DataLoader(  # noqa: E731
        ds, batch_size=cfg["data"]["batch_size"], shuffle=shuf,
        num_workers=cfg["data"]["num_workers"], pin_memory=True, drop_last=shuf,
    )
    return dl(train_ds, True), dl(val_ds, False)


def build_model(cfg: dict) -> MultimodalDeepfakeDetector:
    m = cfg["model"]
    return MultimodalDeepfakeDetector(
        image_size=cfg["data"]["frame_size"],
        patch_size=m["patch_size"],
        embed_dim=m["embed_dim"],
        spatial_depth=m["spatial_depth"],
        temporal_depth=m["temporal_depth"],
        num_heads=m["num_heads"],
        mlp_ratio=m["mlp_ratio"],
        dropout=m["dropout"],
        max_frames=max(cfg["data"]["num_frames"], 64),
        audio_sample_rate=cfg["data"]["audio_sample_rate"],
        audio_embed_dim=m["audio_embed_dim"],
        fusion_dim=m["fusion_dim"],
        fusion_depth=m.get("fusion_depth", 2),
        fusion_heads=m.get("fusion_heads", 8),
        max_audio_tokens=m.get("max_audio_tokens", 256),
        audio_encoder=m.get("audio_encoder", "cnn"),
        wav2vec_pretrained=m.get("wav2vec_pretrained", "facebook/wav2vec2-base"),
        wav2vec_freeze=m.get("wav2vec_freeze", True),
        use_physio=m.get("use_physio", False),
        physio_embed_dim=m.get("physio_embed_dim", 128),
        physio_fps=m.get("physio_fps", 4.0),
    )


def cosine_lr(step: int, total: int, warmup: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return 0.5 * base_lr * (1 + math.cos(math.pi * p))


def _unwrap(model: nn.Module) -> nn.Module:
    """Return the underlying module if wrapped in DataParallel."""
    return model.module if isinstance(model, nn.DataParallel) else model


def step_batch(model, batch, device, sync_weight: float):
    frames = batch["frames"].to(device, non_blocking=True)
    audio = batch["audio"].to(device, non_blocking=True)
    has_audio = batch["has_audio"].to(device, non_blocking=True)
    labels = batch["label"].to(device, non_blocking=True)
    out = model(frames, audio, has_audio=has_audio)
    ce = F.cross_entropy(out["logits"], labels)
    # DataParallel returns per-replica losses; reduce by mean so backward sees a scalar.
    sync_loss = out["sync_loss"]
    if sync_loss.dim() > 0:
        sync_loss = sync_loss.mean()
    loss = ce + sync_weight * sync_loss
    return loss, ce, sync_loss, out["logits"], labels


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    probs, labels_all = [], []
    for batch in loader:
        _, _, _, logits, labels = step_batch(model, batch, device, sync_weight=0.0)
        probs.append(F.softmax(logits, dim=-1)[:, 1].cpu().numpy())
        labels_all.append(labels.cpu().numpy())
    if not probs:
        return {}
    p = np.concatenate(probs); y = np.concatenate(labels_all)
    metrics = {
        "acc": accuracy_score(y, (p > 0.5).astype(int)),
        "f1": f1_score(y, (p > 0.5).astype(int)),
    }
    if len(np.unique(y)) > 1:
        metrics["auc"] = roc_auc_score(y, p)
    return metrics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    default_device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--device", default=default_device)
    p.add_argument("--resume", type=str, default=None,
                   help="path to a previous checkpoint; restores model, optimizer, "
                        "scaler, step counter, and best_auc.")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])

    train_loader, val_loader = build_loaders(cfg)
    model = build_model(cfg).to(args.device)

    # Multi-GPU on CUDA when device_count > 1 (e.g., Kaggle T4 x2).
    use_dp = args.device == "cuda" and torch.cuda.device_count() > 1
    if use_dp:
        print(f"DataParallel across {torch.cuda.device_count()} CUDA devices")
        model = nn.DataParallel(model)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])
    amp_enabled = cfg["train"]["amp"] and args.device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    total_steps = cfg["train"]["epochs"] * max(len(train_loader), 1)
    warmup = cfg["train"]["warmup_epochs"] * max(len(train_loader), 1)
    sync_w = cfg["train"]["sync_loss_weight"]

    ckpt_dir = Path(cfg["train"]["ckpt_dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_auc = -1.0
    step = 0
    start_epoch = 0

    # ---------------- resume ----------------
    if args.resume:
        ckpt_path = Path(args.resume)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"--resume {ckpt_path} does not exist")
        print(f"resuming from {ckpt_path}")
        # weights_only=False: our own checkpoints embed numpy scalars in the
        # metrics dict; safe because we trust the source.
        ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
        _unwrap(model).load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt and amp_enabled:
            try:
                scaler.load_state_dict(ckpt["scaler"])
            except Exception as e:
                print(f"WARN: scaler state mismatch ({e}); starting scaler fresh")
        step = int(ckpt.get("step", 0))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_auc = float(ckpt.get("metrics", {}).get(
            "auc", ckpt.get("metrics", {}).get("acc", best_auc)))
        print(f"  resumed at epoch {start_epoch}, step {step}, best_auc {best_auc:.4f}")

    # ---------------- train ----------------
    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        model.train()
        for batch in train_loader:
            lr = cosine_lr(step, total_steps, warmup, cfg["train"]["lr"])
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                loss, ce, sl, _, _ = step_batch(model, batch, args.device, sync_w)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            scaler.step(opt); scaler.update()
            if step % 50 == 0:
                print(f"[e{epoch} s{step}] loss={loss.item():.4f} ce={ce.item():.4f} "
                      f"sync={sl.item():.4f} lr={lr:.2e}")
            step += 1

        metrics = evaluate(model, val_loader, args.device)
        print(f"[val e{epoch}] {metrics}")
        auc = metrics.get("auc", metrics.get("acc", 0.0))
        if auc > best_auc:
            best_auc = auc
            torch.save({
                "model": _unwrap(model).state_dict(),
                "optimizer": opt.state_dict(),
                "scaler": scaler.state_dict() if amp_enabled else None,
                "epoch": epoch,
                "step": step,
                "metrics": metrics,
                "cfg": cfg,
            }, ckpt_dir / "best.pt")
            print(f"  saved -> {ckpt_dir/'best.pt'} (auc/acc={auc:.4f})")


if __name__ == "__main__":
    main()
