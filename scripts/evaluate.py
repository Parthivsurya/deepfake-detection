"""End-to-end evaluation pipeline (Task 6).

Loads a trained checkpoint, runs the test manifest through the detector,
and produces a single JSON report covering:

* overall detection metrics: accuracy, F1, precision, recall, AUC, AP, EER,
  confusion matrix
* per-dataset breakdown (FF++, Celeb-DF, DFDC, FakeAVCeleb) when the
  manifest's `dataset` column contains multiple values
* latency analysis: per-clip forward-pass time (mean / p50 / p95 / p99),
  effective FPS, and realtime factor
* optional pruning + quantization knobs so you can evaluate the deployment
  variant directly

Usage:
    python scripts/evaluate.py \
        --config configs/default.yaml \
        --manifest manifests/test.csv \
        --ckpt checkpoints/best.pt \
        --out results/eval.json

    # Evaluate the optimized deployment variant
    python scripts/evaluate.py ... --prune 0.4 --quantize --device cpu
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from data.datasets import VideoManifest, VideoClipDataset  # noqa: E402
from models import (  # noqa: E402
    MultimodalDeepfakeDetector,
    apply_global_unstructured_pruning,
    apply_dynamic_quantization,
    weight_sparsity,
    parameter_count,
)
from utils import compute_metrics, equal_error_rate, latency_summary  # noqa: E402


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


def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
) -> dict:
    """Return per-clip probs, labels, datasets, and per-batch latencies."""
    model.eval()
    cuda = device.startswith("cuda")
    probs, labels, datasets, sync, clip_ids = [], [], [], [], []
    latencies_ms: list[float] = []
    with torch.inference_mode():
        for batch in loader:
            frames = batch["frames"].to(device, non_blocking=True)
            audio = batch["audio"].to(device, non_blocking=True)
            has_audio = batch["has_audio"].to(device, non_blocking=True)
            if cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model(frames, audio, has_audio=has_audio)
            if cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            # per-clip latency, not per-batch — easier to interpret
            latencies_ms.extend([(t1 - t0) * 1000.0 / frames.size(0)] * frames.size(0))
            p = F.softmax(out["logits"], dim=-1)[:, 1].cpu().numpy()
            probs.append(p)
            labels.append(batch["label"].cpu().numpy())
            sync.append(out["sync_score"].cpu().numpy())
            clip_ids.extend(batch["clip_id"])
            # `dataset` column is optional but useful for per-dataset breakdown
            if "dataset" in batch:
                datasets.extend(batch["dataset"])
    return {
        "probs": np.concatenate(probs) if probs else np.array([]),
        "labels": np.concatenate(labels) if labels else np.array([]),
        "sync_scores": np.concatenate(sync) if sync else np.array([]),
        "clip_ids": clip_ids,
        "datasets": datasets,
        "latencies_ms": latencies_ms,
    }


def evaluate(args) -> dict:
    cfg = yaml.safe_load(open(args.config))
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])

    manifest = VideoManifest.load(args.manifest)
    # carry `dataset` through to the loader for per-dataset slicing
    ds = VideoClipDataset(
        manifest,
        num_frames=cfg["data"]["num_frames"],
        frame_size=cfg["data"]["frame_size"],
        audio_sample_rate=cfg["data"]["audio_sample_rate"],
        audio_seconds=cfg["data"]["audio_seconds"],
        training=False,
    )
    dataset_col = list(manifest.df["dataset"]) if "dataset" in manifest.df.columns else None

    def _collate(batch):
        out = torch.utils.data.default_collate([{k: v for k, v in b.items()
                                                 if k != "clip_id"} for b in batch])
        out["clip_id"] = [b["clip_id"] for b in batch]
        return out

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=_collate,
    )

    model = build_model(cfg).to(args.device)
    state = torch.load(args.ckpt, map_location=args.device)
    model.load_state_dict(state["model"] if "model" in state else state)

    params = parameter_count(model)
    if args.prune > 0:
        apply_global_unstructured_pruning(model, amount=args.prune, make_permanent=True)
    sp = weight_sparsity(model)
    if args.quantize:
        if args.device != "cpu":
            raise SystemExit("--quantize requires --device cpu")
        model = apply_dynamic_quantization(model)

    result = run_inference(model, loader, args.device)

    # attach dataset column from manifest by index since the loader is in order
    if dataset_col and not result["datasets"]:
        result["datasets"] = dataset_col[: len(result["labels"])]

    overall = compute_metrics(result["labels"], result["probs"], threshold=args.threshold)
    overall["eer"] = equal_error_rate(result["labels"], result["probs"])

    per_dataset: dict = {}
    if result["datasets"]:
        names = np.asarray(result["datasets"])
        for name in sorted(set(names.tolist())):
            mask = names == name
            per_dataset[name] = compute_metrics(
                result["labels"][mask], result["probs"][mask], threshold=args.threshold)
            per_dataset[name]["eer"] = equal_error_rate(
                result["labels"][mask], result["probs"][mask])

    lat = latency_summary(result["latencies_ms"])
    T_frames = cfg["data"]["num_frames"]
    if lat.get("mean_ms"):
        lat["fps_effective"] = T_frames / (lat["mean_ms"] / 1000.0)
        lat["clips_per_sec"] = 1000.0 / lat["mean_ms"]
        # frames are sampled at 4 fps in the extractor by default
        clip_seconds = T_frames / 4.0
        lat["realtime_factor"] = clip_seconds / (lat["mean_ms"] / 1000.0)

    report = {
        "config": args.config,
        "manifest": args.manifest,
        "ckpt": args.ckpt,
        "device": args.device,
        "prune_amount": args.prune,
        "quantized": bool(args.quantize),
        "params": params,
        "weight_sparsity": sp,
        "overall": overall,
        "per_dataset": per_dataset,
        "latency": lat,
    }
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--manifest", required=True, help="evaluation manifest CSV")
    p.add_argument("--ckpt", required=True, help="trained model checkpoint")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--prune", type=float, default=0.0)
    p.add_argument("--quantize", action="store_true")
    p.add_argument("--out", default=None, help="optional JSON output path")
    args = p.parse_args()

    report = evaluate(args)
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
