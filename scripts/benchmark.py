"""Lightweight inference benchmarking (Task 5).

Builds the detector, optionally applies pruning and/or dynamic quantization,
runs warmup + measurement passes with random inputs (or a real clip), and
reports:

* model size on disk after `state_dict` serialization
* parameter count (total / trainable)
* weight sparsity (after pruning)
* per-window latency: mean, p50, p95
* throughput: clips/sec and effective FPS = T_frames / latency
* "real-time factor" = (clip_seconds) / latency. >1 means realtime-capable.

Examples:
    # FP32 baseline on CPU
    python scripts/benchmark.py --config configs/default.yaml --device cpu

    # Pruned + quantized
    python scripts/benchmark.py --config configs/default.yaml --device cpu \\
        --prune 0.4 --quantize

    # Use a real clip instead of random tensors
    python scripts/benchmark.py --config configs/default.yaml \\
        --video /path/to/clip.mp4 --audio /path/to/clip.wav
"""
from __future__ import annotations
import argparse
import io
import json
import sys
import time
from pathlib import Path
from typing import Optional
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from models import (  # noqa: E402
    MultimodalDeepfakeDetector,
    apply_global_unstructured_pruning,
    apply_dynamic_quantization,
    weight_sparsity,
    parameter_count,
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


def model_size_bytes(model: torch.nn.Module) -> int:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell()


def synth_inputs(cfg: dict, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    d = cfg["data"]
    frames = torch.randn(1, d["num_frames"], 3, d["frame_size"], d["frame_size"], device=device)
    audio = torch.randn(1, int(d["audio_sample_rate"] * d["audio_seconds"]), device=device)
    has_audio = torch.ones(1, device=device)
    return frames, audio, has_audio


def real_inputs(cfg: dict, video: str, audio: Optional[str], device: str):
    """Load one clip (frames + waveform) for representative benchmarking."""
    from data.preprocessing.frame_extraction import _center_crop  # type: ignore
    from utils.video_utils import iter_frames
    from utils.audio_utils import load_waveform

    d = cfg["data"]
    fs = d["frame_size"]
    T = d["num_frames"]
    frames_np = []
    for i, f in enumerate(iter_frames(video, sample_fps=4.0)):
        if len(frames_np) >= T:
            break
        crop = _center_crop(f, fs)
        frames_np.append(crop)
    if len(frames_np) < T:
        frames_np += [frames_np[-1]] * (T - len(frames_np))
    arr = np.stack(frames_np).astype(np.float32) / 255.0
    arr = (arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32)
    frames_t = torch.from_numpy(arr.transpose(0, 3, 1, 2)).unsqueeze(0).to(device)

    if audio:
        w = load_waveform(audio, d["audio_sample_rate"])
        target = int(d["audio_sample_rate"] * d["audio_seconds"])
        w = w[:target] if len(w) >= target else np.pad(w, (0, target - len(w)))
        audio_t = torch.from_numpy(w).unsqueeze(0).to(device)
        has = torch.ones(1, device=device)
    else:
        audio_t = torch.zeros(1, int(d["audio_sample_rate"] * d["audio_seconds"]), device=device)
        has = torch.zeros(1, device=device)
    return frames_t, audio_t, has


def benchmark(
    model: torch.nn.Module,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    iters: int,
    warmup: int,
    device: str,
) -> dict:
    model.eval()
    frames, audio, has = inputs
    cuda = device.startswith("cuda")
    if cuda:
        torch.cuda.synchronize()

    latencies = []
    with torch.inference_mode():
        for i in range(warmup + iters):
            if cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(frames, audio, has_audio=has)
            if cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            if i >= warmup:
                latencies.append((t1 - t0) * 1000.0)
    lat = np.asarray(latencies)
    return {
        "iters": iters,
        "warmup": warmup,
        "latency_ms_mean": float(lat.mean()),
        "latency_ms_p50": float(np.percentile(lat, 50)),
        "latency_ms_p95": float(np.percentile(lat, 95)),
        "latency_ms_std": float(lat.std()),
        "clips_per_sec": float(1000.0 / lat.mean()),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--device", default="cpu", help="cpu or cuda (quantization is CPU-only)")
    p.add_argument("--prune", type=float, default=0.0, help="global unstructured pruning amount in (0,1)")
    p.add_argument("--quantize", action="store_true", help="apply dynamic INT8 quantization")
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--threads", type=int, default=0, help="torch.set_num_threads (0 = leave default)")
    p.add_argument("--ckpt", default=None, help="optional .pt checkpoint to load before benchmarking")
    p.add_argument("--video", default=None, help="optional real video clip for realistic timings")
    p.add_argument("--audio", default=None, help="optional real audio file paired with --video")
    p.add_argument("--out", default=None, help="optional JSON dump path")
    args = p.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    if args.quantize and args.device != "cpu":
        raise SystemExit("dynamic quantization requires --device cpu")

    cfg = yaml.safe_load(open(args.config))
    model = build_model(cfg).to(args.device)
    if args.ckpt:
        state = torch.load(args.ckpt, map_location=args.device, weights_only=False)
        model.load_state_dict(state["model"] if "model" in state else state)

    fp32_params = parameter_count(model)
    fp32_size = model_size_bytes(model)

    if args.prune > 0:
        apply_global_unstructured_pruning(model, amount=args.prune, make_permanent=True)
    sparsity_after = weight_sparsity(model)

    if args.quantize:
        model = apply_dynamic_quantization(model)

    final_size = model_size_bytes(model)

    inputs = (real_inputs(cfg, args.video, args.audio, args.device)
              if args.video else synth_inputs(cfg, args.device))
    metrics = benchmark(model, inputs, iters=args.iters, warmup=args.warmup, device=args.device)

    clip_seconds = cfg["data"]["num_frames"] / 4.0    # frames sampled at 4 fps in extractor
    metrics.update({
        "device": args.device,
        "prune_amount": args.prune,
        "quantized": bool(args.quantize),
        "weight_sparsity": sparsity_after,
        "params_fp32": fp32_params,
        "model_size_fp32_mb": fp32_size / (1024 ** 2),
        "model_size_final_mb": final_size / (1024 ** 2),
        "compression_ratio": fp32_size / max(final_size, 1),
        "fps_effective": cfg["data"]["num_frames"] / (metrics["latency_ms_mean"] / 1000.0),
        "realtime_factor": clip_seconds / (metrics["latency_ms_mean"] / 1000.0),
    })

    print(json.dumps(metrics, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
