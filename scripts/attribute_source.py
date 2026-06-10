"""Attribute a video to its source generator.

Usage (single video):
    python scripts/attribute_source.py \
        --ckpt   checkpoints/attribution_best.pt \
        --video  path/to/clip.mp4 \
        --topk   3

Usage (extracted frames dir):
    python scripts/attribute_source.py --ckpt ... --frames-dir frames/<clip_id>/

Output: ranked generators with softmax confidence, family tag, and OOD score.
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attribution.generators import GENERATOR_REGISTRY                    # noqa: E402
from attribution.model import SourceAttributionModel                     # noqa: E402
from attribution.open_set import EnergyScorer, MahalanobisScorer         # noqa: E402
from data.datasets.base import _IMG_MEAN, _IMG_STD                       # noqa: E402


def load_audio_from_path(path: Path, sr: int, seconds: float) -> tuple[torch.Tensor, int]:
    """Best-effort audio loader: returns (waveform (1, S), has_audio 0/1)."""
    target = int(sr * seconds)
    try:
        from data.preprocessing.audio_extraction import load_audio_clip
        wav = load_audio_clip(str(path), sr, seconds)
        return torch.from_numpy(wav).unsqueeze(0), 1
    except Exception:
        return torch.zeros(1, target), 0


# ----------------------------------------------------------- I/O
def load_frames_from_dir(d: Path, num_frames: int, frame_size: int) -> torch.Tensor:
    paths = sorted(d.glob("frame_*.jpg"))
    if not paths:
        raise FileNotFoundError(f"no frame_*.jpg found in {d}")
    if len(paths) >= num_frames:
        step = len(paths) / num_frames
        idx = [int(math.floor(i * step)) for i in range(num_frames)]
    else:
        idx = list(range(len(paths))) + [len(paths) - 1] * (num_frames - len(paths))
    out = np.empty((num_frames, 3, frame_size, frame_size), dtype=np.float32)
    for i, j in enumerate(idx):
        img = Image.open(paths[j]).convert("RGB").resize((frame_size, frame_size), Image.BILINEAR)
        arr = (np.asarray(img, dtype=np.float32) / 255.0 - _IMG_MEAN) / _IMG_STD
        out[i] = arr.transpose(2, 0, 1)
    return torch.from_numpy(out).unsqueeze(0)              # (1, T, 3, H, W)


def load_frames_from_video(path: Path, num_frames: int, frame_size: int) -> torch.Tensor:
    try:
        import imageio.v3 as iio
    except ImportError as e:
        raise RuntimeError("install imageio[ffmpeg] for direct video input") from e
    arr = iio.imread(path, plugin="pyav")                  # (N, H, W, 3)
    n = arr.shape[0]
    if n >= num_frames:
        step = n / num_frames
        idx = [int(math.floor(i * step)) for i in range(num_frames)]
    else:
        idx = list(range(n)) + [n - 1] * (num_frames - n)
    out = np.empty((num_frames, 3, frame_size, frame_size), dtype=np.float32)
    for i, j in enumerate(idx):
        img = Image.fromarray(arr[j]).convert("RGB").resize((frame_size, frame_size), Image.BILINEAR)
        a = (np.asarray(img, dtype=np.float32) / 255.0 - _IMG_MEAN) / _IMG_STD
        out[i] = a.transpose(2, 0, 1)
    return torch.from_numpy(out).unsqueeze(0)


# ----------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--video", type=str, help="path to mp4")
    g.add_argument("--frames-dir", type=str, help="directory with frame_*.jpg")
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--frame-size", type=int, default=224)
    p.add_argument("--audio", type=str, default=None,
                   help="optional waveform path; if omitted and --video given, audio "
                        "is loaded from the video file")
    p.add_argument("--audio-seconds", type=float, default=4.0)
    p.add_argument("--json", action="store_true", help="output a single JSON line")
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    use_audio = cfg["model"].get("use_audio", True)
    model = SourceAttributionModel(
        image_size=cfg["data"]["frame_size"],
        patch_size=cfg["model"]["patch_size"],
        embed_dim=cfg["model"]["embed_dim"],
        spatial_depth=cfg["model"]["spatial_depth"],
        temporal_depth=cfg["model"]["temporal_depth"],
        num_heads=cfg["model"]["num_heads"],
        mlp_ratio=cfg["model"]["mlp_ratio"],
        dropout=0.0,
        max_frames=max(cfg["data"]["num_frames"], 64),
        spectral_dim=cfg["model"].get("spectral_dim", 256),
        residual_dim=cfg["model"].get("residual_dim", 256),
        head_hidden=cfg["model"].get("head_hidden", 384),
        use_audio=use_audio,
        audio_sample_rate=cfg["data"].get("audio_sample_rate", 16000),
        audio_embed_dim=cfg["model"].get("audio_embed_dim", 256),
        audio_fp_dim=cfg["model"].get("audio_fp_dim", 256),
        audio_n_mels=cfg["model"].get("audio_n_mels", 80),
    )
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval().to(args.device)

    if args.video:
        frames = load_frames_from_video(Path(args.video), args.num_frames, args.frame_size)
    else:
        frames = load_frames_from_dir(Path(args.frames_dir), args.num_frames, args.frame_size)
    frames = frames.to(args.device)

    audio = None
    has_audio = None
    if use_audio:
        sr = cfg["data"].get("audio_sample_rate", 16000)
        src = args.audio or args.video
        if src is None:
            audio = torch.zeros(1, int(sr * args.audio_seconds))
            has_audio = torch.zeros(1)
        else:
            audio, ha = load_audio_from_path(Path(src), sr, args.audio_seconds)
            has_audio = torch.tensor([ha], dtype=torch.float32)
        audio = audio.to(args.device)
        has_audio = has_audio.to(args.device)

    with torch.no_grad():
        out = model(frames, waveform=audio, has_audio=has_audio)

    probs = F.softmax(out["logits"], dim=-1).squeeze(0).cpu()
    energy = EnergyScorer.score(out["logits"]).item()
    ood = None
    if "mahalanobis" in ckpt:
        scorer = MahalanobisScorer()
        scorer.means = ckpt["mahalanobis"]["means"]
        scorer.precision = ckpt["mahalanobis"]["precision"]
        scorer.classes = ckpt["mahalanobis"]["classes"]
        ood = scorer.score(out["embed"].cpu()).item()

    order = probs.argsort(descending=True).tolist()
    ranked = []
    for c in order[:args.topk]:
        info = GENERATOR_REGISTRY.get(int(c))
        ranked.append({
            "rank": len(ranked) + 1,
            "generator_id": int(c),
            "name": info.name if info else f"id{c}",
            "family": info.family if info else "unknown",
            "confidence": float(probs[c]),
        })

    result = {
        "input": args.video or args.frames_dir,
        "top_prediction": ranked[0],
        "ranked": ranked,
        "energy_ood": float(energy),
        "mahalanobis_ood": ood,
    }
    if args.json:
        print(json.dumps(result))
        return

    print("\n=== Source Attribution ===")
    print(f"input: {result['input']}")
    print(f"top:   {ranked[0]['name']:<18} family={ranked[0]['family']:<16} "
          f"confidence={ranked[0]['confidence']:.3f}")
    print("ranked:")
    for r in ranked:
        print(f"  #{r['rank']}  {r['name']:<18} family={r['family']:<16} "
              f"conf={r['confidence']:.3f}")
    print(f"energy OOD score : {energy:.3f}")
    if ood is not None:
        print(f"mahalanobis OOD  : {ood:.3f}")
    print("(higher OOD scores => more likely a generator outside the training set)")


if __name__ == "__main__":
    main()
