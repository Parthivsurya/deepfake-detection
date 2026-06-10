"""Run face-crop frame extraction and audio demuxing over a manifest.

The script enriches the input manifest with `frames_dir` and `audio_path`
columns and writes a new manifest CSV.

Usage:
    python scripts/extract_frames.py \
        --manifest manifests/train.csv \
        --out_frames frames/ \
        --out_audio audio/ \
        --fps 4 --max_frames 64 --crop_size 224
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
from data.preprocessing import extract_face_crops, extract_audio  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--out_frames", required=True)
    p.add_argument("--out_audio", required=True)
    p.add_argument("--fps", type=float, default=4.0)
    p.add_argument("--max_frames", type=int, default=64)
    p.add_argument("--crop_size", type=int, default=224)
    p.add_argument("--no_face", action="store_true",
                   help="skip MTCNN face detection and use center crops only")
    p.add_argument("--audio_sr", type=int, default=16000)
    p.add_argument("--out_manifest", default=None,
                   help="defaults to <manifest>.extracted.csv")
    p.add_argument("--device", default=None,
                   help="device for face detection (cpu, cuda, mps). Defaults to auto-detect.")
    args = p.parse_args()

    import torch
    device = args.device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"Using device: {device} for face crop extraction.")

    df = pd.read_csv(args.manifest)
    out_frames = Path(args.out_frames); out_frames.mkdir(parents=True, exist_ok=True)
    out_audio = Path(args.out_audio); out_audio.mkdir(parents=True, exist_ok=True)

    # Build MTCNN once and reuse across all clips (avoid per-clip model reload).
    detector = None
    if not args.no_face:
        from facenet_pytorch import MTCNN
        detector = MTCNN(keep_all=False, device=device, post_process=False)
        print(f"MTCNN initialized on {device} (reused across all clips)")

    frames_dirs, audio_paths = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="extract"):
        clip = row["clip_id"]
        f_dir = out_frames / clip
        a_path = out_audio / f"{clip}.wav"
        try:
            if args.no_face:
                from data.preprocessing import extract_frames
                extract_frames(row["video_path"], f_dir, sample_fps=args.fps,
                               max_frames=args.max_frames, image_size=args.crop_size)
            else:
                extract_face_crops(row["video_path"], f_dir, sample_fps=args.fps,
                                   max_frames=args.max_frames, crop_size=args.crop_size,
                                   device=device, detector=detector)
            extract_audio(row["video_path"], a_path, sample_rate=args.audio_sr)
        except Exception as e:
            print(f"  ! {clip}: {e}")
        frames_dirs.append(str(f_dir))
        audio_paths.append(str(a_path) if a_path.exists() else "")

    df["frames_dir"] = frames_dirs
    df["audio_path"] = audio_paths
    out = Path(args.out_manifest or args.manifest.replace(".csv", ".extracted.csv"))
    df.to_csv(out, index=False)
    print(f"[done] wrote enriched manifest -> {out}")


if __name__ == "__main__":
    main()
