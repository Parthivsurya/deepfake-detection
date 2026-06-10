"""Frame and face-crop extraction pipeline.

Output convention:
    <out_dir>/<clip_id>/frame_0000.jpg
    <out_dir>/<clip_id>/meta.json
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
from PIL import Image

from utils.video_utils import iter_frames


def extract_frames(
    video_path: str | Path,
    out_dir: str | Path,
    sample_fps: float = 4.0,
    max_frames: Optional[int] = None,
    image_size: Optional[int] = None,
    quality: int = 92,
) -> List[str]:
    """Decode `video_path` at `sample_fps` and write JPEGs to `out_dir`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []
    for i, frame in enumerate(iter_frames(video_path, sample_fps=sample_fps)):
        if max_frames is not None and i >= max_frames:
            break
        img = Image.fromarray(frame)
        if image_size:
            img = _resize_keep_aspect(img, image_size)
        out_path = out_dir / f"frame_{i:04d}.jpg"
        img.save(out_path, quality=quality)
        saved.append(str(out_path))
    with open(out_dir / "meta.json", "w") as f:
        json.dump({"video": str(video_path), "num_frames": len(saved), "fps": sample_fps}, f)
    return saved


def extract_face_crops(
    video_path: str | Path,
    out_dir: str | Path,
    sample_fps: float = 4.0,
    max_frames: Optional[int] = None,
    crop_size: int = 224,
    margin: float = 0.3,
    device: str = "cpu",
    detector=None,
) -> List[str]:
    """Extract face-cropped frames using facenet-pytorch MTCNN.

    Falls back to whole-frame center crop when no face is detected.

    Optimizations vs the naive per-frame path:
      * `detector` can be pre-built once by the caller and reused across clips
        (avoids reloading P/R/O-net weights to GPU per video).
      * All sampled frames of one video are batched through `detector.detect`
        in a single GPU call instead of looping frame-by-frame.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if detector is None:
        from facenet_pytorch import MTCNN
        detector = MTCNN(keep_all=False, device=device, post_process=False)

    # 1) Decode all sampled frames first.
    frames: List[np.ndarray] = []
    for i, frame in enumerate(iter_frames(video_path, sample_fps=sample_fps)):
        if max_frames is not None and i >= max_frames:
            break
        frames.append(frame)
    if not frames:
        with open(out_dir / "meta.json", "w") as f:
            json.dump({"video": str(video_path), "num_frames": 0, "fps": sample_fps,
                       "crop_size": crop_size}, f)
        return []

    # 2) Batch detect across the whole video.
    pils = [Image.fromarray(f) for f in frames]
    boxes_per_frame = _batch_detect(detector, pils)

    # 3) Crop + save.
    saved: List[str] = []
    for i, (frame, boxes) in enumerate(zip(frames, boxes_per_frame)):
        if boxes is None or len(boxes) == 0:
            crop = _center_crop(frame, crop_size)
        else:
            crop = _crop_from_box(frame, boxes[0], crop_size, margin)
        out_path = out_dir / f"frame_{i:04d}.jpg"
        Image.fromarray(crop).save(out_path, quality=92)
        saved.append(str(out_path))

    with open(out_dir / "meta.json", "w") as f:
        json.dump({"video": str(video_path), "num_frames": len(saved), "fps": sample_fps,
                   "crop_size": crop_size}, f)
    return saved


def _batch_detect(detector, pils):
    """Run detector.detect on a list of PIL images, with per-image fallback on error."""
    try:
        boxes, _ = detector.detect(pils)
        return boxes
    except Exception:
        out = []
        for pil in pils:
            try:
                b, _ = detector.detect(pil)
                out.append(b)
            except Exception:
                out.append(None)
        return out


def _crop_from_box(frame: np.ndarray, box, size: int, margin: float) -> np.ndarray:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w / 2, y1 + h / 2
    side = max(w, h) * (1 + margin)
    x1 = max(int(cx - side / 2), 0)
    y1 = max(int(cy - side / 2), 0)
    x2 = min(int(cx + side / 2), frame.shape[1])
    y2 = min(int(cy + side / 2), frame.shape[0])
    crop = frame[y1:y2, x1:x2]
    return np.array(Image.fromarray(crop).resize((size, size), Image.BILINEAR))


def _resize_keep_aspect(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    scale = size / min(w, h)
    return img.resize((int(round(w * scale)), int(round(h * scale))), Image.BILINEAR)


def _detect_and_crop(frame: np.ndarray, detector, size: int, margin: float) -> np.ndarray:
    pil = Image.fromarray(frame)
    try:
        box, _ = detector.detect(pil)
    except Exception:
        # Fallback to CPU if device was not CPU
        import torch
        dev = getattr(detector, "device", None)
        if dev is not None and torch.device(dev).type != "cpu":
            try:
                orig_device = dev
                detector.to("cpu")
                box, _ = detector.detect(pil)
                detector.to(orig_device)
            except Exception:
                box = None
        else:
            box = None

    if box is None or len(box) == 0:
        return _center_crop(frame, size)
    x1, y1, x2, y2 = box[0]
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w / 2, y1 + h / 2
    side = max(w, h) * (1 + margin)
    x1 = max(int(cx - side / 2), 0)
    y1 = max(int(cy - side / 2), 0)
    x2 = min(int(cx + side / 2), frame.shape[1])
    y2 = min(int(cy + side / 2), frame.shape[0])
    crop = frame[y1:y2, x1:x2]
    return np.array(Image.fromarray(crop).resize((size, size), Image.BILINEAR))


def _center_crop(frame: np.ndarray, size: int) -> np.ndarray:
    h, w = frame.shape[:2]
    side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    crop = frame[y0:y0 + side, x0:x0 + side]
    return np.array(Image.fromarray(crop).resize((size, size), Image.BILINEAR))
