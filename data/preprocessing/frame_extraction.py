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
) -> List[str]:
    """Extract face-cropped frames using facenet-pytorch MTCNN.

    Falls back to whole-frame center crop when no face is detected.
    """
    from facenet_pytorch import MTCNN
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detector = MTCNN(keep_all=False, device=device, post_process=False)
    saved: List[str] = []
    for i, frame in enumerate(iter_frames(video_path, sample_fps=sample_fps)):
        if max_frames is not None and i >= max_frames:
            break
        crop = _detect_and_crop(frame, detector, crop_size, margin)
        Image.fromarray(crop).save(out_dir / f"frame_{i:04d}.jpg", quality=92)
        saved.append(str(out_dir / f"frame_{i:04d}.jpg"))
    with open(out_dir / "meta.json", "w") as f:
        json.dump({"video": str(video_path), "num_frames": len(saved), "fps": sample_fps,
                   "crop_size": crop_size}, f)
    return saved


def _resize_keep_aspect(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    scale = size / min(w, h)
    return img.resize((int(round(w * scale)), int(round(h * scale))), Image.BILINEAR)


def _detect_and_crop(frame: np.ndarray, detector, size: int, margin: float) -> np.ndarray:
    pil = Image.fromarray(frame)
    box, _ = detector.detect(pil)
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
