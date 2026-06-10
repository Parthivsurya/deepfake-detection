"""Physiological (rPPG) liveness branch.

Remote photoplethysmography extracts a blood-volume-pulse signal from the
periodic skin-color micro-variations of a real face. Real faces have a
~1 Hz pulse; most deepfake generators destroy it. F_P from the TRINETRA
diagram.
"""
from .rppg import (
    PhysioEncoder,
    extract_face_rgb_means,
    pos_chrominance,
    rppg_temporal_signal,
)

__all__ = [
    "PhysioEncoder",
    "extract_face_rgb_means",
    "pos_chrominance",
    "rppg_temporal_signal",
]
