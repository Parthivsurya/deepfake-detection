"""Source attribution: identify *which generator* produced a flagged deepfake.

Three-branch fingerprint model on top of the existing detector backbone:
    spectral (DCT high-pass) + residual (image - denoise) + semantic (frozen TemporalViT)
    -> N-way classifier + open-set OOD scoring.
"""
from .generators import GENERATOR_REGISTRY, GeneratorInfo, generator_family
from .fingerprint import (
    AudioFingerprintExtractor,
    FingerprintExtractor,
    dct_highpass,
    denoise_residual,
    mel_residual,
)
from .spectral_cnn import AudioFingerprintCNN, ResidualCNN, SpectralCNN
from .attribution_head import AttributionHead
from .losses import SupConLoss
from .open_set import EnergyScorer, MahalanobisScorer
from .model import SourceAttributionModel

__all__ = [
    "GENERATOR_REGISTRY",
    "GeneratorInfo",
    "generator_family",
    "dct_highpass",
    "denoise_residual",
    "mel_residual",
    "FingerprintExtractor",
    "AudioFingerprintExtractor",
    "SpectralCNN",
    "ResidualCNN",
    "AudioFingerprintCNN",
    "AttributionHead",
    "SupConLoss",
    "EnergyScorer",
    "MahalanobisScorer",
    "SourceAttributionModel",
]
