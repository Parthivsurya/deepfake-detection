"""Source attribution: identify *which generator* produced a flagged deepfake.

Three-branch fingerprint model on top of the existing detector backbone:
    spectral (DCT high-pass) + residual (image - denoise) + semantic (frozen TemporalViT)
    -> N-way classifier + open-set OOD scoring.
"""
from .generators import GENERATOR_REGISTRY, GeneratorInfo, generator_family
from .fingerprint import dct_highpass, denoise_residual, FingerprintExtractor
from .spectral_cnn import SpectralCNN, ResidualCNN
from .attribution_head import AttributionHead
from .open_set import EnergyScorer, MahalanobisScorer
from .model import SourceAttributionModel

__all__ = [
    "GENERATOR_REGISTRY",
    "GeneratorInfo",
    "generator_family",
    "dct_highpass",
    "denoise_residual",
    "FingerprintExtractor",
    "SpectralCNN",
    "ResidualCNN",
    "AttributionHead",
    "EnergyScorer",
    "MahalanobisScorer",
    "SourceAttributionModel",
]
