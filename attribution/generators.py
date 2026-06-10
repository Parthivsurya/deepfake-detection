"""Registry of known generators and their forensic families.

Class 0 is reserved for `real` so the same head can be used standalone
(without an upstream binary detector) if desired.

Family taxonomy:
    encoder_decoder  — autoencoder face-swap pipelines (Deepfakes, FaceShifter)
    graphics         — classical CG re-enactment (Face2Face)
    blending         — landmark + blend face-swap (FaceSwap)
    neural_texture   — neural rendering on top of UV maps (NeuralTextures)
    gan              — pure GAN synthesis (StyleGAN family, SimSwap)
    diffusion        — diffusion-based generation (SD, SDXL, img2img)
    lip_sync         — audio-driven lip generation (Wav2Lip, SadTalker)
    custom           — dataset-specific pipelines (Celeb-DF)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class GeneratorInfo:
    id: int
    name: str
    family: str
    source_dataset: str  # which dataset this label comes from


GENERATOR_REGISTRY: Dict[int, GeneratorInfo] = {
    0: GeneratorInfo(0, "real",            "real",            "any"),
    1: GeneratorInfo(1, "FF-Deepfakes",    "encoder_decoder", "faceforensics"),
    2: GeneratorInfo(2, "FF-Face2Face",    "graphics",        "faceforensics"),
    3: GeneratorInfo(3, "FF-FaceSwap",     "blending",        "faceforensics"),
    4: GeneratorInfo(4, "FF-NeuralTex",    "neural_texture",  "faceforensics"),
    5: GeneratorInfo(5, "FF-FaceShifter",  "encoder_decoder", "faceforensics"),
    6: GeneratorInfo(6, "Celeb-DF",        "custom",          "celebdf"),
    7: GeneratorInfo(7, "DFDC-mixed",      "encoder_decoder", "dfdc"),
    8: GeneratorInfo(8, "FakeAVCeleb-LS",  "lip_sync",        "fakeavceleb"),
    # 9+ reserved for self-generated samples (StyleGAN3, SimSwap, SD, ...)
}

# Reverse lookup: registry name -> id (used by the manifest builder).
NAME_TO_ID: Dict[str, int] = {info.name: gid for gid, info in GENERATOR_REGISTRY.items()}

# FF++ "manipulation" column -> registry id.
FF_METHOD_TO_ID: Dict[str, int] = {
    "Deepfakes": 1,
    "Face2Face": 2,
    "FaceSwap": 3,
    "NeuralTextures": 4,
    "FaceShifter": 5,
}


def generator_family(gid: int) -> str:
    info = GENERATOR_REGISTRY.get(int(gid))
    return info.family if info else "unknown"


def num_known_classes() -> int:
    """Number of registered generator classes (incl. real)."""
    return len(GENERATOR_REGISTRY)
