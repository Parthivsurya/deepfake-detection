"""Forensic recovery pipeline: detect → reconstruct → re-verify → trust score.

Full Module 2 inference pipeline:

    frames -> perturbation detector -> p_pert ∈ [0,1]
    frames -> detector -> p_orig (probability of "fake")
    if p_pert > threshold:
        frames -> diffusion.purify -> frames_recon
        frames_recon -> detector -> p_recon
    final probability   = (1 - p_pert) * p_orig + p_pert * p_recon
    trust score         = 1 - p_pert       (how much to trust the raw input)

The trust score is what gets surfaced to downstream users / the UI as the
"system confidence" — high when nothing looks tampered with, low when the
input was likely modified before it reached us.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ddpm import DDPM


@dataclass
class RecoveryResult:
    p_perturbation: torch.Tensor    # (B,) prob input was adversarially perturbed
    p_orig_fake: torch.Tensor       # (B,) detector prob of "fake" on raw input
    p_recon_fake: torch.Tensor      # (B,) detector prob of "fake" on reconstructed input
    p_final_fake: torch.Tensor      # (B,) blended probability
    trust_score: torch.Tensor       # (B,) ∈ [0, 1]
    reconstructed: bool             # whether any sample passed through diffusion
    frames_reconstructed: Optional[torch.Tensor] = None


class ForensicRecoveryPipeline:
    def __init__(
        self,
        detector: nn.Module,
        perturbation_detector: nn.Module,
        diffusion: DDPM,
        recon_threshold: float = 0.5,
        t_star: int = 50,
    ):
        self.detector = detector
        self.perturbation_detector = perturbation_detector
        self.diffusion = diffusion
        self.recon_threshold = float(recon_threshold)
        self.t_star = int(t_star)

    @torch.inference_mode()
    def _classify(self, frames, audio, has_audio) -> torch.Tensor:
        out = self.detector(frames, audio, has_audio=has_audio)
        return F.softmax(out["logits"], dim=-1)[:, 1]

    def run(
        self,
        frames: torch.Tensor,
        audio: torch.Tensor,
        has_audio: Optional[torch.Tensor] = None,
        keep_reconstructed: bool = False,
    ) -> RecoveryResult:
        self.detector.eval()
        self.perturbation_detector.eval()
        self.diffusion.eps_net.eval()

        p_pert = self.perturbation_detector(frames)
        p_orig = self._classify(frames, audio, has_audio)

        needs_recon = p_pert > self.recon_threshold
        frames_recon = frames.clone()
        any_recon = bool(needs_recon.any().item())
        if any_recon:
            chunk = frames[needs_recon]
            chunk_recon = self.diffusion.purify(chunk, t_star=self.t_star)
            frames_recon[needs_recon] = chunk_recon

        p_recon = self._classify(frames_recon, audio, has_audio)

        # convex blend by perturbation probability
        p_final = (1.0 - p_pert) * p_orig + p_pert * p_recon
        trust = 1.0 - p_pert

        return RecoveryResult(
            p_perturbation=p_pert.detach(),
            p_orig_fake=p_orig.detach(),
            p_recon_fake=p_recon.detach(),
            p_final_fake=p_final.detach(),
            trust_score=trust.detach(),
            reconstructed=any_recon,
            frames_reconstructed=(frames_recon.detach() if keep_reconstructed else None),
        )
