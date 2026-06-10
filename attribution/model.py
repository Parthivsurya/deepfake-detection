"""End-to-end source attribution model.

Up to six branches:
  * Semantic video    — frozen TemporalViT (from detector checkpoint)
  * Spectral video    — DCT high-pass -> SpectralCNN
  * Residual video    — denoise residual -> ResidualCNN
  * Semantic audio    — frozen audio encoder (from detector checkpoint)
  * Audio fingerprint — mel-residual -> AudioFingerprintCNN
  * Physio (rPPG)     — frozen PhysioEncoder (from detector checkpoint)

Audio branches are gated by `has_audio` (B,) so videos without a soundtrack
contribute a zero audio embedding and don't poison the head. Physio is always
on when `use_physio=True` (real faces always have a face crop).
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from models.audio_encoder import build_audio_encoder
from models.temporal_vit import TemporalViT

from .attribution_head import AttributionHead
from .fingerprint import AudioFingerprintExtractor, FingerprintExtractor
from .generators import num_known_classes
from .spectral_cnn import AudioFingerprintCNN, ResidualCNN, SpectralCNN


class SourceAttributionModel(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 384,
        spatial_depth: int = 6,
        temporal_depth: int = 4,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_frames: int = 64,
        spectral_dim: int = 256,
        residual_dim: int = 256,
        head_hidden: int = 384,
        num_classes: Optional[int] = None,
        # ----- audio -----
        use_audio: bool = True,
        audio_sample_rate: int = 16000,
        audio_embed_dim: int = 256,
        audio_fp_dim: int = 256,
        audio_n_mels: int = 80,
        audio_encoder_kind: str = "cnn",        # "cnn" | "wav2vec"
        wav2vec_pretrained: str = "facebook/wav2vec2-base",
        wav2vec_freeze: bool = True,
        # ----- physio -----
        use_physio: bool = False,
        physio_embed_dim: int = 128,
        physio_fps: float = 4.0,
    ):
        super().__init__()
        if num_classes is None:
            num_classes = num_known_classes()

        # Video branches
        self.backbone = TemporalViT(
            image_size, patch_size, embed_dim, spatial_depth,
            temporal_depth, num_heads, mlp_ratio, dropout, max_frames,
        )
        self.fingerprint = FingerprintExtractor()
        self.spectral_cnn = SpectralCNN(in_channels=3, embed_dim=spectral_dim)
        self.residual_cnn = ResidualCNN(in_channels=3, embed_dim=residual_dim)

        # Audio branches
        self.use_audio = use_audio
        self.audio_encoder_kind = audio_encoder_kind
        branch_dims = [embed_dim, spectral_dim, residual_dim]
        if use_audio:
            audio_kwargs = {}
            if audio_encoder_kind == "wav2vec":
                audio_kwargs = dict(pretrained=wav2vec_pretrained, freeze=wav2vec_freeze)
            self.audio_encoder = build_audio_encoder(
                kind=audio_encoder_kind,
                sample_rate=audio_sample_rate,
                embed_dim=audio_embed_dim,
                **(dict(n_mels=audio_n_mels) if audio_encoder_kind == "cnn" else {}),
                **audio_kwargs,
            )
            self.audio_fp = AudioFingerprintExtractor(
                sample_rate=audio_sample_rate, n_mels=audio_n_mels,
            )
            self.audio_fp_cnn = AudioFingerprintCNN(n_mels=audio_n_mels, embed_dim=audio_fp_dim)
            branch_dims += [audio_embed_dim, audio_fp_dim]
            self.audio_embed_dim = audio_embed_dim
            self.audio_fp_dim = audio_fp_dim

        # Physio branch
        self.use_physio = use_physio
        if use_physio:
            from physio.rppg import PhysioEncoder
            self.physio = PhysioEncoder(embed_dim=physio_embed_dim, fps=physio_fps)
            branch_dims.append(physio_embed_dim)
            self.physio_embed_dim = physio_embed_dim

        self.head = AttributionHead(
            branch_dims=branch_dims,
            hidden_dim=head_hidden,
            num_classes=num_classes,
            dropout=dropout,
        )

        self._backbone_frozen = False
        self._audio_encoder_frozen = False
        self._physio_frozen = False

    # ------------------------------------------------------------------
    def load_backbone(self, ckpt_path: str | Path, strict: bool = False) -> None:
        """Load detector weights and freeze the semantic branches.

        Detector ckpt stores the full MultimodalDeepfakeDetector state-dict
        under "model"; visual.* -> TemporalViT, audio.* -> AudioEncoder.
        """
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt.get("model", ckpt)

        visual_sd = {k[len("visual."):]: v for k, v in sd.items() if k.startswith("visual.")}
        if not visual_sd:
            raise RuntimeError(
                f"checkpoint {ckpt_path} has no `visual.*` keys "
                "— expected a MultimodalDeepfakeDetector checkpoint"
            )
        self.backbone.load_state_dict(visual_sd, strict=strict)
        self.freeze_backbone()

        if self.use_audio:
            audio_sd = {k[len("audio."):]: v for k, v in sd.items() if k.startswith("audio.")}
            if audio_sd:
                self.audio_encoder.load_state_dict(audio_sd, strict=strict)
                self.freeze_audio_encoder()
            else:
                print("WARN: no `audio.*` keys in detector ckpt — audio encoder "
                      "will be trained from scratch (or stays randomly initialized).")

        if self.use_physio:
            physio_sd = {k[len("physio."):]: v for k, v in sd.items() if k.startswith("physio.")}
            if physio_sd:
                self.physio.load_state_dict(physio_sd, strict=strict)
                self.freeze_physio()
            else:
                print("WARN: no `physio.*` keys in detector ckpt — physio encoder "
                      "will be trained from scratch.")

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        self._backbone_frozen = True

    def freeze_audio_encoder(self) -> None:
        for p in self.audio_encoder.parameters():
            p.requires_grad = False
        self.audio_encoder.eval()
        self._audio_encoder_frozen = True

    def freeze_physio(self) -> None:
        for p in self.physio.parameters():
            p.requires_grad = False
        self.physio.eval()
        self._physio_frozen = True

    def train(self, mode: bool = True):
        super().train(mode)
        if self._backbone_frozen:
            self.backbone.eval()
        if self.use_audio and self._audio_encoder_frozen:
            self.audio_encoder.eval()
        if self.use_physio and self._physio_frozen:
            self.physio.eval()
        return self

    # ------------------------------------------------------------------
    def forward(
        self,
        frames: torch.Tensor,
        waveform: Optional[torch.Tensor] = None,
        has_audio: Optional[torch.Tensor] = None,
    ) -> dict:
        """frames: (B, T, 3, H, W);  waveform: (B, samples);  has_audio: (B,)."""
        # ----- video semantic (frozen) -----
        if self._backbone_frozen:
            with torch.no_grad():
                _, v_tokens = self.backbone(frames)
        else:
            _, v_tokens = self.backbone(frames)
        semantic = v_tokens.mean(dim=1)                  # (B, embed_dim)

        # ----- video fingerprints -----
        fp = self.fingerprint(frames)
        spectral = self.spectral_cnn(fp["spectral"])
        residual = self.residual_cnn(fp["residual"])

        branches = [semantic, spectral, residual]

        # ----- audio -----
        if self.use_audio:
            if waveform is None:
                raise ValueError("model has use_audio=True but no waveform was passed")
            if self._audio_encoder_frozen:
                with torch.no_grad():
                    a_clip, _ = self.audio_encoder(waveform)
            else:
                a_clip, _ = self.audio_encoder(waveform)
            a_mel = self.audio_fp(waveform)
            a_fp = self.audio_fp_cnn(a_mel)

            if has_audio is not None:
                mask = has_audio.to(a_clip.dtype).view(-1, 1)
                a_clip = a_clip * mask
                a_fp = a_fp * mask
            branches += [a_clip, a_fp]

        # ----- physio (rPPG) -----
        physio_clip = None
        if self.use_physio:
            if self._physio_frozen:
                with torch.no_grad():
                    p_out = self.physio(frames)
            else:
                p_out = self.physio(frames)
            physio_clip = p_out["clip"]
            branches.append(physio_clip)

        out = self.head(*branches)
        result = {
            "logits": out["logits"],
            "embed": out["embed"],
            "semantic": semantic,
            "spectral": spectral,
            "residual": residual,
        }
        if self.use_audio:
            result["audio_semantic"] = a_clip
            result["audio_fingerprint"] = a_fp
        if self.use_physio:
            result["physio"] = physio_clip
        return result
