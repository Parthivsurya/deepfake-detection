"""Full multimodal detector: TemporalViT + AudioEncoder + CrossAttnFusion + AVSync + rPPG."""
from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn

from .temporal_vit import TemporalViT
from .audio_encoder import build_audio_encoder
from .av_sync import AVSyncHead
from .cross_attention_fusion import CrossAttentionFusion


class MultimodalDeepfakeDetector(nn.Module):
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
        audio_sample_rate: int = 16000,
        audio_embed_dim: int = 256,
        fusion_dim: int = 512,
        fusion_depth: int = 2,
        fusion_heads: int = 8,
        max_audio_tokens: int = 256,
        num_classes: int = 2,
        # ----- new toggles -----
        audio_encoder: str = "cnn",          # "cnn" | "wav2vec"
        wav2vec_pretrained: str = "facebook/wav2vec2-base",
        wav2vec_freeze: bool = True,
        use_physio: bool = False,
        physio_embed_dim: int = 128,
        physio_fps: float = 4.0,
    ):
        super().__init__()
        self.visual = TemporalViT(image_size, patch_size, embed_dim, spatial_depth,
                                  temporal_depth, num_heads, mlp_ratio, dropout, max_frames)
        audio_kwargs = {}
        if audio_encoder == "wav2vec":
            audio_kwargs = dict(pretrained=wav2vec_pretrained, freeze=wav2vec_freeze)
        self.audio = build_audio_encoder(
            kind=audio_encoder, sample_rate=audio_sample_rate,
            embed_dim=audio_embed_dim, **audio_kwargs,
        )
        self.av_sync = AVSyncHead(embed_dim, audio_embed_dim, proj_dim=128)
        self.fusion = CrossAttentionFusion(
            video_dim=embed_dim,
            audio_dim=audio_embed_dim,
            fusion_dim=fusion_dim,
            depth=fusion_depth,
            num_heads=fusion_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            max_video_tokens=max_frames,
            max_audio_tokens=max_audio_tokens,
        )

        self.use_physio = use_physio
        if use_physio:
            from physio.rppg import PhysioEncoder
            self.physio = PhysioEncoder(embed_dim=physio_embed_dim, fps=physio_fps)
            extra_dim = physio_embed_dim
        else:
            extra_dim = 0

        # +1 for the scalar AV-sync score concatenated as an extra feature
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim + 1 + extra_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes),
        )

    def forward(
        self,
        frames: torch.Tensor,           # (B, T, 3, H, W)
        waveform: torch.Tensor,         # (B, samples)
        has_audio: Optional[torch.Tensor] = None,  # (B,) 0/1
    ) -> dict:
        _, v_tokens = self.visual(frames)
        _, a_tokens = self.audio(waveform)
        sync = self.av_sync(v_tokens, a_tokens, has_audio=has_audio)

        fused_out = self.fusion(v_tokens, a_tokens, has_audio=has_audio)
        parts = [fused_out["fused"], sync["sync_score"].unsqueeze(-1)]

        physio_out = None
        if self.use_physio:
            physio_out = self.physio(frames)
            parts.append(physio_out["clip"])

        fused = torch.cat(parts, dim=-1)
        logits = self.classifier(fused)
        result = {
            "logits": logits,
            "sync_loss": sync["sync_loss"],
            "sync_score": sync["sync_score"],
            "fused": fused,
            "video_tokens_fused": fused_out["video_tokens"],
            "audio_tokens_fused": fused_out["audio_tokens"],
        }
        if physio_out is not None:
            result["physio_embed"] = physio_out["clip"]
            result["physio_signal"] = physio_out["signal"]
        return result
