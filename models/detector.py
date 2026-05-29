"""Full multimodal detector: TemporalViT + AudioEncoder + AVSync + classifier."""
from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn

from .temporal_vit import TemporalViT
from .audio_encoder import AudioEncoder
from .av_sync import AVSyncHead


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
        num_classes: int = 2,
    ):
        super().__init__()
        self.visual = TemporalViT(image_size, patch_size, embed_dim, spatial_depth,
                                  temporal_depth, num_heads, mlp_ratio, dropout, max_frames)
        self.audio = AudioEncoder(audio_sample_rate, embed_dim=audio_embed_dim)
        self.av_sync = AVSyncHead(embed_dim, audio_embed_dim, proj_dim=128)

        # +1 for the scalar sync score appended to the fused vector
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim + audio_embed_dim + 1, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(fusion_dim, num_classes)

    def forward(
        self,
        frames: torch.Tensor,           # (B, T, 3, H, W)
        waveform: torch.Tensor,         # (B, samples)
        has_audio: Optional[torch.Tensor] = None,  # (B,) 0/1
    ) -> dict:
        v_cls, v_tokens = self.visual(frames)
        a_cls, a_seq = self.audio(waveform)
        sync = self.av_sync(v_tokens, a_seq, has_audio=has_audio)

        if has_audio is not None:
            mask = has_audio.float().unsqueeze(-1)
            a_cls = a_cls * mask    # zero-out audio for silent clips

        fused = torch.cat([v_cls, a_cls, sync["sync_score"].unsqueeze(-1)], dim=-1)
        fused = self.fusion(fused)
        logits = self.classifier(fused)
        return {
            "logits": logits,
            "sync_loss": sync["sync_loss"],
            "sync_score": sync["sync_score"],
            "fused": fused,
        }
