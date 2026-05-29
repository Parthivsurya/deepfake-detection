"""Audio-visual synchronization head.

Projects per-frame visual tokens and the audio sequence into a shared space,
aligns them in time, and produces:

* a contrastive sync loss (InfoNCE on paired frame/audio chunks),
* a per-clip sync score used as an extra feature for the detector — a low
  score on a "real" clip is itself a strong fake signal (lip-sync mismatch).

Reference: Chung & Zisserman, "Out of time: automated lip sync in the wild".
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class AVSyncHead(nn.Module):
    def __init__(self, video_dim: int, audio_dim: int, proj_dim: int = 128,
                 temperature: float = 0.1):
        super().__init__()
        self.v_proj = nn.Sequential(nn.Linear(video_dim, proj_dim), nn.GELU(),
                                    nn.Linear(proj_dim, proj_dim))
        self.a_proj = nn.Sequential(nn.Linear(audio_dim, proj_dim), nn.GELU(),
                                    nn.Linear(proj_dim, proj_dim))
        self.temperature = temperature
        self.proj_dim = proj_dim

    @staticmethod
    def _resample_audio_to_video(audio_seq: torch.Tensor, T_v: int) -> torch.Tensor:
        # audio_seq: (B, T_a, D) -> (B, T_v, D) via 1D linear interpolation
        x = audio_seq.transpose(1, 2)                              # (B, D, T_a)
        x = F.interpolate(x, size=T_v, mode="linear", align_corners=False)
        return x.transpose(1, 2)

    def forward(
        self,
        frame_tokens: torch.Tensor,    # (B, T, Dv)
        audio_seq: torch.Tensor,       # (B, T_a, Da)
        has_audio: torch.Tensor | None = None,   # (B,) 0/1 mask
    ) -> dict:
        B, T, _ = frame_tokens.shape
        v = F.normalize(self.v_proj(frame_tokens), dim=-1)        # (B, T, P)
        a_resampled = self._resample_audio_to_video(audio_seq, T)
        a = F.normalize(self.a_proj(a_resampled), dim=-1)         # (B, T, P)

        # Per-frame cosine sim -> per-clip "sync score" in [-1, 1]
        per_frame_sim = (v * a).sum(-1)                           # (B, T)
        sync_score = per_frame_sim.mean(dim=1)                    # (B,)

        # InfoNCE on flattened (B*T) frame embeddings
        v_flat = v.reshape(B * T, -1)
        a_flat = a.reshape(B * T, -1)
        logits = (v_flat @ a_flat.t()) / self.temperature         # (B*T, B*T)
        targets = torch.arange(B * T, device=logits.device)
        loss = 0.5 * (F.cross_entropy(logits, targets) +
                      F.cross_entropy(logits.t(), targets))

        if has_audio is not None:
            mask = has_audio.float()
            if mask.sum() == 0:
                loss = loss.new_zeros(())
            sync_score = sync_score * mask

        return {"sync_score": sync_score, "sync_loss": loss, "per_frame_sim": per_frame_sim}
