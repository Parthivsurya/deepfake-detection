"""Bidirectional cross-modal attention fusion (Task 3).

Given a sequence of video tokens V ∈ R^{B×T_v×D_v} and audio tokens
A ∈ R^{B×T_a×D_a}, we (1) project both to a shared dimension D, (2) run
several blocks of bidirectional cross-attention so each modality is
contextualised by the other, and (3) return a fused clip embedding plus the
refined token sequences (the latter are reused by the AV-sync head and for
explainability / token-level diagnostics).

Equations (per fusion block ℓ):
    Vℓ' = Vℓ + Attn(Q=LN(Vℓ), K=LN(Aℓ), V=LN(Aℓ))     # V attends to A
    Aℓ' = Aℓ + Attn(Q=LN(Aℓ), K=LN(Vℓ), V=LN(Vℓ))     # A attends to V
    V_{ℓ+1} = Vℓ' + MLP(LN(Vℓ'))
    A_{ℓ+1} = Aℓ' + MLP(LN(Aℓ'))

The final clip vector is the concatenation of the mean-pooled refined
sequences, projected back to `fusion_dim`.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class _MLP(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(F.gelu(self.fc1(x)))))


class CrossAttentionBlock(nn.Module):
    """One layer of bidirectional cross-attention + per-stream MLP."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        super().__init__()
        self.ln_v_q = nn.LayerNorm(dim)
        self.ln_a_kv = nn.LayerNorm(dim)
        self.attn_v2a = nn.MultiheadAttention(dim, num_heads, dropout=dropout,
                                              batch_first=True)
        self.ln_a_q = nn.LayerNorm(dim)
        self.ln_v_kv = nn.LayerNorm(dim)
        self.attn_a2v = nn.MultiheadAttention(dim, num_heads, dropout=dropout,
                                              batch_first=True)
        self.ln_v_mlp = nn.LayerNorm(dim)
        self.ln_a_mlp = nn.LayerNorm(dim)
        self.mlp_v = _MLP(dim, int(dim * mlp_ratio), dropout)
        self.mlp_a = _MLP(dim, int(dim * mlp_ratio), dropout)

    def forward(
        self,
        v: torch.Tensor,                       # (B, T_v, D)
        a: torch.Tensor,                       # (B, T_a, D)
        key_padding_mask_a: Optional[torch.Tensor] = None,   # (B, T_a)  True = pad
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # video attends to audio
        v_q = self.ln_v_q(v)
        a_kv = self.ln_a_kv(a)
        v_attn, _ = self.attn_v2a(v_q, a_kv, a_kv,
                                  key_padding_mask=key_padding_mask_a,
                                  need_weights=False)
        v = v + v_attn
        # audio attends to video
        a_q = self.ln_a_q(a)
        v_kv = self.ln_v_kv(v)
        a_attn, _ = self.attn_a2v(a_q, v_kv, v_kv, need_weights=False)
        a = a + a_attn
        # per-stream MLP
        v = v + self.mlp_v(self.ln_v_mlp(v))
        a = a + self.mlp_a(self.ln_a_mlp(a))
        return v, a


class CrossAttentionFusion(nn.Module):
    """Project V/A to a shared dim, run N cross-attention blocks, fuse."""

    def __init__(
        self,
        video_dim: int,
        audio_dim: int,
        fusion_dim: int = 512,
        depth: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_video_tokens: int = 64,
        max_audio_tokens: int = 256,
    ):
        super().__init__()
        self.v_in = nn.Linear(video_dim, fusion_dim)
        self.a_in = nn.Linear(audio_dim, fusion_dim)
        # modality embedding lets the model distinguish V vs A inside attention
        self.mod_v = nn.Parameter(torch.zeros(1, 1, fusion_dim))
        self.mod_a = nn.Parameter(torch.zeros(1, 1, fusion_dim))
        self.pos_v = nn.Parameter(torch.zeros(1, max_video_tokens, fusion_dim))
        self.pos_a = nn.Parameter(torch.zeros(1, max_audio_tokens, fusion_dim))
        nn.init.trunc_normal_(self.mod_v, std=0.02)
        nn.init.trunc_normal_(self.mod_a, std=0.02)
        nn.init.trunc_normal_(self.pos_v, std=0.02)
        nn.init.trunc_normal_(self.pos_a, std=0.02)

        self.blocks = nn.ModuleList([
            CrossAttentionBlock(fusion_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm_v = nn.LayerNorm(fusion_dim)
        self.norm_a = nn.LayerNorm(fusion_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fusion_dim = fusion_dim

    def forward(
        self,
        video_tokens: torch.Tensor,            # (B, T_v, Dv)
        audio_tokens: torch.Tensor,            # (B, T_a, Da)
        has_audio: Optional[torch.Tensor] = None,   # (B,) 0/1
    ) -> dict:
        B, T_v, _ = video_tokens.shape
        T_a = audio_tokens.size(1)

        v = self.v_in(video_tokens) + self.pos_v[:, :T_v] + self.mod_v
        a = self.a_in(audio_tokens) + self.pos_a[:, :T_a] + self.mod_a

        # When the entire batch has no audio (e.g., CDF-only batches), skip the
        # audio cross-attention path entirely. Otherwise the constant Wav2Vec2
        # output from zero waveforms dominates the residual stream and the
        # video encoder receives no gradient signal.
        batch_all_silent = (
            has_audio is not None and bool((has_audio < 0.5).all().item())
        )

        # mask out silent clips' audio so video doesn't get poisoned by it
        kpm_a = None
        if has_audio is not None and not batch_all_silent:
            # True where audio should be ignored
            kpm_a = (has_audio < 0.5).unsqueeze(1).expand(B, T_a)

        for blk in self.blocks:
            if batch_all_silent:
                # Video-only path: self-attention over video tokens via the
                # block's MLP + residual; skip cross-attention to audio.
                v = v + blk.mlp_v(blk.ln_v_mlp(v))
            else:
                v, a = blk(v, a, key_padding_mask_a=kpm_a)

        v = self.norm_v(v)
        a = self.norm_a(a)

        v_pool = v.mean(dim=1)
        if has_audio is not None:
            mask = has_audio.float().unsqueeze(-1)
            a_pool = (a.mean(dim=1)) * mask
        else:
            a_pool = a.mean(dim=1)

        fused = self.out_proj(torch.cat([v_pool, a_pool], dim=-1))
        return {"fused": fused, "video_tokens": v, "audio_tokens": a}
