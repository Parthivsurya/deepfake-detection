"""Full multimodal detector: TemporalViT + AudioEncoder + CrossAttnFusion + AVSync + rPPG."""
from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn

from .temporal_vit import TemporalViT
from .audio_encoder import build_audio_encoder
from .av_sync import AVSyncHead
from .cross_attention_fusion import CrossAttentionFusion
from .pretrained_backbone import build_visual_backbone
from .trust import TrustReliabilityEstimator, TrustAwareSparseGate


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
        backbone: str = "temporal_vit",      # "temporal_vit" | "resnet50"
        backbone_freeze: bool = True,        # only applies to pretrained backbones
        # ----- TRUSTFUSE reliability (TRE + TSF) -----
        use_trust: bool = False,             # False = baseline (no reliability)
        trust_dim: int = 128,                # common trust-space dim for Eqs 1-3
        trust_tau: float = 0.5,              # TSF gate threshold τ (Eq. 7)
    ):
        super().__init__()
        self.visual = build_visual_backbone(
            kind=backbone,
            image_size=image_size,
            embed_dim=embed_dim,
            temporal_depth=temporal_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            max_frames=max_frames,
            patch_size=patch_size,
            spatial_depth=spatial_depth,
            freeze=backbone_freeze,
        )
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

        # ---------------- TRUSTFUSE reliability (TRE + TSF) ----------------
        # TRE estimates a per-modality trust score R_m ∈ [0,1]; TSF gates each
        # modality's contribution by R_m before fusion (Eq. 7). Per-modality
        # linear projections bring V/A/(P) into a common trust space so the
        # cross-modal cosine agreement (Eq. 3) is well-defined across differing
        # feature dims. The gate then scales the *original* features by the
        # scalar R_m, so gating is dimension-agnostic and faithful to Eq. 7.
        self.use_trust = use_trust
        if use_trust:
            self.trust_proj = nn.ModuleDict({
                "V": nn.Linear(embed_dim, trust_dim),
                "A": nn.Linear(audio_embed_dim, trust_dim),
            })
            if use_physio:
                self.trust_proj["P"] = nn.Linear(physio_embed_dim, trust_dim)
            self.tre = TrustReliabilityEstimator(
                num_modalities=3 if use_physio else 2)
            self.trust_gate = TrustAwareSparseGate(tau=trust_tau)

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

        # physio is needed up-front when TRE is on (it consumes the rPPG seq).
        physio_out = self.physio(frames) if self.use_physio else None

        # ---------------- TRE + TSF trust gating ----------------
        trust_loss = None
        trust_scores = None
        if self.use_trust:
            feats = {
                "V": self.trust_proj["V"](v_tokens),          # (B, Tv, trust_dim)
                "A": self.trust_proj["A"](a_tokens),          # (B, Ta, trust_dim)
            }
            if self.use_physio:
                feats["P"] = self.trust_proj["P"](physio_out["seq"])  # (B, Tp, trust_dim)
            # same projected sequences drive both the C/A terms and the temporal
            # stability term T (consecutive-token differences).
            tre_out = self.tre(feats, modality_feats_temporal=feats)
            trust_scores = tre_out["trust"]                   # {name: (B,)}
            trust_loss = tre_out["loss"]
            # Eq. 7 — scale the *original* features by the scalar trust R_m.
            v_tokens = self.trust_gate(v_tokens, trust_scores["V"])
            a_tokens = self.trust_gate(a_tokens, trust_scores["A"])

        sync = self.av_sync(v_tokens, a_tokens, has_audio=has_audio)

        fused_out = self.fusion(v_tokens, a_tokens, has_audio=has_audio)
        parts = [fused_out["fused"], sync["sync_score"].unsqueeze(-1)]

        if physio_out is not None:
            physio_clip = physio_out["clip"]
            if self.use_trust:
                # gate the (B, D) clip embedding by R_P via a temporary token axis
                physio_clip = self.trust_gate(
                    physio_clip.unsqueeze(1), trust_scores["P"]).squeeze(1)
            parts.append(physio_clip)

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
        if trust_loss is not None:
            result["trust_loss"] = trust_loss
            result["trust"] = {k: v.detach() for k, v in trust_scores.items()}
        return result
