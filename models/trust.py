"""Dynamic Trust Reliability Estimation (TRE) + Trust-aware Sparse Fusion (TSF).

This module implements the reliability mechanism of the TRUSTFUSE framework.
Before the video / audio / physiological (rPPG) modalities are fused, TRE asks
"how much should we trust each modality right now?" and produces a per-modality
trust score R_m ∈ [0, 1]. TSF then gates / re-weights the projected features by
R_m so that a manipulated or unreliable modality is suppressed before
cross-modal attention. A trust score close to 1 means the modality is
self-consistent, agrees with the other modalities, and is temporally stable.

Notation. Each modality M ∈ {V, A, P} contributes a sequence of feature
vectors F_M = {f_1, …, f_{N_M}}, f_i ∈ R^d. All equation numbers below refer to
mam's TRUSTFUSE methodology.

    μ_m   = (1/N_M) Σ_i f_i                                       (Eq. 1)
    C_m   = exp( − (1/N_M) Σ_i ||f_i − μ_m||²  )                  (Eq. 2)  intra-modal consistency
    A_m   = (1/(M−1)) Σ_{n≠m} cos(μ_m, μ_n)                       (Eq. 3)  cross-modal agreement
    T_m   = exp( − (1/(T−1)) Σ_{t≥2} ||μ_m^t − μ_m^{t−1}||²  )    (Eq. 4)  temporal stability
    R_m   = sigmoid( w1·C_m + w2·A_m + w3·T_m + b )               (Eq. 5)  trust score
    R̂_m   = (C_m + A_m + T_m) / 3                                 (Eq. 6a) pseudo-target
    L_trust = (1/M) Σ_m (R_m − R̂_m)²                             (Eq. 6)  reliability loss
    F̂_m   = R_m · F̃_m if R_m > τ else 0                          (Eq. 7)  trust-gated fusion

Design notes
------------
* C_m and T_m use a bounded exponential of a mean squared distance, so both live
  in (0, 1]; A_m is a mean cosine similarity in [-1, 1]. We keep the raw values
  for Eq. 5 / Eq. 6 exactly as written, and (optionally) map A_m to [0, 1] only
  when forming the pseudo-target so R̂_m is a valid probability target.
* Temporal stability (Eq. 4) needs per-timestep prototypes μ_m^t. When a caller
  passes a single pooled prototype (no temporal axis) we treat T_m as neutral
  (1.0) so the term drops out rather than being fabricated.
* The gate in Eq. 7 is hard (a step at τ). During training we expose a smooth
  straight-through variant so gradients still flow to TRE; at eval the hard gate
  is used. Toggle via `hard_gate`.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def modality_prototype(feats: torch.Tensor) -> torch.Tensor:
    """Eq. 1 — trust-weighted prototype μ_m = mean over the N_M feature vectors.

    feats: (B, N_M, d)   ->   (B, d)
    """
    return feats.mean(dim=1)


def intra_modal_consistency(feats: torch.Tensor, proto: torch.Tensor) -> torch.Tensor:
    """Eq. 2 — C_m = exp( − mean_i ||f_i − μ_m||² ), in (0, 1].

    feats: (B, N_M, d),  proto: (B, d)   ->   (B,)
    """
    # squared Euclidean distance of every feature to its prototype
    sq = ((feats - proto.unsqueeze(1)) ** 2).sum(dim=-1)   # (B, N_M)
    mean_sq = sq.mean(dim=1)                                # (B,)
    return torch.exp(-mean_sq)


def cross_modal_agreement(protos: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    """Eq. 3 — A_m = mean_{n≠m} cos(μ_m, μ_n) over the other modalities.

    protos: list of M prototypes, each (B, d)   ->   list of M agreements, each (B,)
    """
    M = len(protos)
    if M < 2:
        # a single modality has no one to agree with; neutral agreement
        return [torch.ones(protos[0].size(0), device=protos[0].device)]
    agreements: List[torch.Tensor] = []
    for m in range(M):
        sims = []
        for n in range(M):
            if n == m:
                continue
            sims.append(F.cosine_similarity(protos[m], protos[n], dim=-1))
        agreements.append(torch.stack(sims, dim=0).mean(dim=0))  # (B,)
    return agreements


def temporal_stability(protos_t: Optional[torch.Tensor]) -> torch.Tensor:
    """Eq. 4 — T_m = exp( − mean_{t≥2} ||μ_m^t − μ_m^{t−1}||² ), in (0, 1].

    protos_t: (B, T, d) per-timestep prototypes, or None.
    Returns (B,). If fewer than 2 timesteps are available, returns neutral 1.0
    (the term is undefined, so we do not invent a value).
    """
    if protos_t is None or protos_t.size(1) < 2:
        # caller must supply the batch size via a fallback below
        raise ValueError("temporal_stability needs (B, T>=2, d); see TrustReliabilityEstimator")
    diffs = protos_t[:, 1:] - protos_t[:, :-1]              # (B, T-1, d)
    sq = (diffs ** 2).sum(dim=-1)                          # (B, T-1)
    mean_sq = sq.mean(dim=1)                                # (B,)
    return torch.exp(-mean_sq)


class TrustReliabilityEstimator(nn.Module):
    """TRE — per-modality trust scores R_m + reliability loss L_trust.

    Parameters
    ----------
    num_modalities : number of modalities M (3 for V/A/P).
    agreement_to_unit : if True, map cosine agreement from [-1, 1] to [0, 1]
        as (A_m + 1) / 2 when building the pseudo-target R̂_m (Eq. 6a) so the
        regression target is a valid probability. The raw A_m still feeds Eq. 5.
    """

    def __init__(self, num_modalities: int = 3, agreement_to_unit: bool = True):
        super().__init__()
        self.num_modalities = num_modalities
        self.agreement_to_unit = agreement_to_unit
        # w1, w2, w3 and bias b from Eq. 5 — shared across modalities so a single
        # learned reliability rule applies to V/A/P.
        self.w = nn.Parameter(torch.ones(3))   # [w1, w2, w3]
        self.b = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        modality_feats: Dict[str, torch.Tensor],
        modality_feats_temporal: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute R_m for every modality and the reliability loss.

        modality_feats : {name: (B, N_M, d)} pooled feature sequences.
        modality_feats_temporal : optional {name: (B, T, d)} per-timestep
            prototypes for Eq. 4. If a modality is missing here, its temporal
            stability is treated as neutral (1.0).

        Returns dict with:
            trust     : {name: (B,)} trust scores R_m
            C, A, T   : {name: (B,)} the three components
            loss      : scalar L_trust (Eq. 6)
        """
        names = list(modality_feats.keys())
        protos = [modality_prototype(modality_feats[n]) for n in names]
        B = protos[0].size(0)
        device = protos[0].device

        C = {n: intra_modal_consistency(modality_feats[n], protos[i])
             for i, n in enumerate(names)}

        agreements = cross_modal_agreement(protos)
        A = {n: agreements[i] for i, n in enumerate(names)}

        T: Dict[str, torch.Tensor] = {}
        for n in names:
            pt = None if modality_feats_temporal is None else modality_feats_temporal.get(n)
            if pt is not None and pt.size(1) >= 2:
                T[n] = temporal_stability(pt)
            else:
                # undefined without >=2 timesteps -> neutral, does not fabricate
                T[n] = torch.ones(B, device=device)

        w1, w2, w3 = self.w[0], self.w[1], self.w[2]
        trust: Dict[str, torch.Tensor] = {}
        pseudo: Dict[str, torch.Tensor] = {}
        for n in names:
            trust[n] = torch.sigmoid(w1 * C[n] + w2 * A[n] + w3 * T[n] + self.b)  # Eq. 5
            a_hat = (A[n] + 1.0) / 2.0 if self.agreement_to_unit else A[n]
            pseudo[n] = (C[n] + a_hat + T[n]) / 3.0                               # Eq. 6a

        # Eq. 6 — mean over modalities of (R_m − R̂_m)²
        losses = torch.stack([(trust[n] - pseudo[n].detach()) ** 2 for n in names], dim=0)
        loss_trust = losses.mean()

        return {"trust": trust, "C": C, "A": A, "T": T, "loss": loss_trust}


class TrustAwareSparseGate(nn.Module):
    """TSF gate — Eq. 7: F̂_m = R_m · F̃_m if R_m > τ else 0.

    Applied to each modality's projected features before cross-modal attention.
    A hard step at τ is used at eval; a straight-through smooth gate is used in
    training so gradients still reach the TRE parameters.
    """

    def __init__(self, tau: float = 0.5, hard_gate: bool = False):
        super().__init__()
        self.tau = tau
        self.hard_gate = hard_gate

    def forward(self, feats: torch.Tensor, trust: torch.Tensor) -> torch.Tensor:
        """feats: (B, N_M, d)   trust: (B,)   ->   gated (B, N_M, d)."""
        r = trust.view(-1, 1, 1)                       # (B,1,1)
        keep = (trust > self.tau).float().view(-1, 1, 1)
        if self.hard_gate or not self.training:
            gate = keep * r                            # hard: exactly Eq. 7
        else:
            # straight-through: forward = hard gate, backward = smooth (r) so
            # TRE keeps receiving gradient even for suppressed modalities.
            soft = r
            gate = soft + (keep * r - soft).detach()
        return feats * gate
