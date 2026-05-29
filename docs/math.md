# Mathematical Modeling (Task 4)

Formal equations for the two core building blocks of Module 1: **Temporal
Feature Encoding** (the visual backbone) and **Cross-Attention Fusion** (the
multimodal aggregator). The notation mirrors the implementation so each symbol
maps to an identifier in code.

## Notation

| Symbol | Meaning | Code reference |
|---|---|---|
| $B$ | batch size | — |
| $T$ | number of sampled video frames per clip | `cfg.data.num_frames` |
| $H, W$ | frame height / width (after resize) | `cfg.data.frame_size` |
| $P$ | patch side length | `cfg.model.patch_size` |
| $N = (H/P)(W/P)$ | spatial patches per frame | — |
| $D$ | visual embedding dim | `cfg.model.embed_dim` |
| $L_s, L_t$ | spatial / temporal transformer depth | `spatial_depth`, `temporal_depth` |
| $D_a$ | audio embedding dim | `cfg.model.audio_embed_dim` |
| $T_a$ | audio token sequence length | output of `AudioEncoder` |
| $D_f$ | fused embedding dim | `cfg.model.fusion_dim` |
| $L_f$ | cross-attention depth | `cfg.model.fusion_depth` |
| $\tau$ | InfoNCE temperature | `AVSyncHead.temperature` |
| $\lambda$ | sync-loss weight | `cfg.train.sync_loss_weight` |

Throughout, $\mathrm{LN}(\cdot)$ is layer norm, $\mathrm{MLP}(x) = W_2\,\mathrm{GELU}(W_1 x)$,
and $\mathrm{MHA}_h(Q,K,V)$ is multi-head attention with $h$ heads.

## 1. Temporal Feature Encoding

Implementation: `models/temporal_vit.py`. Factorized space-time attention in the
spirit of TimeSformer: per-frame spatial ViT, then a temporal Transformer over
$T$ per-frame summaries.

### 1.1 Patch embedding

Given a clip $X \in \mathbb{R}^{T \times 3 \times H \times W}$, partition each
frame $X_t$ into $N$ non-overlapping patches $X_{t,n} \in \mathbb{R}^{3 \times P \times P}$
and linearly project them to dimension $D$:

$$
e_{t,n} \;=\; W_E\,\mathrm{vec}(X_{t,n}) + b_E, \quad n = 1,\dots,N.
$$

A learnable spatial class token $c^{\mathrm{sp}} \in \mathbb{R}^{D}$ is prepended,
and a learnable spatial positional embedding $P^{\mathrm{sp}} \in \mathbb{R}^{(N+1) \times D}$
is added:

$$
Z_t^{(0)} \;=\; \big[\,c^{\mathrm{sp}};\; e_{t,1};\;\dots;\; e_{t,N}\,\big] + P^{\mathrm{sp}}
\;\in\; \mathbb{R}^{(N+1) \times D}.
$$

### 1.2 Spatial Transformer (per frame)

For $\ell = 1, \dots, L_s$:

$$
\begin{aligned}
\tilde Z_t^{(\ell)} &= Z_t^{(\ell-1)} + \mathrm{MHA}\!\big(\mathrm{LN}(Z_t^{(\ell-1)})\big), \\
Z_t^{(\ell)}      &= \tilde Z_t^{(\ell)} + \mathrm{MLP}\!\big(\mathrm{LN}(\tilde Z_t^{(\ell)})\big).
\end{aligned}
$$

The per-frame summary is the spatial class token after the final block:

$$
f_t \;=\; \mathrm{LN}\!\big(Z_t^{(L_s)}\big)_{0} \;\in\; \mathbb{R}^{D}.
$$

### 1.3 Temporal Transformer (across frames)

Stack the $T$ per-frame summaries with a clip class token $c^{\mathrm{cl}}$ and
add a temporal positional embedding $P^{\mathrm{tm}} \in \mathbb{R}^{(T+1) \times D}$:

$$
F^{(0)} \;=\; \big[\,c^{\mathrm{cl}};\; f_1;\; \dots;\; f_T\,\big] + P^{\mathrm{tm}}_{:\,T+1}.
$$

For $\ell = 1, \dots, L_t$ apply the same residual MHA + MLP recipe as
in §1.2, producing $F^{(L_t)} \in \mathbb{R}^{(T+1) \times D}$.
The visual outputs of the backbone are:

$$
v^{\mathrm{cls}} \;=\; F^{(L_t)}_{0}, \qquad
V \;=\; F^{(L_t)}_{1:T+1} \;\in\; \mathbb{R}^{T \times D}.
$$

$v^{\mathrm{cls}}$ is a clip-level summary; $V$ is the per-frame token sequence
used by the fusion stage and the AV-sync head.

## 2. Audio Encoder

Implementation: `models/audio_encoder.py`. Given a mono waveform
$w \in \mathbb{R}^{S}$ at sample rate $r$,

$$
M \;=\; \mathrm{MelSpec}_{n_{\mathrm{fft}},\,h,\,n_{\mathrm{mels}}}(w) \;\in\; \mathbb{R}^{n_{\mathrm{mels}} \times T'},
\qquad \tilde M \;=\; \mathrm{AmpToDB}(M).
$$

A 1-D CNN stack $\Phi_{\mathrm{cnn}}$ (five Conv1D + BN + GELU blocks with two
strided downsamples) produces

$$
A \;=\; \Phi_{\mathrm{cnn}}(\tilde M)^{\top} \;\in\; \mathbb{R}^{T_a \times D_a},
\qquad a^{\mathrm{cls}} \;=\; \tfrac{1}{T_a}\!\sum_{j=1}^{T_a} A_j.
$$

## 3. Cross-Attention Fusion

Implementation: `models/cross_attention_fusion.py`. Bidirectional cross-attention
between the visual sequence $V$ and the audio sequence $A$.

### 3.1 Input projection

Project both modalities into the shared fusion dimension $D_f$ and add learnable
positional ($P^V, P^A$) and modality ($m_V, m_A$) embeddings:

$$
V^{(0)} \;=\; V W_V + P^V_{:\,T} + m_V, \qquad
A^{(0)} \;=\; A W_A + P^A_{:\,T_a} + m_A.
$$

### 3.2 Bidirectional cross-attention block

For $\ell = 1, \dots, L_f$, given $V^{(\ell-1)}$ and $A^{(\ell-1)}$:

$$
\begin{aligned}
V' &= V^{(\ell-1)} + \mathrm{MHA}\!\Big(Q=\mathrm{LN}(V^{(\ell-1)}),\; K=\mathrm{LN}(A^{(\ell-1)}),\; V=\mathrm{LN}(A^{(\ell-1)});\; M_A\Big), \\[2pt]
A' &= A^{(\ell-1)} + \mathrm{MHA}\!\Big(Q=\mathrm{LN}(A^{(\ell-1)}),\; K=\mathrm{LN}(V'),\; V=\mathrm{LN}(V')\Big), \\[2pt]
V^{(\ell)} &= V' + \mathrm{MLP}\!\big(\mathrm{LN}(V')\big), \\[2pt]
A^{(\ell)} &= A' + \mathrm{MLP}\!\big(\mathrm{LN}(A')\big).
\end{aligned}
$$

$M_A \in \{0, -\infty\}^{B \times T_a}$ is a key-padding mask that suppresses
audio tokens for silent clips (`has_audio = 0`), so the video stream is not
contaminated by zero waveforms.

### 3.3 Pooled fusion

Let $\mathbb{1}_a \in \{0,1\}^{B}$ indicate whether each clip carries audio. The
clip-level fused embedding is

$$
\bar V \;=\; \tfrac{1}{T}\!\sum_{t=1}^{T} V^{(L_f)}_t, \qquad
\bar A \;=\; \mathbb{1}_a \odot \tfrac{1}{T_a}\!\sum_{j=1}^{T_a} A^{(L_f)}_j,
$$

$$
f \;=\; \mathrm{GELU}\!\Big(W_O\,[\bar V \,\Vert\, \bar A]\Big) \;\in\; \mathbb{R}^{D_f}.
$$

### 3.4 Audio-visual synchronization head

Implementation: `models/av_sync.py`. Project the visual and audio token
sequences from the backbone (not the fusion outputs) and align in time:

$$
\hat V \;=\; \mathrm{Norm}\!\big(g_V(V)\big), \qquad
\hat A \;=\; \mathrm{Norm}\!\big(g_A(\mathrm{Resample}_T(A))\big),
$$

where $\mathrm{Resample}_T$ is linear interpolation along the time axis to
length $T$, and Norm denotes $\ell_2$ normalization. The per-clip sync score is

$$
s \;=\; \frac{1}{T}\sum_{t=1}^{T} \langle \hat V_t,\,\hat A_t \rangle \;\in\; [-1, 1],
$$

and the InfoNCE alignment loss (over the flattened $BT \times BT$ logit matrix
with temperature $\tau$) is the symmetric average of the two cross-entropy
directions:

$$
\mathcal{L}_{\mathrm{sync}}
\;=\; -\tfrac{1}{2}\,\Big(
\underbrace{\tfrac{1}{BT}\!\sum_{i} \log
\frac{\exp(\langle \hat V_i, \hat A_i\rangle / \tau)}
     {\sum_{j} \exp(\langle \hat V_i, \hat A_j\rangle / \tau)}}_{\text{V}\to\text{A}}
\;+\;
\underbrace{\tfrac{1}{BT}\!\sum_{i} \log
\frac{\exp(\langle \hat A_i, \hat V_i\rangle / \tau)}
     {\sum_{j} \exp(\langle \hat A_i, \hat V_j\rangle / \tau)}}_{\text{A}\to\text{V}}
\Big).
$$

### 3.5 Classifier and joint objective

The final logits are produced by an MLP over the concatenation of the fused
embedding and the scalar sync score (so the classifier can directly leverage
lip-sync mismatch as a fake signal):

$$
\hat y \;=\; W_2\,\mathrm{GELU}\!\big(W_1\,[\,f \,\Vert\, s\,]\big) \;\in\; \mathbb{R}^{2}.
$$

Let $y \in \{0,1\}$ be the binary deepfake label. The training objective is

$$
\boxed{\;\mathcal{L}
\;=\;
\underbrace{\mathrm{CE}(\hat y,\,y)}_{\text{detection}}
\;+\;
\lambda\,\underbrace{\mathcal{L}_{\mathrm{sync}}}_{\text{multimodal alignment}}.\;}
$$

Module 2 will extend this objective with an adversarial-robustness term and a
diffusion-reconstruction term (cf. project plan §2.5).

## 4. Parameter count summary

For the defaults in `configs/default.yaml` ($D=384$, $L_s=6$, $L_t=4$,
$D_a=256$, $D_f=512$, $L_f=2$, frame size $224$, patch size $16$, $T=16$):

| Component | Approx. parameters |
|---|---|
| Patch embed | $3P^2 D \approx 0.22\text{M}$ |
| Spatial ViT ($L_s$ layers) | $\approx 10.7\text{M}$ |
| Temporal Transformer ($L_t$ layers) | $\approx 7.1\text{M}$ |
| Audio encoder (1-D CNN) | $\approx 0.6\text{M}$ |
| Cross-attention fusion ($L_f$ layers) | $\approx 4.2\text{M}$ |
| AV-sync head | $\approx 0.1\text{M}$ |
| Classifier | $\approx 0.27\text{M}$ |
| **Total** | **≈ 23 M** |

Pruning and quantization (Task 5) will target the spatial ViT and the fusion
blocks first, since they dominate this budget.
