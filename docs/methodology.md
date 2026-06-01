# 3. Methodology

## 3.1 Overview of the Proposed Framework

The proposed framework, **Adversarial Robust Real-Time Multimodal Deepfake
Detection using Temporal Cross-Attention Transformers and Diffusion-Based
Forensic Reconstruction**, integrates two coupled modules. Module 1 performs
real-time multimodal detection on synchronized video and audio streams using a
Temporal Vision Transformer with a cross-attention fusion head and an
audio-visual synchronization objective. Module 2 hardens the detector against
adversarial perturbations through attack-based stress testing, diffusion-based
forensic reconstruction, continual learning against distributional drift, and
formal robustness certification.

The end-to-end inference workflow is:

**Input Stream → Temporal Transformer Detection → Cross-Attention Fusion →
Adversarial Perturbation Analysis → Diffusion Forensic Reconstruction →
Continual-Memory Re-Verification → Trust Score**

The architecture aligns with the three-stage formulation of the proposed
system:

- **Stage 1: Primary Live Deepfake Detection.** Temporal ViT + audio encoder +
  cross-attention fusion produce an initial prediction and an audio-visual
  alignment signal.
- **Stage 2: Source Attribution and Contrastive Provenance Intelligence.**
  Adversarial perturbation analysis and failure attribution localize whether
  a low-confidence prediction stems from genuine manipulation, distributional
  drift, or adversarial noise.
- **Stage 3: Adversarial Forensic Recovery and Adaptive Resilience.**
  Diffusion-based reconstruction restores forensic artifacts; continual
  learning preserves accuracy on previously seen forgeries while adapting to
  new ones; and certified-radius analysis quantifies guaranteed robustness.

## 3.2 Dataset Preparation and Preprocessing

The framework is trained and evaluated on four benchmark datasets:

- **FaceForensics++** — high-quality manipulations (Deepfakes, Face2Face,
  FaceSwap, NeuralTextures).
- **Celeb-DF (v2)** — high-resolution celebrity face swaps.
- **DFDC (Deepfake Detection Challenge)** — large-scale, in-the-wild forgeries
  with audio.
- **FakeAVCeleb** — paired real/fake audio-visual content for cross-modal
  evaluation.

Each dataset is split into training, validation, and testing partitions in a
70:15:15 ratio. Splits are stratified by **identity** and **source video**
rather than by clip, eliminating identity leakage and providing a faithful
estimate of generalization to unseen subjects.

The preprocessing pipeline consists of:

1. **Frame extraction.** Video clips are decoded at a fixed sample rate; $T$
   uniformly spaced frames per clip are selected.
2. **Face detection and alignment.** Each frame is cropped around the detected
   face and resized to $H \times W$, with five-point landmark alignment to
   suppress nuisance pose variation.
3. **Audio extraction and resampling.** The audio track is decoded to mono at a
   fixed sample rate and time-aligned to the video clip boundaries.
4. **Log-Mel spectrogram generation.** A short-time Fourier transform with
   $n_{\mathrm{mels}}$ Mel bands is applied, followed by amplitude-to-decibel
   conversion.
5. **Audio-video synchronization.** A common clip clock guarantees temporal
   correspondence so the cross-attention and AV-sync heads receive paired
   tokens.

The pipeline supports both offline preprocessed tensors (for training) and a
streaming loader (for the real-time evaluation track), ensuring deployment
parity between batch experiments and live inference.

## 3.3 Temporal Multimodal Detection Module

To capture temporal inconsistencies introduced by deepfake generation, a
**Temporal Vision Transformer (TVT)** with factorized space–time attention is
employed.

Given a clip $X \in \mathbb{R}^{T \times 3 \times H \times W}$, each frame is
partitioned into $N = (H/P)(W/P)$ non-overlapping patches and linearly
projected to embedding dimension $D$:

$$
e_{t,n} \;=\; W_E\,\mathrm{vec}(X_{t,n}) + b_E, \qquad n = 1,\dots,N.
$$

A learnable spatial class token and a spatial positional embedding are
prepended/added, producing the initial token sequence $Z_t^{(0)}$. The spatial
Transformer of depth $L_s$ applies the standard residual recipe at each layer
$\ell$:

$$
\begin{aligned}
\tilde Z_t^{(\ell)} &= Z_t^{(\ell-1)} + \mathrm{MHA}\!\big(\mathrm{LN}(Z_t^{(\ell-1)})\big), \\
Z_t^{(\ell)}      &= \tilde Z_t^{(\ell)} + \mathrm{MLP}\!\big(\mathrm{LN}(\tilde Z_t^{(\ell)})\big).
\end{aligned}
$$

The per-frame summary $f_t = \mathrm{LN}(Z_t^{(L_s)})_0$ is taken as the
spatial class-token output. The temporal Transformer of depth $L_t$ then
processes the sequence $[c^{\mathrm{cl}}; f_1, \dots, f_T] + P^{\mathrm{tm}}$
with the same residual MHA/MLP blocks, producing a clip-level embedding
$v^{\mathrm{cls}}$ and a temporally-contextualized per-frame sequence $V$.
Temporal attention is computed as:

$$
\mathrm{Attention}(Q,K,V) \;=\; \mathrm{Softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
$$

This factorization allows the model to learn frame-to-frame irregularities —
abnormal blinking, facial flickering, blending boundary artifacts, and motion
inconsistencies — that single-frame approaches systematically miss.

## 3.4 Audio Encoder and Cross-Attention Multimodal Fusion

The audio stream is encoded by a 1-D CNN backbone $\Phi_{\mathrm{cnn}}$ applied
to log-Mel spectrograms:

$$
A \;=\; \Phi_{\mathrm{cnn}}(\tilde M)^{\top} \;\in\; \mathbb{R}^{T_a \times D_a},
\qquad
a^{\mathrm{cls}} \;=\; \tfrac{1}{T_a}\!\sum_{j=1}^{T_a} A_j.
$$

Visual tokens $V$ and audio tokens $A$ are projected into a shared fusion
dimension $D_f$ and fused by **bidirectional cross-attention**. Letting $F_v$
denote visual embeddings and $F_a$ denote audio embeddings, the canonical
cross-attention operator is

$$
CA(F_v, F_a) \;=\; \mathrm{Softmax}\!\left(\frac{F_v F_a^\top}{\sqrt{d}}\right) F_a,
$$

and in each fusion layer the visual stream attends to the audio stream and
vice versa, with residual connections and feed-forward MLPs after each
attention sub-layer. A key-padding mask suppresses contribution from silent
clips so the video stream is not contaminated by zero-energy audio.

The clip-level fused embedding is the projected concatenation of the
modality-pooled features:

$$
\bar V = \tfrac{1}{T}\!\sum_{t=1}^{T} V^{(L_f)}_t, \quad
\bar A = \mathbb{1}_a \odot \tfrac{1}{T_a}\!\sum_{j=1}^{T_a} A^{(L_f)}_j, \quad
f = \mathrm{GELU}\!\big(W_O\,[\bar V \,\Vert\, \bar A]\big).
$$

### 3.4.1 Audio-Visual Synchronization Head

To explicitly exploit lip-sync inconsistency as a discriminative cue, an
**AV-sync head** projects the backbone visual and resampled audio sequences
to a shared embedding space, $\ell_2$-normalizes them, and computes the
per-clip cosine alignment

$$
s \;=\; \tfrac{1}{T}\sum_{t=1}^{T} \langle \hat V_t, \hat A_t \rangle \;\in\; [-1, 1].
$$

Training uses a symmetric **InfoNCE** objective with temperature $\tau$, where
positives are time-aligned visual/audio token pairs and negatives are drawn
across the batch. The scalar sync score $s$ is concatenated to $f$ and passed
to the classifier:

$$
\hat y \;=\; W_2\,\mathrm{GELU}\!\big(W_1\,[f \,\Vert\, s]\big) \;\in\; \mathbb{R}^{2}.
$$

The joint Module 1 objective is

$$
\mathcal{L} \;=\; \mathrm{CE}(\hat y, y) \;+\; \lambda\,\mathcal{L}_{\mathrm{sync}},
$$

combining the detection cross-entropy with the alignment loss weighted by
$\lambda$.

## 3.5 Real-Time Optimization

Because the framework targets live-stream verification, three optimization
techniques are applied **after** training:

- **Structured pruning** of attention heads and MLP channels in the spatial
  ViT and fusion blocks, guided by magnitude- and Taylor-importance criteria.
- **INT8 quantization** of linear and convolutional layers via
  post-training calibration.
- **Lightweight inference optimization** — token reduction, KV-cache reuse
  across overlapping clips in the streaming loader, and operator fusion.

Real-time performance is reported using:

- Accuracy, F1-score, AUC (detection quality),
- **Frames Per Second (FPS)** and **inference latency** (real-time capability),
- Memory footprint and energy per inference.

These optimizations target deployment on commodity GPUs and edge accelerators
without sacrificing detection quality beyond a pre-specified tolerance.

## 3.6 Adversarial Attack Generation and Failure Analysis

To stress-test the detector, adversarial samples are generated under a
white-box, visual-only threat model using four attacks of increasing
strength:

- **FGSM** (Fast Gradient Sign Method) — single-step $\ell_\infty$ attack.
- **PGD** (Projected Gradient Descent) — multi-step $\ell_\infty / \ell_2$
  attack with random restarts.
- **Carlini-Wagner (C&W)** — margin-based $\ell_2$ optimization attack.
- **DeepFool** — minimal-norm decision-boundary attack.

For FGSM, the adversarial example is

$$
x_{\mathrm{adv}} \;=\; x \;+\; \varepsilon\,\mathrm{sign}\!\big(\nabla_x L(F_\theta(x), y)\big),
$$

where $\varepsilon$ controls perturbation strength. To improve resilience,
adversarial training is performed with the standard min–max objective

$$
\min_{\theta}\; \mathbb{E}_{(x,y)}\!\left[\,\max_{\|\delta\|\le\varepsilon}
L\!\big(F_\theta(x + \delta),\, y\big)\right].
$$

### 3.6.1 Failure Analysis

For each attack family, the framework characterizes detector vulnerability
along four axes:

1. **Perturbation sensitivity:** accuracy as a function of $\varepsilon$, and
   minimum $\varepsilon$ for which the prediction flips.
2. **Confidence collapse:** distribution shift of softmax confidence between
   clean and adversarial samples (Wasserstein distance, ECE).
3. **Spatial attribution:** Grad-CAM and integrated-gradient maps to localize
   where attacks deposit perturbation budget, identifying brittle regions
   (typically peri-oral and peri-ocular).
4. **Cross-attack transferability:** matrix of attack-source vs. evaluated
   model accuracies, quantifying whether defenses against one attack
   generalize to others.

These analyses feed both the adversarial-training schedule and the
reconstruction trigger of §3.7.

## 3.7 Diffusion-Based Forensic Reconstruction

When perturbation analysis flags a sample as likely adversarial — either by
low margin, large softmax-confidence shift, or high-frequency anomaly score —
the sample is routed to a **diffusion-based forensic reconstruction**
module. The forward diffusion process gradually injects Gaussian noise to a
chosen timestep $t^\star$:

$$
q(x_t \mid x_{t-1}) \;=\; \mathcal{N}\!\big(\sqrt{1-\beta_t}\,x_{t-1},\, \beta_t I\big),
$$

with variance schedule $\{\beta_t\}_{t=1}^{T}$ (linear or cosine). A learned
denoiser $\epsilon_\phi$ parameterizes the reverse process

$$
p_\phi(x_{t-1} \mid x_t) \;=\; \mathcal{N}\!\big(\mu_\phi(x_t, t),\, \Sigma_\phi(x_t, t)\big),
$$

which is iterated from $t^\star$ back to $0$ to obtain the purified sample
$\hat x$. Intuitively, the noise injection overwhelms the high-frequency
adversarial perturbation while preserving the low-frequency semantic content;
the learned reverse process then restores forensic artifacts (blending
boundaries, frequency-domain footprints) that the attack tried to mask.

The reconstruction timestep $t^\star$ is the central trade-off: small values
under-purify, large values destroy genuine forgery cues. We tune it on a
held-out adversarial validation set to maximize re-verification accuracy.

## 3.8 Continual Learning Module

Deepfake generators evolve rapidly. To remain effective against emerging
forgery families, a continual learning mechanism is incorporated with three
components:

- **Memory replay** — a fixed-capacity buffer of representative clips from
  past generators, sampled at every update step to anchor prior knowledge.
- **Adaptive fine-tuning** with **Elastic Weight Consolidation (EWC)** — a
  regularizer
  $\mathcal{L}_{\mathrm{EWC}} = \tfrac{\lambda}{2} \sum_i F_i (\theta_i - \theta^*_i)^2$
  penalizing drift away from parameters that were important for previous
  tasks, with importance $F_i$ given by the Fisher information.
- **Online distributional drift detection** — a statistical monitor over
  input features and prediction confidences that triggers an update only when
  drift is detected, preventing unnecessary forgetting from stable streams.

This combination preserves backward transfer on prior generators while
admitting new forgery families incrementally.

## 3.9 Mathematical Robustness Modeling

The framework provides two complementary families of formal robustness
guarantees.

### 3.9.1 Lipschitz-Margin Certified Radius (Deterministic)

For an $L$-Lipschitz classifier $f_\theta$ with logit margin
$m(x) = f_\theta(x)_{\hat y} - \max_{j \ne \hat y} f_\theta(x)_j$, the
prediction is provably constant on the $\ell_2$-ball of radius

$$
r(x) \;=\; \frac{m(x)}{\sqrt{2}\, L}.
$$

Per-layer Lipschitz constants are computed by spectral norms (linear/conv via
power iteration), activation constants (1 for ReLU, $\approx 1.13$ for GELU,
etc.), and affine norm parameters; the network bound is the product. Because
self-attention is not globally Lipschitz, the certificate is reported over the
Lipschitz-bounded sub-network.

The empirical **certified accuracy curve** is

$$
\mathrm{CA}(r) \;=\; \frac{1}{N}\sum_{i=1}^{N}
\mathbb{1}\!\big\{\hat y_i = y_i \;\wedge\; r(x_i) \ge r\big\}.
$$

### 3.9.2 Randomized Smoothing (Probabilistic)

For Gaussian noise $\eta \sim \mathcal{N}(0, \sigma^2 I)$, the smoothed
classifier

$$
g(x) \;=\; \arg\max_{c}\; \mathbb{P}_\eta\!\big[f_\theta(x + \eta) = c\big]
$$

admits the Cohen–Rosenfeld–Kolter certificate: if $\underline{p_A}$ is a
Clopper–Pearson lower bound at confidence $1 - \alpha$ on the top-class
probability, then $g(x + \delta) = g(x)$ for all $\|\delta\|_2 \le R(x)$ with

$$
R(x) \;=\; \sigma\,\Phi^{-1}(\underline{p_A}).
$$

Monte Carlo certification uses sample-splitting (selection set then
certification set) and returns ABSTAIN when $\underline{p_A} < 1/2$.

### 3.9.3 Risk Decomposition

Following the natural/adversarial/boundary decomposition,

$$
R_{\mathrm{adv}}(f, \varepsilon) \;=\; R_{\mathrm{nat}}(f) \;+\; R_{\mathrm{bd}}(f, \varepsilon),
$$

the empirical estimator using attack $A_\varepsilon$ from §3.6 is a
**lower bound** on $R_{\mathrm{adv}}$; the gap is the attack-quality gap
and is shrunk by stronger adversaries. The accuracy–robustness trade-off
curve over $\varepsilon$ closes the loop between empirical attacks (§3.6),
diffusion purification (§3.7), continual updates (§3.8), and certified
radii (§3.9).

## 3.10 Re-Verification and Trust Score Generation

After diffusion reconstruction, the purified sample $\hat x$ is re-evaluated
through the Module 1 detector. The final trust score fuses the original and
post-reconstruction predictions:

$$
S \;=\; (1 - \omega)\big(1 - p(x)\big) \;+\; \omega\big(1 - p(\hat x)\big),
$$

where $p(x)$ is the original deepfake probability, $p(\hat x)$ is the
post-reconstruction probability, and $\omega \in [0, 1]$ is a reconstruction-
confidence weight set by the diffusion module (high when adversarial-noise
indicators were strong; low for clean-looking inputs). A decision threshold
on $S$ separates authentic from manipulated content.

## 3.11 Experimental Evaluation Protocol

A unified evaluation orchestrator (`scripts/robustness_eval.py`) ties all of
the above modules into a single end-to-end **robustness ledger** —
a JSON report plus a paper-ready Markdown summary — exercising the full
pipeline of §3.1 in one pass. The framework is evaluated under three metric
families.

**Detection metrics:**

- Accuracy, Precision, Recall, F1-score,
- AUC-ROC and AUC-PR (the latter important under class imbalance).

**Real-time metrics:**

- Frames Per Second (FPS),
- End-to-end inference latency (median and 95th percentile),
- Memory footprint and energy per inference.

**Robustness metrics:**

- Adversarial accuracy under FGSM / PGD / C&W / DeepFool at increasing
  perturbation budgets,
- Perturbation resilience (minimum $\varepsilon$ to flip prediction),
- Reconstruction quality (PSNR, SSIM) of the diffusion module,
- Certified accuracy at radius $r$ (deterministic) and at radius $R$
  (smoothed),
- Robustness AUC (area under the certified-accuracy curve),
- Backward-transfer accuracy and forgetting after continual updates.

**Ablation studies** isolate the contribution of each architectural
component: (i) temporal attention (vs. frame-averaging), (ii) cross-attention
fusion (vs. late concatenation), (iii) the AV-sync auxiliary loss,
(iv) adversarial training, (v) diffusion-based reconstruction, and
(vi) continual learning. Each component is ablated independently and in
combination to attribute the system's gains.

Cross-dataset generalization is evaluated by training on one benchmark
(e.g. FaceForensics++) and testing on others (Celeb-DF, DFDC, FakeAVCeleb)
to expose dataset-specific shortcut learning.

---

In combination, the modules above instantiate a detector that is
*accurate* (multimodal temporal attention with explicit AV alignment),
*real-time* (pruned and quantized inference), *resilient* (adversarial
training plus diffusion-based forensic recovery), *adaptive* (continual
learning with replay and EWC), and *trustworthy* (certified radii and a
fused trust score), addressing the limitations of prior single-objective
approaches summarized in the related-work comparison.
