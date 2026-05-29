# Mathematical Robustness Modeling (Module 2 Task 5)

Formal robustness theory for the multimodal detector, with implementation
cross-references in `robustness/`. Two complementary families of guarantees
are treated:

* **Deterministic, margin-based.** Bounds obtained by combining a logit
  margin with a global Lipschitz constant. Exact but conservative; the
  upper bound on $L$ for transformer self-attention is unbounded in general,
  so this family certifies a *sub-network* (the Lipschitz-bounded layers
  only). See `robustness/lipschitz.py`, `robustness/margins.py`.
* **Probabilistic, smoothing-based.** Randomized smoothing (Cohen et al.,
  ICML 2019) gives a per-sample $\ell_2$ certificate that holds with
  confidence $1 - \alpha$ for the *smoothed* classifier $g$, with no
  Lipschitz assumption on $f$. See `robustness/randomized_smoothing.py`.

Empirical risk decomposition (Tsipras et al., Madry et al.) connects both
of the above to attack-based metrics from Module 2 Task 1/2 (see
`robustness/bounds.py`).

## Notation

| Symbol | Meaning | Code reference |
|---|---|---|
| $f_\theta : \mathcal{X} \to \mathbb{R}^K$ | logit classifier | `MultimodalDeepfakeDetector` |
| $\hat y(x) = \arg\max_c f_\theta(x)_c$ | predicted class | — |
| $m(x) = f_\theta(x)_{\hat y} - \max_{j \ne \hat y} f_\theta(x)_j$ | logit margin | `compute_margins` |
| $L$ | global L2 Lipschitz constant of $f_\theta$ | `LipschitzEstimator.product_bound` |
| $\sigma$ | smoothing noise standard deviation | `SmoothedClassifier.sigma` |
| $\eta \sim \mathcal{N}(0, \sigma^2 I)$ | smoothing noise | — |
| $g(x) = \arg\max_c \mathbb{P}_\eta[f_\theta(x + \eta) = c]$ | smoothed classifier | `SmoothedClassifier` |
| $R_\mathrm{nat}, R_\mathrm{adv}, R_\mathrm{bd}$ | natural / adversarial / boundary risk | `bounds.py` |

## 1. Lipschitz-based deterministic certificate

A function $f$ is $L$-Lipschitz in the $\ell_2$ norm if
$$
\|f(x) - f(y)\|_2 \le L \,\|x - y\|_2, \quad \forall x, y. \tag{1}
$$
For a feedforward composition $f = f_L \circ \dots \circ f_1$ with per-layer
constants $L_i$, eq. (1) gives the product bound
$$
L \le \prod_{i=1}^{L_\mathrm{net}} L_i. \tag{2}
$$

### 1.1 Per-layer constants

| Layer | $L_i$ | Implementation |
|---|---|---|
| Linear $W$ | $\|W\|_2$ (largest singular value) | `linear_spectral_norm` |
| Conv2d / Conv1d $K$ | largest singular value of $u \mapsto K \ast u$ | `conv_spectral_norm` (power iteration) |
| LayerNorm/GroupNorm/BN (affine) | $\|\gamma\|_\infty$ | `_norm_lipschitz` |
| ReLU / Leaky-ReLU | $1$ | `_ACTIVATION_LIPSCHITZ` |
| GELU | $\approx 1.1289$ | as above |
| SiLU | $\approx 1.10$ | as above |
| Sigmoid | $1/4$ | as above |
| Tanh | $1$ | as above |
| **MultiheadAttention** | **unbounded** in general | flagged in `LipschitzEstimator.unbounded_names` |

The self-attention block
$\mathrm{Att}(Q, K, V) = \mathrm{softmax}(QK^\top / \sqrt{d}) V$
is *not* globally Lipschitz: the softmax Jacobian can grow without bound
when two queries are well-separated and the temperature $d^{-1/2}$ is fixed
(Kim, Papamakarios, Mnih, 2021). We therefore report Lipschitz bounds over
the **bounded subset** of the network and treat the result as a
sub-network certificate.

### 1.2 Margin-based certified radius

For a $K$-class $L$-Lipschitz classifier, the worst-case change in the
$(c, j)$-margin under an $\ell_2$ perturbation $\delta$ is at most
$\sqrt{2} L \,\|\delta\|_2$ (Tsuzuku, Sato, Sugiyama, NeurIPS 2018, Lemma 1).
Therefore the prediction is constant on the open $\ell_2$-ball of radius
$$
\boxed{\;r(x) \;=\; \frac{m(x)}{\sqrt{2}\, L},\;} \tag{3}
$$
and the empirical **certified accuracy curve** is
$$
\mathrm{CA}(r) \;=\; \frac{1}{N}\sum_{i=1}^{N}
\mathbb{1}\big\{\hat y_i = y_i \;\wedge\; r(x_i) \ge r \big\}. \tag{4}
$$
Implementation: `certified_accuracy_curve` in `robustness/margins.py`.

## 2. Randomized smoothing (probabilistic certificate)

For Gaussian noise $\eta \sim \mathcal{N}(0, \sigma^2 I)$ define the smoothed
classifier
$$
g(x) \;=\; \arg\max_{c \in \mathcal{Y}} \;
\underbrace{\mathbb{P}_\eta\big[f_\theta(x + \eta) = c\big]}_{p_c(x)}. \tag{5}
$$

**Theorem (Cohen, Rosenfeld, Kolter, ICML 2019).** Let $c_A = g(x)$ and let
$\underline{p_A}, \overline{p_B}$ be lower / upper bounds on the top-two
probabilities $p_{c_A}(x)$ and $\max_{c \ne c_A} p_c(x)$ respectively.
If $\underline{p_A} \ge \overline{p_B}$, then for any $\delta$ with
$\|\delta\|_2 \le R(x)$,
$$
g(x + \delta) \;=\; c_A,
\qquad
R(x) \;=\; \frac{\sigma}{2}\big(\Phi^{-1}(\underline{p_A}) - \Phi^{-1}(\overline{p_B})\big). \tag{6}
$$
With the simplification $\overline{p_B} \le 1 - \underline{p_A}$,
$$
\boxed{\;R(x) \;=\; \sigma \, \Phi^{-1}(\underline{p_A}).\;} \tag{7}
$$

### 2.1 Monte Carlo certification

We estimate $\underline{p_A}$ as a one-sided Clopper–Pearson lower bound on
$n_A / n$ at confidence $1 - \alpha$, where $n_A$ is the number of times the
top class was returned out of $n$ noisy samples. Sample-splitting is used:
$n_0$ samples select $c_A$, and $n$ further samples certify it
(Cohen et al., Algorithm 2). The procedure returns `ABSTAIN` when
$\underline{p_A} < 1/2$ (no certificate is possible).

Implementation: `SmoothedClassifier.certify` in
`robustness/randomized_smoothing.py`.

### 2.2 Threat model

Smoothing in our pipeline acts on the **visual modality** only:
$x + \eta$ refers to the frame tensor, with audio passed through clean.
This matches the attack threat model from Module 2 Task 1/2 (visual-only
perturbations) so the certified radius is directly comparable to the
empirical $\varepsilon$ values used there.

## 3. Adversarial risk decomposition

Following Tsipras, Santurkar, Engstrom, Turner, Madry (ICLR 2019) and
Madry, Makelov, Schmidt, Tsipras, Vladu (ICLR 2018):

| Quantity | Definition |
|---|---|
| Natural risk      | $R_\mathrm{nat}(f) \;=\; \mathbb{E}_{(x, y)}\big[\mathbb{1}\{f(x) \ne y\}\big]$ |
| Adversarial risk  | $R_\mathrm{adv}(f, \varepsilon) \;=\; \mathbb{E}_{(x, y)}\!\big[\sup_{\|\delta\| \le \varepsilon} \mathbb{1}\{f(x + \delta) \ne y\}\big]$ |
| Boundary risk     | $R_\mathrm{bd}(f, \varepsilon) \;=\; R_\mathrm{adv}(f, \varepsilon) - R_\mathrm{nat}(f) \;\ge\; 0$ |

The empirical estimator for $R_\mathrm{adv}$ replaces the supremum with a
concrete attack $A_\varepsilon$:
$$
\widehat R_\mathrm{adv}(f, \varepsilon) \;=\;
\frac{1}{N}\sum_{i=1}^{N}\mathbb{1}\{f(A_\varepsilon(x_i, y_i)) \ne y_i\}
\;\le\; R_\mathrm{adv}(f, \varepsilon). \tag{8}
$$
Equation (8) is a **lower bound**, since $A_\varepsilon$ is not the
worst-case adversary. The gap $R_\mathrm{adv} - \widehat R_\mathrm{adv}$ is
the *attack quality gap*; running stronger attacks (PGD with more steps,
multi-start CW) tightens it.

Implementation: `bounds.py` exposes `natural_risk`, `adversarial_risk`,
`risk_decomposition`, and `accuracy_robustness_tradeoff`. The last builds a
$\varepsilon \mapsto (R_\mathrm{nat}, R_\mathrm{adv}, R_\mathrm{bd})$ table
suitable for the trade-off plots used in §4.

## 4. Accuracy–robustness trade-off

Tsipras et al. (Theorem 2.1) construct a data distribution on which
$R_\mathrm{nat}$ can be made arbitrarily small while
$R_\mathrm{adv}(\varepsilon)$ stays $\ge 1/2$ for any fixed $\varepsilon$
exceeding a data-dependent threshold — i.e., *natural and adversarial
accuracy are fundamentally at odds*. The accuracy–robustness trade-off
curve produced by `accuracy_robustness_tradeoff` is the empirical
counterpart to this result; for hardened detectors the curve flattens
(smaller drop in $R_\mathrm{adv}$ for increased $\varepsilon$) at the cost
of a small uplift in $R_\mathrm{nat}$.

## 5. Composition with Module 2 pipeline

The mathematical model assembles as follows:

1. **Module 2 Task 1/2 (attacks + failure analysis)** supplies an empirical
   $\widehat R_\mathrm{adv}(\varepsilon)$ curve. Combined with
   $R_\mathrm{nat}$ this fixes the trade-off curve of §4.
2. **Module 2 Task 3 (diffusion recovery)** modifies the classifier to
   $f' = f \circ \mathrm{Purify}_{t^\star}$. Purification adds an
   $\ell_2$-bounded denoising step; substituting $f'$ into eqs. (3) and
   (7) yields the **post-recovery** certified radius. Empirically
   `scripts/run_recovery.py` reports `accuracy_raw` vs `accuracy_recovered`,
   which corresponds to $\widehat R_\mathrm{adv}(f, \varepsilon) -
   \widehat R_\mathrm{adv}(f', \varepsilon)$.
3. **Module 2 Task 4 (continual learning)** introduces task-indexed
   classifiers $\{f^{(1)}, \dots, f^{(T)}\}$. Each can be certified
   independently with (3) or (7); the **backward-transfer–certified
   radius** for task $k$ after training $T$ is $R^{(T)}(x)$ for
   $x$ sampled from task $k$, exposing whether continual updates degrade
   prior-task certificates.
4. **This task (Task 5)** produces the standalone certificates and the
   risk decomposition. Together with Tasks 1–4 they form the full
   robustness ledger used in the final evaluation (Task 6).

## 6. References

* Cohen, Rosenfeld, Kolter. *Certified Adversarial Robustness via
  Randomized Smoothing*. ICML 2019.
* Tsuzuku, Sato, Sugiyama. *Lipschitz-Margin Training: Scalable
  Certification of Perturbation Invariance for Deep Neural Networks*.
  NeurIPS 2018.
* Madry, Makelov, Schmidt, Tsipras, Vladu. *Towards Deep Learning Models
  Resistant to Adversarial Attacks*. ICLR 2018.
* Tsipras, Santurkar, Engstrom, Turner, Madry. *Robustness May Be at Odds
  with Accuracy*. ICLR 2019.
* Kim, Papamakarios, Mnih. *The Lipschitz Constant of Self-Attention*.
  ICML 2021.
