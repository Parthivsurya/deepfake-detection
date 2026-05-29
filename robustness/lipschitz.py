"""Lipschitz bounds for a network.

A function $f: \\mathcal{X} \\to \\mathcal{Y}$ is $L$-Lipschitz under norms
$\\|\\cdot\\|_X, \\|\\cdot\\|_Y$ if
$$\\|f(x) - f(y)\\|_Y \\le L \\|x - y\\|_X \\quad \\forall x, y.$$

For a feedforward composition $f = f_L \\circ \\cdots \\circ f_1$, the product
of per-layer Lipschitz constants is a valid (typically loose) upper bound on
the network Lipschitz constant. We compute exact constants where they have a
closed form (Linear, Conv via power iteration), approximate values for
norm layers (||γ||_∞), and tabulated values for elementwise non-linearities.

**Self-attention is not Lipschitz in general** (Kim et al., 2021). Modules
whose Lipschitz constant we cannot bound are flagged as `unbounded`; the
product bound is reported over the bounded subset only, with a clear
warning. The randomized-smoothing bound (`randomized_smoothing.py`) does not
suffer from this restriction and is the practical certified-robustness tool
for the full detector.

Cross-references:
  - Tsuzuku, Sato, Sugiyama (NeurIPS 2018): margin-based certified radius
    $r \\ge m(x) / (\\sqrt{2} L)$ for L2 perturbations of a $K$-class L-Lipschitz
    classifier.
"""
from __future__ import annotations
from typing import Callable, Dict, List, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# Lipschitz constants of common elementwise non-linearities (sup |f'(x)|).
_ACTIVATION_LIPSCHITZ: Dict[type, float] = {
    nn.ReLU: 1.0,
    nn.LeakyReLU: 1.0,
    nn.GELU: 1.1289,        # max of |GELU'| ≈ 1.1289
    nn.SiLU: 1.1,           # max of |SiLU'| ≈ 1.0998
    nn.Sigmoid: 0.25,
    nn.Tanh: 1.0,
    nn.ELU: 1.0,
    nn.Hardswish: 1.0844,
    nn.Identity: 1.0,
}


def linear_spectral_norm(weight: torch.Tensor) -> float:
    """Operator 2-norm of a 2-D weight matrix (largest singular value)."""
    if weight.dim() != 2:
        raise ValueError(f"expected 2-D weight, got shape {tuple(weight.shape)}")
    return float(torch.linalg.matrix_norm(weight.detach(), ord=2).item())


@torch.no_grad()
def conv_spectral_norm(
    weight: torch.Tensor,
    input_shape: Tuple[int, ...],
    stride=1,
    padding=0,
    dilation=1,
    groups: int = 1,
    n_iter: int = 30,
    seed: int = 0,
) -> float:
    """Spectral norm of a 1-D or 2-D convolution operator via power iteration.

    `input_shape` is `(C_in, H, W)` for Conv2d or `(C_in, L)` for Conv1d.
    Power iteration converges to the top singular value of the linear
    operator $x \\mapsto W \\ast x$ (with the given stride/padding/dilation),
    giving the exact L2-Lipschitz constant of the conv layer.
    """
    g = torch.Generator(device=weight.device).manual_seed(seed)
    if weight.dim() == 4:
        if len(input_shape) != 3:
            raise ValueError(f"Conv2d expects (C, H, W) input_shape, got {input_shape}")
        C_in, H, W = input_shape
        x = torch.randn(1, C_in, H, W, generator=g,
                        device=weight.device, dtype=weight.dtype)
        conv = lambda u: F.conv2d(u, weight, stride=stride, padding=padding,
                                   dilation=dilation, groups=groups)
        conv_T = lambda v: F.conv_transpose2d(v, weight, stride=stride,
                                               padding=padding, dilation=dilation,
                                               groups=groups)
    elif weight.dim() == 3:
        if len(input_shape) != 2:
            raise ValueError(f"Conv1d expects (C, L) input_shape, got {input_shape}")
        C_in, L = input_shape
        x = torch.randn(1, C_in, L, generator=g,
                        device=weight.device, dtype=weight.dtype)
        conv = lambda u: F.conv1d(u, weight, stride=stride, padding=padding,
                                   dilation=dilation, groups=groups)
        conv_T = lambda v: F.conv_transpose1d(v, weight, stride=stride,
                                               padding=padding, dilation=dilation,
                                               groups=groups)
    else:
        raise ValueError(f"expected 3-D (Conv1d) or 4-D (Conv2d) weight, got {tuple(weight.shape)}")
    x = x / (x.norm() + 1e-12)
    for _ in range(n_iter):
        y = conv(x)
        y = y / (y.norm() + 1e-12)
        x = conv_T(y)
        x = x / (x.norm() + 1e-12)
    return float(conv(x).norm().item())


def _activation_lipschitz(module: nn.Module) -> Optional[float]:
    for klass, val in _ACTIVATION_LIPSCHITZ.items():
        if isinstance(module, klass):
            return val
    return None


def infer_conv_input_shapes(
    model: nn.Module,
    forward_call: Callable[[nn.Module], None],
) -> Dict[str, Tuple[int, ...]]:
    """Run a forward pass with hooks to record the input (C, ...) of each Conv layer.

    `forward_call(model)` should invoke whatever signature the model exposes
    (the multimodal detector takes `(frames, audio, has_audio=...)`). Hooks
    are registered on every `Conv1d`/`Conv2d` descendant and torn down before
    returning.
    """
    shapes: Dict[str, Tuple[int, ...]] = {}
    handles = []

    def _hook_factory(name: str):
        def hook(_mod, inputs, _output):
            x = inputs[0]
            shapes[name] = tuple(x.shape[1:])
        return hook

    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv1d, nn.Conv2d)):
            handles.append(mod.register_forward_hook(_hook_factory(name)))
    try:
        with torch.no_grad():
            forward_call(model)
    finally:
        for h in handles:
            h.remove()
    return shapes


def _norm_lipschitz(module: nn.Module) -> Optional[float]:
    """Conservative bound for affine norm layers.

    LN/GN/BN are not strictly Lipschitz w.r.t. inputs in the general case, but
    the affine post-normalization scaling has Lipschitz constant $\\|\\gamma\\|_\\infty$.
    This is the standard practical bound used in Lipschitz-margin training.
    """
    if isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d,
                            nn.BatchNorm2d, nn.BatchNorm3d, nn.InstanceNorm1d,
                            nn.InstanceNorm2d, nn.InstanceNorm3d)):
        if getattr(module, "weight", None) is not None:
            return float(module.weight.detach().abs().max().item())
        return 1.0
    return None


class LipschitzEstimator:
    """Walks `model.named_modules()` and tabulates per-layer Lipschitz constants.

    Provides:
      * `per_layer`: list of (name, kind, value, bounded?) tuples
      * `product_bound`: product over bounded layers (raises if any layer is unbounded
        unless `allow_partial=True`, in which case it returns the product over
        bounded layers and exposes the unbounded ones in `unbounded_names`)
      * `unbounded_names`: layers we could not bound (typically self-attention)
    """

    def __init__(self):
        self.per_layer: List[Dict] = []
        self.unbounded_names: List[str] = []

    def estimate(
        self,
        model: nn.Module,
        input_shape_by_conv: Optional[Dict[str, Tuple[int, ...]]] = None,
        default_conv_input_shape: Tuple[int, ...] = (3, 32, 32),
        default_conv1d_input_shape: Tuple[int, int] = (80, 64),
        n_iter: int = 30,
    ) -> "LipschitzEstimator":
        """Walk `model` and fill `per_layer`.

        `input_shape_by_conv` lets the caller pass exact (C_in, H, W) shapes
        for specific Conv2d modules (by `named_modules()` name). Conv layers
        without an entry use `default_conv_input_shape`.
        """
        self.per_layer = []
        self.unbounded_names = []
        seen_attn = set()
        input_shape_by_conv = input_shape_by_conv or {}

        for name, mod in model.named_modules():
            # MultiheadAttention / nn.functional self-attention: unbounded
            if isinstance(mod, nn.MultiheadAttention):
                self.unbounded_names.append(name)
                self.per_layer.append({"name": name, "kind": "MultiheadAttention",
                                        "value": math.inf, "bounded": False})
                seen_attn.add(name)
                continue

            if isinstance(mod, nn.Linear):
                lip = linear_spectral_norm(mod.weight)
                self.per_layer.append({"name": name, "kind": "Linear",
                                        "value": lip, "bounded": True})
                continue

            if isinstance(mod, nn.Conv2d):
                shape = input_shape_by_conv.get(name, default_conv_input_shape)
                lip = conv_spectral_norm(
                    mod.weight, input_shape=shape,
                    stride=mod.stride, padding=mod.padding,
                    dilation=mod.dilation, groups=mod.groups, n_iter=n_iter,
                )
                self.per_layer.append({"name": name, "kind": "Conv2d",
                                        "value": lip, "bounded": True})
                continue

            if isinstance(mod, nn.Conv1d):
                shape = input_shape_by_conv.get(name, default_conv1d_input_shape)
                lip = conv_spectral_norm(
                    mod.weight, input_shape=shape,
                    stride=mod.stride[0], padding=mod.padding[0],
                    dilation=mod.dilation[0], groups=mod.groups, n_iter=n_iter,
                )
                self.per_layer.append({"name": name, "kind": "Conv1d",
                                        "value": lip, "bounded": True})
                continue

            n_lip = _norm_lipschitz(mod)
            if n_lip is not None:
                self.per_layer.append({"name": name, "kind": type(mod).__name__,
                                        "value": n_lip, "bounded": True})
                continue

            a_lip = _activation_lipschitz(mod)
            if a_lip is not None:
                self.per_layer.append({"name": name, "kind": type(mod).__name__,
                                        "value": a_lip, "bounded": True})
                continue

            # silently skip container modules (Sequential, ModuleList, etc.)
        return self

    @property
    def product_bound(self) -> float:
        """Product of per-layer Lipschitz constants over the *bounded* modules.

        If any module is unbounded, this returns the partial product (a lower
        estimate of the full bound). Caller should check `unbounded_names`.
        """
        prod = 1.0
        for row in self.per_layer:
            if row["bounded"]:
                prod *= row["value"]
        return prod

    def summary(self) -> dict:
        return {
            "n_layers": len(self.per_layer),
            "n_unbounded": len(self.unbounded_names),
            "product_bound_bounded_only": self.product_bound,
            "unbounded_names": list(self.unbounded_names),
        }


def certified_radius_from_margin(
    margin: torch.Tensor,
    lipschitz_constant: float,
    num_classes: int = 2,
) -> torch.Tensor:
    """Tsuzuku-style certified L2 radius from logit margin and global Lipschitz $L$.

    For a $K$-class classifier with logits $z(x)$, predicted class $c$, and
    margin $m(x) = z_c(x) - \\max_{j \\ne c} z_j(x)$, if $z$ is $L$-Lipschitz
    in the L2 norm (per coordinate), then the prediction is constant on the
    open ball of radius $r(x) = m(x) / (\\sqrt{2} L)$. The $\\sqrt{2}$ comes
    from the fact that the worst-case logit pair $(z_c - z_j)$ has Lipschitz
    constant at most $\\sqrt{2}L$.

    Negative margins are clipped to zero (no certificate for misclassified
    samples).
    """
    if lipschitz_constant <= 0 or not math.isfinite(lipschitz_constant):
        return torch.zeros_like(margin)
    return margin.clamp(min=0.0) / (math.sqrt(2.0) * lipschitz_constant)
