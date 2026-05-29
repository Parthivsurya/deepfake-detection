"""Real-time inference optimisations: pruning, quantization, parameter counts.

The functions here are deliberately lightweight wrappers around the standard
torch APIs so they compose with any model that uses `nn.Linear` heavily. They
return the same module (mutated in place for pruning, possibly replaced for
quantization) and a `dict` of diagnostics for logging / benchmarking.

Public API:
    apply_global_unstructured_pruning(model, amount, make_permanent=False)
    apply_dynamic_quantization(model, dtype=torch.qint8)
    weight_sparsity(model)
    parameter_count(model)
"""
from __future__ import annotations
from typing import Iterable
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


def _linear_modules(model: nn.Module) -> list[tuple[nn.Module, str]]:
    return [(m, "weight") for m in model.modules() if isinstance(m, nn.Linear)]


def apply_global_unstructured_pruning(
    model: nn.Module,
    amount: float = 0.3,
    make_permanent: bool = False,
) -> nn.Module:
    """L1 magnitude global pruning over every Linear weight.

    `amount` is the fraction of weights zeroed across the whole pool. If
    `make_permanent=True` the reparametrization is removed so the model can be
    saved / quantized afterwards.
    """
    if not 0.0 < amount < 1.0:
        raise ValueError(f"prune amount must be in (0, 1), got {amount}")
    params = _linear_modules(model)
    if not params:
        return model
    prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=amount)
    if make_permanent:
        for module, name in params:
            prune.remove(module, name)
    return model


def apply_dynamic_quantization(
    model: nn.Module,
    dtype: torch.dtype = torch.qint8,
    extra_layers: Iterable[type[nn.Module]] = (),
    engine: str | None = None,
) -> nn.Module:
    """Dynamic INT8 quantization on Linear (and optionally LSTM) layers.

    Dynamic quantization is CPU-only but requires no calibration data, which is
    perfect for a real-time benchmarking pipeline. The MHA submodules contain
    Linear projections that *are* quantized; the surrounding Python plumbing
    runs in FP32.

    `engine` defaults to "qnnpack" on ARM (Apple Silicon, Raspberry Pi) and
    "fbgemm" on x86. On builds where the default engine is "none", picking one
    of the supported engines is mandatory.
    """
    supported = torch.backends.quantized.supported_engines
    if engine is None:
        if torch.backends.quantized.engine in supported and torch.backends.quantized.engine != "none":
            engine = torch.backends.quantized.engine
        elif "fbgemm" in supported:
            engine = "fbgemm"
        elif "qnnpack" in supported:
            engine = "qnnpack"
        else:
            raise RuntimeError(f"no quantization engine available; supported={supported}")
    torch.backends.quantized.engine = engine
    target = {nn.Linear, *extra_layers}
    return torch.ao.quantization.quantize_dynamic(model, target, dtype=dtype)


def weight_sparsity(model: nn.Module) -> float:
    """Fraction of zero-valued weights across all Linear layers."""
    total, zeros = 0, 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            w = m.weight.detach()
            total += w.numel()
            zeros += int((w == 0).sum().item())
    return zeros / max(total, 1)


def parameter_count(model: nn.Module) -> dict[str, int]:
    """Return total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
