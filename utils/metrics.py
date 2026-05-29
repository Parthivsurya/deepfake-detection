"""Reusable detection metrics with safe behaviour on degenerate inputs.

All functions take 1-D numpy arrays: `y` are 0/1 ground-truth labels,
`p` are sigmoid/softmax probabilities for the "fake" class, and `pred`
are 0/1 hard predictions (defaults to `(p > threshold)`).

Where a metric is undefined (e.g. AUC with a single class), the function
returns `None` rather than raising — easier to JSON-serialise.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


def _binarise(p: np.ndarray, threshold: float) -> np.ndarray:
    return (p > threshold).astype(int)


def compute_metrics(
    y: np.ndarray,
    p: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Full metric report for one (y, p) pair."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    pred = _binarise(p, threshold)
    n = len(y)
    out: dict = {
        "n": int(n),
        "n_real": int((y == 0).sum()),
        "n_fake": int((y == 1).sum()),
        "threshold": threshold,
        "accuracy": float(accuracy_score(y, pred)) if n else None,
        "f1": float(f1_score(y, pred, zero_division=0)) if n else None,
        "precision": float(precision_score(y, pred, zero_division=0)) if n else None,
        "recall": float(recall_score(y, pred, zero_division=0)) if n else None,
    }
    if len(np.unique(y)) > 1:
        out["auc"] = float(roc_auc_score(y, p))
        out["ap"] = float(average_precision_score(y, p))
    else:
        out["auc"] = None
        out["ap"] = None
    if n:
        cm = confusion_matrix(y, pred, labels=[0, 1]).tolist()
        out["confusion_matrix"] = {
            "labels": ["real", "fake"],
            "matrix": cm,           # rows = true, cols = pred
        }
    return out


def equal_error_rate(y: np.ndarray, p: np.ndarray) -> Optional[float]:
    """Threshold where FAR == FRR. Returns the EER value (not the threshold).

    Useful as a single-number summary alongside AUC.
    """
    from sklearn.metrics import roc_curve
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return None
    fpr, tpr, _ = roc_curve(y, p)
    fnr = 1 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def latency_summary(latencies_ms: list[float]) -> dict:
    """Mean / p50 / p95 / p99 / std on a list of millisecond latencies."""
    if not latencies_ms:
        return {"n": 0}
    arr = np.asarray(latencies_ms, dtype=float)
    return {
        "n": int(arr.size),
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "std_ms": float(arr.std()),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }
