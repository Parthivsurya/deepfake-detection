"""Metric utilities sanity check."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from utils import compute_metrics, equal_error_rate, latency_summary  # noqa: E402


def main() -> int:
    np.random.seed(0)
    y = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    p = np.array([0.1, 0.4, 0.8, 0.9, 0.3, 0.2, 0.7, 0.6])
    m = compute_metrics(y, p, threshold=0.5)
    assert m["n"] == 8 and m["n_fake"] == 4 and m["n_real"] == 4
    assert 0.0 <= m["accuracy"] <= 1.0
    assert m["auc"] is not None
    assert m["confusion_matrix"]["labels"] == ["real", "fake"]

    # all-real edge case: AUC undefined, accuracy still works
    m_one = compute_metrics(np.zeros(5), np.array([0.1, 0.2, 0.3, 0.4, 0.45]))
    assert m_one["auc"] is None
    assert m_one["accuracy"] == 1.0     # threshold 0.5 -> all predicted real

    eer = equal_error_rate(y, p)
    assert eer is not None and 0.0 <= eer <= 1.0

    lat = latency_summary([10.0, 12.0, 11.0, 50.0, 9.5])
    assert lat["n"] == 5 and lat["p95_ms"] >= lat["p50_ms"]

    print(f"OK  acc={m['accuracy']:.3f}  auc={m['auc']:.3f}  eer={eer:.3f}  "
          f"p95={lat['p95_ms']:.1f}ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
