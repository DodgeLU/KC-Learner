"""NLL, Brier, and AUC used by the EdNet IRT / AR-KT runners.

The formulas are the ones in
``project_c/train/_ednet_irt_ar_kt_dev5000_common.py``.
NLL clips probabilities to ``[1e-12, 1 - 1e-12]``. Brier does not.
AUC is 0.5 when every label is the same class. Tied scores use the
average rank, and the sort is ``mergesort``.
"""

from __future__ import annotations

import numpy as np

_CLIP_EPS = 1e-12


def metric_nll(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), _CLIP_EPS, 1.0 - _CLIP_EPS)
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def metric_brier(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return float(((p - y) ** 2).mean())


def metric_auc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if y.min() == y.max():
        return 0.5
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1, dtype=np.float64)
    sorted_p = p[order]
    i = 0
    while i < len(p):
        j = i
        while j + 1 < len(p) and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    n_pos = int(y.sum())
    n_neg = len(p) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    sr = float(ranks[y == 1].sum())
    return (sr - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
