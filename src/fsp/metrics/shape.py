"""x-statistic and shape gap (§17.2) — how much signal a monotone metric misses."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from optbinning import OptimalBinning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from ._util import clean_pair
from .association import auc


def x_statistic(x: ArrayLike, y: ArrayLike) -> float:
    """c-statistic of a logistic regression on the WoE-transformed, optimally
    binned feature (§17.2). Always ≥ the raw AUC, equal iff monotone."""
    xa, ya = clean_pair(x, y)
    if len(np.unique(ya)) < 2:
        return float("nan")
    # Free (non-monotone) binning so the WoE can capture a U-shape — that is
    # exactly the non-monotone signal shape_gap exists to detect.
    # Quantile pre-bins + no monotonic constraint, so the WoE is free to be
    # non-monotone and capture a U-shape — the whole point of shape_gap.
    # (optbinning's own optimizer collapses non-monotone features to one bin.)
    edges = np.unique(np.quantile(xa.astype(float), np.linspace(0, 1, 11)[1:-1]))
    ob = OptimalBinning(dtype="numerical", monotonic_trend=None, user_splits=edges).fit(xa, ya)
    woe = np.asarray(ob.transform(xa, metric="woe"), dtype=float).reshape(-1, 1)
    lr = LogisticRegression().fit(woe, ya)
    proba = lr.predict_proba(woe)[:, 1]
    return float(roc_auc_score(ya, proba))


def shape_gap(x: ArrayLike, y: ArrayLike) -> float:
    """`x_stat − c_stat` (§17.2): the non-monotone signal a raw AUC discards.
    ≈ 0 for a monotone feature; large for U-shaped ones (→ review/engineer)."""
    return x_statistic(x, y) - auc(x, y)
