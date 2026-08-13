"""Association effect sizes (playbook §8 dispatch; §17.4, §17.9–17.11).

Each takes (feature, target)-style arrays and returns an effect on its native
scale. Orientation-free where the sign is not meaningful for screening (AUC).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import stats
from sklearn.metrics import roc_auc_score

from ._util import clean_pair


def auc(x: ArrayLike, y: ArrayLike) -> float:
    """Single-feature AUC vs a binary target, orientation-free (≥ 0.5)."""
    xa, ya = clean_pair(x, y)
    if len(np.unique(ya)) < 2:
        return float("nan")
    a = float(roc_auc_score(ya, xa))
    return max(a, 1.0 - a)


def mann_whitney_p(x: ArrayLike, y: ArrayLike) -> float:
    """Two-sided Mann–Whitney U p-value for continuous `x` split by binary `y`
    (§17.13). This is AUC's exact companion p (AUC = U / (m·n)), so it gives the
    single-feature AUC a p-value for the BH-FDR q (§17.6)."""
    xa, ya = clean_pair(x, y)
    classes = np.unique(ya)
    if len(classes) != 2:
        return float("nan")
    g1 = xa[ya == classes[1]].astype(float)
    g0 = xa[ya == classes[0]].astype(float)
    if len(g1) < 1 or len(g0) < 1:
        return float("nan")
    return float(stats.mannwhitneyu(g1, g0, alternative="two-sided").pvalue)


def auc_ovr(x: ArrayLike, y: ArrayLike) -> float:
    """Mean one-vs-rest single-feature AUC for a multiclass target."""
    xa, ya = clean_pair(x, y)
    classes = np.unique(ya)
    if len(classes) < 2:
        return float("nan")
    scores = []
    for c in classes:
        yc = (ya == c).astype(int)
        if len(np.unique(yc)) == 2:
            a = float(roc_auc_score(yc, xa))
            scores.append(max(a, 1.0 - a))
    return float(np.mean(scores)) if scores else float("nan")


def spearman(x: ArrayLike, y: ArrayLike) -> tuple[float, float]:
    xa, ya = clean_pair(x, y)
    r = stats.spearmanr(xa, ya)
    return float(r.statistic), float(r.pvalue)


def kendall(x: ArrayLike, y: ArrayLike) -> tuple[float, float]:
    xa, ya = clean_pair(x, y)
    r = stats.kendalltau(xa, ya)
    return float(r.statistic), float(r.pvalue)


def pearson(x: ArrayLike, y: ArrayLike) -> tuple[float, float]:
    xa, ya = clean_pair(x, y)
    r = stats.pearsonr(xa, ya)
    return float(r.statistic), float(r.pvalue)


def point_biserial(x: ArrayLike, y: ArrayLike) -> tuple[float, float]:
    """Binary vs continuous (§17.11); identical to Pearson on the 0/1 coding."""
    xa, ya = clean_pair(x, y)
    r = stats.pointbiserialr(xa, ya)
    return float(r.statistic), float(r.pvalue)


def cliffs_delta(x: ArrayLike, group: ArrayLike) -> float:
    """Cliff's δ (§17.4): dominance of continuous `x` between the two levels of
    binary `group`. δ = (#x1>x0 − #x1<x0) / (m·n), range −1..1, ties shrink |δ|."""
    xa, ga = clean_pair(x, group)
    levels = np.unique(ga)
    if len(levels) != 2:
        return float("nan")
    a = np.sort(xa[ga == levels[1]].astype(float))
    b = np.sort(xa[ga == levels[0]].astype(float))
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return float("nan")
    greater = int(np.searchsorted(b, a, side="left").sum())
    less = int((n - np.searchsorted(b, a, side="right")).sum())
    return (greater - less) / (m * n)


def correlation_ratio(categories: ArrayLike, values: ArrayLike) -> float:
    """η² (§17.9): fraction of continuous `values` variance explained by the
    categorical grouping. η² = SS_between / SS_total, range 0..1."""
    ca, va = clean_pair(categories, values)
    va = va.astype(float)
    ybar = va.mean()
    ss_total = float(((va - ybar) ** 2).sum())
    if ss_total == 0.0:
        return 0.0
    ss_between = 0.0
    for c in pd.unique(ca):
        vc = va[ca == c]
        ss_between += len(vc) * (vc.mean() - ybar) ** 2
    return float(ss_between / ss_total)


def kruskal_eps2(values: ArrayLike, groups: ArrayLike) -> tuple[float, float]:
    """Rank-based ε² (§17.10): ε² = H / (n − 1) from the Kruskal–Wallis H over
    the groups. Robust to non-normality/outliers. Returns (ε², p)."""
    va, ga = clean_pair(values, groups)
    va = va.astype(float)
    parts = [va[ga == c] for c in pd.unique(ga)]
    parts = [p for p in parts if len(p) > 0]
    if len(parts) < 2:
        return float("nan"), float("nan")
    h, p = stats.kruskal(*parts)
    n = len(va)
    return float(h / (n - 1)), float(p)
