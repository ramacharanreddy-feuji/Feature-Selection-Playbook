"""Resampling engines: shadow floor (§17.5), BH-FDR (§17.6), bootstrap CI (§17.12)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike
from statsmodels.stats.multitest import multipletests

# A metric reduced to a single effect value given (feature, target) arrays.
MetricFn = Callable[[np.ndarray, np.ndarray], float]


def bh_fdr(pvals: ArrayLike) -> np.ndarray:
    """Benjamini-Hochberg q-values (§17.6). NaN p-values pass through as NaN."""
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan)
    mask = ~np.isnan(p)
    if mask.any():
        out[mask] = multipletests(p[mask], method="fdr_bh")[1]
    return out


def bootstrap_ci(
    metric_fn: MetricFn,
    x: ArrayLike,
    y: ArrayLike,
    *,
    b: int = 1000,
    seed: int = 42,
    ci: float = 0.95,
) -> tuple[float, float]:
    """Bootstrap percentile CI (§17.12): resample rows `b` times, recompute the
    metric, take the central `ci` percentiles. Fixed seed → reproducible."""
    xa, ya = np.asarray(x), np.asarray(y)
    n = len(xa)
    rng = np.random.default_rng(seed)
    boot = np.empty(b)
    for i in range(b):
        idx = rng.integers(0, n, n)
        boot[i] = metric_fn(xa[idx], ya[idx])
    lo, hi = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    return float(np.nanpercentile(boot, lo)), float(np.nanpercentile(boot, hi))


def shadow_samples(
    metric_fn: MetricFn,
    x: ArrayLike,
    y: ArrayLike,
    *,
    b: int = 50,
    seed: int = 42,
) -> np.ndarray:
    """The raw shadow distribution (§17.5): permute the feature `b` times and
    recompute the same metric each time. `shadow_floor` and the permutation
    p-value (§17.6) are both read off this one array — compute it once."""
    xa, ya = np.asarray(x), np.asarray(y)
    rng = np.random.default_rng(seed)
    shadow = np.empty(b)
    for i in range(b):
        shadow[i] = metric_fn(rng.permutation(xa), ya)
    return shadow


def shadow_floor(
    metric_fn: MetricFn,
    x: ArrayLike,
    y: ArrayLike,
    *,
    pct: float = 95,
    b: int = 50,
    seed: int = 42,
) -> float:
    """The self-calibrating bar (§17.5, §4.3): the `pct`-th percentile of the
    shadow distribution — the effect achievable at random for this column."""
    return float(np.nanpercentile(shadow_samples(metric_fn, x, y, b=b, seed=seed), pct))


def permutation_p(effect: float, shadow: ArrayLike) -> float:
    """One-sided permutation p-value (§17.6) for a higher-is-better effect: the
    shadow-adjusted fraction of permuted effects at least as large as observed,
    `(1 + #{shadow ≥ effect}) / (B + 1)`. Used for metrics with no analytic p
    (IV, η², Cramér's V, Cliff's δ) so every feature gets a q-value."""
    s = np.asarray(shadow, dtype=float)
    s = s[~np.isnan(s)]
    if len(s) == 0 or not np.isfinite(effect):
        return float("nan")
    return float((1 + int(np.sum(s >= effect))) / (len(s) + 1))
