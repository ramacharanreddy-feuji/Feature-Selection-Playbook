"""Drift (PSI, §17.7) and multicollinearity (VIF, §9)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from statsmodels.stats.outliers_influence import variance_inflation_factor


def psi(expected: ArrayLike, actual: ArrayLike, bins: int = 10) -> float:
    """Population Stability Index (§17.7): bin `expected` into quantiles, then
    sum (%actual − %expected)·ln(%actual / %expected) across bins."""
    e = np.asarray(expected, dtype=float)
    a = np.asarray(actual, dtype=float)
    e = e[~np.isnan(e)]
    a = a[~np.isnan(a)]
    if len(e) == 0 or len(a) == 0:
        return float("nan")
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e_frac = np.histogram(e, edges)[0] / len(e)
    a_frac = np.histogram(a, edges)[0] / len(a)
    eps = 1e-6
    e_frac = np.clip(e_frac, eps, None)
    a_frac = np.clip(a_frac, eps, None)
    return float(np.sum((a_frac - e_frac) * np.log(a_frac / e_frac)))


def vif(x: pd.DataFrame) -> pd.Series:
    """Variance inflation factor per column of a numeric frame (§9, flag > 10)."""
    df = pd.DataFrame(x).dropna().astype(float)
    mat = np.column_stack([df.to_numpy(), np.ones(len(df))])
    out = {str(col): float(variance_inflation_factor(mat, i)) for i, col in enumerate(df.columns)}
    return pd.Series(out)
