"""Survival screening: univariate Cox, C-index, log-rank (§17.8)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.utils import concordance_index
from numpy.typing import ArrayLike


def cox_screen(duration: ArrayLike, event: ArrayLike, x: ArrayLike) -> dict[str, float]:
    """Univariate Cox on feature `x`. Returns hazard ratio, p-value, C-index."""
    df = pd.DataFrame(
        {
            "T": np.asarray(duration, dtype=float),
            "E": np.asarray(event, dtype=float),
            "x": np.asarray(x, dtype=float),
        }
    ).dropna()
    cph = CoxPHFitter().fit(df, duration_col="T", event_col="E")
    return {
        "hr": float(np.exp(cph.params_["x"])),
        "p": float(cph.summary.loc["x", "p"]),
        "cindex": float(cph.concordance_index_),
    }


def concordance(duration: ArrayLike, risk: ArrayLike, event: ArrayLike) -> float:
    """Harrell's C-index for a risk score (higher risk ⇒ shorter survival)."""
    return float(
        concordance_index(
            np.asarray(duration, dtype=float),
            -np.asarray(risk, dtype=float),
            np.asarray(event, dtype=float),
        )
    )


def logrank(duration: ArrayLike, event: ArrayLike, groups: ArrayLike) -> tuple[float, float]:
    """Multivariate log-rank test across the groups. Returns (statistic, p)."""
    res = multivariate_logrank_test(
        np.asarray(duration, dtype=float),
        np.asarray(groups),
        np.asarray(event, dtype=float),
    )
    return float(res.test_statistic), float(res.p_value)
