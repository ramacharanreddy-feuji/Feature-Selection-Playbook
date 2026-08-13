"""Categorical association + Information Value (§17.1, §17.3)."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from optbinning import OptimalBinning
from scipy.stats import chi2_contingency

from ._util import clean_pair

Splits = Iterable[tuple[np.ndarray, np.ndarray]]


def cramers_v(a: ArrayLike, b: ArrayLike) -> float:
    """Uncorrected Cramér's V (reference only — use bergsma_v for decisions)."""
    ct = pd.crosstab(*_two_series(a, b)).to_numpy()
    if min(ct.shape) < 2:
        return float("nan")
    chi2 = float(chi2_contingency(ct, correction=False)[0])
    n = float(ct.sum())
    r, k = ct.shape
    return float(np.sqrt((chi2 / n) / min(r - 1, k - 1)))


def bergsma_v(a: ArrayLike, b: ArrayLike) -> float:
    """Bergsma bias-corrected Cramér's V, Ṽ (§17.1)."""
    ct = pd.crosstab(*_two_series(a, b)).to_numpy()
    if min(ct.shape) < 2:
        return float("nan")
    n = float(ct.sum())
    chi2 = float(chi2_contingency(ct, correction=False)[0])
    r, k = ct.shape
    phi2 = chi2 / n
    phi2_t = max(0.0, phi2 - (r - 1) * (k - 1) / (n - 1))
    r_t = r - (r - 1) ** 2 / (n - 1)
    k_t = k - (k - 1) ** 2 / (n - 1)
    denom = min(r_t - 1, k_t - 1)
    return float(np.sqrt(phi2_t / denom)) if denom > 0 else float("nan")


def information_value(x: ArrayLike, y: ArrayLike, dtype: str = "numerical") -> float:
    """IV via optbinning (§17.3). `dtype` is "numerical" or "categorical"."""
    xa, ya = clean_pair(x, y)
    ob = OptimalBinning(dtype=dtype).fit(xa, ya)
    ob.binning_table.build()
    return float(ob.binning_table.iv)


def woe_table(x: ArrayLike, y: ArrayLike, dtype: str = "numerical") -> pd.DataFrame:
    """The optbinning binning table (bins, WoE, IV contribution) for reporting."""
    xa, ya = clean_pair(x, y)
    ob = OptimalBinning(dtype=dtype).fit(xa, ya)
    return ob.binning_table.build()


def iv_oof(x: ArrayLike, y: ArrayLike, folds: Splits, *, dtype: str = "categorical") -> float:
    """Out-of-fold Information Value for a high-cardinality categorical (§17.3).

    Bins are learned on each fold's training rows only, the WoE is applied to the
    held-out rows, and IV is scored there — so no row uses its own target. Report
    the mean over folds. This counters the in-sample inflation that makes a
    high-cardinality noise column look predictive. `y` must be coded 0/1 (NaN for
    a missing target). WoE uses Laplace smoothing (eps) to stay finite.
    """
    xa = np.asarray(x, dtype=object)
    ya = np.asarray(y, dtype=float)
    eps = 0.5
    ivs: list[float] = []
    for tr, te in folds:
        xtr, ytr, xte, yte = _drop_missing(xa[tr], ya[tr], xa[te], ya[te])
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            continue
        try:
            ob = OptimalBinning(dtype=dtype).fit(xtr, ytr)
            btr = np.asarray(ob.transform(xtr, metric="bins"), dtype=object)
            bte = np.asarray(ob.transform(xte, metric="bins"), dtype=object)
        except Exception:
            continue
        woe = _train_woe(btr, ytr, eps)
        ivs.append(_test_iv(bte, yte, woe))
    return float(np.mean(ivs)) if ivs else float("nan")


def target_encoded_eta_oof(x: ArrayLike, y: ArrayLike, folds: Splits) -> float:
    """Out-of-fold target-encoded η² for a high-cardinality categorical vs a
    regression target (§8, §17.3). Each row's category is encoded by the mean
    target of that category on the *training* folds (unseen → global train mean);
    η² is the variance of the continuous target explained by those held-out
    encodings — 1 − SS_res/SS_tot, clamped ≥ 0. Noise categories regress to the
    mean out-of-fold, so their η² collapses (the whole point)."""
    xa = np.asarray(x, dtype=object)
    ya = np.asarray(y, dtype=float)
    enc = np.full(len(ya), np.nan)
    for tr, te in folds:
        ytr = ya[tr]
        ok = ~np.isnan(ytr)
        if not ok.any():
            continue
        means = pd.Series(ytr[ok]).groupby(pd.Series(xa[tr][ok])).mean()
        gmean = float(ytr[ok].mean())
        enc[te] = pd.Series(xa[te]).map(means).fillna(gmean).to_numpy()
    mask = ~np.isnan(enc) & ~np.isnan(ya)
    yv, ev = ya[mask], enc[mask]
    ss_tot = float(((yv - yv.mean()) ** 2).sum())
    if ss_tot == 0.0 or len(yv) == 0:
        return 0.0
    ss_res = float(((yv - ev) ** 2).sum())
    return float(max(0.0, 1.0 - ss_res / ss_tot))


def _drop_missing(
    xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray, yte: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mtr = pd.notna(pd.Series(xtr)).to_numpy() & ~np.isnan(ytr)
    mte = pd.notna(pd.Series(xte)).to_numpy() & ~np.isnan(yte)
    return xtr[mtr], ytr[mtr], xte[mte], yte[mte]


def _train_woe(bins: np.ndarray, y: np.ndarray, eps: float) -> dict[object, float]:
    good, bad = (y == 0), (y == 1)
    tot_g, tot_b = float(good.sum()), float(bad.sum())
    woe: dict[object, float] = {}
    for b in np.unique(bins):
        m = bins == b
        dg = (float(good[m].sum()) + eps) / (tot_g + eps)
        db = (float(bad[m].sum()) + eps) / (tot_b + eps)
        woe[b] = float(np.log(dg / db))
    return woe


def _test_iv(bins: np.ndarray, y: np.ndarray, woe: dict[object, float]) -> float:
    good, bad = (y == 0), (y == 1)
    tot_g, tot_b = float(good.sum()), float(bad.sum())
    if tot_g == 0.0 or tot_b == 0.0:
        return 0.0
    iv = 0.0
    for b in np.unique(bins):
        m = bins == b
        dg = float(good[m].sum()) / tot_g
        db = float(bad[m].sum()) / tot_b
        iv += (dg - db) * woe.get(b, 0.0)
    return iv


def _two_series(a: ArrayLike, b: ArrayLike) -> tuple[pd.Series, pd.Series]:
    aa, bb = clean_pair(a, b)
    return pd.Series(aa), pd.Series(bb)
