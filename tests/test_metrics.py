"""Metric kernel tests — each metric against a known value or proven property."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fsp import metrics as m


@pytest.fixture
def rng():
    return np.random.default_rng(0)


# --- association ------------------------------------------------------------


def test_auc_perfect_and_orientation_free():
    x = np.array([1, 2, 3, 4], float)
    y = np.array([0, 0, 1, 1])
    assert m.auc(x, y) == pytest.approx(1.0)
    assert m.auc(-x, y) == pytest.approx(1.0)  # orientation-free


def test_auc_random_near_half(rng):
    x = rng.normal(size=2000)
    y = rng.integers(0, 2, 2000)
    assert m.auc(x, y) == pytest.approx(0.5, abs=0.05)


def test_spearman_monotone_is_one():
    x = np.arange(20, dtype=float)
    assert m.spearman(x, x**2)[0] == pytest.approx(1.0)


def test_point_biserial_equals_pearson_on_01():
    y = np.array([0, 0, 1, 1, 1, 0, 1, 0], float)
    x = np.array([1.0, 2, 5, 6, 7, 1, 8, 2])
    assert m.point_biserial(x, y)[0] == pytest.approx(m.pearson(x, y)[0])


def test_cliffs_delta_separation_and_overlap():
    a = np.array([4, 5, 6], float)  # group 1
    b = np.array([1, 2, 3], float)  # group 0
    x = np.concatenate([a, b])
    g = np.array([1, 1, 1, 0, 0, 0])
    assert m.cliffs_delta(x, g) == pytest.approx(1.0)
    assert abs(m.cliffs_delta(x, np.array([1, 0, 1, 0, 1, 0]))) < 0.6


def test_correlation_ratio_perfect_and_none():
    cats = np.array([0, 0, 1, 1])
    assert m.correlation_ratio(cats, np.array([1.0, 1, 9, 9])) == pytest.approx(1.0)
    assert m.correlation_ratio(cats, np.array([5.0, 5, 5, 5])) == pytest.approx(0.0)


def test_kruskal_eps2_range(rng):
    vals = np.r_[rng.normal(0, 1, 100), rng.normal(3, 1, 100)]
    grp = np.r_[np.zeros(100), np.ones(100)]
    eps2, p = m.kruskal_eps2(vals, grp)
    assert 0.0 <= eps2 <= 1.0 and p < 0.05


def test_auc_ovr_multiclass(rng):
    y = np.r_[np.zeros(100), np.ones(100), np.full(100, 2)].astype(int)
    x = np.r_[rng.normal(0, 1, 100), rng.normal(3, 1, 100), rng.normal(6, 1, 100)]
    assert m.auc_ovr(x, y) > 0.8  # a clean 3-way separator


def test_kendall_monotone_is_one():
    x = np.arange(20, dtype=float)
    assert m.kendall(x, x**2)[0] == pytest.approx(1.0)


# --- categorical ------------------------------------------------------------


def test_bergsma_v_independence_near_zero(rng):
    a = rng.integers(0, 4, 4000)
    b = rng.integers(0, 4, 4000)
    assert m.bergsma_v(a, b) < 0.05  # bias-corrected → ~0 under independence


def test_bergsma_v_perfect_association():
    a = np.array(list("aabbcc") * 50)
    b = np.array(list("xxyyzz") * 50)
    assert m.bergsma_v(a, b) > 0.95


def test_information_value_strong_separator(rng):
    y = np.r_[np.zeros(300), np.ones(300)].astype(int)
    x = np.r_[rng.normal(0, 1, 300), rng.normal(4, 1, 300)]
    assert m.information_value(x, y) > 0.5


def test_information_value_accepts_string_target(rng):
    # regression: a Yes/No target must score like a 0/1 one, not raise an opaque
    # optbinning TypeError. IV magnitude is invariant to the class→code mapping.
    x = np.r_[rng.normal(0, 1, 300), rng.normal(4, 1, 300)]
    y01 = np.r_[np.zeros(300), np.ones(300)].astype(int)
    ystr = np.where(y01 == 1, "Yes", "No")
    assert m.information_value(x, ystr) == pytest.approx(m.information_value(x, y01), rel=1e-9)


def test_information_value_rejects_nonbinary_target(rng):
    x = rng.normal(0, 1, 300)
    y3 = rng.integers(0, 3, 300)  # three classes → typed error, not a crash downstream
    with pytest.raises(ValueError, match="binary"):
        m.information_value(x, y3)


def test_woe_table_returns_binning_frame(rng):
    y = np.r_[np.zeros(200), np.ones(200)].astype(int)
    x = np.r_[rng.normal(0, 1, 200), rng.normal(3, 1, 200)]
    tbl = m.woe_table(x, y)
    assert hasattr(tbl, "columns") and len(tbl) > 0  # optbinning binning table


def _kfolds(n, k=5, seed=0):
    idx = np.random.default_rng(seed).permutation(n)
    return [(np.setdiff1d(idx, f), f) for f in np.array_split(idx, k)]


def test_iv_oof_penalizes_high_card_noise(rng):
    # A high-cardinality *noise* column: many random levels, unrelated to y.
    n = 1200
    y = rng.integers(0, 2, n).astype(float)
    noise = rng.integers(0, 300, n).astype(object)  # ~300 random levels
    # A real signal: level parity tracks y most of the time.
    signal = np.where(rng.random(n) < 0.85, y, 1 - y).astype(object)
    folds = _kfolds(n)
    iv_noise = m.iv_oof(noise, y, folds)
    iv_signal = m.iv_oof(signal, y, folds)
    insample_noise = m.information_value(noise, y, dtype="categorical")
    assert iv_signal > iv_noise
    assert insample_noise > 0.3  # in-sample, high-card noise looks predictive
    assert iv_noise < 0.4 * insample_noise  # out-of-fold strips most of that inflation


def test_target_encoded_eta_oof_signal_vs_noise(rng):
    n = 1200
    cats = rng.integers(0, 40, n)
    y_signal = cats + rng.normal(0, 0.5, n)  # category strongly drives the target
    y_noise = rng.normal(0, 1, n)  # target unrelated to category
    folds = _kfolds(n)
    eta_signal = m.target_encoded_eta_oof(cats.astype(object), y_signal, folds)
    eta_noise = m.target_encoded_eta_oof(cats.astype(object), y_noise, folds)
    assert eta_signal > 0.5 and eta_noise < 0.1


# --- shape ------------------------------------------------------------------


def test_shape_gap_nonnegative_and_small_when_monotone(rng):
    y = np.r_[np.zeros(300), np.ones(300)].astype(int)
    x = np.r_[rng.normal(0, 1, 300), rng.normal(2, 1, 300)]  # monotone signal
    gap = m.shape_gap(x, y)
    assert -0.02 < gap < 0.1  # ~0 for a monotone feature (tiny finite-bin slack)


def test_shape_gap_detects_u_shape(rng):
    # U-shaped: extremes are class 1, middle is class 0 → raw AUC ~0.5, binned high
    x = rng.uniform(-3, 3, 800)
    y = (np.abs(x) > 1.5).astype(int)
    assert m.auc(x, y) < 0.6
    assert m.shape_gap(x, y) > 0.1


# --- resampling -------------------------------------------------------------


def test_bh_fdr_monotone_and_bounded():
    q = m.bh_fdr([0.001, 0.01, 0.5, 0.9])
    assert np.all(q <= 1.0) and np.all(np.diff(q) >= -1e-9)


def test_shadow_floor_for_noise_near_null(rng):
    x = rng.normal(size=1000)
    y = rng.integers(0, 2, 1000)
    floor = m.shadow_floor(m.auc, x, y, b=50, seed=1)
    assert 0.5 <= floor < 0.62  # noise column: random-AUC 95th percentile


def test_bootstrap_ci_orders_and_contains(rng):
    x = np.r_[rng.normal(0, 1, 300), rng.normal(3, 1, 300)]
    y = np.r_[np.zeros(300), np.ones(300)].astype(int)
    lo, hi = m.bootstrap_ci(m.auc, x, y, b=200, seed=2)
    assert lo <= hi and lo > 0.8


def test_mann_whitney_p_separates_and_null(rng):
    y = np.r_[np.zeros(200), np.ones(200)].astype(int)
    x = np.r_[rng.normal(0, 1, 200), rng.normal(3, 1, 200)]
    assert m.mann_whitney_p(x, y) < 1e-10  # AUC's companion p, strong signal
    assert m.mann_whitney_p(rng.normal(size=400), y) > 0.05  # noise


def test_permutation_p_one_sided():
    shadow = np.linspace(0.40, 0.60, 50)
    assert m.permutation_p(0.9, shadow) == pytest.approx(1 / 51)  # nothing beats it
    assert m.permutation_p(0.30, shadow) == pytest.approx(1.0)  # below all → everything ≥
    assert np.isnan(m.permutation_p(float("nan"), shadow))


# --- drift ------------------------------------------------------------------


def test_psi_zero_for_same_distribution(rng):
    e = rng.normal(size=5000)
    assert m.psi(e, e) == pytest.approx(0.0, abs=1e-6)


def test_psi_flags_shift(rng):
    e = rng.normal(0, 1, 5000)
    a = rng.normal(2, 1, 5000)
    assert m.psi(e, a) > 0.25


def test_vif_detects_collinearity(rng):
    base = rng.normal(size=500)
    df = pd.DataFrame({"a": base, "b": base + rng.normal(0, 0.01, 500), "c": rng.normal(size=500)})
    v = m.vif(df)
    assert v["a"] > 10 and v["c"] < 5


# --- survival ---------------------------------------------------------------


def test_cox_screen_runs_and_bounds(rng):
    n = 300
    x = rng.normal(size=n)
    duration = rng.exponential(scale=np.exp(-0.5 * x))
    event = rng.integers(0, 2, n)
    out = m.cox_screen(duration, event, x)
    assert set(out) == {"hr", "p", "cindex"} and 0.0 <= out["cindex"] <= 1.0


def test_concordance_and_logrank(rng):
    n = 200
    dur = rng.exponential(size=n)
    event = rng.integers(0, 2, n)
    c = m.concordance(dur, rng.normal(size=n), event)
    assert 0.0 <= c <= 1.0
    stat, p = m.logrank(dur, event, rng.integers(0, 2, n))
    assert stat >= 0.0 and 0.0 <= p <= 1.0
