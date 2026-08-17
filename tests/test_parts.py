"""Part tools A–H + report, on a synthetic dataset with planted structure."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import fsp
from fsp import thresholds as T
from fsp.gates import GateFailure
from fsp.parts import (
    boruta,
    frame,
    inventory,
    leakage,
    partition,
    redundancy,
    relevance,
    values,
    viability,
)


@pytest.fixture
def data():
    rng = np.random.default_rng(7)
    n = 400
    y = rng.integers(0, 2, n)
    df = pd.DataFrame(
        {
            "customer_id": range(n),                       # identifier
            "signal": y + rng.normal(0, 0.4, n),           # real predictor
            "signal_copy": None,                           # exact dup (set below)
            "noise": rng.normal(0, 1, n),                  # pure noise
            "region": rng.choice(list("abcd"), n),         # nominal noise
            "constant": 1,                                 # constant
            "leak_score": y.astype(float),                 # a leak (== target)
            "churn": y,
        }
    )
    df["signal_copy"] = df["signal"]
    return df


def test_frame_and_viability(data):
    assert "churn" in frame.target_candidates(data)
    tf = frame.target_facts(data, "churn")
    assert tf["suggested_type"] == "binary"
    assert "customer_id" in frame.id_candidates(data)
    v = viability.assess(data, "churn", "binary")
    assert v["positives"] >= 1 and viability.tier(v["effective_n"]) == "full"


def test_inventory_structural_flags(data):
    prof = inventory.profile(data)
    assert set(prof["column"]) == set(data.columns)
    assert inventory.structural_flags(data, "constant") == "constant"
    assert inventory.structural_flags(data, "customer_id") == "identifier"
    dups = inventory.duplicate_columns(data)
    assert dups.get("signal_copy") == "signal"
    assert inventory.suggest_semantic_type(inventory.column_facts(data, "region")) == "nominal"


def test_values_missingness_and_leak_guard(data, tmp_path):
    d = data.copy()
    miss = np.zeros(len(d), dtype=bool)
    miss[:120] = True  # opt_a and opt_b share the same partial missingness pattern
    d["opt_a"] = np.where(miss, np.nan, 1.0)
    d["opt_b"] = np.where(miss, np.nan, 2.0)
    clusters = values.comissing_clusters(d)
    assert any({"opt_a", "opt_b"} <= set(c) for c in clusters)
    # leak diagnostic is leakage-guarded (no folds yet)
    ctx = fsp.open_run(data, target="churn", target_type="binary", runs_dir=tmp_path / "runs")
    with pytest.raises(GateFailure):
        values.missingness_predicts_target(ctx, "noise")


def test_partition_and_relevance_guard(data, tmp_path):
    ctx = fsp.open_run(data, target="churn", target_type="binary", runs_dir=tmp_path / "runs")
    # relevance MUST raise before folds exist (the flagship safety feature)
    with pytest.raises(GateFailure):
        relevance.relevance(ctx, "signal", "continuous")
    folds = partition.make_folds(data, "stratified", k=5, target="churn")
    ctx.folds = folds
    r = relevance.relevance(ctx, "signal", "continuous")
    assert r["metric_name"] == "AUC" and r["effect"] > 0.6
    rn = relevance.relevance(ctx, "noise", "continuous")
    assert rn["effect"] < 0.6
    # the leak feature scores near-perfect and above its shadow floor
    rl = relevance.relevance(ctx, "leak_score", "continuous")
    assert rl["effect"] > 0.9


def test_relevance_all_adds_qvalues(data, tmp_path):
    ctx = fsp.open_run(data, target="churn", target_type="binary", runs_dir=tmp_path / "runs")
    ctx.folds = partition.make_folds(data, "stratified", k=5, target="churn")
    out = relevance.relevance_all(ctx, {"signal": "continuous", "noise": "continuous"})
    assert "q_value" in out.columns and len(out) == 2


def test_redundancy_collapses_the_pair(data):
    types = {"signal": "continuous", "signal_copy": "continuous", "noise": "continuous"}
    pairs = redundancy.pairwise(data, list(types), types)
    assert pairs.loc["signal", "signal_copy"] > 0.95
    comps = redundancy.components(pairs, threshold=0.95)
    assert any({"signal", "signal_copy"} <= c for c in comps)
    rep = redundancy.representative(
        {"signal", "signal_copy"},
        effects={"signal": 0.9, "signal_copy": 0.8},
        missing={"signal": 0, "signal_copy": 0},
    )
    assert rep == "signal"


def test_pairwise_survives_string_ordinal(data):
    # regression: a column typed `ordinal` (→ numeric → Spearman) but actually
    # string-valued must score nan→0, not raise AttributeError three parts later.
    d = data.copy()
    d["contract"] = np.where(d["signal"] > d["signal"].median(), "yearly", "monthly")
    types = {"signal": "continuous", "contract": "ordinal"}
    pairs = redundancy.pairwise(d, list(types), types)  # must not raise
    assert pairs.loc["signal", "signal"] == 1.0
    assert np.isfinite(pairs.loc["signal", "contract"])  # 0.0 on the failed pair


def test_fold_positive_counts_and_per_fold_floor(data):
    # the real per-fold minimum is what the §9 floor checks
    folds = partition.make_folds(data, "stratified", k=5, target="churn", run_dir=None)
    counts = partition.fold_positive_counts(folds, data["churn"])
    assert len(counts) == 5 and all(c >= 0 for c in counts)
    minority = data["churn"].value_counts().index[-1]
    assert sum(counts) == int((data["churn"] == minority).sum())  # every positive counted once
    assert T.per_fold_floor_met(counts, min_per_fold=1) is (min(counts) >= 1)


def test_leakage_detectors(data, tmp_path):
    ctx = fsp.open_run(data, target="churn", target_type="binary", runs_dir=tmp_path / "runs")
    leakage.name_signals(ctx)
    assert ctx.leaks.for_column("leak_score")  # name heuristic fired
    effects = {"signal": 0.7, "noise": 0.51, "a": 0.52, "b": 0.5, "c": 0.53, "leak_score": 0.99}
    leakage.outlier_effects(ctx, effects)
    adj = leakage.adjudicate(ctx)
    assert "leak_score" in adj


def test_adjudicate_requires_corroboration(data, tmp_path):
    ctx = fsp.open_run(data, target="churn", target_type="binary", runs_dir=tmp_path / "runs")
    # A lone WEAK signal (effect-above-backstop only) must NOT be adjudicated.
    leakage.backstop_effects(ctx, {"noise": 0.99}, threshold=0.85)
    assert "noise" not in leakage.adjudicate(ctx)
    # A strong structural signal (target-like name) stands alone.
    leakage.name_signals(ctx)  # fires on 'leak_score'
    assert "leak_score" in leakage.adjudicate(ctx)
    # Two distinct weak signals on one column corroborate.
    leakage.outlier_effects(ctx, {"noise": 0.99, "a": 0.5, "b": 0.5, "c": 0.5, "d": 0.5, "e": 0.5})
    assert "noise" in leakage.adjudicate(ctx)  # backstop + outlier → 2 detectors


def test_only_present_for_positives_fires(data, tmp_path):
    d = data.copy()
    y = d["churn"].to_numpy()
    # `receipt` exists only when the outcome happened (present iff y == 1).
    d["receipt"] = np.where(y == 1, 1.0, np.nan)
    ctx = fsp.open_run(d, target="churn", target_type="binary", runs_dir=tmp_path / "runs")
    leakage.only_present_for_positives(ctx)
    assert ctx.leaks.for_column("receipt")
    assert not ctx.leaks.for_column("signal")  # fully-present column does not fire


def test_future_timestamp_fires(tmp_path):
    n = 200
    ref = pd.Timestamp("2024-01-01")
    df = pd.DataFrame(
        {
            "event_date": [ref] * n,
            "resolved_at": [ref + pd.Timedelta(days=30)] * n,  # after prediction time
            "y": np.random.default_rng(0).integers(0, 2, n),
        }
    )
    ctx = fsp.open_run(
        df, target="y", target_type="binary", date_col="event_date", runs_dir=tmp_path / "runs"
    )
    leakage.future_timestamp(ctx)
    assert ctx.leaks.for_column("resolved_at")


def test_missingness_signals_needs_folds_then_fires(data, tmp_path):
    d = data.copy()
    y = d["churn"].to_numpy()
    # A column missing exactly for the positive class → null-mask predicts y.
    d["opt_field"] = np.where(y == 1, np.nan, 1.0)
    ctx = fsp.open_run(d, target="churn", target_type="binary", runs_dir=tmp_path / "runs")
    with pytest.raises(GateFailure):  # guarded: needs the split first
        leakage.missingness_signals(ctx)
    ctx.folds = partition.make_folds(d, "stratified", k=5, target="churn")
    leakage.missingness_signals(ctx)
    assert ctx.leaks.for_column("opt_field")


def test_auc_gets_pvalue_and_qvalue(data, tmp_path):
    ctx = fsp.open_run(data, target="churn", target_type="binary", runs_dir=tmp_path / "runs")
    ctx.folds = partition.make_folds(data, "stratified", k=5, target="churn")
    r = relevance.relevance(ctx, "signal", "continuous")
    assert r["p"] == r["p"] and r["p"] < 0.05  # AUC now carries a Mann–Whitney p
    out = relevance.relevance_all(ctx, {"signal": "continuous", "noise": "continuous"})
    assert out["q_value"].notna().all()  # q-values now populated for AUC features


def test_power_flag_on_small_data(tmp_path):
    rng = np.random.default_rng(1)
    n = 60  # 30 ≤ n < 100 → reduced-power tier
    y = rng.integers(0, 2, n)
    df = pd.DataFrame({"x": y + rng.normal(0, 1, n), "y": y})
    ctx = fsp.open_run(df, target="y", target_type="binary", runs_dir=tmp_path / "runs")
    ctx.folds = partition.make_folds(df, "stratified", k=3, target="y")
    r = relevance.relevance(ctx, "x", "continuous")
    assert r["power_flag"] == "reduced-power"


def test_backstop_and_separation_detectors(data, tmp_path):
    ctx = fsp.open_run(data, target="churn", target_type="binary", runs_dir=tmp_path / "runs")
    leakage.backstop_effects(ctx, {"signal": 0.7, "leak_score": 0.99}, threshold=0.85)
    assert ctx.leaks.for_column("leak_score") and not ctx.leaks.for_column("signal")
    ctx2 = fsp.open_run(data, target="churn", target_type="binary", runs_dir=tmp_path / "runs2")
    leakage.separation_signals(ctx2, ["signal", "leak_score"])
    assert ctx2.leaks.for_column("leak_score")  # perfectly separates the target


def test_derive_datetime_full_set():
    s = pd.Series(pd.date_range("2020-01-01", periods=50, freq="D"), name="signup")
    out = relevance.derive_datetime(s)
    for suff in (
        "year", "month", "dow", "hour", "is_weekend", "days_since_epoch",
        "month_sin", "month_cos", "dow_sin", "dow_cos", "hour_sin", "hour_cos", "recency_days",
    ):
        assert f"signup__{suff}" in out.columns
    assert out["signup__recency_days"].iloc[0] > out["signup__recency_days"].iloc[-1]


def test_boruta_confirms_categorical_signal():
    rng = np.random.default_rng(5)
    n = 400
    y = rng.integers(0, 2, n)
    cat = np.where(y == 1, "a", "b")
    cat = np.where(rng.random(n) < 0.1, "c", cat)  # noisy categorical signal
    df = pd.DataFrame({"catsig": cat, "noise": rng.normal(size=n), "y": y})
    status = boruta.crosscheck(df, ["catsig", "noise"], "y", "binary", max_iter=20)
    assert status.get("catsig") in {"confirmed", "tentative"}  # categorical is cross-checked


def test_structural_flags_protects_target_and_date(data):
    d = data.copy()
    d["signup_date"] = pd.date_range("2020-01-01", periods=len(d), freq="h")
    assert inventory.structural_flags(d, "signup_date") == "identifier"  # all-unique
    assert inventory.structural_flags(d, "signup_date", protected=("signup_date",)) is None


def test_partition_and_viability_helpers():
    assert partition.recommended_k("reduced-power") == 3
    assert partition.recommended_k("full") == 5
    # per-fold floor checks the real minimum across folds, not an average:
    assert T.per_fold_floor_met([12, 11, 13, 10, 14]) is True  # every fold ≥ 10
    assert T.per_fold_floor_met([12, 11, 2, 13, 14]) is False  # one starved fold fails
    assert T.per_fold_floor_met([]) is False  # no folds → not met
    # the average-based old logic would have PASSED this (mean 10.4) — the min (2) fails
    assert T.per_fold_floor_met([2, 18, 18, 18, 18]) is False
    assert partition.suggest_strategy(has_repeating_id=False, has_date=True) == "time"
    assert partition.suggest_strategy(has_repeating_id=True, has_date=False) == "grouped"
    assert partition.suggest_strategy(has_repeating_id=False, has_date=False) == "stratified"


def test_frame_dates_grain_and_target_type(data):
    d = data.copy()
    d["signup"] = pd.date_range("2020-01-01", periods=len(d), freq="h")
    assert "signup" in frame.date_candidates(d)
    assert frame.suggest_target_type(d["churn"]) == "binary"
    assert frame.suggest_target_type(pd.Series(range(100))) == "regression"
    assert frame.grain_facts(d, ["customer_id"])["customer_id"] is False  # unique per row


def test_values_sentinels_and_distribution(data):
    d = data.copy()
    d["age"] = np.where(np.arange(len(d)) < 10, -999.0, 30.0)
    assert "-999" in values.sentinel_candidates(d).get("age", {})
    nulled = values.null_sentinels(d, {"age": [-999.0]})
    assert nulled["age"].isna().sum() == 10
    assert "mean" in values.distribution(d, "signal")
    assert values.missingness(nulled)["age"] > 0


def test_report_tables_and_figures(data):
    from matplotlib.figure import Figure

    vt = fsp.report.tables.viability_table(viability.assess(data, "churn", "binary"))
    assert len(vt) == 1 and "effective_n" in vt.columns
    prof = inventory.profile(data).assign(semantic_type="continuous")
    assert "column" in fsp.report.tables.inventory_table(prof).columns
    led = pd.DataFrame(
        {
            "column": ["a", "b"],
            "verdict": ["keep", "drop"],
            "effect": [0.9, 0.1],
            "semantic_type": ["continuous", "continuous"],
            "metric_name": ["AUC", "AUC"],
            "reason": ["r", "r"],
        }
    )
    assert len(fsp.report.tables.top_keeps(led)) == 1  # only the 'keep' row
    assert isinstance(fsp.report.figures.distribution_hist(data["signal"]), Figure)
    assert isinstance(fsp.report.figures.effect_ranking(led), Figure)
    types = {"signal": "continuous", "noise": "continuous"}
    pairs = redundancy.pairwise(data, ["signal", "noise"], types)
    assert isinstance(fsp.report.figures.redundancy_heatmap(pairs), Figure)


def test_survival_relevance_path(tmp_path):
    rng = np.random.default_rng(4)
    n = 300
    x = rng.normal(size=n)
    df = pd.DataFrame(
        {
            "x": x,
            "T": rng.exponential(scale=np.exp(-0.5 * x)),  # x drives survival time
            "E": rng.integers(0, 2, n),
        }
    )
    ctx = fsp.open_run(
        df, target="T", target_type="survival", event_col="E", runs_dir=tmp_path / "runs"
    )
    ctx.folds = partition.make_folds(df, "stratified", k=5, target="E")  # stratify on event
    r = relevance.relevance(ctx, "x", "continuous")
    assert r["metric_name"] == "C-index"
    assert np.isnan(r["effect"]) or 0.5 <= r["effect"] <= 1.0


def test_rank_within_type_ranks_per_type():
    led = pd.DataFrame(
        {
            "column": ["a", "b", "c"],
            "semantic_type": ["continuous", "continuous", "nominal"],
            "effect": [0.6, 0.8, 0.3],
        }
    )
    ranks = fsp.report.tables.rank_within_type(led)
    assert ranks["b"] == 1 and ranks["a"] == 2 and ranks["c"] == 1


def test_high_card_relevance_uses_oof(tmp_path):
    rng = np.random.default_rng(3)
    n = 1000
    y = rng.integers(0, 2, n)
    df = pd.DataFrame(
        {
            "hc_noise": rng.integers(0, 250, n).astype(str),  # high-card noise
            "y": y,
        }
    )
    ctx = fsp.open_run(df, target="y", target_type="binary", runs_dir=tmp_path / "runs")
    ctx.folds = partition.make_folds(df, "stratified", k=5, target="y")
    r = relevance.relevance(ctx, "hc_noise", "high_card")
    insample = fsp.metrics.information_value(df["hc_noise"].to_numpy(), y, dtype="categorical")
    assert r["metric_name"] == "IV (oof)"
    # out-of-fold IV strips the in-sample high-card inflation (§17.3)
    assert r["effect"] < insample


def test_boruta_confirms_signal(data):
    status = boruta.crosscheck(data, ["signal", "noise"], "churn", "binary", max_iter=20)
    assert status.get("signal") in {"confirmed", "tentative"}


def test_report_scorecard_and_notebook(data, tmp_path):
    ctx = fsp.open_run(data, target="churn", target_type="binary", runs_dir=tmp_path / "runs")
    ctx.ledger.upsert("signal", semantic_type="continuous", verdict="keep", effect=0.8, reason="x")
    ctx.ledger.upsert("constant", semantic_type="constant", verdict="structural-drop", reason="c")
    sc = fsp.report.scorecard.scorecard(ctx.ledger.to_frame())
    assert sc["verdict_counts"]["keep"] == 1
    fig = fsp.report.figures.missingness_bar(data)
    ctx.notebook.add_section("D · Value integrity", body="mapped it", figures=[("missing", fig)])
    assert ctx.notebook.export_html().exists()
