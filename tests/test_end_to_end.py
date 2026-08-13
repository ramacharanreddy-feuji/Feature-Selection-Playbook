"""End-to-end: drive Parts A→H on a known-answer dataset and score the verdicts.

This is the integration proof — the tools compose into a working screening, and
the agent's decision logic (thresholds from §9) lands the right verdicts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import fsp
from fsp import thresholds as T
from fsp.parts import inventory, leakage, partition, redundancy, relevance, viability


def _known_answer_frame() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    n = 500
    y = rng.integers(0, 2, n)
    df = pd.DataFrame(
        {
            "customer_id": range(n),                 # identifier  → structural-drop
            "signal": y + rng.normal(0, 0.4, n),     # real         → keep
            "noise": rng.normal(0, 1, n),            # noise        → drop
            "region": rng.choice(list("abcd"), n),   # nominal noise→ drop
            "constant": 7,                           # constant     → structural-drop
            "leak_score": y.astype(float),           # a leak       → leak-suspect
            "churn": y,
        }
    )
    df["signal_dup"] = df["signal"]                  # exact dup    → structural-drop
    return df


def test_end_to_end_screens_known_answer(tmp_path):
    df = _known_answer_frame()
    ctx = fsp.open_run(df, target="churn", target_type="binary", runs_dir=tmp_path / "runs")

    # B — viability sets the tier
    v = viability.assess(df, "churn", "binary")
    ctx.config.strictness_tier = viability.tier(v["effective_n"])
    assert ctx.config.strictness_tier == "full"

    # C — semantic types + structural drops; name-leak signals fire
    dups = inventory.duplicate_columns(df)
    feature_types: dict[str, str] = {}
    for col in df.columns:
        if col == "churn":
            continue
        facts = inventory.column_facts(df, col)
        stype = inventory.suggest_semantic_type(facts)
        flag = inventory.structural_flags(df, col, duplicate_of=dups.get(col))
        if flag:
            ctx.ledger.upsert(col, semantic_type=stype, verdict="structural-drop", reason=flag)
        else:
            ctx.ledger.upsert(col, semantic_type=stype)
            feature_types[col] = stype
    leakage.name_signals(ctx)

    # E — freeze the split
    ctx.folds = partition.make_folds(df, "stratified", k=5, target="churn")

    # F — relevance on the surviving features, decide verdicts per §9
    testable = {c: t for c, t in feature_types.items() if t not in {"identifier", "constant"}}
    rel = relevance.relevance_all(ctx, testable)
    effects: dict[str, float] = {}
    for row in rel.to_dict("records"):
        col, eff, floor = row["column"], row.get("effect"), row.get("shadow_floor")
        effects[col] = eff if eff is not None else float("nan")
        if eff is None or eff != eff:
            verdict = "review"
        elif eff < (floor or 0) or eff < T.AUTO_DROP["auc"]:
            verdict = "drop"
        else:
            verdict = "keep"
        ctx.ledger.upsert(
            col, verdict=verdict, effect=eff, metric_name=row["metric_name"],
            shadow_floor=floor, reason=f"{row['metric_name']}={eff}",
        )
    leakage.outlier_effects(ctx, effects)

    # G — collapse near-duplicates among the keeps
    keeps = [c for c in testable if ctx.ledger.get(c).get("verdict") == "keep"]
    if len(keeps) >= 2:
        pairs = redundancy.pairwise(df, keeps, feature_types)
        missing = {c: inventory.column_facts(df, c)["missing_rate"] for c in keeps}
        for comp in redundancy.components(pairs, T.REDUNDANCY_COLLAPSE):
            rep = redundancy.representative(comp, effects, missing)
            for c in comp - {rep}:
                ctx.ledger.upsert(c, verdict="redundant", redundant_with=rep, reason=f"~{rep}")

    # H — adjudicate leaks once; give every column a reason
    for col, flag in leakage.adjudicate(ctx).items():
        if ctx.ledger.get(col).get("verdict") != "structural-drop":
            ctx.ledger.upsert(col, verdict="leak-suspect", leak_flag=flag, reason="encodes target")

    # --- score against ground truth ---
    led = ctx.ledger
    assert led.get("signal")["verdict"] == "keep"
    assert led.get("noise")["verdict"] == "drop"
    assert led.get("region")["verdict"] == "drop"
    assert led.get("constant")["verdict"] == "structural-drop"
    assert led.get("customer_id")["verdict"] == "structural-drop"
    assert led.get("signal_dup")["verdict"] == "structural-drop"  # exact dup caught at C
    assert led.get("leak_score")["verdict"] == "leak-suspect"

    # deliverables exist
    ctx.notebook.add_section(
        "H · Verdict",
        body="Screening complete.",
        tables=[("Ledger", fsp.report.tables.ledger_view(led.to_frame()))],
        notes=[fsp.report.scorecard.limits_note()],
    )
    assert ctx.notebook.export_html().exists()
    assert ctx.save_ledger().exists()
    sc = fsp.report.scorecard.scorecard(led.to_frame())
    assert sc["verdict_counts"]["keep"] >= 1
