"""Layer 0 foundation tests: context, gates, ledger, notebook, dispatch, folds."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import fsp
from fsp import cli, dispatch, thresholds
from fsp.folds import Folds
from fsp.gates import GateFailure
from fsp.parts import partition


@pytest.fixture
def sample(tmp_path):
    df = pd.DataFrame({"id": range(20), "x": np.arange(20.0), "churn": [0, 1] * 10})
    return df, tmp_path


def test_open_run_starts_notebook_and_validates(sample):
    df, tmp = sample
    ctx = fsp.open_run(df, target="churn", target_type="binary", runs_dir=tmp / "runs")
    assert (ctx.run_dir / "results.ipynb").exists()
    assert ctx.config.target == "churn" and ctx.config.seed == 42
    with pytest.raises(ValueError, match="not found"):
        fsp.open_run(df, target="nope", runs_dir=tmp / "runs")
    with pytest.raises(ValueError, match="target_type"):
        fsp.open_run(df, target="churn", target_type="weird", runs_dir=tmp / "runs")


def test_gate_pass_and_fail(sample):
    df, tmp = sample
    ctx = fsp.open_run(df, runs_dir=tmp / "runs")
    assert ctx.gate("a", {"ok": True}) is True
    assert (ctx.run_dir / "decision_cards" / "a.json").exists()
    with pytest.raises(GateFailure):
        ctx.gate("b", {"bad": False})
    with pytest.raises(GateFailure):
        ctx.gate("c", {})  # empty never passes vacuously


def test_ledger_upsert_validates_and_saves(sample, tmp_path):
    led = fsp.Ledger()
    led.upsert("x", semantic_type="continuous", verdict="keep", reason="strong")
    with pytest.raises(ValueError):
        led.upsert("y", verdict="nonsense")
    frame = led.to_frame()
    assert "verdict" in frame.columns and frame.loc[0, "column"] == "x"
    p = led.save(tmp_path / "ledger.csv")
    assert pd.read_csv(p).loc[0, "verdict"] == "keep"


def test_notebook_section_and_html(sample):
    df, tmp = sample
    ctx = fsp.open_run(df, runs_dir=tmp / "runs")
    ctx.notebook.add_section("A · Frame", body="Found a target.", facts={"n_rows": 20})
    html = ctx.notebook.export_html()
    assert html.exists() and "A · Frame" in html.read_text()


def test_dispatch_covers_the_table():
    assert dispatch.metric_for("continuous", "binary").name == "AUC"
    assert dispatch.metric_for("nominal", "binary").name == "IV"
    assert dispatch.metric_for("nominal", "regression").name == "η²"
    assert dispatch.metric_for("binary", "regression").name == "point-biserial"
    assert dispatch.metric_for("continuous", "survival").kind == "survival"
    assert dispatch.metric_for("datetime", "binary").kind == "derive"
    # the effect fn actually runs
    spec = dispatch.metric_for("continuous", "binary")
    assert spec.fn is not None
    assert spec.fn(np.array([1.0, 2, 3, 4]), np.array([0, 0, 1, 1])) == pytest.approx(1.0)


def test_thresholds_helpers():
    assert thresholds.tier_for(500) == "full"
    assert thresholds.tier_for(50) == "reduced-power"
    assert thresholds.tier_for(10) == "structural-only"
    assert thresholds.cramers_v_floor(2) == pytest.approx(0.05)
    assert thresholds.cramers_v_floor(5) == pytest.approx(0.025)


def test_folds_roundtrip(tmp_path):
    folds = Folds(splits=[(np.array([0, 1]), np.array([2, 3]))], strategy="kfold", k=1, seed=42)
    p = folds.save(tmp_path / "folds.json")
    back = Folds.load(p)
    assert back.strategy == "kfold" and np.array_equal(back.splits[0][1], np.array([2, 3]))


def test_io_read_csv(tmp_path):
    p = tmp_path / "d.csv"
    pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}).to_csv(p, index=False)
    df = fsp.io.read(p)
    assert list(df.columns) == ["a", "b"] and len(df) == 3


def test_make_folds_persist_and_load(sample, tmp_path):
    df, _ = sample
    folds = partition.make_folds(df, "stratified", k=2, target="churn", run_dir=tmp_path)
    assert (tmp_path / "folds.json").exists()
    back = partition.load_folds(tmp_path)
    assert back.k == 2 and len(back) == 2
    assert np.array_equal(back.splits[0][0], folds.splits[0][0])


def test_provenance_and_calibration(sample, tmp_path):
    df, tmp = sample
    ctx = fsp.open_run(df, target="churn", target_type="binary", runs_dir=tmp / "runs")
    ctx.ledger.upsert("x", semantic_type="continuous", verdict="keep", reason="ok")
    ctx.ledger.upsert("id", semantic_type="identifier", verdict="structural-drop", reason="id col")

    man = fsp.provenance.manifest(ctx)
    assert man["seed"] == 42 and "numpy" in man["libraries"]
    assert fsp.provenance.save(ctx).exists()

    rec = fsp.calibration.log_run(ctx)
    assert rec["verdict_counts"]["keep"] == 1
    assert rec["drop_rate_by_rule"]  # one structural-drop → non-empty
    assert fsp.calibration.append(tmp_path / "cal.jsonl", rec).exists()


def test_scaffold_writes_docs_and_runs(tmp_path):
    written = fsp.scaffold(tmp_path)
    assert set(written) == {"CLAUDE.md", "PLAYBOOK.md", "TOOLS.md"}
    for name in ("CLAUDE.md", "PLAYBOOK.md", "TOOLS.md"):
        assert (tmp_path / name).read_text(encoding="utf-8").strip()  # non-empty
    assert (tmp_path / "runs").is_dir()
    assert "runs/" in (tmp_path / ".gitignore").read_text()
    # idempotent: a second call writes nothing new and doesn't duplicate .gitignore
    assert fsp.scaffold(tmp_path) == []
    assert (tmp_path / ".gitignore").read_text().count("runs/") == 1


def test_cli_init(tmp_path, capsys):
    assert cli.main(["init", str(tmp_path)]) == 0
    assert (tmp_path / "PLAYBOOK.md").exists()
    assert "wrote" in capsys.readouterr().out
