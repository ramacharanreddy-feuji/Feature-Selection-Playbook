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


def test_checkpoint_resume_continues_without_recompute(tmp_path):
    import nbformat

    df = pd.DataFrame({"id": range(20), "amt": [1.0, 2, 3, 4] * 5, "churn": [0, 1] * 10})
    expected_amt = (df["amt"] * 10).tolist()  # capture before the in-place mutation below
    runs = tmp_path / "runs"

    # part 1: set up state (mutated df, scratch, ledger, folds) and checkpoint
    ctx = fsp.open_run(df, target="churn", target_type="binary", run_id="chain", runs_dir=runs)
    ctx.df["amt"] = ctx.df["amt"] * 10  # a mutation that must survive the round-trip
    ctx.state["feature_types"] = {"amt": "continuous"}
    ctx.ledger.upsert("id", semantic_type="identifier", verdict="structural-drop", reason="id")
    ctx.folds = partition.make_folds(ctx.df, "stratified", k=2, target="churn", run_dir=ctx.run_dir)
    ctx.notebook.add_section("C · Inventory", body="typed amt")
    assert ctx.checkpoint().exists()

    # part 2: a fresh process resumes — NO recompute — and everything is restored
    r = fsp.resume_run("chain", runs_dir=runs)
    assert r.df["amt"].tolist() == expected_amt  # mutated df preserved
    assert r.state["feature_types"] == {"amt": "continuous"}  # scratch preserved
    assert r.ledger.get("id")["verdict"] == "structural-drop"  # ledger preserved
    assert r.folds is not None and r.folds.k == 2  # folds reloaded from disk
    r.notebook.add_section("D · Value integrity", body="handled")  # C stays intact
    doc = nbformat.read(str(r.run_dir / "results.ipynb"), as_version=4)
    secs = {(c.get("metadata") or {}).get("fsp_section") for c in doc.cells}
    assert {"C · Inventory", "D · Value integrity"} <= secs

    with pytest.raises(FileNotFoundError, match="no checkpoint"):
        fsp.resume_run("never-ran", runs_dir=runs)


def test_notebook_notes_string_is_one_blockquote(tmp_path):
    # regression: a note passed as a str must be ONE blockquote, not one cell per char
    from fsp.notebook import Notebook

    nb = Notebook(tmp_path / "r.ipynb", title="R")
    nb.add_section("H · Verdict", body="done", notes="Limits. A multi-word note.")
    import nbformat

    doc = nbformat.read(str(tmp_path / "r.ipynb"), as_version=4)
    quotes = [c for c in doc.cells if str(c.source).startswith(">")]
    assert len(quotes) == 1 and "multi-word note" in quotes[0].source


def test_notebook_sections_are_addressable_and_reload(tmp_path):
    from fsp.notebook import Notebook

    p = tmp_path / "results.ipynb"
    nb = Notebook(p, title="R")
    nb.add_section("A · Frame", body="A first.")
    nb.add_section("B · Viability", body="B stuff.")
    nb.add_section("A · Frame", body="A REVISED.")  # replace in place, don't duplicate

    import nbformat

    srcs = [str(c.source) for c in nbformat.read(str(p), as_version=4).cells]
    assert sum(s.startswith("## A · Frame") for s in srcs) == 1  # not duplicated
    assert any("A REVISED." in s for s in srcs) and not any("A first." in s for s in srcs)
    a_i = next(i for i, s in enumerate(srcs) if s.startswith("## A · Frame"))
    b_i = next(i for i, s in enumerate(srcs) if s.startswith("## B · Viability"))
    assert a_i < b_i  # original order preserved on replace

    # a fresh Notebook on the same path (a new process / partial re-run) reloads B,
    # and updating only A leaves B intact — the whole notebook is not regenerated
    nb2 = Notebook(p, title="R")
    nb2.add_section("A · Frame", body="A AGAIN")
    srcs2 = [str(c.source) for c in nbformat.read(str(p), as_version=4).cells]
    assert any("A AGAIN" in s for s in srcs2)  # A updated
    assert any("B stuff." in s for s in srcs2)  # B survived the reload
    assert sum(s.startswith("## A · Frame") for s in srcs2) == 1


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


def test_scaffold_writes_docs_runs_and_starter(tmp_path):
    written = fsp.scaffold(tmp_path)
    assert set(written) == {
        "CLAUDE.md", "PLAYBOOK.md", "TOOLS.md", "analysis/screening.py", "analysis/parts.py",
    }
    for name in ("CLAUDE.md", "PLAYBOOK.md", "TOOLS.md"):
        assert (tmp_path / name).read_text(encoding="utf-8").strip()  # non-empty
    assert (tmp_path / "runs").is_dir()
    gi = (tmp_path / ".gitignore").read_text()
    # run outputs + the regenerable guide docs are gitignored; the user's code is not
    for entry in ("runs/", "CLAUDE.md", "PLAYBOOK.md", "TOOLS.md"):
        assert entry in gi
    assert "analysis/" not in gi and "screening.py" not in gi  # phase code stays tracked
    # the runner resumes single parts (no recompute) and pins a fixed run_id
    runner = (tmp_path / "analysis" / "screening.py").read_text(encoding="utf-8")
    assert "resume_run" in runner and "checkpoint" in runner and 'RUN_ID = "screening"' in runner
    # parts.py: one run_<x> per part, with A→H markers
    parts_py = (tmp_path / "analysis" / "parts.py").read_text(encoding="utf-8")
    assert "import fsp" in parts_py
    assert "def run_a(ctx)" in parts_py and "def run_h(ctx)" in parts_py
    assert "A · Frame" in parts_py and "H · Verdict" in parts_py
    # idempotent: a second call writes nothing new and doesn't duplicate .gitignore lines
    assert fsp.scaffold(tmp_path) == []
    gi2 = (tmp_path / ".gitignore").read_text()
    assert gi2.count("runs/") == 1 and gi2.count("CLAUDE.md") == 1


def test_cli_init(tmp_path, capsys):
    assert cli.main(["init", str(tmp_path)]) == 0
    assert (tmp_path / "PLAYBOOK.md").exists()
    assert "wrote" in capsys.readouterr().out
