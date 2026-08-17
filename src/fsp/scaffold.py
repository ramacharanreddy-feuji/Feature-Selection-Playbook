"""Scaffold a new analysis project.

Drops the three guide docs (CLAUDE / PLAYBOOK / TOOLS), a gitignored `runs/`
output dir, and the phase-code home — `analysis/screening.py` (a runner) plus
`analysis/parts.py` (one `run_<x>(ctx)` per part) — into a target folder, so Claude
Code has its operating instructions *and* a place to write the screening. Exposed as
`fsp.scaffold(...)` and the `fsp init` CLI.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

DOCS = ("CLAUDE.md", "PLAYBOOK.md", "TOOLS.md")

# The runner — run ONE part (resumes prior state, no recompute) or the whole chain.
_RUNNER = '''"""Feature-selection screening — runner. Run ONE part or the whole chain.

    python analysis/screening.py          # A→H in one process (state stays in memory)
    python analysis/screening.py d        # ONLY Part D — resumes C's checkpoint, runs D, saves

It never recomputes a part you did not ask for: a single part resumes the prior
state via fsp.resume_run() (the coerced df, ctx.state, folds, ledger), and a full
run threads one ctx through in memory. A fixed run_id keeps every invocation in the
same runs/<id>/ folder, and results.ipynb updates only the sections you touch — it
is never regenerated wholesale.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # import the sibling parts.py

import fsp
import parts

RUN_ID = "screening"
ORDER = "abcdefgh"


def open_ctx():
    # Part A decides target/grain — state them as corrigible claims, confirm with a human.
    return fsp.open_run("data.csv", target="TARGET", target_type="binary", run_id=RUN_ID)


def main(argv):
    sel = argv[0].lower() if argv else None
    if sel is None:  # full chain — one process, state flows in memory
        ctx = open_ctx()
        for key in ORDER:
            ctx = getattr(parts, f"run_{key}")(ctx)
        ctx.checkpoint()
        return
    if sel not in ORDER:
        raise SystemExit(f"unknown part {sel!r}; pick one of {list(ORDER)} or omit for all")
    ctx = open_ctx() if sel == "a" else fsp.resume_run(RUN_ID)  # resume — no recompute
    getattr(parts, f"run_{sel}")(ctx)
    ctx.checkpoint()


if __name__ == "__main__":
    main(sys.argv[1:])
'''

# The eight parts — one run_<x>(ctx) each, filled ONE at a time, live (§1, §3.1).
_PARTS = '''"""The eight screening parts — one run_<x>(ctx) each (PLAYBOOK.md §7 A–H).

Fill ONE at a time, LIVE (§1, §3.1): call the part's fsp tools, READ the output,
decide from what you see, document the section, pass the gate — THEN the next part.
Do not batch-write the whole chain. Each run_<x> gets the ctx resumed from the
previous part and returns it; touch only this part's notebook section (add_section
rewrites just that one). Stash cross-part state on ctx.state (it is checkpointed),
e.g. ctx.state["feature_types"] = {...} in run_c for run_f/run_g to reuse.

RE-RUN SAFETY — a part's checkpoint stores the *mutated* ctx.df, and resume_run
hands that mutated frame back. So re-running a part sees already-transformed data.
Write every mutation to be idempotent, because some failures are silent: a blank
detector re-run just finds 0 blanks (a gate may catch it), but
`(df["col"] == "Yes").astype(int)` on an already-integer column silently sets every
row to 0 — same shape, same dtype, no error, whole run meaningless. Guard before you
transform (e.g. `if df["col"].dtype == object:`), or make the op a no-op the second
time. When in doubt, re-run the *earlier* part to rebuild the frame from clean data.
"""

import fsp  # noqa: F401
from fsp import thresholds as T  # noqa: F401  (§9 numbers — read, never hardcode)
from fsp.parts import (  # noqa: F401
    boruta, frame, inventory, leakage, partition, redundancy, relevance, values, viability,
)


def run_a(ctx):
    """A · Frame — target, grain, ids, dates (the mandatory human checkpoint)."""
    # frame.target_candidates / target_facts / date_candidates / id_candidates / grain_facts
    # decide ctx.config.* as corrigible claims, confirm with a human
    # ctx.notebook.add_section("A · Frame", body=..., facts=...) ; ctx.gate("A", {...})
    return ctx


def run_b(ctx):
    """B · Viability — positives, effective_n, strictness tier (§10)."""
    return ctx


def run_c(ctx):
    """C · Inventory — semantic type per column + structural drops; name-leak scan.
    Stash the decided types: ctx.state["feature_types"] = {...}."""
    return ctx


def run_d(ctx):
    """D · Value integrity — sentinels (null them), missingness, co-missing; D leak detectors."""
    return ctx


def run_e(ctx):
    """E · Partition — pick the split (§11), freeze folds. The leakage guard opens here."""
    return ctx


def run_f(ctx):
    """F · Relevance + stability — per-feature effect/CI/q/shape/shadow (split frozen)."""
    return ctx


def run_g(ctx):
    """G · Redundancy — collapse near-duplicates ≥ 0.95 among the keeps; VIF flag."""
    return ctx


def run_h(ctx):
    """H · Verdict — adjudicate leaks (§12.3), Boruta cross-check, final verdicts, deliverables."""
    # ... finalize every verdict + reason ...
    # ctx.save_ledger("ledger.parquet"); ctx.save_ledger("ledger.csv")
    # ctx.notebook.export_html(); fsp.provenance.save(ctx)
    return ctx
'''


def _read_doc(name: str) -> str:
    """Read a bundled guide doc: from the installed wheel's `fsp/assets/`
    (hatch force-include), falling back to the repo root in a source checkout."""
    try:
        res = resources.files("fsp").joinpath("assets", name)
        if res.is_file():
            return res.read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError, NotADirectoryError):
        pass
    root = Path(__file__).resolve().parents[2]  # src/fsp/scaffold.py → repo root
    p = root / name
    if p.is_file():
        return p.read_text(encoding="utf-8")
    raise FileNotFoundError(f"bundled doc {name!r} not found — the install may be incomplete")


def scaffold(dest: str | Path = ".", *, overwrite: bool = False) -> list[str]:
    """Copy the guide docs + the `analysis/` phase-code starter (`screening.py`
    runner + `parts.py`) into `dest`, and gitignore `runs/` plus the (regenerable)
    guide docs. Returns the files written; existing files are left untouched unless
    `overwrite=True`."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    for name in DOCS:
        target = dest / name
        if target.exists() and not overwrite:
            continue
        target.write_text(_read_doc(name), encoding="utf-8")
        written.append(name)

    (dest / "runs").mkdir(exist_ok=True)
    # .gitignore: run outputs + the package-provided guide docs (regenerable with
    # `fsp init`), so a user's repo tracks only their own work (analysis/, data).
    gitignore = dest / ".gitignore"
    existing = gitignore.read_text(encoding="utf-8") if gitignore.exists() else ""
    present = {line.strip() for line in existing.splitlines()}
    to_ignore = [e for e in ("runs/", *DOCS) if e not in present]
    if to_ignore:
        with gitignore.open("a", encoding="utf-8") as f:
            if existing and not existing.endswith("\n"):
                f.write("\n")
            f.write("\n".join(to_ignore) + "\n")

    # A home for the phase code (tracked, not gitignored): a runner + one file of parts.
    (dest / "analysis").mkdir(exist_ok=True)
    for rel, text in (("analysis/screening.py", _RUNNER), ("analysis/parts.py", _PARTS)):
        target = dest / rel
        if not target.exists() or overwrite:
            target.write_text(text, encoding="utf-8")
            written.append(rel)

    return written
