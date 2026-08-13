"""Scaffold a new analysis project.

Drops the three guide docs (CLAUDE / PLAYBOOK / TOOLS), a gitignored `runs/`
output dir, and an `analysis/screening.py` starter driver into a target folder —
so Claude Code has its operating instructions *and* a home for the phase code
when it runs the screening. Exposed as `fsp.scaffold(...)` and the `fsp init` CLI.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

DOCS = ("CLAUDE.md", "PLAYBOOK.md", "TOOLS.md")

# Starter driver dropped at `analysis/screening.py` — one file for all parts, filled
# in **one part at a time** (never batch-written; PLAYBOOK.md §1/§3.1).
_STARTER = '''"""Feature-selection screening — driver. Run it LIVE (PLAYBOOK.md §1, §3.1).

Fill in ONE part at a time: call a part's fsp tools, READ the output, decide from
what you see, document it (ctx.notebook.add_section — facts + tables/figures), pass
the gate — THEN write the next part. Do not fill this in all at once. If a run grows,
split a phase into its own file (e.g. analysis/part_f.py) and import what you need.
"""

import fsp
from fsp import thresholds as T  # noqa: F401  (§9 numbers — read, never hardcode)
from fsp.parts import (  # noqa: F401
    boruta, frame, inventory, leakage, partition, redundancy, relevance, values, viability,
)

# Part A decides these — state them as corrigible claims, confirm with a human.
ctx = fsp.open_run("data.csv", target="TARGET", target_type="binary")

# ── A · Frame ──────────────────────────────────────────────────────────────────
# frame.target_candidates / target_facts / date_candidates / id_candidates / grain_facts
# → decide target/grain/date/ids · ctx.notebook.add_section("A · Frame", …) · ctx.gate("A", {…})

# ── B · Viability ──────────────────────────────────────────────────────────────

# ── C · Inventory (semantic types + structural drops; name-leak scan) ───────────

# ── D · Value integrity (sentinels, distributions, co-missing; D leak detectors) ─

# ── E · Partition (freeze folds — the leakage guard opens here) ─────────────────

# ── F · Relevance + stability (effect/CI/q/shape/shadow per feature) ────────────

# ── G · Redundancy (collapse near-duplicates ≥ 0.95 among the keeps) ────────────

# ── H · Verdict (adjudicate leaks §12.3, Boruta cross-check, final verdicts) ─────

# ── Deliverables ────────────────────────────────────────────────────────────────
# ctx.notebook.export_html(); ctx.save_ledger(); fsp.provenance.save(ctx)
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
    """Copy the guide docs + an `analysis/screening.py` starter into `dest`, and
    gitignore `runs/` plus the (regenerable) guide docs. Returns the files written;
    existing files are left untouched unless `overwrite=True`."""
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

    # A home for the phase code (tracked, not gitignored) — one file for all parts.
    (dest / "analysis").mkdir(exist_ok=True)
    starter = dest / "analysis" / "screening.py"
    if not starter.exists() or overwrite:
        starter.write_text(_STARTER, encoding="utf-8")
        written.append("analysis/screening.py")

    return written
