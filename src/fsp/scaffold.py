"""Scaffold a new analysis project.

Drops the three guide docs (CLAUDE / PLAYBOOK / TOOLS) into a target folder so
Claude Code has its operating instructions when it runs the screening against
the user's data. Exposed as `fsp.scaffold(...)` and the `fsp init` CLI.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

DOCS = ("CLAUDE.md", "PLAYBOOK.md", "TOOLS.md")


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
    """Copy the guide docs into `dest`, make a gitignored `runs/`, and return the
    files written. Existing docs are left untouched unless `overwrite=True`."""
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
    gitignore = dest / ".gitignore"
    existing = gitignore.read_text(encoding="utf-8") if gitignore.exists() else ""
    if not any(line.strip() == "runs/" for line in existing.splitlines()):
        with gitignore.open("a", encoding="utf-8") as f:
            if existing and not existing.endswith("\n"):
                f.write("\n")
            f.write("runs/\n")

    return written
