"""The live results notebook — a documented section per part (playbook §14).

Sections are **addressable by title**: re-adding a section REPLACES it in place, and
opening a run reloads any existing `results.ipynb` first. So re-running a single part
rewrites only that part's section and leaves the others intact — the notebook is never
regenerated wholesale, and running the parts in any order (or one at a time) converges
to the same document.
"""

from __future__ import annotations

import base64
import io
import numbers
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import nbformat
import pandas as pd


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, numbers.Integral):
        return str(int(v))
    if isinstance(v, numbers.Real):
        return f"{float(v):.4g}"
    return str(v).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _md_table(df: pd.DataFrame, max_rows: int = 200) -> str:
    shown = df.head(max_rows)
    headers = [str(c) for c in shown.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in shown.itertuples(index=False):
        lines.append("| " + " | ".join(_fmt(v) for v in row) + " |")
    body = "\n".join(lines)
    if len(df) > max_rows:
        body += f"\n\n_…{len(df) - max_rows} more rows._"
    return body


def _kv_table(d: dict[str, Any]) -> str:
    rows = pd.DataFrame({"field": list(d.keys()), "value": [_fmt(v) for v in d.values()]})
    return _md_table(rows)


def _blockquote(text: Any) -> str:
    """Render a (possibly multi-line) note as a single markdown blockquote.

    Guards the most common miswiring: a note passed as a plain string must become
    one quoted block, never one cell per character."""
    lines = str(text).splitlines() or [""]
    return "\n".join(f"> {ln}" if ln else ">" for ln in lines)


def _figure_cell(fig: Any) -> Any:
    """A source-hidden code cell whose only output is the embedded PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    out = nbformat.v4.new_output("display_data", data={"image/png": b64}, metadata={})
    cell = nbformat.v4.new_code_cell(source="")
    cell.outputs = [out]
    cell.metadata = {"jupyter": {"source_hidden": True}}
    return cell


class Notebook:
    """Documented sections, addressable by title. Persists to disk immediately (live);
    re-adding a title replaces that section, and an existing notebook is reloaded on
    open so a partial re-run updates only the sections it touches."""

    def __init__(self, path: str | Path, *, title: str, subtitle: str = "") -> None:
        self.path = Path(path)
        header = f"# {title}\n\n{subtitle}" if subtitle else f"# {title}"
        self._header: Any = nbformat.v4.new_markdown_cell(header)
        self._sections: OrderedDict[str, list[Any]] = OrderedDict()
        if self.path.exists():
            self._load_sections()  # preserve sections written by earlier runs
        self._flush()

    def add_section(
        self,
        title: str,
        *,
        body: str = "",
        facts: dict[str, Any] | None = None,
        tables: list[tuple[str, pd.DataFrame]] | None = None,
        figures: list[tuple[str, Any]] | None = None,
        notes: str | list[str] | None = None,
    ) -> None:
        """Write (or overwrite) the section titled `title`. Re-adding an existing
        title replaces that section in place; a new title is appended. `notes`
        accepts a string or a list of strings — each becomes one blockquote."""
        header = nbformat.v4.new_markdown_cell(f"## {title}")
        header.metadata = {"fsp_section": title}
        cells: list[Any] = [header]
        if body:
            cells.append(nbformat.v4.new_markdown_cell(body))
        if facts:
            cells.append(nbformat.v4.new_markdown_cell("**Facts**\n\n" + _kv_table(facts)))
        if isinstance(notes, str):
            notes = [notes]
        for note in notes or []:
            cells.append(nbformat.v4.new_markdown_cell(_blockquote(note)))
        for caption, tbl in tables or []:
            head = f"**{caption}**\n\n" if caption else ""
            cells.append(nbformat.v4.new_markdown_cell(head + _md_table(tbl)))
        for caption, fig in figures or []:
            if caption:
                cells.append(nbformat.v4.new_markdown_cell(f"**{caption}**"))
            cells.append(_figure_cell(fig))
        self._sections[title] = cells  # replace-in-place (ordered) or append if new
        self._flush()

    def _load_sections(self) -> None:
        """Reload sections from an existing notebook so a partial re-run keeps the
        parts it did not touch. Sections are keyed by the `fsp_section` metadata on
        their heading cell (falling back to a `## ` heading); anything before the
        first heading — the H1 title — is dropped in favor of the fresh header."""
        try:
            nb = nbformat.read(str(self.path), as_version=4)
        except Exception:
            return
        current: str | None = None
        for cell in nb.cells:
            sec = (cell.get("metadata") or {}).get("fsp_section")
            src = str(cell.get("source", ""))
            if sec is None and cell.get("cell_type") == "markdown" and src.startswith("## "):
                sec = src[3:].splitlines()[0].strip()
            if sec is not None:
                current = sec
                self._sections[current] = [cell]
            elif current is not None:
                self._sections[current].append(cell)

    def _cell_list(self) -> list[Any]:
        out: list[Any] = [self._header]
        for cells in self._sections.values():
            out.extend(cells)
        return out

    def _notebook(self) -> Any:
        nb = nbformat.v4.new_notebook()
        nb.cells = self._cell_list()
        return nb

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        nbformat.write(self._notebook(), str(tmp))
        os.replace(tmp, self.path)

    def save(self) -> Path:
        self._flush()
        return self.path

    def export_html(self, path: str | Path | None = None) -> Path:
        from nbconvert import HTMLExporter

        html, _ = HTMLExporter().from_notebook_node(self._notebook())
        out = Path(path) if path else self.path.with_suffix(".html")
        out.write_text(html, encoding="utf-8")
        return out
