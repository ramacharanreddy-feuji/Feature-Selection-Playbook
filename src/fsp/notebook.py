"""The live results notebook — grows a documented section per part (playbook §14)."""

from __future__ import annotations

import base64
import io
import numbers
import os
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
    """Append documented sections; persists to disk immediately (live)."""

    def __init__(self, path: str | Path, *, title: str, subtitle: str = "") -> None:
        self.path = Path(path)
        header = f"# {title}\n\n{subtitle}" if subtitle else f"# {title}"
        self._cells: list[Any] = [nbformat.v4.new_markdown_cell(header)]
        self._flush()

    def add_section(
        self,
        title: str,
        *,
        body: str = "",
        facts: dict[str, Any] | None = None,
        tables: list[tuple[str, pd.DataFrame]] | None = None,
        figures: list[tuple[str, Any]] | None = None,
        notes: list[str] | None = None,
    ) -> None:
        cells: list[Any] = [nbformat.v4.new_markdown_cell(f"## {title}")]
        if body:
            cells.append(nbformat.v4.new_markdown_cell(body))
        if facts:
            cells.append(nbformat.v4.new_markdown_cell("**Facts**\n\n" + _kv_table(facts)))
        for note in notes or []:
            cells.append(nbformat.v4.new_markdown_cell(f"> {note}"))
        for caption, tbl in tables or []:
            head = f"**{caption}**\n\n" if caption else ""
            cells.append(nbformat.v4.new_markdown_cell(head + _md_table(tbl)))
        for caption, fig in figures or []:
            if caption:
                cells.append(nbformat.v4.new_markdown_cell(f"**{caption}**"))
            cells.append(_figure_cell(fig))
        self._cells.extend(cells)
        self._flush()

    def _notebook(self) -> Any:
        nb = nbformat.v4.new_notebook()
        nb.cells = list(self._cells)
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
