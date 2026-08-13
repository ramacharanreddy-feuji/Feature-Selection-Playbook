"""RunContext — the spine every tool reads and writes (see TOOLS.md §6)."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from . import io
from .folds import Folds
from .gates import gate as _gate
from .ledger import Ledger
from .notebook import Notebook

TARGET_TYPES = {"binary", "multiclass", "ordinal", "regression", "survival"}


@dataclass
class RunConfig:
    """Frame config (playbook Part A) — decided by the agent, frozen here."""

    target: str | None = None
    target_type: str | None = None
    event_col: str | None = None  # for survival targets (duration = target)
    date_col: str | None = None
    reference_date: str | None = None  # per-row prediction time (for future-leak, §12.2)
    id_cols: list[str] = field(default_factory=list)
    grain: str | None = None
    prevalence: float | None = None
    strictness_tier: str | None = None
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LeakSignal:
    part: str
    column: str
    detector: str
    ltype: str
    evidence: str = ""


class LeakRegister:
    """Accumulates leak signals across parts C→D→F; adjudicated once at H (§12)."""

    def __init__(self) -> None:
        self._signals: list[LeakSignal] = []

    def add(self, part: str, column: str, detector: str, ltype: str, evidence: Any = "") -> None:
        self._signals.append(LeakSignal(part, column, detector, ltype, str(evidence)))

    def for_column(self, column: str) -> list[LeakSignal]:
        return [s for s in self._signals if s.column == column]

    def signals(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(s) for s in self._signals])

    def __len__(self) -> int:
        return len(self._signals)


class RunContext:
    def __init__(self, df: pd.DataFrame, config: RunConfig, run_id: str, run_dir: Path) -> None:
        self.df = df
        self.config = config
        self.run_id = run_id
        self.run_dir = run_dir
        self.ledger = Ledger()
        self.leaks = LeakRegister()
        self.folds: Folds | None = None
        self.state: dict[str, Any] = {}
        self.notebook = Notebook(
            run_dir / "results.ipynb",
            title="Feature Selection — Results",
            subtitle=f"Run `{run_id}` · {len(df):,} rows × {df.shape[1]} columns",
        )

    def gate(
        self, part: str, conditions: dict[str, bool], *, notes: list[str] | None = None
    ) -> bool:
        return _gate(self, part, conditions, notes=notes)

    def save_ledger(self, name: str = "ledger.parquet") -> Path:
        return self.ledger.save(self.run_dir / name)


def open_run(
    source: str | Path | pd.DataFrame,
    *,
    runs_dir: str | Path = "runs",
    run_id: str | None = None,
    seed: int = 42,
    **frame_hints: Any,
) -> RunContext:
    """Open a run: read the data, make a run directory, start the notebook.

    `frame_hints` are optional RunConfig fields the user supplied (target,
    target_type, date_col, reference_date, event_col, id_cols, grain).
    Everything else is decided by the agent in Part A.
    """
    df = source if isinstance(source, pd.DataFrame) else io.read(source)

    tt = frame_hints.get("target_type")
    if tt is not None and tt not in TARGET_TYPES:
        raise ValueError(f"target_type must be one of {sorted(TARGET_TYPES)}, got {tt!r}")
    named = [frame_hints.get(k) for k in ("target", "date_col", "event_col", "reference_date")]
    for col in named:
        if col is not None and col not in df.columns:
            raise ValueError(f"column {col!r} not found; available: {list(df.columns)}")

    if run_id is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{stamp}-{uuid.uuid4().hex[:6]}"
    run_dir = Path(runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    config = RunConfig(seed=seed, **frame_hints)
    return RunContext(df, config, run_id, run_dir)
