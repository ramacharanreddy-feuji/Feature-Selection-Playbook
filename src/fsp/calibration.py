"""Drop-rate-per-rule log — the primary calibration signal (playbook §9.3)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from .context import RunContext


def drop_rate_by_rule(ledger: pd.DataFrame) -> dict[str, int]:
    """Count how many columns each drop reason removed (§9.3)."""
    dropped = ledger[ledger["verdict"].isin(["drop", "structural-drop", "redundant"])]
    if dropped.empty:
        return {}
    reasons = dropped["reason"].fillna("(unspecified)").astype(str)
    # collapse to the leading phrase before any parenthetical/quote detail
    keys = reasons.str.replace(r"[\(\'\"].*", "", regex=True).str.strip()
    return {str(k): int(v) for k, v in keys.value_counts().items()}


def log_run(ctx: RunContext) -> dict[str, Any]:
    ledger = ctx.ledger.to_frame()
    counts = ledger["verdict"].value_counts(dropna=True) if len(ledger) else pd.Series(dtype=int)
    return {
        "run_id": ctx.run_id,
        "n_columns": int(len(ledger)),
        "verdict_counts": {str(k): int(v) for k, v in counts.items()},
        "drop_rate_by_rule": drop_rate_by_rule(ledger),
    }


def append(path: str | Path, record: dict[str, Any]) -> Path:
    """Append one run's calibration record to a shared JSONL log."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
    return p
