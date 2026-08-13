"""Layer 3 — the closing scorecard and review worklist (playbook §14.1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..calibration import drop_rate_by_rule


def scorecard(ledger: pd.DataFrame) -> dict[str, Any]:
    counts = ledger["verdict"].value_counts(dropna=True) if len(ledger) else pd.Series(dtype=int)
    return {
        "n_columns": int(len(ledger)),
        "verdict_counts": {str(k): int(v) for k, v in counts.items()},
        "drop_rate_by_rule": drop_rate_by_rule(ledger),
    }


def review_export(ledger: pd.DataFrame, path: str | Path) -> Path:
    """Write the `review` band as a tickable CSV for a human to work through."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ledger[ledger["verdict"] == "review"].to_csv(p, index=False)
    return p


def limits_note() -> str:
    """The §13 honesty note to include in the closing section."""
    return (
        "**Limits.** This is a screening tool, not an auto-selector — a human makes the "
        "final call. It cannot catch Part-A *silent* errors (wrong grain, ambiguous "
        "negatives, survivorship) and is blind to **interactions** (features weak alone "
        "but strong together). Dropped columns stay in the ledger; nothing is deleted."
    )
