"""The ledger — one row per column, the audit-trail deliverable (playbook §14)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

# The seven verdicts (playbook §5). Guarded so a typo can't reach the output.
VERDICTS = {
    "keep",
    "review",
    "drop",
    "redundant",
    "engineer",
    "leak-suspect",
    "structural-drop",
}

# §14 schema — always emitted, so the output contract is stable.
LEDGER_FIELDS = [
    "column",
    "semantic_type",
    "verdict",
    "metric_name",
    "effect",
    "effect_ci",
    "q_value",
    "shadow_floor",
    "shape_gap",
    "fold_spread",
    "redundant_with",
    "leak_flag",
    "reason",
    "rank_within_type",
    "power_flag",
]


class Ledger:
    def __init__(self) -> None:
        self._rows: dict[str, dict[str, Any]] = {}

    def upsert(self, column: str, **fields: Any) -> None:
        verdict = fields.get("verdict")
        if verdict is not None and verdict not in VERDICTS:
            raise ValueError(f"invalid verdict {verdict!r}; allowed: {sorted(VERDICTS)}")
        self._rows.setdefault(column, {"column": column}).update(fields)

    def get(self, column: str) -> dict[str, Any]:
        return self._rows.get(column, {})

    def __len__(self) -> int:
        return len(self._rows)

    def to_frame(self) -> pd.DataFrame:
        if not self._rows:
            return pd.DataFrame(columns=LEDGER_FIELDS)
        df = pd.DataFrame(list(self._rows.values()))
        extra = [c for c in df.columns if c not in LEDGER_FIELDS]
        return df.reindex(columns=LEDGER_FIELDS + extra)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        frame = self.to_frame()
        if p.suffix.lower() == ".csv":
            frame.to_csv(p, index=False)
        else:
            frame.to_parquet(p, index=False)
        return p
