"""Gates — self-checks that halt the run (playbook §4.5)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .context import RunContext


class GateFailure(Exception):
    def __init__(self, part: str, reasons: list[str]) -> None:
        self.part = part
        self.reasons = reasons
        super().__init__(f"Gate '{part}' failed: {', '.join(reasons)}")


def gate(
    ctx: RunContext,
    part: str,
    conditions: dict[str, bool],
    *,
    notes: list[str] | None = None,
) -> bool:
    """Record a decision card and raise `GateFailure` unless every condition
    holds. Empty conditions count as a failure (never pass vacuously)."""
    passed = bool(conditions) and all(conditions.values())
    card = {
        "part": part,
        "conditions": conditions,
        "passed": passed,
        "notes": notes or [],
    }
    cards = ctx.run_dir / "decision_cards"
    cards.mkdir(parents=True, exist_ok=True)
    (cards / f"{part}.json").write_text(json.dumps(card, indent=2, default=str))

    if not passed:
        failed = [k for k, v in conditions.items() if not v] or ["no conditions declared"]
        raise GateFailure(part, failed)
    return True
