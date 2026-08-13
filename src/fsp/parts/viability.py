"""Part B — Viability facts (playbook §7 B). Aggregate target facts only."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..thresholds import tier_for


def assess(
    df: pd.DataFrame,
    target: str,
    target_type: str,
    *,
    event_col: str | None = None,
) -> dict[str, Any]:
    """Positives/events, effective n, prevalence, censoring, target nulls (§7 B)."""
    s = df[target]
    target_nulls = int(s.isna().sum())
    present = int(s.notna().sum())

    positives: int | None = None
    prevalence: float | None = None
    censoring: float | None = None
    effective_n = present

    if target_type in {"binary", "multiclass", "ordinal"}:
        vc = s.dropna().value_counts()
        positives = int(vc.min()) if len(vc) else 0
        prevalence = float(positives / vc.sum()) if len(vc) else None
        effective_n = positives
    elif target_type == "survival" and event_col is not None:
        events = int(pd.Series(df[event_col]).fillna(0).astype(float).sum())
        positives = events
        effective_n = events
        censoring = float(1.0 - events / len(df)) if len(df) else None

    return {
        "positives": positives,
        "effective_n": effective_n,
        "prevalence": None if prevalence is None else round(prevalence, 4),
        "censoring": None if censoring is None else round(censoring, 4),
        "target_nulls": target_nulls,
    }


def tier(effective_n: int) -> str:
    return tier_for(effective_n)
