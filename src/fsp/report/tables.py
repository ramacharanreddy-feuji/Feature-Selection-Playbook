"""Layer 3 — tables the agent embeds in the notebook."""

from __future__ import annotations

from typing import Any

import pandas as pd


def viability_table(assess: dict[str, Any]) -> pd.DataFrame:
    """Part B's viability facts as a one-row frame ready to embed (§7 B)."""
    return pd.DataFrame([{k: v for k, v in assess.items() if not isinstance(v, dict)}])


def inventory_table(profile: pd.DataFrame) -> pd.DataFrame:
    wanted = ["column", "semantic_type", "dtype", "n_unique", "missing_rate"]
    keep = [c for c in wanted if c in profile]
    return profile[keep] if keep else profile


def top_keeps(ledger: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    keeps = ledger[ledger["verdict"] == "keep"]
    if "effect" in keeps.columns:
        keeps = keeps.sort_values("effect", ascending=False)
    cols = [c for c in ["column", "semantic_type", "metric_name", "effect", "reason"] if c in keeps]
    return keeps.head(n)[cols]


def ledger_view(ledger: pd.DataFrame) -> pd.DataFrame:
    wanted = ["column", "semantic_type", "verdict", "metric_name", "effect", "reason"]
    return ledger[[c for c in wanted if c in ledger]]


def rank_within_type(ledger: pd.DataFrame) -> dict[str, int]:
    """Position of each column within its semantic type by effect, descending
    (§14 `rank_within_type`). Rows with no effect rank last, by name. Returns
    {column: rank} for Claude to upsert — never ranks across types (§8)."""
    if "semantic_type" not in ledger or "column" not in ledger:
        return {}
    out: dict[str, int] = {}
    eff = ledger["effect"] if "effect" in ledger else pd.Series(index=ledger.index, dtype=float)
    work = ledger.assign(_eff=pd.to_numeric(eff, errors="coerce"))
    for _, grp in work.groupby("semantic_type", dropna=False):
        ordered = grp.sort_values(["_eff", "column"], ascending=[False, True], na_position="last")
        for rank, col in enumerate(ordered["column"], start=1):
            out[str(col)] = rank
    return out
