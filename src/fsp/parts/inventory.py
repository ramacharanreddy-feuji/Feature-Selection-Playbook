"""Part C — Inventory facts + structural flags (playbook §7 C, §9.2)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..thresholds import NEAR_CONSTANT_SHARE, NEAR_EMPTY_MISSING


def _top_share(s: pd.Series) -> float:
    nn = s.dropna()
    return float(nn.value_counts(normalize=True).iloc[0]) if not nn.empty else 0.0


def column_facts(df: pd.DataFrame, col: str) -> dict[str, Any]:
    s = df[col]
    n = len(s)
    nu = int(s.nunique(dropna=True))
    return {
        "column": col,
        "dtype": str(s.dtype),
        "n_unique": nu,
        "missing_rate": round(float(s.isna().mean()), 4),
        "top_share": round(_top_share(s), 4),
        "all_unique": nu == int(s.notna().sum()) and nu == n,
        "sample": [str(v) for v in s.dropna().unique()[:5]],
    }


def profile(df: pd.DataFrame) -> pd.DataFrame:
    """One row of facts per column (vectorised-enough for wide data)."""
    return pd.DataFrame([column_facts(df, c) for c in df.columns])


def duplicate_columns(df: pd.DataFrame) -> dict[str, str]:
    """Map each column that exactly mirrors an earlier one to that earlier name."""
    seen: dict[tuple[str, tuple[Any, ...]], str] = {}
    dup: dict[str, str] = {}
    for col in df.columns:
        key = (str(df[col].dtype), tuple(df[col].fillna("\x00").tolist()))
        if key in seen:
            dup[str(col)] = seen[key]
        else:
            seen[key] = str(col)
    return dup


def structural_flags(
    df: pd.DataFrame,
    col: str,
    *,
    duplicate_of: str | None = None,
    protected: tuple[str, ...] = (),
) -> str | None:
    """A calibration-free structural-drop reason for a column, or None (§9.2).

    `protected` columns (the target and date column, §7 C) are never
    structural-dropped — a date column is usually all-unique and would otherwise
    read as an `identifier`."""
    if str(col) in {str(p) for p in protected}:
        return None
    facts = column_facts(df, col)
    if facts["n_unique"] <= 1:
        return "constant"
    if facts["top_share"] > NEAR_CONSTANT_SHARE:
        return "near_constant"
    if facts["missing_rate"] > NEAR_EMPTY_MISSING:
        return "near_empty"
    if facts["all_unique"] and not pd.api.types.is_float_dtype(df[col]):
        return "identifier"
    if duplicate_of is not None:
        return "duplicate"
    return None


def suggest_semantic_type(facts: dict[str, Any]) -> str:
    """Suggestion only (§8 feature types) — Claude confirms/overrides."""
    nu = facts["n_unique"]
    dtype = str(facts["dtype"])
    if nu <= 1:
        return "constant"
    if "datetime" in dtype:
        return "datetime"
    if "bool" in dtype:
        return "binary"
    if any(t in dtype for t in ("int", "float")):
        if nu == 2:
            return "binary"
        if "int" in dtype and facts["all_unique"]:
            return "identifier"
        if "int" in dtype and nu <= 15:
            return "nominal"
        if "int" in dtype and nu > 15:
            return "count"
        return "continuous"
    # object / string / category
    if facts["all_unique"]:
        return "identifier"
    return "high_card" if nu > 50 else "nominal"
