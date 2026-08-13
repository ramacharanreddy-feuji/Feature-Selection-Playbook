"""Part A — Frame facts (playbook §7 A). Facts + suggestions; Claude decides."""

from __future__ import annotations

from typing import Any

import pandas as pd

_TARGET_HINTS = ("target", "label", "y", "outcome", "churn", "default", "fraud", "class")
_DATE_HINTS = ("date", "time", "timestamp", "dt", "datetime")


def target_candidates(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if str(c).lower() in _TARGET_HINTS]


def suggest_target_type(s: pd.Series) -> str:
    vals = s.dropna()
    nu = int(vals.nunique())
    if nu <= 1:
        return "unknown"
    if nu == 2:
        return "binary"
    if pd.api.types.is_numeric_dtype(vals):
        if pd.api.types.is_integer_dtype(vals) and nu <= 15:
            return "multiclass"
        return "regression"
    return "multiclass"


def target_facts(df: pd.DataFrame, col: str) -> dict[str, Any]:
    s = df[col]
    vc = s.dropna().value_counts()
    return {
        "column": col,
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_nulls": int(s.isna().sum()),
        "class_balance": {str(k): int(v) for k, v in vc.head(10).items()},
        "suggested_type": suggest_target_type(s),
    }


def _looks_like_dates(s: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    sample = s.dropna().head(50)
    if sample.empty or pd.api.types.is_numeric_dtype(sample):
        return False
    parsed = pd.to_datetime(sample, errors="coerce")
    return bool(parsed.notna().mean() >= 0.8)


def date_candidates(df: pd.DataFrame) -> list[str]:
    hinted = [c for c in df.columns if any(h in str(c).lower() for h in _DATE_HINTS)]
    ordered = list(dict.fromkeys([*hinted, *df.columns]))
    return [str(c) for c in ordered if _looks_like_dates(df[c])]


def id_candidates(df: pd.DataFrame) -> list[str]:
    n = len(df)
    out = []
    for c in df.columns:
        name = str(c).lower()
        named = name == "id" or name.endswith("_id")  # avoid covid/grid/valid/paid
        if named or int(df[c].nunique(dropna=True)) == n:
            out.append(str(c))
    return out


def grain_facts(df: pd.DataFrame, id_cols: list[str]) -> dict[str, bool]:
    """For each id column, whether it repeats (grain coarser than the row)."""
    return {c: bool(df[c].duplicated().any()) for c in id_cols}
