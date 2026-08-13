"""Part D — Value integrity (playbook §7 D): sentinels, distributions, missingness."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ..gates import GateFailure

if TYPE_CHECKING:
    from ..context import RunContext

_NUM_SENTINELS = (-9999.0, -999.0, -99.0, 9999.0, 99999.0, 999999.0)
_STR_SENTINELS = {"", "na", "n/a", "nan", "none", "null", "unknown", "missing", "?", "-", "--", "."}


def sentinel_candidates(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Suspected sentinel values per column (corrigible — Claude confirms)."""
    out: dict[str, dict[str, int]] = {}
    for col in df.columns:
        s = df[col]
        hits: dict[str, int] = {}
        if pd.api.types.is_numeric_dtype(s):
            for v in _NUM_SENTINELS:
                c = int((s == v).sum())
                if c:
                    hits[str(int(v))] = c
        elif s.dtype == object:
            low = s.dropna().astype(str).str.strip().str.lower()
            for token, c in low.value_counts().items():
                if token in _STR_SENTINELS:
                    hits[repr(token)] = int(c)
        if hits:
            out[str(col)] = hits
    return out


def null_sentinels(df: pd.DataFrame, register: dict[str, list[Any]]) -> pd.DataFrame:
    """Replace confirmed sentinel values with NaN (§4.4 edge 2). Returns a copy."""
    out = df.copy()
    for col, values in register.items():
        out[col] = out[col].replace(list(values), np.nan)
    return out


def distribution(df: pd.DataFrame, col: str) -> dict[str, Any]:
    s = df[col].dropna()
    if s.empty or not pd.api.types.is_numeric_dtype(s):
        return {"column": col, "flags": ""}
    skew = float(s.skew()) if len(s) > 2 else 0.0
    zero_frac = float((s == 0).mean())
    flags = []
    if zero_frac > 0.5:
        flags.append("zero-inflated")
    if abs(skew) > 2:
        flags.append("skewed")
    return {
        "column": col,
        "min": round(float(s.min()), 4),
        "max": round(float(s.max()), 4),
        "mean": round(float(s.mean()), 4),
        "skew": round(skew, 4),
        "zero_frac": round(zero_frac, 4),
        "flags": ",".join(flags),
    }


def missingness(df: pd.DataFrame) -> pd.Series:
    return df.isna().mean().round(4)


def comissing_clusters(df: pd.DataFrame, threshold: float = 0.95) -> list[list[str]]:
    """Columns that go missing together (correlated null-masks), grouped."""
    miss_cols = [str(c) for c in df.columns if bool(df[c].isna().any())]
    if len(miss_cols) < 2:
        return []
    mask = df[miss_cols].isna().astype(int)
    corr = mask.corr().fillna(0.0)
    g = nx.Graph()
    g.add_nodes_from(miss_cols)
    for i, a in enumerate(miss_cols):
        for b in miss_cols[i + 1 :]:
            if float(corr.loc[a, b]) >= threshold:
                g.add_edge(a, b)
    return [sorted(c) for c in nx.connected_components(g) if len(c) >= 2]


def missingness_predicts_target(ctx: RunContext, col: str) -> float:
    """Leak diagnostic (§12): does a column's missingness predict a binary
    target? Leakage-guarded — requires the split to exist first (§4.4 edge 3)."""
    if ctx.folds is None:
        raise GateFailure("values.missingness_predicts_target", ["folds not frozen (Part E)"])
    target = ctx.config.target
    if target is None or ctx.config.target_type != "binary":
        return float("nan")
    y = ctx.df[target]
    mask = y.notna()
    is_missing = ctx.df[col].isna().astype(int)[mask]
    yv = y[mask]
    if yv.nunique() < 2 or is_missing.nunique() < 2:
        return float("nan")
    a = float(roc_auc_score(yv, is_missing))
    return max(a, 1.0 - a)
