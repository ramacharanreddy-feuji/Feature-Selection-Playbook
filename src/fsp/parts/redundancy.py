"""Part G — Redundancy (playbook §7 G): typed pairwise, components, representative."""

from __future__ import annotations

from collections.abc import Mapping

import networkx as nx
import numpy as np
import pandas as pd

from .. import metrics
from ..metrics import vif  # re-export (§9)

__all__ = ["pairwise", "components", "representative", "vif"]

_NUMERIC = {"continuous", "count", "ordinal"}


def _kind(t: str) -> str:
    return "num" if t in _NUMERIC else "cat"


def _pair_score(df: pd.DataFrame, a: str, b: str, ta: str, tb: str) -> float:
    """A [0,1] similarity for a feature pair, chosen by type (§7 G)."""
    ka, kb = _kind(ta), _kind(tb)
    if ka == "num" and kb == "num":
        return abs(metrics.spearman(df[a], df[b])[0])
    if ka == "cat" and kb == "cat":
        return metrics.bergsma_v(df[a], df[b])
    cat, num = (a, b) if ka == "cat" else (b, a)
    return float(metrics.correlation_ratio(df[cat], df[num]) ** 0.5)  # η ∈ [0,1]


def pairwise(df: pd.DataFrame, features: list[str], types: Mapping[str, str]) -> pd.DataFrame:
    """Symmetric [0,1] similarity matrix over `features`."""
    n = len(features)
    mat = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            s = _pair_score(df, features[i], features[j], types[features[i]], types[features[j]])
            s = 0.0 if not np.isfinite(s) else float(s)
            mat[i, j] = mat[j, i] = s
    return pd.DataFrame(mat, index=features, columns=features)


def components(pairs: pd.DataFrame, threshold: float = 0.95) -> list[set[str]]:
    """Connected components on edges ≥ threshold (playbook §4.1)."""
    cols = [str(c) for c in pairs.columns]
    g = nx.Graph()
    g.add_nodes_from(cols)
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            if float(pairs.loc[a, b]) >= threshold:
                g.add_edge(a, b)
    return [set(c) for c in nx.connected_components(g) if len(c) >= 2]


def representative(
    component: set[str],
    effects: Mapping[str, float],
    missing: Mapping[str, float],
) -> str:
    """Deterministic pick: highest Part-F effect, ties → fewer missing, then name."""
    return sorted(
        component,
        key=lambda c: (-(effects.get(c) or 0.0), missing.get(c) or 0.0, c),
    )[0]
