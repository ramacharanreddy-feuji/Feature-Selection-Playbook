"""Layer 3 — figures the agent embeds in the notebook (via Figure objects)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

_BLUE, _GREEN = "#4C78A8", "#54A24B"


def missingness_bar(df: pd.DataFrame, top: int = 15) -> Figure:
    miss = df.isna().mean().sort_values(ascending=False).head(top)
    fig = Figure(figsize=(6, max(2.0, 0.4 * len(miss) + 1)))
    ax = fig.subplots()
    ax.barh(list(miss.index)[::-1], list(miss.to_numpy())[::-1], color=_BLUE)
    ax.set_xlabel("missing fraction")
    ax.set_title("Missingness by column (top)")
    return fig


def distribution_hist(series: pd.Series) -> Figure:
    fig = Figure(figsize=(5, 3))
    ax = fig.subplots()
    ax.hist(pd.Series(series).dropna().to_numpy(), bins=30, color=_BLUE)
    ax.set_title(str(getattr(series, "name", "")))
    return fig


def effect_ranking(ledger: pd.DataFrame, feature_type: str | None = None, top: int = 20) -> Figure:
    d = ledger.dropna(subset=["effect"])
    if feature_type is not None:
        d = d[d["semantic_type"] == feature_type]
    d = d.sort_values("effect", ascending=False).head(top)
    fig = Figure(figsize=(6, max(2.0, 0.4 * len(d) + 1)))
    ax = fig.subplots()
    ax.barh(list(d["column"])[::-1], list(d["effect"])[::-1], color=_GREEN)
    ax.set_xlabel("effect")
    ax.set_title("Top features by effect")
    return fig


def redundancy_heatmap(pairs: pd.DataFrame) -> Figure:
    fig = Figure(figsize=(5, 4))
    ax = fig.subplots()
    im = ax.imshow(pairs.to_numpy(), vmin=0, vmax=1, cmap="magma")
    ticks = np.arange(len(pairs))
    ax.set_xticks(ticks)
    ax.set_xticklabels(pairs.columns, rotation=90, fontsize=6)
    ax.set_yticks(ticks)
    ax.set_yticklabels(pairs.columns, fontsize=6)
    fig.colorbar(im, ax=ax)
    ax.set_title("Pairwise similarity")
    return fig
