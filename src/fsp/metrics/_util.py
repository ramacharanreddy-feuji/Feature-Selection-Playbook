"""Shared helpers for the metric kernel."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


def clean_pair(x: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Available-case pairing (playbook §7 F): drop rows where either side is
    null, so every metric runs on complete pairs and never imputes."""
    xs = pd.Series(np.asarray(x))
    ys = pd.Series(np.asarray(y))
    mask = (xs.notna() & ys.notna()).to_numpy()
    return xs[mask].to_numpy(), ys[mask].to_numpy()


def as_binary01(y: ArrayLike) -> np.ndarray:
    """Coerce a two-class target to float {0, 1} for the IV / optbinning path.

    A string target (`Yes`/`No`) or non-0/1 codes (`{1, 2}`) would otherwise reach
    optbinning as an object column and surface as an opaque
    `TypeError: agg function failed [how->mean, dtype->object]` — with no mention
    of the target. Numeric y already valued in {0, 1} passes through; any other
    two-class y is factorised (sorted, so the mapping is deterministic — IV
    magnitude is invariant to which class is 1). A target that is *not* binary
    raises a typed error naming the class count, so a mis-typed target fails loudly."""
    ys = pd.Series(np.asarray(y))
    vals = pd.unique(ys.dropna())
    if len(vals) != 2:
        raise ValueError(
            f"information_value needs a binary (0/1) target; got {len(vals)} distinct "
            f"value(s) {list(vals)[:6]}. Coerce the target to two classes first "
            "(multiclass → one-vs-rest)."
        )
    if pd.api.types.is_numeric_dtype(ys) and set(pd.unique(ys.dropna())) <= {0, 1}:
        return np.asarray(ys.to_numpy(dtype=float), dtype=float)
    codes = pd.factorize(ys, sort=True)[0].astype(float)
    codes[ys.isna().to_numpy()] = np.nan
    return np.asarray(codes, dtype=float)
