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
