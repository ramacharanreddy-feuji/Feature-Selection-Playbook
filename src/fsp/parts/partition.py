"""Part E — Partition (playbook §7 E, §11): pick a strategy, freeze the folds."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit

from ..folds import Folds


def suggest_strategy(*, has_repeating_id: bool, has_date: bool) -> str:
    """The §11 decision tree (a suggestion; Claude confirms)."""
    if has_date:
        return "time"
    if has_repeating_id:
        return "grouped"
    return "stratified"


def recommended_k(tier: str) -> int:
    """Fold count for the strictness tier (§10/§11): 3 folds under reduced-power,
    5 otherwise. (`structural-only` runs no F statistics, but returns 3 too.)"""
    return 3 if tier in {"reduced-power", "structural-only"} else 5


def make_folds(
    df: pd.DataFrame,
    strategy: str,
    k: int = 5,
    *,
    seed: int = 42,
    target: str | None = None,
    group: str | None = None,
    date: str | None = None,
    run_dir: str | Path | None = None,
) -> Folds:
    """Build and return frozen (train_idx, test_idx) folds for the strategy.

    If `run_dir` is given, the folds are also written to `run_dir/folds.json`
    (§7 E: freeze to disk so every later part and the modeling team reuse them).
    """
    n = len(df)
    idx = np.arange(n)

    if strategy == "time":
        if date is None:
            raise ValueError("time strategy needs a date column")
        order = np.argsort(pd.to_datetime(df[date], errors="coerce").to_numpy())
        tss = TimeSeriesSplit(n_splits=k)
        raw = [(order[tr], order[te]) for tr, te in tss.split(order)]
    elif strategy == "grouped":
        if group is None:
            raise ValueError("grouped strategy needs a group column")
        gkf = GroupKFold(n_splits=k)
        raw = list(gkf.split(idx, groups=df[group].to_numpy()))
    elif strategy == "stratified":
        if target is None:
            raise ValueError("stratified strategy needs a target column")
        labels = pd.factorize(df[target])[0]  # NaN → -1, keeps StratifiedKFold happy
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        raw = list(skf.split(idx, labels))
    else:
        kf = KFold(n_splits=k, shuffle=True, random_state=seed)
        raw = list(kf.split(idx))

    splits = [(np.asarray(tr, dtype=int), np.asarray(te, dtype=int)) for tr, te in raw]
    folds = Folds(splits=splits, strategy=strategy, k=k, seed=seed)
    if run_dir is not None:
        folds.save(Path(run_dir) / "folds.json")
    return folds


def load_folds(run_dir: str | Path) -> Folds:
    """Reload the frozen split written by `make_folds(run_dir=...)` (§7 E)."""
    return Folds.load(Path(run_dir) / "folds.json")


def fold_positive_counts(
    folds: Folds, y: pd.Series | np.ndarray, *, positive_label: object = None
) -> list[int]:
    """Positives in each fold's held-out (test) block — the counts
    `thresholds.per_fold_floor_met` checks (§9). The positive class defaults to the
    minority level unless `positive_label` is given. `y` is positional (same order
    as the rows the folds index), so it must not be pre-filtered."""
    ys = pd.Series(np.asarray(y))
    if positive_label is None:
        vc = ys.dropna().value_counts()
        if vc.empty:
            return [0 for _ in folds.splits]
        positive_label = vc.index[-1]  # minority class
    pos = (ys == positive_label).to_numpy()
    return [int(pos[te].sum()) for _, te in folds.splits]
