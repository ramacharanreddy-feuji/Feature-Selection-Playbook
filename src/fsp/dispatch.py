"""The §8 dispatch table as code: which effect metric per feature × target type.

Returns the *primary* metric for a cell. Part F also computes the shape gap
(binary target), CI, and shadow floor around this same effect function.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from numpy.typing import ArrayLike

from . import metrics

FEATURE_TYPES = {
    "continuous",
    "count",
    "nominal",
    "ordinal",
    "binary",
    "high_card",
    "datetime",
}
TARGET_TYPES = {"binary", "multiclass", "ordinal", "regression", "survival"}

EffectFn = Callable[[ArrayLike, ArrayLike], float]
PFn = Callable[[ArrayLike, ArrayLike], float]


@dataclass
class MetricSpec:
    name: str
    kind: str  # "assoc" | "survival" | "derive"
    fn: EffectFn | None = None  # effect(feature, target) -> float
    p_fn: PFn | None = None  # optional p-value(feature, target) -> float


# --- effect wrappers (magnitude on the metric's native scale) ---------------


def _iv_cat(x: ArrayLike, y: ArrayLike) -> float:
    return metrics.information_value(x, y, dtype="categorical")


def _abs_spearman(x: ArrayLike, y: ArrayLike) -> float:
    return abs(metrics.spearman(x, y)[0])


def _p_spearman(x: ArrayLike, y: ArrayLike) -> float:
    return metrics.spearman(x, y)[1]


def _abs_cliffs(x: ArrayLike, y: ArrayLike) -> float:
    return abs(metrics.cliffs_delta(x, y))


def _abs_pbs(x: ArrayLike, y: ArrayLike) -> float:
    return abs(metrics.point_biserial(x, y)[0])


def _p_pbs(x: ArrayLike, y: ArrayLike) -> float:
    return metrics.point_biserial(x, y)[1]


def _kruskal_e(x: ArrayLike, y: ArrayLike) -> float:
    return metrics.kruskal_eps2(x, y)[0]


def _kruskal_p(x: ArrayLike, y: ArrayLike) -> float:
    return metrics.kruskal_eps2(x, y)[1]


_NUMERIC = {"continuous", "count"}


def metric_for(feature_type: str, target_type: str) -> MetricSpec:
    """Return the primary metric spec for a feature × target pair (§8)."""
    if feature_type not in FEATURE_TYPES:
        raise ValueError(f"unknown feature_type {feature_type!r}")
    if target_type not in TARGET_TYPES:
        raise ValueError(f"unknown target_type {target_type!r}")

    if feature_type == "datetime":
        return MetricSpec("derive-first", "derive")
    if target_type == "survival":
        return MetricSpec("Cox", "survival")

    if target_type == "binary":
        if feature_type in _NUMERIC:
            return MetricSpec("AUC", "assoc", metrics.auc, metrics.mann_whitney_p)
        return MetricSpec("IV", "assoc", _iv_cat)

    if target_type == "multiclass":
        if feature_type in _NUMERIC:
            return MetricSpec("KW ε²", "assoc", _kruskal_e, _kruskal_p)
        return MetricSpec("Cramér's V", "assoc", metrics.bergsma_v)

    if target_type == "ordinal":
        if feature_type in _NUMERIC or feature_type == "ordinal":
            return MetricSpec("Spearman ρ", "assoc", _abs_spearman, _p_spearman)
        if feature_type == "binary":
            return MetricSpec("Cliff's δ", "assoc", _abs_cliffs)
        return MetricSpec("Cramér's V", "assoc", metrics.bergsma_v)

    # regression
    if feature_type in _NUMERIC or feature_type == "ordinal":
        return MetricSpec("Spearman ρ", "assoc", _abs_spearman, _p_spearman)
    if feature_type == "binary":
        return MetricSpec("point-biserial", "assoc", _abs_pbs, _p_pbs)
    return MetricSpec("η²", "assoc", metrics.correlation_ratio)
