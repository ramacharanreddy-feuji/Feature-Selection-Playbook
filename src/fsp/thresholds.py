"""Playbook §9 thresholds as named constants — read these, never hardcode."""

from __future__ import annotations

# Auto-drop backstops (§9). Higher-is-better metrics: below → drop-eligible.
# η²/ε²/point-biserial sit one tier below their §17.9–17.11 "small" bands so every
# dispatch metric has a backstop (the shadow floor stays primary, §4.3).
AUTO_DROP = {
    "auc": 0.52,
    "iv": 0.01,
    "spearman": 0.05,
    "cliffs": 0.07,
    "cindex": 0.55,
    "pbs": 0.05,  # point-biserial, one tier below 0.1 weak
    "eta2": 0.005,  # correlation ratio η², one tier below 0.01 small
    "eps2": 0.005,  # Kruskal–Wallis ε², one tier below 0.01 small
}

# Leak flags (§9): a single feature this strong is a screaming leak.
LEAK_FLAG = {"auc": 0.85, "iv": 0.50, "cindex": 0.75}

REDUNDANCY_COLLAPSE = 0.95          # §4.1 design commitment — never lower
REDUNDANCY_REVIEW = (0.70, 0.95)    # review band
VIF_FLAG = 10.0                     # flag only, never drop
PSI_DRIFT = 0.25                    # significant drift
COX_SCREEN_P = 0.20                 # liberal univariate Cox screening
SHADOW_PCT = 95.0                   # §17.5 shadow-permutation percentile (tunable)

# Structural drops (§9.2).
NEAR_CONSTANT_SHARE = 0.98
NEAR_EMPTY_MISSING = 0.95

# Viability floors (§9, §7 B) + small-n ladder (§10).
VIABILITY_MIN_POSITIVES = 50
VIABILITY_MIN_PER_FOLD = 10
VIABILITY_MIN_REGRESSION_N = 100
TIER_FULL_N = 100
TIER_REDUCED_N = 30


def cramers_v_floor(min_levels: int) -> float:
    """Cardinality-dependent Cramér's V auto-drop floor (§9.1): one tier below
    Cohen's 'small' = 0.10 / √(min_levels − 1)."""
    if min_levels < 2:
        return float("nan")
    small = 0.10 / (min_levels - 1) ** 0.5
    return float(small / 2.0)


def tier_for(effective_n: int) -> str:
    """Strictness tier from the small-n ladder (§10)."""
    if effective_n >= TIER_FULL_N:
        return "full"
    if effective_n >= TIER_REDUCED_N:
        return "reduced-power"
    return "structural-only"


def viability_floor_met(target_type: str, *, positives: int | None, effective_n: int) -> bool:
    """§9 viability floor by target type."""
    if target_type in {"binary", "multiclass", "ordinal"}:
        return (positives or 0) >= VIABILITY_MIN_POSITIVES
    if target_type == "survival":
        return (positives or 0) >= VIABILITY_MIN_POSITIVES  # positives == events here
    return effective_n >= VIABILITY_MIN_REGRESSION_N


def per_fold_floor_met(
    positives: int, k: int, *, min_per_fold: int = VIABILITY_MIN_PER_FOLD
) -> bool:
    """§9 per-fold viability floor: ≥ `min_per_fold` positives in each of `k`
    folds (checked once folds exist, Part E)."""
    return k > 0 and positives / k >= min_per_fold
