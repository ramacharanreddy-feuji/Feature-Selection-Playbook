"""The metrics kernel — the playbook §17 math as tested functions.

Deterministic and orientation-consistent so results are comparable across runs.
Prefer these over re-implementing a formula (playbook §1).
"""

from .association import (
    auc,
    auc_ovr,
    cliffs_delta,
    correlation_ratio,
    kendall,
    kruskal_eps2,
    mann_whitney_p,
    pearson,
    point_biserial,
    spearman,
)
from .categorical import (
    bergsma_v,
    cramers_v,
    information_value,
    iv_oof,
    target_encoded_eta_oof,
    woe_table,
)
from .drift import psi, vif
from .resampling import bh_fdr, bootstrap_ci, permutation_p, shadow_floor, shadow_samples
from .shape import shape_gap, x_statistic
from .survival import concordance, cox_screen, logrank

__all__ = [
    # association (§17.4, §17.9–17.11)
    "auc",
    "auc_ovr",
    "spearman",
    "kendall",
    "pearson",
    "point_biserial",
    "mann_whitney_p",
    "cliffs_delta",
    "correlation_ratio",
    "kruskal_eps2",
    # categorical (§17.1, §17.3)
    "bergsma_v",
    "cramers_v",
    "information_value",
    "iv_oof",
    "target_encoded_eta_oof",
    "woe_table",
    # shape (§17.2)
    "x_statistic",
    "shape_gap",
    # survival (§17.8)
    "cox_screen",
    "concordance",
    "logrank",
    # drift / collinearity (§17.7, §9)
    "psi",
    "vif",
    # resampling engines (§17.5, §17.6, §17.12)
    "shadow_floor",
    "shadow_samples",
    "permutation_p",
    "bootstrap_ci",
    "bh_fdr",
]
