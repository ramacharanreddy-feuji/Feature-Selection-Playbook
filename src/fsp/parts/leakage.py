"""Part H — leak detectors + adjudication (playbook §12).

Detectors fire at C/D/F and add signals to the run's LeakRegister; adjudicated
once, as a batch, at H. Heuristics, and labelled as such.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .. import metrics
from ..context import LeakRegister as Register  # noqa: F401 — TOOLS.md name alias

if TYPE_CHECKING:
    from ..context import RunContext

_NAME_TOKENS = ("score", "prediction", "proba", "target", "label", "outcome", "_final", "_flag")


def name_signals(ctx: RunContext) -> None:
    """Fire at C: columns named like the target / a post-outcome artefact (§12.2)."""
    target = ctx.config.target
    for col in ctx.df.columns:
        name = str(col).lower()
        if str(col) == str(target):
            continue
        if any(tok in name for tok in _NAME_TOKENS):
            ctx.leaks.add("C", str(col), "target-like-name", "direct", name)


def outlier_effects(ctx: RunContext, effects: Mapping[str, float]) -> None:
    """Fire at F: an effect that is an outlier versus its peers (§12, primary)."""
    vals = np.array([v for v in effects.values() if v is not None and v == v])
    if len(vals) < 5:
        return
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) or 1e-9
    for col, e in effects.items():
        if e is not None and e == e and (e - med) / (1.4826 * mad) > 3.5:
            ctx.leaks.add("F", str(col), "outlier-effect-vs-peers", "direct", f"{e:.3f}")


def backstop_effects(ctx: RunContext, effects: Mapping[str, float], *, threshold: float) -> None:
    """Fire at F: an effect above the fixed §9 leak backstop (§12.2). Pass the
    metric's `LEAK_FLAG` value (AUC 0.85 / IV 0.50 / C-index 0.75). The
    outlier-vs-peers scan stays primary; this is the fixed-threshold backstop."""
    for col, e in effects.items():
        if e is not None and e == e and e >= threshold:
            ctx.leaks.add("F", str(col), "effect-above-backstop", "direct", f"{e:.3f}")


def separation_signals(ctx: RunContext, features: list[str], *, min_count: int = 5) -> None:
    """Fire at F: a feature that (near-)perfectly separates a binary target
    (§12.2). Categorical → any level of ≥ `min_count` rows that is 100% one
    class; numeric → single-feature AUC ≈ 1. Binary target only."""
    target = ctx.config.target
    if target is None or ctx.config.target_type != "binary":
        return
    y = ctx.df[target]
    ymask = y.notna().to_numpy()
    codes = pd.Series(pd.factorize(y[ymask])[0], index=ctx.df.index[ymask])
    for feat in features:
        s = ctx.df[feat][ymask]
        numeric = pd.api.types.is_numeric_dtype(s)
        if not numeric or int(s.nunique()) <= 10:
            grp = pd.DataFrame({"v": s.astype("object"), "y": codes}).dropna()
            if grp.empty:
                continue
            agg = grp.groupby("v")["y"].agg(["mean", "count"])
            pure = (agg["count"] >= min_count) & agg["mean"].isin([0.0, 1.0])
            if bool(pure.any()):
                ctx.leaks.add("F", str(feat), "perfect-separation", "direct", "pure level")
                continue
        if numeric:
            a = metrics.auc(s, y[ymask])
            if a == a and a >= 0.999:
                ctx.leaks.add("F", str(feat), "perfect-separation", "direct", f"AUC {a:.3f}")


def future_timestamp(ctx: RunContext, *, reference: object = None) -> None:
    """Fire at D: a datetime column with values *after* prediction time (§12.2,
    future). Reference precedence: an explicit `reference`, else the frame's
    `reference_date` column (the true prediction time), else `date_col` per row
    (event time — weaker, since event-time ≠ prediction-time; §7 A). A permitted
    pre-split D diagnostic (no target statistic)."""
    ref_col = ctx.config.reference_date or ctx.config.date_col
    ref_series = None
    if reference is not None:
        ref_series = pd.Series(pd.Timestamp(str(reference)), index=ctx.df.index)
    elif ref_col is not None:
        ref_series = pd.to_datetime(ctx.df[ref_col], errors="coerce")
    if ref_series is None:
        return  # no reference to judge against
    skip = {str(ctx.config.date_col), str(ctx.config.reference_date), str(ctx.config.target)}
    for col in ctx.df.columns:
        if str(col) in skip:
            continue
        s = pd.to_datetime(ctx.df[col], errors="coerce")
        if float(s.notna().mean()) < 0.8:
            continue  # not really a datetime column
        frac = float((s > ref_series).mean())
        if frac > 0.0:
            ctx.leaks.add("D", str(col), "future-timestamp", "future", f"{frac:.1%} after ref")


def only_present_for_positives(ctx: RunContext) -> None:
    """Fire at D: a column present almost only for one target class (§12.2,
    execution-dependent). Binary target only; uses the missingness pattern (a
    permitted pre-split D diagnostic), not a per-feature target statistic."""
    target = ctx.config.target
    if target is None or ctx.config.target_type != "binary":
        return
    y = ctx.df[target]
    ymask = y.notna().to_numpy()
    codes = pd.factorize(y[ymask])[0]
    for col in ctx.df.columns:
        if str(col) in {str(target), str(ctx.config.date_col)}:
            continue
        present = ctx.df[col][ymask].notna().to_numpy()
        if present.all() or not present.any():
            continue
        r1 = float(present[codes == 1].mean()) if (codes == 1).any() else 0.0
        r0 = float(present[codes == 0].mean()) if (codes == 0).any() else 0.0
        if (r1 > 0.5 and r0 < 0.02) or (r0 > 0.5 and r1 < 0.02):
            ctx.leaks.add(
                "D", str(col), "present-only-for-one-class", "execution", f"p1={r1:.2f} p0={r0:.2f}"
            )


def missingness_signals(ctx: RunContext, *, threshold: float = 0.70) -> None:
    """Register the missingness-predicts-target leak (§12.2, execution) for any
    column whose null-mask predicts a binary target above `threshold`. Requires
    frozen folds (the diagnostic is leakage-guarded); run after Part E."""
    from .values import missingness_predicts_target

    for col in ctx.df.columns:
        if str(col) == str(ctx.config.target) or not bool(ctx.df[col].isna().any()):
            continue
        auc = missingness_predicts_target(ctx, col)
        if auc == auc and auc >= threshold:
            ctx.leaks.add(
                "D", str(col), "missingness-predicts-target", "execution", f"AUC {auc:.3f}"
            )


# Structural/direct signals reliable enough to stand alone. The rest —
# effect-above-backstop, outlier-vs-peers, missingness-predicts-target — are *weak*
# on a well-separated problem (many honest features clear the bar, or missingness is
# genuinely informative), so a lone one of them needs corroboration (§12.3).
_STRONG_DETECTORS = frozenset(
    {"target-like-name", "future-timestamp", "perfect-separation", "present-only-for-one-class"}
)


def adjudicate(ctx: RunContext) -> dict[str, str]:
    """Resolve the register once (§12.3) → {column: "detector:type"}, **corroborated**:
    a column is adjudicated as a leak-suspect only when a strong structural signal
    fires (target-like name / future timestamp / perfect separation / present-only-for-
    positives) **or** ≥2 distinct detectors agree. A lone weak signal is left in the
    register (still visible in the ledger's `leak_flag`) but not adjudicated — on a
    predictable problem those fire on legitimate strong features and drown the real leak."""
    sigs = ctx.leaks.signals()
    if sigs.empty:
        return {}
    out: dict[str, str] = {}
    for col, grp in sigs.groupby("column"):
        detectors = list(dict.fromkeys(grp["detector"]))
        strong = [d for d in detectors if d in _STRONG_DETECTORS]
        if not strong and len(detectors) < 2:
            continue  # single weak signal → reported only, not adjudicated
        lead = grp[grp["detector"] == (strong[0] if strong else detectors[0])].iloc[0]
        extra = f" (+{len(detectors) - 1})" if len(detectors) > 1 else ""
        out[str(col)] = f"{lead['detector']}:{lead['ltype']}{extra}"
    return out
