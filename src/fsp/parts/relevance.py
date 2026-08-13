"""Part F — Relevance + stability (playbook §7 F). Leakage-guarded.

The flagship safety feature: these functions RAISE if the split does not exist
yet (§4.4 edge 3), so a per-feature target statistic can never run pre-split.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .. import metrics
from ..dispatch import metric_for
from ..gates import GateFailure
from ..thresholds import SHADOW_PCT, tier_for

if TYPE_CHECKING:
    from ..context import RunContext


def derive_datetime(s: pd.Series, *, reference: object = None) -> pd.DataFrame:
    """Derive the full §8 datetime feature set: calendar parts, weekend flag,
    days-since-epoch, cyclical sin/cos (month, day-of-week, hour), and recency to
    a reference (default = the column's own max). Each derivative is routed
    through the dispatch table by its own type."""
    dt = pd.to_datetime(s, errors="coerce")
    if getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_localize(None)
    name = str(s.name) if s.name is not None else "dt"
    ref = pd.Timestamp(reference) if reference is not None else dt.max()
    epoch = pd.Timestamp("1970-01-01")
    tau = 2.0 * np.pi
    month, dow, hour = dt.dt.month, dt.dt.dayofweek, dt.dt.hour
    return pd.DataFrame(
        {
            f"{name}__year": dt.dt.year,
            f"{name}__month": month,
            f"{name}__dow": dow,
            f"{name}__day": dt.dt.day,
            f"{name}__hour": hour,
            f"{name}__is_weekend": (dow >= 5).astype(float),
            f"{name}__days_since_epoch": (dt - epoch).dt.total_seconds() / 86400.0,
            f"{name}__month_sin": np.sin(tau * month / 12.0),
            f"{name}__month_cos": np.cos(tau * month / 12.0),
            f"{name}__dow_sin": np.sin(tau * dow / 7.0),
            f"{name}__dow_cos": np.cos(tau * dow / 7.0),
            f"{name}__hour_sin": np.sin(tau * hour / 24.0),
            f"{name}__hour_cos": np.cos(tau * hour / 24.0),
            f"{name}__recency_days": (ref - dt).dt.total_seconds() / 86400.0,
        }
    )


def _power_flag(pairwise_n: int) -> str | None:
    """Per-feature §10 power flag from its available-case n (blank at full tier)."""
    tier = tier_for(pairwise_n)
    return None if tier == "full" else tier


def hurdle_split(count: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Split a zero-inflated count into (is_zero, positives) (§8)."""
    s = pd.Series(count)
    return (s == 0).astype(float), s.where(s > 0)


def _screening_rows(ctx: RunContext) -> np.ndarray:
    """Row indices the screening statistic is computed on — the union of the
    training folds, which in standard k-fold spans every row. The frozen split's
    role in Part F is to gate ordering (§4.4 edge 3) and supply the per-fold
    stability signal, not to hold rows out of the point estimate. Genuinely
    out-of-fold scoring is used only where in-sample inflates (high-card, §17.3)."""
    assert ctx.folds is not None
    return np.unique(np.concatenate([tr for tr, _ in ctx.folds.splits]))


def relevance(
    ctx: RunContext, feature: str, feature_type: str, *, ci_b: int = 200, shadow_b: int = 50
) -> dict[str, Any]:
    """Effect + CI + q + shape gap + fold spread + shadow floor for one feature.
    The point estimate is on all rows once the split is frozen; the per-fold test
    effects give the stability spread. Raises if folds are not frozen (§4.4).

    `ci_b` / `shadow_b` are the bootstrap-CI and shadow-permutation counts — lower
    them (or `ci_b=0` to skip the CI) to speed up screening on very wide data."""
    if ctx.folds is None:
        raise GateFailure("relevance", ["folds not frozen (Part E) — cannot touch the target"])
    target = ctx.config.target
    ttype = ctx.config.target_type or "binary"
    if feature_type == "high_card" and ttype in {"binary", "regression"}:
        return _high_card_oof(ctx, feature, ttype)
    spec = metric_for(feature_type, ttype)
    if spec.kind == "derive":
        return {"column": feature, "metric_name": "derive-first", "kind": "derive"}
    if spec.kind == "survival":
        return _survival(ctx, feature)

    assert spec.fn is not None and target is not None
    fn = spec.fn
    rows = _screening_rows(ctx)
    x_all = ctx.df[feature].iloc[rows]
    y_all = ctx.df[target].iloc[rows]

    effect = float(fn(x_all, y_all))
    per_fold = []
    for _, te in ctx.folds.splits:
        try:
            per_fold.append(float(fn(ctx.df[feature].iloc[te], ctx.df[target].iloc[te])))
        except Exception:
            per_fold.append(float("nan"))
    fold_spread = float(np.nanstd(per_fold)) if per_fold else float("nan")

    xa, ya = x_all.to_numpy(), y_all.to_numpy()
    ci = (metrics.bootstrap_ci(fn, xa, ya, b=ci_b, seed=ctx.config.seed)
          if ci_b > 0 else (float("nan"), float("nan")))
    samples = metrics.shadow_samples(fn, xa, ya, b=shadow_b, seed=ctx.config.seed)
    floor = float(np.nanpercentile(samples, SHADOW_PCT))
    # Analytic p where the metric has one; else a permutation p off the same
    # shadow samples, so AUC/IV/η²/Cramér's V all get a q-value (§17.6).
    p = float(spec.p_fn(x_all, y_all)) if spec.p_fn else metrics.permutation_p(effect, samples)

    shape = float("nan")
    if ttype == "binary" and feature_type in {"continuous", "count"}:
        try:
            shape = float(metrics.shape_gap(x_all, y_all))
        except Exception:
            shape = float("nan")

    pairwise_n = int((ctx.df[feature].notna() & ctx.df[target].notna()).sum())
    return {
        "column": feature,
        "metric_name": spec.name,
        "effect": effect,
        "effect_ci": (round(ci[0], 4), round(ci[1], 4)),
        "p": p,
        "shape_gap": shape,
        "fold_spread": round(fold_spread, 4),
        "shadow_floor": round(floor, 4),
        "power_flag": _power_flag(pairwise_n),
    }


def _high_card_oof(ctx: RunContext, feature: str, ttype: str) -> dict[str, Any]:
    """High-cardinality relevance via out-of-fold encoding (§8, §17.3) — the
    in-sample IV/η² would be inflated, so score it on held-out folds only. The
    shadow floor permutes the column and re-runs the same out-of-fold metric."""
    target = ctx.config.target
    assert target is not None and ctx.folds is not None
    splits = ctx.folds.splits
    x = ctx.df[feature].to_numpy()

    if ttype == "binary":
        codes = pd.factorize(ctx.df[target])[0].astype(float)
        codes[codes < 0] = np.nan
        y = codes

        def score(xx: np.ndarray) -> float:
            return metrics.iv_oof(xx, y, splits)

        name = "IV (oof)"
    else:  # regression
        y = pd.to_numeric(ctx.df[target], errors="coerce").to_numpy(dtype=float)

        def score(xx: np.ndarray) -> float:
            return metrics.target_encoded_eta_oof(xx, y, splits)

        name = "η² (oof)"

    effect = float(score(x))
    rng = np.random.default_rng(ctx.config.seed)
    shadow = np.array([score(rng.permutation(x)) for _ in range(50)])
    floor = float(np.nanpercentile(shadow, SHADOW_PCT)) if len(shadow) else float("nan")
    pairwise_n = int((ctx.df[feature].notna() & ctx.df[target].notna()).sum())
    return {
        "column": feature,
        "metric_name": name,
        "effect": effect,
        "effect_ci": (float("nan"), float("nan")),
        "p": metrics.permutation_p(effect, shadow),
        "shape_gap": float("nan"),
        "fold_spread": float("nan"),
        "shadow_floor": round(floor, 4),
        "power_flag": _power_flag(pairwise_n),
    }


def _survival(ctx: RunContext, feature: str) -> dict[str, Any]:
    target, event_col = ctx.config.target, ctx.config.event_col
    assert target is not None and event_col is not None
    rows = _screening_rows(ctx)
    dur = ctx.df[target].iloc[rows].to_numpy(dtype=float)
    ev = ctx.df[event_col].iloc[rows].to_numpy(dtype=float)
    x = ctx.df[feature].iloc[rows].to_numpy(dtype=float)
    try:
        out = metrics.cox_screen(dur, ev, x)
        effect, p = max(out["cindex"], 1 - out["cindex"]), out["p"]
    except Exception:
        return {
            "column": feature,
            "metric_name": "C-index",
            "effect": float("nan"),
            "p": float("nan"),
        }
    rng = np.random.default_rng(ctx.config.seed)
    shadow = [metrics.concordance(dur, rng.permutation(x), ev) for _ in range(50)]
    floor = float(np.nanpercentile([max(s, 1 - s) for s in shadow], SHADOW_PCT))
    pairwise_n = int((ctx.df[feature].notna() & ctx.df[event_col].notna()).sum())
    return {
        "column": feature,
        "metric_name": "C-index",
        "effect": effect,
        "p": p,
        "shadow_floor": round(floor, 4),
        "fold_spread": float("nan"),
        "shape_gap": float("nan"),
        "power_flag": _power_flag(pairwise_n),
    }


def relevance_all(
    ctx: RunContext, features: dict[str, str], *,
    n_jobs: int = 1, ci_b: int = 200, shadow_b: int = 50,
) -> pd.DataFrame:
    """Run `relevance` over {feature: feature_type}, then add BH-FDR q-values.

    `n_jobs` fans the per-feature work out over joblib (§11 decision); the default
    (1) stays sequential and deterministic. `ci_b`/`shadow_b` tune the per-feature
    resampling cost for wide data (see `relevance`)."""
    items = list(features.items())
    if n_jobs == 1:
        rows = [relevance(ctx, feat, ftype, ci_b=ci_b, shadow_b=shadow_b) for feat, ftype in items]
    else:
        from joblib import Parallel, delayed

        rows = Parallel(n_jobs=n_jobs)(
            delayed(relevance)(ctx, f, t, ci_b=ci_b, shadow_b=shadow_b) for f, t in items
        )
    df = pd.DataFrame(rows)
    if "p" in df.columns:
        df["q_value"] = metrics.bh_fdr(df["p"].to_numpy())
    return df
