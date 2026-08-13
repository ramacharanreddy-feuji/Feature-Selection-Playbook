"""Part H — Boruta all-relevant cross-check (playbook §16). A safety net."""

from __future__ import annotations

import pandas as pd
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def crosscheck(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    target_type: str,
    *,
    seed: int = 42,
    max_iter: int = 40,
) -> dict[str, str]:
    """Fit Boruta on all features; return confirmed/tentative/rejected.

    Answers one question (§16): did the filter throw away anything you'd use?
    Numeric features pass through; categoricals are ordinal-encoded (factorized)
    so surviving **and** dropped categoricals are cross-checked too, not skipped.
    """
    # Build all columns at once (avoid per-insert DataFrame fragmentation on wide data).
    encoded = {
        feat: (df[feat].astype(float) if pd.api.types.is_numeric_dtype(df[feat])
               else pd.Series(pd.factorize(df[feat])[0], index=df.index).astype(float))
        for feat in features
    }
    if not encoded:
        return {}
    x = pd.DataFrame(encoded, index=df.index)
    x = x.fillna(x.median(numeric_only=True))
    y = df[target]
    mask = y.notna().to_numpy()
    xv = x.to_numpy()[mask]

    if target_type == "regression":
        est = RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1)
        yv = y[mask].to_numpy(dtype=float)
    else:
        est = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
        yv = pd.factorize(y[mask])[0]

    boruta = BorutaPy(est, n_estimators="auto", random_state=seed, max_iter=max_iter)
    boruta.fit(xv, yv)

    out: dict[str, str] = {}
    triples = zip(x.columns, boruta.support_, boruta.support_weak_, strict=False)
    for col, confirmed, weak in triples:
        out[str(col)] = "confirmed" if confirmed else ("tentative" if weak else "rejected")
    return out
