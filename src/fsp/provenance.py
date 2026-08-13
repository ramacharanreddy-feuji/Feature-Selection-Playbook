"""Run manifest — seed, config, and library versions for reproducibility."""

from __future__ import annotations

import contextlib
import importlib.metadata as md
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .context import RunContext

_LIBS = ["numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "lifelines", "optbinning"]


def manifest(ctx: RunContext) -> dict[str, Any]:
    versions: dict[str, str] = {}
    for lib in _LIBS:
        with contextlib.suppress(md.PackageNotFoundError):
            versions[lib] = md.version(lib)
    return {
        "run_id": ctx.run_id,
        "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": ctx.config.seed,
        "config": ctx.config.to_dict(),
        "libraries": versions,
    }


def save(ctx: RunContext) -> Path:
    path = ctx.run_dir / "manifest.json"
    path.write_text(json.dumps(manifest(ctx), indent=2, default=str))
    return path
