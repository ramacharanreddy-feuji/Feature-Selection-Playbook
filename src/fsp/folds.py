"""Frozen cross-validation folds (playbook Part E) — the split everyone reuses."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Folds:
    """A frozen split: a list of (train_idx, test_idx) pairs, plus provenance."""

    splits: list[tuple[np.ndarray, np.ndarray]]
    strategy: str
    k: int
    seed: int

    def __len__(self) -> int:
        return len(self.splits)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "strategy": self.strategy,
            "k": self.k,
            "seed": self.seed,
            "splits": [[tr.tolist(), te.tolist()] for tr, te in self.splits],
        }
        path.write_text(json.dumps(payload))
        return path

    @classmethod
    def load(cls, path: str | Path) -> Folds:
        payload = json.loads(Path(path).read_text())
        splits = [
            (np.asarray(tr, dtype=int), np.asarray(te, dtype=int))
            for tr, te in payload["splits"]
        ]
        return cls(
            splits=splits, strategy=payload["strategy"], k=payload["k"], seed=payload["seed"]
        )
