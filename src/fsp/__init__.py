"""fsp — deterministic tools for the Feature Selection Playbook.

These tools compute facts and the exact §17 math and handle mechanics (ledger,
notebook, gates). They never make a judgment call — Claude decides, guided by
`PLAYBOOK.md`; the human is the final decider. See `TOOLS.md` for the catalogue.
"""

from . import calibration, dispatch, io, metrics, parts, provenance, report, thresholds
from .context import RunConfig, RunContext, open_run, resume_run
from .folds import Folds
from .gates import GateFailure, gate
from .ledger import Ledger
from .notebook import Notebook
from .scaffold import scaffold

__version__ = "0.2.0"

__all__ = [
    # entry + state
    "open_run",
    "resume_run",
    "RunContext",
    "RunConfig",
    "scaffold",
    # process
    "parts",
    "report",
    # foundation
    "metrics",
    "dispatch",
    "thresholds",
    "io",
    "provenance",
    "calibration",
    "Ledger",
    "Notebook",
    "Folds",
    "gate",
    "GateFailure",
    "__version__",
]
