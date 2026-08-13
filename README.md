# Feature Selection Playbook

A guided, **leakage-safe feature-selection screening** you run with Claude Code. It hands back a documented notebook — a ranked, reasoned shortlist with a **verdict + reason for every column**. It is a screening tool, not an auto-selector: the human makes the final call.

Three docs, three jobs:
- **[`PLAYBOOK.md`](PLAYBOOK.md)** — the *guide*: the method (Parts A–H), decision rules, thresholds, and the exact §17 math.
- **[`TOOLS.md`](TOOLS.md)** — the *catalogue* of the `fsp` package: the deterministic tools you call.
- **[`CLAUDE.md`](CLAUDE.md)** — the *entry*: operating instructions Claude Code reads when it runs the screening.

## Install

```sh
uv add "feature-selection-playbook @ git+ssh://git@github-feuji/ramacharanreddy-feuji/Feature-Selection-Playbook.git"
```

(`github-feuji` is an SSH host alias — use whatever alias your key is configured under.)

## Start a new analysis project

```sh
fsp init            # drops CLAUDE.md, PLAYBOOK.md, TOOLS.md + a gitignored runs/ into the folder
```

Then open the folder in Claude Code and, in your notebook:

```python
import fsp

ctx = fsp.open_run("data.csv", target="churn", target_type="binary")
# Claude follows PLAYBOOK.md Parts A→H, calling the fsp tools (see TOOLS.md):
# frame → viability → inventory → values → partition → relevance → redundancy → verdict
```

Claude reads `CLAUDE.md`, runs each part as **compute → decide → document → verify**, and grows the notebook a section per part.

## What you get (under `runs/<run-id>/`)

- **`results.ipynb`** (+ `results.html`) — the documented report, a section per part.
- **`ledger.parquet`** — one row per column: verdict + reason + numbers (dropped columns stay, with their numbers).
- **`folds.json`**, **`manifest.json`** — the frozen split and a reproducibility manifest (fixed seed).

## Develop `fsp`

```sh
uv sync
uv run pytest                              # the test suite
uv run ruff check src tests && uv run mypy src
```

The `fsp` package lives in `src/fsp/`, organized in four layers (foundation → metrics → parts → report); see `TOOLS.md` for the full tool catalogue.
