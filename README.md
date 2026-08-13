# Feature Selection Playbook

A guided, **leakage-safe feature-selection screening** you run with Claude Code. It hands back a documented notebook — a ranked, reasoned shortlist with a **verdict + reason for every column**. It is a screening tool, not an auto-selector: the human makes the final call.

Three docs, three jobs:
- **[`PLAYBOOK.md`](PLAYBOOK.md)** — the *guide*: the method (Parts A–H), decision rules, thresholds, and the exact §17 math.
- **[`TOOLS.md`](TOOLS.md)** — the *catalogue* of the `fsp` package: the deterministic tools you call.
- **[`CLAUDE.md`](CLAUDE.md)** — the *entry*: operating instructions Claude Code reads when it runs the screening.

## Install & start in a new folder

> ⚠️ **Use Python 3.12.** `optbinning` (→ `ortools`) has no Python 3.13 wheel yet, so a 3.13 environment will fail to install. Pin 3.12 as shown.

```sh
# 1. create the project ON Python 3.12 (do it in one step — see the note above)
uv init my-analysis --python 3.12 && cd my-analysis

# 2. install fsp (from the feuji repo; github-feuji is your SSH host alias — use yours)
uv add "feature-selection-playbook @ git+ssh://git@github-feuji/ramacharanreddy-feuji/Feature-Selection-Playbook.git"

# 3. scaffold the guide docs + a starter driver into the folder
uv run fsp init          # writes CLAUDE.md, PLAYBOOK.md, TOOLS.md, analysis/screening.py + a gitignored runs/
```

`fsp init` also drops **`analysis/screening.py`** — one file for the whole run, with A→H section markers. Fill it in **one part at a time** (run a part, read its output, decide, document, gate, then the next — PLAYBOOK.md §3.1); split a phase into its own file under `analysis/` if it grows.

The three guide docs are **gitignored** in your project (they come from the package — regenerate any time with `fsp init`), so your repo tracks only `analysis/` and your data.

Then drop your data in the folder and either **open it in Claude Code** (it reads `CLAUDE.md` and drives the whole screening) or use `fsp` directly in a notebook:

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
