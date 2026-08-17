# Feature Selection Playbook — operating instructions

## What this repo is
Tooling to run **feature-selection screening** on a single dataframe. Three pieces, and you must know all three:
- **[`PLAYBOOK.md`](PLAYBOOK.md)** — the **guide**: the method, order, decision rules, thresholds, and exact math. **Read it in full before you start.** The source of truth for *what to do* and *how much*.
- **[`TOOLS.md`](TOOLS.md)** — the catalogue of the **`fsp` package**: the deterministic tools you call (the §17 math, structural checks, folds, ledger, notebook, gates).
- **`fsp`** — those tools, importable (`import fsp`, once the package is installed in your environment).

## Your job
Given a path to a dataset (CSV/parquet/Excel/SPSS·SAS·Stata) and — if the user provides it — the target column, run the whole screening and hand back a documented notebook:

- Follow **`PLAYBOOK.md` Parts A–H in order**, each as the loop **compute → decide → document → verify** (playbook §3.1).
- **Call `fsp` for the deterministic work** — the §17 math, structural checks, folds, notebook, gates. **Never re-derive a metric `fsp` provides.** Write code only for framing, judgment, adaptation to messy data, and the notebook narrative (playbook §1).
- Build a **documented notebook** (`results.ipynb`) that grows **one section per part** — each section is **markdown prose (what you found + why) + the part's tables + its figures** (`fsp.report.tables` / `fsp.report.figures`) so results are *shown*, not just logged; it ends with the per-column **ledger**. A terse, figure-less section is not done (§14). That notebook is the deliverable.
- **Keep your phase code in `analysis/`** (created by `fsp init`): a `screening.py` **runner** and `parts.py` with one `run_<x>(ctx)` per part. Fill `parts.py` **one part at a time** — run a part, read its output, decide, document, gate, *then* the next; never batch-write the whole run (§1, §3.1). Run a **single part** with `python analysis/screening.py c` — it `resume_run`s the prior state from the checkpoint and runs **only** that part (no recompute); omit the letter to run the whole chain. Each part ends with `ctx.checkpoint()`. Run outputs (notebook, ledger, folds, manifest) land under `runs/<run-id>/`.

## Non-negotiables (from `PLAYBOOK.md`)
- **Use the exact thresholds (§9) and formulas (§17) via `fsp`** (`fsp.thresholds`, `fsp.metrics`). If the guide doesn't specify how to measure something, **STOP and ask** — do not invent a method. If `fsp` lacks a deterministic tool you need, **stop and say so** — don't hand-roll it.
- **No per-feature target statistic before the fold split exists (Part E)** — the most-violated rule (§4.4 edge 3). `fsp`'s relevance tools enforce this; do not bypass them.
- **Never mark `leak-suspect` from a single effect threshold** — corroborate (§12.3): a strong structural signal, or ≥2 detectors. Use `leakage.adjudicate`; don't hand-roll "flag every high effect" (it floods on predictable data).
- **Run the parts live and incrementally — never batch-write the whole A→H run.** Execute each part's tools, read the output, decide from what you see, document the section (facts + tables/figures), pass the gate, *then* proceed (§1, §3.1). Don't skip a step or thin its documentation because you think you know the data.
- **Every dropped column stays in the ledger** with its numbers. `drop` ≠ delete.
- **Part A (Frame) is the mandatory human checkpoint.** State inferences as corrigible claims; confirm target and grain with the user before proceeding.
- **A failing gate is a hard stop** — report which exit condition failed and why (§15); don't guess past it.
- **This is a screening tool, not an auto-selector** — you produce a ranked, reasoned shortlist; the human decides (§16).

## Output
- `results.ipynb` (+ `results.html`) — the documented report, built live (a section per part).
- `ledger.parquet` (or `.csv`) — one row per column: verdict + reason + numbers.
- **Set and state a fixed random seed** so the run is reproducible.
- Write run outputs under `runs/<run-id>/` (gitignored).

## Environment
Install the `fsp` package into your environment **on Python 3.12** (`optbinning`→`ortools` has no 3.13 wheel):

```sh
uv init my-analysis --python 3.12 && cd my-analysis
uv add "feature-selection-playbook @ git+ssh://git@github-feuji/ramacharanreddy-feuji/Feature-Selection-Playbook.git"
uv run fsp init
```

Run your analysis in the project's virtual environment (`uv run python`, a Jupyter kernel, …) and `import fsp`. Outputs land under `runs/<run-id>/` (gitignored).
