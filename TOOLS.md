# `fsp` — Tools Catalogue

**Status:** built (tested; ruff + mypy strict green).
**Companions:** [`PLAYBOOK.md`](PLAYBOOK.md) is the *guide* (what/why/rules/§17 math); [`CLAUDE.md`](CLAUDE.md) is the *entry* (how a run starts); this is the *catalogue* — the deterministic tools you call while running the guide.

`fsp` is a library of **deterministic tools**: they compute facts, the exact §17 math, and mechanics (ledger, notebook, gates). Look here for *what to call, what it returns, and which playbook step it serves*; look in `PLAYBOOK.md` for the method and the math itself.

---

## 1. The contract every tool obeys

- **Deterministic — never decides.** A tool returns numbers/facts (at most a *suggestion*, e.g. `suggest_semantic_type`). Every verdict is Claude's, confirmed by the human.
- **Leakage-safe by construction.** Target-association tools are fold-aware and **raise `GateFailure`** if `ctx.folds is None` — playbook §4.4 edge-3 is a hard error, not a guideline. Guarded: `relevance`, `relevance_all`, `values.missingness_predicts_target`, `leakage.missingness_signals`.
- **Reproducible.** One `seed` threaded everywhere; folds frozen to disk and reused verbatim.
- **`RunContext` is the spine.** Every part tool takes `ctx`; shared state (df, config, folds, ledger, leak register, notebook, seed) lives on it. Metric-kernel functions take plain arrays instead.
- **Suggestions vs decisions.** `suggest_semantic_type`, `suggest_strategy`, `target_facts["suggested_type"]` are hints with evidence; Claude confirms/overrides and records the decision.

---

## 2. Orientation — four layers

```
Layer 3  report/     → the human-facing deliverable (figures, tables, scorecard)
Layer 2  parts/      → the process, stage by stage (A → H)
Layer 1  metrics/    → the correctness core (§17 as tested functions)
Layer 0  foundation  → context, io, ledger, notebook, gates, dispatch, thresholds, folds, provenance, calibration
```

Dependency rule: higher layers import lower ones, never the reverse. `metrics/` imports nothing from `parts/`.

---

## 3. Catalogue

Signatures show the call shape; `ctx` is a `RunContext`. `⛨` marks a leakage-guarded tool (raises before folds exist).

### Layer 0 — foundation

**`context.py`** — the entry and the spine.

| Function | Call | Returns | Playbook |
|---|---|---|---|
| `open_run` | `open_run(source, *, runs_dir="runs", run_id=None, seed=42, **frame_hints)` | `RunContext` (reads data, makes run dir, starts notebook) | entry |
| `resume_run` | `resume_run(run_id, *, runs_dir="runs")` | `RunContext` restored from `checkpoint.pkl` — **continue a later part without recomputing earlier ones** | §3.1 |
| `RunConfig` | dataclass: `target, target_type, event_col, date_col, reference_date, id_cols, grain, prevalence, strictness_tier, seed` | frozen frame config (Part A) | §7 A |
| `RunContext` | attrs: `df, config, run_id, run_dir, ledger, leaks, folds, state, notebook` | — | — |
| `RunContext.gate` | `ctx.gate(part, conditions: dict[str,bool], *, notes=None)` | `bool` (writes card, raises `GateFailure`) | §4.5 |
| `RunContext.checkpoint` | `ctx.checkpoint()` | `Path` — persist df + config + `state` + ledger + leaks so a later part can `resume_run`. Call at the end of each part. | §3.1 |
| `RunContext.save_ledger` | `ctx.save_ledger(name="ledger.parquet")` | `Path` | §14 |
| `LeakRegister` | `ctx.leaks.add(part, col, detector, ltype, evidence)` · `.for_column(col)` · `.signals()` | accumulates leak signals | §12 |

**`io.py`** · `read(path) -> DataFrame` — CSV/parquet/Excel/SPSS·SAS·Stata, encoding-safe.

**`thresholds.py`** — §9 numbers as constants; read these, never hardcode.

| Name | Value / signature | Playbook |
|---|---|---|
| `AUTO_DROP` | `{auc:0.52, iv:0.01, spearman:0.05, cliffs:0.07, cindex:0.55, pbs:0.05, eta2:0.005, eps2:0.005}` | §9 |
| `LEAK_FLAG` | `{auc:0.85, iv:0.50, cindex:0.75}` | §9, §12 |
| `REDUNDANCY_COLLAPSE` / `REDUNDANCY_REVIEW` | `0.95` / `(0.70, 0.95)` | §4.1, §9 |
| `VIF_FLAG` · `PSI_DRIFT` · `COX_SCREEN_P` · `SHADOW_PCT` | `10` · `0.25` · `0.20` · `95.0` | §9, §17.5 |
| `NEAR_CONSTANT_SHARE` · `NEAR_EMPTY_MISSING` | `0.98` · `0.95` | §9.2 |
| `cramers_v_floor(min_levels)` | `float` — cardinality-dependent Cramér floor | §9.1 |
| `tier_for(effective_n)` | `"full" \| "reduced-power" \| "structural-only"` | §10 |
| `viability_floor_met(ttype, *, positives, effective_n)` | `bool` | §9 |
| `per_fold_floor_met(fold_positive_counts, *, min_per_fold=10)` | `bool` — **every** fold ≥ floor (real min, not average) | §9 |

**`dispatch.py`** · `metric_for(feature_type, target_type) -> MetricSpec` — the §8 table as code. `MetricSpec(name, kind, fn, p_fn)`; `kind ∈ {"assoc","survival","derive"}`.

**`ledger.py`** · `Ledger.upsert(column, **fields)` (validates `verdict` ∈ §5) · `.get(col)` · `.to_frame()` · `.save(path)`. `LEDGER_FIELDS` = the §14 schema; `VERDICTS` = the 7 of §5.

**`notebook.py`** · `Notebook.add_section(title, *, body="", facts=None, tables=None, figures=None, notes=None)` (persists live) · `.save()` · `.export_html(path=None)`. **Sections are addressable by `title`:** re-adding a title **rewrites that section in place** (existing sections are reloaded when the run opens), so re-running one part updates only its section — the notebook is never regenerated wholesale. `notes` takes a string **or** a list of strings (each becomes one blockquote). Use a **fixed `run_id`** in `open_run` so re-runs share the folder and this incremental update engages.

**`gates.py`** · `gate(ctx, part, conditions, *, notes=None) -> bool` · `GateFailure`. Empty conditions never pass.

**`folds.py`** · `Folds(splits, strategy, k, seed)` · `.save(path)` · `Folds.load(path)`.

**`provenance.py`** · `manifest(ctx) -> dict` · `save(ctx) -> Path` (seed, config, lib versions).

**`calibration.py`** · `drop_rate_by_rule(ledger) -> dict` · `log_run(ctx) -> dict` · `append(path, record) -> Path` (§9.3 cross-run signal).

### Layer 1 — metrics kernel (§17, each tested)

**`association.py`**

| Function | Call | Returns | § |
|---|---|---|---|
| `auc` | `auc(x, y)` | `float` (orientation-free ≥ 0.5) | 17.13 |
| `auc_ovr` | `auc_ovr(x, y)` | `float` (mean one-vs-rest) | §8 |
| `mann_whitney_p` | `mann_whitney_p(x, y)` | `float` (AUC's exact companion p) | 17.13 |
| `spearman` / `kendall` / `pearson` | `(x, y)` | `(stat, p)` | 17.14 |
| `point_biserial` | `(x, y)` | `(r, p)` | 17.11 |
| `cliffs_delta` | `cliffs_delta(x, group)` | `float` −1..1 | 17.4 |
| `correlation_ratio` | `correlation_ratio(cats, vals)` | `float` η² 0..1 | 17.9 |
| `kruskal_eps2` | `kruskal_eps2(vals, groups)` | `(ε², p)` | 17.10 |

**`categorical.py`**

| Function | Call | Returns | § |
|---|---|---|---|
| `bergsma_v` | `bergsma_v(a, b)` | `float` Ṽ (bias-corrected) | 17.1 |
| `cramers_v` | `cramers_v(a, b)` | `float` (reference only) | 17.1 |
| `information_value` | `information_value(x, y, dtype="numerical")` | `float` IV (in-sample) | 17.3 |
| `woe_table` | `woe_table(x, y, dtype="numerical")` | `DataFrame` | 17.3 |
| `iv_oof` | `iv_oof(x, y, folds, *, dtype="categorical")` | `float` (mean out-of-fold IV) | 17.3 |
| `target_encoded_eta_oof` | `target_encoded_eta_oof(x, y, folds)` | `float` (out-of-fold η²) | 17.3 |

**`shape.py`** · `x_statistic(x, y) -> float` · `shape_gap(x, y) -> float` (17.2).

**`survival.py`** · `cox_screen(dur, event, x) -> {hr, p, cindex}` · `concordance(dur, risk, event) -> float` · `logrank(dur, event, groups) -> (stat, p)` (17.8, 17.16).

**`drift.py`** · `psi(expected, actual, bins=10) -> float` (17.7) · `vif(X: DataFrame) -> Series` (17.15).

**`resampling.py`**

| Function | Call | Returns | § |
|---|---|---|---|
| `bh_fdr` | `bh_fdr(pvals)` | `ndarray` q-values | 17.6 |
| `bootstrap_ci` | `bootstrap_ci(fn, x, y, *, b=1000, seed=42, ci=0.95)` | `(lo, hi)` | 17.12 |
| `shadow_samples` | `shadow_samples(fn, x, y, *, b=50, seed=42)` | `ndarray` (shadow distribution) | 17.5 |
| `shadow_floor` | `shadow_floor(fn, x, y, *, pct=95, b=50, seed=42)` | `float` | 17.5 |
| `permutation_p` | `permutation_p(effect, shadow)` | `float` (one-sided p off the shadow draws) | 17.6 |

### Layer 2 — part tools (A → H)

**`frame.py` (A)** · `target_candidates(df)` · `suggest_target_type(s)` · `target_facts(df, col)` · `date_candidates(df)` · `id_candidates(df)` · `grain_facts(df, id_cols)`.

**`viability.py` (B)** · `assess(df, target, target_type, *, event_col=None) -> dict` (positives, effective_n, prevalence, censoring, target_nulls) · `tier(effective_n) -> str`.

**`inventory.py` (C)** · `profile(df) -> DataFrame` · `column_facts(df, col) -> dict` · `duplicate_columns(df) -> dict` · `structural_flags(df, col, *, duplicate_of=None, protected=()) -> str|None` · `suggest_semantic_type(facts) -> str`.

**`values.py` (D)** · `sentinel_candidates(df) -> dict` · `null_sentinels(df, register) -> DataFrame` · `distribution(df, col) -> dict` · `missingness(df) -> Series` · `comissing_clusters(df, threshold=0.95) -> list` · `missingness_predicts_target(ctx, col) -> float` ⛨.

**`partition.py` (E)** · `suggest_strategy(*, has_repeating_id, has_date) -> str` · `recommended_k(tier) -> int` · `make_folds(df, strategy, k=5, *, seed=42, target=None, group=None, date=None, run_dir=None) -> Folds` · `load_folds(run_dir) -> Folds` · `fold_positive_counts(folds, y, *, positive_label=None) -> list[int]` — positives in each fold's held-out block, for `per_fold_floor_met` (§9).

**`relevance.py` (F)** · `derive_datetime(s, *, reference=None) -> DataFrame` (calendar + cyclical + recency) · `hurdle_split(count) -> (is_zero, positives)` · `relevance(ctx, feature, ftype, *, ci_b=200, shadow_b=50) -> dict` ⛨ · `relevance_all(ctx, features, *, n_jobs=1, ci_b=200, shadow_b=50) -> DataFrame` ⛨ (adds `q_value`; lower `ci_b`/`shadow_b`, or `ci_b=0`, for wide-data speed). See §4 for the `relevance` return shape.

**`redundancy.py` (G)** · `pairwise(df, features, types) -> DataFrame` ([0,1], typed) · `components(pairs, threshold=0.95) -> list[set]` · `representative(component, effects, missing) -> str` · `vif` (re-export).

**`leakage.py` (H)** — detectors add to `ctx.leaks`; adjudicated once at H.

| Function | Call | Fires at | Type |
|---|---|---|---|
| `name_signals` | `name_signals(ctx)` | C | direct |
| `future_timestamp` | `future_timestamp(ctx, *, reference=None)` | D | future |
| `only_present_for_positives` | `only_present_for_positives(ctx)` | D | execution |
| `missingness_signals` | `missingness_signals(ctx, *, threshold=0.70)` ⛨ | D (post-E) | execution |
| `outlier_effects` | `outlier_effects(ctx, effects)` | F | direct |
| `backstop_effects` | `backstop_effects(ctx, effects, *, threshold)` | F | direct |
| `separation_signals` | `separation_signals(ctx, features, *, min_count=5)` | F | direct |
| `adjudicate` | `adjudicate(ctx) -> dict[col, "detector:type"]` | H | — |

`adjudicate` is **corroborated** (§12.3): it returns a column as a leak-suspect only when a strong structural signal fires (name / future-timestamp / perfect-separation / present-only-for-positives) **or** ≥ 2 distinct detectors agree — a lone weak effect signal stays in the register but is not adjudicated. Don't build a "flag every effect ≥ threshold" rule around `backstop_effects`; that floods on predictable data (§12.3). Part-F resampling dominates cost on wide data — pass `relevance_all(..., ci_b=…, shadow_b=…)` (or `ci_b=0`) to trade CI precision for speed.

**`boruta.py` (H)** · `crosscheck(df, features, target, target_type, *, seed=42, max_iter=40) -> dict[col, "confirmed"|"tentative"|"rejected"]` (numeric + factorized categoricals).

### Layer 3 — report & assembly

**`figures.py`** · `missingness_bar(df, top=15)` · `distribution_hist(series)` · `effect_ranking(ledger, feature_type=None, top=20)` · `redundancy_heatmap(pairs)` → matplotlib `Figure`.

**`tables.py`** · `viability_table(assess)` · `inventory_table(profile)` · `top_keeps(ledger, n=10)` · `ledger_view(ledger)` → `DataFrame`; `rank_within_type(ledger) -> dict[col, int]` (§14).

**`scorecard.py`** · `scorecard(ledger) -> dict` · `review_export(ledger, path) -> Path` · `limits_note() -> str` (§13).

### Setup (run once, outside the A→H loop)

**`scaffold.py` / `cli.py`** · `scaffold(dest=".", *, overwrite=False) -> list[str]` and the **`fsp init`** CLI — drop the guide docs + the phase-code starter (**`analysis/screening.py`** runner + **`analysis/parts.py`**, one `run_<x>(ctx)` per part) into a new project, and **gitignore `runs/` + the (regenerable) docs**. Fill `parts.py` **one part at a time** (§3.1); `python analysis/screening.py c` runs only Part C (resumes the checkpoint — no recompute), omit the letter for the whole chain. Code lives in `analysis/`, run outputs in `runs/<run-id>/`.

---

## 4. Cross-cutting contracts

**The `relevance()` return dict → §14 ledger fields.** One row of Part-F facts per feature:

| Key | Meaning | Ledger field |
|---|---|---|
| `column`, `metric_name` | feature + native metric | `column`, `metric_name` |
| `effect` | effect on the metric's scale | `effect` |
| `effect_ci` | bootstrap percentile CI (§17.12) | `effect_ci` |
| `p` → (via `relevance_all`) `q_value` | analytic-or-permutation p, then BH-FDR | `q_value` |
| `shadow_floor` | the permutation bar it cleared (§17.5) | `shadow_floor` |
| `shape_gap` | non-monotone signal, binary target only (§17.2) | `shape_gap` |
| `fold_spread` | cross-fold stability | `fold_spread` |
| `power_flag` | per-feature §10 tier (blank at full) | `power_flag` |

Claude fills the remaining ledger fields by decision: `semantic_type` (C), `verdict` + `reason` (F–H), `redundant_with` (G), `leak_flag` (H), `rank_within_type` (via `tables.rank_within_type`).

**The leakage guard.** `relevance`, `relevance_all`, `values.missingness_predicts_target`, and `leakage.missingness_signals` raise `GateFailure` when `ctx.folds is None`. Freeze folds at Part E (`partition.make_folds`) before any of them.

---

## 5. Call-order reference (A → H) — run incrementally, **not** as a script

This is the sequence of tools each part calls — a **reference, not a script to paste and run blind.** Per playbook §1 / §3.1 you run this **one part at a time**: call a part's tools, **read the output**, decide *from what you see*, document the section (facts + tables/figures), pass the gate, *then* move to the next part. The block below is compressed to show the order; a faithful run is incremental and inspects each part before writing the next — a semantic type can't be chosen before the inventory is seen, nor a verdict before the effect. (Corroboration at H is handled by `leakage.adjudicate`, §12.3 — don't wrap `backstop_effects` in a "flag every high effect" rule.) In a scaffolded project this same sequence lives as `run_a…run_h` in `analysis/parts.py`, each ending `ctx.checkpoint()`; run one part with `python analysis/screening.py c` (it `resume_run`s the prior state — no recompute).

```python
import fsp
from fsp import thresholds as T
from fsp.parts import (frame, viability, inventory, values,
                       partition, relevance, redundancy, leakage, boruta)

ctx = fsp.open_run(path, target="churn", target_type="binary", seed=42, run_id="churn")

# A · Frame — candidates/facts → Claude picks target, grain, date, ids, reference_date
frame.target_facts(ctx.df, ctx.config.target); frame.id_candidates(ctx.df)
ctx.gate("A", {"target_set": ctx.config.target is not None})

# B · Viability — set the strictness tier
v = viability.assess(ctx.df, ctx.config.target, ctx.config.target_type)
ctx.config.strictness_tier = viability.tier(v["effective_n"])
ctx.gate("B", {"has_target": v["effective_n"] > 0})

# C · Inventory — semantic types + structural drops; name-leak signals
dups = inventory.duplicate_columns(ctx.df)
protected = (ctx.config.target, ctx.config.date_col)
feature_types: dict[str, str] = {}
for col in ctx.df.columns:
    facts = inventory.column_facts(ctx.df, col)
    stype = inventory.suggest_semantic_type(facts)          # Claude confirms
    flag = inventory.structural_flags(ctx.df, col, duplicate_of=dups.get(col),
                                      protected=protected)
    if flag:
        ctx.ledger.upsert(col, semantic_type=stype, verdict="structural-drop", reason=flag)
    else:
        ctx.ledger.upsert(col, semantic_type=stype)
        feature_types[col] = stype                          # non-structural → testable in F
leakage.name_signals(ctx)
# Document the part, THEN gate — EVERY part A–H does this (facts + the tables/figures
# behind the decision, §14); a bare gate with no documented section does not pass (§4.5):
ctx.notebook.add_section("C · Inventory", body="…what the profile showed & why…",
                         tables=[("Profile", fsp.report.tables.inventory_table(inventory.profile(ctx.df)))])
ctx.gate("C", {"every_column_typed": True})

# D · Value integrity — sentinels, missingness, co-missing; D leak detectors
values.sentinel_candidates(ctx.df); values.comissing_clusters(ctx.df)
leakage.future_timestamp(ctx); leakage.only_present_for_positives(ctx)

# E · Partition — freeze the folds (guard opens here)
k = partition.recommended_k(ctx.config.strictness_tier)
ctx.folds = partition.make_folds(ctx.df, partition.suggest_strategy(has_repeating_id=False,
                                 has_date=ctx.config.date_col is not None),
                                 k=k, target=ctx.config.target, run_dir=ctx.run_dir)
leakage.missingness_signals(ctx)                            # guarded: post-E
ctx.gate("E", {"folds_frozen": ctx.folds is not None})

# F · Relevance — effect/CI/q/shape/shadow per feature (split frozen)
rel = relevance.relevance_all(ctx, feature_types)
effects = dict(zip(rel["column"], rel["effect"]))
leakage.outlier_effects(ctx, effects)                       # primary leak signal (§12)
# + backstop_effects PER METRIC (AUC effects vs LEAK_FLAG["auc"], IV vs ["iv"], …) —
#   never one threshold across mixed metrics; adjudicate corroborates at H (§12.3)
# → Claude decides keep / drop / review per §9 and upserts each row

# G · Redundancy — collapse near-duplicates among the keeps
keeps = [c for c in feature_types if ctx.ledger.get(c).get("verdict") == "keep"]
missing = {c: inventory.column_facts(ctx.df, c)["missing_rate"] for c in keeps}
pairs = redundancy.pairwise(ctx.df, keeps, feature_types)
for comp in redundancy.components(pairs, T.REDUNDANCY_COLLAPSE):
    rep = redundancy.representative(comp, effects, missing)
    # → others become 'redundant', naming rep

# H · Verdict — Boruta cross-check, adjudicate leaks, final verdicts + reasons
boruta.crosscheck(ctx.df, list(feature_types), ctx.config.target, ctx.config.target_type)
leakage.adjudicate(ctx)
for col, rank in fsp.report.tables.rank_within_type(ctx.ledger.to_frame()).items():
    ctx.ledger.upsert(col, rank_within_type=rank)
ctx.gate("H", {"every_column_has_verdict": True})           # + non-empty reason

# Deliverables
ctx.notebook.export_html(); ctx.save_ledger()
fsp.provenance.save(ctx); fsp.calibration.log_run(ctx)
```
