# Feature Selection Playbook — Operating Spec

**Status:** v1 — locked 2026-08-11

This document is the *operating spec* — what to do, in what order, with which command. It governs execution.

> This is the procedure an agent (e.g. Claude Code) follows when handed a path to a single dataframe. It reads the file, understands every row and column, and hands a data scientist a **ranked, reasoned shortlist** — a verdict with a reason per feature, not a final feature set. A human still makes the real decision (see §16).

---

## 0. How to read this document

- **Parts A–H** run in order. Most of the order is soft; four edges are hard (§4.4).
- Every statistic is produced by a **`dsp` command** (§6). The agent never writes its own EDA code (§1).
- Each part ends at a **gate** (§4.5). A gate is an exit code, not a suggestion.
- The output is the **ledger** (§14): one row per column, with a verdict and a reason.
- Numbers you must not invent are in the **Thresholds** table (§9) and **Metric definitions** (§17), each with a source.

---

## 1. Prime directive

> **Never write your own EDA code. Every statistic comes from a `dsp` command. If you need a statistic the library does not compute, STOP and tell the user — do not improvise it in pandas.**

Why this is absolute: if the agent improvises statistics, output shape drifts every run, nothing is comparable across projects, and the thresholds below become meaningless because they were calibrated against `dsp`'s exact definitions. The markdown says *when* and *why*; the package guarantees *how*.

Corollary rules:
- **No raw thresholds hard-coded in agent reasoning.** Read them from `dsp` config so a single change propagates.
- **No per-feature target-association statistic before the folds exist (Part E).** Only Part B's aggregate viability facts and Part D's missingness-leak diagnostic may touch the target earlier. See §4.4 edge 3 — the most-violated, least-noticed rule.
- **Every drop stays in the ledger.** `drop` means "excluded from the first model," never "deleted."

---

## 2. Three roles

| Role | Who | Responsibility |
|---|---|---|
| **Orchestrator** | The agent | Reads this playbook, calls `dsp` commands in order, makes the judgment calls the playbook reserves for judgment, asks the human the ≤2 questions at Part A, assembles the ledger. |
| **Engine** | The `dsp` package | Computes every statistic, writes every artifact, enforces every gate. Deterministic and comparable across runs. |
| **Decider** | The human | Reviews the ledger, applies domain knowledge the data cannot contain, makes the final feature decision. Confirms Part A (the one mandatory-human step). |

---

## 3. The run at a glance

```
A Frame ──▶ B Viability ──▶ C Inventory ──▶ D Value integrity ──▶ E Partition ──▶ F Relevance+stability ──▶ G Redundancy ──▶ H Verdict
  (no target)  (target)       (no target)     (partly target)       (no target)     (target, train folds)      (no target)     (assemble)
   config       strictness     semantic         sentinels,            frozen           effect+CI+q+shape+         components+     ledger with
   object       tier           types +          missingness           folds            fold-spread               reps           reason/feature
                               structural        clusters
                               drops
                                    │                │                                        │
                              leakage: name     leakage: future ts,                    leakage: suspicious
                              heuristics fire   missingness-leak fire                  effect sizes fire
                                    └────────────────┴──────────────── accumulate ────────────┴──▶ adjudicated once at H
```

**First five parts describe the data; last three judge it.** You cannot judge what you have not described.

---

## 4. Global rules

### 4.1 All-relevant, not minimal-optimal
Find **every** feature carrying usable signal, including redundant ones. We do **not** use mRMR or RFE (their objective is wrong for us). Part G collapses **only near-duplicates (≥ 0.95)** — deduplication, never compression. The 0.95 cut is a design commitment; **never lower it to "reduce feature count."**

### 4.2 The cost asymmetry governs every threshold
| Error | Cost |
|---|---|
| Wrongly **dropped** a real predictor | Silent, permanent — the data scientist never knows to look. |
| Wrongly **kept** a useless column | One extra row in a review table. ~4 seconds. |

**Tune for high recall on drops.** Every threshold sits **one tier below** the conventional "weak" boundary. We answer "is this so weak that showing it wastes someone's time," not "is this good enough to use." Be visibly, deliberately cowardly about dropping.

### 4.3 Shadow-permutation floor is the primary criterion
For every feature, `dsp` computes an effect against the effect achievable at random — estimated by **permuting that same column** (Boruta's mechanism). The fixed constants in §9 are **backstops reported alongside**, not the primary bar. The shadow floor self-calibrates to cardinality, sample size, skew, and missingness for free.

### 4.4 Hard ordering — four edges that are NOT negotiable
1. **A → B.** Viability sets strictness for everything after.
2. **D before F.** Sentinels must be nulled before any statistic runs — one `-999` spike corrupts every mean and correlation.
3. **E before F.** The split must exist before any per-feature target-association statistic runs (Part B's aggregate viability and Part D's missingness diagnostic are the only permitted earlier target contact). *This is the constraint people violate most and notice least.*
4. **F before G.** Representative selection needs effect sizes.

### 4.5 Gates are exit codes
Each part ends with `dsp.gate("<part>")`. If exit conditions are unmet, it fails and the run stops. "Do not proceed without X" is enforced, not suggested.

### 4.6 Artifacts — every part writes the same shapes
- **Decision card** (JSON): what the part decided and why.
- **Parquet**: any per-row or per-column table (fold indices, ledger rows).
- **JSONL**: append-only event log.
- **Figure + sidecar**: every plot ships with a machine-readable sidecar of the numbers behind it.

---

## 5. Verdict vocabulary

Every column exits with exactly one verdict in the ledger:

| Verdict | Meaning |
|---|---|
| `keep` | Passed relevance, is a representative (or unique) after redundancy. Goes into the first model. |
| `review` | Borderline — survived because of the cost asymmetry. A human should look. Default for anything uncertain. |
| `drop` | Excluded from the first model. **Still in the ledger with its numbers.** Never deleted. |
| `redundant` | Statistically near-duplicate (≥ 0.95) of a kept representative. Which one was kept is recorded. |
| `engineer` | Not usable raw, but flags a human to derive something (e.g. a datetime, a ratio the business believes in). The tool does **not** generate it. |
| `leak-suspect` | Effect so strong / structurally suspicious it likely encodes the target. Flagged, never silently kept or dropped — adjudicated at H. |
| `structural-drop` | Removed by a rule needing no calibration (constant, id, duplicate, near-empty). |

---

## 6. The `dsp` command surface

> This is the contract the library must satisfy. Each command writes its artifacts (§4.6) and returns a typed object the next part reads. Signatures are the target API.

### 6.0 Infrastructure
| Command | Returns / effect |
|---|---|
| `dsp.load(path)` | `RunContext` — the dataframe handle + config, the single object every part reads/writes. |
| `dsp.gate(part)` | Raises `GateFailure` (non-zero) if the part's exit conditions are unmet. |
| `dsp.query.*` | The only sanctioned data-access layer (counts, groupbys, crosstabs). Agent never touches pandas directly. |
| `dsp.ledger` | The growing per-column ledger; every part appends. |

### 6.1 Part A — Frame
| Command | Returns |
|---|---|
| `dsp.frame.infer(ctx)` | Config: `target`, `target_type`, `date_col?`, `id_cols`, `grain`, `prevalence`. Every inference is a **corrigible claim**, not a fact. |
| `dsp.frame.questions(ctx)` | The ≤2 questions to ask the human (only where the data genuinely cannot answer). |
| `dsp.frame.confirm(ctx, answers)` | Freezes the config after the human checkpoint. |

### 6.2 Part B — Viability
| Command | Returns |
|---|---|
| `dsp.viability.assess(ctx)` | `positives` (count, not rate), `target_nulls`, `prevalence`, `censoring`, `effective_n`, and the **strictness tier** (§10). |

### 6.3 Part C — Inventory
| Command | Returns |
|---|---|
| `dsp.inventory.semantic_types(ctx)` | Semantic type per column (§8) — *not* dtype. |
| `dsp.inventory.structural_drops(ctx)` | Constants, >98% single value, identifiers, exact duplicates → `structural-drop`. |
| `dsp.profile.wrap(ctx)` | Wraps `ydata-profiling` / `sweetviz` (optional extra) for dtypes/cardinality/distributions. Never rebuild these. |

### 6.4 Part D — Value integrity
| Command | Returns |
|---|---|
| `dsp.values.sentinels(ctx)` | Sentinel register (`-999`, `9999`, `""`, etc.). **Nulls them before F.** |
| `dsp.values.distributions(ctx)` | Distribution verdicts per column (skew, zero-inflation, spikes). |
| `dsp.values.missingness(ctx)` | Missingness rate + **co-missing clusters** (columns that go missing together). |
| `dsp.values.missingness_predicts_target(ctx)` | For each column: does its missingness indicator predict the target? Reports `AUC`, not a mechanism label. |

### 6.5 Part E — Partition
| Command | Returns |
|---|---|
| `dsp.partition.strategy(ctx)` | Split strategy **derived** from grain + time presence (§11), not chosen. |
| `dsp.partition.make_folds(ctx)` | Frozen fold indices written to disk (parquet). Every later part and the modeling team use these exact folds. |

### 6.6 Part F — Relevance + stability
| Command | Returns |
|---|---|
| `dsp.relevance.run(ctx)` | Per feature, **on training folds only**: native effect (§8 dispatch), CI, **q-value** (BH-FDR), shape gap (`x_stat − c_stat`), and **fold spread** (stability). |
| `dsp.relevance.shadow_floor(ctx, col)` | The permutation floor for that column (§4.3). Primary pass/fail. |

### 6.7 Part G — Redundancy
| Command | Returns |
|---|---|
| `dsp.redundancy.pairs(ctx)` | Native pairwise metric per feature pair (Spearman / Cramér's V / η). |
| `dsp.redundancy.components(ctx, threshold=0.95)` | **Connected components** on edges ≥ 0.95, plus a deterministic representative per component. No dendrogram, no cut height. |

### 6.8 Part H — Verdict
| Command | Returns |
|---|---|
| `dsp.leakage.adjudicate(ctx)` | Resolves the accumulated leak register (§13) once, as a batch. |
| `dsp.boruta.crosscheck(ctx)` | All-relevant safety net: fits Boruta on surviving **and** dropped features; flags anything Boruta confirms that the filter dropped (§16). |
| `dsp.verdict.ledger(ctx)` | The final ledger — verdict + reason per column. |
| `dsp.verdict.handoff(ctx)` | The human-facing summary (§14.1). |

---

## 7. Part-by-part procedure

Each part below: **purpose → run → decide → leak detectors that fire here → gate.**

### Part A — Frame  *(no target contact yet)*
- **Purpose:** state what we are predicting, for whom, at what grain.
- **Run:** `dsp.frame.infer(ctx)`. Auto-derive target, target type, date column, whether IDs repeat, prevalence.
- **Decide:** Ask back only via `dsp.frame.questions` where the data genuinely cannot answer — usually 0–2 questions. Then `dsp.frame.confirm`.
- **Human checkpoint (mandatory):** This is the **only part with no automated error detector**. Detectable errors (non-unique keys, target nulls) get caught later; **silent errors** (wrong grain, ambiguous negatives, prediction-time = event-time, survivorship in the population, target drift) produce a flawless analysis of the *wrong question*. State every inference as a corrigible claim; record every assumption.
- **Leak detectors:** none yet.
- **Gate `A`:** config frozen, target identified (or explicitly "no target"), grain stated in plain language.

### Part B — Viability  *(first target contact — allowed, no per-row modeling)*
- **Purpose:** is there enough target to learn from, and how strict should we be?
- **Run:** `dsp.viability.assess(ctx)`.
- **Decide:** Set the strictness tier consumed by every later part. Cheapest part to run; was missing from early drafts.
- **Gate `B`:** viability floor met — **≥ 50 positives total and ≥ 10 per fold** (§9). Below that, drop to the small-n ladder (§10) or halt with a clear message.

### Part C — Inventory  *(no target)*
- **Purpose:** what is each column, *structurally*?
- **Run:** `dsp.inventory.semantic_types(ctx)`, then `dsp.inventory.structural_drops(ctx)`, optionally `dsp.profile.wrap(ctx)`.
- **Decide:** Semantic type is **not dtype** — an 8-level integer is categorical; a 40,000-level string is an identifier. This decides which test each column gets in F. Get it wrong and every downstream number uses the wrong method.
- **Leak detectors:** **name heuristics** fire here (columns named like the target, `*_flag` created post-outcome, `score`, `prediction`, `target_*`). Record to the register; do not act yet.
- **Gate `C`:** every column has a semantic type; structural drops recorded in the ledger.

### Part D — Value integrity  *(partly target)*
- **Purpose:** which values are real, which absent, and why.
- **Run:** `dsp.values.sentinels` → **null the sentinels** → `dsp.values.distributions` → `dsp.values.missingness` → `dsp.values.missingness_predicts_target`.
- **Decide:** Co-missing clusters reveal shared upstream failures; the *cluster membership indicator* is often a better feature than any member — flag as `engineer`.
- **Leak detectors:** **future timestamps** and **missingness-predicts-target leaks** fire here. Report `missingness_predicts_target: AUC 0.71` — an honest measurement, **not** an `MNAR` label the data cannot support (Little's MCAR test is rejected — it assumes multivariate normality, is unreliable for categoricals, and breaks numerically above ~30 variables).
- **Gate `D`:** all sentinels nulled (**hard prerequisite for F**, §4.4 edge 2); missingness map complete.

### Part E — Partition  *(no target values touched — only indices)*
- **Purpose:** which rows may inform the analysis, held identical for everyone downstream.
- **Run:** `dsp.partition.strategy(ctx)` then `dsp.partition.make_folds(ctx)`.
- **Decide:** Strategy is **derived** from grain + time:
  - Has a usable date → **time-based split** (train past, validate future).
  - Repeating entity id (grain coarser than row) → **grouped folds** (an entity never spans folds).
  - Otherwise → **stratified K-fold** on the target.
  - Small-n → fewer folds per §10.
- **Gate `E`:** folds frozen to disk. **Nothing downstream may touch the target until this gate passes** (§4.4 edge 3).

### Part F — Relevance + stability  *(target — train folds only)*
- **Purpose:** does each feature relate to the target, reliably?
- **Run:** `dsp.relevance.run(ctx)` — dispatches the right test per feature type × target type (§8), **on training folds only**, with:
  - **Effect size** on the native scale, labeled with its metric name.
  - **Confidence interval** on the effect.
  - **q-value** — Benjamini-Hochberg FDR across all features (§17.6).
  - **Shape gap** `x_stat − c_stat` — how much signal a monotone metric is discarding (§17.2). Large gap → non-monotone signal → `review`/`engineer`, never auto-drop.
  - **Fold spread** — stability across folds, nearly free because computed per fold.
  - **Shadow floor** (`dsp.relevance.shadow_floor`) — the primary bar (§4.3).
- **Decide (per feature):**
  - Effect **below both** the shadow floor **and** the §9 backstop, with a stable (narrow) fold spread → `drop`.
  - Effect above the floor → `keep`-eligible (final `keep` after G).
  - Anything borderline, unstable across folds, or with a large shape gap → `review`.
  - Never rank globally — rank **within feature type** (§8).
- **Leak detectors:** **suspicious effect sizes** fire here (outlier-relative-to-peers, §13). Record; adjudicate at H.
- **Gate `F`:** every non-structural column has an effect, CI, q-value, shape gap, fold spread, and a shadow-floor decision.

### Part G — Redundancy  *(no target)*
- **Purpose:** who duplicates whom.
- **Run:** `dsp.redundancy.pairs(ctx)` → `dsp.redundancy.components(ctx, threshold=0.95)`.
- **Decide:** Within each connected component (edges ≥ 0.95), keep the deterministic representative (highest Part-F effect, ties broken by fewer missing, then name order). Others → `redundant`, with the representative recorded. Pairs in **0.70–0.95** → `review` (redundancy-review band), **not** collapsed. VIF > 10 → flag only, never drop (§9).
- **Gate `G`:** every kept feature is either unique or a representative; each `redundant` names its representative.

### Part H — Verdict  *(assemble)*
- **Purpose:** what we hand over, and why.
- **Run:**
  1. `dsp.leakage.adjudicate(ctx)` — resolve the whole leak register at once (§13). Each surviving leak → `leak-suspect` with its detector and type.
  2. `dsp.boruta.crosscheck(ctx)` — the all-relevant safety net (§16). Anything Boruta confirms that the filter dropped → re-flag to `review`.
  3. `dsp.verdict.ledger(ctx)` — assign the final verdict + reason per column.
  4. `dsp.verdict.handoff(ctx)` — the human summary.
- **Gate `H`:** every column has exactly one verdict and a non-empty reason; leak register fully adjudicated; drop-rate-per-rule logged (§9).

---

## 8. The dispatch table

**One question for every target type:** *give me a scalar association between feature X and target Y, on a comparable scale, with a p-value.* Only this lookup changes; the phases are identical.

**Read each row with its own metric name. Rank within a column, never across columns.** `source_segment · IV 0.41` next to `catchlight_score · AUC 0.71` is interpretable but not sortable together.

**Feature types (rows):** continuous, count, nominal, ordinal, binary, high-cardinality categorical, datetime.
**Target types (columns):** binary, multiclass, ordinal, regression, survival.

| Feature ↓ / Target → | **Binary** | **Multiclass** | **Ordinal** | **Regression** | **Survival** |
|---|---|---|---|---|---|
| **Continuous** | Single-feature **AUC** + Cliff's δ; `x_stat−c_stat` for non-monotone | OvR AUC / **Kruskal–Wallis η²** | **Spearman ρ** | **Spearman ρ** (+ Pearson) | **Univariate Cox** (HR, p, C-index) |
| **Count** | Split: `is_zero` (AUC) **+** Spearman on positives (hurdle) | KW η² on positives + `is_zero` | Spearman | Spearman + `is_zero` | Cox on count **+** `is_zero` |
| **Nominal** | **IV / WoE** (optbinning) + **Cramér's V** (Bergsma) | **Cramér's V** (Bergsma) | Cramér's V + KW | **Correlation ratio η²** / KW | Cox w/ dummies / **log-rank** |
| **Ordinal** | IV (monotone bins) + Cramér's V; Kendall τ *diagnostic only* | Cramér's V | **Spearman / Kendall τ** | Spearman | Cox (ordinal as numeric + as factor) |
| **Binary** | **IV** / phi / Cramér's V (2×2) | Cramér's V | Cliff's δ / Mann–Whitney | **point-biserial** / Cliff's δ | Cox / log-rank |
| **High-card categorical** | **cross-fold IV** (mean over folds) + Bergsma V | Bergsma V (bias-corrected) | cross-fold IV | cross-fold **target-encoded η²** (bias-corrected) | Cox w/ cross-fold encoding |
| **Datetime** | **Derive first** (see below), then route each derivative by its own type | ← | ← | ← | ← |

**Datetime is never tested raw.** Derive (year, month, day-of-week, hour, is_weekend, days-since-epoch, cyclical sin/cos, recency-to-reference), then route each derivative through the table. Testing a raw epoch is meaningless.

**Ordinal asymmetry:** nominal is the fallback. Treating ordinal as nominal loses power (still valid). Treating nominal as ordinal produces a *meaningless number* nothing catches. **The ordinal metric (Kendall τ) may never trigger an auto-drop** — it is diagnostic only. Resolve ordinal vs nominal by: value lexicon (`low/medium/high`) → name heuristic → compute both and let the comparison be the diagnostic.

**Folded into existing paths, not new columns:**
- **Counts** → regression path + test `is_zero` separately if zero-inflated.
- **Proportions** → regression path + a logit-transformed variant.
- **Multilabel** → a loop: run binary *k* times, one per label.
- **No target** → Parts A, C, D, G, H run; only F needs a target.

**Deferred (do not attempt — genuinely different math):** time-series forecasting (lags/stationarity/ACF-PACF), learning-to-rank (query groups break row independence), competing risks.

---

## 9. Thresholds reference

**Primary criterion is always the shadow-permutation floor (§4.3). Everything here is a backstop, reported alongside.** Every value sits one tier below the conventional "weak" boundary (§4.2).

| Purpose | Metric | Value | Source |
|---|---|---|---|
| Auto-drop | single-feature AUC | < 0.52 | one tier below 0.55 weak |
| Auto-drop | Information Value | < 0.01 | one tier below Siddiqi's 0.02 "useless" |
| Auto-drop | Spearman ρ | < 0.05 | one tier below weak |
| Auto-drop | Cliff's δ | < 0.07 | one tier below 0.147 negligible |
| Auto-drop | C-index (survival) | < 0.55 | 0.5 = random |
| Auto-drop | Cramér's V | **cardinality-dependent** (below) | Cohen ÷ √(min(r,c)−1) |
| Screening p | Cox (survival) | < 0.20 | liberal univariate Cox before LASSO |
| Leak flag | AUC / IV / C-index | > 0.85 / 0.50 / 0.75 | outlier-relative-to-peers is primary |
| Redundancy collapse | any pairwise | **≥ 0.95** | **design commitment — never lower** |
| Redundancy review | any pairwise | 0.70–0.95 | — |
| Multicollinearity | VIF | > 10, **flag only** | common rule of thumb (context-dependent) |
| Drift | PSI | > 0.25 | 0.1 moderate / 0.25 significant |
| Viability floor | positives | < 50 total, < 10/fold | halt / small-n |

### 9.1 Cardinality-dependent Cramér's V floors
A flat floor silently kills real small effects on any categorical with > 3 levels. Cohen's benchmark divides by √(min(r,c) − 1); auto-drop is set one tier below "small":

| Levels | small | **auto-drop** |
|---|---|---|
| 2 | 0.10 | **0.05** |
| 3 | 0.07 | **0.035** |
| 5 | 0.05 | **0.025** |
| 10 | 0.033 | **0.017** |

**"Levels" means `min(r, c)`** — the shorter side of the feature×target table, i.e. the smaller of (target levels, feature levels). Against a **binary target** this is always 2, so the 0.05 floor applies to every categorical; the finer rows engage only for multiclass/ordinal targets, where the target contributes the extra levels.

**Always apply Bergsma's bias correction (§17.1) first.** Uncorrected V on a 5×5 table reads ≈ 0.02 under independence even at n = 10,000 — without correction the floor is measuring bias, not signal.

### 9.2 Structural drops — no calibration needed
Zero variance · > 98% single value · > 95% missing · exact duplicate column · identifier. These become `structural-drop` immediately in Part C.

### 9.3 Log the drop-rate per rule on every run
**This is the primary calibration signal.** If a rule kills 40% of columns, either the data has no signal or the rule is too aggressive — only watching counts across several real datasets says which. `dsp` writes this to the run's decision card.

### 9.4 Expected behavior (sanity check)
On a 300-column dataset: ~100–150 auto-dropped (mostly structural — constants, IDs, duplicates, near-empty), 5–15 flagged suspicious, the rest ranked in a review band. A real reduction in what a human reads, without the tool making a single genuine judgment call.

---

## 10. Small-n ladder

**Gate on effective sample *per test* (`effective_n` from Part B), not total rows.**

| Effective n | Behavior |
|---|---|
| ≥ 100 | Full pipeline. |
| 30–100 | 3 folds, raise floors, mark every verdict `reduced-power`. |
| **< 30** | **Structural drops only. No statistical verdicts.** Say so in the output. |

Below 30, dropping a feature for AUC 0.51 on 60 rows is noise-driven vandalism dressed as rigor. The tool refuses.

---

## 11. Partition strategy — the decision tree

Derived in Part E, never chosen by the agent:

1. **Usable date column present** → time-based split (train on past, validate on future). Check for window overlap.
2. **Repeating entity id (grain coarser than the row)** → grouped K-fold; an entity never appears in two folds.
3. **Neither** → stratified K-fold on the target.
4. **Effective n in 30–100** → 3 folds and raised floors (§10).

Folds are frozen to disk and reused verbatim by every later part **and** the modeling team.

---

## 12. Leakage is a register, not a part

Detectors need inputs from different stages, so leakage **accumulates across parts and is adjudicated once, as a batch, at H**. Column-level leakage detection has **no established methodology** — these eight detectors are **heuristics and are labeled as such** in the output.

### 12.1 The three-type temporal taxonomy
| Type | Meaning | Example detector |
|---|---|---|
| **Direct outcome encoding** | Column is a transform of the target | name heuristic; near-perfect single-feature effect |
| **Execution-dependent metric** | Value only exists because the outcome happened | missingness-predicts-target; post-outcome `*_flag` |
| **Future information** | Value recorded after prediction time | future timestamp vs reference date |

### 12.2 Where each detector fires
| Detector | Fires at | Type |
|---|---|---|
| Target-like column name | C | direct |
| Post-outcome flag naming (`score`, `prediction`, `*_final`) | C | direct/execution |
| Future timestamp vs prediction-time reference | D | future |
| Missingness predicts target (AUC high) | D | execution |
| Value only present for positives | D | execution |
| Single-feature effect is an **outlier vs peers** | F | direct |
| Effect above fixed leak backstop (§9) | F | direct |
| Perfect/near-perfect separation | F | direct |

**Outlier-based detection is primary; fixed thresholds are backstops.** Where the best honest feature reaches 0.62, an 0.80 is glaring; where several legitimately reach 0.82, it isn't.

---

## 13. Structural error awareness (what the tool cannot catch)

Stated in the output so nobody over-trusts the run:
- **Part A silent errors** (§7 Part A) — wrong grain, ambiguous negatives, prediction-time = event-time, survivorship, target drift. No statistic reaches these; they are contradictions between the data and the world.
- **Interaction blindness** — filter selection is blind to features that are weak alone but strong together (XOR; dose÷weight; lat+long; income+debt). Feature-feature correlation gives **no** information about this. Mitigations: `drop` never deletes; ask for domain ratios at Part A; Boruta cross-check at H. **Narrows the gap; does not close it.**

---

## 14. The ledger — output contract

One row per column. This *is* the deliverable.

| Field | Description |
|---|---|
| `column` | Name. |
| `semantic_type` | From Part C. |
| `verdict` | One of §5. |
| `metric_name` | The native metric used (e.g. `IV`, `AUC`, `Spearman ρ`, `C-index`). |
| `effect` | Value on that metric's scale. |
| `effect_ci` | Confidence interval. |
| `q_value` | BH-FDR-corrected. |
| `shadow_floor` | The permutation bar it was compared against. |
| `shape_gap` | `x_stat − c_stat` (non-monotone signal). |
| `fold_spread` | Stability across folds. |
| `redundant_with` | Representative column, if `redundant`. |
| `leak_flag` | Detector + taxonomy type, if any. |
| `reason` | **Plain-language sentence** — why this verdict. Never empty. |
| `rank_within_type` | Position within its feature type. |
| `power_flag` | `reduced-power` if §10 applied. |

Written as parquet + JSONL. Every `drop`/`redundant`/`structural-drop` row **stays** — the audit trail answers "why isn't `region` in the model?" in three months without reopening a notebook.

### 14.1 Handoff summary (human-facing)
`dsp.verdict.handoff` produces: counts per verdict, drop-rate per rule (§9.3), the top-N `keep` within each feature type, every `leak-suspect` with its reason, every `engineer` candidate, and an explicit **limits** note (§13). It states plainly what the tool did **not** decide.

---

## 15. Stop conditions

The agent halts and reports (never guesses past) when:
- A gate fails (§4.5) — report which exit condition and why.
- Viability floor unmet (§9) and small-n ladder says stop (§10).
- A required statistic is missing from `dsp` (§1) — stop, name it, ask the user.
- Part A cannot resolve target/grain even after the ≤2 questions — escalate to the human; do not assume.

---

## 16. What this tool is not

- **Not an auto-selector.** It produces a ranked, reasoned *shortlist*; the human decides (§2). Building it to silently pick the final set contradicts the design.
- **Not a model.** No modeling, no feature engineering (only `engineer` *flags*).
- **Not a claim of better accuracy.** The honest value is **human hours saved, leakage caught early, reduced pipeline cost, and an audit trail** — not better models.
- **Boruta at H is a safety net, not a selection method.** It answers one question: *did the filter throw away anything you'd have used?*

---

## 17. Metric definitions (so `dsp` and the agent agree exactly)

### 17.1 Bergsma bias-corrected Cramér's V
For an r×c contingency table with Pearson χ² and sample size n, let φ² = χ²/n:

- φ̃² = max(0, φ² − (r−1)(c−1)/(n−1))
- r̃ = r − (r−1)²/(n−1)
- c̃ = c − (c−1)²/(n−1)
- **Ṽ = √( φ̃² / min(r̃−1, c̃−1) )**

Always use Ṽ (not raw V) for categorical association and for the §9.1 floors.

### 17.2 The x-statistic (Bruce Lund) and shape gap
- **c-stat** = single-feature AUC (rank concordance of the raw feature vs a binary target).
- **x-stat** = the c-statistic of a logistic regression on the **WoE-transformed, optimally-binned** feature (via `optbinning`).
- **Proven property:** x-stat ≥ c-stat, with **equality iff the feature is monotone vs the target.**
- **shape_gap = x_stat − c_stat** is exactly the signal a monotone metric discards. A U-shaped feature reads ~0.51 raw and ~0.68 binned → gap ≈ 0.17 → `review`/`engineer`, never auto-drop. Replaces mutual information entirely (MI has no natural scale and swings with the estimator).

### 17.3 Information Value (Siddiqi bands)
IV = Σ_bins (%good − %bad) · WoE, WoE = ln(%good / %bad). Computed with `optbinning`.

| IV | Meaning |
|---|---|
| < 0.02 | Useless (**our auto-drop backstop is one tier below, 0.01**) |
| 0.02–0.1 | Weak |
| 0.1–0.3 | Medium |
| 0.3–0.5 | Strong |
| **> 0.5** | **Suspicious — double-check for leakage** (feeds §12) |

For high-cardinality categoricals, use **cross-fold IV** (mean over folds) to counter overfit inflation.

### 17.4 Cliff's δ (non-parametric effect, binary target vs continuous)
Proportion of non-overlap between the two groups' distributions, range −1..1.

| |δ| | Effect |
|---|---|
| < 0.147 | Negligible (**auto-drop backstop 0.07, one tier below**) |
| 0.147–0.33 | Small |
| 0.33–0.474 | Medium |
| ≥ 0.474 | Large |

### 17.5 Shadow-permutation floor
Permute the column (wiping the feature↔target relationship, preserving the marginal), recompute the *same* native metric, repeat B times; the floor is a high percentile (start permissive — the percentile is an open tuning item) of the shadow distribution. A feature must clear **its own** shadow floor, which automatically rises for high-cardinality noise columns.

### 17.6 Benjamini-Hochberg FDR
Across all features tested in F, convert p-values to **q-values** controlling the false-discovery rate. A feature's q-value is reported in the ledger; used with (not instead of) effect size — significance without a clearing effect is still `review`, not `keep`.

### 17.7 Population Stability Index (drift)
PSI = Σ (%actual − %expected) · ln(%actual / %expected) across bins. < 0.10 stable · 0.10–0.25 moderate · **> 0.25 significant drift** → flag.

### 17.8 C-index (survival discrimination)
Fraction of comparable pairs correctly ranked; 0.5 = random, 0.7–0.8 good, > 0.8 excellent *for a whole model*. So a **single feature** at C-index > 0.75 is a screaming leak (§9). Auto-drop backstop 0.55. Biased upward above ~40% censoring — report censoring alongside.

---

## Sources

- Bergsma, *A bias-correction for Cramér's V and Tschuprow's T* — [stats.lse.ac.uk](https://stats.lse.ac.uk/bergsma/pdf/cramerV3.pdf); formula confirmed via [Cramér's V, Wikipedia](https://en.wikipedia.org/wiki/Cram%C3%A9r's_V)
- Lund, *Information Value Statistic and Predictors for Logistic Regression* (ASA 2014) — [amstat.org](https://ww2.amstat.org/meetings/proceedings/2014/data/assets/pdf/313981_87373.pdf)
- Boruta all-relevant selection / shadow features — [CRAN Boruta](https://cran.r-project.org/web/packages/Boruta/Boruta.pdf)
- Cliff's δ thresholds — [effsize, CRAN](https://cran.r-project.org/web/packages/effsize/effsize.pdf)
- Information Value bands (Siddiqi) — [listendata](https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html)
- Population Stability Index thresholds — [listendata](https://www.listendata.com/2015/05/population-stability-index.html)
- VIF rule of thumb & caution — [Springer, O'Brien 2007](https://link.springer.com/article/10.1007/s11135-006-9018-6)
- Liberal univariate Cox screening (p < 0.20–0.25) — [Lasso-Cox, jsurvival](https://www.serdarbalci.com/jsurvival/articles/09-lassocox-comprehensive.html)
- C-index interpretation — [scikit-survival](https://scikit-survival.readthedocs.io/en/stable/user_guide/evaluating-survival-models.html)
- optbinning (IV/WoE/Gini) — [optbinning docs](https://gnpalencia.org/optbinning/)
