# Feature Selection Playbook — a guide for Claude Code

**Status:** v3 — locked (agent + `fsp` tools; in sync with the built package)

You are Claude Code. Handed a path to a single dataframe, you run the whole feature-selection screening: for each part you **call the `fsp` tools** for the deterministic work and **write code** where judgment or adaptation is needed, then produce a **documented notebook** — a ranked, reasoned shortlist with a verdict and a reason for every column. A human still makes the final feature decision (§16).

This document is the **guide**: the **method** (what to do, in what order), the **rules and thresholds** (how to decide), and the **exact math** (§17). Its companion is the **`fsp` package** (catalogued in [`TOOLS.md`](TOOLS.md)), which implements the deterministic work — the §17 math, structural checks, folds, ledger, notebook, and gates. **Call `fsp` for those; never re-derive them.** You own the framing, the judgment (verdicts), adaptation to messy data, and the notebook narrative.

---

## 0. How to read this document

- You run **Parts A–H** in order, each as the same four-step loop (§3.1): **compute → decide → document → verify.**
- **Compute with `fsp`, decide yourself.** The deterministic work is `fsp` tools (it implements §9/§17 exactly); the judgment is yours. Never re-derive a metric `fsp` provides or improvise a method the guide doesn't define (§1).
- Each part ends at a **gate** (§4.5): a self-check you must pass before moving on. If it fails, **stop**.
- The **deliverable is a documented notebook** that you grow one section per part (§14). The per-column **ledger** is its final section.
- The numbers you must not invent are in the **Thresholds** table (§9); the formulas you must implement exactly are in **Metric definitions** (§17). Both are sourced.

---

## 1. Prime directive

> **Call `fsp` for the deterministic work; write code only for framing, judgment, adaptation, and narrative. Never re-implement a metric `fsp` provides, never hardcode a number the guide fixes, and if the guide doesn't specify how to measure something, STOP and ask the user — do not invent a method.**

Why this is absolute: the value of a screening tool is that runs are **comparable** — same tests, same thresholds, same output shape, across every dataset and project. If you improvise methods or numbers, that comparability is lost and the thresholds below become meaningless (they were chosen against the exact definitions in §17). The guide fixes the **what** and the **numbers**, `fsp` fixes the **how** for the deterministic parts, and you own the **judgment** and the adaptation to the data in front of you.

Corollary rules:
- **Use `fsp` for §9 numbers and §17 formulas** (`fsp.thresholds`, `fsp.metrics`). Do not hardcode a threshold or re-implement (or roughly approximate) a metric `fsp` already computes.
- **No per-feature target-association statistic before the folds exist (Part E).** Only Part B's aggregate viability facts and Part D's missingness-leak diagnostic may touch the target earlier. See §4.4 edge 3 — the most-violated, least-noticed rule.
- **Every drop stays in the ledger.** `drop` means "excluded from the first model," never "deleted."

---

## 2. Four roles

| Role | Who | Responsibility |
|---|---|---|
| **Brain** | You (Claude Code) | Read this guide, **make every decision** (frame, semantic type, drop/keep, verdicts), orchestrate the parts, write code for adaptation and the notebook narrative, and self-check the gate. Ask the human the ≤2 questions at Part A. |
| **Body** | The **`fsp`** package | Does the deterministic work — the §17 math, structural checks, folds, ledger, notebook, gates. Computes and reports; **never makes a judgment call.** Deterministic and reproducible. |
| **Guide** | This document | Fixes the method, order, decision rules, thresholds (§9), and exact math (§17). The source of truth for *what* and *how much*. |
| **Decider** | The human | Reviews the notebook, applies domain knowledge the data cannot contain, makes the final feature decision. Confirms Part A (the one mandatory-human step). |

---

## 3. The run at a glance

```
A Frame ──▶ B Viability ──▶ C Inventory ──▶ D Value integrity ──▶ E Partition ──▶ F Relevance+stability ──▶ G Redundancy ──▶ H Verdict
  (no target)  (target)       (no target)     (partly target)       (no target)     (target, split frozen)     (no target)     (assemble)
   config       strictness     semantic         sentinels,            frozen           effect+CI+q+shape+         components+     ledger with
   object       tier           types +          missingness           split            fold-spread               reps           reason/feature
                               structural        clusters
                               drops
                                    │                │                                        │
                              leakage: name     leakage: future ts,                    leakage: suspicious
                              heuristics fire   missingness-leak fire                  effect sizes fire
                                    └────────────────┴──────────────── accumulate ────────────┴──▶ adjudicated once at H
```

**First five parts describe the data; last three judge it.** You cannot judge what you have not described.

### 3.1 The operating loop — how you run every part

Every part is the **same four steps.** This is the core of how you work:

| Step | You do | With |
|---|---|---|
| **1. Compute** | Call the `fsp` tool for this part's facts/statistics (it uses the exact §8/§17 method). Write code only to adapt to messy data. | `fsp.metrics`, `fsp.parts.*` |
| **2. Decide** | Apply this guide's **rules and thresholds** (§5, §8, §9) to the numbers. This is your judgment. | §5 verdicts, §8 dispatch, `fsp.thresholds` |
| **3. Document** | Append a written section to the live notebook — your prose plus the tables and figures behind it. | `fsp.notebook`, `fsp.report` |
| **4. Verify** | Check the part's **gate conditions** (§7). If any fails, **stop** and report — do not proceed. | `fsp.gate` |

Two habits that make this trustworthy:
- **Document as you go, not at the end.** The notebook is written incrementally so a human can watch the analysis unfold and audit each step.
- **When in doubt, keep and flag.** The cost asymmetry (§4.2) means a wrongly-dropped feature is a silent, permanent loss; a wrongly-kept one costs a reviewer four seconds.

---

## 4. Global rules

### 4.1 All-relevant, not minimal-optimal
Find **every** feature carrying usable signal, including redundant ones. Do **not** use mRMR or RFE (their objective is wrong for us). Part G collapses **only near-duplicates (≥ 0.95)** — deduplication, never compression. The 0.95 cut is a design commitment; **never lower it to "reduce feature count."**

### 4.2 The cost asymmetry governs every threshold
| Error | Cost |
|---|---|
| Wrongly **dropped** a real predictor | Silent, permanent — the data scientist never knows to look. |
| Wrongly **kept** a useless column | One extra row in a review table. ~4 seconds. |

**Tune for high recall on drops.** Every threshold sits **one tier below** the conventional "weak" boundary. We answer "is this so weak that showing it wastes someone's time," not "is this good enough to use." Be visibly, deliberately cowardly about dropping.

### 4.3 Shadow-permutation floor is the primary criterion
For every feature, compare its effect against the effect achievable at random — estimated by **permuting that same column** and recomputing the same metric (Boruta's mechanism, §17.5). The fixed constants in §9 are **backstops you report alongside**, not the primary bar. The shadow floor self-calibrates to cardinality, sample size, skew, and missingness for free.

### 4.4 Hard ordering — four edges that are NOT negotiable
1. **A → B.** Viability sets strictness for everything after.
2. **D before F.** Sentinels must be nulled before any statistic runs — one `-999` spike corrupts every mean and correlation.
3. **E before F.** The split must exist before any per-feature target-association statistic runs (Part B's aggregate viability and Part D's missingness diagnostic are the only permitted earlier target contact). *This is the constraint people violate most and notice least.*
4. **F before G.** Representative selection needs effect sizes.

### 4.5 Gates are self-checks, not suggestions
Each part ends with explicit **exit conditions** (§7). Before you move on, verify them with `fsp.gate` and state the result in the notebook. If a condition fails, **stop the run and report which one and why** (§15). "Do not proceed without X" is a hard stop you enforce on yourself — not a soft guideline.

### 4.6 Outputs — what every part leaves behind
- **A notebook section** (§14): prose + the tables and figures behind your decisions, appended live.
- **Ledger rows** (§14): one row per column, updated as parts touch it — the audit trail.
- **Frozen fold indices** (Part E): written to disk so every later part and the modeling team reuse the identical split.
- **Reproducibility:** set a fixed random seed; state it. Same data + same guide ⇒ same verdicts.

---

## 5. Verdict vocabulary

Every column exits with exactly one verdict in the ledger:

| Verdict | Meaning |
|---|---|
| `keep` | Passed relevance, is a representative (or unique) after redundancy. Goes into the first model. |
| `review` | Borderline — survived because of the cost asymmetry. A human should look. Default for anything uncertain. |
| `drop` | Excluded from the first model. **Still in the ledger with its numbers.** Never deleted. |
| `redundant` | Statistically near-duplicate (≥ 0.95) of a kept representative. Which one was kept is recorded. |
| `engineer` | Not usable raw, but flags a human to derive something (e.g. a datetime, a ratio the business believes in). You do **not** generate it. |
| `leak-suspect` | Effect so strong / structurally suspicious it likely encodes the target. Flagged, never silently kept or dropped — adjudicated at H. |
| `structural-drop` | Removed by a rule needing no calibration (constant, id, duplicate, near-empty). |

---

## 6. What each part must produce

For each part you **compute** the left column (call the `fsp` tools — see [`TOOLS.md`](TOOLS.md) for the API), **decide** using the rules, **record** the result in the ledger/notebook, and **verify** the gate. This table is the contract at a glance; §7 is the detail.

| Part | You compute | You decide | You record | Gate check |
|---|---|---|---|---|
| **A** Frame | candidate target / date / id facts; class balance | target, target_type, date_col, id_cols, grain | the frame config | target resolved (or "no target"); grain stated in plain language |
| **B** Viability | positives (count), target nulls, effective_n, prevalence, censoring | strictness tier (§10) | tier + the numbers | effective_n > 0; tier set |
| **C** Inventory | per-column dtype, cardinality, %missing, uniqueness, duplicate map | semantic type (§8) per column; structural drops (§9.2) | type + verdict per column | every column has a semantic type |
| **D** Value integrity | sentinel candidates, distributions, missingness, co-missing clusters | which sentinels are real; `engineer` flags | sentinel register; nulled sentinels | sentinels nulled; missingness mapped |
| **E** Partition | grain + time presence | split strategy (§11) and k | frozen fold indices to disk | folds exist before any F statistic |
| **F** Relevance + stability | per §8 dispatch, once the split is frozen: effect, CI, q-value, shape gap, fold spread, shadow floor | `keep`-eligible / `drop` / `review` per §9 | metrics + verdict per feature | every non-structural column has metrics + a decision |
| **G** Redundancy | pairwise native metric; components at ≥ 0.95 | representative per component; `review` for 0.70–0.95 | `redundant_with` links | every kept feature is unique or a representative |
| **H** Verdict | leak signals (§12); Boruta cross-check (§16) | adjudicate leaks; final verdict + reason for every column | final ledger; closing section | every column has exactly one verdict + a non-empty reason |

---

## 7. Part-by-part procedure

Each part follows the §3.1 loop. Below, per part: what to **compute**, what to **decide**, which **leak detectors** fire, and the **gate** to verify. Each "Compute" step is provided by an `fsp` tool (see [`TOOLS.md`](TOOLS.md)) — call it rather than re-deriving. Document each part in the notebook before you gate.

### Part A — Frame  *(no target contact yet)*
- **Purpose:** state what we are predicting, for whom, at what grain.
- **Compute:** candidate targets, target class balance / dtype / cardinality, columns that parse as dates, columns that are unique-per-row or named like ids, whether any id repeats.
- **Decide & record:** *you* choose the target, target type (one of §8's five), date column, id columns, and grain — each a **corrigible claim**. Where **prediction time differs from the event/observation date**, also record a **prediction-time reference** (`reference_date`) — the moment a real prediction would be made — which the future-timestamp leak check uses (§12.2). Ask the human only where the data genuinely cannot answer (0–2 questions).
- **Human checkpoint (mandatory):** This is the **only part with no automated error detector**. Detectable errors (non-unique keys, target nulls) get caught later; **silent errors** (wrong grain, ambiguous negatives, prediction-time = event-time, survivorship in the population, target drift) produce a flawless analysis of the *wrong question*. State every inference as a corrigible claim; record every assumption in the notebook.
- **Leak detectors:** none yet.
- **Gate `A`:** config recorded, target identified (or explicitly "no target"), grain stated in plain language.

### Part B — Viability  *(first target contact — allowed, no per-row modeling)*
- **Purpose:** is there enough target to learn from, and how strict should we be?
- **Compute:** positives (the minority-class count for classification; **events** for survival; non-null rows for regression), target nulls, prevalence, censoring, and `effective_n` (defined in §10).
- **Decide & record:** set the **strictness tier** (§10) from `effective_n`; it governs every later part. Cheapest part to run.
- **Gate `B`:** effective_n > 0. **Viability floor:** classification ≥ 50 positives (and ≥ 10/fold); **survival** ≥ 50 events; **regression** effective_n ≥ 100. Below the floor, drop to the small-n ladder (§10) or halt with a clear message.

### Part C — Inventory  *(no target)*
- **Purpose:** what is each column, *structurally*?
- **Compute:** for every column — dtype, cardinality, %missing, top values, sample values, uniqueness; the exact-duplicate-column map.
- **Decide & record:** *you* assign each column's **semantic type** (§8 feature types) from the facts — it is **not dtype** (an 8-level integer is categorical; a 40,000-level string is an identifier), and it decides which test the column gets in F. Mark structural drops (§9.2). The target and date column are never structural-dropped.
- **Leak detectors:** **name heuristics** fire here (columns named like the target, `*_flag` created post-outcome, `score`, `prediction`, `target_*`). Record to the register; do not act yet.
- **Gate `C`:** every column has a semantic type; structural drops recorded in the ledger.

### Part D — Value integrity  *(partly target)*
- **Purpose:** which values are real, which absent, and why.
- **Compute:** sentinel candidates (e.g. `-999`, `9999`, `""`, `"unknown"`), distributions (skew, zero-inflation, spikes), missingness rate, and **co-missing clusters** (columns that go missing together — correlate the null-masks, group at ≥ 0.95). Then **null the confirmed sentinels** (hard prerequisite for F, §4.4 edge 2).
- **Decide & record:** confirm which sentinel candidates are real (corrigible). A co-missing cluster's *membership indicator* is often a better feature than any member — flag it `engineer`.
- **Leak detectors:** **future timestamps**, **value-only-present-for-positives**, and **missingness-predicts-target** fire here. Report `missingness_predicts_target: AUC 0.71` — an honest measurement, **not** an `MNAR` label the data cannot support (Little's MCAR test is rejected — it assumes multivariate normality, is unreliable for categoricals, and breaks numerically above ~30 variables). *Implementation note:* in `fsp` the missingness-predicts-target scan is **fold-guarded** (it computes a target AUC), so it is registered right after Part E rather than mid-D — detection timing shifts, but adjudication is still batched at H, and edge-3 stays a hard error.
- **Gate `D`:** all confirmed sentinels nulled; missingness map complete.

### Part E — Partition  *(no target values touched — only indices)*
- **Purpose:** which rows may inform the analysis, held identical for everyone downstream.
- **Compute:** grain + time presence.
- **Decide & record:** *you* pick the split from grain + time (§11), then **freeze fold indices to disk** so every later part reuses the identical split.
- **Gate `E`:** folds frozen. **Nothing downstream may touch the target with a per-feature statistic until this gate passes** (§4.4 edge 3).

### Part F — Relevance + stability  *(target — split frozen)*
- **Purpose:** does each feature relate to the target, reliably?
- **Compute (per feature, once the split is frozen — §4.4 edge 3):** the point estimate uses all available rows; the per-fold **test** effects give the stability spread, and genuinely out-of-fold scoring is used only where in-sample inflates (high-cardinality IV / target-encoding, §17.3). First derive any datetime feature into its parts and split any zero-inflated count into `is_zero` + positives (§8); then run the native test per feature type × target type (§8 dispatch), with:
  - **Effect size** on the native scale, labeled with its metric name.
  - **Confidence interval** on the effect — the uniform bootstrap-percentile method (§17.12).
  - **q-value** — Benjamini-Hochberg FDR across all features (§17.6).
  - **Shape gap** `x_stat − c_stat` — how much signal a monotone metric is discarding (§17.2). **Binary target only** (leave blank where undefined). Large gap → non-monotone signal → `review`/`engineer`, never auto-drop.
  - **Fold spread** — stability across folds, nearly free because you compute per fold.
  - **Shadow floor** — the primary bar (§4.3, §17.5): permute the column, recompute the same metric, take a high percentile.
- **Missing values — available-case (pairwise):** compute each feature's association on the rows where **both** the feature and the target are present. **Never impute for the association** — imputation biases the estimate. If a feature's pairwise n falls below the §10 floor, mark it `reduced-power` (or structural-only under 30). Rows with a null *target* are excluded from every association.
- **Decide & record (per feature):**
  - Effect **below both** the shadow floor **and** the §9 backstop, with a stable (narrow) fold spread → `drop`.
  - Effect above the floor → `keep`-eligible (final `keep` after G).
  - Anything borderline, unstable across folds, or with a large shape gap → `review`.
  - Never rank globally — rank **within feature type** (§8).
- **Leak detectors:** **suspicious effect sizes** fire here (outlier-relative-to-peers, §12). Record; adjudicate at H.
- **Gate `F`:** every non-structural column has an effect, CI, q-value, fold spread, a shadow-floor decision, and — for a binary target — a shape gap.

### Part G — Redundancy  *(no target)*
- **Purpose:** who duplicates whom.
- **Compute:** the native pairwise metric per feature pair, chosen by the pair's types and normalized to **[0, 1]** so 0.95 means the same thing everywhere — then connected components on edges ≥ 0.95:
  - numeric ↔ numeric → **|Spearman ρ|**
  - numeric ↔ categorical (incl. binary) → **correlation ratio η** (§17.9)
  - categorical ↔ categorical → **Bergsma's Cramér's V** (§17.1); binary ↔ binary is its 2×2 case (|phi|)
  - ordinal involved → **|Spearman| / |Kendall τ|**
  VIF, if reported, is computed among the **kept numeric** features only.
- **Decide & record:** within each connected component keep the deterministic representative (highest Part-F effect, ties broken by fewer missing, then name order). Others → `redundant`, with the representative recorded. Pairs in **0.70–0.95** → `review` (redundancy-review band), **not** collapsed. VIF > 10 → flag only, never drop (§9).
- **Gate `G`:** every kept feature is either unique or a representative; each `redundant` names its representative.

### Part H — Verdict  *(assemble)*
- **Purpose:** what we hand over, and why.
- **Compute:** the accumulated leak register (§12); the Boruta cross-check (§16) — fit Boruta on surviving **and** dropped features.
- **Decide & record:** adjudicate each leak signal once, as a batch → `leak-suspect` (with detector + type); re-flag anything Boruta confirms that you dropped → `review`; give **every** column its final verdict + a non-empty reason. Log the drop-rate per rule (§9.3).
- **Document:** the closing section (§14.1).
- **Gate `H`:** every column has exactly one verdict and a non-empty reason; leak register fully adjudicated; drop-rate per rule logged.

---

## 8. The dispatch table

**One question for every target type:** *give me a scalar association between feature X and target Y, on a comparable scale, with a p-value.* Only this lookup changes; the phases are identical.

**Read each row with its own metric name. Rank within a column, never across columns.** `source_segment · IV 0.41` next to `catchlight_score · AUC 0.71` is interpretable but not sortable together.

**Feature types (rows):** continuous, count, nominal, ordinal, binary, high-cardinality categorical, datetime.
**Target types (columns):** binary, multiclass, ordinal, regression, survival.

| Feature ↓ / Target → | **Binary** | **Multiclass** | **Ordinal** | **Regression** | **Survival** |
|---|---|---|---|---|---|
| **Continuous** | Single-feature **AUC** + Cliff's δ; `x_stat−c_stat` for non-monotone | OvR AUC / **Kruskal–Wallis ε²** | **Spearman ρ** | **Spearman ρ** (+ Pearson) | **Univariate Cox** (HR, p, C-index) |
| **Count** | Split: `is_zero` (AUC) **+** Spearman on positives (hurdle) | KW ε² on positives + `is_zero` | Spearman | Spearman + `is_zero` | Cox on count **+** `is_zero` |
| **Nominal** | **IV / WoE** (optbinning) + **Cramér's V** (Bergsma) | **Cramér's V** (Bergsma) | Cramér's V + KW ε² | **Correlation ratio η²** / KW ε² | Cox w/ dummies / **log-rank** |
| **Ordinal** | IV (monotone bins) + Cramér's V; Kendall τ *diagnostic only* | Cramér's V | **Spearman / Kendall τ** | Spearman | Cox (ordinal as numeric + as factor) |
| **Binary** | **IV** / phi / Cramér's V (2×2) | Cramér's V | Cliff's δ / Mann–Whitney | **point-biserial** / Cliff's δ | Cox / log-rank |
| **High-card categorical** | **cross-fold IV** (mean over folds) + Bergsma V | Bergsma V (bias-corrected) | cross-fold IV | cross-fold **target-encoded η²** (out-of-fold, §17.3) | Cox w/ cross-fold encoding |
| **Datetime** | **Derive first** (see below), then route each derivative by its own type | ← | ← | ← | ← |

**Datetime is never tested raw.** Derive (year, month, day-of-week, hour, is_weekend, days-since-epoch, cyclical sin/cos, recency-to-reference), then route each derivative through the table. Testing a raw epoch is meaningless. (This limited, mechanical derivation is the one exception to "no feature engineering," §16 — you still do not invent new business features.)

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
| Auto-drop | point-biserial \|r_pb\| | < 0.05 | one tier below 0.1 weak |
| Auto-drop | correlation ratio η² | < 0.005 | one tier below 0.01 small (§17.9) |
| Auto-drop | Kruskal–Wallis ε² | < 0.005 | one tier below 0.01 small (§17.10) |
| Auto-drop | C-index (survival) | < 0.55 | 0.5 = random |
| Auto-drop | Cramér's V | **cardinality-dependent** (below) | Cohen ÷ √(min(r,c)−1) |
| Screening p | Cox (survival) | < 0.20 | liberal univariate Cox before LASSO |
| Leak flag | AUC / IV / C-index | > 0.85 / 0.50 / 0.75 | outlier-relative-to-peers is primary |
| Redundancy collapse | any pairwise | **≥ 0.95** | **design commitment — never lower** |
| Redundancy review | any pairwise | 0.70–0.95 | — |
| Multicollinearity | VIF | > 10, **flag only** | common rule of thumb (context-dependent) |
| Drift | PSI | > 0.25 | 0.1 moderate / 0.25 significant |
| Viability floor | positives | < 50 total, < 10/fold | halt / small-n |

**Viability floor by target type:** **classification** < 50 positives (minority class); **survival** < 50 events; **regression** effective_n < 100 (§10, §7 B).

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
**This is the primary calibration signal.** If a rule kills 40% of columns, either the data has no signal or the rule is too aggressive — only watching counts across several real datasets says which. Record it in the closing notebook section.

### 9.4 Expected behavior (sanity check)
On a 300-column dataset: ~100–150 auto-dropped (mostly structural — constants, IDs, duplicates, near-empty), 5–15 flagged suspicious, the rest ranked in a review band. A real reduction in what a human reads — most of it from structural facts, not judgment calls.

---

## 10. Small-n ladder

**Gate on effective sample *per test* (`effective_n` from Part B), not total rows.**

**`effective_n` is the binding count for the test, not the row count:**
- **Binary / multiclass / ordinal target** → the **smallest class count** among rows with a non-null target (the minority "events" are the constraint).
- **Regression target** → the count of **non-null target rows**.
- **Survival target** → the number of **events** (uncensored observations) — censored rows carry less information, so events are the binding resource (the events-per-variable logic).

For a feature with missing values, its per-feature effective_n drops further to the rows where **both** the feature and the target are present (§7 F).

| Effective n | Behavior |
|---|---|
| ≥ 100 | Full pipeline. |
| 30–100 | 3 folds, raise floors, mark every verdict `reduced-power`. |
| **< 30** | **Structural drops only. No statistical verdicts.** Say so in the notebook. |

Below 30, dropping a feature for AUC 0.51 on 60 rows is noise-driven vandalism dressed as rigor. Refuse it.

---

## 11. Partition strategy — the decision tree

You determine the split in Part E from the rule below — not by preference. Grain + time presence decide it; then freeze the folds:

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

The **prediction-time reference** for the future-timestamp check comes from Part A (`reference_date`); absent one, `date_col` is used as a weaker proxy — but event-time ≠ prediction-time (§7 A), so treat those hits as lower-confidence.

**Outlier-based detection is primary; fixed thresholds are backstops.** Where the best honest feature reaches 0.62, an 0.80 is glaring; where several legitimately reach 0.82, it isn't. (Modern practice agrees: a feature with absurdly high importance relative to its peers is the first sign of a label proxy.)

---

## 13. Structural error awareness (what this cannot catch)

State these in the notebook so nobody over-trusts the run:
- **Part A silent errors** (§7 Part A) — wrong grain, ambiguous negatives, prediction-time = event-time, survivorship, target drift. No statistic reaches these; they are contradictions between the data and the world.
- **Interaction blindness** — filter selection is blind to features that are weak alone but strong together (XOR; dose÷weight; lat+long; income+debt). Feature-feature correlation gives **no** information about this. Mitigations: `drop` never deletes; ask for domain ratios at Part A; Boruta cross-check at H. **Narrows the gap; does not close it.**

---

## 14. The deliverable — a documented notebook

The deliverable is the **documented results notebook** you build: a section per part (§3.1), ending with the **ledger** — one row per column. Together they are the report a data scientist reads top to bottom.

**Section convention:** each part's section has a heading (`A · Frame` … `H · Verdict`), a short prose paragraph on what you found and why, then the tables and figures behind it. **Derived datetime features** (§8) get their own ledger rows, named for the derivation (e.g. `signup_date__dow`).

Record every column with these fields:

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
| `shape_gap` | `x_stat − c_stat` (non-monotone signal; binary target, else blank). |
| `fold_spread` | Stability across folds. |
| `redundant_with` | Representative column, if `redundant`. |
| `leak_flag` | Detector + taxonomy type, if any. |
| `reason` | **Plain-language sentence** — why this verdict. Never empty. |
| `rank_within_type` | Position within its feature type. |
| `power_flag` | `reduced-power` if §10 applied. |

Also save the ledger as a file (parquet or CSV). Every `drop`/`redundant`/`structural-drop` row **stays** — the audit trail answers "why isn't `region` in the model?" three months later.

### 14.1 Closing section (Part H)
At H, add the closing notebook section: counts per verdict, drop-rate per rule (§9.3), the top-N `keep` within each feature type, every `leak-suspect` with its reason, every `engineer` candidate, and an explicit **limits** note (§13). State plainly what the tool did **not** decide.

---

## 15. Stop conditions

Halt and report (never guess past) when:
- A gate fails (§4.5) — report which exit condition and why.
- Viability floor unmet (§9) and the small-n ladder says stop (§10).
- The guide does not specify how to measure something you need (§1) — stop, name it, ask the user. Do not invent a method.
- `fsp` lacks a tool for a deterministic step you need — stop and say so; do not hand-roll a metric that belongs in `fsp`.
- Your code cannot compute a required statistic correctly (e.g. a library errors on the data and you cannot get a valid result) — report it, don't paper over it.
- Part A cannot resolve target/grain even after the ≤2 questions — escalate to the human; do not assume.

---

## 16. What this tool is not

- **Not an auto-selector.** It produces a ranked, reasoned *shortlist*; the human decides (§2). Building it to silently pick the final set contradicts the design.
- **Not a model.** No modeling, no feature engineering (only `engineer` *flags*, plus the mechanical datetime derivation of §8).
- **Not a claim of better accuracy.** The honest value is **human hours saved, leakage caught early, reduced pipeline cost, and an audit trail** — not better models.
- **Boruta at H is a safety net, not a selection method.** It answers one question: *did the filter throw away anything you'd have used?*

---

## 17. Metric definitions — the exact math (`fsp` implements it)

These are implemented and tested in **`fsp.metrics`** — call `fsp`, do not re-implement them. The formulas below are the exact spec `fsp` is validated against (and what you'd need to verify a result by hand). Internally `fsp` uses the standard libraries (optbinning for WoE/IV, lifelines for Cox/C-index, statsmodels for VIF/FDR).

### 17.1 Bergsma bias-corrected Cramér's V
For an r×c contingency table with Pearson χ² and sample size n, let φ² = χ²/n:

- φ̃² = max(0, φ² − (r−1)(c−1)/(n−1))
- r̃ = r − (r−1)²/(n−1)
- c̃ = c − (c−1)²/(n−1)
- **Ṽ = √( φ̃² / min(r̃−1, c̃−1) )**

Always use Ṽ (not raw V) for categorical association and for the §9.1 floors.

### 17.2 The x-statistic (Bruce Lund) and shape gap
- **c-stat** = single-feature AUC (rank concordance of the raw feature vs a binary target).
- **x-stat** = the c-statistic of a logistic regression on the **WoE-transformed** feature, binned into **quantile bins with no monotonic constraint** (via `optbinning`) so the WoE is free to be non-monotone and capture shape. (optbinning's own optimizer collapses a non-monotone feature to a single bin, which would defeat the purpose.)
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

For **high-cardinality** categoricals, compute IV / target-encoding **out-of-fold**: category statistics from the training folds only, applied to the held-out fold, so no row uses its own target. Report the **mean over folds**. This counters the overfit inflation that makes a high-cardinality noise column look predictive. (For a **regression** target the same out-of-fold encoding is scored as **η²** — the variance explained by the held-out category means; `fsp.metrics.iv_oof` and `target_encoded_eta_oof`.)

### 17.4 Cliff's δ (non-parametric effect, binary target vs continuous)
Proportion of non-overlap between the two groups' distributions. For groups X (size m) and Y (size n):
**δ = (1/(m·n)) · Σ_{i,j} sign(x_i − y_j)**, range −1..1; ties shrink |δ|.

| \|δ\| | Effect |
|---|---|
| < 0.147 | Negligible (**auto-drop backstop 0.07, one tier below**) |
| 0.147–0.33 | Small |
| 0.33–0.474 | Medium |
| ≥ 0.474 | Large |

### 17.5 Shadow-permutation floor
Permute the column (wiping the feature↔target relationship, preserving the marginal), recompute the *same* native metric, repeat B times (e.g. B ≥ 50; `fsp` default 50); the floor is a **high percentile of the shadow distribution** — start at the **95th** (Boruta's convention) and treat the percentile as an open tuning item. A feature must clear **its own** shadow floor, which automatically rises for high-cardinality noise columns. Use a fixed seed so the floor is reproducible. **The same shadow draws also yield a permutation p-value** for any metric that lacks an analytic one (§17.6).

### 17.6 Benjamini-Hochberg FDR
Across the m features tested in F, sort p-values ascending p₍₁₎ ≤ … ≤ p₍ₘ₎. The q-value at rank k is
**q₍ₖ₎ = min( 1, min_{j ≥ k} ( m · p₍ⱼ₎ / j ) )** — i.e. compute m·p₍ₖ₎/k, then enforce monotonicity from rank m down to 1. Report q per feature. Use it **with** (not instead of) effect size — significance without a clearing effect is still `review`, not `keep`. (`statsmodels.stats.multitest.multipletests(method="fdr_bh")`.)

**Every F metric must supply a p-value so it can get a q.** Where the metric has an analytic test, use it: **AUC → Mann–Whitney U** (§17.13, the exact companion test); Spearman/Kendall/point-biserial/Kruskal–Wallis carry their own; Cox its own. Where it does not — **IV, η², Cramér's V, Cliff's δ** — use a **one-sided permutation p** off the shadow draws (§17.5): **p = (1 + #{shadow ≥ effect}) / (B + 1)**. This closes the gap that would otherwise leave the primary binary metrics (AUC, IV) without a q-value.

### 17.7 Population Stability Index (drift)
PSI = Σ (%actual − %expected) · ln(%actual / %expected) across bins. < 0.10 stable · 0.10–0.25 moderate · **> 0.25 significant drift** → flag.

### 17.8 C-index (survival discrimination)
Fraction of comparable pairs correctly ranked; 0.5 = random, 0.7–0.8 good, > 0.8 excellent *for a whole model*. So a **single feature** at C-index > 0.75 is a screaming leak (§9). Auto-drop backstop 0.55. Biased upward above ~40% censoring — report censoring alongside. (`lifelines`.)

### 17.9 Correlation ratio η² (nominal/categorical → continuous)
Proportion of a continuous target's variance explained by group membership; captures **nonlinear** separation Pearson misses.
**η² = SS_between / SS_total = Σ_g n_g (ȳ_g − ȳ)² / Σ_i (y_i − ȳ)²**, range 0–1 (η = √η²). Use for a nominal feature against a regression target.

### 17.10 Kruskal–Wallis effect size (rank-based, robust)
From the Kruskal–Wallis H statistic over k groups, n total:
- **ε² = H / (n − 1)** — recommended (not bias-corrected).
- η²_H = (H − k + 1) / (n − k) — bias-corrected alternative.

Bands (both): 0.01–0.06 small · 0.06–0.14 moderate · ≥ 0.14 large. Use for nominal/multiclass associations on ranks (non-normal, outlier-robust). Report ε².

### 17.11 Point-biserial r_pb (binary feature → continuous target)
**r_pb = ((ȳ₁ − ȳ₀) / s_y) · √(p₁ · p₀)**, where p₁, p₀ are the group proportions and s_y the target's SD. Numerically identical to Pearson r with the binary coded 0/1, so `scipy.stats.pointbiserialr` (or Pearson on the 0/1 column) is exact.

### 17.12 Confidence intervals — one uniform method for every metric
Use a **bootstrap percentile** CI so every effect — AUC, IV, Cramér's V, Spearman, Cliff's δ, η², C-index — is comparable. Resample the usable training rows (feature **and** target present) with replacement **B = 1000** times, recompute the metric each time, and take the **2.5th and 97.5th percentiles** as the 95% CI. Fix the seed. (`fsp` uses B = 1000 for a standalone CI; its in-loop screening default is **200** for cost, exposed as a parameter.) This is the effect's own uncertainty; **fold spread** (§7 F) is the separate cross-fold stability signal — report both.

### 17.13 Single-feature AUC (and its p-value)
**AUC = P(x⁺ > x⁻)** over all positive/negative feature-value pairs — the rank concordance of feature `x` with a binary target; 0.5 = random, and we report it **orientation-free** as `max(a, 1−a)`. It equals the **Mann–Whitney U** statistic ÷ (m·n), so the Mann–Whitney U test gives its **exact p-value**. Auto-drop backstop 0.52 (§9); a single feature > 0.85 is a leak flag (§9, §12). This is the primary metric for continuous/count features vs a binary target (§8).

### 17.14 Rank / linear correlations (Spearman, Kendall, Pearson)
- **Spearman ρ** — Pearson correlation of the *ranks*; monotone-sensitive and outlier-robust. Reported with its t-based p; auto-drop 0.05.
- **Kendall τ** — (concordant − discordant) pairs ÷ total; **diagnostic only, never triggers an auto-drop** (§8 ordinal asymmetry).
- **Pearson r** — linear correlation; reported *alongside* Spearman for continuous↔continuous, never as the sole screen.

### 17.15 Variance Inflation Factor (VIF)
For numeric feature *j* regressed on the other kept numeric features, **VIF_j = 1 / (1 − R²_j)**. VIF > 10 ⇒ severe multicollinearity — **flag only, never a drop** (§9), and computed among the **kept numeric** features only. (`statsmodels`.)

### 17.16 Univariate Cox and the log-rank test (survival)
- **Univariate Cox** — a single-covariate proportional-hazards fit: report the hazard ratio **e^β**, its p (liberal screen at < 0.20, §9), and the model **C-index** (§17.8).
- **Log-rank** — compares survival curves across the levels of a categorical feature; report the statistic and p. (`lifelines`.)

---

## Sources

- Bergsma, *A bias-correction for Cramér's V and Tschuprow's T* — [stats.lse.ac.uk](https://stats.lse.ac.uk/bergsma/pdf/cramerV3.pdf); formula confirmed via [Cramér's V, Wikipedia](https://en.wikipedia.org/wiki/Cram%C3%A9r's_V)
- Lund, *Information Value Statistic and Predictors for Logistic Regression* (ASA 2014) — [amstat.org](https://ww2.amstat.org/meetings/proceedings/2014/data/assets/pdf/313981_87373.pdf)
- Boruta all-relevant selection / shadow features / 95th-percentile rule — [CRAN Boruta](https://cran.r-project.org/web/packages/Boruta/Boruta.pdf), [boruta_py process](https://deepwiki.com/scikit-learn-contrib/boruta_py/2.2-feature-selection-process)
- Cliff's δ thresholds & formula — [effsize, CRAN](https://cran.r-project.org/web/packages/effsize/effsize.pdf), [rcompanion cliffDelta](https://rdrr.io/cran/rcompanion/man/cliffDelta.html)
- Information Value bands (Siddiqi) — [listendata](https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html)
- Out-of-fold target encoding (leakage-safe) — [scikit-learn TargetEncoder](https://scikit-learn.org/stable/modules/preprocessing.html#target-encoder), [Hasz](https://brendanhasz.github.io/2019/03/04/target-encoding)
- Population Stability Index thresholds — [listendata](https://www.listendata.com/2015/05/population-stability-index.html)
- VIF rule of thumb & caution — [Springer, O'Brien 2007](https://link.springer.com/article/10.1007/s11135-006-9018-6)
- Liberal univariate Cox screening (p < 0.20–0.25) — [Lasso-Cox, jsurvival](https://www.serdarbalci.com/jsurvival/articles/09-lassocox-comprehensive.html)
- C-index interpretation — [scikit-survival](https://scikit-survival.readthedocs.io/en/stable/user_guide/evaluating-survival-models.html)
- Correlation ratio η² — [statology](https://www.statology.org/correlation-between-continuous-categorical-variables/)
- Kruskal–Wallis effect size (ε² / η²) — [rcompanion](https://rcompanion.org/handbook/F_08.html), [rstatix](https://rpkgs.datanovia.com/rstatix/reference/kruskal_effsize.html)
- Point-biserial correlation — [scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pointbiserialr.html)
- Benjamini–Hochberg FDR q-values — [statology](https://www.statology.org/benjamini-hochberg-procedure/)
- Bootstrap percentile confidence intervals — [Machine Learning Mastery](https://machinelearningmastery.com/calculate-bootstrap-confidence-intervals-machine-learning-results-python/)
- Events-per-variable rule (10 EPV, and relaxing it) — [Vittinghoff & McCulloch, AJE 2007](https://academic.oup.com/aje/article/165/6/710/63906)
- Pairwise / available-case deletion for associations — [Statistics Solutions](https://www.statisticssolutions.com/missing-data-listwise-vs-pairwise/)
- Mixed-type association method selection — [smartcor](https://arxiv.org/html/2607.22285)
- optbinning (IV/WoE/Gini) — [optbinning docs](https://gnpalencia.org/optbinning/)
