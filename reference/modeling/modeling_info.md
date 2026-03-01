# Modeling Info Guide (RAPM + DARKO + BKE)

This is a practical, plain-language guide to how our advanced modeling layer works, why each layer exists, and how it translates to basketball decisions.

---

## 1) What each model is trying to answer

### RAPM (Regularized Adjusted Plus-Minus)
- **Basketball question:** “When this player is on the court, how does scoring margin change after accounting for teammates and opponents?”
- **What it measures:** Impact in points per 100 possessions.
- **Why it matters:** Box score stats can miss off-ball value (screening, spacing gravity, rotations, deterrence). RAPM captures lineup-level effect directly from possessions.

### DARKO
- **Basketball question:** “What is this player’s current true talent level, and how is it changing over time?”
- **What it measures:** A daily-updated talent estimate using decayed history + contextual adjustment.
- **Why it matters:** It is stronger for forward-looking signal (who is rising/falling), while RAPM is stronger for on-court lineup impact evidence.

### BKE (our custom model)
- **Basketball question:** “Given all evidence (lineup impact, projection, box/role stats, and context), what is the most reliable, role-aware, and future-facing value estimate for a player?”
- **What it measures:**
  - `BKE_O` (offensive impact)
  - `BKE_D` (defensive impact)
  - `BKE` (net impact)
  - uncertainty/confidence
- **Why it matters:** BKE is a modern, EPM-inspired, prior-driven model that fuses independent signals: lineup impact (RAPM), predictive projection (DARKO), and a strong prior built from archetype, box, and tracking stats. It is not just a blend, but a model that can disagree with its inputs and is robust to overfitting. Archetype and context features are used to calibrate and stratify value, not to define it directly.

---

## 2) The layer-by-layer construction (easy mental model)

Think of this as building a coaching report in 5 levels.

## Layer 0 — Data foundation (what happened on the court)
- Inputs:
  - Possession data + inferred lineups
  - Box score + tracking + matchup context
  - External model feed (DARKO)
- Basketball meaning:
  - We know who shared the floor, who the opponent was, and what scoring outcome happened each possession.

## Layer 1 — Rate stats and context normalization
- We convert raw counts to comparable rates (per 100 possessions, percentages, z-scores).
- Basketball meaning:
  - A bench guard in 1,200 possessions and a starter in 5,500 possessions can be compared fairly, while still tracking confidence separately.

## Layer 2 — Role/behavior layer (archetypes)
- Archetypes are behavior-only labels (creator, movement shooter, versatile defender, etc.).
- Basketball meaning:
  - This answers **what role** a player performs, not how valuable they are.

## Layer 3 — Impact models (RAPM, DARKO)
- RAPM/xRAPM estimate lineup impact from possession outcomes.
- DARKO contributes daily talent projections.
- Basketball meaning:
  - RAPM: “What has your impact looked like in real lineups?”
  - DARKO: “What should we expect going forward?”

## Layer 4 — BKE fusion layer (our final value model)
- Build a prior for each player using archetype, box, and tracking stats (role, usage, efficiency, context), independent of RAPM/DARKO.
- Fuse this prior with RAPM and DARKO using a Bayesian or ensemble model (not just a weighted average), allowing BKE to disagree with its inputs when evidence supports it.
- Calibrate and validate BKE by archetype cohort and context, ensuring no role inflation or circularity.
- Basketball meaning:
  - BKE provides a robust, role-aware, and future-facing value estimate, supporting better decision-making for roster fit, player ranking, and role-specific comparison, while remaining independent and complementary to RAPM and DARKO.

---

## 3) How RAPM is built (step by step)

## Step A: Build a possession matrix
- Each row = one possession.
- Each player gets a column.
- Typical net RAPM encoding:
  - +1 if on offense
  - -1 if on defense
- Target = points scored on that possession.

**Basketball context:**
If a lineup repeatedly outscores opponents across many possessions, the model spreads credit/debit across the 10 players while accounting for co-linearity.

## Step B: Fit ridge regression
- We use **ridge** to stabilize estimates when players share many minutes with the same teammates.
- Without ridge, estimates explode for players who are hard to separate.

**Basketball context:**
Two wings playing 90% of their minutes together can look interchangeable; regularization prevents unrealistic “one +8, one -6” outcomes from small noise.

## Step C: Produce split estimates (ORAPM / DRAPM)
- Correct method uses **opponent-controlled joint design**:
  - offensive player indicators and defensive player indicators in one model.
- `ORAPM` from offensive coefficients.
- `DRAPM` from defensive coefficients with sign flip (positive = better defense).

**Basketball context:**
This avoids over-crediting offense-only stars or under-crediting defenders just because of teammate/opponent mix.

## Step D: Multi-season pooling and target-season refinement
- We pool nearby seasons with decay weights for stability.
- Then refine back to target season using pooled coefficients as prior signal.

**Basketball context:**
Veterans keep stability from multi-year evidence, but we still allow this season’s role/health/team context to move the estimate.

---

## 4) How DARKO fits into our pipeline

We treat DARKO as **external canonical input**, not something we re-create from scratch.

Pipeline path:
1. Manual export from DARKO source (CSV)
2. Local fetch-stage intake (file staging)
3. Normalize to canonical columns (`player_id`, `season`, `darko_dpm`, `darko_odpm`, `darko_ddpm`)
4. Ingest into modeling input table

**Basketball context:**
DARKO gives us a daily-updating expectation for talent trend and regression-to-mean behavior, which is especially useful for noisy stretches (hot/cold shooting runs).

---


## 5) How BKE is computed (modern approach)

BKE is built as a modern, EPM-style, prior-driven model:

1. **Build a strong prior:**
  - Use archetype, box score, and tracking stats (role, usage, efficiency, context) to predict player impact, independent of RAPM/DARKO.
  - This prior is the model’s best estimate before seeing on/off or projection data.
2. **Fuse with RAPM and DARKO:**
  - Use a Bayesian regression or ensemble model to combine the prior, RAPM, and DARKO.
  - The model can “disagree” with any input if the data supports it, avoiding overfitting or circularity.
3. **Calibrate by archetype/context:**
  - After fitting, calibrate BKE by archetype cohort and context to ensure no role inflation and realistic value distribution.
4. **Estimate uncertainty:**
  - Quantify confidence based on sample size, model disagreement, and role stability.

**Conceptual formula:**

`BKE_O = f(Prior_O, ORAPM, DARKO_ODPM, OBPM/OWS, archetype, usage, efficiency, context, reliability)`
`BKE_D = f(Prior_D, DRAPM, DARKO_DDPM, DBPM/DWS, archetype, matchup, context, reliability)`
`BKE = BKE_O + BKE_D`
`BKE_uncertainty = g(sample_size, model_disagreement, role_stability)`

### Why this is strong
- BKE is robust to overfitting and can add value beyond RAPM and DARKO.
- Archetype and context features allow for role-aware calibration without mixing role and value.
- The model is explainable, modular, and future-facing.

**Basketball context example:**
- Player A: strong RAPM, weak DARKO, but prior (archetype/box/tracking) suggests regression—BKE can moderate the estimate.
- Player B: moderate RAPM, strong DARKO, stable prior—BKE can reflect sustainable value.

---

## 6) Statistics concepts you need (minimal, practical set)

## Core regression ideas
- **Linear regression:** finds weighted relationships between features and target.
- **Ridge regularization (L2):** shrinks unstable coefficients to reduce overfitting.
- **Cross-validation:** tests hyperparameters (like alpha) on held-out slices.

Basketball translation:
- We avoid being fooled by lineup noise and tiny samples.

## Signal quality concepts
- **Bias vs variance:**
  - high variance = noisy player estimates
  - high bias = over-smoothed, missing true differences
- **Shrinkage:** pull extreme small-sample estimates toward center.
- **Reliability weighting:** larger possession/minute samples get more trust.

Basketball translation:
- 500-possession outliers should not outrank 6,000-possession stars without strong evidence.

## Distribution / scaling concepts
- **Per-100 possession scaling:** apples-to-apples across pace/minutes.
- **Standardization (z-score/robust z):** align features to common scale.
- **Calibration:** align predicted values with observed outcomes across cohorts.

Basketball translation:
- A high-pace team player is not automatically overrated versus low-pace contexts.

## Uncertainty concepts
- **Confidence interval / uncertainty band:** how precise estimate is.
- **Model disagreement:** if RAPM and DARKO disagree strongly, confidence should decrease.

Basketball translation:
- “High upside, low confidence” and “stable impact” are different profile types.

---

## 7) Role vs value (non-negotiable design rule)

- Archetype = **what the player does** (role/behavior, assigned independently).
- BKE/RAPM/DARKO = **how valuable performance is** (impact/value, estimated after role assignment).

Why this matters:
- If we mix role and value too early, we risk inflating certain archetypes and create circular logic.
- Proper order:
  1. Assign role by behavior (archetype logic, independent of value).
  2. Build a prior for value using archetype, box, and tracking stats.
  3. Fuse with impact/projection models (RAPM, DARKO) to estimate value.

Basketball example:
- A “POA Defender” can be elite, average, or replacement-level. The role label informs the prior, but does not force a value score by itself.

---

## 8) Validation checklist we should run every cycle

1. Distribution sanity (means/spreads realistic)
2. Season-to-season stability by possession tiers
3. Rank agreement/correlation between RAPM, DARKO, BKE
4. Archetype-cohort calibration (no role inflation)
5. Outlier review with uncertainty flags
6. Top/bottom player film sanity checks

Basketball context:
- If model top-10 is full of low-minute noise or impossible defenders, we are overfitting.

---

## 9) Practical interpretation guide for coaches/front office

- Use `BKE` for overall impact ranking.
- Use `BKE_O` and `BKE_D` for role-specific lineup fit.
- Use uncertainty to separate “strong conviction” from “speculative upside.”
- Use archetypes to frame *how* value is created, not *whether* value exists.

---

## 10) What this means for BKE project direction right now

- RAPM remains an in-house core signal.
- DARKO enters as external canonical projection feed.
- BKE becomes the fused decision model with explicit confidence.
- The architecture remains modular, explainable, and basketball-first.
