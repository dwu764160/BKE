# Modelling Stats Plan (RAPM + DARKO + BKE)

## Purpose

Define the advanced player-impact modeling layer of BKE using three core model metrics:

1. RAPM (and xRAPM family)
2. DARKO
3. BKE (custom model metric)

This layer is the modeling foundation of player evaluation and sits on top of the existing linear metrics and archetype system.

---

## 1) Source Strategy: Compute vs Fetch

### RAPM — **Compute in-house (primary)**

**Decision:** Keep RAPM/xRAPM fully computed in our pipeline.

**Why:**
- We already own possession-level inputs and lineup inference.
- RAPM is not available as a stable free API with our exact desired variants.
- In-house compute lets us control priors, season pooling, regularization, and diagnostics.

**Implementation home:**
- `src/modeling/model_rapm.py`

### DARKO — **Fetch canonical values + optional local proxy**

**Decision:** Fetch official DARKO values as an external benchmark input, not re-create DARKO as our primary canonical version.

**Why:**
- DARKO is a proprietary, continuously tuned, daily model with no stable public API.
- Public usage pattern is manual CSV export from the official app.
- Re-implementing full DARKO would be high complexity and low reproducibility confidence versus using official outputs.

**Operational plan:**
- Add `src/data_fetch/fetch_darko_manual.py` (or equivalent) that ingests exported DARKO CSV files into `data/historical/darko/`.
- Version by file date (`DARKO_player_talent_YYYY-MM-DD.csv`) and normalize to `player_id`, `season`, `darko_dpm`, `darko_odpm`, `darko_ddpm` where available.
- Add freshness checks (max age threshold) and schema validation in the validation stage.

**Fallback:**
- If DARKO file missing/stale, run with RAPM+BKE-only and label missing-source flags.

---

## 2) BKE Metric Design (Custom)

## BKE definition

BKE is a modern, EPM-inspired, predictive, uncertainty-aware impact model that fuses:

- A strong prior built from archetype, box score, and tracking stats (role, usage, efficiency, context)
- RAPM/xRAPM signal (lineup impact)
- DARKO signal (external predictive projection)
- Linear metrics (BPM/WS/VORP and local box-score rates)
- Context controls (minutes, role stability, team context)

Key principles:
- BKE is not just a blend of RAPM and DARKO; it is a model that can disagree with its inputs and is robust to overfitting.
- Archetype and context features are used to build the prior and calibrate value, not to define value directly.
- Strict separation: role assignment logic (archetype) is independent of value estimation (BKE).

## BKE output family

- `BKE_O`: offensive impact per 100 (role/context-calibrated)
- `BKE_D`: defensive impact per 100 (role/context-calibrated)
- `BKE`: net impact per 100 (`BKE_O + BKE_D`)
- `BKE_uncertainty`: posterior uncertainty / confidence interval width
- `BKE_tier`: percentile-based tiering for evaluation UX

---

## 3) Modeling Architecture

## Stage A — Signal standardization

- Align all signals to season-normalized z-scores (or robust z-scores).
- Build a strong prior for each player using archetype, box, and tracking stats (role, usage, efficiency, context), independent of RAPM/DARKO.
- Reliability weighting by possessions/minutes/sample size.
- Separate offense and defense channels.

## Stage B — Ensemble core (EPM-style fusion)

Use a two-level model:

1. **Prior model:**
  - Predict player impact using archetype, box, and tracking stats (role, usage, efficiency, context), independent of RAPM/DARKO.
  - This prior is the model’s best estimate before seeing on/off or projection data.
2. **Fusion model:**
  - Use a Bayesian regression or ensemble (e.g., stacking, gradient boosting) to combine the prior, RAPM, and DARKO.
  - The model can “disagree” with any input if the data supports it, avoiding overfitting or circularity.
  - Produces `BKE_O`, `BKE_D` with uncertainty.

Target examples:
- Forward-looking: next-season point differential contribution per 100 (preferred)
- Backward-looking sanity target: stabilized multi-year RAPM proxy

## Stage C — Archetype/context-aware calibration

- Calibrate residual bias by archetype cohort (same role family, minutes tier, season) and context features.
- This is post-model calibration only; archetype does not define value directly, but is used to stratify and validate value estimates.
- Prevent role-value leakage by excluding archetype labels from initial supervised target construction, but use them for prior building and post-hoc calibration.

---

## 4) Integration With Existing System

## Linear metrics integration

Linear metrics (`compute_linear_metrics.py`) remain transparent interpretable features and diagnostics:

- BPM/OBPM/DBPM
- WS/OWS/DWS
- VORP

They are not replaced; they are absorbed into BKE ensemble inputs and audit reports.

## Archetype integration

- Archetypes remain the role ontology and UX explanation layer (role identity, confidence, effectiveness).
- Archetype and context features are used to build the prior for BKE and to calibrate/validate value estimates by cohort.
- BKE provides the impact layer (how much value in that role), independent of the role assignment itself.

## Evaluation system foundation

Final player evaluation object (per season/player) should include:

- Role block: offensive + defensive archetypes
- Linear block: BPM/WS/VORP + box/tracking summaries
- Modeling block: RAPM, DARKO, BKE (+ uncertainty)
- Composite dashboard fields for ranking and comparison

---

## 5) Build Plan (Phased)

## Phase M1 — Data and interfaces

- Finalize RAPM paths under `src/modeling/` (done)
- Add DARKO ingest script + schema validator
- Add merged modeling feature table output:
  - `data/processed/modeling_inputs_{season}.parquet`

## Phase M2 — BKE v1 (interpretable baseline)

- Train ridge/elastic-net blend for `BKE_O` and `BKE_D`
- Reliability weights by possessions + minutes
- Save:
  - `data/processed/player_bke_v1.parquet`
  - `data/processed/player_bke_v1.csv`

## Phase M3 — BKE v2 (stacked predictive)

- Add gradient boosting base model
- Add stacking/meta-learner and uncertainty estimator
- Add archetype-cohort calibration report

## Phase M4 — System integration

- Merge BKE outputs into player evaluation dataset
- Wire into viewers and downstream ranking scripts
- Add per-season model cards in `reference/`

---

## 6) Validation and Governance

Required checks per run:

- Distribution sanity (mean near 0, realistic spread)
- Season-to-season stability vs possessions thresholds
- Correlation and rank-agreement with RAPM and DARKO
- Calibration by archetype cohort and minutes buckets
- Outlier diagnostics with uncertainty flags

Validation artifacts:

- `data/processed/modeling_validation_report.json`
- `data/processed/modeling_validation_summary.txt`

---

## 7) Immediate Repository Actions

1. Keep RAPM/xRAPM in `src/modeling/` as canonical modeling scripts.
2. Treat DARKO as fetched canonical external input.
3. Build BKE as the internal fused model metric using RAPM + DARKO + linear/context signals.
4. Maintain strict separation: role assignment logic stays in archetype scripts; impact estimation stays in modeling scripts.

---

## 8) Deliverables Checklist

- [ ] DARKO ingest script + schema checks
- [ ] Modeling input table (`modeling_inputs_{season}.parquet`)
- [ ] BKE v1 outputs (`player_bke_v1.parquet/.csv`)
- [ ] Validation report (`modeling_validation_report.json`)
- [ ] Viewer integration for RAPM/DARKO/BKE triplet

---

## 9) Companion Reading

- `reference/modeling_info.md` — plain-language walkthrough of model layers, stats concepts, and basketball interpretation.
