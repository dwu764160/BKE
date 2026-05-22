# BKE Master Improvement Plan

> **Last rewritten:** 2026-05-22. **Branch:** `personal`.
> **Purpose:** Single authoritative phase ordering. All other plan docs describe
> *what* to do within a phase; this doc governs *when* each phase runs and *why*.
>
> **Phase ordering is non-negotiable.** The minute model is deliberately late
> (after model optimization) because it depends on canonical positions, clean RAPM,
> and calibrated magnitudes. Do not pull it forward.

---

## Agreed Phase Sequence

```
Phase 0  Data ingestion sprint                  [COMPLETE]
  ↓
Phase 1  Pipeline audit (Tracks A/B/C)          [COMPLETE]
         + Phase 3A in parallel (foundation rules — no data needed)
  ↓
Phase 4  Archetype validation                   [COMPLETE]
  ↓
Phase 2  Minute model rebuild                   [COMPLETE — temporal Ridge, MAE=4.06, r²=0.671]
  ↓
Phase 3B Magnitude calibration                  [PENDING — needs 5+ clean seasons]
  ↓
Phase 5  Walk-forward harness                   [PENDING — measurement gate]
  ↓
         Model optimization (TEAM_SCALE, RAPM alpha, game outcome model)
```

---

## Phase 0 — Data Ingestion Sprint

**Status:** COMPLETE  
**Detail doc:** this file (inline)

### Phase 0 Checklist

| Task | Status | Notes |
|---|---|---|
| Tracking availability probe | DONE | All pre-2022 = rapm_only tier. See `docs/reference/tracking_availability.md` |
| Box score backfill 2017-22 | DONE | `data/historical/complete_player_season_stats_backfill.parquet` |
| PBP fetch + normalize all 8 seasons | DONE | `possessions_clean_{season}.parquet` for 2017-18 → 2024-25 |
| RAPM re-run (8-season pooled) | DONE | `player_rapm.parquet` — 20,506 rows, all 8 seasons |
| Impact profiles re-run | DONE | `player_impact_profiles.parquet` — 6,048 rows, all 8 seasons |
| BKE v28 decomposition re-run | DONE | `bke_v28_decomposition.parquet` — 5,848 rows, all 8 seasons |
| `player_profile_aggregate.parquet` rebuild | **DONE** | 5,848 rows, all 8 seasons (2026-05-22) |
| BKE v27 scores re-run | **DONE** | 8 seasons (re-scored on v28 expansion) |
| BKE v30 defense shrinkage re-run | **DONE** | 8 seasons (re-scored on v28 expansion) |
| `team_feature_aggregation.py` re-run | **DONE** | Validated on rebuilt aggregate |
| External RAPM benchmarks fetch | **DONE** | `reports/rapm_external_validation.json` |

---

## Phase 1 — Pipeline Audit + Phase 3A Foundation Rules (in parallel)

**Status:** COMPLETE (2026-05-21)
**Prerequisites:** Phase 0 complete  
**Detail docs:** `docs/plans/data_pipeline_audit_plan.md`, `docs/plans/basketball_intuition_plan.md`

### Pipeline Audit (Three independent tracks, run in parallel)

| Track | Problem | Root Cause | Key Files | Status |
|---|---|---|---|---|
| A | RAPM over-regularized; elite players suppressed by -4.46 pts/100 | Alpha=200 selected by RidgeCV; slope 0.52-0.59 is acceptable for public RAPM | `src/modeling/model_rapm.py` | VALIDATED (no code change needed) |
| B | BPM hand-tuned on 2-3 players; pre-2022 BPM was NaN | 3 hardcoded constants + Step 7c removed; pre-2022 AST=0 (PBP lacks assistPersonId) | `src/data_compute/compute_linear_metrics.py` | FIXED — BPM 99-100% filled all 8 seasons |
| C | Traded players use wrong team's box stats; lineup coverage 50-78% | derive_lineups.py reconstruction quality; no TOT rows in source | `src/data_normalize/`, `complete_player_season_stats.parquet` | DOCUMENTED (GAP-011) — deep fix deferred |

Track A requires external RAPM benchmarks from Phase 0. Tracks B and C are pure formula fixes (no external data).

**Track B details:**
- Restored 4 B-REF formula constants (INDIVIDUAL_DEDUCTION_SCALE, MIN_REGRESSION_THRESHOLD, MIN_REGRESSION_STRENGTH, VERY_LOW_MIN_PENALTY)
- Removed Step 7c "Efficiency-Based Adjustments" (player-specific hacks for Ivey/Clarkson/Allen)
- Fixed pre-2022 AST=0 issue: fill AST from official box stats (coverage-scaled by PBP/box minute ratio)
- `load_player_team_mapping()` now also reads `complete_player_season_stats.parquet` for pre-2022 seasons

**Track C remaining work (deferred):**
- Fix `derive_lineups.py` lineup reconstruction → re-run RAPM (~3-4hr) — captured in GAP-011

**Validation:** `reports/rapm_external_validation.json` (Track A), rebuilt `aggregate/player_profile_aggregate.parquet` (Track B+C)

### Phase 3A — Foundation Rules (no data needed, runs in parallel with audit)

From `docs/plans/basketball_intuition_plan.md`:
- **A1:** Canonical `position_band_5` / `position_band_3` columns in `compute_position_estimate.py` → propagated to profiles + aggregate — *deferred to Phase 2 prerequisites*
- **A2:** B2B/rest penalty fitted from actual game logs — **DONE**: refitted on 6 non-COVID seasons (was 3), `reports/rest_hca_coefficients.json`
- **A3:** HCA calibrated per-team from game logs — **DONE**: COVID seasons excluded, league_avg_hca=2.147, DEN highest (altitude)

These are structural fixes that downstream everything (minute model, simulation). Complete them here so Phase 2 has clean inputs.

---

## Phase 4 — Archetype Validation

**Status:** COMPLETE (2026-05-22)  
**Prerequisites:** Phase 1 complete (clean pipeline output) ✓  
**Detail doc:** `docs/plans/archetype_validation_plan.md`

### Sub-task Checklist

| Sub-task | Status | Notes |
|---|---|---|
| 4.0 Archetype backfill (all 8 seasons) | **DONE** | `compute_player_archetypes.py` + `compute_defensive_archetypes_v2.py` run on all 8 seasons; 3,394 classified rows |
| 4.1 Minutes threshold → 200 min / 10 GP / 8 MPG | **DONE** | Lowered in both archetype scripts; 2022-23 Insufficient dropped from 43% → 22% |
| 4.2 Secondary archetype documentation | **DONE** | Offensive tags finalized; secondary_archetype column added to aggregate |
| 4.3 Validation tracks (stability, sensitivity, coherence, manual) | **DONE** | reports/archetype_stability.json, sensitivity.json, coherence.json, manual_sample.csv |
| 4.4 Player tier system | **DONE** | BKE-anchored within-season percentile; new column `player_tier` in aggregate |
| 4.5 Soft probability adoption | **DONE** | Design complete; pec_off_prob_emb_* columns available for Phase 2 |
| 4.6 Archetype pair validation + OLS matrix | **DONE** | 0 pairs survive lineup Lasso fit; INTERACTION_MATRIX eliminated (use_interaction=False) |

**Key fixes applied:**
- Pre-2022 box score format detection (per-game MIN → season total via ×GP)
- Missing synergy/tracking columns pre-filled with 0.0 for pre-2022 seasons
- PLAYER_ID type normalization (int64 vs object merge fix)
- BKE spine coalesce: fresh archetype values now preferred over stale "Insufficient Minutes" labels from old 500-min threshold
- **Deviation Audit:** Eliminated `INTERACTION_MATRIX` in 4.6 after empirical lineup-level Lasso fit on 4,914 stints (8 seasons) showed zero significant effects above talent control.

---

## Phase 2 — Minute Model Rebuild

**Status:** COMPLETE (2026-05-22)  
**Prerequisites:** Phase 1 (clean RAPM + canonical positions from 3A) + Phase 4 (stable archetypes) ✓  
**Detail doc:** `docs/plans/minute_model_rebuild_plan.md`  
**Report:** `reports/minute_model_rebuild_report_2026-05-22.md`

**Why it's here and not earlier:**
- Minute model uses archetype probability embeddings → needs stable archetypes (Phase 4)
- Minute model uses RAPM/BKE features → needs clean RAPM (Phase 1 Track A)
- Minute model uses `position_band` → needs canonical positions (Phase 3A)
- Deploying a better minute model before fixing its inputs produces a better-tuned version of a wrong model

**What was built:**
- `train_minute_model.py`: Complete rewrite — temporal season-N → N+1 Ridge pipeline (StandardScaler + RidgeCV, GridSearch alpha selection, GroupKFold by season), rookie lookup table from empirical draft-position buckets, team normalization to 240-minute constraint
- `project_next_season.py`: Stripped all manual minute modifiers (age curves, impact multipliers, salary bumps, carry anchors); `project_minutes` now loads pre-computed Ridge predictions for returning players, falls back to rookie lookup for new entrants
- 16 features (prior MPG, BKE, ORAPM, DRAPM, age, salary, usage, 3PT rate, AST rate, draft position, experience, team BKE rank, GP fraction, 3 archetype embeddings)
- Selected alpha: 100.0 (strong regularization)

**Validated holdout (2023-24 → 2024-25):**
| Metric | Old Manual | New Ridge | 
|---|---|---|
| Correlation | NaN (scaling failure) | r²=0.671 (r≈0.82) |
| MAE | 8.13 MPG | **4.06 MPG** |

---

## Phase 3B — Magnitude Calibration

**Status:** PENDING  
**Prerequisites:** Phase 1 complete + ≥5 clean seasons (Phase 0 + Phase 1 expands this)  
**Detail doc:** Part of `docs/plans/basketball_intuition_plan.md` (Phase B section)

**What gets calibrated empirically:**
- Interaction matrix (121 archetype-pair values) → OLS regression against team offensive efficiency
- Modifier magnitudes (B2B penalty, HCA, scheme amplifiers) → regression against game outcomes
- BKE dimension weights → recalibrated against external RAPM benchmark

**Why it can't run earlier:** Hand-tuned magnitudes must be replaced with regression fits. Regression
needs ≥5 clean season transitions and clean RAPM as ground truth. Both prerequisites land here.

---

## Phase 5 — Walk-Forward Harness (Measurement Gate)

**Status:** Infrastructure exists; re-run needed after each phase  
**Script:** `src/simulation/validate_forecast.py`  
**Output:** `reports/forecast_game_validation.json`  
**Detail doc:** `docs/plans/walk_forward_harness_plan.md`

**Current baseline (5 clean transitions, 2017-25):**
| Metric | Value |
|---|---|
| Brier | **0.2396** |
| Transitions | 5 clean (2 COVID-flagged) |

**Role:** After Phases 1-3B complete, re-run the harness to measure aggregate Brier improvement. This is
the only valid measure of whether a phase was worth applying. A phase that doesn't move Brier is not worth merging.

**Vegas target:** ~0.195-0.210. Gap from current 0.2396 is the entire improvement budget for Phases 1-5.

---

## Model Optimization (Final Step)

**Prerequisites:** Phase 5 harness establishes new Brier baseline post-cleanup  
**Targets:**
- TEAM_SCALE: ~25 implied vs. 20.0 current (calibration scripts ready in `src/player_eval/`)
- RAPM alpha / PRIOR_REFINEMENT_STRENGTH: grid search [10, 20, 30, 50, 80]
- SEASON_DECAY_WEIGHTS: cross-validation vs. walk-forward Brier
- Game outcome model structure (if Brier gap remains large)

**Apply only when:** 5+ clean walk-forward transitions confirm the improvement is not noise.

---

## BKE Version Validation Plan

**When to run:** After Phase 1 (clean RAPM) + Phase 3B (calibrated magnitudes) complete.

**Comparison set:**
| Version | What it is | Current seasons |
|---|---|---|
| v28 + v27 | Production baseline (4-layer decomp + OBKE/DBKE scoring) | 3 seasons (will be 8 after Phase 0 finishes) |
| v28 + v30 | Basketball intuition defense (shrinkage/geometry/balance) | 3 seasons (will be 8 after Phase 0 finishes) |
| v28 + v31 | Experimental layer weighting (55/45, 60/40 OBKE/DBKE) | JSON only |

**Method:** Feed each version's output through `team_feature_aggregation.py` → `validate_forecast.py`.
The version with the lowest Brier on 5+ out-of-sample transitions wins and becomes the new production version.

**Do not validate on corrupted RAPM.** Comparing versions before Phase 1 Track A is fixed measures
which version best compensates for a broken input, not which version has better basketball intuition.

---

## Dependency Graph

```
Phase 0 (data ingestion) ──────────────────────────────────────────────────────┐
  ├── possessions + RAPM + v28 decomp: DONE                                    │
  └── aggregate rebuild + v27/v30 re-run + external benchmarks: PENDING        │
                                                                                ▼
Phase 1 (pipeline audit A/B/C) + Phase 3A (foundation rules) ─────────────────┐
  ├── Track A: RAPM calibration (needs external benchmarks from Phase 0)       │
  ├── Track B: BPM formula fix (independent)                                   │
  ├── Track C: traded player dedup (independent)                               │
  └── 3A: canonical positions + B2B/HCA fitting (independent)                 │
                                                                                ▼
Phase 4 (archetype validation) ────────────────────────────────────────────────┐
  └── validation only, no code changes unless stability fails                  │
                                                                                ▼
Phase 2 (minute model rebuild) ────────────────────────────────────────────────┐
  └── needs clean RAPM (1A) + canonical positions (3A) + stable archetypes (4) │
                                                                                ▼
Phase 3B (magnitude calibration) ──────────────────────────────────────────────┐
  └── needs ≥5 clean seasons + clean RAPM as regression target                 │
                                                                                ▼
Phase 5 (walk-forward harness) ── Brier measurement after all cleanup ─────────┐
                                                                                ▼
Model optimization (TEAM_SCALE, alpha, decay weights)
```

---

## Centralized Season Config

```python
# src/modeling/model_config.py — current state
SEASONS = ["2017-18", "2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]
COVID_SEASONS = ["2019-20", "2020-21"]
SEASONS_GAME_MODEL = [s for s in SEASONS if s not in set(COVID_SEASONS)]
SEASONS_RAPM = SEASONS
```

All scripts import from `model_config.py` — never hardcode season lists elsewhere.
