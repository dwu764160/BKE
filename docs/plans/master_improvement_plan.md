# BKE Master Improvement Plan

> **Last rewritten:** 2026-05-21. **Branch:** `personal`.
> **Purpose:** Single authoritative phase ordering. All other plan docs describe
> *what* to do within a phase; this doc governs *when* each phase runs and *why*.
>
> **Phase ordering is non-negotiable.** The minute model is deliberately late
> (after model optimization) because it depends on canonical positions, clean RAPM,
> and calibrated magnitudes. Do not pull it forward.

---

## Agreed Phase Sequence

```
Phase 0  Data ingestion sprint                  [PARTIALLY COMPLETE — see below]
  ↓
Phase 1  Pipeline audit (Tracks A/B/C)          [PENDING]
         + Phase 3A in parallel (foundation rules — no data needed)
  ↓
Phase 4  Archetype validation                   [PENDING — uses clean pipeline output]
  ↓
Phase 2  Minute model rebuild                   [PENDING — needs canonical positions + clean RAPM]
  ↓
Phase 3B Magnitude calibration                  [PENDING — needs 5+ clean seasons]
  ↓
Phase 5  Walk-forward harness                   [PENDING — measurement gate]
  ↓
         Model optimization (TEAM_SCALE, RAPM alpha, game outcome model)
```

---

## Phase 0 — Data Ingestion Sprint

**Status:** PARTIALLY COMPLETE  
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
| `player_profile_aggregate.parquet` rebuild | **NOT DONE** | Still 3 seasons only (1,971 rows) |
| BKE v27 scores re-run | **NOT DONE** | Still 3 seasons only (950 players) |
| BKE v30 defense shrinkage re-run | **NOT DONE** | Still 3 seasons only |
| `team_feature_aggregation.py` re-run | **NOT DONE** | Needs rebuilt aggregate as input |
| External RAPM benchmarks fetch | **NOT DONE** | ESPN RPM or nbarapm.com — no script exists yet |

**What's left (all fast — RAPM/PBP already done):**
1. Rebuild player profile aggregate (all 8 seasons)
2. Re-score BKE v27 on expanded v28 decomposition
3. Re-run BKE v30 defense shrinkage on expanded v28 decomposition
4. Re-run team feature aggregation
5. Write + run external RAPM benchmark fetch (ESPN RPM / nbarapm.com, 2022-25)

**ETA for remaining tasks:** ~60-90 min total, can be chained in one shell and left unattended.

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

**Status:** PENDING  
**Prerequisites:** Phase 1 complete (clean pipeline output)  
**Detail doc:** `docs/plans/archetype_validation_plan.md`

**Three tracks (run in parallel, validation only — no production code changes):**
- Track 1: Year-over-year stability (transition matrix across consecutive seasons)
- Track 2: Threshold sensitivity analysis (how much does swapping check order change assignments)
- Track 3: External archetype comparison (cross-validate against basketball-reference role labels)

Output: stability report + safeguard recommendations. If archetypes are noisy, fixes go in before Phase 2.

---

## Phase 2 — Minute Model Rebuild

**Status:** PENDING  
**Prerequisites:** Phase 1 (clean RAPM + canonical positions from 3A) + Phase 4 (stable archetypes)  
**Detail doc:** `docs/plans/minute_model_rebuild_plan.md`

**Why it's here and not earlier:**
- Minute model uses archetype probability embeddings → needs stable archetypes (Phase 4)
- Minute model uses RAPM/BKE features → needs clean RAPM (Phase 1 Track A)
- Minute model uses `position_band` → needs canonical positions (Phase 3A)
- Deploying a better minute model before fixing its inputs produces a better-tuned version of a wrong model

**Fix summary:** Rebuild temporal (season-N features → season-N+1 MPG), Ridge Regression, rookie lookup table. Replace `project_next_season.py` manual minute logic with model output.

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
