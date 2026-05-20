# BKE Master Improvement Plan

> **Created:** 2026-05-20. **Branch:** `personal`.
> **Purpose:** Single source of truth for all active improvement initiatives.
> Each phase has its own detailed plan doc linked below.
> **Primary constraint:** Walk-forward Brier (Phase 0 output) is the gate metric
> for all other phases — a phase that doesn't move Brier on out-of-sample data
> is not worth applying.

---

## Plan Overview

```
Phase 0  Walk-Forward Harness          [COMPLETE]  → baseline Brier established
Phase 1  Minute Model Rebuild          [PENDING]   → blocked only by time
Phase 2  Data Backfill                 [IN PROGRESS]
Phase 3  Low-Sample Player Prior       [PENDING]   → blocked by Phase 2 PBP
Phase 4  Hyperparameter Calibration    [DEFERRED]  → needs 5+ transitions
Phase 5  Pipeline Correctness Fixes    [PENDING]   → independent of above
Phase 6  Basketball Intuition Wiring   [PENDING]   → gates on Phase 5 data
Phase 7  Archetype Validation          [PENDING]   → independent track
```

---

## Phase 0 — Walk-Forward Game-Level Harness

**Status:** COMPLETE  
**Script:** `src/simulation/validate_forecast.py`  
**Output:** `reports/forecast_game_validation.json`  
**Detail doc:** `docs/plans/walk_forward_harness_plan.md`

**Baseline (2 transitions, 2455 games):**
| Metric | Value |
|---|---|
| Brier | **0.2320** |
| Log Loss | 0.6561 |
| Accuracy | 61.5% |
| Calibration Error | 0.0779 |
| Margin RMSE | ~14.9 pts |

**What this number means:** 0.2320 is genuinely out-of-sample (prior-season projected
ratings vs. actual next-season games). A coin flip scores 0.25; naive home-team-wins
prior scores ~0.247. We beat the no-information baseline. Vegas closing lines score
~0.195–0.210. The gap between us and Vegas is the target for all future phases.

**Baseline transition detail:**
| Season | Brier | Accuracy | Notes |
|---|---|---|---|
| 2023-24 | 0.2267 | 62.2% | Projected from 2022-23 profiles |
| 2024-25 | 0.2374 | 60.8% | Projected from 2023-24 profiles |

---

## Phase 1 — Minute Model Temporal Rebuild

**Status:** PENDING (not started)  
**Detail doc:** `docs/plans/minute_model_rebuild_plan.md`  
**Prerequisite:** None — can start immediately

**Root problem:** Current minute model trains on same-season features → same-season MPG.
Top feature (`pec_defensive_shrinkage_lambda`, importance=0.476) is literally a function
of possessions played *this* season. All top features are same-season values. In forecast
mode, prior-season values are used → severe distribution mismatch between training and deployment.

**Fix:** Rebuild temporal (season-N features → season-N+1 MPG), simplify to Ridge Regression,
add rookie lookup table. Replace `project_next_season.py` manual minute logic with model output.

**Expected signal:** Holdout MPG MAE improvement; downstream — reduced team-rating noise
because minutes are distributed more realistically.

---

## Phase 2 — Data Backfill

**Status:** IN PROGRESS

### Phase 2a — Tracking Probe
**Status:** COMPLETE  
**Finding:** All pre-2022 seasons = **rapm_only** tier. `leaguedashptstats` returns 0
players for 2017-18 through 2021-22 via any HTTP method. No drives, synergy, or touch
data available for those seasons. Only box score + RAPM backbone usable.  
**Output:** `docs/reference/tracking_availability.md`

### Phase 2b — Box Score + Game Log Backfill
**Status:** COMPLETE  
**Seasons:** 2017-18, 2018-19, 2019-20, 2020-21, 2021-22  
**Outputs:**
- `data/historical/complete_player_season_stats_backfill.parquet` (2,744 rows, 5 seasons)
- `data/official_stats/official_advanced_{season}.parquet` per season
- `data/historical/team_game_logs.parquet` expanded (19,038 rows, 8 seasons)

### Phase 2c — PBP Fetch
**Status:** IN PROGRESS (running overnight 2026-05-20)  
**Method:** `src/data_fetch/fetch_pbp/fetch_pbp_statsapi.py` via `stats.nba.com/stats/playbyplayv3`
+ curl_cffi impersonation. 5 background processes running simultaneously.

| Season | Games | PID | Notes |
|---|---|---|---|
| 2021-22 | 1,230 | 377366 | Clean season — highest priority |
| 2018-19 | 1,230 | 378679 | Clean season |
| 2017-18 | 1,230 | 378805 | Clean season |
| 2019-20 | 1,059 | 378830 | COVID bubble |
| 2020-21 | 1,080 | 378893 | COVID shortened |

**Logs:** `/tmp/pbp_statsapi_{season}.log`  
ETA: ~30 min per season running in parallel.

**After PBP completes:** Run normalization → `possessions_clean_{season}.parquet` → add
season to `SEASONS` in `src/modeling/model_config.py` (currently `["2022-23", "2023-24", "2024-25"]`).

### Target state after Phase 2 complete:
- Clean seasons for RAPM: 2017-18, 2018-19, 2021-22, 2022-23, 2023-24, 2024-25 (6)
- Walk-forward transitions: 5 clean (vs. 2 now)
- COVID seasons (RAPM-only, flagged): 2019-20, 2020-21

---

## Phase 3 — Low-Sample Player Prior (DARKO Substitute)

**Status:** PENDING (blocked by Phase 2 PBP → RAPM → expanded profiles)  
**Detail doc:** Part of main plan; no separate doc yet.

**Root problem:** Players with <500 possessions fall back to league-mean RAPM prior.
This is especially bad for rookies, injury-returnees, and late-season callups. The
pipeline infrastructure for DARKO exists (`src/data_fetch/fetch_darko_manual.py`,
`src/data_normalize/normalize_darko.py`) but has never been fed actual data.

**Fix:** Build substitute prior: draft_position + age + box BPM + xRAPM → predicted
ORAPM/DRAPM. Apply as the per-player prior in `model_rapm.py` for low-sample players.
DARKO CSV drops in via the existing ingest path when available.

---

## Phase 4 — Hyperparameter Calibration

**Status:** DEFERRED (needs 5+ transitions; current 2 transitions have too high variance)  
**Calibration scripts built and run:**
- `src/player_eval/calibrate_team_scale.py` → `reports/team_scale_calibration.json`
- `src/player_eval/year_to_year_bke_deltas.py` → `reports/age_curve_calibration.json`

**Findings (do not apply yet — see `docs/findings/calibration_findings_2026-05-20.md`):**
- TEAM_SCALE: ~25 implied vs. 20.0 current (2-season estimate, noisy)
- AGE_CURVE: 27-30, 30-33, 33-36 brackets show faster decline than hardcoded
- RAPM PRIOR_REFINEMENT_STRENGTH = 30.0: needs grid search [10, 20, 30, 50, 80]
- SEASON_DECAY_WEIGHTS: needs cross-validation vs. walk-forward Brier

**Apply when:** 5+ walk-forward transitions available (i.e., after Phase 2 complete +
RAPM re-run).

---

## Phase 5 — Pipeline Correctness Fixes

**Status:** PENDING (independent of Phases 1-4)  
**Detail doc:** `docs/plans/data_pipeline_audit_plan.md`

Three independent tracks:

| Track | Problem | Root Cause | Priority |
|---|---|---|---|
| A | RAPM over-regularized, elite players suppressed | Alpha too high | High |
| B | BPM hand-tuned on 2-3 players | 3 hardcoded constants overriding B-Ref formula | Medium |
| C | Traded players use wrong team's box stats | Dedup keeps max-minutes team, not TOT row | Medium |

Track A requires external RAPM benchmark (ESPN RPM, nbarapm.com). Tracks B and C are
pure formula fixes.

---

## Phase 6 — Basketball Intuition Wiring

**Status:** PENDING (gates on Phase 5 data being correct)  
**Detail doc:** `docs/plans/basketball_intuition_plan.md`

**Phase A (can start now — no external data needed):**
- Canonical position columns (`position_band_3`, `position_band_5`) added as
  computed outputs in `compute_position_estimate.py`
- B2B/rest penalty fitted from actual game logs (data exists)
- HCA calibrated per-team from game logs

**Phase B (gates on Phase 5 + 5+ clean seasons):**
- BKE dimension weights recalibrated against RAPM
- Archetype feature importance re-estimated
- Interaction matrix (121 pair values) validated empirically

---

## Phase 7 — Archetype Validation

**Status:** PENDING (independent track)  
**Detail doc:** `docs/plans/archetype_validation_plan.md`

Three tracks: year-over-year stability (transition matrix), threshold sensitivity analysis,
and external archetype comparison. Can run in parallel with any other phase. Does not modify
any production code — validation only.

---

## Dependency Graph

```
Phase 2c PBP ──► Re-run RAPM ──► Phase 3 (prior model)
                     │
                     ▼
                Phase 4 (calibration, 5+ transitions needed)

Phase 0 (harness) ── always running, re-run after each change ──►  Brier delta

Phase 1 (minute model) ── independent, any time ──► Phase 0 re-run

Phase 5 (pipeline fixes) ── independent ──► Phase 6 (intuition wiring)

Phase 7 (archetype) ── fully independent
```

---

## Verification Protocol

After any phase completes:
1. Re-run `python3 src/simulation/validate_forecast.py`
2. Compare new Brier against the 0.2320 baseline
3. Record result in `docs/findings/` with date
4. If Brier worsened, investigate before merging

**Target path:** 0.2320 → 0.215 (after Phase 1+2) → 0.205 (after Phase 4) → ~0.200 (aspirational)

---

## Centralized Season Config

All pipeline season lists now import from one place:

```python
# src/modeling/model_config.py
SEASONS: List[str] = ["2022-23", "2023-24", "2024-25"]
COVID_SEASONS: List[str] = ["2019-20", "2020-21"]
SEASONS_GAME_MODEL: List[str] = [s for s in SEASONS if s not in set(COVID_SEASONS)]
SEASONS_RAPM: List[str] = SEASONS
```

**To add a season:** Update `SEASONS` here only. Prerequisite: `possessions_clean_{season}.parquet`
exists (PBP fetch + normalization complete).
