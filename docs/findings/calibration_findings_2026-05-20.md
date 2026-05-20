# Calibration Findings — 2026-05-20

> **Session:** BKE large-scale model improvement (Phase 4 calibration scripts).
> **Status:** Findings recorded; NOT applied to production constants.
> **Apply when:** 5+ walk-forward transitions available. See master plan for trigger.
> **Scripts:** `src/player_eval/calibrate_team_scale.py`, `src/player_eval/year_to_year_bke_deltas.py`
> **Reports:** `reports/team_scale_calibration.json`, `reports/age_curve_calibration.json`

---

## 1. Walk-Forward Baseline (Phase 0)

The first genuine out-of-sample game-level validation. Uses prior-season projected ratings
from the forecast pipeline scored against actual next-season games.

| Transition | Games | Brier | Log Loss | Accuracy | Margin RMSE |
|---|---|---|---|---|---|
| 2022-23 → 2023-24 | 1,230 | 0.2267 | 0.6425 | 62.2% | 14.79 pts |
| 2023-24 → 2024-25 | 1,225 | 0.2374 | 0.6697 | 60.8% | 15.01 pts |
| **Aggregate (2 transitions)** | **2,455** | **0.2320** | **0.6561** | **61.5%** | — |

**Baselines for context:**
- Coin flip: Brier = 0.2500
- Naive home-wins prior (55%): Brier ≈ 0.2475
- Vegas closing lines (aspirational target): Brier ≈ 0.195–0.210
- Our current out-of-sample: **0.2320** — beats no-information, lags Vegas by ~0.035

---

## 2. TEAM_SCALE Calibration

`DEFAULT_TEAM_SCALE = 20.0` in `src/player_eval/constants.py`.

**Method:** OLS fit of `actual_net_rating = a * projected_net_rating + b` per season.
"Actual net rating" estimated from observed win% via: `actual_net ≈ 11.0 * Φ⁻¹(win%)`.

| Season | n_teams | r | OLS slope | OLS intercept | Spread ratio | Implied scale |
|---|---|---|---|---|---|---|
| 2023-24 | 30 | 0.733 | 0.9266 | -2.310 | **1.264** | ~25.3 |
| 2024-25 | 30 | 0.620 | 0.7648 | -1.554 | **1.234** | ~24.7 |
| **Aggregate** | 60 | 0.674 | 0.8411 | -1.906 | **1.249** | **~25.0** |

**Interpretation:**

- **Correlation is decent (r=0.67–0.73):** Projected rankings are directionally correct.
  Teams we project as elite (high BKE, high net rating) are actually elite.
- **Spread ratio 1.25:** Actual team spread (std of actual net ratings) is 25% wider than
  projected spread. The model is under-spreading — it thinks the gap between the Celtics
  and the Wizards is smaller than it actually is.
- **OLS slope 0.84:** Slightly below 1.0, meaning at the extremes the projection is
  over-dispersed (slightly contradicts the spread ratio — this contradiction arises from
  the sigma_game assumption used to convert win% to net rating).
- **Implied scale ~25:** Changing `DEFAULT_TEAM_SCALE` from 20.0 to 25.0 would widen
  projected spreads and likely reduce calibration error.

**Why not applied yet:**
1. Only 60 data points (2 seasons × 30 teams). One dominant team (2023-24 Celtics: +8.1 net)
   or one tanking team can shift the fitted scale by ±3 points.
2. The OLS slope (0.84) and spread ratio (1.25) point in slightly different directions;
   resolving which effect dominates requires more data.
3. `team_net_rating_raw` and `team_net_rating_projected` are IDENTICAL in forecast mode
   (r=1.000 — the calibration step in `team_feature_aggregation.py` is bypassed when
   correlation threshold isn't met). So the current "calibrated" value is just `DEFAULT_TEAM_SCALE`
   applied directly. Fixing the calibration pathway is a prerequisite to testing scale changes.

**Recommended action (when 5+ transitions available):**
Apply the 2-parameter OLS fit `(a, b)` per walk-forward fold. Do NOT just update
`DEFAULT_TEAM_SCALE` — the intercept term `b` (systematic offset) is equally important.

---

## 3. Age Curve Calibration

`AGE_CURVE_DELTAS` and `AGE_CURVE_BREAKPOINTS` in `src/player_eval/constants.py`.

**Method:** Year-over-year BKE deltas from `aggregate/player_profile_aggregate.parquet`.
486 player-year pairs across 2 transitions (2022-23→2023-24, 2023-24→2024-25).
Qualification: 500+ minutes, 20+ games played.

**Current hardcoded breakpoints:** `[21, 24, 27, 30, 33, 36]`

| Bracket | N pairs | Empirical mean | Hardcoded | Drift | Flag |
|---|---|---|---|---|---|
| <21 | 1 | — | +0.0500 | — | insufficient data |
| 21–24 | 62 | +0.0983 | +0.0800 | +0.018 | OK |
| 24–27 | 106 | +0.0337 | +0.0400 | -0.006 | OK |
| **27–30** | **113** | **-0.0295** | **+0.0100** | **-0.040** | **⚠ FLAG** |
| **30–33** | **95** | **-0.0390** | **-0.0100** | **-0.029** | **⚠ FLAG** |
| **33–36** | **62** | **-0.0743** | **-0.0200** | **-0.054** | **⚠ FLAG** |
| 36+ | 47 | -0.0315 | -0.0300 | -0.002 | OK |

**Key finding — 27-30 bracket is wrong sign:**
The model assumes players aged 27-30 still improve slightly (+0.010 BKE/yr). The data says
they decline (-0.030/yr). This is a 4-point swing in the wrong direction. Practically, the
model is currently projecting veteran role players in their late 20s as improving when they
are actually declining — over-weighting them in team projections.

**Why this makes basketball sense:**
The 27-30 "late prime" bracket is now showing earlier decline, likely because:
- The current NBA leans younger (more rookies getting heavy minutes)
- Elite players like LeBron/Steph who extended peaks into late 30s are outliers, not the norm
- Role players (who dominate this sample) genuinely peak at 25-27 and start declining

**Why not applied yet (survivorship bias warning):**
The 486-pair sample only includes players with 500+ minutes both seasons. At ages 30-36,
the players who still meet this threshold are the league's most durable veterans — a positively
selected survivor group. The steepest declines (-0.074 at 33-36) likely OVERestimate the
general decline rate because the worst-declining players in this age band were already cut
or reduced to minimal roles and don't appear in the sample. Applying the raw empirical deltas
would over-correct for ages 33+.

**Recommended action:**
- Apply the 27-30 fix when 5+ transitions available (direction is reliable, magnitude uncertain)
- Use empirical 27-30: -0.020/yr (compromise between -0.030 empirical and +0.010 hardcoded)
- Apply 30-33 and 33-36 conservatively: -0.025 and -0.040 (half the empirical correction)
- Re-derive after survivorship bias is better understood (requires tracking RAPM of cut players)

---

## 4. RAPM Hyperparameters (Not Yet Calibrated)

These were not calibrated this session — scripts not yet built for cross-validation.

**Current values:**
- `PRIOR_REFINEMENT_STRENGTH = 30.0` in `src/modeling/model_rapm.py`
- `SEASON_DECAY_WEIGHTS = {0: 1.00, 1: 0.70, 2: 0.50, 3: 0.35}`
- `COVID_DECAY_WEIGHT` (recommended: 0.25) — not yet implemented

**Known issues from model_weaknesses_report.md:**
- Mean delta of -4.46 pts/100 vs. xRAPM benchmark (internal baseline)
- Elite players (Durant, Davis, Luka, Harden) rated below expected floors
- Suggests slope compression: elite players suppressed MORE than role players
- Possible diagnosis: alpha (PRIOR_REFINEMENT_STRENGTH) too high

**Plan:** Grid search over [10, 20, 30, 50, 80] for PRIOR_REFINEMENT_STRENGTH,
using walk-forward Brier (from Phase 0 harness) as objective. Run after Phase 2
PBP complete and RAPM re-run with expanded season history.

---

## 5. Data Availability Summary

| Season | Box Score | Game Log | Tracking | PBP | Status in Pipeline |
|---|---|---|---|---|---|
| 2024-25 | ✓ | ✓ | ✓ | ✓ | In `SEASONS` |
| 2023-24 | ✓ | ✓ | ✓ | ✓ | In `SEASONS` |
| 2022-23 | ✓ | ✓ | ✓ | ✓ | In `SEASONS` |
| 2021-22 | ✓ backfilled | ✓ backfilled | ✗ (empty) | 🔄 in progress | Add after PBP done |
| 2020-21 | ✓ backfilled | ✓ backfilled | ✗ (empty) | 🔄 in progress | COVID — RAPM only |
| 2019-20 | ✓ backfilled | ✓ backfilled | ✗ (empty) | 🔄 in progress | COVID — RAPM only |
| 2018-19 | ✓ backfilled | ✓ backfilled | ✗ (empty) | 🔄 in progress | Add after PBP done |
| 2017-18 | ✓ backfilled | ✓ backfilled | ✗ (empty) | 🔄 in progress | Add after PBP done |

**Why tracking is empty for pre-2022:** `leaguedashptstats` and `synergyplaytypes` endpoints
return 0 players for all seasons before 2022-23, regardless of API method (curl_cffi,
nba_api, direct HTTP). These endpoints appear to have no historical data backing before
the current-era NBA tracking system. Drives, at-rim frequency, synergy play types, and
touch/passing data cannot be backfilled.

**DARKO:** All 8 DARKO columns are N/A across all seasons. No public API. Requires manual
CSV download from BBall-Index. Pipeline ingest code exists at:
`src/data_fetch/fetch_darko_manual.py`, `src/data_normalize/normalize_darko.py`.

---

## 6. What These Findings Change (and Don't Change)

**Architecture unchanged:**
- BKE metric philosophy (9-dimension portable talent)
- RAPM backbone structure
- Archetype system
- Gaussian game model

**Confirmed correct (from this session's data):**
- Projected team rankings directionally correct (r=0.67–0.73 vs. actual win%)
- Young player development (21-27) tracked approximately correctly
- Team game log data: 100% W-L accuracy verified

**Needs fixing (prioritized):**
1. Minute model leakage (Phase 1) — affects every player's minutes projection
2. TEAM_SCALE / calibration bypass (Phase 4 setup) — affects all team projections
3. Age curve 27-30 bracket direction error (Phase 4) — affects veteran projections
4. RAPM elite player suppression (Phase 5 Track A) — affects top-end player ratings
