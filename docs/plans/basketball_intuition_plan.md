# Basketball Intuition Plan

> **Status:** Brainstormed, not yet implemented. Decisions finalized 2026-05-19.
> **Scope:** Formalize and calibrate all basketball domain knowledge baked into the pipeline.
> **Does not change:** BKE metric philosophy, archetype philosophy, 9-dimension structure.
> **Prerequisite for Phase B:** `docs/data_pipeline_audit_plan.md` complete + data backfill.

---

## Two-Phase Structure

**Phase A — Foundation Rules** (no external data needed, can start immediately)
Fix structural and normalization issues that don't require more data to resolve.

**Phase B — Magnitude Calibration** (gates on clean data + backfilled seasons)
Replace hand-tuned penalty/bonus magnitudes with empirically fitted values.
Phase B cannot produce honest results until the data pipeline is correct (RAPM fixed,
traded players fixed, BPM restored) and we have ≥5 seasons to regress against.

---

## Phase A: Foundation Rules

### A1: Canonical Position Columns (Single Source of Truth)

**Problem:** Position logic is applied ad-hoc in different scripts using different labels
and groupings. There is no single canonical `position_band_3` (smalls/wings/bigs) or
`position_band_5` (Guard/Guard-Forward/Forward/Forward-Center/Center) column that every
downstream script reads from.

**Fix:**
1. Add `position_band_5` and `position_band_3` as computed output columns in
   `src/data_compute/compute_position_estimate.py`
2. Add both to the player profile schema (`build_player_impact_profiles.py`)
3. Add both to the profile aggregate (`build_profile_aggregate.py`)
4. Audit every script that does ad-hoc position grouping and replace with these columns

**Canonical mappings:**
```
position_band_5:  Guard | Guard-Forward | Forward | Forward-Center | Center
position_band_3:
  smalls → Guard
  wings  → Guard-Forward, Forward
  bigs   → Forward-Center, Center
```

**Files to touch:**
- `src/data_compute/compute_position_estimate.py` (add computed columns)
- `src/player_eval/build_player_impact_profiles.py` (add to schema)
- `src/profile_aggregate/build_profile_aggregate.py` (propagate)
- Audit: `src/modeling/layer1_portable_talent.py`, `src/modeling/construct_bke_scores_v27.py`,
  `src/profile_aggregate/team_feature_aggregation.py` for any ad-hoc position logic

**Validation:** Every player-season has a non-null `position_band_3` and `position_band_5`.
No script uses raw NBA position labels (PG/SG/SF/PF/C) for grouping without going through
the canonical columns.

---

### A2: B2B/Rest Penalty — Fit From Game Logs, Then Wire

**Problem:** `src/data_compute/compute_rest_home_back2back.py` correctly computes `days_rest`
and `is_b2b` for every game. This data is **never connected to the game model.**
Rest effects are worth ~1.5–2.5 pts of net rating on B2B second games.

**Fix (two steps):**

**Step 1: Fit B2B coefficient from 3 seasons of game logs**

```python
# For each game in team_game_logs.parquet:
# y = actual_margin (home - away)
# features:
#   is_b2b_home (1 if home team on B2B second game, else 0)
#   is_b2b_away (1 if away team on B2B second game, else 0)
#   days_rest_home, days_rest_away (0-7 scale)
#   team_strength_diff (proxy: prior season win%)
# → OLS coefficients for is_b2b and days_rest

# Expected result based on literature:
#   is_b2b_home: ~ -1.5 to -2.5 pts
#   is_b2b_away: ~ -1.5 to -2.5 pts (independent effect)
#   days_rest: ~ +0.3 to +0.5 pts per additional rest day (up to 4-5 days)
```

New file: `src/data_compute/fit_rest_hca_coefficients.py`
Output: `reports/rest_hca_coefficients.json`

**Step 2: Wire fitted coefficients into game model**

In `src/simulation/simulation_config.py`: add `B2B_PENALTY` and `REST_DAY_BONUS`
loaded from `reports/rest_hca_coefficients.json`.

In `src/profile_aggregate/team_feature_aggregation.py`: when computing
`projected_net_rating`, apply adjustment:
```
adjusted_net_rating = base_net_rating
    + B2B_PENALTY * is_b2b(team, game_date)
    + REST_DAY_BONUS * days_rest(team, game_date)
```

This adjustment is game-specific (varies per matchup), not team-season-specific.
The walk-forward harness (Plan 5) will use this at prediction time per game.

**Validation:** Rerun the Brier/calibration analysis on the 3 existing backtest seasons
(in-sample) and confirm that adding rest adjustment moves calibration in the right direction
(games where BKE predicted higher than actual had more B2B situations than expected, etc.)

---

### A3: Team-Specific Home Court Advantage

**Problem:** HCA is flat 2.0 for all 30 teams. Denver and Utah historically show 3.0–3.5.
We have 3 seasons of actual home/away margin data by team to fit this properly.

**Fix:**

```python
# For each team in 3 seasons:
# home_margin_advantage = mean(home_margins) - mean(away_margins)
# adjusted_hca = home_margin_advantage - talent_adjustment
#   (talent_adjustment accounts for teams being genuinely better/worse)

# Regularize toward league mean (2.0) for teams with <60 home games observed
# shrinkage = n_games / (n_games + 30)
# team_hca = shrinkage * observed_hca + (1 - shrinkage) * LEAGUE_AVG_HCA
```

Add to `fit_rest_hca_coefficients.py`.
Output: `reports/rest_hca_coefficients.json` (extend existing output).

Store as a `team_hca` dictionary in `simulation_config.py`, loaded at runtime.
Fallback to `HOME_COURT_ADVANTAGE = 2.0` for any team with <30 observed games.

**Validation:** Denver and Utah should show HCA ≥ 2.5. No team should show HCA < 0.5
or > 4.5 (these would indicate data issues, not real extreme HCA).

---

### A4: Season-Relative Thresholds (Future-Proofing)

**Problem:** Several thresholds use fixed values that will drift as the NBA evolves:
- `SPACING_3PA_RATE_THRESHOLD = 0.30` (% of shots from 3 to count as a "spacer")
- `SPACING_3P_PCT_THRESHOLD = 0.35` (3P% floor for credible shooter)
- League average 3PT% and usage rates embedded in BPM formula

As the league's 3-point revolution continues, a 0.30 3PA rate that was above-average in
2022 may be below-average by 2027. A player shouldn't lose their "spacer" classification
just because the league caught up to them.

**Fix:**
Replace fixed thresholds with season-relative percentiles computed at pipeline runtime:

```python
# At pipeline execution, compute from qualified player pool for that season:
SPACING_3PA_THRESHOLD = np.percentile(qualified_players["3PA_rate"], 40)
# "A spacer is in the top 60% of 3-point attempt rates for their season"

SPACING_3P_PCT_THRESHOLD = np.percentile(qualified_players["3P_pct"], 35)
# "A credible shooter is in the top 65% of 3-point percentage for their season"
```

Similarly, any hardcoded `LEAGUE_AVG_PPP = 1.14` or `DEFAULT_PACE_PER_48 = 100.0` should
be loaded from the actual season's game log data, not assumed constant.

**Files to touch:**
- `src/player_eval/constants.py` (replace SPACING thresholds)
- `src/simulation/simulation_config.py` (replace LEAGUE_AVG_PPP, DEFAULT_PACE_PER_48)
- `src/profile_aggregate/team_feature_aggregation.py` (load thresholds at runtime)

**Validation:** Run pipeline for 2022-23, 2023-24, 2024-25 and confirm that season-relative
thresholds shift appropriately (3PT floor should be higher in more recent seasons). Number of
players classified as spacers should be stable year-over-year (~8-12 per team), not shrinking.

---

### A5: Pace as Matchup Variable

**Problem:** The game model uses a single `DEFAULT_PACE_PER_48 = 100`. Pace differences
of 5-8 possessions between teams are common and affect total scoring and margin variance.
A slow team vs. a fast team produces a different distribution of margins than two teams
with similar pace.

**Fix:**
Add `team_pace` to team profiles (possessions per 48 min, from historical games):

```python
# Matchup pace = weighted average of home/away team pace:
game_pace = 0.5 * home_team_pace + 0.5 * away_team_pace
# Adjust sigma_league by pace:
sigma_adjusted = SIGMA_LEAGUE * sqrt(game_pace / LEAGUE_AVG_PACE)
# Higher pace → higher scoring variance → slightly wider margins
```

This is a small fix but prevents the game model from treating a Grizzlies-Pistons
game (slow pace, lower variance) identically to a Hawks-Timberwolves game (high pace).

**Files to touch:**
- `src/simulation/simulation_config.py` (add pace parameters)
- `src/profile_aggregate/team_feature_aggregation.py` (compute and expose team_pace)
- Walk-forward harness (Plan 5) uses team_pace per game

---

## Phase B: Magnitude Calibration
*(Gates on: data pipeline audit complete + ≥5 clean seasons from backfill)*

### B1: Interaction Matrix — Simplify to ~10 Rules, Then OLS Validate

**Problem:** 121 pairwise archetype synergy values, all hand-tuned with no empirical validation.

**Step 1: Simplify to ~10 canonical basketball rules**

```
STRONG POSITIVE (+0.08 to +0.12):
  Ball Dominant Creator + Off-Ball Movement Shooter  (off-ball reads collapse defense)
  Ball Dominant Creator + Off-Ball Finisher          (lob threat forces help)
  Playmaking Big + Off-Ball Shooter                  (pick-and-pop spacing)
  PnR Rolling Big + Ball-Dominant Playmaker          (classic two-man game)

MILD POSITIVE (+0.03 to +0.06):
  Any Creator + Spot-Up Shooter                      (basic gravity benefit)
  Interior Scorer + POA Defender                     (two-way contribution)

NEUTRAL (0.0):
  Off-Ball Finisher + Off-Ball Finisher              (both need lobs, neither creates)

MILD NEGATIVE (-0.03 to -0.06):
  Ball Dom Creator + Ball Dom Creator (both weak)    (ball-sharing conflict)
  Two Interior Scorers                               (compete for paint)

STRONG NEGATIVE (-0.08 to -0.10):
  Ball Dom Creator + Ball Dom Creator (one elite)    (only if combined USG > 55%)
  Two Perimeter Scorers with no playmaker            (no creation, clog spacing)
```

Replace the current 121-value matrix with ~10 rule-based conditions.
Maintain the `INTERACTION_CAP = 1.5` guardrail.

**Step 2: OLS validation (after backfill + clean pipeline)**

```python
# For each team-season:
# y = actual_net_rating - BKE_talent_base  (the "residual" the model can't explain)
# X = binary indicator for each of the ~10 canonical rules (is this pair present?)
# → OLS coefficients with standard errors
# Keep rules with |t-stat| > 1.5 (relaxed given thin data)
# Shrink insignificant rules toward 0
```

Target: reduce to 5-7 rules that have statistical support. Remove rules that show no signal.

---

### B2: Modifier Magnitude Calibration (Playmaking, Spacing, Defense)

**Problem:** After aggressive manual tuning, penalty/bonus magnitudes are directionally
correct but quantitatively unvalidated:
- `PLAYMAKING_SOLO_PENALTY = -0.25`
- `SPACING_POOR_PENALTY = -0.35`
- `RP_MISSING_PENALTY = -0.40`
- etc.

**Fix (after ≥5 seasons):**

```python
# For each team-season (after clean data):
# y = actual_net_rating - BKE_talent_base (residual)
# X = [playmaker_count, spacer_count, has_rim_protector, has_poa_defender,
#      has_liability_defender, archetype_diversity_score]
# → OLS: fitted coefficients are the empirically calibrated modifier magnitudes

# Cross-validate: hold out one season, fit on remainder, check magnitude stability
```

Replace hardcoded values in `src/player_eval/constants.py` with fitted values.
Retain guardrail: `MODIFIER_MAX_FRACTION = 0.30` (structure can't dominate talent).

---

### B3: Age Curve Archetype-Stratification

**Problem:** Single universal age curve applied to all players. Guards likely peak
slightly earlier and decline differently than bigs.

**Fix (after backfill provides enough player-season transitions):**

```python
# For each player-season transition (N → N+1):
# y = (BKE_score_N+1 - BKE_score_N) / BKE_score_N  (% change)
# X = [age_N, position_band_3, archetype_primary, experience_years]
# → fit separate age curves by position_band_3 (smalls / wings / bigs)

# Validate: fitted curves should show bigs peaking slightly later (27-29 vs 25-27 for guards)
# due to physical maturity timelines
```

Requires ≥3 seasons of player-season pairs per position band (target: 100+ per band).
Currently achievable only with backfilled data.

---

### B4: Volatility Model Scaling (ALPHA_3PA, ALPHA_CREATION)

**Problem:** `ALPHA_3PA = 5.0` and `ALPHA_CREATION = 5.0` appear too aggressive.

**Fix:**
```python
# For each team-season:
# y = variance(actual_game_margins) across 82 games
# X = [3pa_concentration, creation_concentration, transition_rate]
# → OLS fit of alpha coefficients
# Expected result: alphas ~0.5-1.5, not 5.0
```

---

## Summary: Phase A vs Phase B

| Fix | Phase | Gate | Expected Impact |
|---|---|---|---|
| Canonical position columns | A | None | Data normalization / prevents drift |
| B2B/rest wiring | A | Fit from existing game logs | Direct Brier improvement |
| Team-specific HCA | A | Existing game logs | Direct Brier improvement |
| Season-relative thresholds | A | None | Future-proofing |
| Pace as matchup variable | A | None | Small margin calibration improvement |
| Interaction matrix simplify+OLS | B | Clean data + ≥5 seasons | Team net rating quality |
| Modifier magnitude OLS | B | Clean data + ≥5 seasons | Team net rating quality |
| Age curve archetype-stratification | B | Clean data + backfill | Forecast accuracy |
| Volatility alpha calibration | B | Clean data + ≥5 seasons | Calibration in extreme bins |

---

## Files Changed (Phase A)

| File | Change |
|---|---|
| `src/data_compute/compute_position_estimate.py` | Add `position_band_3`, `position_band_5` output |
| `src/data_compute/fit_rest_hca_coefficients.py` | **New** — fit B2B + rest + team HCA coefficients |
| `src/player_eval/build_player_impact_profiles.py` | Propagate canonical position columns |
| `src/profile_aggregate/build_profile_aggregate.py` | Propagate position columns |
| `src/player_eval/constants.py` | Replace fixed thresholds with season-relative |
| `src/simulation/simulation_config.py` | Load team HCA + B2B coefficients; add pace parameters |
| `src/profile_aggregate/team_feature_aggregation.py` | Apply B2B/rest/HCA adjustments; compute team_pace |

---

## Relationship to Other Plans

- **Plan 1 (data pipeline audit):** Phase B modifier calibration requires RAPM fixed (Track A)
  and traded players fixed (Track C) so the team net rating residuals we regress against are honest
- **Plan 2 (minute model rebuild):** Canonical position columns (A1) are features in the temporal
  minute model; fix A1 before rebuilding
- **Plan 4 (archetype validation):** Archetype stability results gate Phase B interaction matrix
  calibration — unstable archetypes mean the pair-count regression is built on sand
- **Plan 5 (walk-forward harness):** B2B/rest adjustments (A2) and team HCA (A3) are applied
  at game prediction time; harness validates their impact on Brier
