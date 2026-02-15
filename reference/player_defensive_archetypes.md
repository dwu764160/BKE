# Player Defensive Archetypes — v3.3 Logic Documentation

## Overview

This document is the authoritative reference for the defensive archetype classifier implemented in [src/data_compute/compute_defensive_archetypes_v2.py](src/data_compute/compute_defensive_archetypes_v2.py).

The v3.3 system explicitly separates:
- **Archetype (role):** what a player is asked to do on defense
- **Impact (quality):** how well the player performs that role

Archetype assignment is **behavior-only**. Impact metrics (`defensive_effectiveness`, `defensive_fit`) are retained only as post-classification overlays.

---

## 1. Core v3.3 Changes

- Removed impact from archetype assignment flow.
- Replaced hard-threshold routing with a **3-layer confidence-scored decision system**:
  1. Size-band gating (Guard / Wing / Big)
  2. Behavioral role confidence scores (continuous, percentile-based)
  3. Margin stability rule (small top-2 gaps route to rotational role)
- Added explicit rotational fallback archetype for perimeter bands only:
  - `Rotational Defender`
- Removed `Rotational Big` from assignment; big-band players now classify as `Rim Protector`, `Dropping Big`, or `Mobile Big` (with `Low-Activity Defender` override still available).

---

## 2. Data Sources

Same primary sources as v3.0:
- Matchup versatility: `data/matchup/matchup_versatility.parquet`
- Matchup difficulty: `data/matchup/matchup_difficulty.parquet`
- Tracking defense dashboards (`defense_Overall`, `defense_LessThan6Ft`, `defense_3Pointers`)
- Tracking defense player file: `tracking_Defense.parquet`
- Box season stats: `data/historical/complete_player_season_stats.parquet`
- Player profiles: `data/processed/player_profiles_advanced.parquet`
- Tracking movement/rebounding: `tracking_SpeedDistance.parquet`, `tracking_Rebounding.parquet`
- Hustle stats: `hustle_stats.parquet`
- Bios: `data/historical/players.parquet`

---

## 3. Feature & Axis Construction

### Base percentiles (season-wide)
The classifier ranks all qualified players per season for metrics like:
`switch_score`, `elite_matchup_pct`, `STL_PER100_DEF_POSS`, `BLK_PCT`, `DEF_RIM_FG_PCT`,
`RIM_FGA_RATE`, `pct_guards`, `pct_forwards`, `pct_centers`, `AVG_SPEED_DEF`, `height_inches`,
`DEFLECTIONS`, `CONTESTED_SHOTS`, `CONTESTED_SHOTS_3PT`, `DEF_LOOSE_BALLS_RECOVERED`, `DEF_BOXOUTS`,
`engagement_score`, `hustle_score`.

### Behavioral role indices (assignment-only)
v3.3 computes percentile-ranked behavior axes:
- `ball_pressure_index_pctl`
- `screen_navigation_index_pctl`
- `offball_navigation_index_pctl`
- `drop_coverage_index_pctl`
- `switch_index_pctl`
- `rim_protection_index_pctl`
- `help_activity_index_pctl`
- `liability_index_pctl`
- `matchup_diversity_pctl`

These indices are derived from assignment profile, tracking/hustle events, and movement workload only.

---

## 4. Classification Flow (Role-Only, Confidence-Scored)

## Layer 1: Size Band Gating
- Deterministic routing:
  - If `pct_centers >= 0.50` or `height_inches >= 81` => `Big`
  - Else if `pct_guards >= 0.60` => `Guard`
  - Else => `Wing`

## Layer 2: Size-Band Role Confidence Scoring

Instead of "first gate passed wins", v3.3 computes eligible role confidence scores and assigns:

`role = argmax(role_scores)`

Representative examples:
- Guard:
  - `poa_score = 0.35*ball_pressure + 0.25*screen_navigation + 0.20*difficulty + 0.10*matchup_diversity + 0.10*engagement`
    - Anchor: `ball_pressure >= 0.60`
  - `offball_score = 0.35*offball_navigation + 0.25*deflections + 0.20*contest_3pt + 0.10*help_activity + 0.10*engagement`
  - `rotational_score = 1 - std([ball_pressure, offball_navigation, switch_index])`
    - Anchors: `0.30 <= ball_pressure <= 0.75` and `0.30 <= offball_navigation <= 0.75`
  - `low_activity_score = 1 - engagement`
    - Anchor: `engagement <= 0.30`
- Wing:
  - `wing_score = 0.35*difficulty + 0.25*ball_pressure + 0.20*contest_2pt + 0.10*matchup_diversity + 0.10*engagement`
    - Anchor: `difficulty >= 0.60`
  - `versatile_score = 0.30*switch_index + 0.25*matchup_diversity + 0.20*help_activity + 0.15*min(ball_pressure, rim_protection) + 0.10*engagement`
    - Anchors: `switch_index >= 0.65` and `max_position_share <= 0.60`
  - `offball_score` and `rotational_score` definitions mirror guard competition behavior.
- Big:
  - `rim_score = 0.45*rim_protection + 0.20*contest_2pt + 0.15*help_activity + 0.10*drop_coverage + 0.10*engagement`
    - Anchor: `rim_protection >= 0.65`
  - `drop_score = 0.40*drop_coverage + 0.25*rim_protection + 0.15*help_activity + 0.10*difficulty + 0.10*engagement`
    - Anchors: `drop_coverage >= 0.65` and `switch_index <= 0.60`
  - `mobile_score = 0.40*switch_index + 0.20*mobility_metric + 0.15*matchup_diversity + 0.15*help_activity + 0.10*rim_protection`
    - Anchors: `switch_index >= 0.65` and `drop_coverage <= 0.75`
  - Big-band assignment excludes `Rotational Big` in v3.3.
  - If anchor-gated big roles are empty, fallback competition uses all three big roles: `Rim Protector`, `Dropping Big`, `Mobile Big`.

Role assignment uses score competition among anchor-eligible roles only.

## Layer 3: Margin Stability Rule
- Let `top_score` and `second_score` be the top two eligible role scores in-band.
- If `top_score - second_score < 0.05`, assign `Rotational Defender` for guard/wing bands.
- Big band does not use rotational fallback in v3.3.
- This avoids brittle cliff effects and suppresses order bias from branch-style routing.

## Layer 4: Low-Activity Override
- If `engagement < 0.25` and `primary_score < 0.65` => force `Low-Activity Defender`.

## Layer 5: Confidence
- `confidence = 0.5 + 0.5 * (primary_score - second_score)`
- Interpreted on `[0.5, 1.0]` as role separation strength.

---

## 5. Archetypes (v3.3)

1. `POA Defender`
2. `Wing Stopper`
3. `Off-Ball Chaser`
4. `Versatile Defender`
5. `Rim Protector`
6. `Dropping Big`
7. `Mobile Big`
8. `Rotational Defender`
9. `Low-Activity Defender`

Secondary tags are behavior modifiers (e.g., `Screen Navigator`, `Ball Hawk`, `Switchable`, `Helper`, `Liability`).

---

## 6. Impact Overlay (Not Used for Assignment)

`defensive_effectiveness` and `defensive_fit` remain in outputs for downstream value modeling.

- They can still use result-quality signals (`d_results_pctl`, rim suppression, etc.).
- They **must not** feed back into role assignment in this stage.

---

## 7. Outputs

Primary files:
- `data/processed/defensive_archetypes_v2.parquet`
- `data/processed/defensive_archetypes_v2.csv`

Impact report files (baseline vs v3.3 run):
- `data/processed/defensive_archetypes_v2_impact_report.csv`
- `data/processed/defensive_archetypes_v2_impact_report.txt`

Key new/updated fields include:
- `size_band`
- `ball_pressure_index_pctl`
- `screen_navigation_index_pctl`
- `offball_navigation_index_pctl`
- `drop_coverage_index_pctl`
- `switch_index_pctl`
- `rim_protection_index_pctl`
- `help_activity_index_pctl`
- `liability_index_pctl`
- `matchup_diversity_pctl`

---

## 8. Repro

```bash
.venv/bin/python src/data_compute/compute_defensive_archetypes_v2.py
```

Last updated: 2026-02-14 (v3.3)