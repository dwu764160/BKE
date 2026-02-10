# Player Defensive Archetypes — v2.1 Logic Documentation

## Overview

This document is the authoritative reference for the defensive archetype classifier implemented in [src/data_compute/compute_defensive_archetypes_v2.py](src/data_compute/compute_defensive_archetypes_v2.py).
The model classifies players by defensive role using matchup data, tracking data, defensive workload, and box-score proxies.

Defensive archetypes provide **role context** (how a player defends) rather than a single "good/bad" score.  The classifier produces a primary archetype, an optional secondary modifier tag, and a confidence score.

**v2.1 key changes from v2.0:**
- Replaced gate-based if/elif hierarchy with **score-based argmax** (6 role affinity scores)
- Removed **Rotation Defender** and **Hustler** as primary archetypes (Hustler is now a secondary modifier)
- Engagement used **only as a filter** for Low-Activity Defender, not as a role driver
- Added **position assignment shares** (pct_guards, pct_forwards, pct_centers) as key inputs
- Matchup difficulty + versatility now drive every role

---

## 1. Data Sources

1a. Matchup Versatility (from matchup pipeline)
- `matchup_versatility.parquet`
- Key fields: `switch_score`, `positions_guarded`, `pct_guards`, `pct_forwards`, `pct_centers`

1b. Matchup Difficulty (from matchup pipeline)
- `matchup_difficulty.parquet`
- Key fields: `avg_opponent_ppg`, `elite_matchup_pct`

1c. Defensive Tracking (NBA tracking)
- `defense_Overall.parquet`, `defense_3Pointers.parquet`, `defense_LessThan6Ft.parquet`, `tracking_Defense.parquet`
- Key fields: `D_FG_PCT_*`, `PCT_PLUSMINUS_*`, `FREQ_*`, `DEF_RIM_FGA`, `DEF_RIM_FG_PCT`

1d. Box Score (season totals)
- `complete_player_season_stats.parquet`
- Key fields: `STL`, `BLK`, `DREB`, `PF`, `MIN`, `GP`

1e. Player Profiles (defensive workload)
- `player_profiles_advanced.parquet`
- Key fields: `POSS_DEF`, `SECONDS_DEF`

1f. Tracking (movement / rebounding)
- `tracking_SpeedDistance.parquet`, `tracking_Rebounding.parquet`
- Key fields: `DIST_MILES_DEF`, `AVG_SPEED_DEF`, `DREB_CHANCES`, `DREB_CHANCE_PCT`

---

## 2. Feature Engineering

Defensive workload normalization (replaces per-36 for defense):

```
STL_PER100_DEF_POSS = STL / POSS_DEF * 100
BLK_PCT = BLK / DEF_RIM_FGA  # proxy for block rate on rim attempts
DREB_PER100_DEF_POSS = DREB / POSS_DEF * 100
DEF_SECONDS_PER_GAME = SECONDS_DEF / GP
```

Composite signals:

```
ENGAGEMENT_SCORE = z(POSS_DEF/GP) + z(SECONDS_DEF/GP) + z(DIST_MILES_DEF/GP) + z(AVG_SPEED_DEF)
HUSTLE_SCORE = z(DREB_CHANCES) + z(DREB_CHANCE_PCT)
```

Position share percentiles (from matchup versatility):

```
guard_pctl   = pct_guards.rank(pct=True)
forward_pctl = pct_forwards.rank(pct=True)
center_pctl  = pct_centers.rank(pct=True)
speed_pctl   = AVG_SPEED_DEF.rank(pct=True)
```

---

## 3. Classification Method (v2.1 — Score-Based)

### Step 1: Engagement Filter
- If `engagement_pctl <= 0.18` AND `d_results_pctl <= 0.40` → **Low-Activity Defender**
- Everyone else → proceed to scoring

### Step 2: Role Affinity Scores (0–1)
Six weighted percentile sums, one per role:

| Score | Formula | Key Drivers |
|-------|---------|-------------|
| **poa_score** | `elite_matchup_pctl*0.30 + difficulty_pctl*0.25 + guard_pctl*0.25 + d_results_pctl*0.10 + stl_pctl*0.10` | Who they guard (stars, guards) |
| **wing_score** | `forward_pctl*0.25 + difficulty_pctl*0.25 + d_results_pctl*0.30 + (1-versatility_pctl)*0.10 + (1-center_pctl)*0.10` | Forward defense + results |
| **chaser_score** | `stl_pctl*0.35 + speed_pctl*0.15 + (1-difficulty_pctl)*0.25 + (1-center_pctl)*0.10 + d_results_pctl*0.15` | Steals + speed + low difficulty |
| **rim_score** | `blk_pctl*0.35 + (1-rim_fg_pctl)*0.30 + rim_fga_pctl*0.15 + center_pctl*0.20` | Blocks + rim suppression |
| **drop_big_score** | `(1-versatility_pctl)*0.25 + center_pctl*0.25 + reb_pctl*0.25 + rim_fga_pctl*0.15 + (1-guard_pctl)*0.10` | Interior + low versatility |
| **mobile_big_score** | `versatility_pctl*0.30 + reb_pctl*0.20 + center_pctl*0.15 + d_results_pctl*0.20 + forward_pctl*0.15` | Versatile + big + results |

### Step 3: Assignment
- **Primary archetype** = argmax(scores)
- **Confidence** = 0.40 + (top − second) / top × 0.55
- **Secondary tag** = context-dependent modifier (see below)

---

## 4. Archetype Definitions

Primary defensive archetypes (7 total):

1. **POA Defender** — Guards primary ball handlers; high elite matchup share, high guard assignment share
2. **Wing Stopper** — Defends forwards/wings; strong results, specialist (lower versatility)
3. **Off-Ball Chaser** — High steals/speed, lower matchup difficulty (roaming/off-ball)
4. **Rim Protector** — Elite blocks + rim FG suppression; interior anchor
5. **Dropping Big** — Interior-only, low versatility, stays in paint
6. **Mobile Big** — Versatile big who switches across positions
7. **Low-Activity Defender** — Engagement filter: bottom ~18% engagement + poor results

Secondary tags (modifiers, never primary):
- `Switchable` — High versatility, guards multiple positions
- `Lockdown` — Elite defensive results
- `Ball Hawk` — Elite steal rate
- `Hustler` — High hustle score (modifier, not standalone role)
- `Shot Blocker` — Elite block rate
- `Primary` — Default; does the role without notable modifier
- `Active Hands` — Good steal/deflection rate
- `Interior` — Paint-focused
- `Help` — Default for bigs
- `Hidden` — Low-Activity but not terrible
- `Liability` — Low-Activity with poor results

---

## 5. Output Schema

Key output fields in `data/processed/defensive_archetypes_v2.parquet`:

| Column | Description |
|---|---|
| `PLAYER_ID` | NBA player id |
| `PLAYER_NAME` | name |
| `SEASON` | season string |
| `defensive_archetype` | primary defensive archetype |
| `defensive_secondary` | secondary modifier tag |
| `defensive_confidence` | 0–1 confidence score |
| `assignment_difficulty` | High / Medium / Low |
| `switch_score` | versatility metric |
| `versatility_pctl` | switch score percentile |
| `avg_opponent_ppg` | matchup difficulty proxy |
| `elite_matchup_pct` | share vs elite scorers |
| `difficulty_pctl` | difficulty percentile |
| `STL_PER100_DEF_POSS` | steals per 100 defensive possessions |
| `stl_pctl` | steal rate percentile |
| `BLK_PCT` | blocks per defended rim attempt |
| `blk_pctl` | block rate percentile |
| `DEF_RIM_FG_PCT` | opponent FG% at rim |
| `RIM_FGA_RATE` | rim FGA rate |
| `engagement_score` | workload + movement composite |
| `engagement_pctl` | engagement percentile |
| `hustle_score` | rebounding effort composite |
| `hustle_pctl` | hustle percentile |
| `D_FG_DIFF` | opponent FG% impact (PCT_PLUSMINUS_Overall) |
| `d_results_pctl` | defensive results percentile |
| `poa_score` | POA Defender affinity (0–1) |
| `wing_score` | Wing Stopper affinity (0–1) |
| `chaser_score` | Off-Ball Chaser affinity (0–1) |
| `rim_score` | Rim Protector affinity (0–1) |
| `drop_big_score` | Dropping Big affinity (0–1) |
| `mobile_big_score` | Mobile Big affinity (0–1) |

---

## 6. Repro & Commands

Run the defensive archetype pipeline:

```bash
.venv/bin/python src/data_compute/compute_defensive_archetypes_v2.py
```

Outputs:
- `data/processed/defensive_archetypes_v2.parquet`
- `data/processed/defensive_archetypes_v2.csv`

Generate the player data viewer (HTML):

```bash
.venv/bin/python app/player_data_viewer.py
```

---

## 7. Notes & Next Steps

- Add NBA hustle dashboard fields (deflections, loose balls, charges, contests) when data is available.
- Add rim deterrence proxies based on on/off rim attempt rates.
- Add defensive on/off impact (from RAPM/DRAPM) to calibrate confidence.
- Add **help frequency proxies** (rotations, contests not on primary assignment).
- Weight **PnR coverage data** more heavily for bigs when available.
- Normalize within **positional clusters** (guards / wings / bigs) for tighter comparisons.

Last updated: 2026-02-10
