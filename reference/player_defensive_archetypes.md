# Player Defensive Archetypes — v3.0 Logic Documentation

## Overview

This document is the authoritative reference for the defensive archetype classifier implemented in [src/data_compute/compute_defensive_archetypes_v2.py](src/data_compute/compute_defensive_archetypes_v2.py).
The model classifies players by defensive role using matchup data, tracking data, hustle stats, defensive workload, player bios, and box-score proxies.

Defensive archetypes provide **role context** (how a player defends) plus **quality context** (how well they do it) via effectiveness and fit scores.

**v3.0 key changes from v2.1:**
- **Positional feasibility masks** — Guards can't be Rim Protectors, bigs can't be POA Defenders. Soft multipliers (not hard gates) based on player's own bio position + matchup position shares.
- **Versatile Defender** added as 8th primary archetype — absorbs switch-everything defenders (Draymond, Bam, etc.).
- **Role repulsion terms** — High affinity for one role suppresses incompatible roles (e.g., high rim_score ↓ poa_score).
- **Fixed BLK_PCT** — Now uses per-game BLK ÷ per-game DEF_RIM_FGA (both from tracking_Defense). v2.1 had a unit mismatch (season total BLK ÷ per-game DEF_RIM_FGA) producing absurd values.
- **Integrated hustle stats** — Deflections, contested shots (2PT/3PT), charges drawn, loose balls recovered, boxouts from NBA Hustle Dashboard.
- **Defensive effectiveness** (0–1) — Archetype-specific weighted results quality score.
- **Defensive fit** — Elite / Good / Average / Poor tier based on effectiveness.
- **Results damping factor** — All role scores suppressed for players with poor D_FG_DIFF results, improving score separation.
- **Sliding Low-Activity gate** — Expands engagement threshold when results are terrible (catches bad defenders who play heavy minutes).
- **On-ball exclusivity signal** — POA score now includes guard assignment share signal, tightening the archetype.
- **Mobile Big tightened** — Requires actual big evidence (center_pctl, height_pctl, forward_pctl).
- **Confidence formula** unchanged: `0.40 + separation * 0.55` — but masks + repulsion + damping create better separation.

---

## 1. Data Sources

1a. Matchup Versatility
- `data/matchup/matchup_versatility.parquet`
- Key fields: `switch_score`, `positions_guarded`, `pct_guards`, `pct_forwards`, `pct_centers`, `primary_position` (G/F/C from matchup data)

1b. Matchup Difficulty
- `data/matchup/matchup_difficulty.parquet`
- Key fields: `avg_opponent_ppg`, `elite_matchup_pct`

1c. Defensive Tracking (NBA tracking dashboard)
- `defense_Overall.parquet` — `D_FG_PCT`, `PCT_PLUSMINUS`, `D_FGA`, `FREQ`
- `defense_LessThan6Ft.parquet` — `LT_06_PCT`, `FGA_LT_06`, `PLUSMINUS`
- `defense_3Pointers.parquet` — `FG3_PCT`, `FG3A`, `PLUSMINUS`
- `tracking_Defense.parquet` — PER-GAME: `STL`, `BLK`, `DREB`, `DEF_RIM_FGM`, `DEF_RIM_FGA`, `DEF_RIM_FG_PCT`

1d. Box Score (season totals)
- `complete_player_season_stats.parquet`
- Key fields: `STL`, `BLK`, `DREB`, `PF`, `MIN`, `GP`, `REB_PCT`

1e. Player Profiles (defensive workload)
- `player_profiles_advanced.parquet`
- Key fields: `POSS_DEF`, `SECONDS_DEF`

1f. Tracking (movement / rebounding)
- `tracking_SpeedDistance.parquet` — `DIST_MILES_DEF`, `AVG_SPEED_DEF`
- `tracking_Rebounding.parquet` — `DREB_CHANCES`, `DREB_CHANCE_PCT`

1g. Hustle Stats (NEW in v3.0)
- `hustle_stats.parquet` — PER-GAME from NBA Hustle Dashboard
- Key fields: `DEFLECTIONS`, `CONTESTED_SHOTS`, `CONTESTED_SHOTS_2PT`, `CONTESTED_SHOTS_3PT`, `CHARGES_DRAWN`, `DEF_LOOSE_BALLS_RECOVERED`, `DEF_BOXOUTS`

1h. Player Bios (NEW in v3.0)
- `data/historical/players.parquet`
- Key fields: `primary_position` (Guard / Guard-Forward / Forward / Forward-Center / Center), `height_inches`

---

## 2. Feature Engineering

### Defensive workload normalization

```
STL_PER100_DEF_POSS = STL / POSS_DEF * 100
DREB_PER100_DEF_POSS = DREB / POSS_DEF * 100
DEF_SECONDS_PER_GAME = SECONDS_DEF / GP
```

### BLK_PCT (v3.0 fix)
```
BLK_PG = BLK_PG_TRK  (per-game from tracking_Defense)
         fallback: BLK / GP  (box score totals / games)
DEF_RIM_FGA_PG = DEF_RIM_FGA  (per-game from tracking_Defense)
                 fallback: D_FGA_LessThan6Ft → 3.0
BLK_PCT = BLK_PG / DEF_RIM_FGA_PG  (clipped 0–1)
```
Expected range: 0.00–0.60 (Wembanyama ~0.47, Gobert ~0.24)

### Composite signals

```
ENGAGEMENT_SCORE = z(POSS_DEF/GP) + z(SECONDS_DEF/GP) + z(DIST_MILES_DEF/GP) + z(AVG_SPEED_DEF)
HUSTLE_SCORE = z(DREB_CHANCES) + z(DREB_CHANCE_PCT) + 0.5*z(DEFLECTIONS) + 0.5*z(CONTESTED_SHOTS)
```

### On-ball exclusivity (NEW in v3.0)
```
onball_exclusivity = (pct_guards * 2.0 - pct_forwards * 0.5 - pct_centers * 1.5).clip(0, 2)
onball_pctl = onball_exclusivity.rank(pct=True)
```

### Position balance (NEW in v3.0)
```
max_pos_share = max(pct_guards, pct_forwards, pct_centers)
pos_balance_pctl = (1 - max_pos_share).rank(pct=True)
```

### Percentiles computed
All percentiles rank across all qualified players in a season:
`versatility_pctl`, `difficulty_pctl`, `elite_matchup_pctl`, `stl_pctl`, `blk_pctl`,
`rim_fg_pctl`, `rim_fga_pctl`, `d_results_pctl`, `reb_pctl`, `guard_pctl`,
`forward_pctl`, `center_pctl`, `speed_pctl`, `height_pctl`,
`deflections_pctl`, `contested_shots_pctl`, `contested_3pt_pctl`,
`charges_pctl`, `loose_balls_pctl`, `boxouts_pctl`,
`engagement_pctl`, `hustle_pctl`, `onball_pctl`, `pos_balance_pctl`

---

## 3. Classification Method (v3.0 — Score-Based + Masks + Repulsion)

### Step 1: Low-Activity Filter (Sliding Gate)
The engagement threshold slides based on results quality:

| d_results_pctl | max engagement_pctl gate |
|---|---|
| < 0.08 | Always Low-Activity |
| < 0.10 | 0.95 |
| < 0.15 | 0.50 |
| < 0.25 | 0.30 |
| < 0.40 | 0.18 (original) |
| ≥ 0.40 | Never from this gate |

Players caught are classified as **Low-Activity Defender** with secondary "Liability" (d_results ≤ 0.20) or "Hidden".

### Step 2: Role Affinity Scores (0–1)
Seven weighted percentile sums:

| Score | Formula | Key Drivers |
|-------|---------|-------------|
| **poa_score** | `elite_matchup_pctl*0.25 + difficulty_pctl*0.20 + onball_pctl*0.20 + guard_pctl*0.15 + deflections_pctl*0.10 + d_results_pctl*0.10` | On-ball exclusivity, elite matchups |
| **wing_score** | `forward_pctl*0.25 + difficulty_pctl*0.20 + d_results_pctl*0.25 + contested_shots_pctl*0.10 + (1-guard_pctl)*0.10 + (1-center_pctl)*0.10` | Forward defense + results |
| **chaser_score** | `stl_pctl*0.25 + deflections_pctl*0.20 + speed_pctl*0.10 + (1-difficulty_pctl)*0.20 + loose_balls_pctl*0.10 + d_results_pctl*0.15` | Steals + deflections + speed |
| **versatile_score** | `versatility_pctl*0.30 + pos_balance_pctl*0.20 + difficulty_pctl*0.15 + d_results_pctl*0.20 + contested_shots_pctl*0.15` | Switch-everything + results |
| **rim_score** | `blk_pctl*0.30 + (1-rim_fg_pctl)*0.25 + rim_fga_pctl*0.15 + center_pctl*0.15 + contested_shots_pctl*0.10 + height_pctl*0.05` | Blocks + rim suppression |
| **drop_big_score** | `(1-versatility_pctl)*0.25 + center_pctl*0.25 + reb_pctl*0.20 + rim_fga_pctl*0.15 + boxouts_pctl*0.10 + (1-guard_pctl)*0.05` | Interior + low versatility |
| **mobile_big_score** | `versatility_pctl*0.20 + center_pctl*0.20 + reb_pctl*0.15 + d_results_pctl*0.15 + height_pctl*0.10 + forward_pctl*0.10 + contested_shots_pctl*0.10` | Versatile big + results |

### Step 3: Results Damping
All scores are multiplied by a damping factor based on defensive results:
```
results_factor = max(0.35, 0.40 + 0.60 * d_results_pctl)
```
This suppresses all scores for bad defenders, creating better separation.

### Step 4: Positional Feasibility Masks
Soft multipliers (never hard gates) applied based on player's **own bio position** and **matchup assignment shares**:

| Condition | Score Adjustments |
|---|---|
| Guard (own_pos) | rim_score ×0.10, drop_big ×0.05, mobile_big ×0.15 |
| Short guard (< 6'5") | Additional rim ×0.05, drop ×0.02, mobile ×0.05 |
| Center (own_pos) | poa_score ×0.15, chaser ×0.30 |
| Center + low guard share (< 0.20) | poa_score ×0.10 |
| Wing-big (own_pos) | poa_score ×0.40, chaser ×0.60 |
| Low guard share (< 0.30) | poa_score ×0.50 |
| Low center share (< 0.15) | rim ×0.30, drop ×0.20, mobile ×0.25 |
| Tall player (≥ 6'8") | poa_score ×0.50, chaser ×0.70 |
| Guard + not tall | wing_score ×0.50 |

### Step 5: Role Repulsion
High affinity for one role suppresses incompatible roles:

| Trigger | Suppressed Scores |
|---|---|
| rim_score > 0.60 | poa ×(1-excess*0.5), chaser ×(1-excess*0.5) |
| poa_score > 0.55 | rim ×(1-excess*0.5), drop ×(1-excess*0.5), mobile ×(1-excess*0.5) |
| chaser_score > 0.55 | wing ×(1-excess*0.4) |
| drop_big_score > 0.55 | versatile ×(1-excess*0.5), mobile ×(1-excess*0.5) |
| versatile_score > 0.60 | drop ×(1-excess*0.3), wing ×(1-excess*0.3) |

### Step 6: Assignment
- **Primary archetype** = argmax(masked + repulsed scores)
- **Confidence** = 0.40 + (top − second) / top × 0.55
- **Secondary tag** = context-dependent modifier
- **Post-classification override**: if effectiveness < 0.15 → Low-Activity (Liability)

---

## 4. Archetype Definitions

Primary defensive archetypes (8 total):

1. **POA Defender** — Guards primary ball handlers exclusively; high elite matchup share, high on-ball exclusivity, guards guards
2. **Wing Stopper** — Defends forwards/wings; strong results, high contested shots
3. **Off-Ball Chaser** — High steals/deflections, fast, lower matchup difficulty (roaming/off-ball pressurer)
4. **Versatile Defender** — Switches across all positions effectively; balanced position shares, high switch score (NEW in v3.0)
5. **Rim Protector** — Elite blocks + rim FG suppression; interior anchor
6. **Dropping Big** — Interior-only, low versatility, stays in paint
7. **Mobile Big** — Versatile big who switches across positions; must be an actual big (height/position required)
8. **Low-Activity Defender** — Engagement/results filter: poor results + low engagement, or extremely poor results regardless

### Secondary Tags (modifiers)
- `Switchable` — High versatility, guards multiple positions
- `Lockdown` — Elite defensive results
- `Ball Hawk` — Elite steal/deflection rate
- `Hustler` — High hustle score
- `Shot Blocker` — Elite block rate
- `Primary` — Default; does the role without notable modifier
- `Active Hands` — Good steal/deflection rate (Off-Ball Chaser default)
- `Interior` — Paint-focused
- `Help` — Default for bigs
- `Hidden` — Low-Activity but not terrible results
- `Liability` — Low-Activity with poor results

---

## 5. Defensive Effectiveness & Fit (NEW in v3.0)

### Defensive Effectiveness (0–1)
Archetype-specific weighted metric combining results, hustle, and role-relevant stats:

| Archetype | Formula |
|---|---|
| POA Defender | 0.40·d_results + 0.20·stl_pctl + 0.20·deflections + 0.20·contested_shots |
| Wing Stopper | 0.50·d_results + 0.20·contested_shots + 0.15·deflections + 0.15·hustle |
| Off-Ball Chaser | 0.25·d_results + 0.35·stl_pctl + 0.25·deflections + 0.15·hustle |
| Versatile | 0.40·d_results + 0.20·versatility + 0.20·contested_shots + 0.20·deflections |
| Rim Protector | 0.30·d_results + 0.35·blk_pctl + 0.20·(1-rim_fg_pctl) + 0.15·contested_shots |
| Dropping Big | 0.35·d_results + 0.25·blk_pctl + 0.20·reb_pctl + 0.20·contested_shots |
| Mobile Big | 0.35·d_results + 0.25·versatility + 0.20·contested_shots + 0.20·hustle |

### Defensive Fit (tier)
| Tier | Effectiveness Threshold |
|---|---|
| Elite | ≥ 0.75 |
| Good | ≥ 0.55 |
| Average | ≥ 0.35 |
| Poor | < 0.35 |

---

## 6. Output Schema

Key output fields in `data/processed/defensive_archetypes_v2.parquet`:

| Column | Description |
|---|---|
| `defensive_archetype` | Primary defensive archetype (8 types) |
| `defensive_secondary` | Secondary modifier tag |
| `defensive_confidence` | 0–0.95 confidence score |
| `defensive_effectiveness` | 0–1 archetype-specific results quality |
| `defensive_fit` | Elite / Good / Average / Poor |
| `assignment_difficulty` | High / Medium / Low |
| `switch_score` | Versatility metric |
| `versatility_pctl` | Switch score percentile |
| `avg_opponent_ppg` | Matchup difficulty proxy |
| `elite_matchup_pct` | Share vs elite scorers |
| `STL_PER100_DEF_POSS` | Steals per 100 defensive possessions |
| `BLK_PCT` | Per-game blocks / per-game rim FGA (0–1) |
| `DEF_RIM_FG_PCT` | Opponent FG% at rim when defending |
| `RIM_FGA_RATE` | Rim FGA rate (per game) |
| `DEFLECTIONS` | Per-game deflections |
| `CONTESTED_SHOTS` | Per-game contested shots |
| `engagement_score` | Workload + movement composite |
| `hustle_score` | Rebounding + hustle composite |
| `D_FG_DIFF` | Opponent FG% impact (PCT_PLUSMINUS_Overall) |
| `d_results_pctl` | Defensive results percentile |
| `poa_score` ... `mobile_big_score` | 7 role affinity scores (0–1) |

---

## 7. Repro & Commands

```bash
# Run the defensive archetype pipeline
.venv/bin/python src/data_compute/compute_defensive_archetypes_v2.py

# Fetch hustle stats (prerequisite)
.venv/bin/python src/data_fetch/fetch_defensive_metrics.py
```

Outputs:
- `data/processed/defensive_archetypes_v2.parquet`
- `data/processed/defensive_archetypes_v2.csv`

---

## 8. Known Limitations & Future Work

- **Trae Young / Cam Thomas edge cases**: Players with moderate engagement but poor results may not be caught by Low-Activity gate. The `defensive_fit=Poor` label communicates this.
- **Luka Dončić**: D_FG_DIFF classifies him as average (d_results ~0.55), not terrible. May need EPM/RAPM-calibrated defensive metric.
- Add RAPM/DRAPM residuals to calibrate effectiveness beyond D_FG_DIFF alone.
- Add synergy defensive POSS_PCT for PR Ball Handler defense as POA signal.
- Add rim deterrence proxies (on/off rim attempt rates).
- Normalize within positional clusters (guards / wings / bigs) for tighter comparisons.
- Consider scheme-specific adjustments (drop coverage vs switch-everything).

Last updated: 2026-02-11
