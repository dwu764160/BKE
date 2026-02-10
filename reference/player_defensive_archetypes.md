# Player Defensive Archetypes — v1.0 Logic Documentation

## Overview

This document is the authoritative reference for the defensive archetype classifier implemented in [src/data_compute/compute_defensive_archetypes.py](src/data_compute/compute_defensive_archetypes.py).
The model classifies players by defensive role using matchup data, tracking data, defensive synergy playtypes, and box-score proxies.

Defensive archetypes are intended to provide **role context** (how a player defends) rather than a single "good/bad" score. The classifier produces a primary archetype, optional secondary tag, and a confidence score.

---

## 1. Data Sources

1a. Matchup Versatility (from matchup pipeline)
- `matchup_versatility.parquet`
- Key fields: `switch_score`, `positions_guarded`, `pct_guards`, `pct_forwards`, `pct_centers`

1b. Matchup Difficulty (from matchup pipeline)
- `matchup_difficulty.parquet`
- Key fields: `avg_opponent_ppg`, `elite_matchup_pct`

1c. Defensive Tracking (NBA tracking)
- `defense_Overall.parquet`, `defense_3Pointers.parquet`, `defense_LessThan6Ft.parquet`
- Key fields: `D_FG_PCT_*`, `PCT_PLUSMINUS_*`, `FREQ_*`

1d. Defensive Synergy (NBA.com playtypes)
- `synergy_Defensive_{Playtype}.parquet`
- Key fields: `DEF_{PLAYTYPE}_PPP`, `DEF_{PLAYTYPE}_POSS`, `DEF_{PLAYTYPE}_PCT`

1e. Box Score (season totals)
- `complete_player_season_stats.parquet`
- Key fields: `STL`, `BLK`, `DREB`, `PF`, `MIN`, `GP`

---

## 2. Feature Engineering

Per-36 conversions:

```
minutes_factor = 36 / (MIN / GP)
STL_PER36 = (STL / GP) * minutes_factor
BLK_PER36 = (BLK / GP) * minutes_factor
DREB_PER36 = (DREB / GP) * minutes_factor
```

Defensive composite signals:

```
RIM_PROTECTION_SCORE = BLK_PER36 * 2 + abs(PCT_PLUSMINUS_LessThan6Ft) * 50
PERIMETER_D_SCORE = STL_PER36 * 2 + abs(PCT_PLUSMINUS_3Pointers) * 50 + abs(PCT_PLUSMINUS_Overall) * 30
```

Defensive playtype summary:

```
AVG_DEF_PPP = mean(DEF_*_PPP)  # lower is better
```

---

## 3. Thresholds (v1.0)

Static thresholds are used (not percentile-based) to keep the model stable across seasons:

| Threshold | Value | Purpose |
|---|---:|---|
| `MIN_MINUTES` | 500 | Minimum minutes to qualify |
| `MIN_GP` | 20 | Minimum games |
| `HIGH_SWITCH_SCORE` | 0.65 | High versatility |
| `VERY_HIGH_SWITCH_SCORE` | 0.80 | Elite switcher |
| `HIGH_DIFFICULTY` | 15.0 | Avg opponent PPG gate |
| `ELITE_MATCHUP_PCT` | 0.30 | Elite matchup share |
| `GOOD_D_FG_PCT_DIFF` | -0.02 | Opponent FG% impact |
| `ELITE_D_FG_PCT_DIFF` | -0.04 | Elite FG% impact |
| `HIGH_BLOCKS` | 1.5 | BLK/36 gate |
| `ELITE_BLOCKS` | 2.0 | Elite BLK/36 |
| `HIGH_STEALS` | 1.2 | STL/36 gate |
| `ELITE_STEALS` | 1.8 | Elite STL/36 |

---

## 4. Archetype Definitions

Primary defensive archetypes:

1. **Lockdown Perimeter Defender** — Elite perimeter metrics + high difficulty matchups
2. **Switchable Defender** — Very high versatility, effective across positions
3. **Rim Protector** — Elite rim deterrence (blocks + rim FG% impact)
4. **Help/Rotation Defender** — High rim + high perimeter signals
5. **Point-of-Attack Defender** — Guards primary ballhandlers, high steals
6. **Wing Stopper** — Perimeter stopper with forward assignments
7. **Post Defender** — Interior-oriented defensive role vs centers/PFs
8. **Scheme Defender** — Solid but needs scheme support
9. **Defensive Liability** — Poor defensive impact signals
10. **Neutral Defender** — Default when no strong signals

Secondary tags:
- `Versatile`, `Rim Protector`, `Perimeter`, `Primary Stopper`, `Rim Help`, `Perimeter Help`

---

## 5. Classification Hierarchy

```
1. Insufficient Minutes (MIN < 500 or GP < 20)
2. Lockdown Perimeter Defender (elite perimeter + high difficulty)
3. Switchable Defender (elite versatility + strong rim/perimeter)
4. Rim Protector (elite rim metrics)
5. Help/Rotation Defender (high rim + high perimeter)
6. Point-of-Attack Defender (guards >50% + high perimeter)
7. Wing Stopper (forwards >40% + high perimeter)
8. Post Defender (centers >40% + high rim)
9. Scheme Defender (high rim or high perimeter)
10. Defensive Liability or Neutral Defender
```

---

## 6. Output Schema

Key output fields in `data/processed/defensive_archetypes.parquet`:

| Column | Description |
|---|---|
| `PLAYER_ID` | NBA player id |
| `PLAYER_NAME` | name |
| `SEASON` | season string |
| `defensive_archetype` | primary defensive archetype |
| `defensive_secondary` | optional subtype |
| `defensive_confidence` | 0–1 confidence score |
| `switch_score` | versatility metric |
| `avg_opponent_ppg` | matchup difficulty proxy |
| `elite_matchup_pct` | share vs elite scorers |
| `STL_PER36` | steals per 36 |
| `BLK_PER36` | blocks per 36 |
| `PCT_PLUSMINUS_Overall` | opponent FG% impact |

---

## 7. Repro & Commands

Run the defensive archetype pipeline:

```bash
.venv/bin/python src/data_compute/compute_defensive_archetypes.py
```

Outputs:
- `data/processed/defensive_archetypes.parquet`
- `data/processed/defensive_archetypes.csv`

---

## 8. Notes & Next Steps

- Consider converting static thresholds to percentile gates for better season-to-season adaptation.
- Integrate defensive synergy PPP into tier calculations (currently used only as a summary).
- Add on/off defensive impact (from RAPM/DRAPM) as a calibration signal for confidence scores.

Last updated: 2026-02-09