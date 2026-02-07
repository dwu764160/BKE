# Player Offensive Archetypes — v2 Logic Documentation

## Overview

This document explains the **complete logic** behind the v2 offensive archetype classification system (`src/data_compute/compute_player_archetypes.py`). The system classifies every NBA player-season into one of **10 primary archetypes** (plus 1 special designation) based on Synergy playtype data, tracking data, and box score stats.

### v2 Changes from v1

| Area | v1 | v2 |
|---|---|---|
| Ball dominance | ISO + PnRBH + PostUp (equal weight) | ON_BALL (ISO + PnRBH) + PostUp × 0.65 |
| Interior vs Perimeter | Drives-based ratio (broken — P95 = 0.17) | FG2A_RATE = (FGA − FG3A) / FGA |
| Scoring gate | Raw PPG ≥ 15 | PTS_PER36 ≥ 18.0 (per-36 normalized) |
| Efficiency | Not considered | TS% z-score modulates confidence |
| Playmaking | AST/36 only | AST/36 + SecAST × 0.5 + PotAST × 0.3 − TOV × 0.5 |
| Per-36 inflation | No guard | MPG ≥ 15.0 filter |
| Confidence | Linear `min(1.0, x)` (ceiling compression) | `sqrt` saturation for headroom |
| Archetypes | 7 + 2 special | 10 + 1 special (added Connector, Ballhandler, All-Around) |
| Off-ball thresholds | CUT+PNRRM ≥ 0.10, SPOTUP ≥ 0.20 | CUT+PNRRM ≥ 0.20, SPOTUP ≥ 0.25 |
| Movement Shooter | Any movement ≥ 0.10 | Movement ≥ 0.10 AND movement > spot-up |
| USG% | From box score (NaN for 2024-25) | Computed from raw data when missing |
| TS% | From box score (NaN for 2024-25) | Computed as PTS / (2 × (FGA + 0.44 × FTA)) |

---

## 1. Data Sources

We merge **three** data sources per player per season:

### 1a. Synergy Playtype Data (11 official playtypes from NBA.com)

| Playtype | Column Name | Meaning |
|---|---|---|
| Isolation | `ISOLATION_POSS_PCT` | % of possessions in isolation |
| PnR Ball Handler | `PRBALLHANDLER_POSS_PCT` | % as PnR ball handler |
| Post Up | `POSTUP_POSS_PCT` | % posting up |
| Cut | `CUT_POSS_PCT` | % on cuts |
| PnR Roll Man | `PRROLLMAN_POSS_PCT` | % as PnR screener/roller |
| Handoff | `HANDOFF_POSS_PCT` | % on handoffs |
| Off Screen | `OFFSCREEN_POSS_PCT` | % off screens |
| Spot Up | `SPOTUP_POSS_PCT` | % spot-up |
| Transition | `TRANSITION_POSS_PCT` | % in transition |
| Putback | `OFFREBOUND_POSS_PCT` | % on offensive rebound putbacks |
| Misc | `MISC_POSS_PCT` | % miscellaneous |

All `_POSS_PCT` values are **fractions (0.0–1.0)**. Example: `ISOLATION_POSS_PCT = 0.187` = 18.7%.

### 1b. Tracking Data

| Category | Key Columns |
|---|---|
| Drives | `DRIVES`, `DRIVE_PTS`, `DRIVE_FG_PCT`, `DRIVE_AST`, `DRIVE_TOV` |
| Passing | `SECONDARY_AST`, `POTENTIAL_AST`, `AST_POINTS_CREATED` |
| Possessions | `TOUCHES`, `TIME_OF_POSS`, `AVG_SEC_PER_TOUCH`, `AVG_DRIB_PER_TOUCH` |
| Catch & Shoot | `CATCH_SHOOT_FG3_PCT`, `CATCH_SHOOT_FGA`, etc. |
| Rebounding | `OREB_CONTEST`, `DREB_CONTEST` |
| Speed/Distance | `DIST_MILES`, `AVG_SPEED` |

### 1c. Box Score Data

Standard stats from `complete_player_season_stats.parquet`: `GP`, `MIN`, `PTS`, `AST`, `REB`, `FGA`, `FG3A`, `FG3M`, `FTA`, `FTM`, `USG_PCT`, `TS_PCT`, etc.

---

## 2. Feature Engineering (Derived Columns)

### Per-36 Stats + MPG Guard

```
MPG = MIN / GP
minutes_factor = 36 / MPG
PTS_PER36 = (PTS / GP) * minutes_factor
AST_PER36 = (AST / GP) * minutes_factor
(etc.)
```

**Per-36 inflation guard**: Players with `MPG < 15.0` are classified as "Insufficient Minutes" regardless of other stats. This prevents role players who play 8 MPG from appearing as 25 PTS/36 scorers.

### Ball Dominance (Split Formula)

```
ON_BALL_CREATION = ISOLATION_POSS_PCT + PRBALLHANDLER_POSS_PCT
POST_CREATION = POSTUP_POSS_PCT
BALL_DOMINANT_PCT = ON_BALL_CREATION + POST_CREATION * 0.65
```

**Why 0.65 weight for post-ups?** A center posting up is not the same kind of "ball dominance" as a guard running ISO or PnR. Post-ups are partially set up by the team (entry passes, positioning) whereas ISO/PnR are true self-creation. The 0.65 weight ensures post-heavy bigs don't get inflated ball-dominance scores.

**Example**: Luka 2024-25 → `(0.187 + 0.340) + 0.046 * 0.65 = 0.557` (55.7%)

### Playmaking Score (with Turnover Penalty)

```
PLAYMAKING_SCORE = AST_PER36 × 1.0
                 + SECONDARY_AST_PER36 × 0.5
                 + POTENTIAL_AST_PER36 × 0.3
                 − TOV_PER36 × 0.5
```

The turnover penalty prevents careless ball handlers from inflating their playmaking tier. Note that the **classification gates use raw AST_PER36**, not the composite — the composite is used for confidence calculation only.

### Shot Profile: FG2A_RATE

```
FG2A_RATE = (FGA − FG3A) / FGA
```

This replaces the v1 `INTERIOR_RATIO` which was based on `DRIVES_PER36 / (DRIVES_PER36 + FG3A_PER36)`. That formula was broken because tracking DRIVES values are extremely small (P95 = 0.17 per 36), making the ratio useless — nobody ever hit the 0.60 threshold.

`FG2A_RATE` directly measures what fraction of shot attempts are 2-pointers:
- **Gobert**: 1.000 (never shoots 3s)
- **Giannis**: 0.952 (almost all 2s)
- **Kuminga**: 0.734 (mostly interior)
- **KD**: 0.670 (balanced, leans interior)
- **Ant Edwards**: 0.500 (perfectly balanced)
- **Curry**: 0.377 (mostly 3s)

### Efficiency: TS% and USG% Computation

For 2024-25 (and any season with missing values), TS% and USG% are computed from raw data:

```
TS% = PTS / (2 × (FGA + 0.44 × FTA))

USG% (proxy) = (FGA + 0.44 × FTA + TOV) × 2.4 / (MIN × 5)
```

The USG proxy assumes ~100 pace and divides by 5 (5 players sharing possessions) to produce a standard 0-1 fraction where 0.20 = 20% = league average.

**TS Z-Score**: `(TS_PCT − league_avg) / league_std` — used to modulate confidence. Elite efficiency boosts confidence up to 10%; poor efficiency penalizes up to 20%.

### Off-Ball Composites

| Feature | Formula | Purpose |
|---|---|---|
| `CUT_PNRRM_PCT` | `CUT_POSS_PCT + PRROLLMAN_POSS_PCT` | Off-ball finishing frequency |
| `MOVEMENT_SHOOTER_PCT` | `HANDOFF_POSS_PCT + OFFSCREEN_POSS_PCT` | Movement shooting frequency |
| `SPOTUP_PCT` | `SPOTUP_POSS_PCT` | Spot-up frequency |
| `TRANSITION_PCT` | `TRANSITION_POSS_PCT` | Transition frequency |
| `PUTBACK_PCT` | `OFFREBOUND_POSS_PCT` | Putback frequency |

### Efficiency Metrics (Informational)

```
EFG% = (FGM + 0.5 × FG3M) / FGA
TOV% = TOV / (FGA + 0.44 × FTA + TOV)
FT_RATE = FTA / FGA
```

---

## 3. Classification Thresholds

| Threshold | Value | Meaning |
|---|---|---|
| `BALL_DOMINANT_PCT` | **0.18** | Min ball dominance (raised from 0.15) |
| `HIGH_BALL_DOMINANT_PCT` | **0.30** | Very high ball dominance (raised from 0.25) |
| `HIGH_PLAYMAKING` | **5.5** AST/36 | High playmaker (raised from 5.0) |
| `VERY_HIGH_PLAYMAKING` | **7.5** AST/36 | Elite playmaker (raised from 7.0) |
| `CONNECTOR_PLAYMAKING` | **3.5** AST/36 | Moderate playmaking for Connector |
| `HIGH_SCORING_PER36` | **18.0** PTS/36 | High scorer (~P70, replaces 15 PPG) |
| `LOW_SCORING_PER36` | **10.0** PTS/36 | Low scorer |
| `HIGH_USAGE` | **0.22** | High usage rate (22%) |
| `LOW_USAGE` | **0.15** | Low usage rate (15%) |
| `INTERIOR_HEAVY` | **0.70** FG2A_RATE | Interior shot profile |
| `PERIMETER_HEAVY` | **0.45** FG2A_RATE | Perimeter shot profile |
| `HIGH_CUT_PNRRM` | **0.20** | Off-ball finishing (raised from 0.10) |
| `HIGH_SPOTUP` | **0.25** | Spot-up frequency (raised from 0.20) |
| `HIGH_MOVEMENT` | **0.10** | Movement shooting |
| `MIN_MINUTES` | **500** | Minimum season minutes |
| `MIN_GP` | **20** | Minimum games played |
| `MIN_MPG` | **15.0** | Minimum MPG (per-36 guard) |
| `HIGH_EFFICIENCY` | **0.58** TS% | Above-average efficiency |
| `LOW_EFFICIENCY` | **0.52** TS% | Below-average efficiency |

---

## 4. Classification Hierarchy

The system uses a **strict priority hierarchy**. Once a player matches a tier, classification returns immediately.

### Step 0: Minimum Requirements

```
IF MIN < 500 OR GP < 20 OR MPG < 15.0 → "Insufficient Minutes"
```

### Step 1: Tier Assignments

Each player gets bucketed into tiers before classification:

- **Ball Dominance**: Low (< 0.18), High (0.18–0.30), Very High (> 0.30)
- **Playmaking**: Low (< 5.5 AST/36), High (5.5–7.5), Elite (> 7.5)
- **Scoring**: Low (< 10 PTS/36), Medium (10–18), High (> 18)
- **Efficiency**: Low (TS ≤ 0.52), Average (0.52–0.58), High (TS ≥ 0.58)

### Step 2: Classification Logic

#### 1️⃣ Ball Dominant Creator

```
IF ball_dominant (≥ 0.18) AND playmaker (≥ 5.5 AST/36):
    → Ball Dominant Creator
    confidence = min(1.0, (√(ball_dom/0.35) + √(playmaking/10)) / 2 × eff_factor)
    IF high_scorer (≥ 18 PTS/36): secondary = "Primary Scorer"
```

**Examples**: Luka, Jokić, LeBron, Trae Young, Haliburton

#### 1b️⃣ Offensive Hub (sub-path of BDC)

```
IF playmaker (≥ 5.5 AST/36) AND high_scorer (≥ 18 PTS/36) AND NOT ball_dominant:
    → Ball Dominant Creator / Offensive Hub
    confidence = min(1.0, (√(ast36/8) + √(pts36/25)) / 2 × eff_factor)
```

Catches high-scoring playmakers who create through non-ISO/PnR means — post-up hubs, passing bigs, etc.

**Examples**: Sabonis (ball_dom=0.107 but AST/36=6.3, PTS/36=19.8), Vucevic

#### 2️⃣ Ballhandler (Facilitator)

```
IF playmaker (≥ 5.5 AST/36) AND NOT high_scorer:
    → Ballhandler
    confidence = min(1.0, √(playmaking/10) × eff_factor)
```

Pure facilitators who distribute more than they score. BDC already captured ball_dom + playmaker combinations, so this only catches non-ball-dominant or non-scoring playmakers.

**Examples**: Draymond Green (AST/36=7.0, PTS/36=8.5)

#### 3️⃣ Primary Scorers (Ball Dominant, Not Playmaker)

Ball-dominant players who don't distribute enough to be BDC/Ballhandler. Split by shot profile:

```
IF ball_dominant (≥ 0.18) AND NOT playmaker:
    IF FG2A_RATE ≥ 0.70:   → Interior Scorer
    ELIF FG2A_RATE ≤ 0.45: → Perimeter Scorer
    ELSE:                    → All-Around Scorer
    IF high_scorer: secondary = "High Volume"
```

| Archetype | FG2A_RATE | Examples |
|---|---|---|
| Interior Scorer | ≥ 0.70 | Kuminga (0.734), Zion (0.87) |
| Perimeter Scorer | ≤ 0.45 | Buddy Hield, Duncan Robinson |
| All-Around Scorer | 0.45–0.70 | KD (0.67), Ant Edwards (0.50) |

**Key insight**: The `INTERIOR_HEAVY = 0.70` threshold was carefully tuned. At 0.60 (v1), KD (0.67) would have been classified as Interior Scorer — clearly wrong. At 0.70, KD falls to All-Around while Kuminga (0.73), Giannis (0.95), and Zion correctly map to Interior.

#### 4️⃣ Connector

```
IF AST/36 ≥ 3.5 AND NOT ball_dominant AND NOT high_scorer:
    IF has_connector_profile (sec_ast ≥ 0.3 OR AST/36 ≥ 4.0 OR ...):
        → Connector
        confidence = min(1.0, √(ast36/6) × eff_factor)
```

Players who facilitate without ball dominance or volume scoring. The "connector profile" check ensures we don't catch accidental passers.

**Examples**: Derrick White, Jrue Holiday, Kyle Anderson

#### 5️⃣ Off-Ball Finisher

```
offball_finish_score = CUT_PNRRM_PCT + PUTBACK × 0.5 + TRANSITION × 0.3

IF CUT_PNRRM ≥ 0.20 OR finish_score > 0.20:
    → Off-Ball Finisher
    IF TRANSITION > 0.10: secondary = "Transition Player"
```

**Threshold rationale**: Raised from 0.10 → 0.20 for CUT_PNRRM because at 0.10, ~35% of players qualified (too broad). At 0.20, ~27% qualify (still the largest bucket but more meaningful).

**Examples**: Gobert, Clint Capela, Derrick Jones Jr.

#### 6️⃣ Off-Ball Movement Shooter

```
IF MOVEMENT_SHOOTER_PCT ≥ 0.10 AND movement > spotup:
    → Off-Ball Movement Shooter
```

The `movement > spotup` requirement prevents overlap with Stationary Shooter. Players who run off screens and through handoffs more than they spot up.

**Examples**: Rare archetype (~2 per season) — players like Klay Thompson in his prime

#### 7️⃣ Off-Ball Stationary Shooter

```
IF SPOTUP_PCT ≥ 0.25:
    → Off-Ball Stationary Shooter
    IF FG3_PCT > 0.38: secondary = "Elite Shooter"
```

**Examples**: Batum, PJ Tucker, Kentavious Caldwell-Pope

#### 8️⃣ Rotation Piece (Catch-All)

```
→ Rotation Piece
Secondary: "Rebounder" / "Perimeter Defender" / "Rim Protector" / "Glue Guy"
confidence = min(0.65, 0.55 × eff_factor)
```

---

## 5. Confidence Calculation

All v2 confidence formulas use **sqrt saturation** instead of v1's linear `min(1.0, x)`:

```
raw_conf = √(metric / denominator)
confidence = min(1.0, raw_conf × efficiency_factor)
```

**Why sqrt?** Linear formulas like `ball_dom / 0.25` max out at 1.0 quickly — a player with ball_dom=0.50 gets the same confidence as ball_dom=0.80. Sqrt provides better discrimination across the range while still having a natural ceiling.

### Efficiency Factor

```
IF TS ≥ 0.58 (High):    efficiency_factor = min(1.10, 1.0 + TS_ZSCORE × 0.05)
IF TS ≤ 0.52 (Low):     efficiency_factor = max(0.80, 1.0 + TS_ZSCORE × 0.05)
ELSE (Average):          efficiency_factor = 1.0
```

This modulates confidence by ±5-20% based on efficiency. A highly efficient scorer gets a small confidence boost; an inefficient one gets penalized. It does NOT change the archetype — just the confidence.

| Archetype | Confidence Formula |
|---|---|
| Ball Dominant Creator | `(√(ball_dom/0.35) + √(playmaking/10)) / 2` |
| Offensive Hub | `(√(ast36/8) + √(pts36/25)) / 2` |
| Ballhandler | `√(playmaking/10)` |
| Interior Scorer | `√(ball_dom/0.25) × (fg2a_rate/0.75)` |
| Perimeter Scorer | `√(ball_dom/0.25) × ((1−fg2a_rate)/0.65)` |
| All-Around Scorer | `√(ball_dom/0.25)` |
| Connector | `√(ast36/6)` |
| Off-Ball Finisher | `√(finish_score/0.25)` |
| Off-Ball Movement Shooter | `√(movement/0.18)` |
| Off-Ball Stationary Shooter | `√(spotup/0.35)` |
| Rotation Piece | Fixed 0.55 |

---

## 6. Archetype Distribution (2024-25, 329 qualified)

| Archetype | Count | % |
|---|---|---|
| Off-Ball Finisher | 89 | 27.1% |
| Ball Dominant Creator | 65 | 19.8% |
| All-Around Scorer | 64 | 19.5% |
| Off-Ball Stationary Shooter | 55 | 16.7% |
| Interior Scorer | 19 | 5.8% |
| Perimeter Scorer | 15 | 4.6% |
| Connector | 14 | 4.3% |
| Ballhandler | 4 | 1.2% |
| Rotation Piece | 2 | 0.6% |
| Off-Ball Movement Shooter | 2 | 0.6% |

---

## 7. Output

The classification produces `data/processed/player_archetypes.parquet` with all original feature columns plus:

| Column | Description |
|---|---|
| `primary_archetype` | 10 archetypes + "Insufficient Minutes" |
| `secondary_archetype` | Specialization (e.g. "Primary Scorer", "Offensive Hub", "Elite Shooter", "Transition Player") |
| `archetype_confidence` | 0.0–1.0 (sqrt-saturated, efficiency-modulated) |
| `ball_dominance_tier` | "Low", "High", "Very High" |
| `playmaking_tier` | "Low", "High", "Elite" |
| `scoring_tier` | "Low", "Medium", "High" |
| `efficiency_tier` | "Low", "Average", "High", "Unknown" |

Seasons covered: 2022-23, 2023-24, 2024-25.

---

## 8. Classification Flow Diagram

```
Player Season Record
         │
    ┌────▼──────────────────┐
    │ MIN<500 OR GP<20      │──Yes──▶ Insufficient Minutes
    │ OR MPG<15             │
    └────┬──────────────────┘
         │No
    ┌────▼──────────────────────┐
    │ Ball Dom (≥0.18)          │
    │ + Playmaker (≥5.5 AST/36) │──Yes──▶ Ball Dominant Creator
    └────┬──────────────────────┘        (sub: Primary Scorer if PTS/36≥18)
         │No
    ┌────▼──────────────────────┐
    │ Playmaker (≥5.5 AST/36)   │
    │ + High Scorer (≥18 PTS/36)│──Yes──▶ BDC / Offensive Hub
    └────┬──────────────────────┘
         │No
    ┌────▼──────────────────────┐
    │ Playmaker (≥5.5 AST/36)   │
    │ + NOT High Scorer         │──Yes──▶ Ballhandler (Facilitator)
    └────┬──────────────────────┘
         │No
    ┌────▼──────────────────┐         ┌─▶ Interior Scorer (FG2A≥0.70)
    │ Ball Dom (≥0.18)      │──Yes──┬─┤─▶ Perimeter Scorer (FG2A≤0.45)
    │ + NOT Playmaker       │       │ └─▶ All-Around Scorer (0.45-0.70)
    └────┬──────────────────┘       └──── (sub: High Volume if PTS/36≥18)
         │No
    ┌────▼──────────────────┐
    │ AST/36 ≥ 3.5          │
    │ + NOT ball dom         │──Yes──▶ Connector
    │ + NOT high scorer      │         (if sec_ast≥0.3 or AST/36≥4.0)
    │ + connector profile    │
    └────┬──────────────────┘
         │No
    ┌────▼───────────────────────┐
    │ CUT+PnRRm ≥ 0.20          │──Yes──▶ Off-Ball Finisher
    │ OR finish_score > 0.20     │         (sub: Transition if trans>0.10)
    └────┬───────────────────────┘
         │No
    ┌────▼───────────────────────┐
    │ Movement ≥ 0.10            │──Yes──▶ Off-Ball Movement Shooter
    │ AND movement > spotup      │
    └────┬───────────────────────┘
         │No
    ┌────▼───────────────────────┐
    │ SpotUp ≥ 0.25              │──Yes──▶ Off-Ball Stationary Shooter
    └────┬───────────────────────┘         (sub: Elite Shooter if 3P>38%)
         │No
         ▼
    Rotation Piece / Glue Guy
```

---

## 9. Validated Classifications (2024-25)

| Player | Archetype | Sub | Conf | FG2A | Key Metrics |
|---|---|---|---|---|---|
| LeBron James | BDC | Primary Scorer | 99% | 69% | ball_dom=0.43, AST/36≈8 |
| Luka Dončić | BDC | Primary Scorer | 100% | 53% | ball_dom=0.56, AST/36≈8 |
| Nikola Jokić | BDC | Primary Scorer | 98% | 76% | ball_dom=0.39, AST/36≈10 |
| Stephen Curry | BDC | Primary Scorer | 89% | 38% | ball_dom=0.20, AST/36≈6 |
| Sabonis | BDC | Offensive Hub | 96% | 85% | ball_dom=0.11, AST/36=6.3 |
| Kevin Durant | All-Around | High Volume | 100% | 67% | ball_dom=0.27 |
| Ant Edwards | All-Around | High Volume | 100% | 50% | ball_dom=0.19 |
| Kuminga | Interior Scorer | High Volume | 100% | 73% | ball_dom=0.31 |
| Giannis | BDC | Primary Scorer | 93% | 95% | ball_dom=0.47 |
| Draymond Green | Ballhandler | — | 74% | 52% | AST/36=7.0, PTS/36≈8.5 |
| Gobert | Off-Ball Finisher | — | 100% | 100% | cut_pnrrm=high |
| Batum | Stationary Shooter | Elite Shooter | 100% | 15% | spotup=high, 3P>38% |
