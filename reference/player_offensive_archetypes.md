# Player Offensive Archetypes — v3 Logic Documentation

## Overview

This document explains the **complete logic** behind the v3 offensive archetype classification system (`src/data_compute/compute_player_archetypes.py`). The system classifies every NBA player-season into one of **12 primary archetypes** (plus 1 special designation) based on Synergy playtype data, tracking data, and box score stats.

### v3 Changes from v2

| Area | v2 | v3 |
|---|---|---|
| Thresholds | Static values (0.18, 5.5, etc.) | Percentile-based (P50–P95), computed per-season |
| BDC gate | ball_dom ≥ 0.18 + AST/36 ≥ 5.5 | ball_dom ≥ P80 + AST ≥ P75 + PTS ≥ P50 + composite ≥ 0.35 |
| BDC pool | 65 players (2024-25) | 40 players (–38% reduction) |
| All-Around pool | 64 players | 26 players (–59% reduction) |
| BDC subtypes | Primary Scorer / Offensive Hub | Heliocentric Guard / Post Hub / Gravity Engine / Primary Scorer |
| PnR Big | Part of Off-Ball Finisher | Separate: PnR Rolling Big / PnR Popping Big (by FG3A) |
| Confidence | Single `archetype_confidence` | Dual: `role_confidence` (fit) + `role_effectiveness` (production) |
| Movement Shooter | movement > spotup | movement/(movement+spotup) ≥ 0.40 |
| Connector | Basic AST/36 ≥ 3.5 gate | AST ≥ P50 + touches/sec-per-touch activity filter |
| FG2A_RATE | Static 0.70/0.45 | P70 (interior) / P30 (perimeter) |
| USG% | Hard gate at 0.22/0.15 | Soft signal only — never disqualifies |
| Composite | Simple sqrt saturation | Linear interpolation from threshold to P95 |

---

## 1. Data Sources

Same as v2 — three data sources per player per season:

### 1a. Synergy Playtype Data (11 playtypes from NBA.com)

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

Standard stats: `GP`, `MIN`, `PTS`, `AST`, `REB`, `FGA`, `FG3A`, `FG3M`, `FTA`, `FTM`, `USG_PCT`, `TS_PCT`, etc.

---

## 2. Feature Engineering

### Per-36 Stats

```
MPG = MIN / GP
minutes_factor = 36 / MPG
PTS_PER36 = (PTS / GP) × minutes_factor
AST_PER36 = (AST / GP) × minutes_factor
```

### Ball Dominance (Split Formula)

```
ON_BALL_CREATION = ISOLATION_POSS_PCT + PRBALLHANDLER_POSS_PCT
POST_CREATION = POSTUP_POSS_PCT
BALL_DOMINANT_PCT = ON_BALL_CREATION + POST_CREATION × 0.65
```

**Why 0.65 weight?** Post-ups are partially team-set-up plays, unlike pure self-creation (ISO/PnR).

### Playmaking Score

```
PLAYMAKING_SCORE = AST_PER36 × 1.0
                 + SECONDARY_AST_PER36 × 0.5
                 + POTENTIAL_AST_PER36 × 0.3
                 − TOV_PER36 × 0.5
```

### Shot Profile: FG2A_RATE

```
FG2A_RATE = (FGA − FG3A) / FGA
```

### Efficiency: TS% and USG%

```
TS% = PTS / (2 × (FGA + 0.44 × FTA))
USG% (proxy) = (FGA + 0.44 × FTA + TOV) × 2.4 / (MIN × 5)
TS_ZSCORE = (TS_PCT − league_avg) / league_std
```

### Off-Ball Composites

| Feature | Formula |
|---|---|
| `CUT_PNRRM_PCT` | `CUT_POSS_PCT + PRROLLMAN_POSS_PCT` |
| `MOVEMENT_SHOOTER_PCT` | `HANDOFF_POSS_PCT + OFFSCREEN_POSS_PCT` |
| `SPOTUP_PCT` | `SPOTUP_POSS_PCT` |
| `TRANSITION_PCT` | `TRANSITION_POSS_PCT` |
| `PUTBACK_PCT` | `OFFREBOUND_POSS_PCT` |

---

## 3. Percentile-Based Thresholds (v3 Key Innovation)

v3 replaces static thresholds with **per-season percentile thresholds** computed from qualified players (MIN ≥ 500, GP ≥ 20, MPG ≥ 15).

### Percentile Threshold Map

| Threshold Name | Column | Percentile | 2024-25 Value | Purpose |
|---|---|---|---|---|
| `BD_MIN` | BALL_DOMINANT_PCT | P80 | 0.372 | BDC main path min ball dominance |
| `BD_ALL_AROUND` | BALL_DOMINANT_PCT | P60 | 0.224 | All-Around Scorer min ball dominance |
| `BD_BASE` | BALL_DOMINANT_PCT | P50 | 0.178 | Any ball-dom consideration |
| `BD_HELIOCENTRIC` | BALL_DOMINANT_PCT | P85 | 0.413 | Heliocentric Guard subtype |
| `BD_P95` | BALL_DOMINANT_PCT | P95 | 0.510 | Composite normalization cap |
| `PLAYMAKER` | AST_PER36 | P75 | 5.074 | Playmaker flag |
| `ELITE_PLAYMAKER` | AST_PER36 | P85 | 6.138 | Elite playmaker (Hub path) |
| `AST_P95` | AST_PER36 | P95 | 7.867 | Composite normalization cap |
| `HIGH_SCORING` | PTS_PER36 | P70 | 18.95 | High-volume scorer |
| `ELITE_SCORING` | PTS_PER36 | P80 | 21.07 | Elite scoring (Hub path) |
| `MODERATE_SCORING` | PTS_PER36 | P50 | 15.51 | BDC scoring floor |
| `ALL_AROUND_SCORING` | PTS_PER36 | P60 | 17.23 | All-Around volume gate |
| `PTS_P95` | PTS_PER36 | P95 | 25.62 | Composite normalization cap |
| `LOW_SCORING` | PTS_PER36 | P25 | 12.63 | Low scorer |
| `FG2A_INTERIOR` | FG2A_RATE | P70 | 0.673 | Interior shot profile |
| `FG2A_PERIMETER` | FG2A_RATE | P30 | 0.478 | Perimeter shot profile |
| `HIGH_CUT_PNRRM` | CUT_PNRRM_PCT | P75 | 0.154 | Off-ball finisher gate |
| `HIGH_SPOTUP` | SPOTUP_PCT | P50 | 0.213 | Spot-up threshold |
| `HIGH_MOVEMENT` | MOVEMENT_SHOOTER_PCT | P70 | 0.105 | Movement shooter gate |
| `HIGH_PRROLLMAN` | PRROLLMAN_POSS_PCT | P70 | 0.065 | PnR Big gate |
| `FG3A_MEDIAN` | FG3A_PER36 | P50 | 5.813 | PnR pop vs roll split |
| `TOUCHES_MEDIAN` | TOUCHES | P50 | 40.3 | Connector activity |
| `HIGH_FG3A` | FG3A_PER36 | P80 | 8.017 | Gravity Engine 3PA volume |
| `HIGH_FG3_PCT` | FG3_PCT | P70 | 0.378 | Gravity Engine accuracy |
| `HIGH_EFFICIENCY` | TS_PCT | P70 | 0.588 | High efficiency |
| `LOW_EFFICIENCY` | TS_PCT | P30 | 0.540 | Low efficiency |
| `CONNECTOR_AST` | AST_PER36 | P50 | 3.376 | Connector playmaking gate |

### Static Thresholds (Non-Percentile)

| Threshold | Value | Purpose |
|---|---|---|
| `MIN_MINUTES` | 500 | Minimum season minutes |
| `MIN_GP` | 20 | Minimum games played |
| `MIN_MPG` | 15.0 | Per-36 inflation guard |
| `POST_HUB_POSS` | 0.10 | Post Hub minimum (must also dominate) |
| `POST_HUB_DOMINANT` | 0.15 | Post Hub guaranteed (any post ≥ 0.15) |
| `HELIOCENTRIC_ON_BALL` | 0.30 | Heliocentric Guard min ON_BALL |
| `POST_WEIGHT` | 0.65 | Post-up weight in ball dominance |
| `MOVEMENT_RATIO_MIN` | 0.40 | movement/(movement+spotup) min |
| `CONNECTOR_SEC_PER_TOUCH_MAX` | 2.5 | Max avg sec per touch for Connector |

---

## 4. Classification Hierarchy

### Step 0: Minimum Requirements

```
IF MIN < 500 OR GP < 20 OR MPG < 15.0 → "Insufficient Minutes"
```

### Step 1: Ball Dominant Creator (Main Path)

```
gates:
  ball_dom >= P80 (BD_MIN)
  ast_per36 >= P75 (PLAYMAKER)
  pts_per36 >= P50 (MODERATE_SCORING)

composite (linear interpolation):
  bd_score  = (ball_dom  - BD_MIN) / (BD_P95  - BD_MIN)     [0–1]
  pts_score = (pts_per36 - P50)    / (PTS_P95 - P50)        [0–1]
  ast_score = (ast_per36 - P75)    / (AST_P95 - P75)        [0–1]

  volume     = 0.50 × bd_score + 0.50 × pts_score
  confidence = ast_score
  composite  = 0.60 × volume + 0.40 × confidence

  IF composite >= 0.35 → Ball Dominant Creator
```

**BDC Subtypes** (priority order):
1. **Post Hub**: `POSTUP ≥ 0.15` (dominant) OR (`POSTUP ≥ 0.10` AND `POSTUP ≥ ON_BALL`)
2. **Gravity Engine**: `FG3A_PER36 ≥ P80` AND (`TS ≥ P70` OR `FG3% ≥ P70`)
3. **Heliocentric Guard**: `ball_dom ≥ P85` (BD_HELIOCENTRIC)
4. **Primary Scorer**: `pts_per36 ≥ P70`

**Examples**: LeBron (Heliocentric), Luka (Heliocentric), Curry (Gravity Engine), Trae (Heliocentric)

### Step 1b: Offensive Hub (Elite Playmaker + Elite Scorer)

```
ast_per36 >= P85 (ELITE_PLAYMAKER)
pts_per36 >= P80 (ELITE_SCORING)

→ Ball Dominant Creator with full subtype logic
  (Post Hub > Gravity Engine > Offensive Hub)
```

**Examples**: Nikola Jokić (Post Hub), high-scoring playmakers not qualifying via main path

### Step 1c: Playmaking Hub (Elite Playmaker + Post Creator)

```
ast_per36 >= P85 (ELITE_PLAYMAKER)
POSTUP_POSS_PCT >= 0.10
NOT is_ball_dominant_high

→ Ball Dominant Creator / Post Hub
```

**Examples**: Domantas Sabonis (ball_dom=0.11, AST/36=6.3, POST=0.12)

### Step 2: Ballhandler (Facilitator)

```
ast_per36 >= P75 (PLAYMAKER) AND NOT high scorer

→ Ballhandler
```

**Examples**: Draymond Green (AST/36=7.0, PTS/36=8.5)

### Step 3: Primary Scorers

```
ball_dom >= P60 (BD_ALL_AROUND) AND NOT playmaker

IF FG2A_RATE >= P70 → Interior Scorer
ELIF FG2A_RATE <= P30 → Perimeter Scorer
ELSE:
  IF pts_per36 >= P60 (ALL_AROUND_SCORING) → All-Around Scorer
  ELSE → falls through to lower archetypes
```

| Archetype | FG2A_RATE | Examples |
|---|---|---|
| Interior Scorer | ≥ P70 (~0.67) | Kuminga, Zion, Giannis |
| Perimeter Scorer | ≤ P30 (~0.48) | Derrick White, Duncan Robinson |
| All-Around Scorer | P30–P70 + PTS ≥ P60 | KD, Anthony Edwards |

### Step 4: Connector

```
ast_per36 >= P50 (CONNECTOR_AST)
NOT ball_dominant_mid
NOT high_scorer
(touches >= P50 OR sec_per_touch <= 2.5)  ← activity filter
+ playmaking profile check
```

**Examples**: Jrue Holiday, Kyle Anderson

### Step 5: PnR Big (Rolling / Popping)

```
PRROLLMAN_POSS_PCT >= P70 (HIGH_PRROLLMAN)
AND prrollman > cut_poss
AND prrollman >= spotup × 0.5

IF FG3A_PER36 >= P50 → PnR Popping Big
ELSE → PnR Rolling Big
```

**Examples**: Brook Lopez (Rolling), Al Horford (Popping)

### Step 6: Off-Ball Finisher

```
offball_finish_score = CUT_PNRRM + PUTBACK × 0.5 + TRANSITION × 0.3

CUT_PNRRM >= P75 OR finish_score > 0.20

→ Off-Ball Finisher
  sub: "Transition Player" if TRANSITION > 0.15
```

**Examples**: Gobert, Clint Capela

### Step 7: Off-Ball Movement Shooter

```
movement >= P70 (HIGH_MOVEMENT)
AND movement / (movement + spotup) >= 0.40

→ Off-Ball Movement Shooter
```

**Examples**: Klay Thompson, Buddy Hield

### Step 8: Off-Ball Stationary Shooter

```
spotup >= P50 (HIGH_SPOTUP)

→ Off-Ball Stationary Shooter
  sub: "Elite Shooter" if FG3% > 0.38
```

**Examples**: Nicolas Batum (Elite Shooter)

### Step 9: Rotation Piece (Catch-All)

```
→ Rotation Piece
sub: Rebounder / Perimeter Defender / Rim Protector / Glue Guy
```

---

## 5. Dual Confidence System (v3 Key Innovation)

v3 splits the single `archetype_confidence` into two independent metrics:

### Role Confidence (Fit)

**What it measures**: How well the player's profile matches the archetype definition. A player with high fit clearly belongs in their category.

| Archetype | Confidence Formula |
|---|---|
| BDC (main path) | `√(composite / 0.6)` where composite uses linear interpolation |
| BDC (Hub path) | `√(ast/cap) × 0.5 + √(pts/cap) × 0.5` |
| BDC (Playmaking Hub) | `√(ast/cap) × 0.6 + √(post/0.15) × 0.4` |
| Ballhandler | `√(ast/cap)` |
| Interior Scorer | `√(bd/cap) × 0.6 + (fg2a/0.85) × 0.4` |
| Perimeter Scorer | `√(bd/cap) × 0.6 + ((1-fg2a)/0.65) × 0.4` |
| All-Around Scorer | `√(bd/cap) × 0.5 + √(pts/cap) × 0.5` |
| Connector | `√(ast/cap)` |
| PnR Big | `√(prrollman / cap)` |
| Off-Ball Finisher | `√(finish_score / cap)` |
| Movement Shooter | `√(movement / cap)` |
| Stationary Shooter | `√(spotup / cap)` |
| Rotation Piece | Fixed 0.55 |

### Role Effectiveness (Production Quality)

**What it measures**: How productive the player is within their role, independent of classification fit.

| Archetype | Effectiveness Formula |
|---|---|
| Ball Dominant Creator | 35% TS efficiency + 35% volume + 30% primary playtype PPP |
| Interior/Perim/All-Around | 50% TS efficiency + 50% volume |
| Ballhandler | 35% TS + 25% volume + 40% AST quality |
| Connector | 30% TS + 20% volume + 50% connection score |
| PnR Big | 35% TS + 30% volume + 35% PnR roll man PPP |
| Off-Ball Finisher | 35% TS + 30% volume + 35% cut PPP |
| Movement Shooter | 25% TS + 15% vol + 35% FG3% + 25% offscreen PPP |
| Stationary Shooter | 25% TS + 15% vol + 35% FG3% + 25% spotup PPP |

TS efficiency score: `clip((TS_ZSCORE + 2) / 4, 0.05, 1.0)` (normalizes -2σ=0, +2σ=1)

---

## 6. Archetype Distribution (2024-25, 329 Qualified)

| Archetype | Count | % |
|---|---|---|
| Off-Ball Stationary Shooter | 64 | 19.5% |
| Off-Ball Finisher | 62 | 18.8% |
| Ball Dominant Creator | 40 | 12.2% |
| Ballhandler | 31 | 9.4% |
| Connector | 28 | 8.5% |
| All-Around Scorer | 26 | 7.9% |
| PnR Rolling Big | 21 | 6.4% |
| Interior Scorer | 18 | 5.5% |
| Off-Ball Movement Shooter | 17 | 5.2% |
| Rotation Piece | 10 | 3.0% |
| Perimeter Scorer | 7 | 2.1% |
| PnR Popping Big | 5 | 1.5% |

### Subtype Distribution

| Subtype | Count | Context |
|---|---|---|
| Transition Player | 45 | Off-Ball Finisher, PnR Big |
| High Volume | 33 | Interior/Perimeter/All-Around Scorers |
| Heliocentric Guard | 29 | BDC subtype |
| Elite Shooter | 23 | Stationary Shooter |
| Playmaking Big | 15 | Connector subtype |
| Glue Guy | 8 | Rotation Piece |
| Gravity Engine | 6 | BDC subtype |
| Primary Scorer | 3 | BDC subtype |
| Post Hub | 2 | BDC subtype |

---

## 7. Output

The classification produces `data/processed/player_archetypes.parquet` with all feature columns plus:

| Column | Description |
|---|---|
| `primary_archetype` | 12 archetypes + "Insufficient Minutes" |
| `secondary_archetype` | Subtype (Heliocentric Guard, Post Hub, Gravity Engine, High Volume, etc.) |
| `role_confidence` | 0.0–1.0 — how well player fits the archetype definition |
| `role_effectiveness` | 0.0–1.0 — how productive the player is within their role |
| `ball_dominance_tier` | "Low", "Moderate", "High", "Very High" |
| `playmaking_tier` | "Low", "Moderate", "High", "Elite" |
| `scoring_tier` | "Low", "Medium", "High", "Elite" |
| `efficiency_tier` | "Low", "Average", "High" |

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
    ┌────▼──────────────────────────┐
    │ Ball Dom ≥ P80                │
    │ + Playmaker ≥ P75             │
    │ + Scoring ≥ P50               │──Yes──▶ Ball Dominant Creator
    │ + Composite ≥ 0.35            │         Subtypes: Post Hub / Gravity Engine
    └────┬──────────────────────────┘         / Heliocentric Guard / Primary Scorer
         │No
    ┌────▼──────────────────────────┐
    │ AST ≥ P85 (Elite Playmaker)   │
    │ + PTS ≥ P80 (Elite Scorer)    │──Yes──▶ BDC / Hub
    └────┬──────────────────────────┘         (full subtype logic)
         │No
    ┌────▼──────────────────────────┐
    │ AST ≥ P85 + POST ≥ 0.10      │──Yes──▶ BDC / Post Hub
    │ + NOT high ball dom           │
    └────┬──────────────────────────┘
         │No
    ┌────▼──────────────────────────┐
    │ AST ≥ P75 + NOT high scorer   │──Yes──▶ Ballhandler (Facilitator)
    └────┬──────────────────────────┘
         │No
    ┌────▼──────────────────┐         ┌─▶ Interior Scorer (FG2A ≥ P70)
    │ Ball Dom ≥ P60        │──Yes──┬─┤─▶ Perimeter Scorer (FG2A ≤ P30)
    │ + NOT Playmaker       │       │ └─▶ All-Around Scorer (PTS ≥ P60)
    └────┬──────────────────┘       └──── sub: High Volume if PTS ≥ P70
         │No
    ┌────▼──────────────────────┐
    │ AST ≥ P50                 │
    │ + NOT ball dom ≥ P60      │
    │ + NOT high scorer         │──Yes──▶ Connector
    │ + Active (touches/sec)    │         sub: Playmaking Big / Hockey Assist
    └────┬──────────────────────┘
         │No
    ┌────▼──────────────────────┐       ┌─▶ PnR Popping Big (FG3A ≥ P50)
    │ PnR Roll Man ≥ P70       │──Yes─┤
    │ + prrollman > cut_poss   │       └─▶ PnR Rolling Big (FG3A < P50)
    └────┬──────────────────────┘
         │No
    ┌────▼───────────────────────┐
    │ CUT+PnRRm ≥ P75           │──Yes──▶ Off-Ball Finisher
    │ OR finish_score > 0.20     │         sub: Transition Player
    └────┬───────────────────────┘
         │No
    ┌────▼───────────────────────────┐
    │ Movement ≥ P70                 │──Yes──▶ Off-Ball Movement Shooter
    │ AND mvmt/(mvmt+spotup) ≥ 0.40 │
    └────┬───────────────────────────┘
         │No
    ┌────▼───────────────────────┐
    │ SpotUp ≥ P50               │──Yes──▶ Off-Ball Stationary Shooter
    └────┬───────────────────────┘         sub: Elite Shooter (3P% > 38%)
         │No
         ▼
    Rotation Piece / Glue Guy
```

---

## 9. Validated Classifications (2024-25)

| Player | Archetype | Subtype | Conf | Eff | Key Metrics |
|---|---|---|---|---|---|
| LeBron James | BDC | Heliocentric Guard | 100% | 85% | BD=0.43, AST=8.5, ON_BALL=0.34 |
| Luka Dončić | BDC | Heliocentric Guard | 100% | 85% | BD=0.56, AST=7.8, ON_BALL=0.53 |
| Stephen Curry | BDC | Gravity Engine | 100% | 86% | BD=0.34, FG3A=11.0, TS=62% |
| Nikola Jokić | BDC | Post Hub | 100% | 98% | BD=0.39, AST=10.0, POST=0.17 |
| Giannis Antetokounmpo | BDC | Primary Scorer | 99% | 91% | BD=0.39, AST=6.8, POST=0.15 |
| Jayson Tatum | BDC | Heliocentric Guard | 100% | 84% | BD=0.59, AST=5.9 |
| SGA | BDC | Heliocentric Guard | 100% | 93% | BD=0.62, AST=6.7 |
| Trae Young | BDC | Heliocentric Guard | 100% | 79% | BD=0.58, AST=11.6 |
| Tyrese Haliburton | BDC | Gravity Engine | 100% | 88% | BD=0.48, FG3A=8.8, TS=60% |
| Domantas Sabonis | BDC | Post Hub | 96% | 95% | BD=0.11, AST=6.3, POST=0.12 (via 1c) |
| Kevin Durant | All-Around | High Volume | 100% | 92% | BD=0.40, PTS/36=27.4, FG2A=67% |
| Anthony Edwards | All-Around | High Volume | 100% | 80% | BD=0.54, PTS/36=25.3, FG2A=50% |
| Kuminga | Interior Scorer | High Volume | 90% | 65% | BD=0.31, FG2A=73% |
| Draymond Green | Ballhandler | — | 100% | 64% | AST/36=7.0, PTS/36=8.5 |
| Klay Thompson | Movement Shooter | — | 100% | 70% | movement=0.15, ratio=0.63 |
| Brook Lopez | PnR Rolling Big | — | 100% | 82% | PRROLLMAN=0.20, FG3A=4.0 |
| Batum | Stationary Shooter | Elite Shooter | 100% | 86% | spotup=0.43, FG3%=41% |
| Gobert | Off-Ball Finisher | — | 100% | 89% | CUT_PNRRM=0.38 |

---

## 10. Design Philosophy

### Why Percentiles?

Static thresholds like "AST/36 ≥ 5.5" drift as league trends change. By using percentiles computed per-season from qualified players, the system automatically adapts. P80 ball dominance captures the top ~20% ball handlers regardless of whether league-wide creation rates shift.

### Why Linear Interpolation in BDC Composite?

v2 used `√(metric / denominator)` which saturates quickly — a player at the threshold and one far above both score similarly. v3's linear interpolation from threshold to P95 provides meaningful discrimination: a player barely meeting P80 ball dominance scores 0, while one at P95 scores 1.0. The 60/40 volume:confidence split weights ball handling + scoring volume (60%) and playmaking (40%).

### Why Separate Confidence and Effectiveness?

In v2, a player like Jokić (98% efficiency as BDC) and a low-efficiency BDC both had `archetype_confidence` blending fit certainty with production quality. v3 separates these:
- **Role confidence** (fit): "How certain are we this player is a BDC?" — based on how well their profile matches the archetype gates
- **Role effectiveness**: "How good are they at executing this role?" — based on TS z-score, volume, and playtype PPP

This separation prevents conflating classification certainty with player quality, which is critical for the upcoming player evaluation phase.

### Why Split PnR Rolling/Popping Big?

A traditional roll-and-finish big (e.g., Clint Capela) plays an entirely different role than a stretch big who pops to the 3-point line (e.g., Al Horford). The FG3A per 36 at P50 naturally separates these two styles. This distinction matters for lineup construction and matchup analysis.
