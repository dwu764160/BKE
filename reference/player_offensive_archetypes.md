# Player Offensive Archetypes — v4 Logic Documentation

## Overview

This document is the authoritative reference for the v4 offensive archetype classifier implemented in `src/data_compute/compute_player_archetypes.py`.
v4 preserves all of the v3 percentile-driven design while adding shot-zone integration, explicit midrange/rim subtypes, and removing the legacy `Rotation Piece` catch-all in favor of a best-fit fallback.

This file mirrors the depth and structure of the v3 documentation but highlights the v4 changes and provides updated examples, thresholds, and reproducible commands.

### High-level v4 changes (summary)

| Area | v3 | v4 |
|---|---|---|
| Shot zones | Not used | Integrated: `AT_RIM_FREQ`, `MIDRANGE_FREQ`, `PAINT_FREQ`, `CORNER3_FREQ`, `AB3_FREQ` (via `src/data_fetch/fetch_shot_zones.py`) |
| Rotation Piece | Catch-all leftover archetype | Eliminated — replaced by best-fit fallback that assigns the nearest archetype by normalized distance |
| Midrange handling | FG2A_RATE proxy only | Dedicated `Midrange Scorer` subtype and `Rim Finisher` subtype using shot-zone percentiles |
| All-Around gate | P60 | Lowered to P50 and evaluated earlier (before Ballhandler) to capture playmaker+scorer hybrids |
| New percentile gates | v3 thresholds | Added `HIGH_AT_RIM`, `HIGH_MIDRANGE`, `MODERATE_MIDRANGE` (P70/P70/P50) |
| Viewer | Basic archetype display | Viewer updated to show `RIM%` and `MID%` and to remove `Rotation Piece` filter; PnR big filters added |

---

## 1. Data Sources (v4)

v4 uses the same three core data sources as previous versions and adds a shot-zone fetcher.

1a. Synergy playtype data — 11 playtypes from NBA.com (unchanged):

 - Isolation (`ISOLATION_POSS_PCT`)
 - PnR Ball Handler (`PRBALLHANDLER_POSS_PCT`)
 - Post Up (`POSTUP_POSS_PCT`)
 - Cut (`CUT_POSS_PCT`)
 - PnR Roll Man (`PRROLLMAN_POSS_PCT`)
 - Handoff (`HANDOFF_POSS_PCT`)
 - Off Screen (`OFFSCREEN_POSS_PCT`)
 - Spot Up (`SPOTUP_POSS_PCT`)
 - Transition (`TRANSITION_POSS_PCT`)
 - Putback (`OFFREBOUND_POSS_PCT`)
 - Misc (`MISC_POSS_PCT`)

1b. Tracking data — same fields used before (drives, passing, touches, time-per-touch, catch-shoot, reb contest, avg speed, etc.)

1c. Box score — `GP`, `MIN`, `PTS`, `AST`, `REB`, `FGA`, `FG3A`, `FTA`, `FTM`, `USG_PCT`, `TS_PCT`, etc.

1d. Shot-zone data (NEW)

 - Source: NBA `LeagueDashPlayerShotLocations` endpoint (league-wide, single call per season).
 - Implemented in `src/data_fetch/fetch_shot_zones.py` which caches raw JSON in `data/tracking_cache/` and writes per-season parquet files at `data/tracking/{season}/shot_zones.parquet`.
 - Important: the returned `resultSets` is a dict (not list); base columns include `PLAYER_ID, PLAYER_NAME, TEAM_ID, TEAM_ABBREVIATION, AGE, NICKNAME` and then 8 zones × 3 metrics (FGM/FGA/FG_PCT).

Derived shot-zone features added to the archetype merge step:

 - `AT_RIM_FREQ` (fraction of FGA in restricted area)
 - `PAINT_FREQ` (in-the-paint non-RA fraction)
 - `MIDRANGE_FREQ` (fraction of FGA in midrange)
 - `CORNER3_FREQ` (corner 3 fraction)
 - `AB3_FREQ` (above the break 3 fraction)
 - `AT_RIM_PLUS_PAINT_FREQ` (useful for interior share calculations)

These fields are now present in `data/processed/player_archetypes.parquet` for every player-season where shot zone data exists.

---

## 2. Feature Engineering (detailed)

All v3-derived features remain; v4 augments merging logic to include shot-zone features. Key formulas are repeated here for completeness.

Per-36 conversions (used widely):

```
MPG = MIN / GP
minutes_factor = 36 / MPG
PTS_PER36 = (PTS / GP) * minutes_factor
AST_PER36 = (AST / GP) * minutes_factor
```

Ball Dominance (same formula as v3):

```
ON_BALL_CREATION = ISOLATION_POSS_PCT + PRBALLHANDLER_POSS_PCT
POST_CREATION = POSTUP_POSS_PCT
BALL_DOMINANT_PCT = ON_BALL_CREATION + POST_CREATION * POST_WEIGHT  # POST_WEIGHT=0.65
```

Playmaking score (unchanged):

```
PLAYMAKING_SCORE = AST_PER36 * 1.0
         + SECONDARY_AST_PER36 * 0.5
         + POTENTIAL_AST_PER36 * 0.3
         - TOV_PER36 * 0.5
```

Shot profile proxies and shot-zone derived metrics:

```
FG2A_RATE = (FGA - FG3A) / FGA  # used previously as interior/perim proxy
AT_RIM_FREQ, MIDRANGE_FREQ, PAINT_FREQ, CORNER3_FREQ, AB3_FREQ  # NEW
```

Use case examples:

- A player with `MIDRANGE_FREQ >= MODERATE_MIDRANGE (P50)` is a candidate for `Midrange Scorer` subtype.
- A player with `AT_RIM_FREQ >= HIGH_AT_RIM (P70)` is a candidate for `Rim Finisher` subtype.

---

## 3. Percentile-Based Thresholds (v4, full table)

v4 retains the v3 design: percentiles computed from qualified players (MIN ≥ 500, GP ≥ 20, MPG ≥ 15). New shot-zone percentiles were added.

Selected thresholds (2024-25 example values):

| Threshold Name | Column | Percentile | Example Value (2024-25) | Purpose |
|---|---|---:|---:|---|
| `BD_MIN` | BALL_DOMINANT_PCT | P80 | 0.3720 | BDC gating |
| `PLAYMAKER` | AST_PER36 | P75 | 5.0741 | Playmaker gating |
| `HIGH_SCORING` | PTS_PER36 | P70 | 18.9497 | High scorer detection |
| `MODERATE_SCORING` | PTS_PER36 | P50 | 17.2264 | All-Around gate (v4 lowered from P60) |
| `ALL_AROUND_SCORING` | PTS_PER36 | P60 | 17.4899 | Legacy/all-around strict gate (v3) |
| `FG2A_INTERIOR` | FG2A_RATE | P70 | 0.6731 | Interior shot profile |
| `FG2A_PERIMETER` | FG2A_RATE | P30 | 0.4781 | Perimeter shot profile |
| `HIGH_PRROLLMAN` | PRROLLMAN_POSS_PCT | P70 | 0.0646 | PnR big gating |
| `FG3A_MEDIAN` | FG3A_PER36 | P50 | 5.8126 | PnR pop vs roll split |
| `TOUCHES_MEDIAN` | TOUCHES | P50 | 40.3 | Connector activity gate |
| `HIGH_AT_RIM` | AT_RIM_FREQ | P70 | 0.3258 | Rim finisher gate |
| `HIGH_MIDRANGE` | MIDRANGE_FREQ | P70 | 0.1186 | Strong midrange gate |
| `MODERATE_MIDRANGE` | MIDRANGE_FREQ | P50 | 0.0818 | Moderate midrange gate |

Static thresholds (unchanged where applicable):

| Param | Value | Purpose |
|---|---:|---|
| `MIN_MINUTES` | 500 | Minimum minutes to qualify |
| `MIN_GP` | 20 | Minimum games |
| `MIN_MPG` | 15.0 | Minimum minutes per game |
| `POST_WEIGHT` | 0.65 | Post-up weight in ball dominance |
| `MOVEMENT_RATIO_MIN` | 0.40 | movement/(movement+spotup) min |
| `CONNECTOR_SEC_PER_TOUCH_MAX` | 2.5 | Activity filter for Connectors |

Notes: percentile numbers are recomputed per-season; therefore gates adapt automatically to league changes.

---

## 4. Classification Hierarchy (v4): full details

This section mirrors the original detail level and documents the exact logic order, formulas, and subtype decisions used in `compute_player_archetypes.py`.

### Step 0 — Minimum requirements

```
IF MIN < MIN_MINUTES OR GP < MIN_GP OR MPG < MIN_MPG:
  primary_archetype = 'Insufficient Minutes'
  secondary_archetype = None
  role_confidence = 0.0
  role_effectiveness = 0.0
  -> stop
```

### Step 1 — Ball Dominant Creator (BDC) main path

Gates (all percentile-based):

 - `ball_dom >= BD_MIN` (P80)
 - `ast_per36 >= PLAYMAKER` (P75)
 - `pts_per36 >= MODERATE_SCORING` (P50)

Composite scoring (linear interpolation between threshold and P95):

```
bd_score  = clip( (ball_dom - BD_MIN) / (BD_P95 - BD_MIN), 0, 1 )
pts_score = clip( (pts_per36 - MODERATE_SCORING) / (PTS_P95 - MODERATE_SCORING), 0, 1 )
ast_score = clip( (ast_per36 - PLAYMAKER) / (AST_P95 - PLAYMAKER), 0, 1 )

volume     = 0.60 * bd_score + 0.40 * pts_score
confidence = ast_score
composite  = 0.60 * volume + 0.40 * confidence

IF composite >= 0.35 -> classify as Ball Dominant Creator
```

BDC subtypes (priority order):

1. Post Hub: `POSTUP_POSS_PCT >= POST_HUB_POSS` (0.10) or `POSTUP_POSS_PCT >= POST_HUB_DOMINANT` (0.15)
2. Gravity Engine: `FG3A_PER36 >= HIGH_FG3A` (P80) AND (TS >= HIGH_EFFICIENCY OR FG3_PCT >= HIGH_FG3_PCT)
3. Heliocentric Guard: `ball_dom >= BD_HELIOCENTRIC` (P85)
4. Primary Scorer: `pts_per36 >= HIGH_SCORING` (P70)

Examples: Jokić (Post Hub), Curry (Gravity Engine), LeBron (Heliocentric)

### Step 1b — Offensive Hub (elite playmaker + elite scorer)

```
IF ast_per36 >= ELITE_PLAYMAKER (P85) AND pts_per36 >= ELITE_SCORING (P80):
  -> BDC with hub-specific subtype logic
```

This path allows some high-creation, high-scoring players to be captured even if other gates are borderline.

### Step 2 — All-Around Scorer (moved earlier in v4)

Rationale: capture players who are both playmakers and scorers but don't meet strict BDC composite.

```
IF is_playmaker (ast_per36 >= PLAYMAKER OR PLAYMAKING_SCORE >= threshold) AND pts_per36 >= MODERATE_SCORING:
  primary_archetype = 'All-Around Scorer'
  secondary_archetype = 'Midrange Scorer' IF MIDRANGE_FREQ >= MODERATE_MIDRANGE
```

v4 lowers the scoring gate used here to `MODERATE_SCORING` (P50) to be more inclusive while kept earlier in pipeline to avoid misclassifying playmakers as Ballhandlers.

### Step 3 — Ballhandler

```
IF ast_per36 >= PLAYMAKER AND NOT high_scorer:
  primary_archetype = 'Ballhandler'
```

Ballhandlers are high end-playmakers who are not primary scorers.

### Step 4 — Primary Scorers (Interior / Perimeter / All-Around)

```
IF ball_dom >= BD_ALL_AROUND (P60) AND NOT playmaker:
  IF FG2A_RATE >= FG2A_INTERIOR (P70): primary_archetype = 'Interior Scorer'
  ELIF FG2A_RATE <= FG2A_PERIMETER (P30): primary_archetype = 'Perimeter Scorer'
  ELSE: primary_archetype = 'All-Around Scorer' IF pts_per36 >= ALL_AROUND_SCORING (legacy) ELSE fallthrough
```

v4 augment: after assigning `Interior Scorer`, set subtype using shot-zone percentiles:

 - `Rim Finisher`: `AT_RIM_FREQ >= HIGH_AT_RIM (P70)`
 - `Midrange Scorer`: `MIDRANGE_FREQ >= MODERATE_MIDRANGE (P50)` (strong evidence >= P70)

### Step 5 — Connector

Connector logic unchanged: AST >= CONNECTOR_AST (P50) + activity filter (touches >= TOUCHES_MEDIAN OR secs_per_touch <= CONNECTOR_SEC_PER_TOUCH_MAX) and not high scorer.

### Step 6 — PnR Big (Rolling / Popping)

```
IF PRROLLMAN_POSS_PCT >= HIGH_PRROLLMAN (P70) AND prrollman > cut_poss:
  IF FG3A_PER36 >= FG3A_MEDIAN (P50): primary_archetype = 'PnR Popping Big'
  ELSE: primary_archetype = 'PnR Rolling Big'
```

### Step 7 — Off-Ball Finisher

```
finish_score = CUT_PNRRM_PCT + 0.5 * PUTBACK_PCT + 0.3 * TRANSITION_PCT
IF CUT_PNRRM_PCT >= HIGH_CUT_PNRRM (P75) OR finish_score > 0.20:
  primary_archetype = 'Off-Ball Finisher'
```

### Step 8 — Off-Ball Movement Shooter

```
IF MOVEMENT_SHOOTER_PCT >= HIGH_MOVEMENT (P70) AND movement/(movement+spotup) >= MOVEMENT_RATIO_MIN (0.40):
  primary_archetype = 'Off-Ball Movement Shooter'
```

### Step 9 — Off-Ball Stationary Shooter

```
IF SPOTUP_PCT >= HIGH_SPOTUP (P50):
  primary_archetype = 'Off-Ball Stationary Shooter'
  secondary_archetype = 'Elite Shooter' IF FG3_PCT >= HIGH_FG3_PCT (P70)
```

### Fallback — Best-fit (Rotation Piece removed)

When none of the above gates match, v4 computes a distance score to each archetype's canonical gate vector. The distance is normalized by the gate scale (P95–gate) per metric, summed, and the archetype with the smallest normalized distance is selected at low confidence. This deterministic fallback prevents opaque "Rotation Piece" labeling while still assigning a role for downstream tools.

Algorithm sketch:

```
for each archetype A:
  compute normalized_diff over archetype-specific key metrics (ball_dom, ast_per36, pts_per36, prrollman, spotup, midrange, at_rim, etc.)
  dist_A = sqrt(sum(normalized_diff^2))/sqrt(N)
assign archetype = argmin_A(dist_A)
role_confidence = 1 - scaled(dist_A)
```

---

## 5. Subtypes (full listing)

Primary subtypes (applied where relevant):

- `Midrange Scorer` — MIDRANGE_FREQ >= P50 (strong evidence P70)
- `Rim Finisher` — AT_RIM_FREQ >= P70
- `High Volume` — top-tier PTS/36 (>= P70)
- `Heliocentric Guard` — aggressive on-ball top percentile (BD_HELIOCENTRIC)
- `Gravity Engine` — high 3PA/36 (P80) + efficiency signals
- `Post Hub` — POSTUP >= 0.10 (dominant at >=0.15)
- `Transition Player`, `Elite Shooter`, `Playmaking Big`, `Primary Scorer`, `Offensive Hub`, etc. — as in v3

---

## 6. Dual Confidence & Effectiveness (formulas)

v4 keeps the same split between `role_confidence` (fit) and `role_effectiveness` (production quality). Below are representative formulas (implementation uses normalized/scaled variants per archetype).

Role Confidence (fit): archetype-dependent; examples:

- BDC: `role_confidence = sqrt(composite / composite_cap)` where composite is the linear interpolation described earlier.
- Interior Scorer: `role_confidence = 0.6 * sqrt(bd_norm) + 0.4 * (fg2a_norm)`

Role Effectiveness (production): blends TS z-score, volume (PTS/36), and role-specific PPPs. Example for BDC:

```
eff_score = 0.35 * ts_component + 0.35 * pts_component + 0.30 * primary_playtype_ppp
```

TS component uses `TS_ZSCORE` normalized to [0,1] by clipping at ±2σ.

---

## 7. Output Schema (selected)

The final parquet contains the full feature matrix plus archetype outputs:

| Column | Description |
|---|---|
| `PLAYER_ID` | NBA player id |
| `PLAYER_NAME` | name |
| `SEASON` | season string |
| `primary_archetype` | archetype label (v4 set of ~11) |
| `secondary_archetype` | subtype label or NULL |
| `role_confidence` | 0–1 fit score |
| `role_effectiveness` | 0–1 production score |
| `AT_RIM_FREQ` | shot-zone: restricted area fraction |
| `MIDRANGE_FREQ` | shot-zone: midrange fraction |
| `PAINT_FREQ` | shot-zone: in-paint non-RA |
| ... | (many other features like PTS_PER36, AST_PER36, TOUCHES, PRROLLMAN_POSS_PCT) |

Files:

- `data/processed/player_archetypes.parquet`
- `data/tracking/{season}/shot_zones.parquet`
- `data/tracking_cache/shot_locations_{season}.json` (raw cache)

---

## 8. Distribution & Validation (2024-25 qualified)

Qualified players: 329 (MIN ≥ 500, GP ≥ 20, MPG ≥ 15)

Archetype counts (v4):

| Archetype | Count | % |
|---|---:|---:|
| Off-Ball Stationary Shooter | 63 | 19.1% |
| Off-Ball Finisher | 62 | 18.8% |
| All-Around Scorer | 40 | 12.2% |
| Ball Dominant Creator | 40 | 12.2% |
| Connector | 31 | 9.4% |
| Ballhandler | 31 | 9.4% |
| PnR Rolling Big | 20 | 6.1% |
| Interior Scorer | 18 | 5.5% |
| Off-Ball Movement Shooter | 12 | 3.6% |
| Perimeter Scorer | 7 | 2.1% |
| PnR Popping Big | 5 | 1.5% |
| Rotation Piece | 0 | 0.0% (ELIMINATED) |

Subtype totals (all seasons): Midrange Scorer 42, Rim Finisher 14, Transition Player 98, High Volume 85, Heliocentric Guard 80, Elite Shooter 77, Playmaking Big 41, Gravity Engine 19, Post Hub 7, Primary Scorer 7.

Representative validated classifications (v4):

| Player | Archetype | Subtype | Notes |
|---|---|---|---|
| DeMar DeRozan | Interior Scorer | Midrange Scorer | MR≈0.459 (v4 midrange detection) |
| Giannis Antetokounmpo | Ball Dominant Creator | Offensive Hub | AT_RIM≈0.571; ball dominance keeps him BDC |
| Stephen Curry | Ball Dominant Creator | Gravity Engine | High FG3A and efficiency |
| Kevin Durant | All-Around Scorer | High Volume | Mixed 2PT/3PT + high PTS/36 |
| Joel Embiid | Interior Scorer / All-Around | Midrange Scorer / High Volume | consistent with changed gates |
| Rudy Gobert / Clint Capela | Off-Ball Finisher | Rim Finisher | Very high AT_RIM_FREQ (Capela ~0.704) |
| LeBron James | Ball Dominant Creator | Heliocentric Guard | High BD & playmaking |
| Luka Dončić | Ball Dominant Creator | Heliocentric Guard | High BD & playmaking |
| Anthony Edwards | All-Around Scorer | High Volume | High PTS/36 and mixed profile |

Notes: full player lists were printed during the v4 run and are reproducible using the validation scripts in `/tmp` used during development.

---

## 9. Classification Flow Diagram (condensed ASCII)

```
Player season record
   │
  MIN<500 OR GP<20 OR MPG<15? ──Yes──> Insufficient Minutes
   │No
   ▼
  BDC gate? ──Yes──> Ball Dominant Creator (Post Hub / Gravity / Heliocentric / Primary Scorer)
   │No
   ▼
  Offensive Hub (elite AST & PTS)? ──Yes──> BDC (hub)
   │No
   ▼
  All-Around Scorer (playmaker & moderate scorer)? ──Yes──> All-Around (Midrange subtype if MR elevated)
   │No
   ▼
  Ballhandler (playmaker, not scorer)? ──Yes──> Ballhandler
   │No
   ▼
  Primary Scorer (BD >= P60)? ──Yes──> Interior / Perimeter / All-Around (shot-zone subtypes)
   │No
   ▼
  Connector? PnR Big? Off-Ball Finisher? Movement Shooter? Stationary Shooter? ──Yes──> respective archetype
   │No
   ▼
  Best-fit fallback (no Rotation Piece) ──> nearest archetype with low confidence
```

---

## 10. Design Notes & Rationale

- Percentiles: keep the classifier robust to league trends. Pxx gates are recomputed per-season from qualified players.
- Shot zones: give the classifier direct signal for midrange vs rim usage instead of proxying with FG2A_RATE only.
- Rotation Piece removal: opaque catch-alls are harmful for downstream model interpretability; best-fit fallback gives deterministic, low-confidence assignments that are actionable.
- All-Around earlier: captures playmaker+scorer hybrids that previously were split across Ballhandler/BDC.

---

## 11. Repro & Commands

Fetch shot zones (one-time per season):
```bash
.venv/bin/python src/data_fetch/fetch_shot_zones.py
```

Run the v4 archetype pipeline (writes `data/processed/player_archetypes.parquet`):
```bash
.venv/bin/python src/data_compute/compute_player_archetypes.py
```

Generate the viewer HTML:
```bash
.venv/bin/python app/player_archetype_viewer.py
```

Quick validation example (prints select players):
```bash
.venv/bin/python - <<'PY'
import pandas as pd
df = pd.read_parquet('data/processed/player_archetypes.parquet')
for name in ['DeRozan','Antetokounmpo','Curry','Durant','Embiid','Don\'\'cic','Tatum']:
  p = df[df['PLAYER_NAME'].str.contains(name, case=False, na=False)]
  if len(p)>0:
    r = p.iloc[0]
    print(r['PLAYER_NAME'], r['SEASON'], r['primary_archetype'], r['secondary_archetype'], 'AT_RIM=', round(r.get('AT_RIM_FREQ',0),3),'MR=',round(r.get('MIDRANGE_FREQ',0),3))
PY
```

---

## 12. Next Steps

- Add unit tests asserting core expectations: DeRozan→Midrange Scorer, Capela→Rim Finisher, Curry→Gravity Engine.
- Integrate the Grand Unified Model wrapper (`compute_rapm_metrics.py`) to connect Layers 1/2 (linear metrics) and 3 (RAPM) for a combined EPM-like metric.
- Add small docs in `reference/` describing the shot-zone fetch format and how to extend it.

Last updated: 2026-02-07

---

## 1. Data Sources

v4 uses the same three base sources as before plus a new shot-zone source:

### 1a. Synergy Playtype Data (unchanged)
- 11 playtypes (Isolation, PnR Ball Handler, Post Up, Cut, PnR Roll Man, Handoff, Off Screen, Spot Up, Transition, Putback, Misc)

### 1b. Tracking Data (unchanged)
- Drives, Passing, Possessions, Catch & Shoot, Rebounding, Speed/Distance metrics

### 1c. Box Score Data (unchanged)

### 1d. Shot Zone Data (NEW)

Source: NBA.com `LeagueDashPlayerShotLocations` (league-wide single-call per season). Implemented in `src/data_fetch/fetch_shot_zones.py` which caches JSON and writes `data/tracking/{season}/shot_zones.parquet`.

Key derived features:
- `AT_RIM_FREQ` — fraction of FGA in restricted area
- `MIDRANGE_FREQ` — fraction of FGA in midrange
- `PAINT_FREQ` — in-paint non-RA
- `CORNER3_FREQ`, `AB3_FREQ` — corner and above-the-break 3 frequencies
- `AT_RIM_PLUS_PAINT_FREQ` — useful for interior share

Notes on the API: the response is a multi-header object; `resultSets` is a dict (not a list) and includes 6 base columns (PLAYER_ID, PLAYER_NAME, TEAM_ID, TEAM_ABBREVIATION, AGE, NICKNAME) plus 8 zones × 3 stats each.

---

## 2. Feature Engineering (high level)

All previous v3 features remain (per-36 stats, ball dominance, playmaking score, FG2A_RATE, TS, USG proxy, off-ball composites). v4 adds shot-zone fields into the feature merge step so archetype decisions can use `AT_RIM_FREQ` and `MIDRANGE_FREQ` directly instead of only FG2A proxies.

Shot-zone usage examples:
- `RIM` = `AT_RIM_FREQ`
- `MR` = `MIDRANGE_FREQ`
- `PAINT` = `PAINT_FREQ`

These are stored in the final output `data/processed/player_archetypes.parquet` per player-season.

---

## 3. Percentile-Based Thresholds (v4)

v4 continues to compute percentile thresholds per-season from qualified players (MIN ≥ 500, GP ≥ 20, MPG ≥ 15). New thresholds were added for shot zones.

Selected thresholds (2024-25 example values):

| Name | Column | Percentile | Example (2024-25) |
|---|---:|---:|---:|
| `BD_MIN` | BALL_DOMINANT_PCT | P80 | 0.372 |
| `PLAYMAKER` | AST_PER36 | P75 | 5.074 |
| `MODERATE_SCORING` | PTS_PER36 | P50 | 17.23 (All-Around gate lowered to this in v4) |
| `ALL_AROUND_SCORING` (legacy) | PTS_PER36 | P60 | 17.23 (v3) |
| `HIGH_AT_RIM` | AT_RIM_FREQ | P70 | 0.3258 |
| `HIGH_MIDRANGE` | MIDRANGE_FREQ | P70 | 0.1186 |
| `MODERATE_MIDRANGE` | MIDRANGE_FREQ | P50 | 0.0818 |
| `HIGH_PRROLLMAN` | PRROLLMAN_POSS_PCT | P70 | 0.0646 |
| `FG2A_INTERIOR` | FG2A_RATE | P70 | 0.6731 |
| `FG2A_PERIMETER` | FG2A_RATE | P30 | 0.4781 |

All other percentile thresholds from v3 remain; v4 only introduces the shot-zone percentiles and re-uses the MODERATE_SCORING (P50) gate for All-Around in order to expand the All-Around path earlier in the hierarchy while keeping precision acceptable.

---

## 4. Classification Hierarchy (v4)

The hierarchy is mostly the same as v3 with the following important changes:

- Step ordering: **All-Around Scorer** path is now evaluated before **Ballhandler** (to catch playmaker+scorer players).
- The All-Around scoring gate was lowered from P60 → P50.
- Interior Scorer now has explicit subtypes:
  - `Rim Finisher`: `AT_RIM_FREQ >= HIGH_AT_RIM` (P70)
  - `Midrange Scorer`: `MIDRANGE_FREQ >= MODERATE_MIDRANGE` (P50) or `>= HIGH_MIDRANGE` (P70) depending on strictness
- `Rotation Piece` archetype has been removed; a best-fit fallback assigns the nearest archetype when none of the main gates match.

High-level flow (condensed):

0) Minimum requirements: MIN ≥ 500, GP ≥ 20, MPG ≥ 15 → otherwise `Insufficient Minutes`.

1) Ball Dominant Creator (BDC) main path: same composite gating as v3 (ball_dom P80 + PLAYMAKER P75 + PTS P50 + composite ≥ 0.35). Subtypes remain (Post Hub, Gravity Engine, Heliocentric Guard, Primary Scorer).

1b) Offensive Hub (elite playmaker + elite scorer) handled as before — still produces a BDC with subtype logic.

2) **All-Around Scorer** (NEW position in order): if `is_playmaker AND is_high_scorer` → classify as All-Around Scorer. Subtype assignment: Midrange Scorer if `MIDRANGE_FREQ` is elevated.

3) Ballhandler: (playmaker but not high scorer) — moved after All-Around in v4.

4) Primary Scorers: ball_dom ≥ P60 and not playmaker; then split by FG2A_RATE and shot-zone signals. Interior Scorer can be subtyped into Rim Finisher vs Midrange Scorer using shot-zone percentiles.

5) Connector, PnR Rolling/Popping Big, Off-Ball Finisher, Movement/Stationary Shooters: same logic as v3 (with PnR split by FG3A per 36 relative to season median).

Fallback: When none of the gates match, v4 computes a normalized distance to each archetype's gate and assigns the archetype with the smallest distance (low confidence). This replaces the old `Rotation Piece` catch-all.

---

## 5. Subtypes (notable additions)

- `Midrange Scorer` — awarded to interior/perimeter scorers with elevated `MIDRANGE_FREQ` (P50/P70 thresholds used as gating for moderate/strong evidence). Example: DeMar DeRozan (MR≈0.459).
- `Rim Finisher` — interior scorers whose `AT_RIM_FREQ` ≥ P70. Example: Capela (AT_RIM≈0.704).
- Existing subtypes (High Volume, Heliocentric Guard, Gravity Engine, Post Hub, Transition Player, Elite Shooter) remain and are applied as before when applicable.

---

## 6. Dual Confidence & Effectiveness (unchanged except Rotation Piece removal)

v4 preserves the split between `role_confidence` (fit) and `role_effectiveness` (production). The formulas remain the same as v3; rotation-piece-specific default confidence is removed because that archetype no longer exists.

---

## 7. Output and Files

Primary output (unchanged path): `data/processed/player_archetypes.parquet` with these columns included (selected):

`PLAYER_ID`, `PLAYER_NAME`, `SEASON`, `primary_archetype`, `secondary_archetype`, `role_confidence`, `role_effectiveness`, plus all feature columns and shot-zone fields: `AT_RIM_FREQ`, `MIDRANGE_FREQ`, `PAINT_FREQ`, `CORNER3_FREQ`, `AB3_FREQ`.

New data files produced by v4 pipeline:

- `data/tracking/{season}/shot_zones.parquet` — produced by `src/data_fetch/fetch_shot_zones.py` (one file per season)

Viewer changes:
- `app/player_archetype_viewer.py` was updated to show `RIM%` and `MID%` in the player card, to remove `Rotation Piece` from filters, and to include PnR big options.

---

## 8. Distribution (2024-25 qualified players, updated v4 counts)

Qualified: 329 players (MIN ≥ 500, GP ≥ 20, MPG ≥ 15)

| Archetype | Count | % |
|---|---:|---:|
| Off-Ball Stationary Shooter | 63 | 19.1% |
| Off-Ball Finisher | 62 | 18.8% |
| All-Around Scorer | 40 | 12.2% |
| Ball Dominant Creator | 40 | 12.2% |
| Connector | 31 | 9.4% |
| Ballhandler | 31 | 9.4% |
| PnR Rolling Big | 20 | 6.1% |
| Interior Scorer | 18 | 5.5% |
| Off-Ball Movement Shooter | 12 | 3.6% |
| Perimeter Scorer | 7 | 2.1% |
| PnR Popping Big | 5 | 1.5% |
| Rotation Piece | 0 | 0.0% (ELIMINATED) |

Subtype totals (all seasons): Midrange Scorer 42, Rim Finisher 14, Transition Player 98, High Volume 85, Heliocentric Guard 80, Elite Shooter 77, Playmaking Big 41, Gravity Engine 19, Post Hub 7, Primary Scorer 7.

---

## 9. Validation Highlights

- DeMar DeRozan: classified `Interior Scorer / Midrange Scorer` (MR ~0.459) — matches human expectation.
- Giannis Antetokounmpo: remains `Ball Dominant Creator / Offensive Hub` with high `AT_RIM_FREQ` (~0.571) — remains BDC because of high ball dominance.
- Kevin Durant: All-Around Scorer / High Volume in most seasons; Intermediate season-level changes are retained correctly.
- Rotation Piece assignments: 0 across seasons after v4 changes.

---

## 10. Repro & Commands

Fetch shot zones (one-time per season):
```bash
.venv/bin/python src/data_fetch/fetch_shot_zones.py
```

Run v4 archetype classifier:
```bash
.venv/bin/python src/data_compute/compute_player_archetypes.py
```

Regenerate viewer HTML:
```bash
.venv/bin/python app/player_archetype_viewer.py
```

---

## 11. Notes & Next Steps

- Grand Unified Model: next planned step is to integrate `compute_linear_metrics.py` (Layer 1/2) and `compute_rapm.py` (Layer 3) into a `compute_rapm_metrics.py` wrapper that produces a unified EPM-like metric — this is a separate task.
- Consider adding small unit tests verifying key player archetypes (DeRozan → Midrange Scorer, Capela → Rim Finisher, Curry → Gravity Engine) after any thresholds are changed.
- Keep `src/data_fetch/fetch_shot_zones.py` cached outputs in `data/tracking_cache/` to avoid repeated NBA API calls.

Last updated: 2026-02-07
