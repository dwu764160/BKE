# Player Offensive Archetypes — v4.3 Logic Documentation

## Overview

This document is the authoritative reference for the v4.3 offensive archetype classifier implemented in `src/data_compute/compute_player_archetypes.py`.
v4 preserves all of the v3 percentile-driven design while adding shot-zone integration, explicit midrange/rim subtypes, and removing the legacy `Rotation Piece` catch-all in favor of a best-fit fallback.

This file mirrors the depth and structure of the v3 documentation but highlights the v4+ changes and provides updated examples, thresholds, and reproducible commands.

### High-level v4 changes (summary)

| Area | v3 | v4 |
|---|---|---|
| Shot zones | Not used | Integrated: `AT_RIM_FREQ`, `MIDRANGE_FREQ`, `PAINT_FREQ`, `CORNER3_FREQ`, `AB3_FREQ` (via `src/data_fetch/fetch_shot_zones.py`) |
| Rotation Piece | Catch-all leftover archetype | Eliminated — replaced by best-fit fallback that assigns the nearest archetype by normalized distance |
| Midrange handling | FG2A_RATE proxy only | Dedicated `Midrange Scorer` subtype and `Rim Finisher` subtype using shot-zone percentiles |
| All-Around gate | P60 | Lowered to P50 and evaluated earlier (before Ballhandler) to capture playmaker+scorer hybrids |
| New percentile gates | v3 thresholds | Added `HIGH_AT_RIM`, `HIGH_MIDRANGE`, `MODERATE_MIDRANGE` (P70/P70/P50) |
| Viewer | Basic archetype display | Viewer updated to show `RIM%` and `MID%` and to remove `Rotation Piece` filter; PnR big filters added |

### v4.1 changes

| Area | v4 | v4.1 |
|---|---|---|
| Low-BD Interior Scorer | Not supported | Alternate path: `FG2A_RATE >= P70` + `MODERATE_SCORING` + `AT_RIM_PAINT_P60` (confidence capped at 0.80) |
| Midrange subtype split | Single `Midrange Scorer` | P70+ = `Midrange Scorer`, P50–P70 = `Inside-the-Arc` (originally named `Midrange Lean`, renamed v4.3) |
| Fallback mechanism | Ad-hoc candidate scoring | Frozen `FALLBACK_VECTORS` dict — canonical metric vectors per archetype for deterministic best-fit |

### v4.2 changes

| Area | v4.1 | v4.2 |
|---|---|---|
| Interior Scorer skill gate | Any non-playmaker with interior profile | Non-stars must show skill finishing (`paint_freq + midrange_freq >= 0.25`); rim-dependent non-stars fall through to PnR Big / Off-Ball |
| Movement Shooter thresholds | P70, ratio 0.40 | Loosened to P60, ratio 0.30 — captures ~2× more movement shooters from the stationary pool |
| Perimeter Scorer (moderate BD) | Only via Primary Scorer (BD ≥ P60) | New Step 7b: `BD >= P50` + scoring above P25 — redirects self-creating players from Stationary Shooter |
| Off-Ball Stationary Shooter | ~64 players (2024-25) | Reduced to ~45 (70% retained) — more balanced distribution |

### v4.3 changes

| Area | v4.2 | v4.3 |
|---|---|---|
| PnR Big absorption | Strict: requires `prrollman > cut_poss` AND P70 | Added "prominent" path: `prrollman >= P50` + interior profile; PnR does NOT need to be the most common play |
| Off-Ball Finisher redirect | No big-man redirect | Bigs with prominent PnR Roll Man activity inside Off-Ball Finisher step are redirected to PnR Rolling/Popping Big |
| Interior Scorer skill check | `SKILL_FINISHING_MIN = 0.25` | Raised to `0.40`; rim-runners with prominent PnR + low midrange always fall through regardless of scoring volume |
| Midrange Lean subtype | Named "Midrange Lean" | Renamed to "Inside-the-Arc" |
| Archetype Embedding | Not implemented | Soft role model: confidence-weighted 11-dimensional embedding vector per player (stored in `archetype_embeddings.parquet`) |
| PnR Rolling Big count | ~13 (2024-25) | ~40 (absorbed from Off-Ball Finisher and Interior Scorer) |
| Off-Ball Finisher count | ~54 (2024-25) | ~36 (bigs redirected to PnR Big) |
| Interior Scorer count | ~33 (2024-25) | ~19 (rim-runners redirected to PnR Big) |

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
| `HIGH_PRROLLMAN` | PRROLLMAN_POSS_PCT | P70 | 0.0646 | PnR big gating (strict path) |
| `PROMINENT_PRROLLMAN` | PRROLLMAN_POSS_PCT | P50 | 0.0320 | PnR big gating (prominent path, v4.3) |
| `FG3A_MEDIAN` | FG3A_PER36 | P50 | 5.8126 | PnR pop vs roll split |
| `TOUCHES_MEDIAN` | TOUCHES | P50 | 40.3 | Connector activity gate |
| `HIGH_AT_RIM` | AT_RIM_FREQ | P70 | 0.3258 | Rim finisher gate |
| `HIGH_MIDRANGE` | MIDRANGE_FREQ | P70 | 0.1186 | Strong midrange gate |
| `MODERATE_MIDRANGE` | MIDRANGE_FREQ | P50 | 0.0818 | Moderate midrange gate |
| `AT_RIM_PAINT_P60` | AT_RIM_PLUS_PAINT_FREQ | P60 | 0.4484 | Interior shot location (v4.1) |
| `HIGH_MOVEMENT` | MOVEMENT_SHOOTER_PCT | P60 | 0.0808 | Movement shooter gate (v4.2: lowered from P70) |
| `BD_BASE` | BALL_DOMINANT_PCT | P50 | ~0.17 | Perimeter Scorer moderate BD gate (v4.2) |
| `LOW_SCORING` | PTS_PER36 | P25 | ~12.5 | Minimum scoring floor (v4.2) |

Static thresholds (unchanged where applicable):

| Param | Value | Purpose |
|---|---:|---|
| `MIN_MINUTES` | 500 | Minimum minutes to qualify |
| `MIN_GP` | 20 | Minimum games |
| `MIN_MPG` | 15.0 | Minimum minutes per game |
| `POST_WEIGHT` | 0.65 | Post-up weight in ball dominance |
| `MOVEMENT_RATIO_MIN` | 0.30 | movement/(movement+spotup) min (v4.2: lowered from 0.40) |
| `CONNECTOR_SEC_PER_TOUCH_MAX` | 2.5 | Activity filter for Connectors |
| `SKILL_FINISHING_MIN` | 0.40 | Minimum paint_freq + midrange_freq for non-star Interior Scorer (v4.3: raised from 0.25) |

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
  IF FG2A_RATE >= FG2A_INTERIOR (P70):
    # v4.3: Harsher skill-finishing gate; rim-runners with PnR activity always fall through
    non_rim_interior = paint_freq + midrange_freq
    is_rim_runner_big = (AT_RIM_FREQ >= HIGH_AT_RIM AND MIDRANGE_FREQ <= LOW_MIDRANGE
                         AND prrollman >= PROMINENT_PRROLLMAN)
    IF NOT is_rim_runner_big AND (pts_per36 >= ELITE_SCORING OR non_rim_interior >= 0.40):
      primary_archetype = 'Interior Scorer'  (with shot-zone subtypes)
    ELSE:
      # Rim-dependent big → falls through to PnR Big / Off-Ball
      continue to next step
  ELIF FG2A_RATE <= FG2A_PERIMETER (P30): primary_archetype = 'Perimeter Scorer'
  ELSE: primary_archetype = 'All-Around Scorer' IF pts_per36 >= MODERATE_SCORING ELSE fallthrough
```

v4.3 augment: Skill finishing floor raised from 0.25 to 0.40. Rim-runners (HIGH_AT_RIM + LOW_MIDRANGE + prominent PnR) ALWAYS fall through to PnR Big regardless of scoring volume. This catches players like Jarrett Allen and Daniel Gafford.

v4 augment: after assigning `Interior Scorer`, set subtype using shot-zone percentiles:

 - `Rim Finisher`: `AT_RIM_FREQ >= HIGH_AT_RIM (P70)`
 - `Midrange Scorer`: `MIDRANGE_FREQ >= HIGH_MIDRANGE (P70)`
 - `Inside-the-Arc`: `MIDRANGE_FREQ >= MODERATE_MIDRANGE (P50)` (between P50 and P70) (renamed from Midrange Lean in v4.3)

### Step 5 — Connector

Connector logic unchanged: AST >= CONNECTOR_AST (P50) + activity filter (touches >= TOUCHES_MEDIAN OR secs_per_touch <= CONNECTOR_SEC_PER_TOUCH_MAX) and not high scorer.

### Step 6 — PnR Big (Rolling / Popping)

```
# Strict path (v4 original)
IF PRROLLMAN_POSS_PCT >= HIGH_PRROLLMAN (P70) AND prrollman > cut_poss AND prrollman >= spotup * 0.5:
  -> PnR Big

# Prominent path (v4.3 NEW) — looser gate for interior-profile bigs
ELIF PRROLLMAN_POSS_PCT >= PROMINENT_PRROLLMAN (P50) AND is_interior_profile AND NOT playmaker:
  -> PnR Big (confidence capped at 0.85)

# Within either path:
  IF FG3A_PER36 >= FG3A_MEDIAN (P50): primary_archetype = 'PnR Popping Big'
  ELSE: primary_archetype = 'PnR Rolling Big'
```

Where `is_interior_profile = (FG2A_RATE >= FG2A_INTERIOR (P70) OR AT_RIM_FREQ >= HIGH_AT_RIM (P70))`.

v4.3: PnR Roll Man no longer needs to be the player's MOST common play. As long as it's prominent (>= P50) and the player has an interior profile, they're classified as PnR Big. This absorbs many rim-running bigs who were previously Off-Ball Finishers or Interior Scorers.

### Step 7 — Off-Ball Finisher

```
finish_score = CUT_PNRRM_PCT + 0.5 * PUTBACK_PCT + 0.3 * TRANSITION_PCT
IF CUT_PNRRM_PCT >= HIGH_CUT_PNRRM (P75) OR finish_score > 0.20:
  # v4.3: Redirect bigs with prominent PnR activity to PnR Big
  IF prrollman >= PROMINENT_PRROLLMAN AND is_interior_profile AND NOT playmaker:
    -> PnR Rolling/Popping Big (confidence capped at 0.80)
  ELSE:
    primary_archetype = 'Off-Ball Finisher'
```

v4.3: Before assigning Off-Ball Finisher, checks if the player is a big man with PnR Roll Man activity. If so, redirects to PnR Big.

### Step 8 — Off-Ball Movement Shooter

```
IF MOVEMENT_SHOOTER_PCT >= HIGH_MOVEMENT (P60) AND movement/(movement+spotup) >= MOVEMENT_RATIO_MIN (0.30):
  primary_archetype = 'Off-Ball Movement Shooter'
```

v4.2: thresholds loosened from P70/0.40 to P60/0.30 to capture more genuine movement shooters (Tim Hardaway Jr., KCP, Kevin Huerter, Moses Moody, Dalton Knecht, etc.) who were previously lumped into Off-Ball Stationary Shooter.

### Step 8b — Perimeter Scorer (moderate self-creation) [NEW v4.2]

Captures players with enough ball handling to create their own shot but who are not pure spot-up shooters. Prevents over-inflating the Off-Ball Stationary Shooter bucket with self-creating wings.

```
IF ball_dom >= BD_BASE (P50) AND NOT playmaker AND pts_per36 >= LOW_SCORING (P25):
  primary_archetype = 'Perimeter Scorer'
```

Examples: Terry Rozier, Cam Whitmore, Brandon Miller, Derrick White, Brandin Podziemski.

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

- `Midrange Scorer` — MIDRANGE_FREQ >= P70 (strong evidence)
- `Inside-the-Arc` — MIDRANGE_FREQ >= P50 but below P70 (moderate evidence, renamed from Midrange Lean in v4.3)
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
| `emb_{archetype}` | Archetype embedding weight (11 dimensions, sums to ~1.0) |
| `emb_entropy` | Shannon entropy of embedding (higher = more hybrid) |
| `emb_entropy_norm` | Normalized entropy (0–1, relative to max possible) |
| `emb_dominance` | Weight of the strongest archetype component |
| `AT_RIM_FREQ` | shot-zone: restricted area fraction |
| `MIDRANGE_FREQ` | shot-zone: midrange fraction |
| `PAINT_FREQ` | shot-zone: in-paint non-RA |
| ... | (many other features like PTS_PER36, AST_PER36, TOUCHES, PRROLLMAN_POSS_PCT) |

Files:

- `data/processed/player_archetypes.parquet`
- `data/processed/archetype_embeddings.parquet` (standalone embedding file, v4.3)
- `data/tracking/{season}/shot_zones.parquet`
- `data/tracking_cache/shot_locations_{season}.json` (raw cache)

---

## 7b. Archetype Embedding (v4.3 — Soft Role Model)

v4.3 introduces a continuous, confidence-weighted archetype embedding for each player. Instead of a single hard classification, every player gets an 11-dimensional vector where each dimension corresponds to an archetype weight.

### Purpose

- **Player evaluation**: Identify hybrid players (e.g., LeBron has BDC + All-Around + Connector elements)
- **Advanced statistics**: Use embeddings as features in regression/prediction models
- **Similarity search**: Find players with comparable role profiles via cosine similarity
- **Classification refinement**: Embedding entropy flags uncertain classifications

### Algorithm

**Step 1 — Compute raw scores per archetype** (uncapped):

For each archetype, reuse the same normalized components from the classification hierarchy. Example for Ball Dominant Creator:

```
bd_score  = (ball_dom - BD_MIN) / (BD_P95 - BD_MIN)
pts_score = (pts_per36 - MODERATE_SCORING) / (PTS_P95 - MODERATE_SCORING)
ast_score = (ast_per36 - PLAYMAKER) / (AST_P95 - PLAYMAKER)

raw_bdc = 0.6 * (0.5 * bd_score + 0.5 * pts_score) + 0.4 * ast_score
```

Scores are NOT clipped — they can go negative (player is far from archetype) or > 1 (exceeds threshold).

Similar formulas exist for all 11 archetypes, using archetype-specific metric combinations and weights.

**Step 2 — Convert to confidence**:

```
distance = max(0, 1 - raw_score)
confidence = exp(-alpha * distance)    # alpha = 2.0 (decay rate)
```

This gives confidence ≈ 1.0 for raw_score ≥ 1 (strong fit) and confidence → 0.13 for raw_score ≈ 0 (weak fit).

**Step 3 — Build embedding**:

```
weight[A] = max(0, raw_score[A]) * confidence[A]
embedding = weight / sum(weights)    # L1-normalize
```

### Derived metrics

- `emb_entropy` — Shannon entropy of the embedding. Higher = more hybrid player.
- `emb_entropy_norm` — Entropy / log₂(11). Ranges from 0 (pure role) to 1 (uniform across all roles).
- `emb_dominance` — Weight of the strongest archetype. Higher = purer role player.

### Example embeddings (2024-25)

| Player | Dominance | Hybrid | Top components |
|---|---:|---:|---|
| LeBron James | 17% | 94% | All-Around 17%, Movement Shooter 16%, Connector 15%, BDC 11% |
| Stephen Curry | 43% | 75% | Movement Shooter 43%, PnR Pop 16%, Perimeter 11% |
| Jarrett Allen | 35% | 48% | PnR Rolling 35%, Off-Ball Finisher 35%, PnR Pop 29% |
| Klay Thompson | 52% | 54% | Movement Shooter 52%, Stationary Shooter 21%, PnR Pop 19% |
| Clint Capela | 36% | 47% | Off-Ball Finisher 36%, PnR Rolling 35%, PnR Pop 28% |

Interpretation: LeBron's high hybrid score (94%) reflects his versatility — he contributes across many roles. Klay's 52% dominance as a Movement Shooter aligned with his hard classification. Jarrett Allen splits evenly between PnR Rolling and Off-Ball Finisher, matching his real role as a rim-running big.

---

## 8. Distribution & Validation (2024-25 qualified)

Qualified players: 329 (MIN ≥ 500, GP ≥ 20, MPG ≥ 15)

Archetype counts (v4.3):

| Archetype | Count | % |
|---|---:|---:|
| Off-Ball Stationary Shooter | 45 | 13.7% |
| PnR Rolling Big | 40 | 12.2% |
| All-Around Scorer | 40 | 12.2% |
| Ball Dominant Creator | 40 | 12.2% |
| Off-Ball Finisher | 36 | 10.9% |
| Ballhandler | 31 | 9.4% |
| Connector | 29 | 8.8% |
| Off-Ball Movement Shooter | 27 | 8.2% |
| Interior Scorer | 19 | 5.8% |
| Perimeter Scorer | 17 | 5.2% |
| PnR Popping Big | 5 | 1.5% |
| Rotation Piece | 0 | 0.0% (ELIMINATED) |

Subtype totals (2024-25): Transition Player 44, High Volume 30, Heliocentric Guard 29, Elite Shooter 17, Playmaking Big 15, Midrange Scorer 13, Rim Finisher 6, Gravity Engine 6, Inside-the-Arc 5, Primary Scorer 3, Post Hub 2, Rebounder 1.

Representative validated classifications (v4.3):

| Player | Archetype | Subtype | Notes |
|---|---|---|---|
| DeMar DeRozan | Interior Scorer | Midrange Scorer | MR≈0.459 (v4 midrange detection) |
| Giannis Antetokounmpo | Ball Dominant Creator | Primary Scorer | AT_RIM≈0.571; ball dominance keeps him BDC |
| Stephen Curry | Ball Dominant Creator | Gravity Engine | High FG3A and efficiency |
| Kevin Durant | All-Around Scorer | High Volume | Mixed 2PT/3PT + high PTS/36 |
| Rudy Gobert / Clint Capela | PnR Rolling Big | — | v4.3: absorbed into PnR Big (was Off-Ball Finisher) |
| Jarrett Allen | PnR Rolling Big | — | v4.3: absorbed from Interior Scorer (rim-runner, no skill finishing) |
| Daniel Gafford | PnR Rolling Big | — | v4.3: absorbed from Interior Scorer (rim-runner, no skill finishing) |
| LeBron James | Ball Dominant Creator | Heliocentric Guard | High BD & playmaking |
| Luka Dončić | Ball Dominant Creator | Heliocentric Guard | High BD & playmaking |
| Anthony Edwards | All-Around Scorer | High Volume | High PTS/36 and mixed profile |
| Onyeka Okongwu | PnR Rolling Big | — | v4.2: redirected from Interior Scorer (rim-dependent, p+mr=0.21) |
| Christian Braun | Off-Ball Finisher | Transition Player | v4.2: redirected (rim-dependent, p+mr=0.19) |
| Klay Thompson | Off-Ball Movement Shooter | — | High movement ratio |
| Cam Whitmore | Perimeter Scorer | High Volume | v4.2: redirected from Stationary (BD=0.21, PTS=20.8) |
| Derrick White | Perimeter Scorer | — | v4.2: self-creating perimeter wing (BD=0.27) |
| Brandon Miller | Perimeter Scorer | High Volume | v4.2: self-creating wing (BD=0.34, PTS=22.1) |

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
  Primary Scorer (BD >= P60)? ──Yes──> Interior* / Perimeter / All-Around
   │                              *Interior requires skill finishing OR star scoring (v4.2)
   │No
   ▼
  Interior Scorer low-BD (FG2A+scoring+AT_RIM, skill gate)? ──Yes──> Interior Scorer
   │No
   ▼
  Connector? PnR Big? Off-Ball Finisher? ──Yes──> respective archetype
   │No
   ▼
  Movement Shooter (P60, ratio 0.30)? ──Yes──> Off-Ball Movement Shooter
   │No
   ▼
  Perimeter Scorer (BD >= P50, scoring > P25)? ──Yes──> Perimeter Scorer (v4.2)
   │No
   ▼
  Stationary Shooter (spotup >= P50)? ──Yes──> Off-Ball Stationary Shooter
   │No
   ▼
  Best-fit fallback (FALLBACK_VECTORS) ──> nearest archetype with low confidence
```

---

## 10. Design Notes & Rationale

- Percentiles: keep the classifier robust to league trends. Pxx gates are recomputed per-season from qualified players.
- Shot zones: give the classifier direct signal for midrange vs rim usage instead of proxying with FG2A_RATE only.
- Rotation Piece removal: opaque catch-alls are harmful for downstream model interpretability; best-fit fallback gives deterministic, low-confidence assignments that are actionable.
- All-Around earlier: captures playmaker+scorer hybrids that previously were split across Ballhandler/BDC.
- v4.2 Skill finishing gate: rim-dependent non-stars (only dunks/layups, no floaters/hooks/midrange) are better described as PnR Rolling Big or Off-Ball Finisher than Interior Scorer. The gate requires `paint_freq + midrange_freq >= 0.25` unless the player is a star scorer (>= P70 PTS/36).
- v4.2 Movement Shooter loosening: P70 was too restrictive; many genuine movement shooters (KCP, Tim Hardaway Jr., Huerter) were lumped with spot-up shooters. P60 + ratio 0.30 better separates movement from static shooters.
- v4.2 Perimeter Scorer expansion: players with moderate ball dominance (P50+) who create their own shot should not be classified as Off-Ball Stationary Shooters; they’re self-creating perimeter scorers.

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

Last updated: 2026-02-07 (v4.2)

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

## 3. Percentile-Based Thresholds (v4.2)

v4.2 continues to compute percentile thresholds per-season from qualified players (MIN ≥ 500, GP ≥ 20, MPG ≥ 15). v4.2 loosened the Movement Shooter threshold and added new static thresholds.

Selected thresholds (2024-25 example values):

| Name | Column | Percentile | Example (2024-25) |
|---|---:|---:|---:|
| `BD_MIN` | BALL_DOMINANT_PCT | P80 | 0.372 |
| `BD_BASE` | BALL_DOMINANT_PCT | P50 | ~0.17 |
| `PLAYMAKER` | AST_PER36 | P75 | 5.074 |
| `MODERATE_SCORING` | PTS_PER36 | P50 | 17.23 |
| `LOW_SCORING` | PTS_PER36 | P25 | ~12.5 |
| `HIGH_AT_RIM` | AT_RIM_FREQ | P70 | 0.294 |
| `HIGH_MIDRANGE` | MIDRANGE_FREQ | P70 | 0.093 |
| `MODERATE_MIDRANGE` | MIDRANGE_FREQ | P50 | 0.064 |
| `AT_RIM_PAINT_P60` | AT_RIM_PLUS_PAINT_FREQ | P60 | 0.448 |
| `HIGH_MOVEMENT` | MOVEMENT_SHOOTER_PCT | P60 | 0.081 |
| `HIGH_PRROLLMAN` | PRROLLMAN_POSS_PCT | P70 | 0.065 |
| `FG2A_INTERIOR` | FG2A_RATE | P70 | 0.673 |
| `FG2A_PERIMETER` | FG2A_RATE | P30 | 0.478 |

New static thresholds (v4.2): `SKILL_FINISHING_MIN = 0.25` (paint_freq + midrange_freq floor for non-star Interior Scorer), `MOVEMENT_RATIO_MIN = 0.30` (lowered from 0.40).

---

## 4. Classification Hierarchy (v4.2)

The hierarchy is the same as v4 with these v4.2 additions:

- **Interior Scorer skill gate**: non-star Interior Scorers must show `paint_freq + midrange_freq >= 0.25` (skill finishing). Rim-dependent players (only dunks/layups) fall through to PnR Big or Off-Ball Finisher.
- **Movement Shooter loosened**: `HIGH_MOVEMENT` lowered from P70 → P60, `MOVEMENT_RATIO_MIN` from 0.40 → 0.30.
- **Perimeter Scorer (Step 7b)**: new step captures players with `BD >= P50` + scoring above `P25` before they reach Off-Ball Stationary Shooter.

Condensed flow:

0) Minimum requirements.
1) BDC main path / 1b) Offensive Hub.
2) All-Around Scorer (playmaker + high scorer).
3) Ballhandler (playmaker, not high scorer).
4) Primary Scorers (Interior* / Perimeter / All-Around). *Interior requires skill finishing OR star scoring.
4b) Low-BD Interior Scorer (skill gate applies).
5) Connector / PnR Big / Off-Ball Finisher.
7) Movement Shooter (P60, ratio 0.30).
7b) Perimeter Scorer (BD >= P50, PTS >= P25).
8) Stationary Shooter (SPOTUP >= P50).
9) Best-fit fallback (FALLBACK_VECTORS).

---

## 5. Subtypes (notable additions)

- `Midrange Scorer` — `MIDRANGE_FREQ >= HIGH_MIDRANGE (P70)`. Example: DeMar DeRozan (MR≈0.459).
- `Inside-the-Arc` — `MIDRANGE_FREQ` between P50 and P70. Moderate midrange usage (v4.1, renamed from Midrange Lean in v4.3).
- `Rim Finisher` — interior scorers whose `AT_RIM_FREQ` ≥ P70. Example: Mark Williams (AT_RIM≈0.670).
- Existing subtypes (High Volume, Heliocentric Guard, Gravity Engine, Post Hub, Transition Player, Elite Shooter, Playmaking Big) remain.

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

## 8. Distribution (2024-25 qualified players, v4.2 counts)

Qualified: 329 players (MIN ≥ 500, GP ≥ 20, MPG ≥ 15)

| Archetype | Count | % |
|---|---:|---:|
| Off-Ball Finisher | 54 | 16.4% |
| Off-Ball Stationary Shooter | 45 | 13.7% |
| All-Around Scorer | 40 | 12.2% |
| Ball Dominant Creator | 40 | 12.2% |
| Interior Scorer | 33 | 10.0% |
| Ballhandler | 31 | 9.4% |
| Connector | 28 | 8.5% |
| Off-Ball Movement Shooter | 26 | 7.9% |
| Perimeter Scorer | 14 | 4.3% |
| PnR Rolling Big | 13 | 4.0% |
| PnR Popping Big | 5 | 1.5% |
| Rotation Piece | 0 | 0.0% (ELIMINATED) |

v4.2 distribution change vs v4: Stationary −19, Movement +14, Perimeter +7, Interior −4, PnR Rolling +1.

---

## 9. Validation Highlights

- DeMar DeRozan: classified `Interior Scorer / Midrange Scorer` (MR ~0.459) — matches human expectation.
- Giannis Antetokounmpo: remains `Ball Dominant Creator / Primary Scorer` with high `AT_RIM_FREQ` (~0.571).
- Onyeka Okongwu: `PnR Rolling Big` (v4.2 redirected from Interior Scorer — rim-dependent non-star, p+mr=0.21).
- Christian Braun: `Off-Ball Finisher / Transition Player` (v4.2 redirected — rim-dependent, p+mr=0.19).
- Cam Whitmore: `Perimeter Scorer / High Volume` (v4.2 — self-creating wing, BD=0.21, PTS=20.8).
- Brandon Miller: `Perimeter Scorer / High Volume` (v4.2 — BD=0.34, PTS=22.1).
- Tim Hardaway Jr., KCP, Kevin Huerter: `Off-Ball Movement Shooter` (v4.2 loosened thresholds).
- Rotation Piece assignments: 0 across all seasons.

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

Last updated: 2026-02-07 (v4.2)
