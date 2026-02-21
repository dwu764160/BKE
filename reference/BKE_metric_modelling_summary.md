# BKE Metric Modelling Summary (v2.6)

## Overview — Portable Talent vs Role-Dependent Impact Decomposition Engine

BKE v2.6 is a multi-layer hierarchical decomposition engine that separates scalable talent from system amplification. v2.6 builds on v2.5 and adds calibrated hierarchical shrinkage in Layer 1, implemented year-over-year backtesting, an updated RUE blend, and scheme-stability handling that is bonus-only in Total Impact.

For every player-season, it produces:

```
Total Impact = Portable Talent + Role-Dependent Impact
```

Where Role-Dependent Impact is further decomposed:

```
Role-Dependent Impact = RUE + Archetype Elevation + Scheme Amplification
```

**v2.6 aggregation pipeline:** Raw → Z-score → Archetype-Neutralized/Position-Conditional Z → Hierarchical Shrinkage → Weighted Sum → Final Z → Percentile (presentation only)

Percentiles are strictly terminal — never used as inputs to any layer or aggregation step (Fix #2).

**v2.6 key additions over v2.5:**
- **Bayesian hierarchical shrinkage (implemented):** Player-level shrinkage toward archetype/position priors with calibration by GP × minutes reliability.
- **Backtesting framework (implemented):** Spearman rank stability, tier accuracy, RMSE, archetype stability, position-split diagnostics.
- **RUE refinement (implemented):** Blended RUE = 40% role alignment cosine + 30% efficiency surplus quality + 30% volume-weighted utilization.
- **Scheme contribution fix (implemented):** Scheme stability in Total Impact is bonus-only (negative scheme z no longer subtracts TI).

**v2.5 key additions over v2.0 (retained):**
- **Archetype-conditional neutralization (Fix #3):** Subtract archetype-expected z-score from 6 of 8 dimensions, so Layer 1C measures ability *above what's expected for the player's role*
- **Three-level z-scores (Fix #1):** League, positional, and archetype z-scores for all dimension columns
- **portability_ratio removed from public outputs (Fix #4):** Retained internally as alias for portability_index
- **MF-1:** Enhanced turnover control (TOV_PER_TOUCH, DRIVE_TOV_RATE)
- **MF-4:** Expanded defensive playmaking (CHARGES_DRAWN, DEF_LOOSE_BALLS_RECOVERED from hustle stats)
- **MF-5:** Playmaking split into creation (AST/36, Playmaking Score) + pressure (Potential AST/36, Secondary AST/36, DRIVE_AST_RATIO, PASSES_MADE_PER36)
- **Bug fix:** elevation_z preserved through Layer 3 (was being dropped)

**Guiding principles (unchanged from v2.0):**
- Strict separation: role assignment (archetype) is independent from value estimation (BKE)
- Portable talent captures context-neutral skill; role-dependent impact captures environment-specific value
- Z-score aggregation preserves interval meaning; percentiles are presentation-only
- Portability is measured structurally (variance across contexts), not compositionally (ratio of scores)
- Small-cohort Bayesian shrinkage and noise reduction for defensive/hustle metrics
- Cross-layer independence enforced: each layer must be measurable with others held constant

---

## Architecture — 7 Modules in `src/modeling/`

| Module | Purpose |
|--------|---------|
| `model_config.py` | Centralized config: paths, weights, thresholds, z-score/shrinkage settings, 8-dimension definitions, NeutralizationConfig |
| `percentile_engine.py` | Three-level percentile standardization + z-score functions + Bayesian shrinkage + archetype-conditional neutralization |
| `layer1_portable_talent.py` | Context-neutral talent estimation (25% RAPM + 20% Playtype + 55% 8-Dimension Model) with neutralization |
| `layer2_role_utilization.py` | Playtype surplus + Role Utilization Efficiency (RUE) + z-score outputs |
| `layer3_archetype_elevation.py` | Player impact above/below archetype baseline + elevation z-scores |
| `layer4_scheme_amplification.py` | Context sensitivity estimation + scheme stability z-scores |
| `decomposition_engine.py` | Final z-score assembly + True Portability Index + output generation |

---

## Step-by-Step Process

### Layer 0 — Data Foundation

**Inputs consumed:**
- `player_rapm.parquet` — Multi-year Bayesian RAPM (pooled + single-season, O/D split)
- `modeling_inputs_all.parquet` — DARKO projections, linear stats (TS%, eFG%, USG%, AST%, TOV%, OREB%, DREB%, per-36 rates)
- `player_profiles_advanced.parquet` — Box score profiles, four factors, on/off, ORTG/DRTG
- `player_archetypes.parquet` — Offensive archetypes + playtype columns + tracking data (drives, rim attempts, paint frequency, FT rate, potential/secondary assists, turnovers, passes made, touches, pcv entropy)
- `defensive_archetypes_v2.parquet` — Defensive archetypes, switch score, versatility, hustle score, engagement score, deflections, matchup diversity, d_results, assignment difficulty
- `player_position_estimates.parquet` — Positional percentage estimates for bucketing
- **NEW in v2.5:** `data/tracking/{season}/hustle_stats.parquet` — CHARGES_DRAWN, DEF_LOOSE_BALLS_RECOVERED, LOOSE_BALLS_RECOVERED, SCREEN_ASSISTS

**Qualification filters:** MIN >= 500, GP >= 20, MPG >= 15.0, Possessions >= 500

---

### Layer 1 — True Portable Talent (Context-Neutral)

> "How good is this player independent of role volume and scheme?"

**1A. RAPM Backbone (25%):** Prefers pooled-split RAPM for O/D separation; falls back to single-season-split where missing. Z-scored within season.

**1B. Playtype Efficiency (20%):** Aggregated playtype z-scores weighted by possession share.

**1C. Portable Dimension Model (55%) — 8 Independent Dimensions (v2.5 Neutralized):**

| # | Dimension | Domain | Sub-metrics | Weight | Neutralized? |
|---|-----------|--------|------------|--------|--------------|
| 1 | Shooting Gravity | Offense | TS% (adj), FG3% (adj), FG3A/36, C&S FG3% | 0.125 | Yes |
| 2 | Driving Gravity | Offense | DRIVES/36, AT_RIM_FREQ, FT_RATE, PAINT_FREQ | 0.125 | Yes |
| 3 | Playmaking Creation | Offense | **Creation** (50%): AST/36, Playmaking Score; **Pressure** (50%): Potential AST/36, Secondary AST/36, DRIVE_AST_RATIO, PASSES_MADE_PER36 | 0.125 | Yes |
| 4 | Extra Possession | Cross | OREB%, DREB%, REB/36 (split 40% offense / 60% defense) | 0.125 | Yes |
| 5 | Defensive Playmaking | Defense | STL/100, BLK%, Deflections, hustle_score, engagement_score, **CHARGES_DRAWN**, **DEF_LOOSE_BALLS_RECOVERED** (Bayesian shrinkage) | 0.125 | No (already context-neutral) |
| 6 | Defensive Impact | Defense | DRAPM, DRTG (inverted), d_results_pctl | 0.125 | No (already context-neutral) |
| 7 | Turnover Control | Offense | TOV%, TOV/36, **TOV_PER_TOUCH**, **DRIVE_TOV_RATE** (all inverted) | 0.125 | Yes |
| 8 | Defensive Versatility | Defense | switch_score, versatility_pctl, assignment_difficulty, matchup_diversity_pctl | 0.125 | No (structural) |

**Bold** items are new in v2.5.

Each sub-metric is z-scored within season (winsorized at ±3.5σ). Missing components default to 0 (league average). Dimension composites are averaged z-scores of their sub-metrics.

**v2.5 Archetype-Conditional Neutralization (Fix #3):**

After computing raw dimension z-scores, 6 of 8 dimensions are neutralized:

```
Neutralized_z = Observed_z - E[z | archetype, season]
```

Where `E[z | archetype, season]` is the mean z-score for that archetype cohort in that season. This ensures Layer 1C measures ability *above what's expected for the player's role*, preventing archetype-typical skills from inflating portable talent scores.

Small-cohort handling: cohorts < 8 shrink expected z toward league mean (0) by 50%. Cohorts < 2 assume expected = 0.

Dimensions **not** neutralized: Defensive Impact (RAPM is already regularized/context-neutral), Defensive Versatility (structural metric).

Pre-neutralization values stored as `{dim}_z_raw` columns for diagnostics.

**v2.5 Three-Level Z-Scores (Fix #1):**

For each dimension z-score column:
- **League z**: the raw (or neutralized) z-score within season
- **Positional z**: z-score within season × position_bucket groups
- **Archetype z**: z-score within season × archetype groups (min cohort 5)

Stored as `{col}_positional` and `{col}_archetype` suffixed columns.

**v2.5 Derived Metrics (computed before dimension model):**

| Metric | Formula | Used In |
|--------|---------|---------|
| TOV_PER_TOUCH | TOV / TOUCHES | Turnover Control (MF-1) |
| DRIVE_TOV_RATE | DRIVE_TOV / DRIVES | Turnover Control (MF-1) |
| DRIVE_AST_RATIO | DRIVE_AST / DRIVES | Playmaking Pressure (MF-5) |
| PASSES_MADE_PER36 | PASSES_MADE * 36 / MPG | Playmaking Pressure (MF-5) |

**1D. Portable Talent Score (PTS):**

```
PTS_z = 0.25 * RAPM_z + 0.20 * Playtype_z + 0.55 * Dimension_z
```

Z-score composite converted to percentile rank for presentation. CDF-based percentile also computed for cross-era comparison.

**Output:** PTS range 0.3-100.0 across 950 qualified player-seasons (3 seasons), 0 NaN.

---

### Layer 2 — Role Utilization Efficiency

> "How much of their talent is actually being expressed in their current role?"

**2A. Usage-Conditioned Efficiency:**

For each of 9 playtypes (Isolation, PnR Ball Handler, Post-Up, Cut, PnR Roll Man, Handoff, Off Screen, Spot-Up, Transition):

```
Surplus = Player_PPP - League_Avg_PPP_at_same_usage_bucket
```

Usage buckets: [0, 0.05, 0.10, 0.20, 0.35, 1.0]. Minimum 0.3 possessions per game per playtype (≈25 season total). Possession-weighted total surplus computed.

**v2.0 fix:** v1.5 used threshold of 25, but data stores per-game possession counts (max ~10). Corrected to 0.3 per game. Surplus now has meaningful non-zero values.

Z-scores added for all surplus metrics via `add_league_z_scores()`.

**2B. Role Utilization Efficiency (RUE):**

```
RUE = cosine_similarity(observed_playtype_vector, optimal_playtype_vector)
```

Where the optimal vector is the mean playtype distribution of the top-half (by RAPM) of the player's archetype cohort. Cosine similarity is already interval-scaled and does not suffer from percentile aggregation distortion.

Z-scores added for RUE raw scores.

**Output:** RUE range 0.3-100.0. Surplus now active with meaningful variance.

---

### Layer 3 — Archetype Elevation

> "How much better is this player than the average player in their archetype?"

**3A. Archetype Baselines:** For each archetype × season: mean/median/std RAPM, mean playtype surplus, mean TS%.

**3B. Elevation Score:**

```
Elevation_RAPM = Player_RAPM - Archetype_mean_RAPM
```

Same for surplus and efficiency. Each z-scored within season, then combined:

```
Elevation_composite = 0.50 * z_RAPM + 0.30 * z_surplus + 0.20 * z_efficiency
```

v2.0 adds `elevation_z` output for z-score aggregation in final decomposition.

Classified into tiers: Below Baseline < Baseline < Above Average < Elite < Archetype-Best.

**Output:** Elevation range 0.3-100.0. 12 archetypes with baselines per season.

---

### Layer 4 — Scheme Amplification

> "How much of this player's impact depends on their specific environment?"

**Full mode** (with possession data): Uses lineup-chunk variance and on/off splits.

**Fast mode** (proxy features, no possession data): Approximates context sensitivity using:
- **RAPM vs DARKO agreement** — If two independent models agree, the player is more likely portable
- **Playtype entropy** — Diversified usage = harder to scheme against = more portable
- **Role confidence** — Higher archetype certainty = less context-dependent

v2.0 adds `scheme_stability_z` output for z-score aggregation.

**Output:** Stability range 0.3-100.0. Classification: ~30% portable, ~40% moderate, ~30% system-dependent.

---

### Final Decomposition

**Role-Dependent Impact z-score:**

```
RDIS_z = 0.40 * RUE_z + 0.40 * Elevation_z + 0.20 * (-Scheme_z)
```

(Inverted scheme z: high stability = low role-dependence)

**Total Impact z-score:**

```
Total_z = (w_PTS * PTS_z + w_RUE * RUE_z + w_Elev * Elev_z + w_Scheme * Scheme_z) / sum(w)
```

All weights default to 1.0. Percentile-ranked within season (qualified only) for presentation.

CDF-based percentile (normal distribution) also computed for cross-era comparisons.

**True Portability Index (v2.0 Fix #2):**

```
Portability Index = Weighted sum of 4 structural components (each 0-1)
```

| Component | Weight | What It Measures |
|-----------|--------|-----------------|
| Lineup Stability | 30% | Multi-model agreement (RAPM vs DARKO) + dimensional consistency |
| Role Elasticity | 25% | Surplus consistency across playtypes + usage entropy |
| Archetype Transfer | 20% | Offensive/defensive balance + role confidence proximity |
| Context Sensitivity | 25% | Scheme stability + ORAPM/DRAPM balance |

Classification: ≥0.70 = Scalable Star, ≤0.40 = System Player, between = Context-Moderate.

This is **structural** (variance of impact across contexts), not **compositional** (ratio of one score to another).

**Impact Tiers:** Elite (90-100th), All-Star (75-90th), Starter (50-75th), Rotation (25-50th), Fringe (0-25th).

---

## Mathematical Definitions

- **Z-score:** z = (x - μ) / σ, winsorized at ±3.5σ
- **Bayesian shrinkage:** x_adj = (1 - a) * x + a * prior, where a = shrinkage_strength * max(0, 1 - GP/min_gp)
- **Z-to-percentile (CDF):** P = Φ(z) * 100, where Φ is the standard normal CDF
- **Percentile rank:** P(x_i) = rank(x_i) / n * 100 (average method) — presentation only
- **Small-cohort shrinkage:** P_adj = (1-a) * P_group + a * P_league, where a = 0.30 for groups < 10
- **Cosine similarity:** cos(a, b) = (a · b) / (|a| * |b|)
- **Shannon entropy:** H = -sum(p_i * ln(p_i))
- **Luck regression:** x_adj = μ + (1-r) × (x - μ) where r is regression rate (3PT: 40%, TS: 30%)
- **Weighted z composite:** Σ(w_i * z_i) / Σ(w_i)

---

## Example Output Card (v2.5)

```
Nikola Jokic (2022-23):
  Portable Talent:     Score 89.2 | z=+1.41 | CDF 92.1%
                       League 100th | Position 100th | Archetype 100th
  Role-Dependent:      Score 63.0 | z=+0.52
  Elevation:           Archetype-Best
  Scheme Stability:    Portable
  Total Impact:        Score 85.0 | z=+0.71 | CDF 76.1%
  Portability Index:   0.82 (Scalable Star)
    Lineup Stability:  0.89 | Role Elasticity: 0.91
    Archetype Transfer: 0.72 | Context Sensitivity: 0.76
  RAPM: +5.91 | ORAPM: +3.72 | DRAPM: +2.19

  v2.5 Dimension Detail (neutralized z-scores):
    Shooting Gravity:      z=-0.32 (raw: +0.45, archetype expects +0.77)
    Playmaking Creation:   z=+2.10 (creation: +2.30, pressure: +1.90)
    Turnover Control:      z=+0.85 (TOV_PER_TOUCH: +1.1, DRIVE_TOV: +0.6)
    Def Playmaking:        z=+1.45 (includes CHARGES_DRAWN, LOOSE_BALLS)
```

---

## Outputs

| File | Format | Description |
|------|--------|-------------|
| `bke_v26_decomposition.parquet` | Parquet | Full decomposition table (~344 columns, 1971 rows) |
| `bke_v26_decomposition.csv` | CSV | Same as above, human-readable |
| `bke_v26_report.json` | JSON | Summary report with top-10 lists, z-score/distribution stats |
| `bke_v26_backtest.json` | JSON | Year-over-year backtesting diagnostics |

---

## Validation Results (v2.5)

| Metric | All Seasons |
|--------|-------------|
| Qualified players | 950 |
| Seasons | 2022-23, 2023-24, 2024-25 |
| PTS range | 0.3-100.0 |
| RUE range | 0.3-100.0 |
| Elevation range | 0.3-100.0 |
| Scheme stability range | 0.3-100.0 |
| NaN in final output (all key cols) | 0 |
| dim_shooting_gravity_z std | 0.522 |
| dim_driving_gravity_z std | 0.462 |
| dim_playmaking_creation_z std | 0.370 |
| dim_extra_possession_z std | 0.674 |
| dim_defensive_playmaking_z std | 0.520 |
| dim_defensive_impact_z std | 0.825 |
| dim_turnover_control_z std | 0.450 |
| dim_defensive_versatility_z std | 0.769 |
| dimension_model_z std | 0.247 |
| portable_talent_z std | 0.461 |
| total_impact_z std | 0.453 |
| portability_index_raw mean | 0.611 |
| portability_index_raw std | 0.076 |
| Surplus abs mean | 15.42 |
| Neutralization raw cols | 6 (dim 1-4, 5 def playmaking, 7 TO) |
| Positional z-score cols | 11 |
| Archetype z-score cols | 14 |
| Runtime | ~8.3s |

Portability class distribution: 286 Scalable Star (30%), 285 Context-Moderate (30%), 379 System Player (40%).

Note: Neutralized dimension z-score stds are slightly lower than v2.0 raw stds, which is expected — removing archetype-expected variation compresses the distribution marginally.

---

## Changes from v2.0 → v2.5

### Fix #1: Three-Level Z-Scores
**v2.0:** League-level z-scores only for dimension columns.
**v2.5:** For each dimension z-score, three levels are computed:
- **League z**: z-score within season (same as v2.0)
- **Positional z**: z-score within season × position_bucket (11 columns)
- **Archetype z**: z-score within season × primary_archetype, min cohort 5 (14 columns)

### Fix #2: Percentiles Strictly Terminal
**v2.0:** Percentiles were already mostly terminal, but some code paths could feed percentiles into aggregation.
**v2.5:** Enforced strictly — percentiles are ONLY computed for presentation at the very end. No percentile is ever used as an input to any layer or aggregation step.

### Fix #3: Archetype-Conditional Neutralization (THE BIG ONE)
**v2.0:** Dimension z-scores were raw league z-scores. A player's shooting gravity z-score compared them to all players, not to players in their role. This inflated portability for archetype-typical skills.
**v2.5:** 6 of 8 dimensions are neutralized by subtracting the archetype-expected z-score:
```
Neutralized_z = Observed_z - E[z | archetype, season]
```
Dimensions neutralized: Shooting Gravity, Driving Gravity, Playmaking Creation, Extra Possession, Turnover Control, Defensive Playmaking.
Dimensions NOT neutralized: Defensive Impact (RAPM already regularized), Defensive Versatility (structural).
Small-cohort shrinkage: cohorts < 8 shrink toward league mean (50%), cohorts < 2 assume expected = 0.

### Fix #4: portability_ratio Removed from Public Outputs
**v2.0:** Both `portability_ratio` and `portability_index` were in player cards and reports.
**v2.5:** `portability_ratio` removed from player card and report. Retained internally as alias for `portability_index` for backward compatibility.

### Fix #5: Archetype Instability (Noted, Not Implemented)
**Deferred beyond v2.6:** Archetype assignment should use probabilistic membership (soft assignment) instead of hard classification to reduce instability at decision boundaries.

### MF-1: Enhanced Turnover Control
**v2.0:** TOV%, TOV/36 only.
**v2.5:** Adds TOV_PER_TOUCH (turnovers per touch — ball security) and DRIVE_TOV_RATE (drive turnovers per drive — decision-making under pressure). Both inverted (lower is better). Derived from existing TOUCHES and DRIVES tracking data.

### MF-4: Expanded Defensive Playmaking
**v2.0:** STL/100, BLK%, Deflections, hustle_score, engagement_score.
**v2.5:** Adds CHARGES_DRAWN and DEF_LOOSE_BALLS_RECOVERED from `hustle_stats.parquet` tracking data. Both are Bayesian-shrunk alongside existing metrics. No new data fetcher was needed — hustle stats already existed in `data/tracking/{season}/hustle_stats.parquet`.

### MF-5: Playmaking Split (Creation + Pressure)
**v2.0:** Single "Playmaking" dimension averaging AST/36, Playmaking Score, Potential AST/36, Secondary AST/36.
**v2.5:** Split into two 50/50 sub-components:
- **Creation** (direct assist generation): AST/36, Playmaking Score
- **Pressure** (collapse/rotation forcing): Potential AST/36, Secondary AST/36, DRIVE_AST_RATIO, PASSES_MADE_PER36
Sub-component z-scores stored for diagnostics (`dim_playmaking_creation_sub_z`, `dim_playmaking_pressure_sub_z`).

### Bug Fix: elevation_z Preserved Through Layer 3
**v2.0:** Layer 3's cleanup code dropped ALL `_z` columns (`z_cols = [c for c in result.columns if c.endswith("_z")]`), including `elevation_z` which the decomposition engine needed.
**v2.5:** Cleanup excludes `elevation_z` from the drop list.

---

## Changes from v1.5 → v2.0

### 1. Z-Score Aggregation (Fix #1)
**v1.5:** Added percentile scores directly (`Final = avg(percentiles)`), distorting tail comparisons due to rank-based compression.
**v2.0:** All internal aggregation uses z-scores. Raw → Z → Weighted Sum → Final Z → Percentile (presentation only). This preserves interval meaning and fixes the tail-compression artifact.

### 2. True Portability Index (Fix #2)
**v1.5:** `Portability = PTS / (PTS + RDIS)` — a compositional ratio that measured score proportion, not transfer stability. The denominator was endogenous (contains portable component), and the ratio shrank for high-impact players even if truly portable.
**v2.0:** Variance-based Portability Index with 4 structural components: Lineup Stability, Role Elasticity, Archetype Transfer, Context Sensitivity. Measures `1 - Normalized Impact Variance Across Contexts`.

### 3. 8-Dimension Portable Model (Layer 1C Expansion)
**v1.5:** 6 skill components with min-max normalization: Shooting Gravity, Passing Efficiency, Rim Protection, Defensive Versatility, Turnover Control, Rebounding.
**v2.0:** 8 z-scored dimensions: Shooting Gravity, Driving Gravity (revised), Playmaking, Extra Possession Creation (cross-domain), Defensive Playmaking (expanded), Defensive Impact (RAPM-informed), Turnover Control (standalone), Defensive Versatility.

### 4. Layer 1 Weight Rebalance
**v1.5:** 35% RAPM + 65% Skill Components.
**v2.0:** 25% RAPM + 20% Playtype + 55% Dimension Model. Layer 1C is now the primary portability engine.

### 5. Driving Gravity Revision
**v1.5:** Included FT% and perimeter initiation filtering.
**v2.0:** Removed FT%, focused on rim pressure creation: DRIVES/36, AT_RIM_FREQ, FT_RATE (fouls drawn rate), PAINT_FREQ.

### 6. Turnover Control Restored
**v1.5:** Embedded in Passing Efficiency dimension.
**v2.0:** Standalone offensive dimension. TOV% and TOV/36 inverted.

### 7. Playtype Surplus Fix
**v1.5:** `min_playtype_poss=25` never matched per-game data (max ~10/game). All surplus values were zero.
**v2.0:** `min_playtype_poss=0.3` (per-game threshold ≈ 25 season total). Surplus now produces meaningful non-zero values.

### 8. Bayesian Shrinkage
**v1.5:** Not implemented.
**v2.0:** Empirical Bayes shrinkage for defensive playmaking metrics and small-sample signals. Configurable shrinkage strength and GP-based weighting.

### 9. NaN Handling in Dimensions
**v1.5:** Missing z-components propagated NaN through dimension averages, causing 107 NaN in 3 dimensions.
**v2.0:** Missing z-components filled with 0 (league average) before averaging. All dimensions have 0 NaN for qualified players.

### 10. Expanded Data Loading
**v1.5:** Offensive archetypes loader only kept core playtype columns. Missing tracking data (drives, rim attempts, paint frequency, assists per 36, entropy).
**v2.0:** Loaders expanded to include driving/tracking/turnover/entropy columns from archetypes and hustle/engagement/deflection/matchup columns from defensive archetypes.

---

## Key Differences from v2.5 Plan → Implementation

### 1. Portability Components Use Proxy Approximations (unchanged from v2.0)
**Plan:** Specified lineup stability from "variance across teammate clusters," on/off splits from PBP, and usage shift simulation.
**Implementation:** Approximates using (a) multi-model agreement (RAPM vs DARKO variance), (b) dimensional consistency (variance across 8 dimensions), (c) surplus CoV + usage entropy, and (d) scheme stability + ORAPM/DRAPM balance.

### 2. Stability-Weighted Dimension Scaling Deferred (unchanged from v2.0)
**Plan:** Specified year-to-year stability weighting, predictive correlation with future RAPM, and cross-team transfer reliability.
**Implementation:** Equal weighting (0.125 each) across 8 dimensions. Stability-based and predictive weighting deferred to v2.6+ (requires multi-year backtesting infrastructure).

### 3. Some Planned Metrics Not Available (unchanged from v2.0)
**Plan:** Listed and-1 frequency, bad pass frequency, live-ball turnover rate, on/off spacing effect, box creation, advantage creation events, disruption rate, cross-match success, shot quality allowed.
**Implementation:** Uses available tracking data proxies. v2.5 added TOV_PER_TOUCH, DRIVE_TOV_RATE, DRIVE_AST_RATIO, PASSES_MADE_PER36, CHARGES_DRAWN, DEF_LOOSE_BALLS_RECOVERED — narrowing the gap but not closing it entirely.

### 4. Bayesian Hierarchical Shrinkage Implemented in v2.6
**Plan:** Structured hierarchy (Player > Archetype > Position).
**Implementation:** Hierarchical shrinkage now implemented in Layer 1 with reliability calibration by games played and minutes; full probabilistic MCMC regression remains out of scope.

### 5. Backtesting Implemented in v2.6
**Plan:** Multi-year stabilization and transfer diagnostics.
**Implementation:** Implemented with rank correlation, tier accuracy, RMSE, archetype stability, position splits, and big-mover reporting.

### 6. Neutralization Applied to 6 of 8 Dimensions (partially different from plan)
**Plan:** Implied neutralization for all dimensions.
**Implementation:** Defensive Impact and Defensive Versatility are NOT neutralized — they are already context-neutral by construction (RAPM is regularized; versatility is structural). This is an intentional design decision, not a limitation.

### 7. Playmaking Split is 50/50 Creation+Pressure (implementation detail)
**Plan:** Specified separate creation and pressure sub-components.
**Implementation:** Implemented as a 50/50 blend of creation sub-z and pressure sub-z within the single playmaking_creation dimension. Sub-component z-scores stored for diagnostics.

---

## Deliverables (v2.6)

- 4-layer hierarchical decomposition with z-score aggregation backbone
- 9-dimension portable talent model (Layer 1C) with archetype-conditional neutralization and self-creation
- Three-level z-scores (league, positional, archetype) for all dimensions
- True Portability Index (4 structural components, variance-based)
- Enhanced turnover control (TOV_PER_TOUCH, DRIVE_TOV_RATE)
- Expanded defensive playmaking (CHARGES_DRAWN, DEF_LOOSE_BALLS_RECOVERED)
- Playmaking creation+pressure split with sub-component diagnostics
- portability_ratio removed from public outputs
- elevation_z bug fix in Layer 3
- Three-level percentile standardization (presentation layer only)
- Bayesian shrinkage for noisy defensive/hustle metrics
- Fixed playtype surplus (corrected per-game threshold)
- Impact tiers (Elite / All-Star / Starter / Rotation / Fringe)
- Scheme classification (Portable / Context-Moderate / System-Dependent)
- All 950 qualified player-seasons across 3 seasons scored with zero NaN
- Full output: parquet (~344 cols) + CSV + JSON report + JSON backtest diagnostics
- Modular, explainable, and basketball-first architecture

## Known Issues / Missing Features (Deferred to Later Iterations)

1. **Fix #5: Archetype assignment instability** → Deferred beyond v2.6 (probabilistic archetype membership / soft assignment)
2. **MF-3: Archetype confidence intervals** → Deferred
3. **Stability-weighted dimension scaling** (year-to-year, predictive correlation) → v2.6+
4. **Extended backtesting** (team-changers, playoff portability, aging curves) → Next iteration
5. **Full probabilistic hierarchical regression** (Player > Archetype > Position with full posterior inference) → Next iteration
6. **PCV entropy from PCV computation** (currently uses emb_entropy_norm as proxy)
7. **Some plan metrics not available:** live-ball TO rate, on-ball TOs, bad pass frequency, on/off spacing effect, box creation, advantage creation events, disruption rate, cross-match success, shot quality allowed
