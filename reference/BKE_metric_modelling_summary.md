# BKE Metric Modelling Summary (v2.0)

## Overview — Portable Talent vs Role-Dependent Impact Decomposition Engine

BKE v2.0 is a multi-layer hierarchical decomposition engine that separates scalable talent from system amplification. v2.0 fixes the mathematical distortions of v1.5 by switching to z-score aggregation, implementing a true variance-based portability index, and expanding to an 8-dimension portable model.

For every player-season, it produces:

```
Total Impact = Portable Talent + Role-Dependent Impact
```

Where Role-Dependent Impact is further decomposed:

```
Role-Dependent Impact = RUE + Archetype Elevation + Scheme Amplification
```

**v2.0 aggregation pipeline:** Raw → Z-score → Weighted Sum → Final Z → Percentile (presentation only)

Percentiles are strictly presentation-layer. All internal aggregation uses z-scores to preserve interval meaning.

**Guiding principles:**
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
| `model_config.py` | Centralized config: paths, weights, thresholds, z-score/shrinkage settings, 8-dimension definitions |
| `percentile_engine.py` | Three-level percentile standardization + z-score functions + Bayesian shrinkage |
| `layer1_portable_talent.py` | Context-neutral talent estimation (25% RAPM + 20% Playtype + 55% 8-Dimension Model) |
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
- `player_archetypes.parquet` — Offensive archetypes + playtype columns + tracking data (drives, rim attempts, paint frequency, FT rate, potential/secondary assists, turnovers, pcv entropy)
- `defensive_archetypes_v2.parquet` — Defensive archetypes, switch score, versatility, hustle score, engagement score, deflections, matchup diversity, d_results, assignment difficulty
- `player_position_estimates.parquet` — Positional percentage estimates for bucketing

**Qualification filters:** MIN >= 500, GP >= 20, MPG >= 15.0, Possessions >= 500

---

### Layer 1 — True Portable Talent (Context-Neutral)

> "How good is this player independent of role volume and scheme?"

**1A. RAPM Backbone (25%):** Prefers pooled-split RAPM for O/D separation; falls back to single-season-split where missing. Z-scored within season.

**1B. Playtype Efficiency (20%):** Aggregated playtype z-scores weighted by possession share.

**1C. Portable Dimension Model (55%) — 8 Independent Dimensions:**

| # | Dimension | Domain | Sub-metrics | Weight |
|---|-----------|--------|------------|--------|
| 1 | Shooting Gravity | Offense | TS% (adj), FG3% (adj), FG3A/36, C&S FG3% | 0.125 |
| 2 | Driving Gravity | Offense | DRIVES/36, AT_RIM_FREQ, FT_RATE, PAINT_FREQ | 0.125 |
| 3 | Playmaking | Offense | AST/36, Playmaking Score, Potential AST/36, Secondary AST/36 | 0.125 |
| 4 | Extra Possession | Cross | OREB%, DREB%, REB/36 (split 40% offense / 60% defense) | 0.125 |
| 5 | Defensive Playmaking | Defense | STL/100, BLK%, Deflections, hustle_score, engagement_score (Bayesian shrinkage) | 0.125 |
| 6 | Defensive Impact | Defense | DRAPM, DRTG (inverted), d_results_pctl (no rim protection — redundancy removed) | 0.125 |
| 7 | Turnover Control | Offense | TOV%, TOV/36 (both inverted — lower is better) | 0.125 |
| 8 | Defensive Versatility | Defense | switch_score, versatility_pctl, assignment_difficulty, matchup_diversity_pctl | 0.125 |

Each sub-metric is z-scored within season (winsorized at ±3.5σ). Missing components default to 0 (league average). Dimension composites are averaged z-scores of their sub-metrics.

**v2.0 changes from v1.5:**
- Expanded from 6 skill components to 8 dimensions
- Driving Gravity revised: removed FT%, focused on rim pressure (drives, rim FGA, fouls drawn, paint frequency)
- Turnover Control restored as standalone offensive dimension (was embedded in passing)
- Bayesian shrinkage applied to defensive playmaking metrics
- Assignment difficulty converted from categorical (Low/Medium/High) to numeric scale
- Extra Possession split cross-domain: 40% offensive, 60% defensive
- Rim protection removed from Defensive Impact to avoid redundancy with Defensive Playmaking

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

## Example Output Card

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
```

---

## Outputs

| File | Format | Description |
|------|--------|-------------|
| `bke_v20_decomposition.parquet` | Parquet | Full decomposition table (~280+ columns, 1971 rows) |
| `bke_v20_decomposition.csv` | CSV | Same as above, human-readable |
| `bke_v20_report.json` | JSON | Summary report with top-10 lists, z-score/distribution stats |

---

## Validation Results (v2.0)

| Metric | All Seasons |
|--------|-------------|
| Qualified players | 950 |
| Seasons | 2022-23, 2023-24, 2024-25 |
| PTS range | 0.3-100.0 |
| RUE range | 0.3-100.0 |
| Elevation range | 0.3-100.0 |
| Scheme stability range | 0.3-100.0 |
| NaN in final output (all key cols) | 0 |
| dim_shooting_gravity_z std | 0.607 |
| dim_driving_gravity_z std | 0.639 |
| dim_playmaking_z std | 0.848 |
| dim_extra_possession_z std | 0.892 |
| dim_defensive_playmaking_z std | 0.638 |
| dim_defensive_impact_z std | 0.825 |
| dim_turnover_control_z std | 0.853 |
| dim_defensive_versatility_z std | 0.769 |
| dimension_model_z std | 0.277 |
| portable_talent_z std | 0.478 |
| total_impact_z std | 0.455 |
| portability_index_raw mean | 0.605 |
| portability_index_raw std | 0.078 |
| Surplus abs mean | 0.325 |
| Runtime | ~8s |

Portability class distribution: ~30% Scalable Star, ~30% Context-Moderate, ~40% System Player.

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

## Key Differences from v2.0 Plan → Implementation

### 1. Portability Components Use Proxy Approximations
**Plan:** Specified lineup stability from "variance across teammate clusters," on/off splits from PBP, and usage shift simulation.
**Implementation:** Approximates using (a) multi-model agreement (RAPM vs DARKO variance), (b) dimensional consistency (variance across 8 dimensions), (c) surplus CoV + usage entropy, and (d) scheme stability + ORAPM/DRAPM balance. True lineup-level variance requires per-lineup possession data not readily available.

### 2. Stability-Weighted Dimension Scaling Deferred
**Plan:** Specified year-to-year stability weighting, predictive correlation with future RAPM, and cross-team transfer reliability.
**Implementation:** Equal weighting (0.125 each) across 8 dimensions. Stability-based and predictive weighting deferred to v2.1 (requires multi-year backtesting infrastructure).

### 3. Some Planned Metrics Not Available
**Plan:** Listed and-1 frequency, bad pass frequency, live-ball turnover rate, on/off spacing effect, box creation, advantage creation events, disruption rate, cross-match success, shot quality allowed.
**Implementation:** Uses available tracking data proxies: drives/36, AT_RIM_FREQ, FT_RATE for driving; POTENTIAL_AST/SECONDARY_AST for playmaking; hustle_score/engagement_score for defensive playmaking. Remaining metrics require tracking data feeds not currently in the pipeline.

### 4. No Formal Bayesian Hierarchical Regression
**Plan:** Implied structured hierarchical regression (Player > Archetype > Position).
**Implementation:** Uses additive z-score weighted composition with empirical Bayes shrinkage at the metric level. Full hierarchical regression deferred to v2.1.

### 5. No Backtesting Yet
**Plan:** Specifies multi-year stabilization, playoff portability testing, aging curve integration, and cross-team transfer case studies.
**Implementation:** Not yet implemented. Planned for v2.1+.

---

## Deliverables (v2.0)

- 4-layer hierarchical decomposition with z-score aggregation backbone
- 8-dimension portable talent model (Layer 1C) as the primary portability engine
- True Portability Index (4 structural components, variance-based)
- Three-level percentile standardization (presentation layer only)
- Bayesian shrinkage for noisy defensive/hustle metrics
- Fixed playtype surplus (corrected per-game threshold)
- Impact tiers (Elite / All-Star / Starter / Rotation / Fringe)
- Scheme classification (Portable / Context-Moderate / System-Dependent)
- All 950 qualified player-seasons across 3 seasons scored with zero NaN
- Full output: parquet + CSV + JSON report
- Modular, explainable, and basketball-first architecture
