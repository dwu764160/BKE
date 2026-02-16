# BKE Metric Modelling Summary (v1.5)

## Overview — Portable Talent vs Role-Dependent Impact Decomposition Engine

BKE v1.5 is a multi-layer hierarchical decomposition engine that separates scalable talent from system amplification, standardized via three-level percentile architecture.

For every player-season, it produces:

```
Total Impact = Portable Talent + Role-Dependent Impact
```

Where Role-Dependent Impact is further decomposed:

```
Role-Dependent Impact = RUE + Archetype Elevation + Scheme Amplification
```

All outputs are percentile-standardized at three levels:
- **League** (macro comparison)
- **Position** (role fairness)
- **Archetype** (micro peer comparison)

**Guiding principles:**
- Strict separation: role assignment (archetype) is independent from value estimation (BKE)
- Portable talent captures context-neutral skill; role-dependent impact captures environment-specific value
- Percentile standardization is structural infrastructure, not cosmetic presentation
- Small-cohort Bayesian shrinkage toward league percentiles prevents archetype inflation

---

## Architecture — 7 Modules in `src/modeling/`

| Module | Purpose |
|--------|---------|
| `model_config.py` | Centralized config: paths, weights, thresholds, qualification filters |
| `percentile_engine.py` | Three-level percentile standardization (league, position, archetype) |
| `layer1_portable_talent.py` | Context-neutral talent estimation (RAPM + skill components) |
| `layer2_role_utilization.py` | Playtype surplus + Role Utilization Efficiency (RUE) |
| `layer3_archetype_elevation.py` | Player impact above/below archetype baseline |
| `layer4_scheme_amplification.py` | Context sensitivity estimation |
| `decomposition_engine.py` | Final assembly + output generation |

---

## Step-by-Step Process

### Layer 0 — Data Foundation

**Inputs consumed:**
- `player_rapm.parquet` — Multi-year Bayesian RAPM (pooled + single-season, O/D split)
- `modeling_inputs_all.parquet` — DARKO projections, linear stats (TS%, eFG%, USG%, AST%, TOV%, OREB%, DREB%, per-36 rates)
- `player_profiles_advanced.parquet` — Box score profiles, four factors, on/off, ORTG/DRTG
- `player_archetypes.parquet` — Offensive archetypes v4.3 + all 30 playtype columns (PPP, POSS_PCT, POSS per playtype)
- `defensive_archetypes_v2.parquet` — Defensive archetypes, switch score, versatility, rim protection, engagement
- `player_position_estimates.parquet` — Positional percentage estimates for bucketing

**Qualification filters:** MIN >= 500, GP >= 20, MPG >= 15.0, Possessions >= 500

---

### Layer 1 — True Portable Talent (Context-Neutral)

> "How good is this player independent of role volume and scheme?"

**1A. RAPM Selection:** Prefers pooled-split RAPM for O/D separation; falls back to single-season-split where missing.

**1B. Luck Adjustment:** Regresses extreme shooting toward league mean:
- 3PT%: 40% regression rate
- TS%: 30% regression rate (lighter touch — includes FT component)

**1C. Portable Skill Components:** Six dimensionally-independent skills that scale across contexts:

| Component | Sub-metrics | Weight |
|-----------|------------|--------|
| Shooting gravity | TS%, FG3A/36, FG3% | 0.15 |
| Passing efficiency | AST/36, playmaking score, TOV% | 0.12 |
| Rim protection | BLK/36, rim protection index pctl | 0.10 |
| Defensive versatility | Switch score, versatility pctl, assignment difficulty | 0.10 |
| Turnover control | TOV%, TOV/36 (inverted) | 0.08 |
| Rebounding | REB/36, OREB%, DREB% | 0.05 |

Each sub-metric is min-max normalized within season, then combined.

**1D. Portable Talent Score (PTS):**

PTS = 0.35 * RAPM_pctl + sum(w_k * SkillComponent_k) + 0.05 * Stability

Converted to league percentile rank (0-100).

**Output:** PTS range 0.3-100.0 across 950 qualified player-seasons (3 seasons).

---

### Layer 2 — Role Utilization Efficiency

> "How much of their talent is actually being expressed in their current role?"

**2A. Usage-Conditioned Efficiency:**

For each of 9 playtypes (Isolation, PnR Ball Handler, Post-Up, Cut, PnR Roll Man, Handoff, Off Screen, Spot-Up, Transition):

```
Surplus = Player_PPP - League_Avg_PPP_at_same_usage_bucket
```

Usage buckets: [0, 0.05, 0.10, 0.20, 0.35, 1.0]. Minimum 25 possessions per playtype. Possession-weighted total surplus computed.

**Playtype Contribution Vector (PCV):** Each playtype surplus converted to league percentile. PCV entropy (Shannon) captures usage diversity. Dominant playtype identified.

**2B. Role Utilization Efficiency (RUE):**

```
RUE = cosine_similarity(observed_playtype_vector, optimal_playtype_vector)
```

Where the optimal vector is the mean playtype distribution of the top-half (by RAPM) of the player's archetype cohort.

High RUE = well-utilized in current role. Low RUE = potentially miscast.

Converted to league + archetype percentile.

**Output:** RUE range 0.3-100.0.

---

### Layer 3 — Archetype Elevation

> "How much better is this player than the average player in their archetype?"

**3A. Archetype Baselines:** For each archetype x season: mean/median/std RAPM, mean playtype surplus, mean TS%.

**3B. Elevation Score:**

```
Elevation_RAPM = Player_RAPM - Archetype_mean_RAPM
```

Same for surplus and efficiency. Each z-scored within season, then combined:

```
Elevation_composite = 0.50 * z_RAPM + 0.30 * z_surplus + 0.20 * z_efficiency
```

Classified into tiers: Below Baseline < Baseline < Above Average < Elite < Archetype-Best.

**Output:** Elevation range 0.3-100.0. 12 archetypes with baselines per season.

---

### Layer 4 — Scheme Amplification

> "How much of this player's impact depends on their specific environment?"

**Full mode** (with possession data): Uses lineup-chunk variance and on/off splits to measure context sensitivity.

**Fast mode** (proxy features, no possession data): Approximates context sensitivity using:
- **RAPM vs DARKO agreement** — If two independent models agree, the player is more likely portable
- **Playtype entropy** — Diversified usage = harder to scheme against = more portable
- **Role confidence** — Higher archetype certainty = less context-dependent

**Scheme Stability Index:**

```
SSI = 0.40 * LineupInteraction + 0.30 * OnOffStability + 0.30 * TeammateIndependence
```

High stability = portable. Low stability = system-dependent.

**Output:** Stability range 0.3-100.0. Classification: ~30% portable, ~40% moderate, ~30% system-dependent.

---

### Final Decomposition

**Role-Dependent Impact Score (RDIS):**

```
RDIS_raw = 0.40 * RUE + 0.40 * Elevation + 0.20 * SchemeAmplification
```

Percentile-ranked within season (qualified players only).

**Total Impact:**

```
TotalImpact_raw = (w_PTS*PTS + w_RUE*RUE + w_Elev*Elevation + w_Scheme*(100-SchemeAmp)) / sum(weights)
```

All weights default to 1.0. Percentile-ranked within season (qualified only).

**Portability Ratio:**

```
Portability = PTS / (PTS + RDIS)
```

Centered at 0.50 for average players. Classified: >0.70 = Scalable Star, <0.40 = System Player.

**Impact Tiers:** Elite (90-100th), All-Star (75-90th), Starter (50-75th), Rotation (25-50th), Fringe (0-25th).

---

## Mathematical Definitions

- **Percentile rank:** P(x_i) = rank(x_i) / n * 100 (average method)
- **Small-cohort shrinkage:** P_adj = (1-a) * P_group + a * P_league, where a = 0.30 for groups < 10
- **Cosine similarity:** cos(a, b) = (a . b) / (|a| * |b|)
- **Shannon entropy:** H = -sum(p_i * ln(p_i))
- **Luck regression:** x_adj = mu + (1-r) * (x - mu) where r is regression rate

---

## Example Output Card

```
Nikola Jokic (2022-23):
  Portable Talent:     League 100th | Position 100th | Archetype 100th
  Role-Dependent:      League 63rd
  Elevation:           Archetype-Best
  Scheme Stability:    Portable
  Total Impact:        League 62nd
  Portability Ratio:   0.61 (Context-Moderate)
  RAPM: +5.91 | ORAPM: +3.72 | DRAPM: +2.19
```

---

## Outputs

| File | Format | Description |
|------|--------|-------------|
| `bke_v15_decomposition.parquet` | Parquet | Full decomposition table (232 columns, 1971 rows) |
| `bke_v15_decomposition.csv` | CSV | Same as above, human-readable |
| `bke_v15_report.json` | JSON | Summary report with top-10 lists, distribution stats |

---

## Validation Results (v1.5)

| Metric | 2022-23 | 2023-24 | 2024-25 |
|--------|---------|---------|---------|
| Qualified players | 315 | 309 | 326 |
| PTS range | 0.3-100.0 | 0.3-100.0 | 0.3-100.0 |
| RUE range | 0.3-100.0 | 0.3-100.0 | 0.3-100.0 |
| Elevation range | 0.3-100.0 | 0.3-100.0 | 0.3-100.0 |
| Scheme stability range | 0.3-100.0 | 0.3-100.0 | 0.3-100.0 |
| NaN in final output | 0 | 0 | 0 |

Portability ratio distribution: mean 0.50, std 0.20, min 0.004, max 0.99.

---

## Key Differences from Plan (v1.5 Plan -> Implementation)

### 1. No Formal Bayesian Hierarchical Regression
**Plan:** Specified `Impact_it = B1(PTS) + B2(RUE) + B3(Scheme) + e` with Bayesian hierarchical structure (Player > Archetype > Position) and shrinkage at each level.
**Implementation:** Uses additive percentile-weighted composition without formal regression fit. Shrinkage is applied at the percentile level (small-cohort Bayesian shrinkage toward league percentiles) rather than through a hierarchical regression model.
**Rationale:** The additive composition is more interpretable and avoids overfitting with limited cross-validation data. The percentile-level shrinkage achieves the same anti-inflation goal.

### 2. Scheme Estimation Uses Proxies Instead of Raw Lineup Data
**Plan:** Specified lineup possession clustering, on/off splits from PBP, defensive scheme analysis, and lineup interaction coefficients from raw possession data.
**Implementation:** Provides two modes — full (raw possession data) and fast (proxy features: RAPM vs DARKO agreement, playtype entropy, role confidence). Fast mode was validated; full mode is available but optional.
**Rationale:** Proxy features provide good differentiation (0.3-100.0 range) using already-computed metrics, avoiding slow I/O for raw possession files.

### 3. No Shot Quality Model
**Plan:** Lists "shot quality models" as a data input.
**Implementation:** Uses existing TS%/eFG% with luck regression adjustments instead of a dedicated shot quality model.
**Rationale:** Shot quality models require spatial shot data not currently in the pipeline. TS%/eFG% with regression achieves similar smoothing.

### 4. Tracking Data Used Indirectly
**Plan:** Lists tracking data as a direct input to Layer 1 skill components.
**Implementation:** Defensive archetypes (which are derived from tracking data: contest %, matchup difficulty, switch rates) are used, but raw tracking metrics are not directly consumed in skill component calculations.
**Rationale:** Tracking data is already distilled into the defensive archetypes pipeline; re-ingesting at the raw level would duplicate work.

### 5. Percentiles as Standardization, Not Regression Inputs
**Plan:** States percentiles are "used in regression inputs" and "as normalized predictors."
**Implementation:** Percentiles are used for output standardization, cross-cohort comparison, and shrinkage — but not as direct regression inputs (since there is no formal regression step).
**Rationale:** Without a formal regression step, percentiles serve the same normalization purpose at the composition stage.

### 6. Portability Ratio Formula Differs
**Plan:** Specifies `Portable Talent / Total Impact`.
**Implementation:** Uses `PTS / (PTS + RDIS)`, which produces a cleaner 0-1 distribution centered at 0.50 for average players. The plan's formula (percentile/percentile) produces degenerate values near 1.0 for most players.
**Rationale:** The additive formulation properly captures the fraction of impact attributable to portable talent, rather than comparing two correlated percentile scales.

### 7. No Backtesting Yet
**Plan:** Specifies validation on team-changers, role shifts, usage spikes, and scheme changes.
**Implementation:** Not yet implemented. Planned for v1.6.
**Rationale:** Requires cross-season player tracking and role-change identification infrastructure.

---

## Deliverables (v1.5)

- 4-layer hierarchical decomposition: Portable Talent -> Role Utilization -> Archetype Elevation -> Scheme Amplification
- Three-level percentile standardization (league, position, archetype) with small-cohort Bayesian shrinkage
- Portability Ratio as a clean scalar for trade valuation and scalability analysis
- Impact tiers (Elite / All-Star / Starter / Rotation / Fringe)
- Scheme classification (Portable / Context-Moderate / System-Dependent)
- All 950 qualified player-seasons across 3 seasons scored with zero NaN
- Full output: parquet + CSV + JSON report
- Modular, explainable, and basketball-first architecture
