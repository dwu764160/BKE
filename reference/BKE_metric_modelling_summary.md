# BKE Metric Modelling Summary (v2.7)

## Overview — Portable Talent vs Role-Dependent Impact Decomposition Engine

BKE v2.7 is a statistical maturity pass on top of the v2.6 decomposition engine. The architecture, layer boundaries, and decomposition objective remain unchanged; v2.7 upgrades role-conditioning math, archetype handling, and stability controls.

For every player-season, it produces:

```
Total Impact = Portable Talent + Role-Dependent Impact
```

Where Role-Dependent Impact is further decomposed:

```
Role-Dependent Impact = RUE + Archetype Elevation + Scheme Amplification
```

**v2.7 aggregation pipeline:** Raw → Z-score → Soft-Archetype Conditional Standardization → Empirical-Bayes Shrinkage → Weighted Sum → Final Z → Percentile (presentation only)

Percentiles remain strictly terminal — never used as inputs to any layer or aggregation step.

**v2.7 key additions over v2.6:**
- **Soft archetype membership (implemented):** Hard single-role internals replaced with embedding-driven `arch_prob_*` memberships.
- **Full conditional neutralization (implemented):** Layer 1 dimension conditioning now supports mean + variance adjustment.
- **Empirical-Bayes variance-component path (implemented):** Shared shrinkage utility now supports within/between variance hooks.
- **Defensive conditioning symmetry (implemented):** Defensive dimensions participate in the same conditional framework as offensive dimensions.
- **RUE source decoupling (implemented):** Archetype-optimal templates can be sourced from portable talent / dimension model / RAPM.
- **Portability transfer tightening (implemented):** Added explicit soft-membership entropy transfer signal.
- **Dimension variance restoration (implemented):** `dimension_model_z` can be scaled back to a target standard deviation.

**v2.6 key additions over v2.5 (retained):**
- Bayesian hierarchical shrinkage calibration and reliability weighting
- Backtesting framework integration
- Blended RUE scoring structure
- Scheme bonus-only behavior in Total Impact

**v2.5 key additions over v2.0 (retained):**
- Archetype-conditional neutralization foundation
- Three-level z-score outputs
- `portability_ratio` removed from public outputs
- MF-1 / MF-4 / MF-5 metric expansions
- `elevation_z` preservation bug fix

**Guiding principles (unchanged):**
- Archetype = role assignment (behavior); impact = value estimation
- Portable talent remains context-neutral; role-dependent impact remains context-specific
- Z-score aggregation remains the internal math backbone
- Portability remains structural (not compositional ratio)

---

## Architecture — 7 Modules in `src/modeling/`

| Module | Purpose |
|--------|---------|
| `model_config.py` | Centralized config: paths, weights, thresholds, z-score/shrinkage settings, neutralization, soft-membership, output versioning |
| `percentile_engine.py` | Three-level percentile + z-score utilities, empirical-Bayes shrinkage, hard/soft neutralization paths |
| `layer1_portable_talent.py` | Context-neutral talent estimation with embedding ingestion, soft memberships, conditional neutralization, variance restoration |
| `layer2_role_utilization.py` | Playtype surplus + RUE with configurable optimal-template source and soft-membership weighting |
| `layer3_archetype_elevation.py` | Baseline-relative elevation with soft-membership weighted archetype expectations |
| `layer4_scheme_amplification.py` | Context sensitivity estimation + scheme stability z-scores |
| `decomposition_engine.py` | Final z-score assembly + structural portability index + transfer signal integration + output generation |

---

## Step-by-Step Process

### Layer 0 — Data Foundation

**Inputs consumed:**
- `player_rapm.parquet` — Multi-year Bayesian RAPM (pooled + single-season, O/D split)
- `modeling_inputs_all.parquet` — DARKO projections, linear stats (TS%, eFG%, USG%, AST%, TOV%, OREB%, DREB%, per-36 rates)
- `player_profiles_advanced.parquet` — Box score profiles, four factors, on/off, ORTG/DRTG
- `player_archetypes.parquet` — Offensive archetypes + playtype columns + tracking data
- `defensive_archetypes_v2.parquet` — Defensive archetypes and matchup/context metrics
- `player_position_estimates.parquet` — Positional percentage estimates for bucketing
- `archetype_embeddings.parquet` — **NEW in v2.7:** embedding vectors used for soft archetype memberships
- `data/tracking/{season}/hustle_stats.parquet` — CHARGES_DRAWN, DEF_LOOSE_BALLS_RECOVERED, etc.

**Qualification filters:** MIN >= 500, GP >= 20, MPG >= 15.0, Possessions >= 500

---

### Layer 1 — True Portable Talent (Context-Neutral)

> "How good is this player independent of role volume and scheme?"

**1A. RAPM Backbone (25%):** Prefers pooled-split RAPM for O/D separation; falls back to single-season-split where missing. Z-scored within season.

**1B. Playtype Efficiency (20%):** Aggregated playtype z-scores weighted by possession share.

**1C. Portable Dimension Model (55%) — 9 Dimensions (v2.6+):**
- Core dimensions and basketball weighting structure from v2.6 are retained.
- v2.7 adds centralized global dimension-weight tuning hooks (`DimensionWeightTuningConfig`) before final Layer 1C composite.

**v2.7 Soft Archetype Memberships (new internal conditioning substrate):**
- Embedding columns (`emb_*`) are mapped to archetype centroids.
- Distances are converted with temperature-scaled softmax:

```
p_i(j) = exp(-d_i(j)/tau) / sum_k exp(-d_i(k)/tau)
```

- Probability floor is applied and renormalized.
- Exposes `arch_prob_*`, `soft_archetype_entropy`, `soft_archetype_entropy_norm`.

**v2.7 Conditional Neutralization Upgrade:**

Legacy mean-centering path remains available, but default v2.7 behavior supports full conditional standardization:

```
z'_i = (z_i - mu_i) / sigma_i
mu_i = sum_j p_i(j) * mu_j
sigma_i = sum_j p_i(j) * sigma_j
```

- This replaces hard one-label conditioning when `arch_prob_*` columns are present.
- Defensive dimensions are no longer exempt from conditioning in config.

**v2.7 Dimension Variance Restoration:**

After Layer 1C aggregation:

```
scale = target_std / observed_std
dimension_model_z <- dimension_model_z * scale
```

Default target std: `0.42`.

**1D. Portable Talent Score (PTS):**

```
PTS_z = 0.25 * RAPM_z + 0.20 * Playtype_z + 0.55 * Dimension_z
```

Z-score composite converted to percentile rank for presentation. CDF-based percentile also computed for cross-era comparison.

---

### Layer 2 — Role Utilization Efficiency

> "How much of their talent is actually being expressed in their current role?"

**2A. Usage-Conditioned Efficiency:**

For each of 9 playtypes:

```
Surplus = Player_PPP - League_Avg_PPP_at_same_usage_bucket
```

Usage bucket logic, minimum-playtype thresholds, and surplus z-score outputs are unchanged from v2.6.

**2B. Role Utilization Efficiency (RUE):**

```
RUE = cosine_similarity(observed_playtype_vector, optimal_playtype_vector)
```

**v2.7 updates in this section:**
- Archetype-optimal template source is configurable via `rue_optimal_source`:
  - `portable_talent` (default)
  - `dimension_model`
  - `rapm`
- When soft memberships exist, per-player optimal vectors are probability-weighted blends across archetype templates.

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

**v2.7 updates in this section:**
- Baselines can be computed with soft-membership weighted moments.
- Player-level expected baseline can be computed as a weighted archetype expectation rather than hard label lookup.

---

### Layer 4 — Scheme Amplification

> "How much of this player's impact depends on their specific environment?"

Full-mode / fast-mode structure is unchanged.

**v2.7 update in decomposition coupling:**
- Layer 4 behavior remains bonus-only in Total Impact, and now receives an additional transfer-related signal via soft-membership entropy in final portability assembly.

---

### Final Decomposition

**Role-Dependent Impact z-score:**

```
RDIS_z = 0.40 * RUE_z + 0.40 * Elevation_z + 0.20 * (-Scheme_z)
```

**Total Impact z-score:**

```
Total_z = (w_PTS * PTS_z + w_RUE * RUE_z + w_Elev * Elev_z + w_Scheme * Scheme_z) / sum(w)
```

All weights default to 1.0. Percentile-ranked within season (qualified only) for presentation.

**Structural Portability Index (retained core, updated transfer term):**

```
Portability Index = Weighted sum of structural components (0-1)
```

| Component | Weight | What It Measures |
|-----------|--------|-----------------|
| Dimensional Breadth | 35% | Breadth and diffusion of portable skill profile |
| Universal Skill Presence | 25% | Shooting/playmaking/creation portability core |
| Two-Way Balance | 20% | Offensive + defensive value balance |
| Scheme Independence (+ transfer blend) | 20% | Scheme stability signals with soft-membership transfer integration |

**v2.7 portability update:**
- Adds `portability_archetype_transfer` from soft-membership entropy (+ optional confidence term) and blends into scheme-independence behavior.

---

## Mathematical Definitions

- **Z-score:** $z = (x - \mu) / \sigma$, winsorized at configured bounds
- **Soft-membership neutralization (full):**

```
z'_i = (z_i - mu_i) / sigma_i
mu_i = sum_j p_i(j) * mu_j
sigma_i = sum_j p_i(j) * sigma_j
```

- **Empirical-Bayes shrinkage (variance-component form):**

```
a = sigma_within^2 / (sigma_within^2 + sigma_between^2 / n)
x_post = (1 - a) * x + a * mu_prior
```

- **Cosine similarity:** $\cos(a,b) = (a \cdot b) / (\|a\|\|b\|)$
- **Shannon entropy:** $H = -\sum p_i \ln p_i$
- **Weighted z composite:** $\sum (w_i z_i) / \sum w_i$

---

## Final Comprehensive Output — BKE (v2.7 Addendum)

This addendum defines the final OBKE/DBKE/BKE constructor used for league-wide final ranking output in `BKE_Scores_v27.json`.

### Inputs

- Source file: `data/processed/bke_v27_decomposition.parquet`
- Population: **eligible players only** (`qualified == True`)
- Signals are raw/z layer composites only (no earlier percentile reuse)

### Dimension Scores Bundle

Per eligible player, the constructor carries forward raw + z for the 9 portable dimensions:
- Shooting Gravity
- Driving Gravity
- Playmaking Creation
- Extra Possession
- Defensive Playmaking
- Defensive Impact
- Turnover Control
- Defensive Versatility
- Self Creation

### Layer Composites (Pre-transform)

The script builds layer-level offensive/defensive composites before final transforms:
- `layer1_offensive_raw = offensive_portable_z`
- `layer1_defensive_raw = defensive_portable_z`
- `layer2_rue_raw = role_utilization_raw_z` (fallback to z-score of `role_utilization_raw`)
- `layer3_off_elevation_raw = zscore(elevation_orapm)`
- `layer3_def_elevation_raw = zscore(elevation_drapm)`
- `layer4_scheme_bonus_raw = max(scheme_stability_z, 0)` (bonus-only)

### OBKE/DBKE Construction

`OBKE_raw` (weighted offensive composite):

```
OBKE_raw = 0.55*layer1_offensive_raw
         + 0.25*layer2_rue_raw
         + 0.20*layer3_off_elevation_raw
```

`DBKE_raw` (weighted defensive composite):

```
DBKE_raw = 0.60*layer1_defensive_raw
         + 0.25*layer3_def_elevation_raw
         + 0.15*layer4_scheme_bonus_raw
```

`BKE_raw`:

```
BKE_raw = OBKE_raw + DBKE_raw
```

### Monotonic Transform + Final Percentiles

Monotonic transform is applied to OBKE/DBKE/BKE:

```
transformed_x = sign(x) * log(1 + |x|)
```

Final percentiles are computed league-wide across **eligible players only**:
- `final_OBKE_percentile`
- `final_DBKE_percentile`
- `final_BKE_percentile`

Final ranking uses only terminal BKE percentile:

```
rank = descending_rank(final_BKE_percentile)
```

This is the only percentile used for ranking.

### Output Artifact

- Script: `src/modeling/construct_bke_scores_v27.py`
- Output: `data/processed/BKE_Scores_v27.json`

Per player entry includes:
- `raw_OBKE`, `raw_DBKE`
- `transformed_OBKE`, `transformed_DBKE`
- `final_OBKE_percentile`, `final_DBKE_percentile`
- `raw_BKE`, `transformed_BKE`, `final_BKE_percentile`
- `rank`

---

## Example Output Card (updated for v2.7 fields)

```
Player (Season):
  Portable Talent:     Score XX.X | z=...
  Role-Dependent:      Score XX.X | z=...
  Total Impact:        Score XX.X | z=...
  Portability Index:   0.XX
    Dimensional Breadth: ...
    Universal Skill: ...
    Two-Way Balance: ...
    Scheme Independence: ...
    Archetype Transfer: ... (v2.7)
  Soft Membership Entropy (norm): ... (v2.7)
```

---

## Outputs

| File | Format | Description |
|------|--------|-------------|
| `bke_v27_decomposition.parquet` | Parquet | Full decomposition table |
| `bke_v27_decomposition.csv` | CSV | Same as above, human-readable |
| `bke_v27_report.json` | JSON | Summary report with top-10 lists, z-score/distribution stats |
| `bke_v27_backtest.json` | JSON | Year-over-year backtesting diagnostics |

---

## Validation Results (v2.7 run)

| Metric | Value |
|--------|-------|
| Qualified players | 950 |
| Seasons | 2022-23, 2023-24, 2024-25 |
| Output artifacts | `bke_v27_decomposition.parquet/.csv`, `bke_v27_report.json`, `bke_v27_backtest.json` |
| Decomposition runtime | ~63s (full v2.7 run) |
| Backtest execution | Success (after indentation fix in save block) |

Selected backtest diagnostics from latest run:
- PTS Spearman $\rho \approx 0.590$
- Portability Index Spearman $\rho \approx 0.685$
- Total Impact within-1-tier accuracy $\approx 76.8\%$

---

## Changes from v2.6 → v2.7

### 1) Soft Archetype Memberships (Fix for boundary instability)
**v2.6:** Hard archetype label used in internal conditioning paths.
**v2.7:** Soft membership vectors (`arch_prob_*`) from embeddings drive conditioning and weighted expectations.

### 2) Neutralization Upgrade (location-only → location+scale)
**v2.6:** Primarily mean-centering by archetype expectation.
**v2.7:** Optional full conditional standardization with expected mean and expected variance.

### 3) Empirical-Bayes Variance Components
**v2.6:** Simpler shrinkage control with reliability scaling.
**v2.7:** Shared shrinkage function supports variance-component EB path (`use_empirical_bayes`, `estimate_variance_components`).

### 4) Defensive Symmetry in Conditioning
**v2.6:** Defensive neutralization exemptions remained in config.
**v2.7:** Defensive dimensions are included in the unified conditioning target list.

### 5) RUE Endogeneity Reduction
**v2.6:** Archetype-optimal distributions effectively RAPM-tethered in practice.
**v2.7:** Source is configurable and defaults to portable-talent-aligned templates.

### 6) Elevation Baseline Softening
**v2.6:** Baselines keyed to hard archetype label.
**v2.7:** Baselines and expected values can be probability-weighted by soft memberships.

### 7) Portability Transfer Signal
**v2.6:** Transferability proxy was indirect.
**v2.7:** Explicit transfer signal from soft-membership entropy added and surfaced as `portability_archetype_transfer`.

### 8) Output Versioning
**v2.6:** `bke_v26_*`
**v2.7:** `bke_v27_*`

---

## Changes from v2.0 → v2.5 (historical baseline retained)

### Fix #1: Three-Level Z-Scores
**v2.0:** League-level z-scores only for dimension columns.
**v2.5:** For each dimension z-score, three levels are computed:
- League z
- Positional z
- Archetype z

### Fix #2: Percentiles Strictly Terminal
**v2.0:** Some paths risked percentile reuse.
**v2.5:** Percentiles enforced as presentation-only.

### Fix #3: Archetype-Conditional Neutralization (foundational)
**v2.5:** Established role-expectation subtraction baseline that v2.7 extends to soft-membership full conditioning.

### Fix #4: portability_ratio Removed from Public Outputs
Retained as internal compatibility alias only.

### MF-1 / MF-4 / MF-5
Turnover-control expansion, defensive playmaking expansion, and playmaking creation/pressure split were added in v2.5 and retained in v2.7.

---

## Changes from v1.5 → v2.0 (historical baseline retained)

- Z-score-first aggregation
- Structural portability index foundation
- Portable dimension model expansion and weighting changes
- Playtype surplus threshold correction
- Bayesian shrinkage introduction
- Data-loader expansion for tracking/context signals

---

## Key Differences from v2.7 Plan → Implementation

### 1. Soft Membership Uses Embedding-Centroid Distance
**Plan:** Probabilistic archetype assignment with robust smoothing.
**Implementation:** Distance-to-centroid softmax with temperature + probability floor.

### 2. Neutralization Uses Shared Utility Path
**Plan:** Conditional standardization throughout role-conditioned dimensions.
**Implementation:** Added soft-membership conditional neutralization utility with fallback to hard-label path.

### 3. RUE Optimal Source Is Configurable
**Plan:** Reduce RAPM endogeneity.
**Implementation:** Added `rue_optimal_source` (`portable_talent` default), with runtime source-candidate fallback chain.

### 4. Elevation Uses Weighted Baselines When Probabilities Exist
**Plan:** Replace brittle hard-label comparisons.
**Implementation:** Weighted baseline expectation from `arch_prob_*`, with hard-label fallback retained.

### 5. Portability Transfer Is Explicitly Surfaced
**Plan:** Better transferability handling.
**Implementation:** Added `portability_archetype_transfer` and blended it into scheme-independence component.

---

## Deliverables (v2.7)

- 4-layer decomposition with z-score backbone retained
- Soft-membership archetype conditioning integrated across layers
- Full conditional neutralization option (mean + variance)
- Empirical-Bayes variance-component shrinkage path in shared utilities
- Configurable RUE optimal-template source
- Soft-membership weighted elevation baselines and expectations
- Structural portability index with explicit archetype transfer signal
- Output/version migration to `bke_v27_*` artifacts
- Updated report, backtesting, readme references, and loop context notes

## Known Issues / Missing Features (Deferred)

1. Full probabilistic hierarchical regression (posterior inference) remains out of scope
2. Stability-weighted dimension scaling is still a future calibration track
3. Extended backtesting slices (team-changers, playoff transfer, aging curves) are pending
4. Some planned metrics remain unavailable in source data (live-ball TO subtype details, richer spacing/on-off decomposition, disruption micro-events)
