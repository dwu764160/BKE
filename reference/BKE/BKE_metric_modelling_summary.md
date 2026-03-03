# BKE Metric Modelling Summary (v2.9)

## Overview — Portable Talent vs Role-Dependent Impact Decomposition Engine

BKE v2.8 is a distribution-integrity pass on top of the v2.7 decomposition engine. The architecture, layer boundaries, and decomposition objective remain unchanged; v2.8 upgrades terminal score shaping, variance anchoring, and diagnostics while preserving the v2.7 role-conditioning framework.

For every player-season, it produces:

```
Total Impact = Portable Talent + Role-Dependent Impact
```

Where Role-Dependent Impact is further decomposed:

```
Role-Dependent Impact = RUE + Archetype Elevation + Scheme Amplification
```

**v2.8 aggregation pipeline:** Raw → Z-score → Soft-Archetype Conditional Standardization → Empirical-Bayes Shrinkage → Weighted Sum → Anchored Z → Logistic Transform → Percentile (presentation only)

Percentiles remain strictly terminal — never used as inputs to any layer or aggregation step.

**v2.8 key additions over v2.7:**
- **Terminal transformed ranking (implemented):** Portable talent, role-dependent impact, and total impact use a monotonic logistic transform before percentile ranking.
- **Seasonwise variance anchoring (implemented):** Role-dependent and total impact z-variance are anchored to floor retention constraints to avoid over-compression.
- **Distribution integrity diagnostics (implemented):** Added variance/compression/tail-separation diagnostics and OBKE/DBKE balance checks.
- **Expanded output artifacts (implemented):** Decomposition now exports dedicated dimension/layer/OBKE-DBKE JSON bundles and distribution reports.

**v2.7 key additions over v2.6 (retained):**
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
| `model_config.py` | Centralized config: paths, weights, thresholds, z-score/shrinkage settings, neutralization, soft-membership, distribution-integrity controls, output versioning |
| `percentile_engine.py` | Three-level percentile + z-score utilities, empirical-Bayes shrinkage, hard/soft neutralization paths |
| `layer1_portable_talent.py` | Context-neutral talent estimation with embedding ingestion, soft memberships, conditional neutralization, variance restoration |
| `layer2_role_utilization.py` | Playtype surplus + RUE with configurable optimal-template source and soft-membership weighting |
| `layer3_archetype_elevation.py` | Baseline-relative elevation with soft-membership weighted archetype expectations |
| `layer4_scheme_amplification.py` | Context sensitivity estimation + scheme stability z-scores |
| `decomposition_engine.py` | Final assembly with anchored/transformed terminal scoring + structural portability index + transfer signal integration + output generation |

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

**Total Impact z-score (pre-anchor):**

```
Total_z = (w_PTS * PTS_z + w_RUE * RUE_z + w_Elev * Elev_z + w_Scheme * Scheme_z) / sum(w)
```

All weights default to 1.0.

**v2.8 variance anchoring and transform path:**
- `role_dependent_impact_z` and `total_impact_z` are seasonwise re-scaled if their post-processing variance falls below configured retention floors.
- Terminal ranking uses a monotonic logistic mapping:

```
transformed_z = 2 * sigmoid(alpha * z) - 1
```

- Percentiles are then computed from transformed values (qualified-only, within season) for:
  - `portable_talent_percentile`
  - `role_dependent_impact_percentile`
  - `total_impact_percentile`

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
- **Logistic transform (v2.8 terminal shaping):** $z_t = 2\sigma(\alpha z)-1$

---

## Final Comprehensive Output — BKE (v2.7 Addendum)

This addendum defines the final OBKE/DBKE/BKE constructor used for league-wide final ranking output in `BKE_Scores_v27.json`.

Note: v2.8 decomposition changes are upstream and diagnostic-focused; the final constructor artifact remains `construct_bke_scores_v27.py` / `BKE_Scores_v27.json` in the current pipeline.

### Inputs

- Source file: `data/processed/bke/bke_v27_decomposition.parquet`
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
- Output: `data/processed/bke/BKE_Scores_v27.json`

Per player entry includes:
- `raw_OBKE`, `raw_DBKE`
- `transformed_OBKE`, `transformed_DBKE`
- `final_OBKE_percentile`, `final_DBKE_percentile`
- `raw_BKE`, `transformed_BKE`, `final_BKE_percentile`
- `rank`

---

## Example Output Card (updated for v2.8 decomposition fields)

```
Player (Season):
  Portable Talent:     Score XX.X | z=...
  Role-Dependent:      Score XX.X | z=...
  Total Impact:        Score XX.X | z=...
  Total Impact (xform): ...
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
| `bke_v28_decomposition.parquet` | Parquet | Full decomposition table (anchored + transformed terminal columns) |
| `bke_v28_decomposition.csv` | CSV | Same as above, human-readable |
| `bke_v28_report.json` | JSON | Summary report with top-10 lists and transformed metric summaries |
| `bke_v28_variance_report.json` | JSON | Variance retention diagnostics for anchored layers |
| `bke_v28_compression_report.json` | JSON | Compression/tail diagnostics and dimension correlation matrix |
| `dimension_scores_v28.json` | JSON | Per-player portable-dimension score bundle |
| `layer_scores_v28.json` | JSON | Per-player layer score bundle |
| `obke_dbke_scores_v28.json` | JSON | Per-player OBKE/DBKE/BKE decomposition bundle |
| `BKE_Scores_v27.json` | JSON | Final constructor output (terminal league-wide ranking artifact) |
| `bke_v29_diagnostic_master.json` | JSON | v2.9 diagnostic suite output — 8-domain audit results + player-level metrics |

**Directory structure:**
- Decomposition data (`parquet`, `csv`, score bundles): `data/processed/bke/`
- Reports and diagnostics (`report.json`, `variance_report.json`, etc.): `reports/`

---

## Validation Results (v2.8 run)

| Metric | Value |
|--------|-------|
| Qualified players | 950 |
| Seasons | 2022-23, 2023-24, 2024-25 |
| Output artifacts | `bke_v28_decomposition.parquet/.csv`, `bke_v28_report.json`, `bke_v28_variance_report.json`, `bke_v28_compression_report.json`, `dimension_scores_v28.json`, `layer_scores_v28.json`, `obke_dbke_scores_v28.json` |
| Decomposition runtime | ~63s (full v2.8 run) |
| Backtest execution | Not rerun in this v2.8 pass (constructor remains v27 path) |

Selected decomposition diagnostics from latest run:
- Qualified players: 950 / 1971 total rows
- Runtime: ~62.8s
- Top impact list and portability classifications produced successfully

---

## V2.9 — Diagnostic Suite (Tests Only)

v2.9 is a measurement-only iteration. No weights, structure, or logic changes. A central diagnostic script runs 8 audit domains against the v2.8 decomposition output and writes results to a unified JSON report.

### Diagnostic Script

- Script: `tests/bke_v29_diagnostics.py`
- Input: `data/processed/bke/bke_v28_decomposition.parquet`
- Output: `reports/bke_v29_diagnostic_master.json`

### 8 Diagnostic Domains

| Domain | Tests | Purpose |
|--------|-------|---------|
| 1. OBKE/DBKE Variance Asymmetry Audit | 1.1–1.3 | Quantify whether defense dominates ranking movement |
| 2. Defensive Signal Quality Audit | 2.1–2.3 | Validate DBKE correlates with true defensive impact |
| 3. Counting Stats vs On/Off Dominance | 3.1–3.3 | Test whether counting stats dominate on/off signal |
| 4. Variance Anchor Stress Test | 4.1–4.2 | Verify seasonwise variance anchoring and year-to-year stability |
| 5. Offense Weight Bias Experiment | 5.1 | Simulate 55/45 and 60/40 offense-biased composites |
| 6. Archetype Coefficient Audit | 6.1–6.3 | Detect hidden archetype-weight biases |
| 7. Portability vs Role-Dependent Drag | 7.1 | Measure whether offensive stars are over-dragged by role component |
| 8. Rank Movement Driver Decomposition | 8 | Decompose BKE rank into per-component contributions |

### Diagnostic Results Summary (v2.9 run)

| Metric | Value |
|--------|-------|
| Qualified players | 950 |
| Seasons | 2022-23, 2023-24, 2024-25 |
| Runtime | ~0.15s |
| Output | `reports/bke_v29_diagnostic_master.json` (525KB) |

**Domain 1 — Variance Asymmetry:**
- Offensive variance share: 48.1%, Defensive: 51.7%, Covariance: 0.2%
- Verdict: **SYMMETRIC** — no defense-dominant asymmetry detected
- Tail sensitivity: defensive bottom-5% penalty exists but proportionate to offensive top-5%

**Domain 2 — Defensive Signal Quality:**
- corr(DBKE, DRAPM) = 0.871 — strong alignment
- corr(def_impact, DRAPM) = 0.824 >> corr(def_playmaking, DRAPM) = 0.351 — proper separation
- Partial correlation (controlling for STL/BLK) = 0.859 — counting stats are NOT diluting
- Counting noise risk: LOW

**Domain 3 — Counting Stats vs On/Off:**
- OBKE correlates well with orapm and rate metrics
- Standardized regression coefficients (DBKE): beta_DRAPM dominates over beta_STL
- No steals overreliance detected

**Domain 4 — Variance Anchor:**
- Per-season std ratios are stable, no anchor leakage
- Year-to-year OBKE stability: ~0.64; DBKE stability: 0.52–0.63
- Noise asymmetry: moderate in 2022-23→2023-24, symmetric in 2023-24→2024-25

**Domain 5 — Offense Weight Bias:**
- 55/45 and 60/40 simulations produce small mean rank shifts
- Top-10 overlap remains high across simulations
- Offensive specialists gain moderately under biased composites; two-way players are stable

**Domain 6 — Archetype Coefficient Audit:**
- No per-archetype multipliers or discrete boosts found in pipeline
- All conditioning is probabilistic via soft memberships
- DBKE variance range across archetypes is proportionate to OBKE range

**Domain 7 — Portability vs Role Drag:**
- Mean rank delta (portable vs total) is near zero — no systemic over-drag
- Top offensive players' drag is examined per-player in the report

**Domain 8 — Rank Movement Decomposition:**
- Offensive driver share: 48.2%, Defensive: 51.8%
- Primary rank driver: **BALANCED**
- Role compression ratio is within expected bounds

### Changes from v2.8 → v2.9

### 1) Central Diagnostic Suite
**v2.8:** Distribution integrity diagnostics embedded in decomposition engine.
**v2.9:** Standalone 8-domain diagnostic suite (`tests/bke_v29_diagnostics.py`) produces unified JSON report.

### 2) No Structural Changes
No weights, layer boundaries, transforms, or archetype logic were modified. v2.9 is measurement-only.

### 3) Output Artifact
New: `reports/bke_v29_diagnostic_master.json` — includes domain-level audit results and per-player diagnostic metrics (OBKE_z, DBKE_z, BKE_equal_var, rank simulations).

---

## V3.1 Experiment 2 Rerun (2026-03-01) — Expanded Production Proxy + 15 Lambda Sweep

### Scope
Experiment 2 was rerun with an expanded production proxy to explicitly include raw offensive box-score signal in addition to efficiency/impact proxies.

### Script and Output
- Script: `src/modeling/experiment2_production_tilt.py`
- Output: `reports/bke_v31_experiment2_production_tilt.json`
- Input: `data/processed/bke/bke_v28_decomposition.parquet` (qualified players only)

### Baseline Profile Used
- Base profile: v3.1 Layer 3 + Layer 6 (`60/40` blend)
- Config loaded from second-pass report:
  - `layer3_exponent = 1.08`
  - `layer6_k = 0.20`

### Expanded Production Proxy
Season-normalized weighted proxy over available columns:
- `orapm` (0.22)
- `TS_PCT` (0.14)
- `PTS` (0.18)
- `AST` (0.12)
- `FGM` (0.08)
- `FGA` (0.08)
- `FG3M` (0.06)
- `FG3A` (0.04)
- `FTM` (0.04)
- `FTA` (0.04)

### Lambda Sweep
- Evaluated 15 lambdas: `0.01` through `0.15`.
- Recommendation rule: maximize predictive rho with `mean_abs_rank_shift_vs_base <= 3.5`.

### Key Results
- Base predictive rho: `0.31523`
- Recommended lambda: `0.03`
  - predictive rho: `0.317888` (`+0.002658` vs base)
  - mean abs rank shift: `3.250526`
  - low-prod top100 count: `16 -> 11` (`-5`)
  - high-prod top100 count: `165 -> 174` (`+9`)

### Interpretation
The rerun indicates both effects are present, but the dominant behavior is **rewarding higher-production profiles**, with a secondary reduction in low-production top-100 exposure. Larger lambdas continue improving predictive rho but introduce progressively larger rank movement.

---

## Changes from v2.7 → v2.8

### 1) Terminal Transform Before Percentile Rank
**v2.7:** Percentiles were computed directly from terminal z values.
**v2.8:** Percentiles are computed from monotonic logistic-transformed terminal values.

### 2) Seasonwise Variance Anchoring
**v2.7:** No explicit variance floor enforcement after layer assembly.
**v2.8:** `role_dependent_impact_z` and `total_impact_z` are anchored per season to retain configured variance floor.

### 3) Distribution Integrity Diagnostics
**v2.7:** Report focused on summary/top-player outputs.
**v2.8:** Adds dedicated variance/compression reports, tail separation, OBKE/DBKE balance, and dimension correlation diagnostics.

### 4) Expanded Artifacts for Downstream Consumption
**v2.7:** Primary decomposition table/report plus constructor/backtest artifacts.
**v2.8:** Adds structured JSON exports for dimensions, layers, and OBKE/DBKE decomposition.

### 5) Output Versioning
**v2.7:** `bke_v27_*`
**v2.8:** `bke_v28_*` (constructor output remains `BKE_Scores_v27.json`)

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

## Deliverables (v2.8)

- 4-layer decomposition with z-score backbone retained
- Soft-membership archetype conditioning integrated across layers
- Full conditional neutralization option (mean + variance)
- Empirical-Bayes variance-component shrinkage path in shared utilities
- Configurable RUE optimal-template source
- Soft-membership weighted elevation baselines and expectations
- Structural portability index with explicit archetype transfer signal
- Seasonwise variance anchoring for role-dependent and total impact layers
- Logistic terminal transforms before percentile ranking
- Distribution-integrity diagnostics and expanded decomposition artifacts
- Output/version migration to `bke_v28_*` decomposition artifacts
- Updated report/readme/loop context notes

## Deliverables (v2.9)

- Central diagnostic suite: `tests/bke_v29_diagnostics.py`
- 8-domain audit covering variance asymmetry, defensive signal quality, counting stats dominance, variance anchoring, offense bias, archetype coefficients, portability drag, and rank movement decomposition
- Unified JSON report: `reports/bke_v29_diagnostic_master.json`
- Per-player diagnostic metrics (OBKE_z, DBKE_z, BKE_equal_var, rank simulations)
- No structural changes to decomposition engine or scoring pipeline

## Known Issues / Missing Features (Deferred)

1. Full probabilistic hierarchical regression (posterior inference) remains out of scope
2. Stability-weighted dimension scaling is still a future calibration track
3. Extended backtesting slices (team-changers, playoff transfer, aging curves) are pending
4. Some planned metrics remain unavailable in source data (live-ball TO subtype details, richer spacing/on-off decomposition, disruption micro-events)

---

## V3.1 — Experimental Layer Suite (Phase 2, Independent Tests)

v3.1 is implemented as an experimental runner (non-destructive to production artifacts):

- Script: `src/modeling/bke_v31_experimental_layers.py`
- Input: `data/processed/bke/bke_v28_decomposition.parquet`
- Output: `reports/bke_v31_experimental_layers.json`
- Population: 950 qualified player-seasons (2022-23, 2023-24, 2024-25)

Each layer is tested independently against a fixed baseline, then a constrained combined model is evaluated with max 3 structural changes.

### Baseline Snapshot (v3.1 experiment baseline)

Baseline uses OBKE/DBKE blend `0.60 / 0.40` and current decomposition internals.

| Metric | Baseline |
|--------|----------|
| Global YoY rank corr | 0.5816 |
| Specialist YoY (Pearson / Spearman) | 0.3133 / 0.2487 |
| Top-10 retention | 0.5242 |
| Top-20 retention | 0.5684 |
| Predictive rho (next-season RAPM) | 0.3296 |
| Center next-season rho | 0.3292 |
| Archetype confidence mean | 0.7594 |
| Std(DBKE final) | 0.4891 |
| Penalty asymmetry (|p5|/p95) | 0.8609 |
| Mean absolute rank shift | 63.9991 |
| Defensive driver share | 0.4986 |

### Layer 1 — Offense/Defense Weight Grid (Independent)

Grid evaluated: 50/50, 53/47, 55/45, 57/43, 60/40.

Observed behavior:
- Offense-heavier settings increased predictive rho but reduced global YoY in this data pass.
- Best weighted candidate under the script objective: **55/45**.

Winner (55/45) vs baseline deltas:
- Predictive rho: **+0.0170**
- Global YoY rank corr: **-0.0115**
- Top-20 retention: **+0.0082**
- Mean absolute rank shift: **+0.9154** (worse)

Interpretation: improves short-horizon predictive fit but does not improve stability metrics in isolation on this snapshot.

### Layer 2 — Specialty-Aware Continuous Dampening (Independent)

Alpha grid evaluated: 0.03, 0.05, 0.07.

Winner: **alpha = 0.07**.

Winner deltas vs baseline:
- Predictive rho: **+0.0044**
- Global YoY rank corr: **-0.0005** (near-flat)
- Center next-season rho: **+0.0009**
- Top-20 retention: **-0.0238**

Interpretation: very small changes; mild predictive benefit, modest retention cost.

### Layer 3 — Defensive Tail Micro-Convex Scaling (Independent)

Exponents evaluated: 1.03, 1.05, 1.08 with:

$$
D_{new}=\mu+\operatorname{sign}(D-\mu)\cdot |D-\mu|^{\gamma}
$$

Winner: **gamma = 1.08**.

Winner deltas vs baseline:
- Global YoY rank corr: **+0.0017**
- Specialist YoY Pearson: **+0.0160**
- Center next-season rho: **+0.0031**
- Std(DBKE final): **-0.0133**
- Mean absolute rank shift: **-0.3629** (improves)

Interpretation: best stability-oriented single layer in this run; modestly improves defensive specialist persistence while keeping variance controlled.

### Layer 4 — Confidence-Weighted Defensive Blend (Independent)

Implemented as:

$$
w_{rapm}=0.45\cdot(0.9+0.2\cdot conf),\quad w_{rapm}\in[0.40,0.50]
$$

and blended between defensive portable signal and stabilized defensive RAPM.

Observed deltas vs baseline:
- Global YoY: **+0.0400**
- Specialist YoY Pearson: **+0.2811**
- Top-10 retention: **-0.3367**
- Top-20 retention: **-0.3938**
- Std(DBKE final): **+0.5236**
- Defensive driver share: **+0.3114**

Interpretation: materially destabilizes distribution geometry and overwhelms defensive share despite strong YoY improvement; **flagged as non-viable for combined deployment** under guardrails.

### Layer 5 — Defensive Dimensional Cleanup (Independent)

Procedure:
1) Identify high-correlation defensive axes (|r| > 0.85)
2) Drop collinear axis from each pair
3) Project remaining defensive axes to whitened PCA space
4) Recompute archetype confidence proxy in orthogonal space

Findings:
- Rating metrics unchanged by design.
- Orthogonal confidence proxy dropped from 0.7594 to 0.3533 in this implementation.

Interpretation: this first-pass confidence proxy is too conservative; keep as experimental geometry probe only (no production use yet).

### Layer 6 — Historical Axis Volatility Scaling (Independent)

Grid evaluated: `k = 0.10, 0.15, 0.20`, with axis scaling:

$$
AxisScale=\frac{1}{1+k\cdot \sigma^2_{historical}},\quad
x_{new}=\mu + AxisScale\cdot(x-\mu)
$$

Winner: **k = 0.20**.

Winner deltas vs baseline:
- Global YoY: **+0.0098**
- Specialist YoY Pearson: **+0.0010**
- Center next-season rho: **+0.0017**
- Std(DBKE final): **-0.0479**
- Mean absolute rank shift: **-0.7208**
- Predictive rho: **-0.0104**

Interpretation: strongest rank-stability improvement among variance controls; small predictive tradeoff.

### Combined Model Evaluation (Constrained, Max 3 Structural Changes)

Guardrail selection now excludes non-viable layers (notably Layer 4 in this run).

Applied layers:
- Layer 1 winner: **55/45**
- Layer 6 winner: **k=0.20**
- Layer 3 winner: **gamma=1.08**

Combined model metrics:
- Global YoY rank corr: **0.5719**
- Specialist YoY (Pearson / Spearman): **0.3293 / 0.2487**
- Top-10 retention: **0.5086**
- Top-20 retention: **0.5765**
- Predictive rho: **0.3436**
- Center next-season rho: **0.3275**
- Std(DBKE final): **0.4758**
- Penalty asymmetry: **0.8585**
- Mean absolute rank shift: **64.6450**

Combined deltas vs baseline:
- Predictive rho: **+0.0140**
- Specialist YoY Pearson: **+0.0160**
- Top-20 retention: **+0.0082**
- Global YoY rank corr: **-0.0097**
- Center next-season rho: **-0.0017**

Interpretation:
- The constrained stack improves predictive signal and specialist persistence.
- Global rank stability remains a tradeoff in this pass (YoY dip), so this should stay in experimental mode pending further calibration.

### v3.1 Experimental Conclusion (Current Pass)

Recommended next implementation path:
1) Keep Layer 3 (gamma=1.08) and Layer 6 (k=0.20) as primary stability tools.
2) Keep Layer 1 as a tunable switch between predictive gain (55/45) and stability anchor (60/40).
3) Rework Layer 4 before any production inclusion (it currently causes severe retention and variance distortion).
4) Revisit Layer 5 confidence reconstruction method before using it as a confidence replacement.

This preserves controlled-impact experimentation and keeps v3.1 aligned with the “refinement, not reconstruction” objective.

### V3.1 Policy Update — Dual Split Co-Production (60/40 + 55/45)

As of the latest v3.1 run, split policy is now explicit and persistent:

- `60/40` remains the default stability anchor split.
- `55/45` remains the predictive-tilted companion split.
- Both are now co-produced in dedicated v3.1 score files so future sessions do not need to infer or re-derive this policy.

Generated baseline score artifacts:
- `data/processed/bke/BKE_Scores_v31_60_40.json`
- `data/processed/bke/BKE_Scores_v31_55_45.json`

This formalizes the Layer 1 interpretation: `55/45` is a predictive lever, while `60/40` is the stability baseline.

### Second-Pass Experiment — Layer 3 + Layer 6 Only (No Weight Shift)

Requested second-pass was run with only:
- Layer 6 axis volatility scaling (`k=0.20`)
- Layer 3 defensive tail scaling (`gamma=1.08`)
- No Layer 1 weight shift (kept at `60/40`)

Run timestamp: `2026-02-28 23:15:58`.

Second-pass metrics:
- Global YoY rank corr: **0.5937** (delta **+0.0120**)
- Specialist YoY Pearson: **0.3308** (delta **+0.0175**)
- Top-10 retention: **0.5565** (delta **+0.0323**)
- Predictive rho: **0.3152** (delta **-0.0144**)
- Center next-season rho: **0.3336** (delta **+0.0043**)
- Std(DBKE final): **0.4257** (delta **-0.0634**)
- Mean absolute rank shift: **63.1297** (delta **-0.8693**)
- Defensive driver share: **0.4297** (delta **-0.0690**)

Additivity gate outcome:
- `global_yoy_not_canceled`: pass
- `rank_shift_not_canceled`: pass
- `dbke_std_not_canceled`: pass
- Baseline-relative stability checks: pass
- **Decision: additive benefits confirmed; promoted to v3.1 output artifacts**

Promoted second-pass artifacts:
- `data/processed/bke/BKE_Scores_v31_60_40_layer36.json`
- `data/processed/bke/BKE_Scores_v31_55_45_layer36.json`
- `reports/bke_v31_layer36_second_pass.json`

Interpretation:
- Layer 3 and Layer 6 are additive on volatility-control targets (YoY/stability/variance), even without Layer 1 reweighting.
- Predictive rho is lower than baseline in this stack, so this should be treated as a stability-first profile.
