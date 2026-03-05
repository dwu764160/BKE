# Simulation Core Summary

## Append-Only Rule
- Never delete sections, only update content of the sections.
- If it is a new version update, track the version number and date every time you update a section.

---

## Entry: 2026-03-04 — Simulation Core v1 Step 1 (Margin-Based Team Simulation)

### Philosophy & Design Principles

**Margin-based, team-level.** Step 1 of the simulation core uses projected team net ratings and volatility from the Player Evaluation Core (PEC Step 3) to simulate NBA season outcomes. No lineup logic, no playoffs adjustments, no player stat simulation — pure margin math.

**Normal distribution game model.** Each game's margin is drawn from a normal distribution where the mean is the expected strength gap between teams (adjusted for home court) and the variance combines each team's projected volatility plus an irreducible league noise floor. Win probability is computed analytically via the standard normal CDF.

**Monte Carlo for distributions, analytics for point estimates.** The deterministic game model produces exact win probabilities analytically. Monte Carlo simulation (10,000 seasons) is used only to derive distributional outputs: win distributions, playoff probability, percentile bands.

### What was built

Three scripts in `src/simulation/`:

#### game_model.py — Layers 1-3 (Parameter, Deterministic Game Model, Schedule Engine)

**Layer 1: Parameter Layer**
- `SimConfig` dataclass: `sigma_league=3.0`, `home_court_advantage=2.0`, `n_simulations=10000`, `random_seed=42`
- Constants stored centrally in `src/player_eval/constants.py`

**Layer 2: Deterministic Game Model**
- `compute_game_distribution(team_a, team_b, is_home_a, config)` — stateless, pure function
- Computes: `delta_mu = (mu_A + H) - mu_B`, `sigma_game = sqrt(sigma_A² + sigma_B² + sigma_league²)`, `win_prob_a = Phi(delta_mu / sigma_game)`
- Returns: delta_mu, sigma_game, win_prob_a, z_score

**Layer 3: Schedule Engine**
- `build_schedule(season)` — parses `data/historical/team_game_logs.parquet`
- Identifies home games via MATCHUP field ("vs." = home, "@" = away)
- Returns list of `Game` dataclasses with actual margins for validation

**Data loading:**
- `load_team_params()` — reads `team_feature_aggregation.parquet` for team `mu` (team_net_rating_projected) and `sigma` (vol_total)
- Filters out invalid team abbreviations (NAN/empty from traded player artifacts)

**Output:** `reports/simulation_step1_results.json`

#### season_sim.py — Layers 4-5 (Monte Carlo Engine, Aggregation)

**Layer 4: Monte Carlo Simulation Engine**
- `simulate_season(schedule, team_params, config)` — fully vectorized NumPy implementation
- Precomputes `delta_mu` and `sigma_game` arrays for entire schedule
- Samples all game margins at once: `margins = rng.normal(loc, scale, size=(n_games, n_sims))`
- Accumulates wins per team via index mapping

**Layer 5: Aggregation Layer**
- `aggregate_results(win_distributions, team_params, playoff_cutoff=42)`
- Per-team outputs: projected wins, win_std, percentiles (p5/p25/median/p75/p95), min/max, playoff probability (P(wins >= 42)), 50+ win probability, 60+ win probability
- Merges with actual records for comparison

**Output:** `reports/simulation_step1_season_results.json`

#### validate_sim.py — Layer 6 (Validation)

**A. Game-Level Validation:** Brier score, log loss, accuracy, margin RMSE/MAE, home win rate comparison
**B. Calibration:** 10-bin overconfidence test — does a predicted 70% WP correspond to ~70% actual?
**C. Season-Level Validation:** MAE, RMSE, correlation of projected vs actual wins
**D. Aggregate Calibration Error:** weighted mean calibration error across all bins and seasons

**Output:** `reports/simulation_step1_validation.json`

### Mathematical Concepts

1. **Game margin distribution:** $M_{A,B} \sim \mathcal{N}(\Delta\mu, \sigma_{game})$ where $\Delta\mu = (\mu_A + H) - \mu_B$ and $\sigma_{game} = \sqrt{\sigma_A^2 + \sigma_B^2 + \sigma_{league}^2}$

2. **Win probability (analytical):** $P(A \text{ wins}) = \Phi\left(\frac{\Delta\mu}{\sigma_{game}}\right)$ where $\Phi$ is the standard normal CDF

3. **League noise floor:** $\sigma_{league} = 3.0$ — irreducible per-game randomness (refereeing, shooting variance, in-game injuries, etc.)

4. **Home court advantage:** $H = 2.0$ net rating points added to home team's projected rating

5. **Brier score:** $\frac{1}{n}\sum(p_i - y_i)^2$ — measures calibration of probabilistic predictions (lower = better, 0.25 = naive coin flip)

6. **Log loss:** $-\frac{1}{n}\sum[y_i\log(p_i) + (1-y_i)\log(1-p_i)]$ — information-theoretic prediction quality (0.693 = coin flip)

### Constants (centralized in constants.py)

| Constant | Value | Purpose |
|---|---|---|
| SIGMA_LEAGUE | 3.0 | Irreducible game-level noise floor |
| HOME_COURT_ADVANTAGE | 2.0 | Net rating points for home team |
| SEASON_SIMULATIONS | 10,000 | Monte Carlo iterations per season |
| RANDOM_SEED | 42 | Reproducibility seed |
| Playoff cutoff | 42 wins | Threshold for playoff probability |

### Validation Results (3 seasons: 2022-23, 2023-24, 2024-25)

#### Game-Level Metrics

| Season | Games | Brier | Log Loss | Accuracy | Margin RMSE | Home WR (act/pred) |
|---|---|---|---|---|---|---|
| 2022-23 | 1230 | 0.2281 | 0.6536 | 64.9% | 12.90 | 0.581 / 0.580 |
| 2023-24 | 1230 | 0.2148 | 0.6191 | 65.6% | 14.13 | 0.543 / 0.575 |
| 2024-25 | 1225 | 0.2053 | 0.5975 | 69.2% | 13.81 | 0.544 / 0.573 |

Brier scores well below 0.25 naive baseline. Log loss well below 0.693 coin flip. Accuracy 65-69%.

#### Season-Level Metrics

| Season | MAE | RMSE | Correlation | Mean Error | Max Over | Max Under |
|---|---|---|---|---|---|---|
| 2022-23 | 5.57 | 6.80 | 0.821 | 0.0 | +14.1 | -9.1 |
| 2023-24 | 6.06 | 7.05 | 0.867 | 0.0 | +12.3 | -17.2 |
| 2024-25 | 3.76 | 4.34 | 0.951 | -0.17 | +8.1 | -8.4 |
| **Overall** | **5.13** | **6.19** | **0.886** | **-0.06** | — | — |

90 team-seasons evaluated. Overall correlation 0.886 with MAE of 5.13 wins.

#### Calibration
- Aggregate calibration error: 0.060 (6.0% average deviation from perfect calibration)
- Slight overconfidence in extreme bins (>90% predicted WP)
- Well-calibrated in middle range (40-70% predicted WP)

### Data Sources

| Source | Path | Fields Used |
|---|---|---|
| Team feature aggregation | data/processed/player_eval/team_feature_aggregation.parquet | team_net_rating_projected (mu), vol_total (sigma) |
| Team game logs | data/historical/team_game_logs.parquet | MATCHUP, PTS, OPP_PTS, WL, GAME_ID, GAME_DATE |

### Outputs

| File | Contents |
|---|---|
| reports/simulation_step1_results.json | Team parameters + game predictions for all seasons |
| reports/simulation_step1_season_results.json | Full simulation results: per-team projected wins, distributions, actual records |
| reports/simulation_step1_validation.json | Validation metrics: Brier, log loss, calibration, season-level comparison |

### Known Issues & Future Work

1. **NAN team artifact:** `team_feature_aggregation.parquet` contains a phantom "NAN" team in 2023-24 and 2024-25 from traded players without team assignments. Filtered out in `load_team_params()`.
2. **Home court advantage calibration:** Predicted home win rate (0.573-0.580) slightly exceeds actual (0.543-0.581) in recent seasons. HCA may need downward adjustment from 2.0 to ~1.5.
3. **Overconfidence in extremes:** 90%+ predicted WP games only win ~79% — sigma_league may need slight increase.

---

## Entry: 2026-03-05 — Simulation Core v1 Step 2 (Lineup Projection)

### Philosophy & Design Principles

**Phase-structured team strength.** Step 2 transforms player-level PEC features into three game-phase team strengths (starters, rotation, clutch) so downstream simulation can move beyond one blended team mean.

**Role/value separation preserved.** Position-role eligibility and lineup constraints are structural (Guard/Wing/Big coverage + usage filters). Impact is applied as a weighted signal after eligibility to avoid role inflation from pure value metrics.

The step produces three team phases for each team-season:

1. Starter unit
2. Rotation unit
3. Clutch unit

Each phase emits mean strength (`mu`) and uncertainty (`sigma`) so downstream simulation layers can weight game phases without collapsing to a single team scalar.

### What was built

New script in `src/simulation/`:

#### lineup_projection.py — Step 2 (Lineup Modeling + Validation)

- Loads inputs from:
	- `data/processed/player_eval/player_impact_profiles.parquet`
	- `data/processed/player_position_estimates.parquet`
	- `data/historical/player_clutch_stats_all.parquet`
	- `data/processed/metrics_lineups.parquet`
	- `data/historical/teams.parquet`
- Builds team-season player pools and computes:
- `clutch_score` from clutch share + normalized impact/minutes blend
- `starter_score` from minutes + impact + positional scarcity + clutch share
- player volatility proxy from impact stability (bounded floor/ceiling)
- rotation proxy using staggered starter impact + bench impact blend
- Selects predicted starter and clutch lineups under structural constraints.
- Writes a team-level validation payload with overlap/rotation diagnostics and frontend-ready name/archetype metadata.

### Mathematical Concepts

1. **Clutch share:**
   \( C_i = \frac{\text{clutch minutes}_i}{\sum_j \text{clutch minutes}_j} \)

2. **Normalization:**
   \( I_i = \text{minmax}(impact_i), \quad M_i = \frac{minutes_i}{\max(minutes)} \)

3. **Clutch score:**
   \( S^{clutch}_i = 0.80 \cdot C_i + 0.20 \cdot (0.65 \cdot I_i + 0.35 \cdot M_i) \)

4. **Starter score:**
   \( S^{start}_i = 0.50 \cdot M_i + 0.25 \cdot I_i + 0.10 \cdot Pos_i + 0.15 \cdot C_i \)
   where \(Pos_i\) is normalized positional scarcity within team pool.

5. **Lineup optimization constraint:**
   Select 5-player lineup maximizing score sum subject to at least one Guard, one Wing, and one Big.

6. **Rotation mean strength:**
   \( \mu_{rotation} = \alpha \cdot \mu_{stagger} + (1-\alpha) \cdot \mu_{bench} \), with \(\alpha = 0.55\).

7. **Player volatility proxy:**
   \( \sigma_i = \text{clip}(3.5 + 10.0 \cdot (0.60 - stability_i), 1.5, 6.5) \)

### Constants (centralized in simulation_config.py)

| Constant | Value | Purpose |
|---|---|---|
| LINEUP_SIZE | 5 | Number of players per projected lineup |
| MIN_MPG_FOR_POOL | 3.0 | Player-pool floor for lineup optimization |
| CLUTCH_WEIGHT_C | 0.80 | Clutch share weight in clutch score |
| CLUTCH_WEIGHT_TALENT | 0.20 | Talent block weight in clutch score |
| CLUTCH_WEIGHT_I / M | 0.65 / 0.35 | Impact/minutes mix inside clutch talent block |
| STARTER_WEIGHT_M / I / POS / C | 0.50 / 0.25 / 0.10 / 0.15 | Starter score composition |
| ROTATION_ALPHA | 0.55 | Stagger-vs-bench blend weight |
| STAGGER_MINUTES_THRESHOLD | 24.0 | Minute threshold for staggered starter contribution |
| STEP2_IMPACT_COLUMN | impact_total_impact | Player impact feature source |
| STEP2_MINUTES_COLUMN | mpg | Minutes feature source |
| STEP2_STABILITY_COLUMN | impact_stability | Stability feature source |

### Validation Results (3 seasons: 2022-23, 2023-24, 2024-25)

#### Season-Level Metrics

| Season | Teams | Starter Overlap Rate | Starter Overlap Count | Clutch Overlap Rate | Clutch Overlap Count | Rotation Corr |
|---|---|---|---|---|---|---|
| 2022-23 | 30 | 0.7000 | 3.500 | 0.8600 | 4.300 | 0.3778 |
| 2023-24 | 30 | 0.7600 | 3.800 | 0.8733 | 4.367 | 0.7134 |
| 2024-25 | 30 | 0.6733 | 3.367 | 0.8400 | 4.200 | 0.5013 |
| **Overall** | **90** | **0.7111** | **3.556** | **0.8578** | **4.289** | **0.5367** |

Starter overlap target (>=0.80) is not met in any season. Clutch overlap target (>=3.5 of 5) is met in all seasons. Rotation target (corr >=0.70) is met in 2023-24 only.

#### Coverage
- Starter overlap available: 90/90 team-seasons
- Clutch overlap available: 90/90 team-seasons
- Rotation benchmark available: 90/90 team-seasons

### Data Sources

| Source | Path | Fields Used |
|---|---|---|
| PEC player profiles | data/processed/player_eval/player_impact_profiles.parquet | `player_id`, `player_name`, `team_abbreviation`, `mpg`, `impact_total_impact`, `impact_stability`, `off_primary_archetype`, `def_primary_archetype` |
| Position estimates | data/processed/player_position_estimates.parquet | `pct_pg`, `pct_sg`, `pct_sf`, `pct_pf`, `pct_c`, `primary_position_estimate` |
| Clutch stats | data/historical/player_clutch_stats_all.parquet | `clutch_minutes`, `clutch_gp` |
| Lineup metrics | data/processed/metrics_lineups.parquet | `NET_RTG`, `total_poss`, `lineup_ids` |
| Team metadata | data/historical/teams.parquet | `abbreviation`, `conference`, `full_name` |

### Outputs

| File | Contents |
|---|---|
| data/processed/simulation/simulation_step2_lineup_profiles.parquet | Team-season Step 2 phase outputs (`mu_start`, `mu_rotation`, `mu_clutch`, `sigma_*`, overlap metrics) |
| reports/simulation_step2_lineup_profiles.json | Frontend-ready Step 2 payload (season summaries, team cards, predicted lineup players with offensive/defensive archetypes, observed named proxies) |
| reports/simulation_step2_validation.json | Aggregate Step 2 validation summary metrics |

### Frontend Integration

- `app/simulation_viewer.py` now generates a two-view simulation UI:
	- Step 1 tab: season simulation tables/charts + single-game simulator
	- Step 2 tab: season-level lineup summary cards and clickable team cards
- Team cards open a modal showing:
	- predicted starters and clutch lineups with offensive and defensive archetypes
	- phase means/volatilities
	- validation overlaps and rotation benchmark comparison
	- observed proxy player names only (no ID display)

### Known Issues & Future Work

1. **Starter and rotation gaps remain:** Clutch identification is strong, but starter overlap and rotation correlation are below target in two of three seasons.
2. **Lineup volatility on weak teams:** Teams with high in-season roster churn/experimentation create lower lineup stability and noisier validation targets.
3. **Proxy limitations:** Highest-possession lineup and bench-heavy NET_RTG are practical proxies, but they are not full coaching-intent ground truth.
4. **Next upgrades:** Add lineup stability priors, injury/transaction continuity features, and role-aware starter priors (coach tendency + position-band flexibility).

---

## Entry: 2026-03-05 — Step 2 Position-Band Policy Pass (Minimal-Impact)

### Philosophy Update

Step 2 now explicitly follows a two-layer position policy:

1. **Canonical identity layer (five-band):** `Guard`, `Guard-Forward`, `Forward`, `Forward-Center`, `Center`.
2. **Structural lineup layer (coarse):** `Guard`, `Wing`, `Big` for lineup constraint solving only.

Canonical identity must remain visible in payloads and docs even when coarse structural roles are used for optimization constraints.

### Minimal-Impact Changes Applied

- `src/simulation/lineup_projection.py`
	- Hardened role parsing so canonical/legacy hybrid labels map consistently to structural roles.
	- Added `position_band` to Step 2 player payloads (`starter_players`, `clutch_players`, `pool_players`).
- `app/simulation_viewer.py`
	- Player rows now render as `role · position_band` when available.
- `src/modeling/model_config.py`
	- Position bucket definitions list canonical labels first and retain legacy aliases for compatibility.
- `src/data_compute/compute_defensive_archetypes.py`
	- Added explicit legacy-script warning directing users to v2 script.

### Impact Audit and Reproduction Results

- **Layer 1 hybrid-collapse exposure (existing modeling output):**
	- Hybrid rows in `bke_v28_decomposition.parquet`: `198`
	- Hybrid rows collapsed into non-hybrid `position_bucket`: `198/198` (100%)
- **Step 2 fallback exposure:**
	- Rows with missing `role_from_position` fallback usage: `0/1680`
- **Post-change Step 2 metrics (no drift):**
	- Starter overlap rate mean: `0.7111`
	- Clutch overlap rate mean: `0.8578`
	- Rotation correlation: `0.5367`
- **Post-change v27 output drift check:**
	- Raw file MD5 varies per run because payload includes `generated_at`.
	- Stable-content MD5 (after removing `generated_at`) remains unchanged across reruns: `d5588c20dcdb083d6b9724e50982b87d`.

### Deferred (High-Impact) Items Requiring Approval

1. Preserve hybrid bands end-to-end inside decomposition cohorting (`position_bucket`) rather than alias-collapse.
2. Redesign matchup/lineup taxonomy inputs to carry explicit canonical bands through all downstream grouped-percentile layers.

These are held for explicit approval because they can alter grouped percentile baselines and rank distributions.