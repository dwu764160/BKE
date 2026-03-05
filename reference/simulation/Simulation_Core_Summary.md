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