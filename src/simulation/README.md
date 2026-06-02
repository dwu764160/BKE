# src/simulation/ — Simulation Core

Generative possession-level basketball simulation. Step 1 (cross-team matchup engine) and Step 2 (possession engine) are both complete and calibrated.

---

## Architecture

```
game_model.py          Layer 1-3: SimConfig, TeamParams, game distribution, schedule engine
season_sim.py          Layer 4-5: Monte Carlo season simulation, aggregation (margin + PPP models)
lineup_projection.py   Step 2: Starter / rotation / clutch lineup selection from player profiles
possession_engine.py   Step 2.1: Generative engine — RateModel, OutcomeSamplerResolver, BoxScoreAggregator
player_stats_sim.py    Player stat simulation helpers
simulation_config.py   All path constants and hyperparameter defaults
gbdt_game_model.py     Step 5: Walk-forward GBDT + Elo ensemble (LightGBM)
validate_forecast.py   Walk-forward game-level forecast harness + rest/YTD helpers
validate_sim.py        Backtest validation vs actual records (Brier, log-loss, margin RMSE)
run_forecast.py        Forecast pipeline orchestrator (multi-scenario)
train_minute_model.py  (alternate) minute model training in simulation context
```

---

## Key concepts

**Step 1 — Cross-team matchup engine:** 3-pair archetype assignment (Primary Perimeter Threat, Interior Threat + Big, Band+Talent Residual). Adjustments stored in `data/processed/bke/cross_team_interactions.parquet`. Run via `scripts/compute_matchup_adj.py`.

**Step 2 — Generative possession engine:** Each possession sampled possession-by-possession; box-score emerges from rate model. `OutcomeSamplerResolver` is the production sampler; `MarkovEventResolver` is the planned Game track upgrade (same seam). Run via `scripts/run_possession_engine.py`.

**Step 5 — GBDT + Elo ensemble:** Walk-forward, out-of-sample. LightGBM stacks Gaussian + Elo + rest features. Elo is currently the strongest single game-level model (Brier 0.2193 aggregate / 0.2092 on 2025-26).

---

## Production scripts (run via scripts/)

| Runner | Module entry | Purpose |
|--------|-------------|---------|
| `scripts/run_possession_engine.py` | `possession_engine.py` | Calibrate + simulate; `--mode v1` is production (9-man) |
| `scripts/validate_possession_engine.py` | `possession_engine.py` | Step 2.1 calibration gate |
| `scripts/compute_matchup_adj.py` | (standalone) | Build `cross_team_interactions.parquet` |

---

## Calibration status (Step 2.1, 2026-06-02)

Walk-forward 2024-25 holdout, v1 mode, 200 sims:

| Metric | Value | Oracle floor |
|--------|-------|-------------|
| Player pts MAE | 5.79 | 4.80 |
| Player ast MAE | 1.75 | 1.42 |
| Player reb MAE | 2.53 | 1.93 |
| Player P10-P90 coverage | 0.754 | — |
| Team pts MAE | 10.33 | — |
| Team bias | +1.56 | — |
| Team P10-P90 coverage | 0.925 | — |

Residual bias in Star/Starter tier is forecast-limited (2024-25 breakout players) — not engine-limited. Do not reduce it by fitting on the holdout.

---

## Key outputs

```
data/processed/simulation/
  possession_box_distributions.parquet  ← per-(game, team, player) box-score distributions
  simulation_step2_lineup_profiles.parquet ← projected starter/rotation/clutch lineups
  simulation_step1_player_season_stats.parquet
  simulation_step1_player_game_samples.parquet
  forecast_step{1,2}_*.parquet (scenario-specific)

reports/
  possession_engine_validation.json
  simulation_step1_results.json
  simulation_step1_season_results.json
  gbdt_forecast_validation.json
  forecast_game_validation.json
```

---

## Simulation config

All paths and hyperparameter defaults live in `simulation_config.py`. Key defaults:

```python
SEASON_SIMULATIONS = 10_000
SIMULATION_RANDOM_SEED = 42
HOME_COURT_ADVANTAGE = 2.5          # pts/100 poss
SIGMA_LEAGUE = 11.0                 # game-level variance
B2B_PENALTY_HOME = -2.1             # pts/100 poss (fitted)
B2B_PENALTY_AWAY = -2.6
```
