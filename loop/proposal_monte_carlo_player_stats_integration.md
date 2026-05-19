# Proposal: Monte Carlo ↔ Player Stats Integration

## Current Architecture

The BKE simulation core has **two independent subsystems** that do not communicate:

### 1. Monte Carlo Season Simulation (`season_sim.py`)
- Runs **10,000 vectorized season simulations** (configurable).
- Each simulation plays the full 82-game schedule.
- Uses a normal draw per game: `margin ~ N(delta_mu, sigma_game)`.
- Produces **team-level** W/L records, playoff probabilities, and confidence intervals.
- Does **not** simulate individual player stats — only team outcomes.

### 2. Player Stat Simulation (`player_stats_sim.py`)
- Runs a **single deterministic pass** through the 82-game schedule.
- For each game, allocates minutes to a 10-player rotation, generates per-player box scores.
- Uses archetype-based matchup adjustments (scoring efficiency, 3PT efficiency, TOV multiplier, etc.).
- Reconciles team totals to a derived score that is **not connected** to the Monte Carlo margin draws.
- Produces per-player season averages (PPG, APG, RPG, etc.) from this single sample.

### Key Problem
The player stat sim produces one deterministic season, while the Monte Carlo produces 10K probabilistic seasons. There is no feedback loop:

- Player stats do not vary across Monte Carlo iterations.
- The team score in the player stat sim is derived independently from Monte Carlo margin draws.
- Injury/rest scenarios affect Monte Carlo via roster changes but player stats don't reflect those dynamics.

---

## Option A: Lightweight Coupling (Recommended — Phase 1)

**Goal:** Use Monte Carlo season outcomes to calibrate player stat distributions without making the full sim per-player.

### How It Works
1. Run Monte Carlo as-is (10K sims) → produces W/L distribution + per-game margin arrays.
2. **Sample K representative seasons** (e.g., K=25) from the Monte Carlo output:
   - 5 from the 10th percentile win outcome (bad seasons)
   - 15 from the 40th–60th percentile (median seasons)
   - 5 from the 90th percentile (good seasons)
3. For each sampled season, run the player stat sim using the **actual margin array** from that Monte Carlo iteration as the game outcomes (instead of deriving team scores independently).
4. Aggregate player stats across the K samples → produce median, P10, P90 player stat projections.

### Benefits
- Player stats now reflect the range of team outcomes.
- Star players have different stat lines in 50-win vs 35-win seasons.
- Maintains the speed advantage of vectorized Monte Carlo (no per-player work in the 10K loop).
- Only adds K=25 player stat passes, which takes ~2-5 seconds.

### Implementation Scope (updated for forecast-first emphasis)

- Recently implemented (already in the codebase):
   - `src/simulation/player_stats_sim.py`: team-level stat budget sampling, Dirichlet usage allocation, capped multinomial allocation, Beta‑Binomial shooting, and reconciliation to team totals.
   - `src/simulation/simulation_config.py`: tuning knobs added (e.g., `PLAYER_GAME_ARCHETYPE_EFFECT_SCALE`, `PLAYER_GAME_DIRICHLET_SCALE`, `PLAYER_GAME_BETA_CONCENTRATION`, `PLAYER_GAME_USAGE_CAP`).
   - `app/simulation_viewer.py`: frontend single-game sim mirrored to backend for parity.

- Remaining work to enable Option A (forecast coupling):
   - `src/simulation/season_sim.py` / `src/simulation/run_forecast.py`: expose per-season margin arrays and add `sample_representative_seasons(n_samples=25, percentiles=[10,50,90])` to pick representative MC iterations.
   - Add a driver that calls `player_stats_sim` for each sampled margin array (e.g., `simulate_detailed_season(margin_array)`) and writes per-sample player-season artifacts.
   - Aggregation: compute per-player P10/P50/P90 across the K samples and emit `reports/player_stat_sim_ranges.json` plus per-sample parquet outputs under `data/processed/simulation/`.
   - Validation: ensure `scripts/validate_player_stat_sim.py` verifies forecast-mode artifacts and writes `reports/player_stat_sim_validation_forecast.json`.

### Estimated Complexity: Medium

---

## Option B: Full Per-Player Monte Carlo (Phase 2+)

**Goal:** Run player-level stat generation inside each Monte Carlo iteration.

### How It Works
1. For each of the 10K season simulations, after drawing the margin for each game, also simulate per-player box scores.
2. Accumulate player stat distributions across all 10K iterations.
3. Produce full probabilistic player projections with confidence intervals.

### Drawbacks
- **Performance:** 10K × 82 games × ~10 players/team × 30 teams = ~246M player-game simulations. This would take 10-30 minutes vs. the current ~15 seconds.
- **Diminishing returns:** The marginal information gain from 10K player stat samples vs. 25 representative samples is small for most use cases.
- **Complexity:** Requires restructuring the vectorized Monte Carlo loop to include per-player logic.

### When to Consider
- Only if downstream products require full probabilistic player stat distributions (e.g., DFS optimization, player prop modeling).
- Could be done as a separate "deep sim" mode that runs overnight.

---

## Recommendation

**Implement Option A (Lightweight Coupling) first.** It provides 90% of the benefit at 5% of the cost. The key wins:

1. Player projections reflect team outcome uncertainty.
2. P10/P50/P90 ranges give a realistic spread for player stats.
3. The Monte Carlo margin draws become the single source of truth for game outcomes across both subsystems.
4. The frontend single-game sim can also use this model: when a user simulates ATL vs BOS, the backend can provide a margin draw and the player stat sim uses that exact margin.

### Migration Path
```
Phase 1: Option A (25 representative seasons)
   ↓
Phase 2: If Option A proves insufficient, implement Option B as an optional "deep sim" mode
```

---

## Forecast Performance (latest run)

- **Season-level (2023-24 forecast):** margin MAE = 7.71, RMSE = 9.74, correlation = 0.7057. See `reports/forecast_season_results.json`.
- **Player-level (forecast mode):** PTS MAE = 2.64, RMSE = 3.39, correlation = 0.9002; AST MAE = 0.71, correlation = 0.8811; REB MAE = 1.00, correlation = 0.8777. See `reports/player_stat_sim_validation_forecast.json`.
- **Lineup / rotation:** starter overlap ≈ 0.67, rotation_corr ≈ 0.36; starter/clutch targets not fully met. See `reports/forecast_lineup_profiles.json`.
- **Known weakness:** the single-game score reconciliation path can still overshoot the target on rare seeds when the downward correction has no FT makes to remove. The frontend viewer mirrors the same logic, so this output should be treated as review-only until the reconciliation path or regression coverage is hardened.

Notes: These forecast artifacts were produced after applying the team-stat budget + Dirichlet + Beta‑Binomial changes in `src/simulation/player_stats_sim.py`. Forecast artifacts should be treated as the canonical evaluation for forecasting quality.

## Updated Plan & Next Steps (forecast-focused)

1. Implement Option A sampling in `src/simulation/run_forecast.py` / `src/simulation/season_sim.py`:
   - Add `sample_representative_seasons(n_samples=25, percentiles=[10,50,90])` to select representative Monte Carlo iterations.
   - For each sampled iteration, call `player_stats_sim.simulate_detailed_season(margin_array)` (or an equivalent wrapper) and persist per-sample player-season outputs.
2. Re-run the forecast pipeline to refresh canonical forecast artifacts:

```bash
python3 src/simulation/run_forecast.py --scenarios end_of_season --skip-preseason-fetch
```

3. Aggregate K-sample player stat distributions and emit:
   - `reports/player_stat_sim_ranges.json` (P10/P50/P90 per player)
   - Per-sample parquet outputs under `data/processed/simulation/` for auditing.
4. Validate forecast ranges:

```bash
python3 scripts/validate_player_stat_sim.py --mode forecast --scenario end_of_season
```

   - Inspect `reports/player_stat_sim_validation_forecast.json` for player/team biases and correlations.
5. Tune modeling knobs (grid/sweep) to improve forecast targets:
   - `PLAYER_GAME_DIRICHLET_SCALE`, `PLAYER_GAME_BETA_CONCENTRATION`, `PLAYER_GAME_USAGE_CAP`, `PLAYER_GAME_ARCHETYPE_EFFECT_SCALE` (in `src/simulation/simulation_config.py`).
6. If Option A does not meet downstream requirements, schedule Option B (deep per-player Monte Carlo) as an overnight job.


## Files Affected

| File | Change |
|------|--------|
| `src/simulation/season_sim.py` | Expose per-season margin arrays; add `sample_representative_seasons()` |
| `src/simulation/player_stats_sim.py` | Accept pre-drawn margins; multi-season aggregation |
| `src/simulation/simulation_config.py` | Add `PLAYER_SIM_REPRESENTATIVE_SAMPLES = 25` |
| `scripts/validate_player_stat_sim.py` | Validate P10/P50/P90 ranges instead of single point |
| `app/simulation_viewer.py` | Display stat ranges in player stat sim tab |
| `reports/` | New artifact: `player_stat_sim_ranges.json` |

---

*Created: Session context — Monte Carlo and Player Stat Sim are currently independent subsystems.*
