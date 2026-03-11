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

### Implementation Scope
- Modify `season_sim.py` to expose per-season margin arrays (currently internal).
- Add a `sample_representative_seasons(n_samples, percentiles)` utility.
- Modify `player_stats_sim.py.simulate_detailed_season()` to accept a pre-drawn margin array instead of computing its own.
- Add aggregation logic to compute P10/P50/P90 player stat projections.
- Update validation to compare ranges vs. single-point estimates.

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
