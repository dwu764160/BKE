# Walk-Forward Harness Plan

> **Status:** Brainstormed, not yet implemented. Decisions finalized 2026-05-19.
> **Scope:** Build the measurement infrastructure that scores forecast pipeline output
> against actual game outcomes. This is the prerequisite for measuring whether any other
> plan's fixes actually improve prediction accuracy.

---

## Why This Plan Exists

Currently, the only Brier/accuracy numbers in the project are **in-sample retrodictions**
from `src/simulation/validate_sim.py`. The forecast pipeline correctly projects team net
ratings using prior-season data, but then simulates a **synthetic schedule with no real
opponents**, producing no game-level metrics.

Without a real-schedule game-level harness:
- Cannot tell if RAPM fix (Plan 1) improved anything
- Cannot tell if B2B wiring (Plan 3) reduced Brier
- Cannot tell if temporal minute model (Plan 2) is better than the leaky version
- Cannot calibrate TEAM_SCALE on out-of-sample data
- Cannot benchmark against prediction markets later

This plan is the gate between "we made changes" and "we know they helped."

---

## Core Script: `src/simulation/validate_forecast.py`

### Inputs
1. **Projected team net ratings** from forecast pipeline output
   (`data/processed/team_feature_aggregation_forecast.parquet` or equivalent)
   — projected using prior-season player profiles + projection logic
2. **Actual game schedule + outcomes** for the test season
   (`data/historical/team_game_logs.parquet`)
3. **Game-day context features** (per game): is_b2b_home, is_b2b_away, days_rest_home,
   days_rest_away, home_team_id (for team-specific HCA lookup)
4. **Model parameters** (from `simulation_config.py`): SIGMA_LEAGUE, HOME_COURT_ADVANTAGE
   (or team-specific HCA if available), B2B_PENALTY, REST_DAY_BONUS

### Logic (Per Game)

```python
# For each actual game in test_season:
mu_home = team_ratings.loc[home_team, "projected_net_rating"]
mu_away = team_ratings.loc[away_team, "projected_net_rating"]

# Apply team-specific HCA if available, else flat 2.0
hca = team_hca.get(home_team, HOME_COURT_ADVANTAGE)

# Apply B2B / rest adjustments
adj_home = B2B_PENALTY * is_b2b_home + REST_DAY_BONUS * days_rest_home
adj_away = B2B_PENALTY * is_b2b_away + REST_DAY_BONUS * days_rest_away

# Compute expected margin and win probability
mu_margin = (mu_home + adj_home) - (mu_away + adj_away) + hca
sigma = SIGMA_LEAGUE * sqrt(game_pace / LEAGUE_AVG_PACE)  # pace-adjusted

p_home_win = scipy.stats.norm.cdf(mu_margin / sigma)
actual = 1 if (home_score > away_score) else 0
```

### Outputs

**Per-season aggregate metrics** (matching existing `validate_sim.py` schema for consistency):
- `n_games`
- `brier` = mean((p_home_win - actual)²)
- `log_loss` = mean(-(actual*log(p) + (1-actual)*log(1-p)))
- `accuracy` = mean(round(p_home_win) == actual)
- `margin_rmse` = sqrt(mean((mu_margin - actual_margin)²))
- `margin_mae` = mean(|mu_margin - actual_margin|)
- `home_wr_actual`, `home_wr_predicted`

**10-bin calibration table:**
- For each bin (0.0-0.1, 0.1-0.2, ..., 0.9-1.0):
  - `count` (games in this bin)
  - `mean_predicted` (avg predicted P(home_win))
  - `mean_actual` (avg actual outcome)
  - `calibration_error` = |mean_predicted - mean_actual|

**Output file:** `reports/forecast_game_validation.json`

---

## Ablation Testing (Measure Impact of Each Fix)

The harness must support running with different parameter configurations to isolate
the impact of each fix. Design:

```python
# Scenarios to run (in single execution):
SCENARIOS = [
    "baseline",           # flat HCA=2.0, no B2B, no rest, flat pace
    "team_hca",           # team-specific HCA, no B2B
    "team_hca_b2b",       # team-specific HCA + B2B penalty
    "team_hca_b2b_rest",  # full Phase A adjustments
    "with_pace",          # baseline + pace-adjusted sigma
    "full",               # all adjustments
]
```

Each scenario produces its own report with the same schema. The summary report shows
side-by-side Brier comparison:

```json
{
  "scenarios": {
    "baseline":             {"brier": 0.245, "logloss": 0.69, "accuracy": 0.62},
    "team_hca":             {"brier": 0.238, "logloss": 0.68, "accuracy": 0.63},
    "team_hca_b2b":         {"brier": 0.231, "logloss": 0.66, "accuracy": 0.65},
    "team_hca_b2b_rest":    {"brier": 0.229, "logloss": 0.66, "accuracy": 0.65},
    "with_pace":            {"brier": 0.244, "logloss": 0.69, "accuracy": 0.62},
    "full":                 {"brier": 0.227, "logloss": 0.65, "accuracy": 0.66}
  },
  "deltas_vs_baseline": {
    "team_hca":             {"brier_delta": -0.007, "interpretation": "fitted HCA helped"},
    "team_hca_b2b":         {"brier_delta": -0.014, "interpretation": "B2B is a strong signal"},
    ...
  }
}
```

This tells us directly which fixes matter and by how much.

---

## Walk-Forward Execution (Multiple Transitions)

The harness must execute across all available season transitions, not just the latest:

```python
# Available transitions (initially):
#   2022-23 → 2023-24 (project using 2022-23 data, score against 2023-24 games)
#   2023-24 → 2024-25 (project using 2023-24 data, score against 2024-25 games)
# After backfill:
#   2017-18 → 2018-19
#   2018-19 → 2021-22
#   2021-22 → 2022-23
#   ... etc.

# For each transition:
#   1. Run forecast pipeline in N→N+1 mode to project ratings
#   2. Run validate_forecast.py against actual N+1 schedule
#   3. Collect per-transition metrics

# Aggregate across transitions (count-weighted):
#   overall_brier = sum(brier_t * n_games_t) / sum(n_games_t)
```

The walk-forward aggregate is the headline number that gets reported.

---

## Baseline Comparison Suite

For every scenario, also report comparison against naive baselines:

| Baseline | Brier (expected) | Why It Matters |
|---|---|---|
| Coin flip (p=0.5) | 0.25 | Absolute floor |
| Always pick home (p=1) | ~0.45 | Worst-case extreme prediction |
| Home rate prior (p=0.55) | ~0.247 | "No information" benchmark |
| Talent-only Gaussian (current) | ?  | Current model |
| + team HCA | ? | Phase A2 impact |
| + B2B/rest | ? | Phase A3 impact |
| Vegas closing line (future) | ~0.21 | Aspirational target |

If our scenario doesn't beat home-rate prior (Brier 0.247), the model is worthless.
If it beats home-rate prior but doesn't approach Vegas (~0.21), there's edge to find.

---

## Implementation Sketch

```python
# src/simulation/validate_forecast.py

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm

from src.simulation.simulation_config import (
    SIGMA_LEAGUE, HOME_COURT_ADVANTAGE, LEAGUE_AVG_PPP, DEFAULT_PACE_PER_48,
)

def load_projected_ratings(season: str) -> pd.DataFrame:
    """Load team net ratings projected for `season` using prior-season data."""
    # Path to forecast pipeline output

def load_actual_games(season: str) -> pd.DataFrame:
    """Load actual games with outcomes, B2B flags, rest days."""
    # Filter team_game_logs for season

def score_game(p_home_win, actual, mu_margin, actual_margin):
    """Single-game metric contributions."""
    brier = (p_home_win - actual) ** 2
    if 0 < p_home_win < 1:
        logloss = -(actual * np.log(p_home_win) + (1 - actual) * np.log(1 - p_home_win))
    else:
        logloss = 0.0
    return brier, logloss, abs(mu_margin - actual_margin)

def run_scenario(season: str, scenario: str, params: dict) -> dict:
    """Run one scenario for one season, return metrics."""
    ratings = load_projected_ratings(season)
    games = load_actual_games(season)
    # For each game, compute predicted p_home_win using scenario params
    # Aggregate metrics
    return {...}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", required=True)
    parser.add_argument("--scenarios", nargs="+", default=["baseline", "full"])
    args = parser.parse_args()
    
    results = {}
    for scenario in args.scenarios:
        results[scenario] = {}
        for season in args.seasons:
            results[scenario][season] = run_scenario(season, scenario, get_scenario_params(scenario))
        results[scenario]["aggregate"] = aggregate_across_seasons(results[scenario])
    
    Path("reports/forecast_game_validation.json").write_text(json.dumps(results, indent=2))
    print_summary(results)

if __name__ == "__main__":
    main()
```

---

## Files Touched

| File | Change |
|---|---|
| `src/simulation/validate_forecast.py` | **New** — primary harness script |
| `src/simulation/forecast_scenarios.py` | **New** — scenario parameter definitions |
| `reports/forecast_game_validation.json` | **New** — output |

No existing files modified. The harness is purely read-only against existing pipeline outputs.

---

## Verification

1. Run `python3 src/simulation/validate_forecast.py --seasons 2023-24 2024-25 --scenarios baseline full`
2. Confirm output JSON has expected structure
3. Sanity check: baseline Brier should be roughly 0.22–0.25 (out-of-sample forecast quality);
   if it's ≥0.30 something is wrong with the pipeline
4. Sanity check: home_wr_predicted should be close to home_wr_actual (within ±0.05)
5. After each Phase A fix is wired: re-run harness, verify that "full" scenario Brier
   improves vs "baseline"

---

## Dependencies & Sequencing

**Can be developed:** Immediately, in parallel with all other plans
(it's purely read-only against forecast pipeline outputs that already exist).

**Becomes critical when:** Any other plan completes — without this harness, we can't
measure the impact of those changes.

**Should be operational before:** Plan 1 (pipeline audit) Track A completes, so we can
measure RAPM fix impact directly.

---

## What This Harness Does NOT Do

- Compare against Kalshi/sportsbook closing lines (that's prediction market work, Phase 6+)
- Compute player-level metrics (this is game-level only)
- Simulate seasons (uses actual schedule, not Monte Carlo)
- Replace `validate_sim.py` (in-sample backtest stays for within-season consistency checks;
  this is the out-of-sample forecast metric)
