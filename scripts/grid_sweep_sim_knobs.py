"""
scripts/grid_sweep_sim_knobs.py
================================================================
Focused grid sweep over the three most-sensitive simulation knobs:
  - LINEUP_TEAM_BONUS_CLIP
  - MATCHUP_TEAM_INTERACTION_SCALE (also applies MATCHUP_SPREAD_BONUS_CLIP)
  - PPP_CONTEXT_BONUS_SCALE

For each combination the script builds the lineup bonus map and matchup map
once with the trial's knob values, then runs a lightweight (1 000 sims) backtest
across all available seasons.

Usage:
    python3 scripts/grid_sweep_sim_knobs.py
================================================================
"""
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Patch simulation config BEFORE importing consumers
import src.simulation.simulation_config as sim_cfg
import src.simulation.player_stats_sim as ps_sim
import src.simulation.season_sim as ss

from src.simulation.game_model import Game, SimConfig, build_schedule, load_team_params
from src.simulation.season_sim import (
    _build_margin_game_arrays,
    _build_ppp_game_arrays,
    _estimate_impact_to_net_scale,
    _load_team_ppp_components,
    _build_predicted_pace_map,
    _simulate_from_game_arrays,
    _valid_games,
    aggregate_results,
    get_actual_records,
    _attach_actuals_and_stats,
)
from src.simulation.lineup_projection import build_projected_lineup_rows
from src.simulation.player_stats_sim import (
    build_lineup_bonus_map,
    build_pairwise_matchup_map,
    load_player_stat_profiles,
)

# ── Grid ──────────────────────────────────────────────────────────
CLIP_VALUES       = [0.30, 0.40, 0.55, 0.70]
MATCHUP_VALUES    = [0.15, 0.25, 0.35, 0.50]
PPP_CTX_VALUES    = [0.0, 0.0015, 0.0025, 0.005]

N_SIMS = 1_000
SEED   = 42


def _set_knobs(clip_v, matchup_v, ppp_v):
    """Patch knobs in both the config module AND all consumer modules."""
    # Config
    sim_cfg.LINEUP_TEAM_BONUS_CLIP = clip_v
    sim_cfg.MATCHUP_TEAM_INTERACTION_SCALE = matchup_v
    sim_cfg.MATCHUP_SPREAD_BONUS_CLIP = matchup_v
    sim_cfg.PPP_CONTEXT_BONUS_SCALE = ppp_v
    # player_stats_sim (imported early, cached at module level)
    ps_sim.LINEUP_TEAM_BONUS_CLIP = clip_v
    ps_sim.MATCHUP_TEAM_INTERACTION_SCALE = matchup_v
    ps_sim.MATCHUP_SPREAD_BONUS_CLIP = matchup_v
    # season_sim
    ss.PPP_CONTEXT_BONUS_SCALE = ppp_v


def main():
    print("Grid sweep: simulation knobs calibration")
    print(f"  CLIP values:    {CLIP_VALUES}")
    print(f"  MATCHUP values: {MATCHUP_VALUES}")
    print(f"  PPP_CTX values: {PPP_CTX_VALUES}")
    total = len(CLIP_VALUES) * len(MATCHUP_VALUES) * len(PPP_CTX_VALUES)
    print(f"  Total trials:   {total}")

    # Load heavy artifacts once with default knobs
    all_params = load_team_params()
    ppp_comps = _load_team_ppp_components(forecast_mode=False)
    pace_all = _build_predicted_pace_map(all_params)
    scale = _estimate_impact_to_net_scale(all_params, ppp_comps)
    print(f"  Impact scale:   {scale:.4f}")

    # Build lineup rows ONCE (Step 2 doesn't depend on the sweep knobs)
    profile_path = sim_cfg.PLAYER_PROFILES_PATH
    lineup_rows, _, _, _, _ = build_projected_lineup_rows(
        forecast_mode=False, profiles_path=profile_path,
    )
    rosters = load_player_stat_profiles(profile_path, forecast_mode=False)
    print(f"  Lineup rows:    {len(lineup_rows)}")
    print(f"  Rosters:        {len(rosters)}")

    # Pre-build schedules per season
    schedules_by_season = {}
    actuals_by_season = {}
    for season in sorted(all_params.keys()):
        schedules_by_season[season] = build_schedule(season)
        actuals_by_season[season] = get_actual_records(season)

    config = SimConfig(n_simulations=N_SIMS, random_seed=SEED)
    print()

    result_rows = []
    trial_idx = 0
    t0 = time.time()

    for clip_v, matchup_v, ppp_v in itertools.product(CLIP_VALUES, MATCHUP_VALUES, PPP_CTX_VALUES):
        trial_idx += 1
        _set_knobs(clip_v, matchup_v, ppp_v)

        # Rebuild bonus and matchup maps with current knob values
        lineup_bonus_all = build_lineup_bonus_map(lineup_rows)
        matchup_bonus_all = build_pairwise_matchup_map(rosters)

        for season in sorted(all_params.keys()):
            params = all_params[season]
            schedule = schedules_by_season[season]
            games = _valid_games(schedule, params)
            all_teams = sorted(params.keys())
            actual = actuals_by_season[season]
            season_lineup   = lineup_bonus_all.get(season, {})
            season_matchup  = matchup_bonus_all.get(season, {})
            season_ppp_comp = ppp_comps.get(season, {})
            season_pace     = pace_all.get(season, {})

            for mi, model_key in enumerate(["margin", "ppp"]):
                if model_key == "ppp":
                    dm, sg, ht, at = _build_ppp_game_arrays(
                        games, params, config, season_ppp_comp, season_pace, scale,
                        season_lineup, season_matchup,
                    )
                else:
                    dm, sg, ht, at = _build_margin_game_arrays(
                        games, params, config, season_lineup, season_matchup,
                    )
                wdists = _simulate_from_game_arrays(dm, sg, ht, at, all_teams, N_SIMS, SEED + mi)
                summaries = aggregate_results(wdists, params, model_key,
                                              season_ppp_comp, season_pace, scale)
                stats = _attach_actuals_and_stats(summaries, actual)
                result_rows.append({
                    "clip": clip_v,
                    "matchup": matchup_v,
                    "ppp_ctx": ppp_v,
                    "season": season,
                    "model": model_key,
                    "mae": stats.get("mae"),
                    "rmse": stats.get("rmse"),
                    "r": stats.get("correlation"),
                })

        margin_rows = [r for r in result_rows[-6:] if r["model"] == "margin"]
        avg_mae = np.mean([r["mae"] for r in margin_rows if r["mae"] is not None])
        avg_r   = np.mean([r["r"] for r in margin_rows if r["r"] is not None])
        print(f"  [{trial_idx:3d}/{total}] clip={clip_v:.2f} match={matchup_v:.2f} ppp={ppp_v:.4f}"
              f"  margin avgMAE={avg_mae:.2f} avgR={avg_r:.4f}")

    elapsed = time.time() - t0
    print(f"\nCompleted {total} trials in {elapsed:.1f}s")

    # Build summary DataFrame
    df = pd.DataFrame(result_rows)
    report_path = ROOT / "reports" / "grid_sweep_sim_knobs.json"
    df.to_json(report_path, orient="records", indent=2)
    print(f"Saved: {report_path}")

    # Print top-10 combos by average margin MAE
    margin_df = df[df["model"] == "margin"].copy()
    agg = (
        margin_df.groupby(["clip", "matchup", "ppp_ctx"])
        .agg(avg_mae=("mae", "mean"), avg_rmse=("rmse", "mean"), avg_r=("r", "mean"))
        .reset_index()
        .sort_values("avg_mae")
    )
    print("\n=== Top-10 combos by average margin MAE ===")
    print(agg.head(10).to_string(index=False))

    ppp_df = df[df["model"] == "ppp"].copy()
    agg_ppp = (
        ppp_df.groupby(["clip", "matchup", "ppp_ctx"])
        .agg(avg_mae=("mae", "mean"), avg_rmse=("rmse", "mean"), avg_r=("r", "mean"))
        .reset_index()
        .sort_values("avg_mae")
    )
    print("\n=== Top-10 combos by average PPP MAE ===")
    print(agg_ppp.head(10).to_string(index=False))

    # Best combined (sum of margin and PPP avg MAE)
    combined = agg.merge(agg_ppp, on=["clip", "matchup", "ppp_ctx"], suffixes=("_margin", "_ppp"))
    combined["combined_mae"] = combined["avg_mae_margin"] + combined["avg_mae_ppp"]
    combined = combined.sort_values("combined_mae")
    print("\n=== Top-10 combos by combined (margin+PPP) MAE ===")
    print(combined[["clip", "matchup", "ppp_ctx", "avg_mae_margin", "avg_mae_ppp", "combined_mae", "avg_r_margin", "avg_r_ppp"]].head(10).to_string(index=False))

    # Restore defaults
    _set_knobs(0.55, 0.35, 0.0025)


if __name__ == "__main__":
    main()
