"""
src/simulation/season_sim.py
=============================================================================
Simulation Core — Step 1, Layers 4-5

Layer 4: Monte Carlo Simulation Engine — simulate full seasons
Layer 5: Aggregation Layer — projected wins, playoff probability, distributions

Uses team net ratings and volatility from PEC Step 3 to simulate NBA seasons.
Each season is simulated N times (default 10,000). For each simulation:
  - Every game margin is drawn from N(delta_mu, sigma_game)
  - If margin > 0 → home team wins
  - Standings are updated

Inputs:
  data/processed/player_eval/team_feature_aggregation.parquet
  data/historical/team_game_logs.parquet

Output:
  reports/simulation_step1_season_results.json

Usage:
  python3 src/simulation/season_sim.py
=============================================================================
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.simulation.game_model import (
    Game,
    SimConfig,
    TeamParams,
    build_schedule,
    load_team_params,
)
from src.simulation.simulation_config import (
    DIRECT_PLAYOFF_RANK,
    HISTORICAL_DIR,
    PLAY_IN_RANK,
    REPORTS_DIR,
)


# ═════════════════════════════════════════════════════════════════════
# Layer 4 — Monte Carlo Simulation Engine
# ═════════════════════════════════════════════════════════════════════

def simulate_season(
    schedule: List[Game],
    team_params: Dict[str, TeamParams],
    config: SimConfig,
) -> Dict[str, np.ndarray]:
    """Simulate a full season N times using vectorized sampling.

    Returns:
      dict of {team_abbr: np.array of win counts, shape (N,)}
    """
    rng = np.random.default_rng(config.random_seed)

    # Filter to games where both teams have parameters
    valid_games = [
        g for g in schedule
        if g.home_team in team_params and g.away_team in team_params
    ]
    n_games = len(valid_games)
    n_sims = config.n_simulations

    # Precompute delta_mu and sigma_game for ALL games (vectorized)
    delta_mus = np.zeros(n_games)
    sigma_games = np.zeros(n_games)
    home_teams = []
    away_teams = []

    for i, game in enumerate(valid_games):
        home_p = team_params[game.home_team]
        away_p = team_params[game.away_team]
        delta_mus[i] = (home_p.mu + config.home_court_advantage) - away_p.mu
        sigma_games[i] = np.sqrt(
            home_p.sigma ** 2 + away_p.sigma ** 2 + config.sigma_league ** 2
        )
        home_teams.append(game.home_team)
        away_teams.append(game.away_team)

    # Sample all margins at once: shape (n_games, n_sims)
    margins = rng.normal(
        loc=delta_mus[:, np.newaxis],
        scale=sigma_games[:, np.newaxis],
        size=(n_games, n_sims),
    )

    # Home team wins when margin > 0
    home_wins = (margins > 0).astype(np.int32)

    # Accumulate wins per team
    all_teams = sorted(team_params.keys())
    team_idx = {t: i for i, t in enumerate(all_teams)}
    win_matrix = np.zeros((len(all_teams), n_sims), dtype=np.int32)

    for i in range(n_games):
        h_idx = team_idx[home_teams[i]]
        a_idx = team_idx[away_teams[i]]
        win_matrix[h_idx] += home_wins[i]
        win_matrix[a_idx] += (1 - home_wins[i])

    return {team: win_matrix[team_idx[team]] for team in all_teams}


# ═════════════════════════════════════════════════════════════════════
# Layer 5 — Aggregation Layer
# ═════════════════════════════════════════════════════════════════════

def aggregate_results(
    win_distributions: Dict[str, np.ndarray],
    team_params: Dict[str, TeamParams],
) -> List[Dict]:
    """Aggregate simulation results into per-team summaries.
    """

    def _round_or_none(value: float, decimals: int = 4):
        if value is None or not np.isfinite(value):
            return None
        return round(float(value), decimals)

    def _rank_probability_metrics() -> Dict[str, Dict[str, float]]:
        """Compute conference rank probabilities from simulated wins."""
        by_conf = {}
        for team, params in team_params.items():
            conf = str(getattr(params, "conference", "Unknown") or "Unknown").title()
            by_conf.setdefault(conf, []).append(team)

        metrics = {}
        for conf, teams in by_conf.items():
            conf_teams = sorted([t for t in teams if t in win_distributions])
            if not conf_teams:
                continue

            wins_matrix = np.vstack([win_distributions[t] for t in conf_teams]).astype(float)
            n_teams, n_sims = wins_matrix.shape

            # Deterministic tiny jitter avoids ambiguous tie ordering in rank cutoffs.
            conf_seed = 1_000 + sum(ord(ch) for ch in conf)
            jitter_rng = np.random.default_rng(conf_seed)
            wins_jittered = wins_matrix + jitter_rng.uniform(0.0, 1e-6, size=wins_matrix.shape)

            order = np.argsort(-wins_jittered, axis=0)
            ranks = np.empty_like(order, dtype=np.int32)
            sim_idx = np.arange(n_sims)
            ranks[order, sim_idx] = np.arange(1, n_teams + 1, dtype=np.int32)[:, np.newaxis]

            play_in_rank = min(PLAY_IN_RANK, n_teams)
            for idx, team in enumerate(conf_teams):
                team_ranks = ranks[idx]
                direct_prob = float(np.mean(team_ranks <= DIRECT_PLAYOFF_RANK))
                top10_prob = float(np.mean(team_ranks <= play_in_rank))
                playin_only_prob = float(
                    np.mean((team_ranks > DIRECT_PLAYOFF_RANK) & (team_ranks <= play_in_rank))
                )
                metrics[team] = {
                    "conference": conf,
                    "expected_conference_rank": float(np.mean(team_ranks)),
                    "direct_playoff_probability": direct_prob,
                    "top_10_probability": top10_prob,
                    "playin_only_probability": playin_only_prob,
                }
        return metrics

    rank_metrics = _rank_probability_metrics()

    summaries = []
    for team, wins in sorted(win_distributions.items()):
        params = team_params[team]
        r = rank_metrics.get(team, {})
        mean_wins = float(np.mean(wins))
        std_wins = float(np.std(wins, ddof=1))
        summaries.append({
            "team": team,
            "conference": str(getattr(params, "conference", "Unknown") or "Unknown").title(),
            "mu": round(params.mu, 4),
            "sigma": round(params.sigma, 4),
            "projected_wins": round(mean_wins, 1),
            "win_std": round(std_wins, 2),
            "win_p5": int(np.percentile(wins, 5)),
            "win_p25": int(np.percentile(wins, 25)),
            "win_median": int(np.median(wins)),
            "win_p75": int(np.percentile(wins, 75)),
            "win_p95": int(np.percentile(wins, 95)),
            "win_min": int(np.min(wins)),
            "win_max": int(np.max(wins)),
            "projected_conf_rank": _round_or_none(r.get("expected_conference_rank"), decimals=2),
            "direct_playoff_probability": _round_or_none(r.get("direct_playoff_probability")),
            "top_10_probability": _round_or_none(r.get("top_10_probability")),
            "playin_only_probability": _round_or_none(r.get("playin_only_probability")),
            # Backward-compatible alias for existing consumers.
            "playoff_probability": _round_or_none(r.get("direct_playoff_probability")),
            "win_50_plus_prob": round(float(np.mean(wins >= 50)), 4),
            "win_60_plus_prob": round(float(np.mean(wins >= 60)), 4),
        })

    summaries.sort(key=lambda x: x["projected_wins"], reverse=True)
    for i, s in enumerate(summaries):
        s["projected_rank"] = i + 1

    return summaries


def get_actual_records(season: str) -> Dict[str, Dict]:
    """Get actual W-L records from team game logs."""
    gl_path = HISTORICAL_DIR / "team_game_logs.parquet"
    if not gl_path.exists():
        return {}

    gl = pd.read_parquet(gl_path)
    gl = gl[gl["SEASON"] == season].copy()
    gl["TEAM_ABBREVIATION"] = gl["TEAM_ABBREVIATION"].astype(str).str.upper()

    records = {}
    for team, group in gl.groupby("TEAM_ABBREVIATION"):
        wins = (group["WL"] == "W").sum()
        losses = (group["WL"] == "L").sum()
        records[team] = {
            "actual_wins": int(wins),
            "actual_losses": int(losses),
            "actual_games": int(wins + losses),
        }
    return records


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════

def main() -> None:
    print("Simulation Core — Step 1: Season Simulation (Layers 4-5)")
    config = SimConfig()

    all_params = load_team_params()
    full_results = {"config": {
        "sigma_league": config.sigma_league,
        "home_court_advantage": config.home_court_advantage,
        "n_simulations": config.n_simulations,
        "random_seed": config.random_seed,
        "direct_playoff_rank": DIRECT_PLAYOFF_RANK,
        "play_in_rank": PLAY_IN_RANK,
    }, "seasons": {}}

    for season in sorted(all_params.keys()):
        print(f"\n{'='*60}")
        print(f"  Season: {season}")
        params = all_params[season]
        schedule = build_schedule(season)
        print(f"  Teams: {len(params)}, Games: {len(schedule)}")

        t0 = time.time()
        win_dists = simulate_season(schedule, params, config)
        elapsed = time.time() - t0
        print(f"  Simulated {config.n_simulations:,} seasons in {elapsed:.2f}s")

        summaries = aggregate_results(win_dists, params)
        actual = get_actual_records(season)

        # Merge actual records
        for s in summaries:
            if s["team"] in actual:
                s.update(actual[s["team"]])
                s["win_error"] = round(s["projected_wins"] - actual[s["team"]]["actual_wins"], 1)

        # Season-level stats
        teams_with_actual = [s for s in summaries if "actual_wins" in s]
        season_stats = {}
        if teams_with_actual:
            errors = [s["win_error"] for s in teams_with_actual]
            pred = [s["projected_wins"] for s in teams_with_actual]
            act = [s["actual_wins"] for s in teams_with_actual]
            season_stats = {
                "mae": round(float(np.mean(np.abs(errors))), 2),
                "rmse": round(float(np.sqrt(np.mean(np.array(errors) ** 2))), 2),
                "correlation": round(float(np.corrcoef(pred, act)[0, 1]), 4),
                "mean_error": round(float(np.mean(errors)), 2),
            }
            print(f"  Season stats: MAE={season_stats['mae']}, "
                  f"RMSE={season_stats['rmse']}, r={season_stats['correlation']}")

        # Print top 5 and bottom 5
        print(f"\n  Top 5:")
        for s in summaries[:5]:
            act_str = f" (actual: {s.get('actual_wins', '?')})" if "actual_wins" in s else ""
            print(f"    {s['projected_rank']:2d}. {s['team']} — {s['projected_wins']:.1f} wins "
                  f"[{s['win_p5']}-{s['win_p95']}]{act_str}")
        print(f"  Bottom 5:")
        for s in summaries[-5:]:
            act_str = f" (actual: {s.get('actual_wins', '?')})" if "actual_wins" in s else ""
            print(f"    {s['projected_rank']:2d}. {s['team']} — {s['projected_wins']:.1f} wins "
                  f"[{s['win_p5']}-{s['win_p95']}]{act_str}")

        full_results["seasons"][season] = {
            "season_stats": season_stats,
            "team_results": summaries,
        }

    # Save
    out_path = REPORTS_DIR / "simulation_step1_season_results.json"
    out_path.write_text(json.dumps(full_results, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
