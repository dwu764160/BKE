"""
src/simulation/game_model.py
=============================================================================
Simulation Core — Step 1, Layers 1-3

Layer 1: Parameter Layer — model constants and hyperparameters
Layer 2: Deterministic Game Model — expected margin + win probability
Layer 3: Schedule Engine — parse real NBA schedule from game logs

Inputs:
  data/processed/player_eval/team_feature_aggregation.parquet
  data/historical/team_game_logs.parquet

Output:
  reports/simulation_step1_results.json  (team parameters + schedule)

Usage:
  python3 src/simulation/game_model.py

  Can also be imported:
    from src.simulation.game_model import (
        SimConfig, TeamParams, Game,
        compute_game_distribution, load_team_params, build_schedule,
    )
=============================================================================
"""

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.simulation.simulation_config import (
    HISTORICAL_DIR,
    REPORTS_DIR,
    TEAM_FEATURES_PATH,
    SIGMA_LEAGUE,
    HOME_COURT_ADVANTAGE,
    SEASON_SIMULATIONS,
    SIMULATION_RANDOM_SEED,
)


# ═════════════════════════════════════════════════════════════════════
# Layer 1 — Parameter Layer
# ═════════════════════════════════════════════════════════════════════

@dataclass
class SimConfig:
    """Immutable simulation configuration."""
    sigma_league: float = SIGMA_LEAGUE
    home_court_advantage: float = HOME_COURT_ADVANTAGE
    n_simulations: int = SEASON_SIMULATIONS
    random_seed: int = SIMULATION_RANDOM_SEED


@dataclass
class TeamParams:
    """Per-team simulation parameters for one season."""
    team_abbreviation: str
    season: str
    mu: float        # projected net rating (points per 100 possessions)
    sigma: float     # projected volatility (std dev per 100 possessions)
    conference: str = "UNKNOWN"


@dataclass
class Game:
    """Single game in a season schedule."""
    game_id: str
    date: str
    home_team: str
    away_team: str
    season: str
    # Actual result (for validation; NaN if not available)
    home_margin: float = np.nan
    home_win: float = np.nan


# ═════════════════════════════════════════════════════════════════════
# Layer 2 — Deterministic Game Model
# ═════════════════════════════════════════════════════════════════════

def compute_game_distribution(
    team_a: TeamParams,
    team_b: TeamParams,
    is_home_a: bool = True,
    config: SimConfig = None,
) -> Dict[str, float]:
    """Compute expected margin, game variance, and win probability.

    Returns:
      delta_mu: expected margin (positive = favors team A)
      sigma_game: game-level standard deviation
      win_prob_a: P(team A wins)
      z_score: standardized expected margin
    """
    if config is None:
        config = SimConfig()

    h = config.home_court_advantage if is_home_a else -config.home_court_advantage

    delta_mu = (team_a.mu + h) - team_b.mu
    sigma_game = np.sqrt(
        team_a.sigma ** 2 + team_b.sigma ** 2 + config.sigma_league ** 2
    )

    z = delta_mu / sigma_game if sigma_game > 0 else 0.0
    win_prob_a = float(norm.cdf(z))

    return {
        "delta_mu": float(delta_mu),
        "sigma_game": float(sigma_game),
        "win_prob_a": float(win_prob_a),
        "z_score": float(z),
    }


# ═════════════════════════════════════════════════════════════════════
# Layer 3 — Schedule Engine
# ═════════════════════════════════════════════════════════════════════

def build_schedule(season: str, game_logs_path: Path = None) -> List[Game]:
    """Build season schedule from team game logs.

    Each real game appears once (home team's perspective).
    The MATCHUP field uses 'vs.' for home and '@' for away.
    """
    if game_logs_path is None:
        game_logs_path = HISTORICAL_DIR / "team_game_logs.parquet"

    gl = pd.read_parquet(game_logs_path)
    gl = gl[gl["SEASON"] == season].copy()
    gl["TEAM_ABBREVIATION"] = gl["TEAM_ABBREVIATION"].astype(str).str.upper()

    # Keep only home games (MATCHUP contains 'vs.')
    home_games = gl[gl["MATCHUP"].str.contains("vs.", na=False)].copy()

    games = []
    for _, row in home_games.iterrows():
        matchup = str(row["MATCHUP"])
        parts = matchup.split(" vs. ")
        home = parts[0].strip().upper()
        away = parts[1].strip().upper() if len(parts) > 1 else ""

        pts = float(row.get("PTS", np.nan))
        opp_pts = float(row.get("OPP_PTS", np.nan))
        margin = pts - opp_pts if not (np.isnan(pts) or np.isnan(opp_pts)) else np.nan

        games.append(Game(
            game_id=str(row.get("GAME_ID", "")),
            date=str(row.get("GAME_DATE", "")),
            home_team=home,
            away_team=away,
            season=season,
            home_margin=margin,
            home_win=1.0 if margin > 0 else (0.0 if margin < 0 else 0.5),
        ))

    return sorted(games, key=lambda g: g.date)


def load_team_params(season: str = None) -> Dict[str, Dict[str, TeamParams]]:
    """Load team parameters from Step 3 team feature aggregation.

    Returns:
      dict of {season: {team_abbr: TeamParams}}
    """
    tf = pd.read_parquet(TEAM_FEATURES_PATH)
    tf["season"] = tf["season"].astype(str)
    tf["team_abbreviation"] = tf["team_abbreviation"].astype(str).str.upper()

    # Build team abbreviation -> conference mapping when metadata exists.
    abbr_to_conf = {}
    teams_path = HISTORICAL_DIR / "teams.parquet"
    if teams_path.exists():
        teams = pd.read_parquet(teams_path)
        teams.columns = [str(c).lower() for c in teams.columns]
        if "abbreviation" in teams.columns and "conference" in teams.columns:
            tmp = teams[["abbreviation", "conference"]].copy()
            tmp["abbreviation"] = tmp["abbreviation"].astype(str).str.upper()
            tmp["conference"] = tmp["conference"].astype(str).str.title()
            abbr_to_conf = dict(tmp.itertuples(index=False, name=None))

    if season is not None:
        tf = tf[tf["season"] == season]

    result = {}
    for _, row in tf.iterrows():
        s = row["season"]
        team = row["team_abbreviation"]
        if team in ("NAN", "nan", "", "NONE"):
            continue
        if s not in result:
            result[s] = {}
        result[s][team] = TeamParams(
            team_abbreviation=team,
            season=s,
            mu=float(row["team_net_rating_projected"]),
            sigma=float(row["vol_total"]),
            conference=abbr_to_conf.get(team, "Unknown"),
        )
    return result


# ═════════════════════════════════════════════════════════════════════
# Main — build and export
# ═════════════════════════════════════════════════════════════════════

def main() -> None:
    print("Simulation Core — Step 1: Game Model (Layers 1-3)")
    config = SimConfig()
    print(f"  Config: sigma_league={config.sigma_league}, HCA={config.home_court_advantage}, "
          f"N_sim={config.n_simulations}, seed={config.random_seed}")

    # Load team parameters
    all_params = load_team_params()
    print(f"  Loaded team parameters for {len(all_params)} seasons")

    results = {
        "config": asdict(config),
        "seasons": {},
    }

    for season in sorted(all_params.keys()):
        params = all_params[season]
        schedule = build_schedule(season)
        print(f"  {season}: {len(params)} teams, {len(schedule)} home games")

        # Compute game-level predictions for all games
        game_predictions = []
        for game in schedule:
            if game.home_team in params and game.away_team in params:
                dist = compute_game_distribution(
                    params[game.home_team],
                    params[game.away_team],
                    is_home_a=True,
                    config=config,
                )
                game_predictions.append({
                    "game_id": game.game_id,
                    "date": game.date,
                    "home": game.home_team,
                    "away": game.away_team,
                    "delta_mu": round(dist["delta_mu"], 4),
                    "sigma_game": round(dist["sigma_game"], 4),
                    "win_prob_home": round(dist["win_prob_a"], 4),
                    "actual_margin": round(game.home_margin, 1) if not np.isnan(game.home_margin) else None,
                    "actual_home_win": game.home_win if not np.isnan(game.home_win) else None,
                })

        results["seasons"][season] = {
            "n_teams": len(params),
            "n_games": len(schedule),
            "n_predicted": len(game_predictions),
            "team_params": {
                t: {
                    "mu": round(p.mu, 4),
                    "sigma": round(p.sigma, 4),
                    "conference": p.conference,
                }
                for t, p in sorted(params.items())
            },
            "sample_predictions": game_predictions[:10],
        }

    # Save
    out_path = REPORTS_DIR / "simulation_step1_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
