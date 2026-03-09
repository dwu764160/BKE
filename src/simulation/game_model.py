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
from typing import Dict, List, Set, Tuple

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


def load_team_params(
    season: str = None,
    features_path: Path = None,
) -> Dict[str, Dict[str, TeamParams]]:
    """Load team parameters from team feature aggregation.

    Args:
        season: Optional season filter.
        features_path: Override path for team features parquet.
            Defaults to TEAM_FEATURES_PATH (backtest) or can be set to
            FORECAST_TEAM_FEATURES_PATH for forecast mode.

    Returns:
      dict of {season: {team_abbr: TeamParams}}
    """
    src = features_path or TEAM_FEATURES_PATH
    tf = pd.read_parquet(src)
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


def _build_extra_pairings(
    teams: List[str],
    extra_games_per_team: int,
    rng: np.random.RandomState,
    max_attempts: int = 400,
) -> Set[frozenset]:
    """Construct simple undirected extra pairings with fixed per-team degree."""
    if extra_games_per_team == 0:
        return set()

    if extra_games_per_team < 0 or extra_games_per_team >= len(teams):
        raise ValueError(
            f"Invalid extra_games_per_team={extra_games_per_team} for n_teams={len(teams)}"
        )

    for _ in range(max_attempts):
        remaining = {team: int(extra_games_per_team) for team in teams}
        edges: Set[frozenset] = set()
        ok = True

        while True:
            active = [t for t, d in remaining.items() if d > 0]
            if not active:
                break

            # Highest unmet degree first, randomized tie-break.
            active.sort(key=lambda t: (-remaining[t], rng.rand()))
            team = active[0]
            need = remaining[team]

            candidates = [
                opp
                for opp in active[1:]
                if remaining[opp] > 0 and frozenset((team, opp)) not in edges
            ]
            if len(candidates) < need:
                ok = False
                break

            candidates.sort(key=lambda t: (-remaining[t], rng.rand()))
            opponents = candidates[:need]

            for opp in opponents:
                edges.add(frozenset((team, opp)))
                remaining[team] -= 1
                remaining[opp] -= 1
                if remaining[opp] < 0:
                    ok = False
                    break
            if not ok or remaining[team] != 0:
                ok = False
                break

        if ok and all(v == 0 for v in remaining.values()):
            return edges

    raise RuntimeError(
        "Unable to construct balanced extra pairings for synthetic forecast schedule"
    )


def _orient_extra_pairings(
    extra_edges: Set[frozenset],
    extra_home_need: Dict[str, int],
    rng: np.random.RandomState,
    max_attempts: int = 600,
) -> List[Tuple[str, str]]:
    """Assign home/away for extra edges to satisfy per-team home needs."""
    edge_list = [tuple(e) for e in extra_edges]
    for _ in range(max_attempts):
        need = {k: int(v) for k, v in extra_home_need.items()}
        order = edge_list.copy()
        rng.shuffle(order)
        oriented: List[Tuple[str, str]] = []
        feasible = True

        for a, b in order:
            na = need.get(a, 0)
            nb = need.get(b, 0)

            if na < 0 or nb < 0:
                feasible = False
                break
            if na == 0 and nb == 0:
                feasible = False
                break

            if na == 0:
                home, away = b, a
            elif nb == 0:
                home, away = a, b
            elif na > nb:
                home, away = a, b
            elif nb > na:
                home, away = b, a
            else:
                home, away = (a, b) if rng.rand() < 0.5 else (b, a)

            need[home] -= 1
            oriented.append((home, away))

        if feasible and all(v == 0 for v in need.values()):
            return oriented

    # Fallback: orient greedily by current need even if exact home targets are missed.
    need = {k: int(v) for k, v in extra_home_need.items()}
    oriented = []
    for a, b in edge_list:
        if need.get(a, 0) >= need.get(b, 0):
            home, away = a, b
        else:
            home, away = b, a
        need[home] = need.get(home, 0) - 1
        oriented.append((home, away))
    return oriented


def generate_balanced_schedule(
    season: str,
    teams: List[str],
    games_per_team: int = 82,
    seed: int = 42,
) -> List[Game]:
    """Generate a balanced synthetic schedule for forecast mode.

    Each pair of teams plays roughly equal home/away matchups.
    Total games = (n_teams * games_per_team) / 2.
    """
    rng = np.random.RandomState(seed)
    teams = sorted([str(t).upper() for t in teams if str(t).strip()])
    n = len(teams)
    if n < 2:
        return []

    # Base double round-robin gives each team 2*(n-1) games (home+away vs each opponent).
    base_games_per_team = 2 * (n - 1)
    if games_per_team < base_games_per_team:
        raise ValueError(
            f"games_per_team={games_per_team} is too small for n_teams={n}; "
            f"minimum is {base_games_per_team}"
        )

    extra_games_per_team = games_per_team - base_games_per_team
    if (n * extra_games_per_team) % 2 != 0:
        raise ValueError(
            f"n_teams*extra_games_per_team must be even; got {n}*{extra_games_per_team}"
        )

    # Base home counts are exactly n-1 for each team.
    target_home = {t: games_per_team // 2 for t in teams}
    if games_per_team % 2 == 1:
        # Distribute one extra home game to half the teams for odd schedules.
        bump = teams.copy()
        rng.shuffle(bump)
        for t in bump[: n // 2]:
            target_home[t] += 1
    extra_home_need = {t: target_home[t] - (n - 1) for t in teams}

    extra_edges = _build_extra_pairings(teams, extra_games_per_team, rng)
    oriented_extra = _orient_extra_pairings(extra_edges, extra_home_need, rng)

    games: List[Game] = []
    game_num = 0

    # Base schedule: exactly 2 games per pair, one at each venue.
    for i in range(n):
        for j in range(i + 1, n):
            home, away = teams[i], teams[j]
            games.append(
                Game(
                    game_id=f"FORECAST_{season}_{game_num:05d}",
                    date=f"{season[:4]}-10-01",
                    home_team=home,
                    away_team=away,
                    season=season,
                )
            )
            game_num += 1
            games.append(
                Game(
                    game_id=f"FORECAST_{season}_{game_num:05d}",
                    date=f"{season[:4]}-10-02",
                    home_team=away,
                    away_team=home,
                    season=season,
                )
            )
            game_num += 1

    # Extra schedule: one additional game for selected pairs.
    for home, away in oriented_extra:
        games.append(
            Game(
                game_id=f"FORECAST_{season}_{game_num:05d}",
                date=f"{season[:4]}-10-03",
                home_team=home,
                away_team=away,
                season=season,
            )
        )
        game_num += 1

    return games


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
