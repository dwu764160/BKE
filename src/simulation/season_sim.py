"""
src/simulation/season_sim.py
=============================================================================
Simulation Core — Step 1, Layers 4-5

Layer 4: Monte Carlo Simulation Engine — simulate full seasons
Layer 5: Aggregation Layer — projected wins, playoff probability, distributions

Runs two models in parallel:
    - margin: existing net-rating margin model
    - ppp: possession-based model (team offense/defense PPP + pace)

Each season is simulated N times (default 10,000). For each simulation:
    - Every game margin is drawn from N(delta_mu, sigma_game)
    - If margin > 0 -> home team wins
    - Standings are updated

Inputs:
  data/processed/player_eval/team_feature_aggregation.parquet
    data/processed/player_eval/player_impact_profiles.parquet
    data/processed/forecast/projected_player_profiles.parquet
  data/historical/team_game_logs.parquet

Output:
    reports/simulation_step1_season_results.json (backtest)
    reports/forecast_season_results.json (forecast mode)

Usage:
  python3 src/simulation/season_sim.py
=============================================================================
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.schema_contract import load_standardized

from src.simulation.game_model import (
    Game,
    SimConfig,
    build_schedule,
    generate_balanced_schedule,
    load_team_params,
)
from src.simulation.lineup_projection import build_projected_lineup_rows
from src.simulation.player_stats_sim import (
    build_lineup_bonus_map,
    build_pairwise_matchup_map,
    load_player_stat_profiles,
    simulate_detailed_season,
)
from src.simulation.simulation_config import (
    CLUTCH_MARGIN_TRIGGER,
    DEFAULT_PACE_PER_48,
    DIRECT_PLAYOFF_RANK,
    FORECAST_PLAYER_GAME_SAMPLES_PATH,
    FORECAST_PLAYER_SEASON_STATS_PATH,
    FORECAST_SEASON_RESULTS_PATH,
    FORECAST_PLAYER_PROFILES_PATH,
    FORECAST_TEAM_FEATURES_PATH,
    HISTORICAL_DIR,
    LEAGUE_AVG_PPP,
    PACE_PRIOR_SEASON_WEIGHT,
    PACE_REGRESSION_WEIGHT,
    PLAYER_PROFILES_PATH,
    PPP_IMPACT_SCALE_PER100,
    PPP_MAX,
    PPP_MIN,
    PPP_OFF_DEF_BLEND_WEIGHT,
    PPP_CONTEXT_BONUS_SCALE,
    PPP_SIGMA_POSSESSION_EXPONENT,
    POSSESSION_FTA_WEIGHT,
    PLAY_IN_RANK,
    REPORTS_DIR,
    SIM_MODEL_MARGIN,
    SIM_MODEL_PPP,
    SIM_MODELS,
    STEP1_PLAYER_GAME_SAMPLES_PATH,
    STEP1_PLAYER_SEASON_STATS_PATH,
    STEP1_SINGLE_GAME_REPORT_PATH,
    FORECAST_SINGLE_GAME_REPORT_PATH,
)


# ═════════════════════════════════════════════════════════════════════
# Layer 4 — Monte Carlo Simulation Engine
# ═════════════════════════════════════════════════════════════════════

def _season_start_year(season: str) -> int:
    return int(str(season)[:4])


def _replacement_pool_mask(df: pd.DataFrame) -> pd.Series:
    """Identify synthetic replacement-pool rows across forecast schema versions."""
    if df.empty:
        return pd.Series(False, index=df.index, dtype=bool)

    if "player_id" in df.columns:
        pid = df["player_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    else:
        pid = pd.Series("", index=df.index, dtype=str)
    id_mask = pid.str.lower().str.startswith("repl_")

    if "player_name" in df.columns:
        names = df["player_name"].astype(str)
    else:
        names = pd.Series("", index=df.index, dtype=str)
    name_mask = names.str.contains("replacement pool", case=False, na=False)

    flag_mask = pd.Series(False, index=df.index, dtype=bool)
    if "is_replacement_pool" in df.columns:
        flag_mask = pd.to_numeric(df["is_replacement_pool"], errors="coerce").fillna(0).astype(int) == 1

    return id_mask | name_mask | flag_mask


def _previous_season(season: str) -> str:
    start_year = _season_start_year(season) - 1
    end_suffix = str(_season_start_year(season))[-2:]
    return f"{start_year}-{end_suffix}"


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _close_game_probability(delta_mu: float, sigma_game: float, trigger: float = CLUTCH_MARGIN_TRIGGER) -> float:
    sigma = max(float(sigma_game), 1e-6)
    z_hi = (trigger - float(delta_mu)) / sigma
    z_lo = (-trigger - float(delta_mu)) / sigma
    return float(np.clip(_normal_cdf(z_hi) - _normal_cdf(z_lo), 0.0, 1.0))


def _resolve_detailed_output_paths(
    forecast_mode: bool,
    output_path: Optional[Path],
) -> Tuple[Path, Path]:
    if forecast_mode and output_path is not None and output_path.stem.startswith("forecast_season_results_"):
        suffix = output_path.stem.replace("forecast_season_results_", "")
        return (
            FORECAST_PLAYER_SEASON_STATS_PATH.with_name(f"forecast_step1_player_season_stats_{suffix}.parquet"),
            FORECAST_PLAYER_GAME_SAMPLES_PATH.with_name(f"forecast_step1_player_game_samples_{suffix}.parquet"),
        )
    if forecast_mode:
        return FORECAST_PLAYER_SEASON_STATS_PATH, FORECAST_PLAYER_GAME_SAMPLES_PATH
    return STEP1_PLAYER_SEASON_STATS_PATH, STEP1_PLAYER_GAME_SAMPLES_PATH


def _build_simulation_contexts(
    forecast_mode: bool,
    player_profiles_path: Optional[Path],
) -> Tuple[List[Dict], Dict[str, Dict[str, Dict[str, float]]], pd.DataFrame, Dict[str, Dict[Tuple[str, str], Dict[str, float]]]]:
    profile_path = player_profiles_path or (FORECAST_PLAYER_PROFILES_PATH if forecast_mode else PLAYER_PROFILES_PATH)
    lineup_rows, _, _, _, _ = build_projected_lineup_rows(
        forecast_mode=forecast_mode,
        profiles_path=profile_path,
    )
    lineup_bonus_map = build_lineup_bonus_map(lineup_rows)
    rosters = load_player_stat_profiles(profile_path, forecast_mode=forecast_mode)
    matchup_map = build_pairwise_matchup_map(rosters)
    return lineup_rows, lineup_bonus_map, rosters, matchup_map


def _valid_games(schedule: List[Game], team_params: Dict) -> List[Game]:
    return [
        g for g in schedule
        if g.home_team in team_params and g.away_team in team_params
    ]


def _simulate_from_game_arrays(
    delta_mus: np.ndarray,
    sigma_games: np.ndarray,
    home_teams: List[str],
    away_teams: List[str],
    all_teams: List[str],
    n_sims: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Vectorized season simulation from per-game mean and sigma arrays."""
    rng = np.random.default_rng(seed)
    n_games = len(home_teams)
    if n_games == 0:
        return {t: np.zeros(n_sims, dtype=np.int32) for t in all_teams}

    margins = rng.normal(
        loc=delta_mus[:, np.newaxis],
        scale=sigma_games[:, np.newaxis],
        size=(n_games, n_sims),
    )
    home_wins = (margins > 0).astype(np.int32)

    team_idx = {t: i for i, t in enumerate(all_teams)}
    win_matrix = np.zeros((len(all_teams), n_sims), dtype=np.int32)
    for i in range(n_games):
        h_idx = team_idx[home_teams[i]]
        a_idx = team_idx[away_teams[i]]
        win_matrix[h_idx] += home_wins[i]
        win_matrix[a_idx] += (1 - home_wins[i])

    return {team: win_matrix[team_idx[team]] for team in all_teams}


def _build_margin_game_arrays(
    games: List[Game],
    team_params: Dict,
    config: SimConfig,
    lineup_bonus_map: Optional[Dict[str, Dict[str, float]]] = None,
    matchup_bonus_map: Optional[Dict[Tuple[str, str], Dict[str, float]]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    lineup_bonus_map = lineup_bonus_map or {}
    matchup_bonus_map = matchup_bonus_map or {}
    n_games = len(games)
    delta_mus = np.zeros(n_games)
    sigma_games = np.zeros(n_games)
    home_teams = []
    away_teams = []

    for i, game in enumerate(games):
        home_p = team_params[game.home_team]
        away_p = team_params[game.away_team]
        sigma_games[i] = np.sqrt(
            home_p.sigma ** 2 + away_p.sigma ** 2 + config.sigma_league ** 2
        )
        home_lineup = lineup_bonus_map.get(game.home_team, {})
        away_lineup = lineup_bonus_map.get(game.away_team, {})
        matchup = matchup_bonus_map.get((game.home_team, game.away_team), {})

        preclutch_delta = (
            (home_p.mu + config.home_court_advantage) - away_p.mu
            + float(home_lineup.get("team_bonus", 0.0))
            - float(away_lineup.get("team_bonus", 0.0))
            + float(matchup.get("spread_bonus", 0.0))
        )
        close_prob = _close_game_probability(preclutch_delta, sigma_games[i])
        clutch_delta = close_prob * (
            float(home_lineup.get("clutch_bonus", 0.0))
            - float(away_lineup.get("clutch_bonus", 0.0))
        )
        delta_mus[i] = preclutch_delta + clutch_delta
        home_teams.append(game.home_team)
        away_teams.append(game.away_team)

    return delta_mus, sigma_games, home_teams, away_teams


def _load_team_pace_history() -> pd.DataFrame:
    """Compute team pace (possessions per 48) from team game logs."""
    gl_path = HISTORICAL_DIR / "team_game_logs.parquet"
    if not gl_path.exists():
        return pd.DataFrame(columns=["season", "team_abbreviation", "pace_per_48"])

    gl = load_standardized(gl_path)
    gl["season"] = gl["season"].astype(str)
    gl["team_abbreviation"] = gl["team_abbreviation"].astype(str).str.upper()

    for col in ["fga", "oreb", "tov", "fta", "min"]:
        gl[col] = pd.to_numeric(gl[col], errors="coerce")

    poss = gl["fga"] - gl["oreb"] + gl["tov"] + POSSESSION_FTA_WEIGHT * gl["fta"]
    pace = poss * (240.0 / gl["min"].replace(0, np.nan))
    gl["pace_per_48"] = pace

    pace_df = (
        gl.groupby(["season", "team_abbreviation"], as_index=False)["pace_per_48"]
        .mean()
    )
    pace_df["season"] = pace_df["season"].astype(str)
    pace_df["team_abbreviation"] = pace_df["team_abbreviation"].astype(str).str.upper()
    return pace_df


def _build_predicted_pace_map(all_params: Dict[str, Dict]) -> Dict[str, Dict[str, float]]:
    """Predict team pace for each target season from prior season pace + regression-to-mean."""
    pace_hist = _load_team_pace_history()
    if pace_hist.empty:
        return {
            season: {team: DEFAULT_PACE_PER_48 for team in sorted(params.keys())}
            for season, params in all_params.items()
        }

    hist_map = {
        (str(r["season"]), str(r["team_abbreviation"]).upper()): float(r["pace_per_48"])
        for _, r in pace_hist.iterrows()
        if np.isfinite(r["pace_per_48"])
    }
    league_by_season = (
        pace_hist.groupby("season")["pace_per_48"].mean().astype(float).to_dict()
    )
    overall_league = float(pace_hist["pace_per_48"].mean()) if len(pace_hist) else DEFAULT_PACE_PER_48

    out: Dict[str, Dict[str, float]] = {}
    for season, params in all_params.items():
        prev = _previous_season(season)
        prev_league = float(league_by_season.get(prev, overall_league))
        out[season] = {}
        for team in sorted(params.keys()):
            prev_team_pace = hist_map.get((prev, team), np.nan)
            if np.isfinite(prev_team_pace):
                pred = (
                    PACE_PRIOR_SEASON_WEIGHT * float(prev_team_pace)
                    + PACE_REGRESSION_WEIGHT * prev_league
                )
            else:
                pred = prev_league
            out[season][team] = float(np.clip(pred, 90.0, 110.0))

    return out


def _load_team_ppp_components(
    forecast_mode: bool,
    player_profiles_path: Path = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Build team offense/defense components from the full impact profile dataset."""
    profile_path = player_profiles_path or (FORECAST_PLAYER_PROFILES_PATH if forecast_mode else PLAYER_PROFILES_PATH)
    if not profile_path.exists():
        return {}

    df = pd.read_parquet(profile_path)
    if df.empty:
        return {}

    repl_mask = _replacement_pool_mask(df)
    if int(repl_mask.sum()) > 0:
        df = df.loc[~repl_mask].copy()

    df["season"] = df["season"].astype(str)
    df["team_abbreviation"] = df["team_abbreviation"].astype(str).str.upper()
    df = df[df["team_abbreviation"].notna() & (df["team_abbreviation"] != "")].copy()

    # Use profile-native offensive/defensive impact columns with robust fallbacks.
    off_col = "impact_obke" if "impact_obke" in df.columns else "impact_bke"
    def_col = "impact_dbke" if "impact_dbke" in df.columns else "impact_bke"
    df["impact_off"] = pd.to_numeric(df.get(off_col), errors="coerce").fillna(0.0)
    df["impact_def"] = pd.to_numeric(df.get(def_col), errors="coerce").fillna(0.0)
    df["minutes"] = pd.to_numeric(df.get("minutes"), errors="coerce")
    mpg = pd.to_numeric(df.get("mpg"), errors="coerce").fillna(0.0)
    games = pd.to_numeric(df.get("games"), errors="coerce").fillna(72.0)
    fallback_minutes = mpg * games
    df["minutes_weight"] = df["minutes"].fillna(fallback_minutes).clip(lower=0.0)

    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for (season, team), g in df.groupby(["season", "team_abbreviation"]):
        weights = g["minutes_weight"].to_numpy(dtype=float)
        if not np.isfinite(weights).all() or weights.sum() <= 0:
            weights = np.ones(len(g), dtype=float)
        off = float(np.average(g["impact_off"].to_numpy(dtype=float), weights=weights))
        deff = float(np.average(g["impact_def"].to_numpy(dtype=float), weights=weights))
        out.setdefault(season, {})[team] = {
            "offense": off,
            "defense": deff,
        }
    return out


def _build_ppp_game_arrays(
    games: List[Game],
    team_params: Dict,
    config: SimConfig,
    season_ppp_components: Dict[str, Dict[str, float]],
    season_pace_map: Dict[str, float],
    impact_to_net_scale: float,
    lineup_bonus_map: Optional[Dict[str, Dict[str, float]]] = None,
    matchup_bonus_map: Optional[Dict[Tuple[str, str], Dict[str, float]]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    lineup_bonus_map = lineup_bonus_map or {}
    matchup_bonus_map = matchup_bonus_map or {}
    n_games = len(games)
    delta_mus = np.zeros(n_games)
    sigma_games = np.zeros(n_games)
    home_teams = []
    away_teams = []

    for i, game in enumerate(games):
        home_team = game.home_team
        away_team = game.away_team
        home_params = team_params[home_team]
        away_params = team_params[away_team]

        home_off = float(season_ppp_components.get(home_team, {}).get("offense", 0.0))
        home_def = float(season_ppp_components.get(home_team, {}).get("defense", 0.0))
        away_off = float(season_ppp_components.get(away_team, {}).get("offense", 0.0))
        away_def = float(season_ppp_components.get(away_team, {}).get("defense", 0.0))

        # impact_* fields are on per-100-style scale; convert to per-possession deltas.
        home_off_ppp = (home_off * impact_to_net_scale) / PPP_IMPACT_SCALE_PER100
        home_def_ppp = (home_def * impact_to_net_scale) / PPP_IMPACT_SCALE_PER100
        away_off_ppp = (away_off * impact_to_net_scale) / PPP_IMPACT_SCALE_PER100
        away_def_ppp = (away_def * impact_to_net_scale) / PPP_IMPACT_SCALE_PER100

        home_pace = float(season_pace_map.get(home_team, DEFAULT_PACE_PER_48))
        away_pace = float(season_pace_map.get(away_team, DEFAULT_PACE_PER_48))
        possessions = float(np.clip((home_pace + away_pace) / 2.0, 88.0, 112.0))

        home_lineup = lineup_bonus_map.get(home_team, {})
        away_lineup = lineup_bonus_map.get(away_team, {})
        matchup = matchup_bonus_map.get((home_team, away_team), {})

        expected_ppp_home = LEAGUE_AVG_PPP + PPP_OFF_DEF_BLEND_WEIGHT * (home_off_ppp - away_def_ppp)
        expected_ppp_away = LEAGUE_AVG_PPP + PPP_OFF_DEF_BLEND_WEIGHT * (away_off_ppp - home_def_ppp)
        expected_ppp_home += PPP_CONTEXT_BONUS_SCALE * (
            float(home_lineup.get("team_bonus", 0.0)) + float(matchup.get("home_offense_bonus", 0.0))
        )
        expected_ppp_away += PPP_CONTEXT_BONUS_SCALE * (
            float(away_lineup.get("team_bonus", 0.0)) + float(matchup.get("away_offense_bonus", 0.0))
        )
        expected_ppp_home = float(np.clip(expected_ppp_home, PPP_MIN, PPP_MAX))
        expected_ppp_away = float(np.clip(expected_ppp_away, PPP_MIN, PPP_MAX))

        preclutch_delta = (expected_ppp_home - expected_ppp_away) * possessions
        preclutch_delta += config.home_court_advantage
        preclutch_delta += float(matchup.get("spread_bonus", 0.0))
        close_prob = _close_game_probability(preclutch_delta, max(1.0, possessions ** 0.5))
        clutch_delta = close_prob * (
            float(home_lineup.get("clutch_bonus", 0.0)) - float(away_lineup.get("clutch_bonus", 0.0))
        )
        delta_margin = preclutch_delta + clutch_delta

        sigma_base = np.sqrt(
            home_params.sigma ** 2 + away_params.sigma ** 2 + config.sigma_league ** 2
        )
        sigma_margin = sigma_base * ((possessions / 100.0) ** PPP_SIGMA_POSSESSION_EXPONENT)

        delta_mus[i] = delta_margin
        sigma_games[i] = max(0.5, sigma_margin)
        home_teams.append(home_team)
        away_teams.append(away_team)

    return delta_mus, sigma_games, home_teams, away_teams


def _estimate_impact_to_net_scale(
    all_params: Dict[str, Dict],
    ppp_components_all: Dict[str, Dict[str, Dict[str, float]]],
) -> float:
    """Estimate linear mapping from profile impact units to net-rating units (per 100)."""
    xs = []
    ys = []
    for season, teams in all_params.items():
        season_components = ppp_components_all.get(season, {})
        for team, params in teams.items():
            comp = season_components.get(team)
            if not comp:
                continue
            raw_sum = float(comp.get("offense", 0.0)) + float(comp.get("defense", 0.0))
            if np.isfinite(raw_sum) and np.isfinite(params.mu):
                xs.append(raw_sum)
                ys.append(float(params.mu))

    if len(xs) < 10:
        return 12.0

    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    denom = float(np.dot(x, x))
    if denom <= 1e-9:
        return 12.0

    slope = float(np.dot(x, y) / denom)
    if not np.isfinite(slope) or slope <= 0:
        return 12.0
    return float(np.clip(slope, 1.0, 40.0))


def simulate_season(
    schedule: List[Game],
    team_params: Dict,
    config: SimConfig,
) -> Dict[str, np.ndarray]:
    """Backward-compatible margin simulation wrapper."""
    games = _valid_games(schedule, team_params)
    delta_mus, sigma_games, home_teams, away_teams = _build_margin_game_arrays(games, team_params, config)
    all_teams = sorted(team_params.keys())
    return _simulate_from_game_arrays(
        delta_mus=delta_mus,
        sigma_games=sigma_games,
        home_teams=home_teams,
        away_teams=away_teams,
        all_teams=all_teams,
        n_sims=config.n_simulations,
        seed=config.random_seed,
    )


def _sample_game_environment_rows(
    games: List[Game],
    team_params: Dict,
    config: SimConfig,
    season_ppp_components: Dict[str, Dict[str, float]],
    season_pace_map: Dict[str, float],
    impact_to_net_scale: float,
    lineup_bonus_map: Dict[str, Dict[str, float]],
    matchup_bonus_map: Dict[Tuple[str, str], Dict[str, float]],
    model_key: str,
    seed: int,
) -> List[Dict[str, float]]:
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, float]] = []

    for game in games:
        if game.home_team not in team_params or game.away_team not in team_params:
            continue

        home_team = game.home_team
        away_team = game.away_team
        home_params = team_params[home_team]
        away_params = team_params[away_team]
        home_lineup = lineup_bonus_map.get(home_team, {})
        away_lineup = lineup_bonus_map.get(away_team, {})
        matchup = matchup_bonus_map.get((home_team, away_team), {})

        home_off = float(season_ppp_components.get(home_team, {}).get("offense", 0.0))
        home_def = float(season_ppp_components.get(home_team, {}).get("defense", 0.0))
        away_off = float(season_ppp_components.get(away_team, {}).get("offense", 0.0))
        away_def = float(season_ppp_components.get(away_team, {}).get("defense", 0.0))

        home_off_ppp = (home_off * impact_to_net_scale) / PPP_IMPACT_SCALE_PER100
        home_def_ppp = (home_def * impact_to_net_scale) / PPP_IMPACT_SCALE_PER100
        away_off_ppp = (away_off * impact_to_net_scale) / PPP_IMPACT_SCALE_PER100
        away_def_ppp = (away_def * impact_to_net_scale) / PPP_IMPACT_SCALE_PER100

        home_pace = float(season_pace_map.get(home_team, DEFAULT_PACE_PER_48))
        away_pace = float(season_pace_map.get(away_team, DEFAULT_PACE_PER_48))
        possessions = float(np.clip((home_pace + away_pace) / 2.0, 88.0, 112.0))

        expected_ppp_home = LEAGUE_AVG_PPP + PPP_OFF_DEF_BLEND_WEIGHT * (home_off_ppp - away_def_ppp)
        expected_ppp_away = LEAGUE_AVG_PPP + PPP_OFF_DEF_BLEND_WEIGHT * (away_off_ppp - home_def_ppp)
        expected_ppp_home += PPP_CONTEXT_BONUS_SCALE * (
            float(home_lineup.get("team_bonus", 0.0)) + float(matchup.get("home_offense_bonus", 0.0))
        )
        expected_ppp_away += PPP_CONTEXT_BONUS_SCALE * (
            float(away_lineup.get("team_bonus", 0.0)) + float(matchup.get("away_offense_bonus", 0.0))
        )
        expected_ppp_home = float(np.clip(expected_ppp_home, PPP_MIN, PPP_MAX))
        expected_ppp_away = float(np.clip(expected_ppp_away, PPP_MIN, PPP_MAX))

        sigma_margin = float(np.sqrt(home_params.sigma ** 2 + away_params.sigma ** 2 + config.sigma_league ** 2))
        base_margin = (home_params.mu + config.home_court_advantage) - away_params.mu
        if model_key == SIM_MODEL_PPP:
            base_margin = (expected_ppp_home - expected_ppp_away) * possessions + config.home_court_advantage

        preclutch_margin = (
            base_margin
            + float(home_lineup.get("team_bonus", 0.0))
            - float(away_lineup.get("team_bonus", 0.0))
            + float(matchup.get("spread_bonus", 0.0))
        )
        sampled_margin = float(rng.normal(preclutch_margin, sigma_margin))
        close_game = abs(sampled_margin) <= CLUTCH_MARGIN_TRIGGER
        if close_game:
            sampled_margin += float(home_lineup.get("clutch_bonus", 0.0)) - float(away_lineup.get("clutch_bonus", 0.0))

        total_points_mean = possessions * (expected_ppp_home + expected_ppp_away)
        total_points = float(rng.normal(total_points_mean, 11.0))
        total_points = max(total_points, 168.0)
        home_score = int(round((total_points + sampled_margin) / 2.0))
        away_score = int(round(total_points - home_score))
        home_score = max(home_score, 80)
        away_score = max(away_score, 80)

        rows.append(
            {
                "game_id": str(game.game_id),
                "date": str(game.date),
                "season": str(game.season),
                "home_team": home_team,
                "away_team": away_team,
                "model": model_key,
                "margin_mean": round(float(preclutch_margin), 4),
                "preclutch_margin": round(float(preclutch_margin), 4),
                "sampled_margin": round(float(sampled_margin), 4),
                "possessions": round(possessions, 3),
                "home_score": int(home_score),
                "away_score": int(away_score),
            }
        )
    return rows


# ═════════════════════════════════════════════════════════════════════
# Layer 5 — Aggregation Layer
# ═════════════════════════════════════════════════════════════════════

def aggregate_results(
    win_distributions: Dict[str, np.ndarray],
    team_params: Dict,
    model_key: str = SIM_MODEL_MARGIN,
    ppp_components: Dict[str, Dict[str, float]] = None,
    pace_map: Dict[str, float] = None,
    impact_to_net_scale: float = 12.0,
) -> List[Dict]:
    """Aggregate simulation results into per-team summaries."""
    ppp_components = ppp_components or {}
    pace_map = pace_map or {}

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

        ppp_off = float(ppp_components.get(team, {}).get("offense", np.nan))
        ppp_def = float(ppp_components.get(team, {}).get("defense", np.nan))
        ppp_net = (
            PPP_OFF_DEF_BLEND_WEIGHT * impact_to_net_scale * (ppp_off + ppp_def)
            if np.isfinite(ppp_off) and np.isfinite(ppp_def)
            else np.nan
        )
        pace_pred = float(pace_map.get(team, np.nan))

        mu_display = float(params.mu)
        if model_key == SIM_MODEL_PPP and np.isfinite(ppp_net):
            mu_display = ppp_net

        summaries.append({
            "team": team,
            "conference": str(getattr(params, "conference", "Unknown") or "Unknown").title(),
            "mu": round(mu_display, 4),
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
            "model": model_key,
            "ppp_offense": round(ppp_off, 4) if np.isfinite(ppp_off) else None,
            "ppp_defense": round(ppp_def, 4) if np.isfinite(ppp_def) else None,
            "ppp_net_rating": round(ppp_net, 4) if np.isfinite(ppp_net) else None,
            "predicted_pace": round(pace_pred, 2) if np.isfinite(pace_pred) else None,
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

    gl = load_standardized(gl_path)
    gl = gl[gl["season"] == season].copy()
    gl["team_abbreviation"] = gl["team_abbreviation"].astype(str).str.upper()

    records = {}
    for team, group in gl.groupby("team_abbreviation"):
        wins = (group["wl"] == "W").sum()
        losses = (group["wl"] == "L").sum()
        records[team] = {
            "actual_wins": int(wins),
            "actual_losses": int(losses),
            "actual_games": int(wins + losses),
        }
    return records


def _attach_actuals_and_stats(summaries: List[Dict], actual: Dict[str, Dict]) -> Dict:
    """Attach actual records to team summaries and compute season-level errors."""
    if not actual:
        return {}

    for s in summaries:
        if s["team"] in actual:
            s.update(actual[s["team"]])
            s["win_error"] = round(s["projected_wins"] - actual[s["team"]]["actual_wins"], 1)

    teams_with_actual = [s for s in summaries if "actual_wins" in s]
    if not teams_with_actual:
        return {}

    errors = np.array([s["win_error"] for s in teams_with_actual], dtype=float)
    pred = np.array([s["projected_wins"] for s in teams_with_actual], dtype=float)
    act = np.array([s["actual_wins"] for s in teams_with_actual], dtype=float)
    corr = float(np.corrcoef(pred, act)[0, 1]) if len(pred) > 1 else np.nan
    return {
        "mae": round(float(np.mean(np.abs(errors))), 2),
        "rmse": round(float(np.sqrt(np.mean(errors ** 2))), 2),
        "correlation": round(corr, 4) if np.isfinite(corr) else None,
        "mean_error": round(float(np.mean(errors)), 2),
    }


def _model_alignment_diagnostics(
    margin_delta_mus: np.ndarray,
    ppp_delta_mus: np.ndarray,
) -> Dict[str, float]:
    """Quick sanity diagnostics comparing margin-model and PPP-model expected spreads."""
    if margin_delta_mus.size == 0 or ppp_delta_mus.size == 0:
        return {}

    diff = ppp_delta_mus - margin_delta_mus
    abs_diff = np.abs(diff)
    denom = np.maximum(np.abs(margin_delta_mus), 1e-6)
    rel_abs = abs_diff / denom
    corr = np.corrcoef(margin_delta_mus, ppp_delta_mus)[0, 1] if margin_delta_mus.size > 1 else np.nan

    return {
        "mean_abs_gap": round(float(abs_diff.mean()), 4),
        "median_abs_gap": round(float(np.median(abs_diff)), 4),
        "p90_abs_gap": round(float(np.percentile(abs_diff, 90)), 4),
        "mean_relative_abs_gap": round(float(rel_abs.mean()), 4),
        "delta_mu_correlation": round(float(corr), 4) if np.isfinite(corr) else None,
    }


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════

def main(
    forecast_mode: bool = False,
    features_path: Path = None,
    player_profiles_path: Path = None,
    output_path: Path = None,
    single_game_home: Optional[str] = None,
    single_game_away: Optional[str] = None,
    single_game_season: Optional[str] = None,
) -> None:
    mode_label = "FORECAST" if forecast_mode else "BACKTEST"
    print(f"Simulation Core — Step 1: Season Simulation (Layers 4-5) [{mode_label}]")
    config = SimConfig()

    src_path = features_path or (FORECAST_TEAM_FEATURES_PATH if forecast_mode else None)
    all_params = load_team_params(features_path=src_path)
    profile_path = player_profiles_path or (FORECAST_PLAYER_PROFILES_PATH if forecast_mode else PLAYER_PROFILES_PATH)

    ppp_components_all = _load_team_ppp_components(
        forecast_mode=forecast_mode,
        player_profiles_path=profile_path,
    )
    pace_pred_all = _build_predicted_pace_map(all_params)
    impact_to_net_scale = _estimate_impact_to_net_scale(all_params, ppp_components_all)
    lineup_rows, lineup_bonus_all, rosters, matchup_bonus_all = _build_simulation_contexts(
        forecast_mode=forecast_mode,
        player_profiles_path=profile_path,
    )
    print(f"  PPP impact->net scale: {impact_to_net_scale:.4f}")
    print(f"  Step 2 lineup contexts: {len(lineup_rows)} team-seasons")
    print(f"  Player stat profiles: {len(rosters)} rows")

    if single_game_home and single_game_away:
        season = single_game_season or (sorted(all_params.keys())[-1] if all_params else None)
        if not season or season not in all_params:
            raise ValueError(f"Single-game season not available: {season}")
        schedule = [
            Game(
                game_id=f"SINGLE_{season}_{single_game_home}_{single_game_away}",
                date=f"{season[:4]}-10-01",
                home_team=str(single_game_home).upper(),
                away_team=str(single_game_away).upper(),
                season=season,
            )
        ]
        env_rows = _sample_game_environment_rows(
            games=schedule,
            team_params=all_params[season],
            config=config,
            season_ppp_components=ppp_components_all.get(season, {}),
            season_pace_map=pace_pred_all.get(season, {}),
            impact_to_net_scale=impact_to_net_scale,
            lineup_bonus_map=lineup_bonus_all.get(season, {}),
            matchup_bonus_map=matchup_bonus_all.get(season, {}),
            model_key=SIM_MODEL_MARGIN,
            seed=config.random_seed + 777,
        )
        season_rosters = rosters[rosters["season"] == season].copy()
        season_df, game_df = simulate_detailed_season(
            schedule=schedule,
            rosters=season_rosters,
            lineup_bonus_map={season: lineup_bonus_all.get(season, {})},
            matchup_map={season: matchup_bonus_all.get(season, {})},
            game_environment_rows=env_rows,
            rng=np.random.default_rng(config.random_seed + 888),
            model_key=SIM_MODEL_MARGIN,
        )
        report_path = FORECAST_SINGLE_GAME_REPORT_PATH if forecast_mode else STEP1_SINGLE_GAME_REPORT_PATH
        report = {
            "season": season,
            "home_team": str(single_game_home).upper(),
            "away_team": str(single_game_away).upper(),
            "environment": env_rows[0] if env_rows else {},
            "player_game_rows": game_df.to_dict(orient="records"),
            "player_summary": season_df.to_dict(orient="records"),
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved single-game simulation: {report_path}")
        return

    full_results = {"config": {
        "sigma_league": config.sigma_league,
        "home_court_advantage": config.home_court_advantage,
        "n_simulations": config.n_simulations,
        "random_seed": config.random_seed,
        "direct_playoff_rank": DIRECT_PLAYOFF_RANK,
        "play_in_rank": PLAY_IN_RANK,
        "mode": mode_label,
        "models": list(SIM_MODELS),
        "league_avg_ppp": LEAGUE_AVG_PPP,
        "default_pace_per_48": DEFAULT_PACE_PER_48,
        "pace_prior_season_weight": PACE_PRIOR_SEASON_WEIGHT,
        "pace_regression_weight": PACE_REGRESSION_WEIGHT,
        "ppp_impact_scale_per100": PPP_IMPACT_SCALE_PER100,
        "ppp_off_def_blend_weight": PPP_OFF_DEF_BLEND_WEIGHT,
        "ppp_sigma_possession_exponent": PPP_SIGMA_POSSESSION_EXPONENT,
        "ppp_clip_min": PPP_MIN,
        "ppp_clip_max": PPP_MAX,
        "ppp_impact_to_net_scale": round(impact_to_net_scale, 6),
    }, "seasons": {}}
    player_season_frames: List[pd.DataFrame] = []
    player_game_frames: List[pd.DataFrame] = []

    for season in sorted(all_params.keys()):
        print(f"\n{'='*60}")
        print(f"  Season: {season}")
        params = all_params[season]
        season_lineup_bonus = lineup_bonus_all.get(season, {})
        season_matchup_bonus = matchup_bonus_all.get(season, {})

        # Build schedule: use actual games for backtest, synthetic for forecast.
        if forecast_mode:
            teams = sorted(params.keys())
            schedule = generate_balanced_schedule(season, teams)
            print(f"  Teams: {len(params)}, Synthetic games: {len(schedule)}")
        else:
            schedule = build_schedule(season)
            print(f"  Teams: {len(params)}, Games: {len(schedule)}")

        games = _valid_games(schedule, params)
        all_teams = sorted(params.keys())
        season_actual = get_actual_records(season)

        season_team_results_by_model = {}
        season_stats_by_model = {}
        season_delta_mu_by_model = {}
        t0 = time.time()

        for model_idx, model_key in enumerate(SIM_MODELS):
            if model_key == SIM_MODEL_PPP:
                season_ppp = ppp_components_all.get(season, {})
                season_pace = pace_pred_all.get(season, {})
                delta_mus, sigma_games, home_teams, away_teams = _build_ppp_game_arrays(
                    games=games,
                    team_params=params,
                    config=config,
                    season_ppp_components=season_ppp,
                    season_pace_map=season_pace,
                    impact_to_net_scale=impact_to_net_scale,
                    lineup_bonus_map=season_lineup_bonus,
                    matchup_bonus_map=season_matchup_bonus,
                )
            else:
                season_ppp = ppp_components_all.get(season, {})
                season_pace = pace_pred_all.get(season, {})
                delta_mus, sigma_games, home_teams, away_teams = _build_margin_game_arrays(
                    games=games,
                    team_params=params,
                    config=config,
                    lineup_bonus_map=season_lineup_bonus,
                    matchup_bonus_map=season_matchup_bonus,
                )

            win_dists = _simulate_from_game_arrays(
                delta_mus=delta_mus,
                sigma_games=sigma_games,
                home_teams=home_teams,
                away_teams=away_teams,
                all_teams=all_teams,
                n_sims=config.n_simulations,
                seed=config.random_seed + model_idx,
            )
            summaries = aggregate_results(
                win_distributions=win_dists,
                team_params=params,
                model_key=model_key,
                ppp_components=season_ppp,
                pace_map=season_pace,
                impact_to_net_scale=impact_to_net_scale,
            )
            season_stats = _attach_actuals_and_stats(summaries, season_actual)

            season_team_results_by_model[model_key] = summaries
            season_stats_by_model[model_key] = season_stats
            season_delta_mu_by_model[model_key] = delta_mus

            if season_stats:
                print(
                    f"  [{model_key}] MAE={season_stats['mae']}, "
                    f"RMSE={season_stats['rmse']}, r={season_stats['correlation']}"
                )
            else:
                print(f"  [{model_key}] No actual W-L available for this season")

        elapsed = time.time() - t0
        print(f"  Simulated {config.n_simulations:,} seasons/model in {elapsed:.2f}s")

        alignment_diag = {}
        if SIM_MODEL_MARGIN in season_delta_mu_by_model and SIM_MODEL_PPP in season_delta_mu_by_model:
            alignment_diag = _model_alignment_diagnostics(
                margin_delta_mus=season_delta_mu_by_model[SIM_MODEL_MARGIN],
                ppp_delta_mus=season_delta_mu_by_model[SIM_MODEL_PPP],
            )
            if alignment_diag:
                print(
                    "  [diagnostic] margin-vs-ppp delta_mu "
                    f"corr={alignment_diag.get('delta_mu_correlation')} "
                    f"mean_abs_gap={alignment_diag.get('mean_abs_gap')}"
                )

        default_model = SIM_MODEL_MARGIN if SIM_MODEL_MARGIN in season_team_results_by_model else list(season_team_results_by_model.keys())[0]
        default_summaries = season_team_results_by_model[default_model]

        detailed_env_rows = _sample_game_environment_rows(
            games=games,
            team_params=params,
            config=config,
            season_ppp_components=ppp_components_all.get(season, {}),
            season_pace_map=pace_pred_all.get(season, {}),
            impact_to_net_scale=impact_to_net_scale,
            lineup_bonus_map=season_lineup_bonus,
            matchup_bonus_map=season_matchup_bonus,
            model_key=default_model,
            seed=config.random_seed + _season_start_year(season),
        )
        season_rosters = rosters[rosters["season"] == season].copy()
        season_player_stats, season_game_stats = simulate_detailed_season(
            schedule=games,
            rosters=season_rosters,
            lineup_bonus_map={season: season_lineup_bonus},
            matchup_map={season: season_matchup_bonus},
            game_environment_rows=detailed_env_rows,
            rng=np.random.default_rng(config.random_seed + 1000 + _season_start_year(season)),
            model_key=default_model,
        )
        if not season_player_stats.empty:
            player_season_frames.append(season_player_stats)
        if not season_game_stats.empty:
            player_game_frames.append(season_game_stats)

        print(f"\n  Top 5 ({default_model}):")
        for s in default_summaries[:5]:
            act_str = f" (actual: {s.get('actual_wins', '?')})" if "actual_wins" in s else ""
            print(
                f"    {s['projected_rank']:2d}. {s['team']} — {s['projected_wins']:.1f} wins "
                f"[{s['win_p5']}-{s['win_p95']}]{act_str}"
            )
        print(f"  Bottom 5 ({default_model}):")
        for s in default_summaries[-5:]:
            act_str = f" (actual: {s.get('actual_wins', '?')})" if "actual_wins" in s else ""
            print(
                f"    {s['projected_rank']:2d}. {s['team']} — {s['projected_wins']:.1f} wins "
                f"[{s['win_p5']}-{s['win_p95']}]{act_str}"
            )

        full_results["seasons"][season] = {
            "season_stats": season_stats_by_model.get(default_model, {}),
            "season_stats_by_model": season_stats_by_model,
            "model_alignment_diagnostics": alignment_diag,
            "team_results": default_summaries,
            "team_results_by_model": season_team_results_by_model,
            "default_model": default_model,
            "detailed_simulation": {
                "player_rows": int(len(season_player_stats)),
                "game_rows": int(len(season_game_stats)),
                "sample_games": int(season_game_stats["game_id"].nunique()) if not season_game_stats.empty else 0,
            },
        }

    # Save
    dst = output_path or (FORECAST_SEASON_RESULTS_PATH if forecast_mode else REPORTS_DIR / "simulation_step1_season_results.json")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(full_results, indent=2), encoding="utf-8")
    season_stats_path, game_samples_path = _resolve_detailed_output_paths(forecast_mode, dst)
    if player_season_frames:
        pd.concat(player_season_frames, ignore_index=True).to_parquet(season_stats_path, index=False)
        print(f"Saved player season stats: {season_stats_path}")
    if player_game_frames:
        pd.concat(player_game_frames, ignore_index=True).to_parquet(game_samples_path, index=False)
        print(f"Saved player game samples: {game_samples_path}")
    print(f"\nSaved: {dst}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run season simulation and detailed player stat simulation")
    parser.add_argument("--forecast-mode", action="store_true", help="Use forecast artifacts instead of backtest artifacts")
    parser.add_argument("--features-path", type=str, default=None, help="Optional team feature parquet override")
    parser.add_argument("--player-profiles-path", type=str, default=None, help="Optional player profile parquet override")
    parser.add_argument("--output-path", type=str, default=None, help="Optional season-results JSON output path override")
    parser.add_argument("--single-game-home", type=str, default=None, help="Run a single detailed game simulation for the specified home team")
    parser.add_argument("--single-game-away", type=str, default=None, help="Run a single detailed game simulation for the specified away team")
    parser.add_argument("--single-game-season", type=str, default=None, help="Season for single-game mode (defaults to latest available)")
    args = parser.parse_args()

    main(
        forecast_mode=args.forecast_mode,
        features_path=Path(args.features_path) if args.features_path else None,
        player_profiles_path=Path(args.player_profiles_path) if args.player_profiles_path else None,
        output_path=Path(args.output_path) if args.output_path else None,
        single_game_home=args.single_game_home,
        single_game_away=args.single_game_away,
        single_game_season=args.single_game_season,
    )
