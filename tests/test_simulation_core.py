import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.simulation.game_model import generate_balanced_schedule
from src.simulation.player_stats_sim import (
    build_pairwise_matchup_map,
    build_team_descriptor_map,
    simulate_single_game_player_stats,
)


REQUIRED_SIM_COLUMNS = {
    "player_id",
    "player_name",
    "season",
    "team_abbreviation",
    "role",
    "off_archetype",
    "def_archetype",
    "base_mpg",
    "games_target",
    "availability_rate",
    "impact_total_impact",
    "impact_obke",
    "impact_dbke",
    "behavioral_usage",
    "behavioral_assist_rate",
    "behavioral_turnover_rate",
    "behavioral_three_point_rate",
    "pts_per36",
    "ast_per36",
    "reb_per36",
    "stl_per36",
    "blk_per36",
    "tov_per36",
    "base_fga_per36",
    "base_fta_per36",
    "fg3a_per36",
    "oreb_per36",
    "dreb_per36",
    "pf_per36",
    "fg2_pct",
    "fg3_pct",
    "ft_pct",
}


def _player(team, player_id, role, off_archetype, def_archetype, mpg, usage, three_rate):
    row = {
        "player_id": str(player_id),
        "player_name": f"P{player_id}",
        "season": "2025-26",
        "team_abbreviation": team,
        "role": role,
        "off_archetype": off_archetype,
        "def_archetype": def_archetype,
        "base_mpg": mpg,
        "games_target": 75.0,
        "availability_rate": 0.91,
        "impact_total_impact": 55.0 + mpg,
        "impact_obke": 0.3,
        "impact_dbke": 0.2,
        "behavioral_usage": usage,
        "behavioral_assist_rate": 0.18 if role == "Guard" else 0.08,
        "behavioral_turnover_rate": 0.12,
        "behavioral_three_point_rate": three_rate,
        "pts_per36": 18.0,
        "ast_per36": 5.0 if role == "Guard" else 2.5,
        "reb_per36": 9.0 if role == "Big" else 5.0,
        "stl_per36": 1.5,
        "blk_per36": 1.8 if role == "Big" else 0.5,
        "tov_per36": 2.1,
        "base_fga_per36": 13.0,
        "base_fta_per36": 3.5,
        "fg3a_per36": 5.0 if three_rate >= 0.30 else 1.5,
        "oreb_per36": 2.2 if role == "Big" else 0.8,
        "dreb_per36": 6.5 if role == "Big" else 3.4,
        "pf_per36": 2.7,
        "fg2_pct": 0.53,
        "fg3_pct": 0.36,
        "ft_pct": 0.79,
    }
    return row


def _roster(team, poa=False, low_activity=False):
    defenders = [
        "POA Defender" if poa else ("Low-Activity Defender" if low_activity else "Rotational Defender"),
        "Wing Stopper" if poa else "Rotational Defender",
        "Rim Protector",
        "Off-Ball Chaser" if poa else "Rotational Defender",
        "Mobile Big",
    ]
    rows = [
        _player(team, f"{team}1", "Guard", "Ball Dominant Creator", defenders[0], 35.0, 0.31, 0.28),
        _player(team, f"{team}2", "Guard", "Perimeter Scorer", defenders[1], 33.0, 0.24, 0.42),
        _player(team, f"{team}3", "Wing", "Off-Ball Movement Shooter", defenders[3], 31.0, 0.17, 0.46),
        _player(team, f"{team}4", "Wing", "Connector", "Versatile Defender", 28.0, 0.15, 0.32),
        _player(team, f"{team}5", "Big", "PnR Rolling Big", defenders[2], 30.0, 0.16, 0.08),
        _player(team, f"{team}6", "Big", "PnR Popping Big", defenders[4], 24.0, 0.14, 0.35),
        _player(team, f"{team}7", "Guard", "Ballhandler", "Rotational Defender", 20.0, 0.20, 0.25),
        _player(team, f"{team}8", "Wing", "Off-Ball Stationary Shooter", "Rotational Defender", 18.0, 0.13, 0.48),
    ]
    return pd.DataFrame(rows)


def test_generate_balanced_schedule_hits_target_games_per_team():
    teams = [f"T{i:02d}" for i in range(30)]
    schedule = generate_balanced_schedule("2025-26", teams)
    counts = {team: 0 for team in teams}
    for game in schedule:
        counts[game.home_team] += 1
        counts[game.away_team] += 1

    assert len(schedule) == 1230
    assert all(value == 82 for value in counts.values())


def test_matchup_map_penalizes_creator_against_poa():
    offense = _roster("OFF")
    poa = _roster("POA", poa=True)
    low = _roster("LOW", low_activity=True)
    rosters = pd.concat([offense, poa, low], ignore_index=True)

    matchup_map = build_pairwise_matchup_map(rosters)
    vs_poa = matchup_map["2025-26"][("OFF", "POA")]["home_offense_bonus"]
    vs_low = matchup_map["2025-26"][("OFF", "LOW")]["home_offense_bonus"]

    assert vs_poa < vs_low


def test_single_game_player_stats_reconcile_team_points_and_minutes():
    home = _roster("HOM")
    away = _roster("AWY", poa=True)
    descriptors = build_team_descriptor_map(pd.concat([home, away], ignore_index=True))
    away_desc = descriptors["2025-26"]["AWY"]
    lineup_context = {
        "starter_ids": home.head(5)["player_id"].tolist(),
        "clutch_ids": home.head(5)["player_id"].tolist(),
        "team_bonus": 0.3,
        "clutch_bonus": 0.2,
    }

    box = simulate_single_game_player_stats(
        season="2025-26",
        team_abbreviation="HOM",
        opponent_abbreviation="AWY",
        team_score=112,
        opponent_score=106,
        possessions=99.5,
        team_roster=home,
        opponent_descriptor=away_desc,
        lineup_context=lineup_context,
        rng=np.random.default_rng(42),
        close_game=True,
        is_home=True,
        game_id="G1",
        date="2025-10-20",
        model_key="margin",
    )

    assert not box.empty
    assert REQUIRED_SIM_COLUMNS.issubset(set(home.columns))
    assert int(box["pts"].sum()) == 112
    assert math.isclose(float(box["sim_minutes"].sum()), 240.0, rel_tol=0.0, abs_tol=1e-6)
    assert (box[["pts", "reb", "ast", "stl", "blk", "tov", "fga", "fta"]] >= 0).all().all()