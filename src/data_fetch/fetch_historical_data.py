"""
src/data_fetch/fetch_historical_data.py
=============================================================================
Utilities for fetching historical play-by-play and game-log data for past seasons.
Input: season identifiers (e.g., '2022-23')
Output: saved CSV/Parquet files under data/historical
=============================================================================
"""

import requests
import os
import math
import random
from pathlib import Path
from curl_cffi import requests as curl_requests

HISTORICAL_DIR = Path("data/historical")


def _player_map_candidates(season: str):
    """Return player-map candidates with season-specific files first."""
    season = str(season)
    candidates = [
        HISTORICAL_DIR / f"player_id_name_map_{season}.csv",
        HISTORICAL_DIR / "player_id_name_map_all.csv",
        HISTORICAL_DIR / "player_id_name_map_['2022-23', '2023-24', '2024-25'].csv",
    ]
    candidates.extend(sorted(HISTORICAL_DIR.glob("player_id_name_map_*.csv")))

    seen = set()
    ordered = []
    for path in candidates:
        norm = str(path)
        if norm not in seen:
            seen.add(norm)
            ordered.append(path)
    return ordered


def _load_player_id_name_map(season: str):
    import csv

    id_to_name = {}
    for map_path in _player_map_candidates(season):
        if not map_path.exists():
            continue
        try:
            with open(map_path, newline="", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    person_id = row.get("PERSON_ID")
                    display_name = row.get("DISPLAY_FIRST_LAST")
                    if person_id is None or display_name is None:
                        continue
                    try:
                        id_to_name[int(float(person_id))] = display_name
                    except (TypeError, ValueError):
                        continue
            if id_to_name:
                print(f"Loaded {len(id_to_name)} player-name mappings from {map_path}")
                return id_to_name
        except Exception as e:
            print(f"Warning: Could not load player map {map_path}: {e}")

    return id_to_name

def save_player_id_name_mapping(season):
    # Accept a single season or a list/tuple of seasons.
    if isinstance(season, (list, tuple, set)):
        for s in season:
            save_player_id_name_mapping(str(s))
        return

    import os
    os.makedirs("data/historical", exist_ok=True)
    players_df = commonallplayers.CommonAllPlayers(is_only_current_season=0, season=season).get_data_frames()[0]
    mapping_df = players_df[["PERSON_ID", "DISPLAY_FIRST_LAST"]].drop_duplicates()
    mapping_df.to_csv(HISTORICAL_DIR / f"player_id_name_map_{season}.csv", index=False)
    print(f"Saved player ID-name mapping for {season} season.")

def fetch_ten_players_game_logs(season):
    print(f"Testing: Fetching game logs for 10 players from {season} season...")
    players_df = commonallplayers.CommonAllPlayers(is_only_current_season=0, season=season).get_data_frames()[0]
    active_players = players_df[players_df["ROSTERSTATUS"] == 1]
    player_ids = active_players["PERSON_ID"].tolist()[:10]

    all_players = []
    for player_id in player_ids:
        try:
            logs = playergamelog.PlayerGameLog(player_id, season, timeout=3)
            df = logs.get_data_frames()[0]
            df["SEASON"] = season
            df["PLAYER_ID"] = player_id
            all_players.append(df)
            print(f"Fetched logs for player {player_id}")
            time.sleep(2)  # longer delay for testing
        except Exception as e:
            print(f"Failed to fetch logs for player {player_id}: {e}")
            continue

    if all_players:
        result_df = pd.concat(all_players, ignore_index=True)
        result_df.to_parquet(f"data/historical/ten_player_game_logs_{season}.parquet", index=False)
        print(f"Saved logs for 10 players from {season} season.")
    else:
        print("No logs fetched for test players.")
# src/data_fetch/fetch_historical_data.py

from nba_api.stats.endpoints import teamgamelog, playergamelog, commonallplayers
import pandas as pd
import time


def _fetch_leaguegamelog_direct(season):
    """Fetch all team game logs for one season in a single stats.nba.com call."""
    url = "https://stats.nba.com/stats/leaguegamelog"
    params = {
        "Counter": "0",
        "DateFrom": "",
        "DateTo": "",
        "Direction": "ASC",
        "LeagueID": "00",
        "PlayerOrTeam": "T",
        "Season": season,
        "SeasonType": "Regular Season",
        "Sorter": "DATE",
    }
    headers = {
        'Accept': 'application/json, text/plain, */*',
        'Connection': 'keep-alive',
        'Origin': 'https://www.nba.com',
        'Referer': 'https://www.nba.com/stats/teams/traditional',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'x-nba-stats-origin': 'stats',
        'x-nba-stats-token': 'true',
    }

    resp = curl_requests.get(
        url,
        params=params,
        headers=headers,
        impersonate="chrome110",
        timeout=60,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"leaguegamelog status={resp.status_code}")

    data = resp.json()
    if isinstance(data, dict) and "resultSets" in data and data["resultSets"]:
        rs = data["resultSets"][0]
        return pd.DataFrame(rs.get("rowSet", []), columns=rs.get("headers", []))
    if isinstance(data, dict) and "resultSet" in data:
        rs = data["resultSet"]
        return pd.DataFrame(rs.get("rowSet", []), columns=rs.get("headers", []))
    raise RuntimeError("Unexpected leaguegamelog JSON format")

def fetch_team_game_logs(seasons):
    from nba_api.stats.static import teams
    all_seasons = []
    for season in seasons:
        print(f"Fetching all team game logs for {season} season...")

        # Primary path: one-shot league endpoint
        got_season = False
        for attempt in range(3):
            try:
                df = _fetch_leaguegamelog_direct(season)
                if not df.empty:
                    df["SEASON"] = season
                    all_seasons.append(df)
                    got_season = True
                    print(f"  ✅ Direct leaguegamelog rows: {len(df)}")
                    break
            except Exception as e:
                wait = 2 ** attempt
                print(f"  [WARN] Direct leaguegamelog failed (attempt {attempt+1}/3): {e}; retrying in {wait}s")
                time.sleep(wait)

        if got_season:
            continue

        # Fallback path: per-team TeamGameLog
        team_list = teams.get_teams()
        print(f"  Falling back to TeamGameLog calls; teams: {len(team_list)}")
        team_ids = [team['id'] for team in team_list]
        season_team_logs = []
        for team_id in team_ids:
            try:
                logs = teamgamelog.TeamGameLog(team_id=team_id, season=season, timeout=60)
                df = logs.get_data_frames()[0]
                if not df.empty:
                    df["SEASON"] = season
                    df["TEAM_ID"] = team_id
                    season_team_logs.append(df)
                time.sleep(0.3)
            except Exception as e:
                print(f"  [WARN] Could not fetch team {team_id} in {season}: {e}")

        if season_team_logs:
            season_df = pd.concat(season_team_logs, ignore_index=True)
            print(f"  ✅ Fallback TeamGameLog rows: {len(season_df)}")
            all_seasons.append(season_df)
        else:
            print(f"  ❌ No team logs fetched for {season}")

    if all_seasons:
        print(f"Total team log DataFrames to concat: {len(all_seasons)}")
        return pd.concat(all_seasons, ignore_index=True)
    else:
        print("[DEBUG] No team logs collected across all seasons.")
        return pd.DataFrame()

def fetch_player_game_logs(seasons):
    all_seasons_players = []
    for season in seasons:
        print(f"Fetching player game logs for {season} season...")
        players_df = commonallplayers.CommonAllPlayers(is_only_current_season=0, season=season).get_data_frames()[0]
        active_players = players_df[players_df["ROSTERSTATUS"] == 1]
        player_ids = active_players["PERSON_ID"].tolist()

        # Load player_id to name mapping for this season
        id_to_name = _load_player_id_name_map(season)
        if not id_to_name:
            print(f"Warning: Could not load any player_id_name_map for {season}")

        all_players = []
        failed_count = 0
        success_count = 0
        for player_id in player_ids:
            player_name = id_to_name.get(player_id, "Unknown")
            for attempt in range(4):
                try:
                    logs = playergamelog.PlayerGameLog(player_id, season, timeout=10)
                    df = logs.get_data_frames()[0]
                    if not df.empty:
                        df["SEASON"] = season
                        df["PLAYER_ID"] = player_id
                        all_players.append(df)
                        success_count += 1
                    break
                except requests.exceptions.ReadTimeout:
                    wait = 2 ** attempt
                    print(f"Timeout for {player_id} ({player_name}), retrying in {wait}s...")
                    time.sleep(wait)
                except Exception as e:
                    print(f"Failed to fetch logs for player {player_id} ({player_name}): {e}")
                    failed_count += 1
                    break
            time.sleep(random.uniform(1.5, 3.0))
        print(f"Season {season}: Successfully fetched logs for {success_count} players, failed for {failed_count} players.")
        if all_players:
            all_seasons_players.extend(all_players)
        else:
            print(f"No player logs were fetched for {season}.")
    print("All seasons complete.")
    if all_seasons_players:
        return pd.concat(all_seasons_players, ignore_index=True)
    else:
        return pd.DataFrame()

if __name__ == "__main__":
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    from src.modeling.model_config import SEASONS as seasons
    # Test fetching 10 players from 2023-24 season
    #fetch_ten_players_game_logs("2023-24")

    #save player names, already ran
    #save_player_id_name_mapping(seasons)

    teams_df = fetch_team_game_logs(seasons)
    teams_df.to_parquet("data/historical/team_game_logs.parquet", index=False)

    #players_df = fetch_player_game_logs(seasons)
    #players_df.to_parquet("data/historical/final_player_game_logs.parquet", index=False)

    print("✅ Historical team and player data successfully saved!")
