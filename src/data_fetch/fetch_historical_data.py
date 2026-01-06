"""
src/data_fetch/fetch_historical_data.py
Utilities for fetching historical play-by-play and game-log data for past seasons.
Input: season identifiers (e.g., '2022-23')
Output: saved CSV/Parquet files under data/historical
"""

import requests
import os
import math
import random

def save_player_id_name_mapping(season):
    import os
    os.makedirs("data/historical", exist_ok=True)
    players_df = commonallplayers.CommonAllPlayers(is_only_current_season=0, season=season).get_data_frames()[0]
    mapping_df = players_df[["PERSON_ID", "DISPLAY_FIRST_LAST"]].drop_duplicates()
    mapping_df.to_csv(f"data/historical/player_id_name_map_{season}.csv", index=False)
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

def fetch_team_game_logs(seasons):
    from nba_api.stats.static import teams
    all_seasons = []
    for season in seasons:
        print(f"Fetching all team game logs for {season} season...")
        team_list = teams.get_teams()
        print(f"  Number of teams found: {len(team_list)}")
        team_ids = [team['id'] for team in team_list]
        season_team_logs = []
        for team_id in team_ids:
            try:
                logs = teamgamelog.TeamGameLog(team_id=team_id, season=season)
                df = logs.get_data_frames()[0]
                print(f"    Team {team_id} ({season}) logs shape: {df.shape}")
                if not df.empty:
                    df["SEASON"] = season
                    df["TEAM_ID"] = team_id
                    season_team_logs.append(df)
                else:
                    print(f"    [DEBUG] No logs for team {team_id} in {season}")
                time.sleep(0.6)
            except Exception as e:
                print(f"[ERROR] Could not fetch logs for team {team_id} in {season}: {e}")
        if season_team_logs:
            print(f"  Teams with logs for {season}: {len(season_team_logs)}")
            all_seasons.append(pd.concat(season_team_logs, ignore_index=True))
        else:
            print(f"  [DEBUG] No team logs for {season}")
    if all_seasons:
        print(f"Total team log DataFrames to concat: {len(all_seasons)}")
        return pd.concat(all_seasons, ignore_index=True)
    else:
        print("[DEBUG] No team logs collected across all seasons.")
        return pd.DataFrame()

def fetch_player_game_logs(seasons):
    import csv
    all_seasons_players = []
    for season in seasons:
        print(f"Fetching player game logs for {season} season...")
        players_df = commonallplayers.CommonAllPlayers(is_only_current_season=0, season=season).get_data_frames()[0]
        active_players = players_df[players_df["ROSTERSTATUS"] == 1]
        player_ids = active_players["PERSON_ID"].tolist()

        # Load player_id to name mapping for this season
        id_to_name = {}
        try:
            with open(f"data/historical/player_id_name_map_['2022-23', '2023-24', '2024-25'].csv", newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    id_to_name[int(row["PERSON_ID"])] = row["DISPLAY_FIRST_LAST"]
        except Exception as e:
            print(f"Warning: Could not load player_id_name_map for {season}: {e}")

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
    seasons = ["2022-23", "2023-24", "2024-25"]
    # Test fetching 10 players from 2023-24 season
    #fetch_ten_players_game_logs("2023-24")

    #save player names, already ran
    #save_player_id_name_mapping(seasons)

    teams_df = fetch_team_game_logs(seasons)
    teams_df.to_parquet("data/historical/team_game_logs.parquet", index=False)

    #players_df = fetch_player_game_logs(seasons)
    #players_df.to_parquet("data/historical/final_player_game_logs.parquet", index=False)

    print("✅ Historical team and player data successfully saved!")
