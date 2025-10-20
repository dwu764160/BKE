import pandas as pd
from nba_api.stats.endpoints import teamgamelog

if __name__ == "__main__":
    from nba_api.stats.static import teams
    # Pick a known team, e.g., Atlanta Hawks
    team_info = teams.find_teams_by_full_name('Atlanta Hawks')[0]
    team_id = team_info['id']
    print(f"[DEBUG] Using team: {team_info['full_name']} (ID: {team_id})")
    season = "2022-23"
    try:
        logs = teamgamelog.TeamGameLog(team_id=team_id, season=season)
        df = logs.get_data_frames()[0]
        print(f"[DEBUG] DataFrame shape for {season}: {df.shape}")
        print(f"[DEBUG] Columns: {list(df.columns)}")
        print(f"[DEBUG] First 10 rows for {season}:")
        print(df.head(10))
    except Exception as e:
        print(f"[ERROR] Could not fetch team logs for {team_id} in {season}: {e}")
