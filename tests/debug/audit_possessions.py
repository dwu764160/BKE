"""
tests/debug/audit_possessions_bulk.py
Performs a Large Scale Audit (N=100) to check for:
1. Accuracy vs NBA (MAE)
2. Internal Consistency (Team A vs Team B Balance)
"""

import pandas as pd
import os
import sys
import json
import time
import random
import numpy as np
from curl_cffi import requests

# Adjust path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

DATA_DIR = "data/historical"
HEADERS_FILE = "data/nba_headers.json"
SEASON = "2023-24"
SAMPLE_SIZE = 100  # Increased from 5

def load_local_data():
    raw_path = os.path.join(DATA_DIR, f"possessions_{SEASON}.parquet")
    if not os.path.exists(raw_path): return None
    print(f"Loading local data {SEASON}...")
    return pd.read_parquet(raw_path)

def load_headers():
    if not os.path.exists(HEADERS_FILE): return {}
    try:
        with open(HEADERS_FILE, 'r') as f: return json.load(f)
    except: return {}

def fetch_season_gamelogs():
    url = "https://stats.nba.com/stats/teamgamelogs"
    params = {
        "MeasureType": "Advanced", "PerMode": "Totals", "Season": SEASON,
        "SeasonType": "Regular Season", "LeagueID": "00"
    }
    
    base_headers = {
        'Accept': 'application/json', 'Connection': 'keep-alive',
        'Origin': 'https://www.nba.com', 'Referer': 'https://www.nba.com/',
        'x-nba-stats-origin': 'stats'
    }
    
    captured = load_headers()
    if 'x-nba-stats-token' in captured:
        base_headers['x-nba-stats-token'] = captured['x-nba-stats-token']
        base_headers['User-Agent'] = captured.get('User-Agent', '')
    else:
        base_headers['x-nba-stats-token'] = 'true'

    try:
        print("Fetching Bulk Game Logs...", end=" ")
        response = requests.get(url, params=params, headers=base_headers, impersonate="chrome110", timeout=20)
        if response.status_code != 200: return None
        
        json_data = response.json()
        headers = json_data['resultSets'][0]['headers']
        rows = json_data['resultSets'][0]['rowSet']
        print(f"✅ ({len(rows)} rows)")
        return pd.DataFrame(rows, columns=headers)
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def audit():
    raw_df = load_local_data()
    nba_df = fetch_season_gamelogs()
    
    if raw_df is None or nba_df is None: return
    
    # Pre-process Local Data
    # Count possessions per Game-Team
    local_counts = raw_df.groupby(['game_id', 'off_team_id']).size().reset_index(name='OUR_POSS')
    
    # Pre-process Official Data
    nba_df['GAME_ID'] = nba_df['GAME_ID'].astype(str)
    nba_df['TEAM_ID'] = nba_df['TEAM_ID'].astype(int)
    nba_df['POSS'] = nba_df['POSS'].astype(float) # Possessions are often floats in API
    
    # Sample Games
    all_games = nba_df['GAME_ID'].unique()
    sample_games = random.sample(list(all_games), min(SAMPLE_SIZE, len(all_games)))
    
    results = []
    
    print(f"\n--- Analysis of {len(sample_games)} Games ---")
    
    for gid in sample_games:
        # Get Official Rows (2 per game)
        game_nba = nba_df[nba_df['GAME_ID'] == gid]
        if len(game_nba) != 2: continue
        
        # Merge our data
        merged = pd.merge(game_nba, local_counts, left_on=['GAME_ID', 'TEAM_ID'], right_on=['game_id', 'off_team_id'], how='left')
        merged['OUR_POSS'] = merged['OUR_POSS'].fillna(0)
        
        # Calculate Game-Level Metrics
        team_a = merged.iloc[0]
        team_b = merged.iloc[1]
        
        # 1. Accuracy vs NBA
        diff_a = team_a['OUR_POSS'] - team_a['POSS']
        diff_b = team_b['OUR_POSS'] - team_b['POSS']
        
        # 2. Internal Consistency (The "Impossible Gap")
        # In a real game, Team A Poss approx equals Team B Poss
        our_gap = abs(team_a['OUR_POSS'] - team_b['OUR_POSS'])
        nba_gap = abs(team_a['POSS'] - team_b['POSS'])
        
        results.append({
            "Game": gid,
            "TeamA": team_a['TEAM_ABBREVIATION'],
            "TeamB": team_b['TEAM_ABBREVIATION'],
            "Our_A": int(team_a['OUR_POSS']),
            "Our_B": int(team_b['OUR_POSS']),
            "NBA_A": int(round(team_a['POSS'])),
            "NBA_B": int(round(team_b['POSS'])),
            "Diff_A": round(diff_a, 1),
            "Diff_B": round(diff_b, 1),
            "Our_Gap": int(our_gap), # Should be near 0
            "Abs_Error": abs(diff_a) + abs(diff_b) # Total error magnitude
        })
        
    res_df = pd.DataFrame(results)
    
    # Sort by worst Internal Gap
    res_df = res_df.sort_values("Our_Gap", ascending=False)
    
    print("\n=== WORST 10 GAMES (Internal Consistency Failure) ===")
    print(res_df[['Game', 'TeamA', 'Our_A', 'TeamB', 'Our_B', 'Our_Gap', 'Abs_Error']].head(10).to_string(index=False))
    
    # Stats
    mae = res_df['Abs_Error'].mean() / 2 # Per team
    avg_gap = res_df['Our_Gap'].mean()
    
    print("\n=== DIAGNOSTICS ===")
    print(f"Mean Absolute Error (vs NBA): {mae:.2f} per team")
    print(f"Avg Internal Gap (A vs B):    {avg_gap:.2f} (Should be < 2.0)")
    
    if avg_gap > 3.0:
        print("🔴 CRITICAL: Massive Possession Misattribution.")
        print("   Logic is assigning possessions to the wrong team.")
    else:
        print("🟢 Logic seems consistent.")

if __name__ == "__main__":
    audit()