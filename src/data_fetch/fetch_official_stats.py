"""
src/data_fetch/fetch_official_stats.py
Stream C: Fetches "Official" Advanced Stats for Display.
Matches NBA.com / Basketball-Reference exactly.
"""

import pandas as pd
import time
import os
import sys
import random
import json
from curl_cffi import requests
from pathlib import Path

# Adjust path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

DATA_DIR = Path("data/official_stats")
CACHE_DIR = Path("data/tracking_cache") # Reuse cache logic
SEASONS = ["2022-23", "2023-24", "2024-25"]

def ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

def fetch_official_advanced(season):
    print(f"\n🏆 Fetching Official Advanced Stats for {season}...")
    
    url = "https://stats.nba.com/stats/leaguedashplayerstats"
    params = {
        "MeasureType": "Advanced", # The Key Parameter
        "PerMode": "PerGame",
        "PlusMinus": "N",
        "PaceAdjust": "N",
        "Rank": "N",
        "LeagueID": "00",
        "Season": season,
        "SeasonType": "Regular Season",
        "PorRound": "0",
        "Outcome": "",
        "Location": "",
        "Month": "0",
        "SeasonSegment": "",
        "DateFrom": "",
        "DateTo": "",
        "OpponentTeamID": "0",
        "VsConference": "",
        "VsDivision": "",
        "TeamID": "0",
        "Conference": "",
        "Division": "",
        "GameSegment": "",
        "Period": "0",
        "ShotClockRange": "",
        "LastNGames": "0",
        "GameScope": "",
        "PlayerExperience": "",
        "PlayerPosition": "",
        "StarterBench": "",
        "DraftYear": "",
        "DraftPick": "",
        "College": "",
        "Country": "",
        "Height": "",
        "Weight": ""
    }
    
    # Headers (Chrome 110 Impersonation)
    headers = {
        'Accept': 'application/json, text/plain, */*',
        'Connection': 'keep-alive',
        'Origin': 'https://www.nba.com',
        'Referer': 'https://www.nba.com/stats/players/advanced',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'x-nba-stats-origin': 'stats',
        'x-nba-stats-token': 'true',
    }

    try:
        resp = requests.get(
            url, params=params, headers=headers, 
            impersonate="chrome110", timeout=30
        )
        
        if resp.status_code != 200:
            print(f"❌ Status {resp.status_code}")
            return
            
        json_data = resp.json()
        headers = json_data['resultSets'][0]['headers']
        rows = json_data['resultSets'][0]['rowSet']
        
        df = pd.DataFrame(rows, columns=headers)
        
        # Save
        outfile = DATA_DIR / f"official_advanced_{season}.parquet"
        df.to_parquet(outfile, index=False)
        print(f"✅ Saved {len(df)} rows to {outfile}")
        
        # Preview Embiid for Verification
        if "Joel Embiid" in df['PLAYER_NAME'].values:
            embiid = df[df['PLAYER_NAME'] == "Joel Embiid"][['PLAYER_NAME', 'USG_PCT', 'TS_PCT', 'AST_PCT', 'REB_PCT']]
            print("   --- Verification (Joel Embiid) ---")
            print(embiid.to_string(index=False))
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    ensure_dirs()
    for s in SEASONS:
        fetch_official_advanced(s)
        time.sleep(2)