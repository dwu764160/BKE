"""
src/data_fetch/fetch_tracking_data.py
Fetches "Style" and "Tracking" data using TLS Impersonation (curl_cffi).
BYPASSES: NBA Akamai/Cloudflare TLS Fingerprinting blocks.
"""

import pandas as pd
import time
import os
import sys
import random
from curl_cffi import requests # The magic library

# Adjust path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

DATA_DIR = "data/tracking"
SEASONS = ["2022-23", "2023-24", "2024-25"]

# --- CONFIGURATION ---
TRACKING_MEASURES = {
    "Drives": "Drives",
    "Defense": "Defense", 
    "CatchShoot": "CatchShoot",
    "PullUpShot": "PullUpShot",
    "Passing": "Passing",
    "Possessions": "Possessions",
    "Rebounding": "Rebounding",
    "Efficiency": "Efficiency",
    "SpeedDistance": "SpeedDistance"
}

SYNERGY_TYPES = [
    "Isolation", "Transition", "PRBallHandler", "PRRollman", 
    "Postup", "Spotup", "Handoff", "Cut", "OffScreen", "Putback", "Misc"
]

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def smart_sleep():
    time.sleep(random.uniform(1.0, 3.0))

def get_nba_data(url, params, referer_suffix):
    """
    Fetches data using curl_cffi to impersonate a real Chrome browser.
    """
    # Headers that look exactly like Chrome
    headers = {
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Connection': 'keep-alive',
        'Origin': 'https://www.nba.com',
        'Referer': f'https://www.nba.com/stats/players/{referer_suffix}',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-site',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'x-nba-stats-origin': 'stats',
        'x-nba-stats-token': 'true', # We try 'true' first; usually works with valid TLS
    }

    try:
        # impersonate="chrome110" is the Key to bypassing TLS blocks
        response = requests.get(
            url, 
            params=params, 
            headers=headers, 
            impersonate="chrome110", 
            timeout=30
        )
        
        if response.status_code != 200:
            print(f"⚠️ Status {response.status_code}", end=" ")
            return None
            
        json_data = response.json()
        
        # Parse NBA's ResultSets format
        result_sets = json_data.get('resultSets', [])
        if not result_sets:
            return pd.DataFrame()
            
        # Usually the first result set is what we want
        headers = result_sets[0]['headers']
        row_set = result_sets[0]['rowSet']
        
        return pd.DataFrame(row_set, columns=headers)

    except Exception as e:
        print(f"⚠️ Error: {e}", end=" ")
        return None

def fetch_tracking(season):
    print(f"\n🏀 Fetching Tracking Data for {season}...")
    season_dir = os.path.join(DATA_DIR, season)
    ensure_dir(season_dir)
    
    url = "https://stats.nba.com/stats/leaguedashptstats"
    
    for measure_name, api_param in TRACKING_MEASURES.items():
        outfile = os.path.join(season_dir, f"tracking_{measure_name}.parquet")
        if os.path.exists(outfile): continue
            
        print(f"   Fetching {measure_name}...", end=" ")
        
        params = {
            "College": "", "Conference": "", "Country": "", "DateFrom": "", "DateTo": "",
            "Division": "", "DraftPick": "", "DraftYear": "", "GameScope": "", "GameSegment": "",
            "Height": "", "LastNGames": "0", "LeagueID": "00", "Location": "",
            "Month": "0", "OpponentTeamID": "0", "Outcome": "", "PORound": "0",
            "PerMode": "PerGame", "Period": "0", "PlayerExperience": "",
            "PlayerOrTeam": "Player", "PlayerPosition": "", "PtMeasureType": api_param,
            "Season": season, "SeasonSegment": "", "SeasonType": "Regular Season",
            "StarterBench": "", "TeamID": "0", "VsConference": "", "VsDivision": "", "Weight": ""
        }
        
        df = get_nba_data(url, params, "drives") # 'drives' is a safe generic referer
        
        if df is not None and not df.empty:
            df.columns = [c.upper() for c in df.columns]
            df.to_parquet(outfile, index=False)
            print(f"✅ ({len(df)} rows)")
        else:
            print("❌ Failed or Empty")
        
        smart_sleep()

def fetch_synergy(season):
    print(f"\n🧠 Fetching Synergy Play Types for {season}...")
    season_dir = os.path.join(DATA_DIR, season)
    ensure_dir(season_dir)
    
    url = "https://stats.nba.com/stats/synergyplaytypes"
    
    # "Offensive" = Player stats, "Defensive" = How they defend it
    for side in ["Offensive", "Defensive"]:
        for ptype in SYNERGY_TYPES:
            filename = f"synergy_{side}_{ptype}.parquet"
            outfile = os.path.join(season_dir, filename)
            
            if os.path.exists(outfile): continue
                
            print(f"   Fetching {side} {ptype}...", end=" ")
            
            params = {
                "LeagueID": "00",
                "PerMode": "PerGame",
                "PlayType": ptype,
                "PlayerOrTeam": "P",
                "SeasonType": "Regular Season",
                "SeasonYear": season,
                "TypeGrouping": side
            }
            
            df = get_nba_data(url, params, "isolation")
            
            if df is not None and not df.empty:
                df.to_parquet(outfile, index=False)
                print(f"✅")
            else:
                print("⚠️ Empty/Skipped")
            
            smart_sleep()

def main():
    print("=== Starting Stream B: TLS Impersonation Fetch ===")
    
    for season in SEASONS:
        fetch_tracking(season)
        fetch_synergy(season)
        
    print("\n✅ Stream B Complete. Data saved.")

if __name__ == "__main__":
    main()