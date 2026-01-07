"""
src/data_fetch/fetch_tracking_data.py
Fetches "Style" and "Tracking" data using TLS Impersonation (curl_cffi).
UPDATED: Implements Strict Referer Matching to fix Status 500 errors.
"""

import pandas as pd
import time
import os
import sys
import json
import random
from curl_cffi import requests
from pathlib import Path

# Adjust path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

DATA_DIR = Path("data/tracking")
CACHE_DIR = Path("data/tracking_cache")
SEASONS = ["2022-23", "2023-24", "2024-25"]

# --- CONFIGURATION ---

# Map API Parameter -> Website URL Slug (Strict Referer Matching)
TRACKING_MEASURES = {
    "Drives": ("Drives", "drives"),
    "CatchShoot": ("CatchShoot", "catch-shoot"),
    "PullUpShot": ("PullUpShot", "pullup"),
    "Passing": ("Passing", "passing"),
    "Possessions": ("Possessions", "touches"),
    "Rebounding": ("Rebounding", "rebounding"),
    "Efficiency": ("Efficiency", "shooting-efficiency"),
    "SpeedDistance": ("SpeedDistance", "speed-distance")
}

SYNERGY_TYPES = [
    "Isolation", "PRBallHandler", "PRRollman", 
    "Postup", "Spotup", "Handoff", "OffScreen"
]

# Map Category -> Slug
DEFENSE_CATEGORIES = {
    "Overall": "defense-dash-overall",
    "Less Than 6Ft": "defense-dash-lt6",
    "3 Pointers": "defense-dash-3pt"
}

def ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

def smart_sleep():
    time.sleep(random.uniform(1.0, 2.5))

def fetch_url_cached(url, params, referer_suffix, cache_name):
    cache_path = CACHE_DIR / f"{cache_name}.json"
    
    if cache_path.exists():
        try:
            with open(cache_path, "r") as f:
                json_data = json.load(f)
                # Valid cache check
                if 'resultSets' in json_data:
                    return parse_json(json_data)
        except:
            pass # corrupted cache, re-fetch

    # Fetch
    headers = {
        'Accept': 'application/json, text/plain, */*',
        'Connection': 'keep-alive',
        'Origin': 'https://www.nba.com',
        'Referer': f'https://www.nba.com/stats/players/{referer_suffix}',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'x-nba-stats-origin': 'stats',
        'x-nba-stats-token': 'true',
    }
    
    try:
        # Debug URL printing for failed requests
        # print(f"DEBUG URL: {url}?{requests.compat.urlencode(params)}")
        
        resp = requests.get(
            url, params=params, headers=headers, 
            impersonate="chrome110", timeout=30
        )
        
        if resp.status_code != 200:
            print(f"⚠️ Status {resp.status_code}", end=" ")
            return None
        
        json_data = resp.json()
        
        with open(cache_path, "w") as f:
            json.dump(json_data, f)
            
        return parse_json(json_data)
            
    except Exception as e:
        print(f"⚠️ Error: {e}", end=" ")
        return None

def parse_json(json_data):
    try:
        result_sets = json_data.get('resultSets', [])
        if not result_sets: return pd.DataFrame()
        headers = result_sets[0]['headers']
        row_set = result_sets[0]['rowSet']
        return pd.DataFrame(row_set, columns=headers)
    except:
        return pd.DataFrame()

def fetch_tracking(season):
    print(f"\n🏀 Fetching Tracking Data (PtStats) for {season}...")
    season_dir = DATA_DIR / season
    season_dir.mkdir(exist_ok=True)
    
    url = "https://stats.nba.com/stats/leaguedashptstats"
    
    for measure_name, (api_param, slug) in TRACKING_MEASURES.items():
        outfile = season_dir / f"tracking_{measure_name}.parquet"
        cache_key = f"tracking_{measure_name}_{season}"
        
        if outfile.exists(): continue
            
        print(f"   Fetching {measure_name}...", end=" ")
        
        params = {
            "LeagueID": "00", "PerMode": "PerGame", "PlayerOrTeam": "Player",
            "PtMeasureType": api_param, "Season": season, "SeasonType": "Regular Season"
        }
        
        df = fetch_url_cached(url, params, slug, cache_key)
        
        if df is not None and not df.empty:
            df.columns = [c.upper() for c in df.columns]
            df.to_parquet(outfile, index=False)
            print(f"✅ ({len(df)} rows)")
        else:
            print("❌ Empty")
        
        smart_sleep()

def fetch_defense_dashboard(season):
    print(f"\n🛡️ Fetching Defense Dashboard for {season}...")
    season_dir = DATA_DIR / season
    season_dir.mkdir(exist_ok=True)
    
    url = "https://stats.nba.com/stats/leaguedashptdefend"
    
    for category, slug in DEFENSE_CATEGORIES.items():
        cat_file = category.replace(" ", "").replace("<", "Lt")
        outfile = season_dir / f"defense_{cat_file}.parquet"
        cache_key = f"defense_{cat_file}_{season}"
        
        if outfile.exists(): continue
            
        print(f"   Fetching {category}...", end=" ")
        
        params = {
            "LeagueID": "00", "PerMode": "PerGame", "DefenseCategory": category,
            "Season": season, "SeasonType": "Regular Season"
        }
        
        df = fetch_url_cached(url, params, slug, cache_key)
        
        if df is not None and not df.empty:
            df.columns = [c.upper() for c in df.columns]
            df.to_parquet(outfile, index=False)
            print(f"✅ ({len(df)} rows)")
        else:
            print("❌ Empty")
            
        smart_sleep()

def fetch_synergy(season):
    print(f"\n🧠 Fetching Synergy Play Types for {season}...")
    season_dir = DATA_DIR / season
    season_dir.mkdir(exist_ok=True)
    
    url = "https://stats.nba.com/stats/synergyplaytypes"
    
    OFFENSIVE_TYPES = [
        "Isolation", "Transition", "PRBallHandler", "PRRollman", 
        "Postup", "Spotup", "Handoff", "Cut", "OffScreen", 
        "OffRebound", "Misc"
    ]
    
    DEFENSIVE_TYPES = [
        "Isolation", "PRBallHandler", "PRRollman", 
        "Postup", "Spotup", "Handoff", "OffScreen"
    ]
    
    for side in ["Offensive", "Defensive"]:
        target_types = OFFENSIVE_TYPES if side == "Offensive" else DEFENSIVE_TYPES
        
        for ptype in target_types:
            filename = f"synergy_{side}_{ptype}.parquet"
            outfile = season_dir / filename
            cache_key = f"synergy_{side}_{ptype}_{season}"
            
            if outfile.exists(): continue
                
            print(f"   Fetching {side} {ptype}...", end=" ")
            
            params = {
                "LeagueID": "00", "PerMode": "PerGame", "PlayType": ptype,
                "PlayerOrTeam": "P", "SeasonType": "Regular Season",
                "SeasonYear": season, "TypeGrouping": side
            }
            
            df = fetch_url_cached(url, params, "isolation", cache_key)
            
            if df is not None and not df.empty:
                df.to_parquet(outfile, index=False)
                print(f"✅")
            else:
                print("⚠️ Empty/Skipped")
            
            smart_sleep()

def main():
    print("=== Starting Stream B: Strict Referer Fetch ===")
    ensure_dirs()
    
    for season in SEASONS:
        fetch_tracking(season)
        fetch_defense_dashboard(season)
        fetch_synergy(season)
        
    print("\n✅ Stream B Complete.")

if __name__ == "__main__":
    main()