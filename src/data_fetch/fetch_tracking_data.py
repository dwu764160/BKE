"""
src/data_fetch/fetch_tracking_data.py
=============================================================================
Fetches "Style" and "Tracking" data using TLS Impersonation (curl_cffi).

v2.6 adds:
  - PostTouch, ElbowTouch, PaintTouch tracking measures
  - Additional defense dashboard categories (2PT, LT10, GT15)
  - Hustle stats (disruption-related metrics)
=============================================================================
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

from src.modeling.model_config import SEASONS

# --- CONFIGURATION ---

TRACKING_MEASURES = {
    "Drives": ("Drives", "drives"),
    "Defense": ("Defense", "defense"),
    "CatchShoot": ("CatchShoot", "catch-shoot"),
    "PullUpShot": ("PullUpShot", "pullup"),
    "Passing": ("Passing", "passing"),
    "Possessions": ("Possessions", "touches"),
    "Rebounding": ("Rebounding", "rebounding"),
    "Efficiency": ("Efficiency", "shooting-efficiency"),
    "SpeedDistance": ("SpeedDistance", "speed-distance"),
    # Note: PostTouch/ElbowTouch/PaintTouch data is included in the
    # Possessions tracking file (POST_TOUCHES, ELBOW_TOUCHES, PAINT_TOUCHES,
    # PTS_PER_POST_TOUCH, PTS_PER_ELBOW_TOUCH, PTS_PER_PAINT_TOUCH columns).
}

DEFENSE_CATEGORIES = {
    "Overall": "defense-dash-overall",
    "Less Than 6Ft": "defense-dash-lt6",
    "3 Pointers": "defense-dash-3pt",
    # v2.6: Additional defense dashboard categories
    "2 Pointers": "defense-dash-2pt",
    "Less Than 10Ft": "defense-dash-lt10",
    "Greater Than 15Ft": "defense-dash-gt15",
}

# leaguedashptstats now requires all optional filter params to be present.
# Sending only 6 minimal params causes HTTP 500; full set causes HTTP 200.
_PTSTATS_DEFAULTS = {
    "College": "", "Conference": "", "Country": "",
    "DateFrom": "", "DateTo": "", "Division": "",
    "DraftPick": "", "DraftYear": "", "GameScope": "",
    "GameSegment": "", "Height": "", "ISTRound": "",
    "LastNGames": 0, "Location": "", "Month": 0,
    "OpponentTeamID": 0, "Outcome": "", "PORound": 0,
    "Period": 0, "PlayerExperience": "", "PlayerPosition": "",
    "SeasonSegment": "", "StarterBench": "", "TeamID": 0,
    "VsConference": "", "VsDivision": "", "Weight": "",
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
                if isinstance(json_data, dict) and (
                    "resultSets" in json_data or "resultSet" in json_data
                ):
                    return parse_json(json_data)
        except:
            pass 

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
        resp = requests.get(
            url, params=params, headers=headers,
            impersonate="chrome124", timeout=30
        )
        
        if resp.status_code != 200:
            # Return None to signal failure (triggering fallback)
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
        if not isinstance(json_data, dict):
            return pd.DataFrame()

        payload = None
        result_sets = json_data.get("resultSets")
        if isinstance(result_sets, list) and result_sets:
            payload = result_sets[0]
        elif isinstance(json_data.get("resultSet"), dict):
            payload = json_data.get("resultSet")

        if not isinstance(payload, dict):
            return pd.DataFrame()

        headers = payload.get("headers", [])
        row_set = payload.get("rowSet", [])
        if not headers:
            return pd.DataFrame(row_set)
        return pd.DataFrame(row_set, columns=headers)
    except:
        return pd.DataFrame()

def fetch_fallback_catch_shoot(season):
    """
    Fallback: Uses '0 Dribbles' from leaguedashplayerptshot as proxy for Catch & Shoot.
    """
    url = "https://stats.nba.com/stats/leaguedashplayerptshot"
    params = {
        "LeagueID": "00", "PerMode": "PerGame", "PlayerOrTeam": "Player",
        "Season": season, "SeasonType": "Regular Season",
        "DribbleRange": "0 Dribbles" # The Proxy
    }
    
    print("   🚑 Using '0 Dribbles' Proxy...", end=" ")
    df = fetch_url_cached(url, params, "shots-dribbles", f"tracking_CatchShoot_Fallback_{season}")
    
    if df is not None and not df.empty:
        # RENAME columns to match standard CatchShoot format
        # e.g. FGM -> CATCH_SHOOT_FGM
        new_cols = {}
        for col in df.columns:
            if col in ['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'GP', 'G', 'MIN']:
                continue
            new_cols[col] = f"CATCH_SHOOT_{col}"
        
        df = df.rename(columns=new_cols)
        # Ensure we have the standard columns expected
        if 'CATCH_SHOOT_FG3M' in df.columns:
            return df
            
    return None

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
            **_PTSTATS_DEFAULTS,
            "LeagueID": "00", "PerMode": "PerGame", "PlayerOrTeam": "Player",
            "PtMeasureType": api_param, "Season": season, "SeasonType": "Regular Season",
        }

        df = fetch_url_cached(url, params, slug, cache_key)
        
        # --- FALLBACK LOGIC ---
        if (df is None or df.empty) and measure_name == "CatchShoot":
            df = fetch_fallback_catch_shoot(season)
        # ----------------------
        
        if df is not None and not df.empty:
            df.columns = [c.upper() for c in df.columns]
            df.to_parquet(outfile, index=False)
            print(f"✅ ({len(df)} rows)")
        else:
            print("❌ Empty/Failed")
        
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


def fetch_hustle_stats(season):
    """
    v2.6: Fetch hustle stats (LeagueHustleStatsPlayer).

    Includes: CONTESTED_SHOTS, DEFLECTIONS, CHARGES_DRAWN,
    SCREEN_ASSISTS, LOOSE_BALLS_RECOVERED, BOX_OUTS, etc.
    These are key inputs for disruption rate and defensive playmaking.
    """
    print(f"\n💪 Fetching Hustle Stats for {season}...")
    season_dir = DATA_DIR / season
    season_dir.mkdir(exist_ok=True)

    url = "https://stats.nba.com/stats/leaguehustlestatsplayer"

    for per_mode, suffix in [("PerGame", ""), ("Totals", "_totals")]:
        outfile = season_dir / f"hustle_stats{suffix}.parquet"
        cache_key = f"hustle_stats{suffix}_{season}"

        if outfile.exists():
            continue

        print(f"   Fetching hustle stats ({per_mode})...", end=" ")

        params = {
            "LeagueID": "00",
            "PerMode": per_mode,
            "Season": season,
            "SeasonType": "Regular Season",
        }

        df = fetch_url_cached(url, params, "hustle", cache_key)

        if df is not None and not df.empty:
            df.columns = [c.upper() for c in df.columns]
            df.to_parquet(outfile, index=False)
            print(f"✅ ({len(df)} rows)")
        else:
            print("❌ Empty/Failed")

        smart_sleep()

def main():
    print("=== Starting Stream B: Robust Fetch with Fallbacks ===")
    ensure_dirs()
    
    for season in SEASONS:
        fetch_tracking(season)
        fetch_defense_dashboard(season)
        fetch_synergy(season)
        fetch_hustle_stats(season)
        
    print("\n✅ Stream B Complete.")

if __name__ == "__main__":
    main()