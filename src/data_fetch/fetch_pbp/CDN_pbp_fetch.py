"""
Fetch NBA play-by-play using the NBA.com CDN (Fast & Stable).

Usage:
  python src/data_fetch/fetch_pbp/fetch_play_by_play.py --seasons 2023-24 2024-25

Source:
  https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{game_id}.json
"""

import pandas as pd
import argparse
import time
import os
import json
import httpx
from pathlib import Path

# --- CONFIG ---
DATA_DIR = "data/historical"
PBP_CACHE_DIR = f"{DATA_DIR}/pbp_cache"
CACHE_FILE = f"{DATA_DIR}/pbp_fetched.json"

# The CDN is extremely fast and has no strict rate limits like the stats API.
BASE_URL = "https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{game_id}.json"

SOURCE_CANDIDATES = [
    "data/historical/team_game_logs.parquet",
    "data/team_game_logs.parquet",
]

# -----------------------------
# Utilities
# -----------------------------

def load_team_game_logs():
    for p in SOURCE_CANDIDATES:
        if os.path.exists(p):
            return pd.read_parquet(p)
    raise FileNotFoundError("team_game_logs.parquet not found. Run fetch_historical_data.py first.")

def load_cache() -> set:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return set(json.load(f))
    return set()

def save_cache(cache: set):
    with open(CACHE_FILE, "w") as f:
        json.dump(sorted(list(cache)), f)

def save_game_pbp(game_id: str, df: pd.DataFrame):
    Path(PBP_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    out = f"{PBP_CACHE_DIR}/pbp_{game_id}.parquet"
    df.to_parquet(out, index=False)

# -----------------------------
# Fetch Logic
# -----------------------------

def fetch_game_pbp(game_id: str, client: httpx.Client):
    url = BASE_URL.format(game_id=game_id)
    
    try:
        resp = client.get(url)
        
        # 404 means the game file doesn't exist (yet?) or invalid ID
        if resp.status_code == 404:
            print(f"  ⚠️ 404 Not Found: {game_id}")
            return None
        
        resp.raise_for_status()
        data = resp.json()
        
        # Extract actions list
        # Structure: { "game": { "actions": [ ... ] } }
        actions = data.get("game", {}).get("actions", [])
        
        if not actions:
            return None
            
        df = pd.DataFrame(actions)
        
        # RENAME columns to match your existing parser's expectations
        # Your parser looks for "RAW_TEXT" or "DESCRIPTION"
        rename_map = {
            "description": "DESCRIPTION",
            "period": "PERIOD",
            "clock": "clock",         # Keep original lower case if preferred, or rename
            "actionNumber": "EVENTNUM"
        }
        df = df.rename(columns=rename_map)
        
        # Ensure GAME_ID is attached
        df["GAME_ID"] = game_id
        
        # Add RAW_TEXT column for compatibility with your parser
        if "DESCRIPTION" in df.columns:
            df["RAW_TEXT"] = df["clock"] + "\n" + df["DESCRIPTION"]
        
        return df

    except Exception as e:
        print(f"  ❌ Error fetching {game_id}: {e}")
        return None

def fetch_season(season, game_ids, fetched_cache):
    newly_fetched = 0
    
    # Use a persistent client for connection pooling (much faster)
    with httpx.Client(timeout=10.0, http2=True) as client:
        
        total = len(game_ids)
        print(f"[{season}] Found {total} games. Checking cache...")
        
        # Filter out already fetched
        to_fetch = [gid for gid in game_ids if gid not in fetched_cache]
        print(f"[{season}] Need to fetch: {len(to_fetch)} games")

        for idx, gid in enumerate(to_fetch, 1):
            print(f"[{season}] Fetching {idx}/{len(to_fetch)} → {gid}", end="\r")
            
            df = fetch_game_pbp(gid, client)
            
            if df is not None and not df.empty:
                save_game_pbp(gid, df)
                fetched_cache.add(gid)
                newly_fetched += 1
                
                # Update cache file every 50 games to be safe
                if newly_fetched % 50 == 0:
                    save_cache(fetched_cache)
            
            # Tiny sleep to be polite, though CDN handles load well
            time.sleep(0.05) 
            
    print(f"\n[{season}] Finished. Fetched {newly_fetched} new games.")
    save_cache(fetched_cache)
    return newly_fetched

# -----------------------------
# Main
# -----------------------------

def main(seasons):
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    Path(PBP_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    fetched_cache = load_cache()
    
    # Load game list
    try:
        games = load_team_game_logs()
    except Exception as e:
        print(f"Error: {e}")
        return

    # Ensure Game IDs are strings (e.g. "0022300001")
    games["GAME_ID"] = games["GAME_ID"].astype(str).str.zfill(10)

    for season in seasons:
        print(f"\n--- Processing Season: {season} ---")
        season_games = games[games["SEASON"] == season]
        
        if season_games.empty:
            print(f"No games found in logs for season {season}")
            continue
            
        game_ids = season_games["GAME_ID"].unique().tolist()
        fetch_season(season, game_ids, fetched_cache)

        # Re-combine all cached files for this season into one parquet
        print(f"Combining parquet files for {season}...")
        dfs = []
        for gid in game_ids:
            p = f"{PBP_CACHE_DIR}/pbp_{gid}.parquet"
            if os.path.exists(p):
                dfs.append(pd.read_parquet(p))

        if dfs:
            out = pd.concat(dfs, ignore_index=True)
            out_path = f"{DATA_DIR}/play_by_play_{season}.parquet"
            out.to_parquet(out_path, index=False)
            print(f"✅ Season {season} saved to {out_path} ({len(out)} rows)")
        else:
            print(f"⚠️ No data found to combine for {season}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", required=True, help="Seasons to fetch (e.g. 2023-24)")
    args = parser.parse_args()
    main(args.seasons)