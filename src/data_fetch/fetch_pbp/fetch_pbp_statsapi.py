"""
src/data_fetch/fetch_pbp/fetch_pbp_statsapi.py
=============================================================================
Fetch NBA play-by-play via stats.nba.com/stats/playbyplayv3 using curl_cffi
TLS impersonation. Fallback for environments where NBA.com frontend and CDN
are unreachable (403/ERR_HTTP2).

API: stats.nba.com/stats/playbyplayv3?GameID={id}&StartPeriod=0&EndPeriod=14
Response: {"game": {"actions": [...]}}  (same structure as CDN JSON)

Usage:
  python3 src/data_fetch/fetch_pbp/fetch_pbp_statsapi.py --seasons 2021-22
  python3 src/data_fetch/fetch_pbp/fetch_pbp_statsapi.py --seasons 2021-22 2018-19
=============================================================================
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import pandas as pd
from curl_cffi import requests as cffi_requests

DATA_DIR = "data/historical"
PBP_CACHE_DIR = f"{DATA_DIR}/pbp_cache"
CACHE_FILE = f"{DATA_DIR}/pbp_fetched_statsapi.json"

SOURCE_CANDIDATES = [
    "data/historical/team_game_logs.parquet",
    "data/team_game_logs.parquet",
]

PBP_URL = (
    "https://stats.nba.com/stats/playbyplayv3"
    "?GameID={game_id}&StartPeriod=0&EndPeriod=14"
)

HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
    "Host": "stats.nba.com",
    "Origin": "https://www.nba.com",
    "Referer": "https://www.nba.com/stats/players/traditional",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
}

SLEEP_BETWEEN_GAMES = (1.0, 2.5)
SLEEP_ON_ERROR = (5.0, 10.0)
MAX_RETRIES = 3


# -------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------

def load_game_logs() -> pd.DataFrame:
    for p in SOURCE_CANDIDATES:
        if os.path.exists(p):
            return pd.read_parquet(p)
    raise FileNotFoundError("team_game_logs.parquet not found")


def load_cache() -> set:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            return set(json.load(f))
    return set()


def save_cache(cache: set):
    with open(CACHE_FILE, "w") as f:
        json.dump(sorted(cache), f)


def save_game_pbp(game_id: str, df: pd.DataFrame):
    Path(PBP_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    df.to_parquet(f"{PBP_CACHE_DIR}/pbp_{game_id}.parquet", index=False)


# -------------------------------------------------------------------
# Fetch one game
# -------------------------------------------------------------------

def fetch_game(game_id: str) -> pd.DataFrame | None:
    url = PBP_URL.format(game_id=game_id)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = cffi_requests.get(
                url, headers=HEADERS, impersonate="chrome120", timeout=30
            )
            if resp.status_code != 200:
                print(f"  HTTP {resp.status_code} — attempt {attempt}/{MAX_RETRIES}")
                time.sleep(random.uniform(*SLEEP_ON_ERROR))
                continue

            data = resp.json()
            actions = data.get("game", {}).get("actions", [])
            if not actions:
                return None

            df = pd.DataFrame(actions)
            rename_map = {
                "description": "DESCRIPTION",
                "period": "PERIOD",
                "clock": "PCTIMESTRING",
                "actionNumber": "EVENTNUM",
                "actionType": "EVENTMSGTYPE",
                "subType": "EVENTMSGACTIONTYPE",
                "personId": "PLAYER1_ID",
                "playerName": "PLAYER1_NAME",
                "teamTricode": "PLAYER1_TEAM_ABBREVIATION",
                "teamId": "PLAYER1_TEAM_ID",
            }
            df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
            df["GAME_ID"] = game_id
            if "DESCRIPTION" in df.columns and "PCTIMESTRING" in df.columns:
                df["RAW_TEXT"] = df["PCTIMESTRING"].astype(str) + "\n" + df["DESCRIPTION"].astype(str)
            return df

        except Exception as e:
            print(f"  Error attempt {attempt}/{MAX_RETRIES}: {e}")
            time.sleep(random.uniform(*SLEEP_ON_ERROR))

    return None


# -------------------------------------------------------------------
# Fetch season
# -------------------------------------------------------------------

def fetch_season(season: str, game_ids: list, cache: set) -> int:
    to_fetch = [g for g in game_ids if g not in cache]
    print(f"[{season}] {len(game_ids)} games total, {len(to_fetch)} to fetch")
    newly = 0

    for idx, gid in enumerate(to_fetch, 1):
        print(f"[{season}] {idx}/{len(to_fetch)} → {gid}", end=" ", flush=True)
        df = fetch_game(gid)

        if df is not None and not df.empty:
            save_game_pbp(gid, df)
            cache.add(gid)
            newly += 1
            print(f"✓ {len(df)} rows")
        else:
            print("✗ no data")

        if newly % 50 == 0 and newly > 0:
            save_cache(cache)

        time.sleep(random.uniform(*SLEEP_BETWEEN_GAMES))

    save_cache(cache)
    print(f"[{season}] Done — {newly} new games fetched")
    return newly


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main(seasons: list[str]):
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    Path(PBP_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    cache = load_cache()
    print(f"Cache: {len(cache)} games already fetched")

    games = load_game_logs()
    games["GAME_ID"] = games["GAME_ID"].astype(str).str.zfill(10)

    for season in seasons:
        print(f"\n=== Season {season} ===")
        season_games = games[games["SEASON"] == season]
        if season_games.empty:
            print(f"  No games in logs for {season}")
            continue

        game_ids = season_games["GAME_ID"].unique().tolist()
        fetch_season(season, game_ids, cache)

        # Combine cached game files into season parquet
        dfs = []
        for gid in game_ids:
            p = f"{PBP_CACHE_DIR}/pbp_{gid}.parquet"
            if os.path.exists(p):
                dfs.append(pd.read_parquet(p))

        if dfs:
            out = pd.concat(dfs, ignore_index=True)
            out_path = f"{DATA_DIR}/play_by_play_{season}.parquet"
            out.to_parquet(out_path, index=False)
            print(f"[{season}] Saved → {out_path} ({len(out):,} rows)")
        else:
            print(f"[{season}] No parquet files to combine")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", required=True)
    args = parser.parse_args()
    main(args.seasons)
