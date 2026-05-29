"""
scripts/fetch_pbp_2025_26_statsapi.py
=============================================================================
Fetch 2025-26 play-by-play via stats.nba.com/stats/playbyplayv3.

The NBA CDN (liveData/) only serves active/recent seasons; 2025-26 data was
purged after the season ended. The stats.nba.com API is the durable historical
store for all seasons.

Output schema matches CDN_pbp_fetch.py so the existing normalization chain
(run_normalization.py → derive_lineups.py → derive_possessions.py) works
without modification.

Writes:
  data/historical/pbp_cache/pbp_{game_id}.parquet   (per-game)
  data/historical/play_by_play_2025-26.parquet       (combined season)

Usage:
  python3 scripts/fetch_pbp_2025_26_statsapi.py [--dry-run] [--limit N]
=============================================================================
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

TARGET_SEASON = "2025-26"
DATA_DIR = REPO / "data/historical"
PBP_CACHE_DIR = DATA_DIR / "pbp_cache"
GAME_LOGS_PATH = DATA_DIR / "team_game_logs.parquet"
SEASON_OUT_PATH = DATA_DIR / f"play_by_play_{TARGET_SEASON}.parquet"
FETCH_CACHE_PATH = DATA_DIR / "pbp_fetched_statsapi_2025_26.json"

PLAYBYPLAY_URL = "https://stats.nba.com/stats/playbyplayv3"

HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Connection": "keep-alive",
    "Origin": "https://www.nba.com",
    "Referer": "https://www.nba.com/",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}


def _curl_get(game_id: str):
    try:
        from curl_cffi import requests as curl_requests
    except ImportError:
        raise ImportError("curl_cffi required: pip install curl_cffi")

    params = {
        "GameID": game_id,
        "StartPeriod": 0,
        "EndPeriod": 0,
        "RangeType": 0,
        "StartRange": 0,
        "EndRange": 0,
    }
    resp = curl_requests.get(
        PLAYBYPLAY_URL,
        params=params,
        headers=HEADERS,
        impersonate="chrome110",
        timeout=30,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
    return resp.json()


def _actions_to_df(actions: list, game_id: str) -> pd.DataFrame:
    """Map playbyplayv3 actions to the CDN PBP schema expected by pbp_parser.py."""
    rows = []
    for a in actions:
        clock = str(a.get("clock", ""))
        desc = str(a.get("description", ""))
        row = {
            # Keys pbp_parser.normalize_pbp_row reads directly
            "GAME_ID":     game_id,
            "PERIOD":      a.get("period"),
            "clock":       clock,
            "scoreHome":   a.get("scoreHome"),
            "scoreAway":   a.get("scoreAway"),
            "DESCRIPTION": desc,
            "personId":    a.get("personId"),
            "playerName":  a.get("playerName"),
            "playerNameI": a.get("playerNameI"),
            "teamId":      a.get("teamId"),
            "teamTricode": a.get("teamTricode"),
            # RAW_TEXT gives pbp_parser a fallback if DESCRIPTION is empty
            "RAW_TEXT":    f"{clock}\n{desc}".strip(),
            # Pass-through fields from stats API
            "actionNumber":    a.get("actionNumber"),
            "orderNumber":     a.get("orderNumber"),
            "actionType":      a.get("actionType"),
            "subType":         a.get("subType"),
            "qualifiers":      a.get("qualifiers"),
            "isFieldGoal":     a.get("isFieldGoal"),
            "shotResult":      a.get("shotResult"),
            "shotDistance":    a.get("shotDistance"),
            "assistPersonId":  a.get("assistPersonId"),
            "assistPlayerNameInitial": a.get("assistPlayerNameInitial"),
            "blockPersonId":   a.get("blockPersonId"),
            "blockPlayerName": a.get("blockPlayerName"),
            "stealPersonId":   a.get("stealPersonId"),
            "stealPlayerName": a.get("stealPlayerName"),
            "foulDrawnPersonId":  a.get("foulDrawnPersonId"),
            "foulDrawnPlayerName": a.get("foulDrawnPlayerName"),
            "reboundTotal":    a.get("reboundTotal"),
            "turnoverTotal":   a.get("turnoverTotal"),
            "pointsTotal":     a.get("pointsTotal"),
            "side":            a.get("side"),
            "area":            a.get("area"),
            "areaDetail":      a.get("areaDetail"),
            "x":               a.get("x"),
            "y":               a.get("y"),
            "possession":      a.get("possession"),
            "edited":          a.get("edited"),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def fetch_game(game_id: str, dry_run: bool = False) -> pd.DataFrame | None:
    cache_path = PBP_CACHE_DIR / f"pbp_{game_id}.parquet"
    if cache_path.exists():
        return None  # Already cached

    try:
        data = _curl_get(game_id)
    except Exception as e:
        print(f"  ✗ {game_id}: {e}")
        return None

    actions = data.get("game", {}).get("actions", [])
    if not actions:
        print(f"  ✗ {game_id}: empty actions")
        return None

    df = _actions_to_df(actions, game_id)

    if not dry_run:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)

    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Max games to fetch (testing)")
    parser.add_argument("--delay", type=float, default=1.2, help="Seconds between requests")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Fetching {TARGET_SEASON} PBP via stats.nba.com/stats/playbyplayv3")
    print("=" * 60)

    if not GAME_LOGS_PATH.exists():
        print(f"ERROR: {GAME_LOGS_PATH} not found")
        return

    gl = pd.read_parquet(GAME_LOGS_PATH)
    gl26 = gl[gl["SEASON"] == TARGET_SEASON]
    game_ids = sorted(gl26["GAME_ID"].unique().tolist())
    print(f"Game IDs from game logs: {len(game_ids)}")

    # Skip already-cached games
    already = {p.stem.replace("pbp_", "") for p in PBP_CACHE_DIR.glob("pbp_002250*.parquet")}
    to_fetch = [g for g in game_ids if g not in already]
    print(f"Already cached: {len(already)} | To fetch: {len(to_fetch)}")

    if args.limit:
        to_fetch = to_fetch[: args.limit]
        print(f"Limit applied: fetching {len(to_fetch)}")

    if not to_fetch:
        print("Nothing to fetch — combining cached files.")
    else:
        newly_fetched = 0
        errors = 0
        for idx, gid in enumerate(to_fetch, 1):
            print(f"  [{idx}/{len(to_fetch)}] {gid}", end="\r")
            result = fetch_game(gid, dry_run=args.dry_run)
            if result is not None:
                newly_fetched += 1
            else:
                if not (PBP_CACHE_DIR / f"pbp_{gid}.parquet").exists():
                    errors += 1
            time.sleep(args.delay)

        print(f"\nFetched: {newly_fetched} new games | Errors: {errors}")

    if args.dry_run:
        print("[DRY RUN] Skipping season file combine.")
        return

    # Combine all cached files for this season into one parquet
    print(f"\nCombining all {TARGET_SEASON} games into {SEASON_OUT_PATH.name}...")
    parts = []
    for gid in game_ids:
        p = PBP_CACHE_DIR / f"pbp_{gid}.parquet"
        if p.exists():
            parts.append(pd.read_parquet(p))

    if parts:
        combined = pd.concat(parts, ignore_index=True)
        combined.to_parquet(SEASON_OUT_PATH, index=False)
        print(f"✓ Saved {len(combined)} rows across {len(parts)} games to {SEASON_OUT_PATH}")
    else:
        print("✗ No games found to combine")


if __name__ == "__main__":
    main()
