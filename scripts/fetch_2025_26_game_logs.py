"""
scripts/fetch_2025_26_game_logs.py
=============================================================================
Fetches 2025-26 regular season team game logs from NBA Stats API and appends
to data/historical/team_game_logs.parquet.

Why: team_game_logs.parquet currently ends at 2024-25. Without 2025-26 game
logs, Step 4 YTD blending cannot be applied to the Kalshi CLV test (we'd have
projections but no in-season actuals to blend toward).

The 2025-26 regular season is complete (May 2026). This is a one-time fetch.

Usage:
    python3 scripts/fetch_2025_26_game_logs.py [--dry-run]
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
GAME_LOGS_PATH = REPO / "data/historical/team_game_logs.parquet"
NBA_HEADERS_PATH = REPO / "data/nba_headers.json"

LEAGUEGAMELOG_URL = "https://stats.nba.com/stats/leaguegamelog"


def _load_headers() -> dict:
    defaults = {
        "Accept": "application/json, text/plain, */*",
        "Connection": "keep-alive",
        "Origin": "https://www.nba.com",
        "Referer": "https://www.nba.com/stats/teams/traditional",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "x-nba-stats-origin": "stats",
        "x-nba-stats-token": "true",
    }
    if NBA_HEADERS_PATH.exists():
        try:
            payload = json.loads(NBA_HEADERS_PATH.read_text())
            cand = payload.get("headers", payload) if isinstance(payload, dict) else {}
            if isinstance(cand, dict):
                defaults.update({k: v for k, v in cand.items() if isinstance(k, str) and isinstance(v, str)})
        except Exception:
            pass
    return defaults


def fetch_season(season: str) -> pd.DataFrame:
    """Fetch one full season of team game logs from stats.nba.com."""
    try:
        from curl_cffi import requests as curl_requests
    except ImportError:
        raise ImportError("curl_cffi required: pip install curl_cffi")

    params = {
        "Counter": "0",
        "DateFrom": "",
        "DateTo": "",
        "Direction": "ASC",
        "LeagueID": "00",
        "PlayerOrTeam": "T",
        "Season": season,
        "SeasonType": "Regular Season",
        "Sorter": "DATE",
    }
    print(f"  Fetching {LEAGUEGAMELOG_URL} for season={season}...")
    resp = curl_requests.get(
        LEAGUEGAMELOG_URL,
        params=params,
        headers=_load_headers(),
        impersonate="chrome110",
        timeout=90,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")

    data = resp.json()
    if "resultSets" in data and data["resultSets"]:
        rs = data["resultSets"][0]
    elif "resultSet" in data:
        rs = data["resultSet"]
    else:
        raise RuntimeError(f"Unexpected JSON format: {list(data.keys())}")

    df = pd.DataFrame(rs.get("rowSet", []), columns=rs.get("headers", []))
    print(f"  Got {len(df)} raw rows, {df['GAME_ID'].nunique() if 'GAME_ID' in df.columns else '?'} unique games")
    return df


def normalize(df: pd.DataFrame, season: str) -> pd.DataFrame:
    """Normalize raw API response to match team_game_logs schema."""
    out = df.copy()

    rename = {"Game_ID": "GAME_ID", "Team_ID": "TEAM_ID"}
    out = out.rename(columns=rename)

    out["SEASON"] = season

    for col in ("GAME_ID",):
        if col in out.columns:
            out[col] = out[col].astype(str)
    if "TEAM_ID" in out.columns:
        out["TEAM_ID"] = pd.to_numeric(out["TEAM_ID"], errors="coerce").round().astype("Int64")
    if "GAME_DATE" in out.columns:
        out["GAME_DATE"] = pd.to_datetime(out["GAME_DATE"]).dt.strftime("%Y-%m-%d")

    for col in ("PTS", "PLUS_MINUS", "MIN"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["GAME_ID", "TEAM_ID", "PTS"]).copy()
    out["TEAM_ID"] = out["TEAM_ID"].astype(int)

    # Compute OPP_PTS by joining each game's two team rows
    base = out[["SEASON", "GAME_ID", "TEAM_ID", "PTS"]].copy()
    opp = base.rename(columns={"TEAM_ID": "OPP_TEAM_ID", "PTS": "OPP_PTS"})
    paired = base.merge(opp, on=["SEASON", "GAME_ID"]).query("TEAM_ID != OPP_TEAM_ID")
    paired = paired.sort_values(["SEASON", "GAME_ID", "TEAM_ID", "OPP_PTS"], ascending=[True, True, True, False])
    paired = paired.drop_duplicates(subset=["SEASON", "GAME_ID", "TEAM_ID"], keep="first")

    out = out.drop(columns=["OPP_PTS"], errors="ignore")
    out = out.merge(paired[["SEASON", "GAME_ID", "TEAM_ID", "OPP_PTS"]], on=["SEASON", "GAME_ID", "TEAM_ID"], how="left")
    out["OPP_PTS"] = pd.to_numeric(out["OPP_PTS"], errors="coerce")
    out = out.dropna(subset=["OPP_PTS"]).copy()
    out["OPP_PTS"] = out["OPP_PTS"].astype(int)
    out["PTS"] = out["PTS"].astype(int)
    out["margin"] = out["PTS"] - out["OPP_PTS"]

    out = out.drop_duplicates(subset=["SEASON", "GAME_ID", "TEAM_ID"]).copy()
    return out


def append_to_logs(new_df: pd.DataFrame, dry_run: bool = False) -> None:
    """Merge new season rows into team_game_logs.parquet."""
    if not GAME_LOGS_PATH.exists():
        print(f"  WARNING: {GAME_LOGS_PATH} not found. Creating from scratch.")
        existing = pd.DataFrame()
    else:
        existing = pd.read_parquet(GAME_LOGS_PATH)
        print(f"  Existing logs: {len(existing)} rows, seasons: {sorted(existing['SEASON'].unique())}")

    # Drop any existing rows for this season (idempotent re-run)
    existing = existing[existing["SEASON"] != TARGET_SEASON].copy() if not existing.empty else existing

    # Align columns — keep all columns from existing, fill NaN for extras in new
    if not existing.empty:
        combined = pd.concat([existing, new_df], ignore_index=True, sort=False)
    else:
        combined = new_df.copy()

    print(f"  Combined: {len(combined)} rows, seasons: {sorted(combined['SEASON'].unique())}")

    if dry_run:
        print("  [DRY RUN] Would save — skipping write.")
        return

    GAME_LOGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(GAME_LOGS_PATH, index=False)
    print(f"  Saved to {GAME_LOGS_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Fetch and normalize but don't write")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Fetching {TARGET_SEASON} regular season game logs")
    print("=" * 60)

    raw = fetch_season(TARGET_SEASON)
    print(f"\nNormalizing {len(raw)} raw rows...")
    clean = normalize(raw, TARGET_SEASON)
    print(f"  Normalized: {len(clean)} team-game rows")
    print(f"  Teams: {sorted(clean['TEAM_ABBREVIATION'].unique()) if 'TEAM_ABBREVIATION' in clean.columns else '?'}")
    print(f"  Date range: {clean['GAME_DATE'].min()} to {clean['GAME_DATE'].max()}")

    # Quick sanity
    expected_games = 1230  # 30 teams × 82 games / 2 = 1230 home games → 2460 team-game rows
    if len(clean) < expected_games:
        print(f"  WARNING: Only {len(clean)} rows — expected ~2460 for a full season")

    b2b_check = (clean.groupby("TEAM_ID").size() <= 82).all()
    if not b2b_check:
        print("  WARNING: Some teams have >82 games — check for playoffs leaking in")

    print(f"\nAppending to {GAME_LOGS_PATH}...")
    append_to_logs(clean, dry_run=args.dry_run)

    if not args.dry_run:
        print("\nDone. Run scripts/build_ytd_team_ratings.py next.")


if __name__ == "__main__":
    main()
