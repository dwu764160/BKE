"""
src/data_fetch/fetch_player_clutch_stats.py
=============================================================================
Fetch player clutch statistics from stats.nba.com (LeagueDashPlayerClutch).

Clutch definition used:
  - ClutchTime: Last 5 Minutes
  - PointDiff: 7  (score margin <= 7)
  - AheadBehind: Ahead or Behind
  - SeasonType: Regular Season

Inputs:
  - Optional cached headers: data/nba_headers.json

Outputs:
  - data/historical/player_clutch_stats_{season}.parquet
  - data/historical/player_clutch_stats_all.parquet
  - data/tracking_cache/player_clutch_stats_{season}.json (raw cache)

Usage:
  python3 src/data_fetch/fetch_player_clutch_stats.py
  python3 src/data_fetch/fetch_player_clutch_stats.py --seasons 2024-25 --force
=============================================================================
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from curl_cffi import requests


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "historical"
CACHE_DIR = ROOT_DIR / "data" / "tracking_cache"
HEADERS_PATH = ROOT_DIR / "data" / "nba_headers.json"

import sys
sys.path.insert(0, str(ROOT_DIR))
from src.modeling.model_config import SEASONS
ENDPOINT = "https://stats.nba.com/stats/leaguedashplayerclutch"

DEFAULT_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Connection": "keep-alive",
    "Origin": "https://www.nba.com",
    "Referer": "https://www.nba.com/stats/players/clutch-traditional",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}

REQUIRED_COLS = [
    "PLAYER_ID",
    "PLAYER_NAME",
    "TEAM_ID",
    "TEAM_ABBREVIATION",
    "GP",
    "MIN",
    "PTS",
    "AST",
    "REB",
    "PLUS_MINUS",
]


def _normalize_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True)


def _load_headers() -> Dict[str, str]:
    headers = dict(DEFAULT_HEADERS)
    if not HEADERS_PATH.exists():
        return headers

    try:
        payload = json.loads(HEADERS_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"  [WARN] Could not parse {HEADERS_PATH}: {exc}")
        return headers

    allow = {
        "accept",
        "accept-language",
        "connection",
        "origin",
        "referer",
        "user-agent",
        "x-nba-stats-origin",
        "x-nba-stats-token",
    }
    for k, v in payload.items():
        if k.lower() in allow and isinstance(v, str) and v.strip():
            headers[k] = v.strip()

    # Keep these hard requirements explicit.
    headers["x-nba-stats-origin"] = "stats"
    headers["x-nba-stats-token"] = "true"
    return headers


def _params_for_season(season: str) -> Dict[str, str]:
    return {
        "AheadBehind": "Ahead or Behind",
        "ClutchTime": "Last 5 Minutes",
        "College": "",
        "Conference": "",
        "Country": "",
        "DateFrom": "",
        "DateTo": "",
        "Division": "",
        "DraftPick": "",
        "DraftYear": "",
        "GameScope": "",
        "GameSegment": "",
        "Height": "",
        "LastNGames": "0",
        "LeagueID": "00",
        "Location": "",
        "MeasureType": "Base",
        "Month": "0",
        "OpponentTeamID": "0",
        "Outcome": "",
        "PORound": "0",
        "PaceAdjust": "N",
        "PerMode": "Totals",
        "Period": "0",
        "PlayerExperience": "",
        "PlayerPosition": "",
        "PlusMinus": "N",
        "PointDiff": "7",
        "Rank": "N",
        "Season": season,
        "SeasonSegment": "",
        "SeasonType": "Regular Season",
        "ShotClockRange": "",
        "StarterBench": "",
        "TeamID": "0",
        "VsConference": "",
        "VsDivision": "",
        "Weight": "",
    }


def _extract_result_frame(payload: Dict) -> pd.DataFrame:
    if not isinstance(payload, dict):
        return pd.DataFrame()

    result_sets = payload.get("resultSets")
    if isinstance(result_sets, list) and result_sets:
        block = result_sets[0]
        return pd.DataFrame(block.get("rowSet", []), columns=block.get("headers", []))

    result_set = payload.get("resultSet")
    if isinstance(result_set, dict):
        return pd.DataFrame(result_set.get("rowSet", []), columns=result_set.get("headers", []))

    return pd.DataFrame()


def _load_cached_payload(cache_path: Path) -> Dict | None:
    if not cache_path.exists():
        return None
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _fetch_payload(season: str, headers: Dict[str, str]) -> Dict:
    params = _params_for_season(season)
    response = requests.get(
        ENDPOINT,
        params=params,
        headers=headers,
        impersonate="chrome110",
        timeout=45,
    )
    if response.status_code != 200:
        raise RuntimeError(f"status={response.status_code}")
    return response.json()


def _normalize_df(df: pd.DataFrame, season: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "player_id",
                "player_name",
                "season",
                "team_id",
                "team_abbreviation",
                "clutch_gp",
                "clutch_minutes",
                "clutch_pts",
                "clutch_ast",
                "clutch_reb",
                "clutch_plus_minus",
            ]
        )

    out = df.copy()
    out.columns = [str(c).upper() for c in out.columns]

    for col in REQUIRED_COLS:
        if col not in out.columns:
            out[col] = 0 if col in {"GP", "MIN", "PTS", "AST", "REB", "PLUS_MINUS"} else ""

    for col in ["GP", "MIN", "PTS", "AST", "REB", "PLUS_MINUS"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    normalized = pd.DataFrame(
        {
            "player_id": _normalize_id(out["PLAYER_ID"]),
            "player_name": out["PLAYER_NAME"].astype(str),
            "season": season,
            "team_id": _normalize_id(out["TEAM_ID"]),
            "team_abbreviation": out["TEAM_ABBREVIATION"].astype(str).str.upper(),
            "clutch_gp": out["GP"].astype(float),
            "clutch_minutes": out["MIN"].astype(float),
            "clutch_pts": out["PTS"].astype(float),
            "clutch_ast": out["AST"].astype(float),
            "clutch_reb": out["REB"].astype(float),
            "clutch_plus_minus": out["PLUS_MINUS"].astype(float),
        }
    )

    # For traded players, keep the team row with the largest clutch-minute sample.
    normalized = normalized.sort_values(
        ["clutch_minutes", "clutch_gp", "clutch_pts"],
        ascending=False,
    ).drop_duplicates(subset=["player_id", "season"], keep="first")

    normalized = normalized.sort_values(
        ["season", "team_abbreviation", "clutch_minutes"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    return normalized


def _season_paths(season: str) -> Tuple[Path, Path]:
    out_path = DATA_DIR / f"player_clutch_stats_{season}.parquet"
    cache_path = CACHE_DIR / f"player_clutch_stats_{season}.json"
    return out_path, cache_path


def fetch_season(season: str, headers: Dict[str, str], force: bool = False) -> pd.DataFrame:
    out_path, cache_path = _season_paths(season)

    payload = None
    if not force:
        payload = _load_cached_payload(cache_path)

    if payload is None:
        try:
            payload = _fetch_payload(season, headers)
            cache_path.write_text(json.dumps(payload), encoding="utf-8")
        except Exception as exc:
            payload = _load_cached_payload(cache_path)
            if payload is None:
                raise RuntimeError(f"Could not fetch season {season}: {exc}") from exc
            print(f"  [WARN] Season {season}: request failed, used cached payload")

    raw_df = _extract_result_frame(payload)
    normalized = _normalize_df(raw_df, season)
    normalized.to_parquet(out_path, index=False)

    total_minutes = float(normalized["clutch_minutes"].sum()) if not normalized.empty else 0.0
    print(
        f"  {season}: rows={len(normalized)}, "
        f"players={normalized['player_id'].nunique() if not normalized.empty else 0}, "
        f"total_clutch_min={total_minutes:.1f}"
    )
    print(f"    saved -> {out_path}")
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch player clutch stats by season")
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=SEASONS,
        help="Seasons in YYYY-YY format (default: 2022-23 2023-24 2024-25)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore existing cache and force fresh API fetches",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seasons = list(dict.fromkeys([str(s).strip() for s in args.seasons if str(s).strip()]))

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching player clutch stats (LeagueDashPlayerClutch)")
    print("  definition: Last 5 Minutes, Ahead or Behind, PointDiff <= 7")

    headers = _load_headers()
    frames: List[pd.DataFrame] = []
    for season in seasons:
        frames.append(fetch_season(season, headers=headers, force=args.force))

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    combined_path = DATA_DIR / "player_clutch_stats_all.parquet"
    combined.to_parquet(combined_path, index=False)
    print(f"Saved combined clutch stats -> {combined_path} (rows={len(combined)})")


if __name__ == "__main__":
    main()
