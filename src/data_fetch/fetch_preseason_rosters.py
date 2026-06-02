"""
src/data_fetch/fetch_preseason_rosters.py
=============================================================================
Fetch preseason roster snapshots (Option B) using CommonTeamRoster.

The endpoint is season-scoped and team-scoped; this script pulls all teams for
requested seasons and stores:
  - a combined snapshot parquet
  - one per-season parquet for fast lookup in projection scripts

Inputs:
  data/historical/teams.parquet

Outputs:
  data/historical/preseason_rosters.parquet
  data/historical/preseason_rosters/preseason_rosters_<season>.parquet
  reports/preseason_rosters_report.json

Usage:
  python3 src/data_fetch/fetch_preseason_rosters.py --season 2025-26
  python3 src/data_fetch/fetch_preseason_rosters.py --seasons 2023-24,2024-25,2025-26
=============================================================================
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
from nba_api.stats.endpoints import commonteamroster

import random
try:
    from curl_cffi import requests
except Exception:
    import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.player_eval.constants import (  # noqa: E402
    HISTORICAL_DIR,
    PRESEASON_ROSTERS_DIR,
    PRESEASON_ROSTERS_PATH,
    PRESEASON_ROSTERS_REPORT,
)
from src.data.schema_contract import load_standardized, save_standardized


def _norm_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()


def _load_teams() -> pd.DataFrame:
    teams_path = HISTORICAL_DIR / "teams.parquet"
    if not teams_path.exists():
        raise FileNotFoundError(f"Missing teams file: {teams_path}")

    teams = load_standardized(teams_path)
    cols = {c.lower(): c for c in teams.columns}
    tid_col = cols.get("team_id") or cols.get("id")
    abbr_col = cols.get("abbreviation")
    full_col = cols.get("full_name")
    if not tid_col or not abbr_col:
        raise ValueError("teams.parquet is missing team_id/id or abbreviation columns")

    out = pd.DataFrame(
        {
            "team_id": _norm_id(teams[tid_col]),
            "team_abbreviation": teams[abbr_col].astype(str).str.upper().str.strip(),
            "team_name": teams[full_col].astype(str).str.strip() if full_col else "",
        }
    )
    out = out[out["team_abbreviation"].str.fullmatch(r"[A-Z]{3}", na=False)].copy()
    out = out.drop_duplicates(subset=["team_id", "team_abbreviation"], keep="first")
    return out


def _fetch_team_roster(team_id: str, season: str, retries: int = 3, timeout: int = 60) -> pd.DataFrame:
    """Try nba_api CommonTeamRoster with retries; fall back to direct HTTP fetch if that fails.

    The HTTP fallback uses `stats.nba.com` with browser-like headers and a simple
    per-team JSON cache to improve resilience against transient network failures.
    """
    # Prefer HTTP fallback (browser headers + caching) — faster and more resilient
    try:
        df = _fetch_team_roster_http(team_id=team_id, season=season, timeout=timeout, retries=retries)
        if df is not None and not df.empty:
            return df
    except Exception:
        pass

    # If HTTP fails, fall back to nba_api CommonTeamRoster with exponential backoff
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            # increase timeout slightly on each attempt
            attempt_timeout = timeout + (attempt - 1) * 10
            endpoint = commonteamroster.CommonTeamRoster(
                team_id=int(team_id),
                season=season,
                league_id_nullable="00",
                timeout=attempt_timeout,
            )
            frames = endpoint.get_data_frames()
            if not frames:
                return pd.DataFrame()
            roster = frames[0].copy()
            if roster.empty:
                return roster
            return roster
        except Exception as exc:
            last_err = exc
            wait = (2 ** attempt) + random.uniform(0, 1)
            time.sleep(wait)

    print(f"  [WARN] Team roster fetch failed for team_id={team_id}, season={season}: {last_err}")
    return pd.DataFrame()


def _parse_result_payload(json_data: dict) -> pd.DataFrame:
    """Parse a stats.nba.com JSON payload into a DataFrame (robust to shapes)."""
    if not isinstance(json_data, dict):
        return pd.DataFrame()

    # Prefer resultSets list
    result_sets = json_data.get("resultSets")
    if isinstance(result_sets, list) and result_sets:
        for table in result_sets:
            headers = table.get("headers") or []
            rows = table.get("rowSet") or []
            if headers and rows:
                try:
                    return pd.DataFrame(rows, columns=headers)
                except Exception:
                    return pd.DataFrame(rows)

    # Fallback to resultSet dict
    result_set = json_data.get("resultSet")
    if isinstance(result_set, dict):
        headers = result_set.get("headers") or []
        rows = result_set.get("rowSet") or []
        if headers and rows:
            try:
                return pd.DataFrame(rows, columns=headers)
            except Exception:
                return pd.DataFrame(rows)

    return pd.DataFrame()


def _fetch_team_roster_http(team_id: str, season: str, timeout: int = 30, retries: int = 3) -> pd.DataFrame:
    """Fetch roster via direct stats.nba.com call with headers, caching, and retries."""
    cache_dir = PRESEASON_ROSTERS_DIR / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"roster_{season}_{team_id}.json"

    # Use cached response when available
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as fh:
                js = json.load(fh)
                return _parse_result_payload(js)
        except Exception:
            pass

    url = "https://stats.nba.com/stats/commonteamroster"
    params = {"TeamID": str(team_id), "Season": season, "LeagueID": "00"}

    headers = {
        "Accept": "application/json, text/plain, */*",
        "Connection": "keep-alive",
        "Origin": "https://www.nba.com",
        "Referer": "https://www.nba.com",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "x-nba-stats-origin": "stats",
        "x-nba-stats-token": "true",
    }

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, headers=headers, impersonate="chrome110", timeout=timeout)
            if getattr(resp, "status_code", None) and resp.status_code != 200:
                last_err = Exception(f"Status {resp.status_code}")
                wait = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(wait)
                continue

            js = resp.json() if hasattr(resp, "json") else json.loads(resp.text)
            # cache and return
            try:
                with open(cache_file, "w", encoding="utf-8") as fh:
                    json.dump(js, fh)
            except Exception:
                pass

            return _parse_result_payload(js)
        except Exception as exc:
            last_err = exc
            wait = (2 ** attempt) + random.uniform(0, 1)
            time.sleep(wait)

    raise last_err if last_err is not None else Exception("Unknown HTTP error fetching roster")


def _normalize_roster_df(roster_df: pd.DataFrame, season: str, team_id: str, team_abbreviation: str, team_name: str) -> pd.DataFrame:
    if roster_df.empty:
        return pd.DataFrame(
            columns=[
                "season",
                "team_id",
                "team_abbreviation",
                "team_name",
                "player_id",
                "player_name",
                "position",
                "exp",
                "num",
                "height",
                "weight",
                "age",
                "school",
            ]
        )

    # Ensure integer 0..N-1 index for proper broadcasting of scalar columns
    roster_df = roster_df.reset_index(drop=True)
    cols = {c.lower(): c for c in roster_df.columns}
    player_id_col = cols.get("player_id") or "PLAYER_ID"
    player_name_col = cols.get("player") or cols.get("player_name") or "PLAYER"

    # Build output dataframe from roster rows first, then assign scalar/team columns
    out = pd.DataFrame()
    out["player_id"] = _norm_id(roster_df[player_id_col]).astype(str)
    out["player_name"] = roster_df[player_name_col].astype(str)
    out["position"] = roster_df.get(cols.get("position"), "").astype(str)
    out["exp"] = roster_df.get(cols.get("exp"), "")
    out["num"] = roster_df.get(cols.get("num"), "")
    out["height"] = roster_df.get(cols.get("height"), "")
    out["weight"] = roster_df.get(cols.get("weight"), "")
    out["age"] = pd.to_numeric(roster_df.get(cols.get("age")), errors="coerce")
    out["school"] = roster_df.get(cols.get("school"), "").astype(str)

    # Now set scalar/team columns (broadcast to rows)
    out["season"] = str(season)
    out["team_id"] = str(team_id)
    out["team_abbreviation"] = str(team_abbreviation).upper() if team_abbreviation is not None else None
    out["team_name"] = str(team_name) if team_name is not None else None

    out = out[out["player_id"].str.len() > 0].copy()
    # Reorder columns to canonical layout
    cols_order = ["season", "team_id", "team_abbreviation", "team_name", "player_id", "player_name", "position", "exp", "num", "height", "weight", "age", "school"]
    return out[[c for c in cols_order if c in out.columns]]


def _load_existing_combined() -> pd.DataFrame:
    if not PRESEASON_ROSTERS_PATH.exists():
        return pd.DataFrame()
    try:
        existing = load_standardized(PRESEASON_ROSTERS_PATH)
    except Exception:
        return pd.DataFrame()
    if existing.empty:
        return existing
    if "player_id" in existing.columns:
        existing["player_id"] = _norm_id(existing["player_id"])
    if "team_id" in existing.columns:
        existing["team_id"] = _norm_id(existing["team_id"])
    if "season" in existing.columns:
        existing["season"] = existing["season"].astype(str)
    return existing


def fetch_preseason_rosters(seasons: List[str], force_refresh: bool = False) -> pd.DataFrame:
    seasons = sorted({str(s).strip() for s in seasons if str(s).strip()})
    if not seasons:
        raise ValueError("No seasons provided")

    teams = _load_teams()
    existing = _load_existing_combined() if not force_refresh else pd.DataFrame()
    have_seasons = set(existing["season"].astype(str).unique().tolist()) if not existing.empty and "season" in existing.columns else set()

    season_frames = []
    for season in seasons:
        if season in have_seasons and not force_refresh:
            season_df = existing[existing["season"] == season].copy()
            if not season_df.empty:
                print(f"  Using cached preseason rosters for {season} ({len(season_df)} rows)")
                season_frames.append(season_df)
                continue

        print(f"  Fetching preseason rosters for {season}...")
        rows = []
        for _, t in teams.iterrows():
            team_id = str(t["team_id"])
            team_abbr = str(t["team_abbreviation"])
            team_name = str(t.get("team_name", ""))
            roster_df = _fetch_team_roster(team_id=team_id, season=season)
            norm = _normalize_roster_df(roster_df, season, team_id, team_abbr, team_name)
            if not norm.empty:
                rows.append(norm)
            time.sleep(0.10)

        season_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        if not season_df.empty:
            season_df = season_df.drop_duplicates(subset=["season", "team_abbreviation", "player_id"], keep="first")
            season_df = season_df.sort_values(["team_abbreviation", "player_name"]).reset_index(drop=True)
        season_frames.append(season_df)

    fetched = pd.concat(season_frames, ignore_index=True) if season_frames else pd.DataFrame()
    if not fetched.empty:
        fetched["season"] = fetched["season"].astype(str)
        fetched["team_id"] = _norm_id(fetched["team_id"])
        fetched["player_id"] = _norm_id(fetched["player_id"])

    # Merge fetched + existing for non-requested seasons.
    if not existing.empty and not force_refresh:
        keep_old = existing[~existing["season"].astype(str).isin(seasons)].copy()
        combined = pd.concat([keep_old, fetched], ignore_index=True)
    else:
        combined = fetched.copy()

    if not combined.empty:
        combined = combined.drop_duplicates(subset=["season", "team_abbreviation", "player_id"], keep="first")
        combined = combined.sort_values(["season", "team_abbreviation", "player_name"]).reset_index(drop=True)

    PRESEASON_ROSTERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_standardized(combined, PRESEASON_ROSTERS_PATH)

    PRESEASON_ROSTERS_DIR.mkdir(parents=True, exist_ok=True)
    for season in sorted(combined["season"].unique()) if not combined.empty else []:
        one = combined[combined["season"] == season].copy()
        season_path = PRESEASON_ROSTERS_DIR / f"preseason_rosters_{season}.parquet"
        save_standardized(one, season_path)

    report = {
        "seasons_requested": seasons,
        "rows_total": int(len(combined)),
        "rows_by_season": {
            season: int((combined["season"] == season).sum())
            for season in sorted(combined["season"].unique())
        }
        if not combined.empty
        else {},
        "teams_by_season": {
            season: int(combined[combined["season"] == season]["team_abbreviation"].nunique())
            for season in sorted(combined["season"].unique())
        }
        if not combined.empty
        else {},
    }
    PRESEASON_ROSTERS_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch preseason roster snapshots by season")
    parser.add_argument("--season", type=str, default=None, help="Single season, e.g. 2025-26")
    parser.add_argument(
        "--seasons",
        type=str,
        default=None,
        help="Comma-separated seasons, e.g. 2023-24,2024-25,2025-26",
    )
    parser.add_argument("--force-refresh", action="store_true", help="Refetch requested seasons even if cached")
    args = parser.parse_args()

    seasons = []
    if args.season:
        seasons.append(args.season)
    if args.seasons:
        seasons.extend([s.strip() for s in args.seasons.split(",") if s.strip()])

    if not seasons:
        raise ValueError("Provide --season or --seasons")

    result = fetch_preseason_rosters(seasons=seasons, force_refresh=args.force_refresh)
    print(f"Saved combined preseason rosters: {PRESEASON_ROSTERS_PATH}")
    print(f"Saved preseason report: {PRESEASON_ROSTERS_REPORT}")
    print(f"Rows: {len(result)}")


if __name__ == "__main__":
    main()
