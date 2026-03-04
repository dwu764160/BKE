"""
src/data_fetch/derive_team_game_logs.py

Builds/repairs data/historical/team_game_logs.parquet with quality gates.

Priority order:
1) Use existing team_game_logs.parquet if it passes quality checks.
2) Fetch authoritative logs via nba_api TeamGameLog and rebuild.
3) Fallback to PBP derivation only if API fetch fails.

This prevents corrupted team margin distributions from propagating downstream.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from typing import Dict, List, Tuple

import pandas as pd
from curl_cffi import requests as curl_requests
from nba_api.stats.endpoints import leaguegamelog, teamgamelog
from nba_api.stats.static import teams

# Adjust path to find src if run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

DATA_DIR = "data/historical"
OUTPUT_FILE = os.path.join(DATA_DIR, "team_game_logs.parquet")


def _default_nba_headers() -> Dict[str, str]:
    return {
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


def _load_nba_headers() -> Dict[str, str]:
    headers = _default_nba_headers()
    headers_path = os.path.join("data", "nba_headers.json")
    if not os.path.exists(headers_path):
        return headers

    try:
        payload = json.loads(open(headers_path, "r", encoding="utf-8").read())
        if isinstance(payload, dict):
            # Accept either flat dict or nested {"headers": {...}}
            cand = payload.get("headers") if isinstance(payload.get("headers"), dict) else payload
            if isinstance(cand, dict):
                for k, v in cand.items():
                    if isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip():
                        headers[k] = v
    except Exception:
        pass
    return headers


def _fetch_leaguegamelog_direct(season: str) -> pd.DataFrame:
    """Fetch one full season of team game logs in a single call to stats.nba.com."""
    url = "https://stats.nba.com/stats/leaguegamelog"
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
    headers = _load_nba_headers()

    resp = curl_requests.get(
        url,
        params=params,
        headers=headers,
        impersonate="chrome110",
        timeout=60,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"leaguegamelog status={resp.status_code}")

    data = resp.json()

    # Handle both resultSets[] and resultSet object formats.
    if isinstance(data, dict) and "resultSets" in data and data["resultSets"]:
        rs = data["resultSets"][0]
        return pd.DataFrame(rs.get("rowSet", []), columns=rs.get("headers", []))

    if isinstance(data, dict) and "resultSet" in data:
        rs = data["resultSet"]
        return pd.DataFrame(rs.get("rowSet", []), columns=rs.get("headers", []))

    raise RuntimeError("Unexpected leaguegamelog JSON format")


def _normalize_team_logs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out

    rename_map = {
        "Game_ID": "GAME_ID",
        "Team_ID": "TEAM_ID",
    }
    out = out.rename(columns=rename_map)

    if "SEASON" not in out.columns:
        out["SEASON"] = "UNKNOWN"

    out["SEASON"] = out["SEASON"].astype(str)
    if "GAME_ID" in out.columns:
        out["GAME_ID"] = out["GAME_ID"].astype(str)
    if "TEAM_ID" in out.columns:
        out["TEAM_ID"] = (
            pd.to_numeric(out["TEAM_ID"], errors="coerce")
            .round()
            .astype("Int64")
        )

    if "PTS" in out.columns:
        out["PTS"] = pd.to_numeric(out["PTS"], errors="coerce")

    required = ["SEASON", "GAME_ID", "TEAM_ID", "PTS"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns for team logs: {missing}")

    out = out.dropna(subset=["SEASON", "GAME_ID", "TEAM_ID", "PTS"]).copy()
    out["TEAM_ID"] = out["TEAM_ID"].astype(int)

    if "GAME_DATE" not in out.columns:
        out["GAME_DATE"] = None

    # Recompute OPP_PTS from paired team rows per game.
    base = out[["SEASON", "GAME_ID", "TEAM_ID", "PTS"]].copy()
    opp = base.rename(columns={"TEAM_ID": "OPP_TEAM_ID", "PTS": "OPP_PTS"})
    paired = base.merge(opp, on=["SEASON", "GAME_ID"], how="inner")
    paired = paired[paired["TEAM_ID"] != paired["OPP_TEAM_ID"]].copy()

    # If malformed game has >2 teams, pick opponent with maximum points (defensive choice).
    paired = paired.sort_values(["SEASON", "GAME_ID", "TEAM_ID", "OPP_PTS"], ascending=[True, True, True, False])
    paired = paired.drop_duplicates(subset=["SEASON", "GAME_ID", "TEAM_ID"], keep="first")

    out = out.drop(columns=["OPP_PTS"], errors="ignore")
    out = out.merge(
        paired[["SEASON", "GAME_ID", "TEAM_ID", "OPP_PTS"]],
        on=["SEASON", "GAME_ID", "TEAM_ID"],
        how="left",
    )
    out["OPP_PTS"] = pd.to_numeric(out["OPP_PTS"], errors="coerce")
    out = out.dropna(subset=["OPP_PTS"]).copy()
    out["OPP_PTS"] = out["OPP_PTS"].astype(int)
    out["PTS"] = out["PTS"].astype(int)
    out["margin"] = out["PTS"] - out["OPP_PTS"]

    out = out.drop_duplicates(subset=["SEASON", "GAME_ID", "TEAM_ID"]).copy()
    return out


def _quality_report(df: pd.DataFrame) -> Dict[str, object]:
    if df.empty:
        return {"ok": False, "reason": "empty"}

    rep: Dict[str, object] = {"ok": True, "seasons": {}}

    # Global game-level integrity checks.
    rows_per_game = df.groupby(["SEASON", "GAME_ID"])["TEAM_ID"].nunique()
    pct_two_rows = float((rows_per_game == 2).mean()) if len(rows_per_game) else 0.0
    rep["pct_games_with_exactly_2_teams"] = pct_two_rows

    team_margins = df.groupby(["SEASON", "TEAM_ID"], as_index=False)["margin"].mean()

    for season, sdf in df.groupby("SEASON"):
        teams_n = int(sdf["TEAM_ID"].nunique())
        games_per_team = sdf.groupby("TEAM_ID")["GAME_ID"].nunique()
        season_margin_std = float(
            team_margins.loc[team_margins["SEASON"] == season, "margin"].std()
        )

        rep["seasons"][str(season)] = {
            "teams": teams_n,
            "games_per_team_min": int(games_per_team.min()) if len(games_per_team) else 0,
            "games_per_team_max": int(games_per_team.max()) if len(games_per_team) else 0,
            "team_margin_std": season_margin_std,
        }

    # Quality gates chosen to catch corrupted synthetic/inverted logs.
    if pct_two_rows < 0.995:
        rep["ok"] = False
        rep["reason"] = f"Only {pct_two_rows:.2%} games have exactly two team rows"
        return rep

    for season, srep in rep["seasons"].items():
        if srep["teams"] < 30:
            rep["ok"] = False
            rep["reason"] = f"{season}: expected 30 teams, got {srep['teams']}"
            return rep
        if srep["games_per_team_min"] < 80 or srep["games_per_team_max"] > 84:
            rep["ok"] = False
            rep["reason"] = (
                f"{season}: games/team out of range "
                f"[{srep['games_per_team_min']}, {srep['games_per_team_max']}]"
            )
            return rep
        if not (2.5 <= srep["team_margin_std"] <= 8.5):
            rep["ok"] = False
            rep["reason"] = f"{season}: implausible team margin std={srep['team_margin_std']:.3f}"
            return rep

    return rep


def _detect_seasons() -> List[str]:
    # Explicit env takes precedence.
    env_seasons = os.environ.get("SEASONS", "").strip()
    if env_seasons:
        return [s.strip() for s in env_seasons.split(",") if s.strip()]

    pattern = os.path.join(DATA_DIR, "play_by_play_*.parquet")
    files = sorted(glob.glob(pattern))
    seasons = []
    for f in files:
        match = re.search(r"(\d{4}-\d{2})", os.path.basename(f))
        if match:
            seasons.append(match.group(1))
    if seasons:
        return sorted(set(seasons))

    # Project default.
    return ["2022-23", "2023-24", "2024-25"]


def _fetch_from_teamgamelog_api(seasons: List[str]) -> pd.DataFrame:
    rows = []
    fetched_seasons = set()

    # Primary: direct stats.nba.com calls via curl_cffi (most resilient in this project).
    for season in seasons:
        print(f"[DIRECT] Fetching leaguegamelog for {season}...")
        try:
            df = _fetch_leaguegamelog_direct(season)
            if not df.empty:
                df["SEASON"] = season
                rows.append(df)
                fetched_seasons.add(season)
            time.sleep(0.2)
        except Exception as exc:
            print(f"  [WARN] direct leaguegamelog failed for {season}: {exc}")

    # Primary: one call per season via LeagueGameLog (more reliable and faster).
    for season in [s for s in seasons if s not in fetched_seasons]:
        print(f"[API] Fetching LeagueGameLog for {season}...")
        try:
            df = leaguegamelog.LeagueGameLog(
                counter=0,
                direction="ASC",
                league_id="00",
                player_or_team_abbreviation="T",
                season=season,
                season_type_all_star="Regular Season",
                sorter="DATE",
                timeout=90,
            ).get_data_frames()[0]
            if not df.empty:
                df["SEASON"] = season
                rows.append(df)
                fetched_seasons.add(season)
            time.sleep(0.3)
        except Exception as exc:
            print(f"  [WARN] LeagueGameLog failed for {season}: {exc}")

    # Fallback: per-team TeamGameLog calls.
    missing_seasons = [s for s in seasons if s not in fetched_seasons]
    team_ids = [t["id"] for t in teams.get_teams()]
    for season in missing_seasons:
        print(f"[API Fallback] Fetching TeamGameLog for {season}...")
        for tid in team_ids:
            try:
                df = teamgamelog.TeamGameLog(
                    team_id=tid,
                    season=season,
                    season_type_all_star="Regular Season",
                    timeout=60,
                ).get_data_frames()[0]
                if df.empty:
                    continue
                df["SEASON"] = season
                rows.append(df)
                fetched_seasons.add(season)
                time.sleep(0.15)
            except Exception as exc:
                print(f"  [WARN] Team {tid} season {season} failed: {exc}")

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    out = _normalize_team_logs(out)
    return out


def _fallback_from_pbp(seasons: List[str]) -> pd.DataFrame:
    print("[FALLBACK] Rebuilding team logs from PBP (less reliable than API)...")
    all_rows = []

    for season in seasons:
        path = os.path.join(DATA_DIR, f"play_by_play_{season}.parquet")
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path)
        if df.empty or "GAME_ID" not in df.columns:
            continue

        for gid, gdf in df.groupby("GAME_ID"):
            if "scoreHome" not in gdf.columns or "scoreAway" not in gdf.columns:
                continue
            gdf = gdf.copy()
            gdf["scoreHome"] = pd.to_numeric(gdf["scoreHome"], errors="coerce")
            gdf["scoreAway"] = pd.to_numeric(gdf["scoreAway"], errors="coerce")
            gdf = gdf.dropna(subset=["scoreHome", "scoreAway"])
            if gdf.empty:
                continue

            home_score = int(gdf["scoreHome"].iloc[-1])
            away_score = int(gdf["scoreAway"].iloc[-1])

            team_col = "teamId" if "teamId" in gdf.columns else "TEAM_ID" if "TEAM_ID" in gdf.columns else None
            if team_col is None:
                continue
            teams_seen = (
                pd.to_numeric(gdf[team_col], errors="coerce")
                .dropna()
                .astype(int)
                .unique()
                .tolist()
            )
            if len(teams_seen) != 2:
                continue

            # Heuristic only in fallback path.
            t1, t2 = teams_seen[0], teams_seen[1]
            game_date = None
            if "timeActual" in gdf.columns:
                game_date = str(gdf.iloc[0]["timeActual"]).split("T")[0]

            all_rows.append({
                "SEASON": season,
                "GAME_ID": str(gid),
                "TEAM_ID": t1,
                "PTS": home_score,
                "OPP_PTS": away_score,
                "GAME_DATE": game_date,
            })
            all_rows.append({
                "SEASON": season,
                "GAME_ID": str(gid),
                "TEAM_ID": t2,
                "PTS": away_score,
                "OPP_PTS": home_score,
                "GAME_DATE": game_date,
            })

    if not all_rows:
        return pd.DataFrame()

    out = pd.DataFrame(all_rows)
    out = _normalize_team_logs(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive/repair team game logs with quality gates")
    parser.add_argument("--force-rebuild", action="store_true", help="Ignore existing file and rebuild")
    parser.add_argument("--skip-api", action="store_true", help="Do not call nba_api; fallback to PBP only")
    args = parser.parse_args()

    seasons = _detect_seasons()
    print(f"Seasons: {seasons}")

    if os.path.exists(OUTPUT_FILE) and not args.force_rebuild:
        try:
            existing = _normalize_team_logs(pd.read_parquet(OUTPUT_FILE))
            rep = _quality_report(existing)
            present = sorted(existing["SEASON"].astype(str).unique().tolist())
            missing = [s for s in seasons if s not in present]
            if rep.get("ok", False) and not missing:
                existing = existing.sort_values(["SEASON", "GAME_ID", "TEAM_ID"]).reset_index(drop=True)
                existing.to_parquet(OUTPUT_FILE, index=False)
                print(f"Existing team_game_logs passed quality checks. Keeping {OUTPUT_FILE}")
                print(rep)
                return
            if missing:
                print(f"Existing team_game_logs missing seasons {missing}; rebuilding")
            else:
                print(f"Existing team_game_logs failed quality checks: {rep.get('reason')}")
        except Exception as exc:
            print(f"Existing team_game_logs unreadable/invalid: {exc}")

    rebuilt = pd.DataFrame()

    if not args.skip_api:
        rebuilt = _fetch_from_teamgamelog_api(seasons)
        if not rebuilt.empty:
            present = sorted(rebuilt["SEASON"].astype(str).unique().tolist())
            missing = [s for s in seasons if s not in present]
            if missing:
                print(f"API-rebuilt logs missing seasons {missing}; refusing partial write")
                rebuilt = pd.DataFrame()
            else:
                rep = _quality_report(rebuilt)
                if rep.get("ok", False):
                    rebuilt = rebuilt.sort_values(["SEASON", "GAME_ID", "TEAM_ID"]).reset_index(drop=True)
                    rebuilt.to_parquet(OUTPUT_FILE, index=False)
                    print(f"Wrote authoritative team logs to {OUTPUT_FILE}; rows={len(rebuilt)}")
                    print(rep)
                    return
                print(f"API-rebuilt logs failed quality checks: {rep.get('reason')}")

    # Last resort fallback.
    rebuilt = _fallback_from_pbp(seasons)
    if rebuilt.empty:
        raise RuntimeError("Could not rebuild team_game_logs from API or PBP")

    rep = _quality_report(rebuilt)
    if not rep.get("ok", False):
        raise RuntimeError(f"Fallback team logs failed quality checks: {rep.get('reason')}")

    rebuilt = rebuilt.sort_values(["SEASON", "GAME_ID", "TEAM_ID"]).reset_index(drop=True)
    rebuilt.to_parquet(OUTPUT_FILE, index=False)
    print(f"Wrote fallback team logs to {OUTPUT_FILE}; rows={len(rebuilt)}")
    print(rep)


if __name__ == "__main__":
    main()