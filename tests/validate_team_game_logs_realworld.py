"""
tests/validate_team_game_logs_realworld.py

Validate local team game logs against official NBA endpoint data.

Checks:
1) Team season W-L records per season.
2) Random sampled game scores by GAME_ID and TEAM_ID.

Usage:
  python3 tests/validate_team_game_logs_realworld.py
  python3 tests/validate_team_game_logs_realworld.py --seasons 2022-23,2023-24,2024-25 --sample-size 20 --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
from curl_cffi import requests as curl_requests


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_PATH = REPO_ROOT / "data/historical/team_game_logs.parquet"
DEFAULT_REPORT_PATH = REPO_ROOT / "reports/team_game_logs_realworld_validation.json"


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
    headers_path = REPO_ROOT / "data/nba_headers.json"
    if not headers_path.exists():
        return headers

    try:
        payload = json.loads(headers_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            cand = payload.get("headers") if isinstance(payload.get("headers"), dict) else payload
            if isinstance(cand, dict):
                for k, v in cand.items():
                    if isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip():
                        headers[k] = v
    except Exception:
        pass
    return headers


def _fetch_leaguegamelog_direct(season: str) -> pd.DataFrame:
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
    resp = curl_requests.get(
        url,
        params=params,
        headers=_load_nba_headers(),
        impersonate="chrome110",
        timeout=90,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"leaguegamelog status={resp.status_code}")

    data = resp.json()
    if isinstance(data, dict) and "resultSets" in data and data["resultSets"]:
        rs = data["resultSets"][0]
        return pd.DataFrame(rs.get("rowSet", []), columns=rs.get("headers", []))
    if isinstance(data, dict) and "resultSet" in data:
        rs = data["resultSet"]
        return pd.DataFrame(rs.get("rowSet", []), columns=rs.get("headers", []))
    raise RuntimeError("Unexpected leaguegamelog JSON format")


def _fetch_official_season(season: str) -> pd.DataFrame:
    try:
        df = _fetch_leaguegamelog_direct(season)
    except Exception:
        try:
            from nba_api.stats.endpoints import leaguegamelog  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                f"Direct leaguegamelog fetch failed and nba_api fallback unavailable: {exc}"
            )

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
    df["SEASON"] = season
    return df


def _normalize_logs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).upper() for c in out.columns]

    if "SEASON" not in out.columns:
        out["SEASON"] = "UNKNOWN"
    out["SEASON"] = out["SEASON"].astype(str)

    for c in ["GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION", "WL", "PTS", "PLUS_MINUS", "OPP_PTS"]:
        if c not in out.columns:
            out[c] = None

    out["GAME_ID"] = out["GAME_ID"].astype(str)
    out["TEAM_ID"] = pd.to_numeric(out["TEAM_ID"], errors="coerce").astype("Int64")
    out["TEAM_ABBREVIATION"] = out["TEAM_ABBREVIATION"].astype(str).str.upper()
    out["PTS"] = pd.to_numeric(out["PTS"], errors="coerce")
    out["PLUS_MINUS"] = pd.to_numeric(out["PLUS_MINUS"], errors="coerce")
    out["OPP_PTS"] = pd.to_numeric(out["OPP_PTS"], errors="coerce")

    # Derive opponent points if missing.
    needs_opp = out["OPP_PTS"].isna() & out["GAME_ID"].notna() & out["PTS"].notna()
    if needs_opp.any():
        total_pts = out.groupby("GAME_ID")["PTS"].transform("sum")
        out.loc[needs_opp, "OPP_PTS"] = total_pts[needs_opp] - out.loc[needs_opp, "PTS"]

    # Derive WL if missing.
    wl_missing = out["WL"].isna() | (out["WL"].astype(str).str.strip() == "")
    out.loc[wl_missing & out["PTS"].notna() & out["OPP_PTS"].notna(), "WL"] = out.loc[
        wl_missing & out["PTS"].notna() & out["OPP_PTS"].notna(),
        "PTS",
    ].gt(out.loc[wl_missing & out["PTS"].notna() & out["OPP_PTS"].notna(), "OPP_PTS"]).map({True: "W", False: "L"})

    out = out.dropna(subset=["SEASON", "GAME_ID", "TEAM_ID", "PTS"]).copy()
    out["TEAM_ID"] = out["TEAM_ID"].astype(int)

    # Keep one team row per game.
    out = out.drop_duplicates(subset=["SEASON", "GAME_ID", "TEAM_ID"])
    return out


def _wl_table(df: pd.DataFrame) -> pd.DataFrame:
    t = df.copy()
    t["W"] = (t["WL"].astype(str).str.upper() == "W").astype(int)
    wl = (
        t.groupby(["SEASON", "TEAM_ID", "TEAM_ABBREVIATION"], as_index=False)
        .agg(GAMES=("GAME_ID", "nunique"), W=("W", "sum"))
    )
    wl["L"] = wl["GAMES"] - wl["W"]
    return wl


def _game_score_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["SEASON", "GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION", "PTS", "OPP_PTS", "PLUS_MINUS"]
    out = df[cols].copy()
    out["PTS"] = out["PTS"].astype(int)
    out["OPP_PTS"] = pd.to_numeric(out["OPP_PTS"], errors="coerce").astype("Int64")
    out["PLUS_MINUS"] = pd.to_numeric(out["PLUS_MINUS"], errors="coerce")
    return out


@dataclass
class SeasonResult:
    season: str
    team_wl_rows_compared: int
    team_wl_exact_match_rows: int
    team_wl_mismatch_rows: int
    team_wl_max_abs_w_diff: int
    team_wl_max_abs_l_diff: int
    full_rows_compared: int
    full_pts_exact_match_rows: int
    full_plus_minus_exact_match_rows: int
    full_pts_mismatch_rows: int
    sampled_games: int
    sampled_rows_compared: int
    sampled_pts_exact_match_rows: int
    sampled_opp_pts_exact_match_rows: int
    sampled_plus_minus_exact_match_rows: int
    sampled_pts_mismatch_rows: int


def validate(local_df: pd.DataFrame, seasons: List[str], sample_size: int, seed: int) -> Dict[str, object]:
    rng = random.Random(seed)
    local_norm = _normalize_logs(local_df)

    season_results: List[SeasonResult] = []
    mismatch_examples: Dict[str, Dict[str, List[dict]]] = {}

    for season in seasons:
        print(f"\n[FETCH] Official league game logs for {season}...")
        official_norm = _normalize_logs(_fetch_official_season(season))

        l_season = local_norm[local_norm["SEASON"] == season].copy()
        o_season = official_norm[official_norm["SEASON"] == season].copy()

        wl_local = _wl_table(l_season)
        wl_off = _wl_table(o_season)

        wl_cmp = wl_local.merge(
            wl_off[["SEASON", "TEAM_ID", "W", "L"]],
            on=["SEASON", "TEAM_ID"],
            how="inner",
            suffixes=("_local", "_official"),
        )
        wl_cmp["W_DIFF"] = wl_cmp["W_local"] - wl_cmp["W_official"]
        wl_cmp["L_DIFF"] = wl_cmp["L_local"] - wl_cmp["L_official"]
        wl_cmp["EXACT"] = (wl_cmp["W_DIFF"] == 0) & (wl_cmp["L_DIFF"] == 0)

        # Full-season row-level score and plus-minus comparison.
        full_local = _game_score_table(l_season)
        full_off = _game_score_table(o_season)
        full_cmp = full_local.merge(
            full_off,
            on=["SEASON", "GAME_ID", "TEAM_ID"],
            how="inner",
            suffixes=("_local", "_official"),
        )
        full_pts_exact = (full_cmp["PTS_local"] == full_cmp["PTS_official"]).sum()
        full_pts_mismatch = (full_cmp["PTS_local"] != full_cmp["PTS_official"]).sum()
        full_pm_local = full_cmp["PLUS_MINUS_local"]
        full_pm_off = full_cmp["PLUS_MINUS_official"]
        full_pm_exact = ((full_pm_local.isna() & full_pm_off.isna()) | (full_pm_local == full_pm_off)).sum()

        # Random game sampling per season.
        game_ids = sorted(l_season["GAME_ID"].dropna().unique().tolist())
        if not game_ids:
            sampled_ids: List[str] = []
        else:
            sampled_ids = rng.sample(game_ids, k=min(sample_size, len(game_ids)))

        sample_local = _game_score_table(l_season[l_season["GAME_ID"].isin(sampled_ids)])
        sample_off = _game_score_table(o_season[o_season["GAME_ID"].isin(sampled_ids)])

        sample_cmp = sample_local.merge(
            sample_off,
            on=["SEASON", "GAME_ID", "TEAM_ID"],
            how="inner",
            suffixes=("_local", "_official"),
        )

        pts_exact = (sample_cmp["PTS_local"] == sample_cmp["PTS_official"]).sum()
        opp_exact = (sample_cmp["OPP_PTS_local"] == sample_cmp["OPP_PTS_official"]).sum()

        pm_local = sample_cmp["PLUS_MINUS_local"]
        pm_off = sample_cmp["PLUS_MINUS_official"]
        pm_exact = ((pm_local.isna() & pm_off.isna()) | (pm_local == pm_off)).sum()
        pts_mismatch = (sample_cmp["PTS_local"] != sample_cmp["PTS_official"]).sum()

        wl_bad = wl_cmp.loc[~wl_cmp["EXACT"], ["TEAM_ID", "TEAM_ABBREVIATION", "W_local", "L_local", "W_official", "L_official", "W_DIFF", "L_DIFF"]]
        score_bad = sample_cmp.loc[
            sample_cmp["PTS_local"] != sample_cmp["PTS_official"],
            [
                "GAME_ID",
                "TEAM_ID",
                "TEAM_ABBREVIATION_local",
                "PTS_local",
                "PTS_official",
                "OPP_PTS_local",
                "OPP_PTS_official",
            ],
        ]

        mismatch_examples[season] = {
            "wl": wl_bad.head(10).to_dict(orient="records"),
            "scores": score_bad.head(10).to_dict(orient="records"),
        }

        season_results.append(
            SeasonResult(
                season=season,
                team_wl_rows_compared=int(len(wl_cmp)),
                team_wl_exact_match_rows=int(wl_cmp["EXACT"].sum()),
                team_wl_mismatch_rows=int((~wl_cmp["EXACT"]).sum()),
                team_wl_max_abs_w_diff=int(wl_cmp["W_DIFF"].abs().max() if len(wl_cmp) else 0),
                team_wl_max_abs_l_diff=int(wl_cmp["L_DIFF"].abs().max() if len(wl_cmp) else 0),
                full_rows_compared=int(len(full_cmp)),
                full_pts_exact_match_rows=int(full_pts_exact),
                full_plus_minus_exact_match_rows=int(full_pm_exact),
                full_pts_mismatch_rows=int(full_pts_mismatch),
                sampled_games=int(len(sampled_ids)),
                sampled_rows_compared=int(len(sample_cmp)),
                sampled_pts_exact_match_rows=int(pts_exact),
                sampled_opp_pts_exact_match_rows=int(opp_exact),
                sampled_plus_minus_exact_match_rows=int(pm_exact),
                sampled_pts_mismatch_rows=int(pts_mismatch),
            )
        )

    total_wl_rows = sum(r.team_wl_rows_compared for r in season_results)
    total_wl_exact = sum(r.team_wl_exact_match_rows for r in season_results)
    total_full_rows = sum(r.full_rows_compared for r in season_results)
    total_full_pts_exact = sum(r.full_pts_exact_match_rows for r in season_results)
    total_full_pm_exact = sum(r.full_plus_minus_exact_match_rows for r in season_results)
    total_sample_rows = sum(r.sampled_rows_compared for r in season_results)
    total_pts_exact = sum(r.sampled_pts_exact_match_rows for r in season_results)

    report = {
        "seasons": seasons,
        "local_path": str(DEFAULT_LOCAL_PATH),
        "sample_size_per_season": sample_size,
        "seed": seed,
        "summary": {
            "team_wl_rows_compared": total_wl_rows,
            "team_wl_exact_match_rows": total_wl_exact,
            "team_wl_exact_match_rate": (float(total_wl_exact / total_wl_rows) if total_wl_rows else None),
            "full_rows_compared": total_full_rows,
            "full_pts_exact_match_rows": total_full_pts_exact,
            "full_pts_exact_match_rate": (float(total_full_pts_exact / total_full_rows) if total_full_rows else None),
            "full_plus_minus_exact_match_rows": total_full_pm_exact,
            "full_plus_minus_exact_match_rate": (float(total_full_pm_exact / total_full_rows) if total_full_rows else None),
            "sample_rows_compared": total_sample_rows,
            "sample_pts_exact_match_rows": total_pts_exact,
            "sample_pts_exact_match_rate": (float(total_pts_exact / total_sample_rows) if total_sample_rows else None),
        },
        "season_results": [r.__dict__ for r in season_results],
        "mismatch_examples": mismatch_examples,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate local team_game_logs against official NBA game logs")
    parser.add_argument("--seasons", default="2022-23,2023-24,2024-25", help="Comma-separated seasons")
    parser.add_argument("--sample-size", type=int, default=15, help="Random games to sample per season")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--local-path", default=str(DEFAULT_LOCAL_PATH), help="Path to local team_game_logs.parquet")
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH), help="Path for JSON output report")
    args = parser.parse_args()

    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    local_path = Path(args.local_path)
    report_path = Path(args.report_path)

    if not local_path.exists():
        raise FileNotFoundError(f"Local logs not found: {local_path}")

    local_df = pd.read_parquet(local_path)
    report = validate(local_df, seasons=seasons, sample_size=args.sample_size, seed=args.seed)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print("TEAM GAME LOGS VS OFFICIAL NBA DATA")
    print("=" * 72)
    print(f"Local file: {local_path}")
    print(f"Seasons: {seasons}")
    print(f"W-L exact match: {report['summary']['team_wl_exact_match_rows']}/{report['summary']['team_wl_rows_compared']} ({report['summary']['team_wl_exact_match_rate']:.2%})")
    print(f"Full-row PTS exact: {report['summary']['full_pts_exact_match_rows']}/{report['summary']['full_rows_compared']} ({report['summary']['full_pts_exact_match_rate']:.2%})")
    print(f"Full-row +/- exact: {report['summary']['full_plus_minus_exact_match_rows']}/{report['summary']['full_rows_compared']} ({report['summary']['full_plus_minus_exact_match_rate']:.2%})")
    print(f"Sampled score exact: {report['summary']['sample_pts_exact_match_rows']}/{report['summary']['sample_rows_compared']} ({report['summary']['sample_pts_exact_match_rate']:.2%})")

    for sres in report["season_results"]:
        print(
            f"{sres['season']}: W-L {sres['team_wl_exact_match_rows']}/{sres['team_wl_rows_compared']} exact, "
            f"full +/- rows {sres['full_plus_minus_exact_match_rows']}/{sres['full_rows_compared']} exact, "
            f"sample score rows {sres['sampled_pts_exact_match_rows']}/{sres['sampled_rows_compared']} exact"
        )

    print(f"Report written: {report_path}")


if __name__ == "__main__":
    main()
