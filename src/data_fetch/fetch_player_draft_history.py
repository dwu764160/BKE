"""
src/data_fetch/fetch_player_draft_history.py
=============================================================================
Fetch player draft metadata (draft class year + draft position) for the
pipeline player universe.

Primary source:
  - stats.nba.com DraftHistory endpoint (single fetch for drafted players)

Fallback source:
  - CommonPlayerInfo endpoint for players not covered by DraftHistory
    (typically undrafted players). Results are cached between runs.

Outputs:
  - data/historical/player_draft_history.parquet
  - data/historical/player_draft_history.csv

Usage:
  python3 src/data_fetch/fetch_player_draft_history.py
  python3 src/data_fetch/fetch_player_draft_history.py --no-common-info
=============================================================================
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from curl_cffi import requests as curl_requests
from nba_api.stats.endpoints import commonplayerinfo, drafthistory

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.player_eval.constants import (  # noqa: E402
    COMPLETE_STATS_PATH,
    DRAFT_COMMONPLAYERINFO_CACHE_PATH,
    PLAYER_DRAFT_HISTORY_PATH,
    PLAYER_PROFILES_PARQUET,
    PLAYERS_META_PATH,
)


def _norm_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()


def _safe_read(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def _extract_result_payload(payload: Dict) -> Optional[Dict]:
    """Return first result payload across known NBA response shapes."""
    if not isinstance(payload, dict):
        return None

    result_sets = payload.get("resultSets")
    if isinstance(result_sets, list) and result_sets:
        block = result_sets[0]
        return block if isinstance(block, dict) else None

    result_set = payload.get("resultSet")
    if isinstance(result_set, dict):
        return result_set

    return None


def _to_int_or_nan(value) -> float:
    if value is None:
        return np.nan
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "undrafted"}:
        return np.nan
    try:
        return float(int(float(text)))
    except Exception:
        return np.nan


def _draft_tier(overall_pick: float) -> str:
    if pd.isna(overall_pick):
        return "undrafted"
    p = int(overall_pick)
    if p <= 14:
        return "lottery"
    if p <= 25:
        return "mid_first"
    if p <= 30:
        return "late_first"
    if p <= 60:
        return "second_round"
    return "undrafted"


def _build_player_universe() -> pd.DataFrame:
    """Build local universe of players from core pipeline tables."""
    frames = []

    if PLAYERS_META_PATH.exists():
        p = _safe_read(PLAYERS_META_PATH)
        if not p.empty and "player_id" in p.columns:
            cols = ["player_id"]
            if "full_name" in p.columns:
                cols.append("full_name")
            tmp = p[cols].copy()
            tmp = tmp.rename(columns={"full_name": "player_name"})
            frames.append(tmp)

    if COMPLETE_STATS_PATH.exists():
        p = _safe_read(COMPLETE_STATS_PATH)
        if not p.empty and "PLAYER_ID" in p.columns:
            cols = ["PLAYER_ID"]
            if "PLAYER_NAME" in p.columns:
                cols.append("PLAYER_NAME")
            tmp = p[cols].copy().rename(columns={"PLAYER_ID": "player_id", "PLAYER_NAME": "player_name"})
            frames.append(tmp)

    if PLAYER_PROFILES_PARQUET.exists():
        p = _safe_read(PLAYER_PROFILES_PARQUET)
        if not p.empty and "player_id" in p.columns:
            cols = ["player_id"]
            if "player_name" in p.columns:
                cols.append("player_name")
            tmp = p[cols].copy()
            frames.append(tmp)

    if not frames:
        return pd.DataFrame(columns=["player_id", "player_name"])

    universe = pd.concat(frames, ignore_index=True)
    universe["player_id"] = _norm_id(universe["player_id"])
    universe["player_name"] = universe.get("player_name", "").astype(str)

    # Keep one name per id (first non-empty name wins).
    universe = universe.sort_values("player_name", ascending=False)
    universe = universe.drop_duplicates(subset=["player_id"], keep="first")
    universe = universe[universe["player_id"].str.len() > 0].copy()
    return universe


def _fetch_draft_history() -> pd.DataFrame:
    """Fetch DraftHistory endpoint (all drafted players)."""
    # Primary: direct HTTP call with browser impersonation (more robust in CI/WSL).
    url = "https://stats.nba.com/stats/drafthistory"
    params = {
        "College": "",
        "LeagueID": "00",
        "OverallPick": "",
        "RoundNum": "",
        "RoundPick": "",
        "Season": "",
        "TeamID": "",
        "TopX": "",
    }
    headers = {
        "Accept": "application/json, text/plain, */*",
        "Connection": "keep-alive",
        "Origin": "https://www.nba.com",
        "Referer": "https://www.nba.com/stats/draft/history",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "x-nba-stats-origin": "stats",
        "x-nba-stats-token": "true",
    }

    for attempt in range(1, 4):
        try:
            resp = curl_requests.get(
                url,
                params=params,
                headers=headers,
                impersonate="chrome110",
                timeout=120,
            )
            if resp.status_code == 200:
                payload = resp.json()
                rs = _extract_result_payload(payload)
                if rs:
                    df = pd.DataFrame(rs.get("rowSet", []), columns=rs.get("headers", []))
                    if not df.empty:
                        return _normalize_draft_history_df(df)
        except Exception as exc:
            wait_s = 2 ** attempt
            print(f"  [WARN] curl DraftHistory attempt {attempt}/3 failed: {exc}; retrying in {wait_s}s")
            time.sleep(wait_s)

    # Fallback: nba_api wrapper.
    last_err: Optional[Exception] = None
    for attempt in range(1, 4):
        try:
            resp = drafthistory.DraftHistory(league_id="00", timeout=60)
            df = resp.get_data_frames()[0]
            break
        except Exception as exc:
            last_err = exc
            wait_s = 2 ** attempt
            print(f"  [WARN] DraftHistory attempt {attempt}/3 failed: {exc}; retrying in {wait_s}s")
            time.sleep(wait_s)
    else:
        raise RuntimeError(f"DraftHistory fetch failed after retries: {last_err}")

    if df.empty:
        return pd.DataFrame(
            columns=[
                "player_id", "player_name", "draft_class_year", "draft_round",
                "draft_pick_in_round", "draft_pick_overall", "draft_team_id",
                "draft_team_abbreviation", "draft_type", "draft_source",
            ]
        )

    return _normalize_draft_history_df(df)


def _normalize_draft_history_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize DraftHistory response to canonical schema."""
    if df.empty:
        return pd.DataFrame()

    out = pd.DataFrame({
        "player_id": _norm_id(df["PERSON_ID"]),
        "player_name": df["PLAYER_NAME"].astype(str),
        "draft_class_year": pd.to_numeric(df["SEASON"], errors="coerce"),
        "draft_round": pd.to_numeric(df["ROUND_NUMBER"], errors="coerce"),
        "draft_pick_in_round": pd.to_numeric(df["ROUND_PICK"], errors="coerce"),
        "draft_pick_overall": pd.to_numeric(df["OVERALL_PICK"], errors="coerce"),
        "draft_team_id": _norm_id(df["TEAM_ID"]),
        "draft_team_abbreviation": df["TEAM_ABBREVIATION"].astype(str).str.upper(),
        "draft_type": df["DRAFT_TYPE"].astype(str),
        "draft_source": "drafthistory",
    })
    out = out.drop_duplicates(subset=["player_id"], keep="first")
    return out


def _load_common_info_cache() -> pd.DataFrame:
    if not DRAFT_COMMONPLAYERINFO_CACHE_PATH.exists():
        return pd.DataFrame(columns=[
            "player_id", "draft_class_year", "draft_round", "draft_pick_overall",
            "draft_pick_in_round", "draft_source", "fetched_at",
        ])
    try:
        cached = pd.read_parquet(DRAFT_COMMONPLAYERINFO_CACHE_PATH)
        if "player_id" in cached.columns:
            cached["player_id"] = _norm_id(cached["player_id"])
        return cached
    except Exception:
        return pd.DataFrame(columns=[
            "player_id", "draft_class_year", "draft_round", "draft_pick_overall",
            "draft_pick_in_round", "draft_source", "fetched_at",
        ])


def _fetch_common_player_info(player_id: str) -> Tuple[float, float, float]:
    """Fetch draft year/round/number for one player via CommonPlayerInfo."""
    last_err: Optional[Exception] = None
    for attempt in range(1, 4):
        try:
            resp = commonplayerinfo.CommonPlayerInfo(player_id=int(player_id), timeout=60)
            data = resp.get_normalized_dict().get("CommonPlayerInfo", [])
            if not data:
                return np.nan, np.nan, np.nan
            row = data[0]
            return (
                _to_int_or_nan(row.get("DRAFT_YEAR")),
                _to_int_or_nan(row.get("DRAFT_ROUND")),
                _to_int_or_nan(row.get("DRAFT_NUMBER")),
            )
        except Exception as exc:
            last_err = exc
            wait_s = 1.5 * attempt
            time.sleep(wait_s)
    print(f"  [WARN] CommonPlayerInfo failed for {player_id}: {last_err}")
    return np.nan, np.nan, np.nan


def _enrich_missing_with_common_info(
    universe: pd.DataFrame,
    draft_df: pd.DataFrame,
    use_common_info: bool,
) -> pd.DataFrame:
    """Fill missing draft metadata from CommonPlayerInfo + cache."""
    cache = _load_common_info_cache()
    cache = cache.drop_duplicates(subset=["player_id"], keep="last")

    merged = universe[["player_id"]].merge(
        draft_df[["player_id", "draft_class_year", "draft_round", "draft_pick_overall"]],
        on="player_id",
        how="left",
    )
    missing_ids = merged.loc[merged["draft_class_year"].isna(), "player_id"].astype(str).unique().tolist()

    to_fetch = []
    if use_common_info:
        cached_ids = set(cache["player_id"].astype(str).tolist()) if not cache.empty else set()
        to_fetch = [pid for pid in missing_ids if pid not in cached_ids]

    if to_fetch:
        print(f"  Enriching {len(to_fetch)} players via CommonPlayerInfo fallback...")
        rows = []
        for i, pid in enumerate(to_fetch, 1):
            year, rnd, num = _fetch_common_player_info(pid)
            rows.append({
                "player_id": pid,
                "draft_class_year": year,
                "draft_round": rnd,
                "draft_pick_overall": num,
                "draft_pick_in_round": num,
                "draft_source": "commonplayerinfo",
                "fetched_at": datetime.utcnow().isoformat(timespec="seconds"),
            })
            if i % 50 == 0:
                print(f"    fetched {i}/{len(to_fetch)}")
            time.sleep(0.15)

        new_cache = pd.DataFrame(rows)
        if cache.empty:
            cache = new_cache
        else:
            cache = pd.concat([cache, new_cache], ignore_index=True)
        cache = cache.drop_duplicates(subset=["player_id"], keep="last")

        DRAFT_COMMONPLAYERINFO_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        cache.to_parquet(DRAFT_COMMONPLAYERINFO_CACHE_PATH, index=False)
        print(f"  Updated cache: {DRAFT_COMMONPLAYERINFO_CACHE_PATH}")

    return cache


def build_player_draft_history(use_common_info: bool = True) -> pd.DataFrame:
    print("Building player draft history...")
    universe = _build_player_universe()
    if universe.empty:
        raise RuntimeError("No local players found to build draft history.")
    print(f"  Local player universe: {len(universe)}")

    drafted = _fetch_draft_history()
    print(f"  DraftHistory rows: {len(drafted)}")

    common_cache = _enrich_missing_with_common_info(universe, drafted, use_common_info)
    cache_cols = [
        "player_id", "draft_class_year", "draft_round",
        "draft_pick_in_round", "draft_pick_overall", "draft_source",
    ]
    if common_cache.empty:
        common_cache = pd.DataFrame(columns=cache_cols)
    else:
        for c in cache_cols:
            if c not in common_cache.columns:
                common_cache[c] = np.nan
        common_cache = common_cache[cache_cols].copy()

    out = universe.merge(drafted, on=["player_id"], how="left", suffixes=("", "_drafted"))
    out = out.merge(common_cache, on=["player_id"], how="left", suffixes=("", "_cache"))

    # Resolve names.
    out["player_name"] = out["player_name"].fillna(out.get("player_name_drafted", ""))

    # Prefer DraftHistory for drafted players, fallback to CommonPlayerInfo cache.
    for col in ["draft_class_year", "draft_round", "draft_pick_in_round", "draft_pick_overall"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(
            pd.to_numeric(out.get(f"{col}_cache"), errors="coerce")
        )

    out["draft_source"] = out["draft_source"].fillna(out.get("draft_source_cache", "unknown"))
    out["draft_source"] = out["draft_source"].fillna("unknown")

    out["draft_team_id"] = out.get("draft_team_id", "")
    out["draft_team_abbreviation"] = out.get("draft_team_abbreviation", "")
    out["draft_type"] = out.get("draft_type", "")

    out["is_drafted"] = out["draft_pick_overall"].notna()
    out["is_undrafted"] = ~out["is_drafted"]
    out["draft_tier"] = out["draft_pick_overall"].map(_draft_tier)

    final_cols = [
        "player_id",
        "player_name",
        "draft_class_year",
        "draft_round",
        "draft_pick_in_round",
        "draft_pick_overall",
        "draft_tier",
        "is_drafted",
        "is_undrafted",
        "draft_team_id",
        "draft_team_abbreviation",
        "draft_type",
        "draft_source",
    ]

    for c in final_cols:
        if c not in out.columns:
            out[c] = np.nan

    out = out[final_cols].copy()
    out["player_id"] = _norm_id(out["player_id"])
    out["player_name"] = out["player_name"].astype(str)
    out = out.drop_duplicates(subset=["player_id"], keep="first")

    # Normalize textual nulls.
    out["draft_team_abbreviation"] = out["draft_team_abbreviation"].fillna("").astype(str).str.upper()
    out["draft_type"] = out["draft_type"].fillna("").astype(str)
    out["draft_source"] = out["draft_source"].fillna("unknown").astype(str)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch player draft history for local pipeline players")
    parser.add_argument(
        "--no-common-info",
        action="store_true",
        help="Skip CommonPlayerInfo fallback for missing players",
    )
    args = parser.parse_args()

    result = build_player_draft_history(use_common_info=not args.no_common_info)

    PLAYER_DRAFT_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(PLAYER_DRAFT_HISTORY_PATH, index=False)
    csv_path = PLAYER_DRAFT_HISTORY_PATH.with_suffix(".csv")
    result.to_csv(csv_path, index=False)

    print("\nSaved draft history:")
    print(f"  {PLAYER_DRAFT_HISTORY_PATH}")
    print(f"  {csv_path}")
    print(f"  rows={len(result)}")
    print(f"  drafted={int(result['is_drafted'].sum())}, undrafted={int(result['is_undrafted'].sum())}")
    print(f"  with_draft_class_year={int(result['draft_class_year'].notna().sum())}")


if __name__ == "__main__":
    main()
