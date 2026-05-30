"""
src/data_fetch/fetch_pregame_lineups.py
=============================================================================
Step 0b — Live pre-game lineup + injury fetch (MARKET-TRACK infra).

Fetches, for each game on a given date, the ACTUAL confirmed starters and the
injury/availability status ~T-60/30 min before tip. The market track uses this
to OVERRIDE the projected starters from `lineup_projection.py` (the projector's
season-level guess) with the real five, then re-prices props/totals. The game
track does NOT use this — it runs on fully projected lineups (see
`docs/phase_v1_architecture.md` §8 track divergence).

  ┌─ projected lineup (offline, lineup_projection.py)
  │     └── GAME track consumes as-is
  └─ live override (THIS script, T-60/30)
        └── MARKET track replaces projected starters + flags injuries

NETWORK NOTE — runs in the user's normal (residential-IP) environment, NOT the
dev sandbox: `stats.nba.com` TCP-connects from the sandbox but the HTTP layer is
silently throttled (Akamai datacenter-IP filtering) and times out. `cdn.nba.com`
and `github.com` work from the sandbox; only the stats host is blocked. Verify
this fetch from an environment whose IP the stats host accepts. See
`docs/findings/network_stats_nba_throttle_2026-05-30.md`.

Schema (shared with the market track):
  data/processed/forecast/pregame_lineups_<date>.parquet
    game_id, game_date, team_id, team_abbreviation, home_away,
    actual_starters (list[player_id]), confirmed (bool),
    out_player_ids (list), doubtful_player_ids (list), questionable_player_ids (list),
    source, minutes_to_tip, fetched_at_utc

Usage (in a networked env):
  python3 src/data_fetch/fetch_pregame_lineups.py --date 2026-01-15
  python3 src/data_fetch/fetch_pregame_lineups.py            # defaults to today
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

try:
    from curl_cffi import requests  # impersonates a real browser TLS fingerprint
    _IMPERSONATE = "chrome124"
except Exception:  # pragma: no cover - fallback for envs without curl_cffi
    import requests  # type: ignore
    _IMPERSONATE = None

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "data" / "processed" / "forecast"
CACHE_DIR = ROOT / "data" / "tracking_cache"

SCOREBOARD_ENDPOINT = "https://stats.nba.com/stats/scoreboardv2"
BOXSUMMARY_ENDPOINT = "https://stats.nba.com/stats/boxscoresummaryv2"
# Confirmed starters appear in the traditional boxscore (START_POSITION populated)
# once lineups are posted (~T-30 to first tip).
BOXTRAD_ENDPOINT = "https://stats.nba.com/stats/boxscoretraditionalv2"

DEFAULT_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Connection": "keep-alive",
    "Origin": "https://www.nba.com",
    "Referer": "https://www.nba.com/",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}


def _norm_id(value) -> str:
    return str(value).strip().replace(".0", "")


def _get(url: str, params: Dict) -> Optional[Dict]:
    kw = dict(headers=DEFAULT_HEADERS, params=params, timeout=30)
    if _IMPERSONATE:
        kw["impersonate"] = _IMPERSONATE
    try:
        resp = requests.get(url, **kw)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:  # network-throttled env -> surfaces clearly
        print(f"  ! request failed ({url.split('/')[-1]}): {exc}")
        return None


def _result_frame(payload: Dict, name: str) -> pd.DataFrame:
    for rs in payload.get("resultSets", []):
        if rs.get("name") == name:
            return pd.DataFrame(rs["rows"], columns=rs["headers"])
    return pd.DataFrame()


def fetch_scoreboard(date_str: str) -> pd.DataFrame:
    """Today's games: game_id, tip time, home/away team ids."""
    mmddyyyy = datetime.strptime(date_str, "%Y-%m-%d").strftime("%m/%d/%Y")
    payload = _get(SCOREBOARD_ENDPOINT, {"GameDate": mmddyyyy, "LeagueID": "00", "DayOffset": 0})
    if not payload:
        return pd.DataFrame()
    games = _result_frame(payload, "GameHeader")
    if games.empty:
        return games
    keep = ["GAME_ID", "GAME_DATE_EST", "GAME_STATUS_TEXT", "HOME_TEAM_ID", "VISITOR_TEAM_ID"]
    return games[[c for c in keep if c in games.columns]].copy()


def fetch_starters_and_status(game_id: str) -> Dict[str, Dict]:
    """Confirmed starters (START_POSITION set) + inactives per team for one game.

    Returns {team_id: {"starters": [ids], "inactive": [ids], "abbr": str}}.
    Starters populate when lineups are confirmed (~T-30); before that the
    boxscore returns players with empty START_POSITION -> `confirmed=False`.
    """
    out: Dict[str, Dict] = {}
    trad = _get(BOXTRAD_ENDPOINT, {
        "GameID": game_id, "StartPeriod": 0, "EndPeriod": 14,
        "StartRange": 0, "EndRange": 0, "RangeType": 0,
    })
    if trad:
        players = _result_frame(trad, "PlayerStats")
        for tid, tdf in players.groupby("TEAM_ID"):
            starters = [
                _norm_id(p) for p, sp in zip(tdf["PLAYER_ID"], tdf.get("START_POSITION", ""))
                if str(sp).strip()
            ]
            abbr = str(tdf["TEAM_ABBREVIATION"].iloc[0]) if "TEAM_ABBREVIATION" in tdf else None
            out[_norm_id(tid)] = {"starters": starters, "inactive": [], "abbr": abbr}

    summary = _get(BOXSUMMARY_ENDPOINT, {"GameID": game_id})
    if summary:
        inactive = _result_frame(summary, "InactivePlayers")
        if not inactive.empty and "TEAM_ID" in inactive.columns:
            for tid, idf in inactive.groupby("TEAM_ID"):
                key = _norm_id(tid)
                out.setdefault(key, {"starters": [], "inactive": [], "abbr": None})
                out[key]["inactive"] = [_norm_id(p) for p in idf["PLAYER_ID"]]
    return out


def build_pregame_table(date_str: str) -> pd.DataFrame:
    games = fetch_scoreboard(date_str)
    if games.empty:
        print(f"No games found for {date_str} (or stats host unreachable).")
        return pd.DataFrame()

    fetched_at = datetime.now(timezone.utc).isoformat()
    rows: List[Dict] = []
    for g in games.itertuples(index=False):
        gid = _norm_id(g.GAME_ID)
        info = fetch_starters_and_status(gid)
        home_id, away_id = _norm_id(g.HOME_TEAM_ID), _norm_id(g.VISITOR_TEAM_ID)
        for tid, home_away in [(home_id, "home"), (away_id, "away")]:
            d = info.get(tid, {"starters": [], "inactive": [], "abbr": None})
            rows.append({
                "game_id": gid,
                "game_date": date_str,
                "team_id": tid,
                "team_abbreviation": d.get("abbr"),
                "home_away": home_away,
                "actual_starters": d.get("starters", []),
                "confirmed": len(d.get("starters", [])) == 5,
                "out_player_ids": d.get("inactive", []),
                "doubtful_player_ids": [],      # populated when an injury-report source is wired
                "questionable_player_ids": [],  # populated when an injury-report source is wired
                "source": "stats.nba.com/boxscore",
                "minutes_to_tip": None,         # caller can compute from GAME_STATUS_TEXT/tip time
                "fetched_at_utc": fetched_at,
            })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    args = ap.parse_args()

    df = build_pregame_table(args.date)
    if df.empty:
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"pregame_lineups_{args.date}.parquet"
    df.to_parquet(out_path, index=False)
    n_conf = int(df["confirmed"].sum())
    print(f"Saved {len(df)} team rows ({n_conf} with confirmed starters) -> {out_path}")


if __name__ == "__main__":
    main()
