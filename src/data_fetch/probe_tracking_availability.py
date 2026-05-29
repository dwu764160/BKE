"""
src/data_fetch/probe_tracking_availability.py
=============================================================================
Probes NBA API to determine which tracking columns return valid data for
pre-2022 seasons. Run this before committing to a backfill strategy.

Fetches one endpoint per season and checks that key columns have non-null
values for a meaningful fraction of players (>= 50%).

Output:
  docs/tracking_availability.md  — human/AI-readable decision table

Usage:
  python3 src/data_fetch/probe_tracking_availability.py
=============================================================================
"""

import json
import os
import sys
import time
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from curl_cffi import requests
    USE_CURL_CFFI = True
except ImportError:
    import urllib.request
    USE_CURL_CFFI = False

PROBE_SEASONS = ["2021-22", "2018-19", "2017-18"]

# leaguedashptstats requires a full set of filter params (even if all empty/zero).
# Sending only 6 minimal params causes HTTP 500 on all seasons.
_PTSTATS_REQUIRED_DEFAULTS = {
    "College": "", "Conference": "", "Country": "",
    "DateFrom": "", "DateTo": "", "Division": "",
    "DraftPick": "", "DraftYear": "", "GameScope": "",
    "GameSegment": "", "Height": "", "ISTRound": "",
    "LastNGames": 0, "Location": "", "Month": 0,
    "OpponentTeamID": 0, "Outcome": "", "PORound": 0,
    "Period": 0, "PlayerExperience": "", "PlayerPosition": "",
    "SeasonSegment": "", "StarterBench": "", "TeamID": 0,
    "VsConference": "", "VsDivision": "", "Weight": "",
}


PROBE_ENDPOINTS = {
    "drives": {
        "url": "https://stats.nba.com/stats/leaguedashptstats",
        "params": {
            **_PTSTATS_REQUIRED_DEFAULTS,
            "PtMeasureType": "Drives",
            "PerMode": "PerGame",
            "LeagueID": "00",
            "SeasonType": "Regular Season",
            "PlayerOrTeam": "Player",
        },
        "key_columns": ["DRIVES", "DRIVE_FGA", "DRIVE_PTS"],
        "tier": "full_bke",
    },
    "passing": {
        "url": "https://stats.nba.com/stats/leaguedashptstats",
        "params": {
            **_PTSTATS_REQUIRED_DEFAULTS,
            "PtMeasureType": "Passing",
            "PerMode": "PerGame",
            "LeagueID": "00",
            "SeasonType": "Regular Season",
            "PlayerOrTeam": "Player",
        },
        "key_columns": ["PASSES_MADE", "POTENTIAL_AST"],
        "tier": "full_bke",
    },
    "possessions": {
        "url": "https://stats.nba.com/stats/leaguedashptstats",
        "params": {
            **_PTSTATS_REQUIRED_DEFAULTS,
            "PtMeasureType": "Possessions",
            "PerMode": "PerGame",
            "LeagueID": "00",
            "SeasonType": "Regular Season",
            "PlayerOrTeam": "Player",
        },
        "key_columns": ["TOUCHES", "FRONT_CT_TOUCHES", "TIME_OF_POSS"],
        "tier": "full_bke",
    },
    "synergy_isolation": {
        "url": "https://stats.nba.com/stats/synergyplaytypes",
        "params": {
            "PlayType": "Isolation",
            "PerMode": "PerGame",
            "LeagueID": "00",
            "SeasonType": "Regular Season",
            "PlayerOrTeam": "P",
            "TypeGrouping": "offensive",
        },
        "key_columns": ["POSS_PCT", "PPP", "POSS"],
        "tier": "full_bke",
    },
    "synergy_spotup": {
        "url": "https://stats.nba.com/stats/synergyplaytypes",
        "params": {
            "PlayType": "Spotup",
            "PerMode": "PerGame",
            "LeagueID": "00",
            "SeasonType": "Regular Season",
            "PlayerOrTeam": "P",
            "TypeGrouping": "offensive",
        },
        "key_columns": ["POSS_PCT", "PPP", "POSS"],
        "tier": "full_bke",
    },
    "hustle": {
        "url": "https://stats.nba.com/stats/leaguehustlestatsplayer",
        "params": {
            "PerMode": "PerGame",
            "LeagueID": "00",
            "SeasonType": "Regular Season",
        },
        "key_columns": ["CONTESTED_SHOTS", "DEFLECTIONS", "LOOSE_BALLS_RECOVERED"],
        "tier": "full_bke",
    },
    "defense": {
        "url": "https://stats.nba.com/stats/leaguedashptdefend",
        "params": {
            "DefenseCategory": "Overall",
            "PerMode": "PerGame",
            "LeagueID": "00",
            "SeasonType": "Regular Season",
        },
        "key_columns": ["FREQ", "FG_PCT", "FG2_PCT"],
        "tier": "reduced_bke",
    },
    "pull_up": {
        "url": "https://stats.nba.com/stats/leaguedashptstats",
        "params": {
            **_PTSTATS_REQUIRED_DEFAULTS,
            "PtMeasureType": "PullUpShot",
            "PerMode": "PerGame",
            "LeagueID": "00",
            "SeasonType": "Regular Season",
            "PlayerOrTeam": "Player",
        },
        "key_columns": ["PULL_UP_FGA", "PULL_UP_PTS"],
        "tier": "full_bke",
    },
    "catch_shoot": {
        "url": "https://stats.nba.com/stats/leaguedashptstats",
        "params": {
            **_PTSTATS_REQUIRED_DEFAULTS,
            "PtMeasureType": "CatchShoot",
            "PerMode": "PerGame",
            "LeagueID": "00",
            "SeasonType": "Regular Season",
            "PlayerOrTeam": "Player",
        },
        "key_columns": ["CATCH_SHOOT_FGA", "CATCH_SHOOT_PTS"],
        "tier": "full_bke",
    },
}

HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
    "Host": "stats.nba.com",
    "Origin": "https://www.nba.com",
    "Referer": "https://www.nba.com/stats/players/drives",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}


def fetch_endpoint(url: str, params: dict) -> dict | None:
    """Fetch an NBA stats endpoint; return parsed JSON or None on failure."""
    all_params = {**params}
    param_str = "&".join(f"{k}={v}" for k, v in all_params.items())
    full_url = f"{url}?{param_str}"

    try:
        if USE_CURL_CFFI:
            response = requests.get(full_url, headers=HEADERS, impersonate="chrome124", timeout=30)
            if response.status_code == 200:
                return response.json()
            else:
                return None
        else:
            req = urllib.request.Request(full_url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
    except Exception as e:
        print(f"    Error: {e}")
        return None


def parse_result(json_data: dict, key_columns: list) -> dict:
    """Return column availability stats from an NBA stats API response."""
    if json_data is None:
        return {"available": False, "reason": "request_failed"}

    result_sets = json_data.get("resultSets") or json_data.get("resultSet")
    if not result_sets:
        return {"available": False, "reason": "no_result_sets"}

    rs = result_sets[0] if isinstance(result_sets, list) else result_sets
    headers = rs.get("headers", [])
    rows = rs.get("rowSet", [])

    if not rows:
        return {"available": False, "reason": "empty_rowset", "n_players": 0}

    n_players = len(rows)
    col_idx = {h: i for i, h in enumerate(headers)}

    col_stats = {}
    for col in key_columns:
        if col not in col_idx:
            col_stats[col] = {"found": False, "non_null_pct": 0.0}
            continue
        idx = col_idx[col]
        non_null = sum(1 for row in rows if row[idx] is not None and row[idx] != "")
        col_stats[col] = {
            "found": True,
            "non_null_pct": round(non_null / n_players, 3),
        }

    all_found = all(v["found"] for v in col_stats.values())
    min_fill = min((v["non_null_pct"] for v in col_stats.values()), default=0.0)

    return {
        "available": all_found and min_fill >= 0.5,
        "n_players": n_players,
        "column_stats": col_stats,
        "min_fill_rate": round(min_fill, 3),
        "reason": "ok" if (all_found and min_fill >= 0.5) else "low_fill_or_missing_columns",
    }


def determine_tier(endpoint_results: dict) -> str:
    """Determine BKE tier from endpoint probe results."""
    full_bke_endpoints = [k for k, v in PROBE_ENDPOINTS.items() if v["tier"] == "full_bke"]
    reduced_bke_endpoints = [k for k, v in PROBE_ENDPOINTS.items() if v["tier"] == "reduced_bke"]

    full_available = all(
        endpoint_results.get(ep, {}).get("available", False) for ep in full_bke_endpoints
    )
    reduced_available = all(
        endpoint_results.get(ep, {}).get("available", False) for ep in reduced_bke_endpoints
    )

    if full_available:
        return "full"
    elif reduced_available:
        return "reduced"
    else:
        return "rapm_only"


def write_tracking_availability(results: dict) -> None:
    """Write docs/tracking_availability.md from probe results."""
    lines = [
        "# Tracking Data Availability by Season",
        "",
        "**Generated by:** `src/data_fetch/probe_tracking_availability.py`",
        "**Purpose:** Determine BKE tier for each pre-2022 season before committing to a backfill strategy.",
        "",
        "---",
        "",
        "## BKE Tier Definitions",
        "",
        "| Tier | Meaning |",
        "|---|---|",
        "| `full` | All 9 BKE dimensions available (all tracking + synergy endpoints return data) |",
        "| `reduced` | BKE without PCValue surplus (partial tracking — drives+at-rim but no playtype) |",
        "| `rapm_only` | No tracking data — RAPM backbone score only |",
        "| `covid` | Season flagged as COVID-disrupted (always rapm_only, reduced decay weight) |",
        "",
        "---",
        "",
        "## Season Probe Results",
        "",
        "| Season | BKE Tier | Drives | Passing | Synergy | Hustle | Notes |",
        "|---|---|---|---|---|---|---|",
    ]

    for season, season_data in sorted(results.items()):
        tier = season_data.get("tier", "unknown")
        ep = season_data.get("endpoints", {})

        def sym(key):
            r = ep.get(key, {})
            if not r:
                return "?"
            if r.get("available"):
                return f"✓ ({r.get('n_players', '?')} players)"
            else:
                return f"✗ ({r.get('reason', '?')})"

        lines.append(
            f"| {season} | `{tier}` | {sym('drives')} | {sym('passing')} | "
            f"{sym('synergy_isolation')} | {sym('hustle')} | |"
        )

    lines += [
        "",
        "---",
        "",
        "## Endpoint Detail",
        "",
    ]

    for season, season_data in sorted(results.items()):
        lines.append(f"### {season}")
        lines.append("")
        lines.append(f"**Determined tier:** `{season_data.get('tier', 'unknown')}`")
        lines.append("")
        ep = season_data.get("endpoints", {})
        for ep_name, ep_data in ep.items():
            avail = "PASS" if ep_data.get("available") else "FAIL"
            n = ep_data.get("n_players", 0)
            fill = ep_data.get("min_fill_rate", 0)
            lines.append(f"- **{ep_name}**: {avail} (n_players={n}, min_fill={fill:.0%})")
        lines.append("")

    out = Path("docs/tracking_availability.md")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote: {out}")


def main():
    print("NBA API Tracking Availability Probe")
    print("=" * 50)
    print(f"Probing seasons: {PROBE_SEASONS}")
    print()

    all_results = {}

    for season in PROBE_SEASONS:
        print(f"\n--- {season} ---")
        season_results = {}

        for ep_name, ep_config in PROBE_ENDPOINTS.items():
            params = {**ep_config["params"], "Season": season}
            print(f"  {ep_name}... ", end="", flush=True)
            time.sleep(random.uniform(1.5, 3.0))

            json_data = fetch_endpoint(ep_config["url"], params)
            result = parse_result(json_data, ep_config["key_columns"])
            season_results[ep_name] = result

            status = "PASS" if result.get("available") else "FAIL"
            n = result.get("n_players", 0)
            print(f"{status} (n={n})")

        tier = determine_tier(season_results)
        all_results[season] = {
            "tier": tier,
            "endpoints": season_results,
        }
        print(f"  → BKE tier: {tier}")

    write_tracking_availability(all_results)
    print("\nProbe complete.")
    return all_results


if __name__ == "__main__":
    main()
