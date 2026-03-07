"""
src/data_fetch/fetch_defensive_metrics.py
=============================================================================
Fetches additional defensive metrics from the NBA Stats API:
  1. Hustle Stats (deflections, loose balls, charges drawn, contested shots, boxouts)

Outputs:
  - data/tracking/{season}/hustle_stats.parquet
=============================================================================
"""

import pandas as pd
import time
import os
import sys
import json
import random
from curl_cffi import requests
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

DATA_DIR = Path("data/tracking")
CACHE_DIR = Path("data/tracking_cache")
SEASONS = ["2022-23", "2023-24", "2024-25"]


def ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def smart_sleep():
    time.sleep(random.uniform(1.5, 3.0))


def fetch_url_cached(url, params, cache_name, referer="https://www.nba.com/stats/players/hustle"):
    """Fetch URL with caching and TLS impersonation."""
    cache_path = CACHE_DIR / f"{cache_name}.json"

    if cache_path.exists():
        try:
            with open(cache_path, "r") as f:
                json_data = json.load(f)
                if 'resultSets' in json_data:
                    return parse_json(json_data)
        except Exception:
            pass

    headers = {
        'Accept': 'application/json, text/plain, */*',
        'Connection': 'keep-alive',
        'Origin': 'https://www.nba.com',
        'Referer': referer,
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'x-nba-stats-origin': 'stats',
        'x-nba-stats-token': 'true',
    }

    try:
        resp = requests.get(
            url, params=params, headers=headers,
            impersonate="chrome110", timeout=60
        )

        if resp.status_code != 200:
            print(f"  ⚠️ HTTP {resp.status_code}")
            return None

        json_data = resp.json()

        with open(cache_path, "w") as f:
            json.dump(json_data, f)

        return parse_json(json_data)

    except Exception as e:
        print(f"  ⚠️ Error: {e}")
        return None


def parse_json(json_data):
    """Parse NBA API JSON response into DataFrame."""
    try:
        result_sets = json_data.get('resultSets', [])
        if not result_sets:
            return pd.DataFrame()
        headers = result_sets[0]['headers']
        row_set = result_sets[0]['rowSet']
        return pd.DataFrame(row_set, columns=headers)
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Hustle Stats
# ---------------------------------------------------------------------------

def fetch_hustle_stats(season: str, per_mode: str = "PerGame") -> pd.DataFrame:
    """
    Fetch LeagueHustleStatsPlayer endpoint.

    Returns columns including:
    - CONTESTED_SHOTS, CONTESTED_SHOTS_2PT, CONTESTED_SHOTS_3PT
    - DEFLECTIONS
    - CHARGES_DRAWN
    - SCREEN_ASSISTS, SCREEN_AST_PTS
    - OFF_LOOSE_BALLS_RECOVERED, DEF_LOOSE_BALLS_RECOVERED, LOOSE_BALLS_RECOVERED
    - OFF_BOXOUTS, DEF_BOXOUTS, BOX_OUTS
    """
    print(f"  Fetching Hustle Stats ({per_mode}) for {season}...", end=" ")

    url = "https://stats.nba.com/stats/leaguehustlestatsplayer"
    params = {
        "LeagueID": "00",
        "PerMode": per_mode,
        "Season": season,
        "SeasonType": "Regular Season",
        "College": "",
        "Conference": "",
        "Country": "",
        "DateFrom": "",
        "DateTo": "",
        "Division": "",
        "DraftPick": "",
        "DraftYear": "",
        "Height": "",
        "Location": "",
        "Month": "",
        "OpponentTeamID": "",
        "Outcome": "",
        "PORound": "",
        "PlayerExperience": "",
        "PlayerPosition": "",
        "SeasonSegment": "",
        "TeamID": "",
        "VsConference": "",
        "VsDivision": "",
        "Weight": "",
    }

    cache_name = f"hustle_stats_{per_mode}_{season}"
    df = fetch_url_cached(url, params, cache_name)

    if df is not None and not df.empty:
        df.columns = [c.upper() for c in df.columns]
        df["SEASON"] = season
        print(f"✅ {len(df)} players")
        return df
    else:
        print("❌ Failed")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("DEFENSIVE METRICS FETCHER")
    print("  - Hustle Stats (deflections, loose balls, charges, contested shots)")
    print("=" * 70)

    ensure_dirs()

    all_hustle_pg = []
    all_hustle_total = []

    for season in SEASONS:
        print(f"\n📊 Processing {season}...")
        season_dir = DATA_DIR / season
        season_dir.mkdir(exist_ok=True)

        # --- Hustle Stats (PerGame) ---
        outfile_pg = season_dir / "hustle_stats.parquet"
        if outfile_pg.exists():
            print(f"  Hustle stats (PerGame) already exist for {season}, skipping...")
            all_hustle_pg.append(pd.read_parquet(outfile_pg))
        else:
            df = fetch_hustle_stats(season, "PerGame")
            if not df.empty:
                df.to_parquet(outfile_pg, index=False)
                all_hustle_pg.append(df)
            smart_sleep()

        # --- Hustle Stats (Totals) ---
        outfile_totals = season_dir / "hustle_stats_totals.parquet"
        if outfile_totals.exists():
            print(f"  Hustle stats (Totals) already exist for {season}, skipping...")
            all_hustle_total.append(pd.read_parquet(outfile_totals))
        else:
            df = fetch_hustle_stats(season, "Totals")
            if not df.empty:
                df.to_parquet(outfile_totals, index=False)
                all_hustle_total.append(df)
            smart_sleep()

    # Summary
    print("\n" + "=" * 70)
    print("FETCH COMPLETE")
    print("=" * 70)

    if all_hustle_pg:
        combined = pd.concat(all_hustle_pg, ignore_index=True)
        print(f"\nHustle Stats (PerGame): {len(combined)} total player-season rows")
        print(f"  Columns: {list(combined.columns)}")
        # Sample top deflectors
        if 'DEFLECTIONS' in combined.columns:
            s25 = combined[combined['SEASON'] == '2024-25']
            if not s25.empty:
                top = s25.nlargest(10, 'DEFLECTIONS')
                print("\n  Top 10 Deflectors (2024-25):")
                for _, r in top.iterrows():
                    print(f"    {r['PLAYER_NAME']:25} DEF={r['DEFLECTIONS']:.1f} CONT={r.get('CONTESTED_SHOTS',0):.1f} CHG={r.get('CHARGES_DRAWN',0):.1f}")

    if all_hustle_total:
        combined_t = pd.concat(all_hustle_total, ignore_index=True)
        print(f"\nHustle Stats (Totals): {len(combined_t)} total player-season rows")


if __name__ == "__main__":
    main()
