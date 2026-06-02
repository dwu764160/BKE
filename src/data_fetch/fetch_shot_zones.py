"""
src/data_fetch/fetch_shot_zones.py
=============================================================================
Fetches shot zone data from NBA.com using LeagueDashPlayerShotLocations endpoint.

This gives us per-player FGM/FGA/FG_PCT broken down by zone:
  - Restricted Area (at rim)
  - In The Paint (Non-RA) (paint non-restricted)
  - Mid-Range
  - Left Corner 3
  - Right Corner 3
  - Above the Break 3
  - Backcourt

Uses DistanceRange='By Zone' and fetches for all players league-wide in one call.
=============================================================================
"""

import pandas as pd
import numpy as np
import time
import os
import sys
import json
import random
from curl_cffi import requests
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data.schema_contract import load_standardized, save_standardized

DATA_DIR = Path("data/tracking")
CACHE_DIR = Path("data/tracking_cache")

from src.modeling.model_config import SEASONS

SHOT_ZONE_NAMES = [
    "Restricted Area",
    "In The Paint (Non-RA)",
    "Mid-Range",
    "Left Corner 3",
    "Right Corner 3",
    "Above the Break 3",
    "Backcourt",
    "Corner 3",
]

# Short prefixes we'll use in the output columns
ZONE_PREFIXES = {
    "Restricted Area": "RA",
    "In The Paint (Non-RA)": "PAINT",
    "Mid-Range": "MR",
    "Left Corner 3": "LC3",
    "Right Corner 3": "RC3",
    "Above the Break 3": "AB3",
    "Backcourt": "BC",
    "Corner 3": "C3",
}


def smart_sleep():
    time.sleep(random.uniform(1.5, 3.0))


def _extract_result_payload(json_data):
    """Return result payload across known NBA stats response shapes."""
    if not isinstance(json_data, dict):
        return None

    result_sets = json_data.get("resultSets")
    if isinstance(result_sets, list) and result_sets:
        payload = result_sets[0]
        return payload if isinstance(payload, dict) else None
    if isinstance(result_sets, dict):
        return result_sets

    result_set = json_data.get("resultSet")
    if isinstance(result_set, dict):
        return result_set

    return None


def fetch_shot_locations(season: str) -> pd.DataFrame:
    """
    Fetch LeagueDashPlayerShotLocations for a given season.
    Returns a DataFrame with columns:
        PLAYER_ID, PLAYER_NAME, TEAM_ID, TEAM_ABBREVIATION, AGE,
        RA_FGM, RA_FGA, RA_FG_PCT,
        PAINT_FGM, PAINT_FGA, PAINT_FG_PCT,
        MR_FGM, MR_FGA, MR_FG_PCT,
        LC3_FGM, LC3_FGA, LC3_FG_PCT,
        RC3_FGM, RC3_FGA, RC3_FG_PCT,
        AB3_FGM, AB3_FGA, AB3_FG_PCT,
        BC_FGM, BC_FGA, BC_FG_PCT
    """
    cache_path = CACHE_DIR / f"shot_locations_{season}.json"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    json_data = None

    # Check cache first
    if cache_path.exists():
        try:
            with open(cache_path, "r") as f:
                json_data = json.load(f)
                if _extract_result_payload(json_data) is None:
                    json_data = None
        except Exception:
            json_data = None

    if json_data is None:
        url = "https://stats.nba.com/stats/leaguedashplayershotlocations"
        params = {
            "College": "",
            "Conference": "",
            "Country": "",
            "DateFrom": "",
            "DateTo": "",
            "DistanceRange": "By Zone",
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
            "PORound": "",
            "PaceAdjust": "N",
            "PerMode": "PerGame",
            "Period": "0",
            "PlayerExperience": "",
            "PlayerPosition": "",
            "PlusMinus": "N",
            "Rank": "N",
            "Season": season,
            "SeasonSegment": "",
            "SeasonType": "Regular Season",
            "ShotClockRange": "",
            "StarterBench": "",
            "TeamID": "",
            "VsConference": "",
            "VsDivision": "",
            "Weight": "",
        }

        headers = {
            'Accept': 'application/json, text/plain, */*',
            'Connection': 'keep-alive',
            'Origin': 'https://www.nba.com',
            'Referer': 'https://www.nba.com/stats/players/shooting',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'x-nba-stats-origin': 'stats',
            'x-nba-stats-token': 'true',
        }

        print(f"   Fetching from NBA API for {season}...", end=" ", flush=True)
        try:
            resp = requests.get(
                url, params=params, headers=headers,
                impersonate="chrome110", timeout=30
            )
            if resp.status_code != 200:
                print(f"❌ HTTP {resp.status_code}")
                return pd.DataFrame()

            json_data = resp.json()

            # Save to cache
            with open(cache_path, "w") as f:
                json.dump(json_data, f)
            print("✅ cached", flush=True)

        except Exception as e:
            print(f"❌ Error: {e}")
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Parse the peculiar multi-header format
    # ------------------------------------------------------------------
    # resultSets is a DICT (not a list) with keys: name, headers, rowSet
    # headers: [ {name: "SHOT_CATEGORY", columnNames: [zone_names], columnSpan: 3, columnsToSkip: 6},
    #            {name: "columns", columnNames: [col_names], columnSpan: 1} ]
    # rowSet: [[...], ...]
    #
    # Actual columns: 6 base cols + 8 zones * 3 stats = 30 total
    rs = _extract_result_payload(json_data)
    if rs is None:
        print("   ⚠️ Unexpected result payload format")
        return pd.DataFrame()

    row_set = rs.get('rowSet', [])

    if not row_set:
        print("   ⚠️ Empty rowSet")
        return pd.DataFrame()

    # Build column names manually based on the known structure
    base_cols = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "AGE", "NICKNAME"]
    zone_stat_cols = []
    for zone_name in SHOT_ZONE_NAMES:
        prefix = ZONE_PREFIXES[zone_name]
        zone_stat_cols.extend([f"{prefix}_FGM", f"{prefix}_FGA", f"{prefix}_FG_PCT"])

    all_cols = base_cols + zone_stat_cols

    # Verify column count matches
    if len(row_set[0]) != len(all_cols):
        print(f"   ⚠️ Column mismatch: expected {len(all_cols)}, got {len(row_set[0])}")
        # Try to adapt — the row length tells us the real column count
        actual_len = len(row_set[0])
        if actual_len > len(all_cols):
            # Pad with extra columns
            for i in range(actual_len - len(all_cols)):
                all_cols.append(f"EXTRA_{i}")
        else:
            all_cols = all_cols[:actual_len]

    df = pd.DataFrame(row_set, columns=all_cols)

    # Ensure numeric types for stat columns
    for col in zone_stat_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def compute_shot_zone_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived shot zone features from the raw zone data.

    Adds:
        TOTAL_FGA: sum of all zone FGA
        AT_RIM_FREQ: RA_FGA / TOTAL_FGA  (restricted area frequency)
        PAINT_FREQ: PAINT_FGA / TOTAL_FGA  (in-the-paint non-RA frequency)
        MIDRANGE_FREQ: MR_FGA / TOTAL_FGA  (mid-range frequency)
        CORNER3_FREQ: (LC3_FGA + RC3_FGA) / TOTAL_FGA
        AB3_FREQ: AB3_FGA / TOTAL_FGA
        AT_RIM_FG_PCT: RA_FG_PCT (alias for clarity)
        MIDRANGE_FG_PCT: MR_FG_PCT (alias)
        AT_RIM_PLUS_PAINT_FREQ: (RA + PAINT) / TOTAL — interior frequency
    """
    if df.empty:
        return df

    # Total FGA across all zones
    fga_cols = [c for c in df.columns if c.endswith('_FGA')]
    df['TOTAL_FGA'] = df[fga_cols].sum(axis=1)

    total = df['TOTAL_FGA'].replace(0, np.nan)

    df['AT_RIM_FREQ'] = df['RA_FGA'] / total
    df['PAINT_FREQ'] = df['PAINT_FGA'] / total
    df['MIDRANGE_FREQ'] = df['MR_FGA'] / total
    df['CORNER3_FREQ'] = (df.get('LC3_FGA', 0) + df.get('RC3_FGA', 0)) / total
    df['AB3_FREQ'] = df['AB3_FGA'] / total

    # Composite interior frequency (at rim + paint non-RA)
    df['AT_RIM_PLUS_PAINT_FREQ'] = (df['RA_FGA'] + df['PAINT_FGA']) / total

    # Aliases for convenience
    df['AT_RIM_FG_PCT'] = df.get('RA_FG_PCT', np.nan)
    df['MIDRANGE_FG_PCT'] = df.get('MR_FG_PCT', np.nan)

    return df


def main():

    print("=" * 60)
    print("SHOT ZONE DATA FETCH (LeagueDashPlayerShotLocations)")
    print("=" * 60)

    for season in SEASONS:
        season_dir = DATA_DIR / season
        season_dir.mkdir(parents=True, exist_ok=True)
        outfile = season_dir / "shot_zones.parquet"

        if outfile.exists():
            existing = load_standardized(outfile)
            print(f"\n✅ {season}: Already cached ({len(existing)} players)")
            continue

        print(f"\n🏀 Processing {season}...")
        df = fetch_shot_locations(season)

        if df.empty:
            print(f"   ❌ No data for {season}")
            continue

        # Compute derived features
        df = compute_shot_zone_features(df)
        df['SEASON'] = season

        save_standardized(df, outfile)
        print(f"   ✅ Saved {len(df)} players → {outfile}")

        # Print summary stats
        print(f"   Avg AT_RIM_FREQ: {df['AT_RIM_FREQ'].mean():.3f}")
        print(f"   Avg MIDRANGE_FREQ: {df['MIDRANGE_FREQ'].mean():.3f}")
        print(f"   Avg PAINT_FREQ: {df['PAINT_FREQ'].mean():.3f}")
        print(f"   Avg AB3_FREQ: {df['AB3_FREQ'].mean():.3f}")

        smart_sleep()

    print("\n✅ Shot Zone Fetch Complete.")


if __name__ == "__main__":
    main()
