"""
src/data_fetch/fetch_darko_manual.py
=============================================================================
Automated DARKO fetch script.

Fetches DARKO data for 2022-23, 2023-24, and 2024-25 seasons from nbarapm.com MetricHistory page
and saves each season as a separate parquet file in data/historical/darko/raw.
=============================================================================
"""


import os
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup



RAW_DIR = "data/historical/darko/raw"
os.makedirs(RAW_DIR, exist_ok=True)

BASE_URL = "https://www.nbarapm.com/load/DARKO"
SEASON_MAP = {
    "2022-23": 2023,
    "2023-24": 2024,
    "2024-25": 2025,
    "2025-26": 2026,
}



def fetch_darko_season(season_label: str, season_filter: int, all_data: pd.DataFrame) -> pd.DataFrame:
    """
    Filters the DARKO dataset for a given season.
    """
    df = all_data[all_data['season'] == season_filter].copy()
    if df.empty:
        raise RuntimeError(f"No DARKO data found for {season_label} (season={season_filter})")
    return df



def main():
    print(f"Fetching full DARKO dataset from {BASE_URL}...")
    resp = requests.get(BASE_URL)
    resp.raise_for_status()
    data = resp.json()
    all_data = pd.DataFrame(data)
    print(f"Fetched {len(all_data)} rows total.")

    results = []
    for season_label, season_filter in SEASON_MAP.items():
        print(f"Filtering DARKO for {season_label} (season={season_filter})...")
        try:
            df = fetch_darko_season(season_label, season_filter, all_data)
            out_path = os.path.join(RAW_DIR, f"darko_{season_label}.parquet")
            df.to_parquet(out_path, index=False)
            num_players = df['player_name'].nunique() if 'player_name' in df.columns else 'N/A'
            print(f"✅ Saved {season_label}: {out_path} (rows={len(df)}, unique players={num_players})")
            results.append((season_label, out_path, len(df), num_players))
        except Exception as e:
            print(f"❌ Failed to filter {season_label}: {e}")
    print("Done.")



# Entrypoint
if __name__ == "__main__":
    main()
