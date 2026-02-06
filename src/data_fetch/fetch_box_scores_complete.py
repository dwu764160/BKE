"""
src/data_fetch/fetch_box_scores_complete.py
Fetches COMPLETE player box score data for ALL players who appear in possession data.

Key Improvements:
1. Uses leaguedashplayerstats (ALL players in season) instead of individual player logs
2. Fetches both Base and Advanced stats for BPM calculation
3. No ROSTERSTATUS filter - gets everyone who played
"""

import pandas as pd
import numpy as np
import time
import os
import sys
import json
import random
from pathlib import Path
from curl_cffi import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

DATA_DIR = Path("data/historical")
OUTPUT_DIR = Path("data/historical")
SEASONS = ["2022-23", "2023-24", "2024-25"]

def fetch_league_player_stats(season, measure_type="Base"):
    """
    Fetch ALL player stats for a season using leaguedashplayerstats endpoint.
    This returns EVERY player who played, not just active roster.
    
    measure_type: "Base" for traditional stats, "Advanced" for advanced
    """
    url = "https://stats.nba.com/stats/leaguedashplayerstats"
    
    params = {
        "MeasureType": measure_type,
        "PerMode": "Totals",  # Get totals for aggregation
        "PlusMinus": "N",
        "PaceAdjust": "N",
        "Rank": "N",
        "LeagueID": "00",
        "Season": season,
        "SeasonType": "Regular Season",
        "PORound": "0",
        "Outcome": "",
        "Location": "",
        "Month": "0",
        "SeasonSegment": "",
        "DateFrom": "",
        "DateTo": "",
        "OpponentTeamID": "0",
        "VsConference": "",
        "VsDivision": "",
        "TeamID": "0",
        "Conference": "",
        "Division": "",
        "GameSegment": "",
        "Period": "0",
        "ShotClockRange": "",
        "LastNGames": "0",
        "GameScope": "",
        "PlayerExperience": "",
        "PlayerPosition": "",
        "StarterBench": "",
        "DraftYear": "",
        "DraftPick": "",
        "College": "",
        "Country": "",
        "Height": "",
        "Weight": ""
    }
    
    headers = {
        'Accept': 'application/json, text/plain, */*',
        'Connection': 'keep-alive',
        'Origin': 'https://www.nba.com',
        'Referer': 'https://www.nba.com/stats/players/traditional',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'x-nba-stats-origin': 'stats',
        'x-nba-stats-token': 'true',
    }
    
    try:
        print(f"   Fetching {measure_type} stats for {season}...", end=" ")
        resp = requests.get(
            url, params=params, headers=headers,
            impersonate="chrome110", timeout=30
        )
        
        if resp.status_code != 200:
            print(f"❌ Status {resp.status_code}")
            return None
        
        json_data = resp.json()
        cols = json_data['resultSets'][0]['headers']
        rows = json_data['resultSets'][0]['rowSet']
        
        df = pd.DataFrame(rows, columns=cols)
        print(f"✅ {len(df)} players")
        return df
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def fetch_all_seasons():
    """Fetch base and advanced stats for all seasons."""
    all_base = []
    all_advanced = []
    
    for season in SEASONS:
        print(f"\n📊 Fetching {season}...")
        
        # Base stats (PTS, REB, AST, STL, BLK, TOV, etc.)
        base_df = fetch_league_player_stats(season, "Base")
        if base_df is not None:
            base_df['SEASON'] = season
            all_base.append(base_df)
        
        time.sleep(random.uniform(1.5, 2.5))
        
        # Advanced stats (TS_PCT, USG_PCT, AST_PCT, etc.)
        adv_df = fetch_league_player_stats(season, "Advanced")
        if adv_df is not None:
            adv_df['SEASON'] = season
            all_advanced.append(adv_df)
        
        time.sleep(random.uniform(1.5, 2.5))
    
    return all_base, all_advanced


def merge_and_save(base_dfs, adv_dfs):
    """Merge base and advanced stats, save results."""
    if not base_dfs:
        print("❌ No base stats fetched")
        return
    
    base_combined = pd.concat(base_dfs, ignore_index=True)
    
    if adv_dfs:
        adv_combined = pd.concat(adv_dfs, ignore_index=True)
        # Merge on PLAYER_ID + SEASON (keep only unique advanced columns)
        adv_cols = ['PLAYER_ID', 'SEASON'] + [c for c in adv_combined.columns 
                   if c not in base_combined.columns or c in ['PLAYER_ID', 'SEASON']]
        merged = base_combined.merge(
            adv_combined[adv_cols], 
            on=['PLAYER_ID', 'SEASON'], 
            how='left'
        )
    else:
        merged = base_combined
    
    # Standardize player_id column
    merged['Player_ID'] = merged['PLAYER_ID']
    
    # Save
    out_path = OUTPUT_DIR / "complete_player_season_stats.parquet"
    merged.to_parquet(out_path, index=False)
    print(f"\n✅ Saved {len(merged)} rows to {out_path}")
    
    # Also save CSV for inspection
    csv_path = OUTPUT_DIR / "complete_player_season_stats.csv"
    merged.to_csv(csv_path, index=False)
    print(f"✅ CSV saved to {csv_path}")
    
    return merged


def compare_coverage(merged_df):
    """Compare coverage vs possession data."""
    print("\n" + "="*60)
    print("COVERAGE COMPARISON")
    print("="*60)
    
    for season in SEASONS:
        # Get players from new stats
        new_players = set(merged_df[merged_df['SEASON'] == season]['PLAYER_ID'].astype(str).unique())
        
        # Get players from possession data
        try:
            poss_df = pd.read_parquet(f'data/historical/possessions_clean_{season}.parquet')
            poss_players = set()
            for col in ['off_lineup', 'def_lineup']:
                for lineup in poss_df[col]:
                    if isinstance(lineup, (list, np.ndarray)):
                        for pid in lineup:
                            poss_players.add(str(pid).replace('.0', ''))
            poss_players.discard('0')
            
            missing = poss_players - new_players
            coverage = (len(poss_players) - len(missing)) / len(poss_players) * 100
            
            print(f"\n{season}:")
            print(f"   New stats: {len(new_players)} players")
            print(f"   Possession data: {len(poss_players)} players")
            print(f"   Missing: {len(missing)} players")
            print(f"   Coverage: {coverage:.1f}%")
            
            if len(missing) <= 10 and len(missing) > 0:
                print(f"   Missing IDs: {sorted(missing)}")
                
        except Exception as e:
            print(f"{season}: Could not compare - {e}")


def main():
    print("="*60)
    print("COMPLETE PLAYER STATS FETCHER")
    print("="*60)
    print("Using leaguedashplayerstats to get ALL players per season")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    base_dfs, adv_dfs = fetch_all_seasons()
    
    merged = merge_and_save(base_dfs, adv_dfs)
    
    if merged is not None:
        compare_coverage(merged)
        
        # Show sample columns
        print("\n" + "="*60)
        print("AVAILABLE COLUMNS")
        print("="*60)
        print(f"Total columns: {len(merged.columns)}")
        print(f"Key columns: {[c for c in merged.columns if any(x in c for x in ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'MIN', 'USG', 'TS_', 'BPM'])]}")


if __name__ == "__main__":
    main()
