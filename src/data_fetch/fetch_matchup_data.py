"""
src/data_fetch/fetch_matchup_data.py
Fetches player matchup data from NBA Stats API for defensive archetype classification.

Endpoints:
1. LeagueSeasonMatchups - Individual player-vs-player matchup data
2. MatchupsRollup - Aggregated defender matchup statistics
3. LeagueDashPtDefend - Player defense tracking by shot distance

This data enables:
- Matchup Versatility: How many different positions/players a defender guards
- Matchup Difficulty: Quality of offensive players defended (by opponent PPG/usage)
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
CACHE_DIR = Path("data/matchup_cache")
OUTPUT_DIR = Path("data/matchup")
SEASONS = ["2022-23", "2023-24", "2024-25"]

# Defense categories for LeagueDashPtDefend (already fetched in tracking_data.py)
DEFENSE_CATEGORIES = {
    "Overall": "Overall",
    "3Pointers": "3 Pointers", 
    "LessThan6Ft": "Less Than 6Ft",
    "LessThan10Ft": "Less Than 10Ft",
    "GreaterThan15Ft": "Greater Than 15Ft"
}

def ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def smart_sleep():
    time.sleep(random.uniform(1.5, 3.0))

def fetch_url_cached(url, params, cache_name, referer="https://www.nba.com/stats/players/isolation"):
    """Fetch URL with caching and TLS impersonation."""
    cache_path = CACHE_DIR / f"{cache_name}.json"
    
    if cache_path.exists():
        try:
            with open(cache_path, "r") as f:
                json_data = json.load(f)
                if 'resultSets' in json_data:
                    return parse_json(json_data)
        except:
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
            print(f"⚠️ HTTP {resp.status_code}")
            return None
        
        json_data = resp.json()
        
        with open(cache_path, "w") as f:
            json.dump(json_data, f)
            
        return parse_json(json_data)
            
    except Exception as e:
        print(f"⚠️ Error: {e}")
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
    except:
        return pd.DataFrame()


def fetch_league_season_matchups(season: str) -> pd.DataFrame:
    """
    Fetch LeagueSeasonMatchups - individual player vs player matchup data.
    
    Returns columns:
    - OFF_PLAYER_ID, OFF_PLAYER_NAME
    - DEF_PLAYER_ID, DEF_PLAYER_NAME
    - GP, MATCHUP_MIN, PARTIAL_POSS
    - PLAYER_PTS, TEAM_PTS
    - MATCHUP_FGM, MATCHUP_FGA, MATCHUP_FG_PCT
    - MATCHUP_FG3M, MATCHUP_FG3A, MATCHUP_FG3_PCT
    - HELP_BLK, HELP_FGM, HELP_FGA
    """
    print(f"   Fetching LeagueSeasonMatchups for {season}...", end=" ")
    
    url = "https://stats.nba.com/stats/leagueseasonmatchups"
    params = {
        "LeagueID": "00",
        "PerMode": "Totals",
        "Season": season,
        "SeasonType": "Regular Season",
        "DefPlayerID": "",
        "DefTeamID": "",
        "OffPlayerID": "",
        "OffTeamID": ""
    }
    
    cache_name = f"leagueseasonmatchups_{season}"
    df = fetch_url_cached(url, params, cache_name)
    
    if df is not None and not df.empty:
        print(f"✅ {len(df)} matchups")
        df['SEASON'] = season
        return df
    else:
        print("❌ Failed")
        return pd.DataFrame()


def fetch_matchups_rollup_all_defenders(season: str) -> pd.DataFrame:
    """
    Fetch MatchupsRollup - aggregated matchup stats per defender.
    This gives us who each defender guards by position.
    
    Returns columns:
    - DEF_PLAYER_ID, DEF_PLAYER_NAME, POSITION
    - PERCENT_OF_TIME (% of defensive possessions at this position)
    - GP, MATCHUP_MIN, PARTIAL_POSS
    - MATCHUP_FGM, MATCHUP_FGA, MATCHUP_FG_PCT
    """
    print(f"   Fetching MatchupsRollup for {season}...", end=" ")
    
    url = "https://stats.nba.com/stats/matchupsrollup"
    params = {
        "LeagueID": "00",
        "PerMode": "Totals",
        "Season": season,
        "SeasonType": "Regular Season",
        "DefPlayerID": "",
        "DefTeamID": "",
        "OffPlayerID": "",
        "OffTeamID": ""
    }
    
    cache_name = f"matchupsrollup_{season}"
    df = fetch_url_cached(url, params, cache_name)
    
    if df is not None and not df.empty:
        print(f"✅ {len(df)} defender-position entries")
        df['SEASON'] = season
        return df
    else:
        print("❌ Failed")
        return pd.DataFrame()


def fetch_defense_tracking(season: str, category: str, api_category: str) -> pd.DataFrame:
    """
    Fetch LeagueDashPtDefend for additional defense distance categories.
    
    Categories: Overall, 3 Pointers, Less Than 6Ft, Less Than 10Ft, Greater Than 15Ft
    """
    print(f"   Fetching Defense {category} for {season}...", end=" ")
    
    url = "https://stats.nba.com/stats/leaguedashptdefend"
    params = {
        "LeagueID": "00",
        "PerMode": "Totals",
        "Season": season,
        "SeasonType": "Regular Season",
        "DefenseCategory": api_category,
        "College": "",
        "Conference": "",
        "Country": "",
        "DateFrom": "",
        "DateTo": "",
        "Division": "",
        "DraftPick": "",
        "DraftYear": "",
        "GameSegment": "",
        "Height": "",
        "LastNGames": "",
        "Location": "",
        "Month": "",
        "OpponentTeamID": "",
        "Outcome": "",
        "PORound": "",
        "Period": "",
        "PlayerExperience": "",
        "PlayerID": "",
        "PlayerPosition": "",
        "SeasonSegment": "",
        "StarterBench": "",
        "TeamID": "",
        "VsConference": "",
        "VsDivision": "",
        "Weight": ""
    }
    
    cache_name = f"defense_{category}_{season}"
    df = fetch_url_cached(url, params, cache_name)
    
    if df is not None and not df.empty:
        print(f"✅ {len(df)} players")
        df['SEASON'] = season
        return df
    else:
        print("❌ Failed")
        return pd.DataFrame()


def compute_matchup_versatility(rollup_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute matchup versatility score from MatchupsRollup data.
    
    Versatility metrics:
    - positions_guarded: Count of distinct positions (G, F, C)
    - position_entropy: Shannon entropy of position distribution (higher = more versatile)
    - primary_position: Position defended most often
    - switch_score: How evenly spread across positions (max at 33% each)
    """
    import numpy as np
    
    if rollup_df.empty:
        return pd.DataFrame()
    
    # Group by defender
    versatility = []
    
    for (def_id, season), group in rollup_df.groupby(['DEF_PLAYER_ID', 'SEASON']):
        positions = group['POSITION'].unique()
        pct_by_pos = group.groupby('POSITION')['PERCENT_OF_TIME'].sum()
        
        # Normalize percentages
        total_pct = pct_by_pos.sum()
        if total_pct > 0:
            pct_by_pos = pct_by_pos / total_pct
        
        # Shannon entropy for versatility
        entropy = -sum(p * np.log2(p + 1e-10) for p in pct_by_pos if p > 0)
        max_entropy = np.log2(len(pct_by_pos)) if len(pct_by_pos) > 1 else 1
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # Switch score (1 = perfectly even, 0 = single position)
        switch_score = normalized_entropy
        
        # Primary position
        primary_pos = pct_by_pos.idxmax() if not pct_by_pos.empty else 'Unknown'
        
        # Total matchup minutes and possessions
        total_min = group['MATCHUP_MIN'].sum()
        total_poss = group['PARTIAL_POSS'].sum()
        
        versatility.append({
            'DEF_PLAYER_ID': def_id,
            'DEF_PLAYER_NAME': group['DEF_PLAYER_NAME'].iloc[0],
            'SEASON': season,
            'positions_guarded': len(positions),
            'position_entropy': entropy,
            'switch_score': switch_score,
            'primary_position': primary_pos,
            'total_matchup_min': total_min,
            'total_matchup_poss': total_poss,
            'pct_guards': pct_by_pos.get('G', 0),
            'pct_forwards': pct_by_pos.get('F', 0),
            'pct_centers': pct_by_pos.get('C', 0),
        })
    
    return pd.DataFrame(versatility)


def compute_matchup_difficulty(matchups_df: pd.DataFrame, 
                                player_stats_path: str = "data/historical/complete_player_season_stats.parquet") -> pd.DataFrame:
    """
    Compute matchup difficulty score based on quality of offensive players defended.
    
    Difficulty metrics:
    - avg_opponent_ppg: Average PPG of offensive players defended
    - avg_opponent_usage: Average usage rate of offensive players defended
    - weighted_opponent_quality: PPG * minutes weighted average
    - elite_matchup_pct: % of matchup time vs 20+ PPG scorers
    """
    if matchups_df.empty:
        return pd.DataFrame()
    
    # Load player stats for opponent quality
    try:
        player_stats = pd.read_parquet(player_stats_path)
        # Create lookup: (PLAYER_ID, SEASON) -> PPG, USG_PCT
        player_stats['PPG'] = player_stats['PTS'] / player_stats['GP']
        stats_lookup = player_stats.set_index(['PLAYER_ID', 'SEASON'])[['PPG', 'USG_PCT', 'MIN']].to_dict('index')
    except:
        print("⚠️ Could not load player stats for difficulty calculation")
        return pd.DataFrame()
    
    difficulty = []
    
    for (def_id, season), group in matchups_df.groupby(['DEF_PLAYER_ID', 'SEASON']):
        # Get opponent stats for each offensive player
        opp_ppg = []
        opp_usage = []
        weights = []  # matchup minutes as weights
        elite_time = 0
        total_time = 0
        
        for _, row in group.iterrows():
            off_id = row['OFF_PLAYER_ID']
            try:
                # MATCHUP_MIN is in "MM:SS" format, convert to total minutes
                matchup_min_str = str(row['MATCHUP_MIN']) if row['MATCHUP_MIN'] else "0:00"
                if ':' in matchup_min_str:
                    parts = matchup_min_str.split(':')
                    matchup_min = float(parts[0]) + float(parts[1]) / 60
                else:
                    matchup_min = float(matchup_min_str)
            except (ValueError, TypeError):
                matchup_min = 0
            
            key = (off_id, season)
            if key in stats_lookup:
                stats = stats_lookup[key]
                ppg = stats.get('PPG', 0)
                usg = stats.get('USG_PCT', 0.15)
                
                opp_ppg.append(ppg)
                opp_usage.append(usg)
                weights.append(matchup_min)
                
                total_time += matchup_min
                if ppg >= 20:
                    elite_time += matchup_min
        
        if not weights or sum(weights) == 0:
            continue
        
        # Weighted averages
        total_weight = sum(weights)
        avg_ppg = sum(p * w for p, w in zip(opp_ppg, weights)) / total_weight
        avg_usage = sum(u * w for u, w in zip(opp_usage, weights)) / total_weight
        elite_pct = elite_time / total_time if total_time > 0 else 0
        
        difficulty.append({
            'DEF_PLAYER_ID': def_id,
            'DEF_PLAYER_NAME': group['DEF_PLAYER_NAME'].iloc[0] if 'DEF_PLAYER_NAME' in group.columns else 'Unknown',
            'SEASON': season,
            'avg_opponent_ppg': avg_ppg,
            'avg_opponent_usage': avg_usage,
            'elite_matchup_pct': elite_pct,
            'total_matchups': len(group),
            'total_matchup_time': total_time,
        })
    
    return pd.DataFrame(difficulty)


def main():
    print("=" * 70)
    print("MATCHUP DATA FETCHER")
    print("=" * 70)
    
    ensure_dirs()
    
    all_matchups = []
    all_rollups = []
    all_versatility = []
    all_difficulty = []
    
    for season in SEASONS:
        print(f"\n📊 Processing {season}...")
        
        # Fetch LeagueSeasonMatchups (individual matchups)
        matchups = fetch_league_season_matchups(season)
        if not matchups.empty:
            all_matchups.append(matchups)
        smart_sleep()
        
        # Fetch MatchupsRollup (aggregated by position)
        rollup = fetch_matchups_rollup_all_defenders(season)
        if not rollup.empty:
            all_rollups.append(rollup)
        smart_sleep()
        
        # Compute derived metrics
        if not rollup.empty:
            versatility = compute_matchup_versatility(rollup)
            all_versatility.append(versatility)
        
        if not matchups.empty:
            difficulty = compute_matchup_difficulty(matchups)
            all_difficulty.append(difficulty)
    
    # Combine all seasons
    print("\n💾 Saving data...")
    
    if all_matchups:
        matchups_df = pd.concat(all_matchups, ignore_index=True)
        matchups_df.to_parquet(OUTPUT_DIR / "league_season_matchups.parquet", index=False)
        print(f"   ✅ Saved {len(matchups_df)} matchups to league_season_matchups.parquet")
    
    if all_rollups:
        rollups_df = pd.concat(all_rollups, ignore_index=True)
        rollups_df.to_parquet(OUTPUT_DIR / "matchups_rollup.parquet", index=False)
        print(f"   ✅ Saved {len(rollups_df)} rollup entries to matchups_rollup.parquet")
    
    if all_versatility:
        versatility_df = pd.concat(all_versatility, ignore_index=True)
        versatility_df.to_parquet(OUTPUT_DIR / "matchup_versatility.parquet", index=False)
        print(f"   ✅ Saved {len(versatility_df)} versatility scores to matchup_versatility.parquet")
    
    if all_difficulty:
        difficulty_df = pd.concat(all_difficulty, ignore_index=True)
        difficulty_df.to_parquet(OUTPUT_DIR / "matchup_difficulty.parquet", index=False)
        print(f"   ✅ Saved {len(difficulty_df)} difficulty scores to matchup_difficulty.parquet")
    
    print("\n" + "=" * 70)
    print("MATCHUP DATA FETCH COMPLETE")
    print("=" * 70)
    
    # Print sample
    if all_versatility:
        print("\nSample Versatility Scores (2024-25):")
        v = versatility_df[versatility_df['SEASON'] == '2024-25'].nlargest(10, 'switch_score')
        for _, row in v.iterrows():
            print(f"  {row['DEF_PLAYER_NAME']:<25} Switch: {row['switch_score']:.2f} | Pos: {row['positions_guarded']} | Primary: {row['primary_position']}")


if __name__ == "__main__":
    main()
