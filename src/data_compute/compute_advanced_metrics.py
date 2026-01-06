"""
src/data_compute/compute_advanced_metrics.py
Aggregates 'clean' possessions into Team and Lineup level advanced stats.
Enriches output with readable Team Names and Player Names.
Fixed: Handles float/string ID mismatches (e.g., 1234.0 vs "1234").
"""

import pandas as pd
import numpy as np
import os
import sys
import glob

# Adjust path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

DATA_DIR = "data/historical"
OUTPUT_DIR = "data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_reference_data():
    """Loads Team and Player names for enrichment."""
    teams_map = {}
    players_map = {}
    
    # Load Teams
    t_path = os.path.join(DATA_DIR, "teams.parquet")
    if os.path.exists(t_path):
        try:
            df_t = pd.read_parquet(t_path)
            df_t['id'] = df_t['id'].astype(str).str.replace(r'\.0$', '', regex=True)
            teams_map = df_t.set_index('id')['full_name'].to_dict()
        except Exception as e:
            print(f"⚠️ Could not load teams.parquet: {e}")

    # Load Players
    p_path = os.path.join(DATA_DIR, "players.parquet")
    if os.path.exists(p_path):
        try:
            df_p = pd.read_parquet(p_path)
            df_p['id'] = df_p['id'].astype(str).str.replace(r'\.0$', '', regex=True)
            players_map = df_p.set_index('id')['full_name'].to_dict()
        except Exception as e:
            print(f"⚠️ Could not load players.parquet: {e}")
        
    return teams_map, players_map

def clean_id(val):
    """Converts float/int/str to clean string ID (no decimals)."""
    if pd.isna(val): return "0"
    return str(val).replace(".0", "")

def resolve_lineup_names(id_list, p_map):
    """Converts list of IDs to list of Names."""
    if not isinstance(id_list, (list, np.ndarray)):
        return []
    # Clean each ID in the list before looking up
    return [p_map.get(clean_id(pid), str(pid)) for pid in id_list]

def load_all_clean_data():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "possessions_clean_*.parquet")))
    if not files:
        print("❌ No clean possession files found.")
        return pd.DataFrame()
    
    print(f"Loading {len(files)} files...")
    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        if 'season' not in df.columns:
            base = os.path.basename(f)
            season = base.replace("possessions_clean_", "").replace(".parquet", "")
            df['season'] = season
        dfs.append(df)
        
    return pd.concat(dfs, ignore_index=True)

def process_teams(df, team_map):
    print("Computing Team Stats...")
    # Group by Season + Offense Team
    off_stats = df.groupby(['season', 'off_team_id']).agg(
        poss_off=('game_id', 'count'),
        pts_scored=('points', 'sum')
    ).reset_index()
    
    # Group by Season + Defense Team
    def_stats = df.groupby(['season', 'def_team_id']).agg(
        poss_def=('game_id', 'count'),
        pts_allowed=('points', 'sum')
    ).reset_index()
    
    # Merge
    merged = pd.merge(
        off_stats, 
        def_stats, 
        left_on=['season', 'off_team_id'], 
        right_on=['season', 'def_team_id']
    )
    
    merged.rename(columns={'off_team_id': 'team_id'}, inplace=True)
    merged.drop(columns=['def_team_id'], inplace=True)
    
    # Calc Metrics
    merged['ORTG'] = (merged['pts_scored'] / merged['poss_off']) * 100
    merged['DRTG'] = (merged['pts_allowed'] / merged['poss_def']) * 100
    merged['NET_RTG'] = merged['ORTG'] - merged['DRTG']
    
    # Map Names (Clean ID first)
    merged['clean_team_id'] = merged['team_id'].apply(clean_id)
    merged['team_name'] = merged['clean_team_id'].map(team_map).fillna("Unknown")
    
    # Reorder
    cols = ['season', 'team_id', 'team_name', 'ORTG', 'DRTG', 'NET_RTG', 'poss_off']
    return merged[cols].sort_values(['season', 'NET_RTG'], ascending=[True, False])

def process_lineups(df, team_map, player_map):
    print("Computing Lineup Stats...")
    # Convert lineup arrays to tuples for grouping
    df['off_lineup_tuple'] = df['off_lineup'].apply(lambda x: tuple(sorted(x)) if x is not None else None)
    df['def_lineup_tuple'] = df['def_lineup'].apply(lambda x: tuple(sorted(x)) if x is not None else None)
    
    # 1. Offense
    off_stats = df.groupby(['season', 'off_team_id', 'off_lineup_tuple']).agg(
        poss_off=('game_id', 'count'),
        pts_scored=('points', 'sum')
    ).reset_index()
    
    # 2. Defense
    def_stats = df.groupby(['season', 'def_team_id', 'def_lineup_tuple']).agg(
        poss_def=('game_id', 'count'),
        pts_allowed=('points', 'sum')
    ).reset_index()
    
    # 3. Merge
    merged = pd.merge(
        off_stats,
        def_stats,
        left_on=['season', 'off_team_id', 'off_lineup_tuple'],
        right_on=['season', 'def_team_id', 'def_lineup_tuple'],
        how='outer'
    ).fillna(0)
    
    # Cleanup
    merged['team_id'] = merged['off_team_id'].replace(0, np.nan).fillna(merged['def_team_id'])
    merged['lineup_ids'] = merged['off_lineup_tuple'].replace(0, np.nan).fillna(merged['def_lineup_tuple'])
    
    # Convert tuple back to list
    merged['lineup_ids'] = merged['lineup_ids'].apply(list)
    
    # 4. Metrics
    merged['total_poss'] = merged['poss_off'] + merged['poss_def']
    
    # Filter out garbage time
    merged = merged[merged['total_poss'] >= 10].copy()
    
    merged['ORTG'] = np.where(merged['poss_off'] > 0, (merged['pts_scored'] / merged['poss_off']) * 100, 0)
    merged['DRTG'] = np.where(merged['poss_def'] > 0, (merged['pts_allowed'] / merged['poss_def']) * 100, 0)
    merged['NET_RTG'] = merged['ORTG'] - merged['DRTG']
    
    # 5. Add Names
    merged['clean_team_id'] = merged['team_id'].apply(clean_id)
    merged['team_name'] = merged['clean_team_id'].map(team_map).fillna("Unknown")
    
    # Resolve Player Names
    print("  Mapping player names...")
    merged['lineup_names'] = merged['lineup_ids'].apply(lambda x: resolve_lineup_names(x, player_map))
    
    # Columns to keep
    cols = ['season', 'team_name', 'NET_RTG', 'ORTG', 'DRTG', 'total_poss', 'lineup_names', 'lineup_ids']
    return merged[cols].sort_values(['season', 'total_poss'], ascending=[True, False])

def main():
    team_map, player_map = load_reference_data()
    
    df = load_all_clean_data()
    if df.empty: return
    
    # Teams
    team_df = process_teams(df, team_map)
    team_out = os.path.join(OUTPUT_DIR, "metrics_teams.parquet")
    team_df.to_parquet(team_out, index=False)
    print(f"✅ Saved Team Metrics to {team_out}")
    print(team_df[['season', 'team_name', 'ORTG', 'NET_RTG']].head(5))

    # Lineups
    lineup_df = process_lineups(df, team_map, player_map)
    lineup_out = os.path.join(OUTPUT_DIR, "metrics_lineups.parquet")
    lineup_df.to_parquet(lineup_out, index=False)
    print(f"✅ Saved Lineup Metrics to {lineup_out}")
    
    # Show Top Lineup
    print("\n--- Top Lineup (>500 Possessions) ---")
    high_vol = lineup_df[lineup_df['total_poss'] > 500].sort_values('NET_RTG', ascending=False)
    if not high_vol.empty:
        top = high_vol.iloc[0]
        print(f"Season: {top['season']}")
        print(f"Team:   {top['team_name']}")
        print(f"NetRTG: {top['NET_RTG']:.1f}")
        print(f"Lineup: {top['lineup_names']}")

if __name__ == "__main__":
    main()