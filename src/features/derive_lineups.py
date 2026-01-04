"""
src/features/derive_lineups.py
Infers the 5 players on the court for every event.
Scans for 'pbp_normalized_*.parquet' and outputs 'pbp_with_lineups_*.parquet'.
"""

import pandas as pd
import numpy as np
import os
import sys
import glob
import re

# Adjust path to find src if run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

DATA_DIR = "data/historical"

def get_initial_lineup(period_events, team_id):
    """Deduces the 5 players starting the period for a specific team."""
    if not team_id or pd.isna(team_id):
        return []

    team_events = period_events[period_events['team_id'] == team_id]
    
    starters = set()
    subs_in = set()
    
    # Iterate chronologically
    for _, row in team_events.iterrows():
        p1 = row.get('player1_id')
        etype = row.get('event_type')
        
        if not p1:
            continue
            
        desc = str(row.get('event_text', '')).upper()
        
        if etype == 'SUBSTITUTION':
            if 'SUB IN' in desc:
                subs_in.add(p1)
            elif 'SUB OUT' in desc:
                if p1 not in subs_in:
                    starters.add(p1)
        else:
            if p1 not in subs_in:
                starters.add(p1)
                
    return list(starters)

def process_game_period(df_gp):
    """Calculates lineups for a single period of a single game."""
    df_gp = df_gp.sort_index()
    
    teams = df_gp['team_id'].dropna().unique()
    if len(teams) != 2:
        return df_gp

    team_a, team_b = teams[0], teams[1]
    
    # 1. Solve Starters
    starters_a = get_initial_lineup(df_gp, team_a)
    starters_b = get_initial_lineup(df_gp, team_b)
    
    current_a = set(starters_a)
    current_b = set(starters_b)
    
    lineups_a = []
    lineups_b = []
    
    # 2. Iterate
    for _, row in df_gp.iterrows():
        row_team = row.get('team_id')
        p1 = row.get('player1_id')
        etype = row.get('event_type')
        desc = str(row.get('event_text', '')).upper()
        
        if etype == 'SUBSTITUTION':
            if row_team == team_a:
                if 'SUB IN' in desc and p1: current_a.add(p1)
                if 'SUB OUT' in desc and p1 and p1 in current_a: current_a.remove(p1)
            elif row_team == team_b:
                if 'SUB IN' in desc and p1: current_b.add(p1)
                if 'SUB OUT' in desc and p1 and p1 in current_b: current_b.remove(p1)
        
        # Store as string for easy serialization/hashing
        lineups_a.append(sorted([str(int(x)) if pd.notna(x) else str(x) for x in current_a]))
        lineups_b.append(sorted([str(int(x)) if pd.notna(x) else str(x) for x in current_b]))

    # Dynamic Column Names
    df_gp[f'lineup_{int(team_a)}'] = lineups_a
    df_gp[f'lineup_{int(team_b)}'] = lineups_b
    
    return df_gp

def process_file(input_path):
    filename = os.path.basename(input_path)
    match = re.search(r"pbp_normalized_(\d{4}-\d{2})\.parquet", filename)
    
    if match:
        season = match.group(1)
        output_path = os.path.join(DATA_DIR, f"pbp_with_lineups_{season}.parquet")
    else:
        output_path = input_path.replace("normalized", "with_lineups")

    print(f"\n--- Processing Lineups for {filename} ---")
    
    df = pd.read_parquet(input_path)
    
    processed_dfs = []
    grouped = df.groupby('game_id')
    total_games = len(grouped)
    
    print(f"Processing {len(df)} rows across {total_games} games...")

    for i, (gid, game_df) in enumerate(grouped):
        if i % 50 == 0:
            print(f"  Game {i}/{total_games}...", end='\r')
            
        period_chunks = []
        for p, period_df in game_df.groupby('period'):
            period_chunks.append(process_game_period(period_df))
            
        if period_chunks:
            processed_dfs.append(pd.concat(period_chunks))
            
    print(f"\nConcatenating {season}...")
    if processed_dfs:
        final_df = pd.concat(processed_dfs)
        print(f"Saving to {output_path}...")
        final_df.to_parquet(output_path, index=False)
        
        # --- VALIDATION (FIXED) ---
        print("Validating...")
        # Get columns that look like 'lineup_12345'
        lineup_cols = [c for c in final_df.columns if c.startswith('lineup_')]
        
        if lineup_cols:
            # Drop rows where ANY lineup col is null/empty
            # Actually, check specific team IDs
            sample = final_df.dropna(subset=['team_id']).iloc[0]
            tid = sample['team_id']
            
            # FIX: Check NOT NA before casting
            if pd.notna(tid):
                col = f'lineup_{int(tid)}'
                if col in sample:
                    lu = sample[col]
                    print(f"  Sample Team {int(tid)} Lineup: {lu}")
                    print(f"  Count: {len(lu)} (Target: 5)")
                else:
                    print(f"  ⚠️ Column {col} not found in sample row.")
            else:
                print("  Sample row has NaN team_id, skipping specific check.")
        else:
            print("  ⚠️ No lineup columns found.")
            
        print("✅ Done.")
    else:
        print("No data processed.")

def main():
    pattern = os.path.join(DATA_DIR, "pbp_normalized_*.parquet")
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"No normalized files found in {DATA_DIR}")
        return

    for f in files:
        process_file(f)

if __name__ == "__main__":
    main()