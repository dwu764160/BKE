"""
src/features/derive_possessions.py
Groups atomic PBP events into logical Possessions.
Includes 'Smart Lineup Selection' and safeguards against IndexErrors.
"""

import pandas as pd
import numpy as np
import os
import sys
import glob
import re

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

DATA_DIR = "data/historical"

def process_game(game_df):
    # Ensure sorted by index (chronological)
    game_df = game_df.sort_index()
    
    if game_df.empty:
        return pd.DataFrame()

    # Identify the two teams
    valid_teams = game_df['team_id'].dropna().unique()
    valid_teams = [int(t) for t in valid_teams if t != 0]
    
    possessions = []
    
    # Initialize state
    current_poss = {
        'game_id': game_df['game_id'].iloc[0],
        'period': None,
        'start_clock': None,
        'off_team_id': None,
        'points': 0,
        'events': 0,
        'start_index': game_df.index[0] # FIX: Use actual first index, not 0
    }
    
    def find_valid_lineup(poss_rows, team_id):
        """
        Scans rows to find a valid 5-player lineup.
        """
        if poss_rows.empty:
            return []

        col_name = f'lineup_{int(team_id)}'
        if col_name not in poss_rows.columns:
            return []
            
        # 1. Prefer gameplay rows
        gameplay_rows = poss_rows[~poss_rows['event_type'].isin(['SUBSTITUTION'])]
        for _, row in gameplay_rows.iterrows():
            lu = row.get(col_name)
            if isinstance(lu, (list, np.ndarray)) and len(lu) == 5:
                return lu
                
        # 2. Fallback to any row
        for _, row in poss_rows.iterrows():
            lu = row.get(col_name)
            if isinstance(lu, (list, np.ndarray)) and len(lu) == 5:
                return lu
                
        # 3. Last Resort
        first = poss_rows.iloc[0].get(col_name)
        return first if isinstance(first, (list, np.ndarray)) else []

    def finalize_possession(idx_end, end_reason, next_offense_team):
        if current_poss['off_team_id'] is not None and current_poss['events'] > 0:
            # Safe slice
            try:
                poss_slice = game_df.loc[current_poss['start_index']:idx_end]
            except Exception:
                poss_slice = pd.DataFrame()

            if not poss_slice.empty:
                t_off = int(current_poss['off_team_id'])
                t_def = next((t for t in valid_teams if t != t_off), None)
                
                lineup_off = find_valid_lineup(poss_slice, t_off)
                lineup_def = find_valid_lineup(poss_slice, t_def) if t_def else []

                possessions.append({
                    'game_id': current_poss['game_id'],
                    'period': current_poss['period'],
                    'off_team_id': t_off,
                    'def_team_id': t_def,
                    'off_lineup': lineup_off,
                    'def_lineup': lineup_def,
                    'points': current_poss['points'],
                    'start_clock': current_poss['start_clock'],
                    'end_clock': game_df.loc[idx_end, 'clock'],
                    'num_events': current_poss['events'],
                    'end_reason': end_reason
                })

        # Reset
        current_poss['points'] = 0
        current_poss['events'] = 0
        current_poss['start_index'] = idx_end + 1
        current_poss['off_team_id'] = next_offense_team

    # Event Loop
    for idx, row in game_df.iterrows():
        etype = row['event_type']
        team_id = row['team_id']
        period = row['period']
        clock = row['clock']
        
        # Period Change
        if current_poss['period'] != period:
            finalize_possession(idx, "PERIOD_END", None)
            current_poss['period'] = period
            current_poss['start_clock'] = clock
            current_poss['off_team_id'] = None
            current_poss['start_index'] = idx # Reset start index to current row
        
        # Determine Offense
        if current_poss['off_team_id'] is None and pd.notna(team_id) and team_id != 0:
            if etype in ['BLOCK', 'STEAL']:
                pass 
            elif etype == 'REBOUND':
                current_poss['off_team_id'] = team_id
            else:
                current_poss['off_team_id'] = team_id

        current_poss['events'] += 1
        current_poss['points'] += row.get('points', 0)
        
        # End Conditions
        if etype in ['FIELD_GOAL', 'FIELD_GOAL_2PT', 'FIELD_GOAL_3PT'] and row['is_made']:
            finalize_possession(idx, "MAKE", None)
            
        elif etype == 'TURNOVER':
            finalize_possession(idx, "TURNOVER", None)
            
        elif etype == 'REBOUND':
            if current_poss['off_team_id'] and team_id and team_id != 0 and team_id != current_poss['off_team_id']:
                finalize_possession(idx, "DEF_REBOUND", team_id) 
                current_poss['start_clock'] = clock
                current_poss['start_index'] = idx 
            
        elif etype == 'FREE_THROW' and row['is_made']:
            desc = str(row.get('event_text', '')).upper()
            if '1 OF 1' in desc or '2 OF 2' in desc or '3 OF 3' in desc:
                finalize_possession(idx, "FT_MAKE", None)

    return pd.DataFrame(possessions)

def process_file(input_path):
    filename = os.path.basename(input_path)
    match = re.search(r"pbp_with_lineups_(\d{4}-\d{2})\.parquet", filename)
    
    if match:
        season = match.group(1)
        output_path = os.path.join(DATA_DIR, f"possessions_{season}.parquet")
    else:
        output_path = input_path.replace("pbp_with_lineups", "possessions")

    print(f"\n--- Deriving Possessions for {filename} ---")
    try:
        df = pd.read_parquet(input_path)
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return

    # Standardize IDs locally just in case
    if 'team_id' in df.columns:
        df['team_id'] = pd.to_numeric(df['team_id'], errors='coerce').fillna(0).astype(int)
    
    results = []
    grouped = df.groupby('game_id')
    total = len(grouped)
    
    for i, (gid, game_df) in enumerate(grouped):
        if i % 100 == 0:
            print(f"  Game {i}/{total}...", end='\r')
        results.append(process_game(game_df))
        
    print(f"\nConcatenating {season}...")
    if results:
        final_df = pd.concat(results)
        print(f"Saving to {output_path}...")
        final_df.to_parquet(output_path, index=False)
        print("✅ Done.")

def main():
    pattern = os.path.join(DATA_DIR, "pbp_with_lineups_*.parquet")
    files = sorted(glob.glob(pattern))
    for f in files:
        process_file(f)

if __name__ == "__main__":
    main()