"""
src/features/derive_possessions.py
Groups atomic PBP events into logical Possessions.
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
    game_df = game_df.sort_index()
    
    # Identify the two teams playing in this specific game
    valid_teams = game_df['team_id'].dropna().unique()
    valid_teams = [int(t) for t in valid_teams]
    
    possessions = []
    
    current_poss = {
        'game_id': game_df['game_id'].iloc[0],
        'period': None,
        'start_clock': None,
        'off_team_id': None,
        'points': 0,
        'events': 0,
        'start_index': 0
    }
    
    def finalize_possession(idx_end, end_reason, next_offense_team):
        if current_poss['off_team_id'] is not None and current_poss['events'] > 0:
            first_row = game_df.loc[current_poss['start_index']]
            
            t_off = int(current_poss['off_team_id'])
            
            # Defense is the team that isn't Offense
            t_def = next((t for t in valid_teams if t != t_off), None)
            
            l_off_col = f'lineup_{t_off}'
            l_def_col = f'lineup_{t_def}' if t_def else None
            
            # Retrieve lineups (handle missing columns gracefully)
            lineup_off = first_row[l_off_col] if l_off_col in first_row else []
            lineup_def = first_row[l_def_col] if l_def_col and l_def_col in first_row else []

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
        
        # Determine Offense
        if current_poss['off_team_id'] is None and pd.notna(team_id):
            if etype in ['BLOCK', 'STEAL']:
                pass # Defensive stats
            elif etype == 'REBOUND':
                current_poss['off_team_id'] = team_id
            else:
                current_poss['off_team_id'] = team_id

        # Accumulate
        current_poss['events'] += 1
        current_poss['points'] += row.get('points', 0)
        
        # End Conditions
        if etype in ['FIELD_GOAL', 'FIELD_GOAL_2PT', 'FIELD_GOAL_3PT'] and row['is_made']:
            finalize_possession(idx, "MAKE", None)
            
        elif etype == 'TURNOVER':
            finalize_possession(idx, "TURNOVER", None)
            
        elif etype == 'REBOUND':
            if current_poss['off_team_id'] and team_id and team_id != current_poss['off_team_id']:
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
    df = pd.read_parquet(input_path)
    
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