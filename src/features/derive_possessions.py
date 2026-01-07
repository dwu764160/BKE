"""
src/features/derive_possessions.py
Groups atomic PBP events into logical Possessions.
UPDATED: Fixed initialization bug (Error -1).
"""

import pandas as pd
import numpy as np
import os
import sys
import glob
import re

# Adjust path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

DATA_DIR = "data/historical"

def process_game(game_df):
    game_df = game_df.sort_index().reset_index(drop=True)
    if game_df.empty: return pd.DataFrame()

    # Identify the two teams
    valid_teams = [int(t) for t in game_df['team_id'].dropna().unique() if t != 0]
    if len(valid_teams) != 2:
        return pd.DataFrame()
    
    team_a, team_b = valid_teams[0], valid_teams[1]
    
    def get_opponent(tid):
        return team_b if tid == team_a else team_a

    possessions = []
    num_rows = len(game_df)
    
    # Initialize with first row data to prevent "Period Change" on step 0
    first_row = game_df.iloc[0]
    
    current = {
        'game_id': first_row['game_id'],
        'period': first_row['period'],      # FIXED: Init with actual period
        'start_clock': first_row['clock'],  # FIXED: Init with actual clock
        'off_team_id': None,
        'points': 0,
        'events': 0,
        'start_idx': 0,
        'has_play': False
    }
    
    def find_lineup(rows, team_id):
        if rows.empty or team_id is None: return []
        col = f'lineup_{int(team_id)}'
        if col not in rows.columns: return []
        
        # Priority: Gameplay rows -> Any row
        for val in rows[~rows['event_type'].isin(['SUBSTITUTION', 'TIMEOUT'])][col]:
            if isinstance(val, (list, np.ndarray)) and len(val) == 5: return val
        for val in rows[col]:
            if isinstance(val, (list, np.ndarray)) and len(val) == 5: return val
        return []

    def finalize(end_idx, reason, force_next_off):
        if current['off_team_id'] and current['events'] > 0:
            slice_df = game_df.iloc[current['start_idx'] : end_idx + 1]
            t_off = int(current['off_team_id'])
            t_def = get_opponent(t_off)
            
            # Filter Zombie Possessions
            is_valid = True
            if reason == "PERIOD_END" and not current['has_play'] and current['points'] == 0:
                is_valid = False

            if is_valid:
                s_clock = current['start_clock']
                # Safety fallback
                if s_clock is None:
                    s_clock = game_df.at[current['start_idx'], 'clock']

                possessions.append({
                    'game_id': current['game_id'],
                    'period': current['period'],
                    'off_team_id': t_off,
                    'def_team_id': t_def,
                    'off_lineup': find_lineup(slice_df, t_off),
                    'def_lineup': find_lineup(slice_df, t_def),
                    'points': current['points'],
                    'start_clock': s_clock,
                    'end_clock': game_df.at[end_idx, 'clock'],
                    'num_events': current['events'],
                    'end_reason': reason
                })

        # Reset State
        current['points'] = 0
        current['events'] = 0
        current['has_play'] = False
        current['start_idx'] = end_idx + 1
        
        # Strict Handoff
        current['off_team_id'] = force_next_off
        
        # Set start clock for next possession
        if end_idx + 1 < num_rows:
            current['start_clock'] = game_df.at[end_idx, 'clock']

    # --- Event Loop ---
    for i in range(num_rows):
        row = game_df.iloc[i]
        
        etype = row['event_type']
        team_id = row['team_id']
        period = row['period']
        clock = row['clock']
        desc = str(row.get('event_text', '')).upper()
        
        # 1. Period Change
        if current['period'] != period:
            finalize(i - 1, "PERIOD_END", None)
            current['period'] = period
            current['start_clock'] = clock
            current['off_team_id'] = None
            current['start_idx'] = i
            
        # 2. Determine Offense
        if current['off_team_id'] is None and pd.notna(team_id) and team_id != 0:
            if etype not in ['BLOCK', 'STEAL', 'SUBSTITUTION', 'TIMEOUT', 'INSTANT_REPLAY', 'UNKNOWN']:
                current['off_team_id'] = team_id

        # 3. Accumulate
        current['events'] += 1
        current['points'] += row.get('points', 0)
        
        if etype in ['FIELD_GOAL', 'FIELD_GOAL_2PT', 'FIELD_GOAL_3PT', 'FREE_THROW', 'TURNOVER']:
            current['has_play'] = True
            
        # 4. End Logic (Strict Flip)
        
        # A. MADE SHOT
        if 'FIELD_GOAL' in etype and row['is_made']:
            is_and_one = False
            scan_limit = min(i + 6, num_rows)
            for k in range(i + 1, scan_limit):
                next_row = game_df.iloc[k]
                n_type = next_row['event_type']
                if n_type in ['SUBSTITUTION', 'TIMEOUT', 'INSTANT_REPLAY', 'UNKNOWN']: continue
                if n_type == 'FREE_THROW' and next_row['team_id'] == team_id:
                    is_and_one = True
                break
            
            if not is_and_one:
                next_team = get_opponent(current['off_team_id']) if current['off_team_id'] else None
                finalize(i, "MAKE", next_team)

        # B. TURNOVER
        elif etype == 'TURNOVER':
            next_team = get_opponent(current['off_team_id']) if current['off_team_id'] else None
            finalize(i, "TURNOVER", next_team)

        # C. DEFENSIVE REBOUND
        elif etype == 'REBOUND':
            if current['off_team_id'] and team_id and team_id != 0 and team_id != current['off_team_id']:
                finalize(i, "DEF_REBOUND", team_id)
                current['start_clock'] = clock
                current['start_idx'] = i

        # D. FREE THROWS
        elif etype == 'FREE_THROW' and row['is_made']:
            if "TECHNICAL" in desc: continue
            if '1 OF 1' in desc or '2 OF 2' in desc or '3 OF 3' in desc:
                next_team = get_opponent(current['off_team_id']) if current['off_team_id'] else None
                finalize(i, "FT_MAKE", next_team)
                
    finalize(num_rows - 1, "GAME_END", None)
    return pd.DataFrame(possessions)

def process_file(input_path):
    filename = os.path.basename(input_path)
    match = re.search(r"pbp_with_lineups_(\d{4}-\d{2})\.parquet", filename)
    if not match: return
    season = match.group(1)
    output_path = os.path.join(DATA_DIR, f"possessions_{season}.parquet")

    print(f"Deriving Possessions {season}...", end=" ")
    try:
        df = pd.read_parquet(input_path)
        if 'team_id' in df.columns:
            df['team_id'] = pd.to_numeric(df['team_id'], errors='coerce').fillna(0).astype(int)
            
        results = []
        for gid, game_df in df.groupby('game_id'):
            results.append(process_game(game_df))
            
        if results:
            final_df = pd.concat(results)
            final_df.to_parquet(output_path, index=False)
            print(f"✅ ({len(final_df):,} rows)")
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    pattern = os.path.join(DATA_DIR, "pbp_with_lineups_*.parquet")
    files = sorted(glob.glob(pattern))
    print("=== Re-Running Possession Derivation (Error Fixed) ===")
    for f in files:
        process_file(f)

if __name__ == "__main__":
    main()