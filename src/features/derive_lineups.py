"""
src/features/derive_lineups.py
Infers the 5 players on the court for every event.
Includes TARGETED DEBUGGING for Game 0022200001.
"""

import pandas as pd
import numpy as np
import os
import sys
import glob
import re

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

DATA_DIR = "data/historical"

# DEBUG CONSTANTS
DEBUG_GAME_ID = "0022200001"
DEBUG_TEAM_ID = "1610612738"
DEBUG_PLAYER_ID = "201143" # Al Horford

def to_id(val):
    """Standardizes IDs to string integers (removes .0 decimals)."""
    if pd.isna(val) or val == "" or str(val).strip() == "":
        return None
    try:
        return str(int(float(val)))
    except:
        return str(val)

def build_player_team_map(game_df):
    pt_map = {}
    
    # 1. Substitutions
    subs = game_df[game_df['event_type'] == 'SUBSTITUTION']
    for _, row in subs.iterrows():
        tid = row['team_id']
        p1 = row['player1_id']
        if tid and p1 and p1 != '0':
            pt_map[p1] = tid
            
    # 2. Free Throws
    fts = game_df[game_df['event_type'] == 'FREE_THROW']
    for _, row in fts.iterrows():
        tid = row['team_id']
        p1 = row['player1_id']
        if tid and p1 and p1 != '0':
            pt_map[p1] = tid

    # 3. Assists
    assists = game_df[
        (game_df['event_type'].str.contains('FIELD_GOAL', na=False)) & 
        (game_df['player2_id'].notna())
    ]
    for _, row in assists.iterrows():
        tid = row['team_id']
        p2 = row['player2_id']
        if tid and p2 and p2 != '0':
            pt_map[p2] = tid
            
    # 4. Fallback (Player 1 -> Team ID)
    for _, row in game_df.iterrows():
        p1 = row.get('player1_id')
        tid = row.get('team_id')
        if tid and p1 and p1 != '0':
            if p1 not in pt_map:
                pt_map[p1] = tid
                
    # 5. Opposite Team Mapping
    teams = list(set(game_df['team_id'].dropna().unique()) - {'0'})
    if len(teams) == 2:
        t1, t2 = teams[0], teams[1]
        for _, row in game_df.iterrows():
            tid = row.get('team_id')
            if not tid or tid == '0': continue
            opp_tid = t2 if tid == t1 else t1
            
            p2 = row.get('player2_id')
            p3 = row.get('player3_id')
            etype = row.get('event_type')
            
            if p3 and p3 != '0': pt_map[p3] = opp_tid
            if etype == 'FOUL' and p2 and p2 != '0': pt_map[p2] = opp_tid
            if etype == 'TURNOVER' and p2 and p2 != '0': pt_map[p2] = opp_tid

    return pt_map

def get_initial_lineup(period_events, team_id, pt_map, game_id=None):
    if not team_id:
        return []

    starters = set()
    subs_in = set()
    
    # DEBUG TRACE
    is_debug = (game_id == DEBUG_GAME_ID and str(team_id) == DEBUG_TEAM_ID)
    if is_debug:
        print(f"\n[DEBUG] Solving Starters for {team_id} in Game {game_id} (Period {period_events['period'].iloc[0]})")
        print(f"[DEBUG] Map check for Horford ({DEBUG_PLAYER_ID}): {pt_map.get(DEBUG_PLAYER_ID)}")

    for _, row in period_events.iterrows():
        candidates = []
        if row.get('player1_id'): candidates.append(row['player1_id'])
        if row.get('player2_id'): candidates.append(row['player2_id'])
        if row.get('player3_id'): candidates.append(row['player3_id'])
        
        team_candidates = [p for p in candidates if pt_map.get(p) == team_id and p != '0']
        
        etype = row.get('event_type')
        desc = str(row.get('event_text', '')).upper()
        p1 = row.get('player1_id')
        
        if etype == 'SUBSTITUTION':
            if p1 in team_candidates:
                if 'SUB IN' in desc:
                    subs_in.add(p1)
                elif 'SUB OUT' in desc:
                    if p1 not in subs_in:
                        starters.add(p1)
                        if is_debug and p1 == DEBUG_PLAYER_ID:
                            print(f"[DEBUG] Horford SUB OUT -> Added to Starters")
        else:
            for p in team_candidates:
                if p not in subs_in:
                    starters.add(p)
                    if is_debug and p == DEBUG_PLAYER_ID and p in starters:
                        # Only print once to avoid spam
                        pass
    
    if is_debug:
        print(f"[DEBUG] Final Starters: {starters}")
        if DEBUG_PLAYER_ID not in starters:
            print(f"[DEBUG] ❌ Horford MISSING from starters!")

    return list(starters)

def process_game_period(df_gp, pt_map, game_id=None):
    df_gp = df_gp.sort_index()
    teams = list(set(df_gp['team_id'].dropna().unique()) - {'0'})
    
    if len(teams) != 2:
        return df_gp

    team_a, team_b = teams[0], teams[1]
    
    starters_a = get_initial_lineup(df_gp, team_a, pt_map, game_id)
    starters_b = get_initial_lineup(df_gp, team_b, pt_map, game_id)
    
    current_a = set(starters_a)
    current_b = set(starters_b)
    
    lineups_a = []
    lineups_b = []
    
    for _, row in df_gp.iterrows():
        row_team = row.get('team_id')
        p1 = row.get('player1_id')
        etype = row.get('event_type')
        desc = str(row.get('event_text', '')).upper()
        
        if etype == 'SUBSTITUTION':
            if p1 and p1 != '0':
                if row_team == team_a:
                    if 'SUB IN' in desc: current_a.add(p1)
                    if 'SUB OUT' in desc and p1 in current_a: current_a.remove(p1)
                elif row_team == team_b:
                    if 'SUB IN' in desc: current_b.add(p1)
                    if 'SUB OUT' in desc and p1 in current_b: current_b.remove(p1)
        
        lineups_a.append(sorted(list(current_a)))
        lineups_b.append(sorted(list(current_b)))

    df_gp[f'lineup_{team_a}'] = lineups_a
    df_gp[f'lineup_{team_b}'] = lineups_b
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
    
    for col in ['player1_id', 'player2_id', 'player3_id', 'team_id']:
        if col in df.columns:
            df[col] = df[col].apply(to_id)
            
    processed_dfs = []
    grouped = df.groupby('game_id')
    total = len(grouped)
    
    for i, (gid, game_df) in enumerate(grouped):
        if i % 100 == 0:
            print(f"  Game {i}/{total}...", end='\r')
            
        pt_map = build_player_team_map(game_df)
        
        period_chunks = []
        for p, period_df in game_df.groupby('period'):
            # Pass game_id for debugging
            period_chunks.append(process_game_period(period_df, pt_map, gid))
            
        if period_chunks:
            processed_dfs.append(pd.concat(period_chunks))
            
    print(f"\nConcatenating {season}...")
    if processed_dfs:
        final_df = pd.concat(processed_dfs)
        print(f"Saving to {output_path}...")
        final_df.to_parquet(output_path, index=False)
        
        # Validation
        # Check specific broken case if available in this season
        if DEBUG_GAME_ID in final_df['game_id'].values:
            debug_rows = final_df[(final_df['game_id'] == DEBUG_GAME_ID) & (final_df['period'] == 1)]
            if not debug_rows.empty:
                first_row = debug_rows.iloc[0]
                col = f'lineup_{DEBUG_TEAM_ID}'
                if col in first_row:
                    print(f"  [FINAL VALIDATION] Game {DEBUG_GAME_ID} Team {DEBUG_TEAM_ID} Starters: {first_row[col]}")
                    print(f"  [FINAL VALIDATION] Length: {len(first_row[col])}")

def main():
    pattern = os.path.join(DATA_DIR, "pbp_normalized_*.parquet")
    files = sorted(glob.glob(pattern))
    for f in files:
        process_file(f)

if __name__ == "__main__":
    main()