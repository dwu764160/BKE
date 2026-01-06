"""
tests/diagnose_lineups.py
Forensic tool to investigate why a specific lineup has < 5 players.
"""

import pandas as pd
import numpy as np
import sys
import os

# Adjust path to import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
from features.derive_lineups import build_player_team_map, get_initial_lineup, to_id

# Target the 2022-23 file
LINEUP_FILE = "data/historical/pbp_with_lineups_2022-23.parquet"
NORM_FILE = "data/historical/pbp_normalized_2022-23.parquet"

def is_bad_lineup(val):
    # Handle Numpy arrays, Lists, or other iterables
    if val is None: return False
    try:
        # Convert to list to check length safely
        l = list(val)
        return len(l) > 0 and len(l) != 5
    except:
        return False

def find_broken_case():
    print(f"Scanning {LINEUP_FILE} for the first broken lineup...")
    try:
        df = pd.read_parquet(LINEUP_FILE)
    except Exception as e:
        print(f"Failed to read file: {e}")
        return None, None, None

    lineup_cols = [c for c in df.columns if c.startswith('lineup_')]
    
    for idx, row in df.iterrows():
        for col in lineup_cols:
            lineup = row[col]
            if is_bad_lineup(lineup):
                print(f"\n🚨 FOUND ERROR CASE:")
                print(f"  Game ID: {row['game_id']}")
                print(f"  Period:  {row['period']}")
                print(f"  Team ID: {col.split('_')[1]}")
                print(f"  Lineup:  {list(lineup)} (Len: {len(lineup)})")
                return row['game_id'], row['period'], col.split('_')[1]
                
    print("✅ No broken lineups found.")
    return None, None, None

def analyze_case(game_id, period, team_id):
    print(f"\n--- DEEP DIVE: Game {game_id} | Period {period} | Team {team_id} ---")
    
    df = pd.read_parquet(NORM_FILE)
    
    # Standardize IDs (Mimic production logic)
    for col in ['player1_id', 'player2_id', 'player3_id', 'team_id']:
        df[col] = df[col].apply(to_id)
        
    game_df = df[df['game_id'] == game_id].copy()
    period_df = game_df[game_df['period'] == period].copy()
    
    # 1. Check Map
    print("\n1. Building Player-Team Map...")
    pt_map = build_player_team_map(game_df)
    
    team_players = [p for p, t in pt_map.items() if t == team_id]
    print(f"   Mapped {len(team_players)} players to Team {team_id}.")
    print(f"   Players: {team_players}")
    
    # 2. Check Raw Events
    print("\n2. Scanning Period Events for Candidates...")
    candidates = set()
    unmapped_candidates = set()
    
    for _, row in period_df.iterrows():
        parts = [row['player1_id'], row['player2_id'], row['player3_id']]
        for p in parts:
            if p:
                if pt_map.get(p) == team_id:
                    candidates.add(p)
                elif p not in pt_map and p != '0':
                    unmapped_candidates.add(p)
        
    print(f"   Mapped Candidates in Period: {list(candidates)}")
    print(f"   UNMAPPED Players in Period: {list(unmapped_candidates)}")
    
    # 3. Trace Starter Logic
    print("\n3. Tracing Starter Logic...")
    starters = set()
    subs_in = set()
    
    # Pre-fill starters with players who act before subbing in
    # This mimics get_initial_lineup logic
    
    for i, row in period_df.iterrows():
        p1 = row['player1_id']
        etype = row['event_type']
        desc = str(row['event_text']).upper()
        
        # Is P1 on our team?
        if p1 in candidates:
            if etype == 'SUBSTITUTION':
                if 'SUB IN' in desc:
                    subs_in.add(p1)
                    # print(f"   [SUB IN] {p1}")
                elif 'SUB OUT' in desc:
                    if p1 not in subs_in:
                        starters.add(p1)
                        print(f"   [SUB OUT] {p1} -> STARTER (Subbed out before In)")
            else:
                if p1 not in subs_in and p1 not in starters:
                    starters.add(p1)
                    print(f"   [ACTION]  {p1} ({etype}) -> STARTER (Active before Sub In)")

    print(f"\nFinal Calculated Starters: {list(starters)} (Len: {len(starters)})")

if __name__ == "__main__":
    gid, per, tid = find_broken_case()
    if gid:
        analyze_case(gid, per, tid)

