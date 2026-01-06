"""
tests/trace_bad_lineups.py
Scans a processed Lineup file for the FIRST instance of a non-5-man lineup.
Then dumps the raw events for that specific Game/Period to identify the cause.
"""

import pandas as pd
import sys
import os
import numpy as np

# Adjust path to import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
from features.derive_lineups import to_id

# FILE TO DEBUG (Change this to target different seasons)
TARGET_FILE = "data/historical/pbp_with_lineups_2024-25.parquet"
NORM_FILE = "data/historical/pbp_normalized_2024-25.parquet"

def find_first_error():
    print(f"Scanning {TARGET_FILE}...")
    try:
        df = pd.read_parquet(TARGET_FILE)
    except Exception as e:
        print(f"Error reading file: {e}")
        return None, None, None, None

    # Identify lineup columns
    lineup_cols = [c for c in df.columns if c.startswith('lineup_')]
    
    for idx, row in df.iterrows():
        for col in lineup_cols:
            lineup = row[col]
            # Check for bad length (not 5) and ensure it's not just empty/garbage start
            if isinstance(lineup, (list, np.ndarray)):
                if len(lineup) > 0 and len(lineup) != 5:
                    team_id = col.split('_')[1]
                    print(f"\n🚨 FOUND BAD LINEUP at Index {idx}")
                    print(f"  Game ID: {row['game_id']}")
                    print(f"  Period:  {row['period']}")
                    print(f"  Clock:   {row['clock']}")
                    print(f"  Team ID: {team_id}")
                    print(f"  Lineup:  {list(lineup)} (Len: {len(lineup)})")
                    print(f"  Event:   {row['event_type']} | {row['event_text']}")
                    return row['game_id'], row['period'], team_id, set(lineup)
    
    print("✅ No errors found in this file.")
    return None, None, None, None

def dump_game_log(game_id, period, target_team_id, target_lineup):
    print(f"\n--- EVENT LOG for Game {game_id} Period {period} ---")
    df = pd.read_parquet(NORM_FILE)
    
    # Filter
    mask = (df['game_id'] == game_id) & (df['period'] == period)
    events = df[mask].sort_index()
    
    print(f"{'CLOCK':<8} | {'EVENT TYPE':<20} | {'TEAM':<15} | {'PLAYER 1':<15} | DESCRIPTION")
    print("-" * 100)
    
    for idx, row in events.iterrows():
        p1 = to_id(row['player1_id'])
        team = to_id(row['team_id'])
        etype = row['event_type']
        desc = str(row['event_text'])
        
        # Formatting
        p1_str = str(p1) if p1 else "-"
        team_str = str(team) if team else "-"
        
        # Markers
        prefix = "  "
        suffix = ""
        
        # Highlight events involving the target team
        if team == target_team_id:
            prefix = ">>"
            
        # Highlight players in the bad lineup
        if p1 in target_lineup:
            p1_str = f"*{p1_str}*"
            
        # Highlight Substitutions (The likely culprit)
        if etype == 'SUBSTITUTION':
            prefix = "!!"
            
        print(f"{prefix} {row['clock']:<8} | {etype:<20} | {team_str:<15} | {p1_str:<15} | {desc} {suffix}")

if __name__ == "__main__":
    gid, per, tid, lineup_set = find_first_error()
    if gid:
        dump_game_log(gid, per, tid, lineup_set)
