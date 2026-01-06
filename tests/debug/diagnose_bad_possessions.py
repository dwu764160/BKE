"""
tests/diagnose_bad_possession.py
Scans FINAL POSSESSIONS for bad lineups (not 5 players).
Dumps the raw event log for the specific Game+Period to debug the root cause.
"""

import pandas as pd
import numpy as np
import sys
import os
import glob

# Adjust path to import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
from features.derive_lineups import to_id

DATA_DIR = "data/historical"
NORM_DIR = "data/historical" # Assuming pbp_normalized is here too

def find_first_bad_possession():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "possessions_*.parquet")))
    
    for f in files:
        print(f"Scanning {os.path.basename(f)}...")
        df = pd.read_parquet(f)
        
        # Check Offense Lineups
        for idx, row in df.iterrows():
            lu = row['off_lineup']
            if isinstance(lu, (list, np.ndarray)):
                if len(lu) != 5:
                    print(f"\n🚨 FOUND BAD POSSESSION (Offense has {len(lu)} players)")
                    print(f"  File:    {os.path.basename(f)}")
                    print(f"  Game ID: {row['game_id']}")
                    print(f"  Period:  {row['period']}")
                    print(f"  Team ID: {row['off_team_id']}")
                    print(f"  Lineup:  {lu}")
                    return row['game_id'], row['period'], row['off_team_id'], lu
        
        # Check Defense Lineups
        for idx, row in df.iterrows():
            lu = row['def_lineup']
            if isinstance(lu, (list, np.ndarray)):
                if len(lu) != 5:
                    print(f"\n🚨 FOUND BAD POSSESSION (Defense has {len(lu)} players)")
                    print(f"  File:    {os.path.basename(f)}")
                    print(f"  Game ID: {row['game_id']}")
                    print(f"  Period:  {row['period']}")
                    print(f"  Team ID: {row['def_team_id']}")
                    print(f"  Lineup:  {lu}")
                    return row['game_id'], row['period'], row['def_team_id'], lu

    print("✅ No bad possessions found.")
    return None, None, None, None

def dump_raw_log(game_id, period, team_id, bad_lineup):
    # Find matching normalized file (heuristic)
    # We look for the file that contains this game
    norm_files = sorted(glob.glob(os.path.join(NORM_DIR, "pbp_normalized_*.parquet")))
    target_df = None
    
    print("  Searching for raw game logs...")
    for f in norm_files:
        try:
            df = pd.read_parquet(f)
            if game_id in df['game_id'].values:
                target_df = df
                break
        except:
            continue
            
    if target_df is None:
        print("❌ Could not find raw data for this game.")
        return

    print(f"\n--- RAW LOG for Game {game_id} Period {period} ---")
    
    # Filter
    mask = (target_df['game_id'] == game_id) & (target_df['period'] == period)
    events = target_df[mask].sort_index()
    
    # Safely build a set of bad player ids. Avoid numpy array truth-value checks.
    if bad_lineup is None:
        bad_set = set()
    else:
        try:
            bad_list = list(bad_lineup)
        except TypeError:
            bad_list = [bad_lineup]
        # Normalize to strings and filter out nulls
        bad_set = set(str(x) for x in bad_list if pd.notnull(x))
    
    print(f"{'CLOCK':<8} | {'EVENT TYPE':<20} | {'TEAM':<15} | {'PLAYER 1':<15} | DESCRIPTION")
    print("-" * 100)
    
    for idx, row in events.iterrows():
        p1 = to_id(row.get('player1_id'))
        team = to_id(row.get('team_id'))
        etype = row.get('event_type')

        # Highlight logic
        prefix = "  "
        if str(team) == str(team_id):
            prefix = ">>"
        if etype == 'SUBSTITUTION':
            prefix = "!!"

        # Coerce possible None values to safe strings for formatting
        clock = row.get('clock') if pd.notnull(row.get('clock')) else ''
        etype_str = etype if etype is not None else ''
        team_str = str(team) if team is not None else ''
        p1_str = str(p1) if p1 is not None else ''
        if p1_str in bad_set:
            p1_str = f"*{p1_str}*"
        event_text = row.get('event_text') if row.get('event_text') is not None else ''

        print(f"{prefix} {clock:<8} | {etype_str:<20} | {team_str:<15} | {p1_str:<15} | {event_text}")

if __name__ == "__main__":
    gid, per, tid, lu = find_first_bad_possession()
    if gid:
        dump_raw_log(gid, per, tid, lu)
