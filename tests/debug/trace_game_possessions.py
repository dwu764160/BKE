"""
tests/debug/trace_game_possessions.py
Dumps the play-by-play possession log for a specific game.
Used to manually spot where possession attribution flips incorrectly.
"""

import pandas as pd
import sys
import os

# Adjust path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

DATA_DIR = "data/historical"
SEASON = "2023-24"
TARGET_GAME = "0022300013" # The MIN (-10) vs SAS (+3) game from your log

def trace_game():
    path = os.path.join(DATA_DIR, f"possessions_{SEASON}.parquet")
    if not os.path.exists(path):
        print("File not found.")
        return

    df = pd.read_parquet(path)
    game = df[df['game_id'] == TARGET_GAME].sort_values(['period', 'start_clock'], ascending=[True, False])
    
    if game.empty:
        print(f"Game {TARGET_GAME} not found.")
        return

    print(f"\n=== TRACE LOG: {TARGET_GAME} ===")
    print(f"Total Possessions: {len(game)}")
    print(f"Team Counts:\n{game['off_team_id'].value_counts()}\n")
    
    # Print flow
    # Columns: Period, Time, Off_Team, Points, Reason
    print(f"{'PER':<3} {'CLOCK':<6} {'OFF_TEAM':<12} {'PTS':<3} {'REASON':<15} {'EVENTS'}")
    print("-" * 60)
    
    prev_team = None
    
    for _, row in game.iterrows():
        team = str(row['off_team_id'])
        
        # Highlight switching errors
        # Normally teams switch every row. If same team has 2 rows in a row, flag it.
        flag = ""
        if prev_team and team == prev_team:
            flag = "******** (Repeat Poss?)"
            
        print(f"{row['period']:<3} {row['start_clock']:<6} {team:<12} {row['points']:<3} {row['end_reason']:<15} {row['num_events']} {flag}")
        
        prev_team = team

if __name__ == "__main__":
    if len(sys.argv) > 1:
        TARGET_GAME = sys.argv[1]
    trace_game()