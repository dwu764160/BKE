"""
tests/debug/inspect_missing_stats.py
Locates 'Traveling' and 'Steal' events to fix missing stats.
"""

import pandas as pd
import os
import sys

# Adjust path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

DATA_FILE = "data/historical/pbp_with_lineups_2023-24.parquet"

def inspect():
    if not os.path.exists(DATA_FILE):
        print("File not found.")
        return

    print(f"Scanning {DATA_FILE}...")
    df = pd.read_parquet(DATA_FILE)
    
    # 1. SEARCH FOR TRAVELING
    print("\n--- TRAVELING EVENTS (Sample) ---")
    travels = df[df.get('event_text', pd.Series()).astype(str).str.contains("Travel", case=False, na=False)].head(10)

    if travels.empty:
        print("❌ No 'Traveling' text found in dataset.")
    else:
        # Show only columns that exist to avoid KeyError
        candidate_cols = ['game_id', 'event_msg_type', 'event_action_type', 'player1_id', 'event_text']
        cols = [c for c in candidate_cols if c in travels.columns]
        print(travels[cols].to_string(index=False))

    # 2. SEARCH FOR STEALS
    print("\n--- STEAL EVENTS (Sample) ---")
    # Search for text "Steal"
    steals = df[df.get('event_text', pd.Series()).astype(str).str.contains("Steal", case=False, na=False)].head(10)

    if steals.empty:
        print("❌ No 'Steal' text found.")
    else:
        candidate_cols = ['game_id', 'event_msg_type', 'player1_id', 'player2_id', 'event_text']
        cols = [c for c in candidate_cols if c in steals.columns]
        print(steals[cols].to_string(index=False))
        
    # 3. CHECK TURNOVER PLAYER 2 (For Steal derivation)
    print("\n--- TURNOVER PLAYER 2 STATS ---")
    if 'event_msg_type' in df.columns:
        tovs = df[df['event_msg_type'] == 5]
    elif 'event_type' in df.columns:
        tovs = df[df['event_type'].astype(str).str.contains('TURNOVER', case=False, na=False)]
    else:
        tovs = df[df.get('event_text', pd.Series()).astype(str).str.contains('TURNOVER', case=False, na=False)]

    print(f"Total Turnovers: {len(tovs)}")
    if 'player2_id' in tovs.columns:
        print(f"Turnovers with Player 2 (Stealer): {tovs['player2_id'].nunique()}")
        print("Sample Player 2 IDs (Stealers):", tovs['player2_id'].unique()[:5])
    else:
        print("player2_id column not present in data.")

if __name__ == "__main__":
    inspect()