"""
tests/debug/inspect_travel_metadata.py
Inspects the 'event_type' of rows containing 'Traveling'.
"""
import pandas as pd
import os
import sys

# Adjust path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
DATA_FILE = "data/historical/pbp_with_lineups_2023-24.parquet"

def inspect():
    if not os.path.exists(DATA_FILE): return
    df = pd.read_parquet(DATA_FILE)
    
    # Filter for rows with "Traveling" in text
    travels = df[df['event_text'].fillna("").str.contains("Traveling", case=False)]
    
    if travels.empty:
        print("No 'Traveling' text found.")
        return

    print(f"Found {len(travels)} Traveling Events.")
    
    # Show the breakdown of Event Types for these rows
    print("\n--- Event Types for 'Traveling' rows ---")
    print(travels['event_type'].value_counts(dropna=False))
    
    if 'event_msg_type' in df.columns:
        print("\n--- Msg Types for 'Traveling' rows ---")
        print(travels['event_msg_type'].value_counts(dropna=False))
        
    print("\n--- Sample Rows ---")
    print(travels[['game_id', 'event_type', 'player1_id', 'event_text']].head(5).to_string(index=False))

if __name__ == "__main__":
    inspect()