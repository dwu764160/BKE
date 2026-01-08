"""
tests/debug/inspect_foul_codes.py
Scans raw PBP data to identify the exact codes for:
1. Technical Fouls (to exclude from PF)
2. Offensive Fouls (to ensure they count as TOV)
"""

import pandas as pd
import os
import sys

# Adjust path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

DATA_FILE = "data/historical/pbp_with_lineups_2023-24.parquet"
# IDs: Luka (1629029), Giannis (203507)
TARGET_PLAYERS = ['1629029', '203507'] 

def inspect():
    if not os.path.exists(DATA_FILE):
        print("File not found.")
        return

    print(f"Scanning {DATA_FILE}...")
    df = pd.read_parquet(DATA_FILE)
    
    # Filter for events involving our targets
    # Player1 is usually the person committing the foul/turnover
    mask = df['player1_id'].astype(str).str.replace('.0','').isin(TARGET_PLAYERS)
    subset = df[mask].copy()
    
    # 1. Inspect FOULS (Event Type 6)
    print("\n=== FOUL TYPES (Msg Type 6) ===")
    fouls = subset[subset['event_type'] == 'FOUL']

    # Choose grouping columns based on what's available in the parquet
    pref_group = ['event_action_type', 'event_text']
    group_cols = [c for c in pref_group if c in fouls.columns]
    if not group_cols:
        # Fallback: at least group by event_text or raw_text if present
        for alt in ['event_text', 'raw_text']:
            if alt in fouls.columns:
                group_cols = [alt]
                break

    if not group_cols:
        print("No suitable grouping columns found for fouls. Available columns:", list(fouls.columns))
    else:
        foul_stats = fouls.groupby(group_cols).size().reset_index(name='Count')
        print(foul_stats.to_string())

    # 2. Inspect TURNOVERS (Event Type 5)
    print("\n=== TURNOVER TYPES (Msg Type 5) ===")
    tovs = subset[subset['event_type'] == 'TURNOVER']

    group_cols = [c for c in pref_group if c in tovs.columns]
    if not group_cols:
        for alt in ['event_text', 'raw_text']:
            if alt in tovs.columns:
                group_cols = [alt]
                break

    if not group_cols:
        print("No suitable grouping columns found for turnovers. Available columns:", list(tovs.columns))
    else:
        tov_stats = tovs.groupby(group_cols).size().reset_index(name='Count')
        print(tov_stats.to_string())

if __name__ == "__main__":
    inspect()