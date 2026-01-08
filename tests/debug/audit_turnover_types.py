"""
tests/debug/audit_turnover_types.py
Breakdown of Giannis's Turnovers by Description.
Diagnoses exactly which category (Travel vs Offensive Foul) is under-counted.
"""

import pandas as pd
import os
import sys

# Adjust path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

DATA_FILE = "data/historical/pbp_with_lineups_2023-24.parquet"
GIANNIS_ID = "203507"

def inspect():
    if not os.path.exists(DATA_FILE):
        print("File not found.")
        return

    print(f"Loading {DATA_FILE}...")
    df = pd.read_parquet(DATA_FILE)
    
    # Filter for Giannis
    df['player1_id'] = df['player1_id'].astype(str).str.replace('.0','')
    g_events = df[df['player1_id'] == GIANNIS_ID].copy()
    
    # 1. Analyze what we ARE counting (Type 5 or String 'TURNOVER')
    print("\n--- CURRENT TURNOVER BREAKDOWN ---")
    # This mimics your current script logic
    # Ensure we work on a copy to avoid SettingWithCopyWarning
    current_tovs = g_events[g_events['event_type'] == 'TURNOVER'].copy()
    
    # Extract keyword from text for grouping
    def get_category(text):
        text = str(text).upper()
        if "OFFENSIVE" in text or "CHARGE" in text: return "OFFENSIVE FOUL"
        if "TRAVEL" in text: return "TRAVELING"
        if "BAD PASS" in text: return "BAD PASS"
        if "LOST BALL" in text: return "LOST BALL"
        if "STEP OUT" in text: return "STEP OUT"
        if "3 SECOND" in text: return "3 SECONDS"
        return "OTHER"

    breakdown = current_tovs['event_text'].apply(get_category).value_counts()
    print(breakdown.to_string())
    print(f"TOTAL: {breakdown.sum()}")

    # 2. Analyze what we might be MISSING (Type 6 Offensive Fouls)
    print("\n--- OFFENSIVE FOULS (Type 6) ---")
    off_fouls = g_events[
        (g_events['event_type'] == 'FOUL') & 
        (g_events['event_text'].str.contains("OFFENSIVE|CHARGE", case=False, na=False))
    ].copy()
    print(f"Total Offensive Fouls (Type 6): {len(off_fouls)}")
    
    # Check if these are already in the turnover count
    # We match by Game/Period/Clock
    # Create keys using .assign() on copies to avoid SettingWithCopyWarning
    current_tovs = current_tovs.assign(key = current_tovs['game_id'].astype(str) + "_" + current_tovs['clock'].astype(str))
    off_fouls = off_fouls.assign(key = off_fouls['game_id'].astype(str) + "_" + off_fouls['clock'].astype(str))
    
    missing = off_fouls[~off_fouls['key'].isin(current_tovs['key'])]
    print(f"Offensive Fouls NOT in Turnover List: {len(missing)}")
    
    if len(missing) > 0:
        print("\nSample Missing Offensive Fouls:")
        print(missing[['game_id', 'clock', 'event_text']].head().to_string(index=False))

if __name__ == "__main__":
    inspect()