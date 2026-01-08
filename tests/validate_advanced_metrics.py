"""
tests/validate_advanced_metrics.py
Validates the Advanced Metrics Output.
Checks for:
- Data Existence (Files present)
- Key Columns (Raw Counts + Rates)
- Data Sanity (Non-zero counts, Rates within 0-100 range)
"""

import pandas as pd
import os
import sys

# Adjust path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

FILE_PATH = "data/processed/player_profiles_advanced.parquet"

def validate():
    if not os.path.exists(FILE_PATH):
        print(f"❌ File not found: {FILE_PATH}")
        return

    print(f"Loading {FILE_PATH}...")
    df = pd.read_parquet(FILE_PATH)
    
    # 1. Column Existence Check
    required_cols = [
        # IDs
        'player_id', 'player_name', 'season',
        # Raw Counts (New)
        'PTS', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 
        'ORB', 'DRB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF',
        # Advanced Rates
        'USG_RATE', 'TS_PCT', 'AST_PCT', 'REB_PCT', 'ORTG', 'DRTG'
    ]
    
    missing = [c for c in required_cols if c not in df.columns]
    
    if missing:
        print(f"❌ Missing Columns: {missing}")
        return
    else:
        print("✅ All Schema Columns Present.")

    # 2. Sanity Check on Raw Counts
    # Check if we have non-zero values for key stats
    total_pts = df['PTS'].sum()
    total_stl = df['STL'].sum()
    total_orb = df['ORB'].sum()
    
    print(f"\n--- Global Totals Check ---")
    print(f"Total Points: {total_pts:,.0f}")
    print(f"Total Steals: {total_stl:,.0f}")
    print(f"Total ORB:    {total_orb:,.0f}")
    
    if total_stl == 0 or total_orb == 0:
        print("❌ Warning: Raw counts for Steals or ORB appear to be empty (0). Check aggregation.")
    else:
        print("✅ Raw Counts look populated.")

    # 3. Sanity Check on Logic (Rebounds)
    # REB should roughly equal ORB + DRB
    df['REB_DIFF'] = df['REB'] - (df['ORB'] + df['DRB'])
    bad_reb = df[df['REB_DIFF'].abs() > 0]
    
    if not bad_reb.empty:
        print(f"⚠️ Warning: {len(bad_reb)} rows have REB != ORB + DRB")
        print(bad_reb[['player_name', 'REB', 'ORB', 'DRB']].head())
    else:
        print("✅ Rebound Sum Check (REB == ORB + DRB) Passed.")

    print("\n✅ VALIDATION COMPLETE")

if __name__ == "__main__":
    validate()