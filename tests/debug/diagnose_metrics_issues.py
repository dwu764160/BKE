"""
tests/debug/diagnose_metrics_issues.py
1. Checks why names are 'Unknown' (inspects reference parquet files).
2. Audits the 'Extreme Net Rating' lineups to see if they are data errors or real outliers.
"""

import pandas as pd
import numpy as np
import os
import sys
import glob

# Adjust path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DATA_DIR = "data/historical"
CLEAN_POSS_DIR = "data/historical"

# The specific lineup ID list from your error log (Knicks Bench unit)
TARGET_LINEUP = ['1628392', '1628404', '1629628', '1630167', '1630540']
# IDs: Hartenstein, Josh Hart, (Likely RJ/IQ/Grimes?), Obi Toppin, McBride

def check_reference_data():
    print("\n=== 1. Checking Reference Data (Names) ===")
    
    # Check Players
    p_path = os.path.join(DATA_DIR, "players.parquet")
    if not os.path.exists(p_path):
        print("❌ players.parquet NOT found. Run 'src/data_fetch/export_db_to_parquet.py'.")
    else:
        try:
            df = pd.read_parquet(p_path)
            print(f"✅ Loaded players.parquet ({len(df)} rows)")
            
            # Check for Hartenstein (1628392)
            target_id = "1628392"
            # Ensure column is string for search
            if 'id' in df.columns:
                match = df[df['id'].astype(str) == target_id]
            elif 'player_id' in df.columns:
                match = df[df['player_id'].astype(str) == target_id]
            else:
                print("❌ 'id' or 'player_id' column missing in players.parquet")
                match = pd.DataFrame()

            if not match.empty:
                print(f"✅ Found Target Player {target_id}: {match.iloc[0].to_dict()}")
            else:
                print(f"❌ Target Player {target_id} NOT found in players.parquet.")
                print("   Sample IDs in file:", df.iloc[:5, 0].tolist())
        except Exception as e:
            print(f"❌ Error reading players.parquet: {e}")

def audit_extreme_lineup():
    print("\n=== 2. Auditing Extreme Lineup Stats ===")
    
    files = sorted(glob.glob(os.path.join(CLEAN_POSS_DIR, "possessions_clean_*.parquet")))
    found_possessions = []
    
    target_set = set(TARGET_LINEUP)
    
    for f in files:
        try:
            df = pd.read_parquet(f)
            # Find rows where off_lineup matches target
            # Note: stored as array/list in parquet
            
            # Helper to check subset/equality (handling float/str mismatch)
            def is_match(x):
                if not isinstance(x, (list, np.ndarray)): return False
                # Convert to clean strings
                x_clean = set(str(pid).replace(".0", "") for pid in x)
                return x_clean == target_set

            mask = df['off_lineup'].apply(is_match)
            matches = df[mask]
            
            if not matches.empty:
                found_possessions.append(matches)
                print(f"  Found {len(matches)} possessions in {os.path.basename(f)}")
                
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not found_possessions:
        print("❌ Could not find any possessions for this lineup.")
        return

    all_poss = pd.concat(found_possessions)
    
    # Manual Calc
    total_poss = len(all_poss)
    total_pts = all_poss['points'].sum()
    ortg = (total_pts / total_poss) * 100
    
    print(f"\n--- Audit Results ---")
    print(f"Total Possessions: {total_poss}")
    print(f"Total Points:      {total_pts}")
    print(f"Calculated ORTG:   {ortg:.2f}")
    
    print("\n--- Sample Possessions (First 5) ---")
    # Only print columns that actually exist to avoid KeyError
    sample_cols = [c for c in ['period', 'clock', 'points', 'end_reason'] if c in all_poss.columns]
    if sample_cols:
        print(all_poss[sample_cols].head(5))
    else:
        print(all_poss.head(5))
    
    print("\n--- Score Distribution ---")
    print(all_poss['points'].value_counts().sort_index())

if __name__ == "__main__":
    audit_extreme_lineup()