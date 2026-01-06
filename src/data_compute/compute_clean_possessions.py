"""
src/data_compute/compute_clean_possessions.py
Final cleaning step. 
1. Loads derived possessions.
2. Filters out the < 0.1% of rows with invalid lineup counts (3, 4, 6 players).
3. Saves the final 'clean' dataset for modeling.
"""

import pandas as pd
import numpy as np
import os
import sys
import glob

DATA_DIR = "data/historical"

def clean_file(filepath):
    filename = os.path.basename(filepath)
    print(f"\nProcessing {filename}...")
    
    try:
        df = pd.read_parquet(filepath)
    except Exception as e:
        print(f"❌ Error reading {filename}: {e}")
        return

    initial_count = len(df)
    
    # Validation Logic
    def is_valid(lineup):
        if isinstance(lineup, (list, np.ndarray)):
            return len(lineup) == 5
        return False

    valid_off = df['off_lineup'].apply(is_valid)
    valid_def = df['def_lineup'].apply(is_valid)
    
    # Keep only rows where BOTH offense and defense are perfect
    clean_df = df[valid_off & valid_def].copy()
    
    dropped = initial_count - len(clean_df)
    pct = (dropped / initial_count) * 100 if initial_count > 0 else 0
    
    print(f"  Initial: {initial_count:,}")
    print(f"  Clean:   {len(clean_df):,}")
    print(f"  Dropped: {dropped:,} ({pct:.3f}%)")
    
    if dropped > 0:
        # Save 'clean' version
        clean_path = filepath.replace("possessions_", "possessions_clean_")
        clean_df.to_parquet(clean_path, index=False)
        print(f"✅ Saved to {clean_path}")
    else:
        print("✅ File was already perfect.")

def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "possessions_*.parquet")))
    # Filter out already cleaned files to avoid recursion if run multiple times
    files = [f for f in files if "clean" not in f]
    
    if not files:
        print("No possession files found.")
        return
        
    for f in files:
        clean_file(f)

if __name__ == "__main__":
    main()