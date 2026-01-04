"""
tests/pbp_test.py
Validates integrity of NBA Play-by-Play data across ALL fetched seasons.
"""

import pandas as pd
import sys
import os
import glob
import re
from pathlib import Path

DATA_DIR = "data/historical"
MIN_ROWS_PER_GAME = 300 

def check_season(raw_path, norm_path, season_label):
    print(f"\n=== Testing Season: {season_label} ===")
    
    # 1. Raw Check
    if not os.path.exists(raw_path):
        print(f"❌ Raw file missing: {raw_path}")
        return

    df_raw = pd.read_parquet(raw_path)
    n_games = df_raw["GAME_ID"].nunique() if "GAME_ID" in df_raw.columns else 0
    print(f"Raw Rows: {len(df_raw):,} | Unique Games: {n_games}")
    
    if n_games > 0:
        density = len(df_raw) / n_games
        print(f"Avg Rows/Game: {density:.1f}")
        if density < MIN_ROWS_PER_GAME:
            print("⚠️  Warning: Low data density.")
    
    # 2. Normalization Check
    if not os.path.exists(norm_path):
        print(f"❌ Normalized file missing: {norm_path}")
        return

    df_norm = pd.read_parquet(norm_path)
    
    # Check 3PT vs 2PT
    if "event_type" in df_norm.columns:
        counts = df_norm["event_type"].value_counts()
        has_3pt = "FIELD_GOAL_3PT" in counts
        has_2pt = "FIELD_GOAL_2PT" in counts
        
        if has_3pt and has_2pt:
            p3_count = counts.get("FIELD_GOAL_3PT", 0)
            p2_count = counts.get("FIELD_GOAL_2PT", 0)
            total_shots = p3_count + p2_count
            rate = p3_count / total_shots if total_shots else 0
            print(f"✅ 2PT/3PT Distinction: OK. 3PT Rate: {rate:.1%} ({p3_count:,} shots)")
        else:
            print("❌ Failed: 2PT/3PT distinction missing.")
            
    # Check FG% Accuracy
    if "is_made" in df_norm.columns and "event_type" in df_norm.columns:
        # Filter for FG events (including blocks if your logic handles them)
        shots = df_norm[df_norm["event_type"].str.contains("FIELD_GOAL", na=False)]
        if not shots.empty:
            fg_pct = shots["is_made"].mean()
            print(f"✅ FG% Check: {fg_pct:.1%} (Target: ~46-48%)")
        else:
            print("⚠️  No field goals found to check.")

def main():
    # Find all raw files
    raw_pattern = os.path.join(DATA_DIR, "play_by_play_*.parquet")
    raw_files = sorted(glob.glob(raw_pattern))
    
    if not raw_files:
        print("No data found.")
        return

    for raw_f in raw_files:
        # Extract season (e.g. 2022-23)
        match = re.search(r"play_by_play_(\d{4}-\d{2})\.parquet", raw_f)
        if match:
            season = match.group(1)
            # Construct expected normalized path
            norm_f = os.path.join(DATA_DIR, f"pbp_normalized_{season}.parquet")
            check_season(raw_f, norm_f, season)
        else:
            print(f"Skipping non-standard file: {raw_f}")

if __name__ == "__main__":
    main()