"""
tests/pbp_test.py
Validates integrity of NBA Play-by-Play data.
Checks:
1. Raw Game Counts (Completeness)
2. Row Density (Quality)
3. Event Distribution (Normalization Logic, including 2PT vs 3PT)
"""

import pandas as pd
import sys
import os
from pathlib import Path

# Paths
RAW_FILE = Path("data/historical/play_by_play_2022-23.parquet")
NORM_FILE = Path("data/historical/pbp_normalized.parquet")

# Thresholds
MIN_ROWS_PER_GAME = 300  # A complete NBA game usually has 400+ events

def check_raw_completeness():
    print(f"\n--- 1. Raw Data Check ({RAW_FILE}) ---")
    if not RAW_FILE.exists():
        print(f"❌ Raw file not found: {RAW_FILE}")
        return

    df = pd.read_parquet(RAW_FILE)
    n_rows = len(df)
    n_games = df["GAME_ID"].nunique()
    
    print(f"Total Rows:       {n_rows:,}")
    print(f"Unique Games:     {n_games:,}")
    print(f"Avg Rows/Game:    {n_rows / n_games:.1f}")

    # Check for low-density games (potential incomplete scrapes)
    game_counts = df["GAME_ID"].value_counts()
    incomplete = game_counts[game_counts < MIN_ROWS_PER_GAME]
    
    if len(incomplete) > 0:
        print(f"⚠️  Warning: {len(incomplete)} games have < {MIN_ROWS_PER_GAME} rows.")
        print(f"    Examples: {incomplete.head(3).to_dict()}")
    else:
        print("✅ Data density looks healthy (all games > 300 rows).")

def check_normalization_quality():
    print(f"\n--- 2. Normalization Check ({NORM_FILE}) ---")
    if not NORM_FILE.exists():
        print(f"❌ Normalized file not found. Run 'src/data_normalize/run_normalization.py' first.")
        return

    df = pd.read_parquet(NORM_FILE)
    
    # 1. Check Event Types
    if "event_type" not in df.columns:
        print("❌ 'event_type' column missing!")
        return

    counts = df["event_type"].value_counts()
    print("Top Event Types:")
    print(counts.head(10))

    # 2. Verify 2PT vs 3PT Distinction
    has_3pt = "FIELD_GOAL_3PT" in counts.index
    has_2pt = "FIELD_GOAL_2PT" in counts.index
    
    if has_3pt and has_2pt:
        print("\n✅ Success: 2PT and 3PT shots are distinguished.")
        total_shots = counts.get("FIELD_GOAL_2PT", 0) + counts.get("FIELD_GOAL_3PT", 0)
        p3 = counts.get("FIELD_GOAL_3PT", 0) / total_shots if total_shots else 0
        print(f"    3PT Rate: {p3:.1%} of all field goals (League Avg is ~39-40%)")
    else:
        print("\n⚠️  Warning: 2PT/3PT distinction NOT found.")
        print("    Ensure you are using the latest 'pbp_parser.py' and have re-run normalization.")

    # 3. Shot Accuracy Check
    if "is_made" in df.columns:
        # Filter only for shots
        shots = df[df["event_type"].str.contains("FIELD_GOAL", na=False)]
        if not shots.empty:
            fg_pct = shots["is_made"].mean()
            print(f"\n✅ Field Goal % Check: {fg_pct:.1%} (Expected ~47-48%)")
        else:
            print("\n⚠️  No FIELD_GOAL events found to check accuracy.")

if __name__ == "__main__":
    check_raw_completeness()
    check_normalization_quality()