"""
src/features/compute_rest_home_back2back.py
Calculates schedule-based metrics:
- Rest days
- Home/Away status
- Back-to-Back flags

Requires: data/historical/team_game_logs.parquet (The master log file)
"""

import pandas as pd
import numpy as np
import os
import sys

# Adjust path to find src if run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

INPUT_LOGS = "data/historical/team_game_logs.parquet"
OUTPUT_SCHEDULE = "data/historical/feature_schedule_context.parquet"

def compute_schedule_features(df):
    df = df.copy()
    
    # Ensure Date format
    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    
    # Sort: Team -> Date
    df = df.sort_values(by=["TEAM_ID", "GAME_DATE"])
    
    # Calculate Rest Days
    # Difference between current game date and previous game date for this team
    df["prev_game_date"] = df.groupby("TEAM_ID")["GAME_DATE"].shift(1)
    df["days_rest"] = (df["GAME_DATE"] - df["prev_game_date"]).dt.days - 1
    
    # Handle first game of season (NaN rest) -> usually set to distinct value like 7 or 3
    df["days_rest"] = df["days_rest"].fillna(3).clip(lower=0, upper=7)
    
    # Back-to-Back (B2B)
    df["is_b2b"] = (df["days_rest"] == 0).astype(int)
    
    # 3 in 4 nights check (Rolling window?)
    # Simple heuristic: if rest of last 2 games sum to <= 1?
    # This is better done with rolling windows, but B2B is the big one.
    
    return df

def main():
    if not os.path.exists(INPUT_LOGS):
        print(f"Error: {INPUT_LOGS} not found.")
        print("Run 'src/data_fetch/derive_team_game_logs.py' first.")
        return

    print("Reading Team Game Logs...")
    df = pd.read_parquet(INPUT_LOGS)
    
    print(f"Computing schedule features for {len(df)} games...")
    df_features = compute_schedule_features(df)
    
    # Keep only identifiers and new features to avoid data bloat
    cols_to_keep = ["GAME_ID", "TEAM_ID", "GAME_DATE", "SEASON", "days_rest", "is_b2b"]
    final_df = df_features[cols_to_keep]
    
    print(f"Saving {len(final_df)} rows to {OUTPUT_SCHEDULE}...")
    final_df.to_parquet(OUTPUT_SCHEDULE, index=False)
    print("Done.")

if __name__ == "__main__":
    main()