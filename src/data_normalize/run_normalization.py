"""
src/data_normalize/run_normalization.py
Driver script to normalize raw PBP data using the robust pbp_parser.
Automatically detects and processes all season files found in data/historical.
"""
import pandas as pd
import sys
import os
import glob
import re

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data_normalize.pbp_parser import normalize_pbp_row

DATA_DIR = "data/historical"

def process_file(input_path):
    filename = os.path.basename(input_path)
    # Expecting filename format: play_by_play_YYYY-YY.parquet
    match = re.search(r"play_by_play_(\d{4}-\d{2})\.parquet", filename)
    
    if match:
        season = match.group(1)
        output_path = os.path.join(DATA_DIR, f"pbp_normalized_{season}.parquet")
    else:
        # Fallback for non-standard names
        output_path = input_path.replace("play_by_play", "pbp_normalized")
        if output_path == input_path:
            output_path = input_path.replace(".parquet", "_normalized.parquet")

    print(f"\n--- Processing {filename} -> {os.path.basename(output_path)} ---")
    
    try:
        df = pd.read_parquet(input_path)
    except Exception as e:
        print(f"❌ Failed to read {filename}: {e}")
        return

    print(f"Normalizing {len(df)} rows...")
    
    records = df.to_dict(orient="records")
    normalized_records = [normalize_pbp_row(r) for r in records]
    
    df_norm = pd.DataFrame(normalized_records)

    # Basic Validation Stats
    if "event_type" in df_norm.columns:
        print("Event Types (Top 5):")
        print(df_norm["event_type"].value_counts().head(5))
    
    if "event_type" in df_norm.columns and "is_made" in df_norm.columns:
        fg_mask = df_norm["event_type"].isin(["FIELD_GOAL", "FIELD_GOAL_2PT", "FIELD_GOAL_3PT"])
        if fg_mask.any():
            fg_pct = df_norm[fg_mask]["is_made"].mean()
            print(f"FG%: {fg_pct:.1%} (Expected ~46-48%)")

    print(f"Saving to {output_path}...")
    df_norm.to_parquet(output_path, index=False)
    print("✅ Done.")

def main():
    # Find all play_by_play files
    pattern = os.path.join(DATA_DIR, "play_by_play_*.parquet")
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"No raw PBP files found in {DATA_DIR} matching 'play_by_play_*.parquet'")
        return

    print(f"Found {len(files)} season files.")
    for f in files:
        process_file(f)

if __name__ == "__main__":
    main()