"""
src/data_normalize/run_normalization.py
Driver script to normalize raw PBP data using the robust pbp_parser.
"""
import pandas as pd
import sys
import os

# Add project root to path (hack for now until setup.py exists)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data_normalize.pbp_parser import normalize_pbp_row

# Using the file name indicated in your prompt/context
INPUT_FILE = "data/historical/play_by_play_2022-23.parquet"
OUTPUT_FILE = "data/historical/pbp_normalized.parquet"

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file {INPUT_FILE} not found.")
        return

    print(f"Reading {INPUT_FILE}...")
    try:
        df = pd.read_parquet(INPUT_FILE)
    except Exception as e:
        print(f"Failed to read parquet file: {e}")
        return
    
    print(f"Normalizing {len(df)} rows...")
    
    # Convert DataFrame to records, process, and convert back
    records = df.to_dict(orient="records")
    normalized_records = [normalize_pbp_row(r) for r in records]
    
    df_norm = pd.DataFrame(normalized_records)

    # Basic Validation Stats
    print("\n--- Normalization Stats ---")
    if "event_type" in df_norm.columns:
        print(df_norm["event_type"].value_counts())
    
    print("\n--- Shot Results (Field Goals) ---")
    if "event_type" in df_norm.columns and "is_made" in df_norm.columns:
        fg_mask = df_norm["event_type"].isin(["FIELD_GOAL", "FIELD_GOAL_2PT", "FIELD_GOAL_3PT"])
        if fg_mask.any():
            print(df_norm[fg_mask]["is_made"].value_counts(normalize=True).rename("Made %"))
        else:
            print("No field goals found.")

    print(f"\nSaving to {OUTPUT_FILE}...")
    df_norm.to_parquet(OUTPUT_FILE, index=False)
    print("Done.")

if __name__ == "__main__":
    main()