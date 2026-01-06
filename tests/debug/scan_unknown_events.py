"""
tests/scan_unknown_events.py
Scans normalized PBP data for 'UNKNOWN' event types to identify missing parsers.
"""

import pandas as pd
import glob
import os
import sys

DATA_DIR = "data/historical"

def scan_unknowns():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "pbp_normalized_*.parquet")))
    if not files:
        print("No normalized files found.")
        return

    print("Scanning for UNKNOWN events...")
    
    unknown_counts = {}
    examples = {}

    for f in files:
        try:
            df = pd.read_parquet(f)
            unknowns = df[df['event_type'] == 'UNKNOWN']
            
            if not unknowns.empty:
                for _, row in unknowns.iterrows():
                    raw = row.get('raw_text', '').replace('\n', ' ')
                    if raw not in unknown_counts:
                        unknown_counts[raw] = 0
                        examples[raw] = row.to_dict()
                    unknown_counts[raw] += 1
        except Exception as e:
            print(f"Error reading {f}: {e}")

    print(f"\nFound {len(unknown_counts)} unique UNKNOWN event patterns.")
    print("-" * 60)
    
    # Sort by frequency
    sorted_unknowns = sorted(unknown_counts.items(), key=lambda x: x[1], reverse=True)
    
    for raw, count in sorted_unknowns[:20]: # Top 20
        print(f"Count: {count:<5} | Text: {raw[:80]}...")

if __name__ == "__main__":
    scan_unknowns()
