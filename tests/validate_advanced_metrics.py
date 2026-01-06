"""
tests/validate_advanced_metrics.py
Validates the integrity and realism of computed NBA metrics.
"""

import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DATA_DIR = "data/processed"

THRESHOLDS = {
    "TEAM_ORTG_MIN": 100.0,
    "TEAM_ORTG_MAX": 125.0,
    "LINEUP_NET_RTG_WARN": 50.0
}

def validate_teams():
    path = os.path.join(DATA_DIR, "metrics_teams.parquet")
    if not os.path.exists(path):
        return

    print(f"\n=== Validating Team Metrics ({os.path.basename(path)}) ===")
    df = pd.read_parquet(path)
    
    avg_ortg = df['ORTG'].mean()
    print(f"✅ Avg ORTG: {avg_ortg:.1f}")
    
    if 'team_name' in df.columns:
        unknowns = df[df['team_name'] == 'Unknown']
        if not unknowns.empty:
            print(f"⚠️  {len(unknowns)} teams have 'Unknown' name (ID Map failed).")
    
def validate_lineups():
    path = os.path.join(DATA_DIR, "metrics_lineups.parquet")
    if not os.path.exists(path):
        return

    print(f"\n=== Validating Lineup Metrics ({os.path.basename(path)}) ===")
    df = pd.read_parquet(path)
    
    high_vol = df[df['total_poss'] > 100].copy()
    print(f"Analyzed {len(high_vol)} lineups with >100 possessions.")
    
    extreme = high_vol[high_vol['NET_RTG'].abs() > THRESHOLDS['LINEUP_NET_RTG_WARN']]
    if not extreme.empty:
        print(f"⚠️  {len(extreme)} lineups have extreme Net Ratings (> +/- 50):")
        # Safe print: check cols first
        cols_to_print = ['season', 'NET_RTG', 'total_poss']
        if 'team_name' in df.columns: cols_to_print.insert(1, 'team_name')
        if 'lineup_names' in df.columns: cols_to_print.append('lineup_names')
        
        print(extreme[cols_to_print].head())
    else:
        print("✅ No extreme outliers in high-volume lineups.")

if __name__ == "__main__":
    validate_teams()
    validate_lineups()