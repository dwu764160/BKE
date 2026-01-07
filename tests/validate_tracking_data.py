"""
tests/validate_tracking_data.py
Validates the integrity of Stream B (API Fetched Data).
Checks for:
- File existence across all seasons
- Success of the "0 Dribbles" Proxy for 2022-23
- Data structure integrity
"""

import pandas as pd
import os
import sys

# Adjust path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DATA_DIR = "data/tracking"
SEASONS = ["2022-23", "2023-24", "2024-25"]

def validate_catch_shoot_proxy():
    print("\n=== 1. Validating Catch & Shoot Proxy (2022-23) ===")
    path = os.path.join(DATA_DIR, "2022-23", "tracking_CatchShoot.parquet")
    
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        return

    try:
        df = pd.read_parquet(path)
        print(f"Loaded {len(df)} rows.")
        
        # The proxy logic should have renamed DRIBBLE_RANGE_FGA -> CATCH_SHOOT_FGA
        # We check if the columns look "normal" (like the other seasons)
        
        target_col = "CATCH_SHOOT_FGA"
        proxy_artifact = "DRIBBLE_RANGE_FGA"
        
        if target_col in df.columns:
            print(f"✅ Columns successfully normalized: Found '{target_col}'")
        else:
            print(f"❌ Missing expected column '{target_col}'. Rename logic failed.")
            
        if proxy_artifact in df.columns:
            print(f"⚠️ Warning: raw proxy column '{proxy_artifact}' still present.")
        
        # Value Sanity Check
        # 0-dribble shots usually have very high FGP (>35%)
        if "CATCH_SHOOT_FG_PCT" in df.columns:
            avg_fgp = df["CATCH_SHOOT_FG_PCT"].mean()
            print(f"Stats Check: Avg FG% = {avg_fgp:.3f} (Expected ~0.350-0.450)")
            
    except Exception as e:
        print(f"❌ Error reading file: {e}")

def validate_coverage():
    print("\n=== 2. Validating Full Dataset Coverage ===")
    
    expected_files = [
        "tracking_Drives.parquet",
        "tracking_CatchShoot.parquet",
        "tracking_PullUpShot.parquet",
        "tracking_Passing.parquet",
        "tracking_Possessions.parquet",
        "tracking_Rebounding.parquet",
        "tracking_Efficiency.parquet",
        "tracking_SpeedDistance.parquet",
        "defense_Overall.parquet",
        "defense_LessThan6Ft.parquet", # Rim Protection
        "synergy_Offensive_Isolation.parquet",
        "synergy_Offensive_PRBallHandler.parquet",
        "synergy_Offensive_Spotup.parquet"
    ]
    
    missing_count = 0
    
    for season in SEASONS:
        season_dir = os.path.join(DATA_DIR, season)
        if not os.path.exists(season_dir):
            print(f"❌ Missing Season Directory: {season}")
            missing_count += len(expected_files)
            continue
            
        files_in_dir = set(os.listdir(season_dir))
        
        print(f"Checking {season}...", end=" ")
        season_missing = []
        for f in expected_files:
            if f not in files_in_dir:
                season_missing.append(f)
        
        if season_missing:
            print(f"❌ Missing {len(season_missing)} files: {season_missing}")
            missing_count += len(season_missing)
        else:
            print("✅ All Core Files Present")

    if missing_count == 0:
        print("\n✅ STREAM B VALIDATION PASSED")
    else:
        print(f"\n❌ STREAM B ISSUES FOUND ({missing_count} missing files)")

if __name__ == "__main__":
    validate_catch_shoot_proxy()
    validate_coverage()