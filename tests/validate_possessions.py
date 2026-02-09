"""
tests/validate_possessions.py
Deep validation of possession data.
Checks ORTG, Pace consistency, and Lineup completeness.
"""

import pandas as pd
import numpy as np
import glob
import os
import sys

DATA_DIR = "data/historical"

def validate_file(filepath, is_clean=False):
    filename = os.path.basename(filepath)
    tag = "[CLEAN]" if is_clean else "[RAW]"
    print(f"\n--- Validating {filename} {tag} ---")
    
    try:
        df = pd.read_parquet(filepath)
    except Exception as e:
        print(f"❌ Read failed: {e}")
        return

    # 1. Basic Counts
    total_poss = len(df)
    total_pts = df['points'].sum()
    unique_games = df['game_id'].nunique()
    
    print(f"Total Possessions: {total_poss:,}")
    print(f"Total Points:      {total_pts:,}")
    print(f"Unique Games:      {unique_games:,}")

    # 2. Efficiency (Offensive Rating)
    # ORTG = (Points / Possessions) * 100
    # Modern NBA is usually 112 - 117
    ortg = (total_pts / total_poss) * 100
    print(f"Global ORTG:       {ortg:.1f} (Target: 112-117)")
    
    if ortg < 105 or ortg > 125:
        print("⚠️  WARNING: ORTG is suspicious. Check point calculations.")
    else:
        print("✅ Efficiency looks realistic.")

    # 3. Pace Consistency (Possessions per Team per Game)
    # We group by Game + Offense Team
    poss_per_team_game = df.groupby(['game_id', 'off_team_id']).size()
    avg_pace = poss_per_team_game.mean()
    min_pace = poss_per_team_game.min()
    max_pace = poss_per_team_game.max()
    
    print(f"Pace (Poss/Team):  Avg {avg_pace:.1f} | Min {min_pace} | Max {max_pace}")
    
    # Check for broken games (too few possessions usually means missing events)
    bad_games = poss_per_team_game[poss_per_team_game < 80]
    if len(bad_games) > 0:
        print(f"⚠️  {len(bad_games)} teams have < 80 possessions. (Potential incomplete data)")
        print(f"    Examples: {bad_games.head(3).to_dict()}")
    else:
        print("✅ No fragmented games found (>80 poss/game).")

    # 4. Lineup Integrity
    # Check if lineups are actually arrays of length 5
    # Sample a few rows to be fast, or check vectors
    
    # Convert first lineup to list if numpy array, then measure length
    # Note: parquet stores lists as numpy object arrays usually
    
    def check_len(x):
        return len(x) if isinstance(x, (list, np.ndarray)) else 0

    # Check Offense Lineups
    off_lens = df['off_lineup'].apply(check_len)
    bad_off = (off_lens != 5).sum()
    
    # Check Defense Lineups
    def_lens = df['def_lineup'].apply(check_len)
    bad_def = (def_lens != 5).sum()
    
    if bad_off == 0 and bad_def == 0:
        print("✅ Lineup Integrity: Perfect (All rows have 5v5).")
    else:
        pct_bad = (bad_off + bad_def) / (2 * len(df)) * 100
        if is_clean:
            # Clean files should have zero lineup errors
            print(f"❌ Lineup Errors in CLEAN file: {bad_off} bad offense, {bad_def} bad defense lineups.")
        else:
            # Raw files may have some lineup issues that get cleaned
            print(f"⚠️ Lineup Issues (raw data): {bad_off} offense, {bad_def} defense ({pct_bad:.2f}% of rows)")
        if bad_off > 0:
            print("   Sample bad off: ", df[off_lens != 5]['off_lineup'].iloc[0])

def validate_file_adjusted(filepath):
    """Wrapper that adjusts severity for raw vs clean files."""
    filename = os.path.basename(filepath)
    is_clean = "_clean_" in filename
    validate_file(filepath, is_clean=is_clean)

def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "possessions_*.parquet")))
    if not files:
        print("No possession files found.")
        return
        
    for f in files:
        validate_file_adjusted(f)

if __name__ == "__main__":
    main()