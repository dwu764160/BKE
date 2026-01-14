"""
tests/debug/audit_league_constants.py
Diagnoses why League PPP is 1.02 instead of 1.15.
"""
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
DATA_DIR = "data/processed"

def inspect():
    path = os.path.join(DATA_DIR, "player_profiles_advanced.parquet")
    if not os.path.exists(path): return
    df = pd.read_parquet(path)
    
    # Filter 2023-24
    df = df[df['season'] == '2023-24'].copy()
    
    # Method A: Sum of Player Stats (What we were doing)
    sum_pts = df['PTS'].sum()
    sum_poss = (df['FGA'] + 0.44 * df['FTA'] + df['TOV']).sum()
    print(f"\n--- Method A: Sum of Player Stats ---")
    print(f"Total Player PTS:  {sum_pts:,.0f}")
    print(f"Total Player POSS: {sum_poss:,.0f}")
    print(f"Calculated PPP:    {sum_pts / sum_poss:.3f} (The 1.02 Bug)")
    
    # Method B: Sum of Team Context (The Fix)
    # TEAM_PTS_ON_COURT sums the score for all 5 players.
    # POSS_OFF sums the possessions for all 5 players.
    # The ratio should be the true League PPP.
    team_pts = df['TEAM_PTS_ON_COURT'].sum()
    team_poss = df['POSS_OFF'].sum()
    print(f"\n--- Method B: Sum of Team Context ---")
    print(f"Total Team PTS (Agg):  {team_pts:,.0f}")
    print(f"Total Team POSS (Agg): {team_poss:,.0f}")
    print(f"Calculated PPP:        {team_pts / team_poss:.3f} (Target: ~1.15)")
    
    # Check if Team Context columns exist and are non-zero
    if team_pts == 0:
        print("⚠️ TEAM_PTS_ON_COURT is all zeros! Check Stream A.")

if __name__ == "__main__":
    inspect()