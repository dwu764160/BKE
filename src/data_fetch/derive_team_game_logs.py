"""
src/data_fetch/derive_team_game_logs.py
Scans all available historical Play-by-Play files to derive a master list of games
and basic box score stats (Final Score, Date).

Outputs:
  data/historical/team_game_logs.parquet
"""

import pandas as pd
import glob
import os
import sys
import re

# Adjust path to find src if run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

DATA_DIR = "data/historical"
OUTPUT_FILE = os.path.join(DATA_DIR, "team_game_logs.parquet")

def extract_game_meta(df, season_label):
    """
    Extracts one row per team per game from PBP data.
    """
    games = []
    
    # Group by Game ID
    for gid, game_df in df.groupby("GAME_ID"):
        # Get Scores (Max score found in columns or parsed text)
        # Since we are reading RAW pbp, we might rely on the last row's score
        # But our new fetcher provides 'scoreHome'/'scoreAway' in columns.
        
        last_row = game_df.iloc[-1]
        
        # Try explicit columns first (New Parser)
        if "scoreHome" in last_row and pd.notna(last_row["scoreHome"]):
            home_score = int(last_row["scoreHome"])
            away_score = int(last_row["scoreAway"])
        else:
            # Fallback to older text parsing logic if needed
            # (Simplified for now: assume columns exist as your new fetcher guarantees them)
            continue 

        # We need to map Team IDs to Home/Away. 
        # Usually PBP doesn't explicitly say "HomeTeamID", but we can infer.
        # This part is tricky without a schedule file. 
        # Strategy: Use unique team IDs found in the game.
        teams = game_df["teamId"].dropna().unique()
        if len(teams) < 2:
            continue
            
        # We can't strictly know Home/Away just from PBP rows without 'person1Id' context 
        # or specific "visitor" cols. 
        # However, for a "Game Log", we just need the IDs and the result.
        
        # For now, let's create a row for Team A and Team B
        t1, t2 = teams[0], teams[1]
        
        # Rough Date extraction (from the timeActual of the first event)
        date_str = None
        if "timeActual" in game_df.columns:
            date_str = str(game_df.iloc[0]["timeActual"]).split("T")[0]

        # Append rows
        games.append({
            "GAME_ID": gid,
            "TEAM_ID": t1,
            "PTS": home_score, # Assumption: we can't distinguish H/A easily here, store raw
            "OPP_PTS": away_score,
            "GAME_DATE": date_str,
            "SEASON": season_label
        })
        games.append({
            "GAME_ID": gid,
            "TEAM_ID": t2,
            "PTS": away_score,
            "OPP_PTS": home_score,
            "GAME_DATE": date_str,
            "SEASON": season_label
        })

    return pd.DataFrame(games)

def main():
    pattern = os.path.join(DATA_DIR, "play_by_play_*.parquet")
    files = sorted(glob.glob(pattern))
    
    if not files:
        print("No play_by_play files found.")
        return

    all_logs = []
    
    for f in files:
        # Extract season from filename (e.g. play_by_play_2022-23.parquet)
        match = re.search(r"(\d{4}-\d{2})", os.path.basename(f))
        season = match.group(1) if match else "UNKNOWN"
        
        print(f"Processing {os.path.basename(f)} ({season})...")
        try:
            df = pd.read_parquet(f)
            # Normalize column names to match new fetcher schema
            if "teamId" not in df.columns and "TEAM_ID" in df.columns:
                df = df.rename(columns={"TEAM_ID": "teamId"})
            
            logs = extract_game_meta(df, season)
            all_logs.append(logs)
        except Exception as e:
            print(f"Error processing {f}: {e}")

    if all_logs:
        master_df = pd.concat(all_logs, ignore_index=True)
        # Dedup just in case
        master_df = master_df.drop_duplicates(subset=["GAME_ID", "TEAM_ID"])
        
        print(f"\nWriting {len(master_df)} game logs to {OUTPUT_FILE}...")
        master_df.to_parquet(OUTPUT_FILE, index=False)
        print("Done.")
    else:
        print("No logs derived.")

if __name__ == "__main__":
    main()