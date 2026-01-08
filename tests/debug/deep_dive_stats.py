"""
tests/debug/deep_dive_stats.py
Deep dive into specific discrepancies for Giannis (TOV) and Luka (PF). (case studies)
"""

import pandas as pd
import os
import sys

# Adjust path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

DATA_FILE = "data/historical/pbp_with_lineups_2023-24.parquet"
GIANNIS_ID = "203507"
LUKA_ID = "1629029"

def inspect():
    if not os.path.exists(DATA_FILE): return
    print(f"Loading {DATA_FILE}...")
    df = pd.read_parquet(DATA_FILE)
    
    # Clean IDs
    df['player1_id'] = df['player1_id'].astype(str).str.replace('.0','')
    has_msg_type = 'event_msg_type' in df.columns
    if has_msg_type:
        df['event_msg_type'] = df['event_msg_type'].fillna(0).astype(int)
    
    # --- CASE 1: GIANNIS TURNOVERS ---
    print("\n=== CASE 1: GIANNIS (TOV vs OFFENSIVE FOULS) ===")
    g_mask = df['player1_id'] == GIANNIS_ID
    
    # Get all Turnovers (Type 5) — fall back to event_type if numeric msg type missing
    if has_msg_type:
        tovs = df[g_mask & (df['event_msg_type'] == 5)].copy()
    else:
        tovs = df[g_mask & (df['event_type'] == 'TURNOVER')].copy()
    
    # Get all Offensive Fouls (Type 6 + Text Filter)
    # Note: Action Type 26 = Offensive, 4 = Charge
    # Fouls (Type 6) — fall back to event_type == 'FOUL'
    if has_msg_type:
        fouls = df[g_mask & (df['event_msg_type'] == 6)].copy()
    else:
        fouls = df[g_mask & (df['event_type'] == 'FOUL')].copy()
    off_fouls = fouls[fouls['event_text'].str.contains("OFFENSIVE|CHARGE", case=False, na=False)]
    
    print(f"Total Type 5 Turnovers: {len(tovs)}")
    print(f"Total Offensive Fouls:  {len(off_fouls)}")
    
    # Check Overlap (Same Game, Period, Clock)
    # Create merge keys using .assign() to avoid SettingWithCopyWarning
    tovs = tovs.assign(merge_key = tovs['game_id'].astype(str) + "_" + tovs['period'].astype(str) + "_" + tovs['clock'].astype(str))
    off_fouls = off_fouls.assign(merge_key = off_fouls['game_id'].astype(str) + "_" + off_fouls['period'].astype(str) + "_" + off_fouls['clock'].astype(str))
    
    overlap = off_fouls[off_fouls['merge_key'].isin(tovs['merge_key'])]
    solo_fouls = off_fouls[~off_fouls['merge_key'].isin(tovs['merge_key'])]
    
    print(f"Offensive Fouls WITH Matching Turnover:    {len(overlap)}")
    print(f"Offensive Fouls WITHOUT Matching Turnover: {len(solo_fouls)} (Potentially Missing TOVs)")
    
    if len(solo_fouls) > 0:
        print("\nSample 'Solo' Offensive Fouls (No TOV event):")
        print(solo_fouls[['game_id', 'clock', 'event_text']].head(5).to_string(index=False))

    # --- CASE 2: LUKA FOULS ---
    print("\n=== CASE 2: LUKA (FOULS vs REPLAYS) ===")
    l_mask = df['player1_id'] == LUKA_ID
    
    # Get all Fouls (Type 6) — fall back to event_type == 'FOUL'
    if has_msg_type:
        luka_fouls = df[l_mask & (df['event_msg_type'] == 6)].copy()
    else:
        luka_fouls = df[l_mask & (df['event_type'] == 'FOUL')].copy()
    
    # Filter Technicals
    luka_pers = luka_fouls[~luka_fouls['event_text'].str.contains("TECHNICAL|DEFENSIVE 3", case=False, na=False)]
    
    print(f"Total Personal Fouls (Filtered): {len(luka_pers)}")
    
    # Check for Instant Replay in the SAME GAME/PERIOD nearby
    # We look for Event Type 18 (Instant Replay) if present, else search event_type/event_text for replay keywords
    if has_msg_type:
        replays = df[df['event_msg_type'] == 18]
    else:
        replays = df[df['event_type'].astype(str).str.contains('REPLAY|INSTANT', case=False, na=False) | df['event_text'].astype(str).str.contains('REPLAY|INSTANT', case=False, na=False)]
    
    print(f"Total Instant Replays in Luka Games: {len(replays[replays['game_id'].isin(luka_fouls['game_id'].unique())])}")
    
    # Logic: For each foul, is there a Replay event within 60 seconds?
    # This is a heuristic check
    overturn_suspects = 0
    for _, foul in luka_pers.iterrows():
        gid = foul['game_id']
        period = foul['period']
        # Convert clock string "MM:SS" to seconds
        try:
            m, s = map(int, foul['clock'].split(':'))
            foul_sec = m * 60 + s
        except: continue
            
        # Find replays in same game/period
        game_replays = replays[(replays['game_id'] == gid) & (replays['period'] == period)]
        
        for _, rep in game_replays.iterrows():
            try:
                rm, rs = map(int, rep['clock'].split(':'))
                rep_sec = rm * 60 + rs
                # Replay usually happens AFTER foul (so lower clock time), within 60s
                if 0 <= (foul_sec - rep_sec) < 120:
                    if "OVERTURN" in str(rep['event_text']).upper():
                        overturn_suspects += 1
                        print(f"   Possible Overturn: Game {gid} | Foul @ {foul['clock']} | Replay @ {rep['clock']}")
                        break
            except: continue
            
    print(f"Potential Overturned Fouls found: {overturn_suspects}")

if __name__ == "__main__":
    inspect()