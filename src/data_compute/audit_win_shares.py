"""
src/data_compute/audit_win_shares.py
Deep Debugger for Win Shares.
1. Calculates Total League WS (Target: ~1230 for 82-game season x 30 teams).
2. Breaks down Jokic's PProd and DRtg into atomic components.
"""

import pandas as pd
import numpy as np
import os
import sys

# Adjust path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

DATA_DIR = "data/processed"

def load_data():
    path = os.path.join(DATA_DIR, "player_profiles_advanced.parquet")
    if not os.path.exists(path): return None
    return pd.read_parquet(path)

def audit_win_shares(df):
    # Filter for 2023-24 only for clean math
    df = df[df['season'] == '2023-24'].copy()
    
    # --- 1. GLOBAL CONSTANTS ---
    # We calculate these strictly from the data provided to check internal consistency
    total_pts = df['TEAM_PTS_ON_COURT'].sum() # Sum of 5-man units
    total_poss = df['POSS_OFF'].sum()         # Sum of 5-man units
    total_gp = df['GP'].sum()
    
    L_PPP = total_pts / total_poss
    
    # Estimate League Pace (Poss per Team-Game)
    # Total Team Games approx = Total Player Games / 5
    est_team_games = total_gp / 5.0
    L_PACE = (total_poss / 5.0) / est_team_games
    
    # Marginal Points Per Win
    # Formula: 0.32 * L_PPG
    L_PPG = L_PPP * L_PACE
    PTS_PER_WIN = 0.32 * L_PPG
    
    print("\n--- GLOBAL CONSTANTS AUDIT ---")
    print(f"Total Player Games: {total_gp}")
    print(f"Est Team Games:     {est_team_games:.1f} (Target: 1230)")
    print(f"League PPP:         {L_PPP:.4f}")
    print(f"League Pace:        {L_PACE:.2f}")
    print(f"League PPG:         {L_PPG:.2f}")
    print(f"Pts Per Win:        {PTS_PER_WIN:.2f} (Target: ~34-37)")
    
    # --- 2. OFFENSIVE COMPONENT ---
    # Recalculate Ind_Poss
    df['Ind_Poss'] = df['FGA'] + 0.44 * df['FTA'] + df['TOV']
    
    # Recalculate PProd (B-Ref Logic)
    # Scorer Credit: PTS * (1 - 0.5 * qAST) + FTM (unassisted) -> Simplified to 0.75 FG factor
    scorer_pts = (df['PTS'] - df['FTM']) * 0.75 + df['FTM']
    passer_pts = df['AST'] * 2.3 * 0.5
    orb_pts = df['ORB'] * L_PPP
    
    df['PProd'] = scorer_pts + passer_pts + orb_pts
    
    # Marginal Offense
    expected_off = 0.92 * L_PPP * df['Ind_Poss']
    df['Marginal_Off'] = df['PProd'] - expected_off
    df['OWS'] = df['Marginal_Off'] / PTS_PER_WIN
    
    # --- 3. DEFENSIVE COMPONENT ---
    # Stops
    df['Stops'] = df['STL'] + df['BLK'] + 0.6 * df['DRB']
    df['Stop_Rate'] = df['Stops'] / df['POSS_DEF'] * 100
    avg_stop_rate = df['Stop_Rate'].mean()
    
    # Ind DRTG
    # 2.0 multiplier for Stop Rate diff
    df['Ind_DRtg'] = df['DRTG'] - (df['Stop_Rate'] - avg_stop_rate) * 2.0
    
    # Marginal Defense
    baseline_drtg = 1.08 * L_PPP * 100
    ind_def_poss = df['POSS_DEF'] / 5.0
    
    df['Marginal_Def'] = (ind_def_poss / 100) * (baseline_drtg - df['Ind_DRtg'])
    df['DWS'] = df['Marginal_Def'] / PTS_PER_WIN
    
    df['WS'] = df['OWS'] + df['DWS']
    
    # --- 4. AUDIT RESULTS ---
    print("\n--- WIN SHARES SUM CHECK ---")
    print(f"Total OWS Sum: {df['OWS'].sum():.1f}")
    print(f"Total DWS Sum: {df['DWS'].sum():.1f}")
    print(f"Total WS Sum:  {df['WS'].sum():.1f}")
    print(f"Target WS:     {est_team_games:.1f}")
    print(f"Inflation Factor: {df['WS'].sum() / est_team_games:.2f}x")
    
    # --- 5. JOKIC BREAKDOWN ---
    try:
        jokic = df[df['player_name'].str.contains("Jok")].iloc[0]
        print("\n--- JOKIC BREAKDOWN ---")
        print(f"PTS: {jokic['PTS']} | Scorer Credit: {scorer_pts[jokic.name]:.1f}")
        print(f"AST: {jokic['AST']} | Passer Credit: {passer_pts[jokic.name]:.1f}")
        print(f"ORB: {jokic['ORB']} | ORB Credit:    {orb_pts[jokic.name]:.1f}")
        print(f"TOTAL PProd: {jokic['PProd']:.1f}")
        print(f"Expected Prod: {expected_off[jokic.name]:.1f}")
        print(f"Marginal Off: {jokic['Marginal_Off']:.1f} -> OWS: {jokic['OWS']:.2f}")
        print("-" * 20)
        print(f"Team DRTG: {jokic['DRTG']:.1f}")
        print(f"Stop Rate: {jokic['Stop_Rate']:.2f}% (Avg: {avg_stop_rate:.2f}%)")
        print(f"Ind DRTG: {jokic['Ind_DRtg']:.1f} (Baseline: {baseline_drtg:.1f})")
        print(f"Marginal Def: {jokic['Marginal_Def']:.1f} -> DWS: {jokic['DWS']:.2f}")
    except: pass

if __name__ == "__main__":
    df = load_data()
    if df is None or (hasattr(df, 'empty') and df.empty):
        print("No data loaded or DataFrame is empty.")
    else:
        audit_win_shares(df)