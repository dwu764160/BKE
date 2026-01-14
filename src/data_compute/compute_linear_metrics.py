"""
src/data_compute/compute_linear_metrics.py
Layer 1: Calculates Win Shares (WS) using B-Ref / Dean Oliver Logic.
FINAL CALIBRATION v2:
- Cleans Dataframe: Drops old columns to prevent L_PPP shadowing.
- DWS Tuning: Reduces Stop Rate multiplier from 2.5 -> 1.2 (Realistic DRtg impact).
- Constants: Ensures consistent 1.16 L_PPP (116 ORtg) usage across OWS/DWS.
"""

import pandas as pd
import numpy as np
import os
import sys

# Adjust path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

DATA_DIR = "data/processed"
OUTPUT_FILE = os.path.join(DATA_DIR, "metrics_linear.parquet")

def load_data():
    path = os.path.join(DATA_DIR, "player_profiles_advanced.parquet")
    if not os.path.exists(path): return None
    df = pd.read_parquet(path)
    
    # CRITICAL: Drop potential "Ghost Columns" from previous runs
    cols_to_drop = ['L_PPP', 'L_PPG', 'Pts_Per_Win', 'PProd', 'Ind_Poss', 'OWS', 'DWS', 'WS']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
    return df

def compute_win_shares_calibrated(df):
    """
    Computes Win Shares with strictly consistent League Baselines.
    """
    # --- 1. LEAGUE CONTEXT ---
    league = df.groupby('season').agg(
        TOT_PTS=('TEAM_PTS_ON_COURT', 'sum'),
        TOT_POSS=('POSS_OFF', 'sum')
    ).reset_index()
    
    # Robust L_PPP (should be ~1.16 for 2024)
    league['L_PPP'] = league['TOT_PTS'] / league['TOT_POSS']
    
    # Pts Per Win: 0.32 * L_PPP * 100 (Standardized Pace)
    league['Pts_Per_Win'] = 0.32 * (league['L_PPP'] * 100)
    
    df = pd.merge(df, league[['season', 'L_PPP', 'Pts_Per_Win']], on='season', how='left')
    
    # --- 2. OFFENSIVE WIN SHARES ---
    
    # A. Points Produced (Credit Split)
    # Scorer: 100% FTM, 75% FG (25% tax for assists)
    scorer_pts = (df['PTS'] - df['FTM']) * 0.75 + df['FTM']
    # Passer: 50% of 2.3 pts per assist (1.15)
    passer_pts = df['AST'] * 1.15
    # ORB: 1.0 * L_PPP (Putback equity)
    orb_pts = df['ORB'] * df['L_PPP']
    
    df['PProd'] = scorer_pts + passer_pts + orb_pts
    
    # B. Possessions Used (Include ORB)
    df['Ind_Poss'] = df['FGA'] + 0.44 * df['FTA'] + df['TOV'] + df['ORB']
    
    # C. Marginal Offense
    # Expected = 0.92 * L_PPP * Ind_Poss
    expected_off = 0.92 * df['L_PPP'] * df['Ind_Poss']
    df['Marginal_Off'] = df['PProd'] - expected_off
    
    df['OWS'] = np.maximum(0, df['Marginal_Off'] / df['Pts_Per_Win'])
    
    # --- 3. DEFENSIVE WIN SHARES ---
    
    # A. Ind DRTG (Tuned)
    # Stop Rate: (STL + BLK + 0.6*DRB) / Def_Poss
    df['Stops'] = df['STL'] + df['BLK'] + 0.6 * df['DRB']
    df['Stop_Rate'] = df['Stops'] / df['POSS_DEF'] * 100
    avg_stop = df.groupby('season')['Stop_Rate'].transform('mean')
    
    # TUNING: Reduced multiplier from 2.5 to 1.2
    # This prevents stars from getting artificial <100 ratings
    df['Ind_DRtg'] = df['DRTG'] - (df['Stop_Rate'] - avg_stop) * 1.2
    
    # B. Marginal Defense
    # Baseline: 1.08 * L_PPP (Replacement Defense)
    # 2024: 1.08 * 1.16 * 100 = 125.2
    baseline_drtg = 1.08 * df['L_PPP'] * 100
    
    # Ind Def Poss (Strict 1/5th share)
    ind_def_poss = df['POSS_DEF'] / 5.0
    
    df['Marginal_Def'] = (ind_def_poss / 100) * (baseline_drtg - df['Ind_DRtg'])
    
    df['DWS'] = np.maximum(0, df['Marginal_Def'] / df['Pts_Per_Win'])
    
    # --- 4. TOTAL ---
    df['WS'] = df['OWS'] + df['DWS']
    
    # --- 5. BPM/VORP (Re-added) ---
    per_100_stl = df['STL'] / df['POSS_DEF'] * 100
    per_100_blk = df['BLK'] / df['POSS_DEF'] * 100
    per_100_tov = df['TOV'] / df['POSS_OFF'] * 100
    
    box_bpm = (
        0.12 * (df['TS_PCT']*100 - 54) + 
        0.25 * df['AST_PCT'] + 
        0.15 * df['REB_PCT'] + 
        1.30 * per_100_stl + 
        0.80 * per_100_blk - 
        0.80 * per_100_tov
    )
    mean_bpm = (box_bpm * df['POSS_OFF']).sum() / df['POSS_OFF'].sum()
    df['BPM'] = box_bpm - mean_bpm
    df['VORP'] = (df['BPM'] + 2.0) * (df['POSS_OFF'] / 8000) * 2.7

    # Trace for Validation
    try:
        jokic = df[(df['player_name'].str.contains("Jok")) & (df['season'] == '2023-24')].iloc[0]
        print("\n--- TRACE: Nikola Jokić ---")
        print(f"L_PPP: {jokic['L_PPP']:.3f} | Baseline DRtg: {baseline_drtg.mean():.1f}")
        print(f"OWS Calculation: {jokic['Marginal_Off']:.1f} / {jokic['Pts_Per_Win']:.1f} = {jokic['OWS']:.2f}")
        print(f"Ind DRTG: {jokic['Ind_DRtg']:.1f} (Team: {jokic['DRTG']:.1f})")
        print(f"DWS Calculation: {jokic['Marginal_Def']:.1f} / {jokic['Pts_Per_Win']:.1f} = {jokic['DWS']:.2f}")
        print(f"TOTAL WS: {jokic['WS']:.2f} (Target ~17.0)")
    except: pass
    
    return df

def main():
    df = load_data()
    if df is None: return

    print("Computing Linear Metrics (Clean State)...")
    
    # Add GMSC
    df['GMSC_TOTAL'] = (df['PTS'] + 0.4*df['FGM'] - 0.7*df['FGA'] - 0.4*(df['FTA']-df['FTM']) + 
                        0.7*df['ORB'] + 0.3*df['DRB'] + df['STL'] + 0.7*df['AST'] + 
                        0.7*df['BLK'] - 0.4*df['PF'] - df['TOV'])
    df['GMSC_AVG'] = df['GMSC_TOTAL'] / df['GP'].replace(0, 1)

    df = compute_win_shares_calibrated(df)
    
    cols = ['player_id', 'player_name', 'season', 'GP', 'WS', 'OWS', 'DWS', 'BPM', 'VORP', 'GMSC_AVG']
    df[cols].to_parquet(OUTPUT_FILE, index=False)
    
    print(f"✅ Saved to {OUTPUT_FILE}")
    print(df[df['season'] == '2023-24'].sort_values('WS', ascending=False).head(5)[cols].to_string(index=False))

if __name__ == "__main__":
    main()