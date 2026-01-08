"""
src/data_compute/compute_linear_metrics.py
Layer 1 & 2: Calculates Linear Weights and Regression Priors.
- Game Score (GmSc)
- Win Shares (WS, OWS, DWS)
- Box Plus-Minus (BPM)
- Value Over Replacement Player (VORP)
"""

import pandas as pd
import numpy as np
import os
import sys
import glob

# Adjust path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

DATA_DIR = "data/processed"
OUTPUT_FILE = os.path.join(DATA_DIR, "metrics_linear.parquet")

def load_data():
    path = os.path.join(DATA_DIR, "player_profiles_advanced.parquet")
    if not os.path.exists(path):
        print("❌ Advanced Profiles not found. Run Stream A first.")
        return None
    return pd.read_parquet(path)

def compute_game_score(df):
    """
    Hollinger's Game Score (Season Average Approximation).
    Formula: PTS + 0.4*FG - 0.7*FGA - 0.4*(FTA-FTM) + 0.7*ORB + 0.3*DRB + STL + 0.7*AST + 0.7*BLK - 0.4*PF - TOV
    """
    # Note: df contains SEASON TOTALS. To get "Game Score", we typically calculate per game.
    # But since GmSc is linear, 'Total GmSc' / Games = 'Avg GmSc'.
    # We'll calculate Total Productivity first.
    
    # Calculate Total GmSc
    df['GMSC_TOTAL'] = (
        df['PTS'] + 
        0.4 * df['FGM'] - 
        0.7 * df['FGA'] - 
        0.4 * (df['FTA'] - df['FTM']) + 
        0.7 * df['ORB'] + 
        0.3 * df['DRB'] + 
        df['STL'] + 
        0.7 * df['AST'] + 
        0.7 * df['BLK'] - 
        0.4 * df['PF'] - 
        df['TOV']
    )
    
    # We estimate 'Games Played' using a heuristic if not present (min 1 poss per game? No, we need actual GP).
    # Since we aggregate by season, we don't strictly have GP in 'player_profiles_advanced' yet.
    # We can approximate 'Per 36' or just leave it as Total Productivity for sorting.
    # For now, we will output GMSC_TOTAL.
    return df

def compute_win_shares(df):
    """
    Simplified Win Shares Logic based on Dean Oliver.
    WS = OWS + DWS
    
    OWS ~ (Points Produced - 0.92 * League Pts/Poss * Poss) / (0.32 * League Pts/Game)
    We will use a robust approximation using our calculated ORTG/DRTG.
    """
    # 1. League Averages (Per Season)
    # We group by season to get the "League Context"
    league_context = df.groupby('season').agg(
        L_PTS=('PTS', 'sum'),
        L_POSS=('POSS_OFF', 'sum')
    ).reset_index()
    
    league_context['L_PPP'] = league_context['L_PTS'] / league_context['L_POSS']
    
    df = pd.merge(df, league_context[['season', 'L_PPP']], on='season', how='left')
    
    # 2. Marginal Offense
    # Points Produced = (ORTG / 100) * POSS_OFF
    # Marginal Offense = Points Produced - (0.92 * L_PPP * POSS_OFF)
    points_produced = (df['ORTG'] / 100) * df['POSS_OFF']
    marginal_offense = points_produced - (0.92 * df['L_PPP'] * df['POSS_OFF'])
    
    # Marginal Points per Win (approx 0.32 * L_PPG, usually ~34 points)
    # We'll use a standard constant of ~34 for simplicity, or derive it.
    pts_per_win = 34.0 
    
    df['OWS'] = np.maximum(0, marginal_offense / pts_per_win)
    
    # 3. Marginal Defense
    # Marginal Defense = (Team_DRTG_League_Avg - Player_DRTG) * %Poss * ...
    # Simplified: (Player Def Stops - 0.92 * L_PPP * Poss)
    # Alternative: Use the Rating Gap.
    # DWS = (Player Stops * D_Rating_Win_Factor) ... this is complex without play-by-play stops.
    # We will use the Rating Differential method:
    # Marginal Def Points = (L_PPP * 100 - DRTG) * (POSS_DEF / 100)
    
    league_drtg = df['L_PPP'] * 100
    marginal_def_pts = (league_drtg - df['DRTG']) * (df['POSS_DEF'] / 100)
    
    # Defense is harder, traditionally divide by ~constant
    df['DWS'] = np.maximum(0, marginal_def_pts / pts_per_win)
    
    df['WS'] = df['OWS'] + df['DWS']
    
    # Win Shares per 48 Minutes (Assuming ~100 Poss = 48 mins approx)
    # WS/48 = WS / (Poss / (League_Pace / 48))
    # Approximation: WS / (Poss / 100) * 0.1 (rough scaler)
    df['WS_48'] = df['WS'] / (df['POSS_OFF'] / 100) * 0.10 
    
    return df

def compute_bpm_vorp(df):
    """
    Calculates Box Plus-Minus (BPM) and VORP.
    Uses standard regression coefficients (BPM 1.0 style for robustness).
    
    BPM = Sc * ReMPG + Or * ORB% + Dr * DRB% + St * STL% + ...
    """
    # Standard Coefficients (approximate)
    # These prioritize efficiency and volume.
    
    # Convert stats to Per-100 Possessions
    # (Since our rates are already %, e.g., AST_PCT, we use those directly)
    
    # Interactions
    df['USG_x_TS'] = df['USG_RATE'] * df['TS_PCT']
    df['AST_x_TRB'] = df['AST_PCT'] * df['REB_PCT']
    
    # Coefficients (simplified from Daniel Myers)
    # Note: Real BPM solves a regression for every season. We use fixed weights here.
    # Weights for a "Good Player"
    
    raw_bpm = (
        0.123 * df['USG_x_TS'] +
        0.120 * df['AST_PCT'] +
        0.050 * df['REB_PCT'] +
        1.200 * (df['STL'] / df['POSS_DEF'] * 100) +  # STL% approx
        0.800 * (df['BLK'] / df['POSS_DEF'] * 100) -  # BLK% approx
        0.050 * (df['TOV'] / df['POSS_OFF'] * 100)
    )
    
    # Normalize:
    # Top players (Jokic) should be +10 to +12. Average is 0.
    # We create a Z-Score like adjustment or centering.
    
    # Let's adjust based on Team Net Rating contribution (The "Prior" logic)
    # BPM approximates Net Rating.
    # We'll use the calculated NET_RTG from stream A as a base, and regress it towards box stats.
    # Actually, let's just use our calculated NET_RTG as a "Proto-BPM" since it is 
    # literally (Team Pts On - Team Pts Off) which is what BPM *tries* to estimate!
    
    # Wait! 'player_profiles_advanced' NET_RTG is the actual On-Court Net Rating.
    # Pure On-Court Net Rating is noisy.
    # Standard BPM is a regression *to* Net Rating.
    # Since we want a "Metric", let's blend the Box Component (raw_bpm) with the actual Net Rating.
    
    # For now, let's define our BPM as a 50/50 blend of Efficiency (Box) and Impact (Net Rating)
    # This is our own "BKE-PM" (BKE Plus Minus)
    
    # Scaling raw_bpm to NBA range (-5 to +10)
    # Current raw_bpm is likely scale 0-100 or something distinct.
    # Let's use a simpler heuristic for VORP directly:
    
    # "Lineup Adjusted BPM"
    # We take the player's NET_RTG and regress it slightly to mean to remove noise.
    poss_factor = df['POSS_OFF'] / (df['POSS_OFF'] + 1000) # Shrinkage factor
    df['BPM'] = df['NET_RTG'] * poss_factor # Shrink low sample sizes to 0
    
    # VORP Calculation
    # VORP = [BPM - (-2.0)] * (% of Possessions Played) * (Team Games / 82)
    # Since we don't have total team poss easily here, we approximate:
    # VORP ~ (BPM + 2.0) * (POSS_OFF / 7000) * 82 ?? No.
    # VORP is roughly: (BPM + 2.0) * (Minutes% / 100) * 82
    # We'll use Possessions as a proxy for Minutes.
    # Avg Season Possessions ~ 8000 (Team).
    
    df['VORP'] = (df['BPM'] + 2.0) * (df['POSS_OFF'] / 8000) * 2.0 # Factor to match scale
    
    return df

def main():
    df = load_data()
    if df is None: return

    print("Computing Linear Metrics...")
    
    # 1. Game Score
    df = compute_game_score(df)
    
    # 2. Win Shares
    df = compute_win_shares(df)
    
    # 3. BPM / VORP
    df = compute_bpm_vorp(df)
    
    # Save
    cols = [
        'player_id', 'player_name', 'season', 
        'GMSC_TOTAL', 'WS', 'OWS', 'DWS', 'WS_48', 'BPM', 'VORP'
    ]
    
    final_df = df[cols].copy()
    final_df.to_parquet(OUTPUT_FILE, index=False)
    
    print(f"✅ Saved Linear Metrics to {OUTPUT_FILE}")
    
    # Validation
    print("\n--- Validation: Top 10 by VORP (2023-24) ---")
    val = final_df[final_df['season'] == '2023-24'].sort_values('VORP', ascending=False).head(10)
    print(val[['player_name', 'BPM', 'VORP', 'WS', 'GMSC_TOTAL']].to_string(index=False))

if __name__ == "__main__":
    main()