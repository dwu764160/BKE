"""
src/data_compute/compute_linear_metrics.py

Full Basketball-Reference Win Shares Implementation
===================================================
Based on: https://www.basketball-reference.com/about/ws.html
          https://www.basketball-reference.com/about/ratings.html

Key Formulas (Dean Oliver / B-REF):
- OWS = Marginal_Offense / Pts_Per_Win
- DWS = Marginal_Defense / Pts_Per_Win
- Pts_Per_Win = 0.32 * League_PPG * (Team_Pace / League_Pace)

PProd uses qAST to avoid double-counting assisted baskets.
TotPoss does NOT include ORB (ORB creates possessions, doesn't use them).

Target Validation (2023-24):
- Jokić: OWS ~11.0, DWS ~6.0, Total ~17.0
- League Total WS: ~1230 (30 teams * 41 avg wins)
"""

import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

DATA_DIR = "data/processed"
HISTORICAL_DIR = "data/historical"
OUTPUT_FILE = os.path.join(DATA_DIR, "metrics_linear.parquet")


def load_player_data():
    """Load player profiles, dropping any stale computed columns."""
    path = os.path.join(DATA_DIR, "player_profiles_advanced.parquet")
    if not os.path.exists(path):
        print(f"ERROR: {path} not found")
        return None
    df = pd.read_parquet(path)
    
    # Drop ghost columns from previous runs
    cols_to_drop = ['L_PPP', 'L_PPG', 'Pts_Per_Win', 'PProd', 'TotPoss', 'Ind_Poss',
                    'OWS', 'DWS', 'WS', 'qAST', 'Marginal_Off', 'Marginal_Def']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
    return df


def load_league_context():
    """
    Load league-level stats from team_game_logs.parquet.
    This gives us correct League PPG (~114.2 for 2023-24).
    """
    path = os.path.join(HISTORICAL_DIR, "team_game_logs.parquet")
    if not os.path.exists(path):
        print(f"WARNING: {path} not found, using fallback constants")
        return None
    
    logs = pd.read_parquet(path)
    
    # Aggregate by season
    league = logs.groupby('SEASON').agg(
        LEAGUE_PTS=('PTS', 'sum'),
        LEAGUE_GAMES=('GAME_ID', 'nunique'),  # Unique games (each game has 2 rows)
        TEAM_GAMES=('PTS', 'count')  # Total team-game rows
    ).reset_index()
    
    # League PPG = Total Points / Total Team-Games
    league['L_PPG'] = league['LEAGUE_PTS'] / league['TEAM_GAMES']
    
    # Pts Per Win: 0.32 * League_PPG (pace-adjusted, but league avg pace = 1.0)
    league['Pts_Per_Win'] = 0.32 * league['L_PPG']
    
    league = league.rename(columns={'SEASON': 'season'})
    return league[['season', 'L_PPG', 'Pts_Per_Win', 'TEAM_GAMES']]


def build_team_season_totals(df):
    """
    Aggregate team-season totals from player data.
    These are needed for qAST calculation.
    
    Note: We sum player stats to get team totals. This is approximate but
    works well since we have full rosters.
    """
    # We need to identify teams - use a proxy by grouping high-minute players
    # For simplicity, we'll estimate team totals from league averages per team
    
    # Alternative: Sum all players by season to get league totals,
    # then divide by 30 teams for average team
    league_totals = df.groupby('season').agg(
        LEAGUE_FGM=('FGM', 'sum'),
        LEAGUE_FGA=('FGA', 'sum'),
        LEAGUE_FG3M=('FG3M', 'sum'),
        LEAGUE_FG3A=('FG3A', 'sum'),
        LEAGUE_FTM=('FTM', 'sum'),
        LEAGUE_FTA=('FTA', 'sum'),
        LEAGUE_AST=('AST', 'sum'),
        LEAGUE_ORB=('ORB', 'sum'),
        LEAGUE_DRB=('DRB', 'sum'),
        LEAGUE_TOV=('TOV', 'sum'),
        LEAGUE_PTS=('PTS', 'sum'),
        LEAGUE_MIN=('MIN', 'sum'),
        LEAGUE_STL=('STL', 'sum'),
        LEAGUE_BLK=('BLK', 'sum'),
        LEAGUE_PF=('PF', 'sum'),
    ).reset_index()
    
    # Per-team averages (30 teams)
    for col in league_totals.columns:
        if col.startswith('LEAGUE_'):
            league_totals[f'TEAM_AVG_{col[7:]}'] = league_totals[col] / 30
    
    return league_totals


def compute_win_shares_bref(df, league_ctx):
    """
    Full B-REF Win Shares Implementation.
    
    References:
    - https://www.basketball-reference.com/about/ws.html
    - https://www.basketball-reference.com/about/ratings.html
    """
    
    # Build team totals for qAST calculation
    team_totals = build_team_season_totals(df)
    df = pd.merge(df, team_totals, on='season', how='left')
    
    # Merge league context (correct L_PPG and Pts_Per_Win)
    if league_ctx is not None:
        df = pd.merge(df, league_ctx[['season', 'L_PPG', 'Pts_Per_Win']], on='season', how='left')
    else:
        # Fallback: use 114.2 PPG for 2023-24
        df['L_PPG'] = 114.2
        df['Pts_Per_Win'] = 0.32 * 114.2  # ~36.5
    
    # L_PPP from player data (ratio is correct even if sums are 5x)
    l_ppp_season = df.groupby('season').apply(
        lambda x: x['TEAM_PTS_ON_COURT'].sum() / x['POSS_OFF'].sum()
    ).reset_index(name='L_PPP')
    df = pd.merge(df, l_ppp_season, on='season', how='left')
    
    # =========================================================================
    # OFFENSIVE WIN SHARES
    # =========================================================================
    
    # --- Step 1: Calculate qAST (Quality of Assists) ---
    # qAST estimates what fraction of player's FGM were assisted
    # Formula: qAST = ((MP/(Team_MP/5)) * (1.14*((Team_AST-AST)/Team_FGM))) + 
    #                 ((((Team_AST/Team_MP)*MP*5 - AST) / ((Team_FGM/Team_MP)*MP*5 - FGM)) * 
    #                  (1 - (MP/(Team_MP/5))))
    
    # Use average team values as proxy
    Team_MP = df['TEAM_AVG_MIN']
    Team_AST = df['TEAM_AVG_AST']
    Team_FGM = df['TEAM_AVG_FGM']
    Team_FGA = df['TEAM_AVG_FGA']
    Team_FG3M = df['TEAM_AVG_FG3M']
    Team_PTS = df['TEAM_AVG_PTS']
    Team_FTM = df['TEAM_AVG_FTM']
    Team_ORB = df['TEAM_AVG_ORB']
    Team_TOV = df['TEAM_AVG_TOV']
    
    MP = df['MIN']
    AST = df['AST']
    FGM = df['FGM']
    FGA = df['FGA']
    FG3M = df['FG3M']
    PTS = df['PTS']
    FTM = df['FTM']
    ORB = df['ORB']
    TOV = df['TOV']
    
    # Prevent division by zero
    Team_FGM = Team_FGM.replace(0, 1)
    Team_MP = Team_MP.replace(0, 1)
    
    # Minutes fraction (how much of team's minutes player played)
    min_pct = MP / (Team_MP / 5)
    min_pct = min_pct.clip(0, 5)  # Cap at 5 (can't play more than all minutes)
    
    # qAST Part 1: Weighted by minutes
    team_ast_rate = 1.14 * ((Team_AST - AST) / Team_FGM)
    team_ast_rate = team_ast_rate.clip(0, 1.5)
    
    # qAST Part 2: Individual adjustment
    team_ast_per_min = Team_AST / Team_MP
    team_fgm_per_min = Team_FGM / Team_MP
    
    # Avoid division by zero in part 2
    denom = (team_fgm_per_min * MP * 5 - FGM).replace(0, 1)
    numer = team_ast_per_min * MP * 5 - AST
    part2_ratio = (numer / denom).clip(-1, 2)
    
    qAST = min_pct * team_ast_rate + part2_ratio * (1 - min_pct)
    qAST = qAST.clip(0, 1)  # qAST should be between 0 and 1
    df['qAST'] = qAST
    
    # --- Step 2: FG_Part of Scoring Possessions ---
    # FG_Part = FGM * (1 - 0.5 * ((PTS - FTM) / (2 * FGA)) * qAST)
    pts_per_fga = ((PTS - FTM) / (2 * FGA.replace(0, 1))).clip(0, 2)
    FG_Part = FGM * (1 - 0.5 * pts_per_fga * qAST)
    
    # --- Step 3: AST_Part of Scoring Possessions ---
    # AST_Part = 0.5 * (((Team_PTS - Team_FTM) - (PTS - FTM)) / (2 * (Team_FGA - FGA))) * AST
    team_fg_pts = Team_PTS - Team_FTM
    player_fg_pts = PTS - FTM
    team_other_fg_pts = team_fg_pts - player_fg_pts
    team_other_fga = (Team_FGA - FGA).replace(0, 1)
    
    AST_Part = 0.5 * (team_other_fg_pts / (2 * team_other_fga)) * AST
    AST_Part = AST_Part.clip(0, None)
    
    # --- Step 4: FT_Part of Scoring Possessions ---
    # FT_Part = (1 - (1 - (FTM/FTA))^2) * 0.4 * FTA
    ft_pct = (FTM / df['FTA'].replace(0, 1)).clip(0, 1)
    FT_Part = (1 - (1 - ft_pct)**2) * 0.4 * df['FTA']
    
    # --- Step 5: Team Scoring Possessions and ORB factors ---
    Team_Scoring_Poss = Team_FGM + (1 - (1 - Team_FTM / df['TEAM_AVG_FTA'].replace(0, 1))**2) * df['TEAM_AVG_FTA'] * 0.4
    
    # Team ORB% (approximate)
    Team_TRB = Team_ORB + df['TEAM_AVG_DRB']
    Opp_TRB = Team_TRB  # Approximate opponent rebounds as similar
    Team_ORB_pct = Team_ORB / (Team_ORB + (Opp_TRB - Team_ORB / 2).clip(1, None))
    Team_ORB_pct = Team_ORB_pct.clip(0.15, 0.35)  # Reasonable range
    
    # Team Play% (scoring efficiency)
    Team_Play_pct = Team_Scoring_Poss / (Team_FGA + df['TEAM_AVG_FTA'] * 0.4 + Team_TOV).replace(0, 1)
    Team_Play_pct = Team_Play_pct.clip(0.3, 0.7)
    
    # Team ORB Weight
    Team_ORB_Weight = ((1 - Team_ORB_pct) * Team_Play_pct) / \
                      (((1 - Team_ORB_pct) * Team_Play_pct + Team_ORB_pct * (1 - Team_Play_pct)).replace(0, 1))
    Team_ORB_Weight = Team_ORB_Weight.clip(0.3, 0.9)
    
    # ORB_Part
    ORB_Part = ORB * Team_ORB_Weight * Team_Play_pct
    
    # --- Step 6: Scoring Possessions ---
    ScPoss = (FG_Part + AST_Part + FT_Part) * (1 - (Team_ORB / Team_Scoring_Poss.replace(0, 1)) * 
              Team_ORB_Weight * Team_Play_pct) + ORB_Part
    ScPoss = ScPoss.clip(0, None)
    
    # --- Step 7: Missed FG and FT Possessions ---
    FGxPoss = (FGA - FGM) * (1 - 1.07 * Team_ORB_pct)
    FTxPoss = ((1 - ft_pct)**2) * 0.4 * df['FTA']
    
    # --- Step 8: Total Possessions (NO ORB!) ---
    TotPoss = ScPoss + FGxPoss + FTxPoss + TOV
    df['TotPoss'] = TotPoss
    
    # --- Step 9: Points Produced ---
    # PProd_FG_Part = 2 * (FGM + 0.5 * 3PM) * (1 - 0.5 * ((PTS-FTM)/(2*FGA)) * qAST)
    PProd_FG_Part = 2 * (FGM + 0.5 * FG3M) * (1 - 0.5 * pts_per_fga * qAST)
    
    # PProd_AST_Part
    team_fg_factor = ((Team_FGM - FGM + 0.5 * (Team_FG3M - FG3M)) / (Team_FGM - FGM).replace(0, 1)).clip(0.5, 1.5)
    PProd_AST_Part = 2 * team_fg_factor * 0.5 * (team_other_fg_pts / (2 * team_other_fga)) * AST
    PProd_AST_Part = PProd_AST_Part.clip(0, None)
    
    # PProd_ORB_Part
    Team_Pts_per_ScPoss = Team_PTS / Team_Scoring_Poss.replace(0, 1)
    Team_Pts_per_ScPoss = Team_Pts_per_ScPoss.clip(1.5, 2.5)
    PProd_ORB_Part = ORB * Team_ORB_Weight * Team_Play_pct * Team_Pts_per_ScPoss
    
    # Total PProd
    PProd = (PProd_FG_Part + PProd_AST_Part + FTM) * \
            (1 - (Team_ORB / Team_Scoring_Poss.replace(0, 1)) * Team_ORB_Weight * Team_Play_pct) + \
            PProd_ORB_Part
    df['PProd'] = PProd
    
    # --- Step 10: Marginal Offense and OWS ---
    Expected_Prod = 0.92 * df['L_PPP'] * TotPoss
    df['Marginal_Off'] = PProd - Expected_Prod
    
    df['OWS'] = (df['Marginal_Off'] / df['Pts_Per_Win']).clip(lower=None)  # Can be negative
    
    # =========================================================================
    # DEFENSIVE WIN SHARES (Full B-REF Implementation)
    # =========================================================================
    # 
    # B-REF Formula for Individual DRtg (from ratings.html):
    # 
    # 1. Calculate Stops = Stops1 + Stops2
    #    - Stops1 = STL + BLK * FMwt * (1 - 1.07*DOR%) + DRB * (1 - FMwt)
    #    - Stops2 = team credit for non-steal TOVs and non-block misses
    #    - FMwt = (DFG% * (1-DOR%)) / (DFG% * (1-DOR%) + (1-DFG%) * DOR%)
    #    - DOR% = Opp_ORB / (Opp_ORB + Team_DRB)
    #
    # 2. Calculate Stop% = (Stops * Opp_MP) / (Team_Poss * MP)
    #
    # 3. Individual DRtg = Team_DRtg + 0.2 * (100 * D_Pts_per_ScPoss * (1 - Stop%) - Team_DRtg)
    #
    # KEY INSIGHT: B-REF only moves DRtg 20% from team baseline! This is why
    # guards don't get penalized heavily for low rebounding - they still get
    # most of the team's defensive credit.
    # =========================================================================
    
    STL = df['STL']
    BLK = df['BLK']
    DRB = df['DRB']
    PF = df['PF']
    
    # Team minutes and defensive possessions (season totals)
    Team_MP_season = 5 * 48 * df['GP'].clip(upper=82)
    
    # Team defensive possessions: ~100 per game * games played
    Team_Def_Poss = 100 * df['GP'].clip(upper=82)
    
    # --- Opponent/Team Statistics (League Averages as Proxy) ---
    # DFG%: Opponent field goal percentage (league avg ~47%)
    DFG_pct = 0.47
    
    # DOR%: Opponent offensive rebound percentage
    # DOR% = Opp_ORB / (Opp_ORB + Team_DRB)
    # League avg: ~25%
    DOR_pct = 0.25
    
    # FMwt: Forced Miss Weight
    # FMwt = (DFG% * (1-DOR%)) / (DFG% * (1-DOR%) + (1-DFG%) * DOR%)
    FMwt = (DFG_pct * (1 - DOR_pct)) / (DFG_pct * (1 - DOR_pct) + (1 - DFG_pct) * DOR_pct)
    # FMwt ≈ 0.73
    
    # D_Pts_per_ScPoss: Opponent points per scoring possession (~2.0)
    D_Pts_per_ScPoss = 2.0
    
    # --- Stops1: Individual Credit ---
    # Stops1 = STL + BLK * FMwt * (1 - 1.07*DOR%) + DRB * (1 - FMwt)
    Stops1 = STL + BLK * FMwt * (1 - 1.07 * DOR_pct) + DRB * (1 - FMwt)
    
    # --- Stops2: Team Credit (per minute played) ---
    # Stops2 captures credit for opponent misses and turnovers not captured by STL/BLK
    # Stops2 = (((Opp_FGA - Opp_FGM - Team_BLK) / Team_MP) * FMwt * (1 - 1.07*DOR%)
    #          + ((Opp_TOV - Team_STL) / Team_MP)) * MP
    #          + (PF / Team_PF) * 0.4 * Opp_FTA * (1 - (Opp_FTM / Opp_FTA))^2
    
    # Per-game opponent stats (league averages): ~100 FGA, 47 FGM, 14 TOV, 22 FTA, 77% FT
    Opp_FGA_pg = 88
    Opp_FGM_pg = 41
    Opp_TOV_pg = 14
    Opp_FTA_pg = 22
    Opp_FT_pct = 0.77
    
    # Per-game team defensive stats (league averages): ~5 BLK, ~8 STL, ~36 PF
    Team_BLK_pg = 5
    Team_STL_pg = 8
    Team_PF_pg = 36
    
    # Per-minute rates (48 min game)
    missed_non_blk_rate = ((Opp_FGA_pg - Opp_FGM_pg - Team_BLK_pg) / (48 * 5))  # per player-minute
    non_stl_tov_rate = ((Opp_TOV_pg - Team_STL_pg) / (48 * 5))
    
    # Stops2 per minute
    Stops2_rate = missed_non_blk_rate * FMwt * (1 - 1.07 * DOR_pct) + non_stl_tov_rate
    # Additional credit from fouls (force missed FTs)
    PF_credit = (PF / (Team_PF_pg * df['GP'].clip(upper=82)).clip(lower=1)) * \
                0.4 * Opp_FTA_pg * df['GP'] * (1 - Opp_FT_pct) ** 2
    
    Stops2 = Stops2_rate * MP + PF_credit
    
    # --- Total Stops ---
    Stops = Stops1 + Stops2
    df['Stops'] = Stops
    
    # --- Stop% ---
    # Stop% = (Stops * Opp_MP) / (Team_Poss * MP)
    # Opp_MP = 48 * 5 * GP (opponent's total minutes)
    # Simplifies to: Stop% = Stops × Opp_MP / (Team_Poss × MP)
    #              = Stops × (5×48×GP) / (100×GP × MP)
    #              = Stops × 240 / (100 × MP)
    #              = Stops × 2.4 / MP
    # Typical values: 40-60% for starters
    
    Stop_pct = (Stops * 2.4) / MP.replace(0, 1)
    Stop_pct = Stop_pct.clip(0.30, 0.70)  # Reasonable range for NBA players
    df['Stop_pct'] = Stop_pct
    
    # --- Individual DRtg (B-REF Formula) ---
    # DRtg = Team_DRtg + 0.2 * (100 * D_Pts_per_ScPoss * (1 - Stop%) - Team_DRtg)
    #
    # Critical: B-REF uses 0.2 coefficient - individual stats only shift DRtg
    # 20% from team baseline. This prevents over-rewarding/penalizing based
    # on box score stats alone.
    
    Team_DRtg = df['DRTG']  # On-court defensive rating
    
    # "Raw" individual DRtg based purely on Stop%
    Raw_Ind_DRtg = 100 * D_Pts_per_ScPoss * (1 - Stop_pct)
    
    # B-REF blended DRtg: 80% team, 20% individual
    Ind_DRtg = Team_DRtg + 0.2 * (Raw_Ind_DRtg - Team_DRtg)
    Ind_DRtg = Ind_DRtg.clip(95, 125)  # Reasonable bounds
    df['Ind_DRtg'] = Ind_DRtg
    
    # Baseline defensive rating: 1.08 * L_PPP (replacement level)
    Baseline_DRtg = 1.08 * df['L_PPP']
    
    # Marginal Defense: points saved relative to replacement level
    # Formula: (MP / Team_MP) * Team_Def_Poss * (Baseline - Ind_DRtg/100)
    Marginal_Def = (MP / Team_MP_season) * Team_Def_Poss * (Baseline_DRtg - Ind_DRtg / 100)
    df['Marginal_Def'] = Marginal_Def
    
    # DWS = Marginal Defense / Points Per Win
    df['DWS'] = (Marginal_Def / df['Pts_Per_Win']).clip(lower=None)  # Can be negative
    
    # =========================================================================
    # TOTAL WIN SHARES
    # =========================================================================
    df['WS'] = df['OWS'] + df['DWS']
    
    # =========================================================================
    # BPM / VORP (Simplified Box Plus-Minus)
    # =========================================================================
    per_100_stl = STL / df['POSS_DEF'].replace(0, 1) * 100
    per_100_blk = BLK / df['POSS_DEF'].replace(0, 1) * 100
    per_100_tov = TOV / df['POSS_OFF'].replace(0, 1) * 100
    
    box_bpm = (
        0.12 * (df['TS_PCT'] * 100 - 54) +
        0.25 * df['AST_PCT'] +
        0.15 * df['REB_PCT'] +
        1.30 * per_100_stl +
        0.80 * per_100_blk -
        0.80 * per_100_tov
    )
    mean_bpm = (box_bpm * df['POSS_OFF']).sum() / df['POSS_OFF'].sum()
    df['BPM'] = box_bpm - mean_bpm
    df['VORP'] = (df['BPM'] + 2.0) * (df['POSS_OFF'] / 8000) * 2.7
    
    return df


def validate_results(df):
    """Print validation statistics and compare to B-REF targets."""
    print("\n" + "=" * 70)
    print("WIN SHARES VALIDATION")
    print("=" * 70)
    
    for season in df['season'].unique():
        season_df = df[df['season'] == season]
        
        print(f"\n--- {season} ---")
        print(f"  L_PPG: {season_df['L_PPG'].iloc[0]:.1f} (Target: ~114)")
        print(f"  Pts_Per_Win: {season_df['Pts_Per_Win'].iloc[0]:.1f} (Target: ~36.5)")
        print(f"  L_PPP: {season_df['L_PPP'].iloc[0]:.3f} (Target: ~1.16)")
        print(f"  Total OWS: {season_df['OWS'].sum():.1f}")
        print(f"  Total DWS: {season_df['DWS'].sum():.1f}")
        print(f"  Total WS: {season_df['WS'].sum():.1f} (Target: ~1230 for full season)")
        
        # Jokić check
        jokic = season_df[season_df['player_name'].str.contains("Jok", case=False, na=False)]
        if len(jokic) > 0:
            j = jokic.iloc[0]
            print(f"\n  Jokić:")
            print(f"    PProd: {j['PProd']:.1f}")
            print(f"    TotPoss: {j['TotPoss']:.1f}")
            print(f"    qAST: {j['qAST']:.3f}")
            print(f"    OWS: {j['OWS']:.2f} (Target: ~11.0)")
            print(f"    DWS: {j['DWS']:.2f} (Target: ~6.0)")
            print(f"    WS: {j['WS']:.2f} (Target: ~17.0)")
    
    # Top 5 by WS for latest season
    latest = df[df['season'] == df['season'].max()].nlargest(10, 'WS')
    print(f"\n--- Top 10 by WS ({df['season'].max()}) ---")
    print(latest[['player_name', 'GP', 'WS', 'OWS', 'DWS', 'PProd', 'TotPoss']].to_string(index=False))


def main():
    print("Computing Win Shares (Full B-REF Implementation)...")
    print("=" * 70)
    
    # Load data
    df = load_player_data()
    if df is None:
        return
    
    league_ctx = load_league_context()
    if league_ctx is not None:
        print(f"✓ Loaded league context from team_game_logs.parquet")
        print(f"  Seasons: {league_ctx['season'].tolist()}")
        print(f"  L_PPG: {league_ctx['L_PPG'].tolist()}")
    else:
        print("⚠ Using fallback league constants")
    
    # Add Game Score
    df['GMSC_TOTAL'] = (
        df['PTS'] + 0.4 * df['FGM'] - 0.7 * df['FGA'] -
        0.4 * (df['FTA'] - df['FTM']) + 0.7 * df['ORB'] +
        0.3 * df['DRB'] + df['STL'] + 0.7 * df['AST'] +
        0.7 * df['BLK'] - 0.4 * df['PF'] - df['TOV']
    )
    df['GMSC_AVG'] = df['GMSC_TOTAL'] / df['GP'].replace(0, 1)
    
    # Compute Win Shares
    df = compute_win_shares_bref(df, league_ctx)
    
    # Validate
    validate_results(df)
    
    # Save output
    output_cols = ['player_id', 'player_name', 'season', 'GP', 'MIN',
                   'WS', 'OWS', 'DWS', 'BPM', 'VORP', 'GMSC_AVG',
                   'PProd', 'TotPoss', 'qAST', 'Marginal_Off', 'Marginal_Def']
    
    # Only keep columns that exist
    output_cols = [c for c in output_cols if c in df.columns]
    
    df[output_cols].to_parquet(OUTPUT_FILE, index=False)
    print(f"\n✅ Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()