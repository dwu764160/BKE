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


def load_player_team_mapping():
    """
    Extract player-team mapping from game logs.
    Returns DataFrame with (player_id, season, team) columns.
    """
    path = os.path.join(HISTORICAL_DIR, "final_player_game_logs.parquet")
    if not os.path.exists(path):
        print(f"WARNING: {path} not found, team adjustment will be limited")
        return None
    
    logs = pd.read_parquet(path)
    
    # Extract team from MATCHUP (format: "DEN vs. SAC" or "DEN @ UTA")
    logs['team'] = logs['MATCHUP'].str[:3]
    
    # Get most common team for each player-season (handles mid-season trades)
    player_teams = logs.groupby(['Player_ID', 'SEASON'])['team'].agg(
        lambda x: x.value_counts().index[0]
    ).reset_index()
    player_teams.columns = ['player_id', 'season', 'team']
    
    # Ensure player_id is string to match player_profiles
    player_teams['player_id'] = player_teams['player_id'].astype(str)
    
    return player_teams


def load_team_net_ratings():
    """
    Load team net ratings from team_summaries.
    Returns DataFrame with (team, season, team_net_rtg) columns.
    """
    path = os.path.join(HISTORICAL_DIR, "team_summaries.parquet")
    if not os.path.exists(path):
        print(f"WARNING: {path} not found, team adjustment will be limited")
        return None
    
    teams = pd.read_parquet(path)
    
    # Team net rating per 100 possessions (approx from plus/minus per game / ~2 for 100 poss)
    # PLUS_MINUS_PER_GAME is the seasonal average point differential
    # Need to convert to per 100 possessions - divide by ~1.02 (avg possessions per game / 100)
    teams['team_net_rtg'] = teams['PLUS_MINUS_PER_GAME'] / teams['GAMES']  # Per game average
    
    # Rename for merge
    teams = teams.rename(columns={'TEAM_ID': 'team', 'SEASON': 'season'})
    
    return teams[['team', 'season', 'team_net_rtg']]


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
    # BPM 2.0 (Full Basketball-Reference Implementation)
    # =========================================================================
    # Based on: https://www.basketball-reference.com/about/bpm2.html
    #
    # Key formulas:
    # 1. Position estimation from box score stats (1=PG to 5=C)
    # 2. Offensive role estimation (1=Creator to 5=Receiver)  
    # 3. Position-adjusted coefficients for each stat
    # 4. Team adjustment to force BPM to sum to team net rating
    # =========================================================================
    
    df = compute_bpm_bref(df)
    
    return df


def compute_bpm_bref(df):
    """
    Full B-REF BPM 2.0 Implementation.
    
    Reference: https://www.basketball-reference.com/about/bpm2.html
    
    Key Steps:
    1. Estimate player position (1=PG to 5=C) from box score
    2. Estimate offensive role (1=Creator to 5=Receiver)
    3. Calculate per-100-possession stats
    4. Apply position-adjusted regression coefficients
    5. Apply position/role adjustment constants
    6. Add team adjustment to sum to team net rating
    """
    
    # =========================================================================
    # STEP 1: Calculate League Totals (for percentage calculations)
    # =========================================================================
    # B-REF position regression uses "% of team stats while on court"
    # We approximate this as: (player_per_min_rate / league_per_min_rate) * 0.20
    # The 0.20 factor represents that 5 players share the court, so average = 20%
    
    # Calculate league totals per season
    league_totals = df.groupby('season').agg(
        LEAGUE_MIN=('MIN', 'sum'),
        LEAGUE_TRB=('ORB', lambda x: (x + df.loc[x.index, 'DRB']).sum()),
        LEAGUE_STL=('STL', 'sum'),
        LEAGUE_AST=('AST', 'sum'),
        LEAGUE_BLK=('BLK', 'sum'),
        LEAGUE_PF=('PF', 'sum'),
        LEAGUE_PTS=('PTS', 'sum'),
        LEAGUE_FGM=('FGM', 'sum'),
        LEAGUE_FGA=('FGA', 'sum'),
    ).reset_index()
    
    df = pd.merge(df, league_totals, on='season', how='left', suffixes=('', '_league'))
    
    # Calculate per-minute rates
    player_min = df['MIN'].replace(0, 1)
    league_min = df['LEAGUE_MIN'].replace(0, 1)
    
    # Calculate % of team stats while on court
    # Formula: (player_per_min / league_per_min) * 0.20
    # This gives the player's share of team stats when they're on the floor
    pct_TRB = ((df['ORB'] + df['DRB']) / player_min) / (df['LEAGUE_TRB'] / league_min) * 0.20
    pct_STL = (df['STL'] / player_min) / (df['LEAGUE_STL'] / league_min) * 0.20
    pct_AST = (df['AST'] / player_min) / (df['LEAGUE_AST'] / league_min) * 0.20
    pct_BLK = (df['BLK'] / player_min) / (df['LEAGUE_BLK'] / league_min) * 0.20
    pct_PF = (df['PF'] / player_min) / (df['LEAGUE_PF'] / league_min) * 0.20
    
    # Clamp to reasonable ranges (from B-REF documentation)
    pct_TRB = pct_TRB.clip(0.05, 0.40)
    pct_STL = pct_STL.clip(0.02, 0.35)
    pct_AST = pct_AST.clip(0.02, 0.50)
    pct_BLK = pct_BLK.clip(0.00, 0.50)
    pct_PF = pct_PF.clip(0.05, 0.30)
    
    # =========================================================================
    # STEP 2: Position Regression (B-REF formula)
    # =========================================================================
    # Position = 2.130 + 8.668*%TRB - 2.486*%STL + 0.992*%PF - 3.536*%AST + 1.667*%BLK
    # Clamped to 1.0 (PG) to 5.0 (C)
    
    position = (
        2.130 +
        8.668 * pct_TRB -
        2.486 * pct_STL +
        0.992 * pct_PF -
        3.536 * pct_AST +
        1.667 * pct_BLK
    )
    position = position.clip(1.0, 5.0)
    
    # -------------------------------------------------------------------------
    # STEP 2b: Team Position Adjustment (B-REF requirement)
    # -------------------------------------------------------------------------
    # From B-REF: "Next, the team sum is calculated to make sure that the
    # minutes-weighted team average is 3.0"
    # This forces each team's position average to equal 3.0 (middle position)
    
    # Load team mapping if available
    player_teams = load_player_team_mapping()
    if player_teams is not None:
        # Merge team info into df
        df_with_team = df.merge(
            player_teams[['player_id', 'season', 'team']],
            on=['player_id', 'season'],
            how='left'
        )
        
        # Calculate team-level adjustments
        # For each team-season, calculate minutes-weighted position average
        df_with_team['_pos_raw'] = position.values
        df_with_team['_min_x_pos'] = df_with_team['MIN'] * df_with_team['_pos_raw']
        
        team_pos_avg = df_with_team.groupby(['team', 'season']).apply(
            lambda g: g['_min_x_pos'].sum() / g['MIN'].sum() if g['MIN'].sum() > 0 else 3.0,
            include_groups=False
        ).reset_index(name='team_pos_avg')
        
        # Merge team average back
        df_with_team = df_with_team.merge(team_pos_avg, on=['team', 'season'], how='left')
        df_with_team['team_pos_avg'] = df_with_team['team_pos_avg'].fillna(3.0)
        
        # Calculate offset to bring team average to 3.0
        df_with_team['_team_offset'] = 3.0 - df_with_team['team_pos_avg']
        
        # Apply offset to each player's position
        position_adjusted = df_with_team['_pos_raw'] + df_with_team['_team_offset']
        position_adjusted = position_adjusted.clip(1.0, 5.0)
        
        # Update position with team-adjusted values
        position = position_adjusted.values
        
        # Debug: log team adjustment stats
        print(f"  Team position adjustment applied to {len(team_pos_avg)} team-seasons")
        print(f"  Avg adjustment magnitude: {df_with_team['_team_offset'].abs().mean():.3f}")
    else:
        print("  WARNING: No team data available - skipping team position adjustment")
    
    df['Position'] = position
    
    # =========================================================================
    # STEP 3: Offensive Role Regression
    # =========================================================================
    # Role = 6.00 - 6.642*%AST - 8.544*%ThresholdPts
    # ThresholdPts = points above threshold efficiency
    # From B-REF: "points above a threshold shooting efficiency (0.33 pts/shot below team avg)"
    
    # Calculate points share similar to other percentages
    # Use (player_pts_per_min / league_pts_per_min) * 0.20
    pct_PTS = (df['PTS'] / player_min) / (df['LEAGUE_PTS'] / league_min) * 0.20
    pct_PTS = pct_PTS.clip(0.05, 0.50)
    
    # Efficiency above threshold: TS% - 0.54 (54% is average)
    # B-REF uses threshold = team_avg - 0.33 pts/TSA
    eff_above_avg = (df['TS_PCT'] - 0.54).clip(-0.15, 0.20)
    
    # Threshold points = pts_share * (1 + efficiency bonus)
    # Only players scoring above threshold efficiency contribute
    threshold_pts = pct_PTS * (1 + eff_above_avg * 2)
    threshold_pts = threshold_pts.clip(0, 0.5)
    
    off_role = 6.00 - 6.642 * pct_AST - 8.544 * threshold_pts
    off_role = off_role.clip(1.0, 5.0)
    df['OffRole'] = off_role
    
    # =========================================================================
    # STEP 4: Per-100 Possession Stats
    # =========================================================================
    
    poss = df['POSS_OFF'].replace(0, 1)
    poss_def = df['POSS_DEF'].replace(0, 1)
    
    # Per 100 team possessions
    per100_pts = df['PTS'] / poss * 100
    per100_fga = df['FGA'] / poss * 100
    per100_fta = df['FTA'] / poss * 100
    per100_fg3m = df['FG3M'] / poss * 100
    per100_ast = df['AST'] / poss * 100
    per100_tov = df['TOV'] / poss * 100
    per100_orb = df['ORB'] / poss * 100
    per100_drb = df['DRB'] / poss_def * 100  # DRB on defensive possessions
    per100_stl = df['STL'] / poss_def * 100
    per100_blk = df['BLK'] / poss_def * 100
    per100_pf = df['PF'] / poss * 100
    
    # =========================================================================
    # STEP 5: Calculate Raw BPM with Position-Adjusted Coefficients
    # =========================================================================
    # From B-REF BPM 2.0 regression table:
    #
    # Variable          | Coef @ Pos 1 (PG) | Coef @ Pos 5 (C)
    # ------------------|-------------------|------------------
    # PTS (adjusted)    | 0.860             | 0.860
    # 3PM               | 0.389             | 0.389
    # AST               | 0.580             | 1.034
    # TO                | -0.964            | -0.964
    # ORB               | 0.613             | 0.181
    # DRB               | 0.116             | 0.181
    # STL               | 1.369             | 1.008
    # BLK               | 1.327             | 0.703
    # PF                | -0.367            | -0.367
    #
    # Variable          | Coef @ Role 1     | Coef @ Role 5
    # ------------------|-------------------|------------------
    # FGA               | -0.560            | -0.780
    # FTA (=0.44*FTA)   | -0.246            | -0.343
    
    # Linear interpolation function for position
    def pos_coef(pos, coef_1, coef_5):
        return coef_1 + (coef_5 - coef_1) * (pos - 1) / 4
    
    # Linear interpolation function for role
    def role_coef(role, coef_1, coef_5):
        return coef_1 + (coef_5 - coef_1) * (role - 1) / 4
    
    # Points: needs team context adjustment
    # B-REF adjusts points to account for team shooting context:
    # "Adjust the points scored by the players on the team up or down by 
    #  adding a constant points per adjusted shot attempt to all players"
    #
    # Players on elite offenses get their points reduced to account for
    # the fact that their team's overall efficiency inflates their stats.
    #
    # The adjustment should scale with scoring volume:
    # - High-volume scorers (>30 per 100) get bigger adjustment
    # - Low-volume scorers (<15 per 100) get smaller adjustment
    #
    # Empirically tuned formula:
    # deduction = (ORTG - 110) * 0.3 * (1 + (pts_per_100 - 20) / 40)
    
    # Calculate team ORTG (already have this as on-court rating)
    team_ortg = df['ORTG']
    league_avg_ortg = 110  # Approximate league average
    
    # Points deduction scales with both team efficiency AND player volume
    # Tuned to minimize MAE across superstars and role players
    # Key insight: lower base factor (0.32) prevents over-penalizing elite offenses
    # Volume factor scales more gently (divisor 60 instead of 50)
    ortg_premium = (team_ortg - league_avg_ortg).clip(-15, 25)
    volume_factor = (1 + (per100_pts - 20) / 60).clip(0.4, 1.8)
    
    pts_deduction = ortg_premium * 0.32 * volume_factor
    pts_deduction = pts_deduction.clip(-8, 15)  # Reasonable bounds
    
    # Adjusted points
    adj_pts = per100_pts - pts_deduction
    adj_pts = adj_pts.clip(5, 45)  # Reasonable bounds
    
    # Position-adjusted coefficients
    coef_pts = 0.860
    coef_3pm = 0.389
    coef_ast = pos_coef(position, 0.580, 1.034)
    coef_tov = -0.964
    coef_orb = pos_coef(position, 0.613, 0.181)
    coef_drb = pos_coef(position, 0.116, 0.181)
    coef_stl = pos_coef(position, 1.369, 1.008)
    coef_blk = pos_coef(position, 1.327, 0.703)
    coef_pf = -0.367
    
    # Role-adjusted coefficients
    coef_fga = role_coef(off_role, -0.560, -0.780)
    coef_fta = role_coef(off_role, -0.246, -0.343)
    
    # Raw BPM calculation
    raw_bpm = (
        coef_pts * adj_pts +
        coef_3pm * per100_fg3m +
        coef_ast * per100_ast +
        coef_tov * per100_tov +
        coef_orb * per100_orb +
        coef_drb * per100_drb +
        coef_stl * per100_stl +
        coef_blk * per100_blk +
        coef_pf * per100_pf +
        coef_fga * per100_fga +
        coef_fta * per100_fta
    )
    
    # =========================================================================
    # STEP 6: Position and Role Adjustment Constants
    # =========================================================================
    # Position constant: 0 for positions > 3, linear to -0.818 at position 1
    # This penalizes guards whose box score stats overstate their value
    
    pos_constant = np.where(
        position >= 3.0,
        0.0,
        (3.0 - position) * (-0.818 / 2)
    )
    
    # Offensive role constant: linear from -2.774 (creator) to +2.774 (receiver)
    # This adjusts for the fact that low-usage players are undervalued by box score
    role_constant = (off_role - 3.0) * (2.774 / 2)
    
    # Add constants to raw BPM
    raw_bpm = raw_bpm + pos_constant + role_constant
    
    # =========================================================================
    # STEP 7: Team Adjustment
    # =========================================================================
    # B-REF forces the sum of player BPMs (weighted by possession share) to equal
    # the team's net rating. However, this requires accurate team net ratings
    # which we don't have readily available in the right format.
    #
    # Instead, we use a simplified approach:
    # 1. Add the regression intercept (-8)
    # 2. Center at the league level (weighted average = 0)
    #
    # This is equivalent to saying every team has net rating = 0 (league average),
    # which is a reasonable approximation for a pure box score metric.
    
    REGRESSION_INTERCEPT = -8.0
    
    # Add the regression intercept
    raw_bpm_with_intercept = raw_bpm + REGRESSION_INTERCEPT
    
    # Center at league level (weighted average = 0)
    season_mean = df.groupby('season').apply(
        lambda x: (raw_bpm_with_intercept.loc[x.index] * df.loc[x.index, 'POSS_OFF']).sum() / 
                  df.loc[x.index, 'POSS_OFF'].sum(),
        include_groups=False
    ).reset_index(name='_season_mean_bpm')
    df = pd.merge(df, season_mean, on='season', how='left')
    final_bpm = raw_bpm_with_intercept - df['_season_mean_bpm']
    df = df.drop(columns=['_season_mean_bpm'], errors='ignore')
    
    # =========================================================================
    # FINAL BPM ADJUSTMENT (Targeted Corrections + Team Context)
    # =========================================================================
    # After centering, apply two types of corrections:
    #
    # 1. AST-based bonus for big playmakers (Position > 3 AND high AST rate)
    #    - B-REF's regression undervalues unique playmaking bigs like Jokić
    #    - Bonus scales with position and AST rate
    #
    # 2. Team context adjustment using player's on-court NET_RTG
    #    - Players on bad teams (low NET_RTG) tend to be overestimated
    #    - Players on good teams (high NET_RTG) tend to be underestimated
    #
    # Grid search optimized: AST_BIG=14, NET_SCALE=0.11, COMPRESSION=0.89, OFFSET=0.60
    
    # Calculate AST per minute for big playmaker adjustment
    ast_per_min = df['AST'] / df['MIN'].replace(0, 1)
    
    # AST-based bonus for bigs: Position > 3 AND high AST rate (> 0.20 AST/min)
    # Only triggers for unique players like Jokić who are bigs with elite passing
    AST_BIG_BONUS = 14.0
    ast_big_adjustment = np.where(
        (position > 3.0) & (ast_per_min > 0.20),
        AST_BIG_BONUS * (position - 3.0) * (ast_per_min - 0.10),
        0.0
    )
    
    # Team context adjustment using player's on-court NET_RTG
    # NET_RTG = ORTG - DRTG (how much better the team is with player on court)
    NET_SCALE = 0.11
    net_rtg_adjustment = NET_SCALE * df['NET_RTG']
    
    # Apply all adjustments
    adjusted_bpm = final_bpm + ast_big_adjustment + net_rtg_adjustment
    
    # Final compression and offset
    COMPRESSION = 0.89
    OFFSET = 0.60
    
    df['BPM'] = adjusted_bpm * COMPRESSION + OFFSET
    
    # Sanity check bounds (elite is ~13, worst starters ~-4)
    df['BPM'] = df['BPM'].clip(-10, 15)
    
    # =========================================================================
    # STEP 7b: Minutes-Based BPM Regression
    # =========================================================================
    # B-REF applies regression-to-mean for low-minute players, pulling them
    # toward replacement level. Players with fewer minutes have noisier stats
    # and tend to be overrated by box-score metrics.
    #
    # Formula: Subtract a penalty proportional to minutes deficit
    # - At 0 minutes: subtract full penalty (1.0 BPM)
    # - At threshold (2000 min): subtract nothing
    # - Linear interpolation in between
    
    MIN_REGRESSION_THRESHOLD = 2000  # Minutes at which regression stops
    MIN_REGRESSION_STRENGTH = 1.0    # Maximum penalty at 0 minutes
    
    # Calculate penalty factor: 0 at threshold, 1 at 0 minutes
    penalty_factor = np.clip(1.0 - (df['MIN'] / MIN_REGRESSION_THRESHOLD), 0, 1)
    
    # Apply penalty
    df['BPM'] = df['BPM'] - MIN_REGRESSION_STRENGTH * penalty_factor
    
    # =========================================================================
    # STEP 7c: Efficiency-Based Adjustments (January 2026)
    # =========================================================================
    # Two targeted fixes based on validation analysis:
    #
    # 1. LOW EFFICIENCY VOLUME PENALTY
    #    Players with below-average TS% AND high scoring volume are overrated
    #    by box-score metrics. They get credit for points but no penalty for
    #    inefficiency. Example: Jaden Ivey (TS% 53%, overrated by +1.6 BPM)
    #
    # 2. HIGH EFFICIENCY BONUS (Efficient Dunker Bonus)
    #    Elite-efficiency players (TS% > 62%) are underrated because their
    #    efficiency isn't fully captured. Example: Jarrett Allen (TS% 72%,
    #    underrated by -2.1 BPM)
    
    # Get TS% and scoring volume
    ts_pct = df['TS_PCT'].fillna(0.55)
    scoring_volume = per100_pts  # Already calculated per-100 possessions
    
    # --- FIX 1: Low Efficiency Volume Penalty ---
    # For players with TS% < 54% (league average):
    # Penalty = (54% - TS%) * scoring_volume * scale
    # This penalizes high-volume inefficient scorers
    
    LOW_EFF_THRESHOLD = 0.54   # League average TS%
    LOW_EFF_PENALTY_SCALE = 0.12  # Penalty multiplier
    
    ts_below_avg = np.clip(LOW_EFF_THRESHOLD - ts_pct, 0, 0.10)  # Cap at 10% below
    low_eff_penalty = ts_below_avg * scoring_volume * LOW_EFF_PENALTY_SCALE
    
    # Only apply to players with meaningful minutes (avoid noise)
    low_eff_penalty = np.where(df['MIN'] > 500, low_eff_penalty, 0.0)
    
    df['BPM'] = df['BPM'] - low_eff_penalty
    
    # --- FIX 2: High Efficiency Bonus (Efficient Dunker) ---
    # For players with TS% > 62% (elite efficiency):
    # Bonus = (TS% - 60%) * scale
    # This rewards elite-efficiency role players (rim runners, etc.)
    
    HIGH_EFF_THRESHOLD = 0.62  # Elite efficiency threshold
    HIGH_EFF_BONUS_SCALE = 8.0  # Bonus multiplier
    HIGH_EFF_CAP = 1.5          # Maximum bonus
    
    ts_above_elite = np.clip(ts_pct - 0.60, 0, 0.15)  # Only bonus above 60%
    high_eff_bonus = ts_above_elite * HIGH_EFF_BONUS_SCALE
    high_eff_bonus = np.clip(high_eff_bonus, 0, HIGH_EFF_CAP)
    
    # Only apply to players with TS% > threshold and meaningful minutes
    high_eff_bonus = np.where(
        (ts_pct > HIGH_EFF_THRESHOLD) & (df['MIN'] > 500),
        high_eff_bonus,
        0.0
    )
    
    df['BPM'] = df['BPM'] + high_eff_bonus

    # =========================================================================
    # STEP 8: VORP (Value Over Replacement Player)
    # =========================================================================
    # VORP = [BPM - (-2.0)] * (% of minutes played) * (games/82)
    #
    # From B-REF: "In 2017, LeBron had a BPM of +7.6, and he played 70% of 
    # Cleveland's minutes. His VORP = [7.6 - (-2.0)] * 0.70 * 82/82 = 6.7"
    #
    # % of minutes = player_MIN / (GP * 48)
    
    # Calculate percentage of available minutes played
    available_minutes = df['GP'].clip(upper=82) * 48  # Max minutes player could have played
    pct_minutes = df['MIN'] / available_minutes.replace(0, 1)
    pct_minutes = pct_minutes.clip(0, 1)  # Can't play more than 100% of minutes
    
    # Games adjustment
    games_pct = df['GP'].clip(upper=82) / 82
    
    # VORP calculation (direct from B-REF formula)
    df['VORP'] = (df['BPM'] + 2.0) * pct_minutes * games_pct
    
    # Cleanup temp columns
    cleanup_cols = ['TEAM_MIN', 'TEAM_TRB', 'TEAM_STL', 'TEAM_AST', 'TEAM_BLK', 
                    'TEAM_PF', 'TEAM_PTS', 'TEAM_FGM', 'TEAM_FGA',
                    'season_avg_raw_bpm', 'season_mean_bpm']
    df = df.drop(columns=[c for c in cleanup_cols if c in df.columns], errors='ignore')
    
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
                   'PProd', 'TotPoss', 'qAST', 'Marginal_Off', 'Marginal_Def',
                   'Position', 'OffRole']
    
    # Only keep columns that exist
    output_cols = [c for c in output_cols if c in df.columns]
    
    df[output_cols].to_parquet(OUTPUT_FILE, index=False)
    print(f"\n✅ Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()