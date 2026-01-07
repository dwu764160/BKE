"""
src/data_compute/compute_player_profiles.py
Stream A: Computes "Box Score Plus" and "Four Factors" profiles.
FIXED: Calculates USG% using 'Team Plays' (not Possessions) to match B-Ref standards.
"""

import pandas as pd
import numpy as np
import glob
import os
import sys

# Adjust path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

DATA_DIR = "data/historical"
OUTPUT_DIR = "data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_id(val):
    if pd.isna(val): return "0"
    return str(val).replace(".0", "")

def get_season_from_path(path):
    base = os.path.basename(path)
    return base.replace("possessions_clean_", "").replace("pbp_with_lineups_", "").replace(".parquet", "")

def compute_denominators(season):
    """
    Loads CLEAN POSSESSIONS to calculate exact Off/Def possessions.
    Used for Ratings (ORTG/DRTG), but NOT for Usage Rate.
    """
    path = os.path.join(DATA_DIR, f"possessions_clean_{season}.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()

    print(f"   Loading Possessions for {season}...")
    df = pd.read_parquet(path)
    
    # Explode Offense -> Count Possessions
    off_exploded = df.explode('off_lineup')
    off_stats = off_exploded.groupby('off_lineup').agg(
        POSS_OFF=('game_id', 'count'),
        TEAM_PTS_ON_COURT=('points', 'sum')
    ).reset_index().rename(columns={'off_lineup': 'player_id'})
    
    # Explode Defense -> Count Possessions
    def_exploded = df.explode('def_lineup')
    def_stats = def_exploded.groupby('def_lineup').agg(
        POSS_DEF=('game_id', 'count'),
        TEAM_PTS_ALLOWED=('points', 'sum')
    ).reset_index().rename(columns={'def_lineup': 'player_id'})
    
    denom_df = pd.merge(off_stats, def_stats, on='player_id', how='outer').fillna(0)
    denom_df['player_id'] = denom_df['player_id'].apply(clean_id)
    return denom_df

def compute_numerators_and_plays(season):
    """
    Loads PBP EVENTS to calculate:
    1. Individual Stats (PTS, AST, etc)
    2. Team Plays on Court (for USG%)
    3. Team FGM on Court (for AST%)
    4. Total Rebound Chances on Court (for REB%)
    """
    path = os.path.join(DATA_DIR, f"pbp_with_lineups_{season}.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
        
    print(f"   Loading PBP Events for {season}...")
    df = pd.read_parquet(path)
    
    df['player1_id'] = df['player1_id'].apply(clean_id)
    df['player2_id'] = df['player2_id'].apply(clean_id)

    # --- 1. Identify "Usage Events" (FGA, FTA, TOV) ---
    is_fga = df['event_type'].str.contains('FIELD_GOAL')
    is_fta = df['event_type'] == 'FREE_THROW'
    is_tov = df['event_type'] == 'TURNOVER'
    
    # Calculate "Play Weight":
    # FGA = 1, TOV = 1, FTA = 0.44
    df['play_weight'] = 0.0
    df.loc[is_fga | is_tov, 'play_weight'] = 1.0
    df.loc[is_fta, 'play_weight'] = 0.44

    # --- 2. Calculate TEAM PLAYS (For USG%) ---
    usage_events = df[df['play_weight'] > 0].copy()
    team_plays_list = []
    
    for team_id, group in usage_events.groupby('team_id'):
        if team_id == 0: continue
        col_name = f"lineup_{int(team_id)}"
        if col_name in group.columns:
            exploded = group.explode(col_name)
            sums = exploded.groupby(col_name)['play_weight'].sum()
            team_plays_list.append(sums)
            
    if team_plays_list:
        team_plays = pd.concat(team_plays_list).groupby(level=0).sum().rename('TEAM_PLAYS_ON_COURT')
    else:
        team_plays = pd.Series(name='TEAM_PLAYS_ON_COURT')

    # --- 3. Calculate TEAM FGM ON COURT (For AST%) ---
    made_shots = df[df['event_type'].str.contains('FIELD_GOAL') & (df['is_made'] == True)].copy()
    team_fgm_list = []
    
    for team_id, group in made_shots.groupby('team_id'):
        if team_id == 0: continue
        col_name = f"lineup_{int(team_id)}"
        if col_name in group.columns:
            exploded = group.explode(col_name)
            counts = exploded.groupby(col_name).size()
            team_fgm_list.append(counts)
            
    if team_fgm_list:
        team_fgm = pd.concat(team_fgm_list).groupby(level=0).sum().rename('TEAM_FGM_ON_COURT')
    else:
        team_fgm = pd.Series(name='TEAM_FGM_ON_COURT')

    # --- 4. Calculate TOTAL REBOUND CHANCES (For REB%) ---
    rebs = df[df['event_type'] == 'REBOUND'].copy()
    total_reb_list = []
    lineup_cols = [c for c in df.columns if c.startswith('lineup_')]
    
    for col in lineup_cols:
        exploded = rebs.explode(col)
        counts = exploded.groupby(col).size()
        total_reb_list.append(counts)

    if total_reb_list:
        total_rebs = pd.concat(total_reb_list).groupby(level=0).sum().rename('TOTAL_REB_ON_COURT')
    else:
        total_rebs = pd.Series(name='TOTAL_REB_ON_COURT')

    # --- 5. Individual Numerators (Standard) ---
    makes = df[df['event_type'].str.contains('FIELD_GOAL') & (df['is_made'] == True)]
    
    p1_stats = df.groupby('player1_id').agg(
        PTS=('points', 'sum'),
        TOV=('event_type', lambda x: (x == 'TURNOVER').sum())
    )
    
    fgm = makes.groupby('player1_id').size().rename('FGM')
    fga = df[is_fga].groupby('player1_id').size().rename('FGA')
    fg3m = df[(df['event_type'] == 'FIELD_GOAL_3PT') & (df['is_made'] == True)].groupby('player1_id').size().rename('FG3M')
    fg3a = df[df['event_type'] == 'FIELD_GOAL_3PT'].groupby('player1_id').size().rename('FG3A')
    ftm = df[(is_fta) & (df['is_made'] == True)].groupby('player1_id').size().rename('FTM')
    fta = df[is_fta].groupby('player1_id').size().rename('FTA')
    reb = df[df['event_type'] == 'REBOUND'].groupby('player1_id').size().rename('REB')
    asts = makes.groupby('player2_id').size().rename('AST')
    
    nums = pd.concat([p1_stats, fgm, fga, fg3m, fg3a, ftm, fta, reb, asts, team_plays, team_fgm, total_rebs], axis=1).fillna(0)
    nums.index.name = 'player_id'
    
    return nums.reset_index()

def process_season(season):
    print(f"\nProcessing Season: {season}")
    
    denoms = compute_denominators(season)
    if denoms.empty: return pd.DataFrame()
    
    nums = compute_numerators_and_plays(season)
    if nums.empty: return pd.DataFrame()
    
    df = pd.merge(denoms, nums, on='player_id', how='left').fillna(0)
    df['season'] = season
    
    # --- Derived Metrics ---
    
    # Efficiency
    df['TS_PCT'] = df['PTS'] / (2 * (df['FGA'] + 0.44 * df['FTA']))
    df['EFG_PCT'] = (df['FGM'] + 0.5 * df['FG3M']) / df['FGA'].replace(0, 1)
    df['FT_RATE'] = df['FTA'] / df['FGA'].replace(0, 1)
    
    # Usage Rate (FIXED)
    # Formula: (Player Plays) / (Team Plays On Court)
    player_plays = df['FGA'] + 0.44 * df['FTA'] + df['TOV']
    
    # Fallback: If TEAM_PLAYS_ON_COURT is 0 (missing lineup data), use POSS_OFF as approximate
    denom_usg = df['TEAM_PLAYS_ON_COURT'].replace(0, np.nan).fillna(df['POSS_OFF'])
    
    df['USG_RATE'] = (player_plays / denom_usg) * 100
    
    # Advanced: Assist Percentage (AST%)
    # Formula: 100 * AST / (Team FGM - Player FGM)
    teammate_fgm = df['TEAM_FGM_ON_COURT'] - df['FGM']
    df['AST_PCT'] = (df['AST'] / teammate_fgm.replace(0, 1)) * 100
    
    # Advanced: Rebound Percentage (REB% / TRB%)
    # Formula: 100 * REB / (Total Rebounds on Court)
    df['REB_PCT'] = (df['REB'] / df['TOTAL_REB_ON_COURT'].replace(0, 1)) * 100
    df['TOV_PCT'] = df['TOV'] / player_plays.replace(0, 1)
    
    # Ratings
    df['ORTG'] = (df['TEAM_PTS_ON_COURT'] / df['POSS_OFF'].replace(0, 1)) * 100
    df['DRTG'] = (df['TEAM_PTS_ALLOWED'] / df['POSS_DEF'].replace(0, 1)) * 100
    df['NET_RTG'] = df['ORTG'] - df['DRTG']
    
    return df

def enrich_names(df):
    try:
        p_path = os.path.join(DATA_DIR, "players.parquet")
        if os.path.exists(p_path):
            meta = pd.read_parquet(p_path)
            if 'id' in meta.columns:
                meta['id'] = meta['id'].astype(str).apply(clean_id)
                name_map = meta.set_index('id')['full_name'].to_dict()
                df['player_name'] = df['player_id'].map(name_map).fillna("Unknown")
    except:
        df['player_name'] = df['player_id']
    return df

def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "possessions_clean_*.parquet")))
    seasons = [get_season_from_path(f) for f in files]
    
    all_seasons = []
    for s in seasons:
        s_df = process_season(s)
        if not s_df.empty:
            all_seasons.append(s_df)
            
    if all_seasons:
        final_df = pd.concat(all_seasons, ignore_index=True)
        final_df = enrich_names(final_df)
        
        out_path = os.path.join(OUTPUT_DIR, "player_profiles_advanced.parquet")
        final_df.to_parquet(out_path, index=False)
        print(f"\n✅ Saved Advanced Profiles to {out_path}")
        
        # Verify Fix
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print("\n--- Corrected Usage Rates (Top 5) ---")
        cols = ['player_name', 'season', 'USG_RATE', 'TS_PCT', 'EFG_PCT', 'FT_RATE', 
                'AST_PCT', 'TOV_PCT', 'REB_PCT', 'ORTG', 'DRTG', 'NET_RTG']
        print(final_df[final_df['POSS_OFF'] > 1000].sort_values('USG_RATE', ascending=False)[cols].head(5))

if __name__ == "__main__":
    main()