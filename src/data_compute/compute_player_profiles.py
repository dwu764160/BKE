"""
src/data_compute/compute_player_profiles.py
Stream A: Computes "Box Score Plus" and "Four Factors" profiles.
UPDATED: 
- Reverted STL/BLK logic to use explicit event types (Player 1).
- Kept OREB Context Fix.
- Kept Foul/Turnover Fixes.
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
    if pd.isna(val) or val == "": return "0"
    return str(int(float(val)))

def get_season_from_path(path):
    base = os.path.basename(path)
    return base.replace("possessions_clean_", "").replace("pbp_with_lineups_", "").replace(".parquet", "")

def compute_denominators(season):
    path = os.path.join(DATA_DIR, f"possessions_clean_{season}.parquet")
    if not os.path.exists(path): return pd.DataFrame()

    print(f"   Loading Possessions for {season}...")
    df = pd.read_parquet(path)
    
    off_exploded = df.explode('off_lineup')
    off_stats = off_exploded.groupby('off_lineup').agg(
        POSS_OFF=('game_id', 'count'),
        TEAM_PTS_ON_COURT=('points', 'sum')
    ).reset_index().rename(columns={'off_lineup': 'player_id'})
    
    def_exploded = df.explode('def_lineup')
    def_stats = def_exploded.groupby('def_lineup').agg(
        POSS_DEF=('game_id', 'count'),
        TEAM_PTS_ALLOWED=('points', 'sum')
    ).reset_index().rename(columns={'def_lineup': 'player_id'})
    
    denom_df = pd.merge(off_stats, def_stats, on='player_id', how='outer').fillna(0)
    denom_df['player_id'] = denom_df['player_id'].apply(clean_id)
    return denom_df

def compute_numerators_and_plays(season):
    path = os.path.join(DATA_DIR, f"pbp_with_lineups_{season}.parquet")
    if not os.path.exists(path): return pd.DataFrame()
        
    print(f"   Loading PBP Events for {season}...")
    df = pd.read_parquet(path)
    
    # --- 1. Clean IDs & Context ---
    df['player1_id'] = df['player1_id'].apply(clean_id)
    df['player2_id'] = df['player2_id'].apply(clean_id)
    df['team_id'] = df['team_id'].fillna(0).astype(int)
    
    if 'event_text' not in df.columns: df['event_text'] = ""
    df['event_text'] = df['event_text'].fillna("").str.upper()
    
    # OREB Context: Forward Fill Shooting Team
    df['is_shot'] = df['event_type'].str.contains('FIELD_GOAL', na=False) | (df['event_type'] == 'FREE_THROW')
    df['shooting_team'] = np.where(df['is_shot'], df['team_id'], np.nan)
    df['prev_shooting_team'] = df['shooting_team'].ffill().shift(1).fillna(0).astype(int)

    # --- 2. Identify Usage Events ---
    is_fga = df['event_type'].str.contains('FIELD_GOAL', na=False)
    is_fta = df['event_type'] == 'FREE_THROW'
    is_tov = df['event_type'] == 'TURNOVER'
    
    df['play_weight'] = 0.0
    df.loc[is_fga | is_tov, 'play_weight'] = 1.0
    df.loc[is_fta, 'play_weight'] = 0.44

    # --- 3. Team Aggregates ---
    usage_events = df[df['play_weight'] > 0].copy()
    team_plays_list = []
    
    for team_id, group in usage_events.groupby('team_id'):
        if team_id == 0: continue
        col_name = f"lineup_{int(team_id)}"
        if col_name in group.columns:
            exploded = group.explode(col_name)
            sums = exploded.groupby(col_name)['play_weight'].sum()
            team_plays_list.append(sums)
            
    team_plays = pd.concat(team_plays_list).groupby(level=0).sum().rename('TEAM_PLAYS_ON_COURT') if team_plays_list else pd.Series(name='TEAM_PLAYS_ON_COURT')

    # Team FGM
    made_shots = df[(df['event_type'].str.contains('FIELD_GOAL', na=False)) & (df['is_made'] == True)].copy()
    team_fgm_list = []
    for team_id, group in made_shots.groupby('team_id'):
        if team_id == 0: continue
        col_name = f"lineup_{int(team_id)}"
        if col_name in group.columns:
            exploded = group.explode(col_name)
            counts = exploded.groupby(col_name).size()
            team_fgm_list.append(counts)
    team_fgm = pd.concat(team_fgm_list).groupby(level=0).sum().rename('TEAM_FGM_ON_COURT') if team_fgm_list else pd.Series(name='TEAM_FGM_ON_COURT')

    # Rebound Chances
    valid_rebs = df[(df['event_type'] == 'REBOUND') & (df['player1_id'] != "0")].copy()
    total_reb_list = []
    lineup_cols = [c for c in df.columns if c.startswith('lineup_')]
    for col in lineup_cols:
        if col in valid_rebs.columns:
            exploded = valid_rebs.explode(col)
            counts = exploded[exploded[col].notna()].groupby(col).size()
            total_reb_list.append(counts)
    total_rebs = pd.concat(total_reb_list).groupby(level=0).sum().rename('TOTAL_REB_ON_COURT') if total_reb_list else pd.Series(name='TOTAL_REB_ON_COURT')

    # --- 5. INDIVIDUAL STATS ---
    
    # A. Complex Filters (Fouls)
    # Exclude Technicals and Defensive 3 Seconds from PF
    fouls = df[df['event_type'] == 'FOUL'].copy()
    is_tech = fouls['event_text'].str.contains("TECHNICAL", na=False)
    is_def3 = fouls['event_text'].str.contains("DEFENSIVE 3", na=False)
    valid_fouls = fouls[~(is_tech | is_def3)]
    
    # B. Aggregation (PTS, STL, BLK, TOV, PF)
    # STL and BLK are explicit event types in this dataset, attributed to Player 1
    p1_stats = df.groupby('player1_id').agg(
        PTS=('points', 'sum'),
        TOV=('event_type', lambda x: (x == 'TURNOVER').sum()),
        STL=('event_type', lambda x: (x == 'STEAL').sum()),
        BLK=('event_type', lambda x: (x == 'BLOCK').sum())
    )
    
    # Merge PF separately since we pre-filtered it
    pf_count = valid_fouls.groupby('player1_id').size().rename('PF')
    
    # C. Shooting
    fgm = made_shots.groupby('player1_id').size().rename('FGM')
    fga = df[is_fga].groupby('player1_id').size().rename('FGA')
    
    is_3pt = df['event_type'] == 'FIELD_GOAL_3PT'
    fg3m = df[is_3pt & (df['is_made'] == True)].groupby('player1_id').size().rename('FG3M')
    fg3a = df[is_3pt].groupby('player1_id').size().rename('FG3A')
    
    ftm = df[(is_fta) & (df['is_made'] == True)].groupby('player1_id').size().rename('FTM')
    fta = df[is_fta].groupby('player1_id').size().rename('FTA')
    
    # D. Rebounding (Context Aware)
    player_rebs = df[(df['event_type'] == 'REBOUND') & (df['player1_id'] != "0")].copy()
    is_oreb = player_rebs['team_id'] == player_rebs['prev_shooting_team']
    
    orb = player_rebs[is_oreb].groupby('player1_id').size().rename('ORB')
    drb = player_rebs[~is_oreb].groupby('player1_id').size().rename('DRB')
    total_reb = player_rebs.groupby('player1_id').size().rename('REB')
    
    # E. Assists (Player 2 on Makes)
    asts = made_shots[made_shots['player2_id'] != "0"].groupby('player2_id').size().rename('AST')
    
    # Combine
    nums = pd.concat([
        p1_stats, pf_count, fgm, fga, fg3m, fg3a, ftm, fta, 
        orb, drb, total_reb, asts, 
        team_plays, team_fgm, total_rebs
    ], axis=1).fillna(0)
    
    nums.index.name = 'player_id'
    if "0" in nums.index: nums = nums.drop("0")
        
    return nums.reset_index()

def process_season(season):
    print(f"\nProcessing Season: {season}")
    denoms = compute_denominators(season)
    if denoms.empty: return pd.DataFrame()
    nums = compute_numerators_and_plays(season)
    if nums.empty: return pd.DataFrame()
    
    df = pd.merge(denoms, nums, on='player_id', how='left').fillna(0)
    df['season'] = season
    
    # Metrics
    df['TS_PCT'] = df['PTS'] / (2 * (df['FGA'] + 0.44 * df['FTA']))
    df['EFG_PCT'] = (df['FGM'] + 0.5 * df['FG3M']) / df['FGA'].replace(0, 1)
    df['FT_RATE'] = df['FTA'] / df['FGA'].replace(0, 1)
    
    player_plays = df['FGA'] + 0.44 * df['FTA'] + df['TOV']
    denom_usg = df['TEAM_PLAYS_ON_COURT'].replace(0, np.nan).fillna(df['POSS_OFF'])
    df['USG_RATE'] = (player_plays / denom_usg) * 100
    
    teammate_fgm = df['TEAM_FGM_ON_COURT'] - df['FGM']
    df['AST_PCT'] = (df['AST'] / teammate_fgm.replace(0, 1)) * 100
    
    df['REB_PCT'] = (df['REB'] / df['TOTAL_REB_ON_COURT'].replace(0, 1)) * 100
    df['TOV_PCT'] = df['TOV'] / player_plays.replace(0, 1) * 100
    
    df['ORTG'] = (df['TEAM_PTS_ON_COURT'] / df['POSS_OFF'].replace(0, 1)) * 100
    df['DRTG'] = (df['TEAM_PTS_ALLOWED'] / df['POSS_DEF'].replace(0, 1)) * 100
    df['NET_RTG'] = df['ORTG'] - df['DRTG']
    
    return df

def enrich_names(df):
    try:
        p_path = os.path.join(DATA_DIR, "players.parquet")
        if os.path.exists(p_path):
            meta = pd.read_parquet(p_path)
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
        
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 2000)
        print("\n--- Validation (Top 5 Scorers) ---")
        
        preferred = ['player_name', 'season', 'player_id']
        raw_columns = ['PTS', 'TOV', 'STL', 'BLK', 'PF',
                       'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA',
                       'ORB', 'DRB', 'REB', 'AST']

        top5 = final_df.sort_values('PTS', ascending=False).head(5).copy()
        display_cols = [c for c in preferred if c in top5.columns] + [c for c in raw_columns if c in top5.columns]
        fmt_df = top5[display_cols].copy()

        for c in display_cols:
            if c in raw_columns and c in fmt_df.columns:
                fmt_df[c] = fmt_df[c].apply(lambda x: "{:,d}".format(int(round(x))) if pd.notna(x) else "")
            else:
                fmt_df[c] = fmt_df[c].fillna("")

        print(fmt_df.to_string(index=False))

if __name__ == "__main__":
    main()