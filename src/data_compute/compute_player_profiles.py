"""
src/data_compute/compute_player_profiles.py
=============================================================================
Stream A: Computes "Box Score Plus" and "Four Factors" profiles.
FINAL FIX (v3):
- Adds 'gp' (Games Played) count for per-game metrics.
- Retains all previous fixes (Steals, Turnovers, Fouls, OREB).
=============================================================================
"""

import pandas as pd
import numpy as np
import glob
import os
import sys
from pathlib import Path

# Adjust path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils.player_name_normalizer import (
from src.data.schema_contract import load_standardized, save_standardized
    apply_player_name_normalization,
    build_player_name_maps,
)

DATA_DIR = "data/historical"
OUTPUT_DIR = "data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_id(val):
    if pd.isna(val) or val == "": return "0"
    return str(int(float(val)))

def time_to_seconds(val):
    if pd.isna(val) or val == "": return 0.0
    try:
        parts = str(val).split(':')
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(val)
    except:
        return 0.0

def get_season_from_path(path):
    base = os.path.basename(path)
    return base.replace("possessions_clean_", "").replace("pbp_with_lineups_", "").replace(".parquet", "")

def compute_denominators(season):
    path = os.path.join(DATA_DIR, f"possessions_clean_{season}.parquet")
    if not os.path.exists(path): return pd.DataFrame()

    print(f"   Loading Possessions for {season}...")
    df = load_standardized(path)
    
    # Calculate Duration
    df['start_sec'] = df['start_clock'].apply(time_to_seconds)
    df['end_sec'] = df['end_clock'].apply(time_to_seconds)
    df['duration'] = (df['start_sec'] - df['end_sec']).clip(lower=0)
    
    # Explode Offense -> Count Possessions AND Games Played
    off_exploded = df.explode('off_lineup')
    off_stats = off_exploded.groupby('off_lineup').agg(
        gp=('game_id', 'nunique'),          # <--- NEW: Count unique games
        poss_off=('game_id', 'count'),
        seconds_off=('duration', 'sum'),
        team_pts_on_court=('points', 'sum')
    ).reset_index().rename(columns={'off_lineup': 'player_id'})
    
    # Explode Defense
    def_exploded = df.explode('def_lineup')
    def_stats = def_exploded.groupby('def_lineup').agg(
        poss_def=('game_id', 'count'),
        seconds_def=('duration', 'sum'),
        team_pts_allowed=('points', 'sum')
    ).reset_index().rename(columns={'def_lineup': 'player_id'})
    
    denom_df = pd.merge(off_stats, def_stats, on='player_id', how='outer').fillna(0)
    denom_df['player_id'] = denom_df['player_id'].apply(clean_id)
    
    # Compute Total Minutes
    denom_df['min'] = (denom_df['seconds_off'] + denom_df['seconds_def']) / 60.0
    denom_df['mpg'] = denom_df['min'] / denom_df['gp'].replace(0, 1)
    
    return denom_df

def compute_numerators_and_plays(season):
    path = os.path.join(DATA_DIR, f"pbp_with_lineups_{season}.parquet")
    if not os.path.exists(path): return pd.DataFrame()
        
    print(f"   Loading PBP Events for {season}...")
    df = load_standardized(path)
    
    # 1. Clean Data & Context
    df['player1_id'] = df['player1_id'].apply(clean_id)
    df['player2_id'] = df['player2_id'].apply(clean_id)
    df['team_id'] = df['team_id'].fillna(0).astype(int)
    
    if 'event_text' not in df.columns: df['event_text'] = ""
    df['event_text'] = df['event_text'].fillna("").str.upper()
    
    # OREB Context
    df['is_shot'] = df['event_type'].str.contains('FIELD_GOAL', na=False) | (df['event_type'] == 'FREE_THROW')
    df['shooting_team'] = np.where(df['is_shot'], df['team_id'], np.nan)
    df['prev_shooting_team'] = df['shooting_team'].ffill().shift(1).fillna(0).astype(int)

    # 2. Team Aggregates
    is_fga = df['event_type'].str.contains('FIELD_GOAL', na=False)
    is_fta = df['event_type'] == 'FREE_THROW'
    is_tov = df['event_type'] == 'TURNOVER'
    
    df['play_weight'] = 0.0
    df.loc[is_fga | is_tov, 'play_weight'] = 1.0
    df.loc[is_fta, 'play_weight'] = 0.44

    usage_events = df[df['play_weight'] > 0].copy()
    team_plays_list = []
    
    for team_id, group in usage_events.groupby('team_id'):
        if team_id == 0: continue
        col_name = f"lineup_{int(team_id)}"
        if col_name in group.columns:
            exploded = group.explode(col_name)
            sums = exploded.groupby(col_name)['play_weight'].sum()
            team_plays_list.append(sums)
            
    team_plays = pd.concat(team_plays_list).groupby(level=0).sum().rename('team_plays_on_court') if team_plays_list else pd.Series(name='team_plays_on_court')

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
    team_fgm = pd.concat(team_fgm_list).groupby(level=0).sum().rename('team_fgm_on_court') if team_fgm_list else pd.Series(name='team_fgm_on_court')

    # Rebound Chances
    valid_rebs = df[(df['event_type'] == 'REBOUND') & (df['player1_id'] != "0")].copy()
    total_reb_list = []
    lineup_cols = [c for c in df.columns if c.startswith('lineup_')]
    for col in lineup_cols:
        if col in valid_rebs.columns:
            exploded = valid_rebs.explode(col)
            counts = exploded[exploded[col].notna()].groupby(col).size()
            total_reb_list.append(counts)
    total_rebs = pd.concat(total_reb_list).groupby(level=0).sum().rename('total_reb_on_court') if total_reb_list else pd.Series(name='total_reb_on_court')

    # --- 5. INDIVIDUAL STATS ---
    
    # A. Turnovers (Include Violations labeled as Turnovers)
    is_tov_explicit = df['event_type'] == 'TURNOVER'
    is_violation_tov = (df['event_type'] == 'VIOLATION') & (df['event_text'].str.contains("TURNOVER", na=False))
    tov_count = df[is_tov_explicit | is_violation_tov].groupby('player1_id').size().rename('tov')

    # B. Steals & Blocks (Explicit Types, Player 1)
    stl_count = df[df['event_type'] == 'STEAL'].groupby('player1_id').size().rename('stl')
    blk_count = df[df['event_type'] == 'BLOCK'].groupby('player1_id').size().rename('blk')
    
    # C. Personal Fouls (Exclude Techs)
    fouls = df[df['event_type'] == 'FOUL'].copy()
    is_tech = fouls['event_text'].str.contains("TECHNICAL", na=False)
    is_def3 = fouls['event_text'].str.contains("DEFENSIVE 3", na=False)
    pf_count = fouls[~(is_tech | is_def3)].groupby('player1_id').size().rename('pf')

    # D. Shooting
    fgm = made_shots.groupby('player1_id').size().rename('fgm')
    fga = df[df['event_type'].str.contains('FIELD_GOAL', na=False)].groupby('player1_id').size().rename('fga')
    
    is_3pt = df['event_type'] == 'FIELD_GOAL_3PT'
    fg3m = df[is_3pt & (df['is_made'] == True)].groupby('player1_id').size().rename('fg3m')
    fg3a = df[is_3pt].groupby('player1_id').size().rename('fg3a')
    
    ftm = df[(df['event_type'] == 'FREE_THROW') & (df['is_made'] == True)].groupby('player1_id').size().rename('ftm')
    fta = df[df['event_type'] == 'FREE_THROW'].groupby('player1_id').size().rename('fta')
    
    # E. Rebounding
    player_rebs = df[(df['event_type'] == 'REBOUND') & (df['player1_id'] != "0")].copy()
    is_oreb = player_rebs['team_id'] == player_rebs['prev_shooting_team']
    
    orb = player_rebs[is_oreb].groupby('player1_id').size().rename('orb')
    drb = player_rebs[~is_oreb].groupby('player1_id').size().rename('drb')
    total_reb = player_rebs.groupby('player1_id').size().rename('reb')
    
    # F. Assists (Player 2 on Makes)
    asts = made_shots[made_shots['player2_id'] != "0"].groupby('player2_id').size().rename('ast')
    
    # G. Points
    pts = df.groupby('player1_id')['points'].sum().rename('pts')
    
    # Combine
    nums = pd.concat([
        pts, fgm, fga, fg3m, fg3a, ftm, fta, 
        orb, drb, total_reb, asts, stl_count, blk_count, tov_count, pf_count,
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
    df['ts_pct'] = df['pts'] / (2 * (df['fga'] + 0.44 * df['fta']))
    df['efg_pct'] = (df['fgm'] + 0.5 * df['fg3m']) / df['fga'].replace(0, 1)
    df['ft_rate'] = df['fta'] / df['fga'].replace(0, 1)
    
    player_plays = df['fga'] + 0.44 * df['fta'] + df['tov']
    denom_usg = df['team_plays_on_court'].replace(0, np.nan).fillna(df['poss_off'])
    df['usg_rate'] = (player_plays / denom_usg) * 100
    
    teammate_fgm = df['team_fgm_on_court'] - df['fgm']
    df['ast_pct'] = (df['ast'] / teammate_fgm.replace(0, 1)) * 100
    
    df['reb_pct'] = (df['reb'] / df['total_reb_on_court'].replace(0, 1)) * 100
    df['tov_pct'] = df['tov'] / player_plays.replace(0, 1) * 100
    
    df['ortg'] = (df['team_pts_on_court'] / df['poss_off'].replace(0, 1)) * 100
    df['drtg'] = (df['team_pts_allowed'] / df['poss_def'].replace(0, 1)) * 100
    df['net_rtg'] = df['ortg'] - df['drtg']
    
    return df

def enrich_names(df):
    try:
        source_specs = [
            (Path(os.path.join(DATA_DIR, "players.parquet")), ["id", "player_id"], ["full_name", "player_name"], 1),
            (Path(os.path.join(OUTPUT_DIR, "player_archetypes.parquet")), ["player_id", "player_id"], ["player_name", "player_name"], 2),
            (Path(os.path.join("data/processed/bke", "bke_v28_decomposition.parquet")), ["player_id", "player_id"], ["player_name", "player_name"], 3),
        ]
        id_to_name, key_to_name = build_player_name_maps(source_specs)
        out = apply_player_name_normalization(
            df=df,
            player_id_col="player_id",
            player_name_col="player_name",
            id_to_name=id_to_name,
            key_to_name=key_to_name,
        )
        return out
    except Exception:
        if "player_name" not in df.columns:
            df["player_name"] = df["player_id"]
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
        save_standardized(final_df, out_path)
        print(f"\n✅ Saved Advanced Profiles to {out_path}")
        
        pd.set_option('display.max_columns', None)
        pd.set_option('display.float_format', '{:.1f}'.format)
        print("\n--- Validation (Top 5 Scorers) ---")
        print(final_df.sort_values('pts', ascending=False)[['player_name', 'gp', 'min', 'mpg', 'pts', 'tov', 'stl']].head(5))

if __name__ == "__main__":
    main()