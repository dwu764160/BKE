import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.modeling.model_config import PtsV40MultiSeasonConfig

def compute_prior_weights(decay, k):
    return [decay ** i for i in range(k)]

def apply_multiseason_smoothing(pts_df: pd.DataFrame, decomp_df: pd.DataFrame, cfg: PtsV40MultiSeasonConfig) -> pd.DataFrame:
    # Merge decomp to get possessions and archetype
    decomp_sub = decomp_df[['player_id', 'season', 'possessions_played', 'primary_archetype']].copy()
    
    # Clean player_ids to ensure proper joining
    pts_df['player_id'] = pts_df['player_id'].astype(str)
    decomp_sub['player_id'] = decomp_sub['player_id'].astype(str)
    
    df = pts_df.merge(decomp_sub, on=['player_id', 'season'], how='left')
    
    # Fallback for missing possessions
    df['possessions_played'] = pd.to_numeric(df['possessions_played'], errors='coerce')
    df['possessions_played'] = df['possessions_played'].fillna(cfg.poss_fallback)
    df['current_possessions'] = df['possessions_played'].astype(int)
    
    # Calculate cohort means for rookies
    cohort_means_o = df.groupby(['season', 'primary_archetype'])['pts_o_v32'].transform('mean')
    cohort_means_d = df.groupby(['season', 'primary_archetype'])['pts_d_v32'].transform('mean')
    global_mean_o = df.groupby('season')['pts_o_v32'].transform('mean')
    global_mean_d = df.groupby('season')['pts_d_v32'].transform('mean')
    
    df['cohort_mean_o'] = cohort_means_o.fillna(global_mean_o)
    df['cohort_mean_d'] = cohort_means_d.fillna(global_mean_d)
    
    # Sort for historical lookback
    df = df.sort_values(by=['player_id', 'season']).reset_index(drop=True)
    
    # Compute priors
    weights = compute_prior_weights(cfg.geometric_decay, cfg.k_max_prior)
    
    o_priors = []
    d_priors = []
    n_priors = []
    
    for i in range(1, cfg.k_max_prior + 1):
        df[f'pts_o_lag_{i}'] = df.groupby('player_id')['pts_o_v32'].shift(i)
        df[f'pts_d_lag_{i}'] = df.groupby('player_id')['pts_d_v32'].shift(i)
    
    # Vectorized computation of weighted priors
    sum_w = np.zeros(len(df))
    sum_o = np.zeros(len(df))
    sum_d = np.zeros(len(df))
    n_seasons = np.zeros(len(df), dtype=int)
    
    for i in range(1, cfg.k_max_prior + 1):
        w = weights[i-1]
        valid = ~df[f'pts_o_lag_{i}'].isna()
        n_seasons += valid.astype(int)
        
        sum_o += np.where(valid, df[f'pts_o_lag_{i}'] * w, 0)
        sum_d += np.where(valid, df[f'pts_d_lag_{i}'] * w, 0)
        sum_w += np.where(valid, w, 0)
        
    with np.errstate(invalid='ignore', divide='ignore'):
        hist_prior_o = np.where(sum_w > 0, sum_o / sum_w, np.nan)
        hist_prior_d = np.where(sum_w > 0, sum_d / sum_w, np.nan)
    
    df['n_prior_seasons'] = n_seasons
    df['hist_prior_o'] = hist_prior_o
    df['hist_prior_d'] = hist_prior_d
    
    # Determine Prior and Source
    df['prior_estimate_o'] = df['hist_prior_o']
    df['prior_estimate_d'] = df['hist_prior_d']
    df['prior_source'] = 'prior_seasons'
    
    # Apply Rookie fallback
    rookie_mask = df['n_prior_seasons'] == 0
    df.loc[rookie_mask, 'prior_estimate_o'] = df.loc[rookie_mask, 'cohort_mean_o'] if cfg.archetype_mean_prior else 0
    df.loc[rookie_mask, 'prior_estimate_d'] = df.loc[rookie_mask, 'cohort_mean_d'] if cfg.archetype_mean_prior else 0
    df.loc[rookie_mask, 'prior_source'] = 'rookie_archetype_mean'
    
    # Compute weight 'w' (weight on current season)
    w_standard = df['current_possessions'] / (df['current_possessions'] + cfg.tau_possessions)
    df['w'] = w_standard
    
    # Exception: 1 prior season but < 500 poss in current season
    exception_mask = (df['n_prior_seasons'] == 1) & (df['current_possessions'] < cfg.rookie_poss_threshold)
    df.loc[exception_mask, 'w'] = df.loc[exception_mask, 'current_possessions'] / 2000.0
    
    # Clip weights to [0, 1] for safety
    df['w'] = df['w'].clip(0, 1)
    
    # Smooth
    df['pts_o_v40_a'] = df['w'] * df['pts_o_v32'] + (1 - df['w']) * df['prior_estimate_o']
    df['pts_d_v40_a'] = df['w'] * df['pts_d_v32'] + (1 - df['w']) * df['prior_estimate_d']
    df['prior_weight_o'] = 1 - df['w']
    
    out_cols = [
        'player_id', 'season', 'pts_o_v32', 'pts_d_v32',
        'pts_o_v40_a', 'pts_d_v40_a', 'prior_weight_o',
        'n_prior_seasons', 'current_possessions', 'prior_source'
    ]
    return df[out_cols].copy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pts-v32", required=True)
    parser.add_argument("--decomp", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tau", type=int, default=1500)
    parser.add_argument("--decay", type=float, default=0.55)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args()
    
    cfg = PtsV40MultiSeasonConfig(
        tau_possessions=args.tau,
        geometric_decay=args.decay,
        k_max_prior=args.k
    )
    
    if args.print_config:
        print(json.dumps(asdict(cfg), indent=2))
        
    pts = pd.read_parquet(args.pts_v32)
    decomp = pd.read_parquet(args.decomp)
    
    out_df = apply_multiseason_smoothing(pts, decomp, cfg)
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out_df.to_parquet(args.output, index=False)
    print(f"Wrote {len(out_df)} rows to {args.output}")

if __name__ == "__main__":
    main()
