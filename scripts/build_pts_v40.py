import argparse
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import json

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.modeling.model_config import PtsV40CompositeConfig
from scripts.validate_lineup_pts_v2 import build_player_on_court_nrtg
from scripts.pts_v32_posthoc_harness import patch_and_brier
import subprocess
import tempfile

VALIDATE_LINEUP = REPO / "scripts/validate_lineup_pts_v2.py"

def compute_yoy_corr(df_v40, pts_o_col='pts_o_v40', pts_d_col='pts_d_v40'):
    df = df_v40.copy()
    df['pts_total'] = df[pts_o_col] + df[pts_d_col]
    df = df.sort_values(['player_id', 'season'])
    df['pts_total_next'] = df.groupby('player_id')['pts_total'].shift(-1)
    df['poss_next'] = df.groupby('player_id')['current_possessions'].shift(-1)
    
    mask = (df['current_possessions'] >= 1000) & (df['poss_next'] >= 1000)
    valid = df[mask].dropna(subset=['pts_total', 'pts_total_next'])
    
    if len(valid) == 0:
        return 0.0
    return valid['pts_total'].corr(valid['pts_total_next'])

def run_lineup_validation(df_v40, label="v40_composite_tmp"):
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        tmp_path = f.name
    
    df_v40.to_parquet(tmp_path, index=False)
    
    json_path = REPO / f"reports/lineup_v2_{label}.json"
    
    cmd = [
        sys.executable, str(VALIDATE_LINEUP),
        "--pts-file", tmp_path,
        "--pts-col", "pts_o_v40",
        "--pts-col-d", "pts_d_v40",
        "--label", label,
        "--output", str(json_path)
    ]
    subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
    os.remove(tmp_path)
    if not json_path.exists():
        return None, None, None
        
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    pooled = data.get("pooled", {})
    joint_r = pooled.get("pts", {}).get("joint_mean_wr", 0.0)
    off_r = pooled.get("pts", {}).get("offense_mean_wr", 0.0)
    def_r = pooled.get("pts", {}).get("defense_mean_wr", 0.0)
    
    return joint_r, off_r, def_r

def build_composite(df_a, df_c, weight_c):
    weight_a = 1.0 - weight_c
    
    # df_a has pts_o_v40_a, pts_d_v40_a, current_possessions
    # df_c has pts_d_v40_c
    
    # We want a single merged df
    df = df_a[['player_id', 'season', 'pts_o_v32', 'pts_d_v32', 'pts_o_v40_a', 'pts_d_v40_a', 'current_possessions']].merge(
        df_c[['player_id', 'season', 'pts_d_v40_c']],
        on=['player_id', 'season'],
        how='inner'
    )
    
    # Offense: just smoothing
    df['pts_o_v40'] = df['pts_o_v40_a']
    
    # Defense: blend smoothing and redesign
    df['pts_d_v40'] = weight_a * df['pts_d_v40_a'] + weight_c * df['pts_d_v40_c']
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pts-a", required=True)
    parser.add_argument("--pts-c", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()

    df_a = pd.read_parquet(args.pts_a)
    df_c = pd.read_parquet(args.pts_c)
    
    if args.sweep:
        weights_c = [0.3, 0.4, 0.5, 0.6, 0.7]
        print(f"{'w_c':<4} | {'w_a':<4} | {'YoY_r':<6} | {'Joint_r':<7} | {'Def_r':<7} | {'Brier':<7}")
        print("-" * 55)
        
        best_brier = 1.0
        best_w = 0.5
        
        for w_c in weights_c:
            df = build_composite(df_a, df_c, w_c)
            
            yoy_r = compute_yoy_corr(df)
            joint_r, off_r, def_r = run_lineup_validation(df, label=f"v40_comp_{w_c}")
            
            # Brier score
            df_brier = df.copy()
            df_brier['pts_o_v32'] = df_brier['pts_o_v40']
            df_brier['pts_d_v32'] = df_brier['pts_d_v40']
            brier_res = patch_and_brier(df_brier)
            brier_score = brier_res.get("brier", 1.0)
            
            print(f"{w_c:<4.1f} | {1-w_c:<4.1f} | {yoy_r:<6.4f} | {joint_r:<7.4f} | {def_r:<7.4f} | {brier_score:<7.4f}")
            if brier_score < best_brier:
                best_brier = brier_score
                best_w = w_c
                
        print(f"\nBest config: weight_c = {best_w}")
        
    else:
        config = PtsV40CompositeConfig()
        df = build_composite(df_a, df_c, config.defense_v40c_weight)
        df.to_parquet(args.output, index=False)
        
        # Build features for forecast via patch_and_brier but saving
        df_brier = df.copy()
        df_brier['pts_o_v32'] = df_brier['pts_o_v40']
        df_brier['pts_d_v32'] = df_brier['pts_d_v40']
        
        # we can patch directly
        features = pd.read_parquet(REPO / "data/processed/forecast/projected_team_features_v32.parquet")
        # In practice we just write the composite df to output.
        print(f"Saved {args.output}")