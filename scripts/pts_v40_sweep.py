import json
import os
import sys
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.pts_v40_multiseason import apply_multiseason_smoothing, PtsV40MultiSeasonConfig
from scripts.pts_v32_posthoc_harness import patch_and_brier

DECOMP_PATH = REPO / "data/processed/bke/bke_v28_decomposition.parquet"
PTS_V32_PATH = REPO / "data/processed/bke/pts_v32.parquet"
VALIDATE_LINEUP = REPO / "scripts/validate_lineup_pts_v2.py"

def compute_yoy_corr(df_v40):
    df = df_v40.copy()
    df['pts_total'] = df['pts_o_v40_a'] + df['pts_d_v40_a']
    df = df.sort_values(['player_id', 'season'])
    df['pts_total_next'] = df.groupby('player_id')['pts_total'].shift(-1)
    df['poss_next'] = df.groupby('player_id')['current_possessions'].shift(-1)
    
    mask = (df['current_possessions'] >= 1000) & (df['poss_next'] >= 1000)
    valid = df[mask].dropna(subset=['pts_total', 'pts_total_next'])
    
    if len(valid) == 0:
        return 0.0
    return valid['pts_total'].corr(valid['pts_total_next'])

def compute_baseline_ranks(pts, decomp):
    df = pts.merge(decomp[['player_id', 'season', 'player_name']], on=['player_id', 'season'], how='left')
    df['pts_total_v32'] = df['pts_o_v32'] + df['pts_d_v32']
    df['rank_v32'] = df.groupby('season')['pts_total_v32'].rank(ascending=False, method='first')
    return df[['player_id', 'season', 'player_name', 'rank_v32']]

def check_star_sanity(df_v40, baseline_ranks):
    stars = ["Stephen Curry", "Kevin Durant", "Luka Doncic", "Shai Gilgeous-Alexander", "Giannis Antetokounmpo"]
    check_seasons = ["2022-23", "2023-24", "2024-25"]
    
    df = df_v40.copy()
    df['pts_total_v40'] = df['pts_o_v40_a'] + df['pts_d_v40_a']
    df['rank_v40'] = df.groupby('season')['pts_total_v40'].rank(ascending=False, method='first')
    
    merged = df.merge(baseline_ranks, on=['player_id', 'season'], how='inner')
    
    sanity_passed = True
    issues = []
    
    for season in check_seasons:
        season_df = merged[merged['season'] == season]
        for star in stars:
            star_row = season_df[season_df['player_name'] == star]
            if len(star_row) == 0:
                continue
            
            rank_v40 = star_row['rank_v40'].values[0]
            rank_v32 = star_row['rank_v32'].values[0]
            
            if (rank_v40 - rank_v32) > 8:
                sanity_passed = False
                issues.append(f"{star} dropped more than 8 ranks in {season} (v32: {rank_v32}, v40: {rank_v40})")
                
    return sanity_passed, issues

def run_lineup_validation(df_v40):
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        tmp_path = f.name
    
    df_v40.to_parquet(tmp_path, index=False)
    label = "v40_sweep_tmp"
    
    json_path = REPO / f"reports/lineup_v2_{label}.json"
    
    cmd = [
        sys.executable, str(VALIDATE_LINEUP),
        "--pts-file", tmp_path,
        "--pts-col", "pts_o_v40_a",
        "--pts-col-d", "pts_d_v40_a",
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

def run_sweep():
    pts = pd.read_parquet(PTS_V32_PATH)
    decomp = pd.read_parquet(DECOMP_PATH)
    baseline_ranks = compute_baseline_ranks(pts, decomp)
    
    # Grid
    taus = [800, 1200, 1500, 2000, 3000]
    decays = [0.40, 0.55, 0.70]
    ks = [2, 3, 4]
    
    results = []
    
    print("Starting Improvement A Sweep...")
    print(f"{'tau':<5} | {'decay':<5} | {'k':<2} | {'YoY_r':<6} | {'Joint_r':<7} | {'Brier':<7} | {'Sanity'}")
    print("-" * 55)
    
    for tau in taus:
        for decay in decays:
            for k in ks:
                cfg = PtsV40MultiSeasonConfig(tau_possessions=tau, geometric_decay=decay, k_max_prior=k)
                df_v40 = apply_multiseason_smoothing(pts, decomp, cfg)
                
                yoy_r = compute_yoy_corr(df_v40)
                sanity_passed, issues = check_star_sanity(df_v40, baseline_ranks)
                
                joint_r, off_r, def_r = run_lineup_validation(df_v40)
                
                # Brier score via patch_and_brier (requires 'pts_o_v32' and 'pts_d_v32' columns for the new values)
                df_brier = df_v40.copy()
                df_brier['pts_o_v32'] = df_brier['pts_o_v40_a']
                df_brier['pts_d_v32'] = df_brier['pts_d_v40_a']
                brier_res = patch_and_brier(df_brier)
                brier_score = brier_res.get("brier", 1.0)
                
                sanity_str = "PASS" if sanity_passed else "FAIL"
                print(f"{tau:<5} | {decay:<5.2f} | {k:<2} | {yoy_r:<6.4f} | {joint_r:<7.4f} | {brier_score:<7.4f} | {sanity_str}")
                
                results.append({
                    "tau": tau,
                    "decay": decay,
                    "k": k,
                    "yoy_corr": float(yoy_r),
                    "joint_r": float(joint_r),
                    "off_r": float(off_r),
                    "def_r": float(def_r),
                    "brier": float(brier_score),
                    "star_sanity_pass": sanity_passed,
                    "star_issues": issues
                })
                
    out_path = REPO / "reports/v40_sweep.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"\nSweep complete. Results saved to {out_path}")

if __name__ == "__main__":
    run_sweep()
