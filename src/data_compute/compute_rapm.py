"""
src/modeling/compute_rapm.py
Computes Regularized Adjusted Plus-Minus (RAPM) using Ridge Regression.
UPDATED: Calculates RAPM per Season (not aggregated).

Methodology:
- Iterates through each season in the clean possessions data.
- Fits a Ridge Regression model for that specific season.
- X = +1 (Offense) / -1 (Defense) dummy variables.
- Y = Points per 100 Possessions.
"""

import pandas as pd
import numpy as np
import glob
import os
import sys
from scipy.sparse import csr_matrix
from sklearn.linear_model import RidgeCV

# Ensure src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

DATA_DIR = "data/historical"
OUTPUT_DIR = "data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_id(val):
    """Standardizes IDs to strings without decimals."""
    if pd.isna(val): return "0"
    return str(val).replace(".0", "")

def load_clean_possessions():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "possessions_clean_*.parquet")))
    if not files:
        print("❌ No clean possession files found in data/historical/")
        return pd.DataFrame()
    
    print(f"Loading {len(files)} possession files...")
    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        # Ensure season column exists
        if 'season' not in df.columns:
            # Extract from filename if missing: possessions_clean_2022-23.parquet
            base = os.path.basename(f)
            season = base.replace("possessions_clean_", "").replace(".parquet", "")
            df['season'] = season
        dfs.append(df)
    
    if not dfs:
        return pd.DataFrame()
        
    return pd.concat(dfs, ignore_index=True)

def build_sparse_matrix(df):
    """Builds the X (Player Presence) and Y (Points) matrices for a single season."""
    # 1. Identify all unique players involved in this season
    all_players = set()
    
    # Helper to collect IDs from lineup columns
    for col in ['off_lineup', 'def_lineup']:
        if df[col].empty: continue
        # Flatten all lists efficiently
        ids = [clean_id(pid) for lineup in df[col] if isinstance(lineup, (list, np.ndarray)) for pid in lineup]
        all_players.update(ids)
    
    all_players.discard('0')
    sorted_players = sorted(list(all_players))
    player_to_idx = {pid: i for i, pid in enumerate(sorted_players)}
    
    n_players = len(sorted_players)
    n_poss = len(df)
    
    # 2. Construct Matrix
    data = []
    rows = []
    cols = []
    
    off_lineups = df['off_lineup'].values
    def_lineups = df['def_lineup'].values
    
    for i in range(n_poss):
        # Offense (+1)
        off = off_lineups[i]
        if isinstance(off, (list, np.ndarray)):
            for pid in off:
                pid_clean = clean_id(pid)
                if pid_clean in player_to_idx:
                    rows.append(i)
                    cols.append(player_to_idx[pid_clean])
                    data.append(1)
        
        # Defense (-1)
        defn = def_lineups[i]
        if isinstance(defn, (list, np.ndarray)):
            for pid in defn:
                pid_clean = clean_id(pid)
                if pid_clean in player_to_idx:
                    rows.append(i)
                    cols.append(player_to_idx[pid_clean])
                    data.append(-1)
                    
    X = csr_matrix((data, (rows, cols)), shape=(n_poss, n_players))
    
    # Target: Points per 100
    Y = df['points'].values * 100.0 
    
    return X, Y, sorted_players

def run_rapm_for_season(df, season_name):
    print(f"\n--- Processing Season: {season_name} ({len(df):,} possessions) ---")
    
    # 1. Build Matrix
    X, Y, player_ids = build_sparse_matrix(df)
    print(f"   Matrix: {X.shape[0]} poss x {X.shape[1]} players")
    
    # 2. Fit Ridge
    # Note: Single-season RAPM is noisier, so we use higher alphas to regularize aggressive outliers
    alphas = [1000, 2000, 3000, 5000] 
    model = RidgeCV(alphas=alphas, fit_intercept=True)
    model.fit(X, Y)
    
    print(f"   ✅ Best Alpha: {model.alpha_}")
    
    # 3. Format Results
    results = []
    for pid, coef in zip(player_ids, model.coef_):
        results.append({
            "season": season_name,
            "player_id": pid,
            "RAPM": coef,
            "intercept": model.intercept_
        })
        
    return pd.DataFrame(results)

def enrich_names(rapm_df):
    """Attaches player names."""
    try:
        p_path = os.path.join(DATA_DIR, "players.parquet")
        if os.path.exists(p_path):
            meta = pd.read_parquet(p_path)
            # Use fixed ID logic
            if 'id' in meta.columns:
                meta['id'] = meta['id'].astype(str).apply(clean_id)
                name_map = meta.set_index('id')['full_name'].to_dict()
                rapm_df['player_name'] = rapm_df['player_id'].map(name_map).fillna("Unknown")
            else:
                 print("⚠️ 'id' column missing in players.parquet")
        else:
            rapm_df['player_name'] = rapm_df['player_id']
    except Exception as e:
        print(f"⚠️ Could not load names: {e}")
        rapm_df['player_name'] = rapm_df['player_id']
    
    return rapm_df

def main():
    # 1. Load All Data
    full_df = load_clean_possessions()
    if full_df.empty: return
    
    all_results = []
    
    # 2. Loop by Season
    seasons = sorted(full_df['season'].unique())
    for season in seasons:
        season_df = full_df[full_df['season'] == season].copy()
        season_rapm = run_rapm_for_season(season_df, season)
        all_results.append(season_rapm)
        
    final_df = pd.concat(all_results, ignore_index=True)
    
    # 3. Cleanup & Save
    final_df = enrich_names(final_df)
    final_df = final_df.sort_values(['season', 'RAPM'], ascending=[True, False])
    
    out_path = os.path.join(OUTPUT_DIR, "player_rapm.parquet")
    final_df.to_parquet(out_path, index=False)
    
    print(f"\n✅ RAPM saved to {out_path}")
    
    # Show Top 5 per season
    for season in seasons:
        print(f"\nTop 5 RAPM ({season}):")
        print(final_df[final_df['season'] == season].head(5)[['player_name', 'RAPM']])

if __name__ == "__main__":
    main()