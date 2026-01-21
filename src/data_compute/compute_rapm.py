"""
src/data_compute/compute_rapm.py
Computes Regularized Adjusted Plus-Minus (RAPM) using Ridge Regression.

UPDATED (2026-01-19): Enhanced RAPM implementation with:
- Multi-season pooled RAPM with decay weights for past seasons
- Possession duration weighting for better accuracy
- Separate ORAPM and DRAPM calculations
- Cross-validated alpha selection with wide parameter range

Methodology:
- RAPM uses ridge regression to estimate player impact from lineup data
- X = sparse matrix with +1 (offense players) / -1 (defense players) per possession
- Y = Points scored per possession (normalized to per 100 possessions scale)
- Ridge regularization (L2) reduces variance in player estimates
- Multi-year pooling improves stability (~2x more accurate than single-season APM)

References:
- NBAsuffer: https://www.nbastuffer.com/analytics101/regularized-adjusted-plus-minus-rapm/
- Basketball-Reference BPM methodology uses 5-year Bayesian RAPM as regression basis
"""

import pandas as pd
import numpy as np
import glob
import os
import sys
from scipy.sparse import csr_matrix, vstack
from sklearn.linear_model import RidgeCV

# Ensure src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

DATA_DIR = "data/historical"
OUTPUT_DIR = "data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Season decay weights for multi-year pooling (most recent season = 1.0)
# Based on research showing 3+ years optimal with weighted past data
SEASON_DECAY_WEIGHTS = {
    0: 1.0,   # Current season
    1: 0.65,  # Previous season
    2: 0.40,  # Two seasons ago
    3: 0.25,  # Three seasons ago
}

def clean_id(val):
    """Standardizes IDs to strings without decimals."""
    if pd.isna(val): return "0"
    return str(val).replace(".0", "")

def load_clean_possessions():
    """Load all clean possessions data with season labels."""
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
    
    full_df = pd.concat(dfs, ignore_index=True)
    
    # Calculate possession duration for weighting
    full_df = calculate_possession_duration(full_df)
    
    return full_df

def parse_clock_seconds(clock_str):
    """Convert clock string (MM:SS or MM:SS.ms) to total seconds."""
    try:
        if pd.isna(clock_str):
            return 0
        parts = str(clock_str).split(':')
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        return 0
    except:
        return 0

def calculate_possession_duration(df):
    """Calculate possession duration in seconds for sample weighting."""
    # Convert clock strings to seconds
    df['start_seconds'] = df['start_clock'].apply(parse_clock_seconds)
    df['end_seconds'] = df['end_clock'].apply(parse_clock_seconds)
    
    # Duration = start - end (clock counts down)
    df['duration'] = df['start_seconds'] - df['end_seconds']
    
    # Handle period boundaries and invalid durations
    df['duration'] = df['duration'].clip(lower=1, upper=60)  # Min 1 sec, max 60 sec (full shot clock)
    
    # Normalize duration weights (optional - scale to reasonable range)
    # Weight possessions by sqrt of duration to reduce outlier impact
    df['duration_weight'] = np.sqrt(df['duration'])
    
    return df

def build_sparse_matrix(df, player_to_idx=None):
    """
    Builds the X (Player Presence) and Y (Points) matrices for RAPM.
    
    Args:
        df: DataFrame with possession data
        player_to_idx: Optional pre-built player index (for multi-season pooling)
    
    Returns:
        X: Sparse matrix (n_poss x n_players) with +1 offense, -1 defense
        Y: Points per possession array
        sorted_players: List of player IDs (column order)
        sample_weights: Array of weights for each possession
    """
    # 1. Identify all unique players if not provided
    if player_to_idx is None:
        all_players = set()
        for col in ['off_lineup', 'def_lineup']:
            if df[col].empty: continue
            ids = [clean_id(pid) for lineup in df[col] if isinstance(lineup, (list, np.ndarray)) for pid in lineup]
            all_players.update(ids)
        
        all_players.discard('0')
        sorted_players = sorted(list(all_players))
        player_to_idx = {pid: i for i, pid in enumerate(sorted_players)}
    else:
        sorted_players = [pid for pid, _ in sorted(player_to_idx.items(), key=lambda x: x[1])]
    
    n_players = len(sorted_players)
    n_poss = len(df)
    
    # 2. Construct Sparse Matrix
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
    
    # Target: Points per possession (will normalize to per 100 after fitting)
    Y = df['points'].values.astype(float)
    
    # Sample weights from duration
    if 'duration_weight' in df.columns:
        sample_weights = df['duration_weight'].values
    else:
        sample_weights = np.ones(n_poss)
    
    return X, Y, sorted_players, sample_weights, player_to_idx

def build_offensive_matrix(df, player_to_idx=None):
    """
    Builds matrix where only OFFENSIVE presence is captured.
    Used for ORAPM calculation.
    
    Args:
        df: DataFrame with possession data
        player_to_idx: Optional pre-built player index
    
    Returns:
        X: Sparse matrix (n_poss x n_players) with +1 for offense only
        Y: Points per possession array
        sorted_players: List of player IDs
        sample_weights: Array of weights
        player_to_idx: Player to column index mapping
    """
    # Identify all unique players if not provided
    if player_to_idx is None:
        all_players = set()
        for col in ['off_lineup', 'def_lineup']:
            if df[col].empty: continue
            ids = [clean_id(pid) for lineup in df[col] if isinstance(lineup, (list, np.ndarray)) for pid in lineup]
            all_players.update(ids)
        
        all_players.discard('0')
        sorted_players = sorted(list(all_players))
        player_to_idx = {pid: i for i, pid in enumerate(sorted_players)}
    else:
        sorted_players = [pid for pid, _ in sorted(player_to_idx.items(), key=lambda x: x[1])]
    
    n_players = len(sorted_players)
    n_poss = len(df)
    
    # Only record offensive presence
    data = []
    rows = []
    cols = []
    
    off_lineups = df['off_lineup'].values
    
    for i in range(n_poss):
        off = off_lineups[i]
        if isinstance(off, (list, np.ndarray)):
            for pid in off:
                pid_clean = clean_id(pid)
                if pid_clean in player_to_idx:
                    rows.append(i)
                    cols.append(player_to_idx[pid_clean])
                    data.append(1)
    
    X = csr_matrix((data, (rows, cols)), shape=(n_poss, n_players))
    Y = df['points'].values.astype(float)
    
    if 'duration_weight' in df.columns:
        sample_weights = df['duration_weight'].values
    else:
        sample_weights = np.ones(n_poss)
    
    return X, Y, sorted_players, sample_weights, player_to_idx


def build_defensive_matrix(df, player_to_idx=None):
    """
    Builds matrix where only DEFENSIVE presence is captured.
    Used for DRAPM calculation.
    
    Note: We use +1 for defensive presence and the target is points ALLOWED.
    A positive DRAPM means the player ALLOWS more points (bad defense).
    We'll flip the sign at the end so positive = good defense.
    
    Args:
        df: DataFrame with possession data
        player_to_idx: Optional pre-built player index
    
    Returns:
        X: Sparse matrix (n_poss x n_players) with +1 for defense
        Y: Points per possession (points allowed)
        sorted_players: List of player IDs
        sample_weights: Array of weights
        player_to_idx: Player to column index mapping
    """
    # Identify all unique players if not provided
    if player_to_idx is None:
        all_players = set()
        for col in ['off_lineup', 'def_lineup']:
            if df[col].empty: continue
            ids = [clean_id(pid) for lineup in df[col] if isinstance(lineup, (list, np.ndarray)) for pid in lineup]
            all_players.update(ids)
        
        all_players.discard('0')
        sorted_players = sorted(list(all_players))
        player_to_idx = {pid: i for i, pid in enumerate(sorted_players)}
    else:
        sorted_players = [pid for pid, _ in sorted(player_to_idx.items(), key=lambda x: x[1])]
    
    n_players = len(sorted_players)
    n_poss = len(df)
    
    # Only record defensive presence
    data = []
    rows = []
    cols = []
    
    def_lineups = df['def_lineup'].values
    
    for i in range(n_poss):
        defn = def_lineups[i]
        if isinstance(defn, (list, np.ndarray)):
            for pid in defn:
                pid_clean = clean_id(pid)
                if pid_clean in player_to_idx:
                    rows.append(i)
                    cols.append(player_to_idx[pid_clean])
                    data.append(1)
    
    X = csr_matrix((data, (rows, cols)), shape=(n_poss, n_players))
    Y = df['points'].values.astype(float)  # Points allowed by defense
    
    if 'duration_weight' in df.columns:
        sample_weights = df['duration_weight'].values
    else:
        sample_weights = np.ones(n_poss)
    
    return X, Y, sorted_players, sample_weights, player_to_idx


def run_rapm_for_season(df, season_name, multi_season=False, season_weight=1.0):
    """
    Run RAPM for a single season.
    
    Args:
        df: Possession data for the season
        season_name: Season identifier (e.g., "2024-25")
        multi_season: Whether this is part of multi-season pooling
        season_weight: Weight for this season's data
    
    Returns:
        DataFrame with RAPM results for this season
    """
    print(f"\n--- Processing Season: {season_name} ({len(df):,} possessions) ---")
    
    # 1. Build Matrix
    X, Y, player_ids, sample_weights, _ = build_sparse_matrix(df)
    print(f"   Matrix: {X.shape[0]} poss x {X.shape[1]} players")
    
    # Apply season weight to sample weights
    sample_weights = sample_weights * season_weight
    
    # 2. Fit Ridge with expanded alpha range
    # Higher alphas = more regularization = more shrinkage toward zero
    # Single-season RAPM is noisier, use higher alphas
    alphas = [500, 1000, 2000, 3000, 5000, 7500, 10000] 
    model = RidgeCV(alphas=alphas, fit_intercept=True)
    model.fit(X, Y, sample_weight=sample_weights)
    
    print(f"   ✅ Best Alpha: {model.alpha_}")
    
    # 3. Format Results (coefficients are per-possession, multiply by 100 for per-100-poss)
    results = []
    for pid, coef in zip(player_ids, model.coef_):
        results.append({
            "season": season_name,
            "player_id": pid,
            "RAPM": coef * 100,  # Scale to per 100 possessions
            "intercept": model.intercept_ * 100
        })
        
    return pd.DataFrame(results)


def run_orapm_drapm_for_season(df, season_name, season_weight=1.0):
    """
    Run separate ORAPM and DRAPM for a single season.
    
    Args:
        df: Possession data for the season
        season_name: Season identifier
        season_weight: Weight for sample weighting
    
    Returns:
        DataFrame with ORAPM and DRAPM results
    """
    print(f"\n--- ORAPM/DRAPM for {season_name} ---")
    
    alphas = [500, 1000, 2000, 3000, 5000, 7500, 10000]
    
    # ORAPM: Offensive impact
    X_off, Y_off, player_ids, sample_weights, player_to_idx = build_offensive_matrix(df)
    sample_weights_off = sample_weights * season_weight
    
    model_off = RidgeCV(alphas=alphas, fit_intercept=True)
    model_off.fit(X_off, Y_off, sample_weight=sample_weights_off)
    print(f"   ORAPM Alpha: {model_off.alpha_}")
    
    # DRAPM: Defensive impact (lower points allowed = better)
    X_def, Y_def, _, sample_weights_def, _ = build_defensive_matrix(df, player_to_idx)
    sample_weights_def = sample_weights_def * season_weight
    
    model_def = RidgeCV(alphas=alphas, fit_intercept=True)
    model_def.fit(X_def, Y_def, sample_weight=sample_weights_def)
    print(f"   DRAPM Alpha: {model_def.alpha_}")
    
    # Format results
    results = []
    for pid, coef_off, coef_def in zip(player_ids, model_off.coef_, model_def.coef_):
        # ORAPM: positive = good offense (more points scored)
        orapm = coef_off * 100
        # DRAPM: flip sign so positive = good defense (fewer points allowed)
        drapm = -coef_def * 100
        
        results.append({
            "season": season_name,
            "player_id": pid,
            "ORAPM": orapm,
            "DRAPM": drapm,
            "RAPM_type": "single_season_split"
        })
    
    return pd.DataFrame(results)


def run_multi_season_rapm(full_df, target_season, n_prior_seasons=2):
    """
    Run multi-season pooled RAPM for improved accuracy.
    
    Uses the target season plus prior seasons with decay weights.
    Research shows 3+ years of data significantly improves RAPM accuracy.
    
    Args:
        full_df: Full possession DataFrame with all seasons
        target_season: The season to calculate RAPM for (e.g., "2024-25")
        n_prior_seasons: Number of prior seasons to include
    
    Returns:
        DataFrame with pooled RAPM results
    """
    seasons = sorted(full_df['season'].unique())
    
    # Find target season index
    if target_season not in seasons:
        print(f"⚠️ Target season {target_season} not found")
        return pd.DataFrame()
    
    target_idx = seasons.index(target_season)
    
    # Collect seasons to use (target + prior)
    seasons_to_use = []
    for i in range(n_prior_seasons + 1):
        idx = target_idx - i
        if idx >= 0:
            seasons_to_use.append((seasons[idx], SEASON_DECAY_WEIGHTS.get(i, 0.2)))
    
    print(f"\n=== Multi-Season Pooled RAPM for {target_season} ===")
    print(f"   Using {len(seasons_to_use)} seasons: {[(s, f'{w:.0%}') for s, w in seasons_to_use]}")
    
    # Build global player index from all seasons
    all_players = set()
    for season, _ in seasons_to_use:
        season_df = full_df[full_df['season'] == season]
        for col in ['off_lineup', 'def_lineup']:
            ids = [clean_id(pid) for lineup in season_df[col] if isinstance(lineup, (list, np.ndarray)) for pid in lineup]
            all_players.update(ids)
    
    all_players.discard('0')
    sorted_players = sorted(list(all_players))
    player_to_idx = {pid: i for i, pid in enumerate(sorted_players)}
    n_players = len(sorted_players)
    
    print(f"   Total unique players: {n_players}")
    
    # Build combined matrices
    X_list = []
    Y_list = []
    weights_list = []
    total_poss = 0
    
    for season, season_weight in seasons_to_use:
        season_df = full_df[full_df['season'] == season].copy()
        X, Y, _, sample_weights, _ = build_sparse_matrix(season_df, player_to_idx)
        
        X_list.append(X)
        Y_list.append(Y)
        weights_list.append(sample_weights * season_weight)
        total_poss += len(season_df)
        
        print(f"   {season}: {len(season_df):,} possessions (weight: {season_weight:.0%})")
    
    # Stack matrices
    X_combined = vstack(X_list)
    Y_combined = np.concatenate(Y_list)
    weights_combined = np.concatenate(weights_list)
    
    print(f"   Combined: {X_combined.shape[0]:,} possessions x {X_combined.shape[1]} players")
    
    # Fit pooled model with wider alpha range
    # Multi-season data is more stable, can use lower alphas
    alphas = [100, 250, 500, 1000, 2000, 3000, 5000, 7500]
    model = RidgeCV(alphas=alphas, fit_intercept=True)
    model.fit(X_combined, Y_combined, sample_weight=weights_combined)
    
    print(f"   ✅ Best Alpha: {model.alpha_} (R²: {model.score(X_combined, Y_combined, sample_weight=weights_combined):.4f})")
    
    # Format results
    results = []
    for pid, coef in zip(sorted_players, model.coef_):
        results.append({
            "season": target_season,
            "player_id": pid,
            "RAPM": coef * 100,  # Scale to per 100 possessions
            "RAPM_type": "pooled",
            "n_seasons_pooled": len(seasons_to_use),
            "intercept": model.intercept_ * 100,
            "alpha": model.alpha_
        })
    
    return pd.DataFrame(results)


def run_multi_season_orapm_drapm(full_df, target_season, n_prior_seasons=2):
    """
    Run multi-season pooled ORAPM and DRAPM for improved accuracy.
    
    Uses the target season plus prior seasons with decay weights.
    
    Args:
        full_df: Full possession DataFrame with all seasons
        target_season: The season to calculate for
        n_prior_seasons: Number of prior seasons to include
    
    Returns:
        DataFrame with pooled ORAPM and DRAPM results
    """
    seasons = sorted(full_df['season'].unique())
    
    if target_season not in seasons:
        print(f"⚠️ Target season {target_season} not found")
        return pd.DataFrame()
    
    target_idx = seasons.index(target_season)
    
    # Collect seasons to use
    seasons_to_use = []
    for i in range(n_prior_seasons + 1):
        idx = target_idx - i
        if idx >= 0:
            seasons_to_use.append((seasons[idx], SEASON_DECAY_WEIGHTS.get(i, 0.2)))
    
    print(f"\n=== Multi-Season ORAPM/DRAPM for {target_season} ===")
    print(f"   Using {len(seasons_to_use)} seasons")
    
    # Build global player index
    all_players = set()
    for season, _ in seasons_to_use:
        season_df = full_df[full_df['season'] == season]
        for col in ['off_lineup', 'def_lineup']:
            ids = [clean_id(pid) for lineup in season_df[col] if isinstance(lineup, (list, np.ndarray)) for pid in lineup]
            all_players.update(ids)
    
    all_players.discard('0')
    sorted_players = sorted(list(all_players))
    player_to_idx = {pid: i for i, pid in enumerate(sorted_players)}
    
    # Build combined offensive matrices
    X_off_list = []
    Y_off_list = []
    weights_off_list = []
    
    # Build combined defensive matrices
    X_def_list = []
    Y_def_list = []
    weights_def_list = []
    
    for season, season_weight in seasons_to_use:
        season_df = full_df[full_df['season'] == season].copy()
        
        # Offensive
        X_off, Y_off, _, weights_off, _ = build_offensive_matrix(season_df, player_to_idx)
        X_off_list.append(X_off)
        Y_off_list.append(Y_off)
        weights_off_list.append(weights_off * season_weight)
        
        # Defensive
        X_def, Y_def, _, weights_def, _ = build_defensive_matrix(season_df, player_to_idx)
        X_def_list.append(X_def)
        Y_def_list.append(Y_def)
        weights_def_list.append(weights_def * season_weight)
    
    # Stack matrices
    X_off_combined = vstack(X_off_list)
    Y_off_combined = np.concatenate(Y_off_list)
    weights_off_combined = np.concatenate(weights_off_list)
    
    X_def_combined = vstack(X_def_list)
    Y_def_combined = np.concatenate(Y_def_list)
    weights_def_combined = np.concatenate(weights_def_list)
    
    alphas = [100, 250, 500, 1000, 2000, 3000, 5000, 7500]
    
    # Fit ORAPM
    model_off = RidgeCV(alphas=alphas, fit_intercept=True)
    model_off.fit(X_off_combined, Y_off_combined, sample_weight=weights_off_combined)
    print(f"   ORAPM Alpha: {model_off.alpha_}")
    
    # Fit DRAPM
    model_def = RidgeCV(alphas=alphas, fit_intercept=True)
    model_def.fit(X_def_combined, Y_def_combined, sample_weight=weights_def_combined)
    print(f"   DRAPM Alpha: {model_def.alpha_}")
    
    # Format results
    results = []
    for pid, coef_off, coef_def in zip(sorted_players, model_off.coef_, model_def.coef_):
        orapm = coef_off * 100
        drapm = -coef_def * 100  # Flip sign so positive = good defense
        
        results.append({
            "season": target_season,
            "player_id": pid,
            "ORAPM": orapm,
            "DRAPM": drapm,
            "RAPM_type": "pooled_split",
            "n_seasons_pooled": len(seasons_to_use)
        })
    
    return pd.DataFrame(results)


def enrich_names(rapm_df):
    """Attaches player names and calculates player-level minutes for context."""
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
                 rapm_df['player_name'] = rapm_df['player_id']
        else:
            rapm_df['player_name'] = rapm_df['player_id']
    except Exception as e:
        print(f"⚠️ Could not load names: {e}")
        rapm_df['player_name'] = rapm_df['player_id']
    
    return rapm_df


def calculate_player_possessions(full_df):
    """Calculate number of possessions played by each player per season."""
    player_poss = {}
    
    for _, row in full_df.iterrows():
        season = row['season']
        
        # Count offensive possessions
        if isinstance(row['off_lineup'], (list, np.ndarray)):
            for pid in row['off_lineup']:
                pid_clean = clean_id(pid)
                key = (season, pid_clean)
                player_poss[key] = player_poss.get(key, 0) + 1
        
        # Count defensive possessions
        if isinstance(row['def_lineup'], (list, np.ndarray)):
            for pid in row['def_lineup']:
                pid_clean = clean_id(pid)
                key = (season, pid_clean)
                player_poss[key] = player_poss.get(key, 0) + 1
    
    # Divide by 2 since player appears on both off and def sides (counted twice)
    # Actually, each row is one possession from one team's perspective
    # A player appears in off_lineup when their team has the ball, def_lineup when opponent has it
    # So we count each appearance as 0.5 possessions (since possession = both sides)
    poss_df = pd.DataFrame([
        {'season': k[0], 'player_id': k[1], 'possessions_played': v}
        for k, v in player_poss.items()
    ])
    
    return poss_df


def main():
    """Main RAPM calculation pipeline."""
    print("=" * 60)
    print("RAPM (Regularized Adjusted Plus-Minus) Calculation")
    print("=" * 60)
    
    # 1. Load All Data
    full_df = load_clean_possessions()
    if full_df.empty:
        print("❌ No possession data found")
        return
    
    print(f"\n📊 Total: {len(full_df):,} possessions loaded")
    
    # 2. Get available seasons
    seasons = sorted(full_df['season'].unique())
    print(f"📅 Seasons available: {seasons}")
    
    all_results = []
    
    # 3. Calculate both single-season and pooled RAPM for each season
    for season in seasons:
        # Single-season RAPM
        print(f"\n{'='*50}")
        print(f"SINGLE-SEASON RAPM: {season}")
        print(f"{'='*50}")
        
        season_df = full_df[full_df['season'] == season].copy()
        season_rapm = run_rapm_for_season(season_df, season)
        season_rapm['RAPM_type'] = 'single_season'
        season_rapm['n_seasons_pooled'] = 1
        all_results.append(season_rapm)
        
        # Multi-season pooled RAPM (only if we have prior seasons)
        if len(seasons) > 1:
            print(f"\n{'='*50}")
            print(f"POOLED RAPM: {season}")
            print(f"{'='*50}")
            
            pooled_rapm = run_multi_season_rapm(full_df, season, n_prior_seasons=2)
            if not pooled_rapm.empty:
                all_results.append(pooled_rapm)
            
            # Pooled ORAPM/DRAPM
            print(f"\n{'='*50}")
            print(f"POOLED ORAPM/DRAPM: {season}")
            print(f"{'='*50}")
            
            pooled_od = run_multi_season_orapm_drapm(full_df, season, n_prior_seasons=2)
            if not pooled_od.empty:
                all_results.append(pooled_od)
        
        # Single-season ORAPM/DRAPM
        print(f"\n{'='*50}")
        print(f"SINGLE-SEASON ORAPM/DRAPM: {season}")
        print(f"{'='*50}")
        
        season_od = run_orapm_drapm_for_season(season_df, season)
        all_results.append(season_od)
    
    final_df = pd.concat(all_results, ignore_index=True)
    
    # 4. Calculate possession counts for context
    print("\n📊 Calculating possession counts...")
    poss_counts = calculate_player_possessions(full_df)
    
    # Merge possession counts
    final_df = final_df.merge(
        poss_counts, 
        on=['season', 'player_id'], 
        how='left'
    )
    final_df['possessions_played'] = final_df['possessions_played'].fillna(0)
    
    # 5. Enrich with names
    final_df = enrich_names(final_df)
    
    # 6. Sort and organize - handle rows with and without RAPM column
    if 'RAPM' in final_df.columns:
        # Fill NaN for RAPM where it doesn't exist (O/D split rows)
        final_df['RAPM'] = final_df['RAPM'].fillna(0)
        final_df = final_df.sort_values(
            ['season', 'RAPM_type', 'RAPM'], 
            ascending=[True, True, False]
        )
    else:
        final_df = final_df.sort_values(['season', 'RAPM_type'], ascending=[True, True])
    
    # 7. Save results
    out_path = os.path.join(OUTPUT_DIR, "player_rapm.parquet")
    final_df.to_parquet(out_path, index=False)
    print(f"\n✅ RAPM saved to {out_path}")
    
    # Also save CSV for easy inspection
    csv_path = os.path.join(OUTPUT_DIR, "player_rapm.csv")
    final_df.to_csv(csv_path, index=False)
    print(f"✅ RAPM CSV saved to {csv_path}")
    
    # 8. Show summary statistics
    print("\n" + "=" * 60)
    print("RAPM SUMMARY")
    print("=" * 60)
    
    for season in seasons:
        print(f"\n📅 {season}:")
        
        # Single-season top 5
        single = final_df[(final_df['season'] == season) & (final_df['RAPM_type'] == 'single_season')]
        if not single.empty:
            print(f"\n   Top 5 Single-Season RAPM:")
            top5 = single.nlargest(5, 'RAPM')[['player_name', 'RAPM', 'possessions_played']]
            for _, row in top5.iterrows():
                print(f"      {row['player_name']:25s}: {row['RAPM']:+6.2f} ({int(row['possessions_played']):,} poss)")
        
        # Pooled top 5
        pooled = final_df[(final_df['season'] == season) & (final_df['RAPM_type'] == 'pooled')]
        if not pooled.empty:
            print(f"\n   Top 5 Pooled RAPM:")
            top5 = pooled.nlargest(5, 'RAPM')[['player_name', 'RAPM', 'possessions_played']]
            for _, row in top5.iterrows():
                print(f"      {row['player_name']:25s}: {row['RAPM']:+6.2f} ({int(row['possessions_played']):,} poss)")
        
        # Pooled ORAPM/DRAPM top 5
        pooled_od = final_df[(final_df['season'] == season) & (final_df['RAPM_type'] == 'pooled_split')]
        if not pooled_od.empty and 'ORAPM' in pooled_od.columns:
            print(f"\n   Top 5 Pooled ORAPM (Offense):")
            top5_o = pooled_od.nlargest(5, 'ORAPM')[['player_name', 'ORAPM', 'possessions_played']]
            for _, row in top5_o.iterrows():
                print(f"      {row['player_name']:25s}: {row['ORAPM']:+6.2f} ({int(row['possessions_played']):,} poss)")
            
            print(f"\n   Top 5 Pooled DRAPM (Defense):")
            top5_d = pooled_od.nlargest(5, 'DRAPM')[['player_name', 'DRAPM', 'possessions_played']]
            for _, row in top5_d.iterrows():
                print(f"      {row['player_name']:25s}: {row['DRAPM']:+6.2f} ({int(row['possessions_played']):,} poss)")
    
    # 9. Validate against expected ranges
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)
    
    # RAPM should typically range from -5 to +8 for established players
    pooled_only = final_df[final_df['RAPM_type'] == 'pooled']
    if not pooled_only.empty and 'RAPM' in pooled_only.columns:
        min_rapm = pooled_only['RAPM'].min()
        max_rapm = pooled_only['RAPM'].max()
        mean_rapm = pooled_only['RAPM'].mean()
        std_rapm = pooled_only['RAPM'].std()
        
        print(f"   Pooled RAPM range: [{min_rapm:.2f}, {max_rapm:.2f}]")
        print(f"   Mean: {mean_rapm:.2f}, Std: {std_rapm:.2f}")
        
        # Expected: mean near 0, std around 2-3
        if abs(mean_rapm) < 0.5 and 1.0 < std_rapm < 5.0:
            print("   ✅ Distribution looks reasonable")
        else:
            print("   ⚠️ Distribution may need review")
    
    # Validate ORAPM/DRAPM
    pooled_od_val = final_df[final_df['RAPM_type'] == 'pooled_split']
    if not pooled_od_val.empty and 'ORAPM' in pooled_od_val.columns:
        print(f"\n   Pooled ORAPM range: [{pooled_od_val['ORAPM'].min():.2f}, {pooled_od_val['ORAPM'].max():.2f}]")
        print(f"   Pooled DRAPM range: [{pooled_od_val['DRAPM'].min():.2f}, {pooled_od_val['DRAPM'].max():.2f}]")
        print(f"   ORAPM Mean: {pooled_od_val['ORAPM'].mean():.2f}, Std: {pooled_od_val['ORAPM'].std():.2f}")
        print(f"   DRAPM Mean: {pooled_od_val['DRAPM'].mean():.2f}, Std: {pooled_od_val['DRAPM'].std():.2f}")


if __name__ == "__main__":
    main()