"""
src/data_compute/compute_xrapm.py
Computes xRAPM (Extended RAPM) with Box-Score Bayesian Prior.

This implements an ESPN RPM-like approach:
1. Calculate a BPM-like prior from box score stats
2. Use this prior to inform the ridge regression (Bayesian approach)
3. Result: xRAPM = RAPM + Box-Score Prior adjustment

The key insight: pure RAPM is noisy due to collinearity. A box-score prior
helps anchor player estimates to their statistical production, especially
for players with fewer possessions.

Methodology:
- Compute simplified BPM prior: offensive/defensive contribution from box stats
- Use sklearn BayesianRidge with prior mean set to BPM estimate
- This shrinks RAPM estimates toward box-score expectation
"""

import pandas as pd
import numpy as np
import glob
import os
import sys
from scipy.sparse import csr_matrix, vstack, hstack
from sklearn.linear_model import Ridge, RidgeCV
from scipy import sparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

DATA_DIR = "data/historical"
OUTPUT_DIR = "data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEASON_DECAY_WEIGHTS = {
    0: 1.0,
    1: 0.65,
    2: 0.40,
    3: 0.25,
}

# BPM coefficients (simplified version based on Basketball-Reference methodology)
# These approximate the contribution of each stat to points per 100 possessions
BPM_COEFFICIENTS = {
    # Offensive stats (positive contribution)
    'PTS': 0.0,       # Points already captured in RAPM target
    'AST': 0.5,       # Assists create ~1 pt each, but shared credit
    'OREB': 0.5,      # Offensive rebounds create extra possessions
    'FG3M': 0.3,      # 3PT bonus beyond points (spacing value)
    'FTM': 0.1,       # Free throws (slight bonus for drawing fouls)
    # Defensive stats (positive = good defense)
    'STL': 1.0,       # Steals end possessions + fast break
    'BLK': 0.5,       # Blocks (some recovered by offense)
    'DREB': 0.2,      # Defensive rebounds end opponent possessions
    # Negative stats
    'TOV': -1.0,      # Turnovers lose possessions
}

# Prior method: 'box_score' (original) or 'hybrid' (NBA advanced stats + box score)
PRIOR_METHOD = 'box_score'


def clean_id(val):
    if pd.isna(val): return "0"
    return str(val).replace(".0", "")


def load_clean_possessions():
    """Load all clean possessions data."""
    files = sorted(glob.glob(os.path.join(DATA_DIR, "possessions_clean_*.parquet")))
    if not files:
        print("❌ No clean possession files found")
        return pd.DataFrame()
    
    print(f"Loading {len(files)} possession files...")
    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        if 'season' not in df.columns:
            base = os.path.basename(f)
            season = base.replace("possessions_clean_", "").replace(".parquet", "")
            df['season'] = season
        dfs.append(df)
    
    full_df = pd.concat(dfs, ignore_index=True)
    full_df = calculate_possession_duration(full_df)
    return full_df


def parse_clock_seconds(clock_str):
    try:
        if pd.isna(clock_str): return 0
        parts = str(clock_str).split(':')
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return 0
    except:
        return 0


def calculate_possession_duration(df):
    df['start_seconds'] = df['start_clock'].apply(parse_clock_seconds)
    df['end_seconds'] = df['end_clock'].apply(parse_clock_seconds)
    df['duration'] = (df['start_seconds'] - df['end_seconds']).clip(lower=1, upper=60)
    df['duration_weight'] = np.sqrt(df['duration'])
    return df


def load_box_score_data():
    """Load player box score data for BPM prior calculation."""
    # Try new complete stats file first (already aggregated per season)
    complete_path = os.path.join(DATA_DIR, "complete_player_season_stats.parquet")
    if os.path.exists(complete_path):
        df = pd.read_parquet(complete_path)
        # Standardize column names
        if 'PLAYER_ID' in df.columns:
            df['player_id'] = df['PLAYER_ID'].astype(str).apply(clean_id)
        if 'SEASON' in df.columns:
            df['season'] = df['SEASON']
        print(f"✅ Loaded complete stats: {len(df)} player-seasons")
        return df
    
    # Fallback to old game logs
    path = os.path.join(DATA_DIR, "final_player_game_logs.parquet")
    if not os.path.exists(path):
        print("⚠️ No box score data found, using zero priors")
        return None
    
    df = pd.read_parquet(path)
    
    # Standardize column names
    if 'Player_ID' in df.columns:
        df['player_id'] = df['Player_ID'].astype(str).apply(clean_id)
    if 'SEASON' in df.columns:
        df['season'] = df['SEASON']
    
    return df


def compute_bpm_prior(box_df, season):
    """
    Compute simplified BPM-like prior for each player.
    
    Returns dict: player_id -> (offensive_prior, defensive_prior)
    """
    if box_df is None:
        return {}
    
    # Filter to season
    season_df = box_df[box_df['season'] == season].copy()
    if season_df.empty:
        return {}
    
    # Check if data is already aggregated (complete_player_season_stats)
    # or needs aggregation (game logs)
    is_aggregated = 'GP' in season_df.columns  # Season totals have GP
    
    if is_aggregated:
        # Data is already per-player season totals
        player_stats = season_df.copy()
        player_stats = player_stats.rename(columns={'player_id': 'player_id'})
    else:
        # Aggregate per player (game logs)
        agg_cols = ['MIN', 'PTS', 'AST', 'OREB', 'DREB', 'STL', 'BLK', 'TOV', 'FG3M', 'FTM', 'FGA']
        available = [c for c in agg_cols if c in season_df.columns]
        player_stats = season_df.groupby('player_id')[available].sum().reset_index()
    
    # Need minutes for per-possession normalization
    if 'MIN' not in player_stats.columns or player_stats['MIN'].sum() == 0:
        return {}
    
    # Estimate possessions from minutes (rough: ~1 possession per 24 seconds)
    player_stats['est_poss'] = player_stats['MIN'] * 2.5  # ~2.5 poss/min per player
    player_stats = player_stats[player_stats['est_poss'] > 100]  # Min sample
    
    priors = {}
    
    for _, row in player_stats.iterrows():
        pid = row['player_id']
        poss = row['est_poss']
        
        # Compute offensive contribution per 100 possessions
        o_value = 0
        for stat, coef in BPM_COEFFICIENTS.items():
            if stat in row and stat in ['AST', 'OREB', 'FG3M', 'FTM', 'TOV']:
                o_value += (row[stat] / poss) * 100 * coef
        
        # Compute defensive contribution per 100 possessions
        d_value = 0
        for stat, coef in BPM_COEFFICIENTS.items():
            if stat in row and stat in ['STL', 'BLK', 'DREB']:
                d_value += (row[stat] / poss) * 100 * coef
        
        # Combined prior (simple average)
        total_prior = o_value + d_value
        
        priors[pid] = {
            'orapm_prior': o_value,
            'drapm_prior': d_value,
            'total_prior': total_prior,
            'minutes': row['MIN'],
        }
    
    return priors


def compute_hybrid_prior(box_df, season):
    """
    Compute hybrid prior using NBA advanced stats + box score components.
    
    Option D: Combines:
    - NET_RATING (on-court impact, scaled)
    - PIE (player impact estimate)
    - Box score production (AST, STL, BLK, etc.)
    - Efficiency adjustments (TS% vs league avg)
    
    Returns dict: player_id -> prior values
    """
    if box_df is None:
        return {}
    
    season_df = box_df[box_df['season'] == season].copy()
    if season_df.empty:
        return {}
    
    # Helper to safely get numeric value
    def safe_get(row, col, default=0):
        val = row.get(col, default)
        if pd.isna(val):
            return default
        return float(val)
    
    # League averages for normalization
    min_minutes = 200  # Minimum minutes for reliable stats
    qualified = season_df[season_df['MIN'] >= min_minutes]
    
    if len(qualified) == 0:
        return {}
    
    # League averages (handle NaN)
    lg_net_rating = qualified['NET_RATING'].mean() if 'NET_RATING' in qualified.columns else 0
    lg_ts_pct = qualified['TS_PCT'].mean() if 'TS_PCT' in qualified.columns else 0.55
    lg_pie = qualified['PIE'].mean() if 'PIE' in qualified.columns else 0.10
    
    if pd.isna(lg_net_rating): lg_net_rating = 0
    if pd.isna(lg_ts_pct): lg_ts_pct = 0.55
    if pd.isna(lg_pie): lg_pie = 0.10
    
    priors = {}
    
    for _, row in season_df.iterrows():
        pid = row['player_id']
        minutes = safe_get(row, 'MIN', 0)
        
        if minutes < 50:  # Very low minute players get zero prior
            priors[pid] = {'orapm_prior': 0.0, 'drapm_prior': 0.0, 'total_prior': 0.0, 'minutes': minutes}
            continue
        
        # === Component 1: NET_RATING (scaled to ~BPM range) ===
        net_rating = safe_get(row, 'NET_RATING', lg_net_rating)
        net_component = (net_rating - lg_net_rating) * 0.5
        
        # === Component 2: PIE (Player Impact Estimate) ===
        pie = safe_get(row, 'PIE', lg_pie)
        pie_component = (pie - lg_pie) * 40
        
        # === Component 3: Box Score Production (per 100 poss) ===
        est_poss = minutes * 2.5
        if est_poss > 0:
            ast_rate = safe_get(row, 'AST', 0) / est_poss * 100
            stl_rate = safe_get(row, 'STL', 0) / est_poss * 100
            blk_rate = safe_get(row, 'BLK', 0) / est_poss * 100
            tov_rate = safe_get(row, 'TOV', 0) / est_poss * 100
            oreb_rate = safe_get(row, 'OREB', 0) / est_poss * 100
            dreb_rate = safe_get(row, 'DREB', 0) / est_poss * 100
            
            o_box = ast_rate * 0.4 + oreb_rate * 0.4 - tov_rate * 0.8
            d_box = stl_rate * 0.8 + blk_rate * 0.5 + dreb_rate * 0.15
        else:
            o_box, d_box = 0.0, 0.0
        
        # === Component 4: Efficiency Adjustment ===
        ts_pct = safe_get(row, 'TS_PCT', lg_ts_pct)
        usg_pct = safe_get(row, 'USG_PCT', 0.20)
        efficiency_adj = (ts_pct - lg_ts_pct) * usg_pct * 30
        
        # === Combine Components ===
        total_prior = (
            net_component * 0.35 +
            pie_component * 0.25 +
            (o_box + d_box) * 0.25 +
            efficiency_adj * 0.15
        )
        
        # Split O/D using OFF_RATING/DEF_RATING
        off_rating = safe_get(row, 'OFF_RATING', 110)
        def_rating = safe_get(row, 'DEF_RATING', 110)
        
        off_share = (off_rating - 100) / 20
        def_share = (110 - def_rating) / 20
        
        if off_share + def_share != 0:
            o_ratio = max(0.2, min(0.8, 0.5 + (off_share - def_share) * 0.2))
        else:
            o_ratio = 0.5
        
        o_prior = total_prior * o_ratio + o_box * 0.3
        d_prior = total_prior * (1 - o_ratio) + d_box * 0.3
        
        # Final NaN check
        if np.isnan(o_prior): o_prior = 0.0
        if np.isnan(d_prior): d_prior = 0.0
        if np.isnan(total_prior): total_prior = 0.0
        
        priors[pid] = {
            'orapm_prior': float(o_prior),
            'drapm_prior': float(d_prior),
            'total_prior': float(total_prior),
            'minutes': float(minutes),
        }
    
    return priors


def build_sparse_matrix(df, player_to_idx=None):
    """Build X matrix with +1 offense, -1 defense."""
    if player_to_idx is None:
        all_players = set()
        for col in ['off_lineup', 'def_lineup']:
            ids = [clean_id(pid) for lineup in df[col] if isinstance(lineup, (list, np.ndarray)) for pid in lineup]
            all_players.update(ids)
        all_players.discard('0')
        sorted_players = sorted(list(all_players))
        player_to_idx = {pid: i for i, pid in enumerate(sorted_players)}
    else:
        sorted_players = [pid for pid, _ in sorted(player_to_idx.items(), key=lambda x: x[1])]
    
    n_players = len(sorted_players)
    n_poss = len(df)
    
    data, rows, cols = [], [], []
    off_lineups = df['off_lineup'].values
    def_lineups = df['def_lineup'].values
    
    for i in range(n_poss):
        off = off_lineups[i]
        if isinstance(off, (list, np.ndarray)):
            for pid in off:
                pid_clean = clean_id(pid)
                if pid_clean in player_to_idx:
                    rows.append(i)
                    cols.append(player_to_idx[pid_clean])
                    data.append(1)
        
        defn = def_lineups[i]
        if isinstance(defn, (list, np.ndarray)):
            for pid in defn:
                pid_clean = clean_id(pid)
                if pid_clean in player_to_idx:
                    rows.append(i)
                    cols.append(player_to_idx[pid_clean])
                    data.append(-1)
    
    X = csr_matrix((data, (rows, cols)), shape=(n_poss, n_players))
    Y = df['points'].values.astype(float)
    
    sample_weights = df['duration_weight'].values if 'duration_weight' in df.columns else np.ones(n_poss)
    
    return X, Y, sorted_players, sample_weights, player_to_idx


def run_xrapm_with_prior(full_df, target_season, box_df, n_prior_seasons=2):
    """
    Run xRAPM with Bayesian box-score prior.
    
    The approach: augment the regression with pseudo-observations that
    encode our prior belief (BPM estimates). This is equivalent to
    Bayesian ridge regression with informative prior.
    """
    seasons = sorted(full_df['season'].unique())
    if target_season not in seasons:
        return pd.DataFrame()
    
    target_idx = seasons.index(target_season)
    
    # Collect seasons
    seasons_to_use = []
    for i in range(n_prior_seasons + 1):
        idx = target_idx - i
        if idx >= 0:
            seasons_to_use.append((seasons[idx], SEASON_DECAY_WEIGHTS.get(i, 0.2)))
    
    prior_type = "Hybrid" if PRIOR_METHOD == 'hybrid' else "BPM"
    print(f"\n=== xRAPM with {prior_type} Prior: {target_season} ===")
    print(f"   Seasons: {[s for s, _ in seasons_to_use]}")
    
    # Compute priors for target season (hybrid or box score)
    if PRIOR_METHOD == 'hybrid':
        bpm_priors = compute_hybrid_prior(box_df, target_season)
    else:
        bpm_priors = compute_bpm_prior(box_df, target_season)
    print(f"   {prior_type} priors computed for {len(bpm_priors)} players")
    
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
    n_players = len(sorted_players)
    
    print(f"   Total players: {n_players}")
    
    # Build combined possession matrices
    X_list, Y_list, weights_list = [], [], []
    
    for season, season_weight in seasons_to_use:
        season_df = full_df[full_df['season'] == season].copy()
        X, Y, _, sample_weights, _ = build_sparse_matrix(season_df, player_to_idx)
        X_list.append(X)
        Y_list.append(Y)
        weights_list.append(sample_weights * season_weight)
    
    X_poss = vstack(X_list)
    Y_poss = np.concatenate(Y_list)
    weights_poss = np.concatenate(weights_list)
    
    print(f"   Possessions: {X_poss.shape[0]:,}")
    
    # === BAYESIAN PRIOR AUGMENTATION ===
    # Add pseudo-observations that encode prior beliefs
    # For each player, add a "prior observation" that says:
    #   "player i's coefficient should be close to their BPM prior"
    #
    # This is done by adding rows to X and Y:
    #   X_prior[i, i] = prior_strength
    #   Y_prior[i] = bpm_prior[i] * prior_strength
    #
    # The prior_strength controls how much we trust the box score prior
    # Higher = more weight on box scores, lower = more weight on RAPM
    
    PRIOR_STRENGTH = 50.0  # Equivalent to ~50 pseudo-possessions of evidence
    
    # Build prior matrix (identity for players with priors)
    prior_data, prior_rows, prior_cols, prior_Y = [], [], [], []
    
    for pid, prior_info in bpm_priors.items():
        if pid in player_to_idx:
            idx = player_to_idx[pid]
            prior_data.append(PRIOR_STRENGTH)
            prior_rows.append(len(prior_Y))
            prior_cols.append(idx)
            # Prior target: BPM estimate (already per 100 poss, divide by 100 for per-poss)
            prior_Y.append(prior_info['total_prior'] / 100.0 * PRIOR_STRENGTH)
    
    n_priors = len(prior_Y)
    X_prior = csr_matrix((prior_data, (prior_rows, prior_cols)), shape=(n_priors, n_players))
    Y_prior = np.array(prior_Y)
    weights_prior = np.ones(n_priors)  # Equal weight for prior observations
    
    print(f"   Prior observations: {n_priors}")
    
    # Combine real possessions with prior pseudo-observations
    X_combined = vstack([X_poss, X_prior])
    Y_combined = np.concatenate([Y_poss, Y_prior])
    weights_combined = np.concatenate([weights_poss, weights_prior])
    
    # Fit ridge regression with LOWER alpha (priors provide regularization)
    # We can use lower alpha because the prior helps with collinearity
    alphas = [10, 25, 50, 100, 250, 500, 1000, 2000]
    model = RidgeCV(alphas=alphas, fit_intercept=True)
    model.fit(X_combined, Y_combined, sample_weight=weights_combined)
    
    print(f"   Best Alpha: {model.alpha_}")
    
    # Extract coefficients
    results = []
    for pid, coef in zip(sorted_players, model.coef_):
        prior_info = bpm_priors.get(pid, {'total_prior': 0, 'orapm_prior': 0, 'drapm_prior': 0})
        results.append({
            "season": target_season,
            "player_id": pid,
            "xRAPM": coef * 100,
            "BPM_prior": prior_info['total_prior'],
            "xRAPM_type": "pooled_with_prior",
            "n_seasons_pooled": len(seasons_to_use),
            "alpha": model.alpha_,
        })
    
    return pd.DataFrame(results)


def run_xrapm_split_with_prior(full_df, target_season, box_df, n_prior_seasons=2):
    """Run separate ORAPM and DRAPM with priors."""
    seasons = sorted(full_df['season'].unique())
    if target_season not in seasons:
        return pd.DataFrame()
    
    target_idx = seasons.index(target_season)
    seasons_to_use = []
    for i in range(n_prior_seasons + 1):
        idx = target_idx - i
        if idx >= 0:
            seasons_to_use.append((seasons[idx], SEASON_DECAY_WEIGHTS.get(i, 0.2)))
    
    bpm_priors = compute_bpm_prior(box_df, target_season)
    
    # Build player index
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
    
    PRIOR_STRENGTH = 50.0
    alphas = [10, 25, 50, 100, 250, 500, 1000]
    
    # === OFFENSIVE xRAPM ===
    X_off_list, Y_off_list, w_off_list = [], [], []
    for season, sw in seasons_to_use:
        sdf = full_df[full_df['season'] == season].copy()
        data, rows, cols = [], [], []
        for i, lineup in enumerate(sdf['off_lineup'].values):
            if isinstance(lineup, (list, np.ndarray)):
                for pid in lineup:
                    pc = clean_id(pid)
                    if pc in player_to_idx:
                        rows.append(i)
                        cols.append(player_to_idx[pc])
                        data.append(1)
        X = csr_matrix((data, (rows, cols)), shape=(len(sdf), n_players))
        Y = sdf['points'].values.astype(float)
        w = sdf['duration_weight'].values if 'duration_weight' in sdf.columns else np.ones(len(sdf))
        X_off_list.append(X)
        Y_off_list.append(Y)
        w_off_list.append(w * sw)
    
    X_off = vstack(X_off_list)
    Y_off = np.concatenate(Y_off_list)
    w_off = np.concatenate(w_off_list)
    
    # Add offensive priors
    prior_data, prior_rows, prior_Y = [], [], []
    for pid, info in bpm_priors.items():
        if pid in player_to_idx:
            prior_data.append(PRIOR_STRENGTH)
            prior_rows.append(len(prior_Y))
            prior_Y.append(info['orapm_prior'] / 100.0 * PRIOR_STRENGTH)
    X_off_prior = csr_matrix((prior_data, (prior_rows, [player_to_idx[p] for p in bpm_priors if p in player_to_idx])), 
                              shape=(len(prior_Y), n_players))
    
    X_off_combined = vstack([X_off, X_off_prior])
    Y_off_combined = np.concatenate([Y_off, np.array(prior_Y)])
    w_off_combined = np.concatenate([w_off, np.ones(len(prior_Y))])
    
    model_off = RidgeCV(alphas=alphas, fit_intercept=True)
    model_off.fit(X_off_combined, Y_off_combined, sample_weight=w_off_combined)
    
    # === DEFENSIVE xRAPM ===
    X_def_list, Y_def_list, w_def_list = [], [], []
    for season, sw in seasons_to_use:
        sdf = full_df[full_df['season'] == season].copy()
        data, rows, cols = [], [], []
        for i, lineup in enumerate(sdf['def_lineup'].values):
            if isinstance(lineup, (list, np.ndarray)):
                for pid in lineup:
                    pc = clean_id(pid)
                    if pc in player_to_idx:
                        rows.append(i)
                        cols.append(player_to_idx[pc])
                        data.append(1)
        X = csr_matrix((data, (rows, cols)), shape=(len(sdf), n_players))
        Y = sdf['points'].values.astype(float)
        w = sdf['duration_weight'].values if 'duration_weight' in sdf.columns else np.ones(len(sdf))
        X_def_list.append(X)
        Y_def_list.append(Y)
        w_def_list.append(w * sw)
    
    X_def = vstack(X_def_list)
    Y_def = np.concatenate(Y_def_list)
    w_def = np.concatenate(w_def_list)
    
    # Add defensive priors
    prior_data_d, prior_rows_d, prior_Y_d = [], [], []
    for pid, info in bpm_priors.items():
        if pid in player_to_idx:
            prior_data_d.append(PRIOR_STRENGTH)
            prior_rows_d.append(len(prior_Y_d))
            # For defense, positive prior = allows fewer points
            prior_Y_d.append(-info['drapm_prior'] / 100.0 * PRIOR_STRENGTH)
    X_def_prior = csr_matrix((prior_data_d, (prior_rows_d, [player_to_idx[p] for p in bpm_priors if p in player_to_idx])),
                              shape=(len(prior_Y_d), n_players))
    
    X_def_combined = vstack([X_def, X_def_prior])
    Y_def_combined = np.concatenate([Y_def, np.array(prior_Y_d)])
    w_def_combined = np.concatenate([w_def, np.ones(len(prior_Y_d))])
    
    model_def = RidgeCV(alphas=alphas, fit_intercept=True)
    model_def.fit(X_def_combined, Y_def_combined, sample_weight=w_def_combined)
    
    print(f"   O-xRAPM Alpha: {model_off.alpha_}, D-xRAPM Alpha: {model_def.alpha_}")
    
    # Format results
    results = []
    for pid, coef_o, coef_d in zip(sorted_players, model_off.coef_, model_def.coef_):
        prior_info = bpm_priors.get(pid, {'orapm_prior': 0, 'drapm_prior': 0})
        results.append({
            "season": target_season,
            "player_id": pid,
            "O_xRAPM": coef_o * 100,
            "D_xRAPM": -coef_d * 100,  # Flip so positive = good defense
            "O_prior": prior_info['orapm_prior'],
            "D_prior": prior_info['drapm_prior'],
            "xRAPM_type": "pooled_split_with_prior",
        })
    
    return pd.DataFrame(results)


def enrich_names(df):
    """Add player names."""
    try:
        p_path = os.path.join(DATA_DIR, "players.parquet")
        if os.path.exists(p_path):
            meta = pd.read_parquet(p_path)
            if 'id' in meta.columns:
                meta['id'] = meta['id'].astype(str).apply(clean_id)
                name_map = meta.set_index('id')['full_name'].to_dict()
                df['player_name'] = df['player_id'].map(name_map).fillna("Unknown")
            else:
                df['player_name'] = df['player_id']
        else:
            df['player_name'] = df['player_id']
    except Exception as e:
        print(f"⚠️ Could not load names: {e}")
        df['player_name'] = df['player_id']
    return df


def main():
    print("=" * 60)
    print("xRAPM (RAPM with Box-Score Bayesian Prior)")
    print("=" * 60)
    
    # Load data
    full_df = load_clean_possessions()
    if full_df.empty:
        print("❌ No possession data")
        return
    
    box_df = load_box_score_data()
    
    print(f"\n📊 Possessions: {len(full_df):,}")
    seasons = sorted(full_df['season'].unique())
    print(f"📅 Seasons: {seasons}")
    
    all_results = []
    
    for season in seasons:
        # Total xRAPM
        xrapm = run_xrapm_with_prior(full_df, season, box_df, n_prior_seasons=2)
        if not xrapm.empty:
            all_results.append(xrapm)
        
        # Split O/D xRAPM
        xrapm_split = run_xrapm_split_with_prior(full_df, season, box_df, n_prior_seasons=2)
        if not xrapm_split.empty:
            all_results.append(xrapm_split)
    
    if not all_results:
        print("❌ No results generated")
        return
    
    final_df = pd.concat(all_results, ignore_index=True)
    final_df = enrich_names(final_df)
    
    # Save
    out_path = os.path.join(OUTPUT_DIR, "player_xrapm.parquet")
    final_df.to_parquet(out_path, index=False)
    print(f"\n✅ xRAPM saved to {out_path}")
    
    csv_path = os.path.join(OUTPUT_DIR, "player_xrapm.csv")
    final_df.to_csv(csv_path, index=False)
    print(f"✅ CSV saved to {csv_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("xRAPM SUMMARY")
    print("=" * 60)
    
    for season in seasons:
        pooled = final_df[(final_df['season'] == season) & (final_df['xRAPM_type'] == 'pooled_with_prior')]
        if not pooled.empty:
            print(f"\n📅 {season} - Top 10 xRAPM:")
            top10 = pooled.nlargest(10, 'xRAPM')[['player_name', 'xRAPM', 'BPM_prior']]
            for i, (_, row) in enumerate(top10.iterrows(), 1):
                print(f"   {i:2d}. {row['player_name']:25s} {row['xRAPM']:+6.2f} (prior: {row['BPM_prior']:+5.2f})")
        
        split = final_df[(final_df['season'] == season) & (final_df['xRAPM_type'] == 'pooled_split_with_prior')]
        if not split.empty and 'O_xRAPM' in split.columns:
            print(f"\n   Top 5 O-xRAPM:")
            top5_o = split.nlargest(5, 'O_xRAPM')[['player_name', 'O_xRAPM']]
            for _, row in top5_o.iterrows():
                print(f"      {row['player_name']:25s} {row['O_xRAPM']:+6.2f}")
            
            print(f"\n   Top 5 D-xRAPM:")
            top5_d = split.nlargest(5, 'D_xRAPM')[['player_name', 'D_xRAPM']]
            for _, row in top5_d.iterrows():
                print(f"      {row['player_name']:25s} {row['D_xRAPM']:+6.2f}")


if __name__ == "__main__":
    main()
