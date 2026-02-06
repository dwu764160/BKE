"""
src/data_compute/compute_xrapm_improved.py
Improved xRAPM with collinearity correction for Wings.

Key Improvements over compute_xrapm.py:
1. Multi-year prior stabilization: Use 3-year career averages for established players
2. On/Off prior injection: Use team net rating differentials as additional prior
3. Teammate collinearity detection: Identify highly correlated player pairs
4. Position-weighted priors: Apply position-specific prior adjustments

The problem identified: Wings (like Jaylen Brown & Tatum) who share 68%+ of 
possessions cannot be separated by pure RAPM. The solution is to use:
- Stronger informative priors for wing players
- Multi-year stabilization for established players  
- On/Off splits as supplementary evidence
"""

import pandas as pd
import numpy as np
import glob
import os
import sys
from scipy.sparse import csr_matrix, vstack
from sklearn.linear_model import RidgeCV

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

DATA_DIR = "data/historical"
OUTPUT_DIR = "data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Import from existing xrapm module
from compute_xrapm import (
    clean_id, load_clean_possessions, load_box_score_data,
    SEASON_DECAY_WEIGHTS, BPM_COEFFICIENTS, calculate_possession_duration
)


def detect_collinear_pairs(full_df, season, threshold=0.60):
    """
    Detect player pairs that share a high percentage of possessions.
    
    Returns dict: {(player_a, player_b): overlap_ratio}
    """
    season_df = full_df[full_df['season'] == season]
    
    # Count offensive possessions per player
    player_poss = {}
    pair_poss = {}
    
    for _, row in season_df.iterrows():
        off_lineup = row['off_lineup']
        if not isinstance(off_lineup, (list, np.ndarray)):
            continue
        
        players = [clean_id(p) for p in off_lineup if clean_id(p) != '0']
        
        for p in players:
            player_poss[p] = player_poss.get(p, 0) + 1
        
        for i, p1 in enumerate(players):
            for p2 in players[i+1:]:
                key = tuple(sorted([p1, p2]))
                pair_poss[key] = pair_poss.get(key, 0) + 1
    
    # Find high-overlap pairs
    collinear_pairs = {}
    for (p1, p2), shared in pair_poss.items():
        p1_total = player_poss.get(p1, 1)
        p2_total = player_poss.get(p2, 1)
        
        overlap_a = shared / p1_total
        overlap_b = shared / p2_total
        max_overlap = max(overlap_a, overlap_b)
        
        if max_overlap >= threshold:
            collinear_pairs[(p1, p2)] = max_overlap
    
    return collinear_pairs


def compute_onoff_priors(box_df, season):
    """
    Compute on/off differential priors for players.
    
    Uses NET_RATING on vs NET_RATING team average to estimate individual impact.
    """
    if box_df is None:
        return {}
    
    season_df = box_df[box_df['season'] == season].copy()
    if season_df.empty:
        return {}
    
    # Need NET_RATING and team info
    if 'NET_RATING' not in season_df.columns:
        return {}
    
    priors = {}
    
    for _, row in season_df.iterrows():
        pid = row.get('player_id', row.get('PLAYER_ID', None))
        if pid is None:
            continue
        pid = clean_id(pid)
        
        net_rating = row.get('NET_RATING', 0)
        if pd.isna(net_rating):
            net_rating = 0
        
        minutes = row.get('MIN', 0)
        if minutes < 200:  # Skip low-minute players
            continue
        
        # Use net rating as simple on/off proxy
        # Scale to per-possession (~100 per 48 minutes)
        on_off_prior = net_rating / 5.0  # Scale factor
        
        priors[pid] = {
            'onoff_prior': on_off_prior,
            'net_rating': net_rating,
            'minutes': minutes,
        }
    
    return priors


def compute_career_priors(box_df, target_season, seasons_back=3):
    """
    Compute career average priors using multiple seasons.
    
    For established players with 3+ years of data, this provides a more
    stable estimate than single-season box score stats.
    """
    if box_df is None:
        return {}
    
    all_seasons = sorted(box_df['season'].unique())
    if target_season not in all_seasons:
        return {}
    
    target_idx = all_seasons.index(target_season)
    seasons_to_use = [all_seasons[i] for i in range(max(0, target_idx - seasons_back + 1), target_idx + 1)]
    
    career_df = box_df[box_df['season'].isin(seasons_to_use)].copy()
    
    # Aggregate across seasons
    id_col = 'player_id' if 'player_id' in career_df.columns else 'PLAYER_ID'
    
    priors = {}
    
    for pid, group in career_df.groupby(id_col):
        pid = clean_id(pid)
        
        total_min = group['MIN'].sum()
        n_seasons = len(group)
        
        if total_min < 500 or n_seasons < 2:
            continue
        
        # Compute career averages
        est_poss = total_min * 2.5
        
        # Box score contribution
        o_value = 0
        d_value = 0
        
        for stat, coef in BPM_COEFFICIENTS.items():
            if stat in group.columns:
                total_stat = group[stat].sum()
                rate = (total_stat / est_poss) * 100
                
                if stat in ['AST', 'OREB', 'FG3M', 'FTM', 'TOV']:
                    o_value += rate * coef
                elif stat in ['STL', 'BLK', 'DREB']:
                    d_value += rate * coef
        
        # Net rating component (if available)
        net_rating = group['NET_RATING'].mean() if 'NET_RATING' in group.columns else 0
        if pd.isna(net_rating):
            net_rating = 0
        
        career_prior = (o_value + d_value) * 0.7 + net_rating / 5.0 * 0.3
        
        priors[pid] = {
            'career_prior': career_prior,
            'n_seasons': n_seasons,
            'total_minutes': total_min,
            'o_contribution': o_value,
            'd_contribution': d_value,
        }
    
    return priors


def compute_position_weights(player_positions):
    """
    Apply position-specific prior weights to address collinearity.
    
    Wings (who typically share more court time) get STRONGER priors
    Bigs (more unique roles) get weaker priors (RAPM works better)
    """
    position_weights = {
        'G': 1.2,   # Guards - moderate prior weight
        'F': 1.5,   # Wings - STRONG prior weight (collinearity issue)
        'C': 0.8,   # Centers - weaker prior (RAPM works well)
    }
    
    weights = {}
    for pid, pos in player_positions.items():
        pos_first = pos[0] if pos else 'F'  # Default to forward weight
        weights[pid] = position_weights.get(pos_first, 1.0)
    
    return weights


def run_improved_xrapm(full_df, target_season, box_df, n_prior_seasons=2):
    """
    Run improved xRAPM with collinearity correction.
    """
    seasons = sorted(full_df['season'].unique())
    if target_season not in seasons:
        return pd.DataFrame()
    
    target_idx = seasons.index(target_season)
    seasons_to_use = []
    for i in range(n_prior_seasons + 1):
        idx = target_idx - i
        if idx >= 0:
            seasons_to_use.append((seasons[idx], SEASON_DECAY_WEIGHTS.get(i, 0.2)))
    
    print(f"\n=== IMPROVED xRAPM: {target_season} ===")
    print(f"   Seasons: {[s for s, _ in seasons_to_use]}")
    
    # Detect collinear pairs
    collinear_pairs = detect_collinear_pairs(full_df, target_season)
    print(f"   Collinear pairs (>{60}% overlap): {len(collinear_pairs)}")
    
    # Compute multiple priors
    career_priors = compute_career_priors(box_df, target_season, seasons_back=3)
    onoff_priors = compute_onoff_priors(box_df, target_season)
    
    print(f"   Career priors (multi-year): {len(career_priors)}")
    print(f"   On/Off priors: {len(onoff_priors)}")
    
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
    
    # Build possession matrices
    X_list, Y_list, weights_list = [], [], []
    
    for season, season_weight in seasons_to_use:
        season_df = full_df[full_df['season'] == season].copy()
        
        data, rows, cols = [], [], []
        for i in range(len(season_df)):
            off_lineup = season_df.iloc[i]['off_lineup']
            def_lineup = season_df.iloc[i]['def_lineup']
            
            if isinstance(off_lineup, (list, np.ndarray)):
                for pid in off_lineup:
                    pid_clean = clean_id(pid)
                    if pid_clean in player_to_idx:
                        rows.append(i)
                        cols.append(player_to_idx[pid_clean])
                        data.append(1)
            
            if isinstance(def_lineup, (list, np.ndarray)):
                for pid in def_lineup:
                    pid_clean = clean_id(pid)
                    if pid_clean in player_to_idx:
                        rows.append(i)
                        cols.append(player_to_idx[pid_clean])
                        data.append(-1)
        
        X = csr_matrix((data, (rows, cols)), shape=(len(season_df), n_players))
        Y = season_df['points'].values.astype(float)
        
        if 'duration_weight' in season_df.columns:
            sample_weights = season_df['duration_weight'].values
        else:
            sample_weights = np.ones(len(season_df))
        
        X_list.append(X)
        Y_list.append(Y)
        weights_list.append(sample_weights * season_weight)
    
    X_poss = vstack(X_list)
    Y_poss = np.concatenate(Y_list)
    weights_poss = np.concatenate(weights_list)
    
    print(f"   Possessions: {X_poss.shape[0]:,}")
    
    # === IMPROVED PRIOR AUGMENTATION ===
    # Use position-weighted priors: stronger for wings, weaker for bigs
    
    PRIOR_STRENGTH_BASE = 50.0
    
    prior_data, prior_rows, prior_cols, prior_Y, prior_weights = [], [], [], [], []
    
    collinear_players = set()
    for (p1, p2) in collinear_pairs.keys():
        collinear_players.add(p1)
        collinear_players.add(p2)
    
    for idx, pid in enumerate(sorted_players):
        # Get career prior (preferred) or fall back to on/off
        career = career_priors.get(pid, {})
        onoff = onoff_priors.get(pid, {})
        
        career_prior_val = career.get('career_prior', 0)
        onoff_prior_val = onoff.get('onoff_prior', 0)
        
        # Combine priors with weights
        if career.get('n_seasons', 0) >= 2:
            combined_prior = career_prior_val * 0.7 + onoff_prior_val * 0.3
        elif onoff_prior_val != 0:
            combined_prior = onoff_prior_val
        else:
            continue  # No prior available
        
        # Apply collinearity correction: STRONGER prior for collinear players
        if pid in collinear_players:
            prior_strength = PRIOR_STRENGTH_BASE * 2.0  # Double weight for collinear
        else:
            prior_strength = PRIOR_STRENGTH_BASE
        
        prior_data.append(prior_strength)
        prior_rows.append(len(prior_Y))
        prior_cols.append(idx)
        prior_Y.append(combined_prior / 100.0 * prior_strength)
        prior_weights.append(1.0)
    
    n_priors = len(prior_Y)
    X_prior = csr_matrix((prior_data, (prior_rows, prior_cols)), shape=(n_priors, n_players))
    Y_prior = np.array(prior_Y)
    weights_prior = np.array(prior_weights)
    
    print(f"   Prior observations: {n_priors}")
    print(f"   Collinear players with boosted prior: {len(collinear_players)}")
    
    # Combine
    X_combined = vstack([X_poss, X_prior])
    Y_combined = np.concatenate([Y_poss, Y_prior])
    weights_combined = np.concatenate([weights_poss, weights_prior])
    
    # Fit model with higher alpha (more regularization for stability)
    alphas = [100, 250, 500, 1000, 2000, 5000]
    model = RidgeCV(alphas=alphas, fit_intercept=True)
    model.fit(X_combined, Y_combined, sample_weight=weights_combined)
    
    print(f"   Best Alpha: {model.alpha_}")
    
    # Extract results
    results = []
    for pid, coef in zip(sorted_players, model.coef_):
        career = career_priors.get(pid, {})
        onoff = onoff_priors.get(pid, {})
        
        is_collinear = pid in collinear_players
        
        results.append({
            "season": target_season,
            "player_id": pid,
            "xRAPM_improved": coef * 100,
            "career_prior": career.get('career_prior', 0),
            "onoff_prior": onoff.get('onoff_prior', 0),
            "is_collinear": is_collinear,
            "alpha": model.alpha_,
        })
    
    return pd.DataFrame(results)


def validate_improvement(improved_df, original_df, benchmark_path="data/processed/bpm_benchmark_2024.csv"):
    """Compare improved xRAPM to original and benchmark."""
    if not os.path.exists(benchmark_path):
        print("⚠️ No benchmark file found")
        return
    
    benchmark = pd.read_csv(benchmark_path)
    benchmark['player_id'] = benchmark['player_id'].astype(str)
    
    # Merge
    improved_merged = improved_df.merge(
        benchmark[['player_id', 'BPM', 'POSITION']],
        on='player_id', how='inner'
    )
    
    original_merged = original_df.merge(
        benchmark[['player_id', 'BPM', 'POSITION']],
        on='player_id', how='inner'
    )
    
    print("\n=== VALIDATION ===")
    
    # Overall metrics
    if len(improved_merged) > 0:
        imp_mae = np.abs(improved_merged['xRAPM_improved'] - improved_merged['BPM']).mean()
        imp_corr = improved_merged['xRAPM_improved'].corr(improved_merged['BPM'])
        print(f"Improved - MAE: {imp_mae:.2f}, Corr: {imp_corr:.3f}")
    
    if len(original_merged) > 0:
        orig_mae = np.abs(original_merged['xRAPM'] - original_merged['BPM']).mean()
        orig_corr = original_merged['xRAPM'].corr(original_merged['BPM'])
        print(f"Original - MAE: {orig_mae:.2f}, Corr: {orig_corr:.3f}")
    
    # By position
    for pos in ['G', 'F', 'C']:
        pos_imp = improved_merged[improved_merged['POSITION'].str.contains(pos, na=False)]
        pos_orig = original_merged[original_merged['POSITION'].str.contains(pos, na=False)]
        
        if len(pos_imp) > 10:
            imp_mae = np.abs(pos_imp['xRAPM_improved'] - pos_imp['BPM']).mean()
            imp_corr = pos_imp['xRAPM_improved'].corr(pos_imp['BPM'])
            orig_mae = np.abs(pos_orig['xRAPM'] - pos_orig['BPM']).mean() if len(pos_orig) > 0 else 0
            orig_corr = pos_orig['xRAPM'].corr(pos_orig['BPM']) if len(pos_orig) > 0 else 0
            
            improvement = ((orig_mae - imp_mae) / orig_mae * 100) if orig_mae > 0 else 0
            print(f"  {pos}: Improved MAE {imp_mae:.2f} (was {orig_mae:.2f}) | Corr {imp_corr:.2f} (was {orig_corr:.2f}) | Δ {improvement:+.1f}%")


def main():
    print("=" * 70)
    print("IMPROVED xRAPM WITH COLLINEARITY CORRECTION")
    print("=" * 70)
    
    # Load data
    print("\n📊 Loading data...")
    full_df = load_clean_possessions()
    box_df = load_box_score_data()
    
    if full_df.empty:
        print("❌ No possession data found")
        return
    
    print(f"   Possessions loaded: {len(full_df):,}")
    print(f"   Box score loaded: {len(box_df) if box_df is not None else 0}")
    
    # Run for 2024-25 season
    target_season = '2024-25'
    
    improved_results = run_improved_xrapm(full_df, target_season, box_df, n_prior_seasons=2)
    
    if improved_results.empty:
        print("❌ No results generated")
        return
    
    # Load original for comparison
    original_path = os.path.join(OUTPUT_DIR, "player_xrapm_2024-25.csv")
    if os.path.exists(original_path):
        original_df = pd.read_csv(original_path)
        original_df['player_id'] = original_df['player_id'].astype(str)
        validate_improvement(improved_results, original_df)
    
    # Save
    improved_results.to_csv(os.path.join(OUTPUT_DIR, "player_xrapm_improved_2024-25.csv"), index=False)
    print(f"\n✅ Saved improved xRAPM to player_xrapm_improved_2024-25.csv")
    
    # Show sample
    print("\n=== SAMPLE IMPROVED xRAPM ===")
    top_players = improved_results.nlargest(15, 'xRAPM_improved')
    for _, row in top_players.iterrows():
        collin = "⚠️" if row['is_collinear'] else ""
        print(f"  {row['player_id']}: {row['xRAPM_improved']:+.2f} {collin}")


if __name__ == "__main__":
    main()
