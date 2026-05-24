import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import os

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.modeling.model_config import PtsV40DefenseConfig
from scripts.validate_lineup_pts_v2 import build_player_on_court_nrtg

def safe_z(series, sign=1.0):
    s = series.astype(float)
    valid = s.dropna()
    if valid.std() < 1e-9 or len(valid) < 2:
        return pd.Series(np.zeros(len(s)), index=s.index)
    z = sign * (s - valid.mean()) / valid.std()
    return z.fillna(0.0)

def apply_defense_redesign(pts: pd.DataFrame, decomp: pd.DataFrame, def_arch: pd.DataFrame, config: PtsV40DefenseConfig, seasons: list) -> pd.DataFrame:
    df = pts.copy()
    def_arch = def_arch.copy()
    
    # Rename columns to match
    if 'PLAYER_ID' in def_arch.columns:
        def_arch = def_arch.rename(columns={'PLAYER_ID': 'player_id'})
    if 'SEASON' in def_arch.columns:
        def_arch = def_arch.rename(columns={'SEASON': 'season'})
    
    # Ensure ID formats match
    df['player_id'] = df['player_id'].astype(str)
    decomp['player_id'] = decomp['player_id'].astype(str)
    def_arch['player_id'] = def_arch['player_id'].astype(str)
    
    # 1. Compute matchup_z per season
    matchup_dfs = []
    for season in seasons:
        df_season = def_arch[def_arch['season'] == season].copy()
        if len(df_season) == 0:
            continue
            
        # Fallback for missing metrics
        if config.pre2022_fallback:
            for col in config.matchup_component_weights.keys():
                if col not in df_season.columns:
                    df_season[col] = np.nan
                df_season[col] = df_season[col].fillna(df_season[col].median() if not pd.isna(df_season[col].median()) else 0.0)
                
        composite = pd.Series(0.0, index=df_season.index)
        for col, weight in config.matchup_component_weights.items():
            sign = 1.0 if weight > 0 else -1.0
            z_col = safe_z(df_season[col], sign=sign)
            composite += abs(weight) * z_col
            
        df_season['matchup_z'] = safe_z(composite)
        
        # Source tracking
        if 'D_FG_DIFF' in df_season.columns and not df_season['D_FG_DIFF'].isna().all() and (df_season['D_FG_DIFF'] != 0.0).any():
            df_season['matchup_data_source'] = "live"
        else:
            df_season['matchup_data_source'] = "pre2022_fallback"
            
        matchup_dfs.append(df_season[['player_id', 'season', 'matchup_z', 'matchup_data_source', 'defensive_archetype']])
        
    if matchup_dfs:
        matchups = pd.concat(matchup_dfs, ignore_index=True)
    else:
        matchups = pd.DataFrame(columns=['player_id', 'season', 'matchup_z', 'matchup_data_source', 'defensive_archetype'])

    # 2. Compute lineup_residual_z (OOS: fit OLS on all prior seasons, apply to current)
    # Build per-season merged data first, then apply rolling OOS betas
    season_data = {}
    for season in seasons:
        per100 = build_player_on_court_nrtg(season)
        if per100.empty:
            continue
        per100['player_id'] = per100['player_id'].astype(str)
        m_season = matchups[matchups['season'] == season]
        merged = per100.merge(m_season[['player_id', 'matchup_z']], on='player_id', how='inner')
        if len(merged) >= 5:
            season_data[season] = merged

    sorted_seasons = sorted(season_data.keys())
    residual_dfs = []
    for i, season in enumerate(sorted_seasons):
        prior_seasons = sorted_seasons[:i]
        df_curr = season_data[season].copy()

        if len(prior_seasons) == 0:
            # No prior data for first season — residual is zero (no OOS estimate possible)
            df_curr['lineup_residual_z'] = 0.0
        else:
            prior_data = pd.concat([season_data[s] for s in prior_seasons], ignore_index=True)
            Y_p = prior_data['per100_def'].values
            X_p = np.vstack([np.ones(len(Y_p)), prior_data['matchup_z'].values]).T
            W_p = np.sqrt(prior_data['def_poss'].clip(lower=1).values)
            beta, _, _, _ = np.linalg.lstsq(X_p * W_p[:, np.newaxis], Y_p * W_p, rcond=None)

            Y_c = df_curr['per100_def'].values
            X_c = np.vstack([np.ones(len(Y_c)), df_curr['matchup_z'].values]).T
            expected = X_c.dot(beta)
            # positive residual = better defender than matchup predicts
            df_curr['lineup_residual_z'] = safe_z(pd.Series(expected - Y_c, index=df_curr.index))

        df_curr['season'] = season
        residual_dfs.append(df_curr[['player_id', 'season', 'lineup_residual_z']])

    if residual_dfs:
        residuals = pd.concat(residual_dfs, ignore_index=True)
    else:
        residuals = pd.DataFrame(columns=['player_id', 'season', 'lineup_residual_z'])

    # 3. Compute archetype_baseline_z — individual player defensive dim score
    # Uses each player's own dim scores (not cohort mean), so signal varies per player
    decomp_clean = decomp.copy()
    if 'defensive_archetype' in decomp_clean.columns:
        decomp_clean = decomp_clean.drop(columns=['defensive_archetype'])

    arch_merged = decomp_clean.merge(
        def_arch[['player_id', 'season', 'defensive_archetype']],
        on=['player_id', 'season'], how='left'
    )
    arch_merged['dim_sum'] = (
        arch_merged['dim_defensive_versatility_z'].fillna(0) +
        arch_merged['dim_defensive_impact_z'].fillna(0) +
        arch_merged['dim_defensive_playmaking_z'].fillna(0)
    )

    baseline_dfs = []
    for season in seasons:
        s_df = arch_merged[arch_merged['season'] == season].copy()
        if len(s_df) > 0:
            # z-score each player's individual dim_sum within the season
            s_df['archetype_baseline_z'] = safe_z(s_df['dim_sum'])
            baseline_dfs.append(s_df[['player_id', 'season', 'archetype_baseline_z']])

    if baseline_dfs:
        baselines = pd.concat(baseline_dfs, ignore_index=True)
    else:
        baselines = pd.DataFrame(columns=['player_id', 'season', 'archetype_baseline_z'])

    # 4. Merge all together and compute pts_d_v40_c
    out = df.merge(matchups[['player_id', 'season', 'matchup_z', 'matchup_data_source']], on=['player_id', 'season'], how='left')
    out = out.merge(residuals[['player_id', 'season', 'lineup_residual_z']], on=['player_id', 'season'], how='left')
    out = out.merge(baselines[['player_id', 'season', 'archetype_baseline_z']], on=['player_id', 'season'], how='left')
    
    out['matchup_z'] = out['matchup_z'].fillna(0.0)
    out['lineup_residual_z'] = out['lineup_residual_z'].fillna(0.0)
    out['archetype_baseline_z'] = out['archetype_baseline_z'].fillna(0.0)
    out['matchup_data_source'] = out['matchup_data_source'].fillna("missing")
    
    pts_d_v40_c = (config.gamma_match * out['matchup_z'] +
                   config.gamma_lineup * out['lineup_residual_z'] +
                   config.gamma_arch * out['archetype_baseline_z'])
                   
    out['pts_d_v40_c'] = np.clip(pts_d_v40_c, -config.final_clip, config.final_clip)
    
    return out

def test_gobert_sign():
    """Unit test for sign convention on D_FG_DIFF.
    Rudy Gobert (or any elite defender) should have negative D_FG_DIFF 
    and it should contribute positively to matchup_z.
    """
    config = PtsV40DefenseConfig()
    
    # Mock df_def
    df_def = pd.DataFrame({
        'player_id': ['1', '2', '3'],
        'season': ['2023-24', '2023-24', '2023-24'],
        'd_results_pctl': [90, 50, 10],            # Player 1 is elite
        'D_FG_DIFF': [-5.0, 0.0, 5.0],             # Player 1 has negative (good) diff
        'contested_shots_pctl': [95, 50, 5],       # Player 1 elite
        'rim_protection_index_pctl': [99, 50, 1],  # Player 1 elite
        'defensive_archetype': ['Anchor', 'Wing', 'Guard']
    })
    
    # Replicate the matchup_z logic directly to test
    composite = pd.Series(0.0, index=df_def.index)
    for col, weight in config.matchup_component_weights.items():
        sign = 1.0 if weight > 0 else -1.0
        z_col = safe_z(df_def[col], sign=sign)
        composite += abs(weight) * z_col
        
        if col == 'D_FG_DIFF':
            # Player 1's D_FG_DIFF is -5.0. Mean is 0. 
            # -5.0 is below mean. With sign=-1, z_col should be positive.
            assert z_col[0] > 0, "D_FG_DIFF sign logic failed! Player with negative diff should have positive z-score."
            
    matchup_z = safe_z(composite)
    assert matchup_z[0] > 0, "Elite defender should have positive matchup_z."
    assert matchup_z[2] < 0, "Poor defender should have negative matchup_z."
    print("test_gobert_sign: PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pts-v32", required=False)
    parser.add_argument("--decomp", required=False)
    parser.add_argument("--def-arch", required=False)
    parser.add_argument("--pbp-dir", required=False)
    parser.add_argument("--output", required=False)
    parser.add_argument("--test", action="store_true", help="Run unit test and exit")
    args = parser.parse_args()

    if args.test:
        test_gobert_sign()
        sys.exit(0)

    if not all([args.pts_v32, args.decomp, args.def_arch, args.pbp_dir, args.output]):
        print("Error: all path arguments are required unless --test is provided.")
        sys.exit(1)

    print(f"Loading data...")
    pts = pd.read_parquet(args.pts_v32)
    decomp = pd.read_parquet(args.decomp)
    def_arch = pd.read_parquet(args.def_arch)
    
    # We get the unique seasons from pts
    seasons = pts['season'].unique().tolist()
    
    config = PtsV40DefenseConfig()
    print(f"Applying Improvement C (Defense Redesign)...")
    out = apply_defense_redesign(pts, decomp, def_arch, config, seasons)
    
    print(f"Saving to {args.output}...")
    out.to_parquet(args.output, index=False)
    print("Done.")
