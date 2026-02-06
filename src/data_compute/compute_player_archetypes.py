"""
src/data_compute/compute_player_archetypes.py
Classifies NBA players into offensive archetypes based on tracking and playtype data.

Archetype Definitions (from user specification):
================================================
1. Ball Dominant Creator - High ISO/PnRBH/PostUp freq + high playmaking scores
2. Interior Scorer - Low playmaking + ball-dominant + high drives + low 3pt
3. Perimeter Scorer - Low playmaking + ball-dominant + low drives + high 3pt  
4. All-Around Scorer - Low playmaking + ball-dominant + balanced interior/perimeter
5. Ballhandler - High playmaking + ball dominant + lower scoring volume
6. Off-Ball Finisher - High Cut/PnRRm/Putback/Transition freq
7. Off-Ball Movement Shooter - High Handoff/Offscreen freq
8. Off-Ball Stationary Shooter - High SpotUp freq
9. Rotation Piece/Glue Guy - All-around, low usage rate

Additional Metrics Incorporated:
- Four Factors: Shooting (40%), Turnovers (25%), Rebounding (20%), Free Throws (15%)
- Three-pointer impact: production, efficiency, wide-open, catch-and-shoot
- Driving impact: attempts, efficiency, FT rate
- Transition impact
- Playmaking: time of possession, average dribbles, secondary assists
- Screening impact (less important)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = Path("data")
TRACKING_DIR = DATA_DIR / "tracking"
HISTORICAL_DIR = DATA_DIR / "historical"
OUTPUT_DIR = DATA_DIR / "processed"
OUTPUT_DIR.mkdir(exist_ok=True)

SEASONS = ['2022-23', '2023-24', '2024-25']

# Thresholds for classification (percentiles within season)
THRESHOLDS = {
    # Ball dominance thresholds
    'BALL_DOMINANT_PCT': 0.15,        # Combined ISO + PnRBH + PostUp possession %
    'HIGH_BALL_DOMINANT_PCT': 0.25,   # Very ball dominant
    
    # Playmaking thresholds (per 36 min)
    'HIGH_PLAYMAKING': 5.0,           # AST per 36
    'VERY_HIGH_PLAYMAKING': 7.0,      # Elite playmaker
    'SECONDARY_AST_HIGH': 1.0,        # Secondary assists per 36
    
    # Scoring volume (PTS per game)
    'HIGH_SCORING': 15.0,
    'LOW_SCORING': 8.0,
    
    # Usage thresholds
    'HIGH_USAGE': 0.22,
    'LOW_USAGE': 0.15,
    
    # Shot profile
    'INTERIOR_HEAVY': 0.6,            # Drives / (Drives + FG3A) ratio
    'PERIMETER_HEAVY': 0.4,           # Below this = perimeter heavy
    
    # Off-ball thresholds (possession %)
    'HIGH_CUT_PNRRM': 0.10,           # Cut + PnRRm combined
    'HIGH_SPOTUP': 0.20,              # SpotUp possession %
    'HIGH_MOVEMENT': 0.10,            # Handoff + OffScreen combined
    
    # Minutes threshold
    'MIN_MINUTES': 500,               # Minimum total minutes for classification
    'MIN_GP': 20,                     # Minimum games played
}

# =============================================================================
# DATA LOADING
# =============================================================================
def load_synergy_data(season: str) -> pd.DataFrame:
    """Load and merge all synergy playtype data for a season."""
    season_dir = TRACKING_DIR / season
    
    playtypes = [
        'Isolation', 'PRBallHandler', 'Postup', 'Cut', 'PRRollman',
        'Handoff', 'OffScreen', 'Spotup', 'Transition', 'OffRebound', 'Misc'
    ]
    
    all_data = []
    for playtype in playtypes:
        path = season_dir / f"synergy_Offensive_{playtype}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            # Aggregate duplicates (traded players) by taking sum of possessions, weighted avg PPP
            agg_df = df.groupby('PLAYER_ID').agg({
                'PLAYER_NAME': 'first',
                'POSS_PCT': 'mean',  # Average possession %
                'PPP': 'mean',       # Average PPP
                'POSS': 'sum'        # Total possessions
            }).reset_index()
            
            agg_df = agg_df.rename(columns={
                'POSS_PCT': f'{playtype.upper()}_POSS_PCT',
                'PPP': f'{playtype.upper()}_PPP',
                'POSS': f'{playtype.upper()}_POSS'
            })
            all_data.append(agg_df)
    
    if not all_data:
        return pd.DataFrame()
    
    # Merge all playtypes into single player row
    player_data = all_data[0]
    for df in all_data[1:]:
        player_data = player_data.merge(
            df.drop(columns=['PLAYER_NAME'], errors='ignore'),
            on='PLAYER_ID', how='outer'
        )
    
    player_data['SEASON'] = season
    return player_data


def load_tracking_data(season: str) -> pd.DataFrame:
    """Load and merge tracking data (drives, passing, possessions, catch-shoot)."""
    season_dir = TRACKING_DIR / season
    
    # Specify columns to keep from each file, avoiding conflicts with box score
    tracking_files = {
        'Drives': ['PLAYER_ID', 'DRIVES', 'DRIVE_PTS', 'DRIVE_FG_PCT', 'DRIVE_AST', 'DRIVE_TOV'],
        'Passing': ['PLAYER_ID', 'PASSES_MADE', 'SECONDARY_AST', 'POTENTIAL_AST', 'AST_POINTS_CREATED'],
        'Possessions': ['PLAYER_ID', 'TOUCHES', 'TIME_OF_POSS', 'AVG_SEC_PER_TOUCH', 'AVG_DRIB_PER_TOUCH', 'FRONT_CT_TOUCHES'],
        'CatchShoot': ['PLAYER_ID', 'CATCH_SHOOT_FGM', 'CATCH_SHOOT_FGA', 'CATCH_SHOOT_FG_PCT', 
                       'CATCH_SHOOT_PTS', 'CATCH_SHOOT_FG3M', 'CATCH_SHOOT_FG3A', 'CATCH_SHOOT_FG3_PCT'],
        'Rebounding': ['PLAYER_ID', 'OREB_CONTEST', 'DREB_CONTEST', 'REB_CONTEST'],
        'SpeedDistance': ['PLAYER_ID', 'DIST_MILES', 'AVG_SPEED'],
    }
    
    merged = None
    for track_type, keep_cols in tracking_files.items():
        path = season_dir / f"tracking_{track_type}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            # Only keep specified columns that exist
            available_cols = [c for c in keep_cols if c in df.columns]
            if available_cols:
                df = df[available_cols]
                # Aggregate duplicates (traded players)
                numeric_cols = [c for c in df.columns if c != 'PLAYER_ID']
                df = df.groupby('PLAYER_ID')[numeric_cols].sum().reset_index()
                
                if merged is None:
                    merged = df
                else:
                    merged = merged.merge(df, on='PLAYER_ID', how='outer')
    
    if merged is not None:
        merged['SEASON'] = season
    return merged if merged is not None else pd.DataFrame()


def load_box_score_data() -> pd.DataFrame:
    """Load complete box score stats."""
    path = HISTORICAL_DIR / "complete_player_season_stats.parquet"
    if not path.exists():
        return pd.DataFrame()
    
    df = pd.read_parquet(path)
    # Select relevant columns for archetype classification
    cols = ['PLAYER_ID', 'PLAYER_NAME', 'SEASON', 'GP', 'MIN', 'PTS', 'AST', 'REB', 
            'OREB', 'DREB', 'STL', 'BLK', 'TOV', 'FGA', 'FGM', 'FG3A', 'FG3M', 
            'FTA', 'FTM', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'USG_PCT', 'TS_PCT']
    available_cols = [c for c in cols if c in df.columns]
    return df[available_cols]


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
def compute_archetype_features(synergy: pd.DataFrame, tracking: pd.DataFrame, 
                                box: pd.DataFrame, season: str) -> pd.DataFrame:
    """Compute all features needed for archetype classification."""
    
    # Filter to season
    syn = synergy[synergy['SEASON'] == season].copy() if not synergy.empty else pd.DataFrame()
    trk = tracking[tracking['SEASON'] == season].copy() if not tracking.empty else pd.DataFrame()
    bx = box[box['SEASON'] == season].copy() if not box.empty else pd.DataFrame()
    
    # Start with box score as base (has all players)
    if bx.empty:
        return pd.DataFrame()
    
    features = bx.copy()
    
    # Merge synergy data
    if not syn.empty:
        syn_cols = [c for c in syn.columns if c not in ['PLAYER_NAME', 'SEASON'] or c == 'PLAYER_ID']
        features = features.merge(syn[syn_cols], on='PLAYER_ID', how='left')
    
    # Merge tracking data
    if not trk.empty:
        trk_cols = [c for c in trk.columns if c not in ['PLAYER_NAME', 'SEASON', 'GP', 'MIN'] or c == 'PLAYER_ID']
        features = features.merge(trk[trk_cols], on='PLAYER_ID', how='left')
    
    # Fill NaN playtype data with 0 (player doesn't use that playtype)
    playtype_cols = [c for c in features.columns if '_POSS_PCT' in c or '_PPP' in c]
    features[playtype_cols] = features[playtype_cols].fillna(0)
    
    # ==========================================================================
    # COMPUTE DERIVED FEATURES
    # ==========================================================================
    
    # Per-36 minute stats
    minutes_factor = 36 / (features['MIN'] / features['GP']).replace(0, np.nan)
    features['PTS_PER36'] = (features['PTS'] / features['GP']) * minutes_factor
    features['AST_PER36'] = (features['AST'] / features['GP']) * minutes_factor
    features['REB_PER36'] = (features['REB'] / features['GP']) * minutes_factor
    features['TOV_PER36'] = (features['TOV'] / features['GP']) * minutes_factor
    
    # Ball dominance score (ISO + PnRBH + PostUp possession %)
    features['BALL_DOMINANT_PCT'] = (
        features.get('ISOLATION_POSS_PCT', 0) + 
        features.get('PRBALLHANDLER_POSS_PCT', 0) + 
        features.get('POSTUP_POSS_PCT', 0)
    ).fillna(0)
    
    # Playmaking composite
    if 'SECONDARY_AST' in features.columns:
        features['SECONDARY_AST_PER36'] = (features['SECONDARY_AST'] / features['GP']) * minutes_factor
    else:
        features['SECONDARY_AST_PER36'] = 0
    
    if 'POTENTIAL_AST' in features.columns:
        features['POTENTIAL_AST_PER36'] = (features['POTENTIAL_AST'] / features['GP']) * minutes_factor
    else:
        features['POTENTIAL_AST_PER36'] = 0
    
    features['PLAYMAKING_SCORE'] = (
        features['AST_PER36'] * 1.0 +
        features['SECONDARY_AST_PER36'] * 0.5 +
        features['POTENTIAL_AST_PER36'] * 0.3
    ).fillna(0)
    
    # Shot profile: Interior vs Perimeter
    if 'DRIVES' in features.columns:
        features['DRIVES_PER36'] = (features['DRIVES'] / features['GP']) * minutes_factor
    else:
        features['DRIVES_PER36'] = 0
    
    features['FG3A_PER36'] = (features['FG3A'] / features['GP']) * minutes_factor
    
    # Interior ratio (drives vs 3pt attempts)
    total_shots = features['DRIVES_PER36'] + features['FG3A_PER36']
    features['INTERIOR_RATIO'] = np.where(
        total_shots > 0,
        features['DRIVES_PER36'] / total_shots,
        0.5  # Default to balanced if no data
    )
    
    # Off-ball frequencies
    features['CUT_PNRRM_PCT'] = (
        features.get('CUT_POSS_PCT', 0) + 
        features.get('PRROLLMAN_POSS_PCT', 0)
    ).fillna(0)
    
    features['MOVEMENT_SHOOTER_PCT'] = (
        features.get('HANDOFF_POSS_PCT', 0) + 
        features.get('OFFSCREEN_POSS_PCT', 0)
    ).fillna(0)
    
    features['SPOTUP_PCT'] = features.get('SPOTUP_POSS_PCT', 0).fillna(0)
    
    features['TRANSITION_PCT'] = features.get('TRANSITION_POSS_PCT', 0).fillna(0)
    
    features['PUTBACK_PCT'] = features.get('OFFREBOUND_POSS_PCT', 0).fillna(0)
    
    # Four Factors composite (for player evaluation, not classification)
    features['EFG_PCT'] = np.where(
        features['FGA'] > 0,
        (features['FGM'] + 0.5 * features['FG3M']) / features['FGA'],
        0
    )
    features['TOV_PCT'] = np.where(
        (features['FGA'] + 0.44 * features['FTA'] + features['TOV']) > 0,
        features['TOV'] / (features['FGA'] + 0.44 * features['FTA'] + features['TOV']),
        0
    )
    features['FT_RATE'] = np.where(features['FGA'] > 0, features['FTA'] / features['FGA'], 0)
    
    # Time of possession (ball handling)
    if 'TIME_OF_POSS' in features.columns:
        features['TIME_OF_POSS_PER36'] = (features['TIME_OF_POSS'] / features['GP']) * minutes_factor
    else:
        features['TIME_OF_POSS_PER36'] = 0
    
    if 'AVG_DRIB_PER_TOUCH' in features.columns:
        features['DRIBBLES_PER_TOUCH'] = features['AVG_DRIB_PER_TOUCH']
    else:
        features['DRIBBLES_PER_TOUCH'] = 0
    
    # Points per game
    features['PPG'] = features['PTS'] / features['GP']
    
    return features


# =============================================================================
# ARCHETYPE CLASSIFICATION
# =============================================================================
def classify_archetype(row: pd.Series) -> dict:
    """
    Classify a single player into an offensive archetype.
    
    Classification hierarchy:
    1. Check if enough minutes/games for classification
    2. Determine ball dominance level
    3. Check playmaking level
    4. Classify based on scoring style or off-ball tendencies
    """
    
    result = {
        'primary_archetype': 'Unknown',
        'secondary_archetype': None,
        'archetype_confidence': 0.0,
        'ball_dominance_tier': 'Low',
        'playmaking_tier': 'Low',
        'scoring_tier': 'Low',
    }
    
    # Check minimum requirements
    if row.get('MIN', 0) < THRESHOLDS['MIN_MINUTES'] or row.get('GP', 0) < THRESHOLDS['MIN_GP']:
        result['primary_archetype'] = 'Insufficient Minutes'
        return result
    
    # ==========================================================================
    # TIER ASSIGNMENTS
    # ==========================================================================
    
    # Ball dominance tier
    ball_dom = row.get('BALL_DOMINANT_PCT', 0)
    if ball_dom >= THRESHOLDS['HIGH_BALL_DOMINANT_PCT']:
        result['ball_dominance_tier'] = 'Very High'
    elif ball_dom >= THRESHOLDS['BALL_DOMINANT_PCT']:
        result['ball_dominance_tier'] = 'High'
    else:
        result['ball_dominance_tier'] = 'Low'
    
    # Playmaking tier
    playmaking = row.get('PLAYMAKING_SCORE', 0)
    ast_per36 = row.get('AST_PER36', 0)
    if ast_per36 >= THRESHOLDS['VERY_HIGH_PLAYMAKING']:
        result['playmaking_tier'] = 'Elite'
    elif ast_per36 >= THRESHOLDS['HIGH_PLAYMAKING']:
        result['playmaking_tier'] = 'High'
    else:
        result['playmaking_tier'] = 'Low'
    
    # Scoring tier
    ppg = row.get('PPG', 0)
    if ppg >= THRESHOLDS['HIGH_SCORING']:
        result['scoring_tier'] = 'High'
    elif ppg >= THRESHOLDS['LOW_SCORING']:
        result['scoring_tier'] = 'Medium'
    else:
        result['scoring_tier'] = 'Low'
    
    # Usage tier
    usg = row.get('USG_PCT', 0.15)
    high_usage = usg >= THRESHOLDS['HIGH_USAGE']
    low_usage = usg < THRESHOLDS['LOW_USAGE']
    
    # ==========================================================================
    # CLASSIFICATION LOGIC
    # ==========================================================================
    
    is_ball_dominant = result['ball_dominance_tier'] in ['High', 'Very High']
    is_playmaker = result['playmaking_tier'] in ['High', 'Elite']
    is_high_volume_scorer = result['scoring_tier'] == 'High'
    
    # Interior vs Perimeter
    interior_ratio = row.get('INTERIOR_RATIO', 0.5)
    is_interior = interior_ratio >= THRESHOLDS['INTERIOR_HEAVY']
    is_perimeter = interior_ratio <= THRESHOLDS['PERIMETER_HEAVY']
    
    # Off-ball tendencies
    cut_pnrrm = row.get('CUT_PNRRM_PCT', 0)
    spotup = row.get('SPOTUP_PCT', 0)
    movement = row.get('MOVEMENT_SHOOTER_PCT', 0)
    transition = row.get('TRANSITION_PCT', 0)
    putback = row.get('PUTBACK_PCT', 0)
    
    # ---------------------------------------------------------------------------
    # CLASSIFICATION HIERARCHY
    # ---------------------------------------------------------------------------
    
    # 1. BALL DOMINANT CREATOR
    # High ISO/PnRBH/PostUp + High playmaking
    if is_ball_dominant and is_playmaker:
        result['primary_archetype'] = 'Ball Dominant Creator'
        result['archetype_confidence'] = min(1.0, (ball_dom / 0.25 + playmaking / 8) / 2)
        if is_high_volume_scorer:
            result['secondary_archetype'] = 'Primary Scorer'
        return result
    
    # 2. BALLHANDLER (Facilitator)
    # High playmaking, ball dominant, but lower scoring
    if is_playmaker and (is_ball_dominant or row.get('TIME_OF_POSS_PER36', 0) > 3.0):
        if not is_high_volume_scorer:
            result['primary_archetype'] = 'Ballhandler'
            result['archetype_confidence'] = min(1.0, playmaking / 8)
            return result
    
    # 3. PRIMARY SCORERS (Ball Dominant, Low Playmaking)
    if is_ball_dominant and not is_playmaker:
        if is_interior:
            result['primary_archetype'] = 'Interior Scorer'
            result['archetype_confidence'] = min(1.0, ball_dom / 0.20 * interior_ratio)
        elif is_perimeter:
            result['primary_archetype'] = 'Perimeter Scorer'
            result['archetype_confidence'] = min(1.0, ball_dom / 0.20 * (1 - interior_ratio))
        else:
            result['primary_archetype'] = 'All-Around Scorer'
            result['archetype_confidence'] = min(1.0, ball_dom / 0.20)
        return result
    
    # 4. OFF-BALL FINISHER
    # High Cut, PnRRm, Putback, Transition
    offball_finish_score = cut_pnrrm + putback * 0.5 + transition * 0.3
    if cut_pnrrm >= THRESHOLDS['HIGH_CUT_PNRRM'] or offball_finish_score > 0.12:
        result['primary_archetype'] = 'Off-Ball Finisher'
        result['archetype_confidence'] = min(1.0, offball_finish_score / 0.15)
        if transition > 0.10:
            result['secondary_archetype'] = 'Transition Player'
        return result
    
    # 5. OFF-BALL MOVEMENT SHOOTER
    # High Handoff and OffScreen
    if movement >= THRESHOLDS['HIGH_MOVEMENT']:
        result['primary_archetype'] = 'Off-Ball Movement Shooter'
        result['archetype_confidence'] = min(1.0, movement / 0.15)
        return result
    
    # 6. OFF-BALL STATIONARY SHOOTER
    # High SpotUp
    if spotup >= THRESHOLDS['HIGH_SPOTUP']:
        result['primary_archetype'] = 'Off-Ball Stationary Shooter'
        result['archetype_confidence'] = min(1.0, spotup / 0.30)
        if row.get('FG3_PCT', 0) > 0.38:
            result['secondary_archetype'] = 'Elite Shooter'
        return result
    
    # 7. ROTATION PIECE / GLUE GUY
    # Low usage, all-around contributions
    if low_usage or (not is_ball_dominant and not is_high_volume_scorer):
        result['primary_archetype'] = 'Rotation Piece'
        
        # Determine specialization
        if row.get('REB_PER36', 0) > 8:
            result['secondary_archetype'] = 'Rebounder'
        elif row.get('STL', 0) / row.get('GP', 1) > 1.2:
            result['secondary_archetype'] = 'Perimeter Defender'
        elif row.get('BLK', 0) / row.get('GP', 1) > 1.0:
            result['secondary_archetype'] = 'Rim Protector'
        else:
            result['secondary_archetype'] = 'Glue Guy'
        
        result['archetype_confidence'] = 0.6
        return result
    
    # Fallback
    result['primary_archetype'] = 'Unclassified'
    result['archetype_confidence'] = 0.3
    return result


def classify_all_players(features: pd.DataFrame) -> pd.DataFrame:
    """Classify all players in the features dataframe."""
    
    classifications = []
    for idx, row in features.iterrows():
        result = classify_archetype(row)
        result['PLAYER_ID'] = row['PLAYER_ID']
        result['PLAYER_NAME'] = row['PLAYER_NAME']
        result['SEASON'] = row['SEASON']
        classifications.append(result)
    
    class_df = pd.DataFrame(classifications)
    
    # Merge with original features
    output = features.merge(class_df, on=['PLAYER_ID', 'PLAYER_NAME', 'SEASON'])
    
    return output


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    print("=" * 70)
    print("PLAYER ARCHETYPE CLASSIFICATION")
    print("=" * 70)
    
    # Load all data
    print("\n📊 Loading data...")
    
    all_synergy = []
    all_tracking = []
    
    for season in SEASONS:
        print(f"   Loading {season}...")
        syn = load_synergy_data(season)
        trk = load_tracking_data(season)
        if not syn.empty:
            all_synergy.append(syn)
        if not trk.empty:
            all_tracking.append(trk)
    
    synergy_df = pd.concat(all_synergy, ignore_index=True) if all_synergy else pd.DataFrame()
    tracking_df = pd.concat(all_tracking, ignore_index=True) if all_tracking else pd.DataFrame()
    box_df = load_box_score_data()
    
    print(f"\n   Synergy data: {len(synergy_df)} rows")
    print(f"   Tracking data: {len(tracking_df)} rows")
    print(f"   Box score data: {len(box_df)} rows")
    
    # Compute features and classify for each season
    all_results = []
    for season in SEASONS:
        print(f"\n🔄 Processing {season}...")
        features = compute_archetype_features(synergy_df, tracking_df, box_df, season)
        
        if features.empty:
            print(f"   ⚠️ No data for {season}")
            continue
        
        classified = classify_all_players(features)
        all_results.append(classified)
        
        # Print summary
        archetype_counts = classified['primary_archetype'].value_counts()
        print(f"   Classified {len(classified)} players:")
        for arch, count in archetype_counts.items():
            print(f"      {arch}: {count}")
    
    # Combine all seasons
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        
        # Save results
        final_df.to_parquet(OUTPUT_DIR / "player_archetypes.parquet", index=False)
        final_df.to_csv(OUTPUT_DIR / "player_archetypes.csv", index=False)
        
        print(f"\n✅ Saved {len(final_df)} player-seasons to data/processed/player_archetypes.parquet")
        
        # Print example classifications
        print("\n" + "=" * 70)
        print("SAMPLE CLASSIFICATIONS (2023-24 Stars)")
        print("=" * 70)
        
        s24 = final_df[final_df['SEASON'] == '2023-24'].copy()
        stars = [
            'LeBron James', 'Stephen Curry', 'Kevin Durant', 'Giannis Antetokounmpo',
            'Luka Dončić', 'Nikola Jokić', 'Joel Embiid', 'Jayson Tatum',
            'Shai Gilgeous-Alexander', 'Anthony Edwards', 'Draymond Green',
            'Kawhi Leonard', 'Jalen Brunson', 'Tyrese Haliburton'
        ]
        
        for name in stars:
            player = s24[s24['PLAYER_NAME'].str.contains(name.split()[0], case=False, na=False)]
            if len(player) > 0:
                p = player.iloc[0]
                secondary = f" / {p['secondary_archetype']}" if p['secondary_archetype'] else ""
                print(f"  {p['PLAYER_NAME']:<25} → {p['primary_archetype']}{secondary} ({p['archetype_confidence']:.0%})")
    
    return final_df


if __name__ == "__main__":
    result = main()
