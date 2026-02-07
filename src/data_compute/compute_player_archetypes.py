"""
src/data_compute/compute_player_archetypes.py
Classifies NBA players into offensive archetypes based on tracking and playtype data.

v2 — Major revision addressing:
  - Post-up separated from on-ball creation (weighted 0.65x)
  - FG2A_RATE replaces broken drives-based INTERIOR_RATIO
  - PTS_PER36 replaces raw PPG for scoring gates
  - Efficiency awareness (TS% z-score adjusts confidence)
  - Turnover penalty in playmaking score
  - Per-36 inflation guard (MIN_MPG >= 15)
  - Raised thresholds: ball_dom 0.18, playmaking 5.5, off-ball finisher 0.14
  - New "Connector" archetype for moderate playmakers
  - Movement Shooter must exceed Spot-Up to avoid overlap
  - Confidence uses sqrt saturation for headroom
  - Computes TS%/USG% from raw data when missing (2024-25)

Archetype Definitions:
======================
 1. Ball Dominant Creator  – High on-ball creation + high playmaking
 2. Ballhandler/Facilitator – High playmaking, ball handling, lower scoring
 3. Interior Scorer        – Ball dominant, non-playmaker, interior shot profile
 4. Perimeter Scorer       – Ball dominant, non-playmaker, perimeter shot profile
 5. All-Around Scorer      – Ball dominant, non-playmaker, balanced shot profile
 6. Connector              – Moderate playmaking, not ball dominant, facilitator
 7. Off-Ball Finisher      – High cut/PnR roll/transition
 8. Off-Ball Movement Shooter – High handoff/offscreen, more than spot-up
 9. Off-Ball Stationary Shooter – High spot-up
10. Rotation Piece/Glue Guy – Catch-all for remaining players
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

# Thresholds for classification
THRESHOLDS = {
    # Ball dominance: ON_BALL_CREATION (ISO+PnRBH) + POST_UP * 0.65
    'BALL_DOMINANT_PCT': 0.18,        # Minimum to be "ball dominant" (was 0.15)
    'HIGH_BALL_DOMINANT_PCT': 0.30,   # Very ball dominant (was 0.25)
    
    # Playmaking thresholds (AST per 36 min)
    'HIGH_PLAYMAKING': 5.5,           # High playmaker (was 5.0)
    'VERY_HIGH_PLAYMAKING': 7.5,      # Elite playmaker (was 7.0)
    'CONNECTOR_PLAYMAKING': 3.5,      # Moderate playmaking for Connector archetype
    'SECONDARY_AST_HIGH': 1.0,        # Secondary assists per 36
    
    # Scoring volume (PTS per 36 — replaces raw PPG)
    'HIGH_SCORING_PER36': 18.0,       # High-volume scorer (~P70)
    'LOW_SCORING_PER36': 10.0,        # Low-volume scorer
    
    # Usage thresholds
    'HIGH_USAGE': 0.22,
    'LOW_USAGE': 0.15,
    
    # Shot profile: FG2A_RATE = (FGA - FG3A) / FGA (replaces broken drives ratio)
    'INTERIOR_HEAVY': 0.70,           # FG2A_RATE >= this → interior (mid-range falls to balanced)
    'PERIMETER_HEAVY': 0.45,          # FG2A_RATE <= this → perimeter
    
    # Off-ball thresholds (possession %)
    'HIGH_CUT_PNRRM': 0.20,          # Cut + PnRRm combined (was 0.10, raised from 0.14)
    'HIGH_SPOTUP': 0.25,              # SpotUp possession % (was 0.20, now ~P50)
    'HIGH_MOVEMENT': 0.10,            # Handoff + OffScreen combined
    
    # Minutes / inflation guard
    'MIN_MINUTES': 500,               # Minimum total minutes
    'MIN_GP': 20,                     # Minimum games played
    'MIN_MPG': 15.0,                  # Minimum minutes per game (per-36 inflation guard)
    
    # Efficiency thresholds (TS%)
    'HIGH_EFFICIENCY': 0.58,          # Above-average TS% (~P65)
    'LOW_EFFICIENCY': 0.52,           # Below-average TS% (~P15)
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
    cols = ['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ABBREVIATION', 'SEASON', 'GP', 'MIN', 'PTS', 'AST', 'REB', 
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
    
    # Minutes per game (per-36 inflation guard)
    features['MPG'] = features['MIN'] / features['GP']
    
    # Per-36 minute stats
    minutes_factor = 36 / (features['MIN'] / features['GP']).replace(0, np.nan)
    features['PTS_PER36'] = (features['PTS'] / features['GP']) * minutes_factor
    features['AST_PER36'] = (features['AST'] / features['GP']) * minutes_factor
    features['REB_PER36'] = (features['REB'] / features['GP']) * minutes_factor
    features['TOV_PER36'] = (features['TOV'] / features['GP']) * minutes_factor
    features['STL_PER36'] = (features['STL'] / features['GP']) * minutes_factor
    features['BLK_PER36'] = (features['BLK'] / features['GP']) * minutes_factor
    
    # ---- Ball Dominance (SPLIT on-ball from post-up) ----
    # On-ball creation: ISO + PnR Ball Handler (true self-creation)
    features['ON_BALL_CREATION'] = (
        features.get('ISOLATION_POSS_PCT', 0) + 
        features.get('PRBALLHANDLER_POSS_PCT', 0)
    ).fillna(0)
    
    # Post creation (weighted at 0.65 — post-ups for bigs ≠ guard ISO)
    features['POST_CREATION'] = features.get('POSTUP_POSS_PCT', 0).fillna(0)
    
    # Combined ball dominance score
    features['BALL_DOMINANT_PCT'] = (
        features['ON_BALL_CREATION'] + features['POST_CREATION'] * 0.65
    ).fillna(0)
    
    # ---- Playmaking composite (with turnover penalty) ----
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
        features['POTENTIAL_AST_PER36'] * 0.3 -
        features['TOV_PER36'] * 0.5   # Turnover penalty
    ).fillna(0)
    
    # ---- Shot Profile: FG2A_RATE (replaces broken drives ratio) ----
    # FG2A_RATE = % of field goals that are 2-pointers
    features['FG2A_RATE'] = np.where(
        features['FGA'] > 0,
        (features['FGA'] - features['FG3A']) / features['FGA'],
        0.5  # Default balanced if no shots
    )
    
    features['FG3A_PER36'] = (features['FG3A'] / features['GP']) * minutes_factor
    
    # Keep drives for informational purposes (fix: tracking DRIVES is per-game, not season total)
    if 'DRIVES' in features.columns:
        # DRIVES from tracking is already per-game average, so just scale to per-36
        features['DRIVES_PER36'] = features['DRIVES'] * minutes_factor / (36 / (features['MIN'] / features['GP']).replace(0, np.nan))
        # Simplified: DRIVES is per-game, scale by minutes_factor/mpg_factor
        # Actually DRIVES is season total from tracking aggregation, need to / GP then * factor
        # But tracking load already sums across stints — so DRIVES is raw season total
        features['DRIVES_PER36'] = (features['DRIVES'] / features['GP']) * minutes_factor
    else:
        features['DRIVES_PER36'] = 0
    
    # Legacy INTERIOR_RATIO kept for backward compat but NOT used for classification
    features['INTERIOR_RATIO'] = features['FG2A_RATE']  # Now just an alias
    
    # ---- Off-ball frequencies ----
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
    
    # ---- Efficiency: compute TS% and USG% when missing ----
    # True Shooting %
    tsa = 2 * (features['FGA'] + 0.44 * features['FTA'])
    computed_ts = np.where(tsa > 0, features['PTS'] / tsa, np.nan)
    
    if 'TS_PCT' in features.columns:
        features['TS_PCT'] = features['TS_PCT'].fillna(pd.Series(computed_ts, index=features.index))
    else:
        features['TS_PCT'] = computed_ts
    
    # Effective FG%
    features['EFG_PCT'] = np.where(
        features['FGA'] > 0,
        (features['FGM'] + 0.5 * features['FG3M']) / features['FGA'],
        0
    )
    
    # USG% — compute from raw data when missing
    # Standard USG ≈ (FGA + 0.44*FTA + TOV) / (MIN/48 * TeamPoss/5)
    # Without team data, approximate: pace~100, so TeamPoss ~ 100*GP
    # Simplifies to: (FGA + 0.44*FTA + TOV) * 48 / (MIN * 100 / 5)
    #              = (FGA + 0.44*FTA + TOV) * 240 / (MIN * 100)
    #              = poss_used * 2.4 / (MIN * 5)
    # Result is a 0-1 fraction where 0.20 = 20% = league average
    poss_used = features['FGA'] + 0.44 * features['FTA'] + features['TOV']
    computed_usg = np.where(
        features['MIN'] > 0,
        poss_used * 2.4 / (features['MIN'] * 5),
        np.nan
    )
    
    if 'USG_PCT' in features.columns:
        features['USG_PCT'] = features['USG_PCT'].fillna(pd.Series(computed_usg, index=features.index))
    else:
        features['USG_PCT'] = computed_usg
    
    # Season-level league average TS for z-score
    league_avg_ts = features.loc[
        (features['MIN'] >= 500) & (features['GP'] >= 20) & (features['MPG'] >= 15),
        'TS_PCT'
    ].mean()
    league_std_ts = features.loc[
        (features['MIN'] >= 500) & (features['GP'] >= 20) & (features['MPG'] >= 15),
        'TS_PCT'
    ].std()
    
    if pd.isna(league_avg_ts):
        league_avg_ts = 0.565
    if pd.isna(league_std_ts) or league_std_ts == 0:
        league_std_ts = 0.04
    
    features['TS_ZSCORE'] = (features['TS_PCT'] - league_avg_ts) / league_std_ts
    features['LEAGUE_AVG_TS'] = league_avg_ts
    
    # Four Factors (informational)
    features['TOV_PCT'] = np.where(
        (features['FGA'] + 0.44 * features['FTA'] + features['TOV']) > 0,
        features['TOV'] / (features['FGA'] + 0.44 * features['FTA'] + features['TOV']),
        0
    )
    features['FT_RATE'] = np.where(features['FGA'] > 0, features['FTA'] / features['FGA'], 0)
    
    # Time of possession
    if 'TIME_OF_POSS' in features.columns:
        features['TIME_OF_POSS_PER36'] = (features['TIME_OF_POSS'] / features['GP']) * minutes_factor
    else:
        features['TIME_OF_POSS_PER36'] = 0
    
    if 'AVG_DRIB_PER_TOUCH' in features.columns:
        features['DRIBBLES_PER_TOUCH'] = features['AVG_DRIB_PER_TOUCH']
    else:
        features['DRIBBLES_PER_TOUCH'] = 0
    
    # Points per game (kept for display, not used in classification)
    features['PPG'] = features['PTS'] / features['GP']
    
    return features


# =============================================================================
# ARCHETYPE CLASSIFICATION
# =============================================================================
def classify_archetype(row: pd.Series) -> dict:
    """
    Classify a single player into an offensive archetype.
    
    v2 Classification hierarchy:
    0. Check minimum requirements (minutes, games, MPG)
    1. Ball Dominant Creator  — on-ball creation + high playmaking
    2. Ballhandler            — high playmaking, not high scorer
    3. Primary Scorers        — ball dominant, not playmaker → interior/perim/all-around
    4. Connector              — moderate playmaking, not ball dominant, not scorer
    5. Off-Ball Finisher      — high cut/PnR roll
    6. Off-Ball Movement Shooter — handoff+offscreen > spot-up
    7. Off-Ball Stationary Shooter — high spot-up
    8. Rotation Piece         — catch-all
    """
    
    result = {
        'primary_archetype': 'Unknown',
        'secondary_archetype': None,
        'archetype_confidence': 0.0,
        'ball_dominance_tier': 'Low',
        'playmaking_tier': 'Low',
        'scoring_tier': 'Low',
        'efficiency_tier': 'Average',
    }
    
    # ==========================================================================
    # STEP 0: MINIMUM REQUIREMENTS
    # ==========================================================================
    mpg = row.get('MPG', 0)
    if (row.get('MIN', 0) < THRESHOLDS['MIN_MINUTES'] or 
        row.get('GP', 0) < THRESHOLDS['MIN_GP'] or
        mpg < THRESHOLDS['MIN_MPG']):
        result['primary_archetype'] = 'Insufficient Minutes'
        return result
    
    # ==========================================================================
    # TIER ASSIGNMENTS
    # ==========================================================================
    
    # Ball dominance tier (using new split formula)
    ball_dom = row.get('BALL_DOMINANT_PCT', 0)
    on_ball = row.get('ON_BALL_CREATION', 0)
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
    
    # Scoring tier (PTS_PER36 replaces raw PPG)
    pts_per36 = row.get('PTS_PER36', 0)
    if pts_per36 >= THRESHOLDS['HIGH_SCORING_PER36']:
        result['scoring_tier'] = 'High'
    elif pts_per36 >= THRESHOLDS['LOW_SCORING_PER36']:
        result['scoring_tier'] = 'Medium'
    else:
        result['scoring_tier'] = 'Low'
    
    # Efficiency tier (TS%)
    ts = row.get('TS_PCT', 0)
    ts_z = row.get('TS_ZSCORE', 0)
    if pd.isna(ts) or ts == 0:
        result['efficiency_tier'] = 'Unknown'
        efficiency_factor = 1.0
    elif ts >= THRESHOLDS['HIGH_EFFICIENCY']:
        result['efficiency_tier'] = 'High'
        efficiency_factor = min(1.1, 1.0 + ts_z * 0.05)  # Small boost
    elif ts <= THRESHOLDS['LOW_EFFICIENCY']:
        result['efficiency_tier'] = 'Low'
        efficiency_factor = max(0.8, 1.0 + ts_z * 0.05)   # Small penalty
    else:
        result['efficiency_tier'] = 'Average'
        efficiency_factor = 1.0
    
    # Usage tier
    usg = row.get('USG_PCT', 0.15)
    if pd.isna(usg):
        usg = 0.15
    high_usage = usg >= THRESHOLDS['HIGH_USAGE']
    low_usage = usg < THRESHOLDS['LOW_USAGE']
    
    # ==========================================================================
    # CLASSIFICATION LOGIC
    # ==========================================================================
    
    is_ball_dominant = result['ball_dominance_tier'] in ['High', 'Very High']
    is_playmaker = result['playmaking_tier'] in ['High', 'Elite']
    is_high_volume_scorer = result['scoring_tier'] == 'High'
    
    # Shot profile: FG2A_RATE
    fg2a_rate = row.get('FG2A_RATE', 0.5)
    is_interior = fg2a_rate >= THRESHOLDS['INTERIOR_HEAVY']
    is_perimeter = fg2a_rate <= THRESHOLDS['PERIMETER_HEAVY']
    
    # Off-ball tendencies
    cut_pnrrm = row.get('CUT_PNRRM_PCT', 0)
    spotup = row.get('SPOTUP_PCT', 0)
    movement = row.get('MOVEMENT_SHOOTER_PCT', 0)
    transition = row.get('TRANSITION_PCT', 0)
    putback = row.get('PUTBACK_PCT', 0)
    
    # ---------------------------------------------------------------------------
    # 1. BALL DOMINANT CREATOR
    # High on-ball creation + High playmaking
    # ---------------------------------------------------------------------------
    if is_ball_dominant and is_playmaker:
        result['primary_archetype'] = 'Ball Dominant Creator'
        # sqrt saturation prevents ceiling compression
        raw_conf = (np.sqrt(ball_dom / 0.35) + np.sqrt(playmaking / 10)) / 2
        result['archetype_confidence'] = min(1.0, raw_conf * efficiency_factor)
        if is_high_volume_scorer:
            result['secondary_archetype'] = 'Primary Scorer'
        return result
    
    # ---------------------------------------------------------------------------
    # 1b. OFFENSIVE HUB (high scoring + high playmaking, not traditional BD)
    # Catches post-up hubs like Sabonis, Vucevic who create through non-ISO means
    # ---------------------------------------------------------------------------
    if is_playmaker and is_high_volume_scorer:
        result['primary_archetype'] = 'Ball Dominant Creator'
        result['secondary_archetype'] = 'Offensive Hub'
        raw_conf = (np.sqrt(ast_per36 / 8.0) + np.sqrt(pts_per36 / 25.0)) / 2
        result['archetype_confidence'] = min(1.0, raw_conf * efficiency_factor)
        return result
    
    # ---------------------------------------------------------------------------
    # 2. BALLHANDLER (Facilitator)
    # High playmaking but not the primary scoring option
    # Since BDC already caught ball_dom + playmaker, this catches
    # non-ball-dominant playmakers (pure facilitators)
    # ---------------------------------------------------------------------------
    if is_playmaker and not is_high_volume_scorer:
        result['primary_archetype'] = 'Ballhandler'
        result['archetype_confidence'] = min(1.0, np.sqrt(playmaking / 10) * efficiency_factor)
        return result
    
    # ---------------------------------------------------------------------------
    # 3. PRIMARY SCORERS (Ball Dominant, Not High Playmaking)
    # Uses FG2A_RATE for interior vs perimeter split
    # ---------------------------------------------------------------------------
    if is_ball_dominant and not is_playmaker:
        if is_interior:
            result['primary_archetype'] = 'Interior Scorer'
            raw_conf = np.sqrt(ball_dom / 0.25) * (fg2a_rate / 0.75)
            result['archetype_confidence'] = min(1.0, raw_conf * efficiency_factor)
        elif is_perimeter:
            result['primary_archetype'] = 'Perimeter Scorer'
            raw_conf = np.sqrt(ball_dom / 0.25) * ((1 - fg2a_rate) / 0.65)
            result['archetype_confidence'] = min(1.0, raw_conf * efficiency_factor)
        else:
            result['primary_archetype'] = 'All-Around Scorer'
            raw_conf = np.sqrt(ball_dom / 0.25)
            result['archetype_confidence'] = min(1.0, raw_conf * efficiency_factor)
        
        if is_high_volume_scorer:
            result['secondary_archetype'] = 'High Volume'
        return result
    
    # ---------------------------------------------------------------------------
    # 4. CONNECTOR (Moderate playmaker, not ball dominant, not primary scorer)
    # Catches facilitative wings/bigs: Draymond, Batum, Derrick White types
    # ---------------------------------------------------------------------------
    if (ast_per36 >= THRESHOLDS['CONNECTOR_PLAYMAKING'] and
        not is_ball_dominant and 
        not is_high_volume_scorer):
        # Additional connector signals: secondary assists, touches
        sec_ast = row.get('SECONDARY_AST_PER36', 0)
        has_connector_profile = (
            sec_ast >= 0.3 or  # Creates for others
            ast_per36 >= 4.0 or  # Strong pure AST
            (ast_per36 >= 3.5 and row.get('DRIBBLES_PER_TOUCH', 0) >= 1.5)
        )
        if has_connector_profile:
            result['primary_archetype'] = 'Connector'
            raw_conf = np.sqrt(ast_per36 / 6.0)
            result['archetype_confidence'] = min(1.0, raw_conf * efficiency_factor)
            if row.get('REB_PER36', 0) > 8:
                result['secondary_archetype'] = 'Playmaking Big'
            elif sec_ast >= 0.8:
                result['secondary_archetype'] = 'Hockey Assist Specialist'
            return result
    
    # ---------------------------------------------------------------------------
    # 5. OFF-BALL FINISHER
    # High Cut, PnR Roll, Putback, Transition
    # ---------------------------------------------------------------------------
    offball_finish_score = cut_pnrrm + putback * 0.5 + transition * 0.3
    if cut_pnrrm >= THRESHOLDS['HIGH_CUT_PNRRM'] or offball_finish_score > 0.20:
        result['primary_archetype'] = 'Off-Ball Finisher'
        raw_conf = np.sqrt(offball_finish_score / 0.25)
        result['archetype_confidence'] = min(1.0, raw_conf * efficiency_factor)
        if transition > 0.10:
            result['secondary_archetype'] = 'Transition Player'
        return result
    
    # ---------------------------------------------------------------------------
    # 6. OFF-BALL MOVEMENT SHOOTER
    # High Handoff+OffScreen, AND must exceed Spot-Up (prevents overlap)
    # ---------------------------------------------------------------------------
    if movement >= THRESHOLDS['HIGH_MOVEMENT'] and movement > spotup:
        result['primary_archetype'] = 'Off-Ball Movement Shooter'
        raw_conf = np.sqrt(movement / 0.18)
        result['archetype_confidence'] = min(1.0, raw_conf * efficiency_factor)
        return result
    
    # ---------------------------------------------------------------------------
    # 7. OFF-BALL STATIONARY SHOOTER
    # High Spot-Up
    # ---------------------------------------------------------------------------
    if spotup >= THRESHOLDS['HIGH_SPOTUP']:
        result['primary_archetype'] = 'Off-Ball Stationary Shooter'
        raw_conf = np.sqrt(spotup / 0.35)
        result['archetype_confidence'] = min(1.0, raw_conf * efficiency_factor)
        fg3_pct = row.get('FG3_PCT', 0)
        if fg3_pct > 0.38:
            result['secondary_archetype'] = 'Elite Shooter'
        return result
    
    # ---------------------------------------------------------------------------
    # 8. ROTATION PIECE / GLUE GUY (Catch-All)
    # ---------------------------------------------------------------------------
    result['primary_archetype'] = 'Rotation Piece'
    
    # Determine specialization
    if row.get('REB_PER36', 0) > 8:
        result['secondary_archetype'] = 'Rebounder'
    elif row.get('STL_PER36', 0) > 1.5:
        result['secondary_archetype'] = 'Perimeter Defender'
    elif row.get('BLK_PER36', 0) > 1.2:
        result['secondary_archetype'] = 'Rim Protector'
    else:
        result['secondary_archetype'] = 'Glue Guy'
    
    result['archetype_confidence'] = min(0.65, 0.55 * efficiency_factor)
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
    print("PLAYER ARCHETYPE CLASSIFICATION (v2)")
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
        qual = classified[
            (classified['MIN'] >= THRESHOLDS['MIN_MINUTES']) &
            (classified['GP'] >= THRESHOLDS['MIN_GP']) &
            (classified['MPG'] >= THRESHOLDS['MIN_MPG'])
        ]
        archetype_counts = qual['primary_archetype'].value_counts()
        print(f"   Classified {len(qual)} qualified players:")
        for arch, count in archetype_counts.items():
            pct = count / len(qual) * 100
            print(f"      {arch}: {count} ({pct:.1f}%)")
    
    # Combine all seasons
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        
        # Save results
        final_df.to_parquet(OUTPUT_DIR / "player_archetypes.parquet", index=False)
        final_df.to_csv(OUTPUT_DIR / "player_archetypes.csv", index=False)
        
        print(f"\n✅ Saved {len(final_df)} player-seasons to data/processed/player_archetypes.parquet")
        
        # Print validation for known players
        print("\n" + "=" * 70)
        print("VALIDATION — KEY PLAYER CLASSIFICATIONS")
        print("=" * 70)
        
        s25 = final_df[final_df['SEASON'] == '2024-25'].copy()
        stars = [
            'LeBron James', 'Stephen Curry', 'Kevin Durant', 'Giannis Antetokounmpo',
            'Luka Dončić', 'Nikola Jokić', 'Jayson Tatum', 'Jonathan Kuminga',
            'Shai Gilgeous-Alexander', 'Anthony Edwards', 'Draymond Green',
            'Jalen Brunson', 'Tyrese Haliburton', 'Derrick White',
            'Joel Embiid', 'Clint Capela', 'Rudy Gobert', 'Nicolas Batum'
        ]
        
        for name in stars:
            player = s25[s25['PLAYER_NAME'].str.contains(name.split()[0], case=False, na=False)]
            if len(player) > 0:
                p = player.iloc[0]
                secondary = f" / {p['secondary_archetype']}" if p['secondary_archetype'] else ""
                eff = f" [{p['efficiency_tier']}]"
                fg2 = f" FG2A={p['FG2A_RATE']:.0%}"
                print(f"  {p['PLAYER_NAME']:<25} → {p['primary_archetype']}{secondary} ({p['archetype_confidence']:.0%}){eff}{fg2}")
            else:
                # Try 2023-24
                s24 = final_df[final_df['SEASON'] == '2023-24']
                player = s24[s24['PLAYER_NAME'].str.contains(name.split()[0], case=False, na=False)]
                if len(player) > 0:
                    p = player.iloc[0]
                    secondary = f" / {p['secondary_archetype']}" if p['secondary_archetype'] else ""
                    print(f"  {p['PLAYER_NAME']:<25} → {p['primary_archetype']}{secondary} ({p['archetype_confidence']:.0%}) [{p['SEASON']}]")
    
    return final_df


if __name__ == "__main__":
    result = main()
