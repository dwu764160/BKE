"""
src/data_compute/compute_player_archetypes.py
Classifies NBA players into offensive archetypes based on tracking, playtype, and shot zone data.

v4 — Shot Zone + Classification Hierarchy Overhaul:
  - Added shot zone data (AT_RIM_FREQ, MIDRANGE_FREQ, PAINT_FREQ) from NBA.com
  - "All-Around Scorer" now placed BEFORE Ballhandler (if playmaker + high scorer)
  - All-Around gate lowered from P60 to P50 scoring
  - Interior Scorer split: Rim Finisher vs Midrange Scorer (using AT_RIM_FREQ)
  - Rotation Piece ELIMINATED — replaced with best-fit fallback
  - Ball dominance + scoring output checked FIRST, then playtype frequency
  - BDC subtypes: Heliocentric Guard | Post Hub | Gravity Engine
  - PnR Rolling/Popping Big split from Off-Ball Finisher
  - Dual confidence: role_confidence (fit certainty) + role_effectiveness (production)

Archetype Definitions (v4):
=============================
 1. Ball Dominant Creator  — High on-ball creation + high playmaking
      Subtypes: Heliocentric Guard | Post Hub | Gravity Engine
 2. Offensive Hub          — Elite playmaker + elite scorer
 3. All-Around Scorer      — Playmaker + high scorer (placed before Ballhandler)
      Subtypes: High Volume | Midrange Scorer
 4. Ballhandler/Facilitator — High playmaking, lower scoring
 5. Interior Scorer        — Ball dominant, non-playmaker, interior-heavy
      Subtypes: Rim Finisher | Midrange Scorer | High Volume
 6. Perimeter Scorer       — Ball dominant, non-playmaker, perimeter FG2A <= P30
 7. Balanced Scorer        — Ball dominant, moderate scoring, mixed shot profile
 8. Connector              — Moderate playmaking, active filter
 9. PnR Rolling Big        — Off-ball, primary roll man, low 3PA
10. PnR Popping Big        — Off-ball, primary roll man, high 3PA
11. Off-Ball Finisher      — High cut/PnR roll/transition
12. Off-Ball Movement Shooter — movement/(movement+spotup) >= 0.4
13. Off-Ball Stationary Shooter — High spot-up
14. Best-fit fallback      — Closest archetype based on scoring signals (no catch-all)
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

# ---------- Static thresholds (only where percentiles don't apply) ----------
STATIC = {
    'MIN_MINUTES': 500,
    'MIN_GP': 20,
    'MIN_MPG': 15.0,
    # BDC subtypes (structural, not volume-based)
    'HELIOCENTRIC_ON_BALL': 0.30,   # ON_BALL_CREATION >= this → Heliocentric Guard
    'POST_HUB_POSS': 0.10,          # POSTUP_POSS_PCT >= this AND must be dominant mode
    'POST_HUB_DOMINANT': 0.15,       # POST >= this always qualifies as Post Hub
    # (original POST_HUB_POSS line replaced)
    # Movement Shooter ratio
    'MOVEMENT_RATIO_MIN': 0.40,     # movement / (movement + spotup) >= 0.40
    # Connector activity filter
    'CONNECTOR_SEC_PER_TOUCH_MAX': 2.5,  # AVG_SEC_PER_TOUCH <= this (active, not holding ball long)
    # Post-up weighting in ball dominance
    'POST_WEIGHT': 0.65,
}

# ---------- Percentile-based thresholds (computed per-season) ----------
# Format: (column_name, percentile)
# These are resolved to actual values at runtime from per-season distributions
PCTILE_THRESHOLDS = {
    # Ball Dominant Creator gates
    'BD_MIN':               ('BALL_DOMINANT_PCT', 80),    # >= P80 for BDC path (tighter v3)
    'BD_ALL_AROUND':        ('BALL_DOMINANT_PCT', 60),    # >= P60 for All-Around
    'BD_BASE':              ('BALL_DOMINANT_PCT', 50),    # >= P50 for any ball-dom consideration
    # Playmaking gates
    'PLAYMAKER':            ('AST_PER36', 75),            # >= P75 for playmaker
    'ELITE_PLAYMAKER':      ('AST_PER36', 85),            # >= P85 elite playmaker
    'CONNECTOR_AST':        ('AST_PER36', 50),            # >= P50 for connector
    # Scoring volume
    'HIGH_SCORING':         ('PTS_PER36', 70),            # >= P70 high volume
    'ELITE_SCORING':        ('PTS_PER36', 80),            # >= P80 elite volume
    'ALL_AROUND_SCORING':   ('PTS_PER36', 60),            # >= P60 for all-around filer
    'LOW_SCORING':          ('PTS_PER36', 25),            # <= P25 low scorer
    # Shot profile
    'FG2A_INTERIOR':        ('FG2A_RATE', 70),            # >= P70 interior heavy
    'FG2A_PERIMETER':       ('FG2A_RATE', 30),            # <= P30 perimeter heavy
    # Off-ball
    'HIGH_CUT_PNRRM':      ('CUT_PNRRM_PCT', 75),       # >= P75 cut + roll man
    'HIGH_SPOTUP':          ('SPOTUP_PCT', 50),           # >= P50 spot-up
    'HIGH_MOVEMENT':        ('MOVEMENT_SHOOTER_PCT', 70), # >= P70 movement shooter
    'HIGH_PRROLLMAN':       ('PRROLLMAN_POSS_PCT', 70),   # >= P70 roll man (for PnR Big)
    # Touches / activity (for Connector)
    'TOUCHES_MEDIAN':       ('TOUCHES', 50),              # >= P50 touches
    # Efficiency
    'HIGH_EFFICIENCY':      ('TS_PCT', 70),               # >= P70 efficient
    'LOW_EFFICIENCY':       ('TS_PCT', 30),               # <= P30 inefficient
    # Gravity Engine
    'HIGH_FG3A':            ('FG3A_PER36', 80),           # >= P80 3PA volume
    'HIGH_FG3_PCT':         ('FG3_PCT', 70),              # >= P70 3P%
    # BDC Heliocentric subtype
    'BD_HELIOCENTRIC':      ('BALL_DOMINANT_PCT', 85),    # >= P85 for Heliocentric subtype
    # P95 caps for composite normalization
    'BD_P95':               ('BALL_DOMINANT_PCT', 95),    # P95 for composite norm
    'PTS_P95':              ('PTS_PER36', 95),            # P95 for composite norm
    'AST_P95':              ('AST_PER36', 95),            # P95 for composite norm
    # Scoring gate for BDC main path
    'MODERATE_SCORING':     ('PTS_PER36', 50),            # >= P50 BDC must be above-avg scorer
    # Playmaking score
    'HIGH_PLAYMAKING_SCORE':('PLAYMAKING_SCORE', 75),     # >= P75
    # FG3A for PnR pop vs roll
    'FG3A_MEDIAN':          ('FG3A_PER36', 50),           # >= P50 → popping, < P50 → rolling
    # Shot zone features (from LeagueDashPlayerShotLocations)
    'HIGH_AT_RIM':          ('AT_RIM_FREQ', 70),          # >= P70 rim finisher
    'HIGH_MIDRANGE':        ('MIDRANGE_FREQ', 70),        # >= P70 midrange scorer
    'LOW_MIDRANGE':         ('MIDRANGE_FREQ', 30),        # <= P30 non-midrange
    'MODERATE_MIDRANGE':    ('MIDRANGE_FREQ', 50),        # >= P50 midrange leaning
    'AT_RIM_PAINT_P60':     ('AT_RIM_PLUS_PAINT_FREQ', 60), # >= P60 interior shot location
}

# ---------- Frozen canonical metric vectors for best-fit fallback ----------
# Prevents silent regressions from feature creep.  Each entry:
#   (row_column, pctile_key_for_denominator_or_None, static_fallback_denom, weight)
FALLBACK_VECTORS = {
    'Off-Ball Stationary Shooter': [
        ('SPOTUP_PCT',           'HIGH_SPOTUP',     0.10, 1.0),
    ],
    'Off-Ball Movement Shooter': [
        ('MOVEMENT_SHOOTER_PCT', 'HIGH_MOVEMENT',   0.05, 1.0),
    ],
    'Off-Ball Finisher': [
        ('CUT_PNRRM_PCT',        None,             0.15, 1.0),
        ('PUTBACK_PCT',           None,             0.15, 0.5),
        ('TRANSITION_PCT',        None,             0.15, 0.3),
    ],
    'Connector': [
        ('AST_PER36',            'CONNECTOR_AST',   2.0,  1.0),
    ],
    'Interior Scorer': [
        ('FG2A_RATE',             None,             0.85, 0.5),
        ('PTS_PER36',            'HIGH_SCORING',    14.0, 0.5),
    ],
    'PnR Rolling Big': [
        ('PRROLLMAN_POSS_PCT',   'HIGH_PRROLLMAN',  0.04, 1.0),
    ],
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
            agg_df = df.groupby('PLAYER_ID').agg({
                'PLAYER_NAME': 'first',
                'POSS_PCT': 'mean',
                'PPP': 'mean',
                'POSS': 'sum'
            }).reset_index()

            agg_df = agg_df.rename(columns={
                'POSS_PCT': f'{playtype.upper()}_POSS_PCT',
                'PPP': f'{playtype.upper()}_PPP',
                'POSS': f'{playtype.upper()}_POSS'
            })
            all_data.append(agg_df)

    if not all_data:
        return pd.DataFrame()

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
            available_cols = [c for c in keep_cols if c in df.columns]
            if available_cols:
                df = df[available_cols]
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
    cols = ['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ABBREVIATION', 'SEASON', 'GP', 'MIN', 'PTS', 'AST', 'REB',
            'OREB', 'DREB', 'STL', 'BLK', 'TOV', 'FGA', 'FGM', 'FG3A', 'FG3M',
            'FTA', 'FTM', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'USG_PCT', 'TS_PCT']
    available_cols = [c for c in cols if c in df.columns]
    return df[available_cols]


def load_shot_zone_data(season: str) -> pd.DataFrame:
    """Load shot zone data (from LeagueDashPlayerShotLocations).
    Returns DataFrame with AT_RIM_FREQ, MIDRANGE_FREQ, PAINT_FREQ, etc."""
    path = TRACKING_DIR / season / "shot_zones.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    # Keep only the columns we need for archetype classification
    keep_cols = ['PLAYER_ID', 'AT_RIM_FREQ', 'PAINT_FREQ', 'MIDRANGE_FREQ',
                 'CORNER3_FREQ', 'AB3_FREQ', 'AT_RIM_PLUS_PAINT_FREQ',
                 'AT_RIM_FG_PCT', 'MIDRANGE_FG_PCT',
                 'RA_FGA', 'MR_FGA', 'PAINT_FGA', 'TOTAL_FGA']
    available = [c for c in keep_cols if c in df.columns]
    return df[available]


# =============================================================================
# PERCENTILE COMPUTATION
# =============================================================================
def compute_season_percentiles(features: pd.DataFrame) -> dict:
    """
    Compute percentile thresholds for a season's qualified players.
    Returns dict mapping threshold_name -> actual_value.
    """
    qualified = features[
        (features['MPG'] >= STATIC['MIN_MPG']) &
        (features['GP'] >= STATIC['MIN_GP']) &
        (features['MIN'] >= STATIC['MIN_MINUTES'])
    ]

    resolved = {}
    for name, (col, pct) in PCTILE_THRESHOLDS.items():
        if col in qualified.columns:
            vals = qualified[col].dropna()
            if len(vals) > 0:
                resolved[name] = float(np.percentile(vals, pct))
            else:
                resolved[name] = 0.0
        else:
            resolved[name] = 0.0

    return resolved


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
def compute_archetype_features(synergy: pd.DataFrame, tracking: pd.DataFrame,
                                box: pd.DataFrame, season: str,
                                shot_zones: pd.DataFrame = None) -> pd.DataFrame:
    """Compute all features needed for archetype classification."""

    syn = synergy[synergy['SEASON'] == season].copy() if not synergy.empty else pd.DataFrame()
    trk = tracking[tracking['SEASON'] == season].copy() if not tracking.empty else pd.DataFrame()
    bx = box[box['SEASON'] == season].copy() if not box.empty else pd.DataFrame()

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

    # Merge shot zone data
    if shot_zones is not None and not shot_zones.empty:
        sz = shot_zones.copy()
        sz_cols = [c for c in sz.columns if c == 'PLAYER_ID' or c not in features.columns]
        features = features.merge(sz[sz_cols], on='PLAYER_ID', how='left')

    # Fill NaN playtype data with 0
    playtype_cols = [c for c in features.columns if '_POSS_PCT' in c or '_PPP' in c]
    features[playtype_cols] = features[playtype_cols].fillna(0)

    # Fill shot zone NaN with 0
    zone_cols = [c for c in features.columns if c in
                 ('AT_RIM_FREQ', 'PAINT_FREQ', 'MIDRANGE_FREQ', 'CORNER3_FREQ',
                  'AB3_FREQ', 'AT_RIM_PLUS_PAINT_FREQ', 'AT_RIM_FG_PCT',
                  'MIDRANGE_FG_PCT', 'RA_FGA', 'MR_FGA', 'PAINT_FGA', 'TOTAL_FGA')]
    for col in zone_cols:
        features[col] = features[col].fillna(0)

    # ==========================================================================
    # COMPUTE DERIVED FEATURES
    # ==========================================================================

    features['MPG'] = features['MIN'] / features['GP']

    # Per-36 minute stats
    minutes_factor = 36 / (features['MIN'] / features['GP']).replace(0, np.nan)
    features['PTS_PER36'] = (features['PTS'] / features['GP']) * minutes_factor
    features['AST_PER36'] = (features['AST'] / features['GP']) * minutes_factor
    features['REB_PER36'] = (features['REB'] / features['GP']) * minutes_factor
    features['TOV_PER36'] = (features['TOV'] / features['GP']) * minutes_factor
    features['STL_PER36'] = (features['STL'] / features['GP']) * minutes_factor
    features['BLK_PER36'] = (features['BLK'] / features['GP']) * minutes_factor

    # ---- Ball Dominance (split on-ball from post-up) ----
    features['ON_BALL_CREATION'] = (
        features.get('ISOLATION_POSS_PCT', 0) +
        features.get('PRBALLHANDLER_POSS_PCT', 0)
    ).fillna(0)

    features['POST_CREATION'] = features.get('POSTUP_POSS_PCT', 0).fillna(0)

    features['BALL_DOMINANT_PCT'] = (
        features['ON_BALL_CREATION'] + features['POST_CREATION'] * STATIC['POST_WEIGHT']
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
        features['TOV_PER36'] * 0.5
    ).fillna(0)

    # ---- Shot Profile: FG2A_RATE ----
    features['FG2A_RATE'] = np.where(
        features['FGA'] > 0,
        (features['FGA'] - features['FG3A']) / features['FGA'],
        0.5
    )

    features['FG3A_PER36'] = (features['FG3A'] / features['GP']) * minutes_factor

    # Drives
    if 'DRIVES' in features.columns:
        features['DRIVES_PER36'] = (features['DRIVES'] / features['GP']) * minutes_factor
    else:
        features['DRIVES_PER36'] = 0

    features['INTERIOR_RATIO'] = features['FG2A_RATE']  # Legacy alias

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
    tsa = 2 * (features['FGA'] + 0.44 * features['FTA'])
    computed_ts = np.where(tsa > 0, features['PTS'] / tsa, np.nan)

    if 'TS_PCT' in features.columns:
        features['TS_PCT'] = features['TS_PCT'].fillna(pd.Series(computed_ts, index=features.index))
    else:
        features['TS_PCT'] = computed_ts

    features['EFG_PCT'] = np.where(
        features['FGA'] > 0,
        (features['FGM'] + 0.5 * features['FG3M']) / features['FGA'],
        0
    )

    # USG proxy (soft signal only — never used as hard gate)
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

    features['PPG'] = features['PTS'] / features['GP']

    return features


# =============================================================================
# ROLE EFFECTIVENESS
# =============================================================================
def compute_role_effectiveness(row: pd.Series, archetype: str, pctiles: dict) -> float:
    """
    Compute how productive a player is within their role.
    Independent of classification certainty — measures performance quality.
    Returns 0.0 to 1.0.
    """
    ts_z = row.get('TS_ZSCORE', 0)
    if pd.isna(ts_z):
        ts_z = 0.0
    # Base efficiency score from TS z-score (normalized: -2sigma=0, +2sigma=1)
    eff_score = float(np.clip((ts_z + 2) / 4, 0.05, 1.0))

    pts36 = row.get('PTS_PER36', 0)
    if pd.isna(pts36):
        pts36 = 0.0
    pts36_cap = max(pctiles.get('ELITE_SCORING', 21.0), 15.0)
    volume_score = float(np.clip(pts36 / pts36_cap, 0, 1.0))

    if archetype == 'Ball Dominant Creator':
        iso_ppp = row.get('ISOLATION_PPP', 0) or 0
        prbh_ppp = row.get('PRBALLHANDLER_PPP', 0) or 0
        primary_ppp = max(iso_ppp, prbh_ppp)
        ppp_score = float(np.clip(primary_ppp / 1.0, 0, 1.0))
        return 0.35 * eff_score + 0.35 * volume_score + 0.30 * ppp_score

    elif archetype in ('Interior Scorer', 'Perimeter Scorer', 'All-Around Scorer'):
        return 0.50 * eff_score + 0.50 * volume_score

    elif archetype == 'Ballhandler':
        ast36 = row.get('AST_PER36', 0) or 0
        ast_cap = max(pctiles.get('ELITE_PLAYMAKER', 6.0), 4.0)
        ast_score = float(np.clip(ast36 / ast_cap, 0, 1.0))
        return 0.35 * eff_score + 0.25 * volume_score + 0.40 * ast_score

    elif archetype == 'Connector':
        ast36 = row.get('AST_PER36', 0) or 0
        sec_ast = row.get('SECONDARY_AST_PER36', 0) or 0
        connect_score = float(np.clip((ast36 + sec_ast * 5) / 8.0, 0, 1.0))
        return 0.30 * eff_score + 0.20 * volume_score + 0.50 * connect_score

    elif archetype in ('PnR Rolling Big', 'PnR Popping Big'):
        prm_ppp = row.get('PRROLLMAN_PPP', 0) or 0
        ppp_score = float(np.clip(prm_ppp / 1.2, 0, 1.0))
        return 0.35 * eff_score + 0.30 * volume_score + 0.35 * ppp_score

    elif archetype == 'Off-Ball Finisher':
        cut_ppp = row.get('CUT_PPP', 0) or 0
        ppp_score = float(np.clip(cut_ppp / 1.3, 0, 1.0))
        return 0.35 * eff_score + 0.30 * volume_score + 0.35 * ppp_score

    elif archetype == 'Off-Ball Movement Shooter':
        fg3 = row.get('FG3_PCT', 0) or 0
        offscr_ppp = row.get('OFFSCREEN_PPP', 0) or 0
        shoot_score = float(np.clip((fg3 - 0.30) / 0.12, 0, 1.0))
        ppp_score = float(np.clip(offscr_ppp / 1.1, 0, 1.0))
        return 0.25 * eff_score + 0.15 * volume_score + 0.35 * shoot_score + 0.25 * ppp_score

    elif archetype == 'Off-Ball Stationary Shooter':
        fg3 = row.get('FG3_PCT', 0) or 0
        spotup_ppp = row.get('SPOTUP_PPP', 0) or 0
        shoot_score = float(np.clip((fg3 - 0.30) / 0.12, 0, 1.0))
        ppp_score = float(np.clip(spotup_ppp / 1.15, 0, 1.0))
        return 0.25 * eff_score + 0.15 * volume_score + 0.35 * shoot_score + 0.25 * ppp_score

    # Default (Rotation Piece etc.)
    return 0.50 * eff_score + 0.50 * volume_score


# =============================================================================
# ARCHETYPE CLASSIFICATION (v3)
# =============================================================================
def classify_archetype(row: pd.Series, pctiles: dict) -> dict:
    """
    Classify a single player into an offensive archetype.

    v3 Classification hierarchy:
    0. Check minimum requirements (minutes, games, MPG)
    1. Ball Dominant Creator  — ball_dom >= P70 AND AST >= P75
       1b. Offensive Hub      — AST >= P80 AND PTS/36 >= P80 (catches post-up hubs)
    2. Ballhandler            — high playmaking, not high scorer
    3. Primary Scorers        — ball dominant (>= P60), scorer → interior/perim/all-around
    4. Connector              — AST >= P50 + active touches filter
    5. PnR Big (Rolling/Popping) — primary off-ball play is PnR Roll Man
    6. Off-Ball Finisher      — high cut/PnR roll/transition
    7. Off-Ball Movement Shooter — movement/(movement+spotup) >= 0.4
    8. Off-Ball Stationary Shooter — spotup >= P50
    9. Rotation Piece         — catch-all
    """

    result = {
        'primary_archetype': 'Unknown',
        'secondary_archetype': None,
        'role_confidence': 0.0,
        'role_effectiveness': 0.0,
        'ball_dominance_tier': 'Low',
        'playmaking_tier': 'Low',
        'scoring_tier': 'Low',
        'efficiency_tier': 'Average',
    }

    # ==========================================================================
    # STEP 0: MINIMUM REQUIREMENTS
    # ==========================================================================
    mpg = row.get('MPG', 0)
    if (row.get('MIN', 0) < STATIC['MIN_MINUTES'] or
        row.get('GP', 0) < STATIC['MIN_GP'] or
        mpg < STATIC['MIN_MPG']):
        result['primary_archetype'] = 'Insufficient Minutes'
        return result

    # ==========================================================================
    # EXTRACT KEY METRICS
    # ==========================================================================
    ball_dom = row.get('BALL_DOMINANT_PCT', 0) or 0
    on_ball = row.get('ON_BALL_CREATION', 0) or 0
    post_up = row.get('POSTUP_POSS_PCT', 0) or 0
    ast_per36 = row.get('AST_PER36', 0) or 0
    pts_per36 = row.get('PTS_PER36', 0) or 0
    playmaking = row.get('PLAYMAKING_SCORE', 0) or 0
    fg2a_rate = row.get('FG2A_RATE', 0.5)
    if pd.isna(fg2a_rate):
        fg2a_rate = 0.5
    fg3a_per36 = row.get('FG3A_PER36', 0) or 0
    fg3_pct = row.get('FG3_PCT', 0) or 0
    ts = row.get('TS_PCT', 0)
    ts_z = row.get('TS_ZSCORE', 0)
    if pd.isna(ts) or ts == 0:
        ts = 0.55
    if pd.isna(ts_z):
        ts_z = 0.0
    usg = row.get('USG_PCT', 0.18)
    if pd.isna(usg):
        usg = 0.18

    # Off-ball
    cut_pnrrm = row.get('CUT_PNRRM_PCT', 0) or 0
    spotup = row.get('SPOTUP_PCT', 0) or 0
    movement = row.get('MOVEMENT_SHOOTER_PCT', 0) or 0
    transition = row.get('TRANSITION_PCT', 0) or 0
    putback = row.get('PUTBACK_PCT', 0) or 0
    prrollman = row.get('PRROLLMAN_POSS_PCT', 0) or 0
    cut_poss = row.get('CUT_POSS_PCT', 0) or 0

    # Tracking
    touches = row.get('TOUCHES', 0) or 0
    sec_per_touch = row.get('AVG_SEC_PER_TOUCH', 3.0)
    if pd.isna(sec_per_touch):
        sec_per_touch = 3.0
    sec_ast = row.get('SECONDARY_AST_PER36', 0) or 0

    # Shot zones (from LeagueDashPlayerShotLocations)
    at_rim_freq = row.get('AT_RIM_FREQ', 0) or 0
    if pd.isna(at_rim_freq):
        at_rim_freq = 0
    midrange_freq = row.get('MIDRANGE_FREQ', 0) or 0
    if pd.isna(midrange_freq):
        midrange_freq = 0
    paint_freq = row.get('PAINT_FREQ', 0) or 0
    if pd.isna(paint_freq):
        paint_freq = 0
    at_rim_plus_paint = row.get('AT_RIM_PLUS_PAINT_FREQ', 0) or 0
    if pd.isna(at_rim_plus_paint):
        at_rim_plus_paint = 0
    midrange_fg_pct = row.get('MIDRANGE_FG_PCT', 0) or 0
    if pd.isna(midrange_fg_pct):
        midrange_fg_pct = 0

    # ==========================================================================
    # RESOLVE PERCENTILE THRESHOLDS
    # ==========================================================================
    p = pctiles  # shorthand

    # ==========================================================================
    # TIER ASSIGNMENTS (informational)
    # ==========================================================================
    if ball_dom >= p.get('BD_MIN', 0.30):
        result['ball_dominance_tier'] = 'Very High'
    elif ball_dom >= p.get('BD_ALL_AROUND', 0.23):
        result['ball_dominance_tier'] = 'High'
    elif ball_dom >= p.get('BD_BASE', 0.17):
        result['ball_dominance_tier'] = 'Moderate'
    else:
        result['ball_dominance_tier'] = 'Low'

    if ast_per36 >= p.get('ELITE_PLAYMAKER', 6.0):
        result['playmaking_tier'] = 'Elite'
    elif ast_per36 >= p.get('PLAYMAKER', 5.0):
        result['playmaking_tier'] = 'High'
    elif ast_per36 >= p.get('CONNECTOR_AST', 3.3):
        result['playmaking_tier'] = 'Moderate'
    else:
        result['playmaking_tier'] = 'Low'

    if pts_per36 >= p.get('ELITE_SCORING', 21.0):
        result['scoring_tier'] = 'Elite'
    elif pts_per36 >= p.get('HIGH_SCORING', 18.5):
        result['scoring_tier'] = 'High'
    elif pts_per36 >= p.get('LOW_SCORING', 13.0):
        result['scoring_tier'] = 'Medium'
    else:
        result['scoring_tier'] = 'Low'

    if ts >= p.get('HIGH_EFFICIENCY', 0.60):
        result['efficiency_tier'] = 'High'
    elif ts <= p.get('LOW_EFFICIENCY', 0.55):
        result['efficiency_tier'] = 'Low'
    else:
        result['efficiency_tier'] = 'Average'

    # ==========================================================================
    # CLASSIFICATION FLAGS (percentile-based)
    # ==========================================================================
    is_ball_dominant_high = ball_dom >= p.get('BD_MIN', 0.30)
    is_ball_dominant_mid = ball_dom >= p.get('BD_ALL_AROUND', 0.23)
    is_playmaker = ast_per36 >= p.get('PLAYMAKER', 5.0)
    is_elite_playmaker = ast_per36 >= p.get('ELITE_PLAYMAKER', 6.0)
    is_high_scorer = pts_per36 >= p.get('HIGH_SCORING', 18.5)
    is_elite_scorer = pts_per36 >= p.get('ELITE_SCORING', 21.0)
    is_interior = fg2a_rate >= p.get('FG2A_INTERIOR', 0.68)
    is_perimeter = fg2a_rate <= p.get('FG2A_PERIMETER', 0.49)

    # ==========================================================================
    # CLASSIFICATION LOGIC
    # ==========================================================================

    # ---------------------------------------------------------------------------
    # 1. BALL DOMINANT CREATOR
    # High on-ball creation (>= P70) + High playmaking (>= P75)
    # Uses 60/40 volume:confidence composite to ensure only top BDCs qualify
    # ---------------------------------------------------------------------------
    moderate_scoring_gate = p.get('MODERATE_SCORING', 15.5)
    if is_ball_dominant_high and is_playmaker and pts_per36 >= moderate_scoring_gate:
        # Composite gate using linear interpolation from threshold to P95
        # Gives 0 at threshold, 1 at P95 — discriminates between borderline and elite
        bd_floor = p.get('BD_MIN', 0.37)
        bd_ceil = max(p.get('BD_P95', 0.51), bd_floor + 0.10)
        pts_floor = p.get('MODERATE_SCORING', 15.5)
        pts_ceil = max(p.get('PTS_P95', 26.0), pts_floor + 5.0)
        ast_floor = p.get('PLAYMAKER', 5.0)
        ast_ceil = max(p.get('AST_P95', 7.8), ast_floor + 1.5)

        bd_score = min(1.0, max(0.0, (ball_dom - bd_floor) / (bd_ceil - bd_floor)))
        pts_score = min(1.0, max(0.0, (pts_per36 - pts_floor) / (pts_ceil - pts_floor)))
        ast_score = min(1.0, max(0.0, (ast_per36 - ast_floor) / (ast_ceil - ast_floor)))

        volume_signal = 0.50 * bd_score + 0.50 * pts_score
        confidence_signal = ast_score
        composite = 0.60 * volume_signal + 0.40 * confidence_signal

        if composite >= 0.35:
            result['primary_archetype'] = 'Ball Dominant Creator'
            result['role_confidence'] = float(min(1.0, np.sqrt(composite / 0.6)))

            # --- BDC Subtypes (priority: Post Hub > Gravity Engine > Heliocentric) ---
            is_post_hub = (
                (post_up >= STATIC['POST_HUB_DOMINANT']) or
                (post_up >= STATIC['POST_HUB_POSS'] and post_up >= on_ball)
            )
            if is_post_hub:
                result['secondary_archetype'] = 'Post Hub'
            elif (fg3a_per36 >= p.get('HIGH_FG3A', 7.0) and
                  (ts >= p.get('HIGH_EFFICIENCY', 0.60) or fg3_pct >= p.get('HIGH_FG3_PCT', 0.37))):
                result['secondary_archetype'] = 'Gravity Engine'
            elif ball_dom >= p.get('BD_HELIOCENTRIC', 0.41):
                result['secondary_archetype'] = 'Heliocentric Guard'
            elif is_high_scorer:
                result['secondary_archetype'] = 'Primary Scorer'

            result['role_effectiveness'] = compute_role_effectiveness(row, 'Ball Dominant Creator', p)
            return result

    # ---------------------------------------------------------------------------
    # 1b. OFFENSIVE HUB (playmaker + elite scorer, moderate ball dominance)
    # Catches post-up hubs: Sabonis, Vucevic, Jokic
    # ---------------------------------------------------------------------------
    if is_elite_playmaker and is_elite_scorer:
        result['primary_archetype'] = 'Ball Dominant Creator'
        result['role_confidence'] = float(min(1.0,
            np.sqrt(ast_per36 / max(p.get('ELITE_PLAYMAKER', 6.0), 4.0)) * 0.5 +
            np.sqrt(pts_per36 / max(p.get('ELITE_SCORING', 21.0), 15.0)) * 0.5
        ))

        # Apply full subtype logic to Hub path too
        is_post_hub = (
            (post_up >= STATIC['POST_HUB_DOMINANT']) or
            (post_up >= STATIC['POST_HUB_POSS'] and post_up >= on_ball)
        )
        if is_post_hub:
            result['secondary_archetype'] = 'Post Hub'
        elif (fg3a_per36 >= p.get('HIGH_FG3A', 7.0) and
              (ts >= p.get('HIGH_EFFICIENCY', 0.60) or fg3_pct >= p.get('HIGH_FG3_PCT', 0.37))):
            result['secondary_archetype'] = 'Gravity Engine'
        else:
            result['secondary_archetype'] = 'Offensive Hub'

        result['role_effectiveness'] = compute_role_effectiveness(row, 'Ball Dominant Creator', p)
        return result

    # -------------------------------------------------------------------------
    # 1c. PLAYMAKING HUB (elite playmaker + post-up creator, any scoring level)
    # Catches Sabonis-type: high AST, post-up heavy, not traditional ball-dom
    # -------------------------------------------------------------------------
    if (is_elite_playmaker and post_up >= STATIC['POST_HUB_POSS'] and
        not is_ball_dominant_high):
        result['primary_archetype'] = 'Ball Dominant Creator'
        result['secondary_archetype'] = 'Post Hub'
        pm_cap = max(p.get('ELITE_PLAYMAKER', 6.0), 4.0)
        result['role_confidence'] = float(min(1.0,
            np.sqrt(ast_per36 / pm_cap) * 0.6 + np.sqrt(post_up / 0.15) * 0.4
        ))
        result['role_effectiveness'] = compute_role_effectiveness(row, 'Ball Dominant Creator', p)
        return result

    # ---------------------------------------------------------------------------
    # 2. ALL-AROUND SCORER (Gap A fix: playmaker + high scorer → before Ballhandler)
    # If a player is both a playmaker AND a high scorer, they are an All-Around Scorer
    # This prevents stars from falling through to Ballhandler or Rotation Piece
    # ---------------------------------------------------------------------------
    moderate_scoring_gate_aa = p.get('MODERATE_SCORING', 15.5)  # P50 gate (lowered from P60)
    if is_playmaker and is_high_scorer:
        result['primary_archetype'] = 'All-Around Scorer'
        bd_cap = max(p.get('BD_MIN', 0.30), 0.25)
        pm_cap = max(p.get('ELITE_PLAYMAKER', 6.0), 4.0)
        result['role_confidence'] = float(min(1.0,
            np.sqrt(pts_per36 / max(p.get('ELITE_SCORING', 21.0), 15.0)) * 0.5 +
            np.sqrt(ast_per36 / pm_cap) * 0.5
        ))
        if is_elite_scorer:
            result['secondary_archetype'] = 'High Volume'
        elif midrange_freq >= p.get('HIGH_MIDRANGE', 0.15):
            result['secondary_archetype'] = 'Midrange Scorer'
        result['role_effectiveness'] = compute_role_effectiveness(row, 'All-Around Scorer', p)
        return result

    # ---------------------------------------------------------------------------
    # 3. BALLHANDLER (Facilitator)
    # High playmaking but not the primary scoring option
    # ---------------------------------------------------------------------------
    if is_playmaker and not is_high_scorer:
        result['primary_archetype'] = 'Ballhandler'
        pm_cap = max(p.get('ELITE_PLAYMAKER', 6.0), 4.0)
        result['role_confidence'] = float(min(1.0, np.sqrt(ast_per36 / pm_cap)))
        result['role_effectiveness'] = compute_role_effectiveness(row, 'Ballhandler', p)
        return result

    # ---------------------------------------------------------------------------
    # 4. PRIMARY SCORERS (Ball Dominant >= P60, Not Playmaker)
    # FG2A_RATE: P70 = interior, P30 = perimeter
    # Interior split: Rim Finisher (AT_RIM high) vs Midrange Scorer (MIDRANGE high)
    # All-Around/Balanced gate lowered to P50 scoring (from P60)
    # ---------------------------------------------------------------------------
    if is_ball_dominant_mid and not is_playmaker:
        bd_cap = max(p.get('BD_MIN', 0.30), 0.25)

        if is_interior:
            result['primary_archetype'] = 'Interior Scorer'
            result['role_confidence'] = float(min(1.0,
                np.sqrt(ball_dom / bd_cap) * 0.6 + (fg2a_rate / 0.85) * 0.4
            ))
            # Subtype: Rim Finisher vs Midrange Scorer (P70) vs Midrange Lean (P50-P70)
            if at_rim_freq >= p.get('HIGH_AT_RIM', 0.30):
                result['secondary_archetype'] = 'Rim Finisher'
            elif midrange_freq >= p.get('HIGH_MIDRANGE', 0.15):
                result['secondary_archetype'] = 'Midrange Scorer'
            elif midrange_freq >= p.get('MODERATE_MIDRANGE', 0.08):
                result['secondary_archetype'] = 'Midrange Lean'
            elif is_high_scorer:
                result['secondary_archetype'] = 'High Volume'
            result['role_effectiveness'] = compute_role_effectiveness(row, 'Interior Scorer', p)
            return result

        elif is_perimeter:
            result['primary_archetype'] = 'Perimeter Scorer'
            result['role_confidence'] = float(min(1.0,
                np.sqrt(ball_dom / bd_cap) * 0.6 + ((1 - fg2a_rate) / 0.65) * 0.4
            ))
            if is_high_scorer:
                result['secondary_archetype'] = 'High Volume'
            result['role_effectiveness'] = compute_role_effectiveness(row, 'Perimeter Scorer', p)
            return result

        else:
            # Balanced Scorer: gate lowered to P50 (MODERATE_SCORING)
            balanced_scoring_gate = p.get('MODERATE_SCORING', 15.5)
            if pts_per36 >= balanced_scoring_gate:
                result['primary_archetype'] = 'All-Around Scorer'
                result['role_confidence'] = float(min(1.0,
                    np.sqrt(ball_dom / bd_cap) * 0.5 +
                    np.sqrt(pts_per36 / max(p.get('ELITE_SCORING', 21.0), 15.0)) * 0.5
                ))
                if is_high_scorer:
                    result['secondary_archetype'] = 'High Volume'
                elif midrange_freq >= p.get('HIGH_MIDRANGE', 0.15):
                    result['secondary_archetype'] = 'Midrange Scorer'
                elif midrange_freq >= p.get('MODERATE_MIDRANGE', 0.08):
                    result['secondary_archetype'] = 'Midrange Lean'
                result['role_effectiveness'] = compute_role_effectiveness(row, 'All-Around Scorer', p)
                return result
            # Falls through to lower archetypes if scoring too low

    # ---------------------------------------------------------------------------
    # 4b. INTERIOR SCORER (low ball-dominance variant)
    # Catches elite interior scorers with low creation but high interior shot
    # profile — e.g. Zion-lite seasons, early Embiid, post-heavy wings.
    # Confidence capped at 0.80 to avoid over-promoting low-creation players.
    # ---------------------------------------------------------------------------
    if (not is_playmaker and
        fg2a_rate >= p.get('FG2A_INTERIOR', 0.68) and
        pts_per36 >= p.get('MODERATE_SCORING', 15.5) and
        at_rim_plus_paint >= p.get('AT_RIM_PAINT_P60', 0.35)):
        result['primary_archetype'] = 'Interior Scorer'
        raw_conf = (
            (fg2a_rate / 0.85) * 0.4 +
            (at_rim_plus_paint / 0.50) * 0.3 +
            (pts_per36 / max(p.get('ELITE_SCORING', 21.0), 15.0)) * 0.3
        )
        result['role_confidence'] = float(min(0.80, raw_conf))  # capped
        if at_rim_freq >= p.get('HIGH_AT_RIM', 0.30):
            result['secondary_archetype'] = 'Rim Finisher'
        elif midrange_freq >= p.get('HIGH_MIDRANGE', 0.15):
            result['secondary_archetype'] = 'Midrange Scorer'
        elif midrange_freq >= p.get('MODERATE_MIDRANGE', 0.08):
            result['secondary_archetype'] = 'Midrange Lean'
        elif is_high_scorer:
            result['secondary_archetype'] = 'High Volume'
        result['role_effectiveness'] = compute_role_effectiveness(row, 'Interior Scorer', p)
        return result

    # ---------------------------------------------------------------------------
    # 5. CONNECTOR
    # Moderate playmaking (>= P50) + active player filter (touches or sec/touch)
    # Not ball dominant, not high scorer
    # ---------------------------------------------------------------------------
    connector_ast_gate = p.get('CONNECTOR_AST', 3.3)
    if (ast_per36 >= connector_ast_gate and
        not is_ball_dominant_mid and
        not is_high_scorer):
        # Activity filter: must be active, not passive
        touches_ok = touches >= p.get('TOUCHES_MEDIAN', 40.0)
        sec_touch_ok = sec_per_touch <= STATIC['CONNECTOR_SEC_PER_TOUCH_MAX']
        has_playmaking_profile = (
            sec_ast >= 0.3 or
            ast_per36 >= p.get('PLAYMAKER', 5.0) * 0.85 or
            (ast_per36 >= connector_ast_gate and (touches_ok or sec_touch_ok))
        )
        if has_playmaking_profile and (touches_ok or sec_touch_ok):
            result['primary_archetype'] = 'Connector'
            pm_cap = max(p.get('PLAYMAKER', 5.0), 3.0)
            result['role_confidence'] = float(min(1.0, np.sqrt(ast_per36 / pm_cap)))
            if row.get('REB_PER36', 0) and row.get('REB_PER36', 0) > 8:
                result['secondary_archetype'] = 'Playmaking Big'
            elif sec_ast >= 0.8:
                result['secondary_archetype'] = 'Hockey Assist Specialist'
            result['role_effectiveness'] = compute_role_effectiveness(row, 'Connector', p)
            return result

    # ---------------------------------------------------------------------------
    # 5. PnR BIG (Rolling / Popping)
    # Primary off-ball play is PnR Roll Man (higher than cuts)
    # Split by FG3A: high 3PA = popper, low = roller
    # ---------------------------------------------------------------------------
    prm_gate = p.get('HIGH_PRROLLMAN', 0.06)
    if prrollman >= prm_gate and prrollman > cut_poss and prrollman >= spotup * 0.5:
        fg3a_mid = p.get('FG3A_MEDIAN', 5.5)
        if fg3a_per36 >= fg3a_mid:
            result['primary_archetype'] = 'PnR Popping Big'
        else:
            result['primary_archetype'] = 'PnR Rolling Big'
        prm_cap = max(p.get('HIGH_PRROLLMAN', 0.06) * 2.5, 0.15)
        result['role_confidence'] = float(min(1.0, np.sqrt(prrollman / prm_cap)))
        if transition > 0.15:
            result['secondary_archetype'] = 'Transition Player'
        result['role_effectiveness'] = compute_role_effectiveness(row, result['primary_archetype'], p)
        return result

    # ---------------------------------------------------------------------------
    # 6. OFF-BALL FINISHER (cuts, transition, putbacks — not primarily roll man)
    # ---------------------------------------------------------------------------
    offball_finish_score = cut_pnrrm + putback * 0.5 + transition * 0.3
    cut_pnrrm_gate = p.get('HIGH_CUT_PNRRM', 0.18)
    if cut_pnrrm >= cut_pnrrm_gate or offball_finish_score > 0.20:
        result['primary_archetype'] = 'Off-Ball Finisher'
        ob_cap = max(cut_pnrrm_gate * 1.5, 0.25)
        result['role_confidence'] = float(min(1.0, np.sqrt(offball_finish_score / ob_cap)))
        if transition > 0.15:
            result['secondary_archetype'] = 'Transition Player'
        result['role_effectiveness'] = compute_role_effectiveness(row, 'Off-Ball Finisher', p)
        return result

    # ---------------------------------------------------------------------------
    # 7. OFF-BALL MOVEMENT SHOOTER
    # movement >= P70 AND movement/(movement+spotup) >= 0.40
    # ---------------------------------------------------------------------------
    movement_gate = p.get('HIGH_MOVEMENT', 0.10)
    total_shoot = movement + spotup
    movement_ratio = movement / total_shoot if total_shoot > 0 else 0
    if movement >= movement_gate and movement_ratio >= STATIC['MOVEMENT_RATIO_MIN']:
        result['primary_archetype'] = 'Off-Ball Movement Shooter'
        result['role_confidence'] = float(min(1.0, np.sqrt(movement / max(movement_gate * 2, 0.15))))
        result['role_effectiveness'] = compute_role_effectiveness(row, 'Off-Ball Movement Shooter', p)
        return result

    # ---------------------------------------------------------------------------
    # 8. OFF-BALL STATIONARY SHOOTER
    # ---------------------------------------------------------------------------
    spotup_gate = p.get('HIGH_SPOTUP', 0.25)
    if spotup >= spotup_gate:
        result['primary_archetype'] = 'Off-Ball Stationary Shooter'
        result['role_confidence'] = float(min(1.0, np.sqrt(spotup / max(spotup_gate * 1.4, 0.35))))
        if fg3_pct > 0.38:
            result['secondary_archetype'] = 'Elite Shooter'
        result['role_effectiveness'] = compute_role_effectiveness(row, 'Off-Ball Stationary Shooter', p)
        return result

    # ---------------------------------------------------------------------------
    # 9. BEST-FIT FALLBACK (replaces Rotation Piece catch-all)
    # Uses scoring signals to assign closest archetype instead of a generic bucket
    # ---------------------------------------------------------------------------
    reb36 = row.get('REB_PER36', 0) or 0
    stl36 = row.get('STL_PER36', 0) or 0
    blk36 = row.get('BLK_PER36', 0) or 0

    # Score each candidate archetype using frozen canonical metric vectors
    # (FALLBACK_VECTORS prevents silent regressions from feature creep)
    candidates = {}
    for arch, metrics in FALLBACK_VECTORS.items():
        score = 0.0
        total_weight = 0.0
        for row_key, pctile_key, static_denom, weight in metrics:
            val = row.get(row_key, 0) or 0
            if pd.isna(val):
                val = 0
            denom = max(p.get(pctile_key, static_denom) if pctile_key else static_denom, 0.001)
            score += weight * (val / denom)
            total_weight += weight
        if total_weight > 0:
            candidates[arch] = score / total_weight

    # Pick the best-fit candidate
    if candidates:
        best_arch = max(candidates, key=candidates.get)
        best_score = candidates[best_arch]
        result['primary_archetype'] = best_arch
        result['role_confidence'] = float(min(0.70, 0.30 + best_score * 0.20))  # capped lower since fallback
    else:
        # True last resort — use scoring/defensive signals
        if pts_per36 >= p.get('MODERATE_SCORING', 15.5):
            result['primary_archetype'] = 'All-Around Scorer'
            result['role_confidence'] = 0.40
        elif reb36 > 8 or blk36 > 1.2:
            result['primary_archetype'] = 'Off-Ball Finisher'
            result['role_confidence'] = 0.35
        else:
            result['primary_archetype'] = 'Off-Ball Stationary Shooter'
            result['role_confidence'] = 0.30

    # Assign a secondary based on defensive/utility signals
    if reb36 > 8:
        result['secondary_archetype'] = 'Rebounder'
    elif stl36 > 1.5:
        result['secondary_archetype'] = 'Perimeter Defender'
    elif blk36 > 1.2:
        result['secondary_archetype'] = 'Rim Protector'
    elif midrange_freq >= p.get('HIGH_MIDRANGE', 0.15):
        result['secondary_archetype'] = 'Midrange Scorer'
    elif midrange_freq >= p.get('MODERATE_MIDRANGE', 0.08):
        result['secondary_archetype'] = 'Midrange Lean'
    elif transition > 0.10:
        result['secondary_archetype'] = 'Transition Player'

    result['role_effectiveness'] = compute_role_effectiveness(row, result['primary_archetype'], p)
    return result


def classify_all_players(features: pd.DataFrame, pctiles: dict) -> pd.DataFrame:
    """Classify all players in the features dataframe using season percentile thresholds."""

    classifications = []
    for idx, row in features.iterrows():
        result = classify_archetype(row, pctiles)
        result['PLAYER_ID'] = row['PLAYER_ID']
        result['PLAYER_NAME'] = row['PLAYER_NAME']
        result['SEASON'] = row['SEASON']
        classifications.append(result)

    class_df = pd.DataFrame(classifications)

    output = features.merge(class_df, on=['PLAYER_ID', 'PLAYER_NAME', 'SEASON'])

    return output


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    print("=" * 70)
    print("PLAYER ARCHETYPE CLASSIFICATION (v4)")
    print("=" * 70)

    print("\n Loading data...")

    all_synergy = []
    all_tracking = []
    all_shot_zones = {}

    for season in SEASONS:
        print(f"   Loading {season}...")
        syn = load_synergy_data(season)
        trk = load_tracking_data(season)
        sz = load_shot_zone_data(season)
        if not syn.empty:
            all_synergy.append(syn)
        if not trk.empty:
            all_tracking.append(trk)
        all_shot_zones[season] = sz

    synergy_df = pd.concat(all_synergy, ignore_index=True) if all_synergy else pd.DataFrame()
    tracking_df = pd.concat(all_tracking, ignore_index=True) if all_tracking else pd.DataFrame()
    box_df = load_box_score_data()

    print(f"\n   Synergy data: {len(synergy_df)} rows")
    print(f"   Tracking data: {len(tracking_df)} rows")
    print(f"   Box score data: {len(box_df)} rows")
    for s, sz in all_shot_zones.items():
        print(f"   Shot zones {s}: {len(sz)} rows")

    all_results = []
    for season in SEASONS:
        print(f"\n Processing {season}...")
        sz = all_shot_zones.get(season, pd.DataFrame())
        features = compute_archetype_features(synergy_df, tracking_df, box_df, season, shot_zones=sz)

        if features.empty:
            print(f"   No data for {season}")
            continue

        # Compute per-season percentile thresholds
        pctiles = compute_season_percentiles(features)
        print(f"   Percentile thresholds computed ({len(pctiles)} metrics)")
        for key in ['BD_MIN', 'PLAYMAKER', 'HIGH_SCORING', 'FG2A_INTERIOR', 'FG2A_PERIMETER',
                     'HIGH_CUT_PNRRM', 'HIGH_SPOTUP', 'HIGH_MOVEMENT', 'CONNECTOR_AST',
                     'ALL_AROUND_SCORING', 'HIGH_PRROLLMAN', 'FG3A_MEDIAN', 'TOUCHES_MEDIAN',
                     'HIGH_AT_RIM', 'HIGH_MIDRANGE', 'MODERATE_MIDRANGE',
                     'AT_RIM_PAINT_P60']:
            print(f"      {key}: {pctiles.get(key, '?'):.4f}")

        classified = classify_all_players(features, pctiles)
        all_results.append(classified)

        # Print summary
        qual = classified[
            (classified['MIN'] >= STATIC['MIN_MINUTES']) &
            (classified['GP'] >= STATIC['MIN_GP']) &
            (classified['MPG'] >= STATIC['MIN_MPG'])
        ]
        archetype_counts = qual['primary_archetype'].value_counts()
        print(f"   Classified {len(qual)} qualified players:")
        for arch, count in archetype_counts.items():
            pct = count / len(qual) * 100
            print(f"      {arch}: {count} ({pct:.1f}%)")

        # Subtypes
        sub_counts = qual[qual['secondary_archetype'].notna()]['secondary_archetype'].value_counts()
        if len(sub_counts) > 0:
            print(f"   Subtypes:")
            for sub, count in sub_counts.items():
                print(f"      {sub}: {count}")

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)

        final_df.to_parquet(OUTPUT_DIR / "player_archetypes.parquet", index=False)
        final_df.to_csv(OUTPUT_DIR / "player_archetypes.csv", index=False)

        print(f"\n Saved {len(final_df)} player-seasons to data/processed/player_archetypes.parquet")

        # Validation
        print("\n" + "=" * 70)
        print("VALIDATION: KEY PLAYER CLASSIFICATIONS")
        print("=" * 70)

        s25 = final_df[final_df['SEASON'] == '2024-25'].copy()
        stars = [
            'LeBron James', 'Stephen Curry', 'Kevin Durant', 'Giannis Antetokounmpo',
            'Luka Dončić', 'Nikola Jokic', 'Jayson Tatum', 'Jonathan Kuminga',
            'Shai Gilgeous-Alexander', 'Anthony Edwards', 'Draymond Green',
            'Jalen Brunson', 'Tyrese Haliburton', 'Derrick White',
            'Joel Embiid', 'Clint Capela', 'Rudy Gobert', 'Nicolas Batum',
            'Domantas Sabonis', 'Trae Young', 'Devin Booker',
            'Klay Thompson', 'Buddy Hield', 'Brook Lopez',
        ]

        for name in stars:
            player = s25[s25['PLAYER_NAME'].str.contains(name, case=False, na=False)]
            if len(player) > 0:
                pp = player.iloc[0]
                sec = f" / {pp['secondary_archetype']}" if pp.get('secondary_archetype') else ""
                eff_tier = pp.get('efficiency_tier', '?')
                rc = pp.get('role_confidence', 0)
                re = pp.get('role_effectiveness', 0)
                fg2 = pp.get('FG2A_RATE', 0)
                bd = pp.get('BALL_DOMINANT_PCT', 0)
                ast = pp.get('AST_PER36', 0)
                print(f"  {pp['PLAYER_NAME']:<25} {pp['primary_archetype']}{sec}  conf={rc:.0%} eff={re:.0%}  [{eff_tier}] BD={bd:.2f} AST={ast:.1f} FG2A={fg2:.0%}")
            else:
                s24 = final_df[final_df['SEASON'] == '2023-24']
                player = s24[s24['PLAYER_NAME'].str.contains(name, case=False, na=False)]
                if len(player) > 0:
                    pp = player.iloc[0]
                    sec = f" / {pp['secondary_archetype']}" if pp.get('secondary_archetype') else ""
                    rc = pp.get('role_confidence', 0)
                    re = pp.get('role_effectiveness', 0)
                    print(f"  {pp['PLAYER_NAME']:<25} {pp['primary_archetype']}{sec}  conf={rc:.0%} eff={re:.0%} [{pp['SEASON']}]")

    return final_df


if __name__ == "__main__":
    result = main()
