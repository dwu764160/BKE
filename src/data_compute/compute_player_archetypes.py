"""
src/data_compute/compute_player_archetypes.py
=============================================================================
Classifies NBA players into offensive archetypes based on tracking, playtype, and shot zone data.

v4.3 — PnR Big Absorption + Harsher Interior Scorer + Archetype Embeddings:
  - PnR Big "prominent" path: interior bigs with >= P50 PnR Roll Man → PnR Big
    (no longer requires PnR to be the MOST common play)
  - Off-Ball Finisher redirect: bigs with prominent PnR activity → PnR Big
  - Harsher Interior Scorer skill check: SKILL_FINISHING_MIN raised to 0.40;
    rim-runners with prominent PnR + low midrange always fall through
  - "Midrange Lean" subtype renamed to "Inside-the-Arc"
  - Archetype Embedding: soft role model producing confidence-weighted
    continuous embedding for each player (11-dimensional)

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

Archetype Definitions (v4.3):
=============================
 1. Ball Dominant Creator  — High on-ball creation + high playmaking
      Subtypes: Heliocentric Guard | Post Hub | Gravity Engine
 3. All-Around Scorer      — Playmaker + high scorer (placed before Ballhandler)
      Subtypes: High Volume | Midrange Scorer
 4. Ballhandler/Facilitator — High playmaking, lower scoring
 5. Interior Scorer        — Ball dominant, non-playmaker, interior-heavy (skill finishing required)
      Subtypes: Rim Finisher | Midrange Scorer | Inside-the-Arc | High Volume
 6. Perimeter Scorer       — Ball dominant, non-playmaker, perimeter FG2A <= P30
 8. Connector              — Moderate playmaking, active filter
 9. PnR Rolling Big        — Off-ball, roll man (strict P70 OR prominent P50 + interior profile)
10. PnR Popping Big        — Off-ball, roll man + high 3PA
11. Off-Ball Finisher      — High cut/PnR roll/transition (bigs with PnR redirected to PnR Big)
12. Off-Ball Movement Shooter — movement/(movement+spotup) >= 0.30
13. Off-Ball Stationary Shooter — High spot-up
14. Best-fit fallback      — Closest archetype based on scoring signals (no catch-all)
=============================================================================
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.modeling.model_config import SEASONS
from src.data.schema_contract import load_standardized, save_standardized

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = Path("data")
TRACKING_DIR = DATA_DIR / "tracking"
HISTORICAL_DIR = DATA_DIR / "historical"
OFFICIAL_DIR = DATA_DIR / "official_stats"
OUTPUT_DIR = DATA_DIR / "processed"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------- Static thresholds (only where percentiles don't apply) ----------
STATIC = {
    'MIN_MINUTES': 200,
    'MIN_GP': 10,
    'MIN_MPG': 8.0,
    # BDC subtypes (structural, not volume-based)
    'HELIOCENTRIC_ON_BALL': 0.30,   # ON_BALL_CREATION >= this → Heliocentric Guard
    'POST_HUB_POSS': 0.10,          # POSTUP_POSS_PCT >= this AND must be dominant mode
    'POST_HUB_DOMINANT': 0.15,       # POST >= this always qualifies as Post Hub
    # Secondary tag gates
    'LOB_THREAT_AT_RIM': 0.65,      # AT_RIM_FREQ >= 65% → Lob Threat (PnR Rolling Big)
    'PLAYMAKING_BIG_AST': 2.0,      # AST_PER36 >= 2.0 → Playmaking Big (PnR bigs)
    'ELITE_SHOOTER_FG3_PCT': 0.38,  # FG3_PCT >= 0.38 for Elite Shooter
    'ELITE_SHOOTER_FG3A': 4.0,      # FG3A_PER36 >= 4.0 volume floor for Elite Shooter
    # (original POST_HUB_POSS line replaced)
    # Movement Shooter ratio (loosened from 0.40 in v4.2)
    'MOVEMENT_RATIO_MIN': 0.30,     # movement / (movement + spotup) >= 0.30
    # Connector activity filter
    'CONNECTOR_SEC_PER_TOUCH_MAX': 2.5,  # AVG_SEC_PER_TOUCH <= this (active, not holding ball long)
    # Post-up weighting in ball dominance
    'POST_WEIGHT': 0.65,
    # Skill finishing floor: paint_freq + midrange_freq minimum for non-star Interior Scorer
    'SKILL_FINISHING_MIN': 0.40,
}

# ---------- Percentile-based thresholds (computed per-season) ----------
# Format: (column_name, percentile)
# These are resolved to actual values at runtime from per-season distributions
PCTILE_THRESHOLDS = {
    # Ball Dominant Creator gates
    'BD_MIN':               ('ball_dominant_pct', 80),    # >= P80 for BDC path (tighter v3)
    'BD_ALL_AROUND':        ('ball_dominant_pct', 60),    # >= P60 for All-Around
    'BD_BASE':              ('ball_dominant_pct', 50),    # >= P50 for any ball-dom consideration
    # Playmaking gates
    'PLAYMAKER':            ('ast_per36', 75),            # >= P75 for playmaker
    'ELITE_PLAYMAKER':      ('ast_per36', 85),            # >= P85 elite playmaker
    'CONNECTOR_AST':        ('ast_per36', 50),            # >= P50 for connector
    # Scoring volume
    'HIGH_SCORING':         ('pts_per36', 70),            # >= P70 high volume
    'ELITE_SCORING':        ('pts_per36', 80),            # >= P80 elite volume
    'ALL_AROUND_SCORING':   ('pts_per36', 60),            # >= P60 for all-around filer
    'LOW_SCORING':          ('pts_per36', 25),            # <= P25 low scorer
    # Shot profile
    'FG2A_INTERIOR':        ('fg2a_rate', 70),            # >= P70 interior heavy
    'FG2A_PERIMETER':       ('fg2a_rate', 30),            # <= P30 perimeter heavy
    # Off-ball
    'HIGH_CUT_PNRRM':      ('cut_pnrrm_pct', 75),       # >= P75 cut + roll man
    'HIGH_SPOTUP':          ('spotup_pct', 50),           # >= P50 spot-up
    'HIGH_MOVEMENT':        ('movement_shooter_pct', 60), # >= P60 movement shooter (loosened from P70 in v4.2)
    'HIGH_PRROLLMAN':       ('prrollman_poss_pct', 70),   # >= P70 roll man (for PnR Big)
    'PROMINENT_PRROLLMAN':  ('prrollman_poss_pct', 50),   # >= P50 prominent roll man (looser PnR Big gate for bigs)
    # Touches / activity (for Connector)
    'TOUCHES_MEDIAN':       ('touches', 50),              # >= P50 touches
    # Efficiency
    'HIGH_EFFICIENCY':      ('ts_pct', 70),               # >= P70 efficient
    'LOW_EFFICIENCY':       ('ts_pct', 30),               # <= P30 inefficient
    # Gravity Engine
    'HIGH_FG3A':            ('fg3a_per36', 80),           # >= P80 3PA volume
    'HIGH_FG3_PCT':         ('fg3_pct', 70),              # >= P70 3P%
    # BDC Heliocentric subtype
    'BD_HELIOCENTRIC':      ('ball_dominant_pct', 85),    # >= P85 for Heliocentric subtype
    # P95 caps for composite normalization
    'BD_P95':               ('ball_dominant_pct', 95),    # P95 for composite norm
    'PTS_P95':              ('pts_per36', 95),            # P95 for composite norm
    'AST_P95':              ('ast_per36', 95),            # P95 for composite norm
    # Scoring gate for BDC main path
    'MODERATE_SCORING':     ('pts_per36', 50),            # >= P50 BDC must be above-avg scorer
    # Playmaking score
    'HIGH_PLAYMAKING_SCORE':('playmaking_score', 75),     # >= P75
    # FG3A for PnR pop vs roll
    'FG3A_MEDIAN':          ('fg3a_per36', 50),           # >= P50 → popping, < P50 → rolling
    # Shot zone features (from LeagueDashPlayerShotLocations)
    'HIGH_AT_RIM':          ('at_rim_freq', 70),          # >= P70 rim finisher
    'HIGH_MIDRANGE':        ('midrange_freq', 70),        # >= P70 midrange scorer
    'LOW_MIDRANGE':         ('midrange_freq', 30),        # <= P30 non-midrange
    'MODERATE_MIDRANGE':    ('midrange_freq', 50),        # >= P50 midrange leaning
    'AT_RIM_PAINT_P60':     ('at_rim_plus_paint_freq', 60), # >= P60 interior shot location
}

# ---------- Frozen canonical metric vectors for best-fit fallback ----------
# Prevents silent regressions from feature creep.  Each entry:
#   (row_column, pctile_key_for_denominator_or_None, static_fallback_denom, weight)
FALLBACK_VECTORS = {
    'Off-Ball Stationary Shooter': [
        ('spotup_pct',           'HIGH_SPOTUP',     0.10, 1.0),
    ],
    'Off-Ball Movement Shooter': [
        ('movement_shooter_pct', 'HIGH_MOVEMENT',   0.05, 1.0),
    ],
    'Off-Ball Finisher': [
        ('cut_pnrrm_pct',        None,             0.15, 1.0),
        ('putback_pct',           None,             0.15, 0.5),
        ('transition_pct',        None,             0.15, 0.3),
    ],
    'Connector': [
        ('ast_per36',            'CONNECTOR_AST',   2.0,  1.0),
    ],
    'Interior Scorer': [
        ('fg2a_rate',             None,             0.85, 0.5),
        ('pts_per36',            'HIGH_SCORING',    14.0, 0.5),
    ],
    'PnR Rolling Big': [
        ('prrollman_poss_pct',   'HIGH_PRROLLMAN',  0.04, 1.0),
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
            df = load_standardized(path)
            agg_df = df.groupby('player_id').agg({
                'player_name': 'first',
                'poss_pct': 'mean',
                'ppp': 'mean',
                'poss': 'sum'
            }).reset_index()

            agg_df = agg_df.rename(columns={
                'poss_pct': f'{playtype.lower()}_poss_pct',
                'ppp': f'{playtype.lower()}_ppp',
                'poss': f'{playtype.lower()}_poss'
            })
            all_data.append(agg_df)

    if not all_data:
        return pd.DataFrame()

    player_data = all_data[0]
    for df in all_data[1:]:
        player_data = player_data.merge(
            df.drop(columns=['player_name'], errors='ignore'),
            on='player_id', how='outer'
        )

    player_data['season'] = season
    return player_data


def load_tracking_data(season: str) -> pd.DataFrame:
    """Load and merge tracking data (drives, passing, possessions, catch-shoot)."""
    season_dir = TRACKING_DIR / season

    tracking_files = {
        'Drives': ['player_id', 'drives', 'drive_pts', 'drive_fg_pct', 'drive_ast', 'drive_tov'],
        'Passing': ['player_id', 'passes_made', 'secondary_ast', 'potential_ast', 'AST_POINTS_CREATED'],
        'Possessions': ['player_id', 'touches', 'time_of_poss', 'avg_sec_per_touch', 'avg_drib_per_touch', 'front_ct_touches'],
        'CatchShoot': ['player_id', 'catch_shoot_fgm', 'catch_shoot_fga', 'catch_shoot_fg_pct',
                       'catch_shoot_pts', 'catch_shoot_fg3m', 'catch_shoot_fg3a', 'catch_shoot_fg3_pct'],
        'Rebounding': ['player_id', 'oreb_contest', 'dreb_contest', 'reb_contest'],
        'SpeedDistance': ['player_id', 'dist_miles', 'avg_speed'],
    }

    merged = None
    for track_type, keep_cols in tracking_files.items():
        path = season_dir / f"tracking_{track_type}.parquet"
        if path.exists():
            df = load_standardized(path)
            available_cols = [c for c in keep_cols if c in df.columns]
            if available_cols:
                df = df[available_cols]
                numeric_cols = [c for c in df.columns if c != 'player_id']
                df = df.groupby('player_id')[numeric_cols].sum().reset_index()

                if merged is None:
                    merged = df
                else:
                    merged = merged.merge(df, on='player_id', how='outer')

    if merged is not None:
        merged['season'] = season
    return merged if merged is not None else pd.DataFrame()


def load_box_score_data() -> pd.DataFrame:
    """Load complete box score stats."""
    path = HISTORICAL_DIR / "complete_player_season_stats.parquet"
    if not path.exists():
        return pd.DataFrame()

    df = load_standardized(path)
    cols = ['player_id', 'player_name', 'team_abbreviation', 'season', 'gp', 'min', 'pts', 'ast', 'reb',
            'oreb', 'dreb', 'stl', 'blk', 'tov', 'fga', 'fgm', 'fg3a', 'fg3m',
            'fta', 'ftm', 'fg_pct', 'fg3_pct', 'ft_pct', 'usg_pct', 'ts_pct']
    available_cols = [c for c in cols if c in df.columns]
    return df[available_cols]


def load_shot_zone_data(season: str) -> pd.DataFrame:
    """Load shot zone data (from LeagueDashPlayerShotLocations).
    Returns DataFrame with AT_RIM_FREQ, MIDRANGE_FREQ, PAINT_FREQ, etc."""
    path = TRACKING_DIR / season / "shot_zones.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = load_standardized(path)
    # Keep only the columns we need for archetype classification
    keep_cols = ['player_id', 'at_rim_freq', 'paint_freq', 'midrange_freq',
                 'corner3_freq', 'ab3_freq', 'at_rim_plus_paint_freq',
                 'at_rim_fg_pct', 'midrange_fg_pct',
                 'ra_fga', 'mr_fga', 'paint_fga', 'total_fga']
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
        (features['mpg'] >= STATIC['MIN_MPG']) &
        (features['gp'] >= STATIC['MIN_GP']) &
        (features['min'] >= STATIC['MIN_MINUTES'])
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

    syn = synergy[synergy['season'] == season].copy() if not synergy.empty else pd.DataFrame()
    trk = tracking[tracking['season'] == season].copy() if not tracking.empty else pd.DataFrame()
    bx = box[box['season'] == season].copy() if not box.empty else pd.DataFrame()

    if bx.empty:
        return pd.DataFrame()

    features = bx.copy()
    # Normalize PLAYER_ID to string in all frames before merging to avoid int/str type conflicts
    features['player_id'] = features['player_id'].astype(str)

    # Pre-2022 complete_player_season_stats stores per-game values (MIN=32.9 MPG, AST=2.3 APG).
    # Post-2022 stores season totals (MIN=2055, AST=203). Detect by median MIN < 50 → per-game.
    _per_game_cols = ['min', 'pts', 'ast', 'reb', 'oreb', 'dreb', 'stl', 'blk', 'tov',
                      'fga', 'fgm', 'fg3a', 'fg3m', 'fta', 'ftm']
    if features['min'].median() < 50 and 'gp' in features.columns:
        for _col in _per_game_cols:
            if _col in features.columns:
                features[_col] = features[_col] * features['gp']

    # Merge synergy data
    if not syn.empty:
        syn['player_id'] = syn['player_id'].astype(str)
        syn_cols = [c for c in syn.columns if c not in ['player_name', 'season'] or c == 'player_id']
        features = features.merge(syn[syn_cols], on='player_id', how='left')

    # Merge tracking data
    if not trk.empty:
        trk['player_id'] = trk['player_id'].astype(str)
        trk_cols = [c for c in trk.columns if c not in ['player_name', 'season', 'gp', 'min'] or c == 'player_id']
        features = features.merge(trk[trk_cols], on='player_id', how='left')

    # Merge shot zone data
    if shot_zones is not None and not shot_zones.empty:
        sz = shot_zones.copy()
        sz['player_id'] = sz['player_id'].astype(str)
        sz_cols = [c for c in sz.columns if c == 'player_id' or c not in features.columns]
        features = features.merge(sz[sz_cols], on='player_id', how='left')

    # Fill NaN playtype data with 0
    playtype_cols = [c for c in features.columns if '_POSS_PCT' in c or '_PPP' in c]
    features[playtype_cols] = features[playtype_cols].fillna(0)

    # Fill shot zone NaN with 0
    zone_cols = [c for c in features.columns if c in
                 ('at_rim_freq', 'paint_freq', 'midrange_freq', 'corner3_freq',
                  'ab3_freq', 'at_rim_plus_paint_freq', 'at_rim_fg_pct',
                  'midrange_fg_pct', 'ra_fga', 'mr_fga', 'paint_fga', 'total_fga')]
    for col in zone_cols:
        features[col] = features[col].fillna(0)

    # Ensure optional synergy/tracking columns exist with 0 defaults.
    # Pre-2022 seasons have no tracking/synergy data; these cols won't be
    # merged in above, so features.get(col, 0) would return int 0 — a Series
    # operation on it then fails. Explicit pre-fill avoids scattered checks.
    _optional_zero_cols = [
        'isolation_poss_pct', 'prballhandler_poss_pct', 'postup_poss_pct',
        'transition_poss_pct', 'spotup_poss_pct', 'offscreen_poss_pct',
        'handoff_poss_pct', 'cut_poss_pct', 'prrollman_poss_pct', 'offrebound_poss_pct',
        'isolation_ppp', 'prballhandler_ppp', 'postup_ppp', 'prrollman_ppp',
        'spotup_ppp', 'transition_ppp', 'cut_ppp',
        'secondary_ast', 'potential_ast', 'touches', 'avg_sec_per_touch',
        'avg_drib_per_touch', 'front_ct_touches', 'drives', 'drive_pts',
        'drive_fg_pct', 'drive_ast', 'drive_tov', 'passes_made',
        'AST_POINTS_CREATED', 'dist_miles', 'avg_speed',
        'at_rim_freq', 'paint_freq', 'midrange_freq', 'corner3_freq',
        'ab3_freq', 'at_rim_plus_paint_freq', 'at_rim_fg_pct', 'midrange_fg_pct',
        'ra_fga', 'mr_fga', 'paint_fga', 'total_fga',
        'catch_shoot_fgm', 'catch_shoot_fga', 'catch_shoot_fg_pct',
        'catch_shoot_pts', 'catch_shoot_fg3m', 'catch_shoot_fg3a', 'catch_shoot_fg3_pct',
        'oreb_contest', 'dreb_contest', 'reb_contest',
    ]
    for _col in _optional_zero_cols:
        if _col not in features.columns:
            features[_col] = 0.0

    # ==========================================================================
    # COMPUTE DERIVED FEATURES
    # ==========================================================================

    features['mpg'] = features['min'] / features['gp']

    # Per-36 minute stats
    minutes_factor = 36 / (features['min'] / features['gp']).replace(0, np.nan)
    features['pts_per36'] = (features['pts'] / features['gp']) * minutes_factor
    features['ast_per36'] = (features['ast'] / features['gp']) * minutes_factor
    features['reb_per36'] = (features['reb'] / features['gp']) * minutes_factor
    features['tov_per36'] = (features['tov'] / features['gp']) * minutes_factor
    features['stl_per36'] = (features['stl'] / features['gp']) * minutes_factor
    features['blk_per36'] = (features['blk'] / features['gp']) * minutes_factor

    # ---- Ball Dominance (split on-ball from post-up) ----
    features['on_ball_creation'] = (
        features.get('isolation_poss_pct', 0) +
        features.get('prballhandler_poss_pct', 0)
    ).fillna(0)

    features['post_creation'] = features.get('postup_poss_pct', 0).fillna(0)

    features['ball_dominant_pct'] = (
        features['on_ball_creation'] + features['post_creation'] * STATIC['POST_WEIGHT']
    ).fillna(0)

    # ---- Playmaking composite (with turnover penalty) ----
    if 'secondary_ast' in features.columns:
        features['secondary_ast_per36'] = (features['secondary_ast'] / features['gp']) * minutes_factor
    else:
        features['secondary_ast_per36'] = 0

    if 'potential_ast' in features.columns:
        features['potential_ast_per36'] = (features['potential_ast'] / features['gp']) * minutes_factor
    else:
        features['potential_ast_per36'] = 0

    features['playmaking_score'] = (
        features['ast_per36'] * 1.0 +
        features['secondary_ast_per36'] * 0.5 +
        features['potential_ast_per36'] * 0.3 -
        features['tov_per36'] * 0.5
    ).fillna(0)

    # ---- Shot Profile: FG2A_RATE ----
    features['fg2a_rate'] = np.where(
        features['fga'] > 0,
        (features['fga'] - features['fg3a']) / features['fga'],
        0.5
    )

    features['fg3a_per36'] = (features['fg3a'] / features['gp']) * minutes_factor

    # Drives
    if 'drives' in features.columns:
        features['drives_per36'] = (features['drives'] / features['gp']) * minutes_factor
    else:
        features['drives_per36'] = 0

    features['interior_ratio'] = features['fg2a_rate']  # Legacy alias

    # ---- Off-ball frequencies ----
    features['cut_pnrrm_pct'] = (
        features.get('cut_poss_pct', 0) +
        features.get('prrollman_poss_pct', 0)
    ).fillna(0)

    features['movement_shooter_pct'] = (
        features.get('handoff_poss_pct', 0) +
        features.get('offscreen_poss_pct', 0)
    ).fillna(0)

    features['spotup_pct'] = features.get('spotup_poss_pct', 0).fillna(0)
    features['transition_pct'] = features.get('transition_poss_pct', 0).fillna(0)
    features['putback_pct'] = features.get('offrebound_poss_pct', 0).fillna(0)

    # ---- Efficiency: compute TS% and USG% when missing ----
    tsa = 2 * (features['fga'] + 0.44 * features['fta'])
    computed_ts = np.where(tsa > 0, features['pts'] / tsa, np.nan)

    if 'ts_pct' in features.columns:
        features['ts_pct'] = features['ts_pct'].fillna(pd.Series(computed_ts, index=features.index))
    else:
        features['ts_pct'] = computed_ts

    features['efg_pct'] = np.where(
        features['fga'] > 0,
        (features['fgm'] + 0.5 * features['fg3m']) / features['fga'],
        0
    )

    # ---- USG%: prefer official NBA.com value, fall back to proxy ----
    # First, try to fill USG_PCT from official advanced stats (covers seasons
    # where the box-score parquet lacks the advanced column).
    official_adv_path = OFFICIAL_DIR / f"official_advanced_{season}.parquet"
    if official_adv_path.exists():
        try:
            off_adv = load_standardized(official_adv_path)
            if 'usg_pct' in off_adv.columns and 'player_id' in off_adv.columns:
                off_usg = off_adv[['player_id', 'usg_pct']].rename(
                    columns={'usg_pct': '_OFFICIAL_USG_PCT'})
                features = features.merge(off_usg, on='player_id', how='left')
                if 'usg_pct' not in features.columns or features['usg_pct'].isna().all():
                    features['usg_pct'] = features['_OFFICIAL_USG_PCT']
                else:
                    features['usg_pct'] = features['usg_pct'].fillna(features['_OFFICIAL_USG_PCT'])
                features.drop(columns=['_OFFICIAL_USG_PCT'], inplace=True, errors='ignore')
        except Exception:
            pass  # silently fall through to proxy

    # Proxy for any remaining NaN: approximate standard USG% formula.
    # Constant 2.0 ≈ league-avg team possessions per minute (100 poss / 48 min ≈ 2.08).
    poss_used = features['fga'] + 0.44 * features['fta'] + features['tov']
    computed_usg = np.where(
        features['min'] > 0,
        poss_used * 2.0 / (features['min'] * 5),
        np.nan
    )

    if 'usg_pct' in features.columns:
        features['usg_pct'] = features['usg_pct'].fillna(pd.Series(computed_usg, index=features.index))
    else:
        features['usg_pct'] = computed_usg

    # Season-level league average TS for z-score
    league_avg_ts = features.loc[
        (features['min'] >= 500) & (features['gp'] >= 20) & (features['mpg'] >= 15),
        'ts_pct'
    ].mean()
    league_std_ts = features.loc[
        (features['min'] >= 500) & (features['gp'] >= 20) & (features['mpg'] >= 15),
        'ts_pct'
    ].std()

    if pd.isna(league_avg_ts):
        league_avg_ts = 0.565
    if pd.isna(league_std_ts) or league_std_ts == 0:
        league_std_ts = 0.04

    features['ts_zscore'] = (features['ts_pct'] - league_avg_ts) / league_std_ts
    features['league_avg_ts'] = league_avg_ts

    # Four Factors (informational)
    features['tov_pct'] = np.where(
        (features['fga'] + 0.44 * features['fta'] + features['tov']) > 0,
        features['tov'] / (features['fga'] + 0.44 * features['fta'] + features['tov']),
        0
    )
    features['ft_rate'] = np.where(features['fga'] > 0, features['fta'] / features['fga'], 0)

    # Time of possession
    if 'time_of_poss' in features.columns:
        features['time_of_poss_per36'] = (features['time_of_poss'] / features['gp']) * minutes_factor
    else:
        features['time_of_poss_per36'] = 0

    if 'avg_drib_per_touch' in features.columns:
        features['dribbles_per_touch'] = features['avg_drib_per_touch']
    else:
        features['dribbles_per_touch'] = 0

    features['ppg'] = features['pts'] / features['gp']

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
    ts_z = row.get('ts_zscore', 0)
    if pd.isna(ts_z):
        ts_z = 0.0
    # Base efficiency score from TS z-score (normalized: -2sigma=0, +2sigma=1)
    eff_score = float(np.clip((ts_z + 2) / 4, 0.05, 1.0))

    pts36 = row.get('pts_per36', 0)
    if pd.isna(pts36):
        pts36 = 0.0
    pts36_cap = max(pctiles.get('ELITE_SCORING', 21.0), 15.0)
    volume_score = float(np.clip(pts36 / pts36_cap, 0, 1.0))

    if archetype == 'Ball Dominant Creator':
        iso_ppp = row.get('isolation_ppp', 0) or 0
        prbh_ppp = row.get('prballhandler_ppp', 0) or 0
        primary_ppp = max(iso_ppp, prbh_ppp)
        ppp_score = float(np.clip(primary_ppp / 1.0, 0, 1.0))
        return 0.35 * eff_score + 0.35 * volume_score + 0.30 * ppp_score

    elif archetype in ('Interior Scorer', 'Perimeter Scorer', 'All-Around Scorer'):
        return 0.50 * eff_score + 0.50 * volume_score

    elif archetype == 'Ballhandler':
        ast36 = row.get('ast_per36', 0) or 0
        ast_cap = max(pctiles.get('ELITE_PLAYMAKER', 6.0), 4.0)
        ast_score = float(np.clip(ast36 / ast_cap, 0, 1.0))
        return 0.35 * eff_score + 0.25 * volume_score + 0.40 * ast_score

    elif archetype == 'Connector':
        ast36 = row.get('ast_per36', 0) or 0
        sec_ast = row.get('secondary_ast_per36', 0) or 0
        connect_score = float(np.clip((ast36 + sec_ast * 5) / 8.0, 0, 1.0))
        return 0.30 * eff_score + 0.20 * volume_score + 0.50 * connect_score

    elif archetype in ('PnR Rolling Big', 'PnR Popping Big'):
        prm_ppp = row.get('prrollman_ppp', 0) or 0
        ppp_score = float(np.clip(prm_ppp / 1.2, 0, 1.0))
        return 0.35 * eff_score + 0.30 * volume_score + 0.35 * ppp_score

    elif archetype == 'Off-Ball Finisher':
        cut_ppp = row.get('cut_ppp', 0) or 0
        ppp_score = float(np.clip(cut_ppp / 1.3, 0, 1.0))
        return 0.35 * eff_score + 0.30 * volume_score + 0.35 * ppp_score

    elif archetype == 'Off-Ball Movement Shooter':
        fg3 = row.get('fg3_pct', 0) or 0
        offscr_ppp = row.get('offscreen_ppp', 0) or 0
        shoot_score = float(np.clip((fg3 - 0.30) / 0.12, 0, 1.0))
        ppp_score = float(np.clip(offscr_ppp / 1.1, 0, 1.0))
        return 0.25 * eff_score + 0.15 * volume_score + 0.35 * shoot_score + 0.25 * ppp_score

    elif archetype == 'Off-Ball Stationary Shooter':
        fg3 = row.get('fg3_pct', 0) or 0
        spotup_ppp = row.get('spotup_ppp', 0) or 0
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
    mpg = row.get('mpg', 0)
    if (row.get('min', 0) < STATIC['MIN_MINUTES'] or
        row.get('gp', 0) < STATIC['MIN_GP'] or
        mpg < STATIC['MIN_MPG']):
        result['primary_archetype'] = 'Insufficient Minutes'
        return result

    # ==========================================================================
    # EXTRACT KEY METRICS
    # ==========================================================================
    ball_dom = row.get('ball_dominant_pct', 0) or 0
    on_ball = row.get('on_ball_creation', 0) or 0
    post_up = row.get('postup_poss_pct', 0) or 0
    ast_per36 = row.get('ast_per36', 0) or 0
    pts_per36 = row.get('pts_per36', 0) or 0
    playmaking = row.get('playmaking_score', 0) or 0
    fg2a_rate = row.get('fg2a_rate', 0.5)
    if pd.isna(fg2a_rate):
        fg2a_rate = 0.5
    fg3a_per36 = row.get('fg3a_per36', 0) or 0
    fg3_pct = row.get('fg3_pct', 0) or 0
    ts = row.get('ts_pct', 0)
    ts_z = row.get('ts_zscore', 0)
    if pd.isna(ts) or ts == 0:
        ts = 0.55
    if pd.isna(ts_z):
        ts_z = 0.0
    usg = row.get('usg_pct', 0.18)
    if pd.isna(usg):
        usg = 0.18

    # Off-ball
    cut_pnrrm = row.get('cut_pnrrm_pct', 0) or 0
    spotup = row.get('spotup_pct', 0) or 0
    movement = row.get('movement_shooter_pct', 0) or 0
    transition = row.get('transition_pct', 0) or 0
    putback = row.get('putback_pct', 0) or 0
    prrollman = row.get('prrollman_poss_pct', 0) or 0
    cut_poss = row.get('cut_poss_pct', 0) or 0

    # Tracking
    touches = row.get('touches', 0) or 0
    sec_per_touch = row.get('avg_sec_per_touch', 3.0)
    if pd.isna(sec_per_touch):
        sec_per_touch = 3.0
    sec_ast = row.get('secondary_ast_per36', 0) or 0

    # Shot zones (from LeagueDashPlayerShotLocations)
    at_rim_freq = row.get('at_rim_freq', 0) or 0
    if pd.isna(at_rim_freq):
        at_rim_freq = 0
    midrange_freq = row.get('midrange_freq', 0) or 0
    if pd.isna(midrange_freq):
        midrange_freq = 0
    paint_freq = row.get('paint_freq', 0) or 0
    if pd.isna(paint_freq):
        paint_freq = 0
    at_rim_plus_paint = row.get('at_rim_plus_paint_freq', 0) or 0
    if pd.isna(at_rim_plus_paint):
        at_rim_plus_paint = 0
    midrange_fg_pct = row.get('midrange_fg_pct', 0) or 0
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

            # --- BDC Subtypes (priority: Post Creator > Gravity Engine > Heliocentric > Downhill Driver) ---
            is_post_hub = (
                (post_up >= STATIC['POST_HUB_DOMINANT']) or
                (post_up >= STATIC['POST_HUB_POSS'] and post_up >= on_ball)
            )
            if is_post_hub:
                result['secondary_archetype'] = 'Post Creator'
            elif (fg3a_per36 >= p.get('HIGH_FG3A', 7.0) and
                  (ts >= p.get('HIGH_EFFICIENCY', 0.60) or fg3_pct >= p.get('HIGH_FG3_PCT', 0.37))):
                result['secondary_archetype'] = 'Gravity Engine'
            elif ball_dom >= p.get('BD_HELIOCENTRIC', 0.41):
                result['secondary_archetype'] = 'Heliocentric'
            elif at_rim_freq >= p.get('HIGH_AT_RIM', 0.30):
                result['secondary_archetype'] = 'Downhill Driver'

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
            result['secondary_archetype'] = 'Post Creator'
        elif (fg3a_per36 >= p.get('HIGH_FG3A', 7.0) and
              (ts >= p.get('HIGH_EFFICIENCY', 0.60) or fg3_pct >= p.get('HIGH_FG3_PCT', 0.37))):
            result['secondary_archetype'] = 'Gravity Engine'
        elif at_rim_freq >= p.get('HIGH_AT_RIM', 0.30):
            result['secondary_archetype'] = 'Downhill Driver'

        result['role_effectiveness'] = compute_role_effectiveness(row, 'Ball Dominant Creator', p)
        return result

    # -------------------------------------------------------------------------
    # 1c. PLAYMAKING HUB (elite playmaker + post-up creator, any scoring level)
    # Catches Sabonis-type: high AST, post-up heavy, not traditional ball-dom
    # -------------------------------------------------------------------------
    if (is_elite_playmaker and post_up >= STATIC['POST_HUB_POSS'] and
        not is_ball_dominant_high):
        result['primary_archetype'] = 'Ball Dominant Creator'
        result['secondary_archetype'] = 'Post Creator'
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
        if post_up >= STATIC['POST_HUB_POSS']:
            result['secondary_archetype'] = 'Post Creator'
        elif is_elite_scorer:
            result['secondary_archetype'] = 'Volume Scorer'
        elif midrange_freq >= p.get('HIGH_MIDRANGE', 0.15):
            result['secondary_archetype'] = 'Midrange Specialist'
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
            # Non-star rim-dependent players (no skill finishing beyond dunks/layups)
            # must show paint floaters/hooks OR midrange game to qualify as Interior Scorer.
            # Otherwise they fall through to PnR Big / Off-Ball Finisher.
            # v4.3: harsher check — rim-runners with prominent PnR Roll Man activity always
            # fall through regardless of scoring; remaining bigs need skill finishing floor of 0.40.
            non_rim_interior = paint_freq + midrange_freq
            is_rim_runner_big = (at_rim_freq >= p.get('HIGH_AT_RIM', 0.30) and
                                 midrange_freq <= p.get('LOW_MIDRANGE', 0.05) and
                                 prrollman >= p.get('PROMINENT_PRROLLMAN', 0.04))
            has_skill_finishing = (not is_rim_runner_big and
                                  (pts_per36 >= p.get('ELITE_SCORING', 21.0) or
                                   non_rim_interior >= STATIC['SKILL_FINISHING_MIN']))
            if has_skill_finishing:
                result['primary_archetype'] = 'Interior Scorer'
                result['role_confidence'] = float(min(1.0,
                    np.sqrt(ball_dom / bd_cap) * 0.6 + (fg2a_rate / 0.85) * 0.4
                ))
                if post_up >= STATIC['POST_HUB_POSS']:
                    result['secondary_archetype'] = 'Post Creator'
                elif midrange_freq >= p.get('HIGH_MIDRANGE', 0.15):
                    result['secondary_archetype'] = 'Midrange Specialist'
                result['role_effectiveness'] = compute_role_effectiveness(row, 'Interior Scorer', p)
                return result
            # else: rim-dependent non-star → falls through to PnR Big / Off-Ball

        elif is_perimeter:
            result['primary_archetype'] = 'Perimeter Scorer'
            result['role_confidence'] = float(min(1.0,
                np.sqrt(ball_dom / bd_cap) * 0.6 + ((1 - fg2a_rate) / 0.65) * 0.4
            ))
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
                if post_up >= STATIC['POST_HUB_POSS']:
                    result['secondary_archetype'] = 'Post Creator'
                elif is_high_scorer:
                    result['secondary_archetype'] = 'Volume Scorer'
                elif midrange_freq >= p.get('HIGH_MIDRANGE', 0.15):
                    result['secondary_archetype'] = 'Midrange Specialist'
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
        # Skill finishing gate: non-star rim-dependent players fall through
        # v4.3: rim-runners with prominent PnR activity always fall through
        non_rim_interior_4b = paint_freq + midrange_freq
        is_rim_runner_big_4b = (at_rim_freq >= p.get('HIGH_AT_RIM', 0.30) and
                                midrange_freq <= p.get('LOW_MIDRANGE', 0.05) and
                                prrollman >= p.get('PROMINENT_PRROLLMAN', 0.04))
        has_skill_finishing_4b = (not is_rim_runner_big_4b and
                                 (pts_per36 >= p.get('ELITE_SCORING', 21.0) or
                                  non_rim_interior_4b >= STATIC['SKILL_FINISHING_MIN']))
        if not has_skill_finishing_4b:
            pass  # rim-dependent non-star → falls through to PnR Big / Off-Ball
        else:
            result['primary_archetype'] = 'Interior Scorer'
            raw_conf = (
                (fg2a_rate / 0.85) * 0.4 +
                (at_rim_plus_paint / 0.50) * 0.3 +
                (pts_per36 / max(p.get('ELITE_SCORING', 21.0), 15.0)) * 0.3
        )
            result['role_confidence'] = float(min(0.80, raw_conf))  # capped
            if post_up >= STATIC['POST_HUB_POSS']:
                result['secondary_archetype'] = 'Post Creator'
            elif midrange_freq >= p.get('HIGH_MIDRANGE', 0.15):
                result['secondary_archetype'] = 'Midrange Specialist'
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
            result['role_effectiveness'] = compute_role_effectiveness(row, 'Connector', p)
            return result

    # ---------------------------------------------------------------------------
    # 5. PnR BIG (Rolling / Popping)
    # Primary off-ball play is PnR Roll Man (higher than cuts)
    # Split by FG3A: high 3PA = popper, low = roller
    # v4.3: Added secondary "prominent" path for interior bigs — PnR Roll Man
    # does NOT need to be the most common play; just prominent (>= P50) + big profile
    # ---------------------------------------------------------------------------
    prm_gate = p.get('HIGH_PRROLLMAN', 0.06)
    prm_prominent = p.get('PROMINENT_PRROLLMAN', 0.04)
    is_interior_profile = (fg2a_rate >= p.get('FG2A_INTERIOR', 0.68) or
                           at_rim_freq >= p.get('HIGH_AT_RIM', 0.30))

    # Strict path: dominant PnR Roll Man (P70, > cuts, >= 50% of spotup)
    pnr_big_strict = (prrollman >= prm_gate and prrollman > cut_poss and
                      prrollman >= spotup * 0.5)
    # Looser path: prominent PnR (P50) + interior profile (rim-running bigs)
    pnr_big_prominent = (prrollman >= prm_prominent and is_interior_profile and
                         not is_playmaker)

    if pnr_big_strict or pnr_big_prominent:
        fg3a_mid = p.get('FG3A_MEDIAN', 5.5)  # P50
        fg3a_pop_gate = p.get('FG3A_MEDIAN', 5.5) * 0.67
        if fg3a_per36 >= fg3a_pop_gate:
            result['primary_archetype'] = 'PnR Popping Big'
        else:
            result['primary_archetype'] = 'PnR Rolling Big'
        prm_cap = max(p.get('HIGH_PRROLLMAN', 0.06) * 2.5, 0.15)
        result['role_confidence'] = float(min(1.0, np.sqrt(prrollman / prm_cap)))
        if not pnr_big_strict:
            result['role_confidence'] = float(min(0.85, result['role_confidence']))
        if result['primary_archetype'] == 'PnR Rolling Big':
            if at_rim_freq >= STATIC['LOB_THREAT_AT_RIM']:
                result['secondary_archetype'] = 'Lob Threat'
            elif ast_per36 >= STATIC['PLAYMAKING_BIG_AST']:
                result['secondary_archetype'] = 'Playmaking Big'
            elif transition > 0.15:
                result['secondary_archetype'] = 'Transition Runner'
        else:  # PnR Popping Big
            if (fg3_pct >= STATIC['ELITE_SHOOTER_FG3_PCT'] and
                    fg3a_per36 >= STATIC['ELITE_SHOOTER_FG3A']):
                result['secondary_archetype'] = 'Elite Shooter'
            elif ast_per36 >= STATIC['PLAYMAKING_BIG_AST']:
                result['secondary_archetype'] = 'Playmaking Big'
        result['role_effectiveness'] = compute_role_effectiveness(row, result['primary_archetype'], p)
        return result

    # ---------------------------------------------------------------------------
    # 6. OFF-BALL FINISHER (cuts, transition, putbacks — not primarily roll man)
    # v4.3: Before assigning Off-Ball Finisher, redirect bigs with prominent
    # PnR Roll Man activity to PnR Big (catches rim-running bigs who cut AND roll)
    # ---------------------------------------------------------------------------
    offball_finish_score = cut_pnrrm + putback * 0.5 + transition * 0.3
    cut_pnrrm_gate = p.get('HIGH_CUT_PNRRM', 0.18)
    if cut_pnrrm >= cut_pnrrm_gate or offball_finish_score > 0.20:
        # Redirect: if player looks like a big with prominent PnR activity → PnR Big
        if (prrollman >= prm_prominent and is_interior_profile and not is_playmaker):
            fg3a_mid = p.get('FG3A_MEDIAN', 5.5)
            if fg3a_per36 >= fg3a_mid:
                result['primary_archetype'] = 'PnR Popping Big'
            else:
                result['primary_archetype'] = 'PnR Rolling Big'
            prm_cap = max(p.get('HIGH_PRROLLMAN', 0.06) * 2.5, 0.15)
            result['role_confidence'] = float(min(0.80, np.sqrt(prrollman / prm_cap)))
            result['role_effectiveness'] = compute_role_effectiveness(row, result['primary_archetype'], p)
            return result

        result['primary_archetype'] = 'Off-Ball Finisher'
        ob_cap = max(cut_pnrrm_gate * 1.5, 0.25)
        result['role_confidence'] = float(min(1.0, np.sqrt(offball_finish_score / ob_cap)))
        result['role_effectiveness'] = compute_role_effectiveness(row, 'Off-Ball Finisher', p)
        return result

    # ---------------------------------------------------------------------------
    # 7. OFF-BALL MOVEMENT SHOOTER
    # movement >= P60 AND movement/(movement+spotup) >= 0.30  (v4.2 loosened)
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
    # 7b. PERIMETER SCORER (moderate self-creation, non-playmaker)
    # Players with enough ball handling to create their own shot — not pure
    # spot-up guys.  Redirects capable ball-handlers away from Off-Ball
    # Stationary Shooter.  Requires BD >= P50 AND scoring above P25.
    # ---------------------------------------------------------------------------
    bd_base_gate = p.get('BD_BASE', 0.17)
    low_scoring_gate = p.get('LOW_SCORING', 13.0)
    if (ball_dom >= bd_base_gate and
        not is_playmaker and
        pts_per36 >= low_scoring_gate):
        result['primary_archetype'] = 'Perimeter Scorer'
        bd_cap = max(p.get('BD_MIN', 0.30), 0.25)
        result['role_confidence'] = float(min(1.0,
            np.sqrt(ball_dom / bd_cap) * 0.5 +
            np.sqrt(pts_per36 / max(p.get('ELITE_SCORING', 21.0), 15.0)) * 0.5
        ))
        result['role_effectiveness'] = compute_role_effectiveness(row, 'Perimeter Scorer', p)
        return result

    # ---------------------------------------------------------------------------
    # 8. OFF-BALL STATIONARY SHOOTER
    # ---------------------------------------------------------------------------
    spotup_gate = p.get('HIGH_SPOTUP', 0.25)
    if spotup >= spotup_gate:
        result['primary_archetype'] = 'Off-Ball Stationary Shooter'
        result['role_confidence'] = float(min(1.0, np.sqrt(spotup / max(spotup_gate * 1.4, 0.35))))
        result['role_effectiveness'] = compute_role_effectiveness(row, 'Off-Ball Stationary Shooter', p)
        return result

    # ---------------------------------------------------------------------------
    # 9. BEST-FIT FALLBACK (replaces Rotation Piece catch-all)
    # Uses scoring signals to assign closest archetype instead of a generic bucket
    # ---------------------------------------------------------------------------
    reb36 = row.get('reb_per36', 0) or 0
    stl36 = row.get('stl_per36', 0) or 0
    blk36 = row.get('blk_per36', 0) or 0

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
        result['secondary_archetype'] = 'Inside-the-Arc'
    elif transition > 0.10:
        result['secondary_archetype'] = 'Transition Player'

    result['role_effectiveness'] = compute_role_effectiveness(row, result['primary_archetype'], p)
    return result


def classify_all_players(features: pd.DataFrame, pctiles: dict) -> pd.DataFrame:
    """Classify all players in the features dataframe using season percentile thresholds."""

    classifications = []
    for idx, row in features.iterrows():
        result = classify_archetype(row, pctiles)
        result['player_id'] = row['player_id']
        result['player_name'] = row['player_name']
        result['season'] = row['season']
        classifications.append(result)

    class_df = pd.DataFrame(classifications)

    output = features.merge(class_df, on=['player_id', 'player_name', 'season'])

    return output


# =============================================================================
# ARCHETYPE EMBEDDING (v4.3 — Soft Role Model)
# =============================================================================
# Reuses the same gate math from classify_archetype() to produce a continuous,
# confidence-weighted embedding vector for each player.  Instead of a one-hot
# classification, each archetype gets a weight ∈ [0, 1] reflecting how well
# the player fits that role.
#
# Algorithm:
#   1. For each archetype, compute a raw_score using the normalised components
#      from the classification hierarchy (uncapped — can go negative or > 1).
#   2. Convert raw_score to confidence via exponential decay:
#        confidence[A] = exp(-α · max(0, 1 − raw_score))
#   3. Build weights:
#        weight[A] = max(0, raw_score) * confidence[A]
#   4. L1-normalise weights to produce a probability-like embedding.
#
# The resulting vector has dimensionality = number of archetypes (11).
# Strong, clean archetype fits dominate; weak / accidental fits fade out.
# =============================================================================

ARCHETYPE_ORDER = [
    'Ball Dominant Creator',
    'All-Around Scorer',
    'Ballhandler',
    'Interior Scorer',
    'Perimeter Scorer',
    'Connector',
    'PnR Rolling Big',
    'PnR Popping Big',
    'Off-Ball Finisher',
    'Off-Ball Movement Shooter',
    'Off-Ball Stationary Shooter',
]

# Temperature / decay rate for confidence transformation
EMBEDDING_ALPHA = 2.0  # higher = steeper falloff for weak fits


def compute_archetype_embedding(row: pd.Series, pctiles: dict) -> dict:
    """
    Compute a soft archetype embedding for a single player.

    Returns dict with:
      - 'emb_{archetype}': float weight for each archetype (sums to ~1.0)
      - 'emb_entropy': Shannon entropy of the embedding (higher = more hybrid)
      - 'emb_dominance': weight of the strongest archetype (higher = purer role)
    """
    p = pctiles

    # ---- Extract metrics (same as classify_archetype) ----
    ball_dom = row.get('ball_dominant_pct', 0) or 0
    on_ball = row.get('on_ball_creation', 0) or 0
    post_up = row.get('postup_poss_pct', 0) or 0
    ast_per36 = row.get('ast_per36', 0) or 0
    pts_per36 = row.get('pts_per36', 0) or 0
    fg2a_rate = row.get('fg2a_rate', 0.5)
    if pd.isna(fg2a_rate):
        fg2a_rate = 0.5
    fg3a_per36 = row.get('fg3a_per36', 0) or 0
    fg3_pct = row.get('fg3_pct', 0) or 0
    ts = row.get('ts_pct', 0)
    if pd.isna(ts) or ts == 0:
        ts = 0.55
    cut_pnrrm = row.get('cut_pnrrm_pct', 0) or 0
    spotup = row.get('spotup_pct', 0) or 0
    movement = row.get('movement_shooter_pct', 0) or 0
    transition = row.get('transition_pct', 0) or 0
    putback = row.get('putback_pct', 0) or 0
    prrollman = row.get('prrollman_poss_pct', 0) or 0
    at_rim_freq = row.get('at_rim_freq', 0) or 0
    if pd.isna(at_rim_freq):
        at_rim_freq = 0
    midrange_freq = row.get('midrange_freq', 0) or 0
    if pd.isna(midrange_freq):
        midrange_freq = 0
    touches = row.get('touches', 0) or 0
    sec_per_touch = row.get('avg_sec_per_touch', 3.0)
    if pd.isna(sec_per_touch):
        sec_per_touch = 3.0
    playmaking = row.get('playmaking_score', 0) or 0

    # ---- Normalisation denominators (threshold → P95 range) ----
    bd_floor = p.get('BD_MIN', 0.37)
    bd_ceil = max(p.get('BD_P95', 0.51), bd_floor + 0.10)
    pts_floor_moderate = p.get('MODERATE_SCORING', 15.5)
    pts_ceil = max(p.get('PTS_P95', 26.0), pts_floor_moderate + 5.0)
    ast_floor = p.get('PLAYMAKER', 5.0)
    ast_ceil = max(p.get('AST_P95', 7.8), ast_floor + 1.5)

    # Helper: uncapped normalisation
    def norm(val, floor, ceil):
        denom = ceil - floor
        if denom <= 0:
            return 0.0
        return (val - floor) / denom

    # ---- Raw scores per archetype (uncapped) ----
    raw = {}

    # 1. Ball Dominant Creator
    bd_s = norm(ball_dom, bd_floor, bd_ceil)
    pts_s = norm(pts_per36, pts_floor_moderate, pts_ceil)
    ast_s = norm(ast_per36, ast_floor, ast_ceil)
    raw['Ball Dominant Creator'] = 0.60 * (0.50 * bd_s + 0.50 * pts_s) + 0.40 * ast_s

    # 2. All-Around Scorer (playmaker + scorer)
    aa_ast = norm(ast_per36, ast_floor, ast_ceil)
    aa_pts = norm(pts_per36, pts_floor_moderate, pts_ceil)
    raw['All-Around Scorer'] = 0.50 * aa_ast + 0.50 * aa_pts

    # 3. Ballhandler (high playmaking, moderate scoring — penalise high scorers)
    bh_ast = norm(ast_per36, ast_floor, ast_ceil)
    # Inverse scoring: high scoring reduces ballhandler fit
    high_scoring_gate = p.get('HIGH_SCORING', 18.5)
    bh_pts_inv = max(0, 1.0 - (pts_per36 / max(high_scoring_gate, 12.0)))
    raw['Ballhandler'] = 0.70 * bh_ast + 0.30 * bh_pts_inv

    # 4. Interior Scorer (ball dominant + interior profile)
    bd_all_around_floor = p.get('BD_ALL_AROUND', 0.23)
    int_bd = norm(ball_dom, bd_all_around_floor, bd_ceil)
    int_fg2a = norm(fg2a_rate, p.get('FG2A_INTERIOR', 0.68), 0.95)
    int_pts = norm(pts_per36, pts_floor_moderate, pts_ceil)
    int_mr = midrange_freq / max(p.get('HIGH_MIDRANGE', 0.15), 0.05)  # skill finishing proxy
    raw['Interior Scorer'] = 0.30 * int_bd + 0.25 * int_fg2a + 0.25 * int_pts + 0.20 * int_mr

    # 5. Perimeter Scorer (ball dominant + perimeter profile)
    perim_base = p.get('BD_BASE', 0.17)
    per_bd = norm(ball_dom, perim_base, bd_ceil)
    per_fg2a_inv = max(0, 1.0 - fg2a_rate) / max(1.0 - p.get('FG2A_PERIMETER', 0.49), 0.20)
    per_pts = norm(pts_per36, p.get('LOW_SCORING', 13.0), pts_ceil)
    raw['Perimeter Scorer'] = 0.35 * per_bd + 0.35 * per_fg2a_inv + 0.30 * per_pts

    # 6. Connector (moderate playmaking + activity)
    conn_ast_floor = p.get('CONNECTOR_AST', 3.3)
    conn_ast = norm(ast_per36, conn_ast_floor, ast_ceil)
    conn_touch = touches / max(p.get('TOUCHES_MEDIAN', 40.0), 10.0)
    conn_sec = max(0, 1.0 - sec_per_touch / STATIC['CONNECTOR_SEC_PER_TOUCH_MAX'])
    conn_pts_inv = max(0, 1.0 - (pts_per36 / max(high_scoring_gate, 12.0)))
    raw['Connector'] = 0.50 * conn_ast + 0.20 * conn_touch + 0.10 * conn_sec + 0.20 * conn_pts_inv

    # 7. PnR Rolling Big (roll man + interior + low 3PA)
    prm_gate = p.get('HIGH_PRROLLMAN', 0.06)
    prm_prominent = p.get('PROMINENT_PRROLLMAN', 0.04)
    prm_s = prrollman / max(prm_gate, 0.01)
    fg3a_mid = p.get('FG3A_MEDIAN', 5.5)
    roll_fg3_inv = max(0, 1.0 - fg3a_per36 / max(fg3a_mid, 1.0))
    roll_interior = fg2a_rate / 0.85
    raw['PnR Rolling Big'] = 0.55 * prm_s + 0.25 * roll_interior + 0.20 * roll_fg3_inv

    # 8. PnR Popping Big (roll man + perimeter shooting)
    pop_fg3 = fg3a_per36 / max(fg3a_mid, 1.0)
    raw['PnR Popping Big'] = 0.55 * prm_s + 0.45 * pop_fg3

    # 9. Off-Ball Finisher (cuts, transition, putbacks)
    cut_gate = p.get('HIGH_CUT_PNRRM', 0.18)
    finish_score = cut_pnrrm + putback * 0.5 + transition * 0.3
    ob_cut = cut_pnrrm / max(cut_gate, 0.05)
    ob_finish = finish_score / 0.25
    raw['Off-Ball Finisher'] = 0.50 * ob_cut + 0.50 * ob_finish

    # 10. Off-Ball Movement Shooter
    move_gate = p.get('HIGH_MOVEMENT', 0.10)
    move_s = movement / max(move_gate, 0.02)
    total_shoot = movement + spotup
    move_ratio = movement / total_shoot if total_shoot > 0 else 0
    move_r = move_ratio / max(STATIC['MOVEMENT_RATIO_MIN'], 0.10)
    raw['Off-Ball Movement Shooter'] = 0.60 * move_s + 0.40 * move_r

    # 11. Off-Ball Stationary Shooter
    spot_gate = p.get('HIGH_SPOTUP', 0.25)
    spot_s = spotup / max(spot_gate, 0.05)
    fg3_s = fg3_pct / max(p.get('HIGH_FG3_PCT', 0.37), 0.30)
    raw['Off-Ball Stationary Shooter'] = 0.65 * spot_s + 0.35 * fg3_s

    # ---- Convert raw scores → confidence-weighted embedding ----
    alpha = EMBEDDING_ALPHA
    weights = {}
    for arch in ARCHETYPE_ORDER:
        r = raw.get(arch, 0.0)
        # Confidence: exponential decay of distance from score=1.0
        # If r >= 1 → confidence = 1; if r ≈ 0 → confidence ≈ exp(-alpha) ≈ 0.13
        distance = max(0.0, 1.0 - r)
        confidence = float(np.exp(-alpha * distance))
        # Weight = positive raw score × confidence
        weights[arch] = max(0.0, r) * confidence

    total = sum(weights.values())

    # Build embedding dict
    embedding = {}
    for arch in ARCHETYPE_ORDER:
        key = f"emb_{arch.lower().replace(' ', '_').replace('-', '_')}"
        embedding[key] = float(weights[arch] / total) if total > 0 else 1.0 / len(ARCHETYPE_ORDER)

    # Compute entropy (Shannon) — higher = more hybrid player
    probs = [embedding[f"emb_{a.lower().replace(' ', '_').replace('-', '_')}"] for a in ARCHETYPE_ORDER]
    entropy = 0.0
    for prob in probs:
        if prob > 1e-10:
            entropy -= prob * np.log2(prob)
    embedding['emb_entropy'] = float(entropy)

    # Max entropy for 11 categories ≈ 3.459
    max_entropy = np.log2(len(ARCHETYPE_ORDER))
    embedding['emb_entropy_norm'] = float(entropy / max_entropy) if max_entropy > 0 else 0.0

    # Dominance: weight of the strongest archetype
    embedding['emb_dominance'] = float(max(probs))

    return embedding


def compute_all_embeddings(features: pd.DataFrame, pctiles: dict) -> pd.DataFrame:
    """Compute archetype embeddings for all players in the features dataframe."""
    embeddings = []
    for idx, row in features.iterrows():
        emb = compute_archetype_embedding(row, pctiles)
        emb['player_id'] = row['player_id']
        emb['player_name'] = row['player_name']
        emb['season'] = row['season']
        embeddings.append(emb)

    return pd.DataFrame(embeddings)


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
                     'ALL_AROUND_SCORING', 'HIGH_PRROLLMAN', 'PROMINENT_PRROLLMAN',
                     'FG3A_MEDIAN', 'TOUCHES_MEDIAN',
                     'HIGH_AT_RIM', 'HIGH_MIDRANGE', 'MODERATE_MIDRANGE',
                     'AT_RIM_PAINT_P60']:
            print(f"      {key}: {pctiles.get(key, '?'):.4f}")

        classified = classify_all_players(features, pctiles)
        all_results.append(classified)

        # Print summary
        qual = classified[
            (classified['min'] >= STATIC['MIN_MINUTES']) &
            (classified['gp'] >= STATIC['MIN_GP']) &
            (classified['mpg'] >= STATIC['MIN_MPG'])
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

        # ---- Compute archetype embeddings per-season ----
        print("\n Computing archetype embeddings...")
        all_emb = []
        for season in SEASONS:
            season_data = final_df[final_df['season'] == season]
            if season_data.empty:
                continue
            # Re-compute pctiles for embedding (use same qualified pool)
            season_pctiles = compute_season_percentiles(season_data)
            emb_df = compute_all_embeddings(season_data, season_pctiles)
            all_emb.append(emb_df)

        if all_emb:
            embedding_df = pd.concat(all_emb, ignore_index=True)
            # Merge embeddings into final_df
            emb_cols = [c for c in embedding_df.columns if c.startswith('emb_')]
            merge_cols = ['player_id', 'player_name', 'season'] + emb_cols
            final_df = final_df.merge(embedding_df[merge_cols],
                                      on=['player_id', 'player_name', 'season'], how='left')
            # Save standalone embedding file
            save_standardized(embedding_df, OUTPUT_DIR / "archetype_embeddings.parquet")
            embedding_df.to_csv(OUTPUT_DIR / "archetype_embeddings.csv", index=False)
            print(f"   Saved {len(embedding_df)} embeddings to data/processed/archetype_embeddings.parquet")

        save_standardized(final_df, OUTPUT_DIR / "player_archetypes.parquet")
        final_df.to_csv(OUTPUT_DIR / "player_archetypes.csv", index=False)

        print(f"\n Saved {len(final_df)} player-seasons to data/processed/player_archetypes.parquet")

        # Validation
        print("\n" + "=" * 70)
        print("VALIDATION: KEY PLAYER CLASSIFICATIONS")
        print("=" * 70)

        s25 = final_df[final_df['season'] == '2024-25'].copy()
        stars = [
            'LeBron James', 'Stephen Curry', 'Kevin Durant', 'Giannis Antetokounmpo',
            'Luka Dončić', 'Nikola Jokic', 'Jayson Tatum', 'Jonathan Kuminga',
            'Shai Gilgeous-Alexander', 'Anthony Edwards', 'Draymond Green',
            'Jalen Brunson', 'Tyrese Haliburton', 'Derrick White',
            'Joel Embiid', 'Clint Capela', 'Rudy Gobert', 'Nicolas Batum',
            'Domantas Sabonis', 'Trae Young', 'Devin Booker',
            'Klay Thompson', 'Buddy Hield', 'Brook Lopez',
            'Jarrett Allen', 'Daniel Gafford',
        ]

        for name in stars:
            player = s25[s25['player_name'].str.contains(name, case=False, na=False)]
            if len(player) > 0:
                pp = player.iloc[0]
                sec = f" / {pp['secondary_archetype']}" if pp.get('secondary_archetype') else ""
                eff_tier = pp.get('efficiency_tier', '?')
                rc = pp.get('role_confidence', 0)
                re = pp.get('role_effectiveness', 0)
                fg2 = pp.get('fg2a_rate', 0)
                bd = pp.get('ball_dominant_pct', 0)
                ast = pp.get('ast_per36', 0)
                print(f"  {pp['player_name']:<25} {pp['primary_archetype']}{sec}  conf={rc:.0%} eff={re:.0%}  [{eff_tier}] BD={bd:.2f} AST={ast:.1f} FG2A={fg2:.0%}")
            else:
                s24 = final_df[final_df['season'] == '2023-24']
                player = s24[s24['player_name'].str.contains(name, case=False, na=False)]
                if len(player) > 0:
                    pp = player.iloc[0]
                    sec = f" / {pp['secondary_archetype']}" if pp.get('secondary_archetype') else ""
                    rc = pp.get('role_confidence', 0)
                    re = pp.get('role_effectiveness', 0)
                    print(f"  {pp['player_name']:<25} {pp['primary_archetype']}{sec}  conf={rc:.0%} eff={re:.0%} [{pp['season']}]")

        # ---- Embedding example output ----
        if 'emb_dominance' in final_df.columns:
            print("\n" + "=" * 70)
            print("ARCHETYPE EMBEDDINGS (sample 2024-25)")
            print("=" * 70)
            emb_cols_display = [c for c in final_df.columns if c.startswith('emb_') and c not in
                                ('emb_entropy', 'emb_entropy_norm', 'emb_dominance')]
            emb_stars = ['LeBron James', 'Stephen Curry', 'Nikola Jokic', 'Jarrett Allen',
                         'Klay Thompson', 'Draymond Green', 'Clint Capela']
            for name in emb_stars:
                player = s25[s25['player_name'].str.contains(name, case=False, na=False)]
                if len(player) == 0:
                    continue
                pp = player.iloc[0]
                if 'emb_dominance' not in pp.index:
                    # Merge back from final_df
                    pp = final_df[(final_df['player_name'].str.contains(name, case=False, na=False)) &
                                  (final_df['season'] == '2024-25')]
                    if len(pp) == 0:
                        continue
                    pp = pp.iloc[0]
                dom = pp.get('emb_dominance', 0)
                ent = pp.get('emb_entropy_norm', 0)
                top3 = []
                for c in emb_cols_display:
                    val = pp.get(c, 0)
                    if val and val > 0.05:
                        arch_name = c.replace('emb_', '').replace('_', ' ').title()
                        top3.append((arch_name, val))
                top3.sort(key=lambda x: x[1], reverse=True)
                top3_str = " | ".join(f"{a}: {v:.0%}" for a, v in top3[:4])
                print(f"  {pp['player_name']:<25} dominance={dom:.0%} hybrid={ent:.0%}  [{top3_str}]")

    return final_df


if __name__ == "__main__":
    result = main()
