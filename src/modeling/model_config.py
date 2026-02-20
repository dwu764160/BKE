"""
src/modeling/model_config.py
=============================================================================
BKE v2.0 — Centralized Configuration for the Portable Talent vs
             Role-Dependent Impact Decomposition Engine.

v2.0 corrections over v1.5:
  - Z-score aggregation replaces percentile averaging (Fix #1)
  - Variance-based portability index replaces compositional ratio (Fix #2)
  - 8 independent portable dimensions (Layer 1C primary engine)
  - Layer 1 weighting: 25% RAPM, 20% Playtype, 55% Dimension Model
  - Driving Gravity revised (no FT%, focus on rim pressure)
  - Turnover Control restored as offensive dimension
  - Bayesian shrinkage for noisy metrics
  - Cross-layer independence enforced

All thresholds, weights, paths, and structural constants live here.
No other module should define model-wide constants.
=============================================================================
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = "data/historical"
PROCESSED_DIR = "data/processed"
OUTPUT_DIR = "data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Input data files
RAPM_PATH = os.path.join(PROCESSED_DIR, "player_rapm.parquet")
MODELING_INPUTS_PATH = os.path.join(PROCESSED_DIR, "modeling_inputs_all.parquet")
PLAYER_PROFILES_PATH = os.path.join(PROCESSED_DIR, "player_profiles_advanced.parquet")
PLAYER_ARCHETYPES_PATH = os.path.join(PROCESSED_DIR, "player_archetypes.parquet")
DEFENSIVE_ARCHETYPES_PATH = os.path.join(PROCESSED_DIR, "defensive_archetypes_v2.parquet")
LINEAR_METRICS_PATH = os.path.join(PROCESSED_DIR, "metrics_linear.parquet")
WIN_SHARES_PATH = os.path.join(PROCESSED_DIR, "metrics_win_shares.parquet")
ARCHETYPE_EMBEDDINGS_PATH = os.path.join(PROCESSED_DIR, "archetype_embeddings.parquet")
POSITION_ESTIMATES_PATH = os.path.join(PROCESSED_DIR, "player_position_estimates.parquet")
POSSESSIONS_GLOB = os.path.join(DATA_DIR, "possessions_clean_*.parquet")

# Output files — v2.0
BKE_OUTPUT_PARQUET = os.path.join(OUTPUT_DIR, "bke_v20_decomposition.parquet")
BKE_OUTPUT_CSV = os.path.join(OUTPUT_DIR, "bke_v20_decomposition.csv")
BKE_REPORT_JSON = os.path.join(OUTPUT_DIR, "bke_v20_report.json")

# ---------------------------------------------------------------------------
# Seasons
# ---------------------------------------------------------------------------
SEASONS: List[str] = ["2022-23", "2023-24", "2024-25"]

# ---------------------------------------------------------------------------
# Qualification filters
# ---------------------------------------------------------------------------
MIN_MINUTES = 500
MIN_GP = 20
MIN_MPG = 15.0
MIN_POSSESSIONS = 500  # minimum possessions for RAPM reliability

# ---------------------------------------------------------------------------
# RAPM configuration
# ---------------------------------------------------------------------------
RAPM_PREFERRED_TYPE = "pooled_split"  # prefer pooled split for O/D separation
RAPM_FALLBACK_TYPE = "single_season_split"

# Multi-year Bayesian RAPM weights (year offset → weight)
SEASON_DECAY_WEIGHTS: Dict[int, float] = {
    0: 1.00,   # current season
    1: 0.70,   # prior season
    2: 0.50,   # 2 seasons ago
}

RIDGE_ALPHAS = [25, 50, 100, 200, 400, 800, 1600, 3200]

# ---------------------------------------------------------------------------
# Portable Talent Score (Layer 1) weights — v2.0 structure
# ---------------------------------------------------------------------------
@dataclass
class PortableTalentConfig:
    """
    Weights for the Portable Talent Score computation.

    v2.0 structure:
      Layer 1 = 25% RAPM Backbone + 20% Playtype Efficiency + 55% Dimension Model (1C)
      Layer 1C = 8 logically independent dimensions, initially equally weighted.
    """
    # --- Layer 1 high-level weights (sum to 1.0) ---
    w_rapm_backbone: float = 0.25       # Layer 1A: RAPM impact
    w_playtype_efficiency: float = 0.20  # Layer 1B: Playtype efficiency composite
    w_dimension_model: float = 0.55      # Layer 1C: 8-dimension portable model

    # --- Layer 1C: 8 dimension weights (equal initially, sum to 1.0) ---
    # Offensive dimensions
    w_dim_shooting_gravity: float = 0.125       # Dim 1: Shooting gravity
    w_dim_driving_gravity: float = 0.125        # Dim 2: Driving / rim pressure
    w_dim_playmaking: float = 0.125             # Dim 3: Playmaking
    w_dim_extra_possession: float = 0.125       # Dim 4: Extra possession creation (cross-domain)
    w_dim_turnover_control: float = 0.125       # Dim 7: Turnover control (offensive)
    # Defensive dimensions
    w_dim_defensive_playmaking: float = 0.125   # Dim 5: Defensive playmaking / chaos
    w_dim_defensive_impact: float = 0.125       # Dim 6: RAPM-informed defensive impact
    w_dim_defensive_versatility: float = 0.125  # Dim 8: Matchup spectrum / switching

    # Extra Possession Creation split between O/D composites
    extra_poss_offensive_share: float = 0.40
    extra_poss_defensive_share: float = 0.60

    # Luck adjustment parameters
    shooting_regression_rate: float = 0.40  # How much to regress 3PT luck
    opponent_3pt_regression: float = 0.50   # Opponent 3PT variance regression
    ts_regression_rate: float = 0.30        # TS% variance regression (renamed from ft_regression_rate)

    # Stability weighting
    min_poss_full_weight: int = 3000  # possessions for full reliability weight
    min_poss_partial: int = 500       # minimum for any weight


PORTABLE_TALENT = PortableTalentConfig()

# ---------------------------------------------------------------------------
# Role Utilization Efficiency (Layer 2) config
# ---------------------------------------------------------------------------
@dataclass
class RoleUtilizationConfig:
    """Config for playtype surplus and Role Utilization Efficiency."""
    # Playtypes tracked for contribution vector
    playtypes: List[str] = field(default_factory=lambda: [
        "ISOLATION",
        "PRBALLHANDLER",
        "POSTUP",
        "CUT",
        "PRROLLMAN",
        "HANDOFF",
        "OFFSCREEN",
        "SPOTUP",
        "TRANSITION",
    ])

    # PPP columns suffix pattern
    ppp_suffix: str = "_PPP"
    poss_pct_suffix: str = "_POSS_PCT"
    poss_suffix: str = "_POSS"

    # Minimum possessions per playtype for reliable efficiency
    # NOTE: Data stores per-game possession counts, not season totals.
    # 0.3 per game ≈ 25 total possessions over an 82-game season.
    min_playtype_poss: float = 0.3

    # Usage bucket boundaries (possession share percentiles for conditioning)
    usage_buckets: List[float] = field(default_factory=lambda: [0.0, 0.05, 0.10, 0.20, 0.35, 1.0])

    # Archetype optimal playtype distributions derived from data
    # These are populated at runtime from archetype cohort means
    archetype_optimal_distributions: Dict[str, Dict[str, float]] = field(default_factory=dict)


ROLE_UTILIZATION = RoleUtilizationConfig()

# ---------------------------------------------------------------------------
# Archetype Elevation (Layer 3) config
# ---------------------------------------------------------------------------
@dataclass
class ArchetypeElevationConfig:
    """Config for archetype elevation scoring."""
    # Impact metrics to compare against archetype baseline
    impact_metrics: List[str] = field(default_factory=lambda: [
        "rapm", "orapm", "drapm",
    ])

    # Playtype surplus metrics for archetype comparison
    surplus_metrics: List[str] = field(default_factory=lambda: [
        "playtype_surplus_total",
    ])

    # Weights for elevation components
    w_rapm_elevation: float = 0.50
    w_playtype_elevation: float = 0.30
    w_efficiency_elevation: float = 0.20


ARCHETYPE_ELEVATION = ArchetypeElevationConfig()

# ---------------------------------------------------------------------------
# Scheme Amplification (Layer 4) config
# ---------------------------------------------------------------------------
@dataclass
class SchemeAmplificationConfig:
    """Config for scheme/context sensitivity estimation."""
    # Minimum lineup observations for variance estimation
    min_lineup_poss: int = 50

    # Number of lineup samples for bootstrapping
    n_bootstrap_samples: int = 100

    # Weights for scheme components
    w_lineup_variance: float = 0.40       # Impact variance across lineups
    w_on_off_variance: float = 0.30       # On/off split variance
    w_teammate_dependency: float = 0.30   # Teammate quality dependency

    # Stability thresholds
    high_stability_threshold: float = 0.70  # Above this = portable
    low_stability_threshold: float = 0.30   # Below this = scheme-dependent


SCHEME_AMPLIFICATION = SchemeAmplificationConfig()

# ---------------------------------------------------------------------------
# Percentile Engine config
# ---------------------------------------------------------------------------
@dataclass
class PercentileConfig:
    """Configuration for the 3-level percentile standardization."""
    # The three standardization levels
    levels: List[str] = field(default_factory=lambda: [
        "league",      # all qualified players
        "position",    # positional cohort (G, F, C, GF, FC)
        "archetype",   # archetype cohort
    ])

    # Position buckets for positional percentiles
    position_buckets: Dict[str, List[str]] = field(default_factory=lambda: {
        "Guard": ["PG", "SG", "G"],
        "Guard-Forward": ["SG-SF", "SF-SG", "GF"],
        "Forward": ["SF", "PF", "F"],
        "Forward-Center": ["PF-C", "C-PF", "FC"],
        "Center": ["C"],
    })

    # Minimum cohort size for reliable percentile computation
    min_cohort_size: int = 10

    # Smoothing for small cohorts (Bayesian shrinkage toward league pctile)
    small_cohort_shrinkage: float = 0.30


PERCENTILE = PercentileConfig()

# ---------------------------------------------------------------------------
# Z-Score Aggregation config (v2.0 Fix #1)
# ---------------------------------------------------------------------------
@dataclass
class ZScoreConfig:
    """
    Configuration for z-score based aggregation.

    v2.0 replaces percentile averaging with z-score weighted sums.
    Raw → Z-score → Weighted Sum → Final Z → Final Percentile.
    Percentiles are presentation only, not aggregation math.
    """
    # Winsorize z-scores at ±N std to prevent outlier distortion
    z_winsorize_limit: float = 3.5

    # Minimum non-null values to compute a reliable z-score
    min_values_for_z: int = 10

    # Fallback z-score when data is insufficient
    fallback_z: float = 0.0


ZSCORE = ZScoreConfig()

# ---------------------------------------------------------------------------
# Bayesian Shrinkage config (v2.0)
# ---------------------------------------------------------------------------
@dataclass
class BayesianShrinkageConfig:
    """
    Configuration for empirical Bayes shrinkage on noisy metrics.

    Applied to:
      - Defensive playmaking (STL%, BLK% can be volatile)
      - On/off metrics (small sample noise)
      - Small-sample role splits
    """
    # Shrinkage strength: 0 = no shrinkage, 1 = full shrinkage to prior
    defensive_playmaking_shrinkage: float = 0.20
    on_off_shrinkage: float = 0.25
    small_sample_shrinkage: float = 0.30

    # Minimum games for full-weight (below this, increase shrinkage)
    min_gp_full_weight: int = 50

    # Prior source: "league_mean" or "positional_mean"
    prior_source: str = "league_mean"


BAYESIAN_SHRINKAGE = BayesianShrinkageConfig()

# ---------------------------------------------------------------------------
# True Portability Index config (v2.0 Fix #2)
# ---------------------------------------------------------------------------
@dataclass
class PortabilityConfig:
    """
    Configuration for variance-based portability measurement.

    v2.0 replaces the fake compositional ratio (PTS/Total) with
    true context-stability measurement:
      Portability Index = 1 - Normalized Impact Variance Across Contexts

    Components:
      1. Lineup Stability Index — variance across teammate contexts
      2. Role Elasticity Test — impact change under ±usage shift
      3. Archetype Transfer Simulation — efficiency in alt archetypes
      4. On/Off Context Sensitivity — variance across environment types
    """
    # Weights for portability components
    w_lineup_stability: float = 0.30
    w_role_elasticity: float = 0.25
    w_archetype_transfer: float = 0.20
    w_context_sensitivity: float = 0.25

    # Role elasticity test: simulate ±X% usage shift
    usage_shift_pct: float = 0.05  # ±5%

    # Archetype transfer: project into N alternative templates
    n_alt_archetypes: int = 3

    # Thresholds for classification
    high_portability: float = 0.70   # Scalable star
    low_portability: float = 0.40    # System-amplified


PORTABILITY = PortabilityConfig()

# ---------------------------------------------------------------------------
# Decomposition Engine (Final) config
# ---------------------------------------------------------------------------
@dataclass
class DecompositionConfig:
    """Config for the final impact decomposition."""
    # Layer weights in final decomposition
    w_portable_talent: float = 1.0     # PTS weight (not normalized, additive)
    w_role_utilization: float = 1.0    # RUE weight
    w_archetype_elevation: float = 1.0 # Elevation weight
    w_scheme_amplification: float = 1.0  # Scheme amplification

    # Portability thresholds (delegated to PortabilityConfig, kept for tier compat)
    high_portability: float = 0.70
    low_portability: float = 0.40

    # Tier boundaries (percentile-based)
    tier_boundaries: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "Elite": (90.0, 100.0),
        "All-Star": (75.0, 90.0),
        "Starter": (50.0, 75.0),
        "Rotation": (25.0, 50.0),
        "Fringe": (0.0, 25.0),
    })


DECOMPOSITION = DecompositionConfig()

# ---------------------------------------------------------------------------
# Portable Dimension Definitions — v2.0 (8 independent dimensions)
# ---------------------------------------------------------------------------
PORTABLE_DIMENSIONS = {
    # --- OFFENSIVE DIMENSIONS ---
    "shooting_gravity": {
        "number": 1,
        "domain": "offensive",
        "description": "Shooting efficiency and gravity (3PT volume, efficiency, TS%)",
        "columns": ["TS_PCT", "FG3_PCT", "FG3A_PER36", "CATCH_SHOOT_FG3_PCT", "MOVEMENT_SHOOTER_PCT"],
        "weight_key": "w_dim_shooting_gravity",
    },
    "driving_gravity": {
        "number": 2,
        "domain": "offensive",
        "description": "Rim pressure creation (drives, rim FGA, fouls drawn — NO FT%)",
        "columns": ["DRIVES_PER36", "AT_RIM_FREQ", "FT_RATE", "PAINT_FREQ"],
        "weight_key": "w_dim_driving_gravity",
    },
    "playmaking": {
        "number": 3,
        "domain": "offensive",
        "description": "Pass creation, assist generation, advantage creation",
        "columns": ["AST_PER36", "PLAYMAKING_SCORE", "POTENTIAL_AST_PER36", "SECONDARY_AST_PER36"],
        "weight_key": "w_dim_playmaking",
    },
    "extra_possession_creation": {
        "number": 4,
        "domain": "cross",
        "description": "Rebounding / extra possessions (cross-domain: 40% off, 60% def)",
        "columns": ["OREB_pct", "DREB_pct", "REB_PER36"],
        "weight_key": "w_dim_extra_possession",
    },
    "turnover_control": {
        "number": 7,
        "domain": "offensive",
        "description": "Ball security and turnover avoidance under any role",
        "columns": ["TOV_PCT", "TOV_PER36"],
        "weight_key": "w_dim_turnover_control",
    },
    # --- DEFENSIVE DIMENSIONS ---
    "defensive_playmaking": {
        "number": 5,
        "domain": "defensive",
        "description": "Chaos creation: steals, blocks, deflections, hustle",
        "columns": ["STL_PER100_DEF_POSS", "BLK_PCT", "DEFLECTIONS",
                     "hustle_score", "engagement_score"],
        "weight_key": "w_dim_defensive_playmaking",
    },
    "defensive_impact": {
        "number": 6,
        "domain": "defensive",
        "description": "RAPM-informed defensive impact (on/off, matchup-adjusted)",
        "columns": ["drapm", "DRTG", "d_results_pctl"],
        "weight_key": "w_dim_defensive_impact",
    },
    "defensive_versatility": {
        "number": 8,
        "domain": "defensive",
        "description": "Matchup spectrum, switching, positional coverage",
        "columns": ["switch_score", "versatility_pctl", "assignment_difficulty",
                     "matchup_diversity_pctl"],
        "weight_key": "w_dim_defensive_versatility",
    },
}

# v1.5 backward compat alias (used by some tests)
PORTABLE_SKILL_COMPONENTS = PORTABLE_DIMENSIONS

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def clean_id(val) -> str:
    """Normalize player ID to clean string."""
    import pandas as pd
    if pd.isna(val):
        return "0"
    return str(val).replace(".0", "")
