"""
src/modeling/model_config.py
=============================================================================
BKE v2.6 — Centralized Configuration for the Portable Talent vs
             Role-Dependent Impact Decomposition Engine.

v2.6 corrections over v2.5:
  - Position-conditional z-scores for biased dimensions (Fix #4)
  - Self-Creation dimension added (Dim 9) — off-dribble, pull-up, usage-based
  - Basketball-informed dimension weights replace equal 0.125 weighting
  - Portability index rewritten: real structural components, not proxies
  - Big-man bias eliminated: shooting/playmaking/creation weighted higher,
    rebounding/rim-protection weighted lower to reflect true portability

Prior versions:
  v2.5: Archetype-conditional neutralization, expanded hustle stats
  v2.0: Z-score aggregation, variance-based portability, 8 dimensions
  v1.5: Initial percentile-based pipeline

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

# Output files — v2.6
BKE_OUTPUT_PARQUET = os.path.join(OUTPUT_DIR, "bke_v26_decomposition.parquet")
BKE_OUTPUT_CSV = os.path.join(OUTPUT_DIR, "bke_v26_decomposition.csv")
BKE_REPORT_JSON = os.path.join(OUTPUT_DIR, "bke_v26_report.json")

# Hustle stats (raw tracking data for MF-4)
TRACKING_DIR = "data/tracking"

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

    v2.6 structure:
      Layer 1 = 25% RAPM Backbone + 20% Playtype Efficiency + 55% Dimension Model (1C)
      Layer 1C = 9 basketball-informed dimensions, weighted by portability.

    v2.6 dimension weighting philosophy:
      Universally portable skills (shooting, playmaking, self-creation) get
      higher weights because they transfer across ANY team/system/context.
      Position-dominated skills (rebounding, rim protection) get lower weights
      because they are strong predictors of POSITION, not TALENT ABOVE position.

    v2.6 position-conditional z-scores:
      Dimensions that heavily correlate with position (driving, rebounding,
      blocks, rim protection, versatility) compute z-scores WITHIN position
      group, not league-wide. This prevents big-man inflation.
    """
    # --- Layer 1 high-level weights (sum to 1.0) ---
    w_rapm_backbone: float = 0.25       # Layer 1A: RAPM impact
    w_playtype_efficiency: float = 0.20  # Layer 1B: Playtype efficiency composite
    w_dimension_model: float = 0.55      # Layer 1C: 9-dimension portable model

    # --- Layer 1C: 9 dimension weights (basketball-informed, sum to 1.0) ---
    # UNIVERSALLY PORTABLE SKILLS (higher weight = 42%)
    w_dim_shooting_gravity: float = 0.14        # Dim 1: Shooting gravity
    w_dim_playmaking: float = 0.14              # Dim 3: Playmaking creation+pressure
    w_dim_self_creation: float = 0.14           # Dim 9: Self-creation (NEW v2.6)

    # VALUABLE BUT POSITIONAL (medium weight = 32%)
    w_dim_defensive_versatility: float = 0.12   # Dim 8: Matchup switching
    w_dim_turnover_control: float = 0.10        # Dim 7: Ball security
    w_dim_defensive_impact: float = 0.10        # Dim 6: RAPM defensive impact

    # POSITION-DEPENDENT (lower weight = 26%)
    w_dim_driving_gravity: float = 0.10         # Dim 2: Driving / rim pressure
    w_dim_extra_possession: float = 0.08        # Dim 4: Rebounding
    w_dim_defensive_playmaking: float = 0.08    # Dim 5: Blocks/deflections/hustle

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
# Archetype-Conditional Neutralization config (v2.5 Fix #3)
# ---------------------------------------------------------------------------
@dataclass
class NeutralizationConfig:
    """
    Configuration for archetype-conditional neutralization of Layer 1C dimensions.

    v2.6 neutralizes all portable dimensions by subtracting the archetype
    expected output, so that Layer 1C measures ability ABOVE what's expected
    for the player's role — not raw observed production.

    Neutralized_z = Observed_z - E[z | archetype]

    v2.6: Position-conditional z-scores REPLACE neutralization for
    position-dominated dimensions. Neutralization is still applied to
    league-wide z-score dimensions for archetype adjustment.
    """
    # Minimum archetype cohort size for reliable expected value
    min_cohort_for_neutralization: int = 8

    # If cohort too small, shrink toward league mean (0.0) instead
    small_cohort_shrinkage: float = 0.50

    # Dimensions to neutralize (league-z dimensions only in v2.6)
    neutralize_dimensions: list = field(default_factory=lambda: [
        "dim_shooting_gravity_z",
        "dim_playmaking_creation_z",
        "dim_self_creation_z",
        "dim_turnover_control_z",
    ])

    # Dimensions using position-conditional z-scores (v2.6) — NOT neutralized
    # because positional z already removes expected positional output
    position_z_dimensions: list = field(default_factory=lambda: [
        "dim_driving_gravity_z",
        "dim_extra_possession_z",
        "dim_defensive_playmaking_z",
        "dim_defensive_impact_z",
        "dim_defensive_versatility_z",
    ])

    # Dimensions that are NOT neutralized (already context-neutral by construction)
    skip_neutralization: list = field(default_factory=lambda: [
        "dim_defensive_impact_z",       # Uses position-conditional z (v2.6)
        "dim_defensive_versatility_z",  # Uses position-conditional z (v2.6)
        "dim_driving_gravity_z",        # Uses position-conditional z (v2.6)
        "dim_extra_possession_z",       # Uses position-conditional z (v2.6)
        "dim_defensive_playmaking_z",   # Uses position-conditional z (v2.6)
    ])


NEUTRALIZATION = NeutralizationConfig()

# ---------------------------------------------------------------------------
# True Portability Index config (v2.0 Fix #2)
# ---------------------------------------------------------------------------
@dataclass
class PortabilityConfig:
    """
    Configuration for structural portability measurement (v2.6).

    v2.6 replaces proxy approximations with structural measurements:
      Portability = How transferable is this player across team contexts?

    Components (v2.6):
      1. Dimensional Breadth (35%) — Breadth of above-average skills
      2. Universal Skill Presence (25%) — Universally portable skills
      3. Two-Way Balance (20%) — Offense + defense balance
      4. Scheme Independence (20%) — Low dependence on specific system
    """
    # Weights for portability components
    w_dimensional_breadth: float = 0.35
    w_universal_skill: float = 0.25
    w_two_way_balance: float = 0.20
    w_scheme_independence: float = 0.20

    # Universal skill dimensions (from PORTABLE_DIMENSIONS)
    universal_dims: list = field(default_factory=lambda: [
        "dim_shooting_gravity_z",
        "dim_playmaking_creation_z",
        "dim_self_creation_z",
    ])

    # All 9 dims for breadth calculation
    all_dims: list = field(default_factory=lambda: [
        "dim_shooting_gravity_z",
        "dim_driving_gravity_z",
        "dim_playmaking_creation_z",
        "dim_extra_possession_z",
        "dim_turnover_control_z",
        "dim_defensive_playmaking_z",
        "dim_defensive_impact_z",
        "dim_defensive_versatility_z",
        "dim_self_creation_z",
    ])

    # Thresholds for classification
    high_portability: float = 0.70   # Scalable star
    low_portability: float = 0.40    # System-amplified


PORTABILITY = PortabilityConfig()

# ---------------------------------------------------------------------------
# Decomposition Engine (Final) config
# ---------------------------------------------------------------------------
@dataclass
class DecompositionConfig:
    """
    Config for the final impact decomposition (v2.6).

    v2.6 rebalances weights so portable talent dominates:
      - Portable talent is the PRIMARY signal (45%)
      - Role utilization is secondary (20%)
      - Elevation and scheme are minor adjustments (20%, 15%)
    """
    # Layer weights in final decomposition (will be normalized to sum=1)
    w_portable_talent: float = 2.25       # 45% — core transferable ability
    w_role_utilization: float = 1.0       # 20% — role-specific efficiency
    w_archetype_elevation: float = 1.0    # 20% — outperforming archetype
    w_scheme_amplification: float = 0.75  # 15% — scheme stability (minor)

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
# Portable Dimension Definitions — v2.6 (9 dimensions, position-conditional z)
# ---------------------------------------------------------------------------
PORTABLE_DIMENSIONS = {
    # --- UNIVERSALLY PORTABLE (league z-scores + archetype neutralization) ---
    "shooting_gravity": {
        "number": 1,
        "domain": "offensive",
        "z_mode": "league",   # v2.6: league z-score (position-neutral skill)
        "description": "Shooting efficiency and gravity (3PT volume, efficiency, TS%)",
        "columns": ["TS_PCT", "FG3_PCT", "FG3A_PER36", "CATCH_SHOOT_FG3_PCT", "MOVEMENT_SHOOTER_PCT"],
        "weight_key": "w_dim_shooting_gravity",
    },
    "playmaking_creation": {
        "number": 3,
        "domain": "offensive",
        "z_mode": "league",   # v2.6: league z-score (universally portable)
        "description": "Playmaking: creation (assists) + pressure (collapse/rotation forcing)",
        "columns": [
            "AST_PER36", "PLAYMAKING_SCORE",
            "POTENTIAL_AST_PER36", "SECONDARY_AST_PER36",
            "DRIVE_AST_RATIO", "PASSES_MADE_PER36",
        ],
        "weight_key": "w_dim_playmaking",
    },
    "self_creation": {
        "number": 9,
        "domain": "offensive",
        "z_mode": "league",   # v2.6: NEW dimension — off-dribble shot creation
        "description": "Self-created offense: pull-up shooting, off-dribble creation, ball dominance",
        "columns": [
            "PULL_UP_FGA_PER36", "PULL_UP_EFG_PCT",
            "AVG_DRIB_PER_TOUCH", "ON_BALL_CREATION",
            "BALL_DOMINANT_PCT",
        ],
        "weight_key": "w_dim_self_creation",
    },
    "turnover_control": {
        "number": 7,
        "domain": "offensive",
        "z_mode": "league",   # v2.6: league z-score (but usage-adjusted metrics)
        "description": "Ball security and turnover avoidance under any role (MF-1 enhanced)",
        "columns": ["TOV_PCT", "TOV_PER36", "TOV_PER_TOUCH", "DRIVE_TOV_RATE"],
        "weight_key": "w_dim_turnover_control",
    },

    # --- POSITION-CONDITIONAL (position z-scores) ---
    "driving_gravity": {
        "number": 2,
        "domain": "offensive",
        "z_mode": "position",  # v2.6: position z-score (rim finishing is positional)
        "description": "Rim pressure creation (drives, rim FGA, fouls drawn — NO FT%)",
        "columns": ["DRIVES_PER36", "AT_RIM_FREQ", "FT_RATE", "PAINT_FREQ"],
        "weight_key": "w_dim_driving_gravity",
    },
    "extra_possession_creation": {
        "number": 4,
        "domain": "cross",
        "z_mode": "position",  # v2.6: position z-score (rebounding is heavily positional)
        "description": "Rebounding / extra possessions (cross-domain: 40% off, 60% def)",
        "columns": ["OREB_pct", "DREB_pct", "REB_PER36"],
        "weight_key": "w_dim_extra_possession",
    },
    "defensive_playmaking": {
        "number": 5,
        "domain": "defensive",
        "z_mode": "position",  # v2.6: position z-score (BLK% dominated by centers)
        "description": "Chaos creation: steals, blocks, deflections, hustle, charges, loose balls",
        "columns": ["STL_PER100_DEF_POSS", "BLK_PCT", "DEFLECTIONS",
                     "hustle_score", "engagement_score",
                     "CHARGES_DRAWN", "DEF_LOOSE_BALLS_RECOVERED"],
        "weight_key": "w_dim_defensive_playmaking",
    },
    "defensive_impact": {
        "number": 6,
        "domain": "defensive",
        "z_mode": "position",  # v2.6: position z-score (DRAPM favors rim protectors)
        "description": "RAPM-informed defensive impact (on/off, matchup-adjusted)",
        "columns": ["drapm", "DRTG", "d_results_pctl"],
        "weight_key": "w_dim_defensive_impact",
    },
    "defensive_versatility": {
        "number": 8,
        "domain": "defensive",
        "z_mode": "position",  # v2.6: position z-score (fix Centers +0.86 vs Guards -0.66)
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
