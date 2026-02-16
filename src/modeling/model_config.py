"""
src/modeling/model_config.py
=============================================================================
BKE v1.5 — Centralized Configuration for the Portable Talent vs
             Role-Dependent Impact Decomposition Engine.

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

# Output files
BKE_OUTPUT_PARQUET = os.path.join(OUTPUT_DIR, "bke_v15_decomposition.parquet")
BKE_OUTPUT_CSV = os.path.join(OUTPUT_DIR, "bke_v15_decomposition.csv")
BKE_REPORT_JSON = os.path.join(OUTPUT_DIR, "bke_v15_report.json")

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
# Portable Talent Score (Layer 1) weights
# ---------------------------------------------------------------------------
@dataclass
class PortableTalentConfig:
    """Weights for the Portable Talent Score computation."""
    # Relative weights for PTS components (sum to 1.0)
    w_rapm: float = 0.35          # Adjusted RAPM impact
    w_shooting: float = 0.15      # Shooting gravity / efficiency
    w_passing: float = 0.12       # Passing efficiency / creation
    w_rim_protection: float = 0.10  # Rim protection / interior D
    w_defensive_versatility: float = 0.10  # Defensive versatility
    w_turnover_control: float = 0.08  # Turnover avoidance
    w_rebounding: float = 0.05   # Rebounding
    w_stability: float = 0.05    # Stability-adjusted impact

    # Luck adjustment parameters
    shooting_regression_rate: float = 0.40  # How much to regress 3PT luck
    opponent_3pt_regression: float = 0.50   # Opponent 3PT variance regression
    ft_regression_rate: float = 0.30        # FT variance regression

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
    min_playtype_poss: int = 25

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

    # Portability ratio thresholds for interpretation
    high_portability: float = 0.70  # Scalable star
    low_portability: float = 0.40   # System-amplified

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
# Portable Skill Component definitions
# ---------------------------------------------------------------------------
# Each skill component maps to data columns and computation logic
PORTABLE_SKILL_COMPONENTS = {
    "shooting_gravity": {
        "description": "Shooting efficiency and gravity",
        "columns": ["TS_PCT", "EFG_PCT", "FG3_PCT", "FG3A_PER36"],
        "weight": PORTABLE_TALENT.w_shooting,
    },
    "rim_protection": {
        "description": "Interior defense and rim protection",
        "columns": ["BLK_PER36", "rim_protection_index_pctl"],
        "weight": PORTABLE_TALENT.w_rim_protection,
    },
    "passing_efficiency": {
        "description": "Passing creation and assist efficiency",
        "columns": ["AST_PER36", "PLAYMAKING_SCORE", "TOV_PCT"],
        "weight": PORTABLE_TALENT.w_passing,
    },
    "defensive_versatility": {
        "description": "Defensive versatility across matchups",
        "columns": ["switch_score", "versatility_pctl", "assignment_difficulty"],
        "weight": PORTABLE_TALENT.w_defensive_versatility,
    },
    "turnover_control": {
        "description": "Ball security and turnover avoidance",
        "columns": ["TOV_PCT", "TOV_PER36"],
        "weight": PORTABLE_TALENT.w_turnover_control,
    },
    "rebounding": {
        "description": "Rebounding contribution",
        "columns": ["REB_PER36", "OREB_pct", "DREB_pct"],
        "weight": PORTABLE_TALENT.w_rebounding,
    },
}

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def clean_id(val) -> str:
    """Normalize player ID to clean string."""
    import pandas as pd
    if pd.isna(val):
        return "0"
    return str(val).replace(".0", "")
