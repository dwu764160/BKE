"""
src/simulation/simulation_config.py
=============================================================================
Centralized configuration and paths for the simulation module.
=============================================================================
"""

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
HISTORICAL_DIR = DATA_DIR / "historical"
REPORTS_DIR = ROOT_DIR / "reports"

TEAM_FEATURES_PATH = PROCESSED_DIR / "player_eval" / "team_feature_aggregation.parquet"
PLAYER_PROFILES_PATH = PROCESSED_DIR / "player_eval" / "player_impact_profiles.parquet"
POSITION_ESTIMATES_PATH = PROCESSED_DIR / "player_position_estimates.parquet"
METRICS_LINEUPS_PATH = PROCESSED_DIR / "metrics_lineups.parquet"
CLUTCH_STATS_ALL_PATH = HISTORICAL_DIR / "player_clutch_stats_all.parquet"
TEAMS_PATH = HISTORICAL_DIR / "teams.parquet"

SIMULATION_PROCESSED_DIR = PROCESSED_DIR / "simulation"
SIMULATION_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

STEP2_LINEUP_PROFILES_PATH = SIMULATION_PROCESSED_DIR / "simulation_step2_lineup_profiles.parquet"
STEP2_LINEUP_REPORT_PATH = REPORTS_DIR / "simulation_step2_lineup_profiles.json"
STEP2_VALIDATION_PATH = REPORTS_DIR / "simulation_step2_validation.json"

# Forecast mode paths
FORECAST_DIR = PROCESSED_DIR / "forecast"
FORECAST_DIR.mkdir(parents=True, exist_ok=True)
FORECAST_TEAM_FEATURES_PATH = FORECAST_DIR / "projected_team_features.parquet"
FORECAST_PLAYER_PROFILES_PATH = FORECAST_DIR / "projected_player_profiles.parquet"
FORECAST_SEASON_RESULTS_PATH = REPORTS_DIR / "forecast_season_results.json"
FORECAST_LINEUP_REPORT_PATH = REPORTS_DIR / "forecast_lineup_profiles.json"
FORECAST_LINEUP_PROFILES_PATH = SIMULATION_PROCESSED_DIR / "forecast_step2_lineup_profiles.parquet"
FORECAST_VALIDATION_PATH = REPORTS_DIR / "forecast_step2_validation.json"

# Ensure report output directory exists when simulation scripts run standalone.
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Core simulation constants
SIGMA_LEAGUE = 3.0
HOME_COURT_ADVANTAGE = 2.0
SEASON_SIMULATIONS = 10_000
SIMULATION_RANDOM_SEED = 42

# Parallel simulation models
SIM_MODEL_MARGIN = "margin"
SIM_MODEL_PPP = "ppp"
SIM_MODELS = (SIM_MODEL_MARGIN, SIM_MODEL_PPP)

# PPP / possession-based model constants
LEAGUE_AVG_PPP = 1.14
DEFAULT_PACE_PER_48 = 100.0
# Pace is noisy year-to-year, so regress aggressively toward league mean.
PACE_PRIOR_SEASON_WEIGHT = 0.40
PACE_REGRESSION_WEIGHT = 0.60
POSSESSION_FTA_WEIGHT = 0.44

# PPP calibration knobs
PPP_IMPACT_SCALE_PER100 = 100.0
PPP_OFF_DEF_BLEND_WEIGHT = 0.50
PPP_SIGMA_POSSESSION_EXPONENT = 0.50
PPP_MIN = 0.85
PPP_MAX = 1.35

# Rank-based postseason cutoffs (per conference)
DIRECT_PLAYOFF_RANK = 6
PLAY_IN_RANK = 10

# Step 2 lineup model constants
LINEUP_SIZE = 5
MIN_MPG_FOR_POOL = 3.0

# Step 2 baseline structural checks
STEP2_STARTER_REQUIRE_TRUE_BIG = True
STEP2_STARTER_TRUE_BIG_BANDS = ("Forward-Center", "Center")

# Starter penalty for low projected minutes (soft penalty, not exclusion)
STEP2_STARTER_LOW_MINUTES_THRESHOLD = 24.0
STEP2_STARTER_LOW_MINUTES_PENALTY = 1.10

# Step 2 v1 optimization toggles
STEP2_ENABLE_CONTINUITY_PRIOR = True
STEP2_ENABLE_CLUTCH_CORE_CONSTRAINT = True
STEP2_ENABLE_ROTATION_REGIME_MODEL = False

# Continuity prior: continuity_factor = 1 - kappa * (1 - R)
STEP2_CONTINUITY_KAPPA = 0.35
STEP2_CONTINUITY_DEFAULT_RETURNING = 1.0

# Clutch core constraint
STEP2_CLUTCH_CANDIDATE_SIZE = 7
STEP2_CLUTCH_MIN_STARTERS = 3
STEP2_CLUTCH_MAX_SWAPS = 2

# Rotation regime model
STEP2_REGIME_STAR_COEF = 0.8
STEP2_REGIME_BENCH_COEF = 0.6
STEP2_ROT_STAGGER_START_W = 0.65
STEP2_ROT_STAGGER_BENCH_W = 0.35
STEP2_ROT_BENCH_START_W = 0.30
STEP2_ROT_BENCH_BENCH_W = 0.70
STEP2_ROT_SIGMA_BASE = 0.60
STEP2_ROT_SIGMA_BENCH_W = 0.40

# Clutch score weights: 0.80 * C + 0.20 * (0.65 * I + 0.35 * M)
CLUTCH_WEIGHT_C = 0.80
CLUTCH_WEIGHT_TALENT = 0.20
CLUTCH_WEIGHT_I = 0.65
CLUTCH_WEIGHT_M = 0.35

# Starter score weights: 0.50 * M + 0.25 * I + 0.10 * Pos + 0.15 * C
STARTER_WEIGHT_M = 0.50
STARTER_WEIGHT_I = 0.25
STARTER_WEIGHT_POS = 0.10
STARTER_WEIGHT_C = 0.15

# Rotation model: mu_rotation = alpha * stagger + (1-alpha) * bench
ROTATION_ALPHA = 0.30
STAGGER_MINUTES_THRESHOLD = 24.0

# Volatility proxy from impact_stability
PLAYER_VOL_BASE = 3.5
PLAYER_VOL_STABILITY_SCALE = 10.0
PLAYER_VOL_STABILITY_CENTER = 0.60
PLAYER_VOL_FLOOR = 1.5
PLAYER_VOL_CEILING = 6.5

# Step 2 feature columns
STEP2_IMPACT_COLUMN = "impact_total_impact"
STEP2_MINUTES_COLUMN = "mpg"
STEP2_STABILITY_COLUMN = "impact_stability"
