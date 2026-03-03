from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT_DIR / "reports"
HISTORICAL_DIR = DATA_DIR / "historical"

PLAYER_EVAL_DATA_DIR = PROCESSED_DIR / "player_eval"
PLAYER_EVAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Profile Aggregate outputs
AGGREGATE_DIR = ROOT_DIR / "aggregate"
AGGREGATE_DIR.mkdir(parents=True, exist_ok=True)
PROFILE_AGGREGATE_PATH = AGGREGATE_DIR / "player_profile_aggregate.parquet"
AGGREGATE_VALIDATION_REPORT = REPORTS_DIR / "profile_aggregate_validation.json"

# Core inputs
BKE_DECOMP_PATH = PROCESSED_DIR / "bke" / "bke_v28_decomposition.parquet"
BKE_SCORES_PATH = PROCESSED_DIR / "bke" / "BKE_Scores_v27.json"
PLAYER_ARCHETYPES_PATH = PROCESSED_DIR / "player_archetypes.parquet"
DEF_ARCHETYPES_PATH = PROCESSED_DIR / "defensive_archetypes_v2.parquet"
POSITION_ESTIMATES_PATH = PROCESSED_DIR / "player_position_estimates.parquet"
COMPLETE_STATS_PATH = HISTORICAL_DIR / "complete_player_season_stats.parquet"
METRICS_LINEAR_PATH = PROCESSED_DIR / "metrics_linear.parquet"
PLAYERS_META_PATH = HISTORICAL_DIR / "players.parquet"
GAME_LOGS_PATH = HISTORICAL_DIR / "final_player_game_logs.parquet"
XRAPM_V2_PATH = PROCESSED_DIR / "player_xrapm_v2.parquet"
XRAPM_V1_PATH = PROCESSED_DIR / "player_xrapm.parquet"
DARKO_DIR = HISTORICAL_DIR / "darko" / "raw"

# Stability / diagnostics
BKE_V29_PLAYER_DIAGNOSTIC_PATH = REPORTS_DIR / "bke_v29_player_diagnostic_report.json"
DBKE_V30_SHRINKAGE_PATH = REPORTS_DIR / "dbke_v30_defense_shrinkage.json"

# Step 1 outputs
PLAYER_PROFILES_PARQUET = PLAYER_EVAL_DATA_DIR / "player_impact_profiles.parquet"
PLAYER_PROFILES_PKL = PLAYER_EVAL_DATA_DIR / "player_profiles_season.pkl"
STEP1_VALIDATION_REPORT = REPORTS_DIR / "player_eval_step1_validation.json"

# Step 2 outputs
MINUTE_MODEL_PATH = PLAYER_EVAL_DATA_DIR / "minute_model_v2.pkl"
MINUTE_PREDICTIONS_PATH = PLAYER_EVAL_DATA_DIR / "minute_model_predictions_v2.parquet"
STEP2_VALIDATION_REPORT = REPORTS_DIR / "player_eval_step2_minute_model_validation.json"

# Modeling constants
RANDOM_SEED = 42
MINUTE_SHARE_CAP = 0.22
MINUTE_SHARE_FLOOR = 0.0
HIGH_USAGE_QUANTILE = 0.75
