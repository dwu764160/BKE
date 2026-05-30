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
AGGREGATE_DIR = ROOT_DIR / "aggregate"

TEAM_FEATURES_PATH = PROCESSED_DIR / "player_eval" / "team_feature_aggregation.parquet"
PLAYER_PROFILES_PATH = PROCESSED_DIR / "player_eval" / "player_impact_profiles.parquet"
POSITION_ESTIMATES_PATH = PROCESSED_DIR / "player_position_estimates.parquet"
METRICS_LINEUPS_PATH = PROCESSED_DIR / "metrics_lineups.parquet"
CLUTCH_STATS_ALL_PATH = HISTORICAL_DIR / "player_clutch_stats_all.parquet"
TEAMS_PATH = HISTORICAL_DIR / "teams.parquet"
PROFILE_AGGREGATE_PATH = AGGREGATE_DIR / "player_profile_aggregate.parquet"

SIMULATION_PROCESSED_DIR = PROCESSED_DIR / "simulation"
SIMULATION_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

STEP2_LINEUP_PROFILES_PATH = SIMULATION_PROCESSED_DIR / "simulation_step2_lineup_profiles.parquet"
STEP2_LINEUP_REPORT_PATH = REPORTS_DIR / "simulation_step2_lineup_profiles.json"
STEP2_VALIDATION_PATH = REPORTS_DIR / "simulation_step2_validation.json"
STEP1_PLAYER_SEASON_STATS_PATH = SIMULATION_PROCESSED_DIR / "simulation_step1_player_season_stats.parquet"
STEP1_PLAYER_GAME_SAMPLES_PATH = SIMULATION_PROCESSED_DIR / "simulation_step1_player_game_samples.parquet"
STEP1_SINGLE_GAME_REPORT_PATH = REPORTS_DIR / "simulation_single_game.json"

# Forecast mode paths
FORECAST_DIR = PROCESSED_DIR / "forecast"
FORECAST_DIR.mkdir(parents=True, exist_ok=True)
FORECAST_TEAM_FEATURES_PATH = FORECAST_DIR / "projected_team_features.parquet"
YTD_RATINGS_PATH = FORECAST_DIR / "team_ratings_ytd.parquet"
FORECAST_PLAYER_PROFILES_PATH = FORECAST_DIR / "projected_player_profiles.parquet"
FORECAST_SEASON_RESULTS_PATH = REPORTS_DIR / "forecast_season_results.json"
FORECAST_LINEUP_REPORT_PATH = REPORTS_DIR / "forecast_lineup_profiles.json"
FORECAST_LINEUP_PROFILES_PATH = SIMULATION_PROCESSED_DIR / "forecast_step2_lineup_profiles.parquet"
FORECAST_VALIDATION_PATH = REPORTS_DIR / "forecast_step2_validation.json"
FORECAST_PLAYER_SEASON_STATS_PATH = SIMULATION_PROCESSED_DIR / "forecast_step1_player_season_stats.parquet"
FORECAST_PLAYER_GAME_SAMPLES_PATH = SIMULATION_PROCESSED_DIR / "forecast_step1_player_game_samples.parquet"
FORECAST_SINGLE_GAME_REPORT_PATH = REPORTS_DIR / "forecast_single_game.json"

# Ensure report output directory exists when simulation scripts run standalone.
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Core simulation constants
SIGMA_LEAGUE = 3.0
HOME_COURT_ADVANTAGE = 2.0  # legacy flat default; overridden per-team via REST_HCA_COEFFICIENTS
SEASON_SIMULATIONS = 10_000
SIMULATION_RANDOM_SEED = 42

# ── Fitted rest / HCA coefficients (from src/data_compute/fit_rest_hca_coefficients.py) ──
# Loaded from reports/rest_hca_coefficients.json if present; otherwise falls back to
# the flat defaults above (HOME_COURT_ADVANTAGE, no B2B/rest adjustments).
#
# Application: for a game with home team H, away team A, on date D:
#   mu_margin = (mu_home + adj_home) - (mu_away + adj_away) + team_hca(H)
#   adj_home  = B2B_PENALTY_HOME * is_b2b_home + REST_DAY_BONUS_HOME * days_rest_home
#   adj_away  = (independent away-team adjustment; sign already encoded in fitted coefs)
import json as _json

REST_HCA_COEFFICIENTS_PATH = REPORTS_DIR / "rest_hca_coefficients.json"
TEAM_PACE_PATH = REPORTS_DIR / "team_pace.json"


def _load_rest_hca_coefficients() -> dict:
    """Load fitted rest+HCA coefficients with safe fallback to flat defaults."""
    if not REST_HCA_COEFFICIENTS_PATH.exists():
        return {
            "league_avg_hca": HOME_COURT_ADVANTAGE,
            "b2b_penalty_home": 0.0,
            "b2b_penalty_away": 0.0,
            "rest_day_bonus_home": 0.0,
            "rest_day_bonus_away": 0.0,
            "team_hca": {},
            "loaded": False,
        }
    try:
        data = _json.loads(REST_HCA_COEFFICIENTS_PATH.read_text())
        data["loaded"] = True
        return data
    except Exception:
        return {
            "league_avg_hca": HOME_COURT_ADVANTAGE,
            "b2b_penalty_home": 0.0,
            "b2b_penalty_away": 0.0,
            "rest_day_bonus_home": 0.0,
            "rest_day_bonus_away": 0.0,
            "team_hca": {},
            "loaded": False,
        }


def _load_team_pace() -> dict:
    """Load fitted per-team-season pace with safe fallback."""
    if not TEAM_PACE_PATH.exists():
        return {
            "league_avg_pace": DEFAULT_PACE_PER_48,
            "team_pace": {},
            "loaded": False,
        }
    try:
        data = _json.loads(TEAM_PACE_PATH.read_text())
        data["loaded"] = True
        return data
    except Exception:
        return {
            "league_avg_pace": DEFAULT_PACE_PER_48,
            "team_pace": {},
            "loaded": False,
        }


REST_HCA = _load_rest_hca_coefficients()
TEAM_PACE = _load_team_pace()

# Convenience extracts (module-level, used by validate_forecast.py and team aggregation)
B2B_PENALTY_HOME = float(REST_HCA.get("b2b_penalty_home", 0.0))
B2B_PENALTY_AWAY = float(REST_HCA.get("b2b_penalty_away", 0.0))
REST_DAY_BONUS_HOME = float(REST_HCA.get("rest_day_bonus_home", 0.0))
REST_DAY_BONUS_AWAY = float(REST_HCA.get("rest_day_bonus_away", 0.0))
LEAGUE_AVG_HCA_FITTED = float(REST_HCA.get("league_avg_hca", HOME_COURT_ADVANTAGE))
TEAM_HCA: dict = dict(REST_HCA.get("team_hca", {}))

# 3-in-4 penalty (hardcoded prior; not yet fitted from data)
# Sign convention: home perspective — negative hurts home team, positive hurts away.
HOME_3IN4_PENALTY = -1.0   # pts/100 extra when home team plays 3rd game in 4 days
AWAY_3IN4_PENALTY =  1.0   # pts/100 extra when away team plays 3rd game in 4 days


def get_team_hca(team_abbr: str) -> float:
    """Return team-specific HCA (shrunk toward league mean for low-sample teams).

    Falls back to LEAGUE_AVG_HCA_FITTED, then HOME_COURT_ADVANTAGE if no fit available.
    """
    if team_abbr in TEAM_HCA:
        return float(TEAM_HCA[team_abbr])
    return LEAGUE_AVG_HCA_FITTED if REST_HCA.get("loaded") else HOME_COURT_ADVANTAGE


def get_team_pace(season: str, team_abbr: str) -> float:
    """Return team-season pace, fallback to season league avg, then DEFAULT_PACE_PER_48."""
    season_paces = TEAM_PACE.get("team_pace", {}).get(str(season), {})
    if team_abbr in season_paces:
        return float(season_paces[team_abbr])
    season_avg = TEAM_PACE.get("league_avg_pace_per_season", {}).get(str(season))
    if season_avg is not None:
        return float(season_avg)
    return float(TEAM_PACE.get("league_avg_pace", DEFAULT_PACE_PER_48))

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
STEP2_CONTINUITY_STARTER_BONUS = 0.08
STEP2_CONTINUITY_ROTATION_BONUS = 0.04

# Clutch core constraint
STEP2_CLUTCH_CANDIDATE_SIZE = 7
STEP2_CLUTCH_MIN_STARTERS = 3
STEP2_CLUTCH_MAX_SWAPS = 2
STEP2_CANDIDATE_LIMIT = 9
STEP2_CANDIDATE_PER_ROLE = 2
STEP2_CANDIDATE_PER_BAND = 2

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

# Starter score weights: 0.80 * M + 0.12 * I + 0.04 * Pos + 0.04 * C
# Tuned in Step 0a (2026-05-30): walk-forward game-by-game validation vs actual
# pbp opening-tip lineups independently re-selected this minutes-heavy weighting
# (config 'starter_m080') for every eval season. Starter hit-rate 0.621 -> 0.647
# walk-forward (+2.5pp), clutch overlap not regressed, minutes corr unchanged.
# Rationale: actual NBA starters are overwhelmingly the highest-minute players;
# the prior 0.50/0.25/0.10/0.15 split diluted that signal. Backtest mode only —
# forecast-mode starter scoring uses separate hardcoded weights.
# Validation: reports/lineup_projection_validation.json;
# docs/findings/lineup_projection_tuning_2026-05-30.md.
STARTER_WEIGHT_M = 0.80
STARTER_WEIGHT_I = 0.12
STARTER_WEIGHT_POS = 0.04
STARTER_WEIGHT_C = 0.04

# Rotation model: mu_rotation = alpha * stagger + (1-alpha) * bench
ROTATION_ALPHA = 0.30
STAGGER_MINUTES_THRESHOLD = 24.0

# Step 2 fit heuristics
STEP2_FIT_CREATOR_BONUS = 0.08
STEP2_FIT_SPACING_BONUS = 0.06
STEP2_FIT_POA_BONUS = 0.05
STEP2_FIT_RIM_BONUS = 0.05
STEP2_FIT_STAGGER_BONUS = 0.03

# Simulation bonus controls
LINEUP_TEAM_BONUS_CLIP = 0.30
LINEUP_STARTER_BONUS_SCALE = 0.20
LINEUP_ROTATION_BONUS_SCALE = 0.15
LINEUP_CONTINUITY_BONUS_SCALE = 0.08
LINEUP_FIT_BONUS_SCALE = 0.08
LINEUP_CLUTCH_BONUS_SCALE = 0.25
CLUTCH_MARGIN_TRIGGER = 7.5
MATCHUP_SPREAD_BONUS_CLIP = 0.15
MATCHUP_PPP_DELTA_CLIP = 0.035
MATCHUP_TEAM_INTERACTION_SCALE = 0.15
PPP_CONTEXT_BONUS_SCALE = 0.0
PLAYER_GAME_TOTAL_MINUTES = 240.0
PLAYER_GAME_ROTATION_SIZE = 10
PLAYER_GAME_MIN_ACTIVE = 8
PLAYER_GAME_MINUTES_STARTER_BONUS = 0.14
PLAYER_GAME_MINUTES_CLUTCH_BONUS = 0.05
PLAYER_GAME_MINUTES_BENCH_PENALTY = 0.08
PLAYER_GAME_ARCHETYPE_EFFECT_SCALE = 0.60
PLAYER_GAME_STARTER_USAGE_BOOST = 1.18
PLAYER_GAME_CLUTCH_USAGE_BOOST = 1.06
PLAYER_GAME_USAGE_CAP = 0.38
PLAYER_GAME_DIRICHLET_SCALE = 85.0
PLAYER_GAME_BETA_CONCENTRATION = 42.0
PLAYER_GAME_BLOWOUT_MARGIN = 15.0
PLAYER_GAME_BLOWOUT_STARTER_PENALTY = 0.10

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
