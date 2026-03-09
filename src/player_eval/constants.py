"""
src/player_eval/constants.py
=============================================================================
Centralized constants, paths, and thresholds for the Player Evaluation
Component (PEC) pipeline. All configurable parameters for player impact
profiles, team feature aggregation, and forecast projection live here.
=============================================================================
"""
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
CLUTCH_STATS_ALL_PATH = HISTORICAL_DIR / "player_clutch_stats_all.parquet"
CLUTCH_STATS_GLOB = "player_clutch_stats_*.parquet"
XRAPM_V2_PATH = PROCESSED_DIR / "player_xrapm_v2.parquet"
XRAPM_V1_PATH = PROCESSED_DIR / "player_xrapm.parquet"
DARKO_DIR = HISTORICAL_DIR / "darko" / "raw"
PLAYER_DRAFT_HISTORY_PATH = HISTORICAL_DIR / "player_draft_history.parquet"
DRAFT_COMMONPLAYERINFO_CACHE_PATH = HISTORICAL_DIR / "player_draft_commonplayerinfo_cache.parquet"

# Player-team stint outputs (derived from game logs)
PLAYER_TEAM_STINTS_PATH = PROCESSED_DIR / "player_team_stints.parquet"
PLAYER_TEAM_STINTS_REPORT = REPORTS_DIR / "player_team_stints_report.json"

# Preseason roster snapshots (Option B forecast mapping)
PRESEASON_ROSTERS_PATH = HISTORICAL_DIR / "preseason_rosters.parquet"
PRESEASON_ROSTERS_REPORT = REPORTS_DIR / "preseason_rosters_report.json"
PRESEASON_ROSTERS_DIR = HISTORICAL_DIR / "preseason_rosters"
PRESEASON_ROSTERS_DIR.mkdir(parents=True, exist_ok=True)

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

# Step 3 outputs
TEAM_FEATURES_PATH = PLAYER_EVAL_DATA_DIR / "team_feature_aggregation.parquet"
STEP3_VALIDATION_REPORT = REPORTS_DIR / "player_eval_step3_team_features_validation.json"

# Step 3 — Team Feature Aggregation constants
# ─── Modifier philosophy ────────────────────────────────────────────
# Talent backbone (TEAM_SCALE × talent_base) MUST dominate net rating.
# Structure + defense modifiers together should contribute <30% of total
# variance. These are "guardrails and tweaks," NOT primary drivers.
# When TEAM_SCALE is data-driven, modifiers stay small relative to it.
#
# BKE compression: player-level BKE std ≈ 0.34 vs real RAPM std ≈ 3.0
# (compression factor ≈ 9×). Minute-weighted team averages further
# compress to std ≈ 0.15.  To produce net ratings on a real NBA scale
# (std ≈ 5-6 per 100 possessions), TEAM_SCALE ≈ 5.5/0.22 ≈ 25.
# Using 20 as conservative default to avoid overprediction given data
# quality uncertainty.
# ─────────────────────────────────────────────────────────────────────

#   Team scaling (default; overridden by data-driven fit when actuals exist)
DEFAULT_TEAM_SCALE = 20.0         # maps compressed BKE to NBA per-100-poss scale
TEAM_SCALE_MIN = 8.0              # data-driven floor (below this, talent can't dominate)
TEAM_SCALE_MAX = 50.0             # data-driven ceiling
TEAM_SCALE_FIT_R_THRESHOLD = 0.20 # minimum |r| for data-driven fit to override default

#   Modifier proportion target — modifiers capped at this fraction of talent std
MODIFIER_MAX_FRACTION = 0.30

#   Interaction matrix scaling
INTERACTION_LAMBDA = 0.75
INTERACTION_CAP = 1.5

#   Offensive structure thresholds
BETA_TOV = 0.12        # turnover penalty coefficient (negligible in practice)
BETA_FTR = 0.15        # free throw rate bonus coefficient (negligible in practice)
PLAYMAKING_MPG_THRESHOLD = 15.0
PLAYMAKING_AST_THRESHOLD = 0.20   # AST% threshold for playmaker
SPACING_MPG_THRESHOLD = 15.0
SPACING_3PA_RATE_THRESHOLD = 0.30 # share of shots from 3
SPACING_3P_PCT_THRESHOLD = 0.35   # 3P make percentage
STRUCTURE_CAP = 1.0               # reduced from 2.0 — structure is a modifier

#   Structure penalty/bonus magnitudes (v2 — reduced to stay subsidiary to talent)
PLAYMAKING_SOLO_PENALTY = -0.25   # (was -0.7) — only 1 playmaker on roster
PLAYMAKING_DEEP_BONUS = 0.10      # (was +0.3) — 3+ playmakers
SPACING_POOR_PENALTY = -0.35      # (was -1.0) — fewer than 2 credible shooters
SPACING_ELITE_BONUS = 0.15        # (was +0.5) — 4+ credible shooters

#   Transition structure term
BETA_TRANSITION = 0.10  # transition success bonus coefficient

#   Defensive controls — magnitudes reduced to stay subsidiary to talent
DEFENSE_CAP = 1.5                 # reduced from 3.0
RP_MPG_THRESHOLD = 15.0
POA_MPG_THRESHOLD = 15.0
DIVERSITY_MPG_THRESHOLD = 15.0
LIABILITY_DBKE_THRESHOLD = -1.0
LIABILITY_MPG_THRESHOLD = 20.0

#   Defense penalty/bonus magnitudes (v2 — proportional to talent, not dominant)
RP_MISSING_PENALTY = -0.40        # (was -1.2) — no rim presence at all
POA_MISSING_PENALTY = -0.25       # (was -0.8) — no POA defender
BOTH_MISSING_PENALTY = -0.15      # (was -0.5) — neither present
ANCHOR_RP_WEIGHT = 0.20           # (was 0.6) — how much top rim protector DBKE matters
ANCHOR_POA_WEIGHT = 0.15          # (was 0.4) — how much top POA defender DBKE matters
DIVERSITY_BONUS_PER_ARCH = 0.05   # (was 0.15) per unique archetype above 3
DIVERSITY_BONUS_CAP = 0.20        # (was 0.6) max diversity bonus
LIABILITY_PER_PLAYER = -0.15      # (was -0.4) per defensive liability
LIABILITY_STACKING_PENALTY = -0.15  # (was -0.4) extra if 2+ liabilities

#   Volatility model — widened to produce realistic team-level variation
#   Target: vol_total should vary by ~1-2 points across teams
VOL_FLOOR = 1.0                   # reduced from 8.0 (was absurdly high)
VOL_CEILING = 6.0                 # reduced from 16.0
ALPHA_3PA = 5.0                   # increased from 1.0 (was negligible)
ALPHA_CREATION = 5.0              # increased from 1.0 (was negligible)
ALPHA_TRANSITION = 3.0            # increased from 0.5 (was negligible)
CREATION_CONCENTRATION_THRESHOLD = 0.28  # slightly lower to catch more variation

#   Transition structure (success-weighted)
TRANSITION_PPP_LEAGUE_AVG = 1.10  # approximate league transition PPP average

# Step 4 outputs — Player Availability Factor
STEP4_VALIDATION_REPORT = REPORTS_DIR / "player_eval_step4_availability_validation.json"

# Step 4 — Availability Factor constants
# ── Age curve: piecewise linear decay ──
# Players under 27 → no age penalty. 27-30 → mild, 30-34 → moderate, 35+ → steep.
AVAIL_AGE_BREAKPOINTS = [27, 30, 34]      # age thresholds
AVAIL_AGE_FACTORS = [1.0, 0.97, 0.92, 0.82]  # factor AT each breakpoint edge
#   <27 → 1.0, 27→0.97 (linear interp to 30→0.92), 30→0.92 (linear to 34→0.82), 35+→0.82

# ── Role expectation: how many games a player is "expected" to play ──
# Major rotation players (high MPG or high salary) are expected to play ~82.
# Lower-minute players have a softer expectation.
AVAIL_MAJOR_ROTATION_MPG = 20.0       # MPG threshold for "major rotation" expectation
AVAIL_HIGH_SALARY_THRESHOLD = 10_000_000  # $10M — players earning this are expected to play 82
AVAIL_FULL_SEASON_GAMES = 82          # NBA regular season game count
AVAIL_MIN_EXPECTED_GAMES = 40         # floor for expected games (bench/two-way players)

# Modeling constants
RANDOM_SEED = 42
MINUTE_SHARE_CAP = 0.22
MINUTE_SHARE_FLOOR = 0.0
HIGH_USAGE_QUANTILE = 0.75

# ═══════════════════════════════════════════════════════════════════
# Forecast Mode — Forward Projection Constants
# ═══════════════════════════════════════════════════════════════════

FORECAST_DIR = PROCESSED_DIR / "forecast"
FORECAST_DIR.mkdir(parents=True, exist_ok=True)

# Projected player profiles output (same schema as player_impact_profiles)
PROJECTED_PROFILES_PATH = FORECAST_DIR / "projected_player_profiles.parquet"
# Projected team features output (same schema as team_feature_aggregation)
PROJECTED_TEAM_FEATURES_PATH = FORECAST_DIR / "projected_team_features.parquet"
# Forecast validation output
FORECAST_VALIDATION_REPORT = REPORTS_DIR / "forecast_validation.json"
# Forecast season simulation output
FORECAST_SEASON_RESULTS = REPORTS_DIR / "forecast_season_results.json"
# Forecast lineup projection output
FORECAST_LINEUP_REPORT = REPORTS_DIR / "forecast_lineup_profiles.json"

# Optional manual rookies input (override path for forecast mode; fallback only).
ROOKIES_INPUT_PATH = DATA_DIR / "forecasting" / "rookies.csv"

# ── Age Curve Constants ──────────────────────────────────────────
# Piecewise linear age adjustment for impact metrics (BKE-scale per year).
# Positive = improvement, negative = decline.
# Calibrated from empirical year-to-year BKE deltas by age bucket.
AGE_CURVE_BREAKPOINTS = [21, 24, 27, 30, 33, 36]
AGE_CURVE_DELTAS = [
    +0.08,   # <21: strong improvement (empirical ~+0.08)
    +0.04,   # 21-24: moderate improvement (empirical ~+0.03-0.05)
    +0.01,   # 24-27: approaching peak, slight improvement
    -0.01,   # 27-30: slight decline
    -0.02,   # 30-33: moderate decline (survivorship bias tempers this)
    -0.02,   # 33-36: decline continues
    -0.03,   # 36+: steeper decline
]

# ── Rookie Projection Constants ─────────────────────────────────
# Expected impact by draft slot tier (BKE-scale).
# Conservative baseline: even strong rookies are often near/below replacement.
ROOKIE_IMPACT_BY_TIER = {
    "lottery": -0.05,
    "mid_first": -0.10,
    "late_first": -0.15,
    "second_round": -0.18,
    "undrafted": -0.22,
}

# Small search grid to tune rookie priors on historical backtests.
ROOKIE_IMPACT_SCALE_GRID = [0.85, 1.0, 1.15]
ROOKIE_IMPACT_SCALE_DEFAULT = 1.0

# Expected MPG by draft slot tier
ROOKIE_MPG_BY_TIER = {
    "lottery": 20.0,
    "mid_first": 13.0,
    "late_first": 9.0,
    "second_round": 6.0,
    "undrafted": 4.0,
}
# Rookie behavioral rate defaults (league average for position)
ROOKIE_DEFAULT_USAGE = 0.18
ROOKIE_DEFAULT_AST_RATE = 0.12
ROOKIE_DEFAULT_TOV_RATE = 0.14
ROOKIE_DEFAULT_3PT_RATE = 0.35
ROOKIE_DEFAULT_EFG = 0.49
ROOKIE_DEFAULT_FTR = 0.25

# ── Forecast Projection Controls ───────────────────────────────

# Impact regression-to-mean controls.
ENABLE_IMPACT_REGRESSION_TO_MEAN = True
IMPACT_REGRESSION_BASE_WEIGHT = 0.12
IMPACT_REGRESSION_STABILITY_WEIGHT = 0.18
IMPACT_REGRESSION_MINUTES_WEIGHT = 0.10
IMPACT_REGRESSION_LOW_MPG_THRESHOLD = 18.0
IMPACT_REGRESSION_MIN_WEIGHT = 0.08
IMPACT_REGRESSION_MAX_WEIGHT = 0.30

# Minutes projection controls (age + impact + salary + team depth competition).
ENABLE_MINUTES_IMPACT_ADJUSTMENT = True
ENABLE_MINUTES_SALARY_ADJUSTMENT = True
ENABLE_MINUTES_TEAM_COMPETITION_ADJUSTMENT = True

MINUTES_IMPACT_ADJUST_SLOPE = 1.5
MINUTES_IMPACT_ADJUST_CLIP = 2.5

MINUTES_SALARY_LEVEL_SCALE_M = 12.0
MINUTES_SALARY_CHANGE_SCALE_M = 10.0
MINUTES_SALARY_LEVEL_WEIGHT = 0.8
MINUTES_SALARY_CHANGE_WEIGHT = 0.6
MINUTES_SALARY_ADJUST_CLIP = 2.0

MINUTES_COMPETITION_PENALTY = 0.40
MINUTES_COMPETITION_MAX_PENALTY = 2.5

# ═══════════════════════════════════════════════════════════════════
# Forecast Structural Improvements — Availability, Replacement, Star
# ═══════════════════════════════════════════════════════════════════

# ── Availability/Health Regression ──────────────────────────────
# Apply expected games-played discount based on prior-season availability +
# role weight.  Stars who play 82 games are rare; injury risk rises with
# age and minute load.  This shrinks projected minutes toward realistic
# expected games.
ENABLE_FORECAST_AVAILABILITY_DISCOUNT = True
# Role-based expected games-played fraction (of 82).
# Stars who played 82 are still projected at < 82 because most won't repeat.
FORECAST_AVAIL_STAR_GAMES_FRACTION = 0.88       # ≥28 MPG prior season
FORECAST_AVAIL_STARTER_GAMES_FRACTION = 0.92    # 20-28 MPG
FORECAST_AVAIL_ROTATION_GAMES_FRACTION = 0.96   # 10-20 MPG
FORECAST_AVAIL_BENCH_GAMES_FRACTION = 0.98      # <10 MPG
# Age-based additional discount (multiplicative on top of role fraction).
FORECAST_AVAIL_AGE_DISCOUNT_START = 30          # start aging out
FORECAST_AVAIL_AGE_DISCOUNT_PER_YEAR = 0.015    # 1.5% per year beyond start
FORECAST_AVAIL_AGE_DISCOUNT_MAX = 0.10           # max 10% extra discount
# Prior-season games-played factor (multiplicative).
# Players who missed significant time last year are likelier to miss again.
FORECAST_AVAIL_PRIOR_GP_DISCOUNT_THRESHOLD = 65 # games below this → extra disc
FORECAST_AVAIL_PRIOR_GP_DISCOUNT_SLOPE = 0.003  # per game below threshold

# ── Replacement-Level Minutes Buffer ───────────────────────────
# Reserve a fraction of team minutes for replacement-level contributors
# (injury replacements, G-League callups, 10-day contracts, etc.).
# These players contribute near-zero or negative impact.
ENABLE_FORECAST_REPLACEMENT_BUFFER = False
FORECAST_REPLACEMENT_BUFFER_FRACTION = 0.12  # 12% of team minutes to replacement
FORECAST_REPLACEMENT_IMPACT_BKE = -0.20      # replacement player BKE (~bottom 25%)
FORECAST_REPLACEMENT_IMPACT_OBKE = -0.10
FORECAST_REPLACEMENT_IMPACT_DBKE = -0.10

# ── Star Concentration / Top-Heavy Adjustment ──────────────────
# When a team's value is heavily concentrated in 1-2 players, apply a
# fragility penalty.  Top-heavy rosters have more downside variance
# (injury to star = catastrophic drop) and historically underperform
# vs. balanced rosters with similar total talent.
ENABLE_FORECAST_STAR_CONCENTRATION = True
# How much weight the top player's share of team BKE carries
FORECAST_STAR_TOP1_SHARE_THRESHOLD = 0.40     # >40% of team net impact = fragile
FORECAST_STAR_TOP2_SHARE_THRESHOLD = 0.65     # top 2 holding >65% = very fragile
FORECAST_STAR_CONCENTRATION_PENALTY_MAX = 2.0 # max net-rating penalty (points)
FORECAST_STAR_CONCENTRATION_PENALTY_SLOPE = 4.0  # penalty rate above threshold
