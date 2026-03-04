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

# Modeling constants
RANDOM_SEED = 42
MINUTE_SHARE_CAP = 0.22
MINUTE_SHARE_FLOOR = 0.0
HIGH_USAGE_QUANTILE = 0.75
