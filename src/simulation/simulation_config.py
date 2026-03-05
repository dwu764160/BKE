"""Centralized configuration and paths for the simulation module."""

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
HISTORICAL_DIR = DATA_DIR / "historical"
REPORTS_DIR = ROOT_DIR / "reports"

TEAM_FEATURES_PATH = PROCESSED_DIR / "player_eval" / "team_feature_aggregation.parquet"

# Ensure report output directory exists when simulation scripts run standalone.
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Core simulation constants
SIGMA_LEAGUE = 3.0
HOME_COURT_ADVANTAGE = 2.0
SEASON_SIMULATIONS = 10_000
SIMULATION_RANDOM_SEED = 42

# Rank-based postseason cutoffs (per conference)
DIRECT_PLAYOFF_RANK = 6
PLAY_IN_RANK = 10
