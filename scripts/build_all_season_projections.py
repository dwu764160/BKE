"""
scripts/build_all_season_projections.py
=============================================================================
Build projected team features for ALL seasons using actual prior-season
game margins — consistent pts/100 possession units throughout.

Strategy (per season N, projecting from N-1 actuals):
  actual_net_rating  = sum(PLUS_MINUS) / sum(MIN) * 48   (pts/48 min ≈ pts/100 poss)
  projected_net_rating = 0.70 * actual_net_rating + 0.30 * 0.0  (regress to mean)

Season coverage:
  2018-19 → 2025-26  (projects from 2017-18 actuals onward)
  2017-18 excluded: no 2016-17 prior-season data in the pipeline.

Replaces the stale multi-era projected_team_features.parquet that mixed
compressed old-BKE units (~0.22 std for 2018-22) with pts/100 units
(~4.0 std for 2023-25). All output is in pts/100 possessions.

Also writes per-season files:
  data/processed/forecast/projected_team_features_v40_{season}.parquet

Output: data/processed/forecast/projected_team_features.parquet
=============================================================================
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.simulation.simulation_config import HISTORICAL_DIR, FORECAST_DIR
from src.modeling.model_config import SEASONS

GAME_LOGS_PATH   = HISTORICAL_DIR / "team_game_logs.parquet"
OUT_ALL          = FORECAST_DIR / "projected_team_features.parquet"

# Seasons we can project (need N-1 actuals; 2017-18 is the earliest in SEASONS)
# 2018-19 uses 2017-18 actuals; 2025-26 uses 2024-25 actuals.
PROJECTABLE = [s for s in SEASONS if s != "2017-18"]

REGRESSION_WEIGHT = 0.30   # fraction pulled toward league mean (0.0)
ACTUAL_WEIGHT     = 0.70


def actual_net_rating(logs: pd.DataFrame, team: str, season: str) -> float | None:
    """Compute season net rating in pts/100 poss from PLUS_MINUS game logs."""
    rows = logs[(logs["TEAM_ABBREVIATION"] == team) & (logs["SEASON"] == season)]
    if rows.empty or rows["MIN"].sum() == 0:
        return None
    # PLUS_MINUS = team pts - opp pts per game. Per 48 min ≈ per 100 poss.
    return float(rows["PLUS_MINUS"].sum() / rows["MIN"].sum() * 48.0)


def project_season(logs: pd.DataFrame, target_season: str,
                   prior_season: str, template_df: pd.DataFrame) -> pd.DataFrame:
    """Build projected_team_features rows for target_season using prior_season actuals."""
    prior_teams = sorted(logs[logs["SEASON"] == prior_season]["TEAM_ABBREVIATION"].unique())
    if not prior_teams:
        print(f"  {target_season}: no prior-season logs for {prior_season} — skipping")
        return pd.DataFrame()

    # Use prior-season template row for structural columns (archetypes, vol, etc.)
    # If no template exists, build a minimal row with just the rating.
    template_season = template_df[template_df["season"] == prior_season]

    rows = []
    missing_template = 0
    for team in prior_teams:
        actual = actual_net_rating(logs, team, prior_season)
        if actual is None:
            projected = 0.0
        else:
            projected = ACTUAL_WEIGHT * actual + REGRESSION_WEIGHT * 0.0

        if not template_season.empty:
            t = template_season[template_season["team_abbreviation"] == team]
            if not t.empty:
                row = t.iloc[0].copy().to_dict()
            else:
                row = template_season.iloc[0].copy().to_dict()
                missing_template += 1
        else:
            row = {}
            missing_template += 1

        row["season"]                   = target_season
        row["team_abbreviation"]        = team
        row["team_net_rating_projected"] = round(projected, 4)
        if actual is not None:
            row["team_net_rating_raw"]   = round(actual, 4)
        rows.append(row)

    if missing_template:
        print(f"  {target_season}: {missing_template} teams had no template row "
              "(used league-average structure columns)")

    df = pd.DataFrame(rows)
    # Ensure key columns are present
    for col in ["team_net_rating_projected", "team_net_rating_raw"]:
        if col not in df.columns:
            df[col] = 0.0
    return df


def main() -> None:
    print("=" * 60)
    print("Building projected team features — all seasons (pts/100 poss)")
    print("=" * 60)

    if not GAME_LOGS_PATH.exists():
        print(f"ERROR: {GAME_LOGS_PATH} not found"); return

    logs = pd.read_parquet(GAME_LOGS_PATH)
    logs["TEAM_ABBREVIATION"] = logs["TEAM_ABBREVIATION"].str.upper()

    # Load existing template so we can copy structural columns for old seasons.
    # We use whatever projected_team_features.parquet currently has (even if stale)
    # only for non-rating columns; we replace the rating column for every season.
    template_df = pd.DataFrame()
    if OUT_ALL.exists():
        template_df = pd.read_parquet(OUT_ALL)
        template_df["team_abbreviation"] = template_df["team_abbreviation"].str.upper()
        template_df["season"] = template_df["season"].astype(str)

    all_parts = []

    # Walk through SEASONS in order to build N from N-1
    season_list = [s for s in SEASONS]
    for i, target in enumerate(season_list):
        if target == "2017-18":
            continue  # no 2016-17 data in pipeline
        prior = season_list[i - 1]
        df = project_season(logs, target, prior, template_df)
        if df.empty:
            continue

        n_teams = len(df)
        std = df["team_net_rating_projected"].std()
        mean = df["team_net_rating_projected"].mean()
        print(f"  {target}: {n_teams} teams | projected μ={mean:+.2f} σ={std:.2f} pts/100")

        all_parts.append(df)

        # Write per-season file for use in build_2025_26_projections / validate_forecast
        season_path = FORECAST_DIR / f"projected_team_features_v40_{target}.parquet"
        df.to_parquet(season_path, index=False)

    if not all_parts:
        print("ERROR: nothing built"); return

    combined = pd.concat(all_parts, ignore_index=True)
    print(f"\nTotal: {len(combined)} rows, {combined['season'].nunique()} seasons")
    print(f"Seasons: {sorted(combined['season'].unique())}")
    print(f"\nScale check (σ of team_net_rating_projected by season):")
    print(combined.groupby("season")["team_net_rating_projected"].std().round(3).to_string())

    FORECAST_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUT_ALL, index=False)
    print(f"\nSaved → {OUT_ALL}")


if __name__ == "__main__":
    main()
