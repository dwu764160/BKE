"""
scripts/build_2025_26_projections.py

Generate 2025-26 team projections from 2024-25 actual season data.
This allows us to test the v4.0 model against real 2025-26 games from Kalshi.

The strategy:
1. Aggregate 2024-25 actual team stats (net rating) from game logs
2. Apply conservative regression to mean for projection
3. Use lineup-weighted team features to build full projected profile
4. Save as projected_team_features_v40.parquet with 2025-26 season

Usage:
    python3 scripts/build_2025_26_projections.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.simulation.simulation_config import HISTORICAL_DIR, FORECAST_DIR

def compute_ytd_net_rating(logs_df, team, season='2024-25'):
    """Compute season-long net rating for a team."""
    team_logs = logs_df[
        (logs_df['TEAM_ABBREVIATION'] == team) &
        (logs_df['SEASON'] == season)
    ]
    if len(team_logs) == 0:
        return np.nan

    # Net rating per game (PLUS_MINUS is team +/-)
    total_plus_minus = team_logs['PLUS_MINUS'].sum()
    total_minutes = team_logs['MIN'].sum()

    # Per 100 possessions: assume ~100 possessions per 48 minutes
    net_rating = (total_plus_minus / total_minutes) * 48.0
    return float(net_rating)

def main():
    print("Building 2025-26 Projections from 2024-25 Actuals")
    print("=" * 70)

    # Load 2024-25 actual game logs
    logs = pd.read_parquet(HISTORICAL_DIR / "team_game_logs.parquet")
    logs_2425 = logs[logs['SEASON'] == '2024-25'].copy()
    print(f"Loaded {len(logs_2425)} games from 2024-25 season")

    # Load latest projected features as template
    features_template = pd.read_parquet(
        FORECAST_DIR / "projected_team_features.parquet"
    )
    features_2425 = features_template[
        features_template['season'] == '2024-25'
    ].copy()

    print(f"Template has {len(features_2425)} teams from 2024-25")

    # For each team, compute observed 2024-25 net rating
    # Then blend toward template value with conservative regression
    teams = sorted(features_2425['team_abbreviation'].unique())
    print(f"\nProcessing {len(teams)} teams...")

    projected_rows = []

    for team in teams:
        # Get 2024-25 template row
        template_row = features_2425[
            features_2425['team_abbreviation'] == team
        ]

        if len(template_row) == 0:
            print(f"  ⚠️ {team}: no template found, skipping")
            continue

        template_row = template_row.iloc[0].copy()

        # Compute actual 2024-25 net rating
        actual_net_rating = compute_ytd_net_rating(logs_2425, team, '2024-25')

        if pd.isna(actual_net_rating):
            print(f"  ⚠️ {team}: no game logs found, using template")
            proj_net_rating = float(template_row['team_net_rating_projected'])
        else:
            # Blend: 70% actual 2024-25, 30% regression to league mean (0)
            # This gives weight to what actually happened but smooths outliers
            lambda_blend = 0.70
            proj_net_rating = lambda_blend * actual_net_rating + (1 - lambda_blend) * 0.0

            print(f"  {team:3s}: actual={actual_net_rating:+6.2f}, "
                  f"projected={proj_net_rating:+6.2f}")

        # Create 2025-26 projection by copying 2024-25 template
        # and updating only the net_rating_projected field
        proj_row = template_row.copy()
        proj_row['season'] = '2025-26'
        proj_row['team_net_rating_projected'] = proj_net_rating

        # Keep all other fields from template (vol, archetype dist, etc.)
        # This is conservative—we're not updating offense/defense components
        # but that's fine for a quick projection

        projected_rows.append(proj_row)

    # Create output dataframe
    projections_2526 = pd.DataFrame(projected_rows)

    # Save
    out_path = FORECAST_DIR / "projected_team_features_v40_2025-26.parquet"
    projections_2526.to_parquet(out_path, index=False)
    print(f"\n✓ Saved {len(projections_2526)} teams to {out_path}")

    # Show summary stats
    print(f"\n2025-26 Projected Net Ratings (70% actual, 30% regression):")
    summary = projections_2526[['team_abbreviation', 'team_net_rating_projected']].copy()
    summary['team_net_rating_projected'] = summary['team_net_rating_projected'].round(2)
    summary = summary.sort_values('team_net_rating_projected', ascending=False)

    print("\nTop 5 (best):")
    for _, row in summary.head(5).iterrows():
        print(f"  {row['team_abbreviation']}: {row['team_net_rating_projected']:+6.2f}")

    print("\nBottom 5 (worst):")
    for _, row in summary.tail(5).iterrows():
        print(f"  {row['team_abbreviation']}: {row['team_net_rating_projected']:+6.2f}")

    print(f"\nMean projection: {projections_2526['team_net_rating_projected'].mean():.2f}")
    print(f"Std dev: {projections_2526['team_net_rating_projected'].std():.2f}")

if __name__ == "__main__":
    main()
