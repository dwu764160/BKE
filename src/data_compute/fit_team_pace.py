"""
src/data_compute/fit_team_pace.py
=============================================================================
Compute per-team-season pace (possessions per 48 minutes) from team_game_logs.

Possessions per game = FGA - OREB + TOV + 0.44 * FTA
Pace per 48 = mean(possessions) / mean(MIN) * 240

Output: reports/team_pace.json
  {
    "league_avg_pace": float,
    "team_pace": {
       "<season>": {"<team_abbrev>": pace_value, ...},
       ...
    }
  }

Pace is a matchup variable — game-level pace is the average of home and away team
pace. The game model uses pace to scale margin variance (sigma).
=============================================================================
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


GAME_LOGS_PATH = Path("data/historical/team_game_logs.parquet")
OUTPUT_PATH = Path("reports/team_pace.json")

# Possessions = FGA - OREB + TOV + 0.44 * FTA  (standard estimator)
POSS_FTA_WEIGHT = 0.44


def compute_possessions(df: pd.DataFrame) -> pd.Series:
    fga = pd.to_numeric(df.get("FGA"), errors="coerce")
    oreb = pd.to_numeric(df.get("OREB"), errors="coerce")
    tov = pd.to_numeric(df.get("TOV"), errors="coerce")
    fta = pd.to_numeric(df.get("FTA"), errors="coerce")
    return fga - oreb + tov + POSS_FTA_WEIGHT * fta


def main():
    print("=" * 72)
    print("TEAM PACE FITTING")
    print("=" * 72)

    df = pd.read_parquet(GAME_LOGS_PATH)
    df = df.copy()
    df["possessions"] = compute_possessions(df)
    df["minutes"] = pd.to_numeric(df["MIN"], errors="coerce")

    # Pace per game per team: possessions normalized to 240 team-minutes (48 min regulation)
    # Each row in team_game_logs is a team-game with team minutes column "MIN"
    # Standard regulation: MIN = 240 (5 players × 48 min); OT adds ~25 per period.
    df["pace_per_48"] = df["possessions"] / df["minutes"] * 240.0

    # Aggregate to team-season
    team_season = df.groupby(["SEASON", "TEAM_ABBREVIATION"]).agg(
        pace_per_48=("pace_per_48", "mean"),
        n_games=("GAME_ID", "count"),
    ).reset_index()

    league_avg_pace_per_season = {}
    team_pace_nested: dict = {}

    for season, season_df in team_season.groupby("SEASON"):
        league_avg = float(season_df["pace_per_48"].mean())
        league_avg_pace_per_season[str(season)] = league_avg
        team_pace_nested[str(season)] = {
            str(row["TEAM_ABBREVIATION"]): float(row["pace_per_48"])
            for _, row in season_df.iterrows()
        }
        # Print summary
        sorted_teams = season_df.sort_values("pace_per_48", ascending=False)
        print(f"\nSeason {season}: league avg pace = {league_avg:.2f} (n_teams={len(season_df)})")
        print(f"  Fastest: {sorted_teams.iloc[0]['TEAM_ABBREVIATION']} @ {sorted_teams.iloc[0]['pace_per_48']:.2f}")
        print(f"  Slowest: {sorted_teams.iloc[-1]['TEAM_ABBREVIATION']} @ {sorted_teams.iloc[-1]['pace_per_48']:.2f}")

    league_avg_overall = float(team_season["pace_per_48"].mean())

    out = {
        "league_avg_pace": league_avg_overall,
        "league_avg_pace_per_season": league_avg_pace_per_season,
        "team_pace": team_pace_nested,
        "n_team_seasons": int(len(team_season)),
        "fta_weight": POSS_FTA_WEIGHT,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nLeague avg pace (overall): {league_avg_overall:.2f}")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
