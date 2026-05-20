"""
src/data_compute/fit_rest_hca_coefficients.py
=============================================================================
Fit empirical coefficients from team_game_logs:
  - Per-team Home Court Advantage (HCA), shrunk toward league mean
  - B2B (back-to-back) penalty per team
  - Rest day bonus per team

Methodology:
  1. Load team_game_logs.parquet (3 seasons).
  2. Derive home/away from MATCHUP string ("vs." = home, "@" = away).
  3. Compute days_rest and is_b2b per team-game (chronologically sorted).
  4. Join home/away rows on GAME_ID to produce one row per game.
  5. Compute team strength control: prior-2-season win rate (or shrunk to .500).
  6. Fit OLS:
        actual_margin ~ team_strength_diff
                      + is_b2b_home + is_b2b_away
                      + days_rest_home + days_rest_away
                      + team_hca_fixed_effects
  7. Extract:
        B2B_PENALTY = coef on is_b2b (shared)
        REST_DAY_BONUS = coef on days_rest (shared)
        team_hca[team] = intercept + fixed_effect[team]  (shrunk to league mean)
  8. Save to reports/rest_hca_coefficients.json.

Output schema:
  {
    "league_avg_hca": float,
    "league_avg_hca_unshrunk": float,
    "b2b_penalty": float,           # negative; subtract from margin
    "rest_day_bonus": float,        # positive; add per day rested
    "team_hca": {team_abbrev: float},
    "n_games": int,
    "n_seasons": int,
    "ols_r_squared": float,
    "ols_coefficients": {...}
  }
=============================================================================
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


GAME_LOGS_PATH = Path("data/historical/team_game_logs.parquet")
OUTPUT_PATH = Path("reports/rest_hca_coefficients.json")

LEAGUE_AVG_HCA_FALLBACK = 2.0
MIN_GAMES_PER_TEAM_FOR_FULL_WEIGHT = 60  # ~one season of home games
SHRINKAGE_K = 30


def _is_home_from_matchup(matchup: str) -> int:
    text = str(matchup or "").upper()
    if " VS." in text or " VS " in text:
        return 1
    if " @ " in text:
        return 0
    return -1  # unknown


def load_games() -> pd.DataFrame:
    df = pd.read_parquet(GAME_LOGS_PATH)
    df = df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["TEAM_ID", "GAME_DATE"])

    # Days rest per team (NaN for first game → fill with 3, clip to [0, 7])
    df["prev_game_date"] = df.groupby("TEAM_ID")["GAME_DATE"].shift(1)
    df["days_rest"] = (df["GAME_DATE"] - df["prev_game_date"]).dt.days - 1
    df["days_rest"] = df["days_rest"].fillna(3).clip(lower=0, upper=7).astype(int)
    df["is_b2b"] = (df["days_rest"] == 0).astype(int)

    df["is_home"] = df["MATCHUP"].apply(_is_home_from_matchup)
    return df


def build_game_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Join home and away rows for each GAME_ID to produce one row per game."""
    home = df[df["is_home"] == 1].copy()
    away = df[df["is_home"] == 0].copy()

    home_cols = {
        "GAME_ID": "GAME_ID",
        "GAME_DATE": "game_date",
        "SEASON": "season",
        "TEAM_ID": "home_team_id",
        "TEAM_ABBREVIATION": "home_team",
        "PTS": "home_pts",
        "OPP_PTS": "away_pts_from_home_row",
        "margin": "home_margin",
        "is_b2b": "is_b2b_home",
        "days_rest": "days_rest_home",
    }
    away_cols = {
        "GAME_ID": "GAME_ID",
        "TEAM_ID": "away_team_id",
        "TEAM_ABBREVIATION": "away_team",
        "PTS": "away_pts",
        "is_b2b": "is_b2b_away",
        "days_rest": "days_rest_away",
    }
    home = home[list(home_cols.keys())].rename(columns=home_cols)
    away = away[list(away_cols.keys())].rename(columns=away_cols)

    games = home.merge(away, on="GAME_ID", how="inner")
    # Sanity: home_margin should equal home_pts - away_pts
    games["actual_margin"] = games["home_pts"] - games["away_pts"]
    return games


def compute_team_strength(games: pd.DataFrame) -> pd.Series:
    """Compute a simple per-team season strength (win rate) for use as control.

    For the same-season strength control, we use the team's full-season win rate.
    This is a leaky control (team strength reflects results we're trying to predict),
    but its purpose here is only to absorb team quality variance so the HCA/B2B/rest
    coefficients are not confounded by team strength. We are NOT using these
    coefficients to predict the same-season games.
    """
    home_pts = games.groupby(["season", "home_team"])["home_pts"].sum()
    home_count = games.groupby(["season", "home_team"]).size()
    away_pts = games.groupby(["season", "away_team"])["away_pts"].sum()
    away_count = games.groupby(["season", "away_team"]).size()

    home_wins = (games["actual_margin"] > 0).groupby([games["season"], games["home_team"]]).sum()
    away_wins = (games["actual_margin"] < 0).groupby([games["season"], games["away_team"]]).sum()

    total_games = home_count.add(away_count, fill_value=0)
    total_wins = home_wins.add(away_wins, fill_value=0)
    win_rate = (total_wins / total_games).rename("season_win_rate")
    return win_rate


def compute_classical_team_hca(games: pd.DataFrame) -> Dict[str, float]:
    """Classical team HCA = (mean home margin - mean away margin) / 2.

    This is the textbook approach: average over 41 home + 41 away games per season
    isolates the venue effect from team quality. A great team's home and away margins
    are BOTH high; a poor team's are BOTH low; the DIFFERENCE between them is the
    venue effect alone.

    For each team:
      home_margin_avg = mean(margins in home games)
      away_margin_avg = mean(margins in away games, from team's perspective = -home_margin from home team perspective)
      HCA[team] = (home_margin_avg - away_margin_avg) / 2

    This is symmetric and orthogonal to team strength.
    """
    # Build per-team-game records: (team, season, is_home, team_margin_from_team_perspective)
    home_records = games[["home_team", "season", "actual_margin"]].rename(
        columns={"home_team": "team", "actual_margin": "margin_team_perspective"}
    )
    home_records["is_home"] = True

    away_records = games[["away_team", "season", "actual_margin"]].rename(
        columns={"away_team": "team"}
    )
    away_records["margin_team_perspective"] = -away_records["actual_margin"]  # flip sign for away team perspective
    away_records["is_home"] = False
    away_records = away_records.drop(columns=["actual_margin"])

    all_records = pd.concat([home_records, away_records], ignore_index=True)
    home_avg = all_records[all_records["is_home"]].groupby("team")["margin_team_perspective"].mean()
    away_avg = all_records[~all_records["is_home"]].groupby("team")["margin_team_perspective"].mean()

    team_hca_raw = ((home_avg - away_avg) / 2.0).to_dict()
    return team_hca_raw


def fit_coefficients(games: pd.DataFrame, win_rate: pd.Series) -> dict:
    """Two-step fit:

    Step A: OLS for B2B and rest coefficients (which are matchup-specific, not absorbed
            into team strength). Uses team season strength as a control.
    Step B: Classical HCA per team — orthogonal to team strength by construction.
    """
    # --- Step A: B2B and rest coefficients via OLS with strength control ---
    games = games.copy()
    home_wr = games.set_index(["season", "home_team"]).index.map(win_rate.to_dict()).astype(float)
    away_wr = games.set_index(["season", "away_team"]).index.map(win_rate.to_dict()).astype(float)
    games["home_win_rate"] = pd.Series(home_wr, index=games.index).fillna(0.5)
    games["away_win_rate"] = pd.Series(away_wr, index=games.index).fillna(0.5)
    games["strength_diff"] = games["home_win_rate"] - games["away_win_rate"]

    features = {
        "intercept": np.ones(len(games)),
        "strength_diff": games["strength_diff"].to_numpy(),
        "is_b2b_home": games["is_b2b_home"].to_numpy(),
        "is_b2b_away": games["is_b2b_away"].to_numpy(),
        "days_rest_home": games["days_rest_home"].to_numpy(),
        "days_rest_away": games["days_rest_away"].to_numpy(),
    }
    feature_names = list(features.keys())
    X = np.column_stack([features[n] for n in feature_names])
    y = games["actual_margin"].to_numpy()

    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ coef
    ss_res = float(((y - y_hat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    coef_dict = dict(zip(feature_names, coef.tolist()))

    # --- Step B: Classical per-team HCA ---
    team_hca_raw = compute_classical_team_hca(games)
    home_games_per_team = games.groupby("home_team").size().to_dict()

    # League-average HCA from classical estimates
    league_avg_hca_classical = float(np.mean(list(team_hca_raw.values())))

    # Shrink team HCA toward classical league mean based on sample size
    team_hca_shrunk: Dict[str, float] = {}
    for team, raw in team_hca_raw.items():
        n = home_games_per_team.get(team, 0)
        shrinkage = n / (n + SHRINKAGE_K)
        team_hca_shrunk[team] = shrinkage * raw + (1 - shrinkage) * league_avg_hca_classical

    return {
        "league_avg_hca": league_avg_hca_classical,
        "league_avg_hca_method": "classical_home_minus_away_margin_div2",
        "league_avg_hca_intercept": coef_dict["intercept"],  # for reference
        "b2b_penalty_home": coef_dict["is_b2b_home"],
        "b2b_penalty_away": coef_dict["is_b2b_away"],
        "rest_day_bonus_home": coef_dict["days_rest_home"],
        "rest_day_bonus_away": coef_dict["days_rest_away"],
        "team_hca": team_hca_shrunk,
        "team_hca_unshrunk": team_hca_raw,
        "home_games_per_team": home_games_per_team,
        "n_games": int(len(games)),
        "n_seasons": int(games["season"].nunique()),
        "ols_r_squared": r_squared,
        "ols_coefficients": coef_dict,
        "shrinkage_k": SHRINKAGE_K,
    }


def main():
    print("=" * 72)
    print("REST + HCA COEFFICIENT FITTING")
    print("=" * 72)
    print(f"Reading {GAME_LOGS_PATH}...")
    df = load_games()
    print(f"Loaded {len(df)} team-games across {df['SEASON'].nunique()} seasons")

    n_unknown_home = int((df["is_home"] == -1).sum())
    if n_unknown_home > 0:
        print(f"WARNING: {n_unknown_home} games with unknown home/away from MATCHUP")

    games = build_game_pairs(df)
    print(f"Joined into {len(games)} game records")

    # Sanity: home_margin vs actual_margin should match for all rows
    mismatch = (games["home_margin"] != games["actual_margin"]).sum()
    if mismatch > 0:
        print(f"WARNING: {mismatch} games where home_margin != home_pts - away_pts")

    win_rate = compute_team_strength(games)
    print(f"Computed season win rates for {len(win_rate)} team-seasons")

    coef = fit_coefficients(games, win_rate)
    print()
    print(f"League avg HCA:       {coef['league_avg_hca']:.3f}")
    print(f"B2B penalty (home):   {coef['b2b_penalty_home']:.3f}")
    print(f"B2B penalty (away):   {coef['b2b_penalty_away']:.3f}")
    print(f"Rest bonus (home):    {coef['rest_day_bonus_home']:.3f} per day")
    print(f"Rest bonus (away):    {coef['rest_day_bonus_away']:.3f} per day")
    print(f"OLS R²:               {coef['ols_r_squared']:.3f}")
    print()
    print("Top 5 highest team HCA (shrunk):")
    sorted_hca = sorted(coef["team_hca"].items(), key=lambda kv: -kv[1])
    for team, hca in sorted_hca[:5]:
        n = coef["home_games_per_team"].get(team, 0)
        print(f"  {team}: {hca:+.3f} ({n} home games)")
    print("Bottom 5 lowest team HCA (shrunk):")
    for team, hca in sorted_hca[-5:]:
        n = coef["home_games_per_team"].get(team, 0)
        print(f"  {team}: {hca:+.3f} ({n} home games)")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(coef, indent=2))
    print()
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
