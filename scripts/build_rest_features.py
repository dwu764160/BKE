"""
scripts/build_rest_features.py
=============================================================================
Computes per-game rest/schedule-stress features from team_game_logs.parquet.

Output: data/processed/forecast/game_rest_features.parquet

Schema:
  game_id, game_date, season, home_team, away_team,
  home_days_rest, away_days_rest,   -- 0 = B2B, 1 = standard, 2+ = extra rest
  home_b2b, away_b2b,               -- bool (days_rest == 0)
  home_3in4, away_3in4              -- bool (3rd game in 4 calendar days)

Convention (consistent with fit_rest_hca_coefficients.py):
  days_rest = (game_date - prev_game_date).days - 1
  is_b2b    = (days_rest == 0)     i.e., consecutive calendar days
  is_3in4   = calendar_span_2_games ≤ 3 AND is_b2b

Usage:
    python3 scripts/build_rest_features.py [--input ...] [--output ...]
=============================================================================
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.schema_contract import load_standardized  # noqa: E402

DEFAULT_INPUT  = REPO / "data/historical/team_game_logs.parquet"
DEFAULT_OUTPUT = REPO / "data/processed/forecast/game_rest_features.parquet"


def _compute_team_rest(df: pd.DataFrame) -> pd.DataFrame:
    """Add days_rest, is_b2b, is_3in4 per team-game row."""
    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["team_abbreviation", "game_date"]).reset_index(drop=True)

    df["_prev_date"] = df.groupby("team_abbreviation")["game_date"].shift(1)
    # Convention: days_rest = calendar_day_difference - 1  (0 = B2B, 1 = standard)
    raw_diff = (df["game_date"] - df["_prev_date"]).dt.days - 1
    df["days_rest"] = raw_diff.fillna(3).clip(lower=0, upper=7).astype(int)
    df["is_b2b"] = df["days_rest"] == 0

    df["_prev2_date"] = df.groupby("team_abbreviation")["game_date"].shift(2)
    calendar_span = (df["game_date"] - df["_prev2_date"]).dt.days.fillna(99)
    df["is_3in4"] = (calendar_span <= 3) & df["is_b2b"]

    return df


def build_game_rest_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per game (home perspective) with both-team rest features."""
    df = _compute_team_rest(df)

    # Per-team-game lookup
    per_team: dict[tuple[str, str], dict] = {}
    for _, row in df.iterrows():
        per_team[(str(row["game_id"]), str(row["team_abbreviation"]).upper())] = {
            "days_rest": int(row["days_rest"]),
            "is_b2b":    bool(row["is_b2b"]),
            "is_3in4":   bool(row["is_3in4"]),
        }

    _default = {"days_rest": 1, "is_b2b": False, "is_3in4": False}

    # Build one row per game from home-team rows only
    home_rows = df[df["matchup"].str.contains("vs.", na=False)].copy()
    rows = []
    for _, row in home_rows.iterrows():
        gid  = str(row["game_id"])
        home = str(row["team_abbreviation"]).upper()
        parts = str(row["matchup"]).split(" vs. ")
        away  = parts[1].strip().upper() if len(parts) > 1 else ""

        h = per_team.get((gid, home), _default)
        a = per_team.get((gid, away), _default)

        rows.append({
            "game_id":        gid,
            "game_date":      row["game_date"].strftime("%Y-%m-%d"),
            "season":         str(row["season"]),
            "home_team":      home,
            "away_team":      away,
            "home_days_rest": h["days_rest"],
            "away_days_rest": a["days_rest"],
            "home_b2b":       h["is_b2b"],
            "away_b2b":       a["is_b2b"],
            "home_3in4":      h["is_3in4"],
            "away_3in4":      a["is_3in4"],
        })

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"ERROR: {input_path} not found. Run data pipeline first.")
        sys.exit(1)

    print(f"Reading {input_path}...")
    df = load_standardized(input_path)
    print(f"  {len(df)} team-game rows, {df['season'].nunique()} seasons")

    out = build_game_rest_features(df)
    print(f"  Built {len(out)} game-level rest rows")

    # Validation: B2B frequency should be 15-25%
    b2b_rate = (out["home_b2b"] | out["away_b2b"]).mean()
    print(f"  Games with ≥1 B2B team: {b2b_rate:.1%}  (expect 15–25%)")

    _3in4_rate = (out["home_3in4"] | out["away_3in4"]).mean()
    print(f"  Games with ≥1 3-in-4 team: {_3in4_rate:.1%}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
