"""
scripts/build_ytd_team_ratings.py
=============================================================================
Step 4 — YTD Blended Team Ratings

For each team-game in team_game_logs.parquet, compute a blended net-rating
that transitions from the preseason BKE projection toward the team's actual
year-to-date performance as the season progresses.

Blend formula:
  alpha = sqrt(games_played_before / 82)          # 0 at game 1 → ~1.0 at game 82
  ytd_bke = ytd_avg_margin * (preseason_std / 6.0) # scale margin pts → BKE units
  blended_mu = (1 - alpha) * preseason_mu + alpha * ytd_bke

Leakage safety: ytd_avg_margin uses only the prior G-1 games via cumsum().shift(1).
The current game's margin is never used to predict itself.

Output: data/processed/forecast/team_ratings_ytd.parquet
  Columns: season, game_id, game_date, team_abbreviation,
           games_played_before, ytd_avg_margin, preseason_mu, alpha, blended_mu

Usage:
  python3 scripts/build_ytd_team_ratings.py
=============================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.schema_contract import load_standardized

from src.simulation.simulation_config import (
    FORECAST_DIR,
    HISTORICAL_DIR,
    YTD_RATINGS_PATH,
)

GAME_LOGS_PATH = HISTORICAL_DIR / "team_game_logs.parquet"
PROJ_FEATURES_PATH = FORECAST_DIR / "projected_team_features.parquet"
PROJ_FEATURES_2026_PATH = FORECAST_DIR / "projected_team_features_v40_2025-26.parquet"

# Cross-team std of full-season avg margins (pts/game), fitted from 2022-25 data.
# Used to convert actual margin to BKE-rating space for blending.
LEAGUE_MARGIN_STD_PRIOR = 6.0


def _load_projected(season: str) -> pd.DataFrame:
    """Load projected team features for a season from either features file."""
    if season == "2025-26" and PROJ_FEATURES_2026_PATH.exists():
        df = load_standardized(PROJ_FEATURES_2026_PATH)
        df["season"] = season
        df["team_abbreviation"] = df["team_abbreviation"].str.upper()
        return df

    if not PROJ_FEATURES_PATH.exists():
        return pd.DataFrame()
    df = load_standardized(PROJ_FEATURES_PATH)
    df["season"] = df["season"].astype(str)
    df["team_abbreviation"] = df["team_abbreviation"].str.upper()
    return df[df["season"] == season]


def build_season_ytd(gl_season: pd.DataFrame, season: str) -> pd.DataFrame:
    """Compute per-game blended ratings for one season.

    Returns a DataFrame with one row per (team, game), indexed by game_id.
    """
    proj = _load_projected(season)
    if proj.empty or "team_net_rating_projected" not in proj.columns:
        return pd.DataFrame()

    proj_idx = proj.set_index("team_abbreviation")["team_net_rating_projected"]

    preseason_std = proj_idx.std()
    if preseason_std < 0.1:
        preseason_std = 0.1  # safety floor for old compressed BKE versions
    margin_to_bke = preseason_std / LEAGUE_MARGIN_STD_PRIOR

    gl = gl_season.copy()
    gl["game_date"] = pd.to_datetime(gl["game_date"])
    gl["team_abbreviation"] = gl["team_abbreviation"].str.upper()
    gl = gl.sort_values(["team_abbreviation", "game_date"]).reset_index(drop=True)

    # OOS cumulative margin: shift(1) excludes the current game
    gl["_n"] = gl.groupby("team_abbreviation").cumcount()  # 0 = first game of season
    gl["_cum"] = gl.groupby("team_abbreviation")["margin"].cumsum()
    gl["_cum_before"] = gl.groupby("team_abbreviation")["_cum"].shift(1)

    gl["games_played_before"] = gl["_n"]
    gl["ytd_avg_margin"] = (gl["_cum_before"] / gl["games_played_before"]).fillna(0.0)
    gl["ytd_bke"] = gl["ytd_avg_margin"] * margin_to_bke
    gl["alpha"] = np.sqrt(gl["games_played_before"].clip(lower=0).astype(float) / 82.0)

    gl["preseason_mu"] = gl["team_abbreviation"].map(lambda t: float(proj_idx.get(t, 0.0)))
    gl["blended_mu"] = (1.0 - gl["alpha"]) * gl["preseason_mu"] + gl["alpha"] * gl["ytd_bke"]

    out = gl[
        ["season", "game_id", "game_date", "team_abbreviation",
         "games_played_before", "ytd_avg_margin", "preseason_mu", "alpha", "blended_mu"]
    ].copy()
    out["game_date"] = out["game_date"].dt.strftime("%Y-%m-%d")
    out["game_id"] = out["game_id"].astype(str)
    return out


def main() -> None:
    print("=" * 60)
    print("Building YTD blended team ratings (Step 4)")
    print("=" * 60)

    if not GAME_LOGS_PATH.exists():
        print(f"ERROR: {GAME_LOGS_PATH} not found")
        return

    gl = load_standardized(GAME_LOGS_PATH)

    # margin column was only backfilled from 2022-23 onward; derive from plus_minus for older seasons.
    # plus_minus is identical to margin for all populated rows (verified).
    if "plus_minus" in gl.columns:
        gl["margin"] = gl["margin"].fillna(gl["plus_minus"])

    # All seasons that now have margin data (skips 2017-18: no projected features)
    margin_seasons = sorted(
        s for s in gl["season"].unique() if not gl[gl["season"] == s]["margin"].isna().all()
    )
    print(f"Seasons with margin data: {margin_seasons}")

    all_parts = []
    for season in margin_seasons:
        proj = _load_projected(season)
        if proj.empty:
            print(f"  {season}: no projected features — skipping")
            continue

        gl_season = gl[gl["season"] == season].copy()
        part = build_season_ytd(gl_season, season)
        if part.empty:
            print(f"  {season}: build failed — skipping")
            continue

        n_games = part["game_id"].nunique()
        n_teams = part["team_abbreviation"].nunique()
        pre_std = proj["team_net_rating_projected"].std()
        scale = pre_std / LEAGUE_MARGIN_STD_PRIOR

        print(f"  {season}: {n_teams} teams, {n_games} unique games | "
              f"preseason_std={pre_std:.3f} → scale={scale:.4f}")

        # Quick sanity — alpha at game 30 (game #29 played before)
        g30 = part[part["games_played_before"] == 29]
        if len(g30):
            alpha_30 = g30["alpha"].iloc[0]
            print(f"    alpha at game 30: {alpha_30:.3f} (expected ~0.595)")

        all_parts.append(part)

    if not all_parts:
        print("ERROR: No seasons could be built")
        return

    combined = pd.concat(all_parts, ignore_index=True)
    print(f"\nTotal rows: {len(combined)}, seasons: {sorted(combined['season'].unique())}")

    YTD_RATINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(YTD_RATINGS_PATH, index=False)
    print(f"Saved to {YTD_RATINGS_PATH}")

    # Quick validation: blended_mu should track preseason early and diverge late
    for season in combined["season"].unique():
        sub = combined[combined["season"] == season]
        early = sub[sub["games_played_before"] <= 5]["blended_mu"]
        late  = sub[sub["games_played_before"] >= 70]["blended_mu"]
        pre   = sub["preseason_mu"]
        print(f"\n  {season} sanity:")
        print(f"    blended (early ≤5 games):   mean={early.mean():.3f}, std={early.std():.3f}")
        print(f"    blended (late ≥70 games):   mean={late.mean():.3f}, std={late.std():.3f}")
        print(f"    preseason_mu:               mean={pre.mean():.3f}, std={pre.std():.3f}")


if __name__ == "__main__":
    main()
