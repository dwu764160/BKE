"""
src/player_eval/calibrate_team_scale.py
=============================================================================
TEAM_SCALE Calibration Script (Issue 3c)

For each walk-forward transition, fits:
  actual_net_rating = a * projected_net_rating + b  (OLS, per transition)

Compares fitted slope (a) to DEFAULT_TEAM_SCALE / raw_talent_scale.

"Actual net rating" is estimated from observed win% via the inverse-normal
approximation: actual_net_rating ≈ σ_game_adj * Φ^{-1}(win%).

Outputs:
  reports/team_scale_calibration.json

Usage:
  python3 src/player_eval/calibrate_team_scale.py
=============================================================================
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.simulation.simulation_config import REPORTS_DIR
from src.player_eval.constants import DEFAULT_TEAM_SCALE

FORECAST_FEATURES = Path("data/processed/forecast/projected_team_features.parquet")
GAME_LOGS = Path("data/historical/team_game_logs.parquet")

# σ_game estimate: game-level std deviation of margin ≈ 11 pts (league typical)
# Used to convert win% → net rating via Φ^{-1}(win%) * sigma_game
# sigma_game = sqrt(sigma_home^2 + sigma_away^2 + sigma_league^2) ≈ 11
SIGMA_GAME_ESTIMATE = 11.0
WIN_PCT_CLIP = (0.10, 0.90)  # avoid ±∞ from Φ^{-1}


def compute_actual_net_ratings(gl: pd.DataFrame, season: str) -> pd.DataFrame:
    """Estimate actual team net rating from observed win% in a season."""
    season_gl = gl[gl["SEASON"] == season].copy()
    if season_gl.empty:
        return pd.DataFrame()

    # Wins and total games per team
    if "WL" not in season_gl.columns:
        return pd.DataFrame()

    wins = (
        season_gl[season_gl["WL"] == "W"]
        .groupby("TEAM_ABBREVIATION")
        .size()
        .reset_index(name="wins")
    )
    total = (
        season_gl.groupby("TEAM_ABBREVIATION")
        .size()
        .reset_index(name="gp")
    )
    rec = wins.merge(total, on="TEAM_ABBREVIATION", how="right").fillna({"wins": 0})
    rec["win_pct"] = rec["wins"] / rec["gp"]
    rec["team_abbreviation"] = rec["TEAM_ABBREVIATION"].astype(str).str.upper()

    # Actual net rating from win%
    clipped = rec["win_pct"].clip(*WIN_PCT_CLIP)
    rec["actual_net_rating"] = SIGMA_GAME_ESTIMATE * norm.ppf(clipped)

    return rec[["team_abbreviation", "win_pct", "gp", "actual_net_rating"]]


def fit_ols(x: np.ndarray, y: np.ndarray):
    """Fit y = a*x + b via OLS. Returns (a, b, r²)."""
    if len(x) < 5:
        return None, None, None
    X = np.column_stack([x, np.ones(len(x))])
    result = np.linalg.lstsq(X, y, rcond=None)
    a, b = result[0]
    y_pred = a * x + b
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(a), float(b), float(r2)


def main() -> None:
    print("TEAM_SCALE Calibration")
    print("=" * 50)

    if not FORECAST_FEATURES.exists():
        print(f"ERROR: {FORECAST_FEATURES} not found. Run forecast pipeline first.")
        return

    ptf = pd.read_parquet(FORECAST_FEATURES)
    gl = pd.read_parquet(GAME_LOGS)

    ptf["season"] = ptf["season"].astype(str)
    ptf["team_abbreviation"] = ptf["team_abbreviation"].astype(str).str.upper()
    gl["SEASON"] = gl["SEASON"].astype(str)

    seasons = sorted(ptf["season"].unique())
    print(f"Projected seasons: {seasons}")

    transition_results = []
    all_x = []
    all_y = []

    for season in seasons:
        proj = ptf[ptf["season"] == season][
            ["team_abbreviation", "team_net_rating_projected", "off_mean", "def_mean"]
        ].copy()

        actual = compute_actual_net_ratings(gl, season)
        if actual.empty:
            print(f"  {season}: No actual game log data — skipping")
            continue

        merged = proj.merge(actual, on="team_abbreviation", how="inner")
        if len(merged) < 10:
            print(f"  {season}: Only {len(merged)} teams — skipping")
            continue

        x = merged["team_net_rating_projected"].values
        y = merged["actual_net_rating"].values
        a, b, r2 = fit_ols(x, y)

        corr = float(np.corrcoef(x, y)[0, 1])
        proj_spread = float(np.std(x))
        actual_spread = float(np.std(y))
        spread_ratio = actual_spread / proj_spread if proj_spread > 0 else None

        print(f"\n  {season}:")
        print(f"    n_teams = {len(merged)}")
        print(f"    r = {corr:.3f}, r² = {r2:.3f}")
        print(f"    OLS: actual = {a:.4f} * projected + {b:.4f}")
        print(f"    Projected spread (std): {proj_spread:.2f}")
        print(f"    Actual spread (std):    {actual_spread:.2f}")
        if spread_ratio:
            print(f"    Spread ratio (actual/projected): {spread_ratio:.3f}")
            print(f"    → Implied TEAM_SCALE correction: current {DEFAULT_TEAM_SCALE:.1f} → recommended ~{DEFAULT_TEAM_SCALE * spread_ratio:.1f}")

        transition_results.append({
            "season": season,
            "n_teams": len(merged),
            "correlation": round(corr, 4),
            "r_squared": round(r2, 4),
            "ols_slope": round(a, 4),
            "ols_intercept": round(b, 4),
            "projected_std": round(proj_spread, 4),
            "actual_std": round(actual_spread, 4),
            "spread_ratio_actual_over_projected": round(spread_ratio, 4) if spread_ratio else None,
            "implied_team_scale_correction": round(DEFAULT_TEAM_SCALE * spread_ratio, 2) if spread_ratio else None,
        })

        all_x.extend(x.tolist())
        all_y.extend(y.tolist())

    # Aggregate across all transitions
    aggregate = {}
    if all_x:
        all_x_arr = np.array(all_x)
        all_y_arr = np.array(all_y)
        a_agg, b_agg, r2_agg = fit_ols(all_x_arr, all_y_arr)
        corr_agg = float(np.corrcoef(all_x_arr, all_y_arr)[0, 1])
        spread_ratio_agg = float(np.std(all_y_arr) / np.std(all_x_arr)) if np.std(all_x_arr) > 0 else None

        aggregate = {
            "n_team_seasons": len(all_x),
            "correlation": round(corr_agg, 4),
            "r_squared": round(r2_agg, 4),
            "ols_slope": round(a_agg, 4),
            "ols_intercept": round(b_agg, 4),
            "spread_ratio_actual_over_projected": round(spread_ratio_agg, 4) if spread_ratio_agg else None,
            "current_default_team_scale": DEFAULT_TEAM_SCALE,
            "implied_team_scale_correction": round(DEFAULT_TEAM_SCALE * spread_ratio_agg, 2) if spread_ratio_agg else None,
        }

        print(f"\n{'=' * 50}")
        print(f"Aggregate across {len(seasons)} transition(s):")
        print(f"  n_team_seasons = {len(all_x)}")
        print(f"  r = {corr_agg:.3f}, r² = {r2_agg:.3f}")
        print(f"  OLS: actual = {a_agg:.4f} * projected + {b_agg:.4f}")
        print(f"  Current DEFAULT_TEAM_SCALE = {DEFAULT_TEAM_SCALE}")
        if spread_ratio_agg:
            recommended = DEFAULT_TEAM_SCALE * spread_ratio_agg
            print(f"  Recommended correction: ~{recommended:.1f}")
            if abs(recommended - DEFAULT_TEAM_SCALE) > 2.0:
                print(f"  ⚠ Spread ratio differs significantly from 1.0. Consider updating DEFAULT_TEAM_SCALE.")
            else:
                print(f"  Current TEAM_SCALE within reasonable range.")

    output = {
        "note": (
            "Actual net rating estimated from win% via normal approximation "
            f"(sigma_game={SIGMA_GAME_ESTIMATE}). "
            "Only 2 walk-forward transitions available. More data needed for stable calibration."
        ),
        "sigma_game_used": SIGMA_GAME_ESTIMATE,
        "current_default_team_scale": DEFAULT_TEAM_SCALE,
        "transitions": transition_results,
        "aggregate": aggregate,
    }

    out_path = REPORTS_DIR / "team_scale_calibration.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
