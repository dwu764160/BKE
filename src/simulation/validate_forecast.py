"""
src/simulation/validate_forecast.py
=============================================================================
Phase 0 Walk-Forward Game-Level Harness

Validates the forecast pipeline's team ratings against actual next-season games.
This is a GENUINE walk-forward test: projected team features (built from season-N
player profiles) are evaluated against actual season-N+1 game results.

Key distinction from validate_sim.py (backtest):
  - validate_sim.py: same-season team features → retrodiction
  - THIS script: projected_team_features.parquet → true out-of-sample forecast

Walk-forward transitions covered:
  - 2023-24 actual games, using ratings projected from 2022-23 data
  - 2024-25 actual games, using ratings projected from 2023-24 data

Inputs:
  data/processed/forecast/projected_team_features.parquet
  data/historical/team_game_logs.parquet

Output:
  reports/forecast_game_validation.json

Usage:
  python3 src/simulation/validate_forecast.py
=============================================================================
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.simulation.game_model import (
    Game,
    SimConfig,
    TeamParams,
    build_schedule,
    compute_game_distribution,
)
from src.simulation.simulation_config import (
    FORECAST_TEAM_FEATURES_PATH,
    HISTORICAL_DIR,
    REPORTS_DIR,
)


# ═════════════════════════════════════════════════════════════════════
# Load projected team params (forecast pipeline output)
# ═════════════════════════════════════════════════════════════════════

def load_forecast_team_params(
    features_path: Path = None,
) -> Dict[str, Dict[str, TeamParams]]:
    """Load projected team parameters from forecast pipeline output.

    Returns: {season: {team_abbr: TeamParams}}
    where `season` is the PROJECTED season (the season the ratings apply to).
    """
    src = features_path or FORECAST_TEAM_FEATURES_PATH
    if not src.exists():
        raise FileNotFoundError(
            f"Projected team features not found: {src}\n"
            "Run the forecast pipeline first to generate projected ratings."
        )

    tf = pd.read_parquet(src)
    tf["season"] = tf["season"].astype(str)
    tf["team_abbreviation"] = tf["team_abbreviation"].astype(str).str.upper()

    result = {}
    for _, row in tf.iterrows():
        season = row["season"]
        team = row["team_abbreviation"]
        if team in ("NAN", "nan", "", "NONE"):
            continue
        if season not in result:
            result[season] = {}
        result[season][team] = TeamParams(
            team_abbreviation=team,
            season=season,
            mu=float(row["team_net_rating_projected"]),
            sigma=float(row["vol_total"]),
        )
    return result


# ═════════════════════════════════════════════════════════════════════
# Game-level metrics (same as validate_sim.py)
# ═════════════════════════════════════════════════════════════════════

def compute_game_metrics(
    schedule: List[Game],
    team_params: Dict[str, TeamParams],
    config: SimConfig,
) -> Dict:
    """Brier, log-loss, accuracy, margin RMSE for a set of games."""
    probs = []
    actuals = []
    pred_margins = []
    actual_margins = []

    for game in schedule:
        if game.home_team not in team_params or game.away_team not in team_params:
            continue
        if np.isnan(game.home_win):
            continue

        dist = compute_game_distribution(
            team_params[game.home_team],
            team_params[game.away_team],
            is_home_a=True,
            config=config,
        )
        probs.append(dist["win_prob_a"])
        actuals.append(game.home_win)
        pred_margins.append(dist["delta_mu"])
        actual_margins.append(game.home_margin if not np.isnan(game.home_margin) else None)

    if not probs:
        return {"error": "No matching games found"}

    probs = np.array(probs)
    actuals = np.array(actuals)
    eps = 1e-10
    probs_clipped = np.clip(probs, eps, 1 - eps)

    brier = float(np.mean((probs - actuals) ** 2))
    log_loss = float(-np.mean(
        actuals * np.log(probs_clipped) + (1 - actuals) * np.log(1 - probs_clipped)
    ))
    accuracy = float(np.mean((probs >= 0.5).astype(float) == actuals))
    home_win_rate = float(np.mean(actuals))
    avg_predicted = float(np.mean(probs))

    valid = [(p, a) for p, a in zip(pred_margins, actual_margins) if a is not None]
    margin_rmse = None
    margin_mae = None
    if valid:
        pm = np.array([v[0] for v in valid])
        am = np.array([v[1] for v in valid])
        margin_rmse = float(np.sqrt(np.mean((pm - am) ** 2)))
        margin_mae = float(np.mean(np.abs(pm - am)))

    return {
        "n_games": len(probs),
        "brier_score": round(brier, 6),
        "log_loss": round(log_loss, 6),
        "accuracy": round(accuracy, 4),
        "home_win_rate_actual": round(home_win_rate, 4),
        "home_win_rate_predicted": round(avg_predicted, 4),
        "margin_rmse": round(margin_rmse, 4) if margin_rmse is not None else None,
        "margin_mae": round(margin_mae, 4) if margin_mae is not None else None,
    }


def compute_calibration(
    schedule: List[Game],
    team_params: Dict[str, TeamParams],
    config: SimConfig,
    n_bins: int = 10,
) -> List[Dict]:
    """10-bin calibration: does the model's X% confidence match observed X% win rate?"""
    probs = []
    actuals = []

    for game in schedule:
        if game.home_team not in team_params or game.away_team not in team_params:
            continue
        if np.isnan(game.home_win):
            continue

        dist = compute_game_distribution(
            team_params[game.home_team],
            team_params[game.away_team],
            is_home_a=True,
            config=config,
        )
        probs.append(dist["win_prob_a"])
        actuals.append(game.home_win)

    if not probs:
        return []

    probs = np.array(probs)
    actuals = np.array(actuals)

    bins = []
    edges = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        mask = (probs >= edges[i]) & (probs < edges[i + 1])
        if i == n_bins - 1:
            mask = (probs >= edges[i]) & (probs <= edges[i + 1])
        count = int(mask.sum())
        if count > 0:
            bins.append({
                "bin_low": round(float(edges[i]), 2),
                "bin_high": round(float(edges[i + 1]), 2),
                "count": count,
                "predicted_rate": round(float(np.mean(probs[mask])), 4),
                "actual_rate": round(float(np.mean(actuals[mask])), 4),
                "calibration_error": round(abs(float(np.mean(probs[mask])) - float(np.mean(actuals[mask]))), 4),
            })
    return bins


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-path", type=Path, default=None,
                        help="Override projected_team_features.parquet path")
    parser.add_argument("--exclude-seasons", nargs="*", default=[],
                        help="Season strings to skip (e.g. 2019-20 2020-21)")
    args = parser.parse_args()

    print("Phase 0 Walk-Forward Forecast Validation")
    print("=" * 50)

    config = SimConfig()
    forecast_params = load_forecast_team_params(features_path=args.features_path)
    for s in args.exclude_seasons:
        forecast_params.pop(s, None)
    available_seasons = sorted(forecast_params.keys())

    if not available_seasons:
        print("ERROR: No projected team features found.")
        return

    print(f"Projected seasons available: {available_seasons}")

    result = {
        "mode": "walk_forward_forecast",
        "description": (
            "Each season's team ratings come from the PRIOR season's forecast pipeline output. "
            "Games evaluated are ACTUAL next-season results. This is genuine out-of-sample validation."
        ),
        "config": {
            "sigma_league": config.sigma_league,
            "home_court_advantage": config.home_court_advantage,
        },
        "transitions": {},
        "game_level": {},
        "calibration": {},
        "aggregate": {},
    }

    all_probs = []
    all_actuals = []
    all_cal_errors_weighted = []
    total_games_across = 0

    for season in available_seasons:
        params = forecast_params[season]

        # Check that actual game logs exist for this season
        gl_path = HISTORICAL_DIR / "team_game_logs.parquet"
        if not gl_path.exists():
            print(f"  {season}: game logs missing — skipping")
            continue

        gl = pd.read_parquet(gl_path)
        if season not in gl["SEASON"].values:
            print(f"  {season}: no actual game logs — skipping (game logs cover {sorted(gl['SEASON'].unique())})")
            continue

        schedule = build_schedule(season)
        if not schedule:
            print(f"  {season}: empty schedule — skipping")
            continue

        print(f"\n  {season} (projected ratings vs. actual games):")

        metrics = compute_game_metrics(schedule, params, config)
        cal_bins = compute_calibration(schedule, params, config)

        result["game_level"][season] = metrics
        result["calibration"][season] = cal_bins

        n_games = metrics.get("n_games", 0)
        brier = metrics.get("brier_score")
        log_loss = metrics.get("log_loss")
        acc = metrics.get("accuracy")

        print(f"    Games: {n_games}")
        if brier is not None:
            print(f"    Brier: {brier:.4f}, Log Loss: {log_loss:.4f}")
            print(f"    Accuracy: {acc:.3f}")
        margin_rmse = metrics.get("margin_rmse")
        if margin_rmse is not None:
            print(f"    Margin RMSE: {margin_rmse:.2f}")
        print(f"    Home win rate: actual={metrics.get('home_win_rate_actual'):.3f}, "
              f"predicted={metrics.get('home_win_rate_predicted'):.3f}")

        # Aggregate probabilities for cross-season Brier
        for game in schedule:
            if game.home_team in params and game.away_team in params:
                if not np.isnan(game.home_win):
                    dist = compute_game_distribution(
                        params[game.home_team], params[game.away_team],
                        is_home_a=True, config=config,
                    )
                    all_probs.append(dist["win_prob_a"])
                    all_actuals.append(game.home_win)

        for b in cal_bins:
            all_cal_errors_weighted.append(b["calibration_error"] * b["count"])
            total_games_across += b["count"]

        result["transitions"][season] = {
            "description": f"Projected from prior season, evaluated against actual {season} games",
            "n_games": n_games,
            "brier_score": brier,
            "log_loss": log_loss,
            "accuracy": acc,
        }

    # Aggregate across all walk-forward transitions
    if all_probs:
        all_probs = np.array(all_probs)
        all_actuals = np.array(all_actuals)
        eps = 1e-10
        probs_clipped = np.clip(all_probs, eps, 1 - eps)

        agg_brier = float(np.mean((all_probs - all_actuals) ** 2))
        agg_log_loss = float(-np.mean(
            all_actuals * np.log(probs_clipped) + (1 - all_actuals) * np.log(1 - probs_clipped)
        ))
        agg_accuracy = float(np.mean((all_probs >= 0.5).astype(float) == all_actuals))
        agg_cal_error = (
            round(sum(all_cal_errors_weighted) / total_games_across, 4)
            if total_games_across > 0 else None
        )

        result["aggregate"] = {
            "n_transitions": len(available_seasons),
            "n_games_total": len(all_probs),
            "brier_score": round(agg_brier, 6),
            "log_loss": round(agg_log_loss, 6),
            "accuracy": round(agg_accuracy, 4),
            "aggregate_calibration_error": agg_cal_error,
        }

        print(f"\n{'=' * 50}")
        print(f"Aggregate across {len(available_seasons)} walk-forward transition(s):")
        print(f"  Games:    {len(all_probs)}")
        print(f"  Brier:    {agg_brier:.4f}  (lower is better; chance = 0.25)")
        print(f"  Log Loss: {agg_log_loss:.4f}")
        print(f"  Accuracy: {agg_accuracy:.3f}")
        if agg_cal_error is not None:
            print(f"  Calibration Error: {agg_cal_error:.4f}")
        print()
        print("Note: these ratings were built from PRIOR-SEASON projected profiles.")
        print("Compare to validate_sim.py (same-season backtest) to quantify leakage.")

    # Save
    out_path = REPORTS_DIR / "forecast_game_validation.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
