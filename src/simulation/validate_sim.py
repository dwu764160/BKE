"""
src/simulation/validate_sim.py
=============================================================================
Simulation Core — Step 1, Layer 6: Validation

Validates simulation predictions against real NBA records:
  A. Game-Level Validation: Brier score, log loss, calibration
  B. Margin Validation: RMSE, distribution comparison
  C. Season-Level Validation: MAE, RMSE, correlation (pred vs actual wins)
  D. Overconfidence Test: calibration bins
  E. Parameter sensitivity report

Inputs:
  data/processed/player_eval/team_feature_aggregation.parquet
  data/historical/team_game_logs.parquet

Output:
  reports/simulation_step1_validation.json

Usage:
  python3 src/simulation/validate_sim.py
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
    load_team_params,
)
from src.simulation.simulation_config import REPORTS_DIR


# ═════════════════════════════════════════════════════════════════════
# A. Game-Level Validation
# ═════════════════════════════════════════════════════════════════════

def validate_game_level(
    schedule: List[Game],
    team_params: Dict[str, TeamParams],
    config: SimConfig,
) -> Dict:
    """Compute game-level prediction metrics."""
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
        if not np.isnan(game.home_margin):
            actual_margins.append(game.home_margin)
        else:
            actual_margins.append(None)

    probs = np.array(probs)
    actuals = np.array(actuals)
    pred_margins = np.array(pred_margins)

    # Clip probabilities to avoid log(0)
    eps = 1e-10
    probs_clipped = np.clip(probs, eps, 1 - eps)

    # Brier score
    brier = float(np.mean((probs - actuals) ** 2))

    # Log loss
    log_loss = float(-np.mean(
        actuals * np.log(probs_clipped) + (1 - actuals) * np.log(1 - probs_clipped)
    ))

    # Accuracy (threshold at 0.5)
    predictions = (probs >= 0.5).astype(float)
    accuracy = float(np.mean(predictions == actuals))

    # Home win rate
    home_win_rate = float(np.mean(actuals))
    avg_predicted_home_win = float(np.mean(probs))

    # Margin RMSE (where actual margins are available)
    valid_margins = [(p, a) for p, a in zip(pred_margins, actual_margins) if a is not None]
    margin_rmse = None
    margin_mae = None
    if valid_margins:
        pm = np.array([v[0] for v in valid_margins])
        am = np.array([v[1] for v in valid_margins])
        margin_rmse = float(np.sqrt(np.mean((pm - am) ** 2)))
        margin_mae = float(np.mean(np.abs(pm - am)))

    return {
        "n_games": len(probs),
        "brier_score": round(brier, 6),
        "log_loss": round(log_loss, 6),
        "accuracy": round(accuracy, 4),
        "home_win_rate_actual": round(home_win_rate, 4),
        "home_win_rate_predicted": round(avg_predicted_home_win, 4),
        "margin_rmse": round(margin_rmse, 4) if margin_rmse else None,
        "margin_mae": round(margin_mae, 4) if margin_mae else None,
    }


# ═════════════════════════════════════════════════════════════════════
# B. Calibration — Overconfidence Test
# ═════════════════════════════════════════════════════════════════════

def validate_calibration(
    schedule: List[Game],
    team_params: Dict[str, TeamParams],
    config: SimConfig,
    n_bins: int = 10,
) -> List[Dict]:
    """Check if predicted X% games win approximately X% of the time."""
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
            actual_rate = float(np.mean(actuals[mask]))
            predicted_rate = float(np.mean(probs[mask]))
            bins.append({
                "bin_low": round(float(edges[i]), 2),
                "bin_high": round(float(edges[i + 1]), 2),
                "count": count,
                "predicted_rate": round(predicted_rate, 4),
                "actual_rate": round(actual_rate, 4),
                "calibration_error": round(abs(predicted_rate - actual_rate), 4),
            })
    return bins


# ═════════════════════════════════════════════════════════════════════
# C. Season-Level Validation
# ═════════════════════════════════════════════════════════════════════

def validate_season_level(
    season_results_path: Path,
) -> Dict:
    """Compare projected wins vs actual wins from pre-computed season sim results."""
    if not season_results_path.exists():
        return {"error": "Season results not found — run season_sim.py first"}

    data = json.loads(season_results_path.read_text(encoding="utf-8"))
    all_pred = []
    all_actual = []
    per_season = {}

    for season, sdata in data.get("seasons", {}).items():
        pred = []
        actual = []
        for team in sdata.get("team_results", []):
            if "actual_wins" in team:
                pred.append(team["projected_wins"])
                actual.append(team["actual_wins"])
        if pred:
            pred_arr = np.array(pred)
            act_arr = np.array(actual)
            errors = pred_arr - act_arr
            per_season[season] = {
                "n_teams": len(pred),
                "mae": round(float(np.mean(np.abs(errors))), 2),
                "rmse": round(float(np.sqrt(np.mean(errors ** 2))), 2),
                "correlation": round(float(np.corrcoef(pred_arr, act_arr)[0, 1]), 4),
                "mean_error": round(float(np.mean(errors)), 2),
                "max_overestimate": round(float(np.max(errors)), 1),
                "max_underestimate": round(float(np.min(errors)), 1),
            }
            all_pred.extend(pred)
            all_actual.extend(actual)

    overall = {}
    if all_pred:
        all_pred = np.array(all_pred)
        all_actual = np.array(all_actual)
        all_errors = all_pred - all_actual
        overall = {
            "n_team_seasons": len(all_pred),
            "mae": round(float(np.mean(np.abs(all_errors))), 2),
            "rmse": round(float(np.sqrt(np.mean(all_errors ** 2))), 2),
            "correlation": round(float(np.corrcoef(all_pred, all_actual)[0, 1]), 4),
            "mean_error": round(float(np.mean(all_errors)), 2),
        }

    return {
        "overall": overall,
        "per_season": per_season,
    }


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════

def main() -> None:
    print("Simulation Core — Step 1: Validation (Layer 6)")
    config = SimConfig()

    all_params = load_team_params()
    full_validation = {
        "config": {
            "sigma_league": config.sigma_league,
            "home_court_advantage": config.home_court_advantage,
        },
        "game_level": {},
        "calibration": {},
        "season_level": {},
    }

    # A + B: Game-level validation and calibration per season
    for season in sorted(all_params.keys()):
        print(f"\n  {season}:")
        params = all_params[season]
        schedule = build_schedule(season)

        game_val = validate_game_level(schedule, params, config)
        cal_bins = validate_calibration(schedule, params, config)

        full_validation["game_level"][season] = game_val
        full_validation["calibration"][season] = cal_bins

        print(f"    Games: {game_val['n_games']}")
        print(f"    Brier: {game_val['brier_score']:.4f}, Log Loss: {game_val['log_loss']:.4f}")
        print(f"    Accuracy: {game_val['accuracy']:.3f}")
        print(f"    Margin RMSE: {game_val.get('margin_rmse', 'N/A')}")
        print(f"    Home win rate: actual={game_val['home_win_rate_actual']:.3f}, "
              f"predicted={game_val['home_win_rate_predicted']:.3f}")

    # C: Season-level validation
    season_results_path = REPORTS_DIR / "simulation_step1_season_results.json"
    season_val = validate_season_level(season_results_path)
    full_validation["season_level"] = season_val

    if "overall" in season_val and season_val["overall"]:
        print(f"\n  Season-Level Overall:")
        print(f"    MAE: {season_val['overall']['mae']}")
        print(f"    RMSE: {season_val['overall']['rmse']}")
        print(f"    Correlation: {season_val['overall']['correlation']}")

    # D: Aggregate calibration across all seasons
    all_cal_errors = []
    for season, bins in full_validation["calibration"].items():
        for b in bins:
            all_cal_errors.append(b["calibration_error"] * b["count"])
    if all_cal_errors:
        total_games = sum(
            b["count"]
            for bins in full_validation["calibration"].values()
            for b in bins
        )
        if total_games > 0:
            full_validation["aggregate_calibration_error"] = round(
                sum(all_cal_errors) / total_games, 4
            )

    # Save
    out_path = REPORTS_DIR / "simulation_step1_validation.json"
    out_path.write_text(json.dumps(full_validation, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
