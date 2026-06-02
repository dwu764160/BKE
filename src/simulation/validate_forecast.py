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
    compute_game_distribution_with_context,
)
import dataclasses

from src.data.schema_contract import load_standardized

from src.simulation.simulation_config import (
    FORECAST_TEAM_FEATURES_PATH,
    HISTORICAL_DIR,
    REPORTS_DIR,
    YTD_RATINGS_PATH,
)


# ═════════════════════════════════════════════════════════════════════
# Step 3 — Rest / B2B lookup helper
# ═════════════════════════════════════════════════════════════════════

def build_rest_lookup(season: str, game_logs_path: Path = None) -> dict:
    """Return {game_id: {home_b2b, away_b2b, home_3in4, away_3in4,
                          home_days_rest, away_days_rest}} for one season.

    Convention: days_rest = calendar_day_diff - 1  (0 = B2B, 1 = standard).
    Consistent with fit_rest_hca_coefficients.py so fitted coefficients apply correctly.
    """
    gl_path = game_logs_path or HISTORICAL_DIR / "team_game_logs.parquet"
    if not gl_path.exists():
        return {}

    gl = load_standardized(gl_path)
    gl = gl[gl["season"] == season].copy()
    if gl.empty:
        return {}

    gl["game_date"] = pd.to_datetime(gl["game_date"])
    gl = gl.sort_values(["team_abbreviation", "game_date"]).reset_index(drop=True)

    gl["_prev"] = gl.groupby("team_abbreviation")["game_date"].shift(1)
    raw_diff = (gl["game_date"] - gl["_prev"]).dt.days - 1
    gl["days_rest"] = raw_diff.fillna(3).clip(lower=0, upper=7).astype(int)
    gl["is_b2b"] = gl["days_rest"] == 0

    gl["_prev2"] = gl.groupby("team_abbreviation")["game_date"].shift(2)
    span = (gl["game_date"] - gl["_prev2"]).dt.days.fillna(99)
    gl["is_3in4"] = (span <= 3) & gl["is_b2b"]

    per_team: dict = {}
    for _, row in gl.iterrows():
        per_team[(str(row["game_id"]), str(row["team_abbreviation"]).upper())] = {
            "days_rest": int(row["days_rest"]),
            "is_b2b":    bool(row["is_b2b"]),
            "is_3in4":   bool(row["is_3in4"]),
        }

    _def = {"days_rest": 1, "is_b2b": False, "is_3in4": False}
    lookup: dict = {}
    home_rows = gl[gl["matchup"].str.contains("vs.", na=False)]
    for _, row in home_rows.iterrows():
        gid  = str(row["game_id"])
        home = str(row["team_abbreviation"]).upper()
        parts = str(row["matchup"]).split(" vs. ")
        away  = parts[1].strip().upper() if len(parts) > 1 else ""
        h = per_team.get((gid, home), _def)
        a = per_team.get((gid, away), _def)
        lookup[gid] = {
            "home_b2b":       int(h["is_b2b"]),
            "away_b2b":       int(a["is_b2b"]),
            "home_3in4":      int(h["is_3in4"]),
            "away_3in4":      int(a["is_3in4"]),
            "home_days_rest": h["days_rest"],
            "away_days_rest": a["days_rest"],
        }
    return lookup


def _get_dist(
    game: Game,
    team_params: dict,
    config: SimConfig,
    rest_lookup: dict,
    ytd_lookup: dict = None,
) -> dict:
    """Compute game distribution, applying YTD blending and rest context when available."""
    home_p = team_params[game.home_team]
    away_p = team_params[game.away_team]

    if ytd_lookup:
        game_ytd = ytd_lookup.get(game.game_id, {})
        if game.home_team in game_ytd:
            home_p = dataclasses.replace(home_p, mu=game_ytd[game.home_team])
        if game.away_team in game_ytd:
            away_p = dataclasses.replace(away_p, mu=game_ytd[game.away_team])

    ctx = rest_lookup.get(game.game_id) if rest_lookup else None
    if ctx:
        return compute_game_distribution_with_context(
            home_team=home_p,
            away_team=away_p,
            is_b2b_home=ctx["home_b2b"],
            is_b2b_away=ctx["away_b2b"],
            days_rest_home=ctx["home_days_rest"],
            days_rest_away=ctx["away_days_rest"],
            is_3in4_home=ctx["home_3in4"],
            is_3in4_away=ctx["away_3in4"],
            config=config,
        )
    return compute_game_distribution(
        home_p,
        away_p,
        is_home_a=True,
        config=config,
    )


# ═════════════════════════════════════════════════════════════════════
# Step 4 — YTD blending lookup helper
# ═════════════════════════════════════════════════════════════════════

def load_ytd_lookup(seasons: list, ytd_path: Path = None) -> dict:
    """Return {game_id: {team_abbr: blended_mu}} for the given seasons.

    Built by scripts/build_ytd_team_ratings.py. Returns empty dict if the
    file doesn't exist (falls back to static preseason ratings).
    """
    p = ytd_path or YTD_RATINGS_PATH
    if not p.exists():
        return {}

    df = pd.read_parquet(p)
    df = df[df["season"].isin(seasons)].copy()
    if df.empty:
        return {}

    lookup: dict = {}
    for _, row in df.iterrows():
        gid = str(row["game_id"])
        team = str(row["team_abbreviation"]).upper()
        mu = float(row["blended_mu"])
        if gid not in lookup:
            lookup[gid] = {}
        lookup[gid][team] = mu
    return lookup


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
    rest_lookup: dict = None,
    ytd_lookup: dict = None,
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

        dist = _get_dist(game, team_params, config, rest_lookup, ytd_lookup)
        win_prob = dist.get("win_prob_home", dist.get("win_prob_a", 0.5))
        probs.append(win_prob)
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
    rest_lookup: dict = None,
    ytd_lookup: dict = None,
) -> List[Dict]:
    """10-bin calibration: does the model's X% confidence match observed X% win rate?"""
    probs = []
    actuals = []

    for game in schedule:
        if game.home_team not in team_params or game.away_team not in team_params:
            continue
        if np.isnan(game.home_win):
            continue

        dist = _get_dist(game, team_params, config, rest_lookup, ytd_lookup)
        win_prob = dist.get("win_prob_home", dist.get("win_prob_a", 0.5))
        probs.append(win_prob)
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
# Game-level prediction export (for CLV measurement, market testing)
# ═════════════════════════════════════════════════════════════════════

def export_game_predictions(
    schedule: List[Game],
    team_params: Dict[str, TeamParams],
    config: SimConfig,
    rest_lookup: dict = None,
    ytd_lookup: dict = None,
) -> pd.DataFrame:
    """
    Export per-game predictions in CLV format.

    Returns DataFrame with columns:
      - game_date (str, YYYY-MM-DD)
      - home_team (str, 3-letter abbreviation)
      - away_team (str, 3-letter abbreviation)
      - bke_home_win_prob (float, 0–1)
      - home_result (int, 1=home win, 0=away win)

    Only includes games where:
      - Both teams are in team_params
      - Actual result (home_win) is known (not NaN)
    """
    rows = []
    for game in schedule:
        if game.home_team not in team_params or game.away_team not in team_params:
            continue
        if np.isnan(game.home_win):
            continue

        dist = _get_dist(game, team_params, config, rest_lookup, ytd_lookup)
        win_prob = dist.get("win_prob_home", dist.get("win_prob_a", 0.5))
        rows.append({
            "game_date": game.date,
            "home_team": game.home_team,
            "away_team": game.away_team,
            "bke_home_win_prob": float(win_prob),
            "home_result": int(game.home_win),
        })

    return pd.DataFrame(rows)


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

    ytd_lookup_all = load_ytd_lookup(available_seasons)
    if ytd_lookup_all:
        print(f"YTD lookup loaded: {len(ytd_lookup_all)} game entries across seasons")
    else:
        print("YTD lookup not found — using static preseason ratings (run build_ytd_team_ratings.py)")

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
    all_game_predictions = []  # For CLV export

    for season in available_seasons:
        params = forecast_params[season]

        # Check that actual game logs exist for this season
        gl_path = HISTORICAL_DIR / "team_game_logs.parquet"
        if not gl_path.exists():
            print(f"  {season}: game logs missing — skipping")
            continue

        gl = load_standardized(gl_path)
        if season not in gl["season"].values:
            print(f"  {season}: no actual game logs — skipping (game logs cover {sorted(gl['season'].unique())})")
            continue

        schedule = build_schedule(season)
        if not schedule:
            print(f"  {season}: empty schedule — skipping")
            continue

        rest_lookup = build_rest_lookup(season)
        n_b2b = sum(1 for v in rest_lookup.values() if v["home_b2b"] or v["away_b2b"])
        print(f"\n  {season} (projected ratings vs. actual games):")
        if rest_lookup:
            print(f"    Rest context: {len(rest_lookup)} games, {n_b2b} with ≥1 B2B team "
                  f"({n_b2b / len(rest_lookup):.1%})")

        # Pass the full ytd_lookup; _get_dist misses on game_ids not in the dict (no-op).
        n_ytd = sum(1 for g in schedule if str(g.game_id) in ytd_lookup_all)
        if ytd_lookup_all and n_ytd:
            print(f"    YTD blending: {n_ytd} matching game entries")

        metrics = compute_game_metrics(schedule, params, config, rest_lookup=rest_lookup,
                                       ytd_lookup=ytd_lookup_all)
        cal_bins = compute_calibration(schedule, params, config, rest_lookup=rest_lookup,
                                       ytd_lookup=ytd_lookup_all)

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
                    dist = _get_dist(game, params, config, rest_lookup, ytd_lookup_all)
                    win_prob = dist.get("win_prob_home", dist.get("win_prob_a", 0.5))
                    all_probs.append(win_prob)
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

        # Collect game predictions for CLV export
        season_predictions = export_game_predictions(schedule, params, config,
                                                     rest_lookup=rest_lookup,
                                                     ytd_lookup=ytd_lookup_all)
        all_game_predictions.append(season_predictions)

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

    # Save game predictions for CLV measurement
    if all_game_predictions:
        game_preds_df = pd.concat(all_game_predictions, ignore_index=True)
        game_preds_path = REPORTS_DIR / "bke_game_forecasts.parquet"
        game_preds_df.to_parquet(game_preds_path, index=False)
        print(f"\nExported {len(game_preds_df)} game predictions to {game_preds_path}")
        print(f"  Schema: game_date, home_team, away_team, bke_home_win_prob, home_result")

    # Save validation metrics
    out_path = REPORTS_DIR / "forecast_game_validation.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
