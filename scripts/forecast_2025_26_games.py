"""
scripts/forecast_2025_26_games.py

Generate game-level forecasts for 2025-26 season using projected team features.
Then compute CLV vs Kalshi closing lines.

Usage:
    python3 scripts/forecast_2025_26_games.py
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.simulation.game_model import (
    Game,
    SimConfig,
    TeamParams,
    build_schedule,
    compute_game_distribution,
    compute_game_distribution_with_context,
)
import dataclasses

from src.simulation.simulation_config import FORECAST_DIR, REPORTS_DIR, YTD_RATINGS_PATH


def build_rest_lookup_from_schedule(kalshi_df: pd.DataFrame) -> dict:
    """Compute B2B and 3-in-4 flags from Kalshi game schedule.

    Keyed by (game_date_str, home_team, away_team).
    Convention: days_rest = calendar_day_diff - 1  (0 = B2B, 1 = standard).
    """
    home_games = kalshi_df[["game_date", "home_team", "away_team"]].copy()
    home_games["team"] = home_games["home_team"]
    away_games = kalshi_df[["game_date", "home_team", "away_team"]].copy()
    away_games["team"] = away_games["away_team"]

    all_games = pd.concat([home_games, away_games], ignore_index=True)
    all_games["game_date"] = pd.to_datetime(all_games["game_date"])
    all_games = all_games.sort_values(["team", "game_date"]).reset_index(drop=True)

    all_games["_prev"] = all_games.groupby("team")["game_date"].shift(1)
    raw_diff = (all_games["game_date"] - all_games["_prev"]).dt.days - 1
    all_games["days_rest"] = raw_diff.fillna(3).clip(lower=0, upper=7).astype(int)
    all_games["is_b2b"] = all_games["days_rest"] == 0

    all_games["_prev2"] = all_games.groupby("team")["game_date"].shift(2)
    span = (all_games["game_date"] - all_games["_prev2"]).dt.days.fillna(99)
    all_games["is_3in4"] = (span <= 3) & all_games["is_b2b"]

    per_team: dict = {}
    for _, row in all_games.iterrows():
        key = (row["game_date"].strftime("%Y-%m-%d"), str(row["team"]))
        per_team[key] = {
            "days_rest": int(row["days_rest"]),
            "is_b2b":    bool(row["is_b2b"]),
            "is_3in4":   bool(row["is_3in4"]),
        }

    _def = {"days_rest": 1, "is_b2b": False, "is_3in4": False}
    lookup: dict = {}
    for _, row in kalshi_df.iterrows():
        date_str = str(row["game_date"])
        home = str(row["home_team"])
        away = str(row["away_team"])
        h = per_team.get((date_str, home), _def)
        a = per_team.get((date_str, away), _def)
        lookup[(date_str, home, away)] = {
            "home_b2b":       int(h["is_b2b"]),
            "away_b2b":       int(a["is_b2b"]),
            "home_3in4":      int(h["is_3in4"]),
            "away_3in4":      int(a["is_3in4"]),
            "home_days_rest": h["days_rest"],
            "away_days_rest": a["days_rest"],
        }
    return lookup


def load_ytd_lookup_2025_26(ytd_path=None) -> dict:
    """Return {(game_date_str, team_abbr): blended_mu} for 2025-26.

    Keyed by (date, team) because Kalshi games have no NBA game_id.
    Built by scripts/build_ytd_team_ratings.py.
    """
    p = ytd_path or YTD_RATINGS_PATH
    if not p.exists():
        return {}
    df = pd.read_parquet(p)
    df = df[df["season"] == "2025-26"].copy()
    if df.empty:
        return {}
    lookup = {}
    for _, row in df.iterrows():
        key = (str(row["game_date"]), str(row["team_abbreviation"]).upper())
        lookup[key] = float(row["blended_mu"])
    return lookup


def load_team_params_2025_26():
    """Load 2025-26 projected team parameters."""
    proj_path = FORECAST_DIR / "projected_team_features_v40_2025-26.parquet"
    if not proj_path.exists():
        raise FileNotFoundError(f"{proj_path} not found. Run build_all_season_projections.py first.")

    tf = pd.read_parquet(proj_path)
    result = {}
    for _, row in tf.iterrows():
        team = str(row["team_abbreviation"]).upper()
        result[team] = TeamParams(
            team_abbreviation=team,
            season="2025-26",
            mu=float(row["team_net_rating_projected"]),
            sigma=float(row.get("vol_total", 3.0)),  # Default to league sigma if missing
        )
    return result


def generate_2025_26_forecasts(team_params, kalshi_df, rest_lookup=None, ytd_lookup=None):
    """Generate game-level forecasts for 2025-26 using Kalshi as schedule.

    Applies YTD blended ratings (Step 4) when ytd_lookup is provided, then
    rest/B2B/3-in-4 corrections (Step 3) when rest_lookup is provided.
    """
    config = SimConfig()
    rows = []
    n_ytd_applied = 0

    for _, kalshi_game in kalshi_df.iterrows():
        home   = str(kalshi_game["home_team"]).upper()
        away   = str(kalshi_game["away_team"]).upper()
        date   = str(kalshi_game["game_date"])
        result = kalshi_game["home_result"]

        if home not in team_params or away not in team_params:
            continue

        home_p = team_params[home]
        away_p = team_params[away]

        # Step 4: apply YTD blended mu when available
        if ytd_lookup:
            home_blended = ytd_lookup.get((date, home))
            away_blended = ytd_lookup.get((date, away))
            if home_blended is not None:
                home_p = dataclasses.replace(home_p, mu=home_blended)
                n_ytd_applied += 1
            if away_blended is not None:
                away_p = dataclasses.replace(away_p, mu=away_blended)

        ctx = rest_lookup.get((date, str(kalshi_game["home_team"]), str(kalshi_game["away_team"]))) if rest_lookup else None
        if ctx:
            dist = compute_game_distribution_with_context(
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
            win_prob = dist["win_prob_home"]
        else:
            dist = compute_game_distribution(
                home_p, away_p, is_home_a=True, config=config,
            )
            win_prob = dist["win_prob_a"]

        rows.append({
            "game_date":        date,
            "home_team":        home,
            "away_team":        away,
            "bke_home_win_prob": float(win_prob),
            "home_result":      int(result),
        })

    if ytd_lookup and n_ytd_applied:
        print(f"   YTD blending applied to {n_ytd_applied} team-game entries")
    return pd.DataFrame(rows)


def compute_clv_metrics(bke_df, kalshi_df):
    """Compute CLV between BKE and Kalshi."""
    # Merge on game identifiers
    merged = kalshi_df.merge(
        bke_df,
        on=["game_date", "home_team", "away_team"],
        suffixes=("_kalshi", "_bke"),
        how="inner"
    )

    if len(merged) == 0:
        return None

    # Handle column name conflicts after merge
    if "home_result_kalshi" in merged.columns:
        outcomes = merged["home_result_kalshi"].values
    else:
        outcomes = merged["home_result"].values

    kalshi_probs = merged["kalshi_home_win_prob"].values
    bke_probs = merged["bke_home_win_prob"].values

    # Compute Brier for both
    brier_bke = float(np.mean((bke_probs - outcomes) ** 2))
    brier_kalshi = float(np.mean((kalshi_probs - outcomes) ** 2))

    # Compute CLV: BKE advantage on the winning side
    bke_edge = np.where(
        outcomes == 1,
        bke_probs - kalshi_probs,
        (1 - bke_probs) - (1 - kalshi_probs)
    )
    clv_mean = float(np.mean(bke_edge))
    clv_std = float(np.std(bke_edge))
    clv_sem = clv_std / np.sqrt(len(bke_edge))  # Standard error of mean

    # Count CLV wins
    clv_positive = int(np.sum(bke_edge > 0))
    clv_negative = int(np.sum(bke_edge < 0))

    return {
        "n_games": len(merged),
        "brier_bke": round(brier_bke, 6),
        "brier_kalshi": round(brier_kalshi, 6),
        "brier_delta": round(brier_bke - brier_kalshi, 6),
        "clv_mean": round(clv_mean, 6),
        "clv_std": round(clv_std, 6),
        "clv_sem": round(clv_sem, 6),
        "clv_games_positive": clv_positive,
        "clv_games_negative": clv_negative,
        "clv_interpretation": (
            "✓ POSITIVE: BKE beats market" if clv_mean > 0.01
            else "✗ NEGATIVE: Market beats BKE" if clv_mean < -0.01
            else "≈ TIE: No clear edge"
        ),
    }


def main():
    print("2025-26 Forecast & Kalshi CLV Analysis")
    print("=" * 70)

    # Load Kalshi data first (needed for schedule)
    print("\n1. Loading Kalshi closing lines...")
    kalshi_path = REPO / "data/external/kalshi_closing_lines.csv"
    if not kalshi_path.exists():
        print(f"   ✗ {kalshi_path} not found")
        return
    kalshi = pd.read_csv(kalshi_path)
    print(f"   ✓ Loaded {len(kalshi)} Kalshi market games")

    # Load projections
    print("\n2. Loading 2025-26 projected team features...")
    team_params = load_team_params_2025_26()
    print(f"   ✓ Loaded {len(team_params)} teams")

    # Build rest context from Kalshi schedule
    print("\n3. Building rest/B2B context from schedule...")
    rest_lookup = build_rest_lookup_from_schedule(kalshi)
    n_b2b = sum(1 for v in rest_lookup.values() if v["home_b2b"] or v["away_b2b"])
    n_3in4 = sum(1 for v in rest_lookup.values() if v["home_3in4"] or v["away_3in4"])
    print(f"   ✓ {len(rest_lookup)} games, {n_b2b} with ≥1 B2B ({n_b2b / len(rest_lookup):.1%}), "
          f"{n_3in4} with ≥1 3-in-4 ({n_3in4 / len(rest_lookup):.1%})")

    # Load YTD blended ratings (Step 4)
    print("\n4. Loading YTD blended team ratings...")
    ytd_lookup = load_ytd_lookup_2025_26()
    if ytd_lookup:
        n_ytd_teams = len({t for _, t in ytd_lookup})
        print(f"   ✓ {len(ytd_lookup)} team-game entries, {n_ytd_teams} teams")
    else:
        print("   ✗ YTD lookup not found — using static preseason ratings")
        print("     Run: python3 scripts/build_ytd_team_ratings.py")

    # Generate forecasts
    print("\n5. Generating 2025-26 game forecasts (with rest + YTD context)...")
    bke_forecasts = generate_2025_26_forecasts(team_params, kalshi, rest_lookup=rest_lookup,
                                               ytd_lookup=ytd_lookup)
    print(f"   ✓ Generated {len(bke_forecasts)} game predictions")
    print(f"   Date range: {bke_forecasts['game_date'].min()} to {bke_forecasts['game_date'].max()}")

    # Save forecasts
    out_path = REPORTS_DIR / "bke_2025_26_forecasts.parquet"
    bke_forecasts.to_parquet(out_path, index=False)
    print(f"   ✓ Saved to {out_path}")

    # Compute CLV
    print("\n6. Computing CLV (Closing-Line Value)...")
    clv_metrics = compute_clv_metrics(bke_forecasts, kalshi)

    if clv_metrics is None:
        print("   ✗ No games matched between BKE and Kalshi")
        return

    print(f"\n{'=' * 70}")
    print(f"RESULTS: {clv_metrics['n_games']} games matched")
    print(f"{'=' * 70}")

    print(f"\nBrier Scores:")
    print(f"  BKE:    {clv_metrics['brier_bke']:.6f}")
    print(f"  Kalshi: {clv_metrics['brier_kalshi']:.6f}")
    print(f"  Delta:  {clv_metrics['brier_delta']:+.6f}")
    print(f"           {'(BKE worse)' if clv_metrics['brier_delta'] > 0 else '(BKE better)'}")

    print(f"\nClosing-Line Value (CLV):")
    print(f"  Mean CLV:     {clv_metrics['clv_mean']:+.6f} pp per game")
    print(f"  Std Dev:      {clv_metrics['clv_std']:.6f}")
    print(f"  Std Err Mean: {clv_metrics['clv_sem']:.6f}")
    print(f"  Positive CLV: {clv_metrics['clv_games_positive']:4d} games")
    print(f"  Negative CLV: {clv_metrics['clv_games_negative']:4d} games")
    print(f"  Interpretation: {clv_metrics['clv_interpretation']}")

    # Save report
    report_path = REPORTS_DIR / "kalshi_clv_2025_26.json"
    with open(report_path, "w") as f:
        json.dump(clv_metrics, f, indent=2)
    print(f"\n✓ Report saved to {report_path}")

    # Interpretation
    print(f"\n{'=' * 70}")
    print("INTERPRETATION:")
    print(f"{'=' * 70}")

    if clv_metrics['clv_mean'] > 0:
        pct_edge = clv_metrics['clv_mean'] * 100
        print(f"\n✓ POSITIVE EDGE: BKE is +{pct_edge:.2f}pp better calibrated than Kalshi")
        if clv_metrics['clv_mean'] > 0.02:
            print(f"  → Alpha-ready for trading (CLV > 0.02)")
        elif clv_metrics['clv_mean'] > 0.01:
            print(f"  → Tradeable edge exists (CLV > 0.01)")
        else:
            print(f"  → Marginal edge (CLV > 0)")
    else:
        pct_edge = abs(clv_metrics['clv_mean']) * 100
        print(f"\n✗ NEGATIVE EDGE: Kalshi is +{pct_edge:.2f}pp better than BKE")
        print(f"  → No alpha. Market pricing is sharper.")


if __name__ == "__main__":
    main()
