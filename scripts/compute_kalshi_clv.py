"""
Computes closing-line value (CLV) of BKE model vs. Kalshi closing prices.
CLV > 0 means BKE predicted the outcome better than the closing line.

Usage:
    python3 scripts/compute_kalshi_clv.py \
        --kalshi data/external/kalshi_closing_lines.csv \
        --bke-forecast data/processed/forecast/bke_game_forecasts.parquet \
        --output reports/kalshi_clv_report.json
"""
import argparse, json, sys
from pathlib import Path
import pandas as pd
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def brier(probs, outcomes):
    return float(np.mean((np.array(probs) - np.array(outcomes)) ** 2))


def log_loss(probs, outcomes):
    eps = 1e-7
    p = np.clip(probs, eps, 1 - eps)
    return float(-np.mean(outcomes * np.log(p) + (1 - outcomes) * np.log(1 - p)))


def compute_clv(kalshi_probs, bke_probs, outcomes):
    """
    CLV = mean(bke_prob_on_winning_side - kalshi_prob_on_winning_side)
    Positive CLV = BKE was more accurate than closing line.
    """
    outcomes = np.array(outcomes)
    bke_probs = np.array(bke_probs)
    kalshi_probs = np.array(kalshi_probs)

    bke_edge = np.where(
        outcomes == 1,
        bke_probs - kalshi_probs,
        (1 - bke_probs) - (1 - kalshi_probs),
    )
    return float(np.mean(bke_edge)), bke_edge.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kalshi", required=True)
    parser.add_argument("--bke-forecast", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    kalshi = pd.read_csv(args.kalshi)
    bke = pd.read_parquet(args.bke_forecast)

    merged = kalshi.merge(bke, on=["game_date", "home_team", "away_team"], how="inner")
    n = len(merged)
    if n == 0:
        print("ERROR: No games matched between Kalshi and BKE forecast. Check team abbreviation format.")
        sys.exit(1)

    outcomes = merged["home_result"].tolist()
    kalshi_probs = merged["kalshi_home_win_prob"].tolist()
    bke_probs = merged["bke_home_win_prob"].tolist()

    brier_bke = brier(bke_probs, outcomes)
    brier_kalshi = brier(kalshi_probs, outcomes)
    logloss_bke = log_loss(bke_probs, outcomes)
    logloss_kalshi = log_loss(kalshi_probs, outcomes)
    clv_mean, clv_per_game = compute_clv(kalshi_probs, bke_probs, outcomes)

    report = {
        "n_games": n,
        "brier_bke": brier_bke,
        "brier_kalshi": brier_kalshi,
        "brier_delta": brier_bke - brier_kalshi,
        "logloss_bke": logloss_bke,
        "logloss_kalshi": logloss_kalshi,
        "clv_mean": clv_mean,
        "interpretation": (
            "POSITIVE CLV = BKE beats market" if clv_mean > 0
            else "NEGATIVE CLV = market beats BKE"
        ),
        "games_with_positive_clv": int(np.sum(np.array(clv_per_game) > 0)),
        "games_with_negative_clv": int(np.sum(np.array(clv_per_game) < 0)),
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
