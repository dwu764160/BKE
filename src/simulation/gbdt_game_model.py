"""
src/simulation/gbdt_game_model.py
=============================================================================
Step 5 — Walk-Forward GBDT Game Model (LightGBM) + Gaussian/GBDT/Elo Ensemble

Layers a LightGBM classifier on top of the production Gaussian baseline to
reduce game-level Brier toward the CLV-ready threshold (~0.210).

Why this is a genuine out-of-sample test
----------------------------------------
For each target season T, the model trains ONLY on seasons < T and predicts T.
The Gaussian leg is computed with the exact production helpers from
validate_forecast.py, so its numbers are directly comparable to the existing
walk-forward baseline (no re-derivation, no accidental leakage).

Normalization strategy (cross-era scale drift)
----------------------------------------------
blended_mu (the game-time rating used by the Gaussian) mixes compressed
preseason_mu (std ~0.6) with real-margin ytd_avg_margin (std ~5-6), so the
effective delta_mu scale drifts season to season. We standardize delta_mu
WITHIN each season (z-score). This uses only rating features (no outcomes), so
it is leakage-free, and it makes the magnitude feature comparable across eras.
The already-calibrated Gaussian win-prob (gauss_p) is also fed as a feature.

Leakage controls
-----------------
- Walk-forward: train seasons strictly < target season.
- Per-season standardization uses only that season's rating features (no labels).
- LightGBM early stopping + isotonic calibration are fit on a holdout carved
  from the TRAINING seasons (most-recent prior season), never on the target.
- Elo is sequential and only ever sees games before the one being predicted.

Inputs
  data/processed/forecast/projected_team_features.parquet   (team sigma/mu)
  data/processed/forecast/team_ratings_ytd.parquet          (blended_mu)
  data/historical/team_game_logs.parquet                    (outcomes, schedule)

Outputs
  reports/gbdt_forecast_validation.json
  reports/bke_gbdt_game_forecasts.parquet  (CLV schema, all OOS seasons)

Usage
  python3 src/simulation/gbdt_game_model.py
  python3 src/simulation/gbdt_game_model.py --min-prior-seasons 2
=============================================================================
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.simulation.game_model import SimConfig, build_schedule
from src.simulation.simulation_config import REPORTS_DIR, HISTORICAL_DIR, LEAGUE_AVG_HCA_FITTED
from src.simulation.validate_forecast import (
    load_forecast_team_params,
    build_rest_lookup,
    load_ytd_lookup,
    _get_dist,
)

RANDOM_SEED = 42

# Feature columns fed to the GBDT (order is fixed for reproducibility).
# The GBDT STACKS the Gaussian (gauss_p, delta_mu_norm) and the sequential Elo
# (elo_p) signals plus rest context; elo_p is OOS per-game so this is leakage-free.
FEATURE_COLS = [
    "delta_mu_norm",   # per-season z-scored Gaussian margin (era-robust magnitude)
    "gauss_p",         # production Gaussian home win prob (calibrated prior)
    "elo_p",           # sequential walk-forward Elo home win prob (in-season form)
    "home_days_rest",
    "away_days_rest",
    "rest_diff",       # home_days_rest - away_days_rest
    "home_b2b",
    "away_b2b",
    "home_3in4",
    "away_3in4",
]


# ═════════════════════════════════════════════════════════════════════
# Feature assembly — reuse the production Gaussian pipeline per game
# ═════════════════════════════════════════════════════════════════════

def build_feature_frame(config: SimConfig) -> pd.DataFrame:
    """Build the per-game feature/label table across all seasons with YTD ratings.

    One row per played game (home perspective). The Gaussian context (delta_mu,
    win_prob_home) is produced by the exact production helpers so it matches the
    existing walk-forward baseline.
    """
    forecast_params = load_forecast_team_params()
    seasons = sorted(forecast_params.keys())  # 2018-26 (2017-18 has no YTD)

    gl_path = HISTORICAL_DIR / "team_game_logs.parquet"
    gl_seasons = set(pd.read_parquet(gl_path, columns=["SEASON"])["SEASON"].unique())

    rows: List[dict] = []
    for season in seasons:
        if season not in gl_seasons:
            continue
        params = forecast_params[season]
        schedule = build_schedule(season)
        if not schedule:
            continue
        rest_lookup = build_rest_lookup(season)
        ytd_lookup = load_ytd_lookup([season])

        for game in schedule:
            if game.home_team not in params or game.away_team not in params:
                continue
            if np.isnan(game.home_win):
                continue

            dist = _get_dist(game, params, config, rest_lookup, ytd_lookup)
            gauss_p = dist.get("win_prob_home", dist.get("win_prob_a", 0.5))
            ctx = rest_lookup.get(game.game_id, {})

            rows.append({
                "season": season,
                "game_id": str(game.game_id),
                "game_date": str(game.date)[:10],
                "home_team": game.home_team,
                "away_team": game.away_team,
                "delta_mu": float(dist["delta_mu"]),
                "gauss_p": float(gauss_p),
                "home_days_rest": int(ctx.get("home_days_rest", 1)),
                "away_days_rest": int(ctx.get("away_days_rest", 1)),
                "home_b2b": int(ctx.get("home_b2b", 0)),
                "away_b2b": int(ctx.get("away_b2b", 0)),
                "home_3in4": int(ctx.get("home_3in4", 0)),
                "away_3in4": int(ctx.get("away_3in4", 0)),
                "home_win": float(game.home_win),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Per-season z-score of delta_mu (leakage-free: rating features only).
    df["rest_diff"] = df["home_days_rest"] - df["away_days_rest"]
    season_std = df.groupby("season")["delta_mu"].transform("std").replace(0.0, 1.0)
    season_mean = df.groupby("season")["delta_mu"].transform("mean")
    df["delta_mu_norm"] = (df["delta_mu"] - season_mean) / season_std
    return df


# ═════════════════════════════════════════════════════════════════════
# Metrics
# ═════════════════════════════════════════════════════════════════════

def _metrics(probs: np.ndarray, actuals: np.ndarray) -> dict:
    probs = np.asarray(probs, dtype=float)
    actuals = np.asarray(actuals, dtype=float)
    eps = 1e-10
    pc = np.clip(probs, eps, 1 - eps)
    return {
        "n_games": int(len(probs)),
        "brier_score": round(float(np.mean((probs - actuals) ** 2)), 6),
        "log_loss": round(float(-np.mean(actuals * np.log(pc) + (1 - actuals) * np.log(1 - pc))), 6),
        "accuracy": round(float(np.mean((probs >= 0.5).astype(float) == actuals)), 4),
    }


# ═════════════════════════════════════════════════════════════════════
# Walk-forward Elo (sequential, out-of-sample by construction)
# ═════════════════════════════════════════════════════════════════════

def compute_walk_forward_elo(
    df: pd.DataFrame,
    k: float = 20.0,
    hca_elo: float = 55.0,
    carry: float = 0.75,
    base: float = 1500.0,
) -> pd.Series:
    """Return a home-win-prob Series aligned to df.index.

    Ratings update sequentially within each season (only past games inform a
    prediction). Between seasons, ratings regress toward `base`:
        start = carry * end_prev + (1 - carry) * base.
    hca_elo ~= 55 Elo points ≈ the empirical NBA home edge.
    """
    ratings: Dict[str, float] = {}
    preds = pd.Series(0.5, index=df.index, dtype=float)

    for season in sorted(df["season"].unique()):
        # Regress prior-season ratings toward the mean at each new season start.
        ratings = {t: carry * r + (1 - carry) * base for t, r in ratings.items()}
        sdf = df[df["season"] == season].sort_values(["game_date", "game_id"])
        for idx, row in sdf.iterrows():
            h, a = row["home_team"], row["away_team"]
            rh = ratings.get(h, base)
            ra = ratings.get(a, base)
            p_home = 1.0 / (1.0 + 10.0 ** (-((rh + hca_elo) - ra) / 400.0))
            preds.at[idx] = p_home
            outcome = float(row["home_win"])
            ratings[h] = rh + k * (outcome - p_home)
            ratings[a] = ra + k * ((1.0 - outcome) - (1.0 - p_home))
    return preds


# ═════════════════════════════════════════════════════════════════════
# Walk-forward GBDT
# ═════════════════════════════════════════════════════════════════════

def _fit_predict_gbdt(
    train: pd.DataFrame,
    test: pd.DataFrame,
    calibrate: bool = True,
) -> np.ndarray:
    """Train LightGBM on `train`, return calibrated home-win probs for `test`.

    Early stopping and isotonic calibration use a holdout carved from the
    TRAINING seasons (most-recent prior season; random 15% if only one prior
    season is available). The target season is never touched during fitting.
    """
    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression

    prior_seasons = sorted(train["season"].unique())
    if len(prior_seasons) >= 2:
        valid_season = prior_seasons[-1]
        tr = train[train["season"] != valid_season]
        va = train[train["season"] == valid_season]
    else:
        va = train.sample(frac=0.15, random_state=RANDOM_SEED)
        tr = train.drop(va.index)

    X_tr, y_tr = tr[FEATURE_COLS], tr["home_win"].astype(int)
    X_va, y_va = va[FEATURE_COLS], va["home_win"].astype(int)

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
    )

    raw_test = model.predict_proba(test[FEATURE_COLS])[:, 1]
    if not calibrate:
        return raw_test

    # Isotonic calibration fit on the (held-out) validation fold.
    raw_va = model.predict_proba(X_va)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw_va, y_va.values)
    return iso.transform(raw_test)


def walk_forward(
    df: pd.DataFrame,
    min_prior_seasons: int = 1,
    calibrate: bool = True,
) -> pd.DataFrame:
    """Run walk-forward GBDT + Elo across seasons.

    Returns df restricted to evaluated (target) rows, with added columns:
      gbdt_p, elo_p (elo_p is computed globally then sliced to target rows).
    """
    seasons = sorted(df["season"].unique())

    # Elo is global/sequential; compute once, slice to target rows later.
    df = df.copy()
    df["elo_p"] = compute_walk_forward_elo(df)
    df["gbdt_p"] = np.nan

    evaluated_idx: List = []
    for i, target in enumerate(seasons):
        if i < min_prior_seasons:
            continue
        train = df[df["season"].isin(seasons[:i])]
        test = df[df["season"] == target]
        if train.empty or test.empty:
            continue
        df.loc[test.index, "gbdt_p"] = _fit_predict_gbdt(train, test, calibrate=calibrate)
        evaluated_idx.extend(test.index.tolist())

    return df.loc[evaluated_idx].copy()


# ═════════════════════════════════════════════════════════════════════
# Ensembles + reporting
# ═════════════════════════════════════════════════════════════════════

ENSEMBLES = {
    "gaussian_only": {"gauss_p": 1.0},
    "gbdt_only":     {"gbdt_p": 1.0},
    "elo_only":      {"elo_p": 1.0},
    "blend_50_50":   {"gauss_p": 0.50, "gbdt_p": 0.50},
    "blend_30_50_20": {"gauss_p": 0.30, "gbdt_p": 0.50, "elo_p": 0.20},
}


def _blend(df: pd.DataFrame, weights: Dict[str, float]) -> np.ndarray:
    p = np.zeros(len(df), dtype=float)
    for col, w in weights.items():
        p = p + w * df[col].to_numpy(dtype=float)
    return p


def evaluate(eval_df: pd.DataFrame) -> dict:
    seasons = sorted(eval_df["season"].unique())
    result = {"legs": {}, "per_season": {}}

    for name, w in ENSEMBLES.items():
        probs = _blend(eval_df, w)
        result["legs"][name] = _metrics(probs, eval_df["home_win"].to_numpy())

    for season in seasons:
        sdf = eval_df[eval_df["season"] == season]
        result["per_season"][season] = {
            name: _metrics(_blend(sdf, w), sdf["home_win"].to_numpy())
            for name, w in ENSEMBLES.items()
        }
    return result


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-prior-seasons", type=int, default=1,
                        help="Minimum prior seasons before a season is predicted (default 1).")
    parser.add_argument("--no-calibrate", action="store_true",
                        help="Disable isotonic calibration of GBDT output.")
    parser.add_argument("--export-season", default="2025-26",
                        help="Season to export in CLV schema for Kalshi testing.")
    args = parser.parse_args()

    print("Step 5 — Walk-Forward GBDT Game Model")
    print("=" * 56)

    config = SimConfig()
    df = build_feature_frame(config)
    if df.empty:
        print("ERROR: no feature rows built (missing forecast/ytd/game-log data).")
        return
    print(f"Feature rows: {len(df)} across seasons {sorted(df['season'].unique())}")
    print(f"Features: {FEATURE_COLS}")

    eval_df = walk_forward(df, min_prior_seasons=args.min_prior_seasons,
                           calibrate=not args.no_calibrate)
    if eval_df.empty:
        print("ERROR: no OOS rows evaluated.")
        return

    report = evaluate(eval_df)
    report["meta"] = {
        "mode": "walk_forward_gbdt",
        "min_prior_seasons": args.min_prior_seasons,
        "calibrated": not args.no_calibrate,
        "feature_cols": FEATURE_COLS,
        "n_oos_games": int(len(eval_df)),
        "oos_seasons": sorted(eval_df["season"].unique()),
        "normalization": "per-season z-score of delta_mu (leakage-free)",
        "ensemble_weights": ENSEMBLES,
    }

    # Console summary — aggregate OOS.
    print(f"\nAggregate OOS ({len(eval_df)} games, seasons "
          f"{sorted(eval_df['season'].unique())}):")
    print(f"  {'leg':<16}{'Brier':>9}{'LogLoss':>10}{'Acc':>8}")
    for name in ENSEMBLES:
        m = report["legs"][name]
        print(f"  {name:<16}{m['brier_score']:>9.4f}{m['log_loss']:>10.4f}{m['accuracy']:>8.3f}")

    best = min(ENSEMBLES, key=lambda n: report["legs"][n]["brier_score"])
    g = report["legs"]["gaussian_only"]["brier_score"]
    b = report["legs"][best]["brier_score"]
    print(f"\nBest leg: {best} (Brier {b:.4f}); Gaussian baseline {g:.4f}; "
          f"delta {b - g:+.4f}")
    report["meta"]["best_leg"] = best
    report["meta"]["best_brier"] = b
    report["meta"]["gaussian_brier"] = g

    # Per-season Brier table.
    print(f"\nPer-season Brier:")
    hdr = "  season    " + "".join(f"{n.split('_')[0][:7]:>9}" for n in ENSEMBLES)
    print(hdr)
    for season in sorted(report["per_season"].keys()):
        ps = report["per_season"][season]
        line = f"  {season:<10}" + "".join(f"{ps[n]['brier_score']:>9.4f}" for n in ENSEMBLES)
        print(line)

    # Persist JSON report.
    out_json = REPORTS_DIR / "gbdt_forecast_validation.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved report: {out_json}")

    # CLV-schema export for the requested season (GBDT + best ensemble probs).
    exp = eval_df[eval_df["season"] == args.export_season]
    if not exp.empty:
        best_w = ENSEMBLES[best]
        clv = pd.DataFrame({
            "game_date": exp["game_date"].values,
            "home_team": exp["home_team"].values,
            "away_team": exp["away_team"].values,
            "bke_home_win_prob": _blend(exp, best_w),
            "gbdt_home_win_prob": exp["gbdt_p"].values,
            "home_result": exp["home_win"].astype(int).values,
        })
        out_clv = REPORTS_DIR / "bke_gbdt_game_forecasts.parquet"
        clv.to_parquet(out_clv, index=False)
        print(f"Exported {len(clv)} {args.export_season} predictions ({best} as "
              f"bke_home_win_prob) to {out_clv}")
    else:
        print(f"Note: export season {args.export_season} not in OOS set; no CLV export.")


if __name__ == "__main__":
    main()
