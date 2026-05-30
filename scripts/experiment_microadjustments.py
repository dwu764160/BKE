"""
scripts/experiment_microadjustments.py
=============================================================================
Test the three proposed game-model micro-adjustments — walk-forward, OOS —
and classify each as HELP / NOISE / HURT via a PAIRED bootstrap on per-game
squared errors (so we don't mistake noise for signal).

Adjustments (all vs the current production Gaussian baseline, M0):
  M1 = Adj1 "recalibrate the game layer": walk-forward isotonic calibration of
       the Gaussian win prob (fixes the under-confidence / scale mismatch).
  M2 = Adj2 "stop diluting in-season signal": rating = current-season UNDILUTED
       YTD margin in real points (delta = ytd_margin_home - ytd_margin_away),
       walk-forward isotonic-calibrated.
  M4 = Adj3 "Elo as first-class + GBDT stack": existing GBDT stacking gauss/elo/
       rest, plus a variant (M5) that also stacks the undiluted current margin.

Reference: Elo (best single model) and the Gaussian baseline.

Run: python3 scripts/experiment_microadjustments.py
=============================================================================
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.simulation.game_model import SimConfig
from src.simulation.simulation_config import YTD_RATINGS_PATH
from src.simulation.gbdt_game_model import build_feature_frame, compute_walk_forward_elo

RANDOM_SEED = 42


def _gbdt(train, test, feats):
    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression
    seasons = sorted(train["season"].unique())
    if len(seasons) >= 2:
        va = train[train["season"] == seasons[-1]]
        tr = train[train["season"] != seasons[-1]]
    else:
        va = train.sample(frac=0.15, random_state=RANDOM_SEED)
        tr = train.drop(va.index)
    m = lgb.LGBMClassifier(objective="binary", n_estimators=600, learning_rate=0.03,
                           num_leaves=15, min_child_samples=50, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                           random_state=RANDOM_SEED, n_jobs=-1, verbose=-1)
    m.fit(tr[feats], tr["home_win"].astype(int),
          eval_set=[(va[feats], va["home_win"].astype(int))], eval_metric="binary_logloss",
          callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(m.predict_proba(va[feats])[:, 1], va["home_win"].astype(int).values)
    return iso.transform(m.predict_proba(test[feats])[:, 1])


def _iso_1d(train, test, col):
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(train[col].values, train["home_win"].astype(int).values)
    return iso.transform(test[col].values)


def brier(p, y):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def paired_bootstrap(se_model, se_base, n=2000, seed=0):
    """Return (mean_delta, lo, hi) for se_model - se_base. Negative => model better."""
    rng = np.random.RandomState(seed)
    d = np.asarray(se_model) - np.asarray(se_base)
    idx = rng.randint(0, len(d), size=(n, len(d)))
    boots = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def classify(lo, hi):
    if hi < 0:
        return "HELP"
    if lo > 0:
        return "HURT"
    return "NOISE"


def main():
    cfg = SimConfig()
    df = build_feature_frame(cfg)
    df["elo_p"] = compute_walk_forward_elo(df)

    # Merge undiluted current-season YTD margin (real points, strictly prior).
    ytd = pd.read_parquet(YTD_RATINGS_PATH)[["game_id", "team_abbreviation", "ytd_avg_margin"]].copy()
    ytd["team_abbreviation"] = ytd["team_abbreviation"].str.upper()
    ytd["game_id"] = ytd["game_id"].astype(str)
    mh = ytd.rename(columns={"team_abbreviation": "home_team", "ytd_avg_margin": "ytd_h"})
    ma = ytd.rename(columns={"team_abbreviation": "away_team", "ytd_avg_margin": "ytd_a"})
    df = df.merge(mh, on=["game_id", "home_team"], how="left").merge(ma, on=["game_id", "away_team"], how="left")
    df["ytd_h"] = df["ytd_h"].fillna(0.0)
    df["ytd_a"] = df["ytd_a"].fillna(0.0)
    df["delta_current"] = df["ytd_h"] - df["ytd_a"]

    base_feats = ["delta_mu_norm", "gauss_p", "elo_p", "home_days_rest", "away_days_rest",
                  "rest_diff", "home_b2b", "away_b2b", "home_3in4", "away_3in4"]
    plus_feats = base_feats + ["delta_current"]

    seasons = sorted(df["season"].unique())
    parts = []
    for i, T in enumerate(seasons):
        if i < 1:
            continue
        train = df[df["season"].isin(seasons[:i])]
        test = df[df["season"] == T].copy()
        if train.empty or test.empty:
            continue
        test["M1_gauss_cal"] = _iso_1d(train, test, "gauss_p")
        test["M2_curr_margin"] = _iso_1d(train, test, "delta_current")
        test["M4_gbdt"] = _gbdt(train, test, base_feats)
        test["M5_gbdt_plus"] = _gbdt(train, test, plus_feats)
        parts.append(test)
        print(f"  fit {T}: train={len(train)} test={len(test)}")

    ev = pd.concat(parts, ignore_index=True)
    y = ev["home_win"].to_numpy()
    models = {
        "M0_gaussian (baseline)": ev["gauss_p"].to_numpy(),
        "M1_gauss_cal (Adj1)":    ev["M1_gauss_cal"].to_numpy(),
        "M2_curr_margin (Adj2)":  ev["M2_curr_margin"].to_numpy(),
        "Elo (reference)":        ev["elo_p"].to_numpy(),
        "M4_gbdt_stack (Adj3)":   ev["M4_gbdt"].to_numpy(),
        "M5_gbdt+curr_margin":    ev["M5_gbdt_plus"].to_numpy(),
    }
    se = {k: (v - y) ** 2 for k, v in models.items()}
    base = se["M0_gaussian (baseline)"]
    elo = se["Elo (reference)"]

    print(f"\nOOS games: {len(ev)} | seasons {sorted(ev['season'].unique())}")
    print(f"\n{'model':<26}{'Brier':>8}{'Δ vs M0':>10}{'95% CI':>20}{'verdict':>9}")
    print("-" * 75)
    for k, p in models.items():
        b = brier(p, y)
        if k == "M0_gaussian (baseline)":
            print(f"{k:<26}{b:>8.4f}{'—':>10}{'—':>20}{'—':>9}")
            continue
        md, lo, hi = paired_bootstrap(se[k], base)
        print(f"{k:<26}{b:>8.4f}{md:>+10.4f}   [{lo:+.4f},{hi:+.4f}]{classify(lo,hi):>9}")

    # Does any adjustment beat Elo (the bar to clear)?
    print(f"\nvs Elo (does any adjustment beat the best single model?):")
    print(f"{'model':<26}{'Δ vs Elo':>10}{'95% CI':>22}{'verdict':>9}")
    print("-" * 70)
    for k in ["M1_gauss_cal (Adj1)", "M2_curr_margin (Adj2)", "M4_gbdt_stack (Adj3)", "M5_gbdt+curr_margin"]:
        md, lo, hi = paired_bootstrap(se[k], elo)
        v = "BEATS ELO" if hi < 0 else ("worse" if lo > 0 else "tie")
        print(f"{k:<26}{md:>+10.4f}   [{lo:+.4f},{hi:+.4f}]{v:>9}")


if __name__ == "__main__":
    main()
