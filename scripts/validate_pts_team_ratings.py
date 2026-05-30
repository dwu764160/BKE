"""
scripts/validate_pts_team_ratings.py
=============================================================================
Phase 2 — does the PTS-based team rating earn its place? Two performance gates.

Gate 1 (same-season + prior-year signal):
  Compare, as a PRESEASON prior for season N's actual net rating, the production
  signal (N-1 point margin) vs the new PTS rating (N-1 pts_net). If prior-year
  pts_net predicts current net rating better than prior-year margin, PTS is a
  better team prior. Also reports same-season pts_net↔net (descriptive ceiling).
  Bar: beat the prior-year-margin persistence baseline (~r 0.55).

Gate 2 (incremental game-level value vs Elo):
  Walk-forward. Rate each game's teams from season **N-1** PTS (leakage-free),
  add as GBDT features alongside Elo, and report Brier with a PAIRED bootstrap
  vs Elo-only and vs the current GBDT stack. Keep only if the CI excludes 0 (HELP).

Run: python3 scripts/validate_pts_team_ratings.py
=============================================================================
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.simulation.game_model import SimConfig
from src.simulation.simulation_config import HISTORICAL_DIR, FORECAST_DIR
from src.simulation.gbdt_game_model import build_feature_frame, compute_walk_forward_elo

PTS_RATINGS = FORECAST_DIR / "team_pts_ratings.parquet"
RANDOM_SEED = 42

SEASON_ORDER = ["2017-18", "2018-19", "2019-20", "2020-21", "2021-22",
                "2022-23", "2023-24", "2024-25", "2025-26"]


def prior(season):
    i = SEASON_ORDER.index(season)
    return SEASON_ORDER[i - 1] if i > 0 else None


def actual_net_ratings():
    """Per team-season net rating from PLUS_MINUS (available all seasons)."""
    gl = pd.read_parquet(HISTORICAL_DIR / "team_game_logs.parquet",
                         columns=["SEASON", "TEAM_ABBREVIATION", "PLUS_MINUS", "MIN"])
    gl["TEAM_ABBREVIATION"] = gl["TEAM_ABBREVIATION"].str.upper()
    g = gl.groupby(["SEASON", "TEAM_ABBREVIATION"]).agg(
        pm=("PLUS_MINUS", "sum"), mn=("MIN", "sum")).reset_index()
    g["actual_net"] = g["pm"] / g["mn"] * 48.0
    return g.rename(columns={"SEASON": "season", "TEAM_ABBREVIATION": "team"})[
        ["season", "team", "actual_net"]]


def _r(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 3 else float("nan")


def gate1(pts):
    print("=" * 64)
    print("GATE 1 — team-rating signal (correlation with actual net rating)")
    print("=" * 64)
    act = actual_net_ratings()
    p = pts[["season", "team_abbreviation", "pts_net", "pts_net_fit"]].rename(
        columns={"team_abbreviation": "team"})

    # Same-season descriptive ceiling.
    same = act.merge(p, on=["season", "team"], how="inner")
    print(f"\nSame-season (descriptive): pts_net↔net r={_r(same.pts_net.values, same.actual_net.values):.3f} "
          f"| pts_net_fit↔net r={_r(same.pts_net_fit.values, same.actual_net.values):.3f} "
          f"(n={len(same)})")

    # Prior-year priors: production (N-1 margin) vs candidate (N-1 pts_net).
    act2 = act.copy()
    act2["prior_season"] = act2["season"].map(prior)
    margin_prior = act.rename(columns={"season": "prior_season", "actual_net": "margin_prev"})
    pts_prior = p.rename(columns={"season": "prior_season", "pts_net": "pts_prev",
                                  "pts_net_fit": "pts_fit_prev"})
    j = (act2.merge(margin_prior, on=["prior_season", "team"], how="inner")
              .merge(pts_prior, on=["prior_season", "team"], how="inner"))
    r_margin = _r(j.margin_prev.values, j.actual_net.values)
    r_pts = _r(j.pts_prev.values, j.actual_net.values)
    r_pts_fit = _r(j.pts_fit_prev.values, j.actual_net.values)
    print(f"\nAs a PRESEASON prior for season-N net rating (n={len(j)} team-seasons):")
    print(f"  prior-year MARGIN  (production signal): r = {r_margin:.3f}   <-- bar to beat")
    print(f"  prior-year PTS_NET (candidate)        : r = {r_pts:.3f}   {'BEATS' if r_pts>r_margin else 'loses'}")
    print(f"  prior-year PTS_NET+fit                : r = {r_pts_fit:.3f}   {'BEATS' if r_pts_fit>r_margin else 'loses'}")
    return r_margin, r_pts


def _gbdt(train, test, feats):
    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression
    seasons = sorted(train["season"].unique())
    if len(seasons) >= 2:
        va = train[train["season"] == seasons[-1]]; tr = train[train["season"] != seasons[-1]]
    else:
        va = train.sample(frac=0.15, random_state=RANDOM_SEED); tr = train.drop(va.index)
    m = lgb.LGBMClassifier(objective="binary", n_estimators=600, learning_rate=0.03,
                           num_leaves=15, min_child_samples=50, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, reg_lambda=1.0, random_state=RANDOM_SEED,
                           n_jobs=-1, verbose=-1)
    m.fit(tr[feats], tr["home_win"].astype(int),
          eval_set=[(va[feats], va["home_win"].astype(int))], eval_metric="binary_logloss",
          callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(m.predict_proba(va[feats])[:, 1], va["home_win"].astype(int).values)
    return iso.transform(m.predict_proba(test[feats])[:, 1])


def paired(se_a, se_b, n=2000, seed=0):
    rng = np.random.RandomState(seed)
    d = np.asarray(se_a) - np.asarray(se_b)
    idx = rng.randint(0, len(d), size=(n, len(d)))
    b = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def verdict(lo, hi):
    return "HELP" if hi < 0 else ("HURT" if lo > 0 else "NOISE")


def gate2(pts):
    print("\n" + "=" * 64)
    print("GATE 2 — incremental game-level value vs Elo (walk-forward, N-1 PTS)")
    print("=" * 64)
    cfg = SimConfig()
    df = build_feature_frame(cfg)
    df["elo_p"] = compute_walk_forward_elo(df)

    # Attach LAGGED (season N-1) PTS ratings for home & away.
    df["prior_season"] = df["season"].map(prior)
    cols = ["pts_net", "pts_net_fit", "fit_spacing", "fit_rim", "fit_ball_dom"]
    pr = pts[["season", "team_abbreviation"] + cols].copy()
    h = pr.rename(columns={"season": "prior_season", "team_abbreviation": "home_team",
                           **{c: f"h_{c}" for c in cols}})
    a = pr.rename(columns={"season": "prior_season", "team_abbreviation": "away_team",
                           **{c: f"a_{c}" for c in cols}})
    df = df.merge(h, on=["prior_season", "home_team"], how="left").merge(
        a, on=["prior_season", "away_team"], how="left")
    for c in cols:
        df[f"diff_{c}"] = df[f"h_{c}"].fillna(0) - df[f"a_{c}"].fillna(0)

    base_feats = ["delta_mu_norm", "gauss_p", "elo_p", "home_days_rest", "away_days_rest",
                  "rest_diff", "home_b2b", "away_b2b", "home_3in4", "away_3in4"]
    pts_feats = base_feats + ["diff_pts_net"]
    pts_fit_feats = base_feats + ["diff_pts_net", "diff_fit_spacing", "diff_fit_rim", "diff_fit_ball_dom"]

    seasons = sorted(df["season"].unique())
    parts = []
    for i, T in enumerate(seasons):
        if i < 1:
            continue
        tr = df[df["season"].isin(seasons[:i])]; te = df[df["season"] == T].copy()
        if tr.empty or te.empty:
            continue
        te["p_base"] = _gbdt(tr, te, base_feats)
        te["p_pts"] = _gbdt(tr, te, pts_feats)
        te["p_ptsfit"] = _gbdt(tr, te, pts_fit_feats)
        parts.append(te)
        print(f"  fit {T}: train={len(tr)} test={len(te)}")
    ev = pd.concat(parts, ignore_index=True)
    y = ev["home_win"].to_numpy()

    def brier(p): return float(np.mean((ev[p].to_numpy() - y) ** 2))
    se = {k: (ev[k].to_numpy() - y) ** 2 for k in ["p_base", "p_pts", "p_ptsfit"]}
    se_elo = (ev["elo_p"].to_numpy() - y) ** 2

    print(f"\nOOS games: {len(ev)} | seasons {sorted(ev['season'].unique())}")
    print(f"  Elo-only Brier            : {np.mean(se_elo):.4f}")
    print(f"  GBDT base (no PTS)        : {brier('p_base'):.4f}")
    print(f"  GBDT + PTS_net            : {brier('p_pts'):.4f}")
    print(f"  GBDT + PTS_net + fit      : {brier('p_ptsfit'):.4f}")

    print(f"\nPaired bootstrap (negative = better):")
    for name, key in [("PTS vs GBDT-base", ("p_pts", "p_base")),
                      ("PTS+fit vs GBDT-base", ("p_ptsfit", "p_base")),
                      ("PTS vs Elo", ("p_pts", None)),
                      ("PTS+fit vs Elo", ("p_ptsfit", None))]:
        a_se = se[key[0]]
        b_se = se[key[1]] if key[1] else se_elo
        md, lo, hi = paired(a_se, b_se)
        tag = verdict(lo, hi) if key[1] else ("BEATS ELO" if hi < 0 else ("worse" if lo > 0 else "tie"))
        print(f"  {name:<24} Δ={md:+.4f}  [{lo:+.4f},{hi:+.4f}]  {tag}")


def main():
    pts = pd.read_parquet(PTS_RATINGS)
    gate1(pts)
    gate2(pts)


if __name__ == "__main__":
    main()
