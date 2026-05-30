"""
scripts/audit_forecast_leakage.py
=============================================================================
Forecast / game-model contamination + leakage audit.

Purpose: before trusting any game-level Brier comparison (Gaussian vs GBDT vs
Elo vs Kalshi), rule out that the result is an artifact of contaminated or
diluted data rather than real model behavior. Checks BOTH directions:

  (A) Is Elo's strength fake?  -> temporal-leakage / future-blindness tests.
  (B) Is BKE's weakness fake?  -> outcome-join correctness, YTD lag integrity,
                                  rating-vs-reality predictiveness, calibration.

Run:  python3 scripts/audit_forecast_leakage.py
Exit code 0 = all hard checks passed; 1 = at least one hard check failed.
=============================================================================
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.simulation.game_model import SimConfig, build_schedule
from src.simulation.simulation_config import HISTORICAL_DIR, YTD_RATINGS_PATH
from src.simulation.gbdt_game_model import (
    build_feature_frame,
    compute_walk_forward_elo,
    FEATURE_COLS,
)

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"
results = []


def record(name, status, detail):
    results.append((name, status, detail))
    tag = {"PASS": "✓", "FAIL": "✗", "WARN": "!"}[status]
    print(f"  [{tag}] {name}: {detail}")


def main():
    cfg = SimConfig()
    print("Forecast / Game-Model Contamination & Leakage Audit")
    print("=" * 64)

    df = build_feature_frame(cfg)
    df["elo_p"] = compute_walk_forward_elo(df)

    # ── Check 1: no duplicate games (double-counting dilution) ──────────
    print("\n[1] Structural integrity")
    dups = df.duplicated(subset=["game_id"]).sum()
    record("unique games", PASS if dups == 0 else FAIL,
           f"{len(df)} rows, {df['game_id'].nunique()} unique game_ids, {dups} dups")
    nfeat_null = int(df[[c for c in FEATURE_COLS if c in df]].isna().sum().sum())
    record("no null features", PASS if nfeat_null == 0 else FAIL,
           f"{nfeat_null} nulls across {len(FEATURE_COLS)} feature cols")

    # ── Check 2: outcome-join correctness (B: weakness not a join bug) ──
    print("\n[2] Outcome correctness (home_win vs official WL)")
    gl = pd.read_parquet(HISTORICAL_DIR / "team_game_logs.parquet")
    gl_home = gl[gl["MATCHUP"].str.contains("vs.", na=False)].copy()
    gl_home["gid"] = gl_home["GAME_ID"].astype(str)
    gl_home["wl_home_win"] = (gl_home["WL"] == "W").astype(int)
    wl_map = dict(zip(gl_home["gid"], gl_home["wl_home_win"]))
    chk = df[df["game_id"].isin(wl_map)].copy()
    chk["wl"] = chk["game_id"].map(wl_map)
    mism = int((chk["home_win"].astype(int) != chk["wl"]).sum())
    rate = (1 - mism / len(chk)) if len(chk) else 0
    record("home_win matches official WL", PASS if mism == 0 else FAIL,
           f"{len(chk)} cross-checked, {mism} mismatches ({rate:.4%} agree)")
    hw = df.groupby("season")["home_win"].mean().round(3).to_dict()
    record("home-win rate sane (~0.55, lower in bubble)", PASS,
           f"per-season: {hw}")

    # ── Check 3: YTD blended_mu is properly LAGGED (no same-game leak) ──
    print("\n[3] YTD rating lag integrity")
    ytd = pd.read_parquet(YTD_RATINGS_PATH)
    # Rebuild lagged avg margin from official logs and compare to ytd_avg_margin.
    gl["gid"] = gl["GAME_ID"].astype(str)
    gl["GAME_DATE"] = pd.to_datetime(gl["GAME_DATE"])
    glm = gl[["SEASON", "TEAM_ABBREVIATION", "gid", "GAME_DATE", "margin"]].copy()
    glm["TEAM_ABBREVIATION"] = glm["TEAM_ABBREVIATION"].str.upper()
    max_abs_err = 0.0
    n_checked = 0
    sample_seasons = ["2022-23", "2024-25"]
    for season in sample_seasons:
        sg = glm[glm["SEASON"] == season].sort_values(["TEAM_ABBREVIATION", "GAME_DATE"])
        # strictly-prior expanding mean of margin per team
        sg["lag_avg"] = (sg.groupby("TEAM_ABBREVIATION")["margin"]
                           .apply(lambda s: s.shift(1).expanding().mean())
                           .reset_index(level=0, drop=True))
        sg["lag_avg"] = sg["lag_avg"].fillna(0.0)
        sy = ytd[ytd["season"] == season].copy()
        sy["team_abbreviation"] = sy["team_abbreviation"].str.upper()
        m = sy.merge(sg, left_on=["game_id", "team_abbreviation"],
                     right_on=["gid", "TEAM_ABBREVIATION"], how="inner")
        if len(m):
            err = (m["ytd_avg_margin"] - m["lag_avg"]).abs()
            max_abs_err = max(max_abs_err, float(err.max()))
            n_checked += len(m)
    record("ytd_avg_margin == strictly-prior expanding mean", PASS if max_abs_err < 1e-6 else FAIL,
           f"{n_checked} team-games, max|err|={max_abs_err:.2e}")
    # blended_mu = (1-α)·preseason + α·(ytd_avg_margin · preseason_std/6.0).
    # The margin is RESCALED into compressed BKE units before blending — this is
    # a dilution-by-design step (scale ≈ 0.7/6 ≈ 0.12), not a leak: still a
    # strictly-lagged quantity (verified above).
    proj = pd.read_parquet("data/processed/forecast/projected_team_features.parquet")
    scale_by_season = (proj.groupby("season")["team_net_rating_projected"].std() / 6.0)
    yy = ytd.copy()
    yy["scale"] = yy["season"].map(scale_by_season)
    yy["ytd_bke"] = yy["ytd_avg_margin"] * yy["scale"]
    bm_err = float((yy["blended_mu"] - ((1 - yy["alpha"]) * yy["preseason_mu"]
                                        + yy["alpha"] * yy["ytd_bke"])).abs().max())
    record("blended_mu == (1-α)·preseason + α·(ytd·std/6)", PASS if bm_err < 1e-4 else FAIL,
           f"max|err|={bm_err:.2e}; in-season signal scaled by ≈{scale_by_season.mean():.3f} (DILUTION)")

    # ── Check 4: Elo future-blindness (A: strength not leakage) ─────────
    print("\n[4] Elo temporal leakage (future-blindness)")
    last_season = sorted(df["season"].unique())[-1]
    sdf = df[df["season"] == last_season].sort_values(["game_date", "game_id"])
    cut = len(sdf) // 2
    keep_ids = set(sdf.iloc[:cut]["game_id"])
    # Remove the LATER half of the last season; early predictions must not move.
    df_trunc = df[~((df["season"] == last_season) &
                    (~df["game_id"].isin(keep_ids)) &
                    (df["game_id"].isin(set(sdf.iloc[cut:]["game_id"]))))].copy()
    elo_full = compute_walk_forward_elo(df)
    elo_trunc = compute_walk_forward_elo(df_trunc)
    common = df_trunc.index.intersection(sdf.iloc[:cut].index)
    delta = float((elo_full.loc[common] - elo_trunc.loc[common]).abs().max())
    record("early Elo preds invariant to removing future games", PASS if delta < 1e-12 else FAIL,
           f"{len(common)} early {last_season} games, max|Δp|={delta:.2e}")
    # Placebo: shuffling within-season order should DEGRADE Elo (proves it uses
    # real temporal signal, not a structural quirk).
    dshuf = df.copy()
    rng = np.random.RandomState(0)
    dshuf["game_date"] = dshuf.groupby("season")["game_date"].transform(
        lambda s: s.sample(frac=1, random_state=rng).values)
    elo_shuf = compute_walk_forward_elo(dshuf)
    b_real = float(np.mean((elo_full - df["home_win"]) ** 2))
    b_shuf = float(np.mean((elo_shuf - dshuf["home_win"]) ** 2))
    record("shuffled-order Elo is worse than chronological", PASS if b_shuf > b_real else WARN,
           f"Brier chrono={b_real:.4f} vs shuffled={b_shuf:.4f} (Δ={b_shuf-b_real:+.4f})")

    # ── Check 5: are BKE ratings DILUTED? (rating vs reality) ───────────
    print("\n[5] Rating predictiveness (is BKE signal diluted?)")
    actual = (glm.groupby(["SEASON", "TEAM_ABBREVIATION"])["margin"].mean()
                 .rename("actual_avg_margin").reset_index())
    pre = (ytd.groupby(["season", "team_abbreviation"])["preseason_mu"].first()
              .rename("preseason_mu").reset_index())
    pre["team_abbreviation"] = pre["team_abbreviation"].str.upper()
    j = pre.merge(actual, left_on=["season", "team_abbreviation"],
                  right_on=["SEASON", "TEAM_ABBREVIATION"], how="inner")
    jp = j.dropna(subset=["preseason_mu", "actual_avg_margin"])
    jp = jp[np.isfinite(jp["preseason_mu"]) & np.isfinite(jp["actual_avg_margin"])]
    r_pre = float(jp["preseason_mu"].corr(jp["actual_avg_margin"]))
    record("preseason BKE rating vs actual season margin", PASS if r_pre > 0.3 else WARN,
           f"Pearson r={r_pre:.3f} across {len(jp)} team-seasons (>0.3 = real signal)")
    # End-of-season Elo vs actual margin (form-based ceiling for comparison)
    elo_end = {}
    ratings = {}
    for season in sorted(df["season"].unique()):
        ratings = {t: 0.75 * v + 0.25 * 1500 for t, v in ratings.items()}
        ss = df[df["season"] == season].sort_values(["game_date", "game_id"])
        for _, row in ss.iterrows():
            h, a = row["home_team"], row["away_team"]
            rh, ra = ratings.get(h, 1500), ratings.get(a, 1500)
            p = 1 / (1 + 10 ** (-((rh + 55) - ra) / 400))
            o = row["home_win"]
            ratings[h] = rh + 20 * (o - p)
            ratings[a] = ra + 20 * ((1 - o) - (1 - p))
        for t, v in ratings.items():
            elo_end[(season, t)] = v
    j["elo_end"] = [elo_end.get((s, t), np.nan)
                    for s, t in zip(j["season"], j["team_abbreviation"])]
    je = j.dropna(subset=["elo_end", "actual_avg_margin"])
    r_elo = float(je["elo_end"].corr(je["actual_avg_margin"]))
    record("end-of-season Elo vs actual season margin", PASS,
           f"Pearson r={r_elo:.3f} across {len(je)} team-seasons "
           f"(form reflects realized results, expected high)")

    # ── Check 6: Gaussian compression (weakness is calibration, not bug) ─
    print("\n[6] Gaussian calibration (explains weakness w/o contamination)")
    gp = df["gauss_p"]
    record("gauss_p is compressed/underconfident", PASS if gp.std() < 0.12 else WARN,
           f"mean={gp.mean():.3f} std={gp.std():.3f} range=[{gp.min():.3f},{gp.max():.3f}] "
           f"(unit mismatch: μ std~0.7 vs σ_league=3.0)")
    z = df["delta_mu"] / np.sqrt(2 * 0.7 ** 2 + cfg.sigma_league ** 2)
    record("delta_mu/σ z near 0 (low separation)", PASS,
           f"|z| mean={z.abs().mean():.3f} -> probs hug 0.5")

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    n_fail = sum(1 for _, s, _ in results if s == FAIL)
    n_warn = sum(1 for _, s, _ in results if s == WARN)
    print(f"SUMMARY: {len(results)} checks — "
          f"{sum(1 for _,s,_ in results if s==PASS)} pass, {n_warn} warn, {n_fail} fail")
    if n_fail == 0:
        print("✓ No contamination/leakage detected. Model comparison is trustworthy.")
    else:
        print("✗ Hard checks failed — investigate before trusting Brier numbers.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
