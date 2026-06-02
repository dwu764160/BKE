"""
scripts/experiment_ytd_roster_pts.py
=============================================================================
The real PTS test: does a CURRENT-roster YTD PTS rating beat Elo?

Phase 1/2 showed lagged (season N-1) PTS adds nothing. The hypothesis is that
PTS only helps when applied to the CURRENT roster as it forms through the season
(trades, who's actually getting minutes) — the window where Elo lags.

Rating (leakage-free):
  For team T before game G (date D), weight each player's PRIOR-SEASON PTS net by
  their STRICTLY-PRIOR current-season cumulative minutes:
     ytd_pts(T, D) = Σ_{games g<D} Σ_p (min_{p,g} · skill_p)  /  Σ_{g<D} Σ_p min_{p,g}
  skill_p = season N-1 (pts_o_v40 + pts_d_v40); 0 if no prior season (rookies).
  Uses only games strictly before G and prior-season skill → no leakage.

Data limit: local player-game minutes exist only for 2022-23..2024-25, so the
walk-forward test covers OOS seasons 2023-24 and 2024-25 (train on earlier).
Backfilling player_game_logs (network) extends this.

Compares, walk-forward, via logistic stacking on Elo (thin data → logistic, not GBDT):
  A: Elo only
  B: Elo + CURRENT-roster ytd_pts diff      (the hypothesis)
  C: Elo + LAGGED team pts diff             (the Phase-2 stale signal, control)

Run: python3 scripts/experiment_ytd_roster_pts.py
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

SEASON_ORDER = ["2017-18", "2018-19", "2019-20", "2020-21", "2021-22",
                "2022-23", "2023-24", "2024-25", "2025-26"]


def prior(s):
    i = SEASON_ORDER.index(s)
    return SEASON_ORDER[i - 1] if i > 0 else None


def player_skill_by_season():
    """{(season, player_id): pts_net} from pts_v40 (used at season+1 as prior skill)."""
    p = pd.read_parquet("data/processed/bke/pts_v40.parquet",
                        columns=["player_id", "season", "pts_o_v40", "pts_d_v40"])
    p["skill"] = p["pts_o_v40"] + p["pts_d_v40"]
    return {(str(r.season), int(r.player_id)): float(r.skill) for r in p.itertuples()}


def build_ytd_roster_pts():
    """Return DataFrame [game_id, team, ytd_pts] (strictly-prior, current-roster)."""
    pgl = pd.read_parquet(HISTORICAL_DIR / "final_player_game_logs.parquet",
                          columns=["season", "player_id", "game_id", "game_date", "min", "matchup"])
    pgl["game_date"] = pd.to_datetime(pgl["game_date"], format="mixed")
    pgl["team"] = pgl["matchup"].str.split(r"\s+(?:vs\.|@)\s+", regex=True).str[0].str.upper()
    pgl["min"] = pd.to_numeric(pgl["min"], errors="coerce").fillna(0.0)

    skill = player_skill_by_season()
    pgl["skill"] = [skill.get((prior(s), int(pid)), 0.0)
                    for s, pid in zip(pgl["season"], pgl["player_id"])]
    pgl["contrib"] = pgl["min"] * pgl["skill"]

    # Per team-game totals, then strictly-prior cumulative ratio.
    g = (pgl.groupby(["season", "team", "game_id", "game_date"], as_index=False)
            .agg(s_game=("contrib", "sum"), m_game=("min", "sum")))
    g = g.sort_values(["season", "team", "game_date"]).reset_index(drop=True)
    grp = g.groupby(["season", "team"])
    g["cum_s_before"] = grp["s_game"].cumsum() - g["s_game"]
    g["cum_m_before"] = grp["m_game"].cumsum() - g["m_game"]
    g["ytd_pts"] = np.where(g["cum_m_before"] > 0, g["cum_s_before"] / g["cum_m_before"], np.nan)
    g["games_before"] = grp.cumcount()
    return g[["game_id", "team", "ytd_pts", "games_before"]]


def main():
    cfg = SimConfig()
    df = build_feature_frame(cfg)
    df["elo_p"] = compute_walk_forward_elo(df)
    df["game_id"] = df["game_id"].astype(str)

    ytd = build_ytd_roster_pts()
    ytd["game_id"] = ytd["game_id"].astype(str)
    h = ytd.rename(columns={"team": "home_team", "ytd_pts": "h_ytd", "games_before": "h_gb"})
    a = ytd.rename(columns={"team": "away_team", "ytd_pts": "a_ytd", "games_before": "a_gb"})
    df = df.merge(h, on=["game_id", "home_team"], how="left").merge(a, on=["game_id", "away_team"], how="left")
    df["ytd_pts_diff"] = df["h_ytd"] - df["a_ytd"]

    # Lagged team pts diff (control) from Phase-1 ratings.
    pts = pd.read_parquet(FORECAST_DIR / "team_pts_ratings.parquet")
    df["prior_season"] = df["season"].map(prior)
    lh = pts[["season", "team_abbreviation", "pts_net"]].rename(
        columns={"season": "prior_season", "team_abbreviation": "home_team", "pts_net": "h_lag"})
    la = pts[["season", "team_abbreviation", "pts_net"]].rename(
        columns={"season": "prior_season", "team_abbreviation": "away_team", "pts_net": "a_lag"})
    df = df.merge(lh, on=["prior_season", "home_team"], how="left").merge(la, on=["prior_season", "away_team"], how="left")
    df["lag_pts_diff"] = df["h_lag"].fillna(0) - df["a_lag"].fillna(0)

    # Keep all games with the feature (>=2 prior games to avoid div-by-tiny early on).
    # Critically do NOT gate out early season — that's where PTS should beat Elo.
    use = df[df["ytd_pts_diff"].notna() & (df["h_gb"] >= 2) & (df["a_gb"] >= 2)].copy()
    use["min_gb"] = use[["h_gb", "a_gb"]].min(axis=1)
    use["elo_logit"] = np.log(use["elo_p"].clip(1e-6, 1 - 1e-6) / (1 - use["elo_p"].clip(1e-6, 1 - 1e-6)))
    seasons = [s for s in SEASON_ORDER if s in set(use["season"]) and prior(s) in set(use["season"])]
    print(f"Seasons with current-roster feature: {sorted(use['season'].unique())}")
    print(f"OOS test seasons (need a prior season with data): {seasons}")

    from sklearn.linear_model import LogisticRegression

    def fit_eval(feat_cols):
        probs, ys, segs = [], [], []
        for T in seasons:
            tr = use[use["season"].isin([s for s in seasons if s < T] + [prior(T)])]
            tr = tr[tr["season"] < T]
            te = use[use["season"] == T]
            if tr.empty or te.empty:
                continue
            m = LogisticRegression(max_iter=1000)
            m.fit(tr[feat_cols], tr["home_win"].astype(int))
            probs.append(m.predict_proba(te[feat_cols])[:, 1])
            ys.append(te["home_win"].to_numpy())
            segs.append(np.full(len(te), T))
        return np.concatenate(probs), np.concatenate(ys), np.concatenate(segs)

    pA, y, seg = fit_eval(["elo_logit"])
    pB, _, _ = fit_eval(["elo_logit", "ytd_pts_diff"])
    pC, _, _ = fit_eval(["elo_logit", "lag_pts_diff"])

    def brier(p): return float(np.mean((p - y) ** 2))

    def paired(pa, pb, n=2000, seed=0):
        rng = np.random.RandomState(seed)
        d = (pa - y) ** 2 - (pb - y) ** 2
        idx = rng.randint(0, len(d), size=(n, len(d)))
        b = d[idx].mean(axis=1)
        return float(d.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

    raw_elo = use.set_index(use.index)  # for raw-elo Brier on same rows
    # raw Elo brier on the evaluated rows
    ev_mask = use["season"].isin(seasons)
    raw_b = float(np.mean((use.loc[ev_mask, "elo_p"].to_numpy() - use.loc[ev_mask, "home_win"].to_numpy()) ** 2))

    print(f"\nOOS games evaluated: {len(y)}  (test seasons {sorted(set(seg))})")
    print(f"  raw Elo Brier                  : {raw_b:.4f}")
    print(f"  A) logit(Elo)                  : {brier(pA):.4f}")
    print(f"  B) logit(Elo + CURRENT-roster) : {brier(pB):.4f}")
    print(f"  C) logit(Elo + LAGGED pts)     : {brier(pC):.4f}")
    mdB, loB, hiB = paired(pB, pA)
    mdC, loC, hiC = paired(pC, pA)
    print(f"\nPaired bootstrap vs A=logit(Elo) (negative = better):")
    print(f"  B current-roster: Δ={mdB:+.4f} [{loB:+.4f},{hiB:+.4f}]  "
          f"{'HELP' if hiB<0 else ('HURT' if loB>0 else 'NOISE')}")
    print(f"  C lagged control: Δ={mdC:+.4f} [{loC:+.4f},{hiC:+.4f}]  "
          f"{'HELP' if hiC<0 else ('HURT' if loC>0 else 'NOISE')}")

    # Segment by how far into the season we are (the hypothesis: PTS helps EARLY).
    eval_rows = use[use["season"].isin(seasons)].copy().reset_index(drop=True)
    eval_rows = eval_rows.sort_values(["season"])  # align to seg order from fit_eval
    seB = (pB - y) ** 2
    seA = (pA - y) ** 2
    gb = eval_rows["min_gb"].to_numpy()
    print(f"\nBrier(B) - Brier(A) by season-progress bucket (negative = PTS helps):")
    print(f"  {'games_into_season':<20}{'n':>6}{'EloA':>9}{'+PTS B':>9}{'Δ':>9}")
    for lo, hi, lbl in [(2, 10, "early (2-10)"), (11, 25, "mid (11-25)"),
                        (26, 50, "late (26-50)"), (51, 99, "deep (51+)")]:
        m = (gb >= lo) & (gb <= hi)
        if m.sum() > 0:
            print(f"  {lbl:<20}{int(m.sum()):>6}{seA[m].mean():>9.4f}{seB[m].mean():>9.4f}"
                  f"{(seB[m].mean()-seA[m].mean()):>+9.4f}")
    print("\nNote: thin (local player logs = 2022-25 → 2 OOS seasons). Backfill "
          "player_game_logs (network) to extend before drawing a firm conclusion.")


if __name__ == "__main__":
    main()
