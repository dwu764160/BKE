"""
scripts/build_pts_team_ratings.py
=============================================================================
Phase 1 — aggregate player PTS v4.0 into TEAM ratings, for all 9 seasons.

This wires the player-impact metric (PTS v4.0) into a team-strength number for
the first time (the production game model never used it — see
docs/findings/game_model_comparison_2026-05-30.md).

Variants (the meaningful cells of the 2×2 the user asked for):
  A. PTS-only additive      — team_net = Σ minute_share · (pts_o_v40 + pts_d_v40).
                              This is exactly `off_talent_base` from
                              src/profile_aggregate/team_feature_aggregation.py:309
                              with the impact column swapped to PTS.
  B. PTS + archetype-fit     — A plus a *fit-only* composite (spacing / rim
                              protection / creator redundancy) that adds NO talent
                              level (PTS already carries level → no double-count).

Note on C/D ("interaction-aware revived model"): the repo's own Lasso fit found
archetype-PAIR interactions negligible — `INTERACTION_MATRIX = {}`
(team_feature_aggregation.py:167-196). So the revived interaction model reduces
to A + structure term ≈ B. We therefore build A and B (the two distinct cells)
and ALSO emit the raw fit-composition columns so the GBDT can weight them itself
("let the model decide", per the interview).

Leakage: this file produces SAME-SEASON ratings (for the correlation gate). The
walk-forward experiment maps season N ratings built from season **N-1** PTS.

Output: data/processed/forecast/team_pts_ratings.parquet
Run:    python3 scripts/build_pts_team_ratings.py
=============================================================================
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.simulation.simulation_config import FORECAST_DIR

AGG_PATH = Path("aggregate/player_profile_aggregate.parquet")
OUT_PATH = FORECAST_DIR / "team_pts_ratings.parquet"

# Archetype groupings for the fit-only term (offense uses primary_archetype,
# defense uses defensive_archetype).
SHOOTERS = {"Off-Ball Stationary Shooter", "Off-Ball Movement Shooter", "Perimeter Scorer"}
CREATORS = {"Ball Dominant Creator", "Ballhandler", "All-Around Scorer"}
BALL_DOMINANT = {"Ball Dominant Creator", "Ballhandler"}
RIM_PROTECT = {"Rim Protector", "Dropping Big", "Mobile Big"}


def _wshare(g, mask):
    """Minute-weighted share of players whose archetype is in `mask`."""
    return float((g["agg_minute_share"] * mask.astype(float)).sum())


def build() -> pd.DataFrame:
    cols = ["season", "team_abbreviation", "agg_minute_share", "min",
            "pts_o_v40", "pts_d_v40", "primary_archetype", "defensive_archetype"]
    agg = pd.read_parquet(AGG_PATH, columns=cols)
    agg = agg[agg["team_abbreviation"].notna() & (agg["agg_minute_share"] > 0)].copy()
    agg["team_abbreviation"] = agg["team_abbreviation"].str.upper()

    rows = []
    for (season, team), g in agg.groupby(["season", "team_abbreviation"]):
        # Variant A — PTS-only additive (minute-weighted player net impact).
        pts_off = float((g["agg_minute_share"] * g["pts_o_v40"]).sum())
        pts_def = float((g["agg_minute_share"] * g["pts_d_v40"]).sum())
        pts_net = pts_off + pts_def

        # Fit-only composition signals (no talent level).
        prim = g["primary_archetype"].fillna("")
        deff = g["defensive_archetype"].fillna("")
        fit_spacing = _wshare(g, prim.isin(SHOOTERS))
        fit_creator = _wshare(g, prim.isin(CREATORS))
        fit_ball_dom = _wshare(g, prim.isin(BALL_DOMINANT))
        fit_rim = _wshare(g, deff.isin(RIM_PROTECT))
        rotation = g[g["agg_minute_share"] > 0.08]
        fit_diversity = int(rotation["primary_archetype"].dropna().nunique())

        # Variant B — A + an illustrative fit composite (small coefficients;
        # the GBDT path below uses the raw fit_* columns instead of these weights).
        pts_net_fit = (pts_net
                       + 0.50 * (fit_spacing - 0.30)
                       + 0.50 * fit_rim
                       - 0.50 * max(0.0, fit_ball_dom - 0.40))

        rows.append({
            "season": season, "team_abbreviation": team,
            "pts_off": pts_off, "pts_def": pts_def, "pts_net": pts_net,
            "pts_net_fit": pts_net_fit,
            "fit_spacing": fit_spacing, "fit_creator": fit_creator,
            "fit_ball_dom": fit_ball_dom, "fit_rim": fit_rim,
            "fit_diversity": fit_diversity,
            "n_players": int(len(g)),
        })

    df = pd.DataFrame(rows).sort_values(["season", "team_abbreviation"]).reset_index(drop=True)
    return df


def main():
    df = build()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"Wrote {len(df)} team-season ratings to {OUT_PATH}")
    print(f"Seasons: {sorted(df['season'].unique())}")
    print(f"Teams/season: {df.groupby('season').size().to_dict()}")
    print("\npts_net by season (std across teams — should resemble talent spread):")
    print(df.groupby("season")["pts_net"].agg(["mean", "std"]).round(4).to_string())
    print("\nSample (2023-24 top/bottom by pts_net):")
    s = df[df.season == "2023-24"].sort_values("pts_net", ascending=False)
    print(s[["team_abbreviation", "pts_net", "fit_spacing", "fit_rim", "fit_diversity"]]
          .head(4).to_string(index=False))
    print(s[["team_abbreviation", "pts_net", "fit_spacing", "fit_rim", "fit_diversity"]]
          .tail(3).to_string(index=False))


if __name__ == "__main__":
    main()
