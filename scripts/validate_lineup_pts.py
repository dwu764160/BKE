"""
scripts/validate_lineup_pts.py
=============================================================================
Phase 3B Method 2 — Lineup-Level PTS Validation

For each 5-man lineup observed with >= MIN_POSSESSIONS in a season, compare
the lineup's predicted net rating (from the 5 players' PTS scores, scaled
by TEAM_SCALE) against the lineup's actual on-court net rating from PBP.

Why this matters: season-level Brier tests team-talent SUMS; it cannot
detect whether per-player scores are correctly calibrated relative to each
other. A model that inflates all players 2x and deflates TEAM_SCALE by 2x
gets identical Brier but wrong individual scores. Lineup-level r catches
that — only a model that correctly ranks individual contribution within a
team predicts which lineups perform well.

Source: data/historical/pbp_with_lineups_{season}.parquet
        — every row has exactly TWO non-null lineup columns (off + def
          team of that game). team_id on the row = team performing the
          action, so its lineup_{team_id} is the offensive lineup; the
          other non-null lineup_ column is the defensive lineup.

Possession proxy:
    POSS_off = count(FG attempts) + count(TURNOVER) + 0.44 * count(FT trips)
    Points scored = sum of `points` on rows attributed to off team.
    (FT trip = first FT in a sequence; we approximate via FT count / 2.)

The proxy is approximate but stable across teams/seasons, so relative
rankings of lineups are preserved — which is what we need for r.

Metrics output:
    Pearson r (predicted vs actual lineup NR), Spearman r, RMSE,
    possession-weighted Pearson r, calibration slope.

Usage:
    python3 scripts/validate_lineup_pts.py
    python3 scripts/validate_lineup_pts.py --pts-col offensive_portable_z \
        --pts-col-d defensive_portable_z --min-poss 30
    python3 scripts/validate_lineup_pts.py --pts-file /tmp/pts_v32.parquet
=============================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

PBP_GLOB = REPO / "data/historical"
DECOMP_DEFAULT = REPO / "data/processed/bke/bke_v28_decomposition.parquet"
OUTPUT_JSON = REPO / "reports/lineup_pts_validation.json"

SEASONS = ["2017-18", "2018-19", "2019-20", "2020-21", "2021-22",
           "2022-23", "2023-24", "2024-25"]


# ---------------------------------------------------------------------------
# Lineup possession aggregation
# ---------------------------------------------------------------------------

def aggregate_lineup_possessions(pbp: pd.DataFrame) -> pd.DataFrame:
    """For one season's PBP, produce per-(team_id, frozenset(lineup)) totals.

    Returns DataFrame with columns:
        team_id, lineup, off_poss, off_pts, def_poss, def_pts
    """
    lineup_cols = [c for c in pbp.columns if c.startswith("lineup_")]
    # Map team_id (int) → column name for fast lookup
    team_to_col = {int(c.split("_", 1)[1]): c for c in lineup_cols}

    # Mask events that count as offensive plays
    is_fga = pbp["event_type"].isin(["FIELD_GOAL_2PT", "FIELD_GOAL_3PT"])
    is_ft = pbp["event_type"] == "FREE_THROW"
    is_to = pbp["event_type"] == "TURNOVER"

    # Filter to events that bear on possession counting / scoring
    play_mask = is_fga | is_ft | is_to
    plays = pbp.loc[play_mask].copy()

    # Drop rows lacking team_id (e.g., period markers)
    plays = plays[plays["team_id"].notna()].copy()
    plays["team_id"] = plays["team_id"].astype(int)

    # Vectorize: for each row, find OFF and DEF lineup
    # Build an array of (off_team, off_lineup, def_team, def_lineup, poss_weight, points)
    rows_out = []
    # Group by team_id so we read one lineup column at a time (much faster)
    for off_team, sub in plays.groupby("team_id"):
        off_col = team_to_col.get(off_team)
        if off_col is None or off_col not in sub.columns:
            continue

        off_lineups = sub[off_col].values
        sub_event = sub["event_type"].values
        sub_pts = pd.to_numeric(sub.get("points", 0), errors="coerce").fillna(0).values

        # For each row in this off-team group, find the def lineup (the OTHER nonnull lineup col)
        other_cols = [c for c in lineup_cols if c != off_col]
        # Stack other lineup columns; pick the first non-null per row
        other_block = sub[other_cols]
        # Find non-null mask
        notna = other_block.notna()
        # First nonnull index per row
        # Convert to numpy for speed
        col_arr = np.array(other_cols)
        first_nonnull_idx = notna.values.argmax(axis=1)
        any_nonnull = notna.values.any(axis=1)
        def_col_per_row = np.where(any_nonnull, col_arr[first_nonnull_idx], None)

        # Extract def lineups
        def_lineups = np.empty(len(sub), dtype=object)
        for j, c in enumerate(other_cols):
            mask = (def_col_per_row == c)
            if mask.any():
                def_lineups[mask] = other_block[c].values[mask]
        def_teams = np.array([int(c.split("_", 1)[1]) if c else 0 for c in def_col_per_row])

        # Compute possession weights per row
        # FGA = 1, TO = 1, FT = 0.44 / 2 = 0.22 (approximate)
        poss_weight = np.where(
            (sub_event == "FIELD_GOAL_2PT") | (sub_event == "FIELD_GOAL_3PT"), 1.0,
            np.where(sub_event == "TURNOVER", 1.0,
            np.where(sub_event == "FREE_THROW", 0.22, 0.0))
        )

        # Build per-row records
        for i in range(len(sub)):
            ol = off_lineups[i]
            dl = def_lineups[i]
            if ol is None or dl is None:
                continue
            if not (hasattr(ol, "__len__") and len(ol) == 5):
                continue
            if not (hasattr(dl, "__len__") and len(dl) == 5):
                continue
            rows_out.append((
                off_team, frozenset(str(x) for x in ol),
                def_teams[i], frozenset(str(x) for x in dl),
                poss_weight[i], sub_pts[i],
            ))

    if not rows_out:
        return pd.DataFrame()

    df = pd.DataFrame(rows_out, columns=[
        "off_team", "off_lineup", "def_team", "def_lineup", "poss", "points",
    ])

    # Aggregate offensive side
    off_agg = df.groupby(["off_team", "off_lineup"]).agg(
        off_poss=("poss", "sum"),
        off_pts=("points", "sum"),
    ).reset_index().rename(columns={"off_team": "team_id", "off_lineup": "lineup"})

    # Aggregate defensive side
    def_agg = df.groupby(["def_team", "def_lineup"]).agg(
        def_poss=("poss", "sum"),
        def_pts=("points", "sum"),
    ).reset_index().rename(columns={"def_team": "team_id", "def_lineup": "lineup"})

    merged = off_agg.merge(def_agg, on=["team_id", "lineup"], how="outer").fillna(0.0)
    return merged


# ---------------------------------------------------------------------------
# Predicted lineup net rating
# ---------------------------------------------------------------------------

def predict_lineup_net_rating(
    lineups: pd.DataFrame,
    pts_table: pd.DataFrame,
    pts_col_o: str,
    pts_col_d: str,
    team_scale: float,
    off_def_split: Tuple[float, float] = (0.5, 0.5),
) -> pd.DataFrame:
    """For each lineup, predict net rating from 5 players' PTS scores.

    predicted_off = TEAM_SCALE * mean(PTS_O) for the 5 players
    predicted_def = TEAM_SCALE * mean(PTS_D) for the 5 players
    predicted_net = off_split * predicted_off - def_split * predicted_def

    (Net rating convention: higher off, lower def allowed = better.
     Our PTS_D is "more positive = better defender", so we subtract.)
    """
    # Build {player_id: (pts_o, pts_d)} lookup
    p = pts_table[["player_id", pts_col_o, pts_col_d]].dropna(subset=["player_id"]).copy()
    p["player_id"] = p["player_id"].astype(str)
    p[pts_col_o] = pd.to_numeric(p[pts_col_o], errors="coerce").fillna(0.0)
    p[pts_col_d] = pd.to_numeric(p[pts_col_d], errors="coerce").fillna(0.0)
    lookup_o = dict(zip(p["player_id"], p[pts_col_o]))
    lookup_d = dict(zip(p["player_id"], p[pts_col_d]))

    def _avg(lineup: frozenset, table: dict) -> float:
        vals = [table.get(str(pid), 0.0) for pid in lineup]
        return float(np.mean(vals)) if vals else 0.0

    lineups = lineups.copy()
    lineups["pts_o_mean"] = lineups["lineup"].map(lambda lu: _avg(lu, lookup_o))
    lineups["pts_d_mean"] = lineups["lineup"].map(lambda lu: _avg(lu, lookup_d))

    off_w, def_w = off_def_split
    lineups["predicted_off_nr"] = team_scale * lineups["pts_o_mean"]
    lineups["predicted_def_nr"] = team_scale * lineups["pts_d_mean"]
    lineups["predicted_net_rating"] = (
        off_w * lineups["predicted_off_nr"] + def_w * lineups["predicted_def_nr"]
    )
    return lineups


# ---------------------------------------------------------------------------
# Validation metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    df: pd.DataFrame,
    min_poss: int,
) -> Dict:
    """Compute Pearson r, Spearman r, RMSE, weighted r, calibration slope."""
    df = df[(df["off_poss"] >= min_poss) & (df["def_poss"] >= min_poss)].copy()
    if len(df) < 10:
        return {"error": f"Too few lineups (n={len(df)} after filter)"}

    # Actual net rating per 100 possessions
    df["actual_off_nr"] = df["off_pts"] / df["off_poss"] * 100.0
    df["actual_def_nr"] = df["def_pts"] / df["def_poss"] * 100.0
    df["actual_net_rating"] = df["actual_off_nr"] - df["actual_def_nr"]

    # Drop any inf / nan after division
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["actual_net_rating", "predicted_net_rating"])
    if len(df) < 10:
        return {"error": "Too few lineups after NaN filter"}

    pearson_r, pearson_p = stats.pearsonr(df["predicted_net_rating"], df["actual_net_rating"])
    spearman_r, spearman_p = stats.spearmanr(df["predicted_net_rating"], df["actual_net_rating"])

    weights = np.sqrt(df["off_poss"].clip(lower=1))
    cov = np.cov(df["predicted_net_rating"], df["actual_net_rating"], aweights=weights)
    if cov[0, 0] > 0 and cov[1, 1] > 0:
        weighted_r = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    else:
        weighted_r = float("nan")

    rmse = float(np.sqrt(np.mean((df["predicted_net_rating"] - df["actual_net_rating"]) ** 2)))
    mae = float(np.mean(np.abs(df["predicted_net_rating"] - df["actual_net_rating"])))

    # Calibration slope: regress actual on predicted
    slope, intercept = np.polyfit(df["predicted_net_rating"], df["actual_net_rating"], 1)

    # Offense / defense separately
    off_pearson, _ = stats.pearsonr(df["predicted_off_nr"], df["actual_off_nr"])
    def_pearson, _ = stats.pearsonr(df["predicted_def_nr"], df["actual_def_nr"])

    return {
        "n_lineups": int(len(df)),
        "min_possessions": int(min_poss),
        "pearson_r": round(float(pearson_r), 4),
        "spearman_r": round(float(spearman_r), 4),
        "weighted_pearson_r": round(float(weighted_r), 4),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "calibration_slope": round(float(slope), 4),
        "calibration_intercept": round(float(intercept), 4),
        "offense_pearson_r": round(float(off_pearson), 4),
        "defense_pearson_r": round(float(def_pearson), 4),
        "actual_nr_mean": round(float(df["actual_net_rating"].mean()), 4),
        "actual_nr_std": round(float(df["actual_net_rating"].std()), 4),
        "predicted_nr_mean": round(float(df["predicted_net_rating"].mean()), 4),
        "predicted_nr_std": round(float(df["predicted_net_rating"].std()), 4),
    }


# ---------------------------------------------------------------------------
# Per-season pipeline
# ---------------------------------------------------------------------------

def validate_season(
    season: str,
    pts_table: pd.DataFrame,
    pts_col_o: str,
    pts_col_d: str,
    team_scale: float,
    min_poss: int,
    off_def_split: Tuple[float, float],
) -> Optional[Dict]:
    """Run the full pipeline for one season."""
    pbp_path = PBP_GLOB / f"pbp_with_lineups_{season}.parquet"
    if not pbp_path.exists():
        return None

    pbp = pd.read_parquet(pbp_path)
    lineups = aggregate_lineup_possessions(pbp)
    if lineups.empty:
        return {"error": "no lineups aggregated"}

    pts_season = pts_table[pts_table["season"] == season].copy()
    if pts_season.empty:
        return {"error": f"no PTS data for season {season}"}

    lineups = predict_lineup_net_rating(
        lineups, pts_season, pts_col_o, pts_col_d,
        team_scale=team_scale, off_def_split=off_def_split,
    )
    metrics = compute_metrics(lineups, min_poss)
    metrics["season"] = season
    metrics["lineups_seen_total"] = int(len(lineups))
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> Dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pts-file", default=str(DECOMP_DEFAULT),
                        help="Parquet with PTS columns (default: decomp v28)")
    parser.add_argument("--pts-col", default="offensive_portable_z",
                        help="Offensive PTS column name")
    parser.add_argument("--pts-col-d", default="defensive_portable_z",
                        help="Defensive PTS column name")
    parser.add_argument("--team-scale", type=float, default=20.0,
                        help="TEAM_SCALE multiplier (default: 20.0)")
    parser.add_argument("--min-poss", type=int, default=30,
                        help="Minimum possessions per lineup side")
    parser.add_argument("--off-def-split", type=float, nargs=2, default=[0.5, 0.5],
                        help="Off / Def weight in net rating (default: 0.5 0.5)")
    parser.add_argument("--seasons", nargs="*", default=None,
                        help="Subset of seasons to evaluate")
    parser.add_argument("--output", default=str(OUTPUT_JSON))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    seasons = args.seasons or SEASONS
    pts_table = pd.read_parquet(args.pts_file)
    pts_table["player_id"] = pts_table["player_id"].astype(str)
    pts_table["season"] = pts_table["season"].astype(str)

    if not args.quiet:
        print(f"\nLineup PTS Validation")
        print(f"PTS file:   {args.pts_file}")
        print(f"PTS_O col:  {args.pts_col}    PTS_D col: {args.pts_col_d}")
        print(f"TEAM_SCALE: {args.team_scale}    min_poss: {args.min_poss}")
        print(f"Off/Def split: {args.off_def_split}")
        print("-" * 70)

    per_season = {}
    all_pred = []
    all_actual = []
    all_weights = []

    for season in seasons:
        if not args.quiet:
            print(f"  {season:8s} ", end="", flush=True)
        result = validate_season(
            season, pts_table, args.pts_col, args.pts_col_d,
            team_scale=args.team_scale, min_poss=args.min_poss,
            off_def_split=tuple(args.off_def_split),
        )
        if result is None:
            if not args.quiet:
                print(" (no PBP data)")
            continue
        per_season[season] = result
        if "error" in result:
            if not args.quiet:
                print(f" ERROR: {result['error']}")
            continue
        if not args.quiet:
            print(f"n={result['n_lineups']:5d}  r={result['pearson_r']:+.3f}  "
                  f"weighted_r={result['weighted_pearson_r']:+.3f}  "
                  f"rmse={result['rmse']:.2f}")

    # Pooled aggregate
    pooled_metrics = {}
    valid = [(s, r) for s, r in per_season.items() if "error" not in r]
    if valid:
        rs = [r["pearson_r"] for _, r in valid]
        wrs = [r["weighted_pearson_r"] for _, r in valid if not np.isnan(r["weighted_pearson_r"])]
        rmses = [r["rmse"] for _, r in valid]
        ns = [r["n_lineups"] for _, r in valid]
        pooled_metrics = {
            "n_seasons": len(valid),
            "total_lineups": int(sum(ns)),
            "mean_pearson_r": round(float(np.mean(rs)), 4),
            "median_pearson_r": round(float(np.median(rs)), 4),
            "mean_weighted_pearson_r": round(float(np.mean(wrs)), 4) if wrs else None,
            "mean_rmse": round(float(np.mean(rmses)), 4),
            "n_weighted_mean_r": round(float(np.average(rs, weights=ns)), 4),
        }
        if not args.quiet:
            print("-" * 70)
            print(f"  Pooled: mean r={pooled_metrics['mean_pearson_r']:+.3f}  "
                  f"weighted r={pooled_metrics['mean_weighted_pearson_r']}  "
                  f"mean rmse={pooled_metrics['mean_rmse']:.2f}")

    output = {
        "config": {
            "pts_file": args.pts_file,
            "pts_col_o": args.pts_col,
            "pts_col_d": args.pts_col_d,
            "team_scale": args.team_scale,
            "min_poss": args.min_poss,
            "off_def_split": args.off_def_split,
        },
        "per_season": per_season,
        "pooled": pooled_metrics,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=str)
    if not args.quiet:
        print(f"\nSaved → {args.output}")
    return output


if __name__ == "__main__":
    main()
