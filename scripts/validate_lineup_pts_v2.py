"""
scripts/validate_lineup_pts_v2.py
=============================================================================
Lineup-Level PTS Validation v2 (Phase A fixes)

Improvements over scripts/validate_lineup_pts.py:

  A.1  Naive baselines reported alongside PTS:
       - "zero" model    : every player score = 0 (chance prediction)
       - "per100_NR" model: player's actual on-court per-100 Net Rating from
                          team_game_logs (no PTS, no archetype, no shrinkage)
       Tells us the floor that PTS must beat to claim it adds signal.

  A.2  Possession-weighted Pearson r is the primary headline metric.
       Unweighted r mixes 50-poss and 3000-poss lineups equally.
       Also fit LINEUP_TEAM_SCALE empirically per season as the OLS slope
       of actual_NR ~ avg_PTS_O + avg_PTS_D, then re-report RMSE on the
       calibrated scale.

  A.3  Independent offense and defense validation:
       - actual_off_rtg = off_pts / off_poss * 100
       - actual_def_rtg = def_pts / def_poss * 100  (lower = better)
       - report r(predicted_off, actual_off_rtg)
       - report r(predicted_def, -actual_def_rtg)  (sign-flipped: positive r = good)
       Exposes whether defensive PTS is well-calibrated independently.

Output: reports/lineup_v2_<label>.json with per-season + pooled metrics for
        each model variant (pts | per100 | zero) and each axis (joint | off | def).

Usage:
    # Validate v3.2 PTS (production)
    python3 scripts/validate_lineup_pts_v2.py \\
        --pts-file data/processed/bke/pts_v32.parquet \\
        --pts-col pts_o_v32 --pts-col-d pts_d_v32 \\
        --label v32

    # Validate the legacy v2.7 offensive_portable_z / defensive_portable_z
    python3 scripts/validate_lineup_pts_v2.py \\
        --pts-file data/processed/bke/bke_v28_decomposition.parquet \\
        --pts-col offensive_portable_z --pts-col-d defensive_portable_z \\
        --label v27
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

from scripts.validate_lineup_pts import aggregate_lineup_possessions

PBP_DIR = REPO / "data/historical"
PLAYER_GAME_LOGS = REPO / "data/historical/final_player_game_logs.parquet"
DECOMP_DEFAULT = REPO / "data/processed/bke/bke_v28_decomposition.parquet"
DEFAULT_OUTPUT = REPO / "reports/lineup_v2.json"

SEASONS = ["2017-18", "2018-19", "2019-20", "2020-21", "2021-22",
           "2022-23", "2023-24", "2024-25"]


# ---------------------------------------------------------------------------
# Per-player on-court NetRtg baseline (Phase A.1)
# ---------------------------------------------------------------------------

def build_player_on_court_nrtg(season: str) -> pd.DataFrame:
    """For each player active in `season`, compute their season-aggregate
    on-court NetRtg per 100 possessions from PBP-derived lineup totals.

    Approach: aggregate lineup possessions then re-attribute to each of the
    5 players in the lineup. A player's per-100 NetRtg = sum over their
    lineups of (off_pts - def_pts) / sum possessions * 100.
    """
    pbp_path = PBP_DIR / f"pbp_with_lineups_{season}.parquet"
    if not pbp_path.exists():
        return pd.DataFrame()
    pbp = pd.read_parquet(pbp_path)
    lineups = aggregate_lineup_possessions(pbp)
    if lineups.empty:
        return pd.DataFrame()

    # Explode 5 players per lineup
    rows = []
    for _, lu in lineups.iterrows():
        members = list(lu["lineup"])
        for pid in members:
            rows.append({
                "player_id": str(pid),
                "off_pts": lu["off_pts"] / 5.0,    # per-player share
                "off_poss": lu["off_poss"] / 5.0,
                "def_pts": lu["def_pts"] / 5.0,
                "def_poss": lu["def_poss"] / 5.0,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    g = df.groupby("player_id").sum(numeric_only=True).reset_index()
    g["per100_off"] = np.where(g["off_poss"] > 0, g["off_pts"] / g["off_poss"] * 100.0, 0)
    g["per100_def"] = np.where(g["def_poss"] > 0, g["def_pts"] / g["def_poss"] * 100.0, 0)
    g["per100_net"] = g["per100_off"] - g["per100_def"]
    # Centre and unit-std per season so comparable scale to PTS
    g["per100_off_z"] = (g["per100_off"] - g["per100_off"].mean()) / max(g["per100_off"].std(), 1e-6)
    g["per100_def_z"] = -(g["per100_def"] - g["per100_def"].mean()) / max(g["per100_def"].std(), 1e-6)
    g["season"] = season
    return g[["player_id", "season", "per100_off", "per100_def", "per100_net",
              "per100_off_z", "per100_def_z", "off_poss", "def_poss"]]


# ---------------------------------------------------------------------------
# Predicted lineup ORtg / DRtg / Net (Phase A.3)
# ---------------------------------------------------------------------------

def predict_lineup(
    lineups: pd.DataFrame,
    pts_table: pd.DataFrame,
    o_col: str, d_col: str,
    scale: float,
) -> pd.DataFrame:
    """Return lineups with predicted_off_nr, predicted_def_nr, predicted_net_nr."""
    p = pts_table[["player_id", o_col, d_col]].dropna(subset=["player_id"]).copy()
    p["player_id"] = p["player_id"].astype(str)
    p[o_col] = pd.to_numeric(p[o_col], errors="coerce").fillna(0.0)
    p[d_col] = pd.to_numeric(p[d_col], errors="coerce").fillna(0.0)
    look_o = dict(zip(p["player_id"], p[o_col]))
    look_d = dict(zip(p["player_id"], p[d_col]))

    def _avg(lu: frozenset, table: dict) -> float:
        vals = [table.get(str(pid), 0.0) for pid in lu]
        return float(np.mean(vals)) if vals else 0.0

    out = lineups.copy()
    out["pts_o_mean"] = out["lineup"].map(lambda lu: _avg(lu, look_o))
    out["pts_d_mean"] = out["lineup"].map(lambda lu: _avg(lu, look_d))
    out["predicted_off_nr"] = scale * out["pts_o_mean"]
    out["predicted_def_nr"] = scale * out["pts_d_mean"]
    out["predicted_net_nr"] = out["predicted_off_nr"] + out["predicted_def_nr"]
    return out


# ---------------------------------------------------------------------------
# Metrics with off / def split + possession-weighted r
# ---------------------------------------------------------------------------

def compute_split_metrics(lineups: pd.DataFrame, min_poss: int) -> Dict:
    """Phase A.2 + A.3 metrics: joint, off, def each with possession-weighted r."""
    df = lineups[(lineups["off_poss"] >= min_poss) & (lineups["def_poss"] >= min_poss)].copy()
    if len(df) < 10:
        return {"error": f"too few lineups (n={len(df)})"}

    df["actual_off_nr"] = df["off_pts"] / df["off_poss"] * 100.0
    df["actual_def_nr"] = df["def_pts"] / df["def_poss"] * 100.0   # lower = better
    df["actual_net_nr"] = df["actual_off_nr"] - df["actual_def_nr"]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[
        "actual_net_nr", "predicted_net_nr", "actual_off_nr", "actual_def_nr",
    ])
    if len(df) < 10:
        return {"error": "too few after NaN filter"}

    n = len(df)
    weights = np.sqrt(np.minimum(df["off_poss"], df["def_poss"]).clip(lower=1))

    def _r(pred: pd.Series, actual: pd.Series) -> Tuple[float, float]:
        if pred.std() < 1e-9 or actual.std() < 1e-9:
            return 0.0, 0.0
        r, _ = stats.pearsonr(pred, actual)
        cov = np.cov(pred, actual, aweights=weights)
        wr = cov[0, 1] / max(np.sqrt(cov[0, 0] * cov[1, 1]), 1e-9) if (
            cov[0, 0] > 0 and cov[1, 1] > 0
        ) else float("nan")
        return float(r), float(wr)

    # Joint net rating
    joint_r, joint_wr = _r(df["predicted_net_nr"], df["actual_net_nr"])
    # Offense (positive correlation expected)
    off_r, off_wr = _r(df["predicted_off_nr"], df["actual_off_nr"])
    # Defense: predicted_def_nr is positive=good, actual_def_nr is positive=BAD;
    #   so we sign-flip actual so that positive correlation = good model
    def_r, def_wr = _r(df["predicted_def_nr"], -df["actual_def_nr"])

    # Empirical LINEUP scale = slope of actual ~ predicted (joint)
    if df["predicted_net_nr"].std() < 1e-9:
        slope, intercept = 0.0, float(df["actual_net_nr"].mean())
    else:
        slope, intercept = np.polyfit(df["predicted_net_nr"], df["actual_net_nr"], 1)
    # RMSE on calibrated scale
    calibrated_pred = slope * df["predicted_net_nr"] + intercept
    rmse_calibrated = float(np.sqrt(np.mean((calibrated_pred - df["actual_net_nr"]) ** 2)))
    rmse_raw = float(np.sqrt(np.mean((df["predicted_net_nr"] - df["actual_net_nr"]) ** 2)))

    return {
        "n_lineups": int(n),
        "min_possessions": int(min_poss),
        "joint": {
            "pearson_r": round(joint_r, 4),
            "weighted_pearson_r": round(joint_wr, 4),
            "rmse_raw": round(rmse_raw, 4),
            "rmse_calibrated": round(rmse_calibrated, 4),
            "calibration_slope": round(float(slope), 4),
            "calibration_intercept": round(float(intercept), 4),
        },
        "offense": {
            "pearson_r": round(off_r, 4),
            "weighted_pearson_r": round(off_wr, 4),
            "predicted_std": round(float(df["predicted_off_nr"].std()), 4),
            "actual_std": round(float(df["actual_off_nr"].std()), 4),
        },
        "defense": {
            # positive r = model correctly identifies good defenders
            "pearson_r": round(def_r, 4),
            "weighted_pearson_r": round(def_wr, 4),
            "predicted_std": round(float(df["predicted_def_nr"].std()), 4),
            "actual_std": round(float(df["actual_def_nr"].std()), 4),
            "note": "positive r = model ranks better defenders correctly (actual sign flipped)",
        },
    }


# ---------------------------------------------------------------------------
# Per-season pipeline
# ---------------------------------------------------------------------------

def validate_season(
    season: str, pts_table: pd.DataFrame, scale: float,
    min_poss: int, models: dict,
) -> Dict:
    """Run all models for a season and return per-model metrics."""
    pbp_path = PBP_DIR / f"pbp_with_lineups_{season}.parquet"
    if not pbp_path.exists():
        return {"error": "no PBP"}
    pbp = pd.read_parquet(pbp_path)
    lineups = aggregate_lineup_possessions(pbp)
    if lineups.empty:
        return {"error": "no lineups"}

    # Naive baseline: per-100 NetRtg
    per100 = build_player_on_court_nrtg(season)
    per100 = per100[["player_id", "per100_off_z", "per100_def_z"]] if not per100.empty else None

    out = {}
    for name, (o_col, d_col, src) in models.items():
        if src == "pts":
            src_table = pts_table[pts_table["season"] == season]
        elif src == "per100":
            if per100 is None or per100.empty:
                out[name] = {"error": "no per100 data"}
                continue
            src_table = per100
        elif src == "zero":
            src_table = pts_table[pts_table["season"] == season].copy()
            for c in (o_col, d_col):
                src_table[c] = 0.0
        else:
            out[name] = {"error": f"unknown source {src}"}
            continue

        pred = predict_lineup(lineups, src_table, o_col, d_col, scale=scale)
        out[name] = compute_split_metrics(pred, min_poss)
    return out


# ---------------------------------------------------------------------------
# Pool across seasons
# ---------------------------------------------------------------------------

def pool(per_season: dict, model: str) -> Dict:
    rows = []
    for s, m in per_season.items():
        if model not in m:
            continue
        v = m[model]
        if "error" in v:
            continue
        rows.append(v)
    if not rows:
        return {}
    return {
        "n_seasons": len(rows),
        "total_lineups": int(sum(r["n_lineups"] for r in rows)),
        "joint_mean_r": round(float(np.mean([r["joint"]["pearson_r"] for r in rows])), 4),
        "joint_mean_wr": round(float(np.mean([r["joint"]["weighted_pearson_r"] for r in rows])), 4),
        "offense_mean_r": round(float(np.mean([r["offense"]["pearson_r"] for r in rows])), 4),
        "offense_mean_wr": round(float(np.mean([r["offense"]["weighted_pearson_r"] for r in rows])), 4),
        "defense_mean_r": round(float(np.mean([r["defense"]["pearson_r"] for r in rows])), 4),
        "defense_mean_wr": round(float(np.mean([r["defense"]["weighted_pearson_r"] for r in rows])), 4),
        "mean_rmse_raw": round(float(np.mean([r["joint"]["rmse_raw"] for r in rows])), 4),
        "mean_rmse_calibrated": round(float(np.mean([r["joint"]["rmse_calibrated"] for r in rows])), 4),
        "mean_calibration_slope": round(float(np.mean([r["joint"]["calibration_slope"] for r in rows])), 4),
    }


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pts-file", default=str(DECOMP_DEFAULT))
    p.add_argument("--pts-col", default="offensive_portable_z")
    p.add_argument("--pts-col-d", default="defensive_portable_z")
    p.add_argument("--team-scale", type=float, default=20.0)
    p.add_argument("--min-poss", type=int, default=50)
    p.add_argument("--seasons", nargs="*", default=None)
    p.add_argument("--label", default="v2_run")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = p.parse_args()

    pts_table = pd.read_parquet(args.pts_file)
    pts_table["player_id"] = pts_table["player_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    pts_table["season"] = pts_table["season"].astype(str)
    seasons = args.seasons or SEASONS

    # Three models to compare
    models = {
        "pts":     (args.pts_col,    args.pts_col_d,    "pts"),
        "per100":  ("per100_off_z",  "per100_def_z",    "per100"),
        "zero":    (args.pts_col,    args.pts_col_d,    "zero"),
    }

    print(f"\nLineup PTS Validation v2 — label={args.label}")
    print(f"PTS file: {args.pts_file}  | {args.pts_col} / {args.pts_col_d}")
    print(f"scale={args.team_scale}  min_poss={args.min_poss}  seasons={seasons}")
    print("-" * 80)

    per_season = {}
    for s in seasons:
        print(f"  {s} ...", end="", flush=True)
        per_season[s] = validate_season(s, pts_table, args.team_scale, args.min_poss, models)
        if "error" in per_season[s]:
            print(f" {per_season[s]['error']}")
            continue
        m = per_season[s]
        # Brief per-season summary
        pts_m = m.get("pts", {}).get("joint", {})
        p100_m = m.get("per100", {}).get("joint", {})
        zero_m = m.get("zero", {}).get("joint", {})
        print(f" n={m.get('pts',{}).get('n_lineups','?'):>4}  "
              f"PTS r={pts_m.get('pearson_r','?')} wr={pts_m.get('weighted_pearson_r','?')}  "
              f"per100 r={p100_m.get('pearson_r','?')} wr={p100_m.get('weighted_pearson_r','?')}  "
              f"zero r={zero_m.get('pearson_r','?')}")

    pooled = {name: pool(per_season, name) for name in models}
    print("\n=== POOLED ===")
    for name, p_ in pooled.items():
        if not p_:
            continue
        print(f"  {name:8s}  joint r={p_['joint_mean_r']:+.3f} wr={p_['joint_mean_wr']:+.3f}  "
              f"off r={p_['offense_mean_r']:+.3f}  def r={p_['defense_mean_r']:+.3f}  "
              f"slope={p_['mean_calibration_slope']:.2f}  rmse_cal={p_['mean_rmse_calibrated']:.2f}")

    # Empirical lineup scale (mean calibration slope across seasons) for v3.2 PTS
    pts_pool = pooled.get("pts", {})
    if pts_pool.get("mean_calibration_slope"):
        empirical_scale = args.team_scale * pts_pool["mean_calibration_slope"]
        print(f"\nEMPIRICAL LINEUP_TEAM_SCALE for PTS = {empirical_scale:.1f}  "
              f"(team scale {args.team_scale} × mean slope {pts_pool['mean_calibration_slope']:.2f})")

    output = {
        "config": {
            "pts_file": args.pts_file, "pts_col_o": args.pts_col, "pts_col_d": args.pts_col_d,
            "team_scale": args.team_scale, "min_poss": args.min_poss, "label": args.label,
        },
        "per_season": per_season,
        "pooled": pooled,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
