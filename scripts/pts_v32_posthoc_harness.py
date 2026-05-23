"""
scripts/pts_v32_posthoc_harness.py
=============================================================================
v3.2 PTS Post-Hoc Adjustment Harness

Uses existing `offensive_portable_z` and `defensive_portable_z` from the
decomp parquet AS-IS (so it reproduces the published "Pure PTS α=1.00"
baseline exactly: Brier=0.2248). Then applies surgical post-hoc
adjustments and measures the delta:

  Adjustment A — Matchup-based Dim 6 RESIDUAL:
      def_v32 = defensive_portable_z
                + dim_d6_residual_weight * (matchup_dim6_z - dim_defensive_impact_z)
      effectively swaps in matchup signal for the DRAPM-driven dim6 component.

  Adjustment B — Dim 5 weight reduction:
      def_v32 -= dim_d5_reduction * dim_defensive_playmaking_z

  Adjustment C — Offense-heavy global rebalance:
      Apply a scaling factor to offense vs defense PTS.

  Adjustment D — Multi-season smoothing (current * 0.7 + prior * 0.3):
      Cheap noise reduction for early-career players.

  Adjustment E — Star amplification at team aggregation (top-1, top-2).

The baseline (no adjustments) reproduces 0.2248 Brier. We then sweep
adjustments and pick the joint winner.

Usage:
  python3 scripts/pts_v32_posthoc_harness.py --baseline       # confirm 0.2248
  python3 scripts/pts_v32_posthoc_harness.py --label v32_a
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import subprocess
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DECOMP_PATH = REPO / "data/processed/bke/bke_v28_decomposition.parquet"
DEF_ARCH_PATH = REPO / "data/processed/defensive_archetypes_v2.parquet"
PROJ_PROFILES = REPO / "data/processed/forecast/projected_player_profiles.parquet"
PROJ_TEAM_FEAT = REPO / "data/processed/forecast/projected_team_features.parquet"
VALIDATE_FORECAST = REPO / "src/simulation/validate_forecast.py"
VALIDATE_LINEUP = REPO / "scripts/validate_lineup_pts.py"
DEFAULT_TEAM_SCALE = 20.0


# ---------------------------------------------------------------------------

@dataclass
class PostHocConfig:
    # Adjustment A — Matchup Dim 6 residual swap
    use_matchup_dim6_residual: bool = False
    matchup_dim6_strength: float = 0.0       # 0.0 = off; 1.0 = full swap

    # Adjustment B — Dim 5 weight reduction
    dim5_weight_reduction: float = 0.0       # 0.0 = off; 0.03 = remove 3pp of dim5 weight

    # Adjustment C — Offense / Defense scaling
    offense_gain: float = 1.0
    defense_gain: float = 1.0

    # Adjustment D — Multi-season smoothing
    multi_season_smoothing: float = 0.0

    # Adjustment E — Star amplification at team aggregation
    star_amp_top1: float = 1.0
    star_amp_top2: float = 1.0

    # Optional defensive Bayesian shrinkage toward archetype mean
    defensive_archetype_shrinkage: float = 0.0

    # Final clip
    final_clip: float = 4.0


# ---------------------------------------------------------------------------
# Matchup Dim 6 z (from defensive_archetypes_v2)
# ---------------------------------------------------------------------------

def build_matchup_dim6_z(def_arch: pd.DataFrame) -> pd.DataFrame:
    """Return [player_id, season, matchup_dim6_z] composite z-score."""
    df = def_arch[["PLAYER_ID", "SEASON"]].copy()
    df["player_id"] = df["PLAYER_ID"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["season"] = df["SEASON"].astype(str)

    components = {
        "d_results_pctl": 0.35,
        "D_FG_DIFF": 0.30,
        "contested_shots_pctl": 0.15,
        "rim_protection_index_pctl": 0.20,
    }
    invert = {"D_FG_DIFF"}  # low = good → invert

    composite = pd.Series(0.0, index=def_arch.index)
    total_w = 0.0
    for col, w in components.items():
        if col not in def_arch.columns:
            continue
        vals = pd.to_numeric(def_arch[col], errors="coerce")
        z = vals.groupby(def_arch["SEASON"].astype(str)).transform(
            lambda x: (x - x.mean()) / max(x.std(), 1e-6)
        ).fillna(0)
        if col in invert:
            z = -z
        composite = composite + w * z
        total_w += w
    if total_w > 0:
        composite = composite / total_w

    df["matchup_dim6_z"] = composite.values
    df["matchup_dim6_z"] = df.groupby("season")["matchup_dim6_z"].transform(
        lambda x: (x - x.mean()) / max(x.std(), 1e-6)
    )
    return df[["player_id", "season", "matchup_dim6_z"]]


# ---------------------------------------------------------------------------
# Apply adjustments
# ---------------------------------------------------------------------------

def apply_posthoc(
    decomp: pd.DataFrame, def_arch: pd.DataFrame, cfg: PostHocConfig,
) -> pd.DataFrame:
    """Return DataFrame with columns: player_id, season, pts_o_v32, pts_d_v32"""
    df = decomp[[
        "player_id", "season",
        "offensive_portable_z", "defensive_portable_z",
        "dim_defensive_impact_z", "dim_defensive_playmaking_z",
        "primary_archetype",
    ]].copy()
    df["player_id"] = df["player_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["season"] = df["season"].astype(str)
    for c in ["offensive_portable_z", "defensive_portable_z",
              "dim_defensive_impact_z", "dim_defensive_playmaking_z"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    pts_o = df["offensive_portable_z"].copy()
    pts_d = df["defensive_portable_z"].copy()

    # Adjustment A — Matchup Dim 6 residual
    if cfg.use_matchup_dim6_residual and cfg.matchup_dim6_strength > 0:
        m = build_matchup_dim6_z(def_arch)
        # De-duplicate matchup table to avoid one-to-many merge expansion
        m = m.drop_duplicates(subset=["player_id", "season"], keep="last")
        df = df.merge(m, on=["player_id", "season"], how="left")
        # Refresh local series since merge may have re-indexed
        pts_o = df["offensive_portable_z"].copy()
        pts_d_post = df["defensive_portable_z"].copy()
        # Where matchup is missing, residual = 0 (no change)
        matchup_z = df["matchup_dim6_z"].fillna(df["dim_defensive_impact_z"])
        residual = matchup_z - df["dim_defensive_impact_z"]
        d6_share = 0.31
        pts_d = pts_d_post + cfg.matchup_dim6_strength * d6_share * residual

    # Adjustment B — Dim 5 reduction
    if cfg.dim5_weight_reduction > 0:
        # Dim 5 contributes ~0.08/0.32 ≈ 0.25 of defensive_portable_z
        d5_share = 0.25
        reduction_amount = cfg.dim5_weight_reduction / 0.08  # fraction of dim5 weight removed
        pts_d = pts_d - reduction_amount * d5_share * df["dim_defensive_playmaking_z"]

    # Adjustment C — Gain scaling
    pts_o = cfg.offense_gain * pts_o
    pts_d = cfg.defense_gain * pts_d

    # Adjustment D — Defensive archetype shrinkage
    if cfg.defensive_archetype_shrinkage > 0 and "primary_archetype" in df.columns:
        s = cfg.defensive_archetype_shrinkage
        arch_mean_d = df.groupby(["season", "primary_archetype"])
        # Bind back via transform
        arch_mean_d = pts_d.groupby([df["season"], df["primary_archetype"]]).transform("mean")
        pts_d = (1 - s) * pts_d + s * arch_mean_d.fillna(pts_d.mean())

    # Adjustment E — Multi-season smoothing
    if cfg.multi_season_smoothing > 0:
        s = cfg.multi_season_smoothing
        df["pts_o"] = pts_o.values
        df["pts_d"] = pts_d.values
        df = df.sort_values(["player_id", "season"])
        df["pts_o_prev"] = df.groupby("player_id")["pts_o"].shift(1)
        df["pts_d_prev"] = df.groupby("player_id")["pts_d"].shift(1)
        df["pts_o"] = np.where(df["pts_o_prev"].notna(),
                               (1 - s) * df["pts_o"] + s * df["pts_o_prev"],
                               df["pts_o"])
        df["pts_d"] = np.where(df["pts_d_prev"].notna(),
                               (1 - s) * df["pts_d"] + s * df["pts_d_prev"],
                               df["pts_d"])
        pts_o = df["pts_o"]
        pts_d = df["pts_d"]

    # Final clip
    pts_o = pts_o.clip(-cfg.final_clip, cfg.final_clip)
    pts_d = pts_d.clip(-cfg.final_clip, cfg.final_clip)

    out = df[["player_id", "season"]].copy()
    out["pts_o_v32"] = pts_o.values
    out["pts_d_v32"] = pts_d.values
    return out


# ---------------------------------------------------------------------------
# Brier + Lineup harness
# ---------------------------------------------------------------------------

def patch_and_brier(
    pts: pd.DataFrame, team_scale: float = DEFAULT_TEAM_SCALE,
    star_amp_top1: float = 1.0, star_amp_top2: float = 1.0,
) -> Dict:
    projected = pd.read_parquet(PROJ_PROFILES)
    team_feat = pd.read_parquet(PROJ_TEAM_FEAT)
    seasons_sorted = sorted(pts["season"].unique())
    season_to_next = {s: seasons_sorted[i+1] for i, s in enumerate(seasons_sorted[:-1])}
    pts_m = pts.copy()
    pts_m["target_season"] = pts_m["season"].map(season_to_next)
    pts_m = pts_m.dropna(subset=["target_season"])
    pts_m = pts_m.drop(columns=["season"]).rename(columns={"target_season": "season"})
    pts_m["player_id"] = pts_m["player_id"].astype(str)
    pts_m["season"] = pts_m["season"].astype(str)
    pp = projected.copy()
    pp["player_id"] = pp["player_id"].astype(str)
    pp["season"] = pp["season"].astype(str)
    merged = pp.merge(pts_m[["player_id", "season", "pts_o_v32", "pts_d_v32"]],
                      on=["player_id", "season"], how="left")
    for c in ["pts_o_v32", "pts_d_v32"]:
        merged[c] = merged[c].fillna(0.0)

    # Scale to original distribution
    o_std = pp["impact_obke"].std(); o_mean = pp["impact_obke"].mean()
    d_std = pp["impact_dbke"].std(); d_mean = pp["impact_dbke"].mean()
    n_o_std = merged["pts_o_v32"].std(); n_d_std = merged["pts_d_v32"].std()
    if n_o_std > 1e-9:
        merged["new_obke"] = merged["pts_o_v32"] / n_o_std * o_std + o_mean
    else:
        merged["new_obke"] = o_mean
    if n_d_std > 1e-9:
        merged["new_dbke"] = merged["pts_d_v32"] / n_d_std * d_std + d_mean
    else:
        merged["new_dbke"] = d_mean

    # Star amplification
    if star_amp_top1 != 1.0 or star_amp_top2 != 1.0:
        merged["rk"] = merged.groupby(["season", "team_abbreviation"])["new_obke"].rank(
            ascending=False, method="first"
        )
        amp = np.where(merged["rk"] == 1, star_amp_top1,
                np.where(merged["rk"] == 2, star_amp_top2, 1.0))
        merged["new_obke"] *= amp
        merged["new_dbke"] *= amp
        merged = merged.drop(columns=["rk"])

    merged["minutes"] = pd.to_numeric(merged.get("minutes"), errors="coerce").fillna(0.0)
    team_min = merged.groupby(["season", "team_abbreviation"])["minutes"].transform("sum")
    merged["ms"] = merged["minutes"] / team_min.replace(0, np.nan).fillna(1.0)
    merged["w_off"] = merged["ms"] * merged["new_obke"]
    merged["w_def"] = merged["ms"] * merged["new_dbke"]
    talent = merged.groupby(["season", "team_abbreviation"]).agg(
        new_off=("w_off", "sum"), new_def=("w_def", "sum")
    ).reset_index()

    tf = team_feat.copy()
    tf["team_abbreviation"] = tf["team_abbreviation"].astype(str).str.upper()
    talent["team_abbreviation"] = talent["team_abbreviation"].astype(str).str.upper()
    merged2 = tf.merge(talent, on=["season", "team_abbreviation"], how="left")
    delta_off = (merged2["new_off"].fillna(merged2["off_talent_base"])
                 - merged2["off_talent_base"])
    delta_def = (merged2["new_def"].fillna(merged2["def_talent_base"])
                 - merged2["def_talent_base"])
    merged2["team_net_rating_projected"] = (
        merged2["team_net_rating_projected"]
        + team_scale * delta_off + team_scale * delta_def
    )
    merged2["off_talent_base"] = merged2["new_off"].fillna(merged2["off_talent_base"])
    merged2["def_talent_base"] = merged2["new_def"].fillna(merged2["def_talent_base"])
    patched = merged2.drop(columns=["new_off", "new_def"])

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        tmp = f.name
    out_json = REPO / "reports/forecast_game_validation.json"
    try:
        patched.to_parquet(tmp, index=False)
        subprocess.run([sys.executable, str(VALIDATE_FORECAST), "--features-path", tmp],
                       capture_output=True, text=True, cwd=str(REPO))
        if not out_json.exists():
            return {"error": "no JSON"}
        d = json.loads(out_json.read_text())
        agg = d.get("aggregate", {})
        return {
            "brier": agg.get("brier_score"),
            "log_loss": agg.get("log_loss"),
            "accuracy": agg.get("accuracy"),
            "per_transition": {s: m.get("brier_score") for s, m in d.get("game_level", {}).items()},
        }
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def lineup_r(pts: pd.DataFrame, team_scale: float, min_poss: int,
             seasons: Optional[list]) -> Dict:
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        pts_path = f.name
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out_json = f.name
    try:
        pts.to_parquet(pts_path, index=False)
        cmd = [sys.executable, str(VALIDATE_LINEUP),
               "--pts-file", pts_path,
               "--pts-col", "pts_o_v32", "--pts-col-d", "pts_d_v32",
               "--team-scale", str(team_scale), "--min-poss", str(min_poss),
               "--output", out_json, "--quiet"]
        if seasons:
            cmd.append("--seasons"); cmd.extend(seasons)
        subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
        if not os.path.exists(out_json):
            return {}
        d = json.load(open(out_json))
        return d.get("pooled", {})
    finally:
        for p in (pts_path, out_json):
            if os.path.exists(p):
                os.unlink(p)


def run(cfg: PostHocConfig, team_scale: float,
        lineup_seasons: Optional[list], min_poss: int) -> Dict:
    decomp = pd.read_parquet(DECOMP_PATH)
    def_arch = pd.read_parquet(DEF_ARCH_PATH) if DEF_ARCH_PATH.exists() else None
    pts = apply_posthoc(decomp, def_arch, cfg)
    b = patch_and_brier(pts, team_scale=team_scale,
                        star_amp_top1=cfg.star_amp_top1,
                        star_amp_top2=cfg.star_amp_top2)
    l = lineup_r(pts, team_scale=team_scale, min_poss=min_poss,
                 seasons=lineup_seasons)
    return {
        "config": asdict(cfg),
        "team_scale": team_scale,
        "brier": b,
        "lineup": l,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--team-scale", type=float, default=DEFAULT_TEAM_SCALE)
    parser.add_argument("--lineup-seasons", nargs="*",
                        default=["2022-23", "2023-24", "2024-25"])
    parser.add_argument("--min-poss", type=int, default=50)
    parser.add_argument("--label", default="v32_posthoc")
    parser.add_argument("--config", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = PostHocConfig()
    if not args.baseline and args.config:
        with open(args.config) as f:
            for k, v in json.load(f).items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)

    print(f"\n=== {args.label} ===")
    print(f"Config: {asdict(cfg)}")
    r = run(cfg, args.team_scale, args.lineup_seasons, args.min_poss)
    print(f"Brier:   {r['brier']}")
    print(f"Lineup:  {r['lineup']}")
    r["label"] = args.label
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        json.dump(r, open(args.output, "w"), indent=2, default=str)
        print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
