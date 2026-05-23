"""
scripts/pts_v32_harness.py
=============================================================================
v3.2 PTS Dual-Metric Harness

Given a PtsV32Config, runs:
  Method 1 — Season-level Brier (via validate_forecast.py) using the
             experiment_pts_rdis_blend.py patch pattern
  Method 2 — Lineup-level Pearson r (validate_lineup_pts.py)

Returns dict of metrics. Used by pts_v32_sweep.py to score candidates.

Usage:
  python3 scripts/pts_v32_harness.py                       # default config
  python3 scripts/pts_v32_harness.py --config tmp/cfg.json
  python3 scripts/pts_v32_harness.py --label v32_seed --min-poss 50
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.pts_v32_recompute import (
    PtsV32Config, recompute_pts_v32, DECOMP_PATH, DEF_ARCH_PATH,
)

PROJ_PROFILES = REPO / "data/processed/forecast/projected_player_profiles.parquet"
PROJ_TEAM_FEAT = REPO / "data/processed/forecast/projected_team_features.parquet"
VALIDATE_FORECAST = REPO / "src/simulation/validate_forecast.py"
VALIDATE_LINEUP = REPO / "scripts/validate_lineup_pts.py"

DEFAULT_TEAM_SCALE = 20.0


# ---------------------------------------------------------------------------
# Brier (Method 1) — borrowed pattern from experiment_pts_rdis_blend.py
# ---------------------------------------------------------------------------

def _scale_to_original(values: pd.Series, target_mean: float, target_std: float) -> pd.Series:
    std = values.std()
    if std < 1e-9:
        return pd.Series(target_mean, index=values.index)
    return (values / std) * target_std + target_mean


def build_team_talent_v32(
    projected: pd.DataFrame,
    pts: pd.DataFrame,
    team_scale: float = DEFAULT_TEAM_SCALE,
    star_amp_top1: float = 1.0,
    star_amp_top2: float = 1.0,
) -> pd.DataFrame:
    """Replace impact_obke/impact_dbke in projected_player_profiles with v3.2 PTS,
    then compute team off/def talent bases.

    Decomp[season=N] feeds projected[season=N+1] (next-season target).
    """
    seasons_sorted = sorted(pts["season"].unique())
    season_to_next = {s: seasons_sorted[i + 1] for i, s in enumerate(seasons_sorted[:-1])}

    pts_for_merge = pts.copy()
    pts_for_merge["target_season"] = pts_for_merge["season"].map(season_to_next)
    pts_for_merge = pts_for_merge.dropna(subset=["target_season"])
    pts_for_merge = pts_for_merge.drop(columns=["season"]).rename(
        columns={"target_season": "season"}
    )

    pp = projected.copy()
    pp["player_id"] = pp["player_id"].astype(str)
    pp["season"] = pp["season"].astype(str)
    pts_for_merge["player_id"] = pts_for_merge["player_id"].astype(str)
    pts_for_merge["season"] = pts_for_merge["season"].astype(str)

    merged = pp.merge(
        pts_for_merge[["player_id", "season", "pts_o_v32", "pts_d_v32"]],
        on=["player_id", "season"], how="left",
    )
    merged["pts_o_v32"] = merged["pts_o_v32"].fillna(0.0)
    merged["pts_d_v32"] = merged["pts_d_v32"].fillna(0.0)

    # Scale PTS to match original impact_obke/dbke distribution so TEAM_SCALE remains compatible
    orig_obke_std = pp["impact_obke"].std()
    orig_dbke_std = pp["impact_dbke"].std()
    orig_obke_mean = pp["impact_obke"].mean()
    orig_dbke_mean = pp["impact_dbke"].mean()

    merged["new_obke"] = _scale_to_original(merged["pts_o_v32"], orig_obke_mean, orig_obke_std)
    merged["new_dbke"] = _scale_to_original(merged["pts_d_v32"], orig_dbke_mean, orig_dbke_std)

    # Compute minute share per team-season
    merged["minutes"] = pd.to_numeric(merged.get("minutes"), errors="coerce").fillna(0.0)
    team_min = merged.groupby(["season", "team_abbreviation"])["minutes"].transform("sum")
    merged["ms"] = merged["minutes"] / team_min.replace(0, np.nan).fillna(1.0)

    # Star amplification (Section 5E)
    if star_amp_top1 != 1.0 or star_amp_top2 != 1.0:
        merged["rk"] = merged.groupby(["season", "team_abbreviation"])["new_obke"].rank(
            ascending=False, method="first"
        )
        merged["amp"] = np.where(merged["rk"] == 1, star_amp_top1,
                          np.where(merged["rk"] == 2, star_amp_top2, 1.0))
        merged["new_obke"] *= merged["amp"]
        merged["new_dbke"] *= merged["amp"]
        merged = merged.drop(columns=["rk", "amp"])

    merged["w_off"] = merged["ms"] * merged["new_obke"]
    merged["w_def"] = merged["ms"] * merged["new_dbke"]

    team_talent = (
        merged.groupby(["season", "team_abbreviation"])
        .agg(new_off=("w_off", "sum"), new_def=("w_def", "sum"))
        .reset_index()
    )
    return team_talent


def patch_team_features(
    team_features: pd.DataFrame,
    new_talent: pd.DataFrame,
    team_scale: float = DEFAULT_TEAM_SCALE,
) -> pd.DataFrame:
    """Replace off_talent_base / def_talent_base, recompute net rating."""
    tf = team_features.copy()
    tf["team_abbreviation"] = tf["team_abbreviation"].astype(str).str.upper()
    new_talent["team_abbreviation"] = new_talent["team_abbreviation"].astype(str).str.upper()

    merged = tf.merge(new_talent, on=["season", "team_abbreviation"], how="left")
    delta_off = (merged["new_off"].fillna(merged["off_talent_base"]) - merged["off_talent_base"])
    delta_def = (merged["new_def"].fillna(merged["def_talent_base"]) - merged["def_talent_base"])
    merged["team_net_rating_projected"] = (
        merged["team_net_rating_projected"]
        + team_scale * delta_off
        + team_scale * delta_def
    )
    merged["off_talent_base"] = merged["new_off"].fillna(merged["off_talent_base"])
    merged["def_talent_base"] = merged["new_def"].fillna(merged["def_talent_base"])
    return merged.drop(columns=["new_off", "new_def"])


def run_brier(patched_features: pd.DataFrame) -> Dict:
    """Save patched features, run validate_forecast, read aggregate from JSON file."""
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        tmp_path = f.name
    out_json = REPO / "reports/forecast_game_validation.json"
    try:
        patched_features.to_parquet(tmp_path, index=False)
        result = subprocess.run(
            [sys.executable, str(VALIDATE_FORECAST), "--features-path", tmp_path],
            capture_output=True, text=True, cwd=str(REPO),
        )
        if not out_json.exists():
            return {"error": "no JSON written", "stderr": result.stderr[-300:]}
        data = json.loads(out_json.read_text())
        agg = data.get("aggregate", {})
        # Per-transition Brier for diagnostics
        per_trans = {
            s: m.get("brier_score") for s, m in data.get("game_level", {}).items()
        }
        return {
            "brier": agg.get("brier_score"),
            "log_loss": agg.get("log_loss"),
            "accuracy": agg.get("accuracy"),
            "n_games": agg.get("n_games_total"),
            "per_transition_brier": per_trans,
        }
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Lineup r (Method 2)
# ---------------------------------------------------------------------------

def run_lineup_r(
    pts_path: str,
    team_scale: float = DEFAULT_TEAM_SCALE,
    min_poss: int = 50,
    seasons: Optional[list] = None,
) -> Dict:
    """Run validate_lineup_pts using v3.2 PTS columns."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out_json = f.name
    try:
        cmd = [
            sys.executable, str(VALIDATE_LINEUP),
            "--pts-file", pts_path,
            "--pts-col", "pts_o_v32",
            "--pts-col-d", "pts_d_v32",
            "--team-scale", str(team_scale),
            "--min-poss", str(min_poss),
            "--output", out_json,
            "--quiet",
        ]
        if seasons:
            cmd.append("--seasons")
            cmd.extend(seasons)
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
        if not os.path.exists(out_json):
            return {"error": result.stderr[-500:] if result.stderr else "no output"}
        with open(out_json) as f:
            data = json.load(f)
        return data.get("pooled", {})
    finally:
        if os.path.exists(out_json):
            os.unlink(out_json)


# ---------------------------------------------------------------------------
# Top-level harness
# ---------------------------------------------------------------------------

def run_harness(
    cfg: PtsV32Config,
    team_scale: float = DEFAULT_TEAM_SCALE,
    star_amp_top1: float = 1.0,
    star_amp_top2: float = 1.0,
    min_poss: int = 50,
    lineup_seasons: Optional[list] = None,
    skip_lineup: bool = False,
) -> Dict:
    """Single-call harness: recompute → patch → Brier + lineup r."""
    decomp = pd.read_parquet(DECOMP_PATH)
    def_arch = pd.read_parquet(DEF_ARCH_PATH) if DEF_ARCH_PATH.exists() else None
    pts = recompute_pts_v32(decomp, def_arch, cfg)

    # Save PTS to temp for lineup harness
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        pts_path = f.name
    try:
        pts.to_parquet(pts_path, index=False)

        # Brier
        projected = pd.read_parquet(PROJ_PROFILES)
        team_feat = pd.read_parquet(PROJ_TEAM_FEAT)
        talent = build_team_talent_v32(
            projected, pts, team_scale=team_scale,
            star_amp_top1=star_amp_top1, star_amp_top2=star_amp_top2,
        )
        patched = patch_team_features(team_feat, talent, team_scale=team_scale)
        brier_metrics = run_brier(patched)

        # Lineup r
        lineup_metrics = {}
        if not skip_lineup:
            lineup_metrics = run_lineup_r(
                pts_path, team_scale=team_scale, min_poss=min_poss,
                seasons=lineup_seasons,
            )

        return {
            "config": cfg.to_dict(),
            "team_scale": team_scale,
            "star_amp_top1": star_amp_top1,
            "star_amp_top2": star_amp_top2,
            "brier": brier_metrics,
            "lineup": lineup_metrics,
            "pts_o_std": float(pts["pts_o_v32"].std()),
            "pts_d_std": float(pts["pts_d_v32"].std()),
        }
    finally:
        os.unlink(pts_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--team-scale", type=float, default=DEFAULT_TEAM_SCALE)
    parser.add_argument("--star-amp-top1", type=float, default=1.0)
    parser.add_argument("--star-amp-top2", type=float, default=1.0)
    parser.add_argument("--min-poss", type=int, default=50)
    parser.add_argument("--lineup-seasons", nargs="*", default=None,
                        help="Subset of seasons for lineup r (default: all)")
    parser.add_argument("--skip-lineup", action="store_true")
    parser.add_argument("--label", default="v32_run")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = PtsV32Config()
    if args.config:
        with open(args.config) as f:
            overrides = json.load(f)
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    print(f"\n=== {args.label} ===")
    result = run_harness(
        cfg, team_scale=args.team_scale,
        star_amp_top1=args.star_amp_top1, star_amp_top2=args.star_amp_top2,
        min_poss=args.min_poss, lineup_seasons=args.lineup_seasons,
        skip_lineup=args.skip_lineup,
    )
    print(f"Brier:        {result['brier']}")
    print(f"Lineup pool:  {result['lineup']}")
    print(f"PTS std:      o={result['pts_o_std']:.3f}  d={result['pts_d_std']:.3f}")
    result["label"] = args.label

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"Saved → {args.output}")
    return result


if __name__ == "__main__":
    main()
