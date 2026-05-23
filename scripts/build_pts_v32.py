"""
scripts/build_pts_v32.py
=============================================================================
BKE v3.2 PTS Production Build

Single command that:
  1. Reads PTS_V32 config from model_config.PtsV32Config
  2. Loads bke_v28_decomposition.parquet + defensive_archetypes_v2.parquet
  3. Applies v3.2 post-hoc adjustments (Fix 3 weight reduce, def shrinkage,
     multi-season smoothing, optional matchup Dim 6)
  4. Writes data/processed/bke/pts_v32.parquet — per-player PTS_O / PTS_D
  5. Writes data/processed/forecast/projected_team_features_v32.parquet —
     patched team net rating using v3.2 PTS in place of impact_obke/dbke

Downstream consumers (validate_forecast, viewers) can either:
  - Use the v3.2 features file directly:
        python3 src/simulation/validate_forecast.py \\
            --features-path data/processed/forecast/projected_team_features_v32.parquet
  - Continue using the v2.7 baseline (untouched)

The team_feature_aggregation.py pipeline is NOT modified — v3.2 is a
post-processing step that produces a parallel output file.

Usage:
    python3 scripts/build_pts_v32.py
    python3 scripts/build_pts_v32.py --no-matchup-dim6
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.modeling.model_config import (
    PTS_V32, PTS_V32_PARQUET, PROJ_TEAM_FEATURES_V32_PARQUET,
    BKE_OUTPUT_PARQUET, BKE_V28_OUTPUT_PARQUET, DEFENSIVE_ARCHETYPES_PATH,
)
from scripts.pts_v32_posthoc_harness import (
    PostHocConfig, apply_posthoc, patch_and_brier, lineup_r,
)

DECOMP_PATH = REPO / "data/processed/bke/bke_v28_decomposition.parquet"
DEF_ARCH_PATH = REPO / "data/processed/defensive_archetypes_v2.parquet"
PROJ_PROFILES = REPO / "data/processed/forecast/projected_player_profiles.parquet"
PROJ_TEAM_FEAT = REPO / "data/processed/forecast/projected_team_features.parquet"


def cfg_from_pts_v32(pts_v32_cfg, **overrides) -> PostHocConfig:
    """Translate PTS_V32 (the production config) → PostHocConfig (the harness)."""
    base = dict(
        use_matchup_dim6_residual=pts_v32_cfg.use_matchup_dim6,
        matchup_dim6_strength=pts_v32_cfg.matchup_dim6_strength,
        dim5_weight_reduction=pts_v32_cfg.dim5_weight_reduction,
        defensive_archetype_shrinkage=pts_v32_cfg.defensive_archetype_shrinkage,
        multi_season_smoothing=pts_v32_cfg.multi_season_smoothing,
        star_amp_top1=pts_v32_cfg.star_amp_top1,
        star_amp_top2=pts_v32_cfg.star_amp_top2,
        final_clip=pts_v32_cfg.final_pts_clip,
    )
    base.update(overrides)
    return PostHocConfig(**base)


def build_patched_features(pts: pd.DataFrame) -> pd.DataFrame:
    """Replicate the patch step from patch_and_brier without invoking validate_forecast."""
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
    for c in ("pts_o_v32", "pts_d_v32"):
        merged[c] = merged[c].fillna(0.0)

    o_std = pp["impact_obke"].std(); o_mean = pp["impact_obke"].mean()
    d_std = pp["impact_dbke"].std(); d_mean = pp["impact_dbke"].mean()
    n_o = max(merged["pts_o_v32"].std(), 1e-9)
    n_d = max(merged["pts_d_v32"].std(), 1e-9)
    merged["new_obke"] = merged["pts_o_v32"] / n_o * o_std + o_mean
    merged["new_dbke"] = merged["pts_d_v32"] / n_d * d_std + d_mean

    merged["minutes"] = pd.to_numeric(merged.get("minutes"), errors="coerce").fillna(0.0)
    team_min = merged.groupby(["season", "team_abbreviation"])["minutes"].transform("sum")
    merged["ms"] = merged["minutes"] / team_min.replace(0, np.nan).fillna(1.0)
    merged["w_off"] = merged["ms"] * merged["new_obke"]
    merged["w_def"] = merged["ms"] * merged["new_dbke"]
    talent = merged.groupby(["season", "team_abbreviation"]).agg(
        new_off=("w_off", "sum"), new_def=("w_def", "sum"),
    ).reset_index()

    tf = team_feat.copy()
    tf["team_abbreviation"] = tf["team_abbreviation"].astype(str).str.upper()
    talent["team_abbreviation"] = talent["team_abbreviation"].astype(str).str.upper()
    m2 = tf.merge(talent, on=["season", "team_abbreviation"], how="left")
    d_off = m2["new_off"].fillna(m2["off_talent_base"]) - m2["off_talent_base"]
    d_def = m2["new_def"].fillna(m2["def_talent_base"]) - m2["def_talent_base"]
    m2["team_net_rating_projected"] = (
        m2["team_net_rating_projected"] + 20.0 * d_off + 20.0 * d_def
    )
    m2["off_talent_base"] = m2["new_off"].fillna(m2["off_talent_base"])
    m2["def_talent_base"] = m2["new_def"].fillna(m2["def_talent_base"])
    return m2.drop(columns=["new_off", "new_def"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-matchup-dim6", action="store_true",
                        help="Force off the matchup-based Dim 6 swap")
    parser.add_argument("--matchup-strength", type=float, default=None)
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args()

    cfg = PTS_V32
    if args.print_config:
        print(json.dumps(asdict(cfg), indent=2, default=str))

    overrides = {}
    if args.no_matchup_dim6:
        overrides["use_matchup_dim6_residual"] = False
        overrides["matchup_dim6_strength"] = 0.0
    if args.matchup_strength is not None:
        overrides["use_matchup_dim6_residual"] = args.matchup_strength > 0
        overrides["matchup_dim6_strength"] = args.matchup_strength

    ph_cfg = cfg_from_pts_v32(cfg, **overrides)
    print(f"\nBuilding PTS v3.2 with config:")
    for k, v in asdict(ph_cfg).items():
        print(f"  {k:<40}  {v}")
    print()

    # Build PTS table
    decomp = pd.read_parquet(DECOMP_PATH)
    def_arch = pd.read_parquet(DEF_ARCH_PATH) if DEF_ARCH_PATH.exists() else None
    pts = apply_posthoc(decomp, def_arch, ph_cfg)
    out_pts = Path(PTS_V32_PARQUET)
    os.makedirs(out_pts.parent, exist_ok=True)
    pts.to_parquet(out_pts, index=False)
    print(f"PTS v3.2:  {out_pts} ({len(pts)} rows)")
    print(f"  pts_o_v32 mean={pts['pts_o_v32'].mean():+.3f}  std={pts['pts_o_v32'].std():.3f}")
    print(f"  pts_d_v32 mean={pts['pts_d_v32'].mean():+.3f}  std={pts['pts_d_v32'].std():.3f}")

    # Build patched team features
    patched = build_patched_features(pts)
    out_feat = Path(PROJ_TEAM_FEATURES_V32_PARQUET)
    os.makedirs(out_feat.parent, exist_ok=True)
    patched.to_parquet(out_feat, index=False)
    print(f"Patched features: {out_feat}  ({len(patched)} rows)")
    print(f"  team_net_rating_projected mean={patched['team_net_rating_projected'].mean():+.3f}")
    print(f"  team_net_rating_projected std={patched['team_net_rating_projected'].std():.3f}")


if __name__ == "__main__":
    main()
