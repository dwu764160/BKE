"""
scripts/diagnose_brier_baseline.py
=============================================================================
Phase A.4 — Diagnose the 0.2248 ↔ 0.2345 Brier baseline gap

The experiment script (scripts/experiment_pts_rdis_blend.py) at α=1.00
reports Brier = 0.2248. Our post-hoc harness at "baseline" config (no
adjustments, pts_o = offensive_portable_z) reports Brier = 0.2345. We
previously verified the *patched* team_features are bit-identical between
the two paths. So the gap must come from:

  (a) The experiment script's *baseline* run (no patching) gets a different
      Brier than `validate_forecast.py` direct (0.2400 vs 0.2420).
  (b) A subtle difference in the patched-features round-trip (parquet
      dtype, NaN handling) that affects validate_forecast downstream.
  (c) Difference in player-merge coverage between the two paths.

This script runs both end-to-end with verbose dumps and prints the
diff so we can find it.

Output: reports/brier_baseline_diagnostic.json
=============================================================================
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

OUT = REPO / "reports/brier_baseline_diagnostic.json"


def run_validate_forecast(features_path: str) -> dict:
    out_json = REPO / "reports/forecast_game_validation.json"
    subprocess.run(
        [sys.executable, "src/simulation/validate_forecast.py",
         "--features-path", features_path],
        cwd=str(REPO), capture_output=True, text=True,
    )
    if not out_json.exists():
        return {}
    d = json.loads(out_json.read_text())
    return d.get("aggregate", {})


def main():
    PROJ_FEAT = REPO / "data/processed/forecast/projected_team_features.parquet"
    tf = pd.read_parquet(PROJ_FEAT)

    # Test 1: direct validate_forecast on original parquet (no round-trip)
    print("=== Test 1: validate_forecast direct on original parquet ===")
    a1 = run_validate_forecast(str(PROJ_FEAT))
    print(f"  Brier: {a1.get('brier_score')}  log_loss: {a1.get('log_loss')}")

    # Test 2: write parquet to tempfile via to_parquet, read back, validate
    print("\n=== Test 2: parquet round-trip (write → read → validate) ===")
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        tmp_path = f.name
    try:
        tf.to_parquet(tmp_path, index=False)
        a2 = run_validate_forecast(tmp_path)
        print(f"  Brier: {a2.get('brier_score')}  log_loss: {a2.get('log_loss')}")
        # Compare dtypes
        tf_back = pd.read_parquet(tmp_path)
        dtype_diff = {
            c: (str(tf[c].dtype), str(tf_back[c].dtype))
            for c in tf.columns
            if str(tf[c].dtype) != str(tf_back[c].dtype)
        }
        if dtype_diff:
            print(f"  DTYPE DIFFERENCES on round-trip: {dtype_diff}")
        else:
            print("  No dtype differences on round-trip.")

        # Value diff (numeric only)
        numeric_cols = tf.select_dtypes(include=[np.number]).columns
        diffs = {}
        for c in numeric_cols:
            a, b = tf[c], tf_back[c]
            d = (a - b).abs().max()
            if pd.notna(d) and d > 1e-10:
                diffs[c] = float(d)
        if diffs:
            print(f"  NUMERIC DIFFS on round-trip: {diffs}")
        else:
            print("  No numeric value drift on round-trip.")
    finally:
        os.unlink(tmp_path)

    # Test 3: experiment script α=1.0 path
    print("\n=== Test 3: experiment_pts_rdis_blend α=1.0 ===")
    from scripts.experiment_pts_rdis_blend import (
        build_components, compute_new_talent_bases, patch_team_features as exp_patch,
    )
    decomp = pd.read_parquet(REPO / "data/processed/bke/bke_v28_decomposition.parquet")
    proj = pd.read_parquet(REPO / "data/processed/forecast/projected_player_profiles.parquet")
    comps = build_components(decomp)
    exp_talent = compute_new_talent_bases(proj, comps, 1.00, None)
    exp_patched = exp_patch(tf, exp_talent)
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        tmp_exp = f.name
    try:
        exp_patched.to_parquet(tmp_exp, index=False)
        a3 = run_validate_forecast(tmp_exp)
        print(f"  Brier: {a3.get('brier_score')}  log_loss: {a3.get('log_loss')}")
    finally:
        os.unlink(tmp_exp)

    # Test 4: posthoc harness baseline path
    print("\n=== Test 4: posthoc harness baseline (no adjustments) ===")
    from scripts.pts_v32_posthoc_harness import (
        PostHocConfig, apply_posthoc, patch_and_brier,
    )
    def_arch_path = REPO / "data/processed/defensive_archetypes_v2.parquet"
    def_arch = pd.read_parquet(def_arch_path) if def_arch_path.exists() else None
    pts_post = apply_posthoc(decomp, def_arch, PostHocConfig())
    post_brier = patch_and_brier(pts_post, team_scale=20.0)
    print(f"  Brier (posthoc internal): {post_brier.get('brier')}")

    # Direct compare: load the EXACT patched features produced by each path
    # and run validate_forecast separately
    print("\n=== Test 5: byte-by-byte diff of patched team_features (exp vs posthoc) ===")
    # Recompute posthoc patched
    from scripts.build_pts_v32 import build_patched_features
    # But build_pts_v32 uses PTS_V32 (not baseline). We need the baseline equivalent.
    # Inline-patch using zero-adjustment posthoc:
    # Re-use exp_patched as the exp path; for posthoc baseline, replicate logic:
    seasons_sorted = sorted(pts_post["season"].unique())
    season_to_next = {s: seasons_sorted[i + 1] for i, s in enumerate(seasons_sorted[:-1])}
    pts_m = pts_post.copy()
    pts_m["target_season"] = pts_m["season"].map(season_to_next)
    pts_m = pts_m.dropna(subset=["target_season"]).drop(columns=["season"]).rename(
        columns={"target_season": "season"}
    )
    pts_m["player_id"] = pts_m["player_id"].astype(str)
    pts_m["season"] = pts_m["season"].astype(str)
    pp = proj.copy()
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
    tmin = merged.groupby(["season", "team_abbreviation"])["minutes"].transform("sum")
    merged["ms"] = merged["minutes"] / tmin.replace(0, np.nan).fillna(1.0)
    merged["w_off"] = merged["ms"] * merged["new_obke"]
    merged["w_def"] = merged["ms"] * merged["new_dbke"]
    post_talent = merged.groupby(["season", "team_abbreviation"]).agg(
        new_off=("w_off", "sum"), new_def=("w_def", "sum"),
    ).reset_index()
    tf2 = tf.copy()
    tf2["team_abbreviation"] = tf2["team_abbreviation"].astype(str).str.upper()
    post_talent["team_abbreviation"] = post_talent["team_abbreviation"].astype(str).str.upper()
    pm = tf2.merge(post_talent, on=["season", "team_abbreviation"], how="left")
    d_off = pm["new_off"].fillna(pm["off_talent_base"]) - pm["off_talent_base"]
    d_def = pm["new_def"].fillna(pm["def_talent_base"]) - pm["def_talent_base"]
    pm["team_net_rating_projected"] = pm["team_net_rating_projected"] + 20.0 * d_off + 20.0 * d_def
    pm["off_talent_base"] = pm["new_off"].fillna(pm["off_talent_base"])
    pm["def_talent_base"] = pm["new_def"].fillna(pm["def_talent_base"])
    posthoc_patched = pm.drop(columns=["new_off", "new_def"])

    key = ["season", "team_abbreviation"]
    exp_p = exp_patched.set_index(key)[["team_net_rating_projected", "off_talent_base", "def_talent_base"]]
    post_p = posthoc_patched.set_index(key)[["team_net_rating_projected", "off_talent_base", "def_talent_base"]]
    cmp = exp_p.join(post_p, lsuffix="_exp", rsuffix="_post").dropna()
    cmp["delta_net"] = cmp["team_net_rating_projected_exp"] - cmp["team_net_rating_projected_post"]
    print(f"  Rows compared: {len(cmp)}")
    print(f"  delta_net stats: mean={cmp['delta_net'].mean():.2e}  std={cmp['delta_net'].std():.2e}  "
          f"max_abs={cmp['delta_net'].abs().max():.2e}")
    if cmp['delta_net'].abs().max() < 1e-6:
        print("  → Patched team features are bit-identical.")

    # Save full diagnostic
    result = {
        "test1_direct_no_roundtrip": a1,
        "test2_roundtrip": a2,
        "test3_experiment_alpha_1": a3,
        "test4_posthoc_baseline": post_brier,
        "test5_patched_features_max_abs_diff": float(cmp['delta_net'].abs().max()),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved → {OUT}")

    # Summary verdict
    print("\n=== VERDICT ===")
    bs = [
        ("Direct (no round-trip)", a1.get("brier_score")),
        ("After round-trip write-read", a2.get("brier_score")),
        ("Experiment script α=1.0", a3.get("brier_score")),
        ("Posthoc harness baseline", post_brier.get("brier")),
    ]
    for label, v in bs:
        print(f"  {label:<35}  Brier={v}")
    if cmp['delta_net'].abs().max() < 1e-6 and a3.get("brier_score") != post_brier.get("brier"):
        print("  → Patched features identical but Brier differs. Bug is in validate_forecast input handling.")
        print("    Most likely a dtype change on parquet round-trip or NaN promotion.")


if __name__ == "__main__":
    main()
