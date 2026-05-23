"""
scripts/pts_v32_sweep.py
=============================================================================
v3.2 PTS Configuration Sweep

Runs a structured grid of PtsV32Config variants to optimize jointly on:
  Method 1 — season-level Brier (validate_forecast)
  Method 2 — lineup-level Pearson r (validate_lineup_pts)

Pipeline:
  1. Establish baseline: pure PTS (current v2.7 dim weights, no fixes)
  2. Single-fix ablations (Fix 2 alone, Fix 3 alone, weight changes alone)
  3. Combined candidates with promising single-fix gains
  4. Local refinement around best combined config (TEAM_SCALE, star amp)
  5. Save ranked results table

Usage:
  python3 scripts/pts_v32_sweep.py                # full sweep, ~10-15 min
  python3 scripts/pts_v32_sweep.py --lineup-seasons 2023-24 2024-25
  python3 scripts/pts_v32_sweep.py --quick
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.pts_v32_recompute import PtsV32Config
from scripts.pts_v32_harness import run_harness

OUTPUT_JSON = REPO / "reports/pts_v32_sweep.json"


# ---------------------------------------------------------------------------
# Config presets — each ablates a specific change vs the current v2.7 baseline
# ---------------------------------------------------------------------------

def baseline_v27_weights() -> PtsV32Config:
    """Current v2.7 weights (what offensive_portable_z/defensive_portable_z use today)
    but with all v3.2 plumbing OFF — i.e., the cleanest reproduction of current PTS."""
    return PtsV32Config(
        w_shooting_gravity=0.14,
        w_driving_gravity=0.10,
        w_playmaking=0.14,
        w_extra_possession=0.08,
        w_turnover_control=0.10,
        w_defensive_playmaking=0.08,
        w_defensive_impact=0.10,
        w_defensive_versatility=0.12,
        w_self_creation=0.14,
        use_matchup_dim6=False,
        defensive_shrinkage=0.0,
        per_dim_clip=10.0,
        final_pts_clip=10.0,
        enforce_unit_variance=False,
        multi_season_smoothing=0.0,
    )


def fix2_only() -> PtsV32Config:
    """Baseline weights + Fix 2 (matchup-based Dim 6) only."""
    c = baseline_v27_weights()
    c.use_matchup_dim6 = True
    c.matchup_dim6_blend = 0.85
    return c


def fix3_only() -> PtsV32Config:
    """Baseline weights + Fix 3 (Dim 5 weight prune 0.08 → 0.05, redistribute)."""
    c = baseline_v27_weights()
    c.w_defensive_playmaking = 0.05
    # Redistribute 0.03 to def_versatility (better validated portable dim)
    c.w_defensive_versatility = 0.15
    return c


def weight_rebalance_offense_heavy() -> PtsV32Config:
    """Plan-suggested redistribution: more weight to universally portable offense."""
    return PtsV32Config(
        w_shooting_gravity=0.17,        # 0.14 → 0.17
        w_driving_gravity=0.10,
        w_playmaking=0.16,              # 0.14 → 0.16
        w_extra_possession=0.07,        # 0.08 → 0.07
        w_turnover_control=0.10,
        w_defensive_playmaking=0.05,    # 0.08 → 0.05 (Fix 3)
        w_defensive_impact=0.10,
        w_defensive_versatility=0.10,
        w_self_creation=0.15,           # 0.14 → 0.15
        use_matchup_dim6=False,
        defensive_shrinkage=0.0,
        per_dim_clip=3.5,
        final_pts_clip=4.0,
        enforce_unit_variance=False,
    )


def weight_rebalance_v32() -> PtsV32Config:
    """Same offensive boost as `offense_heavy` but with Fix 2 (matchup Dim 6) on,
    moderate blend so legacy Dim 6 absorbs RAPM-residual signal."""
    c = weight_rebalance_offense_heavy()
    c.use_matchup_dim6 = True
    c.matchup_dim6_blend = 0.55
    c.defensive_shrinkage = 0.10
    return c


def conservative_v32() -> PtsV32Config:
    """Minimal-change v3.2: keep all current dim weights, ONLY toggle Fix 2 (50/50 blend)."""
    c = baseline_v27_weights()
    c.use_matchup_dim6 = True
    c.matchup_dim6_blend = 0.50
    return c


def multi_season_smoothed() -> PtsV32Config:
    """Baseline + 70/30 multi-season smoothing (5E)."""
    c = baseline_v27_weights()
    c.multi_season_smoothing = 0.30
    return c


def light_clip() -> PtsV32Config:
    c = baseline_v27_weights()
    c.per_dim_clip = 3.0
    c.final_pts_clip = 3.5
    return c


def aggressive_offense() -> PtsV32Config:
    """Push offense weights to the top of plan ranges; drop defense to 0.32."""
    return PtsV32Config(
        w_shooting_gravity=0.18,
        w_driving_gravity=0.13,
        w_playmaking=0.16,
        w_extra_possession=0.07,
        w_turnover_control=0.10,
        w_defensive_playmaking=0.04,
        w_defensive_impact=0.08,
        w_defensive_versatility=0.10,
        w_self_creation=0.14,
        use_matchup_dim6=False,
        defensive_shrinkage=0.0,
        per_dim_clip=10.0,
        final_pts_clip=10.0,
        enforce_unit_variance=False,
    )


PRESETS = {
    "00_baseline_v27_weights": baseline_v27_weights,
    "01_conservative_fix2_only_50_50_blend": conservative_v32,
    "02_fix2_only_85_15_blend": fix2_only,
    "03_fix3_only_dim5_prune": fix3_only,
    "04_weight_rebalance_offense_heavy": weight_rebalance_offense_heavy,
    "05_weight_rebalance_v32_combined": weight_rebalance_v32,
    "06_multi_season_smoothed_30pct": multi_season_smoothed,
    "07_light_clip_30_35": light_clip,
    "08_aggressive_offense": aggressive_offense,
}


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------

def evaluate_preset(
    name: str, cfg: PtsV32Config, team_scale: float,
    lineup_seasons: List[str], min_poss: int,
) -> Dict:
    print(f"\n[{name}]  TEAM_SCALE={team_scale}", flush=True)
    t0 = time.time()
    try:
        result = run_harness(
            cfg, team_scale=team_scale,
            min_poss=min_poss, lineup_seasons=lineup_seasons,
        )
        dt = time.time() - t0
        brier = result.get("brier", {}).get("brier")
        line_r = result.get("lineup", {}).get("mean_pearson_r")
        line_wr = result.get("lineup", {}).get("mean_weighted_pearson_r")
        print(f"  Brier={brier}  lineup r={line_r}  weighted={line_wr}  ({dt:.1f}s)")
        result["preset"] = name
        result["wall_seconds"] = round(dt, 2)
        return result
    except Exception as e:
        print(f"  ERROR: {e}")
        return {"preset": name, "error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lineup-seasons", nargs="*",
                        default=["2022-23", "2023-24", "2024-25"],
                        help="Subset for lineup r (default: last 3 non-COVID seasons)")
    parser.add_argument("--min-poss", type=int, default=50)
    parser.add_argument("--team-scale", type=float, default=20.0)
    parser.add_argument("--quick", action="store_true",
                        help="Run only first 3 presets")
    parser.add_argument("--output", default=str(OUTPUT_JSON))
    parser.add_argument("--presets", nargs="*", default=None)
    args = parser.parse_args()

    if args.presets:
        presets_to_run = {k: v for k, v in PRESETS.items() if k in args.presets}
    elif args.quick:
        presets_to_run = dict(list(PRESETS.items())[:3])
    else:
        presets_to_run = PRESETS

    print(f"v3.2 PTS Sweep — {len(presets_to_run)} preset(s)")
    print(f"Lineup seasons: {args.lineup_seasons}  min_poss={args.min_poss}")
    print(f"TEAM_SCALE: {args.team_scale}")
    print("=" * 70)

    results = []
    for name, cfg_factory in presets_to_run.items():
        cfg = cfg_factory()
        r = evaluate_preset(name, cfg, args.team_scale,
                            args.lineup_seasons, args.min_poss)
        results.append(r)

    # Build summary table
    print("\n" + "=" * 70)
    print(f"{'preset':<45}  {'Brier':>7}  {'r_pool':>7}  {'wr_pool':>7}")
    print("-" * 70)
    for r in results:
        if "error" in r:
            print(f"  {r['preset']:<43}  ERROR: {r['error'][:30]}")
            continue
        b = r.get("brier", {}).get("brier")
        lr = r.get("lineup", {}).get("mean_pearson_r")
        wr = r.get("lineup", {}).get("mean_weighted_pearson_r")
        print(f"  {r['preset']:<43}  {b if b is not None else 'NA':>7}  "
              f"{lr if lr is not None else 'NA':>7}  {wr if wr is not None else 'NA':>7}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"sweep": results}, f, indent=2, default=str)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
