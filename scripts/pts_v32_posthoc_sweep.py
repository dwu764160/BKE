"""
scripts/pts_v32_posthoc_sweep.py
=============================================================================
v3.2 PTS Post-Hoc Adjustment Sweep

Runs a grid of post-hoc adjustments to existing PTS scores and ranks by
joint Brier + lineup r. Uses scripts/pts_v32_posthoc_harness.py.

Sweep dimensions:
  - Matchup-Dim6 residual strength: {0.0, 0.5, 1.0}
  - Dim5 weight reduction: {0.00, 0.03, 0.06}  (remove up to 3-6pp from Dim 5)
  - Offense/defense gain pairs:
      {(1.0, 1.0), (1.10, 1.0), (1.0, 1.10), (1.10, 0.95)}
  - Multi-season smoothing: {0.0, 0.30}
  - Star amplification (top1, top2): {(1.0, 1.0), (1.10, 1.05)}

Two-stage:
  Stage 1: single-knob sweeps (find best per knob)
  Stage 2: combine 2-3 best knobs

Output: reports/pts_v32_posthoc_sweep.json with ranked table.
=============================================================================
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.pts_v32_posthoc_harness import PostHocConfig, run

OUTPUT = REPO / "reports/pts_v32_posthoc_sweep.json"


def evaluate(label: str, cfg: PostHocConfig, team_scale: float,
             lineup_seasons: List[str], min_poss: int) -> Dict:
    t0 = time.time()
    try:
        r = run(cfg, team_scale, lineup_seasons, min_poss)
        b = r.get("brier", {}).get("brier")
        lr = r.get("lineup", {}).get("mean_pearson_r")
        wr = r.get("lineup", {}).get("mean_weighted_pearson_r")
        dt = time.time() - t0
        print(f"  {label:<55}  Brier={b}  r={lr}  wr={wr}  ({dt:.1f}s)", flush=True)
        return {"label": label, "team_scale": team_scale, **r,
                "wall_seconds": round(dt, 1)}
    except Exception as e:
        print(f"  {label} ERROR: {e}")
        return {"label": label, "error": str(e)}


def stage1_single_knobs(team_scale, lineup_seasons, min_poss) -> List[Dict]:
    """One knob at a time."""
    results = []
    # Baseline
    results.append(evaluate("00_baseline", PostHocConfig(),
                            team_scale, lineup_seasons, min_poss))

    # Matchup Dim 6 sweeps
    for s in (0.5, 1.0):
        c = PostHocConfig(use_matchup_dim6_residual=True, matchup_dim6_strength=s)
        results.append(evaluate(f"01_matchup_d6_strength_{s}", c,
                                team_scale, lineup_seasons, min_poss))

    # Dim 5 reduction sweeps
    for r5 in (0.03, 0.06):
        c = PostHocConfig(dim5_weight_reduction=r5)
        results.append(evaluate(f"02_dim5_reduce_{r5}", c,
                                team_scale, lineup_seasons, min_poss))

    # Offense / defense gains
    for og, dg in [(1.10, 1.0), (1.0, 1.10), (1.10, 0.95), (1.05, 1.05)]:
        c = PostHocConfig(offense_gain=og, defense_gain=dg)
        results.append(evaluate(f"03_gain_o{og}_d{dg}", c,
                                team_scale, lineup_seasons, min_poss))

    # Multi-season smoothing
    for ms in (0.20, 0.30):
        c = PostHocConfig(multi_season_smoothing=ms)
        results.append(evaluate(f"04_smooth_{ms}", c,
                                team_scale, lineup_seasons, min_poss))

    # Star amp
    for t1, t2 in [(1.10, 1.05), (1.20, 1.05)]:
        c = PostHocConfig(star_amp_top1=t1, star_amp_top2=t2)
        results.append(evaluate(f"05_star_amp_{t1}_{t2}", c,
                                team_scale, lineup_seasons, min_poss))

    # Defensive shrinkage
    for ds in (0.10, 0.20):
        c = PostHocConfig(defensive_archetype_shrinkage=ds)
        results.append(evaluate(f"06_def_shrink_{ds}", c,
                                team_scale, lineup_seasons, min_poss))

    return results


def stage2_combos(team_scale, lineup_seasons, min_poss) -> List[Dict]:
    """Combinations of the most promising knobs."""
    results = []

    # Combo A: matchup d6 + smoothing
    for s in (0.5,):
        for ms in (0.20, 0.30):
            c = PostHocConfig(use_matchup_dim6_residual=True,
                              matchup_dim6_strength=s,
                              multi_season_smoothing=ms)
            results.append(evaluate(f"A_matchup{s}_smooth{ms}", c,
                                    team_scale, lineup_seasons, min_poss))

    # Combo B: smoothing + star amp
    for ms in (0.20,):
        c = PostHocConfig(multi_season_smoothing=ms,
                          star_amp_top1=1.10, star_amp_top2=1.05)
        results.append(evaluate(f"B_smooth{ms}_star1.10_1.05", c,
                                team_scale, lineup_seasons, min_poss))

    # Combo C: all-on conservative
    c = PostHocConfig(
        use_matchup_dim6_residual=True, matchup_dim6_strength=0.5,
        multi_season_smoothing=0.20,
        star_amp_top1=1.05, star_amp_top2=1.02,
        offense_gain=1.05, defense_gain=0.98,
    )
    results.append(evaluate("C_full_combo_conservative", c,
                            team_scale, lineup_seasons, min_poss))

    # Combo D: aggressive
    c = PostHocConfig(
        use_matchup_dim6_residual=True, matchup_dim6_strength=1.0,
        multi_season_smoothing=0.30,
        star_amp_top1=1.10, star_amp_top2=1.05,
        offense_gain=1.10, defense_gain=1.0,
    )
    results.append(evaluate("D_full_combo_aggressive", c,
                            team_scale, lineup_seasons, min_poss))

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--team-scale", type=float, default=20.0)
    parser.add_argument("--lineup-seasons", nargs="*",
                        default=["2022-23", "2023-24", "2024-25"])
    parser.add_argument("--min-poss", type=int, default=50)
    parser.add_argument("--stage", choices=["1", "2", "both"], default="both")
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()

    print(f"v3.2 Post-Hoc Sweep")
    print(f"  team_scale={args.team_scale}  lineup_seasons={args.lineup_seasons}")
    print(f"  min_poss={args.min_poss}  stage={args.stage}")
    print("=" * 80)

    all_results = []
    if args.stage in ("1", "both"):
        print("\nStage 1: single-knob sweep")
        all_results.extend(stage1_single_knobs(
            args.team_scale, args.lineup_seasons, args.min_poss))
    if args.stage in ("2", "both"):
        print("\nStage 2: combination sweep")
        all_results.extend(stage2_combos(
            args.team_scale, args.lineup_seasons, args.min_poss))

    # Rank and print
    print("\n" + "=" * 80)
    print(f"{'label':<50}  {'Brier':>8}  {'r':>7}  {'wr':>7}  {'∆Brier':>7}")
    print("-" * 80)
    valid = [r for r in all_results if "brier" in r]
    baseline_brier = None
    for r in valid:
        if r["label"] == "00_baseline":
            baseline_brier = r["brier"]["brier"]
            break
    if baseline_brier is None and valid:
        baseline_brier = valid[0]["brier"]["brier"]

    valid_sorted = sorted(valid, key=lambda r: r["brier"]["brier"] if r["brier"]["brier"] else 99)
    for r in valid_sorted:
        b = r["brier"].get("brier")
        lr = r.get("lineup", {}).get("mean_pearson_r")
        wr = r.get("lineup", {}).get("mean_weighted_pearson_r")
        delta = b - baseline_brier if (b and baseline_brier) else None
        print(f"  {r['label']:<48}  {b:>8.6f}  "
              f"{lr if lr is not None else 'NA':>7}  "
              f"{wr if wr is not None else 'NA':>7}  "
              f"{delta:>+7.4f}" if delta is not None else
              f"  {r['label']:<48}  {b}  NA  NA  NA")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"baseline_brier": baseline_brier,
                   "results": all_results}, f, indent=2, default=str)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
