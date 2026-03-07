"""
src/simulation/run_forecast.py
=============================================================================
Forecast Orchestrator — Run full forward-projection simulation pipeline

Pipeline:
  1. project_next_season → projected player profiles
  2. team_feature_aggregation (forecast) → projected team features
  3. season_sim (forecast) → projected season outcomes
  4. lineup_projection (forecast) → projected lineup profiles

Modes:
  Backtest:  Project known seasons to validate forecast accuracy
  Forecast:  Project next season from latest available data

Usage:
  python3 src/simulation/run_forecast.py                     # Backtest
  python3 src/simulation/run_forecast.py --forecast 2025-26  # True forecast
    python3 src/simulation/run_forecast.py --roster roster.csv  # Optional roster overrides
=============================================================================
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser(description="Run forecast simulation pipeline")
    parser.add_argument("--forecast", type=str, default=None,
                        help="Target season for true forecast (e.g., 2025-26)")
    parser.add_argument("--roster", type=str, default=None,
                        help="Path to roster CSV for team mappings")
    parser.add_argument("--rookies", type=str, default=None,
                        help="Optional rookies CSV fallback override")
    parser.add_argument("--rookie-impact-scale", type=float, default=None,
                        help="Override rookie impact scale multiplier")
    parser.add_argument("--no-rookie-scale-tune", action="store_true",
                        help="Disable historical rookie-scale tuning")
    parser.add_argument("--skip-lineup", action="store_true",
                        help="Skip lineup projection step")
    args = parser.parse_args()

    forecast_mode = args.forecast is not None
    mode_label = "FORECAST" if forecast_mode else "BACKTEST"

    print("=" * 70)
    print(f"  FORECAST PIPELINE — {mode_label}")
    print("=" * 70)
    t0 = time.time()

    # ━━━━ Step 1: Project player profiles ━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n[1/4] Projecting player profiles...")
    from src.player_eval.project_next_season import main as project_main
    sys.argv = ["project_next_season"]
    if args.forecast:
        sys.argv += ["--forecast", args.forecast]
    if args.roster:
        sys.argv += ["--roster", args.roster]
    if args.rookies:
        sys.argv += ["--rookies", args.rookies]
    if args.rookie_impact_scale is not None:
        sys.argv += ["--rookie-impact-scale", str(args.rookie_impact_scale)]
    if args.no_rookie_scale_tune:
        sys.argv += ["--no-rookie-scale-tune"]
    project_main()

    # ━━━━ Step 2: Team feature aggregation (forecast mode) ━━━━━━━
    print("\n[2/4] Computing team features (forecast mode)...")
    from src.profile_aggregate.team_feature_aggregation import main as team_agg_main
    team_agg_main(forecast_mode=True)

    # ━━━━ Step 3: Season simulation (forecast mode) ━━━━━━━━━━━━━━
    print("\n[3/4] Running season simulation (forecast mode)...")
    from src.simulation.season_sim import main as season_sim_main
    season_sim_main(forecast_mode=True)

    # ━━━━ Step 4: Lineup projection (forecast mode) ━━━━━━━━━━━━━━
    if not args.skip_lineup:
        print("\n[4/4] Building lineup projections (forecast mode)...")
        from src.simulation.lineup_projection import main as lineup_main
        lineup_main(forecast_mode=True)
    else:
        print("\n[4/4] Skipping lineup projection (--skip-lineup)")

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  FORECAST PIPELINE COMPLETE — {elapsed:.1f}s")
    print(f"{'=' * 70}")

    # Print summary
    from src.player_eval.constants import PROJECTED_PROFILES_PATH, FORECAST_VALIDATION_REPORT
    from src.simulation.simulation_config import (
        FORECAST_SEASON_RESULTS_PATH,
        FORECAST_TEAM_FEATURES_PATH,
    )

    print(f"\n  Outputs:")
    for p in [PROJECTED_PROFILES_PATH, FORECAST_TEAM_FEATURES_PATH,
              FORECAST_SEASON_RESULTS_PATH, FORECAST_VALIDATION_REPORT]:
        exists = "✓" if p.exists() else "✗"
        print(f"    {exists} {p}")


if __name__ == "__main__":
    main()
