"""
src/simulation/run_forecast.py
=============================================================================
Forecast Orchestrator — Run forward-projection pipeline for roster scenarios.

Scenarios:
  - end_of_season: carry-forward / end-of-season team context (peek)
  - preseason_snapshot: opening-roster snapshot mapping (Option B)

Pipeline per scenario:
  1. project_next_season → projected player profiles
  2. team_feature_aggregation (forecast) → projected team features
  3. season_sim (forecast) → projected season outcomes (margin + PPP)
  4. lineup_projection (forecast) → projected lineup profiles

Writes both scenario-specific artifacts and combined frontend artifacts.

Usage:
  python3 src/simulation/run_forecast.py
  python3 src/simulation/run_forecast.py --forecast 2025-26
=============================================================================
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.player_eval.constants import (  # noqa: E402
    FORECAST_LINEUP_REPORT,
    FORECAST_VALIDATION_REPORT,
    PLAYER_PROFILES_PARQUET,
)
from src.simulation.simulation_config import (  # noqa: E402
    FORECAST_DIR,
    FORECAST_PLAYER_GAME_SAMPLES_PATH,
    FORECAST_PLAYER_SEASON_STATS_PATH,
    FORECAST_SEASON_RESULTS_PATH,
    REPORTS_DIR,
    SIMULATION_PROCESSED_DIR,
)


SCENARIO_LABELS = {
    "end_of_season": "End-of-Season Roster (Peek)",
    "preseason_snapshot": "Preseason Roster Snapshot",
}
DEFAULT_SCENARIOS = ["end_of_season", "preseason_snapshot"]


def _parse_scenarios(raw: str) -> List[str]:
    if not raw:
        return list(DEFAULT_SCENARIOS)
    keys = [s.strip() for s in raw.split(",") if s.strip()]
    out = []
    for k in keys:
        if k not in SCENARIO_LABELS:
            raise ValueError(f"Unknown scenario: {k}. Valid: {sorted(SCENARIO_LABELS)}")
        if k not in out:
            out.append(k)
    return out


def _determine_backtest_target_seasons() -> List[str]:
    if not PLAYER_PROFILES_PARQUET.exists():
        return []
    df = pd.read_parquet(PLAYER_PROFILES_PARQUET, columns=["season"])
    seasons = sorted({str(s) for s in df["season"].dropna().astype(str).tolist()})
    if len(seasons) < 2:
        return []
    return seasons[1:]


def _safe_load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _ensure_preseason_rosters(
    scenarios: List[str],
    target_seasons: List[str],
    preseason_rosters_path: Path,
    skip_fetch: bool,
) -> None:
    if "preseason_snapshot" not in scenarios:
        return
    if skip_fetch:
        print("  Skipping preseason roster fetch (--skip-preseason-fetch)")
        return
    if not target_seasons:
        return

    print("\n[setup] Ensuring preseason roster snapshots...")
    from src.data_fetch.fetch_preseason_rosters import fetch_preseason_rosters

    # Fetch into canonical output; projection can still read user override path.
    fetch_preseason_rosters(seasons=target_seasons, force_refresh=False)
    if preseason_rosters_path:
        print(f"  Projection preseason path override: {preseason_rosters_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run forecast simulation pipeline for roster scenarios")
    parser.add_argument("--forecast", type=str, default=None, help="Target season for true forecast (e.g., 2025-26)")
    parser.add_argument("--roster", type=str, default=None, help="Path to manual roster CSV overrides")
    parser.add_argument("--rookies", type=str, default=None, help="Optional rookies CSV fallback override")
    parser.add_argument("--rookie-impact-scale", type=float, default=None, help="Override rookie impact scale multiplier")
    parser.add_argument("--no-rookie-scale-tune", action="store_true", help="Disable historical rookie-scale tuning")
    parser.add_argument("--skip-lineup", action="store_true", help="Skip lineup projection step")
    parser.add_argument(
        "--scenarios",
        type=str,
        default=",".join(DEFAULT_SCENARIOS),
        help="Comma-separated scenarios: end_of_season,preseason_snapshot",
    )
    parser.add_argument(
        "--preseason-rosters-path",
        type=str,
        default=None,
        help="Optional preseason roster parquet (combined or season-specific dir)",
    )
    parser.add_argument(
        "--skip-preseason-fetch",
        action="store_true",
        help="Skip auto-fetch of preseason rosters before preseason scenario",
    )
    args = parser.parse_args()

    scenarios = _parse_scenarios(args.scenarios)
    forecast_mode = args.forecast is not None
    mode_label = "FORECAST" if forecast_mode else "BACKTEST"
    preseason_rosters_path = Path(args.preseason_rosters_path) if args.preseason_rosters_path else None

    print("=" * 70)
    print(f"  FORECAST PIPELINE — {mode_label} — scenarios={','.join(scenarios)}")
    print("=" * 70)
    t0 = time.time()

    target_seasons = [args.forecast] if args.forecast else _determine_backtest_target_seasons()
    _ensure_preseason_rosters(
        scenarios=scenarios,
        target_seasons=target_seasons,
        preseason_rosters_path=preseason_rosters_path,
        skip_fetch=args.skip_preseason_fetch,
    )

    scenario_outputs: Dict[str, Dict] = {}

    for scenario_idx, scenario_key in enumerate(scenarios, start=1):
        scenario_label = SCENARIO_LABELS[scenario_key]
        print(f"\n{'-' * 70}")
        print(f"[{scenario_idx}/{len(scenarios)}] Scenario: {scenario_key} ({scenario_label})")
        print(f"{'-' * 70}")

        # Scenario-specific artifact paths.
        profiles_path = FORECAST_DIR / f"projected_player_profiles_{scenario_key}.parquet"
        team_features_path = FORECAST_DIR / f"projected_team_features_{scenario_key}.parquet"
        season_results_path = REPORTS_DIR / f"forecast_season_results_{scenario_key}.json"
        validation_path = REPORTS_DIR / f"forecast_validation_{scenario_key}.json"
        lineup_report_path = REPORTS_DIR / f"forecast_lineup_profiles_{scenario_key}.json"
        lineup_profiles_path = SIMULATION_PROCESSED_DIR / f"forecast_step2_lineup_profiles_{scenario_key}.parquet"
        lineup_validation_path = REPORTS_DIR / f"forecast_step2_validation_{scenario_key}.json"
        player_season_stats_path = FORECAST_PLAYER_SEASON_STATS_PATH.with_name(
            f"forecast_step1_player_season_stats_{scenario_key}.parquet"
        )
        player_game_samples_path = FORECAST_PLAYER_GAME_SAMPLES_PATH.with_name(
            f"forecast_step1_player_game_samples_{scenario_key}.parquet"
        )

        # Step 1: Project player profiles.
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
        if preseason_rosters_path:
            sys.argv += ["--preseason-rosters-path", str(preseason_rosters_path)]
        sys.argv += ["--team-mapping-mode", scenario_key]
        sys.argv += ["--output-path", str(profiles_path)]
        sys.argv += ["--validation-output-path", str(validation_path)]
        project_main()

        # Step 2: Team feature aggregation.
        print("\n[2/4] Computing team features (forecast mode)...")
        from src.profile_aggregate.team_feature_aggregation import main as team_agg_main

        team_agg_main(
            forecast_mode=True,
            profiles_path=profiles_path,
            output_path=team_features_path,
        )

        # Step 3: Season simulation.
        print("\n[3/4] Running season simulation (forecast mode)...")
        from src.simulation.season_sim import main as season_sim_main

        season_sim_main(
            forecast_mode=True,
            features_path=team_features_path,
            player_profiles_path=profiles_path,
            output_path=season_results_path,
        )

        # Step 4: Lineup projection.
        if not args.skip_lineup:
            print("\n[4/4] Building lineup projections (forecast mode)...")
            from src.simulation.lineup_projection import main as lineup_main

            lineup_main(
                forecast_mode=True,
                profiles_path=profiles_path,
                profiles_output_path=lineup_profiles_path,
                report_output_path=lineup_report_path,
                validation_output_path=lineup_validation_path,
            )
        else:
            print("\n[4/4] Skipping lineup projection (--skip-lineup)")

        scenario_outputs[scenario_key] = {
            "label": scenario_label,
            "team_mapping_mode": scenario_key,
            "paths": {
                "projected_profiles": str(profiles_path),
                "projected_team_features": str(team_features_path),
                "season_results": str(season_results_path),
                "player_season_stats": str(player_season_stats_path),
                "player_game_samples": str(player_game_samples_path),
                "validation": str(validation_path),
                "lineup_report": str(lineup_report_path),
                "lineup_profiles": str(lineup_profiles_path),
                "lineup_validation": str(lineup_validation_path),
            },
            "season_results": _safe_load_json(season_results_path),
            "validation": _safe_load_json(validation_path),
            "lineup": _safe_load_json(lineup_report_path),
        }

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  FORECAST PIPELINE COMPLETE — {elapsed:.1f}s")
    print(f"{'=' * 70}")

    default_scenario = scenarios[0] if scenarios else "end_of_season"
    default_season_payload = scenario_outputs.get(default_scenario, {}).get("season_results", {})
    default_validation_payload = scenario_outputs.get(default_scenario, {}).get("validation", {})
    default_lineup_payload = scenario_outputs.get(default_scenario, {}).get("lineup", {})

    combined_season_results = {
        "config": default_season_payload.get("config", {}),
        "seasons": default_season_payload.get("seasons", {}),
        "default_scenario": default_scenario,
        "scenario_labels": {k: SCENARIO_LABELS[k] for k in scenarios},
        "scenarios": {
            key: {
                "label": payload["label"],
                "team_mapping_mode": payload["team_mapping_mode"],
                "paths": payload["paths"],
                "config": payload["season_results"].get("config", {}),
                "seasons": payload["season_results"].get("seasons", {}),
            }
            for key, payload in scenario_outputs.items()
        },
    }

    combined_validation = {
        "default_scenario": default_scenario,
        "scenario_labels": {k: SCENARIO_LABELS[k] for k in scenarios},
        "mode": default_validation_payload.get("mode", mode_label.lower()),
        "projections": default_validation_payload.get("projections", {}),
        "scenarios": {k: v.get("validation", {}) for k, v in scenario_outputs.items()},
    }

    combined_lineup = {
        "default_scenario": default_scenario,
        "scenario_labels": {k: SCENARIO_LABELS[k] for k in scenarios},
        "seasons": default_lineup_payload.get("seasons", {}),
        "validation_overall": default_lineup_payload.get("validation_overall", {}),
        "scenarios": {k: v.get("lineup", {}) for k, v in scenario_outputs.items()},
    }

    FORECAST_SEASON_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FORECAST_SEASON_RESULTS_PATH.write_text(json.dumps(combined_season_results, indent=2), encoding="utf-8")
    FORECAST_VALIDATION_REPORT.write_text(json.dumps(combined_validation, indent=2), encoding="utf-8")
    FORECAST_LINEUP_REPORT.write_text(json.dumps(combined_lineup, indent=2), encoding="utf-8")

    print("\n  Combined outputs:")
    print(f"    ✓ {FORECAST_SEASON_RESULTS_PATH}")
    print(f"    ✓ {FORECAST_VALIDATION_REPORT}")
    print(f"    ✓ {FORECAST_LINEUP_REPORT}")

    for key in scenarios:
        paths = scenario_outputs.get(key, {}).get("paths", {})
        print(f"\n  Scenario artifacts [{key}]:")
        for p in paths.values():
            exists = "✓" if Path(p).exists() else "✗"
            print(f"    {exists} {p}")


if __name__ == "__main__":
    main()
