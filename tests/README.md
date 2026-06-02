# tests/

Active tests and debug/validation scripts.

---

## Active tests (always runnable, data-free)

```bash
pytest -q tests/
```

| File | What it tests |
|------|--------------|
| `test_simulation_core.py` | 3 simulation-core unit tests — data-free, always green |

---

## Validation scripts (require data pipeline outputs)

These are not run by `pytest`. Run them manually after pipeline changes.

| Script | Validates |
|--------|-----------|
| `validate_possessions.py` | Possession ORTG, pace, lineup completeness |
| `validate_rapm.py` | RAPM distribution, external benchmarks, stability |
| `validate_advanced_metrics.py` | WS/BPM/VORP vs B-Ref ground truth |
| `validate_ws_broad.py` | Win Shares for 7 key players vs B-Ref |
| `validate_official_stats.py` | Official stats schema, ranges, player counts |
| `validate_tracking_data.py` | Tracking file existence, CatchShoot proxy check |
| `validate_gamelogs.py` | Game log integrity (row counts, season coverage) |
| `validate_data_integrity.py` | Broad data integrity sweep (`--season 2024-25`) |
| `validate_team_game_logs_realworld.py` | Team game log real-world sanity checks |
| `pbp_test.py` | PBP parse correctness |
| `test_pbp_parser_ids.py` | PBP player ID consistency |
| `VALIDATION_REPORT.md` | Human-readable snapshot of last validation run |

---

## Debug scripts (tests/debug/)

One-off diagnostic scripts for pipeline investigation. Not part of CI.

| Script | What it diagnoses |
|--------|------------------|
| `audit_league_constants.py` | League-wide rate constants (pace, OREB%, FT%) |
| `audit_possessions.py` | Possession count vs expected |
| `audit_turnover_types.py` | TOV type distribution |
| `debug_lineup_solver.py` | Lineup inference edge cases |
| `debug_rapm_names.py` | Player name matching in RAPM inputs |
| `deep_dive_stats.py` | Deep per-player stat breakdown |
| `diagnose_bad_possessions.py` | Bad possession root-cause analysis |
| `diagnose_metrics_issues.py` | Metric value anomalies |
| `inspect_foul_codes.py` | PBP foul code mapping |
| `inspect_possessions_api_structure.py` | Raw PBP API response structure |
| `inspect_steal_metadata.py` | Steal event metadata |
| `scan_unknown_events.py` | Unrecognized PBP event types |
| `trace_bad_lineups.py` | Lineup inference failures |
| `trace_game_possessions.py` | Game-level possession trace |
