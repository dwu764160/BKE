# scripts/

One-off analysis, experiment, and validation scripts. These are not imported by the pipeline — they read from pipeline outputs and write to `reports/` or `data/processed/`.

---

## Build scripts (run after pipeline stages)

| Script | Runs after | Output |
|--------|-----------|--------|
| `build_pts_v40.py` | `decomposition_engine`, archetypes | `data/processed/bke/pts_v40.parquet` |
| `build_all_season_projections.py` | `build_profile_aggregate` | `data/processed/forecast/projected_team_features.parquet` |
| `build_2025_26_projections.py` | `build_profile_aggregate` | `projected_team_features_v40_2025-26.parquet` (current season) |
| `build_rest_features.py` | `team_game_logs.parquet` | `game_rest_features.parquet` |
| `build_ytd_team_ratings.py` | `build_all_season_projections` | `team_ratings_ytd.parquet` |
| `build_pts_team_ratings.py` | `pts_v40.parquet` | team PTS-based net ratings |
| `compute_matchup_adj.py` | `cross_team_interactions_matrix.parquet`, lineups | `cross_team_interactions.parquet` |

## Simulation runners

| Script | Purpose |
|--------|---------|
| `run_possession_engine.py` | Generative possession engine — calibrate + simulate. `--mode v0` (starters + bench unit) or `--mode v1` (9-man rotation, production). `--season`, `--sims` flags. |
| `phase4_apply_schema_contract.py` | One-time utility: apply `canonicalize()` to a parquet directory. |

## Validation / gates

| Script | What it checks |
|--------|---------------|
| `validate_possession_engine.py` | Step 2 calibration gate — player-prop MAE + team-total MAE vs 2024-25 actuals |
| `validate_lineup_pts.py` | Lineup-level PTS attribution vs PBP ground truth |
| `validate_lineup_pts_v2.py` | v2 lineup PTS validation (pts_v40-aware) |
| `validate_lineup_projection.py` | Starter/rotation overlap vs observed PBP lineups |
| `validate_matchup_interactions.py` | Walk-forward PPP MAE for cross-team interaction cells (2024-25 holdout) |
| `validate_player_stat_sim.py` | Player stat simulation vs actual box scores |
| `validate_pts_team_ratings.py` | PTS-derived team ratings vs actual net ratings |
| `validate_backup_integrity.py` | Row count + SHA-256 checksum vs `data_backup_20260601/` |

## Archetype / interaction pre-flights (locked — do not re-run)

| Script | Output | Status |
|--------|--------|--------|
| `test_archetype_interactions.py` | `archetype_interaction_signal_test.json` | **LOCKED** — two-way FE signal test, 8 seasons, 71 sig cells. Do not re-run. |
| `fit_archetype_interactions_v2.py` | `cross_team_interactions_matrix.parquet` | EB-shrunk production matrix. Re-run only if raw matchup data changes. |
| `test_softmax_vs_argmax.py` | `softmax_vs_argmax_test.json` | Argmax vs softmax archetype assignment comparison |

## Experiment scripts

These were written to test specific hypotheses. Results are locked in `docs/findings/` — rerun only if you need to re-investigate.

| Script | What it tested |
|--------|---------------|
| `experiment_microadjustments.py` | Schedule micro-adjustments (B2B stacking, travel) |
| `experiment_pts_rdis_blend.py` | PTS/RDIS blending approaches for team ratings |
| `experiment_ytd_roster_pts.py` | YTD roster PTS as a blended rating signal |
| `pts_v32_harness.py` | PTS v3.2 sweep harness (baseline comparison) |
| `pts_v32_posthoc_harness.py` | Post-hoc v3.2 analysis with matchup dim-6 |
| `pts_v32_recompute.py` | v3.2 recompute with config sweeps |
| `pts_v40_defense.py` | PTS v4.0 defense component testing |
| `pts_v40_multiseason.py` | PTS v4.0 cross-season stability |
| `pts_v40_sweep.py` | PTS v4.0 composite weight sweep |
| `grid_sweep_sim_knobs.py` | Simulation parameter sweep |
| `diagnose_brier_baseline.py` | Brier score baseline decomposition |
| `oos_check.py` | Out-of-sample check for model features |
| `audit_forecast.py` | Forecast pipeline audit |
| `audit_forecast_leakage.py` | Temporal leakage audit for game-level forecasts |

## Analysis / comparison

| Script | Purpose |
|--------|---------|
| `compare_bref_advanced_metrics.py` | BKE vs Basketball-Reference WS/BPM/VORP comparison |
| `compare_data_batches.py` | Compare two data directories (old vs fresh pipeline run) |
| `forecast_2025_26_games.py` | 2025-26 game-level forecast export |
| `fetch_kalshi_closing_lines.py` | Fetch Kalshi market closing lines for CLV comparison |
| `fill_bref_ground_truth.py` | Populate B-Ref ground-truth data |
| `backfill_tracking_missing.py` | Backfill missing tracking seasons |
| `fetch_2025_26_game_logs.py` | Fetch current season game logs |
| `fetch_pbp_2025_26_statsapi.py` | Fetch 2025-26 PBP via Stats API |

## Utility

| Script | Purpose |
|--------|---------|
| `log_read_file.py` | Log a file read event to the session log |
| `log_skill_usage.py` | Log a skill invocation |
| `preload_skills.py` | Pre-load skill files into context |
| `scan_copilot_models.py` | Scan available Copilot model endpoints |
| `validate_skills.py` | Validate skill file structure |
| `watch_update_copilot_reasoning_effort.py` | Watch and update Copilot reasoning effort setting |
