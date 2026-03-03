# BKE — Basketball KPI Engine

Lightweight analytics pipeline for possession-level RAPM (Regularized Adjusted Plus-Minus), ORAPM and DRAPM.

# Recreate BKE (commands)

Prereqs
- Python 3.9+ and `pip`
- Optional: `graphviz` for rendering dot files, `playwright` if using the DOM PBP fetcher

Quick setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Pipeline (order matters)

## Fetch / Ingest
```bash
python3 src/data_fetch/fetch_pbp/bootstrap_nba_session.py        # Init NBA session (cookies, headers)
python3 src/data_fetch/fetch_pbp/capture_nba_headers.py          # Save NBA API headers
python3 src/data_fetch/fetch_historical_data.py                  # Fetch historical game/team data
python3 src/data_fetch/fetch_players.py                          # Fetch player metadata
python3 src/data_fetch/fetch_teams.py                            # Fetch team metadata
python3 src/data_fetch/fetch_profiles.py                         # Fetch player profiles
python3 src/data_fetch/fetch_player_salaries.py                  # Fetch player salary data (per season, ESPN)
python3 src/data_fetch/fetch_pbp/CDN_pbp_fetch.py                # Fetch play-by-play (CDN)
python3 src/data_fetch/fetch_pbp/fetch_play_by_play.py           # Fetch play-by-play (DOM fallback)
python3 src/data_fetch/fetch_official_stats.py                   # Fetch official NBA stats
python3 src/data_fetch/fetch_tracking_data.py                    # Fetch NBA tracking data
python3 src/data_fetch/fetch_box_scores_complete.py              # Fetch full box scores
python3 src/data_fetch/fetch_matchup_data.py                     # Fetch matchup data
python3 src/data_fetch/fetch_shot_zones.py                       # Fetch shot zone data
python3 src/data_fetch/fetch_darko_manual.py --input <path_or_dir> # Stage manual DARKO CSV exports
```

## Derive / Normalize / Features
```bash
python3 src/data_fetch/derive_team_game_logs.py           # Derive team game logs
python3 src/data_fetch/summarize_team_logs.py             # Summarize team logs
python3 src/utils/export_db_to_parquet.py                 # Export DB tables to parquet
python3 src/data_normalize/run_normalization.py           # Normalize raw data
python3 src/data_normalize/normalize_darko.py             # Normalize DARKO exports to canonical schema
python3 src/features/derive_lineups.py                    # Derive lineups
python3 src/features/derive_possessions.py                # Derive possessions
python3 src/features/compute_rest_home_back2back.py       # Compute rest/home/back-to-back
```

## Compute / Metrics
```bash
python3 src/data_compute/compute_clean_possessions.py         # Clean/validate possessions
python3 src/data_compute/compute_local_metrics.py            # Compute local metrics
python3 src/data_compute/compute_linear_metrics.py           # Compute linear metrics (WS, BPM, VORP)
python3 src/data_compute/compute_advanced_metrics.py         # Compute advanced metrics
python3 src/data_compute/compute_player_profiles.py          # Compute player profiles
python3 src/data_compute/compute_player_archetypes.py        # Compute offensive archetypes
python3 src/data_compute/compute_position_estimate.py        # Compute position estimate (PG/SG/SF/PF/C shares)
python3 src/data_compute/compute_defensive_archetypes_v2.py  # Compute defensive archetypes (v2, balanced confidence: 0.50*top + 0.50*(0.5+0.5*margin^0.4) + 0.03*max(0, top-0.75); margin fallback gap=0.03)
```

## Modeling / Impact Metrics
```bash
python3 src/modeling/model_rapm.py                           # Compute RAPM / ORAPM / DRAPM
python3 src/modeling/ingest_darko.py                         # Build modeling_inputs_{season} tables
python3 src/modeling/decomposition_engine.py                 # Run BKE v2.8 full decomposition pipeline
#   Includes: Layer 1 (Portable Talent w/ Bayesian shrinkage)
#             Layer 2 (Role Utilization Efficiency)
#             Layer 3 (Archetype Elevation)
#             Layer 4 (Scheme Stability, bonus-only)
python3 src/modeling/construct_bke_scores_v27.py            # Build final OBKE/DBKE/BKE output JSON (terminal ranking)
python3 -m src.modeling.backtesting                          # Run year-over-year prediction backtest
python3 -m src.modeling.backtesting --all-pairs              # Run all season-pair backtests
python3 tests/bke_v29_diagnostics.py                         # Run v2.9 diagnostic suite (8 domains → JSON report)
python3 src/modeling/dbke_v30_defense_shrinkage.py          # Run v3.0 defense reconstruction phases (A/B/C) with stop-on-fail gates (Phase B stable geometry + Phase C reliability-weighted scaling + specialist rank-stability diagnostics)
python3 src/modeling/bke_v31_experimental_layers.py         # Run v3.1 experimental layer suite (independent layer tests + constrained combined model)
python3 src/modeling/experiment2_production_tilt.py         # Run v3.1 Experiment 2 rerun (15-λ production-tilt sweep using ORAPM/TS + raw offensive box-score stats)
```

## Player Evaluation (PEC v1)
```bash
python3 src/player_eval/build_player_impact_profiles.py     # PEC Step 1: build PlayerImpactProfile (47+ fields from 9 sources, canonical offensive/defensive archetype labels + embeddings, behavioral fingerprint)
python3 src/profile_aggregate/build_profile_aggregate.py    # Profile Aggregate: merge all 15 pipeline sources into aggregate/player_profile_aggregate.parquet (1971 rows x 908 cols)
python3 src/player_eval/train_minute_model.py               # PEC Step 2: train MPG prediction model (GBDT, 70 features, GroupKFold CV, temporal holdout; excludes volume stats)
```

## Visualization / Export
```bash
python3 app/player_archetype_viewer.py    # Generate archetype viewer
python3 app/player_data_viewer.py         # Generate player data viewer (with salary + BKE split toggle: 60/40 or 55/45)
python3 scripts/export_bke_components.py  # Export per-player BKE components JSON for interactive viewer
python3 app/player_bke_viewer.py          # Generate standalone BKE interactive explorer (lambda slider + split toggle)
python3 app/player_eval_viewer.py         # Generate PEC viewer (player cards, team view, detail modal, predicted vs actual MPG)
python3 src/utils/export_db_to_parquet.py # Export DB tables to parquet
```

# Validation & tests

```bash
python3 tests/validate_rapm.py
python3 tests/validate_data_integrity.py --season 2024-25
pytest -q
```

# Diagram

```bash
dot -Tpng scheme_diagrams/flow_diagram_pre_possession.dot -o scheme_diagrams/flow_diagram_pre_possession.png
```

# Data layout (locations used by scripts)
- `data/historical/` — raw + normalized PBP, possessions, caches; per-season salary files: `player_salaries_2022-23.parquet`, `player_salaries_2023-24.parquet`, etc. (columns: player_id, player_name, team, team_id, season, salary)
- `data/processed/` — core pipeline outputs: `player_rapm.parquet`, `player_rapm.csv`, `modeling_inputs_all.parquet/.csv`, `modeling_inputs_{season}.parquet`, `player_position_estimates_2022-23.parquet/.csv`, `player_position_estimates_2023-24.parquet/.csv`, `player_position_estimates_2024-25.parquet/.csv`, combined compatibility `player_position_estimates.parquet/.csv`, `defensive_archetypes_v2.parquet`, `defensive_archetypes_v2.csv`, `player_archetypes.parquet`, `archetype_embeddings.parquet`, `metrics_linear.parquet`, `metrics_win_shares.parquet`
- `data/processed/player_eval/` — PEC Step outputs: `player_impact_profiles.parquet`, `player_profiles_season.pkl`, `minute_model_v2.pkl`, `minute_model_predictions_v2.parquet`
- `aggregate/` — comprehensive player profile aggregate: `player_profile_aggregate.parquet` (1971 rows x 908 cols, all 15 pipeline sources merged)
- `data/processed/bke/` — BKE decomposition data: `bke_v28_decomposition.parquet`, `bke_v28_decomposition.csv`, `BKE_Scores_v27.json` (includes terminal league-wide percentiles plus grouped transformed percentiles by `position_bucket`, `primary_archetype`, and `defensive_archetype`), `dimension_scores_v28.json`, `layer_scores_v28.json`, `obke_dbke_scores_v28.json`, v3.1 split artifacts `BKE_Scores_v31_60_40.json`, `BKE_Scores_v31_55_45.json`, and (when Layer 3+6 second-pass is additive) `BKE_Scores_v31_60_40_layer36.json`, `BKE_Scores_v31_55_45_layer36.json`, `bke_v31_components.json` (per-player components for interactive viewer)
- `reports/` — all report/diagnostic/validation outputs: `bke_v28_report.json`, `bke_v28_variance_report.json`, `bke_v28_compression_report.json`, `bke_v27_backtest.json`, `bke_v29_diagnostic_master.json`, `dbke_v30_defense_shrinkage.json`, `bke_v31_experimental_layers.json`, `bke_v31_layer36_second_pass.json`, `modeling_inputs_report.json`, `player_eval_step1_validation.json`, `player_eval_step2_minute_model_validation.json`, `profile_aggregate_validation.json`, `bref_metric_comparison_{season}.csv`, `defensive_archetypes_v2_impact_report.csv/.txt`, `validation_report_*.json`
- `data/tracking/` — tracking-derived JSONs
- `src/player_eval/` — new phase workspace for upcoming player evaluation engine scripts
- `data_backup_YYYYMMDD/` — dated snapshot backup folders (use `bash scripts/backup_data_snapshot.sh [YYYYMMDD]`)

# Reference docs structure
- `reference/` now supports contribution-based organization with subfolders:
	- `reference/archetypes/`
	- `reference/data_fetch/`
	- `reference/data_schema/`
	- `reference/modeling/`
	- `reference/pipeline/`
	- `reference/player_eval/`
	- `reference/app/`
- Root-level canonical modeling plan/summary files are retained for compatibility:
	- `reference/BKE_metric_modelling_plan.md`
	- `reference/BKE_metric_modelling_summary.md`

# Notes
- Inspect `src/*` scripts for CLI flags and optional args (season filters, caching).
- Tweak `SEASON_DECAY_WEIGHTS` and alpha grids in `src/modeling/model_rapm.py` to change pooling/regularization.
- BKE split policy from v3.1 onward: co-produce and support both `60/40` (stability anchor, default) and `55/45` (predictive variant) in outputs/viewers.

# Future Upgrade Ideas

## Central Archetype Tuning Knob File

Create a single file (e.g., archetype_coefficients.json) containing coefficients for each defensive and offensive archetype. Changing a coefficient in this file would directly control the prevalence/distribution of each archetype in the pipeline. Both compute_defensive_archetypes_v2.py and compute_player_archetypes.py would read from this file and apply the coefficients during role assignment. This enables rapid, unified, and transparent tuning of archetype distributions.

## Consolidated Player Profile File

After player evaluation is complete, merge all player data (bios, advanced stats, archetypes, RAPM, position, etc.) into a single consolidated file per season (e.g., player_profiles_2024-25.parquet). This file would serve as the authoritative, denormalized source for all downstream tools and viewers, enabling fast, reliable, and simple access to the complete player record for any season.

# Known Bugs / Issues

## Player Names such as Jokic and Doncic breaking name-based matching

## Standardized nicknames such as Herb Jones vs. Herbert Jones, Carlton Carrington vs. Bub Carrington, etc.