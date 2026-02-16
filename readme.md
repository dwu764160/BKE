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
python3 src/data_compute/compute_defensive_archetypes_v2.py  # Compute defensive archetypes (v2)
```

## Modeling / Impact Metrics
```bash
python3 src/modeling/model_rapm.py                           # Compute RAPM / ORAPM / DRAPM
python3 src/modeling/ingest_darko.py                         # Build modeling_inputs_{season} tables
```

## Visualization / Export
```bash
python3 app/player_archetype_viewer.py    # Generate archetype viewer
python3 app/player_data_viewer.py         # Generate player data viewer (with salary)
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
- `data/processed/` — outputs: `player_rapm.parquet`, `player_rapm.csv`, `modeling_inputs_all.parquet/.csv`, `modeling_inputs_{season}.parquet`, `player_position_estimates_2022-23.parquet/.csv`, `player_position_estimates_2023-24.parquet/.csv`, `player_position_estimates_2024-25.parquet/.csv`, combined compatibility `player_position_estimates.parquet/.csv`, `defensive_archetypes_v2.parquet`, `defensive_archetypes_v2.csv`, `defensive_archetypes_v2_impact_report.csv`, `defensive_archetypes_v2_impact_report.txt`, validation report
- `data/tracking/` — tracking-derived JSONs

# Notes
- Inspect `src/*` scripts for CLI flags and optional args (season filters, caching).
- Tweak `SEASON_DECAY_WEIGHTS` and alpha grids in `src/modeling/model_rapm.py` to change pooling/regularization.

# Future Upgrade Ideas

## Central Archetype Tuning Knob File

Create a single file (e.g., archetype_coefficients.json) containing coefficients for each defensive and offensive archetype. Changing a coefficient in this file would directly control the prevalence/distribution of each archetype in the pipeline. Both compute_defensive_archetypes_v2.py and compute_player_archetypes.py would read from this file and apply the coefficients during role assignment. This enables rapid, unified, and transparent tuning of archetype distributions.

## Consolidated Player Profile File

After player evaluation is complete, merge all player data (bios, advanced stats, archetypes, RAPM, position, etc.) into a single consolidated file per season (e.g., player_profiles_2024-25.parquet). This file would serve as the authoritative, denormalized source for all downstream tools and viewers, enabling fast, reliable, and simple access to the complete player record for any season.

# Known Bugs / Issues

## Player Names such as Jokic and Doncic breaking name-based matching

## Standardized nicknames such as Herb Jones vs. Herbert Jones, Carlton Carrington vs. Bub Carrington, etc.