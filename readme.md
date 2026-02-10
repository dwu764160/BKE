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
python3 src/data_fetch/fetch_pbp/bootstrap_nba_session.py
python3 src/data_fetch/fetch_pbp/capture_nba_headers.py

# 2. Fetch historical and player/team metadata
python3 src/data_fetch/fetch_historical_data.py
python3 src/data_fetch/fetch_players.py
python3 src/data_fetch/fetch_teams.py
python3 src/data_fetch/fetch_profiles.py
# Optionally fetch play-by-play (choose one):
python3 src/data_fetch/fetch_pbp/CDN_pbp_fetch.py
#    - or DOM fallback (Playwright required)
python3 src/data_fetch/fetch_pbp/fetch_play_by_play.py
python3 src/data_fetch/fetch_official_stats.py
python3 src/data_fetch/fetch_tracking_data.py
python3 src/data_fetch/fetch_box_scores_complete.py
python3 src/data_fetch/fetch_matchup_data.py
python3 src/data_fetch/fetch_shot_zones.py
```

## Derive / Normalize / Features
```bash
python3 src/data_fetch/derive_team_game_logs.py
python3 src/data_fetch/summarize_team_logs.py
python3 src/utils/export_db_to_parquet.py
python3 src/data_normalize/run_normalization.py
python3 src/features/derive_lineups.py
python3 src/features/derive_possessions.py
python3 src/features/compute_rest_home_back2back.py
```

## Compute / Metrics
```bash
python3 src/data_compute/compute_clean_possessions.py
python3 src/data_compute/compute_rapm.py
# xRAPM: Use compute_xrapm_improved.py (supersedes compute_xrapm.py)
python3 src/data_compute/compute_xrapm_improved.py
python3 src/data_compute/compute_local_metrics.py
python3 src/data_compute/compute_linear_metrics.py
python3 src/data_compute/compute_advanced_metrics.py
python3 src/data_compute/compute_player_profiles.py
python3 src/data_compute/compute_player_archetypes.py
# Defensive archetypes: Use compute_defensive_archetypes_v2.py (supersedes compute_defensive_archetypes.py)
python3 src/data_compute/compute_defensive_archetypes_v2.py
```

## Visualization / Export
```bash
python3 app/player_archetype_viewer.py
python3 app/player_data_viewer.py
python3 src/utils/export_db_to_parquet.py
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
- `data/historical/` — raw + normalized PBP, possessions, caches
- `data/processed/` — outputs: `player_rapm.parquet`, `player_rapm.csv`, validation report
- `data/tracking/` — tracking-derived JSONs

# Notes
- Inspect `src/*` scripts for CLI flags and optional args (season filters, caching).
- Tweak `SEASON_DECAY_WEIGHTS` and `alphas` in `src/data_compute/compute_rapm.py` to change pooling/regularization.
