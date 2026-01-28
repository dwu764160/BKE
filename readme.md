
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

Fetch / ingest (order matters)

```bash
# 1. Setup NBA session headers (if needed)
python3 src/data_fetch/fetch_pbp/bootstrap_nba_session.py
python3 src/data_fetch/fetch_pbp/capture_nba_headers.py

# 2. Fetch historical and player/team metadata
python3 src/data_fetch/fetch_historical_data.py
python3 src/data_fetch/fetch_players.py
python3 src/data_fetch/fetch_teams.py
python3 src/data_fetch/fetch_profiles.py

# 3. Optionally fetch play-by-play (choose one):
#    - fast CDN fetch
python3 src/data_fetch/fetch_pbp/CDN_pbp_fetch.py
#    - or DOM fallback (Playwright required)
python3 src/data_fetch/fetch_pbp/fetch_play_by_play.py

# 4. Fetch official stats / tracking as available
python3 src/data_fetch/fetch_official_stats.py
python3 src/data_fetch/fetch_tracking_data.py
```

Normalize & derive features

```bash
# Normalize raw PBP to canonical rows
python3 src/data_normalize/run_normalization.py

# Infer lineups and derive possessions / features
python3 src/features/derive_lineups.py
python3 src/features/derive_possessions.py
python3 src/features/compute_rest_home_back2back.py
```

Compute metrics

```bash
# Optional cleaning step
python3 src/data_compute/compute_clean_possessions.py

# Compute RAPM / ORAPM / DRAPM (single and pooled)
python3 src/data_compute/compute_rapm.py

# Compute local & advanced metrics
python3 src/data_compute/compute_local_metrics.py
python3 src/data_compute/compute_advanced_metrics.py

# Produce player profiles / exports
python3 src/data_compute/compute_player_profiles.py
python3 src/utils/export_db_to_parquet.py
```

Validation & tests

```bash
# Run RAPM validation (produces data/processed/rapm_validation_report.json)
python3 tests/validate_rapm.py

# Run unit tests (if any)
pytest -q
```

Diagram

```bash
# Render flow diagram
dot -Tpng scheme_diagrams/flow_diagram_pre_possession.dot -o scheme_diagrams/flow_diagram_pre_possession.png
```

Data layout (locations used by scripts)
- `data/historical/` — raw + normalized PBP, possessions, caches
- `data/processed/` — outputs: `player_rapm.parquet`, `player_rapm.csv`, validation report
- `data/tracking/` — tracking-derived JSONs

Notes
- Inspect `src/*` scripts for CLI flags and optional args (season filters, caching).
- Tweak `SEASON_DECAY_WEIGHTS` and `alphas` in `src/data_compute/compute_rapm.py` to change pooling/regularization.
