#!/usr/bin/env bash
set -euo pipefail

# compute_data.sh — Run normalization, features, metrics, and export scripts for BKE pipeline
# Usage: bash scripts/compute_data.sh

# Derive / Normalize / Features
echo "[COMPUTE] derive_team_game_logs.py"
python3 src/data_fetch/derive_team_game_logs.py
echo "[COMPUTE] summarize_team_logs.py"
python3 src/data_fetch/summarize_team_logs.py
echo "[COMPUTE] export_db_to_parquet.py"
python3 src/utils/export_db_to_parquet.py
echo "[COMPUTE] run_normalization.py"
python3 src/data_normalize/run_normalization.py
echo "[COMPUTE] derive_lineups.py"
python3 src/features/derive_lineups.py
echo "[COMPUTE] derive_possessions.py"
python3 src/features/derive_possessions.py
echo "[COMPUTE] compute_rest_home_back2back.py"
python3 src/features/compute_rest_home_back2back.py

# Compute / Metrics
echo "[COMPUTE] compute_clean_possessions.py"
python3 src/data_compute/compute_clean_possessions.py
echo "[COMPUTE] compute_rapm.py"
python3 src/data_compute/compute_rapm.py
# xRAPM: Use compute_xrapm_improved.py (supersedes compute_xrapm.py)
echo "[COMPUTE] compute_xrapm_improved.py"
python3 src/data_compute/compute_xrapm_improved.py
echo "[COMPUTE] compute_local_metrics.py"
python3 src/data_compute/compute_local_metrics.py
echo "[COMPUTE] compute_linear_metrics.py"
python3 src/data_compute/compute_linear_metrics.py
echo "[COMPUTE] compute_advanced_metrics.py"
python3 src/data_compute/compute_advanced_metrics.py
echo "[COMPUTE] compute_player_profiles.py"
python3 src/data_compute/compute_player_profiles.py
echo "[COMPUTE] compute_player_archetypes.py"
python3 src/data_compute/compute_player_archetypes.py
# Defensive archetypes: Use compute_defensive_archetypes_v2.py (supersedes compute_defensive_archetypes.py)
echo "[COMPUTE] compute_defensive_archetypes_v2.py"
python3 src/data_compute/compute_defensive_archetypes_v2.py

# Visualization / Export
echo "[COMPUTE] player_archetypes_viewer.py"
python3 app/player_archetypes_viewer.py
echo "[COMPUTE] player_data_viewer.py"
python3 app/player_data_viewer.py
echo "[COMPUTE] export_db_to_parquet.py"
python3 src/utils/export_db_to_parquet.py
