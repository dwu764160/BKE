#!/usr/bin/env bash
# Rerun RAPM pipeline so 2025-26 data lands in player_profile_aggregate.parquet
# Run: bash scripts/rerun_2025_26_rapm.sh &> logs/rerun_2025_26.log
set -euo pipefail
cd "$(dirname "$0")/.."

LOG="logs/rerun_2025_26.log"
mkdir -p logs

ts() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run() {
    local label="$1"; shift
    ts ">>> $label"
    if python3 "$@" >> "$LOG" 2>&1; then
        ts "✓ $label"
    else
        ts "WARN: $label exited non-zero — continuing"
    fi
}

ts "============================================================"
ts "2025-26 RAPM + BKE Re-Run Start"
ts "============================================================"

# ---- Lineups (all seasons, ~18 min) ----
run "derive_lineups"             src/features/derive_lineups.py

# ---- Possessions (all seasons, ~20 min) ----
run "derive_possessions"         src/features/derive_possessions.py

# ---- RAPM (all seasons, ~60 min) ----
run "model_rapm"                 src/modeling/model_rapm.py

# ---- BKE layers ----
run "compute_local_metrics"      src/data_compute/compute_local_metrics.py
run "compute_linear_metrics"     src/data_compute/compute_linear_metrics.py
run "compute_advanced_metrics"   src/data_compute/compute_advanced_metrics.py
run "compute_player_profiles"    src/data_compute/compute_player_profiles.py
run "compute_player_archetypes"  src/data_compute/compute_player_archetypes.py
run "compute_def_archetypes_v2"  src/data_compute/compute_defensive_archetypes_v2.py
run "construct_bke_scores_v27"   src/modeling/construct_bke_scores_v27.py
run "layer1_portable_talent"     src/modeling/layer1_portable_talent.py
run "layer2_role_utilization"    src/modeling/layer2_role_utilization.py
run "layer3_archetype_elevation" src/modeling/layer3_archetype_elevation.py
run "layer4_scheme_amplification" src/modeling/layer4_scheme_amplification.py
run "decomposition_engine"       src/modeling/decomposition_engine.py
run "dbke_v30_defense_shrinkage" src/modeling/dbke_v30_defense_shrinkage.py
run "build_pts_v40"              scripts/build_pts_v40.py \
    --pts-a data/processed/bke/pts_v40_a.parquet \
    --pts-c data/processed/bke/pts_v40_c.parquet \
    --output data/processed/bke/pts_v40.parquet

# ---- Aggregate ----
run "build_profile_aggregate"    src/profile_aggregate/build_profile_aggregate.py

# ---- Team projections: rebuild all seasons in consistent pts/100 units ----
run "build_all_season_projections" scripts/build_all_season_projections.py

# ---- Rest/B2B features (all seasons) ----
run "build_rest_features"        scripts/build_rest_features.py

# ---- YTD blended ratings (all seasons with projected features: 2018-26) ----
run "build_ytd_team_ratings"     scripts/build_ytd_team_ratings.py

# ---- Forecast + CLV ----
run "forecast_2025_26_games"     scripts/forecast_2025_26_games.py

ts "============================================================"
ts "Re-Run Complete — check logs/rerun_2025_26.log for WARNs"
ts "============================================================"
