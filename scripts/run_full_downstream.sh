#!/usr/bin/env bash
# scripts/run_full_downstream.sh
# =============================================================================
# Full downstream compute pipeline — runs after all data fetches are done.
# Produces: normalize → features → RAPM → archetypes → BKE layers → aggregate
#
# Skips: team_game_logs rebuild (has known quality-check failure on 2022-23),
#        DARKO (not available), viewer apps (no GUI).
#
# Usage:
#   bash scripts/run_full_downstream.sh 2>&1 | tee logs/downstream.log
# =============================================================================

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

mkdir -p logs
LOG="$REPO/logs/downstream.log"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }
run() {
    local label="$1"; shift
    log "  >>> $label"
    if python3 "$@" >> "$LOG" 2>&1; then
        log "  ✓ $label"
    else
        log "  WARN: $label exited non-zero — continuing"
    fi
}

log "============================================================"
log "Full Downstream Pipeline Start"
log "============================================================"

# ---- Normalize & Features ----
run "summarize_team_logs"        src/data_fetch/summarize_team_logs.py
run "export_db_to_parquet"       src/utils/export_db_to_parquet.py
run "run_normalization"          src/data_normalize/run_normalization.py
run "normalize_darko (optional)" src/data_normalize/normalize_darko.py
run "derive_lineups"             src/features/derive_lineups.py
run "derive_possessions"         src/features/derive_possessions.py
run "compute_rest_home_b2b"      src/features/compute_rest_home_back2back.py

# ---- Compute ----
run "compute_clean_possessions"  src/data_compute/compute_clean_possessions.py
run "model_rapm"                 src/modeling/model_rapm.py
run "ingest_darko (optional)"    src/modeling/ingest_darko.py
run "compute_local_metrics"      src/data_compute/compute_local_metrics.py
run "compute_linear_metrics"     src/data_compute/compute_linear_metrics.py
run "compute_advanced_metrics"   src/data_compute/compute_advanced_metrics.py
run "compute_player_profiles"    src/data_compute/compute_player_profiles.py
run "compute_player_archetypes"  src/data_compute/compute_player_archetypes.py
run "compute_def_archetypes_v2"  src/data_compute/compute_defensive_archetypes_v2.py

# ---- BKE Scoring Chain ----
run "construct_bke_scores_v27"   src/modeling/construct_bke_scores_v27.py
run "layer1_portable_talent"     src/modeling/layer1_portable_talent.py
run "layer2_role_utilization"    src/modeling/layer2_role_utilization.py
run "layer3_archetype_elevation" src/modeling/layer3_archetype_elevation.py
run "layer4_scheme_amplification" src/modeling/layer4_scheme_amplification.py
run "decomposition_engine"       src/modeling/decomposition_engine.py
run "dbke_v30_defense_shrinkage" src/modeling/dbke_v30_defense_shrinkage.py

# ---- PTS ----
# Baseline: PTS v3.2 (input to v4.0 components; must run first to cover all seasons)
run "build_pts_v32"              scripts/build_pts_v32.py
# Component A: Bayesian multi-season smoothing
run "build_pts_v40_a"            scripts/pts_v40_multiseason.py \
    --pts-v32 data/processed/bke/pts_v32.parquet \
    --decomp  data/processed/bke/bke_v28_decomposition.parquet \
    --output  data/processed/bke/pts_v40_a.parquet
# Component C: Defense redesign (matchup-weighted)
run "build_pts_v40_c"            scripts/pts_v40_defense.py \
    --pts-v32  data/processed/bke/pts_v32.parquet \
    --decomp   data/processed/bke/bke_v28_decomposition.parquet \
    --def-arch data/processed/defensive_archetypes_v2.parquet \
    --pbp-dir  data/ \
    --output   data/processed/bke/pts_v40_c.parquet
# Composite: 50% A + 50% C
run "build_pts_v40"              scripts/build_pts_v40.py \
    --pts-a data/processed/bke/pts_v40_a.parquet \
    --pts-c data/processed/bke/pts_v40_c.parquet \
    --output data/processed/bke/pts_v40.parquet

# ---- Aggregate ----
run "build_profile_aggregate"    src/profile_aggregate/build_profile_aggregate.py

# ---- Team projections: rebuild all seasons in consistent pts/100 units ----
# Must run after profile aggregate so BKE scores are current.
run "build_all_season_projections" scripts/build_all_season_projections.py

# ---- Rest/B2B features (all seasons, required for Step 5 GBDT) ----
run "build_rest_features"        scripts/build_rest_features.py

# ---- YTD blended ratings (all seasons with projected features: 2018-26) ----
run "build_ytd_team_ratings"     scripts/build_ytd_team_ratings.py

# ---- Forecast + CLV ----
run "forecast_2025_26_games"     scripts/forecast_2025_26_games.py

log "============================================================"
log "Downstream Pipeline Complete"
log "Check logs/downstream.log for any WARN lines."
log "============================================================"
