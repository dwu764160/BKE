#!/usr/bin/env bash
# scripts/pipeline_2025_26.sh
# =============================================================================
# Full 2025-26 data fetch + compute pipeline.
#
# Run order matters:
#   1. Team game logs   → parquet updated, CDN PBP can read game IDs
#   2. CDN PBP          → play_by_play_2025-26.parquet (fast, no rate limits)
#   3. Box scores       → needs 2025-26 in SEASONS (patched before this step)
#   4. Official stats   → same SEASONS dependency
#   5. Compute pipeline → normalize → features → possessions → RAPM → archetypes
#
# Time estimate:
#   Step 1  team game logs       ~2  min
#   Step 2  CDN PBP (1,230 gms)  ~60-90 min
#   Step 3  box scores            ~90-120 min (rate-limited NBA Stats API)
#   Step 4  official stats        ~30-60 min
#   Step 5  compute pipeline      ~3-5 hr
#   TOTAL                         ~6-9 hr
#
# Usage:
#   bash scripts/pipeline_2025_26.sh 2>&1 | tee logs/pipeline_2025_26.log
# =============================================================================

set -uo pipefail   # no -e: allow individual steps to fail; errors are logged

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

LOG_DIR="$REPO/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/pipeline_2025_26.log"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

log() {
    echo "[$(ts)] $*" | tee -a "$LOG"
}

fail() {
    log "ERROR: $*"
    log "Pipeline aborted. Check $LOG for details."
    exit 1
}

log "============================================================"
log "2025-26 Full Pipeline Start"
log "============================================================"

# -------------------------------------------------------------------------
# STEP 1 — Team game logs (2-3 min)
# Adds 2025-26 rows to data/historical/team_game_logs.parquet.
# Required by CDN PBP (game IDs) and Step 4 YTD blending (margins).
# -------------------------------------------------------------------------
log "[STEP 1/5] Fetching 2025-26 team game logs..."
if python3 scripts/fetch_2025_26_game_logs.py >> "$LOG" 2>&1; then
    log "[STEP 1/5] ✓ Team game logs updated"
else
    fail "fetch_2025_26_game_logs.py failed"
fi

# -------------------------------------------------------------------------
# STEP 2 — CDN PBP (60-90 min)
# Reads game IDs from the updated team_game_logs.parquet.
# Outputs: data/historical/pbp_cache/pbp_*.parquet + play_by_play_2025-26.parquet
# -------------------------------------------------------------------------
log "[STEP 2/5] Fetching 2025-26 PBP via CDN (~60-90 min)..."
if python3 src/data_fetch/fetch_pbp/CDN_pbp_fetch.py --seasons 2025-26 >> "$LOG" 2>&1; then
    log "[STEP 2/5] ✓ CDN PBP fetch complete"
else
    log "[STEP 2/5] WARN: CDN PBP fetch had errors (may be partial) — continuing"
fi

# -------------------------------------------------------------------------
# STEP 3 — Add 2025-26 to model_config.py SEASONS
# Done here (after data is fetched) so box_scores / official_stats include it.
# -------------------------------------------------------------------------
log "[STEP 3/5] Patching model_config.py SEASONS to include 2025-26..."
SEASONS_LINE='    "2017-18", "2018-19", "2019-20", "2020-21", "2021-22",'
NEW_SEASONS='    "2017-18", "2018-19", "2019-20", "2020-21", "2021-22",'
CONFIG="$REPO/src/modeling/model_config.py"

# Idempotent: only patch if 2025-26 not already present
if grep -q '"2025-26"' "$CONFIG"; then
    log "[STEP 3/5] 2025-26 already in SEASONS — skipping patch"
else
    python3 - <<'PYEOF'
import re, pathlib, sys

config = pathlib.Path("src/modeling/model_config.py")
text = config.read_text()

old = '    "2022-23", "2023-24", "2024-25",\n]'
new = '    "2022-23", "2023-24", "2024-25", "2025-26",\n]'

if old not in text:
    print("ERROR: Could not find SEASONS closing line to patch", file=sys.stderr)
    sys.exit(1)

config.write_text(text.replace(old, new))
print("Patched SEASONS to include 2025-26")
PYEOF
    if [ $? -eq 0 ]; then
        log "[STEP 3/5] ✓ model_config.py SEASONS updated"
    else
        fail "Failed to patch model_config.py SEASONS"
    fi
fi

# -------------------------------------------------------------------------
# STEP 4 — Box scores + official stats (120-180 min combined)
# Both import SEASONS from model_config; 2025-26 is now included.
# -------------------------------------------------------------------------
log "[STEP 4/5] Fetching box scores for 2025-26 (~90-120 min)..."
if python3 src/data_fetch/fetch_box_scores_complete.py >> "$LOG" 2>&1; then
    log "[STEP 4/5] ✓ Box scores complete"
else
    log "[STEP 4/5] WARN: Box score fetch had errors — continuing"
fi

log "[STEP 4/5] Fetching official advanced stats for 2025-26 (~30-60 min)..."
if python3 src/data_fetch/fetch_official_stats.py >> "$LOG" 2>&1; then
    log "[STEP 4/5] ✓ Official stats complete"
else
    log "[STEP 4/5] WARN: Official stats fetch had errors — continuing"
fi

log "[STEP 4/5] Fetching tracking data for 2025-26 (~20 min)..."
if python3 src/data_fetch/fetch_tracking_data.py >> "$LOG" 2>&1; then
    log "[STEP 4/5] ✓ Tracking data complete"
else
    log "[STEP 4/5] WARN: Tracking fetch had errors (data may be unavailable) — continuing"
fi

log "[STEP 4/5] Fetching shot zones for 2025-26 (~20 min)..."
if python3 src/data_fetch/fetch_shot_zones.py >> "$LOG" 2>&1; then
    log "[STEP 4/5] ✓ Shot zones complete"
else
    log "[STEP 4/5] WARN: Shot zones fetch had errors — continuing"
fi

# -------------------------------------------------------------------------
# STEP 5 — Compute pipeline (~3-5 hr)
# Full normalize → features → possessions → RAPM → archetypes chain.
# -------------------------------------------------------------------------
log "[STEP 5/5] Running compute pipeline (~3-5 hr)..."

run_compute() {
    local label="$1"; shift
    log "  [compute] $label"
    if python3 "$@" >> "$LOG" 2>&1; then
        log "  [compute] ✓ $label"
        return 0
    else
        log "  [compute] WARN: $label failed — continuing"
        return 1
    fi
}

run_compute "derive_team_game_logs.py"    src/data_fetch/derive_team_game_logs.py
run_compute "summarize_team_logs.py"      src/data_fetch/summarize_team_logs.py
run_compute "export_db_to_parquet.py"     src/utils/export_db_to_parquet.py
run_compute "run_normalization.py"        src/data_normalize/run_normalization.py
run_compute "derive_lineups.py"           src/features/derive_lineups.py
run_compute "derive_possessions.py"       src/features/derive_possessions.py
run_compute "compute_rest_home_back2back" src/features/compute_rest_home_back2back.py
run_compute "compute_clean_possessions"   src/data_compute/compute_clean_possessions.py
run_compute "model_rapm.py"               src/modeling/model_rapm.py
run_compute "compute_local_metrics.py"    src/data_compute/compute_local_metrics.py
run_compute "compute_linear_metrics.py"   src/data_compute/compute_linear_metrics.py
run_compute "compute_advanced_metrics.py" src/data_compute/compute_advanced_metrics.py
run_compute "compute_player_profiles.py"  src/data_compute/compute_player_profiles.py
run_compute "compute_player_archetypes.py" src/data_compute/compute_player_archetypes.py
run_compute "compute_defensive_archetypes_v2.py" src/data_compute/compute_defensive_archetypes_v2.py
run_compute "export_db_to_parquet.py (final)" src/utils/export_db_to_parquet.py

log "============================================================"
log "Pipeline complete. Check $LOG for any WARN lines."
log "Next: python3 scripts/build_ytd_team_ratings.py  (Step 4 YTD blending)"
log "      python3 scripts/forecast_2025_26_games.py   (re-run CLV with YTD)"
log "============================================================"
