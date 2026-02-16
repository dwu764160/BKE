#!/usr/bin/env bash
set -euo pipefail

# ===========================================================================
# reproduce_pipeline.sh — Full BKE data reproduction pipeline
#
# Run from workspace root:
#   bash scripts/reproduce_pipeline.sh [--skip-fetch] [--layer N]
#
# This script:
#   1. Creates data_temp_reprod/ with correct subdirectories
#   2. For each pipeline script:
#      a. Copies it to /tmp/bke_reprod/
#      b. Patches hardcoded DATA_DIR / OUTPUT_DIR paths via sed
#      c. Runs the patched version
#      d. Output goes to data_temp_reprod/
#   3. Runs comparison at the end
# ===========================================================================

WORKSPACE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WORKSPACE"

PYTHON="${WORKSPACE}/.venv/bin/python"
FRESH="data_temp_reprod"
LOG_DIR="${FRESH}/logs"
ORIGINAL="data"
TMP_SCRIPTS="/tmp/bke_reprod"
SKIP_FETCH=false
START_LAYER=0

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-fetch) SKIP_FETCH=true; shift ;;
        --layer) START_LAYER="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "============================================"
echo "BKE Data Reproduction Pipeline"
echo "Workspace: $WORKSPACE"
echo "Output:    $FRESH/"
echo "Skip fetch: $SKIP_FETCH"
echo "Start layer: $START_LAYER"
echo "Started:   $(date)"
echo "============================================"

# --- 0. Setup fresh output tree ---
mkdir -p "$FRESH"/{historical,processed,tracking/{2022-23,2023-24,2024-25},matchup,official_stats,features}
mkdir -p "$LOG_DIR"
mkdir -p "$TMP_SCRIPTS"

# --- Helper: patch and run a script ---
# Copies script to temp, applies sed replacements to redirect output paths,
# then runs the patched version.
run_patched() {
    local step_id="$1"
    local script_rel="$2"
    shift 2
    local extra_args=("$@")

    local script_name
    script_name="$(basename "$script_rel")"
    local patched="${TMP_SCRIPTS}/${step_id}_${script_name}"

    echo ""
    echo ">>> [$step_id] Patching & running: $script_rel"

    # Copy original script
    cp "$script_rel" "$patched"

    # Universal sed patches: replace ALL occurrences of "data/ and 'data/ paths
    # This catches every pattern: DATA_DIR = "data/...", Path("data/..."),
    # os.path.join("data/...", ...), hardcoded file paths, etc.
    sed -i "s|\"data/|\"${FRESH}/|g" "$patched"
    sed -i "s|'data/|'${FRESH}/|g" "$patched"

    # Also catch Path("data") / "subdir" style (no trailing slash)
    sed -i "s|\"data\"|\"${FRESH}\"|g" "$patched"

    # Instead of injecting sys.path into the script (which breaks when line 2
    # falls inside a docstring), use PYTHONPATH to make both `from src.xxx` and
    # sibling imports (e.g. `from compute_xrapm import`) work correctly.
    local script_src_dir
    script_src_dir="$(cd "$(dirname "$script_rel")" && pwd)"

    # Run the patched script
    echo "    Running patched script..."
    if PYTHONPATH="${WORKSPACE}:${script_src_dir}" $PYTHON "$patched" "${extra_args[@]}" 2>&1 | tee "$LOG_DIR/${step_id}.log"; then
        echo "<<< [$step_id] ✅ Done"
    else
        echo "<<< [$step_id] ❌ FAILED (exit $?)"
        echo "    See log: $LOG_DIR/${step_id}.log"
    fi
}

# ===========================================================================
# LAYER 0: FETCH (Copy raw data from original — no API calls needed)
# ===========================================================================
if [[ $START_LAYER -le 0 ]]; then
    echo ""
    echo "=== LAYER 0: COPY RAW FETCHED DATA ==="

    # --- Smart copy: skip if key files already exist in FRESH ---
    _key_file="$FRESH/historical/play_by_play_2024-25.parquet"
    if [[ -f "$_key_file" ]]; then
        echo "  ✅ Raw data already present in $FRESH/ — skipping copy phase"
    else
        echo "Copying raw fetched data from $ORIGINAL/ to $FRESH/..."

        # Copy raw PBP parquets
        cp "$ORIGINAL/historical/play_by_play_"*.parquet "$FRESH/historical/" 2>/dev/null || echo "  ⚠️ No play_by_play files"

        # Copy player game logs
        cp "$ORIGINAL/historical/player_game_logs_"*.parquet "$FRESH/historical/" 2>/dev/null || true
        cp "$ORIGINAL/historical/final_player_game_logs.parquet" "$FRESH/historical/" 2>/dev/null || true
        cp "$ORIGINAL/historical/complete_player_season_stats.parquet" "$FRESH/historical/" 2>/dev/null || true

        # Copy SQLite DB
        cp "$ORIGINAL/player_team_profiles.db" "$FRESH/player_team_profiles.db" 2>/dev/null || true

        # Copy official stats (fetched from API, deterministic)
        cp "$ORIGINAL/official_stats/"*.parquet "$FRESH/official_stats/" 2>/dev/null || true

        # Copy tracking data per season
        for season in 2022-23 2023-24 2024-25; do
            mkdir -p "$FRESH/tracking/$season"
            cp "$ORIGINAL/tracking/$season/"*.parquet "$FRESH/tracking/$season/" 2>/dev/null || true
        done

        # Copy matchup data
        cp "$ORIGINAL/matchup/"*.parquet "$FRESH/matchup/" 2>/dev/null || true

        # Copy features
        cp "$ORIGINAL/features/"*.parquet "$FRESH/features/" 2>/dev/null || true

        # Copy session files for any API calls
        cp "$ORIGINAL/nba_headers.json" "$FRESH/" 2>/dev/null || true
        cp "$ORIGINAL/nba_session.json" "$FRESH/" 2>/dev/null || true

        # Copy caches (skip pbp_cache — it's huge and not needed for recompute)
        # cp -r "$ORIGINAL/historical/pbp_cache" "$FRESH/historical/pbp_cache" 2>/dev/null || true
        mkdir -p "$FRESH/tracking_cache" "$FRESH/matchup_cache"
        cp "$ORIGINAL/tracking_cache/"*.json "$FRESH/tracking_cache/" 2>/dev/null || true
        cp "$ORIGINAL/matchup_cache/"*.json "$FRESH/matchup_cache/" 2>/dev/null || true
    fi

    # Run derive scripts that produce team_game_logs and summaries
    run_patched F11 src/data_fetch/derive_team_game_logs.py
    run_patched F12 src/data_fetch/summarize_team_logs.py
    run_patched F13 src/utils/export_db_to_parquet.py

    echo ""
    echo "=== LAYER 0 COMPLETE ==="
fi

# ===========================================================================
# LAYER 1: NORMALIZE + FEATURES
# ===========================================================================
if [[ $START_LAYER -le 1 ]]; then
    echo ""
    echo "=== LAYER 1: NORMALIZE + FEATURES ==="

    run_patched N1 src/data_normalize/run_normalization.py
    run_patched N2 src/features/derive_lineups.py
    run_patched N3 src/features/derive_possessions.py
    run_patched N4 src/data_compute/compute_clean_possessions.py
    run_patched N5 src/features/compute_rest_home_back2back.py

    echo ""
    echo "=== LAYER 1 COMPLETE ==="
fi

# ===========================================================================
# LAYER 2: COMPUTE
# ===========================================================================
if [[ $START_LAYER -le 2 ]]; then
    echo ""
    echo "=== LAYER 2: COMPUTE ==="

    run_patched C1 src/data_compute/compute_player_profiles.py
    run_patched C2 src/data_compute/compute_advanced_metrics.py
    run_patched C3 src/data_compute/compute_linear_metrics.py
    run_patched C4 src/data_compute/compute_local_metrics.py
    run_patched C5 src/modeling/model_rapm.py
    run_patched C6 src/modeling/ingest_darko.py
    # C8 (archetypes) and C9 (defensive archetypes) excluded per user request
    # run_patched C8 src/data_compute/compute_player_archetypes.py
    # run_patched C9 src/data_compute/compute_defensive_archetypes_v2.py

    echo ""
    echo "=== LAYER 2 COMPLETE ==="
fi

# ===========================================================================
# LAYER 3: COMPARE
# ===========================================================================
echo ""
echo "=== LAYER 3: COMPARISON (old vs new) ==="
if [[ -f scripts/compare_data_batches.py ]]; then
    $PYTHON scripts/compare_data_batches.py "$ORIGINAL" "$FRESH" 2>&1 | tee "$LOG_DIR/comparison.log"
else
    echo "⚠️ scripts/compare_data_batches.py not found — skipping comparison"
fi

echo ""
echo "============================================"
echo "Pipeline complete: $(date)"
echo "Logs: $LOG_DIR/"
echo "Output: $FRESH/"
echo "============================================"
