# Data Reproduction & Verification Plan

> **⚠ HISTORICAL PLANNING DOC — 2026-05-xx vintage**
> The comparison manifest in §4 uses old uppercase column names (`GAME_ID`, `TEAM_ID`, `PTS`, etc.)
> that are now obsolete. All parquets use canonical lowercase names since the parquet
> standardization (Sessions A-C, 2026-06-01/02). The workflow concepts in §1-3 and §5-9
> are still valid; update column references before using the §4 manifest.
> Current schema reference: `docs/reference/data_schemas.md`.

> **Original goal:** Re-run the entire BKE pipeline from raw fetch → computed metrics, without
> destroying any current data. Compare old vs new for completeness and correctness.
> Identify and fix any problems before moving to player evaluation.

---

## 0. Inventory — What We Have Today

### Fetch Scripts (Layer 0 — External Data)

| # | Script | Output Location | Description |
|---|--------|----------------|-------------|
| F1 | `src/data_fetch/fetch_pbp/CDN_pbp_fetch.py` | `data/historical/play_by_play_{season}.parquet` | Raw PBP events from NBA CDN (3 seasons) |
| F2 | `src/data_fetch/fetch_historical_data.py` | `data/historical/player_game_logs_{season}.parquet` | Player game logs + ID/name mapping |
| F3 | `src/data_fetch/fetch_players.py` | `data/player_team_profiles.db` (players table) | Player metadata → SQLite |
| F4 | `src/data_fetch/fetch_teams.py` | `data/player_team_profiles.db` (teams table) | Team metadata → SQLite |
| F5 | `src/data_fetch/fetch_profiles.py` | `data/player_team_profiles.db` (profiles) | Player bio (height, weight, etc) via NBA API + B-Ref scraping |
| F6 | `src/data_fetch/fetch_official_stats.py` | `data/official_stats/official_advanced_{season}.parquet` | Official NBA.com advanced stats (OFF_RTG, DEF_RTG, USG%, TS%, etc) |
| F7 | `src/data_fetch/fetch_tracking_data.py` | `data/tracking/{season}/tracking_*.parquet`, `synergy_*.parquet` | Tracking (drives, passing, etc) + Synergy playtypes (11 off, 7 def) |
| F8 | `src/data_fetch/fetch_shot_zones.py` | `data/tracking/{season}/shot_zones.parquet` | Shot zone data (AT_RIM, MIDRANGE, PAINT, etc) |
| F9 | `src/data_fetch/fetch_matchup_data.py` | `data/matchup/*.parquet` | Matchup data (versatility, difficulty) |
| F10 | `src/data_fetch/fetch_box_scores_complete.py` | `data/historical/complete_player_season_stats.parquet` | League-wide player base+advanced stats |
| F11 | `src/data_fetch/derive_team_game_logs.py` | `data/historical/team_game_logs.parquet` | Team game logs derived from PBP |
| F12 | `src/data_fetch/summarize_team_logs.py` | `data/historical/team_summaries.parquet` + `.csv` | Per-team-per-season summaries |
| F13 | `src/utils/export_db_to_parquet.py` | `data/historical/players.parquet`, `teams.parquet` | SQLite → parquet export |

### Normalize & Feature Scripts (Layer 1 — Derived Features)

| # | Script | Output Location | Description |
|---|--------|----------------|-------------|
| N1 | `src/data_normalize/run_normalization.py` | `data/historical/pbp_normalized_{season}.parquet` | Canonical PBP rows from raw data |
| N2 | `src/features/derive_lineups.py` | `data/historical/pbp_with_lineups_{season}.parquet` | PBP with inferred 5-man lineups |
| N3 | `src/features/derive_possessions.py` | `data/historical/possessions_{season}.parquet` | Grouped logical possessions |
| N4 | `src/data_compute/compute_clean_possessions.py` | `data/historical/possessions_clean_{season}.parquet` | Cleaned possessions (valid 5v5 only) |
| N5 | `src/features/compute_rest_home_back2back.py` | `data/historical/feature_schedule_context.parquet` | Schedule features (rest, home/away, B2B) |

### Compute Scripts (Layer 2 — Metrics & Models)

| # | Script | Output Location | Description |
|---|--------|----------------|-------------|
| C1 | `src/data_compute/compute_player_profiles.py` | `data/processed/player_profiles_advanced.parquet` | Box Score Plus + Four Factors profiles |
| C2 | `src/data_compute/compute_advanced_metrics.py` | `data/processed/metrics_teams.parquet`, `metrics_lineups.parquet` | Team ORTG/DRTG/NET, lineup stats |
| C3 | `src/data_compute/compute_linear_metrics.py` | `data/processed/metrics_linear.parquet` | Win Shares (OWS/DWS/WS), BPM, VORP |
| C4 | `src/data_compute/compute_local_metrics.py` | `data/advanced_local_metrics.parquet` | Box-score-derived advanced metrics |
| C5 | `src/modeling/model_rapm.py` | `data/processed/player_rapm.parquet` + `.csv` | RAPM / ORAPM / DRAPM (ridge regression) |
| C6 | `src/modeling/ingest_darko.py` | `data/processed/modeling_inputs_all.parquet` + `modeling_inputs_{season}.parquet` | DARKO + RAPM modeling input merge |
| C8 | `src/data_compute/compute_player_archetypes.py` | `data/processed/player_archetypes.parquet` + `.csv`, `archetype_embeddings.parquet` + `.csv` | Offensive archetypes v4.3 + embeddings |
| C9 | `src/data_compute/compute_defensive_archetypes_v2.py` | `data/processed/defensive_archetypes_v2.parquet` + `.csv` | Defensive archetypes (5 types) |

### Existing Validation Tests

| Test | Validates |
|------|-----------|
| `tests/validate_possessions.py` | Possession ORTG, pace, lineup completeness |
| `tests/validate_rapm.py` | RAPM distribution, external benchmarks, stability |
| `tests/validate_advanced_metrics.py` | WS/BPM/VORP vs B-Ref ground truth |
| `tests/validate_ws_broad.py` | Win Shares for broad set (7 players vs B-Ref) |
| `tests/validate_official_stats.py` | Official stats schema, ranges, player counts |
| `tests/validate_tracking_data.py` | Tracking file existence, CatchShoot proxy |
| `tests/validate_gamelogs.py` | Game log integrity |

### All Data Artifacts (~158 files)

```
data/
├── player_team_profiles.db          # SQLite (players, teams, fetch_cache)
├── advanced_local_metrics.parquet   # Local box-score metrics
├── nba_headers.json                 # Session config
├── nba_session.json                 # Session cookies
├── historical/                      # ~32 parquet + CSV files
│   ├── play_by_play_{season}.parquet         (3 files)
│   ├── pbp_normalized_{season}.parquet       (3 files)
│   ├── pbp_with_lineups_{season}.parquet     (3 files)
│   ├── possessions_{season}.parquet          (3 files)
│   ├── possessions_clean_{season}.parquet    (3 files)
│   ├── player_game_logs_{season}.parquet     (2 files)
│   ├── complete_player_season_stats.parquet
│   ├── final_player_game_logs.parquet
│   ├── team_game_logs.parquet
│   ├── team_summaries.parquet + .csv
│   ├── team_game_details.parquet + .csv
│   ├── feature_schedule_context.parquet
│   ├── players.parquet, teams.parquet
│   └── pbp_cache/ (raw JSON cache)
├── processed/                       # ~22 files
│   ├── player_profiles_advanced.parquet
│   ├── metrics_linear.parquet
│   ├── metrics_teams.parquet, metrics_lineups.parquet
│   ├── metrics_win_shares.parquet
│   ├── player_rapm.parquet + .csv
│   ├── player_xrapm.parquet + .csv
│   ├── player_xrapm_v2.parquet + .csv
│   ├── player_xrapm_improved_{season}.csv    (2 files)
│   ├── player_archetypes.parquet + .csv
│   ├── archetype_embeddings.parquet + .csv
│   ├── defensive_archetypes_v2.parquet + .csv
│   ├── defensive_archetypes.parquet + .csv    (v1, legacy)
│   └── rapm_validation_report.json
├── tracking/{season}/               # ~29 parquet per season (87 total)
├── tracking_cache/                  # Raw JSON API responses
├── matchup/                         # 4 parquet files
├── matchup_cache/                   # Raw JSON matchup responses
├── official_stats/                  # 3 parquet files (per season)
└── features/                        # 2 parquet files
```

---

## 1. Backup Strategy — Protect Current Data

### 1a. Create a timestamped snapshot

```bash
# From workspace root
BACKUP_DIR="data_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Copy everything except raw caches (they can always be re-fetched)
rsync -av --progress \
  --exclude='pbp_cache/' \
  --exclude='tracking_cache/' \
  --exclude='matchup_cache/' \
  --exclude='__pycache__/' \
  data/ "$BACKUP_DIR/"

echo "✅ Backup saved to $BACKUP_DIR"
```

This preserves:
- All parquet/CSV outputs (the stuff we care about comparing)
- SQLite database
- Official stats, matchup, and feature files

It skips raw JSON caches (~100s of MB) since those are API response caches
that get re-created on fetch. The caches themselves are also still in `data/`
and won't be touched by the re-run (see §2).

### 1b. Verify the backup

```bash
# Quick sanity check
echo "=== Backup ==="
find "$BACKUP_DIR" -name '*.parquet' | wc -l
echo "=== Original ==="
find data/ -name '*.parquet' -not -path '*/cache/*' | wc -l
```

Both counts should match.

### 1c. Optional — external copy

If you want an off-machine copy:
```bash
# Compress for portability
tar czf "${BACKUP_DIR}.tar.gz" "$BACKUP_DIR"
# Move to external drive, cloud, etc.
```

---

## 2. Re-Run Strategy — New Data Goes to `data_fresh/`

**Key principle:** We do NOT overwrite `data/`. Instead, we create a parallel
output tree `data_fresh/` and point scripts at it. After comparison, we either
promote `data_fresh/` → `data/` or keep the old data.

### 2a. Environment Variables

Create a small wrapper that redirects output dirs:
```bash
export BKE_DATA_DIR="data_fresh"
export BKE_HISTORICAL_DIR="data_fresh/historical"
export BKE_PROCESSED_DIR="data_fresh/processed"
export BKE_TRACKING_DIR="data_fresh/tracking"
export BKE_MATCHUP_DIR="data_fresh/matchup"
export BKE_OFFICIAL_DIR="data_fresh/official_stats"
export BKE_FEATURES_DIR="data_fresh/features"
```

**However**, most scripts have hardcoded paths. So the practical approach
is a thin wrapper script (see §3) that:
1. Copies the script to a temp location
2. Patches `DATA_DIR`/`OUTPUT_DIR` paths via sed
3. Runs the patched version
4. Outputs to `data_fresh/`

Alternatively (simpler): **symlink swap**:
```bash
mv data data_original
mkdir -p data_fresh
ln -s data_original data          # scripts read from original
# Then after each compute step, move output files to data_fresh/
```

**Recommended approach:** Use a **runner script** (§3) that handles all of this.

---

## 3. Master Re-Run Script — `scripts/reproduce_pipeline.sh`

This is the single entry point. It:
1. Creates `data_fresh/` with correct subdirectories
2. Copies raw inputs (PBP, caches) from `data/` so fetchers can skip re-downloading
3. Runs each layer in order, outputting to `data_fresh/`
4. Runs the comparison script at the end

### Execution Order (respects dependency chain)

```
┌─────────────────────────────────────────────────────────────────────┐
│ LAYER 0: FETCH (External API calls — can skip if caches exist)     │
│                                                                     │
│  F1  CDN PBP fetch          → play_by_play_{season}.parquet        │
│  F2  Historical game logs   → player_game_logs_{season}.parquet    │
│  F3  Players metadata       → player_team_profiles.db              │
│  F4  Teams metadata         → player_team_profiles.db              │
│  F5  Player profiles (bio)  → player_team_profiles.db              │
│  F6  Official advanced      → official_advanced_{season}.parquet   │
│  F7  Tracking + Synergy     → tracking/{season}/*.parquet          │
│  F8  Shot zones             → tracking/{season}/shot_zones.parquet │
│  F9  Matchup data           → matchup/*.parquet                    │
│  F10 Complete box scores    → complete_player_season_stats.parquet  │
│  F11 Derive team game logs  → team_game_logs.parquet               │
│  F12 Summarize team logs    → team_summaries.parquet               │
│  F13 Export DB to parquet   → players.parquet, teams.parquet       │
└─────────────┬───────────────────────────────────────────────────────┘
              │
┌─────────────▼───────────────────────────────────────────────────────┐
│ LAYER 1: NORMALIZE + FEATURES                                       │
│                                                                     │
│  N1  Normalize PBP          → pbp_normalized_{season}.parquet      │
│  N2  Derive lineups         → pbp_with_lineups_{season}.parquet    │
│  N3  Derive possessions     → possessions_{season}.parquet         │
│  N4  Clean possessions      → possessions_clean_{season}.parquet   │
│  N5  Schedule features      → feature_schedule_context.parquet     │
└─────────────┬───────────────────────────────────────────────────────┘
              │
┌─────────────▼───────────────────────────────────────────────────────┐
│ LAYER 2: COMPUTE (Metrics, Models, Archetypes)                      │
│                                                                     │
│  C1  Player profiles        → player_profiles_advanced.parquet     │
│  C2  Advanced metrics       → metrics_teams, metrics_lineups       │
│  C3  Linear metrics (WS)    → metrics_linear.parquet               │
│  C4  Local metrics          → advanced_local_metrics.parquet       │
│  C5  RAPM                   → player_rapm.parquet                  │
│  C6  xRAPM                  → player_xrapm.parquet                 │
│  C7  xRAPM improved (v2)   → player_xrapm_v2.parquet              │
│  C8  Offensive archetypes   → player_archetypes, embeddings        │
│  C9  Defensive archetypes   → defensive_archetypes_v2              │
└─────────────────────────────────────────────────────────────────────┘
```

### Script Template

```bash
#!/usr/bin/env bash
set -euo pipefail

# ===========================================================================
# reproduce_pipeline.sh — Full BKE data reproduction
# Run from workspace root: bash scripts/reproduce_pipeline.sh
# ===========================================================================

WORKSPACE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WORKSPACE"

PYTHON=".venv/bin/python"
FRESH="data_fresh"
LOG_DIR="$FRESH/logs"
ORIGINAL="data"

echo "============================================"
echo "BKE Data Reproduction Pipeline"
echo "Workspace: $WORKSPACE"
echo "Output:    $FRESH/"
echo "Started:   $(date)"
echo "============================================"

# --- 0. Setup fresh output tree ---
mkdir -p "$FRESH"/{historical,processed,tracking/{2022-23,2023-24,2024-25},matchup,official_stats,features}
mkdir -p "$LOG_DIR"

# --- 0b. Copy raw inputs (caches) so fetchers skip API calls ---
echo "Copying caches from original data..."
cp -r "$ORIGINAL/pbp_cache/"      "$FRESH/historical/pbp_cache/" 2>/dev/null || true
cp -r "$ORIGINAL/tracking_cache/" "$FRESH/tracking_cache/"       2>/dev/null || true
cp -r "$ORIGINAL/matchup_cache/"  "$FRESH/matchup_cache/"        2>/dev/null || true
cp    "$ORIGINAL/nba_headers.json" "$FRESH/" 2>/dev/null || true
cp    "$ORIGINAL/nba_session.json" "$FRESH/" 2>/dev/null || true

# Helper: run a script with DATA_DIR patched
run_step() {
    local step_id="$1"
    local script="$2"
    shift 2
    echo ""
    echo ">>> [$step_id] Running: $script $*"
    $PYTHON "$script" "$@" 2>&1 | tee "$LOG_DIR/${step_id}.log"
    echo "<<< [$step_id] Done (exit $?)"
}

# ===========================================================================
# LAYER 0: FETCH
# ===========================================================================
# NOTE: Most fetchers use caches. If caches exist, they skip the API call.
# For a TRUE re-fetch, delete the relevant *_cache/ dir first.
# For re-computation only, skip Layer 0 entirely and just copy the raw
# fetched parquets from data/ → data_fresh/.

echo ""
echo "=== LAYER 0: FETCH (skip if using cached data) ==="

# Option A: COPY raw fetched data from original (fast, no API calls)
echo "Copying raw fetched data from original..."
cp "$ORIGINAL/historical/play_by_play_"*.parquet "$FRESH/historical/" 2>/dev/null || true
cp "$ORIGINAL/historical/player_game_logs_"*.parquet "$FRESH/historical/" 2>/dev/null || true
cp "$ORIGINAL/historical/final_player_game_logs.parquet" "$FRESH/historical/" 2>/dev/null || true
cp "$ORIGINAL/historical/complete_player_season_stats.parquet" "$FRESH/historical/" 2>/dev/null || true
cp "$ORIGINAL/player_team_profiles.db" "$FRESH/player_team_profiles.db" 2>/dev/null || true
cp "$ORIGINAL/official_stats/"*.parquet "$FRESH/official_stats/" 2>/dev/null || true
cp -r "$ORIGINAL/tracking/" "$FRESH/tracking_copy_raw/" 2>/dev/null || true
# Copy tracking parquets into fresh tracking dirs
for season in 2022-23 2023-24 2024-25; do
    cp "$ORIGINAL/tracking/$season/"*.parquet "$FRESH/tracking/$season/" 2>/dev/null || true
done
cp "$ORIGINAL/matchup/"*.parquet "$FRESH/matchup/" 2>/dev/null || true
cp "$ORIGINAL/historical/players.parquet" "$FRESH/historical/" 2>/dev/null || true
cp "$ORIGINAL/historical/teams.parquet"   "$FRESH/historical/" 2>/dev/null || true

# Option B: UNCOMMENT to re-fetch from API (slow, requires session cookies)
# run_step F1  src/data_fetch/fetch_pbp/CDN_pbp_fetch.py
# run_step F2  src/data_fetch/fetch_historical_data.py
# run_step F3  src/data_fetch/fetch_players.py
# run_step F4  src/data_fetch/fetch_teams.py
# run_step F5  src/data_fetch/fetch_profiles.py
# run_step F6  src/data_fetch/fetch_official_stats.py
# run_step F7  src/data_fetch/fetch_tracking_data.py
# run_step F8  src/data_fetch/fetch_shot_zones.py
# run_step F9  src/data_fetch/fetch_matchup_data.py
# run_step F10 src/data_fetch/fetch_box_scores_complete.py
# run_step F11 src/data_fetch/derive_team_game_logs.py
# run_step F12 src/data_fetch/summarize_team_logs.py
# run_step F13 src/utils/export_db_to_parquet.py

# ===========================================================================
# LAYER 1: NORMALIZE + FEATURES
# ===========================================================================
echo ""
echo "=== LAYER 1: NORMALIZE + FEATURES ==="

run_step N1  src/data_normalize/run_normalization.py
run_step N2  src/features/derive_lineups.py
run_step N3  src/features/derive_possessions.py
run_step N4  src/data_compute/compute_clean_possessions.py
run_step N5  src/features/compute_rest_home_back2back.py
run_step F11 src/data_fetch/derive_team_game_logs.py
run_step F12 src/data_fetch/summarize_team_logs.py
run_step F13 src/utils/export_db_to_parquet.py

# ===========================================================================
# LAYER 2: COMPUTE
# ===========================================================================
echo ""
echo "=== LAYER 2: COMPUTE ==="

run_step C1  src/data_compute/compute_player_profiles.py
run_step C2  src/data_compute/compute_advanced_metrics.py
run_step C3  src/data_compute/compute_linear_metrics.py
run_step C4  src/data_compute/compute_local_metrics.py
run_step C5  src/modeling/model_rapm.py
run_step C6  src/modeling/ingest_darko.py
run_step C8  src/data_compute/compute_player_archetypes.py
run_step C9  src/data_compute/compute_defensive_archetypes_v2.py

# ===========================================================================
# LAYER 3: VALIDATE + COMPARE
# ===========================================================================
echo ""
echo "=== LAYER 3: VALIDATION ==="

run_step V1  tests/validate_possessions.py
run_step V2  tests/validate_rapm.py
run_step V3  tests/validate_advanced_metrics.py
run_step V4  tests/validate_ws_broad.py
run_step V5  tests/validate_official_stats.py
run_step V6  tests/validate_tracking_data.py

echo ""
echo "=== LAYER 4: COMPARISON (old vs new) ==="
$PYTHON scripts/compare_data_batches.py "$ORIGINAL" "$FRESH" 2>&1 | tee "$LOG_DIR/comparison.log"

echo ""
echo "============================================"
echo "Pipeline complete: $(date)"
echo "Logs: $LOG_DIR/"
echo "Comparison: $LOG_DIR/comparison.log"
echo "============================================"
```

**Important caveat:** The scripts above have hardcoded `DATA_DIR` paths pointing
to `data/`. For the `data_fresh/` approach to work, you need ONE of:
1. **Symlink swap** before running (recommended):
   ```bash
   mv data data_original
   ln -s data_fresh data
   # run pipeline
   # afterward: rm data && mv data_original data
   ```
2. **Parameterize scripts** to accept `--data-dir` (future improvement)
3. **Copy raw inputs** into `data_fresh/`, then rename `data_fresh/` → `data/`
   temporarily, run, rename back

---

## 4. Comparison Script — `scripts/compare_data_batches.py`

This is the most critical piece. It compares `data/` (old) vs `data_fresh/` (new).

### What It Checks

#### A. Completeness Checks
- File existence: every file in old exists in new (and vice versa)
- Row counts: each parquet should have same or more rows
- Column schema: same columns, same dtypes
- Season coverage: all 3 seasons present where expected
- Player coverage: no players dropped, no ghosts added

#### B. Correctness Checks
- **Numeric drift**: For every numeric column, compute:
  - Mean absolute difference (MAD)
  - Max absolute difference
  - Pearson correlation (should be ≥ 0.999 for deterministic pipelines)
  - Percent of values changed by > 1%
- **Categorical stability**: For label columns (archetype, subtype):
  - Count of changed labels
  - Confusion matrix of old vs new
  - List of players whose labels changed
- **Key player spot checks**: Hard-coded reference players and their expected
  values (LeBron, Curry, Jokic, etc.) — must match within tolerance
- **Distribution checks**: KS-test between old and new distributions per column
- **Aggregate checks**: Season-level totals (e.g., league total WS, avg RAPM)

### Files to Compare

```python
COMPARISON_MANIFEST = {
    # --- Layer 1 outputs ---
    "historical/pbp_normalized_{season}.parquet": {
        "seasons": ["2022-23", "2023-24", "2024-25"],
        "key_cols": ["game_id", "event_type", "player_id"],
        "check": "row_count_and_schema",
    },
    "historical/pbp_with_lineups_{season}.parquet": {
        "seasons": ["2022-23", "2023-24", "2024-25"],
        "key_cols": ["game_id", "off_lineup", "def_lineup"],
        "check": "row_count_and_schema",
    },
    "historical/possessions_clean_{season}.parquet": {
        "seasons": ["2022-23", "2023-24", "2024-25"],
        "key_cols": ["game_id", "off_team_id", "points"],
        "check": "row_count_and_numeric_drift",
    },
    "historical/team_game_logs.parquet": {
        "key_cols": ["GAME_ID", "TEAM_ID", "PTS"],
        "check": "exact_match",
    },
    "historical/team_summaries.parquet": {
        "key_cols": ["TEAM_ID", "SEASON"],
        "check": "numeric_drift",
    },
    # --- Layer 2 outputs ---
    "processed/player_profiles_advanced.parquet": {
        "join_on": ["player_id", "season"],
        "numeric_cols": ["PTS", "AST", "REB", "STL", "BLK", "TOV", "MIN", "GP",
                         "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
                         "TS_PCT", "USG_PCT", "ORB", "DRB"],
        "tolerance": 0.01,
        "check": "full_numeric",
    },
    "processed/metrics_linear.parquet": {
        "join_on": ["player_id", "season"],
        "numeric_cols": ["WS", "OWS", "DWS", "BPM", "VORP"],
        "tolerance": 0.05,
        "check": "full_numeric",
        "ground_truth": True,   # compare vs B-Ref
    },
    "processed/metrics_teams.parquet": {
        "join_on": ["team_id", "season"],
        "numeric_cols": ["ORTG", "DRTG", "NET_RTG"],
        "tolerance": 0.5,
        "check": "full_numeric",
    },
    "processed/player_rapm.parquet": {
        "join_on": ["player_id"],
        "numeric_cols": ["RAPM", "ORAPM", "DRAPM"],
        "tolerance": 0.1,
        "check": "full_numeric",
    },
    "processed/player_xrapm.parquet": {
        "join_on": ["player_id"],
        "numeric_cols": ["xRAPM", "ORAPM", "DRAPM"],
        "tolerance": 0.1,
        "check": "full_numeric",
    },
    "processed/player_xrapm_v2.parquet": {
        "join_on": ["player_id"],
        "numeric_cols": ["xRAPM", "ORAPM", "DRAPM"],
        "tolerance": 0.1,
        "check": "full_numeric",
    },
    "processed/player_archetypes.parquet": {
        "join_on": ["PLAYER_ID", "SEASON"],
        "label_cols": ["primary_archetype", "secondary_archetype"],
        "numeric_cols": ["role_confidence", "role_effectiveness"],
        "tolerance": 0.02,
        "check": "labels_and_numeric",
    },
    "processed/archetype_embeddings.parquet": {
        "join_on": ["PLAYER_ID", "SEASON"],
        "numeric_cols": ["emb_ball_dominant_creator", "emb_all_around_scorer",
                         "emb_ballhandler", "emb_interior_scorer",
                         "emb_perimeter_scorer", "emb_connector",
                         "emb_pnr_rolling_big", "emb_pnr_popping_big",
                         "emb_off_ball_finisher", "emb_off_ball_movement_shooter",
                         "emb_off_ball_stationary_shooter",
                         "emb_entropy", "emb_dominance"],
        "tolerance": 0.005,
        "check": "full_numeric",
    },
    "processed/defensive_archetypes_v2.parquet": {
        "join_on": ["PLAYER_ID", "SEASON"],
        "label_cols": ["defensive_archetype"],
        "check": "labels_and_numeric",
    },
    "advanced_local_metrics.parquet": {
        "join_on": ["Player_ID", "SEASON"],
        "check": "full_numeric",
        "tolerance": 0.02,
    },
    "official_stats/official_advanced_{season}.parquet": {
        "seasons": ["2022-23", "2023-24", "2024-25"],
        "check": "exact_match",
    },
    "tracking/{season}/shot_zones.parquet": {
        "seasons": ["2022-23", "2023-24", "2024-25"],
        "check": "exact_match",
    },
}
```

### Comparison Output Format

The script should produce:

1. **Console summary** with pass/fail per file
2. **`data_fresh/logs/comparison_report.json`** with machine-readable results
3. **`data_fresh/logs/comparison_report.txt`** with human-readable detail

Example output:
```
=== DATA COMPARISON REPORT ===
Date: 2026-02-08
Old: data/    New: data_fresh/

FILE                                    STATUS    ROWS_OLD  ROWS_NEW  DRIFT   NOTES
─────────────────────────────────────────────────────────────────────────────────
historical/possessions_clean_2024-25    ✅ PASS   185,432   185,432   0.000
processed/player_profiles_advanced      ✅ PASS   1,680     1,680     0.001   Max Δ: TS_PCT 0.003
processed/metrics_linear                ⚠️ WARN   1,680     1,680     0.042   WS MAD=0.04, 3 players Δ>5%
processed/player_rapm                   ✅ PASS   1,432     1,432     0.008
processed/player_archetypes             ✅ PASS   1,680     1,680     —       0 label changes
processed/archetype_embeddings          ✅ PASS   1,680     1,680     0.002
processed/defensive_archetypes_v2       ✅ PASS   1,687     1,687     —       0 label changes

LABEL CHANGES:
  (none)

KEY PLAYER SPOT CHECKS:
  LeBron James     WS: 8.2→8.2  BPM: 7.1→7.1  RAPM: +3.2→+3.2  Archetype: BDC/Helio ✅
  Stephen Curry    WS: 9.1→9.1  BPM: 6.8→6.8  RAPM: +4.1→+4.1  Archetype: BDC/Grav  ✅
  Nikola Jokic     WS: 17.0→17.0 BPM: 11.5→11.5                  Archetype: BDC/Post  ✅
  ...
```

---

## 5. Review Checklist — What to Look For Per Script

Before or during the re-run, review each script for these common issues:

### Fetch Scripts (Layer 0)

| Issue | Where to Check | Risk |
|-------|---------------|------|
| **Stale session cookies** | `data/nba_headers.json`, `data/nba_session.json` | API 403s — update before re-fetch |
| **Rate limiting** | All `fetch_*.py` scripts — look for `time.sleep()` | Getting IP-banned by NBA.com |
| **Missing seasons** | `SEASONS` list in each script | 2024-25 data may be incomplete (mid-season) |
| **Schema changes** | NBA API may have added/removed columns since last fetch | Column mismatch errors |
| **Duplicate rows** | `fetch_box_scores_complete.py` fetches league-wide — check for player dupes | Inflated aggregates |
| **ID type mismatches** | `player_id` as int vs str varies across scripts | Silent merge failures |
| **CatchShoot 2022-23 proxy** | `fetch_tracking_data.py` has a fallback for broken endpoint | Verify proxy still needed |

### Normalize/Feature Scripts (Layer 1)

| Issue | Where to Check | Risk |
|-------|---------------|------|
| **Lineup inference accuracy** | `derive_lineups.py` — bench techs, ejections | Wrong 5-man lineups → bad RAPM |
| **Possession boundary logic** | `derive_possessions.py` — turnovers, end-of-period | Missed possessions or double-counting |
| **Clean possession filter** | `compute_clean_possessions.py` — rejects non-5v5 | Losing too many valid possessions |
| **Game log completeness** | `derive_team_game_logs.py` — depends on PBP coverage | Missing games = wrong team totals |

### Compute Scripts (Layer 2)

| Issue | Where to Check | Risk |
|-------|---------------|------|
| **qAST calculation** | `compute_linear_metrics.py` lines 200-250 | B-Ref WS accuracy depends on this |
| **Team average proxies** | `compute_linear_metrics.py` uses league/30 | Traded players get wrong team context |
| **BPM coefficients** | `compute_linear_metrics.py` | Must match B-Ref 2.0 spec exactly |
| **RAPM regularization** | `model_rapm.py` — alpha selection range | Over/under-regularization |
| **Archetype hierarchy order** | `compute_player_archetypes.py` — Steps 1-14 | Wrong order = wrong classifications |
| **Defensive archetype thresholds** | `compute_defensive_archetypes_v2.py` | Distribution may shift with updated data |

---

## 6. Improvement Opportunities (Flag While Reviewing)

As you review each script, tag with one of:

- `[CORRECTNESS]` — Logic bug or wrong formula
- `[COMPLETENESS]` — Missing data, players, or seasons
- `[ROBUSTNESS]` — Fragile code (hardcoded paths, no error handling)
- `[PERFORMANCE]` — Slow or wasteful (unnecessary re-reads, no caching)
- `[QUALITY]` — Could produce better output with small tweaks

Keep a running list in `data_fresh/logs/review_findings.md` during the process.

---

## 7. Execution Plan — Step by Step

```
- [ ] Step 1: Create backup (§1a, §1b)
- [ ] Step 2: Review fetch scripts (§5 Layer 0 checklist) — flag issues
- [ ] Step 3: Review normalize/feature scripts (§5 Layer 1 checklist) — flag issues
- [ ] Step 4: Review compute scripts (§5 Layer 2 checklist) — flag issues
- [ ] Step 5: Fix any [CORRECTNESS] issues found in Steps 2-4
- [ ] Step 6: Create scripts/reproduce_pipeline.sh (§3)
- [ ] Step 7: Create scripts/compare_data_batches.py (§4)
- [ ] Step 8: Run Layer 0 (copy or re-fetch, per preference)
- [ ] Step 9: Run Layer 1 (normalize + features)
- [ ] Step 10: Run Layer 2 (compute all metrics)
- [ ] Step 11: Run existing validation tests against fresh data
- [ ] Step 12: Run comparison script (old vs new)
- [ ] Step 13: Review comparison report — investigate any WARN/FAIL items
- [ ] Step 14: Fix issues found, re-run affected steps
- [ ] Step 15: Final comparison — all PASS
- [ ] Step 16: Decide: promote data_fresh → data, or keep original
- [ ] Step 17: Update loop/ context files and commit
```

### Estimated Time

| Phase | Human Time | Machine Time |
|-------|-----------|-------------|
| Backup (§1) | 2 min | 5-10 min (copy) |
| Script review (§5) | 30-60 min | — |
| Fix issues (Step 5) | 15-45 min | — |
| Create runner + comparator | 15 min | — |
| Layer 0 (copy mode) | 1 min | 5 min |
| Layer 0 (re-fetch mode) | 1 min | 30-60 min (API calls) |
| Layer 1 (normalize) | — | 10-20 min |
| Layer 2 (compute) | — | 15-30 min |
| Validation + comparison | 5 min | 5 min |
| Investigation + fixes | 15-30 min | depends |
| **Total** | **~1.5-3 hours** | **~1-2 hours compute** |

---

## 8. Decision Framework — What Constitutes "Pass"

### Deterministic Outputs (must be exact or near-exact)
- PBP normalization: row count identical, schema identical
- Lineups: same lineups inferred (±0.1% tolerance for edge cases)
- Possessions: same count (±0.5% for boundary cases)
- Official stats, tracking, matchup data: byte-identical if not re-fetched

### Stochastic Outputs (allow tolerance)
- RAPM/xRAPM: correlation ≥ 0.995, MAD ≤ 0.1 per-100
- Win Shares: MAD ≤ 0.3 vs old data, MAD ≤ 0.5 vs B-Ref ground truth
- BPM: MAD ≤ 0.5
- Archetype labels: ≤ 2% of players change archetype
- Embeddings: cosine similarity ≥ 0.99 per player

### Red Flags (immediate investigation)
- Any file missing entirely
- Row count differs by > 5%
- RAPM correlation < 0.98
- WS MAD > 1.0
- > 10% of archetype labels changed
- Key player (LeBron, Curry, Jokic) classification changed

---

## 9. After Comparison — Promotion or Rollback

### If everything passes:
```bash
# Promote fresh data
mv data data_old_$(date +%Y%m%d)
mv data_fresh data
# Clean up
rm -rf data_old_*  # after confirming
```

### If issues found:
```bash
# Keep original, fix scripts, re-run
# data/ is untouched since we used data_fresh/
# Fix issues in src/ scripts, re-run affected steps only
```

### If you want to keep both:
```bash
# Original stays at data/
# Fresh stays at data_fresh/ for A/B comparison
# Use data_fresh/ for new evaluation work
```
