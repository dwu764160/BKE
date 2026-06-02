# Parquet Data Standardization Plan

**Status:** DECISIONS LOCKED — ready to implement (2026-06-01)  
**Owner:** Daniel + Claude Code  
**Branch:** personal  
**Strategy:** Option B — In-place parquet rewrite + full script update  
**Stats scope:** Full lowercase — ALL column names including stats (PTS→pts, REB→reb, etc.)  
**Phase order:** Phase 0 → 5 → 4 → 3 → 2 → 1 → 6 → 7 (root cause first)  
**Session plan:** 3 sessions with Sonnet (see Session Grouping below)  

---

## Problem Statement

The BKE data layer has accumulated four distinct schema inconsistency classes across its parquet corpus. These cause silent bugs, require per-script workarounds, and blocked Step-2 calibration work (the `Player_ID`/`PLAYER_ID` collision, the `SEASON_ID` numeric format, float-string team IDs).

### Issue Classes Found (full audit 2026-06-01)

| Class | Affected Files | Example |
|-------|---------------|---------|
| **Case-dup columns** | `player_game_logs_*.parquet` (2 files), `final_player_game_logs.parquet`, `complete_player_season_stats.parquet`, `bke_v{15,20,25,26,27,28,30}_decomposition.parquet` (7 files) | `Player_ID` + `PLAYER_ID` both present; `EFG_PCT` + `eFG_pct` both present |
| **SEASON_ID numeric format** | `player_game_logs_*.parquet`, `final_player_game_logs.parquet`, `league_season_matchups.parquet`, `matchups_rollup.parquet` | `SEASON_ID = '22024'` instead of `'2024-25'`; no `season` column |
| **Float-string IDs** | `player_impact_profiles.parquet`, `forecast/projected_player_profiles.parquet`, `player_profile_aggregate.parquet` | `team_id = '1610612744.0'` |
| **UPPERCASE vs lowercase** | Most of `data/historical/`, `data/matchup/`, `data/processed/` raw inputs | `GAME_ID`, `PLAYER_ID`, `SEASON` vs `game_id`, `player_id`, `season` |

### Migration Surface (from audit)

- **95** Python files read at least one parquet/CSV
- **86** Python files reference uppercase column names  
- **71** files do both — these are the full migration targets

---

## Locked Decisions

### Strategy: Option B — In-place rewrite

Rewrite ALL parquet files to canonical lowercase schema. Update all 71 consumer scripts to use lowercase column names. `load_standardized()` exists as a permanent safety net but becomes effectively a no-op on clean files.

**Pre-condition (mandatory before ANY parquet is touched):**
1. Create `data_backup_20260601/` — full copy of all gitignored data
2. Run validation script to verify backup is complete and intact (row counts + checksums)
3. Only proceed after validation passes

### Stats columns: Full lowercase — everything

ALL column names normalized to lowercase, including stat columns:

```
PTS → pts       REB → reb       AST → ast       MIN → min
GP → gp         FGA → fga       FGM → fgm       FG3A → fg3a
FG3M → fg3m     FTA → fta       FTM → ftm       OREB → oreb
DREB → dreb     STL → stl       BLK → blk       TOV → tov
WL → wl         MATCHUP → matchup    GAME_DATE → game_date
... (all caps stat columns → lowercase)
```

### Phase order: Root cause first (5 → 4 → 3 → 2 → 1 → 6 → 7)

Start at the data source (fetch layer) and work downstream. Each completed phase means newly regenerated files downstream are already canonical.

### Phase 7: In scope

After Phases 0–6 complete, rewrite the historical parquets that cannot be regenerated from the pipeline (PBP files, game logs). `data_backup_20260301` + `data_backup_20260601` both serve as safety nets.

### Session grouping: 3 Sonnet sessions (Sonnet is fine)

Sonnet handles mechanical find-and-replace work equally well to Opus. All 7 phases in one session is too much context — splitting into 3 keeps each session focused and verifiable.

| Session | Phases | Scope |
|---------|--------|-------|
| **Session A** | 0 + 5 | Foundation (schema contract) + data fetch layer (root cause fix) |
| **Session B** | 4 + 3 + 2 | Data compute → modeling → player eval/aggregate |
| **Session C** | 1 + 6 + 7 | Simulation + analysis scripts + historical parquet rewrite |

---

## Canonical Schema Rules (target)

```
Column names:    all lowercase, underscore-separated (no exceptions)
Season format:   'YYYY-YY'  (e.g. '2024-25', not '22024' or '2024')
Season ID:       season_id column renamed to season; '22024' → '2024-25'
ID columns:      plain integer strings, no trailing .0  (e.g. '1610612744')
Stat columns:    all lowercase  (pts, reb, ast, min, gp, fga, fg3m, ...)
```

---

## Pre-Flight: Backup + Validation (before Session A)

### Step 1: Create 2026-06-01 backup

```bash
# From repo root — copy all gitignored data directories
cp -r data/ data_backup_20260601/
cp -r reports/ reports_backup_20260601/ 2>/dev/null || true
cp -r aggregate/ aggregate_backup_20260601/ 2>/dev/null || true
```

### Step 2: Validation script

`scripts/validate_backup_integrity.py` — new file, created in Phase 0.

Checks:
- Row count match between `data/` and `data_backup_20260601/` for every parquet
- SHA-256 checksum match for every parquet file
- Prints a PASS/FAIL summary — abort if any FAIL

### Step 3: Post-rewrite validation

Same script run again after each phase's parquet rewrites. Checks:
- All parquets still readable (not corrupted)
- Row counts unchanged (no rows dropped)
- Column names all lowercase (schema rule enforced)
- No null values introduced in key join columns (player_id, game_id, season)

---

## Phase 0 — Foundation (Session A, first)

**Goal:** Build the schema contract and validation tooling before touching a single file.

**Deliverables:**

### `src/data/schema_contract.py`

```python
# Public API:
load_standardized(path, **kwargs) -> pd.DataFrame
save_standardized(df, path, **kwargs) -> None
canonicalize(df) -> pd.DataFrame          # apply rules without I/O
_season_to_standard(val: str) -> str      # '22024' → '2024-25', passthrough
_normalize_id_col(series) -> pd.Series   # strip .0 from float IDs
```

**`canonicalize(df)` transform sequence:**
1. Lowercase all column names
2. Drop case-variant duplicate columns (keep first by column order)
3. If `season_id` present and `season` absent: rename to `season`
4. If both `season` and `season_id` present: drop `season_id`
5. Normalize `season` column: `'22024'` → `'2024-25'`
6. Normalize all `*_id` columns: strip trailing `.0` from string representations

**`load_standardized(path, **kwargs)`:** `pd.read_parquet(path, **kwargs)` → `canonicalize(df)`  
**`save_standardized(df, path, **kwargs)`:** `canonicalize(df)` → `df.to_parquet(path, **kwargs)`

### `scripts/validate_backup_integrity.py`

Standalone script, no imports from `src/`. Compares `data/` vs `data_backup_20260601/` on row counts + checksums. Used before and after every parquet rewrite phase.

### `docs/reference/data_schemas.md`

One-page canonical schema reference. Covers all files in the Step-2 critical path plus canonical column name table.

**Files created:** 3  
**Files updated:** 0  
**Parquets rewritten:** 0

---

## Phase 5 — Data Fetch Layer (Session A, second)

**Goal:** Fix the root cause — fetch scripts write canonical schema going forward.

**Consumer scripts to update (12 files):**
```
src/data_fetch/derive_team_game_logs.py
src/data_fetch/fetch_backfill_box_scores.py
src/data_fetch/fetch_box_scores_complete.py
src/data_fetch/fetch_matchup_data.py
src/data_fetch/fetch_profiles.py
src/data_fetch/fetch_players.py
src/data_fetch/fetch_preseason_rosters.py
src/data_fetch/fetch_player_salaries.py
src/data_fetch/fetch_player_draft_history.py
src/data_fetch/fetch_shot_zones.py
src/data_fetch/summarize_team_logs.py
src/data_fetch/fetch_pbp/fetch_play_by_play.py
```

**Changes per script:**
- Import `save_standardized` from `src.data.schema_contract`
- Replace `df.to_parquet(path)` → `save_standardized(df, path)`
- Replace `pd.read_parquet(path)` → `load_standardized(path)` where fetch scripts read existing files
- Update all column references in the script body to lowercase

**Parquets rewritten (existing fetch outputs):**
```
data/historical/team_game_logs.parquet
data/historical/player_game_logs_2023-24.parquet
data/historical/player_game_logs_2024-25.parquet
data/historical/final_player_game_logs.parquet
data/matchup/league_season_matchups.parquet
data/matchup/matchups_rollup.parquet
data/matchup/matchup_difficulty.parquet
data/matchup/matchup_versatility.parquet
data/official_stats/*.parquet (all)
data/tracking/*.parquet (all)
data/features/*.parquet (if any)
```

**Verification gate:** `python3 scripts/validate_backup_integrity.py` PASS; all row counts unchanged.

---

## Phase 4 — Data Compute Layer (Session B, first)

**Goal:** Fix compute scripts — they read historical (now canonical after Phase 5 rewrite) and write to `data/processed/`.

**Consumer scripts to update (8 files):**
```
src/data_compute/compute_player_archetypes.py
src/data_compute/compute_defensive_archetypes_v2.py
src/data_compute/compute_linear_metrics.py
src/data_compute/compute_player_profiles.py
src/data_compute/compute_position_estimate.py
src/data_compute/compute_local_metrics.py
src/data_compute/fit_rest_hca_coefficients.py
src/data_compute/fit_team_pace.py
```

**Parquets rewritten (compute outputs in data/processed/):**
```
data/processed/player_archetypes.parquet
data/processed/defensive_archetypes*.parquet (all versions)
data/processed/metrics_linear.parquet
data/processed/metrics_lineups.parquet
data/processed/metrics_teams.parquet
data/processed/metrics_win_shares.parquet
data/processed/modeling_inputs_*.parquet (all seasons + all)
data/processed/player_position_estimates*.parquet (all)
```

**Verification gate:** `python3 scripts/validate_backup_integrity.py` PASS; spot-check 3 files' schemas in Python.

---

## Phase 3 — Modeling Layer (Session B, second)

**Consumer scripts to update (~10 files):**
```
src/modeling/layer1_portable_talent.py
src/modeling/construct_bke_scores_v27.py
src/modeling/bke_v31_experimental_layers.py
src/modeling/decomposition_engine.py
src/modeling/fit_archetype_interactions_v1_team_season.py
src/modeling/fit_archetype_interactions_v2.py
src/modeling/ingest_darko.py
src/modeling/model_rapm.py
src/modeling/backtesting.py
scripts/build_pts_v40.py
```

**Parquets rewritten:**
```
data/processed/bke/bke_v{15,20,25,26,27,28,30}_decomposition.parquet  — fix case-dup stat cols
data/processed/bke/pts_v40.parquet
data/processed/bke/cross_team_interactions.parquet
data/processed/player_rapm.parquet
data/processed/player_xrapm.parquet
data/processed/player_xrapm_v2.parquet
data/processed/archetype_embeddings.parquet
```

**Special case — bke_v* decompositions:** Case-dup stat columns (`EFG_PCT`/`eFG_pct`) resolved by `canonicalize()` (first occurrence kept, duplicate dropped). After rewrite all columns are lowercase and deduped.

---

## Phase 2 — Player Eval + Profile Aggregate (Session B, third)

**Consumer scripts to update (6 files):**
```
src/player_eval/build_player_impact_profiles.py
src/player_eval/project_next_season.py
src/player_eval/calibrate_team_scale.py
src/player_eval/year_to_year_bke_deltas.py
src/profile_aggregate/build_profile_aggregate.py
src/profile_aggregate/team_feature_aggregation.py
```

**Parquets rewritten:**
```
data/processed/player_eval/player_impact_profiles.parquet   — float team_id
data/processed/player_eval/minute_model_predictions_v*.parquet
data/processed/player_eval/team_feature_aggregation.parquet
data/processed/forecast/projected_player_profiles.parquet   — float team_id
data/processed/forecast/forecast_step*.parquet (all)
aggregate/player_profile_aggregate.parquet                  — float team_id_2
```

---

## Phase 1 — Simulation Layer (Session C, first)

**Goal:** Unblock Step-2.1 calibration work. The highest-value layer, done last because it reads from layers 2–5 — those must be clean first.

**Consumer scripts to update (7 files):**
```
scripts/run_possession_engine.py
scripts/validate_possession_engine.py
src/simulation/lineup_projection.py
src/simulation/game_model.py
src/simulation/gbdt_game_model.py
src/simulation/season_sim.py
src/simulation/validate_forecast.py
```

**Parquets rewritten:**
```
data/processed/simulation/possession_box_distributions.parquet
data/processed/simulation/simulation_step*.parquet (all)
data/processed/simulation/forecast_step*.parquet (all)
```

**Verification gate:** `pytest -q tests/` 3/3; `python3 scripts/validate_possession_engine.py` reports same MAE numbers as pre-standardization baseline.

---

## Phase 6 — Analysis and Experiment Scripts (Session C, second)

**Consumer scripts to update (~24 files):**
```
scripts/audit_forecast_leakage.py
scripts/build_2025_26_projections.py
scripts/build_all_season_projections.py
scripts/build_pts_team_ratings.py
scripts/build_rest_features.py
scripts/build_ytd_team_ratings.py
scripts/compare_bref_advanced_metrics.py
scripts/compare_data_batches.py
scripts/compute_matchup_adj.py
scripts/diagnose_brier_baseline.py
scripts/experiment_microadjustments.py
scripts/experiment_pts_rdis_blend.py
scripts/experiment_ytd_roster_pts.py
scripts/fit_archetype_interactions_v2.py
scripts/pts_v32_harness.py
scripts/pts_v32_posthoc_harness.py
scripts/pts_v32_recompute.py
scripts/pts_v40_defense.py
scripts/pts_v40_multiseason.py
scripts/pts_v40_sweep.py
scripts/test_archetype_interactions.py
scripts/test_softmax_vs_argmax.py
scripts/validate_lineup_pts.py
scripts/validate_lineup_pts_v2.py
scripts/validate_matchup_interactions.py
scripts/validate_player_stat_sim.py
scripts/validate_pts_team_ratings.py
```

No parquets rewritten in this phase — these are read-only analysis scripts.

---

## Phase 7 — Historical Parquet Rewrite (Session C, third)

**Goal:** Rewrite the ~15 historical files that cannot be regenerated from the pipeline. After this, `load_standardized()` is a true no-op on all files.

**Pre-conditions:**
- Both `data_backup_20260301/` and `data_backup_20260601/` verified intact
- `scripts/validate_backup_integrity.py` PASS on current state
- Phases 0–6 all complete and verified

**Files to rewrite:**
```
data/historical/player_game_logs_2023-24.parquet    — UPPERCASE + case-dup + SEASON_ID
data/historical/player_game_logs_2024-25.parquet    — UPPERCASE + case-dup + SEASON_ID
data/historical/final_player_game_logs.parquet      — UPPERCASE + case-dup + SEASON_ID
data/historical/complete_player_season_stats.parquet — UPPERCASE + case-dup
data/historical/complete_player_season_stats_backfill.parquet — mixed case
data/historical/team_game_logs.parquet              — UPPERCASE + SEASON_ID (has both)
data/historical/feature_schedule_context.parquet    — UPPERCASE
data/historical/team_game_details.parquet           — UPPERCASE
data/historical/team_summaries.parquet              — UPPERCASE
data/historical/play_by_play_2017-18.parquet        — mixed case
data/historical/play_by_play_2018-19.parquet        — mixed case
data/historical/play_by_play_2019-20.parquet        — mixed case
data/historical/play_by_play_2020-21.parquet        — mixed case
data/historical/play_by_play_2021-22.parquet        — mixed case
data/historical/play_by_play_2022-23.parquet        — mixed case
data/historical/play_by_play_2023-24.parquet        — mixed case
data/historical/play_by_play_2024-25.parquet        — mixed case
data/historical/play_by_play_2025-26.parquet        — mixed case
data/historical/pbp_normalized_*.parquet (9 files)  — mixed case
data/historical/pbp_with_lineups_*.parquet (9 files) — mixed case
```

**Rewrite procedure:** `canonicalize(pd.read_parquet(f)).to_parquet(f, index=False)` per file.  
**Verification:** Row count + checksum vs `data_backup_20260601/` after each file.

---

## Session Log

| Date | Session | Phases | Status | Notes |
|------|---------|--------|--------|-------|
| 2026-06-01 | Planning | Audit + Plan | ✅ Done | Full corpus audit; 71 migration targets; decisions locked |
| 2026-06-01 | A | 0 + 5 | ✅ Done | schema_contract.py + validator + docs; 12 fetch scripts updated; 343 parquets rewritten; pytest 3/3; Phase 5 validator PASS |
| 2026-06-01 | B | 4 + 3 + 2 | ✅ Done | 81 data/processed/ + aggregate/ parquets rewritten; 8 compute scripts + 10 modeling scripts + 6 player_eval/aggregate scripts updated; pytest 3/3; post-rewrite PASS (1 pre-existing null in cross_team_interactions, present in backup) |
| 2026-06-02 | C | 1 + 6 + 7 | ✅ Done | 7 simulation scripts + 18 analysis scripts + 4 PBP historical parquets updated; possession engine MAE unchanged; pytest 3/3; validator PASS |
