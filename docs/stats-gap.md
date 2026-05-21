# Player Profile Aggregate — Data Coverage & Gap Index

**Aggregate:** `aggregate/player_profile_aggregate.parquet`  
**Seasons covered:** 2017-18 through 2024-25  
**Shape:** 5,848 rows × 946 columns  
**Tracking tier split:** `rapm_only` (pre-2022-23) | `full` (2022-23 onward)

---

## Update Log

| Date | Action | Who |
|---|---|---|
| 2026-05-21 | Initial audit — Phase 0 complete, 8-season aggregate built | Session |
| 2026-05-21 | BRef BPM/OBPM/DBPM/WS/VORP fetched for all 8 seasons → `data/reference/external_rapm_*.parquet` | Session |
| 2026-05-21 | Pre-2022 salary fetch attempted — blocked (ESPN 202 challenge page, BRef salary pages removed). Gap documented in GAP-003. | Session |
| 2026-05-21 | Phase 1 Track B complete — BPM formula restored (4 constants + Step 7c removed). r=0.93-0.94 vs BRef. Report: `reports/rapm_external_validation.json`. | Session |
| 2026-05-21 | GAP-011 added: lineup reconstruction covers 50-78% of traded player game appearances. Root in `derive_lineups.py`. | Session |
| 2026-05-21 | GAP-012 fixed: Pre-2022 PBP lacks `assistPersonId` → AST=0 → BPM NaN for 2017-22. Fixed in `compute_linear_metrics.py`: fill AST from box stats (coverage-scaled by PBP minute ratio). BPM now 99-100% filled all 8 seasons. | Session |

> **How to update this file:** After rebuilding `player_profile_aggregate.parquet`, run:
> ```bash
> .venv/bin/python3 -c "
> import pandas as pd; df = pd.read_parquet('aggregate/player_profile_aggregate.parquet')
> seasons = sorted(df['season'].unique())
> for col in ['orapm','bke_raw_obke','salary','weight_lbs','wingspan_inches','obpm','drives']:
>     by_s = df.groupby('season')[col].apply(lambda x: round(x.notna().mean(),2)) if col in df.columns else 'MISSING'
>     print(col, dict(by_s) if hasattr(by_s,'items') else by_s)
> "
> ```
> Then append a new row to the Update Log above and update any fill rates that changed.

---

## Metric Index by Domain

Fill rates shown as: `17-18 | 18-19 | 19-20 | 20-21 | 21-22 | 22-23 | 23-24 | 24-25`

---

### Identity (10 cols)

Core player-season keys. All critical fields 100% filled.

| Column | Fill by Season | Notes |
|---|---|---|
| `player_id` | 1.00 all seasons | |
| `player_name` | 1.00 all seasons | |
| `season` | 1.00 all seasons | |
| `position_bucket` | 1.00 all seasons | Position from RAPM model (PG/SG/SF/PF/C) |
| `team_abbreviation_2` | 1.00 all seasons | |
| `primary_position_estimate` | 0.99 \| 0.80 \| 0.68 \| 0.72 \| 0.78 \| 0.70 \| 0.71 \| 0.74 | From compute_position_estimate.py; ~25% unfilled = low-minute players |
| `age` | 0.99 \| 0.80 \| 0.68 \| 0.72 \| 0.78 \| 0.70 \| 0.71 \| 0.74 | NBA API bio |

---

### RAPM (11 cols)

All 8 seasons, core impact metrics.

| Column | Fill by Season | Notes |
|---|---|---|
| `orapm` | 1.00 all seasons | Pooled multi-season ridge regression RAPM |
| `drapm` | 1.00 all seasons | |
| `rapm` | 1.00 all seasons | Net RAPM |
| `rapm_type` | 1.00 all seasons | `rapm_only` pre-2022 / `full` post-2022 |
| `rapm_position_bucket_pctl` | 1.00 all seasons | Percentile within position |
| `alpha_pooled` | 0.53–0.67 all seasons | Shrinkage factor; lower fill = low-sample players excluded |

---

### Player-Eval Computed / `pec_` (87 cols)

**Near-100% fill all 8 seasons.** These are computed by `build_player_impact_profiles.py` with fallback logic — every player with possessions data gets a value. The `pec_` prefix means "player eval computed."

| Column | Fill by Season | Notes |
|---|---|---|
| `pec_impact_orapm` | 1.00 all seasons | |
| `pec_impact_bke` | 1.00 all seasons | Uses RAPM-fallback when BKE not available |
| `pec_behavioral_three_point_rate` | 1.00 all seasons | Pre-2022: box-score 3PA/FGA; 22% of pre-2022 rows = league-avg fallback (0.405) for low-data players |
| `pec_behavioral_efg` | 1.00 all seasons | Same: box-score derived pre-2022 |
| `pec_minute_share_potential` | 1.00 all seasons | |
| `pec_salary` | 0.00 pre-2022 \| 0.54 post-2022 | Mirrors raw salary column |

---

### BKE Scores / `bke_` (26 cols)

**30-42% fill** — expected. BKE has a minimum-possession qualification threshold (~500 possessions). Only qualified players receive BKE scores. Low-minute players, short-stint injuries, and developmental contracts are intentionally excluded.

| Column | Fill by Season | Notes |
|---|---|---|
| `bke_raw_obke` | 0.40 \| 0.34 \| 0.27 \| 0.31 \| 0.31 \| 0.41 \| 0.38 \| 0.42 | |
| `bke_raw_dbke` | same | |
| `bke_final_pctl` | same | League-wide percentile 0-100 |
| `bke_dim_shooting_gravity` | same | |
| `bke_dim_defensive_impact` | same | |
| `obke_v30_reference` | 0.00 pre-2022 \| 0.39 post-2022 | v30 defense shrinkage output; Phase C failed on expanded dataset — see Known Gaps |

---

### BKE Decomposition / Layer Scores (32 cols)

Same qualification pattern as BKE Scores (~30-42% fill). All 8 seasons present.

| Column | Fill by Season | Notes |
|---|---|---|
| `offensive_portable_z` | 0.40 \| 0.34 \| 0.27 \| 0.31 \| 0.31 \| 0.41 \| 0.38 \| 0.42 | Layer 1 portable talent |
| `defensive_portable_z` | same | |
| `portable_talent_z` | same | Combined Layer 1 |
| `elevation_z` | same | Layer 3: archetype elevation |
| `rue_z` | same | Layer 2: role utilization efficiency |
| `scheme_stability_z` | same | Layer 4: scheme amplification |

---

### Archetypes (132 cols)

Mixed fill. Soft archetype probabilities are 100% filled all seasons. Hard archetype assignments and v28+ columns are qualification-gated (~40% fill). Tracking-dependent archetype features are 0% pre-2022.

| Column | Fill by Season | Notes |
|---|---|---|
| `arch_prob_ball_dominant_creator` | 1.00 all seasons | Soft probability; all players get one |
| `soft_archetype_entropy` | 1.00 all seasons | |
| `primary_archetype` | 0.00 pre-2022 \| 0.70-0.74 post-2022 | Hard assignment; 0% pre-2022 is a gap (see Known Gaps) |
| `portability_archetype_transfer` | 0.27-0.42 all seasons | BKE-qualified players only |
| `off_rating` | 0.00 pre-2022 \| 0.70 post-2022 (2024-25 also 0!) | **BUG NOTE: 2024-25 fill=0.00** — investigate |
| `def_rating` | same as off_rating | same bug |

---

### Bio (5 cols)

Degraded pre-2022. Source is `players.parquet` which is a current-season roster snapshot (686 active players). Retired players have no bio data.

| Column | Fill by Season | Notes |
|---|---|---|
| `height_inches` | 0.99 \| 0.80 \| 0.68 \| 0.72 \| 0.78 \| 0.70 \| 0.71 \| 0.74 | From `players.parquet` (current roster) + position_estimates |
| `weight_lbs` | 0.37 \| 0.37 \| 0.39 \| 0.48 \| 0.57 \| 0.67 \| 0.74 \| 0.89 | Gradient = retired players have no weight; only current players do |
| `wingspan_inches` | 0.00 all seasons | **Completely missing** — see Known Gaps |
| `bio_height_inches` | same gradient as weight_lbs | Duplicate from players.parquet |
| `experience_years` | same gradient as weight_lbs | Same source limitation |

---

### Box Advanced Stats (7 cols)

`bpm`, `obpm`, `dbpm`, `vorp`, `ws`, `ws_per48`, `ts_pct`

| Column | Fill by Season | Notes |
|---|---|---|
| `ts_pct` | 0.99 \| 0.79 \| 0.67 \| 0.72 \| 0.77 \| 0.70 \| 0.70 \| 0.73 | From NBA API box stats; available all 8 seasons |
| `bpm`, `obpm`, `dbpm` | 0.00 pre-2022 \| 0.70 post-2022 | BRef scrape; **pre-2022 fetch running now** |
| `vorp`, `ws`, `ws_per48` | 0.00 pre-2022 \| 0.70 post-2022 | Same; pre-2022 fetch running now |

---

### Salary (1 col)

| Column | Fill by Season | Notes |
|---|---|---|
| `salary` | 0.00 pre-2022 \| 0.53-0.59 post-2022 | ESPN scrape; **pre-2022 fetch running now** |

---

### Tracking (27 cols)

**Correctly 0% pre-2022** — tracking data (`leaguedashptstats`) unavailable from NBA API for pre-2022-23 seasons. This is a confirmed data availability gap, not a pipeline bug. See `docs/reference/tracking_availability.md`.

| Column | Fill by Season | Notes |
|---|---|---|
| `drives`, `drive_pts`, `drive_fg_pct` | 0.00 pre-2022 \| 0.70 post-2022 | |
| `catch_shoot_fga`, `catch_shoot_fg_pct` | 0.00 pre-2022 \| 0.71 post-2022 | |
| `pull_up_fga`, `pull_up_fg_pct` | 0.00 pre-2022 \| 0.71 post-2022 | |
| `avg_speed`, `dist_miles` | 0.00 pre-2022 \| 0.70 post-2022 | |
| `touches` | 0.00 pre-2022 \| 0.70 post-2022 | |

---

### Clutch Stats (8 cols)

**0% pre-2022.** Fetch script (`fetch_player_clutch_stats.py`) only covers 2022-25. Pre-2022 clutch data is fetchable from `leaguedashplayerclutch` endpoint but was never run for older seasons.

---

### Game Log Detail / `gl_` (10 cols)

`gl_games_total`, `gl_mpg_mean`, `gl_mpg_std`, etc.

| Column | Fill by Season | Notes |
|---|---|---|
| `agg_mpg`, `agg_minute_share` | 0.99-1.00 pre-2022 \| 0.70 post-2022 | From team_game_logs — available all seasons |
| `gl_games_total`, `gl_mpg_*` | 0.00 pre-2022 \| 0.54-0.64 post-2022 | Detailed game log aggregates only for 2022-25 |

---

### Draft (9 cols)

| Column | Fill by Season | Notes |
|---|---|---|
| `draft_tier`, `is_drafted` | 0.40-0.52 pre-2022 \| 0.73-1.00 post-2022 | Draft history only covers current players; gradient = retired players missing |
| `draft_class_year` | 0.34-0.48 pre-2022 \| 0.54-0.69 post-2022 | Same source limitation |
| `draft_pick_overall` | same | |

---

### xRAPM (1 col)

| Column | Fill by Season | Notes |
|---|---|---|
| `xrapm_alpha` | 0.00 except 2023-24=0.82, 2024-25=1.00 | xRAPM v2 only computed for recent 2 seasons |

---

### Other / Miscellaneous (570 cols)

Catch-all bucket. Includes archetype probability embeddings (`arch_prob_*`, 100% fill all seasons), model internals, position probabilities (`prob_PG`, `prob_C`, etc.), and various computed z-scores. Pre-2022 fill ranges from 0% (tracking-dependent) to 100% (RAPM-derived).

Notable fully-filled columns in this group: `possessions_played`, `arch_prob_*` (11 archetypes), `prob_PG/SG/SF/PF/C`, `n_seasons_pooled`.

---

## Known Gaps / Issues

### GAP-001 — Wingspan: Completely Missing (0% all seasons)
**Severity:** Medium  
**Column:** `wingspan_inches`  
**Root cause:** `players.parquet` is fetched from NBA API `/players` endpoint which does not return wingspan. The column exists in the schema but is always null.  
**Impact:** Phase 2 minute model and Phase 3B calibration cannot use wingspan as a feature.  
**Fix:** Fetch from DraftExpress archive or Basketball-Reference player pages. Needs a new `fetch_player_bios_historical.py` script.  
**Blocked by:** Nothing — independent data fetch task.

---

### GAP-002 — Weight / Height / Experience: Gradient Fill (37% → 89% 2017-18 → 2024-25)
**Severity:** Low  
**Columns:** `weight_lbs`, `bio_height_inches`, `experience_years`  
**Root cause:** `players.parquet` is a current-season roster dump (686 active players only). Players who retired before 2024-25 are missing bio data for their historical seasons.  
**Impact:** Low — `height_inches` (from position_estimates) is well-filled. Weight is not currently used as a model feature.  
**Fix:** Run `commonplayerinfo` NBA API endpoint for all historical player IDs. A batch script over all player IDs in `data/processed/player_rapm.parquet`.  
**Blocked by:** Nothing — independent data fetch task.

---

### GAP-003 — Salary: Only 2022-25 (0% pre-2022)
**Severity:** Low-Medium  
**Column:** `salary`  
**Root cause:** Pre-2022 salary data is not programmatically accessible from free sources. ESPN salary pages now return a JS challenge (202). BRef historical salary pages (`NBA_YEAR_salaries.html`) return 404 — those pages were removed. NBA API has no public salary endpoint.  
**Impact:** Cannot use salary as a feature in pre-2022 model training. Phase 3B magnitude calibration may need salary as a control variable.  
**Fix options:** (a) Manual CSV download from Spotrac.com for 2017-22, save to `data/historical/player_salaries_{season}.parquet`; (b) Accept the gap — salary not used in any model through Phase 4. Deferred until Phase 3B confirms it's needed.

---

### GAP-004 — BPM / WS / VORP: Only 2022-25 in aggregate (RESOLVED for external files)
**Severity:** Resolved for Phase 1 Track A  
**Columns:** `bpm`, `obpm`, `dbpm`, `vorp`, `ws`, `ws_per48`  
**Status as of 2026-05-21:** `data/reference/external_rapm_{season}.parquet` now exists for all 8 seasons (2017-18 through 2024-25), fetched from Basketball-Reference. These files are ready for Phase 1 Track A calibration.  
**Remaining gap:** The `bpm`/`ws` columns in `player_profile_aggregate.parquet` still show 0% pre-2022 because `build_profile_aggregate.py` pulls from `data/processed/metrics_linear.parquet` (which only covered 2022-25), not from the external reference files. To close this gap in the aggregate: extend `compute_linear_metrics.py` to cover all 8 seasons using BRef data, then rebuild aggregate. This is Phase 1 Track B work.

---

### GAP-005 — Primary Archetype: 0% Pre-2022
**Severity:** Medium  
**Column:** `primary_archetype` (hard assignment)  
**Root cause:** `primary_archetype` is produced by `compute_player_archetypes.py` which only ran on 2022-25. The archetype computation requires tracking features that are unavailable pre-2022. The soft probabilities (`arch_prob_*`) ARE filled all 8 seasons via RAPM-backbone-only path.  
**Impact:** Phase 4 archetype validation will have no hard-assignment data for pre-2022. Year-over-year stability analysis limited to 2022-25 transitions (3 pairs).  
**Fix:** This is intentional for tracking-dependent archetypes. A reduced `rapm_only` archetype classifier (using only RAPM + box stats) could be built. Deferred to Phase 4.

---

### GAP-006 — off_rating / def_rating: 0% for 2024-25
**Severity:** Investigate  
**Columns:** `off_rating`, `def_rating`  
**Observation:** Both are 0% for 2024-25 despite being 70% filled for 2022-23 and 2023-24. Was 0.99+ for pre-2022 seasons (different source?).  
**Root cause:** Unknown — likely a source change or merge key mismatch in the 2024-25 data. Needs investigation before Phase 1.  
**Fix:** Check what source provides `off_rating` / `def_rating` and trace why 2024-25 is missing.

---

### GAP-007 — Clutch Stats: 0% Pre-2022
**Severity:** Low  
**Columns:** `clutch_*` (8 cols)  
**Root cause:** `fetch_player_clutch_stats.py` only covers 2022-25.  
**Fix:** Extend to earlier seasons if needed. Deferred — clutch data not currently used in any model.

---

### GAP-008 — BKE v30 Phase C: Failed on 8-Season Dataset
**Severity:** Medium (affects v30 model validity)  
**Output:** `reports/dbke_v30_defense_shrinkage.json` — `status: stopped_after_phase_c`  
**Root cause:** Phase C global balance correction (60/40 OBKE/DBKE weighting) makes penalty asymmetry worse when applied to the full 8-season dataset. `penalty_asymmetry_post` (0.315) > `penalty_asymmetry_pre` (0.188). COVID seasons likely drive this.  
**Impact:** v30 defense shrinkage cannot be used as a production BKE version until Phase C parameters are re-calibrated.  
**Fix:** Phase 3B magnitude calibration task — fit Phase C parameters on expanded dataset. Until then, use v28+v27 as production BKE.

---

### GAP-009 — xRAPM: Only 2023-25 (2 seasons)
**Severity:** Low  
**Column:** `xrapm_alpha`  
**Root cause:** xRAPM v2 was only computed for 2023-24 and 2024-25.  
**Fix:** Deferred. xRAPM needs pre-computed priors that require model training.

---

### GAP-010 — Behavioral Rates Pre-2022: League-Avg Imputation for ~22% of Rows
**Severity:** Low  
**Columns:** `pec_behavioral_three_point_rate`, `pec_behavioral_efg`  
**Observation:** 781 of 3,498 pre-2022 rows have exactly `0.405` (league-avg fallback) for `behavioral_three_point_rate`. These are low-sample or injury-return players where box-score rates are unreliable.  
**Impact:** Minimal — `pec_` columns use fallback logic by design. Not used directly in RAPM or BKE scoring.  
**Fix:** None needed. This is intended behavior.

---

### GAP-011 — Lineup Reconstruction: Traded Players Show Partial Season Coverage
**Severity:** Medium  
**Scope:** `possessions_clean_{season}.parquet` → `player_profiles_advanced.parquet` → RAPM  
**Observation (confirmed 2026-05-21):**  
`derive_lineups.py` lineup reconstruction only captures 50–78% of game appearances for players traded mid-season. Example: OG Anunoby 2023-24 — played 36 TOR games + 50 NYK games = 86 total, but the possession data only shows him in ~28 TOR games + ~22 NYK games = 50 games. His RAPM and BKE are computed from only ~58% of his actual season.  
**Root Cause:** `derive_lineups.py` lineup reconstruction logic fails to identify certain players in lineup events, especially early-season games and specific PBP event formats. Not specific to traded players — even full-season TOR players showed ~83% coverage (Barnes 60/72 games).  
**Impact:** Medium — player-level RAPM has higher variance, team aggregates slightly underweight players with reconstruction failures. Walk-forward Brier impact estimated small (~0.002-0.005).  
**Fix Required:** Improve `derive_lineups.py` to handle all PBP event formats and player ID edge cases. Then re-run possession extraction → RAPM → BKE pipeline (~3-4 hr background job). Deferred to focused `derive_lineups.py` session.  
**Workaround:** Current RAPM estimates are directionally correct but noisy. Proceed with Phase 2+ using current data; re-run after `derive_lineups.py` fix.
