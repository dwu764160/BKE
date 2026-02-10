# BKE Data Integrity Validation Report

**Date:** February 9, 2026 (updated after possession/USG fixes)  
**Season tested:** 2024-25 (primary), 2022-23 and 2023-24 (secondary)  
**Sample size:** 64 players across all archetypes, positions, and minutes tiers

---

## Executive Summary

After fixing two root-cause issues (possession parser gaps + archetype USG formula), our pipeline produces **reliable box-score totals**, **internally consistent rate stats**, and **near-exact archetype USG%**.

### Fixes Applied
1. **derive_possessions.py**: Added handlers for VIOLATION-turnovers (traveling, 5-sec, 8-sec, 3-sec) and mid-game JUMP_BALL held balls
2. **compute_player_archetypes.py**: USG_PCT now loaded from official NBA.com advanced stats; proxy constant fixed from 2.4→2.0 for edge-case fallback

### Before → After

| Metric | Before | After | Δ |
|--------|--------|-------|---|
| Possession drift | **-2.3%** | **-0.24%** | 10× improvement |
| Archetype USG vs NBA.com | **+3.68pp** | **+0.00pp** | Fixed |
| ORTG vs NBA.com | **+2.4 pts** | **+0.0 pts** | Fixed |
| DRTG vs NBA.com | **+2.6 pts** | **+0.6 pts** | Improved |
| Profile USG vs NBA.com | **+0.77pp** | **+0.77pp** | Residual |
| League-wide possession deficit | **-2.5%** | **-0.7%** | 3.5× improvement |
| Pace (poss/team/game) | **98.8** | **100.8** | Matches NBA avg |

---

## Tier 1: Box Score Totals ✅ PASS

| Metric | Result |
|--------|--------|
| FGM | **63/63 exact matches** |
| FGA | 61/63 exact (2 traded-player edge cases) |
| GP | 62/63 exact (1 edge case) |
| Minutes | Mean drift **0.25%**, max 1.71% |

---

## Tier 2: Internal Consistency ✅ PERFECT

| Check | Result |
|-------|--------|
| PTS = (FGM-FG3M)×2 + FG3M×3 + FTM | **ALL 63 PASS** |
| REB = ORB + DRB | **ALL 63 PASS** |
| TS% recomputed from raw | **Max error: 0.000000** |
| eFG% recomputed from raw | **Max error: 0.000000** |

---

## Tier 3: Rate Stats vs NBA.com ✅ GOOD (was ⚠️)

| Stat | Mean Δ vs NBA.com | Status |
|------|-------------------|--------|
| TS% | **0 (exact)** | ✅ |
| eFG% | **0 (exact)** | ✅ |
| USG% (profiles) | **+0.77pp** | ⚠️ residual |
| USG% (archetypes) | **+0.00pp** | ✅ **FIXED** |
| ORTG | **+0.0 pts** | ✅ **FIXED** (was +2.4) |
| DRTG | **+0.6 pts** | ✅ improved (was +2.6) |

---

## Tier 4: Possession Count ✅ FIXED (was ❌)

| Metric | Before Fix | After Fix |
|--------|-----------|-----------|
| Mean drift | -2.3% | **-0.24%** |
| Median drift | -2.1% | **-0.05%** |
| Range | [-5.0%, -0.9%] | **[-3.1%, +1.9%]** |
| Players >2% drift | ~30/63 | **2/63** |

New possession end reasons: JUMP_BALL (~2,100/season), VIOLATION-TURNOVER (+2,000/season)

---

## Tier 5: USG% Reconciliation ✅ (archetype FIXED)

| Source | Mean Δ vs NBA.com | Status |
|--------|-------------------|--------|
| Archetype (USG_PCT) | **+0.00pp** | ✅ **FIXED** — now uses official NBA.com values |
| Profile (USG_RATE) | **+0.77pp** | ⚠️ residual — uses PBP-derived TEAM_PLAYS_ON_COURT |

---

## Tier 6: League-Wide Sanity

| Metric | Before | After |
|--------|--------|-------|
| Total player-POSS deficit | -31,628 (-2.5%) | **-8,363 (-0.7%)** |
| Pace (poss/team/game) | 98.8 | **100.8** |

---

## Tier 7: vs Basketball-Reference Ground Truth

| Stat | Exact Matches |
|------|--------------|
| GP | 62/63 |
| FGM | **63/63** |
| FGA | 61/63 |
| PTS | 62/63 |
| AST | 60/63 |
| MIN | mean Δ0.25% |

---

## Remaining Items

### Low-priority residual issues:
1. **Profile USG_RATE** still +0.77pp vs NBA.com — uses PBP-derived TEAM_PLAYS_ON_COURT which is slightly inflated
2. **~0.7% league-wide possession deficit** — remaining edge cases (FT-endings without rebounds, some game-specific PBP anomalies)
3. **DRTG** still +0.6 pts vs NBA.com on average — related to #2

### Validation infrastructure:
4. Run `tests/validate_data_integrity.py` after any pipeline changes to track regression
5. Consider adding automated CI checks for possession count drift

---

## Files

| File | Purpose |
|------|---------|
| `tests/validate_data_integrity.py` | Main validation script (64-player, 7-tier) |
| `tests/bref_ground_truth_2024-25.csv` | Basketball-Reference data for 64 players |
| `tests/VALIDATION_REPORT.md` | This report |
| `src/features/derive_possessions.py` | Fixed: VIOLATION-TURNOVER + JUMP_BALL handlers |
| `src/data_compute/compute_player_archetypes.py` | Fixed: USG_PCT from official stats |
