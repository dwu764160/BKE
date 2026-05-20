# Data Pipeline Audit & Fix Plan

> **Status:** Brainstormed, not yet implemented. Decisions finalized 2026-05-19.
> **Scope:** Validate foundation data, fix correctness bugs, restore proper formula implementations.
> **Does not change:** BKE metric philosophy, archetype structure, dimension weights.

---

## Overview

The foundation data (box scores, team game logs, PBP events) is trustworthy — 100% W-L match,
PBP within 0.7% of NBA official. Problems are in derived metrics and aggregation logic.

Three independent tracks, can proceed in parallel:

| Track | Problem | Root Cause | Files |
|---|---|---|---|
| A | RAPM over-regularized, elite players suppressed | Alpha too high; no external calibration | `src/modeling/model_rapm.py` |
| B | BPM hand-tuned on 2-3 players | 3 hardcoded constants overriding B-REF formula | `src/data_compute/compute_linear_metrics.py` |
| C | Traded players use wrong team's box stats | Dedup keeps max-minutes team, not TOT row | `src/data_normalize/`, `complete_player_season_stats.parquet` |

---

## Track A: RAPM External Validation & Calibration

### Problem

`rapm_validation_report.json` shows mean delta = -4.46 pts/100 vs xRAPM benchmark.
But xRAPM is computed from **our own box stats** — internal consistency, not external ground truth.
Elite player checks **failed**: Kevin Durant, Anthony Davis, Luka Dončić, James Harden all
rated below expected RAPM floors. This pattern suggests slope compression (elite players
suppressed *more* than role players), which is NOT absorbed by BKE's within-season z-scoring.

### Investigation Plan

**Step 1: Obtain two external RAPM benchmarks**

Source 1 — ESPN RPM (O-RPM + D-RPM, published annually):
- URL pattern: `espn.com/nba/statistics/rpm/_/season/YEAR`
- Or use a cached academic dataset if available (ESPN RPM is widely scraped)

Source 2 — nbarapm.com public RAPM:
- Site publishes downloadable RAPM data per season
- Check: `nbarapm.com` for current download format

Target seasons: 2022-23, 2023-24, 2024-25 for all qualified players (>500 min).

**Step 2: Cross-validate the two external sources first**

Before comparing against our RAPM, check that ESPN RPM and nbarapm.com agree:
- Pearson r between ESPN O-RPM and nbarapm ORAPM (expect >0.80)
- If they disagree significantly, investigate which is more internally consistent
- Use the more reliable one as the primary benchmark

**Step 3: Compare our RAPM against the consensus benchmark**

For each player-season in our dataset:
```python
# Stratify by player tier
tiers = {
    "All-Star": players where external_rapm > 3.0,
    "Starter": 0.5 < external_rapm <= 3.0,
    "Bench": -1.0 < external_rapm <= 0.5,
    "Fringe": external_rapm <= -1.0
}
# Measure our_rapm vs external_rapm by tier
# Key metric: slope of OLS regression (our_rapm ~ external_rapm)
# slope < 1.0 → slope compression (elite players suppressed more)
# intercept != 0 → uniform shift (absorbed by z-scoring)
```

**Step 4: Diagnose based on findings**

| Finding | Diagnosis | Fix |
|---|---|---|
| Slope ≈ 1.0, large negative intercept | Uniform suppression | Accept — z-scoring absorbs it |
| Slope < 1.0 (e.g., 0.5-0.7) | Elite players suppressed more | Reduce ridge alpha |
| Slope ≈ 1.0, intercept ≈ 0, but MAE high | Noise/collinearity | Extend pooling window or BPM prior |
| Systematic elite player fails | Alpha definitely too high | Cross-validate alpha on external benchmark |

**Step 5: Fix (if slope compression confirmed)**

The RAPM pipeline cross-validates alpha via `RidgeCV` on `RAPM_ALPHAS_POOLED = [10, 25, 50, 100, 200, 400, 800, 1600]`.
If external validation shows slope compression, the selected alpha is too high.

Fix options (choose based on diagnosis):
- **Reduce alpha range**: try `[1, 5, 10, 25, 50, 100]` and re-run external validation
- **Add BPM prior**: augment target season possessions with a BPM-derived pseudo-prior
  (already suggested in rapm_validation_report.json, infrastructure partially exists)
- **Apply post-hoc rescaling** as a temporary fix while the root cause is investigated

### Output

New file: `src/modeling/validate_rapm_external.py`
- Loads our RAPM + external benchmark
- Produces per-tier correlation, slope/intercept, MAE table
- Saves to `reports/rapm_external_validation.json`

---

## Track B: BPM Formula Audit & Restoration

### Problem

Three constants in `src/data_compute/compute_linear_metrics.py` are empirically tuned on 2-3 players:
```python
INDIVIDUAL_DEDUCTION_SCALE = 0.28    # "calibrated for Ivey/Clarkson"
MIN_REGRESSION_THRESHOLD = 1500       # changed from 2000 (undocumented)
VERY_LOW_MIN_PENALTY = 1.0            # changed from 3.0 (undocumented)
```

BPM feeds into:
- BKE as a cross-validation/secondary signal
- The planned young player prior (low-sample RAPM stabilization)
- xRAPM computation (RAPM predicted from box stats)

A hand-tuned BPM doesn't generalize across seasons or player-types.

### Fix Plan

**Step 1: Source the correct B-REF formula**
- Basketball-Reference documents their BPM methodology
- Paper: Daniel Myers, "Box Plus/Minus: A Simple Regression-Based +/- Statistic" (2014)
- Cross-reference with B-REF's published update notes

**Step 2: Compare implementation line-by-line against formula**
- Read every coefficient in `compute_linear_metrics.py` (~1300 lines)
- Flag every place our values differ from documented formula
- Specifically audit the three suspect constants

**Step 3: Restore correct formula values**
- Remove `INDIVIDUAL_DEDUCTION_SCALE = 0.28` or restore its documented value
- Restore `MIN_REGRESSION_THRESHOLD` to its formula-specified threshold (likely 2000)
- Restore `VERY_LOW_MIN_PENALTY` to formula-specified value

**Step 4: External validation**
- Scrape B-REF BPM values for all players in our dataset (2022-23 through 2024-25)
- Compare our restored BPM vs B-REF published BPM
- Accept target: Pearson r > 0.80, MAE < 1.5 for qualified players

**Step 5: Validate DWS separately**
The DWS normalization (post-hoc rescaling to 1230 total team WS) was flagged as
potentially shrinking variance artificially. After the BPM fix, check whether DWS
distribution is reasonable vs. B-REF published DWS values.

### Output

- Corrected `src/data_compute/compute_linear_metrics.py` (no player-specific tuning)
- New report: `reports/bpm_external_validation.json` (our BPM vs B-REF BPM)

---

## Track C: Traded Player Handling Fix

### Problem

When loading `complete_player_season_stats.parquet`, traded players have multiple rows
(one per team). The current dedup logic:
```python
# build_player_impact_profiles.py line 587-589
stats = stats.sort_values("minutes_box", ascending=False).drop_duplicates(
    subset=["player_id", "season"], keep="first"
)
```
This keeps only the **highest-minutes team**, discarding the player's stats on other teams.

**Impact:** Trade-deadline buyers (typically playoff contenders who acquire stars at the deadline)
have understated team quality because the acquired player's stats don't reflect their games with
the new team. This affects exactly the teams that matter most for playoff and game predictions.

### Fix Plan

**Step 1: Verify NBA API TOT rows exist in source data**

The NBA API returns a "TOT" team row for traded players — a season-aggregate across all teams.
Check whether `complete_player_season_stats.parquet` includes these TOT rows:
```python
stats = pd.read_parquet("data/historical/complete_player_season_stats.parquet")
traded = stats[stats["TEAM_COUNT"] > 1]
has_tot = (traded["TEAM_ABBREVIATION"] == "TOT").any()
# If True: TOT rows exist but we're discarding them
# If False: TOT rows never fetched — need to re-fetch
```

**Step 2a (if TOT rows exist): Update dedup to use TOT for aggregate metrics**
```python
# New logic:
def resolve_traded_players(stats):
    traded_ids = stats[stats["TEAM_COUNT"] > 1]["player_id"].unique()
    # For traded players: use TOT row for aggregate season stats
    tot_rows = stats[(stats["player_id"].isin(traded_ids)) & (stats["TEAM_ABBREVIATION"] == "TOT")]
    non_traded = stats[~stats["player_id"].isin(traded_ids)]
    # For per-team context: keep individual team rows (for stint system)
    team_rows = stats[(stats["player_id"].isin(traded_ids)) & (stats["TEAM_ABBREVIATION"] != "TOT")]
    return tot_rows, team_rows, non_traded
```

**Step 2b (if TOT rows missing): Add to fetch script**
- Update `src/data_fetch/` to request per-team and TOT rows for all players
- Re-run fetch for 2022-23 through 2024-25

**Step 3: Update stint system to use correct team-specific row for per-team aggregation**

When aggregating player contributions to a team, use:
- **Season-aggregate metrics** (RAPM, BKE, BPM): from TOT row (full season performance)
- **Team-specific context** (games played with team, team synergy): from per-team row

The stint system already has the architecture to do this (`derive_player_team_stints.py`).
The fix is ensuring it reads per-team stats (not the season-aggregate) for team-level
contribution weighting, while using the season-aggregate for individual player quality metrics.

**Step 4: Validate**

Before/after check on known trade-deadline acquisitions:
- 2023-24: Dejounte Murray (ATL → NOP at deadline) — check if NOP's roster quality improves
- 2022-23: Jae'Sean Tate / Eric Gordon (HOU → DEN) — check DEN improvement
- Compare BKE-projected team net ratings before/after fix for buyer teams

### Output

- Updated `src/data_normalize/` fetch/dedup logic for traded players
- Updated `src/player_eval/build_player_impact_profiles.py` to use TOT rows correctly
- Validation: before/after team net rating comparison for 5 known trade-deadline buyer teams

---

## Known Additional Issues (Lower Priority)

These were identified but are not blocking:

| Issue | Severity | Status |
|---|---|---|
| USG_RATE in profiles +0.77pp vs NBA.com | Medium | Use archetype USG_PCT instead until fixed |
| Archetype classification not externally validated | Medium | Exploratory use only; don't treat as hard classification |
| Possession count drift ±3% for some teams | Medium | Acceptable; document in validation report |
| Defensive archetypes: Mikal Bridges 8x duplication in source parquet | Low | Dedup exists in pipeline; clean source file |
| DRTG +0.6 pts/100 vs NBA.com | Low | Systematic but small; document |
| on_off_diff is team-level broadcast as player-level | Low | Rename or document to prevent misinterpretation |

---

## Verification Sequence (After All Three Tracks Complete)

1. Re-run `python3 src/modeling/model_rapm.py` (Track A fix)
2. Run `python3 src/modeling/validate_rapm_external.py` → confirm slope ≈ 1.0, r > 0.75 vs external
3. Re-run `python3 src/data_compute/compute_linear_metrics.py` (Track B fix)
4. Run BPM external validation → confirm r > 0.80 vs B-REF
5. Re-run profile aggregate with fixed traded player handling (Track C fix)
6. Validate team net ratings for trade-deadline buyer teams show expected improvement
7. Re-run BKE scoring pipeline with corrected inputs
8. Re-run `python3 src/simulation/validate_sim.py` → compare backtest metrics before/after
   (expecting improvement in season-win MAE and cross-season player stability)

---

## Relationship to Minute Model Rebuild Plan

See `docs/minute_model_rebuild_plan.md`.

The data pipeline fixes feed into the minute model rebuild:
- Track A (RAPM fix) → improves `prior_orapm` and `prior_drapm` features in temporal minute model
- Track B (BPM fix) → improves the young player prior (xRAPM and BPM used as prior inputs)
- Track C (traded player fix) → improves roster competition features in minute model

The minute model rebuild can proceed before these fixes (it will train on whatever RAPM/BPM values
exist), but the fixes will improve the quality of its inputs and should produce measurably better
holdout MAE after they're applied.
