# Phase 3B Plan — BKE v3.2: PTS / RDIS Architecture Split

> **Last updated:** 2026-05-22  
> **Branch:** `personal`  
> **Status:** PENDING — ready for Opus execution  
> **Prerequisite experiment:** `scripts/experiment_pts_rdis_blend.py` — DONE (results in `reports/experiment_pts_rdis_blend.json`)

---

## v3.2 Philosophy

Three output modes, one underlying decomposition:

| Output | Purpose | Consumer | Notes |
|---|---|---|---|
| **PTS** (Portable Talent Score) | Game outcome prediction + player ranking | Game model pipeline, viewer | Feeds `off_talent_base` / `def_talent_base`; replaces current BKE in aggregation |
| **RDIS** (Role-Dependent Impact Score) | Contextual production analysis | Trade/FA analysis, role fit diagnostics, scouting | Not in game model pipeline; surfaced as secondary viewer signal |
| **BKE** (OBKE + DBKE combined) | Cosmetic display score | Viewer only | Still computed and shown; no longer a pipeline input |

**Why this split:**  
The blend experiment (α grid, 9 variants) showed a monotone gradient: Brier improves as RDIS weight decreases, with pure PTS (α=1.00) giving Brier=0.2248 vs. baseline 0.2400. Continuity-discounted RDIS (Approach B) performed worse than a fixed split at the same α. RDIS carries noise for game prediction but carries signal for interpreting role execution — these are different jobs requiring different designs.

**RDIS is not a ranking metric.** It measures "what did you produce in your specific context this season?" — good for role execution, bad for ordering player ability. PTS (after fixing portability contamination) answers "what can this player do regardless of context," which is the right ranking question.

**Key data point:** Current experiment used `offensive_portable_z` / `defensive_portable_z` as PTS proxies — these still contain RAPM contamination (Layer 1A = 25%). The −0.015 Brier gain is with contaminated PTS. Clean PTS should improve further.

---

## Measurement Framework

Opus must validate every material change against **both** of the following. A change that improves one at the cost of the other should be flagged and discussed, not silently adopted.

### Method 1 — Season-Level Brier (existing)

**Script:** `src/simulation/validate_forecast.py`  
**Input:** `data/processed/forecast/projected_team_features.parquet`  
**Experiment harness:** `scripts/experiment_pts_rdis_blend.py` (accepts patched features)

**Chain:**
```
Player PTS scores × minute shares
    → off_talent_base / def_talent_base per team-season   (team_feature_aggregation.py)
    → team_net_rating_projected = TEAM_SCALE × (off + def) + structure terms
    → per-game win_prob = Φ((μ_home − μ_away + HCA) / σ)   [Gaussian margin model]
    → Brier = mean((win_prob − actual_outcome)²)
```

**Baseline:** 0.2400 (current BKE blend)  
**PTS-only baseline:** 0.2248 (α=1.00 from experiment, still with RAPM contamination)  
**Target:** < 0.220 after architecture fixes  
**Transitions:** 2 clean (2022-23→2023-24, 2023-24→2024-25); never computed within-season

---

### Method 2 — Lineup-Level Plus-Minus (new)

**Script to write:** `scripts/validate_lineup_pts.py`  
**Source data:** `data/historical/pbp_with_lineups_{season}.parquet` (all 8 seasons; lineup columns are lists of player IDs per team_id)

**What it measures:** For each 5-man lineup observed with ≥ 30 possessions in a season, compare the lineup's predicted net rating (from the 5 players' PTS scores) against the lineup's actual on-court net rating from PBP.

**Script specification:**

```python
"""
scripts/validate_lineup_pts.py
=============================================================================
Lineup-Level PTS Validation

For each 5-man lineup observed ≥ MIN_POSSESSIONS in a season:
  predicted_net_rating = TEAM_SCALE * mean(PTS_z for each of 5 players)
  actual_net_rating    = (off_pts - def_pts) / possessions * 100

Metrics:
  - Pearson r (predicted vs. actual lineup net rating)
  - Spearman r (rank correlation)
  - RMSE
  - Calibration: does TEAM_SCALE need to be different at lineup level?

Usage:
  python3 scripts/validate_lineup_pts.py [--season 2023-24] [--min-poss 30]
=============================================================================
"""

# Implementation notes for Opus:
# 1. Load pbp_with_lineups_{season}.parquet — lineup columns are lists stored per team_id
#    column names like 'lineup_1610612754' contain frozen sets of player IDs for that team
# 2. For each event row, extract the two active lineups (home and away)
#    Each lineup column contains a list like [1626167, 1628971, ...]
# 3. Derive possession-level scoring from event sequences:
#    - A possession ends when possession changes (turnover, made shot, defensive rebound, FT sequence)
#    - Score change per possession = points_scored_on_that_possession
# 4. Group by (season, team_id, frozenset(lineup)) → sum(off_possessions), sum(off_points), sum(def_points)
#    net_rating = (off_points - def_points) / possessions * 100
# 5. Load player PTS scores from aggregate/player_profile_aggregate.parquet (portable_talent_z)
# 6. For each lineup, compute mean(PTS_z) of the 5 players → predicted_net_rating = TEAM_SCALE * mean
# 7. Filter to lineups with possessions >= MIN_POSSESSIONS (default 30)
# 8. Compute Pearson r, Spearman r, RMSE, mean calibration error
# 9. Also compute pos-weighted Pearson r (weight each lineup by sqrt(possessions))
# 10. Output to reports/lineup_pts_validation.json

# Key columns in pbp_with_lineups:
#   game_id, period, clock, away_score, home_score, points, event_type
#   lineup_{team_id}: list of 5 player IDs currently on court for that team
#   (lineup columns exist only for the team's own games — others are None)
```

**Metrics and targets:**

| Metric | Current (estimated) | Target after v3.2 |
|---|---|---|
| Pearson r (predicted vs. actual lineup NR) | ~0.10–0.15 (raw BKE, estimated) | ≥ 0.25 |
| Spearman r | ~0.10–0.15 | ≥ 0.25 |
| Possession-weighted Pearson r | ~0.15–0.20 | ≥ 0.30 |
| RMSE (net rating pts/100) | ~12–15 (estimated) | < 10 |

*Estimates: lineup-level is inherently noisy due to small samples. Realistic ceiling is ~0.40 r for a season-average model.*

**Why lineup-level validation matters:**  
Season-level Brier tests whether team-level talent sums predict wins — it cannot detect whether individual player scores are correctly calibrated relative to each other. A model that inflates all players by 2× and deflates TEAM_SCALE by 2× gets the same Brier but has wrong individual scores. Lineup-level plus-minus catches this: only a model that correctly ranks individual contribution within a team will predict which lineups perform well.

---

## PTS Architecture Fixes

These are structural changes required for correctness, independent of the tuning sweep below.  
**All fixes stay within PTS (portable talent) — RDIS is not modified in this phase.**

### Fix 1 — Remove RAPM from Layer 1A (highest priority)

**Current:** `portable_talent_z = 0.25*rapm_z + 0.20*playtype_z + 0.55*dim_z`  
**Proposed:** `portable_talent_z = 0.20*playtype_z + 0.80*dim_z` OR `1.00*dim_z`  
**File:** `src/modeling/layer1_portable_talent.py`, `src/modeling/model_config.py`

RAPM increases 0.17–0.66 pts/100 per unit of teammate quality (PMC evidence). Including it in a "portable" score contaminates PTS with whoever the player's teammates were. Every known metric that claims portability (EPM, LEBRON) uses RAPM as an adjustment on top of a box-score prior, not as a direct input.

**Tuning candidates (see Section 5):** Keep 0.0–0.15 RAPM weight in PTS if it consistently helps both Brier AND lineup r. Remove entirely if evidence is mixed.

### Fix 2 — Replace DRAPM in Dim 6

**Current:** `dim6 = f(drapm, DRTG, d_results_pctl)`  
**Proposed:** Replace with matchup-level signals from `data/matchup/` and `data/tracking/`  
**File:** `src/modeling/layer1_portable_talent.py`

Candidate inputs (use what's available; not all may exist):
- `opponent_efg_on_ball`: opponent eFG% when this player is primary defender
- `contested_shot_rate`: proportion of opponent attempts contested by this player
- `dfg_pct`: defensive field goal percentage against
- `matchup_pts_per_poss`: opponent scoring when matched up against this player
- Fallback if matchup data unavailable for pre-2022: shrink toward archetype Dim 6 mean

### Fix 3 — Prune Dim 5 (ball pressure only)

**Current:** `dim5 = f(STL%, BLK%, deflections, hustle_score, charges, loose_balls)`  
**Proposed:** Remove BLK% and loose_balls (scheme-dependent). Keep: STL_per100, deflections, charges drawn.  
**File:** `src/modeling/layer1_portable_talent.py`

Reduce Dim 5 weight proportionally: 8% → 5–6%. BLK% is highly scheme-driven (drop coverage vs. hedge); STL% is more portable instinct. Loose balls recovered is physically portable but very noisy per season.

### Fix 4 — Move Layer 1B (Playtype Efficiency) out of PTS

**Current:** Playtype PPP surplus is 20% of PTS.  
**Proposed:** Remove from PTS; route to RDIS where context-dependent production belongs.  
**File:** `src/modeling/layer1_portable_talent.py`, `src/modeling/model_config.py`  

Playtype efficiency depends on: quality of screens set for you, play calls the team runs for you, and teammate passing. Keeping it in PTS inflates ballhandlers on pass-heavy teams (Steph, Luka) but penalizes those on isolation-heavy systems. It belongs in RDIS.

---

## Section 5 — Wide Tuning Candidate Table

This section defines every numerical parameter Opus is free to tune in order to optimize PTS on both measurement levels. **Constraints are listed under each group** — stay within them. Test each candidate isolated if possible, or in a controlled sweep; do not randomly change 10 things at once.

### 5A — Layer 1 Top-Level Weights

| Parameter | Current | Candidate Range | Constraint |
|---|---|---|---|
| `w_rapm_backbone` | 0.25 | 0.00 → 0.15 | Remove or minimize (Fix 1 above) |
| `w_playtype_efficiency` | 0.20 | 0.00 → 0.10 | Remove or minimize (Fix 4 above) |
| `w_dimension_model` | 0.55 | 0.75 → 1.00 | Must increase as other weights decrease; sum must = 1.0 |

### 5B — Dimension Weights (Layer 1C)

All 9 weights must remain positive and sum to 1.00. Constraint: offensive dims (1, 2, 3, 7, 9) must sum to ≥ 0.52. Defensive dims (4, 5, 6, 8) ceiling at 0.40 total.

| Dim | Name | Current | Candidate Range |
|---|---|---|---|
| 1 | Shooting Gravity | 0.14 | 0.12 → 0.20 |
| 2 | Playmaking | 0.14 | 0.12 → 0.18 |
| 3 | Self-Creation | 0.14 | 0.12 → 0.18 |
| 4 | Defensive Versatility | 0.12 | 0.08 → 0.14 |
| 5 | Defensive Playmaking (ball pressure only post-fix) | 0.08 | 0.04 → 0.08 |
| 6 | Defensive Impact (post matchup-data fix) | 0.10 | 0.06 → 0.12 |
| 7 | Driving Gravity | 0.10 | 0.08 → 0.14 |
| 8 | Extra Possession | 0.08 | 0.06 → 0.10 |
| 9 | Turnover Control | 0.10 | 0.06 → 0.12 |

*Calibration method: after architecture fixes, run ridge regression of dim scores against held-out team offensive / defensive efficiency differentials across 6 non-COVID seasons. Use regression coefficients (renormalized to sum to 1.0, subject to constraints above) as the new weights. This is data-driven, not hand-tuned.*

### 5C — Archetype Neutralization

Neutralization removes archetype baseline from each player's raw dimension score. Currently applies uniformly at strength 1.0.

| Parameter | Current | Candidate Range | Notes |
|---|---|---|---|
| Neutralization strength | 1.0 | 0.5 → 1.5 | < 1.0 = partial neutralization; > 1.0 = overcorrects |
| Apply to offensive dims only | No | Yes / No | Defensive dims may not need neutralization (less archetype-predictable) |
| Apply to all 9 dims | Yes | Yes / selective | Can disable per-dim if correlation between dim and archetype is low |

### 5D — Dim Score Clipping

| Parameter | Current | Candidate Range |
|---|---|---|
| Per-dim z-score clip | None | ±2.0 → ±3.5 |
| Final PTS_z clip before aggregation | None | ±3.0 → ±4.0 |
| Defensive dim clip tighter than offensive | No | Try ±2.5 def, ±3.5 off |

### 5E — Team Aggregation (team_feature_aggregation.py)

| Parameter | Current | Candidate Range | Notes |
|---|---|---|---|
| `TEAM_SCALE` | 20.0 | 18.0 → 28.0 | Implied ~25 from external benchmarks; tune on Brier + lineup calibration jointly |
| Minute-share weighting | linear (min/total) | linear, sqrt, log | sqrt/log compresses star dominance; test if it improves lineup-level r |
| Minimum minute threshold for inclusion | ~200 min | 150 → 300 min | Players with < threshold are noise; affects team totals |
| Star amplification (top-N players) | None | 1.0 → 1.3× for top-2 | Test: does upweighting top players improve lineup-level prediction? |
| PTS score recency averaging | 1 season | 1 season, 0.7×current + 0.3×prior | Multi-season smoothing may reduce noise; test on Brier only |

### 5F — OBKE/DBKE Overall Split

Applied in `construct_bke_scores_v27.py`. Affects cosmetic BKE display AND the offensive/defensive talent bases separately.

| Parameter | Current | Candidate Range | Notes |
|---|---|---|---|
| OBKE weight in final BKE | 0.60 | 0.55 → 0.65 | Offense is more predictive empirically; ceiling at 0.65 |
| PTS_O / PTS_D relative weight in game model | 60/40 | 55/45 → 65/35 | May differ from cosmetic BKE split |

### 5G — Small-Sample Bayesian Shrinkage

Currently not applied to PTS. Adding per-dimension shrinkage toward the archetype mean for players with < N seasons of data may reduce noise in early-career or low-minute players.

| Parameter | Candidate Range | Notes |
|---|---|---|
| Shrinkage strength (per dim) | 0.0 → 0.4 | 0.0 = off; 0.4 = 40% toward archetype mean |
| Minimum seasons before full trust | 1 → 3 | Players with < N seasons get stronger shrinkage |
| Apply only to defensive dims | True / False | Defensive stats are noisier; shrinkage may help more |

### 5H — Dim 6 Input Weighting (post Fix 2)

If multiple matchup signals are available, blend them:

| Signal | Candidate weight | Notes |
|---|---|---|
| `opponent_efg_on_ball` | 0.4 → 0.6 | Core matchup outcome |
| `contested_shot_rate` | 0.2 → 0.3 | Activity level; portable |
| `matchup_pts_per_poss` | 0.1 → 0.3 | Outcome; partially portable |
| `dfg_pct` | 0.0 → 0.2 | Noisier; use if others missing |
| DRAPM residual (after matchup controls) | 0.0 → 0.15 | Only the residual not explained by matchup data |

---

## Validation Gates

Before committing any change, measure both methods. Report the delta in both.

| Gate | Condition | Action |
|---|---|---|
| Architecture fix (Fixes 1–4) | Brier improves OR lineup r improves, other metric does not regress | Accept |
| Tuning change (Section 5) | Both metrics improve OR one improves and other unchanged | Accept |
| Either metric regresses | Revert; note why |
| Large Brier gain but lineup r decreases | Flag to user — may be overfitting to season-level signal |

**Braun sanity check** (not a gate, a diagnostic): After each material PTS change, print Christian Braun's PTS_z ranking percentile within Off-Ball Finisher / 3-and-D Wing archetypes. Should be 50th–70th percentile, not 90th+. If still 90th+, RAPM contamination is not resolved.

**Star sensitivity check**: After each material change, print PTS_z for Curry, Luka, KD. Should be top-15 among all qualified players in their peak seasons without any explicit OBKE/DBKE reweighting hack.

---

## Optimization Mandate for Opus

**You have full freedom to:**
- Modify weights in Section 5 within stated ranges
- Test combinations of fixes from Section 4
- Write new helper functions in any modeling file
- Add new measurement columns to the decomposition output
- Design and run the lineup validation script
- Run the Brier experiment harness with any patched features

**You must not:**
- Change the archetype system (offensive v4.3, defensive v3.4) — no archetype redefinitions
- Change the Gaussian game model structure or HCA/B2B coefficients
- Remove the OBKE/DBKE split — it must remain as two separable components
- Change the minute model (Phase 2 complete) or the walk-forward harness
- Introduce signals that are not available for pre-2022 seasons without a pre-2022 fallback
- Make PTS a black box — all 9 dimensions and their weights must remain interpretable and logged
- Adopt any change that fails both measurement gates simultaneously

**Do not break basketball intuition.** The dimension model has meaning: Dim 1 is shooting gravity, Dim 3 is self-creation. If a proposed weight makes Shooting Gravity worth 3% of PTS, reject it regardless of Brier — it violates what the metric is supposed to measure. When in doubt, bias toward the basketball-rational explanation over the regression coefficient.

---

## Sub-task Checklist

| Sub-task | Priority | Depends on | Status |
|---|---|---|---|
| 3B.0 Write `validate_lineup_pts.py` | HIGH | pbp_with_lineups data | PENDING |
| 3B.1 Establish pre-fix baselines on both metrics | HIGH | 3B.0 | PENDING |
| 3B.2 Fix 1: Remove Layer 1A RAPM; re-run both metrics | HIGH | 3B.1 | PENDING |
| 3B.3 Fix 4: Move playtype efficiency (1B) to RDIS | HIGH | 3B.2 | PENDING |
| 3B.4 Fix 2: Replace Dim 6 DRAPM with matchup data | MEDIUM | 3B.1 (independent) | PENDING |
| 3B.5 Fix 3: Prune Dim 5 to ball-pressure only | MEDIUM | 3B.1 (independent) | PENDING |
| 3B.6 Dimension weight calibration (ridge regression) | MEDIUM | 3B.4 + 3B.5 complete | PENDING |
| 3B.7 Sweep Section 5 tuning candidates (5A–5H) | MEDIUM | 3B.2 + 3B.3 | PENDING |
| 3B.8 Adopt best config; update model_config.py | HIGH | 3B.6 + 3B.7 | PENDING |
| 3B.9 Wire PTS into team_feature_aggregation.py | HIGH | 3B.8 | PENDING |
| 3B.10 Final Brier + lineup validation; write report | HIGH | 3B.9 | PENDING |

**Run order:** 3B.0 → 3B.1 → (3B.2, 3B.3 sequential) then (3B.4, 3B.5 parallel) → 3B.6 → 3B.7 → 3B.8 → 3B.9 → 3B.10

---

## Files to Modify

| File | Changes |
|---|---|
| `src/modeling/layer1_portable_talent.py` | Remove Layer 1A (RAPM), Fix Dim 6 (matchup data), Fix Dim 5 (ball pressure only) |
| `src/modeling/model_config.py` | Updated weights for all Section 5 candidates; `w_rapm_backbone=0.0`, `w_playtype_efficiency=0.0`, `w_dimension_model=1.0` as starting point |
| `src/profile_aggregate/team_feature_aggregation.py` | Use PTS scores (`portable_talent_z`) instead of BKE scores for talent base computation |
| `scripts/validate_lineup_pts.py` | **New** — lineup-level plus-minus validation script (see Method 2 spec above) |
| `scripts/experiment_pts_rdis_blend.py` | Already complete (blend sweep done) |
| `docs/reference/basketball-intuitions.md` | Update Section 2 (architecture), Section 6 (net rating construction uses PTS, not BKE) |
| `docs/multiple-versions.md` | Add v3.2 entry: PTS/RDIS split, BKE cosmetic-only |

---

## What Does NOT Change

- Archetype definitions (v4.3 offensive, v3.4 defensive)
- OBKE/DBKE decomposition structure (inputs change; structure stays)
- Game model (Gaussian margin, HCA, B2B coefficients)
- Interaction matrix (eliminated in Phase 4.6 — stays eliminated)
- Minute model (Phase 2 complete)
- Walk-forward harness (`validate_forecast.py`) — re-run only
- RDIS computation — redesign deferred; RDIS is cosmetic for now
