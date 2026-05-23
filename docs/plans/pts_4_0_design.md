# PTS 4.0 Design — Implementer-Ready Spec

**Date:** 2026-05-23  
**Status:** DRAFT — pending user approval before implementation  
**Prereq:** `docs/v3_3_phase_a_validation.md` (Phase A findings, especially the OOS reversal)  
**Target reader:** an implementer (Sonnet) who has not seen this conversation and must build the system from this document plus the cited files.

---

## 1. Background

Phase A established four facts that shape this design:

- **OOS validation reversal:** PTS v3.2 beats a leakage-free naive per-100 NetRtg baseline by +0.118 weighted Pearson r on lineup prediction (pooled across 7 walk-forward transitions, 5,143 lineups). The dimension-model abstraction works — it just needs to be evolved at the margins, not rebuilt around on-court signal.
- **Defense is the weakest axis** (lineup wr ≈ 0.28 vs offense ≈ 0.32).
- **Year-over-year PTS_z correlation is r ≈ 0.56**, lower than a portable talent score should have (target ≥ 0.70).
- **No per-player uncertainty** exists — a 200-min rookie and a 3,000-min veteran get equal weight in team aggregation.

v4.0 implements **four independent improvements**, each behind its own config flag, each with its own ablation harness. Improvements compose linearly; the user must be able to A/B test each one independently before stacking. **Do not stack improvements without first measuring each in isolation against the locked Phase-A baseline.**

This doc covers **Improvements A and C only** (defense redesign and multi-season Bayesian smoothing) per user direction; improvements B (uncertainty) and D (orthogonalization) are noted at the end of §13 as v3.4 / v4.0 follow-ons.

---

## 2. Non-Goals (Things v4.0 MUST NOT Change)

These are out of scope. Touching them is grounds for rejecting the implementation:

- The dimension definitions (Dim 1–9), archetype neutralization step, position-conditional z-scoring, or any code in `src/modeling/layer1_portable_talent.py`. v4.0 is a new module that consumes layer1's output, not a rewrite.
- `src/profile_aggregate/team_feature_aggregation.py` — untouched.
- `src/simulation/validate_forecast.py` — untouched.
- `src/simulation/game_model.py` — untouched.
- Existing `data/processed/bke/bke_v28_decomposition.parquet` — used read-only.
- Existing `data/processed/bke/pts_v32.parquet` — v3.2 production output, preserved as a comparator.
- Existing `model_config.py` `PtsV32Config` block — preserved; new v4.0 config is additive.
- The `scripts/validate_lineup_pts_v2.py` harness is the **canonical** validation tool. Do not modify it. If a new validation feature is needed, add it as a new flag.

---

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  v2.7 BKE pipeline (unchanged)                                       │
│  → data/processed/bke/bke_v28_decomposition.parquet (5,848 rows)     │
└─────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
       ┌───────────────────────────────────────────────┐
       │  v3.2 post-hoc adjustments (unchanged)         │
       │  scripts/build_pts_v32.py                      │
       │  → data/processed/bke/pts_v32.parquet          │
       └───────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  NEW: v4.0 post-processing layer                                     │
│                                                                      │
│  Improvement A — Multi-season Bayesian smoothing                     │
│    scripts/pts_v40_multiseason.py                                   │
│    Input:  pts_v32.parquet (+ historical decomp for prior seasons)  │
│    Output: data/processed/bke/pts_v40_a.parquet                     │
│                                                                      │
│  Improvement C — Defense redesign                                    │
│    scripts/pts_v40_defense.py                                       │
│    Input:  pts_v32.parquet + defensive_archetypes_v2.parquet +      │
│            matchups_rollup.parquet + pbp_with_lineups_*.parquet     │
│    Output: data/processed/bke/pts_v40_c.parquet                     │
│                                                                      │
│  Composite v4.0 (after individual ablations pass):                   │
│    scripts/build_pts_v40.py                                         │
│    Input:  pts_v40_a.parquet + pts_v40_c.parquet                    │
│    Output: data/processed/bke/pts_v40.parquet +                     │
│            data/processed/forecast/projected_team_features_v40.pqt  │
└─────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
       ┌───────────────────────────────────────────────┐
       │  Validation (existing scripts/validate_lineup_pts_v2.py    │
       │  + src/simulation/validate_forecast.py)        │
       │  → reports/lineup_v2_v40_<variant>.json        │
       │  → reports/forecast_game_validation.json       │
       └───────────────────────────────────────────────┘
```

Key principle: **each improvement writes its OWN parquet output**. The composite is built only after each individual output passes its ablation gate. This makes regressions trivially attributable.

---

## 4. Improvement A — Multi-Season Bayesian Smoothing

### 4.1 Goal

Raise the year-over-year correlation of `portable_talent_z` (and `pts_o_v32`, `pts_d_v32`) from the current r ≈ 0.56 to ≥ 0.70 for players with ≥ 2 prior seasons of data, **without degrading the level of player differentiation** (PTS_z std stays in the 0.15–0.35 range per season).

### 4.2 Method

Hierarchical empirical-Bayes smoothing per player:

```
PTS_o_v40_a(player, season) = w(player, season) · PTS_o_v32(player, season)
                            + (1 − w(player, season)) · PriorEstimate(player, season)

PriorEstimate(player, season) = weighted average of last K prior seasons' PTS,
                                with weights decaying geometrically:
                                  season t−1: weight 1.0
                                  season t−2: weight 0.55
                                  season t−3: weight 0.30

w(player, season) = current_season_possessions / (current_season_possessions + tau)

tau = 1500   (hyperparameter; see §4.4 for sweep)
K   = 3      (max prior seasons looked back)
```

For players with no prior seasons (rookies), `PriorEstimate = league-mean PTS for their archetype`.

For players with 1 prior season but < 500 possessions in current season, weight w defaults to current-poss / 2000 (slightly less reactive).

### 4.3 Data sources

| Required column | Source parquet | Notes |
|---|---|---|
| `player_id`, `season` | `data/processed/bke/pts_v32.parquet` | join key |
| `pts_o_v32`, `pts_d_v32` | `data/processed/bke/pts_v32.parquet` | current PTS |
| Prior seasons' `pts_o_v32`, `pts_d_v32` | derived by sorting same file by `player_id, season` | shift(1), shift(2), shift(3) |
| `possessions_played` | `data/processed/bke/bke_v28_decomposition.parquet` | join on `player_id, season`. Required for `w` calculation. |
| `primary_archetype` | `bke_v28_decomposition.parquet` | for rookie cohort means |

If `possessions_played` is missing or NaN for a player-season, default to 1500 (treat as full weight). Document this fallback in the report.

### 4.4 Hyperparameters (sweep)

| Param | Default | Sweep values |
|---|---|---|
| `tau` (poss equiv of prior) | 1500 | 800, 1200, 1500, 2000, 3000 |
| Geometric decay rate | 0.55 | 0.40, 0.55, 0.70 |
| `K` max prior seasons | 3 | 2, 3, 4 |
| Min poss for rookie shrinkage exception | 500 | fixed |

Sweep by Brier + lineup wr (Phase A canonical harness). Adopt the joint optimum.

### 4.5 Code spec

**File:** `scripts/pts_v40_multiseason.py`

**Config dataclass** (in `model_config.py` as `PtsV40MultiSeasonConfig`):

```python
@dataclass
class PtsV40MultiSeasonConfig:
    tau_possessions: int = 1500
    geometric_decay: float = 0.55
    k_max_prior: int = 3
    rookie_poss_threshold: int = 500
    poss_fallback: int = 1500  # when possessions_played NaN
    archetype_mean_prior: bool = True  # rookies → archetype cohort mean
```

**CLI:**

```bash
python3 scripts/pts_v40_multiseason.py \
    --pts-v32 data/processed/bke/pts_v32.parquet \
    --decomp data/processed/bke/bke_v28_decomposition.parquet \
    --output data/processed/bke/pts_v40_a.parquet \
    [--tau 1500] [--decay 0.55] [--k 3]
```

**Output parquet schema:**

| Column | Dtype | Notes |
|---|---|---|
| `player_id` | string | |
| `season` | string | |
| `pts_o_v32` | float64 | unmodified, for reference |
| `pts_d_v32` | float64 | unmodified, for reference |
| `pts_o_v40_a` | float64 | smoothed offense |
| `pts_d_v40_a` | float64 | smoothed defense |
| `prior_weight_o` | float64 | (1 − w) used for offense |
| `n_prior_seasons` | int | number of prior seasons available for this player |
| `current_possessions` | int | possessions in current season |
| `prior_source` | string | "prior_seasons", "rookie_archetype_mean", "fallback" |

### 4.6 Validation gate (must pass before adoption)

Run `scripts/validate_lineup_pts_v2.py` with `--pts-file data/processed/bke/pts_v40_a.parquet --pts-col pts_o_v40_a --pts-col-d pts_d_v40_a --label v40a`.

Pass criteria — **all four** must hold against `pts_v32.parquet` baseline:

1. **YoY PTS_z correlation** ≥ 0.70 (vs 0.56 baseline). Diagnostic: per-player PTS_total[N] vs PTS_total[N+1] correlation, restricted to players with ≥ 1000 possessions in both seasons.
2. **Lineup weighted Pearson r (joint)** ≥ pts_v32 baseline 0.331 (no regression).
3. **Aggregate Brier** ≤ pts_v32 baseline 0.2352 (no regression). Use `build_patched_features` from `scripts/build_pts_v32.py` adapted for v40_a columns.
4. **Star sanity:** Stephen Curry / Kevin Durant / Luka / SGA / Giannis all in top-30 PTS_total per season (current PTS_total = pts_o_v40_a + pts_d_v40_a) for 2022-23..2024-25. None drops more than 8 ranks vs v3.2.

If 1–3 pass but star sanity (4) fails, the smoothing is too aggressive — reduce tau and retry.

### 4.7 Known risks

- **Overshrinking to prior:** if tau too small, stars regress toward mean too hard and lineup r drops.
- **Rookie cohort mean is poor prior:** archetype-mean works for veterans but rookies may be mis-archetyped. Mitigation: use position-bucket mean if archetype confidence < 0.5.
- **Geometric decay choice is fragile:** sweep both `0.40` and `0.70` and confirm `0.55` is not at the edge of a cliff.

---

## 5. Improvement C — Defense Redesign

### 5.1 Goal

Raise lineup-defense weighted Pearson r (against actual lineup DRtg, sign-flipped) from the current 0.282 to ≥ 0.32. This is the largest measurable single gap in v3.2.

### 5.2 Method

Replace the current `pts_d_v32` for high-minute defenders with a composite that blends three orthogonal defensive signals, each shrunken individually before combination:

```
pts_d_v40_c(player, season) =
    γ_match  * matchup_z(player, season)        # opponent-controlled box differential
  + γ_lineup * lineup_residual_z(player, season) # within-team on/off after matchup control
  + γ_arch   * archetype_baseline_z(player, season) # archetype-conditional prior

where the three γ's sum to 1.0 (free hyperparameters), and each input z is:

  matchup_z          = z(0.40 · z(d_results_pctl, sign +1)        # opp eFG% allowed pctl
                          + 0.30 · z(D_FG_DIFF, sign −1)          # def FG% vs expected
                          + 0.15 · z(contested_shots_pctl, sign +1)
                          + 0.15 · z(rim_protection_index_pctl, sign +1))

  lineup_residual_z  = z(per100_def_observed − per100_def_expected_from_matchup)
                       per season, possession-weighted within season

  archetype_baseline_z = z(season-archetype mean of dim_defensive_versatility_z
                            + dim_defensive_impact_z + dim_defensive_playmaking_z)
                       computed per (season, primary_defensive_archetype) cohort

Final clip: ±3.5 σ.
```

`per100_def_expected_from_matchup` is a per-player season-level regression of observed lineup-aggregated per-100 DRtg against a player's matchup_z, fit on the same data (within-season; OOS validation only at the lineup harness level). This is the key insight: it isolates the "lineup chemistry" residual that pure matchup misses.

### 5.3 Data sources

| Required column | Source parquet | Notes |
|---|---|---|
| `player_id`, `season` | `data/processed/bke/pts_v32.parquet` | join key |
| `pts_d_v32` | `data/processed/bke/pts_v32.parquet` | for offense unchanged path |
| `d_results_pctl`, `D_FG_DIFF`, `contested_shots_pctl`, `rim_protection_index_pctl` | `data/processed/defensive_archetypes_v2.parquet` | matchup signals; available all 8 seasons |
| `defensive_archetype` | `data/processed/defensive_archetypes_v2.parquet` | for archetype baseline |
| `dim_defensive_versatility_z`, `dim_defensive_impact_z`, `dim_defensive_playmaking_z` | `data/processed/bke/bke_v28_decomposition.parquet` | for archetype baseline composite |
| Lineup PBP | `data/historical/pbp_with_lineups_{season}.parquet` | for per100 DRtg per player (use the helper `build_player_on_court_nrtg` from `scripts/validate_lineup_pts_v2.py`) |

### 5.4 Hyperparameters (sweep)

| Param | Default | Sweep values |
|---|---|---|
| `γ_match` | 0.45 | 0.30, 0.45, 0.60 |
| `γ_lineup` | 0.35 | 0.20, 0.35, 0.50 |
| `γ_arch` | 0.20 | 0.10, 0.20, 0.30 |
| Final clip | ±3.5 | ±2.5, ±3.5, ±5.0 |

Sweep on the joint (lineup wr_defense + aggregate Brier). Pick the joint optimum within constraint that no individual gamma drops below 0.10 (interpretability bound).

### 5.5 Code spec

**File:** `scripts/pts_v40_defense.py`

**Config dataclass** (in `model_config.py` as `PtsV40DefenseConfig`):

```python
@dataclass
class PtsV40DefenseConfig:
    gamma_match: float = 0.45
    gamma_lineup: float = 0.35
    gamma_arch: float = 0.20
    matchup_component_weights: Dict[str, float] = field(default_factory=lambda: {
        "d_results_pctl": 0.40,                # sign +1 (higher pctl = better defense)
        "D_FG_DIFF": -0.30,                    # sign -1 (lower DFG_DIFF = better defense; invert)
        "contested_shots_pctl": 0.15,          # sign +1
        "rim_protection_index_pctl": 0.15,     # sign +1
    })
    final_clip: float = 3.5
    # Pre-2022 fallback: if matchup data missing for season < 2022-23,
    # use defensive_archetypes_v2 d_results_pctl which IS in older seasons too
    pre2022_fallback: bool = True
```

**CLI:**

```bash
python3 scripts/pts_v40_defense.py \
    --pts-v32 data/processed/bke/pts_v32.parquet \
    --decomp data/processed/bke/bke_v28_decomposition.parquet \
    --def-arch data/processed/defensive_archetypes_v2.parquet \
    --pbp-dir data/historical \
    --output data/processed/bke/pts_v40_c.parquet
```

**Output parquet schema:**

| Column | Dtype | Notes |
|---|---|---|
| `player_id` | string | |
| `season` | string | |
| `pts_o_v32` | float64 | unmodified; offense is NOT changed by Improvement C |
| `pts_d_v32` | float64 | preserved for reference |
| `pts_d_v40_c` | float64 | new defense PTS |
| `matchup_z` | float64 | trace col |
| `lineup_residual_z` | float64 | trace col |
| `archetype_baseline_z` | float64 | trace col |
| `matchup_data_source` | string | "live", "pre2022_fallback" |

### 5.6 Validation gate

Run `scripts/validate_lineup_pts_v2.py` with `--pts-file pts_v40_c.parquet --pts-col pts_o_v32 --pts-col-d pts_d_v40_c --label v40c`.

Pass criteria — **all three** must hold:

1. **Defense weighted r** ≥ 0.32 (vs pts_v32 baseline 0.282). At least +0.038 wr improvement.
2. **Offense weighted r** within 0.005 of baseline (must not be harmed; offense isn't touched but Phase A showed indirect effects via team aggregation).
3. **Aggregate Brier** ≤ pts_v32 baseline 0.2352 (no regression).

Bonus diagnostics (not gates):
- Top-15 defenders sanity: Jrue Holiday, Marcus Smart, Draymond Green, Bam Adebayo, Rudy Gobert should all be top-30 by `pts_d_v40_c` in 2023-24.
- Archetype breakout: report defense r per defensive archetype to see if any cohort regresses.

### 5.7 Known risks

- **Pre-2022 matchup data gaps:** `defensive_archetypes_v2.parquet` covers all 8 seasons per Phase A discovery, so this is small. The `D_FG_DIFF` and `contested_shots_pctl` may be NaN for some early-season players; fallback to median is acceptable.
- **lineup_residual_z is a within-season regression:** introduces a degree-of-freedom risk if the regression overfits. Mitigation: use simple linear OLS only; no interactions; document `n_obs` per season.
- **Sign convention bugs are easy here:** test `D_FG_DIFF` sign on a known elite defender — Rudy Gobert should have negative `D_FG_DIFF` (opponents shoot worse than expected against him), so the sign-inverted `−D_FG_DIFF` gives him a positive contribution. Verify in unit test before commit.

---

## 6. Composite v4.0 Build (only after both A and C pass individually)

**Sequence required:** A must pass its gate before C is built on top; both must pass before composite.

If both pass, compose:

```
pts_o_v40 = pts_o_v40_a       # offense gets multi-season smoothing only
pts_d_v40 = blend(pts_d_v40_a, pts_d_v40_c, weight=0.4)   # defense gets smoothing AND redesign
              where blend is convex: 0.4 · v40_a + 0.6 · v40_c
              (defense redesign dominates; smoothing adds stability on top)
```

The 0.6/0.4 split is a hyperparameter; sweep at composite time over {0.5/0.5, 0.6/0.4, 0.7/0.3}.

**File:** `scripts/build_pts_v40.py`

Reads `pts_v40_a.parquet` and `pts_v40_c.parquet`, produces `pts_v40.parquet` and `projected_team_features_v40.parquet` (using the same patch logic as `scripts/build_pts_v32.py`).

### 6.1 Final composite validation

All four gates required:

1. Aggregate Brier ≤ 0.230 (improvement of −0.005 over v3.2's 0.2352)
2. Lineup weighted Pearson r (joint) ≥ 0.36 (improvement of +0.025 over 0.331)
3. Lineup weighted defense r ≥ 0.32 (matches Improvement C alone)
4. YoY PTS_z r ≥ 0.70 (matches Improvement A alone)

If any composite gate fails but individual gates passed, the composite blend ratio is wrong — re-sweep.

---

## 7. Validation Methodology

**Canonical harness for everything:** `scripts/validate_lineup_pts_v2.py` (Phase A output, locked).

**Brier:** `python3 src/simulation/validate_forecast.py --features-path <v40 features parquet>`.

**Baseline comparators** (already produced in Phase A):
- `reports/lineup_v2_v32.json` — v3.2 PTS lineup metrics (use as "no change" comparator)
- `reports/lineup_baseline_v28.json` — v2.7 PTS lineup metrics
- Live `validate_forecast.py` baseline: 0.2420 (full v2.7 BKE pipeline output)
- Live `validate_forecast.py` v3.2 production: 0.2352

**Per improvement ablation reports go to:**
- `reports/v40_a_multiseason.json` — Improvement A standalone
- `reports/v40_c_defense.json` — Improvement C standalone
- `reports/v40_composite.json` — final composite

**OOS naive baseline check** must be repeated for each variant (to catch any regression to leaky behavior):

```bash
python3 -c "
from scripts.validate_lineup_pts_v2 import build_player_on_court_nrtg, ...
# Use season N-1 per-100 to predict season N lineups
# Expected: v4.0 PTS still beats by ≥ +0.10 wr (vs Phase A baseline +0.118)
"
```

If v4.0 falls below +0.10 wr OOS advantage vs naive, it means the smoothing/defense changes are eating into the abstraction's value — investigate before adoption.

---

## 8. Implementation Order (Strict Sequence)

The user explicitly required that improvements be tested individually. **Do not skip steps.**

1. **Improvement A — Multi-season smoothing**
   - Implement `scripts/pts_v40_multiseason.py` per §4
   - Sweep hyperparameters per §4.4 (5 tau × 3 decay × 3 K = 45 configs at default; pruned by ablation)
   - Pick winner; produce `pts_v40_a.parquet`
   - Run all 4 gates per §4.6
   - If pass → proceed. If fail → diagnose, narrow sweep, retry.

2. **Improvement C — Defense redesign**
   - Implement `scripts/pts_v40_defense.py` per §5
   - Sweep γ's per §5.4 (3 × 3 × 3 = 27 configs)
   - Pick winner; produce `pts_v40_c.parquet`
   - Run all 3 gates per §5.6
   - If pass → proceed. If fail → diagnose.

3. **Composite v4.0** (only after both A and C individually pass)
   - Implement `scripts/build_pts_v40.py` per §6
   - Sweep blend ratio
   - Run all 4 composite gates per §6.1
   - If pass → adopt; update `PTS_V40` block in `model_config.py`; write `docs/bke_v40_report.md`.

4. **Re-validate OOS naive baseline test** — repeat the season N-1 per-100 → season N comparison; expect v4.0 to still beat by ≥ +0.10 wr.

5. **Star sanity audit** — Curry/KD/Luka/SGA/Giannis top-30; Braun within-archetype 50–80p.

---

## 9. File Map (precise paths)

New files to create:

| Path | Purpose |
|---|---|
| `scripts/pts_v40_multiseason.py` | Improvement A standalone builder |
| `scripts/pts_v40_defense.py` | Improvement C standalone builder |
| `scripts/build_pts_v40.py` | composite builder + features patcher |
| `scripts/pts_v40_sweep.py` | runs hyperparameter sweeps for A and C |
| `docs/plans/pts_4_0_design.md` | this doc |
| `docs/bke_v40_report.md` | final report after composite passes |

New files PRODUCED:

| Path | Purpose |
|---|---|
| `data/processed/bke/pts_v40_a.parquet` | Improvement A output |
| `data/processed/bke/pts_v40_c.parquet` | Improvement C output |
| `data/processed/bke/pts_v40.parquet` | composite output |
| `data/processed/forecast/projected_team_features_v40.parquet` | patched features for game model |
| `reports/v40_a_multiseason.json` | A ablation results |
| `reports/v40_c_defense.json` | C ablation results |
| `reports/v40_composite.json` | composite results |
| `reports/v40_sweep.json` | full hyperparameter sweep dumps |

Files to be ADDED to (config dataclasses appended):

| Path | What to add |
|---|---|
| `src/modeling/model_config.py` | `PtsV40MultiSeasonConfig`, `PtsV40DefenseConfig`, `PtsV40CompositeConfig`, output path constants |

---

## 10. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Multi-season smoothing pulls stars too hard toward mean → lineup r drops | Medium | High | Sweep tau; star sanity gate; fallback to current PTS for top-30 ranked players if smoothed version under-ranks them |
| Defense redesign sign-flips signal accidentally | Medium | High | Unit test on known-elite defenders (Gobert, Smart, Holiday) BEFORE pipeline run |
| Lineup residual regression overfits | Medium | Medium | Linear OLS only, no interactions; document n_obs; use train/test split within season if necessary |
| `possessions_played` missing for many players | Low | Medium | Fallback to 1500 (full weight); document fallback rate; if > 10% of player-seasons, raise to user |
| Composite blend ratio is wrong → composite worse than either piece alone | Medium | Medium | Sweep at composite step; if no ratio improves over both standalones, use whichever standalone is better |
| OOS naive baseline starts beating v4.0 | Low | Critical | If happens, v4.0 has regressed toward leakiness; revert to v3.2 and re-design |

---

## 11. v4.1 — Next After A and C (Improvements B and D)

These two improvements were deliberately deferred from v4.0 to keep the first ablation clean. **They are the explicit next step after v4.0 ships** (composite passes §6.1 gates and the report in §13 is written). They should be planned and built as v4.1 with the same A/B isolation discipline used here.

### 11.1 Improvement B — Per-Player Uncertainty Propagation

#### Goal

Replace minutes-only team aggregation with **inverse-variance weighted aggregation** so a 200-min rookie no longer carries the same per-minute influence as a 3,000-min veteran with much tighter posterior. Expected to help defense more than offense (defense estimates are noisier).

#### Method

For each player-season, compute a posterior σ on `pts_o` and `pts_d` from variance components:

```
σ_player² = σ_residual² / n_eff_possessions  +  σ_archetype²   (between-player variance within archetype)

where:
  σ_residual²  = season-wide residual variance after fixed effects
  n_eff        = possessions_played  (proxy for sample size)
  σ_archetype² = empirical-Bayes prior variance from existing
                 src/modeling/bayesian_hierarchical.py
```

Aggregation downstream becomes:

```
team_off_v41 = Σ_player ( minute_share · pts_o_v40 · 1/σ_o² )
             / Σ_player ( minute_share · 1/σ_o² )

(precision-weighted, then minute-share weighted — replaces minute-share-only)
```

#### Data sources

| Required column | Source | Notes |
|---|---|---|
| `pts_o_v40`, `pts_d_v40` | `data/processed/bke/pts_v40.parquet` | v4.0 composite |
| `possessions_played` | `bke_v28_decomposition.parquet` | for n_eff |
| variance components | `src/modeling/bayesian_hierarchical.py` | already exposes per-dim posterior variance hooks; extend to expose `pts_o_sigma_post`, `pts_d_sigma_post` |
| `primary_archetype` | `bke_v28_decomposition.parquet` | for σ_archetype² estimation |

#### Code spec

**Files:**

- `scripts/pts_v41_uncertainty.py` — produces `data/processed/bke/pts_v41_b.parquet` with the extra σ columns
- Modify `scripts/build_pts_v32.py` pattern: add an alternate `build_patched_features_v41(pts, weighting='precision')` that swaps minute-only for precision-weighted aggregation
- New config dataclass `PtsV41UncertaintyConfig` in `model_config.py`

**Output schema** adds to v40 columns:

| Column | Dtype | Notes |
|---|---|---|
| `pts_o_sigma` | float64 | posterior σ on pts_o |
| `pts_d_sigma` | float64 | posterior σ on pts_d |
| `pts_o_precision` | float64 | 1/σ² |
| `pts_d_precision` | float64 | 1/σ² |
| `precision_weight_used` | float64 | the effective aggregation weight per player |

#### Sweep / hyperparameters

| Param | Default | Sweep |
|---|---|---|
| Precision-weighting strength γ in `weight = minute_share · (precision^γ)` | 1.0 | 0.0, 0.5, 1.0, 1.5 (γ=0 reproduces v4.0; γ=1 is full Bayes; γ>1 over-confidence) |
| σ_archetype² floor (when archetype cohort small) | 0.02 | 0.005, 0.02, 0.05 |
| Posterior σ minimum (clip floor) | 0.05 | fixed |

#### Validation gates

Apply `scripts/validate_lineup_pts_v2.py` with `--pts-file pts_v41_b.parquet --pts-col pts_o_v40 --pts-col-d pts_d_v40 --label v41b` AND a modified harness that uses precision-weighted lineup prediction (not just simple mean of 5 players). Pass criteria:

1. **Lineup defense wr** ≥ v4.0 baseline (no regression). Expected: small improvement (low-minute defenders downweighted appropriately).
2. **Aggregate Brier** ≤ v4.0 baseline (no regression). Expected: ≤ −0.001 Brier improvement.
3. **Bench-player downweighting visible:** mean `precision_weight_used` for players with < 500 possessions is at least 30% lower than for players with ≥ 2000 possessions. Diagnostic only, not a gate.
4. **Star precision** is high: Curry/Luka/SGA precision ≥ 90th percentile. Confidence ranking matches what we expect.

#### Known risks

- **Archetype variance estimate is fragile** when archetype cohorts are small. Mitigation: minimum cohort size 15, fallback to position-bucket variance.
- **Precision dominates minutes too hard:** if γ > 1, garbage minutes from elite bench players outvote starter rotation. Sweep γ carefully.
- **Variance is not independent of mean:** a player whose true PTS is far from league mean has higher residual variance. Empirical Bayes corrects this; verify by checking the σ-vs-|PTS_z| scatter is roughly flat post-correction.

### 11.2 Improvement D — Source Orthogonalization

#### Goal

Eliminate double-counting where the same basketball action contributes to a dimension via two correlated inputs (box stat + tracking stat). Currently `dim_shooting_gravity_z` averages TS_PCT and CATCH_SHOOT_FG3_PCT — both are "shooting efficiency"; players who excel at both get a 2× boost on the same skill.

#### Method

For each dimension with multiple input columns, identify input pairs with within-season Pearson r > 0.5 and **residualize the secondary input against the primary** before z-score averaging:

```
For dim_shooting_gravity_z:
  primary   = z(TS_PCT)                                          # full weight, anchor
  secondary = z(residual(CATCH_SHOOT_FG3_PCT against TS_PCT))    # only the part NOT explained by TS%
  
  dim_shooting_gravity_z_v41 = 0.6 · primary + 0.4 · secondary
```

Repeat for every dim where input collinearity > 0.5. Examples likely to need orthogonalization:

| Dim | Likely collinearity (to verify) | Primary | Secondary (residualized) |
|---|---|---|---|
| 1 — Shooting Gravity | TS_PCT × CATCH_SHOOT_FG3_PCT | TS_PCT | catch-shoot residual |
| 2 — Driving Gravity | DRIVES_PER36 × AT_RIM_FREQ | DRIVES_PER36 | rim-freq residual |
| 3 — Playmaking | AST_PER36 × POTENTIAL_AST_PER36 | AST_PER36 | pot-ast residual |
| 7 — Turnover Control | TOV_PCT × TOV_PER_TOUCH | TOV_PER_TOUCH (more usage-adjusted) | TOV_PCT residual |

For dimensions with only one input source, no orthogonalization (no-op).

#### Data sources

All from `data/processed/bke/bke_v28_decomposition.parquet`. The raw inputs are stored as `dim_*_z_raw` columns already; orthogonalization can be done from those columns directly.

#### Code spec

**File:** `scripts/pts_v41_orthogonal.py`

**Config dataclass** `PtsV41OrthogonalConfig`:

```python
@dataclass
class PtsV41OrthogonalConfig:
    collinearity_threshold: float = 0.5  # only orthogonalize pairs above this
    primary_weight: float = 0.6           # weight on the primary input
    secondary_weight: float = 0.4         # weight on the residualized secondary
    # Per-dim override map (computed once, then locked):
    dim_orthogonalize: Dict[str, Tuple[str, List[str]]] = field(default_factory=lambda: {
        # dim → (primary_col, [secondaries_to_residualize])
        "dim_shooting_gravity_z": ("TS_PCT", ["CATCH_SHOOT_FG3_PCT", "FG3_PCT"]),
        "dim_driving_gravity_z":  ("DRIVES_PER36", ["AT_RIM_FREQ", "PAINT_FREQ"]),
        "dim_playmaking_creation_z": ("AST_PER36", ["POTENTIAL_AST_PER36", "SECONDARY_AST_PER36"]),
        "dim_turnover_control_z":  ("TOV_PER_TOUCH", ["TOV_PCT", "TOV_PER36"]),
    })
```

Discovery step (run once, document in report): compute the collinearity matrix for inputs within each dim across seasons, choose primary by highest signal-to-noise, list secondaries.

**Output:** `data/processed/bke/pts_v41_d.parquet` with new `dim_*_z_v41` columns and a recomputed `pts_o_v41_d`, `pts_d_v41_d`.

#### Sweep

| Param | Default | Sweep |
|---|---|---|
| `primary_weight` | 0.6 | 0.5, 0.6, 0.7, 0.8 |
| `collinearity_threshold` | 0.5 | 0.3, 0.5, 0.7 (controls how many pairs get residualized) |

#### Validation gates

Run `scripts/validate_lineup_pts_v2.py` with `--pts-file pts_v41_d.parquet --label v41d`. Pass criteria:

1. **YoY PTS_z correlation** ≥ v4.0 baseline (orthogonalization should improve stability by removing duplicated noise).
2. **Lineup weighted r (joint)** ≥ v4.0 baseline.
3. **Aggregate Brier** ≤ v4.0 baseline.
4. **Diagnostic — diminished double-counting:** for the 4 orthogonalized dims, the within-season correlation between the primary and the residualized secondary should be < 0.1 (by construction of residualization). Verify in unit test.

#### Known risks

- **Residualization removes signal alongside noise:** if the secondary input truly measures a different skill (e.g., spot-up shooting vs total shooting), residualizing washes that out. Mitigation: sweep `primary_weight` and check that lineup r doesn't drop.
- **Inputs are not always well-defined per dim:** some dims have NaN inputs for parts of the population. Use within-season fitted residualization (skip NaN rows in the OLS, apply coefficients to all rows).
- **Order matters with > 2 inputs:** residualize secondary against primary, then tertiary against (primary + residualized secondary). Document the order per dim in the config.

### 11.3 Composite v4.1 (after both B and D pass individually)

Same composition pattern as §6 — only after Improvements B and D each pass their standalone gates, build the v4.1 composite:

```
pts_o_v41 = orthogonalized_dim_recompute(pts_v40_inputs)            # D effect
pts_d_v41 = orthogonalized_dim_recompute(pts_v40_inputs)            # D effect  
team aggregation uses precision-weighting from B                    # B effect
```

Sweep blend strategies (D vs no-D on top of B; full-stack vs partial).

### 11.4 v4.1 Implementation Order (Strict Sequence)

Mirror the v4.0 discipline. **Do not skip steps.**

1. Improvement B — implement, sweep γ, run 4 gates → adopt or revise
2. Improvement D — discover collinearity matrix, implement, sweep weights, run 4 gates → adopt or revise
3. Composite v4.1 — sweep blend, run all gates including OOS naive baseline re-check
4. Adopt; update `PTS_V41` config block; write `docs/bke_v41_report.md`

### 11.5 v4.1 Files

| Path | Purpose |
|---|---|
| `scripts/pts_v41_uncertainty.py` | Improvement B builder |
| `scripts/pts_v41_orthogonal.py` | Improvement D builder |
| `scripts/build_pts_v41.py` | composite builder + features patcher (v41 variant) |
| `scripts/pts_v41_sweep.py` | sweeps for B and D |
| `docs/plans/pts_4_1_design.md` | detailed design (this section formalized + collinearity matrix appendix) |
| `docs/bke_v41_report.md` | final report after composite passes |
| `data/processed/bke/pts_v41_b.parquet` | B output |
| `data/processed/bke/pts_v41_d.parquet` | D output |
| `data/processed/bke/pts_v41.parquet` | v4.1 composite |
| `data/processed/forecast/projected_team_features_v41.parquet` | patched features for game model |
| `reports/v41_b_uncertainty.json`, `reports/v41_d_orthogonal.json`, `reports/v41_composite.json` | ablation + composite results |

### 11.6 v4.1 Risks

| Risk | Mitigation |
|---|---|
| Precision-weighting collapses team aggregation when most players have low precision (early season, COVID seasons) | Floor on `pts_*_precision` ≥ 0.05 ⁻²; document |
| Orthogonalization discovery is overfit to in-sample collinearity matrix | Use only pairs with stable r > 0.5 across ≥ 6 of 8 seasons; lock the dim_orthogonalize dict before sweep |
| v4.1 composite over-shrinks bench players, hurting depth-driven team predictions (Heat, Spurs late-season) | Diagnostic gate: team aggregate precision-weighted vs minute-only should differ < 10 % for healthy teams; flag teams where it differs more |
| Need to re-run Phase A OOS naive baseline at v4.1 to confirm no regression | Mandatory gate in §11.4 step 3 |

### 11.7 What Would Make Us Skip v4.1

If v4.0 alone reaches the composite gates of §6.1 (Brier ≤ 0.230, lineup wr ≥ 0.36, defense wr ≥ 0.32, YoY r ≥ 0.70) AND the Phase C meta-model alone closes the remaining Brier gap to Vegas-academic-band (≤ 0.215), v4.1 may be deprioritized in favor of:

- Real-time data ingestion (manual CSV for Kalshi closing-line backtest, per the user's earlier direction)
- Phase C meta-model build (Improvements B/D help PTS quality marginally; the meta-model is where the larger remaining Brier gain lives)

User decision required at v4.0 completion: ship v4.1 next or pivot to Phase C? Both are good follow-on candidates; choose based on whichever metric ceiling was hit first.

---

## 12. Sub-task Checklist for Implementer (v4.0)

Execute in order. Mark each completed in TaskCreate / TaskUpdate before proceeding.

- [ ] **4.0.1** Read `docs/v3_3_phase_a_validation.md` end to end.
- [ ] **4.0.2** Read `scripts/validate_lineup_pts_v2.py` and `scripts/build_pts_v32.py` to understand the harness and patch path.
- [ ] **4.0.3** Add `PtsV40MultiSeasonConfig` to `src/modeling/model_config.py`.
- [ ] **4.0.4** Implement `scripts/pts_v40_multiseason.py` per §4.5. Include CLI args, output schema, --print-config.
- [ ] **4.0.5** Implement `scripts/pts_v40_sweep.py` covering the §4.4 grid; output `reports/v40_sweep.json`.
- [ ] **4.0.6** Pick A winner from sweep; run §4.6 gates; write `reports/v40_a_multiseason.json`.
- [ ] **4.0.7** A passes? → continue to C. Fails? → diagnose, narrow sweep, retry. Stop and report to user if still failing.
- [ ] **4.0.8** Add `PtsV40DefenseConfig` to `src/modeling/model_config.py`.
- [ ] **4.0.9** Implement `scripts/pts_v40_defense.py` per §5.5. Include unit test for sign convention on Rudy Gobert.
- [ ] **4.0.10** Extend `scripts/pts_v40_sweep.py` for §5.4 γ-grid.
- [ ] **4.0.11** Pick C winner from sweep; run §5.6 gates; write `reports/v40_c_defense.json`.
- [ ] **4.0.12** C passes? → continue to composite. Fails? → diagnose, narrow sweep, retry. Stop and report if still failing.
- [ ] **4.0.13** Add `PtsV40CompositeConfig` to `src/modeling/model_config.py`.
- [ ] **4.0.14** Implement `scripts/build_pts_v40.py`; sweep blend ratio.
- [ ] **4.0.15** Run §6.1 composite gates; write `reports/v40_composite.json`.
- [ ] **4.0.16** Run OOS naive baseline check per §7.
- [ ] **4.0.17** Run star sanity audit per §8.5.
- [ ] **4.0.18** Composite passes all 4 + OOS check + star sanity? → adopt. Write `docs/bke_v40_report.md` per §13.
- [ ] **4.0.19** If composite fails, document why in the report and propose v4.0.1 fix path.

### 12.1 Sub-task Checklist for Implementer (v4.1 — only after v4.0 ships)

- [ ] **4.1.1** Read `docs/bke_v40_report.md` and confirm v4.0 composite gates passed.
- [ ] **4.1.2** Write `docs/plans/pts_4_1_design.md` formalizing §11 into a standalone implementer doc (collinearity matrix appendix included).
- [ ] **4.1.3** Add `PtsV41UncertaintyConfig` to `src/modeling/model_config.py`.
- [ ] **4.1.4** Implement `scripts/pts_v41_uncertainty.py` per §11.1. Extend `src/modeling/bayesian_hierarchical.py` to expose `pts_o_sigma_post`, `pts_d_sigma_post`.
- [ ] **4.1.5** Implement `scripts/pts_v41_sweep.py` for the γ × σ_archetype × σ floor grid (§11.1 sweep).
- [ ] **4.1.6** Pick B winner; run all 4 §11.1 gates; write `reports/v41_b_uncertainty.json`.
- [ ] **4.1.7** B passes? → continue to D. Fails? → diagnose, narrow sweep, retry.
- [ ] **4.1.8** Add `PtsV41OrthogonalConfig` to `src/modeling/model_config.py`. Run the collinearity discovery step (per-dim, across all 8 seasons) and lock the `dim_orthogonalize` dict.
- [ ] **4.1.9** Implement `scripts/pts_v41_orthogonal.py` per §11.2. Unit test: post-orthogonalization, primary vs residualized-secondary within-season r should be < 0.1.
- [ ] **4.1.10** Extend `scripts/pts_v41_sweep.py` for §11.2 `primary_weight` × `collinearity_threshold` grid.
- [ ] **4.1.11** Pick D winner; run all 4 §11.2 gates; write `reports/v41_d_orthogonal.json`.
- [ ] **4.1.12** D passes? → continue to composite. Fails? → diagnose.
- [ ] **4.1.13** Implement `scripts/build_pts_v41.py`; sweep blend strategies (B-only, D-only, full-stack).
- [ ] **4.1.14** Run §11.3 composite gates including OOS naive baseline re-check.
- [ ] **4.1.15** Star sanity audit (same bar as v4.0).
- [ ] **4.1.16** Composite passes? → adopt. Write `docs/bke_v41_report.md` per §13 (same skeleton, v4.1 metrics).
- [ ] **4.1.17** If composite fails, document why and propose v4.1.1 fix path or recommend pivot to Phase C meta-model.

---

## 13. Final Report Spec

`docs/bke_v40_report.md` must include (at minimum):

- §1 TL;DR table: Brier, lineup wr (joint/off/def), YoY r, star sanity — all four versions (v2.7 / v3.2 / v40_a alone / v40_c alone / v40 composite).
- §2 Improvement A standalone results + winning hyperparameters + sweep diagnostic table.
- §3 Improvement C standalone results + winning hyperparameters + sweep diagnostic table.
- §4 Composite results + blend ratio + composite gate scores.
- §5 OOS naive baseline re-check.
- §6 Star + Braun audit, top-30 lineup.
- §7 Downstream effects: which downstream consumers should switch to v4.0 features, which stay on v3.2.
- §8 Updated comparison vs market (v40 expected Brier vs Vegas, academic benchmarks).
- §9 Roadmap to v4.1 (Improvements B and D) and Phase C (meta-model).

---

## 14. Approval Required Before Implementation

This is a DRAFT. The implementer must NOT write code until the user has reviewed this doc and signaled approval. Specifically, approval is sought on:

1. The 4-improvement scope reduced to **2** (A and C only) per user direction.
2. The math of §4.2 (Bayesian smoothing formula) and §5.2 (defense composite formula).
3. The hyperparameter sweep grids in §4.4 and §5.4.
4. The validation gate thresholds in §4.6 and §5.6.
5. The composite blend strategy in §6.

If any of those need adjustment, edit this doc before coding.
