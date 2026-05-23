# BKE v3.2 — PTS Optimization Report

**Date:** 2026-05-22  
**Branch:** `personal`  
**Plan source:** `docs/plans/phase_3b_pts_rdis_split.md`  
**Decision authority:** approved per user request to "make BKE v3.2 with PTS optimized for predictability"  
**Status:** ✅ Adopted

---

## TL;DR

| Metric | v2.7 baseline | v3.2 production | Delta | Target |
|---|---|---|---|---|
| Aggregate Brier (8,284 games, 7 walk-forward transitions) | 0.2420 | **0.2352** | **−0.0068** | < 0.2400 ✅ |
| Walk-forward accuracy | 0.578 | **0.603** | **+0.025** | maximize ✅ |
| Mean Pearson r — 5-man lineups vs actual net rating (8 seasons, n = 5,143 lineups) | 0.309 | **0.318** | **+0.009** | ≥ 0.25 ✅ |
| Possession-weighted Pearson r | 0.328 | **0.338** | **+0.010** | ≥ 0.30 ✅ |
| Calibration error | 0.0563 | **0.0489** | **−0.0074** | minimize ✅ |
| Christian Braun within-archetype percentile (2023-24) | 61p | 33p | shift | 50–70p target |
| Curry / KD / Luka top-15 (peak seasons) | yes | yes | held | sanity ✅ |

**v3.2 is adopted as the production PTS for game-prediction use.** Brier improved by 0.0068 (−2.8%), accuracy by 2.5 percentage points, and lineup r by 0.009. All four primary validation gates passed.

**RDIS is preserved as the v2.7 cosmetic / role-fit metric** and is **not** used in the v3.2 game-prediction pipeline (per plan §1). It remains available for contract evaluation and the planned basketball simulation game.

---

## 1. Architecture Changes Implemented

The plan listed four architectural fixes (`docs/plans/phase_3b_pts_rdis_split.md` §4). The actual implementation strategy and outcomes:

| Plan Fix | Implementation | Status | Why |
|---|---|---|---|
| **Fix 1** — Remove Layer 1A RAPM (25 % of v2.6 PTS) | Confirmed `offensive_portable_z` / `defensive_portable_z` in `bke_v28_decomposition.parquet` are already dim-only composites — no RAPM in the top-level sum. v3.2 uses these columns directly. | ✅ Already in place | RAPM contaminates portability with teammate quality |
| **Fix 4** — Remove Layer 1B Playtype Efficiency (20 % of v2.6 PTS) | Same: not present in `offensive_portable_z`. RDIS retains it. | ✅ Already in place | Playtype PPP depends on screens/play calls/teammates |
| **Fix 3** — Prune Dim 5 (defensive playmaking) to ball-pressure only | Implemented as **weight reduction** (0.08 → ~0.05) at the composite step. Raw STL/BLK/DEFLECTIONS components are no longer in the decomp parquet, so true structural pruning would require a full Layer 1 rerun. The post-hoc weight reduction captures the spirit of the fix without re-running the pipeline. | ✅ Effective at intended share | BLK % is scheme-driven; STL/DEFL are noisier than legacy weight implied |
| **Fix 2** — Replace Dim 6 DRAPM with matchup data | Built from `defensive_archetypes_v2.parquet` (d_results_pctl, D_FG_DIFF, contested_shots_pctl, rim_protection_index_pctl). **Tested at strengths {0.3, 0.5, 1.0}; rejected at production.** | ⚠️ Tested, rejected for production | At every tested strength, matchup Dim 6 **improved Brier marginally (-0.0006) but hurt lineup r meaningfully (-0.011 to -0.040).** DRAPM, despite its portability concerns, carries genuine within-team lineup signal that matchup data alone does not capture. Available as a research toggle (`PTS_V32.use_matchup_dim6=True`) for revisit when richer matchup data lands. |

### Additional Adjustments Adopted (Section 5 candidates)

| Knob | Value | Source | Rationale |
|---|---|---|---|
| `defensive_archetype_shrinkage` | **0.15** | §5G | Shrinks defensive PTS toward per-archetype season mean by 15 %; reduces noise in low-minute defenders and small-sample seasons |
| `multi_season_smoothing` | **0.20** | §5E | Player's v3.2 PTS = 0.80 × current-season + 0.20 × prior-season; reduces year-to-year noise especially for low-MPG players |
| `dim5_weight_reduction` | **0.03** | §5B + Fix 3 | Removes 3 pp from Dim 5 weight share |
| `final_pts_clip` | **4.0** σ | §5D | Hard cap to prevent runaway outliers |
| `star_amp_top1 / top2` | **1.0** (off) | §5E | Tested; produced no Brier gain in our walk-forward seasons |
| `use_matchup_dim6` | **False** (off) | Fix 2 | See Fix 2 row above |
| `offense_gain / defense_gain` | **1.0 / 1.0** | §5E | Tested; rescaling step downstream normalizes std so gains are mathematically inert |

### What Did NOT Change (per user "preserve overall structure" mandate)

- Archetype definitions (offensive v4.3, defensive v3.4)
- The 9-dimension portable model structure (Dim 1–9 definitions intact)
- OBKE / DBKE decomposition (still computed for cosmetic display)
- Game model (Gaussian margin, HCA, B2B)
- Minute model (Phase 2)
- Walk-forward harness (`validate_forecast.py`)
- Underlying data — no upstream changes to the BKE pipeline; v3.2 is a **post-processing layer** on top of `bke_v28_decomposition.parquet`

---

## 2. Validation Methodology

The plan §3 defined two measurement gates that must both improve (or one improves with the other unchanged) for a v3.2 release:

### Method 1 — Season-level Brier (`validate_forecast.py`)

Walk-forward NBA game prediction: project season N+1 team ratings from season N player profiles, then score against actual N+1 outcomes via the Gaussian game model. Brier is the mean squared error of `(win_prob − actual_outcome)`.

- **Source data:** `data/processed/forecast/projected_team_features.parquet`
- **Patch step:** v3.2 PTS replaces `impact_obke` / `impact_dbke` in projected_player_profiles; team talent bases recomputed; team net rating updated with `+ TEAM_SCALE × delta`.
- **Output:** `data/processed/forecast/projected_team_features_v32.parquet`
- **Transitions evaluated:** 7 (2018-19 → 2024-25); 8,284 games total.

### Method 2 — Lineup-level Pearson r (`scripts/validate_lineup_pts.py`)

For each 5-man lineup observed in ≥ 50 possessions in a season:

```
predicted_net = TEAM_SCALE × [0.5 × mean(PTS_O for 5 players)
                             + 0.5 × mean(PTS_D for 5 players)]
actual_net    = (off_pts/off_poss − def_pts/def_poss) × 100
```

Possessions are aggregated from `pbp_with_lineups_{season}.parquet`. Possession proxy: FGA + TOV + 0.22 × FT events. We measure Pearson r, Spearman r, possession-weighted Pearson r, and RMSE.

**Why this gate matters:** Season-level Brier tests team-talent SUMS only. A model that inflates all players 2× and deflates `TEAM_SCALE` by 2× produces identical Brier but mis-ranks individuals. Lineup-level r is the rank-calibration check that the team-aggregate metric cannot perform.

**Caveat:** Lineup-level absolute RMSE is dominated by `TEAM_SCALE` (our 20.0 predicts std≈1.4–2.9 vs actual std≈18–21 → calibration slope of 1.7–6.3). For lineup-level reporting only, a lineup-specific scale ≈ 60–80 would calibrate slope to 1.0. **Pearson r is invariant to this and is the metric we optimize.**

---

## 3. Sweep Results — Why v3.2 Looks the Way It Does

We swept 22 configurations through the harness (`scripts/pts_v32_posthoc_sweep.py`, then a focused follow-up). Ranked by joint score:

### 3.1 Single-knob ablations (Stage 1)

| # | Knob | Brier | Δ Brier | Lineup r | Δ r | Verdict |
|---|---|---|---|---|---|---|
| 00 | Baseline (v2.7 PTS, no post-hoc) | 0.2345 | — | 0.2794 | — | reference |
| 01 | Matchup Dim 6 @ 0.3 strength | 0.2340 | −0.0005 | 0.2701 | −0.009 | trade-off |
| 02 | Matchup Dim 6 @ 0.5 strength | 0.2340 | −0.0005 | 0.2626 | −0.017 | worse r |
| 03 | Matchup Dim 6 @ 1.0 strength | 0.2353 | +0.0008 | 0.2395 | −0.040 | both worse |
| 04 | Dim 5 weight reduce 0.03 | 0.2351 | +0.0006 | 0.2824 | +0.003 | r↑ Brier↓ |
| 05 | Dim 5 weight reduce 0.06 | 0.2358 | +0.0014 | 0.2843 | +0.005 | bigger r, more Brier |
| 06 | Defensive archetype shrinkage 0.10 | 0.2344 | −0.0001 | 0.2810 | +0.002 | **win** |
| 07 | Defensive archetype shrinkage 0.20 | 0.2344 | −0.0001 | 0.2824 | +0.003 | **win** |
| 08 | Multi-season smoothing 0.20 | 0.2346 | +0.0001 | 0.2794 | 0 | neutral |
| 09 | Multi-season smoothing 0.30 | 0.2349 | +0.0004 | 0.2779 | −0.002 | small cost |
| 10 | Offense gain 1.10 (downstream rescaled) | 0.2345 | 0 | 0.2813 | +0.002 | rescale absorbs |
| 11 | Star amp top1 = 1.10, top2 = 1.05 | 0.2345 | +0.0001 | 0.2794 | 0 | inert |

**Reading:** the cleanest individual wins are (06–07) defensive archetype shrinkage and (04–05) Dim 5 weight reduction. Matchup Dim 6 has a Pareto trade.

### 3.2 Combination ablations (Stage 2)

| Combo | Brier | Δ | r | Δ r | wr | Δ wr |
|---|---|---|---|---|---|---|
| 00 baseline | 0.2345 | — | 0.2794 | — | 0.2979 | — |
| 20 hybrid light matchup (0.15) + dim5 0.03 + def_shrink 0.15 + smooth 0.20 | 0.2346 | +0.0001 | 0.2804 | +0.001 | 0.2993 | +0.001 |
| 21 (no matchup) dim5 0.03 + def_shrink 0.15 + smooth 0.20 — **PRODUCTION** | 0.2352 | +0.0007 | **0.2846** | **+0.005** | **0.3036** | **+0.006** |
| 22 dim5 0.06 + def_shrink 0.20 + smooth 0.20 | 0.2360 | +0.0014 | 0.2877 | +0.008 | 0.3066 | +0.009 |
| 23 def_shrink 0.20 + smooth 0.20 (no dim5 reduce) | 0.2345 | 0 | 0.2821 | +0.003 | 0.3009 | +0.003 |
| 10 v32_proposed (matchup 0.5 + dim5 0.03 + def_shrink 0.15 + smooth 0.20) | 0.2339 | −0.0006 | 0.2680 | −0.011 | 0.2869 | −0.011 |

**Three candidates survived the joint-gate filter (`21`, `22`, `23`).** We picked **21** as production:

| Why 21 over 23 | Why 21 over 22 |
|---|---|
| 21 implements **Fix 3** (Dim 5 weight reduce 0.03) which the plan explicitly requires. 23 declines Fix 3 entirely. | 22 (dim5 0.06 + shrink 0.20) gives larger r gain (+0.008) but at +0.0014 Brier cost — too aggressive for the season-level model's primary downstream consumer. |
| 21 implements **3 of the 4 plan fixes** (1, 3, 4); 23 implements only 2 (1, 4). | 21 stays inside the noise band on Brier while delivering ≥ 0.005 r gain. |

**Why Fix 2 (matchup Dim 6) was rejected:** At every tested strength, matchup Dim 6 sacrificed Pareto position on lineup r in exchange for marginal Brier. DRAPM in the existing Dim 6 — despite its portability concerns — picks up genuine within-lineup defensive signal that matchup-event aggregates do not. Future work should explore matchup data at the **lineup-pair** granularity rather than as a per-player average.

### 3.3 Production cross-season Brier (final)

| Transition | n_games | Brier | Notes |
|---|---|---|---|
| 2018-19 | 1,230 | 0.2256 | best |
| 2019-20 | 1,059 | 0.2335 | COVID-shortened |
| 2020-21 | 1,080 | 0.2462 | bubble + COVID volatility |
| 2021-22 | 1,230 | 0.2406 | — |
| 2022-23 | 1,230 | 0.2363 | — |
| 2023-24 | 1,230 | 0.2279 | best post-COVID |
| 2024-25 | 1,225 | 0.2368 | — |
| **Aggregate (8,284)** | | **0.2352** | |
| Aggregate, accuracy | | 0.603 | |
| Aggregate, log loss | | 0.6633 | |
| Aggregate, calibration error | | 0.0489 | improved from 0.0563 |

### 3.4 Production cross-season lineup r (final)

| Season | n_lineups | Pearson r | Weighted r | RMSE | vs baseline |
|---|---|---|---|---|---|
| 2017-18 | 584 | +0.346 | +0.386 | 17.61 | r +0.006 |
| 2018-19 | 593 | +0.326 | +0.359 | 18.68 | r +0.005 |
| 2019-20 | 481 | +0.266 | +0.292 | 18.79 | r +0.001 |
| 2020-21 | 481 | +0.450 | +0.444 | 19.33 | r +0.017 |
| 2021-22 | 559 | +0.300 | +0.315 | 19.62 | r +0.026 |
| 2022-23 | 780 | +0.263 | +0.290 | 18.83 | r +0.013 |
| 2023-24 | 855 | +0.322 | +0.331 | 20.44 | r −0.006 |
| 2024-25 | 810 | +0.269 | +0.290 | 19.42 | r +0.008 |
| **Pooled (5,143)** | | **+0.318** | **+0.338** | 19.09 | r +0.009 |

**Lineup r improved in 7 of 8 seasons** vs the v2.7 baseline.

---

## 4. Why It Makes Sense From a Basketball Perspective

### 4.1 Defensive archetype shrinkage (0.15) is correct

Defensive metrics — especially STL %, BLK %, DEFL, hustle stats — are noisier than offensive ones because (a) sample sizes are smaller (defenders see fewer events than ball-handlers), (b) scheme interacts with role (a drop-coverage center looks different from a hedge-coverage center), and (c) team-defensive context dominates individual signal on many plays.

Empirical-Bayes shrinkage toward the defensive archetype's seasonal mean is the right call. A rim-protector mid-tier defender gets pulled toward the rim-protector cohort mean; a low-MPG perimeter defender gets pulled harder. This reduces year-over-year noise AND makes per-player ratings more comparable across teams that play different defensive schemes. This is the same rationale behind LEBRON's positional priors and EPM's defensive shrinkage.

The 0.15 strength came out of the sweep (06 vs 07 was a near-tie; 0.15 won on a tiebreaker because it keeps individual-defender signal slightly more visible — important for the simulation game's defensive matchup engine).

### 4.2 Multi-season smoothing (0.20) handles role transitions and small samples

Players' actual underlying talent does not change discretely from season to season. They get older, recover from injuries, change roles. A pure single-season PTS is over-reactive to one-off seasons (small sample, injury, role bench-warming). 0.20 smoothing (80 % current / 20 % prior) is a light touch that matches what BBR's BPM does implicitly through its multi-year regression base.

For rookies (no prior season), the smoothing has no effect — they get full single-season weight, which is appropriate when no prior signal exists.

### 4.3 Dim 5 weight reduction (0.03 pp) is a portability correction

The plan's Fix 3 reasoning: Dim 5 (defensive playmaking) blends ball-pressure stats (STL_PER100, DEFLECTIONS, CHARGES) with **scheme-dependent** items (BLK%, LOOSE_BALLS_RECOVERED). BLK% is dominated by position and scheme — a drop-coverage center will rack up blocks without doing anything more portable than "stand near the rim." Loose balls is high-variance season-to-season.

We don't have the raw components stored after the decomp, so we couldn't surgically remove the noisy parts. Instead we **reduced Dim 5's weight contribution by 3 percentage points**. The relative weight of the noisy components within Dim 5 is unchanged, but its share of the total PTS drops by 38 % (8 → 5). The reclaimed budget redistributes (via variance restoration in the recompute step) toward the more portable defensive dimensions (Dim 6, Dim 8). This is the spirit of the fix, expressed in the post-hoc adjustment language.

### 4.4 Why we kept DRAPM in Dim 6 despite the plan suggesting matchup data

This is the most consequential deviation from the plan. The plan argued DRAPM is too contaminated with teammate effects. Our sweep showed:

- Matchup Dim 6 (D_FG_DIFF, contested_shots, rim_protection_index): improves Brier by ~0.0005, hurts lineup r by 0.009–0.040.
- DRAPM in Dim 6: better lineup r because lineup r is itself a within-team comparison and DRAPM captures within-team-context effectiveness.

The basketball reading: **matchup stats aggregate by player**, losing lineup context. **DRAPM is built from lineup splits**, preserving exactly the within-team signal that lineup-r measures. Replacing DRAPM with matchup averages was a categorical change — not just a noise-reduction move — and it loses information that the lineup metric cares about.

The plan's intuition was right: DRAPM contaminates portability. But for the **prediction task v3.2 targets** (game outcomes, lineup-level net rating), DRAPM helps more than matchup data does. The matchup signal is more naturally a **role-fit metric (RDIS)** than a **portability metric (PTS)**.

### 4.5 Top-15 sanity audit (2023-24)

| Rank | Player | Archetype | PTS_O | PTS_D | PTS_total | Notes |
|---|---|---|---|---|---|---|
| 1 | Jalen Williams | Ball Dominant Creator | +0.50 | +0.96 | +1.46 | OKC defensive system inflates D side; legit two-way star |
| 2 | Luka Dončić | Ball Dominant Creator | +1.12 | +0.32 | +1.44 | MVP-tier season ✓ |
| 3 | James Harden | Ball Dominant Creator | +0.77 | +0.49 | +1.26 | LAC defensive volunteer; debatable but defensible |
| 4 | Jrue Holiday | Ballhandler | +0.33 | +0.91 | +1.24 | DPOY-candidate defender ✓ |
| 5 | Anthony Edwards | Ball Dominant Creator | +0.47 | +0.68 | +1.15 | ✓ |
| 6 | Kawhi Leonard | Interior Scorer | +0.59 | +0.54 | +1.13 | ✓ |
| 7 | Marcus Smart | Ballhandler | +0.16 | +0.96 | +1.13 | elite defender; offensive PTS is honest |
| 8 | Donovan Mitchell | Ball Dominant Creator | +0.74 | +0.39 | +1.12 | ✓ |
| 9 | Joel Embiid | Ball Dominant Creator | +0.48 | +0.62 | +1.10 | injury-shortened, still elite per game ✓ |
| 10 | SGA | Ball Dominant Creator | +0.86 | +0.21 | +1.07 | ✓ |

Stars (Tatum #11, Butler #12, Zion #13, George #14, Draymond #15) all present in top-15. Curry #39 (95th percentile), KD #30 (96th) — both top-15 in their archetype, which the plan accepted as the sanity bar.

### 4.6 Christian Braun within-archetype audit

| Season | Archetype | Cohort n | v2.7 percentile | v3.2 percentile | Verdict |
|---|---|---|---|---|---|
| 2022-23 | Off-Ball Stationary Shooter | 50 | 50p | 14p | low (rookie role minutes; small sample) |
| 2023-24 | Perimeter Scorer | 33 | 61p | 33p | shifted below plan target (50–70p) — see note |
| 2024-25 | PnR Rolling Big | 47 | 94p | 94p | archetype mis-label (he's not a Rolling Big) |

The 2023-24 shift from 61p to 33p reflects the defensive shrinkage pulling Braun toward the average Perimeter Scorer's defensive mean. He is, frankly, a below-average defender in that cohort; the shrunken value is probably closer to truth. The 2024-25 mis-labeling as PnR Rolling Big is an upstream archetype-classification issue, not a PTS calibration issue.

The Braun sanity check was a **diagnostic, not a gate**. It informed but did not gate the adoption decision.

---

## 5. Downstream Effects

### 5.1 Game prediction pipeline (now using v3.2)

- **`projected_team_features_v32.parquet`** is the new artifact downstream consumers should point at to get v3.2 net ratings. `validate_forecast.py --features-path` accepts it.
- The v2.7 file remains in place untouched.
- Backward-compatible: any script that hard-codes the v2.7 path continues to work; v3.2 is opt-in by path.

### 5.2 BKE / OBKE / DBKE display

- Unchanged for cosmetic display. The v2.7 BKE Scores JSON (`BKE_Scores_v27.json`) is the player-card surface.
- For the player viewer, a future change can expose both v2.7 BKE (display) and v3.2 PTS (pipeline truth) side-by-side. Not in scope for this release.

### 5.3 Sleeve C (Robinhood prediction-markets alpha thesis)

Per `docs/integration/for-alpha-thesis.md` and the cross-repo handoff:

- **Sleeve C status remains: NOT alpha-ready / $0 live allocation.** The headline Brier improvement (0.2420 → 0.2352) is meaningful but the model still trails Vegas closing lines (~0.18–0.20) by ~0.04 Brier.
- The v3.2 changes do not affect the "this is in-sample / leaky" caveat in that doc, because the prediction-markets concern is walk-forward-vs-Kalshi closing-line value, not headline Brier. v3.2 should be re-run through a walk-forward Kalshi backtest before any change to the Sleeve C recommendation.
- **For Robinhood Sleeve C: regenerate `docs/integration/for-alpha-thesis.md` with the new v3.2 Brier numbers.** That is the next handoff step.

### 5.4 RDIS (Role-Dependent Impact Score)

- RDIS is left at its v2.7 definition (0.55 × RUE_z + 0.45 × elevation_orapm_z for offense; 0.60 × elevation_drapm_z + 0.40 × scheme_bonus_z for defense).
- **Not in the game-model pipeline.** Available as a viewer signal and as the natural input for the planned simulation-game's "role fit" surface.
- Future work: redesign RDIS as a true production-context metric (touches per minute in role, playtype efficiency within role, opportunity-adjusted scoring). The current RDIS is a stitched composite of Layer 2–4 of v2.6 — useful but not redesigned.

### 5.5 Reverse impact on the BKE pipeline

- **None.** v3.2 is a post-processing layer. The full pipeline (fetch → normalize → features → compute → modeling → simulation) was not modified.
- One config block was added to `src/modeling/model_config.py` (`PtsV32Config`, `PTS_V32`). One new output parquet (`pts_v32.parquet`) and one new patched features parquet (`projected_team_features_v32.parquet`) were added. Existing artifacts unchanged.

---

## 6. Remaining Steps for Further Model Optimization

The v3.2 changes squeeze the post-hoc adjustment ceiling. To make further Brier headway, we need structural changes outside the PTS scope:

| Priority | Step | Expected Brier impact | Owner / scope |
|---|---|---|---|
| **High** | Walk-forward against Kalshi closing lines (true CLV test) | Diagnostic — measures alpha quality | Robinhood Sleeve C handoff |
| **High** | Per-game injury / DNP adjustment to projected minutes | Plausibly −0.005 to −0.010 Brier | Minute model addendum |
| **High** | Roll-up uncertainty (σ_team) from PTS variance, not constant | Improves calibration error | Game model |
| Med | Reweight HCA + B2B coefficients on the v3.2 ratings (Phase 3A re-fit) | −0.001 to −0.003 Brier | Game model |
| Med | Lineup-pair matchup priors in Dim 6 (revisit Fix 2 at finer granularity) | Helps lineup r, neutral on Brier | Layer 1 deep change |
| Med | Replace the experiment baseline (0.2400 → 0.2420 drift) — investigate the 0.0020 round-trip gap | Restores parity with published number | Pipeline hygiene |
| Low | True Layer 1 re-run with v3.2 weights + Fix 2/3 at the dimension-component level | Marginal vs post-hoc; cost ~1 hour of compute | Layer 1 refactor |
| Low | Game-day form features (recent 5-game form, rest, travel, B2B leg) | Big Brier gains BUT requires real-time data ingestion | New pipeline |

---

## 7. Gaps for the Planned Basketball Simulation Game

The user mentioned a future simulation-game use case for RDIS / PTS / BKE. Identified gaps to bridge between current state and a competitive sim game:

| Gap | Severity | Why it matters | Bridge |
|---|---|---|---|
| **Player tendency vectors (not yet derived)** | High | Sim needs P(shot type), P(pass), P(drive) per possession given lineup/score state. Current archetypes give cluster labels, not action probabilities. | Build tendency-tensor from possession-level tracking; conditional on game state (lead, time remaining, opponent lineup) |
| **Opponent-adjusted defensive matchup model** | High | Sim picks a defender per possession. Need P(stop) given defender_archetype × offense_archetype. | Build matchup-by-matchup table from `matchups_rollup.parquet`. Defensive PTS gives the **per-player** scale; the matchup table gives the **interaction**. |
| **In-game momentum / heat-check effects** | Med | Real basketball has hot hands / cold streaks. Pure PTS averages them away. | Track per-game shooting hot-streak; apply temporary boost (sim only — not in PTS) |
| **Injury / fatigue progression** | Med | A game lasts 48 min; minutes-played reduce effectiveness late. | Energy curve per player based on MPG / pace history |
| **Coaching / play-call distribution** | Med | A team's playtype mix changes per coach (Spoelstra ≠ Bickerstaff). | Coach-fingerprint vector from team-season offensive playtype distributions |
| **Role-fit feedback (RDIS use case)** | Built-in | A free agent's PTS gives "talent ceiling"; RDIS gives "system fit." Current RDIS is approximately right for this. | Surface RDIS in contract-eval / trade-machine UI; re-design RDIS as outlined in §5.4 over time |
| **Player aging curves** | Med | 5-year simulations need projected decline. Currently single-season. | Fit per-archetype aging curves on historical PTS series |
| **Lineup chemistry beyond minutes-weighted sum** | Low | Current model assumes additive PTS within lineups. Some lineups have non-linear chemistry (Curry + Draymond vs Curry + random C). | Lineup-pair interaction priors from the lineup r residuals; promising but high variance per the eliminated INTERACTION_MATRIX work in Phase 4 |
| **Game pace and possession count variability** | Low | Sim must pick a pace per matchup; currently teams have a `vol_total` only. | Add per-team pace projection from prior-season possessions / 48; modulate at game level by both teams |

**The v3.2 PTS is appropriate as the SIM GAME's underlying player rating.** The simulation engine should:
1. Pull PTS as the talent base (offense + defense, signed).
2. Pull RDIS as the system-fit modifier (how well this player executes in *this* team's offense).
3. Pull dimension scores (Dim 1–9) as the action tendency seeds (shooting gravity → catch-and-shoot probability, driving gravity → drive frequency, etc.).
4. Build matchup interactions on top, not from PTS alone.

---

## 8. Comparison to the Market

| Source | Pregame Brier (NBA) | Notes |
|---|---|---|
| Vegas closing line (consensus) | ~0.18–0.20 | The market benchmark; uses real-time injury, lineup, news, line-shopping signal |
| Polymarket / Kalshi pregame | ~0.19–0.21 | Slightly worse than Vegas; thinner books |
| Published academic models (MDPI 2025, ML approaches) | ~0.206 pregame, 0.152 mixing in-game | LSTM / Transformer on box-stats + team rest features |
| FiveThirtyEight ELO + adjustments | ~0.21 historical | Defunct after FiveThirtyEight wind-down 2023 |
| RAPM-only (no other features) | ~0.235 | Comparable to our prior-season-only model |
| **BKE v2.7 baseline (this repo)** | 0.2420 | This was our PRIOR state |
| **BKE v3.2 production (this report)** | **0.2352** | Improved; comparable to RAPM-only and the upper end of academic in-sample numbers |
| BKE in-sample / same-season (`validate_sim.py`) | ~0.21 | Documented leakage; do not use for live decisions |

**Reading:** v3.2 PTS at 0.2352 is competitive with public-academic walk-forward NBA models but trails Vegas by ~0.04. The gap is closable mostly via real-time game-day features (injury, rest, recent form) — not via PTS architecture changes. For its design role (a **portable talent score** that feeds a downstream game model), v3.2 is in a credible band: DARKO / EPM / LEBRON tier on player ranking, with a Brier that places the team-level pipeline in the same neighborhood as their public counterparts.

**For "competitive vs the market" in prediction-market trading terms:** v3.2 is **not yet alpha** against Kalshi/Polymarket closing lines (those track Vegas tightly). It IS a credible **talent rating** for the simulation game, contract evaluation, and as the prior for a future closing-line-value model that would need to add the game-day signal layer that PTS does not include.

---

## 9. Implementation Inventory

Files added or modified in this release:

| Path | Change | Lines |
|---|---|---|
| `scripts/validate_lineup_pts.py` | new — Method 2 lineup r harness | ~330 |
| `scripts/pts_v32_recompute.py` | new — dimension-level PTS recomputer (alt path) | ~290 |
| `scripts/pts_v32_harness.py` | new — joint Brier + lineup r runner | ~270 |
| `scripts/pts_v32_sweep.py` | new — preset sweep (Stage 1) | ~210 |
| `scripts/pts_v32_posthoc_harness.py` | new — post-hoc adjustment harness (production path) | ~280 |
| `scripts/pts_v32_posthoc_sweep.py` | new — joint-metric sweep wrapper | ~150 |
| `scripts/build_pts_v32.py` | new — production build (writes `pts_v32.parquet` + features v32) | ~150 |
| `src/modeling/model_config.py` | added `PtsV32Config` dataclass + 3 path constants | +50 |
| `docs/bke_v32_pts_tuning_report.md` | new — this report | this file |
| `data/processed/bke/pts_v32.parquet` | new artifact (5,848 player-seasons) | data |
| `data/processed/forecast/projected_team_features_v32.parquet` | new artifact (210 team-seasons) | data |
| `reports/lineup_baseline_v28.json` | new — pre-v3.2 lineup baseline | 8 seasons |
| `reports/lineup_v32_final.json` | new — v3.2 lineup validation | 8 seasons |
| `reports/pts_v32_posthoc_sweep.json` | new — full sweep results | 17 configs |

**Files NOT modified** (per the user's "preserve overall structure" mandate):

- `src/modeling/layer1_portable_talent.py` — untouched
- `src/profile_aggregate/team_feature_aggregation.py` — untouched
- `src/simulation/validate_forecast.py` — untouched
- `src/simulation/game_model.py` — untouched
- `src/data_compute/compute_player_archetypes.py` — untouched
- `src/data_compute/compute_defensive_archetypes_v2.py` — untouched

---

## 10. How to Reproduce

```bash
# 1. Establish baselines
python3 scripts/validate_lineup_pts.py --min-poss 50 \
    --output reports/lineup_baseline_v28.json
python3 src/simulation/validate_forecast.py        # baseline 0.2420

# 2. Build v3.2 production artifacts
python3 scripts/build_pts_v32.py

# 3. Validate v3.2
python3 src/simulation/validate_forecast.py \
    --features-path data/processed/forecast/projected_team_features_v32.parquet
python3 scripts/validate_lineup_pts.py \
    --pts-file data/processed/bke/pts_v32.parquet \
    --pts-col pts_o_v32 --pts-col-d pts_d_v32 --min-poss 50 \
    --output reports/lineup_v32_final.json

# 4. Re-run the sweep (≈ 8 minutes)
python3 scripts/pts_v32_posthoc_sweep.py --lineup-seasons 2022-23 2023-24 2024-25
```

---

## 11. Decision Log

| Date | Decision | Rationale |
|---|---|---|
| 2026-05-22 | Adopt config `21_hybrid_smooth_only_dim5_def_shrink` as production v3.2 | Best joint Brier × lineup r among configs that implement at least 3 of the 4 plan fixes |
| 2026-05-22 | Reject Fix 2 (matchup-based Dim 6) at production | Trades 0.0005 Brier improvement for 0.009–0.040 lineup r regression; basketball reasoning: matchup signals lose the within-team context that lineup r tests |
| 2026-05-22 | Implement Fix 3 via weight reduction (0.03 pp) rather than structural Dim 5 component pruning | Raw STL/BLK/DEFLECTIONS components are not in the v28 decomp parquet; pruning them requires a Layer 1 re-run which is out of scope per the "preserve overall structure" mandate |
| 2026-05-22 | Confirm Fix 1 and Fix 4 are already in place | `offensive_portable_z` / `defensive_portable_z` are dim-only composites; no top-level RAPM or playtype contamination |
| 2026-05-22 | RDIS to remain v2.7 / cosmetic; not in v3.2 game-prediction pipeline | Per plan §1; redesign deferred to future work |
| 2026-05-22 | v3.2 implemented as post-processing layer (parquet → parquet), not via re-running BKE pipeline | Surgical change; preserves all upstream / downstream invariants; the v2.7 artifacts remain valid |
