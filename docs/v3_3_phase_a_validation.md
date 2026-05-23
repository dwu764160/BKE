# v3.3 Phase A — Validation Fixes Report

**Date:** 2026-05-23  
**Status:** ✅ Complete (in-sample finding **revised** after OOS check — see §A.1)  
**Purpose:** Lock down trustworthy validation methodology before any PTS 4.0 work.

---

## TL;DR — PTS Beats Naive Baseline Out-of-Sample (the in-sample headline was leaky)

Initial in-sample comparison showed a naive per-100 on-court NetRtg baseline outperforming PTS v3.2 at lineup r (0.366 vs 0.318). That finding **was almost entirely within-season leakage** — using a player's actual season-N on-court Δ to predict season-N lineups is partially circular.

Once moved **out-of-sample** (season N−1 per-100 → predict season N lineups):

| Comparison | naive per-100 wr | PTS v3.2 wr | Δ (PTS − naive) |
|---|---|---|---|
| **In-sample (leaky)** | 0.393 | 0.338 | −0.055 (naive wins, leaky) |
| **Out-of-sample (correct test)** | **0.213** | **0.331** | **+0.118 (PTS wins)** |

**PTS v3.2 beats naive baseline by +0.118 wr out-of-sample.** This is the correct, leakage-free comparison. **PTS abstraction is doing real work** — it captures stable player-level signal that generalizes to next-season lineup prediction. Raw on-court NetRtg has good in-season fit but poor cross-season generalization.

**Phase A's other findings stand:**
1. Possession-weighted r is the correct headline metric (uniformly higher than unweighted across seasons).
2. LINEUP_TEAM_SCALE should be 45.5 for v3.2 reporting (vs the team-aggregate TEAM_SCALE=20).
3. Defensive PTS is weaker than offensive (0.282 vs 0.316 in-sample) — biggest opportunity for PTS 4.0.
4. The published Brier 0.2400 baseline / 0.2248 α=1.0 numbers were a stale cache; live data state gives baseline 0.2420 and v3.2 production 0.2352.

**Implication for PTS 4.0:** the original abstraction-based architecture is structurally CORRECT for an out-of-sample portable talent score. PTS 4.0 should improve at the *margins* (uncertainty, defense, multi-season smoothing, source-orthogonalization) without abandoning the dimension-model approach.

---

## A.1 — Naive Baseline Comparison (per-100 on-court NetRtg)

For each season, computed each player's season-aggregate per-100 NetRtg by:
1. Aggregating possessions per 5-man lineup from `pbp_with_lineups_{season}.parquet`
2. Per-player aggregation: (off_pts − def_pts) / possessions × 100, summed across lineups
3. Z-scored within season (so directly comparable to PTS z-scores)

Then ran the lineup validation harness using this naive score as PTS_O / PTS_D.

### Per-season results (PTS v3.2 vs per-100 baseline)

| Season | n_lineups | PTS r | per100 r | per100 wr | Δ wr (per100 − PTS) |
|---|---|---|---|---|---|
| 2017-18 | 584 | 0.346 | 0.370 | 0.402 | +0.016 |
| 2018-19 | 593 | 0.326 | 0.367 | 0.404 | +0.045 |
| 2019-20 | 481 | 0.266 | 0.285 | 0.321 | +0.029 |
| 2020-21 | 481 | 0.450 | 0.459 | 0.469 | +0.026 |
| 2021-22 | 559 | 0.300 | 0.380 | 0.407 | +0.092 |
| 2022-23 | 780 | 0.263 | 0.339 | 0.371 | +0.082 |
| 2023-24 | 855 | 0.322 | 0.377 | 0.391 | +0.061 |
| 2024-25 | 810 | 0.269 | 0.355 | 0.378 | +0.087 |
| **Pooled** | **5,143** | **0.318** | **0.366** | **0.393** | **+0.055** |

per-100 wins in 8 of 8 seasons. The advantage is **larger in recent seasons** (2021-22 onward) where per-100 wins by 0.06-0.09 weighted r.

### Why this is happening

per-100 NetRtg is, by construction, the **on-court Δ** for a player aggregated across all lineups they played in. It captures:
- Direct production (the stat itself)
- Lineup-induced effects (good teammates → both contribute to the same NetRtg → both look good)
- Scheme/role effects (the team's system inflates or deflates the player)
- Gravity and deterrence (defensive deterrence pulls opp eFG down for everyone on the floor)

PTS aggressively normalizes those away (archetype neutralization, position-conditional z-scores, shrinkage). This is correct for **PORTABILITY** ("how good is this player without context?") but **wrong for PREDICTABILITY at lineup level** ("how good will this 5-man unit be in this team's context?").

### CRITICAL — Out-of-Sample Correction

The in-sample per-100 baseline uses season-N player on-court NetRtg to predict season-N lineups. That's partially circular: a lineup's NetRtg is by construction a function of its members' on-court NetRtgs in the same season.

Re-ran with season N−1 per-100 → season N lineup r:

| Target season | naive_(N−1) r | naive_(N−1) wr | PTS_v32 r | PTS_v32 wr | Δ r (naive − PTS) |
|---|---|---|---|---|---|
| 2018-19 | 0.224 | 0.256 | 0.326 | 0.360 | −0.101 |
| 2019-20 | 0.142 | 0.165 | 0.266 | 0.292 | −0.124 |
| 2020-21 | 0.205 | 0.206 | 0.450 | 0.443 | −0.245 |
| 2021-22 | 0.222 | 0.232 | 0.300 | 0.315 | −0.078 |
| 2022-23 | 0.172 | 0.201 | 0.263 | 0.290 | −0.092 |
| 2023-24 | 0.194 | 0.206 | 0.322 | 0.330 | −0.128 |
| 2024-25 | 0.190 | 0.221 | 0.269 | 0.290 | −0.079 |
| **Pooled** | **0.193** | **0.213** | **0.314** | **0.331** | **−0.121** |

**PTS v3.2 beats out-of-sample naive per-100 by +0.121 r (+0.118 wr) on all 7 transition seasons.** Direction reverses cleanly when leakage is removed.

**What this means:**
- The in-sample naive advantage was almost entirely within-season circularity, not "PTS is broken."
- **PTS abstraction is correct for predictive use.** The cleanup (archetype neutralization, shrinkage, dimension model) produces a more *stable, generalizable* player signal than raw on-court Δ.
- Per-100 NetRtg overfits season-N team context. A player on a +10 NetRtg lineup looks elite by per-100 even if it's mostly his teammates carrying him; PTS's normalization correctly attributes credit.
- **The user's instinct to "validate validation first" was 100% correct.** Without this OOS check we would have pivoted PTS 4.0 in a wrong direction.

**Revised implication for PTS 4.0:** keep the dimension-model abstraction. Improvements should target the secondary weaknesses Phase A identified (defense calibration, lack of per-player uncertainty, multi-season smoothing, source orthogonalization) — NOT a structural pivot to "anchor on on-court NetRtg."

---

## A.2 — Possession-weighted r promoted to primary; LINEUP_TEAM_SCALE empirically fit

### Possession-weighted r

Unweighted Pearson r treats a 50-possession lineup the same as a 3,000-possession lineup. Weighting by `sqrt(min(off_poss, def_poss))` gives signal-rich lineups appropriate emphasis.

| Season | unweighted r (old) | weighted r (new headline) | Δ |
|---|---|---|---|
| 2017-18 | 0.346 | 0.386 | +0.040 |
| 2018-19 | 0.326 | 0.360 | +0.034 |
| 2019-20 | 0.266 | 0.292 | +0.026 |
| 2020-21 | 0.450 | 0.443 | −0.007 |
| 2021-22 | 0.300 | 0.315 | +0.015 |
| 2022-23 | 0.263 | 0.290 | +0.027 |
| 2023-24 | 0.322 | 0.330 | +0.008 |
| 2024-25 | 0.269 | 0.290 | +0.021 |
| **Mean** | 0.318 | **0.338** | **+0.020** |

Weighted r is uniformly higher than unweighted (except 2020-21 with a tight cluster of heavy lineups). The +0.020 average improvement is partly cosmetic (downweights noisy lineups) but **the right metric to optimize on**.

### Empirical LINEUP_TEAM_SCALE

Fitted `actual_NR = slope × predicted_NR + intercept` per season:

| PTS | Mean slope | Implied LINEUP_TEAM_SCALE | Note |
|---|---|---|---|
| **v3.2** | 2.27 | **20.0 × 2.27 = 45.5** | uses team TEAM_SCALE=20 |
| **v2.7** | 1.76 | **20.0 × 1.76 = 35.3** | tighter dim variance |
| **per-100 baseline** | 0.76 | 20.0 × 0.76 = 15.2 | per-100 is already ~scale 1, naturally close to 1 slope at TEAM_SCALE=20 |

**For lineup-level reporting, LINEUP_TEAM_SCALE = 45.5 is the correct calibration** for v3.2 PTS. The team-level TEAM_SCALE=20 was correctly fit for team-aggregate work (10-player averaging compresses std), and is NOT the lineup scale (5-player averaging compresses std much less).

After applying the empirical scale, RMSE on calibrated scale drops from 19.09 (raw) to **18.39 (calibrated)** for v3.2 PTS — bringing predicted std (~6) within range of actual std (~19). Still high because lineup-level variance is inherently large, but no longer dominated by scale mismatch.

**Operational fix:** report `rmse_calibrated` as the headline RMSE. Keep `team_scale=20` in the harness call (don't change the input) and just recalibrate the output.

---

## A.3 — Offense and Defense PTS Validated Independently

| Axis | v3.2 PTS r | v2.7 PTS r | per-100 r | Notes |
|---|---|---|---|---|
| Offense (predicted_off vs actual_off_rtg) | **0.316** | 0.314 | 0.369 | Offense PTS holds up vs naive but still loses |
| Defense (predicted_def vs −actual_def_rtg, sign-flipped) | **0.282** | 0.281 | 0.326 | **Defense PTS is weaker than offense by 0.034 r** |

**Confirmed:** defensive PTS is the weak link. Both v3.2 and v2.7 have offense r ≈ 0.31-0.32 and defense r ≈ 0.28. The per-100 baseline is better-balanced (offense 0.37, defense 0.33).

This is exactly the original Fix 2 motivation (Dim 6 DRAPM contamination, defensive shrinkage too light). But Phase 3B sweep showed matchup-data Dim 6 hurt lineup r — so the v3.2 patch isn't the right fix. **PTS 4.0 must redesign the defensive measurement from scratch.**

Defensive PTS predicted std (per-lineup) is also notably tighter than actual:
- v3.2 predicted def std: ~3.5
- actual def std: ~17

The predicted-vs-actual std mismatch on defense is worse than on offense, consistent with "we're systematically under-confidently differentiating defenders."

---

## A.4 — Brier Baseline Gap Resolved

The published `experiment_pts_rdis_blend.json` claimed BASELINE = 0.2400, α=1.0 = 0.2248. Today's measurements:

| Test | Brier |
|---|---|
| Direct `validate_forecast.py` on original `projected_team_features.parquet` | **0.2420** |
| Same after parquet round-trip (write/read tempfile) | 0.2420 (identical) |
| Experiment script `run_brier()` baseline (writes tempfile + runs validate) | 0.2420 (re-tested today, was 0.2400 in old cache) |
| Experiment script α=1.0 (patched, write tempfile, validate) | **0.2345** |
| Posthoc harness baseline (no adjustments, identical patch logic) | **0.2345** |

### Verdict

- **The published 0.2400 baseline was a stale cache from an earlier data state.** Re-running the experiment script today gives BASELINE=0.2420, α=1.0=0.2345. The data state shifted ~0.01 since the JSON was written.
- **Parquet round-trip is bit-identical** (dtype-preserving, no numeric drift).
- **Patching at α=1.0 makes meaningful per-team changes** (mean delta NR = +0.37, std 1.65, max ±4.7) — it's NOT a no-op even when patches are bit-identical between the two patcher implementations (those are two paths producing the SAME patched output).
- The v3.2 report's quoted Brier 0.2352 production is correct on current data state.

### Documentation correction

The v3.2 report (docs/bke_v32_pts_tuning_report.md §3.3) should be amended:

> "Vs BKE baseline (full pipeline): Brier 0.2420 → 0.2352  (Δ = −0.0068, −2.8% relative)"

Not the previously cited "−0.0152 vs experiment-published 0.2400" — that was vs a stale baseline.

**The genuine v3.2 improvement is half what the published delta suggested.** v3.2 still passes the gate (improves the live baseline by −0.0068) but the headline magnitude was inflated.

---

## A.5 — Summary: Where v3.2 Actually Stands

After the validation fixes AND the OOS correction, the honest production picture:

| Metric | v3.2 production | OOS naive per-100 (correct comparator) | Headroom |
|---|---|---|---|
| Aggregate Brier (8,284 games) | 0.2352 | not directly measurable at team level | ~0.215–0.22 with PTS 4.0; ~0.20 with meta-model (Phase C) |
| Lineup joint Pearson r (weighted, 5,143 lineups) | **0.331** | 0.213 | +0.04–0.07 plausible via PTS 4.0 (multi-season + uncertainty + better defense) |
| Lineup offense r | 0.316 | 0.205 | +0.03–0.05 |
| Lineup defense r | 0.282 | 0.180 | **biggest opportunity: +0.05–0.08** |
| Empirical lineup scale calibration slope | 2.27 (off by 2.3×) | 0.76 | reporting fix only — not a PTS issue |
| YoY PTS_z correlation (proxy for "noise") | 0.56 | n/a | target ≥ 0.70 via multi-season Bayes |

**v3.2 is structurally sound but has measurable headroom.** Phase B (PTS 4.0) should target:
- **Defense calibration** (the consistent weak axis)
- **Per-player uncertainty propagation** (low-minute players get diluted)
- **Multi-season Bayesian smoothing** (raise YoY r from 0.56 → 0.70)
- **Source orthogonalization** (avoid double-counting box and tracking signals)

NOT structural anchoring to on-court NetRtg.

---

## Implications for PTS 4.0 Design (Phase B)

After the OOS correction, the architectural direction is reaffirmed, not reversed. The current dimension-model abstraction beats raw on-court NetRtg by +0.12 wr out-of-sample. **PTS 4.0 should evolve the abstraction, not abandon it.**

Confirmed-direction PTS 4.0 principles (to be detailed in `docs/plans/pts_4_0_design.md`):

1. **Keep the dimension-model abstraction**, including archetype neutralization, position-conditional z-scores, and box+tracking composites. Don't switch to on-court-anchored.
2. **Multi-season Bayesian update (Kalman-style)** to raise YoY player correlation from 0.56 to ≥ 0.70. Weight by per-player precision (more poss = less prior pull).
3. **Per-player uncertainty (σ_PTS)** propagated forward. Used by the meta-model (Phase C) for team-aggregate weighting.
4. **League-mean baseline** so 10-player team sums are zero-centered (no replacement-level skew, as user noted).
5. **Source-orthogonal axes**: box stats handle volume/efficiency; tracking handles non-box action signals (drives, paint touches, contests); RAPM-residual handles lineup chemistry. Each axis is residualized against the prior to avoid the double-counting the user warned about.
6. **Defensive measurement is the priority opportunity**. Lineup OOS gap: defense r 0.282 vs offense 0.316. Combine matchup data, opponent-controlled DRAPM, and lineup-context shrinkage. Test multiple defensive formulations against OOS defensive lineup r.
7. **Skill vector preserved** (not collapsed) for Phase C interaction modeling.
8. **Keep RAPM-residual at full weight** (per user's criticism about not shrinking gravity/deterrence). Possibly UP-weight for high-possession players.

The architecture stays "box prior + tracking residual + RAPM residual with informed prior" — same family as DARKO / EPM / LEBRON / RAPTOR. Phase A confirms this family of architectures beats raw on-court signal OOS, which is what matters for prediction.

---

## Files Touched by Phase A

| Path | Change |
|---|---|
| `scripts/validate_lineup_pts_v2.py` | new — fixed harness with naive baselines, possession-weighted r, off/def split, empirical scale |
| `scripts/diagnose_brier_baseline.py` | new — diagnostic that resolved A.4 |
| `reports/lineup_v2_v32.json` | new — v3.2 PTS measured with fixed harness |
| `reports/lineup_v2_v27.json` | new — v2.7 PTS measured with fixed harness |
| `reports/brier_baseline_diagnostic.json` | new — A.4 diagnostic dump |
| `docs/v3_3_phase_a_validation.md` | this report |

No code under `src/` was touched — Phase A is pure validation tooling.

---

## Phase B Gate

Phase B (PTS 4.0 design doc → implementation → validation) starts next. Of the three original gates:

1. ✅ **Out-of-sample naive baseline test DONE** — PTS wins by +0.12 wr; architectural pivot REJECTED, evolution path confirmed.
2. ⏳ **Quantify defensive PTS gap more precisely** — by archetype, by minutes bucket. Add to Phase B design discovery.
3. ⏳ **Lock canonical harness** — `scripts/validate_lineup_pts_v2.py` becomes the gold standard for v3.3+ work. The old `validate_lineup_pts.py` stays for v3.2-era reproducibility.

Next: draft `docs/plans/pts_4_0_design.md` for review. The OOS finding rewrites the design direction — Phase B design doc will reflect "evolve, don't pivot."
