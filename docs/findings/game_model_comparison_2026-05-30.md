# Game-Level Model Comparison — Gaussian vs GBDT vs Elo

**Date:** 2026-05-30
**Author:** Step 5 (walk-forward GBDT game model)
**Code:** `src/simulation/gbdt_game_model.py`
**Audit:** `scripts/audit_forecast_leakage.py` (12/12 checks pass)
**Raw report:** `reports/gbdt_forecast_validation.json`

---

> **⚠ CORRECTION (2026-05-30, post-publication):** This report originally framed
> the Gaussian leg as "the BKE player-impact pipeline." That attribution is
> **wrong**. Code inspection (`build_all_season_projections.py:50-78`,
> `build_2025_26_projections.py:27-92`) shows the game model's team strength is
> `team_net_rating_projected = 0.70 × (prior-season PLUS_MINUS point margin)`,
> regressed 30% to the mean. **Neither PTS v4.0 nor BKE player-impact is in the
> game-prediction path at all** — the archetype/talent columns in
> `projected_team_features` are vestigial template fields the forecaster never
> reads. So the comparison below is really *last year's margin* vs *this year's
> margin (Elo)*, not BKE vs Elo. Passages affected: §0, §3, §5 (corrected inline).

## 0. TL;DR

We trained a LightGBM "stacking" model on top of the production Gaussian
forecaster to push game-level Brier toward the 0.210 CLV gate. It worked
(−0.016 Brier vs Gaussian). But the headline is uncomfortable:

> **A zero-cost sequential Elo rating beats the production game model** — 0.2193
> aggregate OOS Brier vs the Gaussian's 0.2408, and 0.2092 on 2025-26 (the only
> model that clears the ≤ 0.210 gate).

A full contamination/leakage audit (Section 4) confirms this is **real model
behavior, not diluted or leaked data**. But the deeper finding is *what the
production model actually uses*: its "team rating" is **70% of last season's
point margin** — not PTS, not BKE. So Elo (this season's margin, updated game by
game) winning is almost tautological: current form beats year-old form. The
r ≈ 0.55 "rating ↔ actual margin" correlation is just **year-over-year team
persistence**, and says nothing about PTS/BKE quality — *because those metrics
were never wired into the game model.*

---

## 1. What each model is

| Model | Information used | How it forms a win probability |
|---|---|---|
| **Gaussian** (production baseline) | Preseason BKE team rating + YTD-blended margin + rest/HCA | `Φ(Δμ / σ_game)`, σ_game = √(σ_home² + σ_away² + σ_league²) |
| **GBDT stack** (new, Step 5) | All of the above (`gauss_p`, per-season-z-scored `Δμ`) **plus Elo** (`elo_p`) plus rest flags | LightGBM, walk-forward, isotonic-calibrated |
| **Elo** (new, Step 5) | Only the sequence of prior game results this season (+ 0.75 carry from last season) | Logistic on rating diff: `1/(1+10^(-(Rh+HCA−Ra)/400))`, K=20 |

All three are evaluated **walk-forward, strictly out-of-sample**: for target
season T, train only on seasons < T, predict T. 8,279 games over 2019-20…2025-26.

---

## 2. Results

### Aggregate OOS Brier (8,279 games, 2019-26)

| Model | Brier | Log-loss | Accuracy |
|---|---|---|---|
| Gaussian only | 0.2408 | 0.6746 | 0.574 |
| GBDT stack | 0.2250 | 0.6978 | 0.631 |
| **Elo only** | **0.2193** | **0.6283** | **0.648** |
| blend 50/50 (G+GBDT) | 0.2264 | 0.6441 | 0.627 |
| blend 30/50/20 (G+GBDT+Elo) | 0.2223 | 0.6353 | 0.633 |

Reference points: chance = 0.25; constant home-rate (0.557) ≈ 0.247;
Kalshi closing lines (2025-26) = 0.2045.

### Per-season Brier

| Season | Gaussian | GBDT | **Elo** | blend 30/50/20 | Note |
|---|---|---|---|---|---|
| 2019-20 | 0.2500 | 0.2397 | 0.2198 | 0.2271 | COVID stoppage + bubble |
| 2020-21 | 0.2479 | 0.2363 | 0.2292 | 0.2339 | no/limited fans |
| 2021-22 | 0.2431 | 0.2291 | 0.2229 | 0.2267 | |
| 2022-23 | 0.2375 | 0.2293 | 0.2278 | 0.2264 | |
| 2023-24 | 0.2398 | 0.2163 | 0.2137 | 0.2178 | |
| 2024-25 | 0.2338 | 0.2145 | 0.2139 | 0.2133 | |
| **2025-26** | 0.2358 | 0.2133 | **0.2092** | 0.2128 | clears ≤ 0.210 gate |

Elo is best or tied-best in **every** season. The gap narrows over time as
prior-year margins became more representative (Gaussian 0.250 → 0.236), but
never closes.

---

## 3. Does it make basketball sense? Yes.

**Why Elo wins.** NBA outcomes at the game level are dominated by *current* team
strength, which drifts within a season (trades, injuries, role changes, chemistry,
tanking). Elo is a pure recency-weighted record: every result nudges a team's
rating, so by midseason it reflects exactly how good a team is *right now*. Its
end-of-season rating correlates with realized season margin at **r = 0.958** — it
is, by construction, an almost-perfect mirror of what happened. Published NBA Elo
systems (e.g., FiveThirtyEight) land at ~0.21–0.22 Brier, so our 0.2193 is exactly
where a competent vanilla Elo should be. Nothing surprising — it's a strong,
well-understood baseline.

**Why the production game model lags.** It rates a team as 70% of *last season's*
point margin — a reasonable preseason prior (year-over-year persistence r ≈ 0.55)
but blind to everything that changed this year until the YTD blend slowly catches
up. Three compounding issues:

0. **Stale, talent-free signal.** The rating is prior-year scoreboard margin —
   no PTS, no BKE, no roster awareness. A team that added a star or lost one to
   injury looks identical to last year until games accumulate.
1. **Scale bug in the rating.** `actual_net_rating = PLUS_MINUS.sum()/MIN.sum()×48`
   divides by **team-minutes (240/game)**, returning margin **÷ 5**, not pts/100
   (the code comment "per 100 poss" is wrong). A +5/game team shows as +1.0.
2. **In-season signal compression (dilution by design).** The YTD blend further
   rescales current margin into compressed units:
   `ytd_bke = ytd_avg_margin × (preseason_std/6.0)`, factor ≈ **0.115** — shrinking
   "this team is actually great this year" almost 9×.
3. **Oversized variance.** σ_league = 3.0 in *points*, but μ lives in the
   ÷5-compressed scale (std ≈ 0.7). So `z = Δμ/σ_game` averages ≈ 0.8 and `Φ(z)`
   rarely leaves 0.45–0.65 — **systematically under-confident** (gauss_p std =
   0.083 vs the ~0.16 a calibrated model needs). Under-confidence is pure Brier tax.

In basketball terms: the game layer is quoting a year-old, 5×-muted team rating
and then refusing to commit to it. Elo speaks in current scoreboard-margin units
and updates every night, so it says "this team is clearly better tonight" out loud.

**Why the GBDT helps but can't win.** The tree re-expands the compressed signal
and reads Elo as a feature, recovering most of the gap (0.2408 → 0.2250). But
on ~1–8k rows it can't out-resolve the single feature (`elo_p`) that already
carries the dominant signal; mixing in the miscalibrated Gaussian features adds
variance. A strong lone feature beating a model that dilutes it is a textbook
outcome, not a bug.

---

## 4. Contamination & leakage audit — was the data diluted or contaminated?

Run: `python3 scripts/audit_forecast_leakage.py` → **12/12 checks pass, 0 fail.**
We checked *both* directions: is Elo's strength fake (leakage), and is BKE's
weakness fake (a join/lag bug)?

| # | Check | Result | Rules out |
|---|---|---|---|
| 1 | No duplicate games / no null features | 9,509 unique IDs, 0 dups, 0 nulls | Double-counting dilution |
| 2 | `home_win` == official W/L | **100.00%** agree (9,509 games) | Mislabeled outcomes / bad join |
| 2 | Home-win rate per season | 0.54–0.59 (lower in bubble) | Side/label swap |
| 3 | `ytd_avg_margin` == strictly-prior expanding mean | max err **0.0** | YTD using the current game (leak) |
| 3 | `blended_mu` formula reproduces exactly | max err 4e-16 | Hidden rating contamination |
| 4 | **Elo future-blindness**: drop later half of season, early preds unchanged | max Δp = **0.0** | Elo peeking at future games |
| 4 | **Placebo**: shuffling within-season order degrades Elo | 0.2192 → 0.2204 | Elo "winning" via a structural quirk |
| 5 | Prior-year-margin rating ↔ actual margin | **r = 0.545** (stable 0.52–0.59/season) | The rating being pure noise |
| 5 | End-of-season Elo ↔ actual margin | r = 0.958 | (sanity: form mirrors results) |
| 6 | `gauss_p` compressed / under-confident | std 0.083 | Confirms weakness = calibration, not data |

**Conclusion:** the comparison is trustworthy. Elo is not leaking (future-blind,
and temporally meaningful). The production rating is not pure noise — prior-year
margin carries real r ≈ 0.55 persistence — but it is **stale, 5×-scale-bugged,
9×-diluted, and under-confident**, and (critically) **does not contain PTS or
BKE at all**. The weakness is a calibration/scaling/staleness problem in the game
layer, on top of leakage-free data — not corrupted data.

> Leakage-safety note: per-season z-scoring of `Δμ` is transductive feature
> scaling (uses that season's rating spread, **no outcomes**); a causal
> expanding-window variant moves Brier negligibly because `gauss_p` and `elo_p`
> dominate. No outcome information crosses the train/test boundary.

---

## 5. What it means for our data, metrics, and pipeline

**The data is fine.** Outcomes, schedule, rest, and YTD lag are all clean and
correct. No remediation needed at the ingest/normalization layer.

**The biggest finding is a gap, not a bug: PTS/BKE was never wired into the game
model.** The prediction path uses prior-year margin. So we have *not yet tested*
whether the player-impact engine predicts games — the premise the whole
Vegas-beating goal rests on. Two tracks follow:

*Track 1 — mechanical fixes (cheap, but converge to ~Elo, not Vegas):*

1. **Recalibrate the game layer.** *Measured result: NOISE (see §5b).* Walk-forward
   isotonic calibration of `gauss_p` gave Δ = −0.0005 (95% CI straddles 0). Any
   monotonic scale/σ fix has the same ceiling: **you cannot calibrate your way out
   of a stale, low-resolution signal.** The Gaussian's problem is lack of
   *resolution* (prior-year margin barely separates games), not miscalibration.
   Drop this one.
2. **Stop diluting the in-season signal.** The 0.115 margin→unit rescale throttles
   exactly the information that wins games; express the rating in real points and
   let current-season form drive late-season games. *Note: this largely reinvents
   Elo* — current margin is current margin.
3. **Adopt Elo as a first-class pipeline citizen.** Free, leakage-free, best
   single model, strong GBDT feature. Interim production game model: **GBDT stack
   with Elo**, or Elo alone. CLV export already ships Elo as `bke_home_win_prob`
   (`reports/bke_gbdt_game_forecasts.parquet`).

*Track 2 — the actual experiment (the only credible alpha source):*

4. **Wire PTS/BKE player-impact into team strength and test incremental CLV vs
   Elo — specifically where Elo is blind** (early season, post-trade,
   post-injury). This is the unrun experiment. If player-impact adds nothing even
   there, prediction is at a true bottleneck; if it adds CLV in those windows,
   that is the wedge.

**For Sleeve C / alpha-readiness.** Still **not alpha-ready.** Elo's 2025-26
Brier (0.2092) reaches *parity* with Kalshi (0.2045), not superiority → expected
CLV ≈ 0 — a massive improvement over the Gaussian's −0.069 CLV, but the mechanical
fixes top out at ~Elo and cannot cross the ~0.01 Brier gap to the market. Path to
CLV > 0.02 runs through Track 2, then a walk-forward game-level CLV test vs Kalshi
closing lines, before any allocation.

---

## 5b. Micro-adjustment experiment (measured 2026-05-30)

`python3 scripts/experiment_microadjustments.py` — walk-forward, 8,279 OOS games,
paired bootstrap (2,000 resamples) on per-game squared error vs the Gaussian
baseline. Verdict = HELP only if the 95% CI excludes 0.

| Adjustment | Brier | Δ vs Gaussian | 95% CI | Verdict |
|---|---|---|---|---|
| M0 Gaussian (baseline) | 0.2408 | — | — | — |
| **Adj1** isotonic-recalibrated Gaussian | 0.2403 | −0.0005 | [−0.0019, +0.0008] | **NOISE** |
| **Adj2** undiluted current-season margin | 0.2225 | −0.0183 | [−0.0215, −0.0152] | **HELP** |
| Elo (reference) | 0.2193 | −0.0215 | [−0.0247, −0.0185] | HELP |
| **Adj3** GBDT stack | 0.2250 | −0.0158 | [−0.0190, −0.0126] | **HELP** |
| Adj3+ GBDT stack **+ current margin feature** | 0.2235 | −0.0173 | [−0.0205, −0.0141] | HELP |

**Does any adjustment beat Elo?** No — all are significantly *worse* than Elo
(M2 closest at +0.0032, CI [+0.0011, +0.0053]). The mechanical fixes **converge
toward Elo and stop there**, empirically confirming the ~0.01 Brier wall to the
market.

**Takeaways:**
- **Adj1 = drop.** Calibration can't add resolution to a stale signal.
- **Adj2 = keep as a feature, not a standalone.** "Undiluting" the in-season
  signal recovers most of the gap — but it's just Elo-lite (flat season margin
  vs Elo's recency weighting), so prefer Elo.
- **Adj3 = keep.** Best multi-signal model; adding the undiluted current margin
  as a feature nudges it from 0.2250 → 0.2235. Still below Elo alone, because its
  stale Gaussian features add noise. A stack of only {Elo, current-margin, rest}
  is the natural next tweak — but that is still Track-1 plumbing, not alpha.

---

## 6. Reproduce

```bash
python3 src/simulation/gbdt_game_model.py        # train + evaluate all legs
python3 scripts/audit_forecast_leakage.py        # 12-check contamination audit
python3 scripts/experiment_microadjustments.py   # Adj1/2/3 with paired-bootstrap verdicts
# then (optional) CLV vs market:
python3 scripts/fetch_kalshi_closing_lines.py    # compare 2025-26 export to Kalshi
```
