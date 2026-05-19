# BKE Performance Handoff — Sleeve C (Prediction Market Model Edge)

> **Audience:** A Claude Code session in the Robinhood trading-bot repo making a
> capital-allocation decision about Sleeve C. This document is self-contained.
> **Generated:** 2026-05-19. **Source branch:** `personal`.
> **Investigator's honesty note (first pass — 2026-05-19):** First-pass checkout
> contained **code and reference docs only** — no `reports/*.json`, no `data/`.
> All numbers were transcribed from narrative snapshots and flagged as unverified.
> **This is the data-present re-verification pass (also 2026-05-19).** All four
> required data artifacts were confirmed present. `validate_sim.py` ran
> successfully and regenerated `reports/simulation_step1_validation.json`.
> `pytest -q` passed (3 tests, 176s). Every metric below marked with ✓ was
> directly extracted from the live JSON this session. Numbers that differ from
> the prior reference-doc values are flagged **[CORRECTED]**. Numbers that
> confirm the reference-doc are flagged **[CONFIRMED]**.

---

## Section 1: What BKE Is

BKE (Basketball KPI Engine) is a possession-level NBA player-impact model that
fuses regularized plus-minus (RAPM), an external talent feed (DARKO), and a
box/tracking/archetype prior into one season-normalized player value score per
player-season. Those player scores roll up into projected team net ratings,
which feed a Gaussian game-margin model that outputs a win probability for any
NBA matchup — the quantity relevant to prediction-market (Kalshi/Polymarket)
moneyline pricing.

---

## Section 2: Pipeline Completeness Matrix

| Stage | Complete? | Last Run | Output Location | Present in this checkout? |
|---|---|---|---|---|
| Data ingest (PBP, team logs, players) | Yes | 2026-03 | `data/historical/` | **Yes** — `team_game_logs.parquet` confirmed ✓ |
| RAPM / ORAPM / DRAPM modeling | Yes | 2026-03 | `data/processed/player_rapm.parquet` | **Partial** — parquet present via player_impact_profiles |
| BKE decomposition v1.5–v3.1 | Yes | 2026-03-01 | `data/processed/bke/`, `reports/bke_v*.json` | **Yes** — 7 BKE reports present ✓ |
| Player Eval Core Step 1–3 | Yes | 2026-03-08 | `data/processed/player_eval/`, `reports/player_eval_step*.json` | **Yes** — all present ✓ |
| Simulation Step 1 (game model) | Yes | 2026-05-19 (re-run) | `reports/simulation_step1_*.json` | **Yes** — re-generated this session ✓ |
| Simulation Step 2 (lineup proj.) | Yes | 2026-03-08 | `reports/simulation_step2_*.json` | **Yes** — present ✓ |
| Forecast pipeline (walk-forward) | Yes | 2026-03-11 | `reports/forecast_*.json` | **Yes** — 6 forecast reports present ✓ |
| Player-stat sim (box scores) | Yes, with known blocker | 2026-03-11 | `data/processed/simulation/*` | Code yes / reports present ✓ |
| **Game model vs. Vegas/Kalshi** | **NEVER DONE** | — | — | — |
| **Genuine out-of-sample game-level Brier** | **NEVER MEASURED** | — | — | — |
| Phase 4 "Model Development" (overall_plan) | **Unchecked / incomplete** | — | — | — |
| Phase 5 "Evaluation & Interpretability" | **Unchecked / incomplete** | — | — | — |

`overall_plan.DO_NOT_CHANGE.txt` shows Phases 1–3d checked `[x]`; **Phase 4
(Model Development) and Phase 5 (Evaluation) are still `[ ]`**. The simulation
work is a parallel "Stream F/G" effort, not the planned Phase 4 game-outcome
model.

---

## Section 3: Game-Level Prediction Performance

All figures below extracted directly from `reports/simulation_step1_validation.json`
regenerated this session (2026-05-19) by running `python3 src/simulation/validate_sim.py`.

| Season | Games | Brier | Log Loss | Accuracy | Margin RMSE | Margin MAE | Home WR act / pred |
|---|---|---|---|---|---|---|---|
| 2022-23 | 1230 | **0.2270** ✓ | **0.6507** ✓ | **64.1%** [CORRECTED] | **12.87** ✓ | 10.11 | 0.581 / 0.580 ✓ |
| 2023-24 | 1230 | **0.2154** [CORRECTED] | **0.6209** ✓ | **65.4%** [CORRECTED] | **14.14** ✓ | 11.06 | 0.543 / 0.574 [CORRECTED] |
| 2024-25 | 1225 | **0.2053** ✓ | **0.5979** [CORRECTED] | **69.3%** [CORRECTED] | **13.80** ✓ | 10.75 | 0.544 / 0.573 ✓ |

Discrepancies vs. prior reference doc: Brier 2022-23 was 0.2281 → actual 0.2270;
accuracy 2022-23 was 64.9% → actual 64.1%; Brier 2023-24 was 0.2148 → actual
0.2154; accuracy 2023-24 was 65.6% → actual 65.4%; accuracy 2024-25 was 69.2%
→ actual 69.3%. All differences are within 0.001 Brier / <1% accuracy — the
prior session's transcribed values were nearly correct but not precisely so.
**Margin MAE was missing from the prior doc; freshly extracted values are
10.1/11.1/10.7 across the three seasons.**

### Is this a genuine hold-out test? **NO — it is in-sample / look-ahead-contaminated.**

**Confirmed by source code this session.** `src/profile_aggregate/team_feature_aggregation.py`,
line 873:

```python
src_path = profiles_path or (PROJECTED_PROFILES_PATH if forecast_mode else PLAYER_PROFILES_PARQUET)
```

In BACKTEST mode (`forecast_mode=False`), the function reads from
`PLAYER_PROFILES_PARQUET` — the **same-season** player profiles built from
end-of-season actual player performance. The model is thus asked "given how
these teams actually performed this season, who won each game" — a retrodiction,
not a forecast. The Brier ≈ 0.205–0.227 figures are **not usable as evidence
of market edge.** (Line 865 also confirms: `mode_label = "FORECAST" if
forecast_mode else "BACKTEST"`.)

The genuinely leakage-free path is the **forecast pipeline** (Section 5). It
yields no game-level Brier because it simulates a synthetic schedule.

---

## Section 4: Full Calibration Table

Data extracted directly from `reports/simulation_step1_validation.json` this session.
`aggregate_calibration_error` (count-weighted MCE across all bins/seasons) = **0.0562** ✓
(Prior doc stated 0.060 — **[CORRECTED]**: prior doc rounded up; exact value is 0.0562.)

### Per-season 10-bin calibration tables

**2022-23** (n=1230 games)

| Bin (predicted WP) | Count | Pred. rate | Actual rate | Calib. error | Naive err (|0.5−act|) |
|---|---|---|---|---|---|
| 0.0–0.1 | 4 | 0.0383 | 0.2500 | 0.2117 | 0.2500 |
| 0.1–0.2 | 66 | 0.1576 | 0.2879 | 0.1303 | 0.2121 |
| 0.2–0.3 | 73 | 0.2496 | 0.2877 | 0.0381 | 0.2123 |
| 0.3–0.4 | 66 | 0.3548 | 0.3485 | 0.0064 | 0.1515 |
| 0.4–0.5 | 165 | 0.4581 | 0.5212 | 0.0631 | 0.0212 |
| 0.5–0.6 | 260 | 0.5522 | 0.6000 | 0.0478 | 0.1000 |
| 0.6–0.7 | 271 | 0.6480 | 0.6458 | 0.0022 | 0.1458 |
| 0.7–0.8 | 154 | 0.7417 | 0.6299 | 0.1118 | 0.1299 |
| 0.8–0.9 | 89 | 0.8531 | 0.8090 | 0.0441 | 0.3090 |
| 0.9–1.0 | 82 | 0.9283 | 0.7805 | **0.1478** | 0.2805 |

**2023-24** (n=1230 games)

| Bin (predicted WP) | Count | Pred. rate | Actual rate | Calib. error | Naive err (|0.5−act|) |
|---|---|---|---|---|---|
| 0.0–0.1 | 31 | 0.0699 | 0.1290 | 0.0591 | 0.3710 |
| 0.1–0.2 | 58 | 0.1510 | 0.3103 | 0.1594 | 0.1897 |
| 0.2–0.3 | 114 | 0.2478 | 0.2982 | 0.0504 | 0.2018 |
| 0.3–0.4 | 132 | 0.3554 | 0.3485 | 0.0069 | 0.1515 |
| 0.4–0.5 | 135 | 0.4461 | 0.4815 | 0.0354 | 0.0185 |
| 0.5–0.6 | 157 | 0.5492 | 0.4841 | 0.0651 | 0.0159 |
| 0.6–0.7 | 158 | 0.6484 | 0.5759 | 0.0725 | 0.0759 |
| 0.7–0.8 | 176 | 0.7538 | 0.6477 | **0.1061** | 0.1477 |
| 0.8–0.9 | 168 | 0.8519 | 0.7798 | 0.0722 | 0.2798 |
| 0.9–1.0 | 101 | 0.9423 | 0.8812 | 0.0611 | 0.3812 |

**2024-25** (n=1225 games)

| Bin (predicted WP) | Count | Pred. rate | Actual rate | Calib. error | Naive err (|0.5−act|) |
|---|---|---|---|---|---|
| 0.0–0.1 | 21 | 0.0563 | 0.0000 | 0.0563 | 0.5000 |
| 0.1–0.2 | 79 | 0.1554 | 0.1646 | 0.0091 | 0.3354 |
| 0.2–0.3 | 107 | 0.2571 | 0.3364 | 0.0793 | 0.1636 |
| 0.3–0.4 | 109 | 0.3516 | 0.3486 | 0.0030 | 0.1514 |
| 0.4–0.5 | 154 | 0.4504 | 0.3701 | 0.0803 | 0.1299 |
| 0.5–0.6 | 150 | 0.5501 | 0.5600 | 0.0099 | 0.0600 |
| 0.6–0.7 | 169 | 0.6472 | 0.6213 | 0.0259 | 0.1213 |
| 0.7–0.8 | 177 | 0.7488 | 0.6949 | 0.0539 | 0.1949 |
| 0.8–0.9 | 156 | 0.8492 | 0.7756 | 0.0735 | 0.2756 |
| 0.9–1.0 | 103 | 0.9372 | 0.8738 | 0.0634 | 0.3738 |

### Calibration pattern summary

The extreme bins (0.0–0.1 and 0.9–1.0) are consistently the most miscalibrated
across all seasons. The 0.9–1.0 bin shows overconfidence: the model assigns >90%
win probability but actual win rates are only 78–88%. The mid-range bins
(0.4–0.7) are generally well-calibrated. This is the in-sample pattern — actual
out-of-sample calibration is unknown.

### Baseline comparisons (computed from JSON data this session)

| Predictor | Season | Brier | MCE | Notes |
|---|---|---|---|---|
| Always 0.5 (coin flip) | — | 0.2500 | — | Definitional |
| Always pick home (p=1) | 2022-23 | 0.4195 | — | = 1 − 0.5805 |
| Always pick home (p=1) | 2023-24 | 0.4569 | — | = 1 − 0.5431 |
| Always pick home (p=1) | 2024-25 | 0.4555 | — | = 1 − 0.5445 |
| Home-rate prior (~0.55) | 2022-23 | 0.2435 | — | p(1−p) |
| Home-rate prior (~0.54) | 2023-24 | 0.2481 | — | p(1−p) |
| Home-rate prior (~0.54) | 2024-25 | 0.2480 | — | p(1−p) |
| Naive MCE (predict 0.5) | All seasons | — | 0.1657 | Count-weighted overall |
| **BKE model (in-sample)** | 2022-23 | **0.2270** | 0.0564 | In-sample only |
| **BKE model (in-sample)** | 2023-24 | **0.2154** | 0.0660 | In-sample only |
| **BKE model (in-sample)** | 2024-25 | **0.2053** | 0.0461 | In-sample only |
| **BKE model (overall MCE)** | All | — | **0.0562** | In-sample only |
| BKE model (genuine OOS) | — | **UNKNOWN** | **UNKNOWN** | Never measured |

**The model's MCE of 0.0562 vs. naive MCE of 0.1657 looks impressive — a 66%
reduction.** But this advantage is in-sample and in the calibration-error metric
specifically (which measures bin-level accuracy, not raw predictive power). The
Brier gain over a home-rate constant (0.205–0.227 vs. 0.244–0.250) is more
modest: roughly 0.015–0.045 Brier improvement, and entirely in-sample.

---

## Section 5: Season-Level Accuracy

### In-sample backtest (same caveat as Section 3 — leaky)

Extracted directly from `reports/simulation_step1_validation.json` this session.

| Season | MAE (wins) | RMSE | Correlation | Max Over | Max Under |
|---|---|---|---|---|---|
| 2022-23 | **5.47** [CORRECTED] | **6.74** ✓ | **0.831** [CORRECTED] | **+15.7** [CORRECTED] | **−8.7** [CORRECTED] |
| 2023-24 | **5.93** [CORRECTED] | **7.15** [CORRECTED] | **0.868** ✓ | **+12.4** ✓ | **−18.6** [CORRECTED] |
| 2024-25 | **3.71** [CORRECTED] | **4.42** [CORRECTED] | **0.948** [CORRECTED] | **+9.0** [CORRECTED] | **−8.8** [CORRECTED] |
| **Overall** | **5.03** [CORRECTED] | **6.22** ✓ | **0.887** ✓ | — | — |

(Prior doc showed MAE 5.57/6.06/3.76/5.13; freshly extracted values are
5.47/5.93/3.71/5.03. All discrepancies are small — prior reference doc was
from a slightly older run.)

### 5 largest absolute errors — in-sample backtest

From `reports/simulation_step1_season_results.json` (in-sample, same-season data):

| Team-Season | Projected W | Actual W | Error | Note |
|---|---|---|---|---|
| CHI 2023-24 | 20.4 | 39 | **−18.6** | Significant underestimate |
| WAS 2022-23 | 50.7 | 35 | **+15.7** | Overestimate |
| UTA 2022-23 | 50.9 | 37 | **+13.9** | Overestimate |
| SAS 2023-24 | 34.4 | 22 | **+12.4** | Overestimate |
| DET 2023-24 | 25.7 | 14 | **+11.7** | Overestimate |

Note: these are **in-sample** errors (same-season player data → same-season wins).
Even with the leaky rating, errors of 10–19 wins occur. The in-sample errors
are smaller than the walk-forward errors below precisely because of leakage.

### Genuine walk-forward forecast (leakage-free — the number that matters)

Extracted from `reports/forecast_season_results.json` (end_of_season scenario),
generated by the forecast pipeline using prior-season player data to project
the next season.

| Season projected | Forecast MAE (wins) | Forecast RMSE | Forecast r |
|---|---|---|---|
| 2023-24 | **7.71** ✓ | 9.74 | 0.706 |
| 2024-25 | **8.89** ✓ | 11.25 | 0.617 |

The repo's own benchmark: *"Vegas lines typically achieve MAE of 5–6 wins."*
The leakage-free model is **~2–4 wins worse per team than the Vegas/market
baseline it would have to beat** to make money on season-win or derived game
markets. The 2024-25 correlation (0.617) is lower than 2023-24 (0.706),
suggesting instability across the two available walk-forward transitions.

### 5 worst walk-forward forecast errors (from `forecast_season_results.json`)

| Team-Season | Projected W | Actual W | Error | Likely cause |
|---|---|---|---|---|
| NOP 2024-25 | 44.9 | 21 | **+23.9** | Zion/Ingram injuries, roster collapse |
| MEM 2023-24 | 50.3 | 27 | **+23.3** | Ja Morant suspension |
| PHI 2024-25 | 46.0 | 24 | **+22.0** | Embiid played only 13 games |
| HOU 2023-24 | 20.1 | 41 | **−20.9** | Young-team breakout |
| POR 2024-25 | 17.0 | 36 | **−19.0** | Breakout development |

Note: prior doc cited values from loop narrative notes (MEM +26.8, NOP +21.1,
etc.) which differ from the JSON. **The JSON values are authoritative.**
The top errors still cluster on injury/suspension events and young-team
breakouts — categories the model cannot anticipate.

---

## Section 6: BKE Player Metric Stability and Predictive Value

### BKE version predictive_rho table

`predictive_rho` is defined in the experimental layers reports as the **Spearman
correlation between the BKE-derived production proxy and a predictive target
within the same dataset** — specifically, correlation of the composite BKE
score against a production-weighted score (orapm/PTS/TS%/AST) that proxies
for "observed value." This is **not** a game-outcome prediction metric and
**not** a market-price metric. It measures how well BKE rank-orders players
by their observed production.

The BKE backtest reports (v2.6, v2.7, v3.0) measure **cross-season player rank
stability** (train on 2022-24, test on 2024-25, n=263 returning players). This
is also not a game-outcome metric.

| BKE Version | Generated | qualified_players | predictive_rho | Source | Notes |
|---|---|---|---|---|---|
| v1.5 | 2026-02-16 | 950 | — | bke_v15_report.json | Basic report, no rho field |
| v2.0 | 2026-02-16 | 950 | — | bke_v20_report.json | Basic report |
| v2.5 | 2026-02-20 | 950 | — | bke_v25_report.json | Basic report |
| v2.6 | 2026-02-20 | 950 | — | bke_v26_report.json | Basic report; backtest: pts_rho=0.553 |
| v2.7 | 2026-02-20 | 950 | — | bke_v27_report.json | Basic report; backtest: pts_rho=0.590 |
| v2.8 | 2026-02-28 | 950 | — | bke_v28_report.json | Basic report |
| v3.0 | 2026-02-20 | 950 | — | bke_v30_report.json | Basic report; backtest: pts_rho=0.553 |
| v3.1 experimental (baseline 60/40) | 2026-02-28 | 950 | **0.3296** | bke_v31_experimental_layers.json | Baseline config |
| v3.1 experimental (best layer1, 55/45) | 2026-02-28 | 950 | **0.3466** | bke_v31_experimental_layers.json | Best single-layer improvement |
| v3.1 experimental (combined layers 1+6+3) | 2026-02-28 | 950 | **0.3436** | bke_v31_experimental_layers.json | Combined model |
| **v3.1 exp2 rerun base profile** | 2026-03-01 | 950 | **0.3152** | bke_v31_experiment2_production_tilt.json | Layer36 second pass |
| **v3.1 exp2 recommended (λ=0.03)** | 2026-03-01 | 950 | **0.3179** | bke_v31_experiment2_production_tilt.json | Production config |

The BKE backtest cross-season rank correlations (player rank stability,
train→test next season):

| Version | Test season | portable_talent spearman | total_impact spearman | Within-1 tier accuracy |
|---|---|---|---|---|
| v2.6 | 2024-25 | 0.553 | 0.511 | 76.0% |
| v2.7 | 2024-25 | 0.590 | 0.537 | 76.8% |
| v3.0 | 2024-25 | 0.553 | 0.511 | 76.0% |

**Cross-season player BKE carry stability** (year-over-year r of player BKE,
from `reports/forecast_validation.json`):

| Transition | BKE carry r | MPG carry r |
|---|---|---|
| 2022-23 → 2023-24 | 0.4301 | 0.8452 |
| 2023-24 → 2024-25 | 0.4919 | 0.8340 |

BKE carry r ≈ 0.43–0.49 is modest — typical for impact metrics; not exceptional.
MPG carry r ≈ 0.83–0.85 is strong, as expected (playing time is stable year-to-year).

**What `predictive_rho` predicts — clarification:** it is the correlation between
the BKE composite score and a production proxy (orapm=0.22, PTS=0.18, TS%=0.14,
AST=0.12, FGM=0.08, FGA=0.08, FG3M=0.06, FG3A=0.04, FTM=0.04, FTA=0.04
weights) within the experiment. ρ ≈ 0.315–0.347 is moderate internal validity,
not a game-outcome or market metric.

---

## Section 7: Known Gaps and Failure Modes (Phase 6 — explicit Yes/No/Partially)

1. **Compared against Vegas lines or Kalshi-implied prices on specific games?**
   **No.** Zero market comparison exists anywhere in the repo. The only Vegas
   reference is a one-line prose aside ("Vegas ≈ 5–6 win MAE"). No game-by-game,
   no line ingestion, no closing-line-value analysis.

2. **True walk-forward test (train ≤N, test N+1, no N+1 data in BKE ratings)?**
   **Partially.** The *forecast pipeline* is genuinely walk-forward for
   season-win projection (MAE ≈ 7.7–8.9, r ≈ 0.62–0.71). But it runs a
   **synthetic schedule with no real opponents/outcomes**, so it yields **no
   game-level win-probability, Brier, log-loss, or calibration**. The only
   game-level metrics that exist are the **in-sample (leaky)** ones (Section 3).

3. **Known data-completeness issues — injuries, trades, back-to-backs?**
   **Yes, multiple.** (a) No injury/availability forecasting — the 5 worst
   walk-forward errors are all injury/suspension driven. (b) Mid-season trades
   handled via "stint" splitting but flagged as a known bug for name matching.
   (c) `compute_rest_home_back2back.py` exists but rest/B2B is **not** an input
   to the game model — the win-prob formula uses only team `mu`, team `sigma`,
   and a flat 2.0 HCA. (d) A "phantom NAN team" artifact from traded players
   is filtered, not fixed. (e) DARKO data availability rate = 0.0 per
   `modeling_inputs_report.json` — DARKO is entirely absent from the current
   data; the model runs on RAPM+box only. ✓ **[NEW — from modeling_inputs_report.json]**

4. **Known weakest scenario?** **Yes — injury/roster-shock team-seasons**
   (NOP/MEM/PHI: +22–24 win errors in walk-forward) and **high-confidence games**:
   the model is **overconfident in the >90% predicted-WP bin** (actual ≈ 78–88%
   across seasons) — precisely the favorites bucket where prediction markets are
   most efficient and mispriced edges are smallest. HCA is also slightly
   miscalibrated (predicted home WR 0.573–0.580 vs actual 0.543–0.581).

5. **Earliest training season / latest tested season?** Earliest: **2022-23**.
   Latest tested: **2024-25**. Only **3 seasons** of data exist total — a very
   thin base for any walk-forward claim (effectively 2 forecast transitions).

6. **Next planned improvement (loop files)?** `overall_plan` Phase 4 (proper
   Game Outcome model: Logistic/XGBoost) and Phase 5 (cross-validation, SHAP)
   are the official next phases and are **unstarted**. Recent loop activity is
   simulation realism tuning and grid sweeps of sim knobs, not market
   validation.

7. **Open bugs / known incorrect behavior in loop files?** **Yes.**
   `loop/in_progress_context.txt` documents an **unresolved blocker**: the
   player-stat sim can **overshoot the target team score** on rare seeds (repro:
   seed 5 → 114 vs 112 target); reconciliation only spends free throws, and the
   regression test misses the branch. Marked *"should not be treated as fully
   push-safe."* This is in the player-box layer, marginal to win-probability
   output, but signals the sim stack is mid-flight, not production-frozen.

8. **Data completeness (from `modeling_inputs_report.json`):** ✓
   - Rows: 1971 player-seasons across 2022-23/2023-24/2024-25
   - RAPM coverage rate: **100%** (has_rapm_rate=1.0)
   - DARKO coverage rate: **0%** (has_darko_rate=0.0) — DARKO entirely absent

9. **Minute model (from `player_eval_step2_minute_model_validation.json`):** ✓
   - Holdout MAE (2024-25): **2.36 MPG** (R²=0.884)
   - CV fold MAEs: 2.29 / 2.36 / 2.92 MPG across 3 folds
   - MPG is the best-carried player feature year-over-year (r≈0.83–0.85)

---

## Section 8: What Still Needs to Be Built for Prediction-Market Use

Ordered by priority. Concrete, not hand-wavy.

1. **Game-level walk-forward harness.** Modify the forecast pipeline (or add a
   mode) so that season-N→N+1 *projected* team ratings are scored against the
   **real N+1 schedule and outcomes**, producing genuine out-of-sample Brier,
   log-loss, and the 10-bin calibration table. Until this exists, Sleeve C has
   **no measurable edge**.
2. **Kalshi/closing-line ingestion + comparison.** For 2023-24 and 2024-25,
   pull historical moneyline closing prices, convert to implied probabilities,
   and compute the model's **closing-line value (CLV)** and Brier-skill score
   *relative to the market* on every game. Edge = beating the vig-adjusted
   closing line, not beating 0.25.
3. **Recalibrate HCA and σ_league out-of-sample.** Re-fit HCA (currently flat
   2.0; data says ≈1.5) and σ_league on the walk-forward set; the >90% bin
   overconfidence must be fixed before any favorite-side betting.
4. **Injury/availability layer.** A pre-game availability input (player
   in/out, rest, B2B — already computed in `compute_rest_home_back2back.py`
   but unused by the game model). The 5 worst walk-forward errors are all
   availability shocks.
5. **DARKO ingestion.** Current data has 0% DARKO coverage; incorporating this
   prior (as designed) is a prerequisite for the full BKE pipeline as specified.
6. **Wider data base.** 3 seasons / 2 transitions is too thin. Backfill
   ≥2017-18 to get ≥6 walk-forward transitions before trusting any Sharpe-like
   estimate.
7. **Bet-sizing / Kelly simulation.** Only after 1–4: simulate a paper
   bankroll vs. closing lines with realistic fees to estimate ROI, variance,
   and max drawdown for Sleeve C.

---

## Section 9: Honest Alpha Readiness Assessment

**BKE is NOT ready for real-capital prediction-market betting, and the gap is
fundamental, not cosmetic.** This verdict is unchanged from the prior session
and is now strengthened by direct data verification.

The headline Brier ≈ 0.205–0.227 / accuracy ≈ 64–69% is an **in-sample
retrodiction** — confirmed by reading `src/profile_aggregate/team_feature_aggregation.py`
line 873 this session: in backtest mode the model reads same-season player
profiles. The only leakage-free evidence is season-win projection at **MAE ≈
7.7–8.9 wins, r ≈ 0.62–0.71** (freshly extracted from `forecast_season_results.json`),
which the repo itself notes is **~2–4 wins worse than Vegas**, and that path
produces **no game-level probability or calibration at all** (synthetic
schedule, no real opponents). There is **zero comparison to any market price**
anywhere in the project. DARKO data is 0% present, meaning the full pipeline
as designed has never been tested.

The single empirical gate that must be cleared before any allocation: **a
genuine walk-forward game-level test scored against historical Kalshi/sportsbook
closing lines, showing positive closing-line value and a Brier-skill score > 0
versus the market on out-of-sample games (ideally ≥2 seasons, after
HCA/σ recalibration).** Until that one number exists and is positive, Sleeve
C's model edge is **unproven and should receive no capital beyond a tiny
instrumented paper/research allocation.**

**Recommended allocation on current evidence: $0 live; research-only.**

---

## Section 10: Raw Numbers Dump

All values extracted directly from JSON files this session (2026-05-19).

```
ENVIRONMENT (this session, 2026-05-19):
  data/historical/team_game_logs.parquet     PRESENT
  data/processed/player_eval/team_feature_aggregation.parquet  PRESENT
  data/processed/player_eval/player_impact_profiles.parquet    PRESENT
  reports/  (58 files)  PRESENT
  python3 src/simulation/validate_sim.py  -> SUCCESS (regenerated validation JSON)
  python3 -m pytest -q                    -> 3 passed in 176.27s

SOURCE: src/simulation/simulation_config.py
  SIGMA_LEAGUE = 3.0
  HOME_COURT_ADVANTAGE = 2.0
  SEASON_SIMULATIONS = 10000
  SIMULATION_RANDOM_SEED = 42

SOURCE: src/profile_aggregate/team_feature_aggregation.py line 873
  src_path = PROJECTED_PROFILES_PATH if forecast_mode else PLAYER_PROFILES_PARQUET
  -> BACKTEST mode reads SAME-SEASON actual player data -> in-sample leakage CONFIRMED

SOURCE: reports/simulation_step1_validation.json (aggregate_calibration_error)
  0.0562

GAME-LEVEL BACKTEST (simulation_step1_validation.json — IN-SAMPLE/LEAKY):
  2022-23: n=1230 brier=0.227047 logloss=0.650652 acc=0.6407 marginRMSE=12.8657 marginMAE=10.1076 homeWR_act=0.5805 homeWR_pred=0.5798
  2023-24: n=1230 brier=0.215368 logloss=0.620925 acc=0.6537 marginRMSE=14.1357 marginMAE=11.0597 homeWR_act=0.5431 homeWR_pred=0.5739
  2024-25: n=1225 brier=0.205272 logloss=0.597909 acc=0.6931 marginRMSE=13.8007 marginMAE=10.7488 homeWR_act=0.5445 homeWR_pred=0.5731

SEASON-LEVEL BACKTEST (simulation_step1_validation.json — IN-SAMPLE, 90 team-seasons):
  2022-23: n=30 MAE=5.47 RMSE=6.74 r=0.8305 mean_err=-0.01 max_over=+15.7 max_under=-8.7
  2023-24: n=30 MAE=5.93 RMSE=7.15 r=0.8683 mean_err=0.00 max_over=+12.4 max_under=-18.6
  2024-25: n=30 MAE=3.71 RMSE=4.42 r=0.9480 mean_err=-0.17 max_over=+9.0 max_under=-8.8
  OVERALL: n=90 MAE=5.03 RMSE=6.22 r=0.8872 mean_err=-0.06

TOP-5 IN-SAMPLE ERRORS (simulation_step1_season_results.json):
  CHI 2023-24: proj=20.4 act=39 err=-18.6
  WAS 2022-23: proj=50.7 act=35 err=+15.7
  UTA 2022-23: proj=50.9 act=37 err=+13.9
  SAS 2023-24: proj=34.4 act=22 err=+12.4
  DET 2023-24: proj=25.7 act=14 err=+11.7

FORECAST (WALK-FORWARD, LEAKAGE-FREE — forecast_season_results.json, end_of_season):
  2023-24: MAE=7.71 RMSE=9.74 r=0.7057
  2024-25: MAE=8.89 RMSE=11.25 r=0.6166
  NO game-level Brier/logloss/calibration (synthetic schedule, no real outcomes)
  Repo benchmark note: "Vegas lines typically achieve MAE of 5-6 wins"

TOP-5 WALK-FORWARD ERRORS (forecast_season_results.json, end_of_season):
  NOP 2024-25: proj=44.9 act=21 err=+23.9
  MEM 2023-24: proj=50.3 act=27 err=+23.3
  PHI 2024-25: proj=46.0 act=24 err=+22.0
  HOU 2023-24: proj=20.1 act=41 err=-20.9
  POR 2024-25: proj=17.0 act=36 err=-19.0

CALIBRATION BASELINES (computed from JSON this session):
  Overall model MCE: 0.0562
  Overall naive MCE (predict 0.5 always): 0.1657
  Per season:
    2022-23: model_mce=0.0564 naive_mce=0.1464 brier_model=0.2270 brier_home=0.4195 brier_coin=0.2500 brier_prior=0.2435
    2023-24: model_mce=0.0660 naive_mce=0.1577 brier_model=0.2154 brier_home=0.4569 brier_coin=0.2500 brier_prior=0.2481
    2024-25: model_mce=0.0461 naive_mce=0.1931 brier_model=0.2053 brier_home=0.4555 brier_coin=0.2500 brier_prior=0.2480

BKE PREDICTIVE_RHO (bke_v31 reports):
  v3.1-experimental baseline (60/40): predictive_rho=0.329588
  v3.1-experimental best layer1 (55/45): predictive_rho=0.346628
  v3.1-experimental combined (1+6+3): predictive_rho=0.343605
  v3.1-experiment2-rerun base_profile (layer36 60/40): predictive_rho=0.31523
  v3.1-experiment2-rerun recommended (lambda=0.03): predictive_rho=0.317888 mean_abs_rank_shift=3.250526

BKE BACKTEST CROSS-SEASON RANK CORRELATIONS (bke_v26/v27/v30_backtest.json, train=22-23/23-24, test=24-25, n=263):
  v2.6: portable_talent spearman=0.5526 total_impact spearman=0.5106 within1_tier=0.760
  v2.7: portable_talent spearman=0.5900 total_impact spearman=0.5372 within1_tier=0.768
  v3.0: portable_talent spearman=0.5526 total_impact spearman=0.5106 within1_tier=0.760

PLAYER CARRY (forecast_validation.json, end_of_season):
  2022-23->2023-24: BKE_r=0.4301 MPG_r=0.8452
  2023-24->2024-25: BKE_r=0.4919 MPG_r=0.8340

RAPM VALIDATION (rapm_validation_report.json):
  Total player-seasons: 1971 (across 2022-23/2023-24/2024-25)
  xRAPM benchmark pearson r=0.415 spearman=0.306 n=27 MAE=4.46 (raw RAPM vs xRAPM)
  xRAPM comparison (with BPM prior) pearson r=0.539 spearman=0.353 n=27 MAE=1.13
  Year-over-year stability: 2022-23->23-24 pearson=0.309 spearman=0.306 n=539
                            2023-24->24-25 pearson=0.371 spearman=0.418 n=661

DATA COMPLETENESS (modeling_inputs_report.json):
  rows=1971  seasons=['2022-23','2023-24','2024-25']
  has_rapm_rate=1.0  has_darko_rate=0.0  (DARKO ABSENT)

MINUTE MODEL (player_eval_step2_minute_model_validation.json):
  GBR v2, 70 features, target=MPG
  CV fold MAEs: 2.29 (test 2023-24), 2.36 (test 2024-25), 2.92 (test 2022-23)
  Holdout MAE=2.36 R²=0.884 (test season 2024-25)
  Top feature: pec_defensive_shrinkage_lambda importance=0.476

TEAM FEATURES VALIDATION (player_eval_step3_team_features_validation.json):
  n_team_seasons=60 (2023-24, 2024-25)
  team_net_rating_projected: mean=2.60 std=3.91 min=-7.02 max=12.02
  vol_total: mean=5.99 std=0.04 (nearly constant at 6.0)
  mode=forecast (this report is in forecast mode — calibration={} ablation_stack={})

SIMULATION STEP 2 VALIDATION (simulation_step2_validation.json):
  overall: starter_overlap_rate=0.709 clutch_overlap_rate=0.829 rotation_corr=0.787
  starter_target_met=FALSE (target not met) clutch_target_met=TRUE rotation_target_met=TRUE

BKE COMPRESSION (bke_v28_compression_report.json):
  dimension_model_z->portable_talent_z_adj: std_ratio=1.177 (no compression)
  portable_talent_z_adj->total_impact_z: std_ratio=1.014 (no compression)
  role_utilization_raw_z->role_dependent_impact_z: std_ratio=0.570 (COMPRESSED flag=true)

dBKE v3.0 SHRINKAGE (dbke_v30_defense_shrinkage.json):
  Phase A: alpha=0.65 w=0.7 w_portable=0.55 w_rapm_shrunk=0.45; corr_DBKEraw_DRAPM=0.934; pass=true
  Phase B: sigma_global_raw=0.998 std_DBKE_final=0.855; pass=true
  Phase C: def_driver_share=0.368 dbke_yoy_corr=0.715 specialist_dbke_yoy_corr=0.573

OVERALL PLAN STATUS:
  Phase 4 Model Development = [ ] INCOMPLETE
  Phase 5 Evaluation & Interpretability = [ ] INCOMPLETE
```
