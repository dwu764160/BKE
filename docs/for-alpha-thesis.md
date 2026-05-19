# BKE Performance Handoff — Sleeve C (Prediction Market Model Edge)

> **Audience:** A Claude Code session in the Robinhood trading-bot repo making a
> capital-allocation decision about Sleeve C. This document is self-contained.
> **Generated:** 2026-05-19. **Source branch:** `personal`.
> **Investigator's honesty note:** This checkout contains **code and reference
> docs only**. `.gitignore` excludes `data/`, `reports/`, `models/`, and
> `aggregate/`. There are **zero `reports/*.json` files present** and **no
> `data/` directory**. `validate_sim.py` was executed and **failed with
> `FileNotFoundError`** (missing `data/processed/player_eval/team_feature_aggregation.parquet`).
> Therefore **not a single performance number below was reproduced or
> independently verified in this session.** Every metric is transcribed from
> dated narrative snapshots committed to the repo's reference/loop docs by
> prior sessions. Treat all numbers as *claimed, unverified, and possibly
> stale*. The one data artifact that does exist locally
> (`aggregate/player_profile_aggregate.parquet`, 1971×933) is a player-feature
> table, not a validation report, and does not let us recompute game-level
> accuracy.

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

| Stage | Complete? | Last Run (claimed) | Output Location | Present in this checkout? |
|---|---|---|---|---|
| Data ingest (PBP, team logs, players) | Yes (per loop) | 2026-03 | `data/historical/` | **No** (gitignored, absent) |
| RAPM / ORAPM / DRAPM modeling | Yes | 2026-03 | `data/processed/player_rapm.parquet` | **No** |
| BKE decomposition v2.8–v3.1 | Yes | 2026-03-01 | `data/processed/bke/`, `reports/bke_v*.json` | **No** |
| Player Eval Core Step 1–3 | Yes | 2026-03-08 | `aggregate/player_profile_aggregate.parquet` | **Partial** — only the aggregate parquet exists locally |
| Simulation Step 1 (game model) | Yes (code present, validated 2026-03-04) | 2026-03-11 | `reports/simulation_step1_*.json` | Code yes / reports **No** |
| Simulation Step 2 (lineup proj.) | Yes | 2026-03-08 | `reports/simulation_step2_*.json` | Code yes / reports **No** |
| Forecast pipeline (walk-forward) | Yes | 2026-03-11 | `reports/forecast_*.json` | Code yes / reports **No** |
| Player-stat sim (box scores) | Yes, with known blocker | 2026-03-11 | `data/processed/simulation/*` | Code yes / data **No** |
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

All figures below come from `reference/simulation/Simulation_Core_Summary.md`,
entry dated **2026-03-04**, describing the contents of the absent
`reports/simulation_step1_validation.json`.

| Season | Games | Brier | Log Loss | Accuracy | Margin RMSE | Home WR act / pred |
|---|---|---|---|---|---|---|
| 2022-23 | 1230 | 0.2281 | 0.6536 | 64.9% | 12.90 | 0.581 / 0.580 |
| 2023-24 | 1230 | 0.2148 | 0.6191 | 65.6% | 14.13 | 0.543 / 0.575 |
| 2024-25 | 1225 | 0.2053 | 0.5975 | 69.2% | 13.81 | 0.544 / 0.573 |

- Margin MAE: not separately documented (only RMSE ≈ 12.9–14.1 points).
- Test games: ~3,685 across 3 seasons (2022-23 → 2024-25).

### Is this a genuine hold-out test? **NO — it is in-sample / look-ahead-contaminated.**

This is the single most important finding for capital allocation, and the
repo's own documentation admits it. The Forecast Pipeline entry in
`Simulation_Core_Summary.md` (2026-03-06) states verbatim that the forecast
pipeline *"eliminates the backtesting leakage inherent in the original
simulation pipeline, **where same-season data was used to predict same-season
outcomes**."*

Mechanism (confirmed by reading `src/simulation/game_model.py` and
`validate_sim.py`):

1. `load_team_params()` reads each team's `mu = team_net_rating_projected` and
   `sigma = vol_total` from `team_feature_aggregation.parquet`.
2. That parquet is built from `player_impact_profiles.parquet` →
   `aggregate/player_profile_aggregate.parquet`, which is derived from the
   **same season's** player performance.
3. `validate_game_level()` then scores those ratings against the **same
   season's** actual game results.

So the model is effectively asked "given how these teams actually performed
this season, who won each game" — a retrodiction, not a forecast. The Brier
≈ 0.21 figure is **not usable** as evidence of market edge.

The genuinely leakage-free path is the **forecast pipeline** (project season N
→ N+1 using only ≤N data). Its accuracy is far weaker (Section 5) and,
critically, **it produces no game-level Brier/log-loss/calibration at all**
because it simulates a *synthetic balanced schedule* with no real opponents or
outcomes. **There is currently no honest game-level win-probability accuracy
measurement anywhere in this project.**

---

## Section 4: Full Calibration Table

**The per-bin calibration table is NOT available in this checkout.** It lives
only in the absent `reports/simulation_step1_validation.json`.
`validate_sim.py` (read in full) computes **10 equal-width bins** (0.0–0.1,
0.1–0.2, … 0.9–1.0) with `predicted_rate`, `actual_rate`, and
`calibration_error = |predicted − actual|` per bin, plus a count-weighted
`aggregate_calibration_error`. Only the aggregated/qualitative summary was
recorded in the reference docs:

| Calibration metric (in-sample backtest) | Value |
|---|---|
| Aggregate calibration error (count-weighted MCE over all bins/seasons) | **0.060** (6.0% mean deviation) |
| Mid-range (40–70% predicted WP) | Well-calibrated (qualitative) |
| Extreme bin (>90% predicted WP) | **Overconfident — ~79% actual** vs >90% predicted |

### Baseline comparison (analytic, computable without the data)

| Predictor | Brier | Accuracy | Notes |
|---|---|---|---|
| Always 0.5 (coin flip) | 0.2500 | ~50% | Definitional |
| Always pick home (p→1 for home) | ≈ 0.42–0.46 | ≈ 0.54–0.58 | = 1 − home_win_rate; home WR ≈ 0.54–0.58 from the table |
| Always predict p = home_win_rate (≈0.55) | ≈ 0.2475 | n/a | Brier ≈ p(1−p) ≈ 0.2475 |
| BKE model (in-sample) | **0.205–0.228** | **65–69%** | Beats all three — **but in-sample only** |
| BKE model (genuine hold-out) | **UNKNOWN — never measured** | UNKNOWN | — |

**Conclusion:** The model beats every naive baseline *on the leaky in-sample
backtest*. On a genuine hold-out, the game-level Brier is unmeasured, so **no
claim of market-beating calibration can be substantiated.** MCE-vs-naive
caveat: a "predict 0.55 every game" model already achieves Brier ≈ 0.2475 and
MCE near the home-rate deviation, so the in-sample improvement to 0.205–0.228
is real but modest in absolute terms, and the calibration edge is concentrated
in the mid-probability range where prediction-market vig is hardest to beat.

---

## Section 5: Season-Level Accuracy

### In-sample backtest (same caveat as Section 3 — leaky)

`Simulation_Core_Summary.md` 2026-03-04, 90 team-seasons:

| Season | MAE (wins) | RMSE | Correlation | Max Over | Max Under |
|---|---|---|---|---|---|
| 2022-23 | 5.57 | 6.80 | 0.821 | +14.1 | −9.1 |
| 2023-24 | 6.06 | 7.05 | 0.867 | +12.3 | −17.2 |
| 2024-25 | 3.76 | 4.34 | 0.951 | +8.1 | −8.4 |
| **Overall** | **5.13** | **6.19** | **0.886** | — | — |

### Genuine walk-forward forecast (leakage-free — the number that matters)

From `Simulation_Core_Summary.md` (2026-03-06/07) and `loop/context_summary.txt`
(2026-03-07 sprint), forecast-style projected wins vs. actual:

| Season | Forecast MAE (wins) | Forecast correlation |
|---|---|---|
| 2023-24 | ≈ 7.5 – 9.1 | 0.64 – 0.69 |
| 2024-25 | ≈ 8.9 – 9.1 | 0.64 – 0.65 |

The repo's own benchmark note: *"Vegas lines typically achieve MAE of 5–6
wins."* So the leakage-free model is **~3–4 wins worse per team than the
Vegas/market baseline it would have to beat** to make money on season-win or
derived game markets.

### 5 worst team-season forecast errors (from `loop/context_summary.txt`, "Remaining Large Errors")

| Team-Season | Error (wins) | Cause (uncontrollable event) |
|---|---|---|
| MEM 2023-24 | +26.8 | Ja Morant 25-game suspension |
| NOP 2024-25 | +21.1 | Zion / Ingram season-ending injuries |
| TOR 2023-24 | +21.0 | Mid-season tank / rebuild |
| PHI 2024-25 | +18.0 | Embiid played only 13 games |
| POR 2024-25 | −18.1 | Young-team breakout after prior tank |

All five are injury/roster shocks the model has no mechanism to anticipate —
directly relevant to prediction markets, where these are exactly the spots a
naive model gets steamrolled by sharper money.

---

## Section 6: BKE Player Metric Stability and Predictive Value

From `reference/BKE/bke_ingest_manifest.json` and
`reference/bke_repo_context_summary_for_ai.md`, sourced from the absent
`reports/bke_v31_experiment2_production_tilt.json` (v3.1-experiment2-rerun,
generated 2026-03-01, `qualified_players = 950`):

| Metric | Value | Meaning |
|---|---|---|
| `base_profile.predictive_rho` | **0.31523** | Spearman/Pearson ρ of the BKE production-proxy vs. the predictive target |
| Recommended λ = 0.03 → `predictive_rho` | **0.317888** | Best λ under heuristic (max ρ s.t. rank-shift ≤ 3.5) |
| `mean_abs_rank_shift_vs_base` | **3.250526** | Avg. player rank movement vs. base profile at recommended λ |
| Production-proxy top weights | orapm 0.22, PTS 0.18, TS% 0.14, AST 0.12 | What the proxy leans on |

**Cross-season player BKE carry stability** (year-over-year r of player BKE,
from forecast carry validation, `loop/context_summary.txt` 2026-03-08):

| Transition | BKE carry r | MPG carry r |
|---|---|---|
| 2022-23 → 2023-24 | 0.43 (0.42 preseason) | 0.78 (0.73) |
| 2023-24 → 2024-25 | 0.49 (0.48 preseason) | 0.79 (0.73) |

**What `predictive_rho` predicts — clarification:** it is the correlation
between the BKE-derived *production proxy* (a box-weighted score:
orapm/PTS/TS%/AST) and the model's predictive target within the v3.1
experiment. It is **not** a measure of "future team wins" and **not** a market
metric. ρ ≈ 0.32 is modest — it indicates BKE rank-orders player production
with moderate signal, with ~0.43–0.49 season-to-season player stability
(typical for impact metrics; not exceptional).

---

## Section 7: Known Gaps and Failure Modes (Phase 6 — explicit Yes/No/Partially)

1. **Compared against Vegas lines or Kalshi-implied prices on specific games?**
   **No.** Zero market comparison exists anywhere in the repo. The only Vegas
   reference is a one-line prose aside ("Vegas ≈ 5–6 win MAE"). No game-by-game,
   no line ingestion, no closing-line-value analysis.

2. **True walk-forward test (train ≤N, test N+1, no N+1 data in BKE ratings)?**
   **Partially.** The *forecast pipeline* is genuinely walk-forward for
   season-win projection (MAE ≈ 9, r ≈ 0.65). But it runs a **synthetic
   schedule with no real opponents/outcomes**, so it yields **no game-level
   win-probability, Brier, log-loss, or calibration**. The only game-level
   metrics that exist are the **in-sample (leaky)** ones.

3. **Known data-completeness issues — injuries, trades, back-to-backs?**
   **Yes, multiple.** (a) No injury/availability forecasting — the 5 worst
   errors are all injury/suspension driven. (b) Mid-season trades handled via
   "stint" splitting but flagged as a known bug for name matching
   (Jokic/Doncic, Herb/Herbert Jones). (c) `compute_rest_home_back2back.py`
   exists but rest/B2B is **not** an input to the game model — the win-prob
   formula uses only team `mu`, team `sigma`, and a flat 2.0 HCA. (d) A
   "phantom NAN team" artifact from traded players is filtered, not fixed.

4. **Known weakest scenario?** **Yes — injury/roster-shock team-seasons**
   (MEM/NOP/PHI: +18 to +27 win errors) and **high-confidence games**: the
   model is **overconfident in the >90% predicted-WP bin (actual ≈ 79%)** —
   precisely the favorites bucket where prediction markets are most efficient
   and mispriced edges are smallest. HCA is also slightly miscalibrated
   (predicted home WR 0.573–0.580 vs actual 0.543–0.581; docs suggest dropping
   HCA 2.0 → ~1.5, not yet done).

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
   player-stat sim and its mirrored frontend can **overshoot the target team
   score** on rare seeds (repro: seed 5 → 114 vs 112 target); reconciliation
   only spends free throws, and the regression test misses the branch. Marked
   *"should not be treated as fully push-safe."* Plus the documented
   name-matching bugs (readme "Known Bugs"). Note: this blocker is in the
   player-box layer, which is *marginal* to the win-probability output, but it
   signals the sim stack is mid-flight, not production-frozen.

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
   but unused by the game model). The 5 worst errors are all availability
   shocks.
5. **Wider data base.** 3 seasons / 2 transitions is too thin. Backfill
   ≥2017-18 to get ≥6 walk-forward transitions before trusting any Sharpe-like
   estimate.
6. **Bet-sizing / Kelly simulation.** Only after 1–4: simulate a paper
   bankroll vs. closing lines with realistic fees to estimate ROI, variance,
   and max drawdown for Sleeve C.

---

## Section 9: Honest Alpha Readiness Assessment

**BKE is NOT ready for real-capital prediction-market betting, and the gap is
fundamental, not cosmetic.** The headline Brier ≈ 0.21 / accuracy ≈ 67% is an
**in-sample retrodiction** — the repo's own docs admit same-season data leaks
into the team ratings used to "predict" that season's games. The only
leakage-free evidence is season-win projection at **MAE ≈ 9 wins, r ≈ 0.65**,
which the repo itself notes is **~3–4 wins worse than Vegas**, and that path
produces **no game-level probability or calibration at all**. There is **zero
comparison to any market price** anywhere in the project. The single empirical
gate that must be cleared before any allocation: **a genuine walk-forward
game-level test scored against historical Kalshi/sportsbook closing lines,
showing positive closing-line value and a Brier-skill score > 0 versus the
market on out-of-sample games (ideally ≥2 seasons, after HCA/σ recalibration).**
Until that one number exists and is positive, Sleeve C's model edge is
**unproven and should receive no capital beyond a tiny instrumented
paper/research allocation.** Recommended allocation on current evidence:
**$0 live; research-only.**

---

## Section 10: Raw Numbers Dump

All values verbatim from repo docs (no `reports/` JSON exists to cross-check).

```
SOURCE: src/simulation/simulation_config.py (READ DIRECTLY THIS SESSION)
  SIGMA_LEAGUE = 3.0
  HOME_COURT_ADVANTAGE = 2.0
  SEASON_SIMULATIONS = 10000
  SIMULATION_RANDOM_SEED = 42
  TEAM_FEATURES_PATH = data/processed/player_eval/team_feature_aggregation.parquet

SOURCE: src/simulation/game_model.py (READ DIRECTLY THIS SESSION)
  win_prob_home = Phi( ((mu_home + HCA) - mu_away) / sqrt(sigma_home^2 + sigma_away^2 + sigma_league^2) )
  mu  = team_net_rating_projected   (from team_feature_aggregation.parquet)
  sigma = vol_total                 (from team_feature_aggregation.parquet)
  Backtest team ratings derive from SAME-SEASON player data -> in-sample.

SESSION CHECKS (run this session, 2026-05-19):
  python3 src/simulation/validate_sim.py  -> FileNotFoundError: data/processed/player_eval/team_feature_aggregation.parquet
  python3 -m pytest -q                    -> 3 passed in 1.36s (tests/test_simulation_core.py; data-free regression checks only)
  app/{simulation,player_eval,player_bke}_viewer.py --help -> all FileNotFoundError (HTML generators, no CLI, need absent data)
  aggregate/player_profile_aggregate.parquet -> shape (1971, 933), seasons ['2022-23','2023-24','2024-25'] (ONLY data artifact present)

GAME-LEVEL BACKTEST (Simulation_Core_Summary.md 2026-03-04 — IN-SAMPLE/LEAKY):
  2022-23: games=1230 brier=0.2281 logloss=0.6536 acc=0.649 marginRMSE=12.90 homeWR act=0.581 pred=0.580
  2023-24: games=1230 brier=0.2148 logloss=0.6191 acc=0.656 marginRMSE=14.13 homeWR act=0.543 pred=0.575
  2024-25: games=1225 brier=0.2053 logloss=0.5975 acc=0.692 marginRMSE=13.81 homeWR act=0.544 pred=0.573
  aggregate_calibration_error = 0.060 ; >90% bin actual ~0.79 (overconfident) ; 40-70% well-calibrated

SEASON-LEVEL BACKTEST (IN-SAMPLE, 90 team-seasons):
  2022-23: MAE=5.57 RMSE=6.80 r=0.821 maxOver=+14.1 maxUnder=-9.1
  2023-24: MAE=6.06 RMSE=7.05 r=0.867 maxOver=+12.3 maxUnder=-17.2
  2024-25: MAE=3.76 RMSE=4.34 r=0.951 maxOver=+8.1  maxUnder=-8.4
  OVERALL: MAE=5.13 RMSE=6.19 r=0.886 meanErr=-0.06

FORECAST (WALK-FORWARD, LEAKAGE-FREE; Simulation_Core_Summary.md 2026-03-06/07, loop 2026-03-07):
  2023-24 margin: MAE ~8.20 (improved to 7.52 in later sprint) r ~0.67-0.69
  2024-25 margin: MAE ~9.02 (~9.06) r ~0.64-0.65
  2024-25 PPP   : MAE ~8.27 r ~0.689
  NO game-level Brier/logloss/calibration (synthetic schedule, no real outcomes)
  Repo benchmark note: "Vegas lines typically achieve MAE of 5-6 wins"

PLAYER CARRY (forecast validation, loop 2026-03-08):
  end_of_season  2022-23->23-24: BKE r=0.4301 MPG r=0.7833
  end_of_season  2023-24->24-25: BKE r=0.4919 MPG r=0.7932
  preseason_snap 2022-23->23-24: BKE r=0.4211 MPG r=0.7342
  preseason_snap 2023-24->24-25: BKE r=0.4765 MPG r=0.7301

BKE PREDICTIVE (bke_ingest_manifest.json / bke_v31_experiment2, gen 2026-03-01):
  qualified_players=950 base_predictive_rho=0.31523
  recommended lambda=0.03 predictive_rho=0.317888 mean_abs_rank_shift_vs_base=3.250526
  production_proxy weights: orapm=0.22 PTS=0.18 TS_PCT=0.14 AST=0.12

WORST FORECAST ERRORS (loop/context_summary.txt):
  MEM 2023-24 +26.8 (Morant suspension); NOP 2024-25 +21.1 (Zion/Ingram inj);
  TOR 2023-24 +21.0 (tank); PHI 2024-25 +18.0 (Embiid 13 GP); POR 2024-25 -18.1 (breakout)

KNOWN BLOCKER (loop/in_progress_context.txt 2026-03-11):
  player_stats_sim score-overshoot on rare seeds (seed 5: 114 vs 112 target);
  reconciliation only spends FTs; regression test misses branch; "not push-safe".

OVERALL PLAN STATUS (overall_plan.DO_NOT_CHANGE.txt):
  Phase 4 Model Development = [ ] INCOMPLETE
  Phase 5 Evaluation & Interpretability = [ ] INCOMPLETE
```
