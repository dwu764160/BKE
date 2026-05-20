# BKE Model Weaknesses — Full Diagnostic Report

> **Generated:** 2026-05-19. **Author:** Claude Code analysis with live data verification and external research.
> **Purpose:** Honest, detailed catalog of architectural and empirical weaknesses, organized by scope —
> large-scale (player/team/season prediction) and small-scale (game-by-game prediction market utility).

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Part A — Large-Scale Weaknesses](#2-part-a--large-scale-weaknesses-playerteamseason-prediction)
   - A1: RAPM Foundation
   - A2: BKE Composite Score
   - A3: Data Volume / Training History
   - A4: Minute Model
   - A5: Season-Win Forecast Performance
   - A6: No In-Season Adaptation
3. [Part B — Small-Scale Weaknesses](#3-part-b--small-scale-weaknesses-game-by-game-prediction-market-utility)
   - B1: Fundamental Evidence Gap
   - B2: Gaussian Game Model Architecture
   - B3: Overconfidence in Extreme Predictions
   - B4: Missing Game-Level Features
   - B5: Market Efficiency Ceiling
   - B6: Vig and Transaction Costs
   - B7: No Live / In-Game Capability
4. [Cross-Cutting Structural Problems](#4-cross-cutting-structural-problems)
5. [What Each Weakness Would Take to Fix](#5-what-each-weakness-would-take-to-fix)

---

## 1. Executive Summary

BKE has a well-designed pipeline and a principled modeling philosophy. The core failure is that
**the hardest parts of the pipeline remain unbuilt and the most important number — out-of-sample
game-level predictive accuracy vs. a market price — has never been measured.** The model has
also never been stress-tested against the information that real markets use (confirmed lineups,
injury reports, rest data), so it is structurally blind to the inputs that drive 80%+ of
short-run game outcome variance.

For **large-scale prediction** (season wins, player rankings), the model is hampered by a tiny
data history (3 seasons / 2 walk-forward transitions), an unstable player-impact foundation
(RAPM year-over-year r ≈ 0.31–0.37; BKE carry r ≈ 0.43–0.49), and a completely absent DARKO
integration (0% coverage), meaning the full designed pipeline has never been tested. The
leakage-free season forecast is already 2–4 wins worse than Vegas per team.

For **game-level prediction market utility**, the problems compound: every Brier/accuracy
metric that exists is in-sample retrodiction; the game model ignores rest, B2B, and injury
availability; the Gaussian architecture is overconfident on heavy favorites (the most
efficiently priced market segment); and there is literally zero comparison to any market price
in the entire project. Phases 4 and 5 of the official development plan — a proper game outcome
model and cross-validation harness — are unchecked and unstarted.

---

## 2. Part A — Large-Scale Weaknesses (Player/Team/Season Prediction)

### A1: RAPM Foundation Weaknesses

**The player-impact metric that anchors everything is built on a statistically fragile base.**

RAPM (Regularized Adjusted Plus-Minus) uses ridge regression over lineup stints. The
fundamental problems are well-documented in the academic literature:

**Collinearity.** When two players consistently share or avoid the court, ridge regression
cannot separate their individual contributions. The regularization term (λ) shrinks both
coefficients toward zero rather than correctly attributing the shared effect. In practice, this
means star players on top-heavy rosters (e.g., a team where the franchise player accounts for
60%+ of minutes) and backup players who never see the same lineup as the stars are both
systematically mis-rated. BKE uses stint-level RAPM without documenting how it handles lineup
collinearity, and the `modeling_inputs_report.json` shows it runs on only 1971 player-seasons
across 3 seasons — too thin to resolve collinearity via volume alone.

**Year-over-year instability.** The `rapm_validation_report.json` shows raw year-over-year
RAPM Pearson r = 0.31–0.37 (n=539–661 player-seasons). This is low — barely better than
chance for individual players. The ridge regression penalty stabilizes estimates within a season
but cannot fix the underlying noisiness of the signal. Published research confirms: RAPM MSE
of ~1.4 when predicting next-year RAPM from current-year RAPM, with the difference between
the 30th and 50th player only ≈ 0.4 — smaller than the prediction error.

**DARKO absent (0% coverage).** BKE's design calls for DARKO as a Bayesian prior that
stabilizes RAPM estimates by weighting toward a player's translated college stats and age curve.
The `modeling_inputs_report.json` shows `has_darko_rate = 0.0` — DARKO is entirely missing
from the current data. The model is running on RAPM+box only. The full designed pipeline has
literally never been tested with the priors it was built to use. This is not a minor gap:
DARKO's age-curve component is specifically designed to address RAPM's year-over-year
instability for young players (development) and veteran players (decline curves) — precisely
the player-types where BKE's walk-forward errors are largest.

**RAPM is descriptive, not predictive by design.** RAPM tells you the net points per 100
possessions when player X was on the court in past games. It is a *historical* decomposition,
not a forecast. Using it as a forecast input requires assuming that past lineup interactions
predict future lineup interactions — an assumption that holds reasonably for veteran players
in stable roles but breaks badly for:
- Young players mid-development trajectory
- Players changing teams or roles
- Players recovering from injury
- Veterans entering decline

These are exactly the categories responsible for BKE's worst walk-forward errors.

**The xRAPM benchmark is weak.** The `rapm_validation_report.json` shows raw RAPM vs xRAPM
(a public external benchmark): Pearson r = 0.415, Spearman r = 0.306, n=27, MAE = 4.46.
The n=27 is extremely small (27 players with enough minutes to compare). With BPM prior: r =
0.539. These are moderate correlations, not strong external validation. The comparison dataset
is also small enough that a few outlier players dominate the correlation.

---

### A2: BKE Composite Score Validity

**The composite BKE score's predictive validity is modest and measured only internally.**

The key metric, `predictive_rho`, is the Spearman correlation between BKE score and a
production proxy (ORAPM=0.22, PTS=0.18, TS%=0.14, AST=0.12, etc.) **within the same dataset**.
At ρ ≈ 0.315–0.347, this is moderate internal consistency — the BKE composite aligns with
observed production reasonably well. But:

- It is **not** a game-outcome metric
- It is **not** a next-year predictive metric (same-dataset correlation)
- It is **not** a market-price metric
- The target is a weighted combo of box stats and ORAPM — inputs that partly feed BKE itself,
  so some circularity is unavoidable

The cross-season player rank stability (backtest: Spearman ≈ 0.55–0.59, within-1-tier ≈ 76%)
is decent but not exceptional for ranking the top 50 players. For the bottom half of the player
pool — backups, role players — the metric is noisier and less validated.

**Role-dependent impact is compressed.** The `bke_v28_compression_report.json` flags that
`role_utilization_raw_z -> role_dependent_impact_z` has std_ratio = 0.570 (compression flag =
true). This means the role-adjustment layer is dramatically narrowing the variance of the
role-based impact signal — compressing genuine spread into a flatter distribution. This
suggests BKE may be overly smoothing genuine differences between high-usage and low-usage
players in a way that masks real talent gaps.

**Version tweaks show diminishing returns.** From v3.1 baseline (ρ = 0.3296) to best single
layer (ρ = 0.3466) to combined layers (ρ = 0.3436), the improvement is ~5% relative. The
second experiment pass actually regresses (ρ = 0.3152–0.3179). This is a strong signal that
the current feature set and architecture are near their ceiling: you're squeezing the last 1%
out of a fixed signal base rather than adding genuinely new information.

**BKE carry r ≈ 0.43–0.49 vs. MPG carry r ≈ 0.83–0.85.** The single most stable player
feature year-over-year is how many minutes a player plays — not their impact per minute. BKE's
year-over-year stability is only half as strong as raw usage. A simpler model that projected
teams based on prior-year minutes × prior-year efficiency (and regressed both toward the mean)
might achieve comparable or better season-forecast results with far less complexity. This should
be tested as a baseline but has not been.

---

### A3: Data Volume and Training History

**Three seasons is not enough data to trust any statistical claim from this model.**

With 3 seasons (2022-23, 2023-24, 2024-25) and only 2 walk-forward transitions, BKE's
train/test regime is:
- **Training set:** 1–2 seasons of player history
- **Test set:** 1 season of outcomes
- **Transitions tested:** 2

Two data points cannot establish a stable estimate of out-of-sample accuracy. The r values
(0.706 in 2023-24 → 0.617 in 2024-25) already show significant drop-off across just one
additional test season. A Sharpe-ratio-style ROI estimate would require ≥ 6 transitions
(≥ 7 seasons of data) before the variance of the estimate shrinks to a level where you could
distinguish real edge from luck at 2σ confidence.

**Only 3 seasons also means the model cannot distinguish:**
- Persistent structural features (team culture, coaching system, arena elevation) from noise
- Post-COVID normalization effects (HCA changed significantly 2021-22 through 2023-24) from
  a true long-run HCA estimate
- Young-team development curves (Houston 2023-24: +20.9 win error; Portland 2024-25: -19.0
  win error) from actual roster strength

**70 features in the minute model on 3 seasons.** The Gradient Boosted Regression minute
model uses 70 features to predict MPG. With ~600 player-seasons per season (about 1800 total
train/test rows), the features-to-samples ratio is approximately 1:26. For GBR this is
manageable due to regularization, but the top feature being `pec_defensive_shrinkage_lambda`
(importance = 0.476) — a regularization hyperparameter, not a basketball observation — is a
red flag indicating the model may be fitting to its own regularization artifact rather than
genuine basketball signal.

---

### A4: Minute Model Weaknesses

**The playing-time projection layer accumulates error that propagates through the entire stack.**

The minute model (GBR v2) achieves holdout MAE = 2.36 MPG and R² = 0.884. This looks
impressive in isolation, but consider the propagation:

- A 30-game starter playing 32 MPG might be predicted at 34.36 MPG (+2.36) — that's a 7.4%
  overestimate of his contribution to team net rating
- Across 10+ rotation players, these errors compound and partially cancel, but in the worst
  cases (a team where one player's role is genuinely uncertain), the team-level projected
  net rating can be significantly off

**More critically:** the minute model cannot anticipate:
- Load management decisions (star players sitting ~15-25 games in modern NBA seasons)
- Injury-related minute restriction (a player returning from surgery on a limited plan)
- Coaching changes that shift rotations
- Rookie emergence (a high-draft pick who breaks into the rotation mid-season)

All of these are systematic rather than random errors — they bias the model in predictable ways
(overestimate established stars, underestimate emerging players) that a better model could in
principle correct for.

---

### A5: Season-Win Forecast Performance vs. Benchmarks

**The genuine out-of-sample (leakage-free) season win forecast is materially worse than the market.**

| Metric | BKE (leakage-free) | Vegas benchmark | Gap |
|---|---|---|---|
| MAE (wins/team) | 7.71–8.89 | ~5–6 | 2–4 wins worse |
| Correlation r | 0.617–0.706 | ~0.80–0.85 (implied) | Meaningful |
| Seasons tested | 2 transitions | Many decades | Very thin |

The 5 worst errors all have the same root cause: **availability shocks that BKE cannot
anticipate or detect.**

| Team-Season | Error | Root Cause |
|---|---|---|
| NOP 2024-25 | +23.9 wins | Zion/Ingram injuries, roster collapse |
| MEM 2023-24 | +23.3 wins | Ja Morant suspension |
| PHI 2024-25 | +22.0 wins | Embiid played only 13 games |
| HOU 2023-24 | -20.9 wins | Young-team breakout |
| POR 2024-25 | -19.0 wins | Unexpected development |

The top 3 errors alone represent teams where a star missed >60% of games due to events
knowable within the first 2-4 weeks of the season. A human analyst following sports news
would have corrected these projections immediately. BKE has no mechanism to do so.

---

### A6: No In-Season Adaptation or Dynamic Updating

**BKE is a static model: it makes a pre-season forecast and cannot update.**

Modern NBA prediction systems (FiveThirtyEight RAPTOR, ESPN BPI, number-fire) continuously
re-estimate team quality using recent game results, weighting recent games more heavily and
incorporating:
- Rolling win/loss trends
- Recent injury news
- Load management announcements
- Trade deadline acquisitions

BKE's architecture has no such updating mechanism. The `forecast_mode=True` path uses
prior-season player profiles and projects the entire season as if the pre-season roster is
frozen. This is structurally correct for a pre-season analysis but makes the model
non-competitive for in-season prediction, which is when prediction market contracts are
actually live and tradeable.

---

## 3. Part B — Small-Scale Weaknesses (Game-by-Game Prediction Market Utility)

### B1: The Fundamental Evidence Gap

**The most important number — out-of-sample game-level accuracy vs. a market price — has never been measured.**

This is not a modeling flaw; it is a missing experiment. Everything in Part A above feeds into this:

- The Brier scores (0.205–0.227) are **confirmed in-sample retrodictions** via code inspection of
  `src/profile_aggregate/team_feature_aggregation.py` line 873. In backtest mode, the model reads
  same-season actual player data — it is essentially asked "given how this team actually played
  this year, who won each game." The answer should be good. It is not a forecast.
- The forecast pipeline produces game-level probabilities only for a **synthetic schedule with
  no real opponents or outcomes** — so it cannot produce a Brier score either.
- **Zero game-by-game comparison to any Kalshi, DraftKings, or FanDuel line exists anywhere
  in the codebase.**

The consequence: you do not know whether BKE's game-level probability estimates are better than
or worse than simply copying the market price. Without this number, there is no basis for
betting a single dollar.

---

### B2: Gaussian Game Model Architecture Limitations

**The Gaussian margin model is principled but makes assumptions that break in edge cases.**

BKE models game outcomes as: `margin ~ N(μ_home - μ_away + HCA, σ_league²)` where
σ_league = 3.0 and HCA = 2.0 (flat, universal constants in `simulation_config.py`).

**σ_league = 3.0 is a fixed constant.** This is the spread of team net ratings (points/100
possessions), not the game-to-game variance in outcomes. But real game outcomes have enormous
variance — RMSE of 12.87–14.14 on margins — driven by:
- Random hot/cold shooting nights (3-point shooting variance alone is ≈ ±4–5 pts/game)
- Foul trouble removing key players mid-game
- Overtime (adds ~5 possessions, fundamentally changing margin distribution)
- Blowout dynamics (trailing team increases pace, starters rest)

A flat σ_league cannot capture the game-to-game variance *conditional on game state*. Close
games and blowouts have very different distributions of final margins, and a fixed σ will
systematically overstate the probability of extremely lopsided outcomes.

**HCA = 2.0 is flat across all contexts.** Real home-court advantage varies substantially by:
- Team and venue (altitude in Denver = +4 pts; neutral-level venues = +0.5)
- Travel burden of the away team (transcontinental flight previous night vs. local game)
- Back-to-back status (reduces HCA or even reverses it)
- Crowd noise and referee home-crowd effects (varies by market)

Published research (Wharton, 2012; Bayesian HCA estimation, 2022) shows HCA in the NBA has
been declining and now averages ≈ 1.5–2.0 points — BKE's 2.0 is at the upper end. The
model's predicted home win rate (0.573–0.580) consistently exceeds actual (0.543–0.581),
confirming a slight HCA overcalibration.

**No fat tails.** NBA game margin distributions have slightly heavier tails than Gaussian —
blowouts (30+ point margins) happen at higher rates than a normal distribution would predict.
The model's overconfidence in extreme bins (Section 3 in the alpha thesis) is partly a
consequence of using a light-tailed distribution.

**Gaussian win probability is structurally symmetric.** For games between close teams (μ_diff
≈ 0), the model outputs P(win) ≈ 0.5 ± small δ. This is correct on average but ignores
asymmetric factors like:
- The better team having injury-impacted depth more susceptible to foul trouble
- Historical head-to-head matchup-specific dynamics (coaching adjustments, scheme matchups)

---

### B3: Overconfidence in Extreme Predictions — the Worst Possible Place to Be Wrong

**BKE is overconfident exactly where the market is most efficient and edges are smallest.**

From the calibration tables:

| Predicted WP | 2022-23 actual | 2023-24 actual | 2024-25 actual | Model overconfidence |
|---|---|---|---|---|
| 0.9–1.0 | 78.1% | 88.1% | 87.4% | 5–15% probability gap |
| 0.7–0.8 | 63.0% | 64.8% | 69.5% | 5–10% probability gap |

The 0.9–1.0 bin is where BKE is most wrong. This is also where:
- **Prediction markets are most liquid and most efficiently priced** — professional arbitrageurs
  focus on heavy favorites because large volumes can be deployed
- **Closing-line value (CLV) is hardest to find** — the market knows a team is a huge favorite
  and prices it correctly; finding a genuine edge requires information the market doesn't have
- **Blowout dynamics inflate uncertainty** — when BKE rates a game at 93% home win, the trail
  team often makes lineup decisions (sitting starters, fouling) that change margin distribution

Concretely: if BKE says 93% and Kalshi says 88%, the market is almost certainly right. The
5-point gap is the model's bias, not an edge. Betting BKE's heavy favorites against the market
is the worst possible strategy with this model.

---

### B4: Critical Missing Game-Level Features

**The game model is missing features that professional handicappers treat as primary inputs.**

**Rest and back-to-back (B2B).**
`compute_rest_home_back2back.py` exists in the codebase and computes exactly the rest metrics
needed. The `simulation_step2_validation.json` and the game model configuration confirm it is
**not connected to the win-probability formula.** Rest is one of the most empirically reliable
predictors of game-level performance differential. Teams on 0 days rest (back-to-back second
game) shoot more poorly, commit more turnovers, and have higher injury risk. A 2–3 point
performance differential between a rested team and a B2B team is well-documented and is
incorporated into every professional sportsbook's opening line.

**Injury / lineup availability.**
The model uses full-season average player profiles. At game time, prediction markets trade on
*confirmed lineups* — information available ~60–90 minutes before tip-off (NBA injury reports
are required 30 minutes before tipoff). A top-3 player missing a game shifts the point spread
by 4–8 points and win probability by 15–25 percentage points. BKE cannot use this information
without an injury-feed integration, and no such integration exists.

**Pace and tempo differential.**
Some teams dramatically change the pace of play based on opponent. Fast-pace vs. slow-pace
mismatches create game-to-game variance that is not captured by season-average net ratings.
The BKE pipeline does not include pace adjustment in the game model.

**Referee assignment.**
Referee crew has documented effects on foul rate, pace, and total points. Some crews call
significantly more fouls than others, affecting whether stars stay on the floor and whether
games become high-scoring. This is information professional handicappers use that BKE
completely ignores.

**Motivational / schedule context.**
Teams fighting for playoff seeding, load-managing in meaningless games, or tanking for lottery
odds behave differently from their season-average ratings. The model has no game-context
feature.

---

### B5: The Market Efficiency Ceiling

**NBA closing lines are set by some of the most sophisticated quantitative systems in any
sports betting market. Beating them structurally requires information BKE doesn't have.**

The closing line at Kalshi/DraftKings/FanDuel on an NBA game incorporates:
- Season-long performance metrics (what BKE also has)
- Rolling recent-game performance (last 5–10 games, recency-weighted)
- Confirmed injury/lineup information (confirmed hours before BKE can update)
- Reverse-line movement from sharp bettors (signals that professional models disagree with
  the opening line)
- Public betting percentages (used to shade lines)
- Weather, travel, altitude (venue effects)
- Rest differential (automatically incorporated)

BKE has access to only the first of these. The remaining inputs collectively explain a much
larger share of game-to-game outcome variance than season-average quality.

The efficient-market research on sports betting is clear: closing lines are semi-strong-form
efficient. Models that beat the vig long-term require either (a) a genuinely superior
player-quality model that the market doesn't use, or (b) access to information earlier than
the market (opening-line value, not closing-line value). BKE's approach is (a), and the
evidence that it has superior quality is the in-sample Brier — which is contaminated.

**The closing-line value (CLV) test is the correct benchmark**, not Brier vs. 0.25. CLV
measures whether your probability estimate at bet-time was better than the market's probability
at close. Even with a genuine game-level edge, CLV requires you to act before the market
corrects — within hours of line opening for marquee games. BKE has no mechanism for this:
it generates probabilities pre-season and does not update.

---

### B6: Vig and Transaction Costs

**The arithmetic of prediction market betting makes break-even much harder than it looks.**

Kalshi's NBA game contracts trade with a bid-ask spread. For a standard moneyline contract,
the effective vig is approximately 2–5% of notional on liquid games and higher on less-liquid
contracts (props, parlays). To be profitable net of vig at Kelly-criterion sizing:

- If true edge is 0%: expected ROI = -(vig) ≈ -2 to -5% per bet
- To break even: need a 2–5% systematic edge on *every bet placed*
- To achieve 10% ROI: need ≈ 7–10% systematic edge per bet

Without a measured out-of-sample Brier vs. closing line, BKE's edge on any individual game
is completely unknown. The Kelly criterion on unknown edge = bet zero. The proper sequence is:

1. Establish out-of-sample game-level Brier vs. closing lines on historical data (2021–2025)
2. Compute CLV on those games (did BKE's opening-time probability beat the closing price?)
3. Simulate a paper bankroll with realistic sizing and vig
4. Only after confirming positive CLV and positive paper-bankroll ROI: consider live allocation

BKE is at step 0 of this sequence.

---

### B7: No Live / In-Game Capability

**An entire class of prediction market edges is structurally inaccessible.**

Prediction markets increasingly offer live contracts: first-half result, leading at end of
third quarter, game winner at halftime. These contracts can have significant inefficiencies —
crowds overreact to early scoring runs, and the market is thinner and less sharp for live
props than pregame lines.

BKE has no in-game update model. It cannot:
- Use current score and game time to update win probability
- Incorporate foul trouble information
- Adjust for lineup changes made mid-game
- Price live contracts at all

This is an architectural limitation, not a missing parameter. Adding live capability would
require building a completely separate real-time inference layer.

---

## 4. Cross-Cutting Structural Problems

These weaknesses apply to both large-scale and small-scale prediction and represent
architectural choices that would need fundamental rethinking:

### C1: Phases 4 and 5 Are Unstarted

The `overall_plan.DO_NOT_CHANGE.txt` shows Phases 4 (Model Development: logistic/XGBoost
game outcome model) and Phase 5 (Evaluation: cross-validation, SHAP analysis) are unchecked.
The current game model is a principled heuristic (Gaussian margin from net ratings) but is
not the planned formal statistical model. The simulation work (Streams F/G) is a parallel
effort, not a replacement for Phases 4–5.

Published research (MDPI 2025) shows that uncertainty-aware ML frameworks (RNNs with Monte
Carlo dropout, trained with strict chronological partitioning) consistently outperform static
Gaussian margin models on NBA prediction. BKE's game model is where 2018-era research was —
the field has moved significantly.

### C2: No SHAP / Feature Attribution

With no SHAP analysis (Phase 5 is unchecked), there is no way to know:
- Which features are actually driving predictions (vs. noise)
- Whether the model is using proxies for information it shouldn't have
- Which games the model is most/least confident on and why

This makes debugging prediction errors purely ad-hoc rather than systematic.

### C3: Phantom NAN Team / Trade Handling Bug

The pipeline filters a "phantom NAN team" artifact from traded players rather than fixing
the root cause. This is documented but unresolved. Mid-season trade handling via stint
splitting has a known name-matching bug. These are data quality issues that could introduce
systematic biases for teams that are active in trades — exactly the teams with the most
uncertain team quality assessments.

### C4: The Simulation Blocker

`loop/in_progress_context.txt` documents an unresolved bug in the player-stat simulation:
seed 5 causes the simulated team score to overshoot the target (114 vs 112 target) because
the reconciliation only spends free throws. This is noted as "should not be treated as
fully push-safe." While this affects the box-score simulation layer rather than win-probability
output directly, it signals that the simulation stack is in an in-progress state.

---

## 5. What Each Weakness Would Take to Fix

| Weakness | Fix Category | Estimated Effort | Priority for PM Use |
|---|---|---|---|
| In-sample leakage in Brier metrics | Run walk-forward game-level harness with real schedule | Medium (2–3 weeks) | **Critical** |
| No market comparison | Ingest historical closing lines, compute CLV | Medium | **Critical** |
| DARKO absent (0% coverage) | Obtain DARKO data feed, integrate | Low (pipe already exists) | High |
| HCA/σ miscalibration | Re-fit on walk-forward set | Low (1 day) | High |
| Rest/B2B not in game model | Wire `compute_rest_home_back2back.py` → game model | Low (1–2 days) | High |
| Injury availability layer | Pre-game lineup feed + win-prob adjustment | High (weeks) | High |
| Only 3 seasons of data | Backfill to 2017-18 (6+ transitions) | High (weeks) | Medium |
| Overconfidence in 0.9–1.0 bin | Calibration recalibration (Platt scaling, isotonic) | Low | Medium |
| No Phase 4 game model | Build logistic/XGB on walk-forward game features | High | Medium |
| No Phase 5 SHAP analysis | Build after Phase 4 | Medium | Low |
| No live in-game capability | Separate architecture | Very High | Low |
| Trade handling bugs | Root-cause fix to name matching | Medium | Low |

---

## Sources

External research informing this analysis:

- [Lineup Regularized Adjusted Plus-Minus (L-RAPM): Basketball Lineup Ratings with Informed Priors](https://arxiv.org/html/2601.15000v1) — arxiv 2026, RAPM collinearity and stabilization
- [Regularized Adjusted Plus-Minus Part III — Squared Statistics](https://squared2020.com/2018/12/24/regularized-adjusted-plus-minus-part-iii-what-had-really-happened-was/) — RAPM limitations and collinearity
- [A Review of Adjusted Plus/Minus and Stabilization](http://godismyjudgeok.com/DStats/2011/nba-stats/a-review-of-adjusted-plusminus-and-stabilization/) — variance and interpretability issues
- [Uncertainty-Aware Machine Learning for NBA Forecasting](https://www.mdpi.com/2078-2489/17/1/56) — MDPI 2025, RNN+MC dropout, strict chronological partitioning
- [Predicting the Outcome of NBA Games — Bryant University](https://digitalcommons.bryant.edu/cgi/viewcontent.cgi?article=1000&context=honors_data_science) — accuracy benchmarks and Gaussian Naïve Bayes
- [NBA Model Math — Yale Undergraduate Sports Analytics](https://sports.sites.yale.edu/nba-model-math) — HCA and Gaussian margin modeling
- [The Role of Rest in NBA Home Court Advantage — Wharton](https://faculty.wharton.upenn.edu/wp-content/uploads/2012/04/Nba.pdf) — rest effects on HCA
- [Closing Line Value — Bet Analytix](https://www.bet-analytix.com/academy/closing-odds-ultimate-indicator) — CLV as the correct edge metric
- [CLV Analysis — SportBot AI](https://www.sportbotai.com/blog/clv-analysis-what-closing-line-value-reveals-about-your-betting-strategy) — CLV and market efficiency
- [Walk-Forward vs. Backtesting — Surmount](https://surmount.ai/blogs/walk-forward-analysis-vs-backtesting-pros-cons-best-practices) — walk-forward analysis rationale
- [Mastering Cross-Validation for Betting Models — oddsonnet.com](https://oddsonnet.com/news/mastering-cross-validation-techniques-for-betting-models-avoid-overfitting-and-boost-profits) — overfitting and leakage pitfalls
- [What Are Sports Prediction Markets — Sportico 2026](https://www.sportico.com/business/sports-betting/2026/prediction-markets-sports-kalshi-robinhood-polymarket-1234858418/) — Kalshi/Polymarket NBA market structure
- [NBA Player Value Models: Calculating RAPM — John Chen / Medium](https://medium.com/@johnchenmbb/calculating-rapm-steps-1-and-2-of-my-summer-plan-1a78e1476b1f) — RAPM year-over-year stability
- [2024-25 NBA Forecast and Estimated RAPTOR Ratings — Neil Paine](https://neilpaine.substack.com/p/2024-25-nba-forecast) — win total forecast accuracy benchmarks
- [A Systematic Review of Machine Learning in Sports Betting — arxiv 2024](https://arxiv.org/html/2410.21484v1) — comprehensive review of ML model pitfalls
