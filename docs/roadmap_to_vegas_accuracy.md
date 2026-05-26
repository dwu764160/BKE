# Roadmap to Vegas-Level Accuracy
**Date:** 2026-05-25  
**Target:** Walk-forward Brier ≤ 0.200 with positive closing-line value (CLV) against Kalshi  
**Current position:** Brier ~0.233 (v4.0-A+C composite, patched, OOS)  
**Vegas benchmark:** Brier ~0.195–0.210  
**Gap:** ~0.025–0.035 Brier

---

## Next Steps (Priority Order)

| # | Step | Severity | Complexity / Model | Expected Brier Gain | Notes |
|---|---|---|---|---|---|
| 1 | Lock C winner + build composite v4.0 | **Required** | Low / Sonnet | ~0.001–0.002 | C winner: γ_match=0.45, γ_lineup=0.35, γ_arch=0.20, brier=0.2344. Build `scripts/build_pts_v40.py`, sweep blend ratio (0.4A+0.6C), run §6.1 gates |
| 2 | Kalshi CLV measurement harness | **Critical** | Medium / Sonnet | N/A (measurement) | Without this, all Brier improvements are unvalidated for alpha. Ingest 2023-24 and 2024-25 Kalshi closing lines as CSV, compute game-by-game CLV and Brier-skill vs market. This is the real test. |
| 3 | Game-day rest/schedule correction layer | **High** | Medium / Sonnet | **−0.010 to −0.015** | Rest days, back-to-backs, 3-in-4s, travel days are fully public and computable from `team_game_logs.parquet`. Each B2B shifts true win prob 5-8 pts. Build a pre-game delta applied to `team_net_rating_projected` before the game model runs. Biggest single ROI improvement. |
| 4 | Replace Gaussian game model with GBDT | **High** | High / Opus | **−0.012 to −0.018** | The Gaussian `norm.cdf(delta_mu / sigma)` is a fixed formula. A LightGBM model trained walk-forward on features (net rating diff, pace diff, rest diff, HCA, season stage, B2B flags, home/away win %, last-10 net rating) learns the actual non-linear relationship between team strength and win prob. Use tree depth ≤ 5, L2 regularization. ~9,800 games across 8 seasons — enough for GBDT, not deep learning. |
| 5 | Season-to-date power rating update | **High** | Medium / Sonnet | **−0.008 to −0.012** | Current model uses only preseason projections. Add a Bayesian updating layer: after each game, adjust team `net_rating_projected` toward actual season-to-date net rating with a shrinkage weight that grows through the season (early October = 5% actual, late March = 60% actual). Kalman filter or exponential decay both work. |
| 6 | Fix DRAPM contamination in Dim 6 | **Medium** | High / Opus | **−0.003 to −0.008** | DRAPM in Dim 6 contradicts the dim-only PTS goal — it carries teammate-quality signal. Previous attempt to replace with matchup data hurt lineup r (−0.011 to −0.040) because Dim 6 was swapped raw. Correct approach: build a proper OOS matchup composite (matchup PPP allowed, rim protection, contested shot rate, assignment difficulty) and blend it with a shrunk DRAPM estimate. Opus should design this freely without the v4.0 Non-Goals constraints. |
| 7 | Per-player uncertainty propagation (v4.1-B) | **Medium** | Medium / Sonnet | **−0.002 to −0.005** | Precision-weight team aggregation: 200-min rookies carry less certainty than 3,000-min veterans. Replace minute-share weighting with `minute_share × precision^γ`. Improves team defense rating accuracy most. `src/modeling/bayesian_hierarchical.py` already has posterior variance hooks. |
| 8 | Real-time injury layer | **Critical for CLV** | Very High / Opus | **−0.012 to −0.020** | This is the hardest piece and where Vegas has the biggest edge. Official NBA injury reports (available on NBA.com 1-2 hours before tip-off) can be scraped. Even a binary "starter out / starter plays" flag per game moves win probability 6-10 pts for a star player. Long-term pipeline goal; design separately from step 3. |
| 9 | Ensemble game model | **Low-Medium** | Medium / Sonnet | **−0.003 to −0.005** | Average win probabilities from 3 diverse game models (Gaussian + GBDT + simple Elo) even when models have similar individual Brier. Ensemble diversity reduces systematic bias by ~0.003. Implement after step 4 has a trained GBDT. |
| 10 | Source orthogonalization (v4.1-D) | **Low** | High / Opus | **−0.001 to −0.003** | Dim 3 (playmaking) and Dim 9 (self-creation) double-count ball-dominance. Residualize secondary inputs against primary per dimension where within-season r > 0.5. Marginal gain; do after higher-ROI items. |

**Rough total if steps 3–7 all land:** Brier from ~0.233 → ~0.195–0.205. That is the Vegas band.

---

## Architectural Suggestions (Existing System)

These are not in the step-by-step roadmap above but are improvements to current architecture, design choices, and pipeline structure that will help meet the Kalshi target.

### PTS Architecture

**A. Separate offense and defense into independent pipelines.**  
Currently `pts_o` and `pts_d` are computed from the same dimension model with a split at the end. Offensive and defensive skill are nearly orthogonal in reality (Bam Adebayo is elite defensively, mediocre offensively). Building two fully separate scoring passes — one optimized to predict offensive lineup production, one for defensive — avoids the tradeoff where a defensive weight hurts offensive calibration and vice versa.

**B. Add within-season decay weighting to the dimension model.**  
Current PTS is a full-season aggregate treated as a constant. Players improve (second-half breakouts), decline (aging vets), or get injured mid-season. Add a recency-weighted version: possessions in the last 30 days get 2× weight; first 30 days get 0.5× weight. Report both stable-season and recent-form PTS. The meta-model can use both as features.

**C. Position-conditional z-scoring is too coarse at 5 buckets.**  
Guard / Guard-Forward / Forward / Forward-Center / Center leaves Nikola Jokic and Robert Williams in the same bucket (both Center). Consider splitting with a continuous position estimate (0=PG, 4=C) and weighting the z-score reference distribution smoothly between adjacent buckets rather than hard-assigning. Prevents the current situation where a 6'11" stretch-4 who rebounds like a C but plays like a F gets penalized in both buckets.

**D. Replace Dim 6 DRAPM with a pure matchup composite, designed correctly this time.**  
The prior attempt (v3.2) swapped raw matchup data in a single step, hurting lineup r. The right architecture: blend DRAPM × `(1 − confidence)` + matchup_composite × `confidence`, where confidence = f(n_matchup_possessions, data_quality). Early seasons (pre-2022) fall back to DRAPM entirely. Later seasons increasingly trust matchup data as it becomes available. This maintains lineup signal continuity while reducing the teammate-quality contamination.

**E. Add a "portability decay" flag per player.**  
Some PTS signals are less portable than they appear. A player's Dim 7 (turnover control) z-score from a low-usage, second-unit role may not transfer to a primary ball-handler role. Add a `usage_transfer_flag` that tags players who changed usage tier ≥1 level (e.g., < 20% usage to > 28%) and applies a larger prior weight (higher shrinkage) on their dimension scores for that season transition.

---

### Game Model

**F. Per-team historical HCA calibration.**  
HCA is currently 2.5 pts/100 for all teams. Denver at altitude is worth 3.5-4.0. Sacramento's crowd noise has a documented effect. Compute per-team rolling HCA from actual home vs. road net ratings over the last 3 seasons and use that instead of the flat constant.

**G. Pace-adjusted game variance.**  
`sigma_game` is a constant ~4.0 pts/100. A 120-pace game has more possessions (and more variance in total margin) than a 100-pace game. Scale sigma proportionally: `sigma_game = base_sigma × sqrt(expected_pace / league_avg_pace)`. Fixes systematic miscalibration on high-pace matchups.

**H. Conference/division rivalry effect.**  
Teams that play each other multiple times per season have less predictive uncertainty — more mutual information. Division rivals have smaller Brier errors in actual Vegas lines (they know each other better). Add a small sigma reduction (~5%) for same-division matchups.

---

### Pipeline Structure

**I. Add season-to-date net rating as a live feature.**  
The biggest single pipeline gap: preseason projections never update. After November (30+ games), actual team net ratings are more predictive than preseason projections. Build a lightweight updater that blends: `projected = α × preseason_projection + (1-α) × actual_ytd_net_rating`, where α decreases from 1.0 (opening day) to 0.3 (March). This can be computed from `team_game_logs.parquet` which you already have.

**J. Player-role stability score.**  
Before applying multi-season smoothing (v4.0-A), flag players whose role changed significantly between seasons (usage, minutes, position bucket). For unstable players, reduce the prior weight even at low τ — their historical score is from a different role. Current smoothing treats a 12-min backup who became a starter equally with a 34-min veteran who had the same role both years.

**K. Decouple validation into three distinct harnesses.**  
Currently Brier is the primary metric for everything. Add:
1. **Lineup harness** (already exists): measures PTS signal quality directly — use for PTS architectural decisions
2. **Game-level Brier** (already exists): measures game model quality — use for game model decisions
3. **CLV harness against Kalshi** (missing): the only metric that measures alpha — use for go/no-go on Sleeve C

Never use the game-level Brier to justify PTS changes (they're weakly linked). Never use the lineup r to justify game model changes (different level of abstraction). The coupling of these has caused decisions like "Improvement A passes because Brier is flat" when the right measure for A is YoY correlation + lineup r.

---

### Evaluation System

**L. Add per-context Brier breakdown.**  
Split the validation Brier by: back-to-back games, conference games, top-10 seed vs. top-10 seed, games within last 5 of season (playoff positioning games), playoff-bound vs. lottery teams. Vegas has tighter calibration in high-stakes games — your model may have systematic errors in specific contexts that aggregate Brier hides.

**M. Add calibration curve reporting.**  
The 10-bin calibration is computed in `validate_forecast.py` but not regularly reported. A calibration curve (predicted win prob on x-axis, actual win rate on y-axis) that deviates from the diagonal tells you exactly where the model is over/under-confident. This should be a standard output every time you run validation.

**N. Track Brier by season separately.**  
Aggregate Brier across all seasons masks year-to-year drift. The per-season Brier from `for-alpha-thesis.md` shows: 2022-23 at 0.235, 2023-24 at 0.215 (in-sample, leaky), 2024-25 at 0.205 (in-sample, leaky). The genuine OOS numbers are higher. Tracking per-season separately will reveal if model quality is trending up or down as the NBA evolves.

---

## Summary Priority Matrix

| Category | Highest ROI item | Est. Brier gain | Timeframe |
|---|---|---|---|
| Game-day context | Rest/B2B correction layer (step 3) | −0.010–0.015 | 1–2 weeks |
| Game model | GBDT replacement (step 4) | −0.012–0.018 | 3–4 weeks |
| Real-time data | Season-to-date update (step 5) | −0.008–0.012 | 1–2 weeks |
| PTS architecture | DRAPM → matchup redesign (step 6) | −0.003–0.008 | 2–3 weeks |
| Evaluation | Kalshi CLV harness (step 2) | N/A (alpha test) | 1 week |
| Long-term | Real-time injury layer (step 8) | −0.012–0.020 | 4–8 weeks |

**The path to Kalshi-level alpha is steps 2 → 3 → 5 → 4, in that order.**  
Steps 1 and 6 are PTS quality work; important but not what closes the remaining Brier gap.
