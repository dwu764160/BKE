# PTS → Game-Model Wiring — Phase 1/2 Result

**Date:** 2026-05-30
**Code:** `scripts/build_pts_team_ratings.py`, `scripts/validate_pts_team_ratings.py`
**Artifact:** `data/processed/forecast/team_pts_ratings.parquet` (270 team-seasons, 9 seasons)
**Follows:** `docs/findings/game_model_comparison_2026-05-30.md`

## TL;DR

We aggregated PTS v4.0 into team ratings for the first time and tested whether it
helps predict games. Verdict, performance-first:

> **PTS is an excellent *same-season* team-strength descriptor (r = 0.84 vs actual
> net rating), but it does NOT beat Elo at the game level — because the only
> leakage-free way to wire it today is a one-season lag, and the stale signal adds
> nothing.** The bottleneck is **data (current-season roster + availability)**, not
> the metric.

Do **not** promote lagged PTS into the game model. The metric is validated; its
value is locked behind a YTD/current-roster computation we cannot run yet.

## Variants built (the meaningful 2×2 cells)

- **A. PTS-only additive** — `team_net = Σ minute_share · (pts_o_v40 + pts_d_v40)`
  (the `off_talent_base` form with PTS swapped in). Face-valid: 2023-24 top-4 =
  BOS (champion), MIN, OKC, LAC; bottom = CHA, UTA, WAS.
- **B. PTS + archetype-fit** — A plus a fit-only composite (spacing / rim
  protection / creator-redundancy). Fit terms emitted raw so the GBDT can weight
  them ("let the model decide").
- *C/D (interaction-aware revived model) were not built separately:* the repo's
  own Lasso fit already found archetype-PAIR interactions negligible
  (`INTERACTION_MATRIX = {}`, `team_feature_aggregation.py:167-196`), so they
  reduce to A + structure ≈ B. Building them would be redundant.

## Gate 1 — team-rating signal (correlation with actual net rating)

| Signal | r vs actual net | Note |
|---|---|---|
| **pts_net, same-season** | **0.840** | metric is a strong contemporaneous descriptor |
| pts_net_fit, same-season | 0.812 | fit term slightly *hurts* |
| prior-year MARGIN (production signal) | **0.542** | the bar to beat as a preseason prior |
| prior-year pts_net (candidate) | 0.514 | **loses** to margin |
| prior-year pts_net+fit | 0.499 | worse |

The split is the whole story: same-season PTS is great (0.84), but **carried
forward one season it is slightly worse than last year's margin** — both are stale
N-1 priors, and realized margin already bakes in health/coaching/luck.

## Gate 2 — incremental game-level value vs Elo (walk-forward, 8,279 OOS games)

Rate each game's teams from season **N-1** PTS (leakage-free), add as GBDT
features alongside Elo; paired bootstrap (2,000 resamples).

| Model | Brier |
|---|---|
| Elo-only | 0.2193 |
| GBDT base (no PTS) | 0.2250 |
| GBDT + PTS_net | 0.2260 |
| GBDT + PTS_net + fit | 0.2256 |

| Comparison | Δ Brier | 95% CI | Verdict |
|---|---|---|---|
| PTS vs GBDT-base | +0.0009 | [−0.0003, +0.0022] | **NOISE** |
| PTS+fit vs GBDT-base | +0.0006 | [−0.0010, +0.0021] | **NOISE** |
| PTS vs Elo | +0.0066 | [+0.0048, +0.0085] | worse |
| PTS+fit vs Elo | +0.0063 | [+0.0044, +0.0083] | worse |

Lagged PTS is statistical noise on top of the GBDT and clearly worse than Elo.

## What this means

1. **The metric is good; the wiring is stale.** Same-season r = 0.84 proves PTS
   captures team strength. The game model can't use it because we only have it at
   a one-season lag (season-aggregated), which Elo and prior-year margin already
   subsume.
2. **The unlock is a YTD/current-roster PTS** — PTS recomputed from games-to-date
   on the *current* roster, updated through the season. That is exactly the signal
   Elo lacks resolution on around roster changes. It cannot be tested with the
   current season-aggregated PTS (that would leak the games being predicted).
3. **Therefore the real dependency is the Phase-3 data track**: current-season
   roster + per-game availability/inactives, plus an incremental PTS computation.
   Until then, PTS at the game level is research-only.

## Recommendation

- Keep `team_pts_ratings.parquet` as a validated **team-quality** artifact (useful
  for the manager-sim product and for preseason power rankings).
- Do **not** wire lagged PTS into the production game model (no measured value).
- Decide whether to invest in the heavy Phase-3 data work (game-day inactives +
  YTD PTS) — that is the only path that could let player-impact beat Elo, and it
  is a multi-step data-engineering effort, not a wiring task.

## Phase 3 (prelim) — current-roster YTD PTS, the real test

`scripts/experiment_ytd_roster_pts.py` builds the signal Phase-1/2 lacked: weight
each player's **prior-season** PTS by their **strictly-prior current-season**
cumulative minutes (`ytd_pts(T,D) = Σ_{g<D} Σ_p min·skill / Σ_{g<D} Σ_p min`). This
captures roster changes (trades, who's actually playing) — leakage-free. Local
player-game minutes exist only for 2022-25, so this is **2 OOS seasons (2023-24,
2024-25), 2,389 games** — preliminary.

Walk-forward logistic stack on Elo:

| Model | Brier |
|---|---|
| raw Elo | 0.2133 |
| A) logit(Elo) | 0.2142 |
| B) logit(Elo + current-roster PTS) | 0.2149 |
| C) logit(Elo + lagged PTS) | 0.2153 |

| Comparison | Δ Brier | 95% CI | Verdict |
|---|---|---|---|
| B current-roster vs Elo | +0.0007 | [−0.0008, +0.0021] | **NOISE** |
| C lagged vs Elo | +0.0011 | [+0.0002, +0.0019] | HURT |

**By season progress** (Δ = Brier(Elo+PTS) − Brier(Elo); negative = PTS helps):

| Window | n | Δ |
|---|---|---|
| early (games 2-10) | 274 | **−0.0004** |
| mid (11-25) | 458 | **−0.0005** |
| late (26-50) | 743 | +0.0015 |
| deep (51+) | 914 | +0.0010 |

**Interpretation:** the *direction* matches the theory exactly — current-roster PTS
helps marginally while Elo is still uninformed (early/mid) and becomes dead weight
once Elo has caught up (late/deep). But the **magnitude is negligible (~0.0005
Brier) and not statistically distinguishable from zero.** Even in its best window,
player-impact gets nowhere near the ~0.01 Brier needed to reach Kalshi.

**Strategic read (mounting evidence):** across lagged-PTS, current-roster PTS, and
every mechanical fix, **nothing in the BKE/PTS pipeline beats Elo at the game level
by a margin that matters.** PTS is excellent for *team quality* (same-season
r=0.84) — ideal for the manager-sim and power rankings — but it does **not** appear
to carry game-level alpha over Elo. The theoretical early-season edge is real in
sign but too small to monetize.

**Data caveat / what would change the verdict:** only 2 OOS seasons are testable
offline (player_game_logs local = 2022-25; stats.nba.com is firewalled from this
environment). A full read needs: (1) `player_game_logs` backfilled to all 9 seasons,
(2) **preseason-roster-projected minutes** for the very-early window (games 1-10,
where YTD minutes are too noisy and a preseason roster prior is the right signal —
needs the roster backfill to 8 seasons), (3) optionally game-day inactives to test
true injury-availability pricing. All three are network-gated fetches to run in an
environment that can reach stats.nba.com.

## Reproduce
```bash
python3 scripts/build_pts_team_ratings.py
python3 scripts/validate_pts_team_ratings.py
python3 scripts/experiment_ytd_roster_pts.py     # current-roster test (2022-25)
```
