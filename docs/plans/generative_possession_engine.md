# Generative Possession Engine — design note (Core Step 2)

**Status:** PLAN — locked after collaborative planning 2026-05-31. Build pending
user green-light.
**Depends on:** Step 0 lineup projector, Step 1 matchup engine
(`cross_team_interactions.parquet`), `pts_v40`, `projected_player_profiles`
(behavioral_* rates), team pace/profile artifacts.
**Architecture ref:** `docs/phase_v1_architecture.md` (RATIFIED). This is the
**last pre-divergence Core step**; after it, the repos fork.

## Goal

Simulate a game as a sequence of possessions so that a full box score *emerges*,
then calibrate the emergent distribution to real box scores. Judged on **both
player-prop and team-total calibration**. Every simulated stat must be
*explainable* via an explicit decomposition — talent × archetype × matchup — not
a black box.

## The one decision that shapes everything: a decoupled resolver seam

The engine is built so the game-track can later swap in a **full event-level
Markov sim** WITHOUT touching the loop, aggregation, or calibration. The seam is
a single interface:

```
PossessionResolver.resolve(offense_lineup, defense_lineup, game_state)
    -> PossessionOutcome
```

| Module (Core, shared) | Responsibility | Now | Later (game track) |
|---|---|---|---|
| `PossessionLoop` / `GameState` | Alternating-possession driver: score, clock, possession count, pace; agnostic to *how* an outcome is made | ✓ | unchanged |
| **`PossessionResolver`** (interface) | Produce one `PossessionOutcome` | `OutcomeSamplerResolver` (sample from conditioned probs) | `MarkovEventResolver` (full event sim) — drop-in, same interface |
| `RateModel` | The **shared stat-line model**: player attrs + matchup → outcome & attribution probabilities. The thing we **calibrate**. | coarse (behavioral_* + talent) | rich per-player tendency model |
| `BoxScoreAggregator` | Accumulate outcomes+attributions → per-player & per-team box score; emit Monte-Carlo distributions | ✓ | unchanged |

**`PossessionOutcome` schema is rich from day one** so the Markov resolver needs
no schema change later: `{points, fga, fgm, fg3a, fg3m, fta, ftm, scorer_id,
assist_id, tov_id, oreb_id, dreb_id, fouler_id, possession_seconds}`.

> **Explicit later direction (documented per decision):**
> - **Game track (post-fork):** `MarkovEventResolver` — full event-level
>   possession sim (inbound → action → drive/pass/shoot → shot quality →
>   make/miss → rebound → recurse), fed by a **rich per-player tendency model**.
>   Lives *exclusively* in the game/sim sister repo, built on top of this seam.
> - **Market track:** consumes the **resulting box-score distributions**
>   (player-line + team-total) — may never need the engine internals; it reads
>   the aggregated output and overrides lineups with live actuals at T-60/30.
> - Slogan: *possession-outcome sampling now → player-tendency + full event-level
>   Markov sim later in the game track.*

## RateModel — explainable decomposition (reuse existing rates)

Per offensive player, possession-outcome and attribution probabilities are built
**multiplicatively** so calibration tunes scale, not structure:

```
rate = base_rate(behavioral_* + pts_v40 talent)   # who he is, opponent-averaged
     × archetype_shape(off_primary_archetype)       # how he scores
     × matchup_multiplier(Step-1 player_matchup_adj) # vs THIS defense
```

- **base_rate** from `projected_player_profiles`: `behavioral_usage` (who uses the
  possession), `behavioral_efg`, `behavioral_three_point_rate`,
  `behavioral_rim_rate`, `behavioral_turnover_rate`, `behavioral_orb_rate` /
  `behavioral_drb_rate`, `behavioral_free_throw_rate`; scaled by `pts_o_v40` /
  `pts_d_v40` talent.
- **matchup_multiplier** from Step 1's `player_matchup_adjs` /
  `lineup_net_rating_adj`. **Use the FULL empirical mismatch magnitude here**
  (diff2 +0.040, diff1 +0.009, diff0 −0.019) — NOT the 0.5× projection discount;
  the engine simulates real-time mismatch hunting. `center_overloaded` raises
  opponent rim-attempt frequency & finish/FT rate.
- Keeping the three terms separate is what makes any simulated line explainable
  ("X scored 28 because: talent base 22 × creator shape × +matchup vs a weak POA").

## Build sequence (smallest surface first, then expand without stopping)

**v0 — starters + aggregate bench unit:**
- `PossessionLoop` over two lineups; pace from team profiles.
- 5 starters at projected MPG (Step 0) + **one synthetic "bench unit"** absorbing
  remaining minutes at team-average bench rates → all 240 player-minutes covered,
  team totals realistic, starter props real.
- `OutcomeSamplerResolver` + decomposed `RateModel`.
- Monte-Carlo N sims/game → player-line + team-total distributions.

**v1 — light 9–11 man rotation (regular season):** *proceed directly if v0
calibration is clean, no stop.* Replace the aggregate bench unit with the top
9–11 by projected minutes individually (no foul-out/OT/clutch swaps yet). The
loop already accepts a minutes/stint schedule, so this is config + a real bench
rotation, not a rewrite.

(Deferred beyond v1: foul accumulation/foul-out, OT, clutch-lineup swaps,
garbage-time subs — all additive on the same loop.)

## Calibration (walk-forward, leakage-free)

Fit `RateModel` scaling parameters on **train seasons ≤ 2023-24** so simulated
box scores match actuals in aggregate (league outcome-rate priors, usage→points
mapping, team pace/ORtg). Structure (the three multiplicative terms) is fixed;
calibration only tunes scale. Apply forward to 2024-25.

## Validation gate — BOTH, on the 2024-25 holdout

1. **Player-prop calibration:** per player-game, sim distribution vs actual —
   PTS/REB/AST median MAE, interval coverage (P10–P90 ≈ 80%), over/under tail
   calibration.
2. **Team-total calibration:** sim team points & total vs actual — MAE, pace
   match, O/U calibration.

Both reported in one `reports/possession_engine_validation.json`. Gate = simulated
distributions are unbiased and well-covered OOS (not a single point-estimate MAE).
(Market CLV is **deferred** — calibrate the engine before pricing against lines.)

## Artifacts (planned)

- `src/simulation/possession_engine.py` — loop, resolver interface,
  `OutcomeSamplerResolver`, `RateModel`, aggregator.
- `scripts/run_possession_engine.py` — generate sims →
  `data/processed/simulation/possession_box_distributions.parquet`.
- `scripts/validate_possession_engine.py` →
  `reports/possession_engine_validation.json`.
- Registry + findings docs on completion.

## Out of scope (this step)

The full event-level Markov sim, the rich per-player tendency model, foul-out/OT,
and the repo fork itself — all post-Step-2 / game-track. Market-CLV pricing.

## Open risks

- **Usage attribution:** assigning which player uses each possession from
  `behavioral_usage` may over-concentrate on stars; calibrate usage curve.
- **Bench-unit realism (v0):** an aggregate bench can bias variance low; v1
  rotation is the fix, hence "proceed directly if clean."
- **Matchup-multiplier scale:** full mismatch magnitude is large; watch for
  team-total inflation and recalibrate the multiplier→rate transfer.
