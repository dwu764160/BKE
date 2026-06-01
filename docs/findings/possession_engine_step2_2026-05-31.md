# Step 2 — Generative Possession Engine: build & validation (2026-05-31)

Implements the locked design in `docs/plans/generative_possession_engine.md`. The
last pre-divergence Core step. Three deliverables, all run clean; both modes PASS
the validation gate.

## Architecture — the decoupling seam (load-bearing)

`src/simulation/possession_engine.py`:

- `PossessionResolver.resolve(off, defense, state, rng) -> PossessionOutcome` — the
  seam. `PossessionLoop`/`BoxScoreAggregator` are agnostic to *how* an outcome is
  made.
- `OutcomeSamplerResolver` — Core's lightweight possession-outcome sampler. Exposes
  the canonical single-possession `resolve()` (seam + unit-test path) AND a
  vectorized `simulate_team_batch()` for full-season Monte-Carlo; both share one
  `RateModel`, so they are consistent by construction.
- `PossessionOutcome` carries a rich schema from day one (scorer/assist/tov/reb/
  foul ids + seconds) so the game track's future `MarkovEventResolver` drops in with
  no schema change.
- `RateModel` — the shared, calibratable stat-line spine. Explainable, multiplicative:
  `rate = base_rate(behavioral_* + pts_v40 talent) × archetype_shape × matchup_mult(Step-1)`.
  The Step-1 matchup adjustment is applied at **full empirical magnitude** here (the
  possession engine simulates real-time mismatch hunting), not the 0.5× projection
  discount used in the season-level artifact.

## Build sequence (smallest surface first, then expand without stopping)

- **v0** — 5 starters at projected MPG + one synthetic aggregate bench unit absorbing
  the remaining minutes at team-average rates (covers all 240 player-minutes).
- **v1** — top 9–11 by projected MPG individually, minutes normalized to 240; no
  foul-out/OT/clutch swaps (deferred, additive on the same loop).

## Calibration (walk-forward, leakage-free)

Global constants set from **train ≤2023-24** team logs: `pace` (mean possessions/
game), `oreb`, `league_ft`, `league_3p`; then a single global `efg_scale` tuned so
simulated mean team PTS matches actual on a train sample. Structure (the three
multiplicative `RateModel` terms) is fixed; calibration only tunes scale. Game-to-
game dispersion comes from per-sim Poisson pace + per-possession binomial outcome
sampling (no explicit shock knobs in this cut — adding per-game minutes/efficiency
dispersion to widen prop intervals is the Step 2.1 item below). Applied forward to
2024-25. Calibrated constants (v1): pace 102.3, efg_scale 0.945, oreb 0.230,
league_3p 0.359, league_ft 0.776.

## Validation — BOTH gates, 2024-25 holdout

`scripts/validate_possession_engine.py` → `reports/possession_engine_validation.json`.

| Gate | v0 (starters+bench) | v1 (9–11 rotation) |
|---|---|---|
| Team PTS MAE | 10.21 | 10.34 |
| Team PTS bias | +1.08 | +1.86 |
| Team P10–P90 coverage | 0.890 | 0.885 |
| Player pts MAE | 6.65 | **5.89** |
| Player AST MAE | 1.98 | 2.09 |
| Player REB MAE | 2.74 | 2.46 |
| Player pts P10–P90 coverage | 0.642 | 0.649 |
| Player-games scored | 7,274 | 13,173 |
| **Verdict** | **PASS** | **PASS** |

v1 is the stronger player-prop model (pts_mae 5.89 vs 6.65, and 13,173 vs 7,274
scorable player-games) — modeling individual rotation players beats the aggregate
bench unit, exactly the v0→v1 expectation. Team totals are well-calibrated in both
(bias +1 to +2 pts, coverage ~0.89, above the 0.80 target). Sim team mean ≈ 115 vs
actual ≈ 114.

## Bugs found & fixed during the build

1. **Defensive rebounds almost never recorded** (sim 0.001/player). In the
   vectorized miss branch, DREB was assigned in an `else` to the OREB test, which
   fired only when *no* shot in the whole batch drew an offensive rebound — so it
   essentially never ran. Fixed to credit DREB on the complement mask (`~oreb`)
   independently. REB MAE dropped 4.20 → ~2.4.
2. **Per-season team-log paths don't exist** — there's a combined
   `team_game_logs.parquet` (all seasons, `SEASON` column), not
   `team_game_logs_<season>.parquet`. Calibration was silently falling back to
   defaults. Switched to the combined file filtered by season.
3. **Player game logs ship both `Player_ID` and `PLAYER_ID`** → collide to a
   duplicate label on `.upper()`. Deduped columns post-rename.

## Known limitation (deferred)

Player-points interval coverage is 0.65 vs the 0.80 ideal — intervals too narrow
(overconfident). The dominant missing variance source is **minutes**: v1 holds
projected MPG fixed and cannot express DNPs, foul trouble, or blowout garbage-time
minute swings, which drive a large share of real player-game point variance. This
cut also has no explicit per-game efficiency shock (dispersion is only Poisson pace
+ per-possession binomial). Adding per-game minutes + efficiency dispersion (and the
deferred foul-out/OT/clutch logic) is the first Step-2.1 calibration item; team
totals already clear the gate, so this is a prop-sharpening task, not a blocker.

## Handoff to the fork

This is the last shared Core step. Both tracks now consume
`possession_box_distributions.parquet`:
- **Markets:** reads the emergent box-score *distributions* (player-line + team-
  total) to price props/totals; overrides projected starters with live actuals at
  T-60/30. Never runs the resolver.
- **Game:** swaps in `MarkovEventResolver` (full event-level sim + rich per-player
  tendency model) on the existing `PossessionResolver` seam.
