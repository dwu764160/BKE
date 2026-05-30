# Step 0 — Lineup Projection Tuning (starters / clutch / rotation strength)

**Status:** PLAN — awaiting approval before commit.
**Date:** 2026-05-30
**Why before Step 1:** the cross-team interaction model and the whole game engine
consume *projected lineups*. If the projector is inaccurate, every downstream
matchup and possession is wrong. We must establish and tune its accuracy first.
**Depends on:** `lineup_projection.py`, `pbp_with_lineups_*` (all 9 seasons),
archetypes, minute model.

## The key decision (answered)

Do Step 0 at a **game-by-game level against actual lineups**? **Yes — high ROI:**
- Ground truth is **free and offline**: `pbp_with_lineups_*` gives the actual
  on-court 5 per team per event for all 9 seasons. Starters = opening-tip lineup;
  clutch lineup = on-court 5 in clutch windows; rotation = per-player actual
  minutes. **No network needed to validate/tune.**
- The **live** injury-report + actual-starter fetch (T-60/30 min) we'll need anyway
  for the market track is the *same* concept forward in time — so building Step 0
  also de-risks that infra. One concept, two payoffs.

## Scope split (offline now vs network later)

- **Step 0a — offline tuning (this build):** derive actual starters/clutch/minutes
  from pbp, score the projector game-by-game, tune to maximize accuracy.
- **Step 0b — live fetch (market infra, user's networked env):** injury report +
  pre-game actual starters at T-60/30 min, reused by the market track to override
  projected starters. Spec + code here; run from an environment whose IP
  `stats.nba.com` accepts. (Note: general internet works in the dev session, but
  `stats.nba.com` specifically times out here — NBA datacenter-IP throttling — so
  the live fetch is verified in the normal working environment, not this session.)

## Ground truth (offline, from `pbp_with_lineups_*`)

- **Starters:** the 5 players in each team's `lineup_<team_id>` at the first event
  of period 1.
- **Clutch lineup:** the on-court 5 during clutch windows (period ≥ 4, |margin| ≤ 5,
  game clock ≤ 5:00) — the standard NBA clutch definition.
- **Rotation strength:** actual per-player minutes (from lineup-stint durations, or
  `player_game_logs` for 2022-25) → who really plays and how much.

## Metrics (game-by-game)

| Target | Metric |
|---|---|
| Starters | mean # of 5 projected starters who actually started (0–5); exact-5 rate |
| Clutch | overlap between projected clutch-5 and actual clutch-5 |
| Rotation | corr(projected minutes, actual minutes); top-9 rotation set overlap |
| Rotational **proxy** strength | does the projected rotation's aggregate PTS track the actual rotation's? (the "proxy" the game track later replaces with full rotation projection) |

Report a baseline (current projector) then tune.

## Tuning levers (in `lineup_projection.py` / `simulation_config`)

Starter-band thresholds, minute-model weighting, continuity κ (returning-player
carry), clutch candidate/swap params (`STEP2_CLUTCH_*`), role/position-band
assignment. Tune to maximize the metrics above with walk-forward discipline
(tune on seasons < T, evaluate on T) so we don't overfit.

## Validation gate (performance-prioritized)

1. Establish baseline accuracy per metric across 9 seasons.
2. Tuned projector must **beat baseline** on starter hit-rate and minutes
   correlation without regressing clutch overlap.
3. Output a per-game projected-vs-actual report → `reports/lineup_projection_validation.json`.

## Track divergence (reminder)

- **Markets:** Step 0b live fetch overrides projected starters with real ones
  (+ injuries) at T-60/30 min.
- **Game:** uses the *projected* lineups only, and later projects **full actual
  rotational lineups** (not just the starter proxy) for the possession engine.

## Output

- `reports/lineup_projection_validation.json` (offline accuracy report).
- Tuned params committed to `simulation_config`.
- Step 0b: `src/data_fetch/fetch_pregame_lineups.py` (live starters + injury) —
  code-complete, run in networked env; schema shared with the market track.

## Reuse

`lineup_projection.py`, `pbp_with_lineups_*`, `train_minute_model.py`,
`derive_lineups.py` (actual-lineup derivation patterns), the walk-forward harness.

## Out of scope

Full rotational-lineup projection for the sim (game track, later); the cross-team
interaction model (Step 1); any model weight changes outside the projector.
