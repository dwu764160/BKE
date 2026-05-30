# Step 0a — Lineup Projection Tuning (game-by-game, offline) — RESULTS

**Date:** 2026-05-30
**Code:** `scripts/validate_lineup_projection.py`
**Projector:** `src/simulation/lineup_projection.py` (backtest mode)
**Raw report:** `reports/lineup_projection_validation.json`
**Plan:** `docs/plans/step0_lineup_projection_tuning.md`

---

## 0. TL;DR

We validated the season-level lineup projector **game by game** against actual
lineups derived offline from `pbp_with_lineups_*` (all 9 seasons; 21,495
game-team rows), then tuned its starter-selection weights **walk-forward**.

> **Gate PASSED.** Walk-forward starter hit-rate **0.621 → 0.647** (+2.5pp),
> clutch overlap not regressed (2.872 → 2.894), minutes correlation unchanged.
> The tuner **independently re-selected the same minutes-heavy weighting for
> every eval season** — a robust, non-overfit win. Locked into
> `simulation_config` (`STARTER_WEIGHT_*`).

The honest read: the projector's **starter** identification got meaningfully
better by trusting minutes more; **minutes correlation and clutch** were already
at their structural ceiling for the levers in scope and did not move.

---

## 1. Ground truth (offline, from pbp only)

Derived per game, per team, from the on-court-5 (`lineup_<team_id>`) columns:

| Target | Definition |
|---|---|
| **Starters** | the 5 on court at the first valid 5-man event of period 1 (opening tip) |
| **Clutch-5** | the most time-weighted 5-man lineup in the clutch window: period ≥ 4, \|margin\| ≤ 5, clock ≤ 5:00 |
| **Minutes** | per-player seconds on court from lineup-stint durations (clock deltas between events) |

Coverage: starters present for 100% of game-teams; a clutch window exists in
~46% of games (realistic — about half of NBA games are "clutch"); minutes for
100%. Sanity: per-team game minutes sum to ≈240 (regulation) / 265 (one OT).

## 2. Metrics (projected season lineup vs each actual game)

- **starter_hit** = |proj_starter_5 ∩ actual_game_starter_5| / 5
- **clutch_overlap** = |proj_clutch_5 ∩ actual_game_clutch_5| (0–5)
- **minutes_corr** = Spearman(proj MPG, actual game minutes) over the player union
- **rotation_top9_overlap** (secondary)

## 3. Baseline (current projector, pre-tuning), 19,037 game-teams (8 seasons)

| Metric | Overall | Range across seasons |
|---|---|---|
| starter hit-rate | **0.623** (~3.1 of 5 start) | 0.582 – 0.666 |
| clutch overlap | **2.88** of 5 | 2.52 – 3.21 |
| minutes corr (Spearman) | **0.486** | 0.406 – 0.573 |
| rotation top-9 overlap | 0.676 | — |

(2025-26 has pbp ground truth but no player profiles yet, so it is excluded from
projector scoring — 8 eval seasons, 2017-18…2024-25.)

## 4. Tuning grid + walk-forward result

10-config grid over the starter/continuity/positional levers. Walk-forward: for
each eval season T, pick the config maximizing (starter_hit + minutes_corr) on
seasons < T with a no-clutch-regression gate, then score on T.

| Config | starter_hit | clutch | min_corr |
|---|---|---|---|
| baseline (0.50/0.25/0.10/0.15) | 0.623 | 2.883 | 0.486 |
| starter_m060 | 0.637 | 2.893 | 0.486 |
| starter_m070 | 0.648 | 2.908 | 0.486 |
| **starter_m080 (0.80/0.12/0.04/0.04)** | **0.653** | **2.910** | 0.486 |
| lowmin_125 / lowmin_150 | 0.628 / 0.631 | ~2.886 | 0.486 |
| kappa_050 (more continuity) | 0.619 | 2.882 | 0.486 |
| no_true_big | 0.621 | 2.883 | 0.486 |
| combo_min_heavy | 0.651 | 2.908 | 0.486 |

**Walk-forward chose `starter_m080` for every season with history** (2018-19→
2024-25; 2017-18 has no prior season → baseline). Aggregate over eval seasons:

| | baseline | tuned | Δ |
|---|---|---|---|
| starter hit-rate | 0.6214 | **0.6468** | **+0.0254** |
| minutes corr | 0.485 | 0.485 | 0.000 |
| clutch overlap | 2.8715 | 2.8935 | +0.022 |

**Gate:** starter_hit improved �myes · minutes_corr not worse ✓ · clutch not
regressed ✓ → **PASS.**

## 5. Why it works (basketball sense)

Actual NBA starters are overwhelmingly a team's **highest-minute** players. The
prior starter score diluted minutes (0.50) with impact (0.25), positional
scarcity (0.10) and clutch share (0.15), so it occasionally elevated a
high-impact / low-minute role player over an actual starter. Pushing minutes to
0.80 aligns the projection with how coaches actually set lineups. That the
walk-forward re-derives this every season — across the COVID-bubble years too —
says it's a stable structural fact, not a fit to one regime.

## 6. Honest limitations

- **Minutes correlation is invariant (0.486) across all configs** — by
  construction. The starter-weight levers only change *which 5 are flagged*, not
  the per-player **minute estimates**, which come straight from profile MPG. To
  actually lift minutes_corr you must tune the **minute model**
  (`train_minute_model.py`), which is **out of Step 0a scope**. So the gate's
  "minutes corr not worse" is satisfied as equality, and the genuine measured
  win is starter identification only.
- **Projection is season-aggregated, evaluated per game.** A single projected
  starting five is scored against every game's actual five, so night-to-night
  variation (rest, injury, matchup) caps the achievable hit-rate well below 1.0
  — 0.65 (~3.25 of 5) is a reasonable ceiling for a static projection. The
  per-game injury/rest reality is exactly what **Step 0b** (live override)
  handles for the market track.
- **Clutch ground truth** uses the most time-weighted clutch-window five; games
  without a clutch window (~54%) contribute no clutch metric.

## 7. What changed

- `simulation_config.py`: `STARTER_WEIGHT_M/I/POS/C` = **0.80 / 0.12 / 0.04 /
  0.04** (was 0.50 / 0.25 / 0.10 / 0.15). Backtest mode only — forecast-mode
  starter scoring uses separate hardcoded weights and is unaffected.
- `lineup_projection.py`: `build_projected_lineup_rows(config_overrides=…)` added
  so the tuner can sweep `Step2Config` fields without mutating module defaults.
- Production `simulation_step2_*` artifacts regenerated; `pytest -q` 3/3 pass.

## 8. Reproduce

```bash
python3 scripts/validate_lineup_projection.py --rebuild-truth   # ~13 min (truth + 10-config grid)
python3 scripts/validate_lineup_projection.py                   # reuse cached truth
```

## 9. Next (Step 0b / Step 1)

- **Step 0b** (`src/data_fetch/fetch_pregame_lineups.py`) — live confirmed
  starters + inactives at T-60/30, market-track override. Code-complete; runs in
  the networked env (`stats.nba.com` is throttled in the sandbox — see
  `network_stats_nba_throttle_2026-05-30.md`).
- **Step 1** — cross-team archetype interaction model
  (`docs/plans/cross_team_interaction_model.md`), now that the projected lineups
  it consumes are validated/tuned.
