# Cross-Team Archetype Interaction Model — design note (Core step 1)

**Status:** PLAN — immediate build (ratified build order, `docs/phase_v1_architecture.md` §9).
**Date:** 2026-05-30
**Depends on:** projected lineups (`lineup_projection.py`), archetypes, PTS v4.0, pbp/matchup data.

## Goal

Resolve the value of a matchup *between two lineups* — "a good POA defender vs a
ball-dominant creator: what happens?" — and emit it as a Core artifact both tracks
consume. This is the unused unlock: the lineup projector exists but has never fed
cross-team matchup resolution (verified: it's wired only to `season_sim.py`).

## Critical distinction (avoid a known trap)

The repo's earlier Lasso fit found **within-team** archetype-pair synergy
**negligible** (`INTERACTION_MATRIX = {}`, `team_feature_aggregation.py:167-196`) —
i.e., pairing two offensive archetypes *on the same team* adds little.

**This is a DIFFERENT and untested hypothesis:** **cross-team
attacker-vs-defender** matchups. A POA defender genuinely suppressing a creator,
or a rim protector erasing a roller, is a real basketball effect that the additive
PTS/RAPM model cannot express (RAPM is opponent-averaged). Do **not** assume the
within-team null result transfers — measure it fresh.

## Double-counting guard (performance + correctness)

PTS already carries each player's *opponent-averaged* talent level. The interaction
model must add **only the matchup-dependent delta** — how a player's efficiency
deviates from his average *specifically against a given defender archetype* — never
re-add talent. Construction: interaction term is mean-zero over defender archetypes
by definition (Σ over opponents of the adjustment = 0), so it cannot inflate a
player's baseline.

## Estimation (from data we have)

1. From pbp / `fetch_matchup_data` + lineups, build attacker-archetype ×
   defender-archetype efficiency cells: PPP (or eFG / TOV) of offensive archetype A
   when primarily guarded by defensive archetype D, vs A's overall average.
2. `interaction[A,D] = PPP(A | guarded by D) − PPP(A overall)` (shrunk toward 0 by
   sample size — empirical-Bayes, like the RAPM shrinkage already in the repo).
3. Per game, given two projected lineups, assign likely matchups (by position band /
   minutes overlap), sum the minute-weighted interaction deltas → a team-level
   `cross_team_interaction_adj` (and a per-matchup breakdown for the game engine).

## Output (Core artifact)

`data/processed/bke/cross_team_interactions.parquet`:
- the shrunk `interaction[A,D]` matrix (small, reusable), and
- a per-(game_id, team) `interaction_adj` once lineups are attached.

## Validation gate (performance-prioritized, same discipline as PTS)

1. **Signal exists?** Do interaction cells differ from 0 beyond shrinkage noise
   (e.g., POA-vs-creator meaningfully negative)? If the matrix is ≈0, stop — report
   "cross-team matchups also negligible" and move to the stat-line model.
2. **Incremental value?** Add `interaction_adj` (home−away) as a feature in the
   walk-forward GBDT harness (`scripts/experiment_microadjustments.py` pattern);
   paired bootstrap vs Elo and vs the current stack. Keep only if HELP.
   *Note:* it may still not beat Elo at the game level (consistent with prior
   findings) yet be valuable for the **game engine's per-possession realism** and
   for **player props** — so judge it on BOTH the game-level gate and a lineup/
   matchup-level fit, not game Brier alone.

## Track divergence (where this feeds)

- **Markets:** override projected starters with **live actual starters + injury**
  (~T-60/30 min), recompute `interaction_adj`, price props/totals.
- **Game:** use **fully projected** lineups (full rotations + clutch), feed the
  per-matchup interaction breakdown into the possession engine.

## Reuse

`lineup_projection.py` (projector), `fit_archetype_interactions_v2.py` (fit
scaffolding + shrinkage), `fetch_matchup_data` outputs, pbp_normalized (all 9
seasons), the GBDT harness for the incremental gate.

## Out of scope

The possession engine and the generative stat-line model (later Core steps). This
note covers only the interaction matrix + per-game adjustment and its validation.

---

## Empirical grounding (2026-05-30) — gate #1 PASSED, build sequence locked

Before committing to a build, we ran the signal test
(`scripts/test_archetype_interactions.py`, full writeup:
`docs/findings/archetype_interaction_signal_2026-05-30.md`). Result: **real,
talent-purged signal** — 79/99 archetype-pair cells significant at 95% CI (≈5 by
chance), 76 pass BH-FDR, using **two-way player fixed effects** on NBA matchup
data (per off×def-player possessions, 2022-23…2024-25). The motivating cells hold:
POA Defender vs Ball Dominant Creator **−0.038 ppp** [−0.047,−0.027]; PnR Rolling
Big vs Rim Protector **−0.091** (talent was *masking* this one). **Gate #1 is met
— do not re-litigate "does signal exist"; it does.**

Two things the test forces into the build:

1. **The dominant axis is size/position.** The biggest cells are cross-position
   (guard offense vs big defender, +0.14…+0.19 ppp) and only occur on **switches**.
   ⇒ Apply the matrix through **position-band matchup assignment**; hold the
   cross-position cells as a **separate switch/mismatch term** gated by an
   assignment flag — never apply the full off×def matrix to every pair.
2. **Positive-vs-creator artifacts exist** (e.g., Versatile Defender +0.057) —
   shrink hard, mean-zero per offensive archetype over defenders, and eyeball the
   underlying player lists for any positive suppression cell before trusting it.

### Revised, concrete build steps

1. **Estimation = the test's method, promoted to an artifact.** Reuse the
   two-way-FE residual `ppp` cells from `test_archetype_interactions.py` (NOT a
   raw `PPP(A|D) − PPP(A)` diff — that re-imports talent). Add empirical-Bayes
   shrinkage by cell possessions and enforce Σ_D interaction[A,D]·poss = 0 per
   offensive archetype. Emit `data/processed/bke/cross_team_interactions.parquet`
   (the shrunk matrix) + provenance (n_pairs, poss, CI per cell).
2. **Per-game `interaction_adj`:** attach projected lineups (Step 0 projector),
   assign matchups by position band + minutes overlap, sum minute-weighted
   interaction deltas → team-level adj (+ per-matchup breakdown for the engine).
   Switch term applied only where a mismatch is forced.
3. **Gate #2 (incremental value):** add `interaction_adj` (home−away) to the
   walk-forward GBDT harness (`experiment_microadjustments.py` pattern), paired
   bootstrap vs Elo and vs the current stack. **Expectation, not a failure
   condition:** magnitudes (~−0.04 to −0.09 ppp; a few team points/game) likely
   make this **marginal for game-level Brier** but valuable for **props and
   possession realism** — judge on BOTH (a lineup/matchup-level fit + the game
   gate), per the original note.
4. **Walk-forward caveat:** matchup data starts 2022-23, so the matrix itself
   can't be validated pre-2022-23; it ports to all seasons via archetype labels.

### Reuse (updated)

`scripts/test_archetype_interactions.py` (estimation + FE + shrinkage scaffolding,
already written and validated), `lineup_projection.py` (projector, now Step-0
tuned), `data/matchup/league_season_matchups.parquet`, the GBDT harness.
