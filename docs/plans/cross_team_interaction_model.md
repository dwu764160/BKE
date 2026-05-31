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
talent-purged signal** — confirmed on both the original 3-season run and the
**full 8-season run (2017-18…2024-25, 810k rows, 9.6M partial possessions)**:
71/99 archetype-pair cells significant at 95% CI (≈5 by chance), 68 pass BH-FDR,
using **two-way player fixed effects**. The motivating cells hold with tighter CIs:
POA Defender vs Ball Dominant Creator **−0.031 ppp** [−0.042,−0.019], 166k poss;
PnR Rolling Big vs Rim Protector **−0.078** [−0.083,−0.073], 172k poss. The 8
cells that lost significance vs the 3-season run are marginal wing/perimeter cells
(era-sensitive) — the size/position axis (Tier 1-2) is stable across all 8 seasons.
**Gate #1 is met — do not re-litigate "does signal exist"; it does.**

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
4. **Walk-forward:** matchup data covers 9 seasons (2017-18…2025-26, 1.22M raw
   rows). Player archetype profiles cover 2017-18…2024-25 (8 seasons; 2025-26
   profiles not yet generated). The 8-season signal re-run is **COMPLETE**
   (2026-05-31, `reports/archetype_interaction_signal_test.json`); 71/99 sig
   cells, era-stable size/position axis confirmed. Walk-forward validation across
   seasons is now feasible. **Do not re-run the pre-flight; the 8-season matrix
   is locked.**

### Reuse (updated)

`scripts/test_archetype_interactions.py` (estimation + FE + shrinkage scaffolding,
already written and validated), `lineup_projection.py` (projector, now Step-0
tuned), `data/matchup/league_season_matchups.parquet`, the GBDT harness.

---

## Resolved Design — Step 1 Matchup Engine (2026-05-31)

Decisions locked after brainstorm + empirical testing. Do not re-litigate these
unless a new season of data or a new test overturns a specific finding.

### Terminology (canonical for this step)

Position bands come from `position_band_3` (smalls / wings / bigs) and
`position_proxy` (Guard / Guard-Forward / Forward / Forward-Center / Center).

| Term | Meaning | Data source |
|---|---|---|
| **smalls** | Guard-band players | `position_band_3 = 'smalls'` |
| **wings** | Forward-band players | `position_band_3 = 'wings'` |
| **bigs** | Center-band players | `position_band_3 = 'bigs'` |
| **interior threat** | Bigs-band offensive player (or wings-band with PnR Rolling Big / Interior Scorer archetype) with high `pts_o_v40` | archetype + `pts_o_v40` |
| **big-bodied defender** | Wings-band defender (`position_proxy` = Forward or Forward-Center) who is **above league-average defensively** (`pts_d_v40 > 0.0`) and ranks top-3 on their team by `pts_d_v40` — a physically capable forward who can absorb the interior offensive threat without being the primary center. A Forward-Center with `pts_d_v40 < 0.0` does not qualify; the center must step up (Sub-case B). | `position_proxy` + `pts_d_v40` |
| **rim anchor** | Bigs-band defender classified as Rim Protector or Dropping Big; primarily a paint/zone presence, not a 1-on-1 matchup defender in the traditional sense | `def_primary_archetype` + `position_band_3` |

### Encoding decisions (locked)

1. **Archetype for cell lookup: argmax always.** `off_primary_archetype` /
   `def_primary_archetype` are used to index the interaction matrix. The softmax
   test (8 seasons, 810k rows) showed softmax weighting degrades prediction at
   every metric and every confidence tier except near-zero-gain at low confidence
   — adding noise from off-archetype cells outweighs any benefit from soft
   assignment.
2. **Softmax for Unknown-archetype fallback only.** If `off_primary_archetype`
   is "Unknown" or "Insufficient Minutes" (21.6% of minutes), fall back to the
   position-band-average cell: average `interaction_ppp` over all archetypes in
   that position band for the given `def_primary_archetype`. Do not apply
   softmax probs as weights in any other context.
3. **Talent rank metric: `pts_o_v40`** (offensive PTS 4.0 score from
   `data/processed/bke/pts_v40.parquet`). Used to rank offensive players by
   matchup threat within and across position bands. `impact_obke` is a BKE
   composite that diverges at r=0.51 from `pts_o_v40` — do not substitute.

### Data note: what "matchup" data actually measures

`data/matchup/league_season_matchups.parquet` is NBA `LeagueSeasonMatchups`,
powered by Second Spectrum spatial tracking (closest-defender proximity
attribution). It records **emergent outcomes** — who was physically nearest
the ball-handler — not intended defensive assignment. Double-team possessions
are attributed to the primary (pre-double) defender, not split. Scheme-driven
assignments (zone, help-and-recover) are invisible. **Every use of this data
is labeled `source: emergent_matchup` in the output artifact provenance.** This
caveat does not invalidate the signal (the interaction cells are real and large)
but means the "assignment" we compute is a probabilistic estimate, not a
ground-truth label.

### Matchup assignment engine — three pairs

**Pair 1: Primary perimeter threat**

*What it models:* The opponent's top offensive ball-handler (smalls or wings,
primary creator archetype) against the defense's top perimeter defender.

*Assignment logic:*
1. Identify the offensive team's **primary perimeter threat**: highest
   `pts_o_v40` player among smalls-band players with `off_primary_archetype` ∈
   {Ball Dominant Creator, All-Around Scorer, Ballhandler}. If none qualify,
   extend to wings-band.
2. Assign the defense's best available POA Defender or Wing Stopper
   (`def_primary_archetype` + highest `pts_d_v40` among smalls/wings defenders).
3. Look up `interaction[off_arch][def_arch]` (argmax, FE-adjusted matrix).
4. Emit: `player_matchup_adj_ppp` (per-player) and contribute to lineup net
   rating via `pts_o_v40`-weighted sum.

*Source tag:* `assignment_proxy` — the POA-on-creator assignment is the most
empirically stable pair (166k possessions, era-stable) but still inferred.

---

**Pair 2: Interior threat + defensive big behavior**

*What it models:* The opponent's top interior offensive player against the
defense's big-man resource allocation. This pair has two sub-cases determined
by defensive roster composition.

*Identifying the interior offensive threat:* highest `pts_o_v40` among bigs-band
players with `off_primary_archetype` ∈ {PnR Rolling Big, Interior Scorer,
PnR Popping Big}. If the offense has a wings-band player with one of these
archetypes and a high `pts_o_v40` (a physical wing playing interior), include
them as a candidate.

**Sub-case A — Big-bodied defender available:**
If the defensive team has a **big-bodied defender** (wings-band, `position_proxy`
= Forward or Forward-Center, top-3 `pts_d_v40` on their roster AND
`pts_d_v40 > 0.0`): that player guards the interior offensive threat.
A below-average defensive forward does not qualify — putting a bad defender
on a high-`pts_o_v40` interior threat is worse than a center overload. The rim anchor (bigs-band center) remains
in the paint focused on rim protection, not assigned 1-on-1. Look up:
`interaction[off_arch][big_bodied_def_arch]` — typically a wings-vs-interior
cell (band_diff=1; expected residual adj ~+0.009 ppp from mismatch test).

**Sub-case B — No big-bodied defender available:**
The rim anchor must guard the interior offensive threat directly (band_diff=0 or
1 depending on whether the threat is a forward-center vs true big). Flag:
`center_overloaded = True`. Cell lookup: `interaction[off_arch][rim_anchor_def_arch]`.
This is the scenario that degrades team defense — the center cannot simultaneously
guard the interior scorer AND protect the rim from roll-man attacks. Model the
rim protection role as a lineup composition term (always active) separately from
the 1-on-1 pair cell.

*Rim protection as lineup composition:*
The rim anchor's effect is partly categorical — does this defense have a Rim
Protector or Dropping Big in the rotation? If yes: apply the Rim Protector /
Dropping Big cell to roll-man / big-roller matchups regardless of explicit
assignment (the paint presence is real even without formal 1-on-1 assignment).
*Source tag:* `scheme_inferred` for the sub-case logic; `emergent_matchup` for
the cell values.

---

**Pairs 3–5: Position band + talent rank + mismatch flag**

*What it models:* The three remaining offensive players, matched to the three
remaining defenders after pairs 1–2 are assigned. This is the residual set where
per-player matchup data is sparse and assignment is noisiest. The approach is
intentionally lower-fidelity — the primary signal comes from physical compatibility
(position band), not archetype, because most players here are mid-rotation role
players with flat archetype softmax distributions.

*Assignment logic:*
1. Rank the remaining offensive players by `pts_o_v40` within their position
   band (smalls, wings, bigs). Do the same for remaining defenders.
2. **Default: match rank-to-rank within the same band.** 1st smalls offender →
   1st smalls defender, etc. Projection assumes coaches will avoid sustained
   cross-band mismatches — teams switch back, call timeouts, and adjust
   rotations. Do not project a mismatch unless no same-band option exists.
3. If no same-band defender is available (cross-band forced): set
   `band_diff = |off_band_n − def_band_n|` and apply a **discounted** mismatch
   adjustment. Coaching adjustments (switching back, loading help) partially
   neutralize mismatches — the full empirical magnitude assumes the mismatch
   was sustained, which is not a safe projection assumption.

*Mismatch adjustment — empirically quantified (2026-05-31 test, 8 seasons):*

| Band diff | Observed FE-resid | Archetype cell | Raw gap | **Projection discount (×0.5)** |
|---|---|---|---|---|
| 0 (same) | −0.026 | −0.007 | −0.019 | **−0.010** |
| 1 (adjacent) | +0.011 | +0.002 | +0.009 | **+0.005** |
| 2 (cross) | +0.058 | +0.018 | +0.040 | **+0.020** |

The 0.5 discount factor is an initial calibration parameter, not a hard
constant — adjust in validation against 2024-25 holdout. The full gap
(+0.040 / +0.009 / −0.019) is retained in the possession engine where real
player mismatch hunting is simulated, not projected.

Direction: **+** = offense scores above archetype prediction (defender undersized);
**−** = defense wins beyond archetype (defender oversized). The diff=0 same-band
penalty (−0.010 discounted) reflects that physically matched defenders outperform
archetype averages — not a mismatch, just a calibration correction.

**Directionality nuance:** The diff=2 adjustment is an average. For cells
where the archetype already encodes the mismatch mechanism (BDC vs Rim Protector,
PnR Rolling Big vs POA Defender), the cell value itself captures most of the
gap — apply the discounted adj only to the unexplained residual. For generic
cross-band assignments (wings guarding a small with no matching archetype cell),
the full discounted gap applies.

**Unknown archetype fallback:** Use `position_band_3`-average interaction_ppp
over all archetypes in that band, for the defender's archetype. Document as
`source: band_average_fallback`.

*Source tag for pairs 3-5:* `assignment_proxy` (band-based) with
`mismatch: measured` for the gap terms.

### Output artifact schema

`data/processed/bke/cross_team_interactions.parquet` — per (game_id, team) row:

| Column | Description |
|---|---|
| `pair1_off_player_id` | Primary perimeter threat player |
| `pair1_def_arch` | Assigned defender's archetype |
| `pair1_interaction_ppp` | FE-adjusted cell value |
| `pair1_source` | assignment_proxy / emergent_matchup |
| `pair2_off_player_id` | Interior offensive threat |
| `pair2_sub_case` | A (big-bodied avail) or B (center overloaded) |
| `pair2_interaction_ppp` | Cell value + rim-anchor lineup term |
| `pair3_5_interaction_ppp` | Sum of pairs 3-5 cell values + mismatch adjs |
| `band_mismatch_flags` | JSON: {pair_n: band_diff} per residual pair |
| `lineup_net_rating_adj` | Sum of all pairs, minute-weighted |
| `player_matchup_adjs` | JSON: {player_id: ppp_adj} for props / stat-line use |
| `center_overloaded` | Bool — Pair 2 Sub-case B fired |

### Validation (per earlier decision)

Primary target: player-game-level PPP prediction improvement (MAE), holdout
2024-25 season. Game-level Brier is a secondary diagnostic, not the gate.
Walk-forward: train on 2017-18→2023-24, validate on 2024-25 player-game stats
using projected starters. Archetype cell source tag used to stratify results
(emergent_matchup cells expected to outperform assignment_proxy cells).
