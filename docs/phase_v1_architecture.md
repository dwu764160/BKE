# Phase v1 — Architecture, Track Split, and the Lineup Unlock

**Status:** RATIFIED 2026-05-30 — Core + 2 sister repos approved; game repo created
after Core step 2; build order = lineup projector → cross-team interactions first;
simulation = bottom-up generative (§10).
**Date:** 2026-05-30
**Context docs:** `docs/findings/game_model_comparison_2026-05-30.md`,
`docs/findings/pts_wiring_phase1_2026-05-30.md`, `docs/roadmap_to_vegas_accuracy.md`

---

## 1. Why this document exists

We are at ~v0.8. Two hard, *measured* findings reset the strategy:

1. **Game-winner prediction is a dead end vs the market.** Elo (0.219 Brier, 0.209
   on 2025-26) is the best model; PTS/BKE add nothing over it at the game level
   (lagged = noise; current-roster = noise, direction-correct but negligible).
   The NBA moneyline is the most efficient market and prices exactly what we model.
2. **PTS is an excellent player/team *evaluation* metric** (same-season team
   r=0.84) — wrong tool for moneylines, right tool for player props, team totals,
   and a simulation engine.

This forces a deliberate split into two products that share one engine:
- **Markets track** — a prediction-market bot that *scans the long tail* (props,
  totals, secondary lines) for exploitable mispricings, not a beat-the-closing-line
  moneyline model.
- **Game track** — a Tier-2 generative possession-level basketball simulation
  (manager game) in a sister repo.

This doc locks the architecture, the precise divergence point, and the sequencing
so the two tracks can fork cleanly without duplicating or corrupting the core.

---

## 2. Finding: the lineup projector is NOT used in prediction (correct), and is the unlock

Verified 2026-05-30:
- `src/simulation/lineup_projection.py` (the starter / rotation / clutch projector,
  "Step 2") is imported **only** by `season_sim.py`, `run_forecast.py`,
  `grid_sweep_sim_knobs.py` — the **season Monte-Carlo track**.
- The **game-level prediction path** (`game_model.py`, `validate_forecast.py`,
  `forecast_2025_26_games.py`, `build_all_season_projections.py`) uses **no
  lineups** — only the prior-year-margin team rating. **No under-the-hood misuse.**
- It produces `simulation_step2_lineup_profiles` (starter/rotation/clutch) but has
  **never** fed game-level prediction or **cross-team archetype interactions**.

→ Projecting starter/rotation/clutch lineups unlocks **cross-team archetype
interaction** ("a good POA defender vs a ball-dominant superstar — what happens?"),
which neither track currently exploits. This is the **next slot-in, before** the
prop/stat-line model (§7).

---

## 3. Repo & architecture strategy (DECISION NEEDED)

**Recommendation: one Core repo + two consumer sister repos** (matches the existing
Robinhood/Kalshi sister-repo pattern in `CLAUDE.md`, and the user's instinct).

```
            ┌──────────────────────────────────────────────┐
            │  BKE CORE (this repo)                          │
            │  data pipeline → RAPM → PTS v4.0 → archetypes  │
            │  → lineup projector (starter/rotation/clutch)  │
            │  → cross-team archetype interaction model      │
            │  → generative player stat-line model (NEW)     │
            │  Publishes VERSIONED artifacts + a stable API. │
            └───────────────┬───────────────┬───────────────┘
                            │               │
              ┌─────────────▼──┐         ┌──▼──────────────────┐
              │ MARKETS track   │        │ GAME track (NEW repo)│
              │ (Kalshi bot —   │        │ Tier-2 generative    │
              │  existing       │        │ possession engine +  │
              │  Robinhood repo)│        │ manager meta-systems │
              │ live lineups +  │        │ + UI. Projected      │
              │ injury fetch +  │        │ lineups only.        │
              │ edge-scanner    │        │                      │
              └─────────────────┘        └──────────────────────┘
```

**Why core + sisters (not a monorepo):**
- The two products have different deps, cadences, and risk (a betting bot vs a
  game UI). A monorepo couples their release cycles and bloats both.
- The cross-repo contract already exists (`gh api ... for-alpha-thesis.md`).
- Core stays a clean, testable engine that publishes artifacts; consumers pin a
  core version.

**Cost to accept:** cross-repo versioning. Mitigation: Core publishes a **versioned
artifact contract** (the parquet/JSON outputs + a documented schema), and each
sister repo records which Core version/artifact set it consumes (extend
`docs/multiple-versions.md` with a "Core API surface" section).

**Alternatives considered:** (a) monorepo with `core/ markets/ game/` boundaries —
simpler sync, but couples cadence and bloats; (b) three independent repos with no
shared core — duplicates the engine, guaranteed drift. Both rejected.

---

## 4. The shared Core (everything before the divergence)

The engine both tracks consume, in dependency order:
1. Data pipeline (fetch → normalize → possessions/lineups) — built.
2. RAPM → PTS v4.0 (player impact) — built, validated (team r=0.84).
3. Archetypes (offensive v4.3, defensive v3.4) — built.
4. **Lineup projector** (starter / rotation / clutch) — built, but only wired to
   season sim; **to be wired as a Core artifact** (§7).
5. **Cross-team archetype interaction model** — NEW (§7): resolves matchup value
   between projected lineups (POA vs creator, rim protector vs roller, etc.).
6. **Generative player stat-line model** — NEW (§8): given player + role + minutes
   + matchup, produce a calibrated stat-line distribution. Serves **both** props
   and the game engine.

Core publishes these as versioned artifacts; tracks never re-derive them.

---

## 5. The divergence point (precise)

Both tracks share the Core through the projected lineups and the interaction +
stat-line models. **They diverge at the lineup *realization* layer and the
consuming application:**

| Aspect | Markets track | Game track |
|---|---|---|
| Lineups | **Fetch ACTUAL starters + injury report ~T-60/30 min**, override the projection, recompute interactions/props fast | **Fully projected** (no real game to fetch); project **actual rotational lineups**, not just the starter proxy |
| Rotation depth | starter-level + live actives is enough to price | needs full rotation + clutch for possession sim |
| Loop | low-latency: news → recompute → scan lines → act | offline: simulate seasons / games deterministically + stochastically |
| Output | edge vs market lines (props/totals), CLV | simulated games, box scores, standings, careers |
| Goal | conditional edge on soft lines | fun, plausible, responsive simulation |

**So:** shared = projector + interaction model + stat-line model. Diverge = the
lineup *input* (live-actual vs projected) and the application around it.

---

## 6. Markets track — plan (props/totals edge-finding, NOT moneylines)

Premise: don't beat the consensus; find lines where our number diverges by more
than the vig, only in markets where we have a *real* informational edge.
1. **Target the long tail:** player props (pts/reb/ast), team totals, secondary
   lines — where the market is lazy on role players and unusual lineups.
2. **Live lineup/injury fetch** at T-60/30 min → override projected starters with
   actuals → recompute the generative stat-line model → produce our prop number.
3. **Edge-scanner:** compare our number to all available lines (Kalshi +
   sportsbooks); flag divergences > threshold; size by edge (Kelly-fraction).
4. **CLV measurement per prop** is the North Star — log every flagged line vs its
   close; positive CLV before any real allocation (consistent with current $0
   research-only stance).
5. **Execution/line-shopping** layer (later): cross-market arb, stale-line capture.
Lives in the existing trading sister repo (Sleeve C consumes Core artifacts).

## 7. Game track — plan (Tier-2 generative possession engine)

What exists (be accurate): `season_sim.py` (team PPP + Gaussian Monte Carlo, 10k
seasons), `player_stats_sim.py` (single-game box-score sim + archetype matchup
adjustments), `train_minute_model.py`, `lineup_projection.py`, and
`pbp_normalized` for all 9 seasons (the training data for a generative engine).

Gaps to a **true generative possession engine**:
1. Possession state machine / event generation (init → action → shot/TO/foul →
   rebound → transition) — train on pbp.
2. Player decision/policy (who shoots/passes/defends from game state).
3. Head-to-head matchup resolution (assignment + mismatch exploitation).
4. In-game dynamics (fatigue, foul trouble, momentum, clutch, garbage time).
5. Tactical/coaching layer (play-calling, PnR coverage, pace control, late-game
   fouling) — the manager's inputs.
6. Manager meta-systems: development/aging, injuries, contracts/cap, trades, draft,
   chemistry, scouting.
7. (Optional) spatial/shot-location model.
Lives in a NEW sister repo; consumes Core (ratings + projector + interaction +
stat-line model).

---

## 8. Sequencing — when we fork

Fork the repos only AFTER the shared Core is locked. Order:
1. **Lineup projector → Core artifact** + **cross-team archetype interaction
   model** (§7 unlock). Validate the interaction signal (does POA-vs-creator etc.
   move outcomes measurably?). *Shared; both tracks need it.*
2. **Generative player stat-line model** (§4.6). Validate vs real box scores.
   *Shared; serves props AND the game.*
3. **FORK.** Markets adds live-lineup fetch + edge-scanner + execution; Game adds
   the possession engine + meta-systems + UI.

**Divergence point in time = after step 2** (the stat-line model validates as a
shared artifact). Before that, both tracks are 100% shared — do not split repos
yet.

---

## 9. Decisions — RATIFIED 2026-05-30

1. **Repo strategy (§3):** ✅ Core + 2 sister repos.
2. **Game repo timing:** ✅ create after Core step 2 (stat-line model validated).
3. **Markets home:** ✅ existing Robinhood/Kalshi repo (Sleeve C).
4. **Build order:** ✅ lineup projector → cross-team interaction model first.
5. **Doc practice:** ✅ lock the Core API contract (§11) + the immediate-build
   design note now; defer full `docs/game/` (possession engine) until just before
   that build and full `docs/markets/` until the fork. Document at coupling +
   uncertainty, just ahead of the build.

---

## 10. Simulation architecture — bottom-up generative, NOT predict-then-backfill

**Decision:** the game track is a **bottom-up generative** possession engine. We do
**not** draw a box score first and backfill a plausible play-by-play to match it.

Why top-down "predict the result then fill in how we got there" is rejected:
- It **kills interactivity** — if the final is fixed first, mid-game manager
  decisions (subs, defensive scheme, fouling) can't change anything.
- It **kills cross-team archetype interactions** — those only fire per-possession
  (POA defender assigned to the ball-handler bends *that* possession). A
  predetermined box score can only fake them as aggregate fudge factors, which is
  exactly today's `player_stats_sim.py` limitation we are transcending.
- **Consistency is harder, not easier** — forcing events to reconcile exactly to a
  pre-drawn box score is brittle constrained generation. Bottom-up, the box score
  is just the sum of events.

**Architecture (hierarchical):**
1. **Macro layer** (cheap, uses ratings): expected pace → possession count; team
   strength → efficiency band. Keeps aggregates calibrated.
2. **Possession engine** (the build): generate events forward — initiation →
   matchup-resolved action → shot / TO / foul / rebound → transition — conditioned
   on on-court lineup, fatigue, score. The box score **emerges**.
3. **Calibration spine:** the shared generative **stat-line model** (Core step 2)
   is the target — the possession engine is tuned so its emergent per-player
   marginals match it.

**The unification:** ONE stat-line model, TWO uses.
- **Markets** *samples* it directly for prop distributions (no possessions; fast).
- **Game** uses it as the *target its possession engine reproduces*.
Same calibrated truth across both products — consistency for free. This is
"generate honestly, calibrated to a shared spine," not "predict then backfill."

---

## 11. Core API / artifact contract (the stable surface both tracks consume)

Core publishes **versioned artifacts**; sister repos pin a Core version and never
re-derive. Initial surface (extend `docs/multiple-versions.md` as these land):

| Artifact | Path | Produced by | Consumed by |
|---|---|---|---|
| Player impact (PTS v4.0) | `data/processed/bke/pts_v40.parquet` | build_pts_v40 | both |
| Off/Def archetypes | `data/processed/bke/{player_archetypes,defensive_archetypes_v2}.parquet` | archetype scripts | both |
| Team PTS ratings | `data/processed/forecast/team_pts_ratings.parquet` | build_pts_team_ratings | both |
| **Projected lineups** (starter/rotation/clutch) | `data/processed/simulation/simulation_step2_lineup_profiles.parquet` | lineup_projection | both (game uses full rotation; markets overrides starters w/ live actuals) |
| **Cross-team interaction model** (NEW) | TBD `data/processed/bke/cross_team_interactions.parquet` | NEW (step 1) | both |
| **Generative stat-line model** (NEW) | TBD `models/statline_model.*` | NEW (step 2) | both |

Contract rules: (1) schemas are versioned and additive; (2) any breaking change
bumps a Core version recorded in `docs/multiple-versions.md`; (3) sister repos read
artifacts, never import Core internals.
