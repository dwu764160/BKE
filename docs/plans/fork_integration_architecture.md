# BKE Fork — Integration Architecture (Markets ⟷ Game)

> **Purpose.** This is the connective-tissue document for the post–Step-2.1 fork. It is
> **not** a task list. It explains how the two downstream tracks — **Markets** (prediction
> markets / props pricing) and **Game** (full event-level simulation) — hang off the same
> calibrated possession-engine spine, what each track adds, what they *share*, and where
> the deferred pieces (in-season YTD blending, Elo, a player rating + tendency model) plug
> in. Read `docs/findings/possession_engine_step2_1_2026-06-02.md` first for the engine
> state; read `docs/plans/generative_possession_engine.md` for the locked Step-2 design.

---

## 1. The shared spine (locked — do not fork it)

Everything below the fork is one artifact and one seam:

```
projected_player_profiles ─┐
pts_v40 (talent) ──────────┤
cross_team_interactions ───┼──▶ RateModel ──▶ OutcomeSamplerResolver ──▶ simulate_game
9-man v1 lineups ──────────┘        │              (PossessionResolver seam)      │
                                     │                                            ▼
                          GlobalConstants (walk-forward                possession_box_distributions.parquet
                          calibrated, train < target)                  (per-game player + team distributions)
```

- **The seam.** `PossessionResolver.resolve(off, defense, state, rng) -> PossessionOutcome`
  is the *only* coupling point. `PossessionLoop` / `BoxScoreAggregator` / the calibration
  harness are agnostic to *how* an outcome is produced. Markets consumes the emergent
  distributions and **never touches the resolver**; Game swaps a richer resolver onto this
  exact interface with **zero change** to the loop, aggregator, or calibration.
- **`PossessionOutcome` already carries the rich schema** (scorer/assist/tov/reb/foul ids +
  possession_seconds) so the Game track's event resolver needs no schema migration.
- **The data contract** both tracks read is `possession_box_distributions.parquet`:
  one row per (game_id, season, team, side∈{off,def}, player_id) with
  `pts_mean/p10/p50/p90/std`, `ast_mean/p10/p50/p90`, `fg*_mean`, `ft*_mean`, `oreb_mean`,
  plus `player_id="TEAM"` rows for team totals, and `def` rows carrying `dreb/stl/blk/pf`.
  **This schema is the API.** Treat additions as backward-compatible; never repurpose a column.

What is **locked** after Step 2.1: the multiplicative `RateModel` structure, the band+mpg
rebound weights, the FTR→trip conversion, the assist/usage shape exponents, the
minutes-dispersion variance model, the 9-man v1 rotation, and walk-forward constant
calibration. Two independent holdouts (2023-24 train ≤22-23, 2024-25 train ≤23-24) both
PASS with near-identical player MAEs — the spine generalizes and is not overfit.

---

## 2. Where the fork happens, and why here

Step 2 / 2.1 was deliberately the **last shared step** because both tracks need the same
thing — a calibrated mapping from (players, matchup) to a distribution of box-score
outcomes — but they consume it at **different resolutions**:

| | Markets | Game |
|---|---|---|
| Question | "What is the *distribution* of this line?" | "How does this game *unfold*, event by event?" |
| Resolver | none — reads emergent distributions | `MarkovEventResolver` on the seam |
| Resolution | game-aggregated player/team distributions | possession/event sequence |
| Latency | batch (pre-tip) + live override | interactive / on-demand |
| Output | prices for totals, spreads, props | play-by-play, win prob trajectory, box score |

Forking here means the expensive, error-prone part (getting the box-score *shape* right) is
solved once and shared. Both tracks inherit the same calibrated shot mix, rebound hierarchy,
assist flow, and tails.

---

## 3. Markets track — how it integrates

**Reads:** `possession_box_distributions.parquet`. **Never runs the resolver.**

```
possession_box_distributions ─▶ price totals (TEAM rows) ─▶ price spreads (team diff)
                              └▶ price player props (player rows, full P10/P50/P90 dist)
                                          ▲
                       live-lineup override at T-60 / T-30 (actives, minutes, scratches)
                                          ▲
                       in-season YTD blend of player rates (DEFERRED — see §5)
```

- **Pricing order (by readiness, strongest first):** totals → spreads → player points props →
  rebound/assist props. Totals/spreads ride the well-calibrated team distribution (bias
  ~+1.5–2.4, coverage ~0.91–0.93 across both holdouts). Player props ride the per-player
  distributions, which are now genuinely calibrated (per-player season-mean corr PTS 0.88 /
  REB 0.81 / AST 0.88; tails match real NBA, P90/mean ≈ real 1.55 median).
- **The live-lineup override is the load-bearing Markets-specific layer.** The engine's
  prop surface is the **9-man *projected* rotation**. Before tip, replace projected actives
  with the announced lineup: scratch DNPs (their distribution → 0), reallocate freed
  minutes/usage to the actual actives (re-normalize the lineup's minutes to 240, re-run the
  cheap batch sampler for that one game, or apply the minutes-share delta analytically).
  This is what converts a season-projection into a game-specific price and is the single
  biggest accuracy lever for same-day props.
- **Why props are research-only until a CLV gate passes:** established-star points show a
  residual over-projection (high-usage scorers, partly role-change driven) and the mid-tier
  shows breakout-driven under-projection. Both are *forecast-layer* effects, not engine
  defects (proven: established-Superstar per-player bias +0.24, exact shot mix). They wash
  out at the team level but matter at the prop level. **Gate:** walk-forward, game-level test
  of prop prices vs Kalshi/Polymarket closing lines showing CLV > 0.01 before any allocation.
- **Sister-repo handoff:** Markets output is the *content* of Sleeve C. The performance
  contract lives in `docs/integration/for-alpha-thesis.md` (consumed by the Robinhood repo).
  Update it whenever a new validation run lands (see §6).

---

## 4. Game track — how it integrates, and what it still needs

**Implements:** a `MarkovEventResolver(PossessionResolver)` dropped onto the seam. The loop,
aggregator, and the entire Step-2.1 calibration come for free; consistency with Markets is
guaranteed by construction because both resolvers share the same `RateModel` primitives.

The Game track is *event-level* where the Sampler is *outcome-level*. To simulate a possession
as a sequence (bring-up → action → help → shot/pass/turnover → crash/transition) rather than
sampling a single terminal outcome, it needs two player-level models the Sampler approximates
implicitly but does not represent explicitly:

1. **Player rating (skill/efficiency).** Largely *already present* and reusable: `pts_v40`
   (offensive/defensive talent), the `behavioral_*` rates, and the Step-1 matchup adjustment.
   The Markov resolver consumes these as the conditional *quality* of each event (make
   probability, finish efficiency, defensive resistance). No new fetch needed for v1 — this
   is a re-projection of existing artifacts onto event conditionals.

2. **Player tendency (behavioral distribution).** This is the genuine *new* requirement. The
   Sampler bakes tendency into archetype shape multipliers + aggregate rates; an event sim
   needs an explicit, conditional tendency model: shot-location/zone propensity, shot-type
   (rim / mid / 3 / FT-draw), pass-vs-shoot on the catch, PnR ball-handler vs roll-man split,
   transition vs half-court. **Seed data exists** — `fetch_shot_zones`, the play-type inputs
   already feeding `compute_player_archetypes`, and the archetype assignments — so v1 tendency
   is an *aggregation/derivation* task on existing data, not a new ingest. The archetype
   system is effectively a coarse tendency prior already; the Game track refines it to
   conditional, state-dependent distributions.

**Integration principle:** the Markov resolver must reproduce the Sampler's *marginals* (it
must pass the same Step-2.1 validation gate) before adding event-level richness — otherwise
the two tracks diverge and Markets/Game prices disagree. The validator is the shared contract:
run `validate_possession_engine.py` against the Markov resolver's emergent distributions and
require parity with the Sampler baseline before layering tendencies.

---

## 5. Shared deferred dependencies (both tracks want these)

These were correctly out of scope for engine calibration but are the next force multipliers,
and they belong to **both** tracks:

- **In-season YTD blending of player rates (the big one).** The engine today applies a single
  *static, preseason-style* projection across all 82 games — it cannot know that a player
  broke out. This is the root cause of the mid-tier under-bias and a chunk of prop error.
  The fix mirrors the team-level YTD machinery (`build_ytd_team_ratings.py`,
  `alpha = sqrt(game_no/82)`): blend the preseason projection toward season-to-date player
  rates as games accrue, **strictly `shift(1)` (prior games only)** to stay leakage-free.
  - *Markets:* this is what makes mid-season props sharp; pairs naturally with the live-lineup
    override (one is rate-level, the other is availability-level).
  - *Game:* the Markov resolver reads the same blended rates, so in-season form propagates to
    event conditionals automatically.
- **Player rating + tendency model (Game-critical, Markets-useful).** See §4. A shared
  `player_event_profiles` artifact (rating + conditional tendencies) would serve the Markov
  resolver directly and let Markets price *prop types the Sampler can't* (e.g. made-threes,
  FT attempts) from explicit tendencies rather than aggregate rates.
- **Elo / momentum.** Elo lives in the **team win-forecast ensemble** (`gbdt_game_model.py`:
  Gaussian 30% / GBDT 50% / Elo 20%), which prices *moneylines* — orthogonal to the possession
  engine. It does **not** feed box-score generation and should not be wired into the resolver.
  It connects to Markets only as the moneyline pricer alongside the engine's totals/spreads,
  and to the sister repo as the existing Sleeve C win-probability signal.

**Dependency ordering that keeps both tracks honest:** YTD blending is the highest-leverage,
lowest-architecture-risk add and benefits both tracks identically — do it on the shared spine
**before** either track hardens. The tendency model is Game-gated and can proceed in parallel.

---

## 6. File / artifact map (the contract surface)

| Concern | Path | Owner | Notes |
|---|---|---|---|
| Engine spine | `src/simulation/possession_engine.py` | shared | locked post-2.1; the seam |
| Runner / calibration | `scripts/run_possession_engine.py` | shared | walk-forward `walk_forward_train(target)` |
| Validation gate | `scripts/validate_possession_engine.py` | shared | `--season`; per-tier + tail; the parity contract |
| Emergent distributions | `data/processed/simulation/possession_box_distributions.parquet` | shared | **the API both tracks read** |
| Per-season gate JSON | `reports/possession_engine_validation_{season}.json` | shared | walk-forward evidence |
| Team win-forecast (Elo/YTD/ensemble) | `src/simulation/gbdt_game_model.py`, `game_model.py`, `build_ytd_team_ratings.py` | Markets-moneyline | orthogonal to the resolver |
| Tendency seed data | `fetch_shot_zones`, archetype/play-type inputs | Game | derive `player_event_profiles` from these |
| Sister-repo handoff | `docs/integration/for-alpha-thesis.md` | Markets→Robinhood | update on each validation run |

---

## 7. One-paragraph summary for whoever picks this up

The possession engine is the calibrated, leakage-free, multi-season-validated spine that both
downstream tracks share through a single parquet contract and a single resolver seam. **Markets**
reads the emergent box-score distributions and prices totals → spreads → props, with a
live-lineup override at tip and an eventual game-level CLV gate before real money. **Game**
swaps a `MarkovEventResolver` onto the seam and must first reproduce the Sampler's validated
marginals, then layer an explicit player rating + tendency model (seed data already in the repo)
to go event-level. The two highest-leverage shared adds are **in-season YTD blending of player
rates** (kills most of the residual prop/mid-tier error, benefits both tracks, must stay
`shift(1)` leakage-free) and the **player tendency model** (Game-critical). **Elo** stays where
it is — the team moneyline ensemble — and does not touch the resolver. Do the shared work on the
spine before hardening either track, and keep the validator as the parity contract between them.
