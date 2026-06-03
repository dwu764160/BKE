# CLAUDE.md — BKE (Basketball KPI Engine)

Guidance for Claude Code sessions working in this repository.

## Narration & Output (persistent, system-level)

**Suppress intermediate step narration.** Do not announce what you are about to do
or pre-narrate tool use ("Now I'll search…", "Let me read…", "I'm going to run…").
Just do it. This applies to every session and every task.

Keep visible:
- **Decision logic** — the *why* behind non-obvious choices, trade-offs, and pivots.
- **Errors and surprises** — failures, unexpected results, and how you handled them.
- **Checkpoint summaries** — at natural breakpoints in long/multi-step work, a short
  recap of what was done and what's next.

Final output: use the **detailed-chat-output** skill structure. It may be slightly
richer than a bare summary, **as long as suppressing the play-by-play still nets a
meaningful token saving** over narrating every step. Optimize for signal per token,
not for a transcript of the process.

## Repo essentials

- Possession-level NBA player-impact pipeline: fetch → normalize → features →
  compute metrics/archetypes → BKE modeling → simulation/forecast → viewers.
- **`data/`, `reports/`, `models/`, `aggregate/` are gitignored.** A fresh
  clone has code + reference docs only. Most scripts (`validate_sim.py`, app
  viewers) will `FileNotFoundError` until the data pipeline is run or data is
  restored from a `data_backup_*` snapshot.
- Recorded validation numbers live in `reference/**/*.md` and `loop/*.txt`
  (dated, append-only narrative snapshots), not in tracked JSON.
- Do not edit `loop/*DO_NOT_CHANGE*.txt` checkbox formats — automation parses
  them.
- **`loop/in_progress_context.txt` is NOT append-only.** It holds ONE task at a time
  (current work description) to save tokens on context. Replace it entirely when
  starting a new task; do NOT append session entries. When a task is complete,
  replace with the next task description.
- **Multiple versions exist** for BKE scoring, archetype scripts, and data artifacts.
  Before touching any versioned file, read `docs/multiple-versions.md` for the canonical
  truth on which version is production.
- Tests: `pytest -q` (currently only data-free `tests/test_simulation_core.py`,
  3 checks).

## Cross-Repo References

### Sister Repo: Robinhood Trading Bot

This repo's simulation and prediction output feeds Sleeve C
(Prediction Markets) in the Robinhood trading bot's alpha thesis.

Robinhood alpha thesis:   docs/alpha-thesis.md in the Robinhood repo
BKE performance handoff:  docs/integration/for-alpha-thesis.md (this repo)

To fetch this repo's performance doc from a Robinhood CC session:

    gh api repos/Daniel-Wu-Github/BKE/contents/docs/integration/for-alpha-thesis.md?ref=personal \
      | python3 -c "import sys,json,base64; d=json.load(sys.stdin); \
        print(base64.b64decode(d['content']).decode())"

To fetch the Robinhood alpha thesis from a BKE CC session:

    gh api repos/Daniel-Wu-Github/robinhood/contents/docs/alpha-thesis.md?ref=main \
      | python3 -c "import sys,json,base64; d=json.load(sys.stdin); \
        print(base64.b64decode(d['content']).decode())"

### When to Regenerate docs/integration/for-alpha-thesis.md

Re-run the full investigation whenever:
  - A new BKE version is released (new bke_v* report appears in reports/)
  - validate_sim.py is re-run on new season data
  - The alpha thesis in the Robinhood repo updates Sleeve C parameters
  - Any loop/ file marks a major milestone complete

### Current Sleeve C status (as of 2026-05-19)

Per `docs/integration/for-alpha-thesis.md`: **NOT alpha-ready.** Headline Brier ≈ 0.21 is
in-sample/leaky (repo docs admit same-season leakage). Leakage-free forecast is
MAE ≈ 9 wins (~3-4 worse than Vegas) with **no game-level Brier** and **zero
market-price comparison**. Recommended live allocation: **$0 / research-only**
until a walk-forward game-level test vs. Kalshi closing lines shows positive
closing-line value.

### Downstream Sister Repos: BKE-Market & BKE-Game (the fork)

This repo is the **upstream engine** for two downstream tracks that fork off the Step-2.1
possession engine. They are local siblings under `~/projects/`:

- **`../BKE-Market`** — prices prediction-market totals/spreads/props. Consumes the emergent
  distributions; never runs the resolver.
- **`../BKE-Game`** — full event-level simulation via a `MarkovEventResolver` on the existing
  `PossessionResolver` seam.

Both consume **`data/processed/simulation/possession_box_distributions.parquet`** (the API)
and are bound by the parity contract `scripts/validate_possession_engine.py`. The connective
architecture lives in `docs/plans/fork_integration_architecture.md` (this repo) and is
mirrored as `docs/integration_with_bke.md` in each sister repo.

**What the sister repos depend on from here (do not break without updating them):**
- the `possession_box_distributions.parquet` schema (treat as a versioned API),
- the `PossessionResolver` seam + `PossessionOutcome` schema,
- the walk-forward validation gate and its oracle floors,
- the calibrated constants block in `src/simulation/possession_engine.py`.

When any of these change, update `docs/plans/fork_integration_architecture.md` and the
sister repos' `docs/ground_truths.md`. Each sister repo points back here via `../BKE-Copy`.

## Skill System

This repo has a comprehensive skill library: **portable governance skills**, **BKE domain skills**, and **context engineering skills**. The canonical single source of truth is `.github/skills/SKILL_MAP.md`.

### MANDATORY: Skill Auto-Loading at Task Start

**Every task MUST follow this sequence before editing or planning:**

> **Note:** Project skills live in `.github/skills/` as markdown files. The harness
> `Skill` tool does **not** support these — invoking them via `Skill` always errors.
> Load them with the `Read` tool instead.

1. **Load mandatory skills** by reading their files directly:
   - `Read .github/skills/scope-creep-guard/SKILL.md` — enforce phase boundaries before any edits
   - `Read .github/skills/detailed-chat-output/SKILL.md` — structured output for every task

2. **Determine task domain** by reading the user request and current work context (`loop/in_progress_context.txt`).

3. **Load applicable domain skills** by:
   - Consulting `.github/skills/SKILL_MAP.md` **Selection Order** (lines 11–60)
   - Reading each matching skill file (`Read .github/skills/<name>/SKILL.md`) before planning or editing

4. **Do not skip or rely on memory:** SKILL_MAP.md is the canonical source; it changes over time. Always consult it before treating a skill as deprecated or adding new trigger conditions.

**Example:** For a modeling change, read: `scope-creep-guard/SKILL.md` → `detailed-chat-output/SKILL.md` → `audit-model/SKILL.md` → then proceed.

For a complete reference of all skills and when to use them, see `.github/skills/SKILL_MAP.md` **Selection Order** and **Skill Registry**.

### Output Format — Mandatory

Every response MUST follow the **detailed-chat-output** structure:

1. **Outcome** — Lead with what was accomplished (1–2 sentences).
2. **Changes** — Concrete file edits and imports (file paths + line numbers).
3. **Verification** — Test results, passes, or "not verified" callout.
4. **Next Steps** — What comes next or what's pending.

Read `loop/in_progress_context.txt` at the start of each session for current task context.

### Handoff Prompts

When asked to produce a handoff prompt: **always display it in chat AND save it to `curr_handoff.md` at the repo root** (overwrite every time). Never save handoff prompts to `docs/` or any subdirectory — `docs/` is for reference material, not session prompts. `curr_handoff.md` is gitignored and ephemeral.

## Basketball Terminology — Decoding Statistical Jargon

This section translates technical metrics and modeling terms into basketball context for clarity in code comments and documentation.

### Model & Feature Terms

| Statistical Term | Basketball Analogy | Explanation |
|---|---|---|
| **Brier Score** | "Calibration accuracy" | How well prediction confidence matches reality. A team predicted to win 70% of the time should actually win ~70% of games. Lower = better. Vegas target: 0.195–0.210. |
| **Net Rating** | "Plus-minus per 100 possessions" | If Team A outscores Team B by 5 points per every 100 possessions played, net rating = +5. The single best predictor of team strength. |
| **YTD (Year-to-Date) Blending** | "Shift from preseason to actual record" | Early season: trust preseason projections (teams haven't played enough games). By game 30: gradually believe what the team actually shows. By season end: 100% actual performance. |
| **OOS (Out-of-Sample)** | "Future prediction validity" | Train on seasons 2020–2022, predict season 2023. This tests whether the model works on data it's never seen (not backfit). Critical for proving genuine edge vs. Vegas. |
| **Leakage** | "Peeking at the answer" | Using information from the current game to predict the current game (e.g., using Q4 stats to predict final outcome). Inflates validation scores; defeats live prediction. |
| **Walk-Forward CV** | "Season-by-season test" | For each season, train on all past seasons, predict that season. Repeats across all seasons. Proves the model works on every new data regime. |
| **Closing-Line Value (CLV)** | "Beat the Vegas closing odds" | On average, were BKE predictions better-calibrated than the market at close? CLV > 0.01 = model has real alpha (1% edge). CLV ≤ 0 = losing strategy. |
| **Back-to-Back (B2B)** | "Fatigue penalty" | Team played yesterday, playing today. Road B2B is worse (travel + fatigue). Reduces predicted net rating ~2.5 pts/100. Data-backed at 57% ATS loss rate. |
| **Season Stage Alpha** | "Ramp weight through season" | `alpha = sqrt(game_number / 82)`. At game 1, alpha ≈ 0.11 (preseason dominates). At game 30, alpha ≈ 0.60 (half preseason, half actual). Smooth transition, not step-function. |

### Model Architecture Terms

| Technical Term | Basketball Analogy | Explanation |
|---|---|---|
| **Improvement A / Improvement C** | "Two different coaching adjustments" | Two separate enhancements to the model tested in isolation. Blend them with a weight: Improvement A at 50%, Improvement C at 50%. |
| **Composite v4.0 (A + C blend)** | "Hybrid starting lineup" | Don't pick one enhancement; run both. Weight them: w_c=0.50 means equal say. Locked in commit e32bad8 after testing w_c ∈ {0.3–0.7}. |
| **Joint Correlation (Joint_r)** | "How well independent stats align" | Two independent data sources (e.g., preseason projections vs. actual net rating correlation). If Joint_r ≥ 0.331, the two signals agree. |
| **YoY Correlation (YoY_r)** | "Year-over-year consistency" | Does the model rank teams the same way across seasons? Curry should be #1 both seasons. If YoY_r < 0.766, the model is season-dependent (unstable). |
| **GBDT (Gradient Boosting Decision Tree)** | "Machine-learning coach adjustments" | Instead of using a single formula (Gaussian), train a tree-based learner on historical games. Learns non-linear patterns: ultra-lopsided matchups are over-predicted; B2B teams underperform formula. |
| **Gaussian Baseline** | "Simple mathematical curve" | `win_prob = norm.cdf(delta_mu / sigma)`. Uses only team strength delta and home-court advantage. Symmetric, stable, but misses non-linear effects. |
| **Elo (in ensemble)** | "Momentum indicator" | Running rating that updates after each game. Sharp start = higher Elo; cold spell = lower. Captures short-term form the other models miss. |
| **Ensemble** | "Three scouts voting" | Don't trust one model; average 3: Gaussian (30%), GBDT (50%), Elo (20%). Reduces systematic bias by ~0.003 Brier. |

### Validation & Readiness Terms

| Term | Basketball Analogy | Explanation |
|---|---|---|
| **Validation Gate** | "Threshold the team must cross" | Brier ≤ 0.235, YoY_r ≥ 0.766, star sanity (Curry ≤ #30). Miss any gate = model isn't ready. |
| **Star Sanity** | "Sense check: elite players rank highly" | Curry #7, KD #8, SGA #2, Giannis #8. If Luka is #412 (he's not in dataset), that's a data problem, not a model problem. |
| **Alpha-Ready** | "Ready for real money" | CLV > 0.02 **and** Brier ≤ 0.210. Below that: research-only, no allocation. |

### Matchup Projection System (Step 1)

The cross-team interaction model assigns five offensive players to five defenders
using a three-tier system. All cell lookups use argmax archetype; position band is
used for assignment and as fallback for Unknown-archetype players. Talent rank
within a band uses `pts_o_v40` (PTS 4.0 offensive score); defensive quality uses
`pts_d_v40`. Terminology: **smalls** = guards (`position_band_3 = 'smalls'`),
**wings** = forwards, **bigs** = centers.

| Term | Basketball Analogy | Explanation |
|---|---|---|
| **Pair 1 — Primary Perimeter Threat** | "Ace vs. ace stopper" | The highest-`pts_o_v40` smalls/wings creator (Ball Dominant Creator, All-Around Scorer, or Ballhandler archetype) matched against the defense's best POA Defender or Wing Stopper. Most empirically stable assignment — 166k+ possession sample, era-stable across 8 seasons. Cell lookup: argmax archetype. |
| **Pair 2 — Interior Threat + Big Behavior** | "Who handles the big man?" | Sub-case A: if defense has a big-bodied forward (wings-band, `position_proxy` = Forward/Forward-Center, `pts_d_v40 > 0.0`, top-3 on team), that player guards the interior threat; center stays as rim anchor. Sub-case B: no qualifying defender → center is **overloaded** — guards interior threat AND protects the rim. Rim anchor's paint-presence effect is always modeled as a lineup composition term regardless of sub-case. |
| **Pairs 3-5 — Band + Talent Residual** | "Everyone else, size then talent" | Remaining players ranked by `pts_o_v40` within position band (smalls/wings/bigs), matched rank-to-rank within the same band. **Projection assumes no mismatch** — coaches switch back and rotate to avoid sustained cross-band assignments. Cross-band forced only when no same-band defender exists; discounted penalty (~50% of empirical magnitude) applied. |
| **Band Mismatch Penalty** | "Size mismatch tax" | Empirical: +4.0 pts/100 FE-residual for smalls↔bigs forced mismatch (archetype cells explain only 31% of this). **Projection applies ~50% discount** (→ +2.0 pts/100) to account for coaching adjustment. Full magnitude used in the possession engine where real-time mismatch hunting is simulated. |
| **Center Overloaded flag** | "One man, two jobs" | Fires in Pair 2 Sub-case B: no qualifying big-bodied defender, center must guard the interior threat and anchor the paint simultaneously. Signals that roll men and big-attackers on the opposing team will get cleaner rim looks — center cannot do both jobs at full effectiveness. |

---

## Improvement Loop

**Write a Debug Entry when you make a real mistake; that is the entire improvement
loop.** Append a `### Debug Entry` block to `.claude/debugging_log.md` (what broke,
root cause, fix, and — optionally — a `**Skill gap:**` line). That is the only
required step. Everything downstream is automatic and lossless: the session-end
hook reads new Debug Entries into persistent memory (`memory/skill_effectiveness.md`,
`memory/debugging_patterns.md`). No pattern scanners, no smoke tests, no
pending-improvements queue — those were removed (2026-05-30) as token-wasting noise.

### Automatic Verification (Stop Hook)

`scripts/session-end.sh` runs at the end of every session and does exactly four
things: (1) runs the verification check (`pytest -q tests/`), (2) appends a session
boundary to `.claude/debugging_log.md`, (3) runs `scripts/update-skill-memory.sh`
(new Debug Entries → persistent memory), (4) sends a push notification.

### Git Hooks Setup (one-time per local clone)

```bash
git config core.hooksPath .githooks
```

This wires the `pre-commit` (secret scan) and `pre-push` (commit log) hooks.

### Push Notifications

`scripts/notify.sh` sends ntfy.sh push notifications. Set `NTFY_CHANNEL_URL` env var to your topic to activate. Leave unset if not needed — scripts fail silently.
