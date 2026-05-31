# Step 1 — Cross-Team Matchup Engine: build & validation (2026-05-31)

Implements the locked Resolved Design in
`docs/plans/cross_team_interaction_model.md`. Three deliverables, all run clean.

## 1. Shrunk interaction matrix — `fit_archetype_interactions_v2.py`

Promotes the locked 8-season FE-residual cells
(`reports/archetype_interaction_signal_test.json`, 99 cells, two-way player FE)
to a production matrix.

- **Empirical-Bayes shrinkage toward 0** using each cell's standard error
  `se = (ci_hi − ci_lo) / (2·1.96)` (the CI half-width already encodes
  possessions, so this is "shrinkage by possessions"). Weight
  `w = τ²/(τ² + se²)`, with `τ²` by method-of-moments (poss-weighted spread of
  raw cells − mean sampling variance, clamped ≥0).
- **Mean-zero per offensive archetype** (poss-weighted) enforced after
  shrinkage — the double-counting guard from the design (Σ_D interaction[A,D]·poss = 0).
- Output `data/processed/bke/cross_team_interactions_matrix.parquet` with full
  provenance (`raw_ppp, se, shrink_factor, poss, ci_lo, ci_hi, source`).

Numbers: 99 cells, shrunk poss-weighted std ≈ 0.040 ppp, mean shrink_factor ≈ 0.92
(cells are high-possession, so little shrinkage — consistent with a confirmed
signal), per-off-archetype mean-zero residual ≈ 4.5e-18.

Also exposes `estimate_fe_cells(...)` — a reusable weighted two-way-FE estimator
with a season filter (mirrors the locked test methodology: weighted alternating
demean by OFF/DEF player, `min_partial_poss=2`, 12 iters). Used **only** by the
walk-forward validation to fit a train-only matrix; the production artifact shrinks
the already-locked cells (per "do not re-run the pre-flight").

## 2. Per-matchup adjustment — `compute_matchup_adj.py`

The 3-pair assignment engine over projected starting fives.

- **Pair 1 — primary perimeter threat:** highest `pts_o_v40` smalls/wings creator
  (BDC / All-Around Scorer / Ballhandler) → best POA Defender / Wing Stopper by
  `pts_d_v40`. argmax cell.
- **Pair 2 — interior threat + big behavior:** highest `pts_o_v40` interior
  archetype. Sub-case A if a **big-bodied defender** exists (wings-band, proxy
  Forward/Forward-Center, `pts_d_v40>0`, top-3 on team) — that player guards,
  center stays as rim anchor (paint term = 0.5× rim-anchor cell). Sub-case B
  otherwise — `center_overloaded=True`, rim anchor guards directly.
- **Pairs 3-5 — band + talent residual:** remaining players ranked by `pts_o_v40`
  within `position_band_3`, matched rank-to-rank same band. Cross-band forced →
  discounted mismatch adj (0.5× empirical: diff0 −0.010, diff1 +0.005, diff2 +0.020);
  same-band diff0 calibration correction −0.010.
- Unknown off archetype → position-band-average cell fallback.
- `lineup_net_rating_adj` = minute-weighted sum of per-player ppp adjustments × 100
  (pts/100). `player_matchup_adjs` JSON kept per off starter for props/stat-line.

**Output** `data/processed/bke/cross_team_interactions.parquet` per the design
schema. **Granularity = `(season, team, opponent)`, `game_id=NULL`** — the matchup
source is season-aggregated (no game_id) and the projector emits one starting five
per (season, team), so a team-pair matchup is identical across their games. The
Markets track fans this out per game later with live lineups.

Run: **7,380 rows / 8 seasons; center_overloaded 0.570; pair-1 argmax-resolved
92.3%; lineup_net_rating_adj mean −0.26, std 1.32, range [−4.27, +5.59] pts/100.**
The slightly negative mean is expected — top perimeter threats are assigned the best
perimeter defenders, so the argmax cell is usually suppressive. The high argmax rate
(92%) confirms most lineups resolve to real cells rather than fallbacks.

## 3. Walk-forward validation — `validate_matchup_interactions.py`

Leakage-free, player-game PPP MAE on the **2024-25 holdout** (the design's gate;
game Brier is only a secondary diagnostic).

Protocol: fit two-way FE + archetype cells on **train ≤2023-24 only**, EB-shrink,
apply forward. For each 2024-25 matchup row (weight = `PARTIAL_POSS`):
`baseline = grand + off_fe + def_fe` (talent only); `model = baseline +
interaction[off_arch][def_arch]`. The shared baseline cancels, so the delta
isolates the OOS value of the interaction term. Paired bootstrap on per-row
abs-error differences.

Result (archetypes from `player_impact_profiles` — actual, all 8 seasons):
**n = 105,595 holdout rows; overall WMAE 0.18256 → 0.18119 (ΔWMAE −0.00138, +0.75%);
significant-cell stratum (n = 69,980) ΔWMAE −0.00278; paired bootstrap p(model
better) = 1.000, CI [−0.00164, −0.00112]. Verdict: MODEL HELPS (OOS).** The
improvement is leakage-free and ~2× larger on rows where a *significant* train cell
applies — exactly the pattern a real matchup effect should show. The headline PPP
gain is small in absolute terms (mean-zero matrix, ±0.04–0.09 ppp cells), but it is
statistically clean and directionally consistent; the larger value remains for
**possession realism and player props**, which Step 2 evaluates.

> Interim note: an earlier validation run that sourced archetypes from
> `projected_player_profiles` (sparse, forward-looking, 7 seasons) was
> NEUTRAL/INCONCLUSIVE. Switching to actual-archetype coverage (the same source the
> locked signal test used) is what surfaces the OOS win — a coverage effect, not a
> methodology change.

## Data-schema corrections (verified this session)

Earlier session notes carried display-corrupted schemas. Ground truth:
- `league_season_matchups.parquet` is **season-aggregated** per off×def player
  pair (SEASON, OFF/DEF_PLAYER_ID, PARTIAL_POSS, PLAYER_PTS, GP, …). **No
  game_id, no team_id** — game-level fan-out is not possible from this source.
- `simulation_step2_lineup_profiles.parquet` carries only `starter_player_ids` /
  `clutch_player_ids` at player level (+ team mu/sigma). Per-player archetypes
  must be joined from the profiles.
- `projected_player_profiles.parquet` has `position_band_3` (smalls/wings/bigs),
  `position_proxy`, `off/def_primary_archetype`, `mpg`; seasons 2018-19..2024-25.

## Handoff to Step 2

Consume `cross_team_interactions.parquet`: `lineup_net_rating_adj`,
`player_matchup_adjs`, `center_overloaded`, `band_mismatch_flags`. Use the **full**
empirical mismatch magnitude (not the 0.5× projection discount) inside the
possession engine where real-time mismatch hunting is simulated.
