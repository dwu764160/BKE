# Multiple Versions Index

> **Purpose:** Single reference for which version of every versioned artifact is canonical.
> Claude Code sessions should read this before touching any script or data file that
> has a version number in its name or header.
>
> **Update this file** whenever a new version is promoted to production, deprecated, or
> experimental status changes.

---

## PTS Scoring Model (Player True Skill)

PTS is the player-level offensive/defensive impact score. It feeds into `projected_team_features`
(via `team_net_rating_projected`), `player_profile_aggregate`, and the BKE layer inputs.

| Version | Script | Output artifact | Status |
|---|---|---|---|
| v3.2 | `scripts/build_pts_v32.py` | `data/processed/bke/pts_v32.parquet` | **BASELINE / DEPRECATED** — cols `pts_o_v32, pts_d_v32`. Retained as a sweep baseline; not used in production pipeline. |
| v4.0-A | `scripts/build_pts_v40.py --pts-a` | `data/processed/bke/pts_v40_a.parquet` | **COMPONENT** — Improvement A defense. Input to composite. |
| v4.0-C | `scripts/build_pts_v40.py --pts-c` | `data/processed/bke/pts_v40_c.parquet` | **COMPONENT** — Improvement C defense. Input to composite. |
| **v4.0** | `scripts/build_pts_v40.py` | `data/processed/bke/pts_v40.parquet` | **CURRENT PRODUCTION** — 50% A + 50% C composite. Cols: `pts_o_v40`, `pts_d_v40`. Locked at `PtsV40CompositeConfig(defense_v40c_weight=0.50)` in `src/modeling/model_config.py`. |

**Canonical production columns used downstream:** `pts_o_v40`, `pts_d_v40` (in `pts_v40.parquet`)

**Season coverage:** 2017-26 (9 seasons after 2025-26 RAPM completes)

**v3.2 retention rule:** `pts_v32.parquet` and `pts_o_v32 / pts_d_v32` columns are kept as
baseline comparison inputs for sweep scripts (`pts_v40_multiseason.py`, `pts_v40_defense.py`,
`validate_lineup_pts_v2.py`). These are dev-only. **No production pipeline step should read
`pts_v32` directly.** If you see a production script importing `pts_v32`, that is a version error.

**Scale note:** PTS scores are in compressed BKE units (std ~0.3 per player). Team-level
`team_net_rating_projected` is in pts/100 possessions (std ~0.6-0.8), computed from actual
game margins by `scripts/build_all_season_projections.py` — not directly from PTS player scores.

---

## Team Projection Artifacts

| Version | Script | Output artifact | Status |
|---|---|---|---|
| legacy (mixed scale) | `src/player_eval/team_feature_aggregation.py` | *(historical)* | **DEPRECATED** — produced compressed BKE units for 2018-22, pts/100 for 2023-25. Do not use to regenerate. |
| **current** | `scripts/build_all_season_projections.py` | `data/processed/forecast/projected_team_features.parquet` | **CURRENT PRODUCTION** — all 8 seasons in consistent pts/100 units from actual game margins. |
| 2025-26 supplement | `scripts/build_2025_26_projections.py` | `data/processed/forecast/projected_team_features_v40_2025-26.parquet` | Separate file for current season; read by `build_ytd_team_ratings.py` for 2025-26. |

**Pipeline order:** `build_profile_aggregate` → `build_all_season_projections` → `build_rest_features` → `build_ytd_team_ratings`

---

## Game-Level Forecast Models

Walk-forward, out-of-sample (train seasons < T, predict T). Aggregate Brier over
2019-20…2025-26 (8,279 games) from `reports/gbdt_forecast_validation.json`.

| Model | Script / entry | Output | Agg Brier | 2025-26 Brier | Status |
|---|---|---|---|---|---|
| Gaussian baseline | `src/simulation/validate_forecast.py` (`compute_game_distribution_with_context`) | `reports/bke_game_forecasts.parquet` | 0.2408 | 0.2358 | **PRODUCTION BASELINE** — underconfident (gauss_p std ≈ 0.083) |
| GBDT stack (Step 5) | `src/simulation/gbdt_game_model.py` | `reports/bke_gbdt_game_forecasts.parquet` | 0.2250 | 0.2133 | **NEW** — LightGBM stacking Gaussian+Elo+rest; −0.0158 vs Gaussian |
| Elo (sequential) | `src/simulation/gbdt_game_model.py` (`compute_walk_forward_elo`) | *(in JSON report)* | **0.2193** | **0.2092** | **BEST SINGLE MODEL** — beats the BKE player-impact pipeline |

**Key finding (2026-05-30):** a zero-cost sequential Elo (K=20, HCA≈55 Elo pts,
0.75 between-season carry) is the strongest game-level model and the only leg
that crosses the Brier ≤ 0.210 gate on 2025-26 (0.2092, near Kalshi's 0.2045).
The GBDT stack helps but cannot beat pure Elo on limited data. Normalization
used: per-season z-score of `delta_mu` (leakage-free) to absorb cross-era scale
drift in `blended_mu`. **Alpha gate (CLV > 0.02 AND Brier ≤ 0.210) still NOT
met** — Elo is near Brier-parity with Kalshi, not ahead, so expected CLV ≈ 0.
Next: run `scripts/fetch_kalshi_closing_lines.py` CLV test against the Elo/GBDT
2025-26 export.

**⚠ Source note (2026-05-30):** the Gaussian/GBDT "team rating"
(`team_net_rating_projected`) is `0.70 × prior-season PLUS_MINUS margin`
(`build_all_season_projections.py`, `build_2025_26_projections.py`) — it contains
**no PTS and no BKE**. The archetype/talent columns in `projected_team_features`
are unused template fields. Wiring PTS/BKE player-impact into team strength is an
open task (Track 2 in `docs/findings/game_model_comparison_2026-05-30.md`).

**Run:** `python3 src/simulation/gbdt_game_model.py`

---

## BKE Scoring Model

| Version | Script | Output artifact | Status |
|---|---|---|---|
| v2.7 | `src/modeling/construct_bke_scores_v27.py` | `data/processed/bke/BKE_Scores_v27.json` | **PRODUCTION OUTPUT** (JSON for viewers) |
| v2.8 | `src/modeling/decomposition_engine.py` | `data/processed/bke/bke_v28_decomposition.parquet` | **PRODUCTION DECOMPOSITION** — canonical pipeline |
| v3.0 | `src/modeling/dbke_v30_defense_shrinkage.py` | `data/processed/bke/bke_v30_decomposition.parquet` | **FAILED** — Phase C broke on 8-season data; do not use (GAP-008 in `docs/stats-gap.md`) |
| v3.1 | `src/modeling/bke_v31_experimental_layers.py` | `data/processed/bke/bke_v31_components.json` | **EXPERIMENTAL ONLY** — does not overwrite production; interactive viewer only |

**Canonical production pipeline:**
```
decomposition_engine.py (v2.8)  →  bke_v28_decomposition.parquet
construct_bke_scores_v27.py     →  BKE_Scores_v27.json   (viewer input)
```

**Decomposition parquet history** (all in `data/processed/bke/`):
`bke_v15`, `bke_v20`, `bke_v25`, `bke_v26` — legacy, superseded.
`bke_v27` — superseded by v28.
`bke_v28` — **CURRENT PRODUCTION**.
`bke_v30` — failed experiment.

---

## BKE Model Config

| File | Version tag | Status |
|---|---|---|
| `src/modeling/model_config.py` | `BKE v2.7` in header | **CURRENT** — active config for production pipeline |

The config version number (v2.7) refers to the scoring schema version it was written for,
not the decomposition version. v2.8 decomposition still reads from this config.

---

## Defensive Archetype Classification

| File | Internal version | Status |
|---|---|---|
| `src/data_compute/compute_defensive_archetypes.py` | v1 | **DEPRECATED** — header explicitly marks it "NOT IN USE, REPLACED BY v2" |
| `src/data_compute/compute_defensive_archetypes_v2.py` | v3.4 | **CURRENT** — all sessions use this |

**Canonical archetype count: 9**
POA Defender · Wing Stopper · Versatile Defender · Off-Ball Chaser ·
Rotational Defender · Rim Protector · Dropping Big · Mobile Big · Low-Activity Defender

The `current_phase_plan.DO_NOT_CHANGE.txt` Stream C note that says "Reduced from 9 to 5"
describes an intermediate revision that was subsequently rolled back. The current code (v3.4)
implements all 9. `docs/reference/basketball-intuitions.md` §4 is the canonical truth.

**Output:** `data/processed/bke/defensive_archetypes_v2.parquet`

---

## Offensive Archetype Classification

| File | Internal version | Status |
|---|---|---|
| `src/data_compute/compute_player_archetypes.py` | v4.3 | **CURRENT** — only one script, no deprecated copy |

**Canonical archetype count: 11**
Ball Dominant Creator · Ballhandler · All-Around Scorer · Interior Scorer ·
Perimeter Scorer · PnR Rolling Big · PnR Popping Big · Off-Ball Finisher ·
Off-Ball Movement Shooter · Off-Ball Stationary Shooter · Connector

"Rotation Piece" referenced in older code comments — **eliminated in v4.3**, replaced by
best-fit fallback. Any player who clears 200 min threshold now falls into one of the 11.

**Output:** `data/processed/bke/player_archetypes_v4.parquet` (or equivalent)

---

## Layer / Dimension Score Artifacts

| Artifact | Version | Status |
|---|---|---|
| `data/processed/bke/dimension_scores_v28.json` | v2.8 | **CURRENT** |
| `data/processed/bke/layer_scores_v28.json` | v2.8 | **CURRENT** |
| `data/processed/bke/obke_dbke_scores_v28.json` | v2.8 | **CURRENT** |
| `data/processed/bke/BKE_Scores_v27.json` | v2.7 | Production viewer input (downstream of v2.8 decomp) |
| `data/processed/bke/BKE_Scores_v31_*.json` | v3.1 | Experimental only |

---

## Cross-Team Interaction Matrix (Simulation Core — Step 1)

The matchup engine resolves attacker-vs-defender archetype matchups between two
projected lineups. Built 2026-05-31.

| Artifact | Script | Output | Status |
|---|---|---|---|
| Signal cells (raw FE) | `scripts/test_archetype_interactions.py` | `reports/archetype_interaction_signal_test.json` | **PRE-FLIGHT (LOCKED)** — two-way player-FE residual cells, 8 seasons (2017-18…2024-25), 99 cells, 71 sig @95% / 68 BH-FDR. Do not re-run. |
| **Shrunk matrix** | `scripts/fit_archetype_interactions_v2.py` | `data/processed/bke/cross_team_interactions_matrix.parquet` | **CURRENT PRODUCTION** — EB-shrunk (per-cell SE) + mean-zero per off archetype. Cols: `off_arch, def_arch, interaction_ppp` (shrunk), `raw_ppp, se, shrink_factor, poss, ci_lo, ci_hi, source`. |
| **Per-matchup adj** | `scripts/compute_matchup_adj.py` | `data/processed/bke/cross_team_interactions.parquet` | **CURRENT PRODUCTION** — 3-pair engine over projected starters. Granularity = `(season, team, opponent)`, `game_id=NULL` (matchup source is season-aggregated, no game_id). |

**Inputs (canonical):** `simulation_step2_lineup_profiles.parquet` (starter IDs),
`projected_player_profiles.parquet` (archetype / `position_band_3` / `position_proxy` /
mpg, fallback `player_impact_profiles.parquet`), `pts_v40.parquet` (`pts_o_v40` /
`pts_d_v40` talent rank — NOT `impact_obke`).

**Validation:** `scripts/validate_matchup_interactions.py` →
`reports/cross_team_interaction_validation.json`. Walk-forward player-game PPP MAE
on the 2024-25 holdout (train FE+cells ≤2023-24, applied forward — leakage-free).
Result: **MODEL HELPS (OOS)** — leakage-free player-game PPP WMAE 0.18256→0.18119
(+0.75% overall, +1.5% on significant-cell rows), paired-bootstrap p=1.000. Gain is
~2× larger where a significant train cell applies. Headline magnitude is small (the
fuller value is possession/props realism, judged in Step 2).

**Data caveat:** all cells are `source = emergent_matchup` — closest-defender
proximity attribution (Second Spectrum), not intentional assignment.

---

## App Viewers

No version forks — each viewer is a single canonical file. They load multiple BKE versions
for **comparison display**, but the primary data always comes from v2.8 decomposition.

| Viewer | Primary BKE input | Notes |
|---|---|---|
| `app/player_bke_viewer.py` | `bke_v31_components.json` | Experimental v3.1 interactive slider — not production scoring |
| `app/player_data_viewer.py` | v27 + v28 + v30 | Side-by-side comparison; v30 column is partial (failed Phase C) |
| `app/player_archetype_viewer.py` | Current archetype outputs | No versioning issue |
| `app/player_eval_viewer.py` | `player_profile_aggregate.parquet` | No versioning issue |
| `app/simulation_viewer.py` | Forecast artifacts | No versioning issue |

---

## Loop / Audit Files

| File | What it is | Superseded? |
|---|---|---|
| `loop/system_audit_v2.6.md` | Post-v2.6 system audit snapshot | Historical — describes v2.6 state only |
| `loop/in_progress_context.txt` | Append-only session log | Always current — newest entry = truth |
| `loop/current_phase_plan.DO_NOT_CHANGE.txt` | Task-level checklist | Always current — checkbox state = truth |
| `loop/overall_plan.DO_NOT_CHANGE.txt` | Phase-level plan | Always current |

**Rule for `loop/in_progress_context.txt`:** When entries contradict each other (e.g. "reduced
to 5 archetypes" in an older entry vs. the code having 9), **always trust the code and
`docs/reference/basketball-intuitions.md` over older loop entries.**

---

## Diagnostics / Test Scripts

| File | Version | Status |
|---|---|---|
| `tests/bke_v29_diagnostics.py` | v2.9 | Legacy diagnostics — not part of active test suite |
| `tests/test_simulation_core.py` | — | **ACTIVE** — the only tests run by `pytest -q` |

---

## Quick-Reference: "What do I use?"

| Task | Use this |
|---|---|
| Compute defensive archetypes | `compute_defensive_archetypes_v2.py` (v3.4) |
| Compute offensive archetypes | `compute_player_archetypes.py` (v4.3) |
| Recompute BKE decomposition | `decomposition_engine.py` (v2.8) |
| Regenerate BKE viewer JSON | `construct_bke_scores_v27.py` |
| Check archetype definitions | `docs/reference/basketball-intuitions.md` §4 |
| Check model config constants | `src/modeling/model_config.py` |
| Run v3.0 defense shrinkage | **Don't** — GAP-008, Phase C failed |
| Run v3.1 experimental layers | OK for exploration, **not for production artifacts** |
