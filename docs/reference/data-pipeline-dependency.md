# BKE Data Pipeline Dependency Reference

> **Last updated:** 2026-05-21. Describes the full pipeline from raw fetch to forecast output.
> **Purpose:** Single reference for what each script produces, what it reads, and where it sits
> in the dependency chain.

---

## Quick Answers

**Are player_profile_aggregate seasons separate rows?**
Yes — one row per player-season. A player appearing in all 8 seasons (2017-18 through 2024-25)
has 8 rows. Total: 5,848 rows across 1,283 unique players.

**Does player_profile_aggregate come before or after BKE and archetypes?**
After. It is the final assembly layer (Layer 9) that joins BKE decomposition, archetypes,
linear metrics, positions, and player metadata into one wide table. It reads from nearly
every upstream artifact.

---

## Layer 0 — Raw Data Fetch

Scripts in `src/data_fetch/`. These hit external APIs (NBA.com, Basketball-Reference) and
write raw parquet files to `data/historical/` and `data/tracking/`.

| Script | Output | Notes |
|---|---|---|
| `fetch_pbp/` | `raw_pbp_{season}.parquet` | One per season; NBA API play-by-play |
| `fetch_box_scores_complete.py` | `complete_player_season_stats.parquet` | Official box stats (pts/reb/ast etc.) all seasons |
| `fetch_official_stats.py` | `official_adv_{season}.parquet`, `official_tracking_{season}.parquet` | Advanced + tracking (2013-14+) |
| `derive_team_game_logs.py` | `final_player_game_logs.parquet` | Per-game player logs (for HCA/B2B fitting) |
| `fetch_players.py` | `players.parquet` | Player metadata (name, DOB, draft info) |
| `fetch_player_draft_history.py` | `player_draft_history.parquet` | Draft position, year, team |
| `fetch_historical_data.py` | Various `data/historical/*.parquet` | Supplemental historical data |

---

## Layer 1 — Data Normalize

Scripts in `src/data_normalize/`. Converts raw API output into structured possession/event tables.

| Script | Reads | Output |
|---|---|---|
| `pbp_parser.py` | `raw_pbp_{season}.parquet` | `possessions_raw_{season}.parquet` |
| `normalize_player_names.py` | Various raw files | Name-canonical mapping used downstream |

---

## Layer 2 — Features

Scripts in `src/features/`. Derives lineup and possession structure from raw events.

| Script | Reads | Output |
|---|---|---|
| `derive_lineups.py` | `possessions_raw_{season}.parquet` | `pbp_with_lineups_{season}.parquet` |
| `derive_possessions.py` | `pbp_with_lineups_{season}.parquet` | Updated possession structure |
| `derive_player_team_stints.py` | `possessions_raw`, box stats | `player_team_stints.parquet` |

---

## Layer 3 — Clean Data

| Script | Reads | Output |
|---|---|---|
| `compute_clean_possessions.py` | `pbp_with_lineups_{season}.parquet` | `possessions_clean_{season}.parquet` |

---

## Layer 4 — Player Profiles (PBP-derived)

| Script | Reads | Output |
|---|---|---|
| `compute_player_profiles.py` | `possessions_clean_{season}.parquet` (all seasons) | `player_profiles_advanced.parquet` |
| `compute_position_estimate.py` | `player_profiles_advanced.parquet` | `player_position_estimates.parquet` |

`player_profiles_advanced.parquet` contains PBP-derived per-player-season stats:
AST, PTS, REB, STL, BLK, TOV, MIN, plus play-type frequencies and shot zone breakdowns.

**Pre-2022 note:** PBP lacks `assistPersonId` → AST=0 for all pre-2022 players in this file.
`compute_linear_metrics.py` fills AST from official box stats in post-processing.

---

## Layer 5 — Linear Metrics (parallel with RAPM)

| Script | Reads | Output |
|---|---|---|
| `compute_linear_metrics.py` | `player_profiles_advanced.parquet`, `complete_player_season_stats.parquet` | `metrics_linear.parquet` |

Computes BPM (B-REF 2.0 formula), position estimates, and other linear composites.
**Reference pool for BPM league averages:** players with ≥500 min (not lowered by archetype threshold).

---

## Layer 6 — RAPM Model (parallel with linear metrics)

| Script | Reads | Output |
|---|---|---|
| `model_rapm.py` | `possessions_clean_{season}.parquet` (pooled), lineup data | `player_rapm.parquet`, `player_rapm.csv` |

Ridge regression (alpha=200) on lineup possession data. Pooled across all 8 seasons.
Produces: `rapm`, `orapm`, `drapm`, `rapm_type` (pooled_split), `possessions_played`.

---

## Layer 7 — BKE Decomposition (needs RAPM)

| Script | Reads | Output |
|---|---|---|
| `decomposition_engine.py` | `player_rapm.parquet`, `player_profiles_advanced.parquet`, `official_tracking_{season}.parquet` | `bke_v28_decomposition.parquet` |

Four-layer decomposition: RAPM backbone (25%) + Playtype (20%) + 9-Dimension model (55%).
Computes all dimension scores, OBKE, DBKE, BKE composite. One row per player-season.

---

## Layer 8 — Archetypes (parallel, both read player_profiles_advanced)

| Script | Reads | Output | Threshold |
|---|---|---|---|
| `compute_player_archetypes.py` | `player_profiles_advanced.parquet`, `official_tracking`, playtype data | `player_archetypes.parquet`, `archetype_embeddings.parquet` | ≥200 min, ≥10 GP, ≥8 MPG |
| `compute_defensive_archetypes_v2.py` | `player_profiles_advanced.parquet`, `official_tracking` | `defensive_archetypes_v2.parquet` | Same |

11 offensive archetypes, 9 defensive archetypes. Players below threshold: `"Insufficient Minutes"`.
Archetypes are **position-agnostic** — based on behavioral role signals, not listed position.

---

## Layer 9 — Player Impact Profiles (final player-level assembly)

| Script | Reads | Output |
|---|---|---|
| `build_player_impact_profiles.py` | `bke_v28_decomposition.parquet`, `complete_player_season_stats.parquet`, `player_archetypes.parquet`, `defensive_archetypes_v2.parquet`, `player_position_estimates.parquet`, `metrics_linear.parquet`, `players.parquet`, `player_team_stints.parquet` | `player_impact_profiles.parquet` |

Joins all player-level signals into one flat table. Computes availability score, archetype
confidence, cohort z-scores, PEC (Player Evaluation Composite) probability embeddings.

---

## Layer 10 — Profile Aggregate

| Script | Reads | Output |
|---|---|---|
| `build_profile_aggregate.py` | `player_impact_profiles.parquet` + most Layer 7-8 artifacts directly | `player_profile_aggregate.parquet` |

Wide table (946 columns) with every signal per player-season. The main analytical surface.
**This is where BKE scores, archetypes, BPM, RAPM, and all dimension scores coexist.**

---

## Layer 11 — Team Feature Aggregation

| Script | Reads | Output |
|---|---|---|
| `team_feature_aggregation.py` | `player_profile_aggregate.parquet`, `preseason_rosters.parquet` | `team_feature_aggregation.parquet` |

Minute-weighted team BKE, structure modifiers (spacing, playmaking depth, rim protection),
and interaction matrix application. Produces projected team net ratings.

---

## Layer 12 — Forecast / Simulation

| Script | Reads | Output |
|---|---|---|
| `project_next_season.py` | `player_profile_aggregate.parquet`, `preseason_rosters.parquet` | `projected_player_profiles.parquet`, `projected_team_features.parquet` |
| `validate_forecast.py` | `projected_team_features.parquet`, game schedule | Walk-forward Brier score |

---

## Dependency Graph (ASCII)

```
Layer 0: Data Fetch
  raw_pbp_{season}            complete_player_season_stats    official_adv/tracking
       │                              │                               │
Layer 1: Normalize                    │                               │
  possessions_raw_{season}            │                               │
       │                              │                               │
Layer 2: Features                     │                               │
  pbp_with_lineups_{season}           │                               │
  player_team_stints ◄────────────────┘                               │
       │                                                               │
Layer 3: Clean Data                                                    │
  possessions_clean_{season}                                           │
       │                                                               │
Layer 4: Player Profiles                                               │
  player_profiles_advanced ◄────────────────────────────────────────  │ (tracking shape)
  player_position_estimates                                            │
       │                              │                               │
       ├──────────────────────────────┤                               │
       │                              │                               │
Layer 5/6 (parallel):                 │                               │
  metrics_linear ◄────────────────────┘ (box stats AST fill)         │
  player_rapm ◄──── possessions_clean                                 │
       │                                                               │
Layer 7: BKE Decomposition                                            │
  bke_v28_decomposition ◄────────────────────────────────────────────┘
       │
       ├─────────────────────────────────────────┐
       │                                         │
Layer 8 (parallel):                              │
  player_archetypes ◄── player_profiles_advanced │
  defensive_archetypes_v2                        │
       │                                         │
       └──────────────────────────────┐          │
                                      │          │
Layer 9: Player Impact Profiles       │          │
  player_impact_profiles ◄────────────┴──────────┘
  (joins: bke_v28_decomp + archetypes + metrics_linear
         + position_estimates + complete_stats + players_meta
         + player_team_stints)
       │
Layer 10: Profile Aggregate
  player_profile_aggregate  ← THE FINAL PLAYER SURFACE
       │
Layer 11: Team Aggregation
  team_feature_aggregation
       │
Layer 12: Forecast / Simulation
  projected_team_features → walk-forward Brier
```

---

## Coefficient Fitting (Side Pipelines)

These run independently and produce coefficients used in the game model:

| Script | Output | Used by |
|---|---|---|
| `fit_rest_hca_coefficients.py` | `reports/rest_hca_coefficients.json` | Game model (HCA, B2B, rest) |
| `fit_team_pace.py` | Pace coefficients | Possession rate adjustments |

---

## Re-run Order After Changes

If you change any file, re-run from the earliest affected layer:

| Change type | Re-run from |
|---|---|
| PBP parse / lineup logic | Layer 1 (possessions_raw) → all downstream |
| Player profile compute | Layer 4 (player_profiles_advanced) → Layer 5+ |
| RAPM model | Layer 6 (player_rapm) → Layer 7+ |
| BKE decomposition | Layer 7 (bke_v28_decomp) → Layer 9+ |
| Archetype thresholds | Layer 8 (archetypes) → Layer 9+ |
| BPM formula | Layer 5 (metrics_linear) → Layer 9+ |
| Aggregate rebuild only | Layer 10 (build_profile_aggregate) |
