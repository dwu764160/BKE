# BKE — Basketball KPI Engine

Possession-level NBA player-impact pipeline: fetch → normalize → compute → model → evaluate → simulate/forecast.

**Branch:** `personal` | **Python 3.10+** | **Tests:** `pytest -q` (3 checks, data-free)

---

## What this repo produces

| Artifact | Script | Description |
|----------|--------|-------------|
| `player_profile_aggregate.parquet` | `build_profile_aggregate.py` | 1,971 rows × ~924 cols — one row per player-season, all signals merged |
| `pts_v40.parquet` | `scripts/build_pts_v40.py` | PTS v4.0 offensive/defensive talent scores |
| `bke_v28_decomposition.parquet` | `decomposition_engine.py` | BKE v2.8 — 4-layer decomposition (RAPM + playtype + 9-dim + scheme) |
| `player_impact_profiles.parquet` | `build_player_impact_profiles.py` | Per-player-season impact profiles with archetype embeddings |
| `possession_box_distributions.parquet` | `scripts/run_possession_engine.py` | Per-game player box-score distributions from generative engine |
| `cross_team_interactions.parquet` | `scripts/compute_matchup_adj.py` | 3-pair archetype matchup adjustments per (season, team, opponent) |

All parquets use canonical **lowercase column names** (`src/data/schema_contract.py`).

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q          # 3/3 data-free smoke tests
```

`data/`, `reports/`, `models/`, `aggregate/` are gitignored. A fresh clone has code + docs only.  
Restore from a `data_backup_*` snapshot, or run the pipeline from scratch (see below).

---

## Pipeline order

```
Layer 0  src/data_fetch/            Raw fetch: NBA.com, B-Ref, tracking, matchup
Layer 1  src/data_normalize/        PBP normalization → pbp_normalized_*.parquet
         src/features/              Lineup inference → pbp_with_lineups_*.parquet
Layer 2  src/data_compute/          Player profiles, archetypes, linear metrics, clean possessions
Layer 3  src/modeling/              RAPM, BKE decomposition, PTS v4.0, interaction cells
Layer 4  src/player_eval/           Impact profiles, minute model, team scale calibration
Layer 5  src/profile_aggregate/     Final wide-table assembly + team feature aggregation
Layer 6  src/simulation/            Game model, possession engine, season sim, forecast
```

Full dependency map with re-run order: `docs/reference/data-pipeline-dependency.md`

### Fetch layer (one-time or per-season refresh)

```bash
python3 src/data_fetch/fetch_historical_data.py          # team + player game logs
python3 src/data_fetch/fetch_players.py                  # player metadata
python3 src/data_fetch/fetch_teams.py                    # team metadata
python3 src/data_fetch/fetch_preseason_rosters.py        # preseason roster snapshots
python3 src/data_fetch/fetch_profiles.py                 # player bio (height, age, etc.)
python3 src/data_fetch/fetch_player_salaries.py          # salary data (ESPN)
python3 src/data_fetch/fetch_player_draft_history.py     # draft class / pick info
python3 src/data_fetch/fetch_official_stats.py           # NBA.com advanced stats
python3 src/data_fetch/fetch_tracking_data.py            # tracking + synergy playtypes
python3 src/data_fetch/fetch_box_scores_complete.py      # full box scores (all seasons)
python3 src/data_fetch/fetch_matchup_data.py             # closest-defender matchup data
python3 src/data_fetch/fetch_shot_zones.py               # shot zone distributions
python3 src/data_fetch/fetch_player_clutch_stats.py      # last-5-min clutch stats
python3 src/data_fetch/fetch_pbp/fetch_play_by_play.py   # play-by-play (DOM fallback)
python3 src/data_fetch/derive_team_game_logs.py          # derive team game logs from PBP
python3 src/data_fetch/summarize_team_logs.py            # per-team-season summaries
python3 src/utils/export_db_to_parquet.py                # SQLite → parquet
```

### Normalize + features

```bash
python3 src/data_normalize/run_normalization.py          # PBP → pbp_normalized_*.parquet
python3 src/features/derive_lineups.py                   # → pbp_with_lineups_*.parquet
python3 src/features/derive_possessions.py               # → possessions_*.parquet
python3 src/data_compute/compute_clean_possessions.py    # → possessions_clean_*.parquet
python3 src/features/compute_rest_home_back2back.py      # → feature_schedule_context.parquet
```

### Compute metrics + archetypes

```bash
python3 src/data_compute/compute_player_profiles.py      # → player_profiles_advanced.parquet
python3 src/data_compute/compute_linear_metrics.py       # → metrics_linear.parquet (WS, BPM, VORP)
python3 src/data_compute/compute_local_metrics.py        # → advanced_local_metrics.parquet
python3 src/data_compute/compute_player_archetypes.py    # → player_archetypes.parquet (11 off types)
python3 src/data_compute/compute_defensive_archetypes_v2.py  # → defensive_archetypes_v2.parquet (9 types)
python3 src/data_compute/compute_position_estimate.py    # → player_position_estimates.parquet
python3 src/data_compute/fit_rest_hca_coefficients.py    # → rest_hca_coefficients.json
python3 src/data_compute/fit_team_pace.py                # → pace coefficients
```

### Modeling

```bash
python3 src/modeling/model_rapm.py                       # → player_rapm.parquet
python3 src/modeling/ingest_darko.py                     # → modeling_inputs_*.parquet
python3 src/modeling/decomposition_engine.py             # → bke_v28_decomposition.parquet
python3 src/modeling/construct_bke_scores_v27.py         # → BKE_Scores_v27.json (viewer input)
python3 scripts/build_pts_v40.py                         # → pts_v40.parquet
python3 scripts/test_archetype_interactions.py           # → archetype_interaction_signal_test.json (pre-flight, locked)
python3 scripts/fit_archetype_interactions_v2.py         # → cross_team_interactions_matrix.parquet
python3 scripts/compute_matchup_adj.py                   # → cross_team_interactions.parquet
```

### Player evaluation + aggregate

```bash
python3 src/player_eval/build_player_impact_profiles.py  # → player_impact_profiles.parquet
python3 src/player_eval/calibrate_team_scale.py          # team scale calibration
python3 src/simulation/train_minute_model.py             # → minute_model_v2.pkl
python3 src/profile_aggregate/team_feature_aggregation.py # → team_feature_aggregation.parquet
python3 src/profile_aggregate/build_profile_aggregate.py  # → player_profile_aggregate.parquet
```

### Simulation + forecast

```bash
python3 scripts/build_all_season_projections.py          # → projected_team_features.parquet (all seasons)
python3 scripts/build_rest_features.py                   # → game_rest_features.parquet
python3 scripts/build_ytd_team_ratings.py                # → team_ratings_ytd.parquet
python3 src/player_eval/project_next_season.py           # → projected_player_profiles.parquet
python3 src/simulation/lineup_projection.py              # → simulation_step2_lineup_profiles.parquet
python3 scripts/run_possession_engine.py --mode v1 --season 2024-25 --sims 200
python3 scripts/validate_possession_engine.py            # Step 2 calibration gate
python3 src/simulation/season_sim.py                     # Monte Carlo season simulation
python3 src/simulation/validate_forecast.py              # Walk-forward game-level Brier
python3 src/simulation/gbdt_game_model.py                # GBDT + Elo ensemble
```

---

## Versioned artifacts

Always read `docs/multiple-versions.md` before touching a versioned script.

| Artifact | Version | Script |
|----------|---------|--------|
| PTS scoring | **v4.0** (production) | `scripts/build_pts_v40.py` |
| BKE decomposition | **v2.8** (production) | `src/modeling/decomposition_engine.py` |
| Offensive archetypes | **v4.3** — 11 types | `src/data_compute/compute_player_archetypes.py` |
| Defensive archetypes | **v3.4** — 9 types | `src/data_compute/compute_defensive_archetypes_v2.py` |
| Game model | Gaussian + GBDT + Elo ensemble | `src/simulation/gbdt_game_model.py` |
| Possession engine | Step 2.1 calibrated | `src/simulation/possession_engine.py` |

---

## Simulation status (Step 2.1 calibrated, 2026-06-02)

Walk-forward 2024-25 holdout (calibrated on ≤2023-24, forward projection):

| Gate | Value | Target |
|------|-------|--------|
| Player pts MAE | 5.79 | < 7.0 |
| Player ast MAE | 1.75 | — (near oracle floor 1.42) |
| Player reb MAE | 2.53 | — (near oracle floor 1.93) |
| Player P10–P90 cov | 0.754 | 0.60–0.95 |
| Team pts MAE | 10.33 | — |
| Team bias | +1.56 | \|bias\| < 2.5 |
| Team P10–P90 cov | 0.925 | 0.60–0.95 |

Next: Markets track — price totals and player props from `possession_box_distributions.parquet`.

---

## Schema contract

```python
from src.data.schema_contract import load_standardized, save_standardized, canonicalize

df = load_standardized("data/historical/team_game_logs.parquet")  # read + canonicalize
save_standardized(df, "data/processed/output.parquet")           # canonicalize + write
```

Rules: all column names lowercase, `season_id` → `season`, season format `'2024-25'`, no `.0` on ID columns.

---

## Key reference docs

| Doc | Purpose |
|-----|---------|
| `docs/multiple-versions.md` | Canonical version for every versioned artifact |
| `docs/reference/data-pipeline-dependency.md` | Full layer-by-layer dependency map |
| `docs/reference/data_schemas.md` | Column name and schema reference |
| `docs/reference/basketball-intuitions.md` | Archetype definitions, position-band policy |
| `docs/phase_v1_architecture.md` | Strategy: Markets track vs Game track |
| `docs/roadmap_to_vegas_accuracy.md` | Path to CLV-positive forecasting |
| `docs/integration/for-alpha-thesis.md` | BKE → Robinhood alpha thesis handoff |

---

## Position-band policy

Canonical bands: `Guard`, `Guard-Forward`, `Forward`, `Forward-Center`, `Center`. Do not collapse hybrid bands in pipeline outputs. Derive a separate coarse role (`Guard`/`Wing`/`Big`) when needed, and keep `position_band` alongside it.

---

## Git hooks

```bash
git config core.hooksPath .githooks
```

---

## Tests

```bash
pytest -q tests/                           # 3/3 data-free tests (always runnable)
python3 scripts/validate_backup_integrity.py  # row count + checksum vs backup
python3 scripts/validate_possession_engine.py # Step 2 calibration gate
```
