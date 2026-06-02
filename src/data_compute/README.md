# src/data_compute/ — Layer 2: Compute

Derives structured metrics from normalized PBP and raw fetched data. Reads from `data/historical/` and `data/tracking/`; writes to `data/processed/`.

---

## Scripts

### Player profiles + metrics

| Script | Reads | Output |
|--------|-------|--------|
| `compute_player_profiles.py` | `possessions_clean_*.parquet`, official stats | `data/processed/player_profiles_advanced.parquet` — PBP-derived per-player-season stats; note: AST=0 for pre-2022 (no `assistPersonId` in PBP), filled downstream by `compute_linear_metrics.py` |
| `compute_linear_metrics.py` | `player_profiles_advanced.parquet`, `complete_player_season_stats.parquet` | `data/processed/metrics_linear.parquet` — Win Shares (OWS/DWS/WS), BPM (B-Ref 2.0 formula), VORP; DWS normalized so season total WS targets 1,230 |
| `compute_local_metrics.py` | box stats + tracking | `advanced_local_metrics.parquet` — box-score-derived advanced metrics |
| `compute_advanced_metrics.py` | possessions, team game logs | `metrics_teams.parquet`, `metrics_lineups.parquet` — team ORTG/DRTG/NET, 5-man lineup stats |
| `compute_position_estimate.py` | `player_profiles_advanced.parquet`, tracking | `player_position_estimates.parquet` — PG/SG/SF/PF/C percentage shares + primary position band |
| `compute_clean_possessions.py` | `possessions_*.parquet` | `possessions_clean_*.parquet` — valid 5v5 possessions only |

### Archetypes

| Script | Output | Notes |
|--------|--------|-------|
| `compute_player_archetypes.py` | `player_archetypes.parquet`, `archetype_embeddings.parquet` | **11 offensive archetypes** (v4.3); threshold ≥200 min, ≥10 GP, ≥8 MPG; below threshold → `"Insufficient Minutes"` |
| `compute_defensive_archetypes_v2.py` | `defensive_archetypes_v2.parquet` | **9 defensive archetypes** (v3.4); `compute_defensive_archetypes.py` (v1) is deprecated — do not use |

**Offensive archetype list:** Ball Dominant Creator · Ballhandler · All-Around Scorer · Interior Scorer · Perimeter Scorer · PnR Rolling Big · PnR Popping Big · Off-Ball Finisher · Off-Ball Movement Shooter · Off-Ball Stationary Shooter · Connector

**Defensive archetype list:** POA Defender · Wing Stopper · Versatile Defender · Off-Ball Chaser · Rotational Defender · Rim Protector · Dropping Big · Mobile Big · Low-Activity Defender

### Coefficient fitting (side pipelines)

| Script | Output | Used by |
|--------|--------|---------|
| `fit_rest_hca_coefficients.py` | `reports/rest_hca_coefficients.json` | Game model HCA, B2B penalty, rest bonus |
| `fit_team_pace.py` | pace coefficients | Possession rate adjustments in simulation |
