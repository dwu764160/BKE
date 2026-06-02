# src/player_eval/ — Layer 4: Player Evaluation

Builds player impact profiles, trains the minute model, and projects forward to next season. Output feeds `src/profile_aggregate/` and `src/simulation/`.

---

## Scripts

| Script | Output | Purpose |
|--------|--------|---------|
| `build_player_impact_profiles.py` | `data/processed/player_eval/player_impact_profiles.parquet` | **Primary output** — 47+ fields from 9 sources: BKE decomposition, archetypes, RAPM, linear metrics, position estimates, box stats, player metadata, clutch stats, simulation Step-1 fields. Includes behavioral fingerprint, availability score, archetype embeddings, PEC probability embeddings. |
| `calibrate_team_scale.py` | scale calibration coefficients | Calibrates BKE → NBA pts/100 team scale (`TEAM_SCALE=20`). Used by `team_feature_aggregation.py`. |
| `project_next_season.py` | `data/processed/forecast/projected_player_profiles.parquet` | Forward projection: regression-to-mean + age-adjusted carry-forward. Uses `minute_model_v2.pkl` for MPG. Supports `end_of_season` and `preseason_snapshot` scenarios. Rookie generation is draft-tier-based (no CSV dependency). |
| `year_to_year_bke_deltas.py` | YoY BKE delta analysis | Year-over-year BKE change analysis and stability checks |
| `train_minute_model.py` | `minute_model_v2.pkl`, `minute_model_predictions_v2.parquet` | Temporal Ridge v3 — season-N features → N+1 MPG prediction. 16 features including archetype embeddings. GroupKFold CV by season. Backtest: r²=0.671, MAE=4.06 MPG on 2023-24→2024-25 holdout. |
| `constants.py` | (config) | Path constants for player eval layer |

---

## Key data paths

```
data/processed/player_eval/
  player_impact_profiles.parquet     ← primary eval output
  player_profiles_season.pkl         ← per-season profile cache
  minute_model_v2.pkl                ← trained minute model
  minute_model_predictions_v2.parquet ← precomputed MPG predictions
  team_feature_aggregation.parquet   ← team-level aggregation

data/processed/forecast/
  projected_player_profiles.parquet  ← next-season projection
```
