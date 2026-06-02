# app/ — Streamlit Viewers

Interactive data viewers. Each loads pipeline outputs directly from `data/processed/` and `aggregate/`.

Run any viewer: `streamlit run app/<viewer>.py`

---

| Viewer | Primary input | Purpose |
|--------|--------------|---------|
| `player_eval_viewer.py` | `player_profile_aggregate.parquet` | Player cards, team view, detail modal, predicted vs actual MPG. Main analytical surface. |
| `player_archetype_viewer.py` | `player_archetypes.parquet`, `defensive_archetypes_v2.parquet` | Archetype distribution, role clustering, player-archetype lookup |
| `player_data_viewer.py` | `BKE_Scores_v27.json`, `bke_v28_decomposition.parquet` | Side-by-side BKE version comparison (v27/v28/v30) + salary overlay |
| `player_bke_viewer.py` | `bke_v31_components.json` | Experimental v3.1 interactive λ slider + split toggle (60/40 or 55/45). Not production scoring. |
| `simulation_viewer.py` | `forecast_season_results.json`, `forecast_lineup_profiles.json`, simulation parquets | Season simulation tabs: Step 1 + Step 2 + Step 1 Forecast + Step 2 Forecast |

---

**Note on `player_bke_viewer.py`:** uses experimental v3.1 components, not the production v2.8 decomposition. Do not treat its scores as production output.
