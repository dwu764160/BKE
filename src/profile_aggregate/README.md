# src/profile_aggregate/ — Layer 5: Profile Aggregate

Final assembly layer. Joins every upstream signal into one wide table per player-season.

---

## Scripts

| Script | Output | Notes |
|--------|--------|-------|
| `build_profile_aggregate.py` | `aggregate/player_profile_aggregate.parquet` | **THE final player surface.** ~924 columns, ~1,971 rows (one per player-season across 9 seasons). Merges 18+ pipeline sources: BKE decomposition, archetypes (off + def), RAPM, linear metrics, position estimates, official box stats, player metadata, clutch stats, draft data, salaries, simulation Step-1 fields, YTD blending. |
| `team_feature_aggregation.py` | `data/processed/player_eval/team_feature_aggregation.parquet` | Minute-weighted team BKE, structure modifiers (spacing, playmaking depth, rim protection), interaction matrix application. Produces projected team net ratings. Supports `forecast_mode=True` for forward projections. |

---

## What's in `player_profile_aggregate.parquet`

One row = one player-season (e.g., Steph Curry in 2023-24).

Column groups:
- **Identity** — player_id, player_name, season, team, position_band, age
- **BKE** — bke_composite, obke, dbke, all dimension scores (dim_*)
- **Archetypes** — off_primary_archetype, def_primary_archetype, confidence, probability embeddings
- **RAPM** — rapm, orapm, drapm
- **PTS v4.0** — pts_o_v40, pts_d_v40
- **Linear** — ws, ows, dws, bpm, vorp
- **Box stats** — pts, reb, ast, stl, blk, tov, min, gp, fga, fgm, fg3a, fg3m, fta, ftm
- **Behavioral** — behavioral_usage, behavioral_efg, behavioral_three_point_rate, behavioral_assist_rate, ...
- **Position** — pct_pg, pct_sg, pct_sf, pct_pf, pct_c, primary_position_estimate
- **Eval** — impact_total_impact, availability_score, mpg, pred_mpg
- **Simulation** — simulation-Step-1 archetype interaction fields
- **Draft + salary** — draft_year, draft_round, draft_pick, salary

This is the input surface for viewers (`app/player_eval_viewer.py`, `app/simulation_viewer.py`) and the simulation team aggregation.
