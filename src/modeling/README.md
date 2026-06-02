# src/modeling/ — Layer 3: Modeling

RAPM, BKE decomposition, PTS scoring, and archetype interaction models.

---

## Production scripts

| Script | Output | Status |
|--------|--------|--------|
| `model_rapm.py` | `player_rapm.parquet` | Ridge regression (α=200) on 5v5 lineup possessions, pooled across seasons |
| `ingest_darko.py` | `modeling_inputs_{season}.parquet`, `modeling_inputs_all.parquet` | DARKO + RAPM merge for BKE input |
| `decomposition_engine.py` | `bke_v28_decomposition.parquet` | **BKE v2.8 — production**. 4-layer decomposition: RAPM backbone (25%) + Playtype (20%) + 9-Dimension (55%). See `model_config.py` for constants. |
| `construct_bke_scores_v27.py` | `BKE_Scores_v27.json` | Terminal OBKE/DBKE/BKE scoring + percentile brackets. Input for all viewers. |
| `layer1_portable_talent.py` | (component) | Layer 1 — portable talent with Bayesian shrinkage |
| `layer2_role_utilization.py` | (component) | Layer 2 — role utilization efficiency |
| `layer3_archetype_elevation.py` | (component) | Layer 3 — archetype elevation modifier |
| `layer4_scheme_amplification.py` | (component) | Layer 4 — scheme stability bonus |
| `percentile_engine.py` | (component) | Percentile → z-score mappings |
| `model_config.py` | (config) | BKE v2.7 config — constants, season list, feature weights. Source of truth for pipeline constants. |

## Experimental / deprecated

| Script | Status |
|--------|--------|
| `bke_v31_experimental_layers.py` | **Experimental only** — v3.1 interactive viewer layers; do not overwrite production artifacts |
| `dbke_v30_defense_shrinkage.py` | **Failed** (Phase C broke on 8-season data) — do not use for production |
| `backtesting.py` | Year-over-year BKE prediction backtest |
| `bayesian_hierarchical.py` | Bayesian hierarchical model (experimental) |
| `validate_archetypes.py` | Archetype label sanity checks |
| `fit_archetype_interactions_v1_team_season.py` | v1 team-season interaction fitter (superseded by v2 in scripts/) |
| `fit_archetype_interactions_v2.py` | v2 interaction fitter (src/modeling copy — canonical is `scripts/fit_archetype_interactions_v2.py`) |

## Version guide

| Artifact | Version | Notes |
|----------|---------|-------|
| `bke_v28_decomposition.parquet` | **v2.8** | Production |
| `BKE_Scores_v27.json` | v2.7 scoring | Production viewer input (built from v2.8 decomp) |
| `bke_v30_decomposition.parquet` | v3.0 | Failed Phase C — do not use |
| `bke_v31_components.json` | v3.1 | Experimental only |

See `docs/multiple-versions.md` for the complete version registry.
