# BKE Repo Context Summary (for AI ingestion)

Purpose: provide an AI model a concise, actionable summary of the repository, focusing on the BKE metric and the offense + defense archetypes, how they are constructed, and where to find validation/stability evidence.

1) Repo overview
- Pipeline: fetch raw NBA data → normalize PBP → derive possessions/features → compute metrics/archetypes → modeling/impact → exports.
- Primary folders of interest:
  - `data/` (raw, caches, features, processed outputs)
  - `src/` (pipeline code: data_fetch, data_normalize, features, data_compute, modeling)
  - `reports/` (versioned produced metrics, backtests, validations)
  - `reference/` (docs & archetype definitions)
  - `loop/` (live context, plans, and required updates)

2) Design principles to respect
- Role vs Impact separation: archetypes = behavior-only; impact/effectiveness layered afterward.
- Behavior-based gating and rarity: archetype assignments use hard percentile gates + structural constraints.
- Centralized thresholds: gating and cutoffs live in specific compute scripts (expected under `src/data_compute/`).
- Reproducibility: use `bash scripts/reproduce_pipeline.sh` → outputs to `data_temp_reprod/`.

3) The BKE metric — what it is and how it is built
- Definition: a composite possession-level player metric that synthesizes model outputs (ORAPM/DRAPM-like) and possession features into a single, season-normalized player score used for ranking and downstream analysis.
- Inputs:
  - Possession-level features produced by the feature stage (`data/features/`).
  - Tracking-derived features and official stats (`data/tracking/`, `data/official_stats/`).
  - Modeling inputs summarized in `reports/modeling_inputs_report.json`.
- Modeling & construction:
  - Separate offensive and defensive contributions (ORAPM, DRAPM) are estimated with regularized models (see `src/modeling/` and `reports/rapm_validation_report.json`).
  - Outputs are normalized to possessions/minutes and undergo shrinkage/compression steps (see `reports/dbke_v30_defense_shrinkage.json`, `reports/bke_v28_compression_report.json`).
  - A decomposition/parquet input like `data/processed/bke/bke_v28_decomposition.parquet` is used to derive production proxies and experiment sweeps (see `reports/bke_v31_experiment2_production_tilt.json`).
- Output: versioned JSON reports (`reports/bke_v*.json`) and processed parquet tables in `data/processed/bke/`.

4) Offense & Defense Archetypes — purpose and construction
- Goal: classify player behavioral roles ("what you are asked to do") independent of effectiveness.
- Sources: aggregation of play-type frequencies, creation/handling metrics, defensive matchup/coverage features, and tracking measures from `data/features/` and tracking caches.
- Assignment rules:
  - Hard percentile thresholds (e.g., top X% on a feature axis) + minimum minute/possession gating.
  - Multi-criterion gates for rare roles (e.g., POA/Versatile requires high percentiles across multiple independent features).
  - Centralized logic lives in `src/data_compute/compute_player_archetypes.py` (or equivalent).
- Outputs: archetype labels and prevalence tables in `data/processed/` and visualized by `app/player_archetype_viewer.py`.

5) Validation, accuracy & stability evidence (where and what to inspect)
- Key report files (primary sources):
  - `reports/bke_v*_report.json` and `reports/bke_v*_backtest.json` (versioned backtests and diagnostics).
  - `reports/rapm_validation_report.json` (RAPM / model diagnostics comparisons).
  - `reports/modeling_inputs_report.json` (input distributions).
  - `reports/bke_v28_variance_report.json` and `reports/bke_v28_compression_report.json` (variance, compression diagnostics).
  - `reports/dbke_v30_defense_shrinkage.json` (shrinkage settings applied to defense estimates).
- Recommended stability & accuracy metrics to compute or retrieve:
  - Cross-season Pearson correlations of BKE (test–retest stability).
  - Backtest rank-recall / top-N predictive enrichment and mean_abs_rank_shift.
  - RMSE / MAE of BKE-derived predictions vs realized outcomes or RAPM baselines.
  - Variance explained (component-wise) and compression ratios.
  - Archetype prevalence by season and minute thresholds (to verify rarity gates).

6) Example: values pre-extracted from `reports/bke_v31_experiment2_production_tilt.json` (useful reference)
- `version`: "v3.1-experiment2-rerun"
- `generated_at`: "2026-03-01 00:35:43"
- `qualified_players`: 950
- `base_profile.predictive_rho`: 0.31523
- Recommended `lambda`: 0.03 with `predictive_rho`: 0.317888 and `mean_abs_rank_shift_vs_base`: 3.250526
- Production proxy effective weights (top contributors): `orapm` 0.22, `PTS` 0.18, `TS_PCT` 0.14, `AST` 0.12

7) Practical ingestion checklist for an AI model
- Read `readme.md` and `loop/context_summary.txt` for pipeline expectations and current phase.
- Load `reports/modeling_inputs_report.json`, `reports/rapm_validation_report.json`, and the latest `bke_v*_report.json` & `*_backtest.json`.
- Load processed outputs in `data/processed/` and feature tables in `data/features/` to inspect exact input distributions.
- Parse `reference/archetypes/*` to learn archetype definitions and gating variables.
- Compute the stability stats listed above and compare them to the values in `reports/` for cross-validation.
- When changing logic, update `loop/in_progress_context.txt`, `loop/context_summary.txt`, and `readme.md` with the new step and outputs.

8) Governance notes
- After any change to pipeline code, update `readme.md` to reflect the new pipeline order and outputs.
- Update `loop` files to summarize rationale and changes.
- Use `pytest -q` for unit checks and `bash scripts/reproduce_pipeline.sh` for end-to-end reproduction.

---
This summary is intended as the canonical human-readable import for an AI or automation agent onboarding to the repository.
