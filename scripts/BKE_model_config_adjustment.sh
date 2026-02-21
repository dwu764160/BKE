#!/bin/bash
# BKE_model_config_adjustment.sh
# Run full modeling/decomposition pipeline after model config changes

set -e

echo "[BKE] Running decomposition pipeline after model config adjustment..."

# Step 1: Run decomposition engine (v2.8)
python3 src/modeling/decomposition_engine.py

# Step 2: Regenerate final constructor output (BKE_Scores_v27.json)
python3 src/modeling/construct_bke_scores_v27.py

# Step 3: Regenerate player data viewer HTML
python3 app/player_data_viewer.py

# Step 4: Print summary of key outputs
ls -lh data/processed/bke_v28_decomposition.parquet data/processed/bke_v28_decomposition.csv data/processed/bke_v28_report.json data/processed/bke_v28_variance_report.json data/processed/bke_v28_compression_report.json data/processed/dimension_scores_v28.json data/processed/layer_scores_v28.json data/processed/obke_dbke_scores_v28.json data/processed/BKE_Scores_v27.json || true
ls -lh app/player_data.html || true

echo "[BKE] Adjustment run complete. Review outputs and reports for validation."
