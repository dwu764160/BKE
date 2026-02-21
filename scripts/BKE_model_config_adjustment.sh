#!/bin/bash
# BKE_model_config_adjustment.sh
# Run all modeling layers and outputs after model_config.py changes
# bash scripts/BKE_model_config_adjustment.sh

set -e

echo "[BKE] Running all modeling layers and outputs after model_config.py adjustment..."

# Step 1: Run RAPM modeling (uses model_config.py)
python3 src/modeling/model_rapm.py

# Step 2: Ingest DARKO (uses model_config.py)
python3 src/modeling/ingest_darko.py

# Step 3: Run full decomposition engine (uses model_config.py)
python3 src/modeling/decomposition_engine.py

# Step 4: Build final BKE/OBKE/DBKE outputs (uses model_config.py)
python3 src/modeling/construct_bke_scores_v27.py

# Step 5: (Optional) Run backtesting for validation
# python3 -m src.modeling.backtesting
# python3 -m src.modeling.backtesting --all-pairs

# Step 6: Regenerate player data viewer HTML
python3 app/player_data_viewer.py

# Step 7: Print summary of key outputs
echo ""
echo "[BKE] === BKE Decomposition Data ==="
ls -lh data/processed/bke/bke_v28_decomposition.parquet data/processed/bke/bke_v28_decomposition.csv data/processed/bke/BKE_Scores_v27.json data/processed/bke/dimension_scores_v28.json data/processed/bke/layer_scores_v28.json data/processed/bke/obke_dbke_scores_v28.json || true
echo ""
echo "[BKE] === Reports ==="
ls -lh reports/bke_v28_report.json reports/bke_v28_variance_report.json reports/bke_v28_compression_report.json reports/modeling_inputs_report.json || true
echo ""
echo "[BKE] === Viewer ==="
ls -lh app/player_data.html || true

echo "[BKE] Adjustment run complete. Review outputs and reports for validation."
