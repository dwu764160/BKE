#!/usr/bin/env bash
set -euo pipefail

# Run BKE v3.1 Experiment 2 rerun (15-lambda production-tilt sweep)
# Usage: bash scripts/run_experiment2_production_tilt.sh

echo "[RUN] src/modeling/experiment2_production_tilt.py"
python3 src/modeling/experiment2_production_tilt.py
