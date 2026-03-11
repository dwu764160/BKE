---
name: audit-model
description: |
  Audit suite for modeling and impact estimation (`src/modeling/`). Runs
  model diagnostics, backtests, and checks for reproducibility and input
  sanity (modeling_inputs_* files and reports).
tags:
  - audit
  - modeling
  - backtest
version: "1.0"
last_updated: "2026-03-09"
persona:
  - "Modeling auditor"
  - "Backtest and diagnostics reviewer"
preferred_tools:
  - read_file
  - grep_search
  - semantic_search
  - file_search
  - run_in_terminal
  - get_errors
avoid_tools:
  - create_new_workspace
job_scope:
  - "Run model diagnostic scripts (e.g., `src/modeling/model_rapm.py`) in smoke mode."
  - "Validate `modeling_inputs_{season}.parquet` shapes and distributions against `reports/modeling_inputs_report.json`."
  - "Run backtest harnesses and summarize core diagnostics (correlations, MAE/RMSE)."
when_to_use:
  - "On-demand, after modeling changes or when requested by the user."
example_prompts:
  - "Run a quick model diagnostics pass for RAPM and summarize key flags."
  - "Run the backtest harness for the last two seasons and produce a comparison report."
log_usage: true
usage_log: loop/skill_usage.log
---
# Model Audit Skill

This skill focuses on ensuring modeling inputs, reproducibility, and diagnostic outputs are trustworthy and documented.
