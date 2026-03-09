---
name: "Backtest vs. Forecast Skill"
description: |
  Ensures that every simulation or prediction phase has both a backtest and a
  forecast component where appropriate. Validates presence of backtest reports
  and forecast artifacts and highlights missing coverage.
tags:
  - forecast
  - backtest
  - governance
version: "1.0"
last_updated: "2026-03-09"
persona:
  - "Forecast/backtest governance specialist"
  - "Projection parity auditor"
preferred_tools:
  - read_file
  - grep_search
  - semantic_search
  - run_in_terminal
  - get_errors
avoid_tools:
  - create_new_workspace
job_scope:
  - "Scan `src/` and `reports/` to find simulation/prediction steps and check for both backtest and forecast outputs."
  - "Run `src/simulation/run_forecast.py --skip-lineup` in smoke mode when safe (user approval required for long runs)."
  - "Validate that forecast artifacts live under `data/processed/forecast/` and that corresponding backtest `reports/` exist."
  - "Summarize gaps and propose a remediation checklist (e.g., add forecast runner or backtest harness)."
when_to_use:
  - "On-demand, when preparing forecasts or validating that modeling changes have both backtest and forecast coverage."
example_prompts:
  - "Verify that every simulation step has both backtest and forecast artifacts and list missing ones."
  - "Run a smoke forecast/backtest parity check for the forecast pipeline."
---
# Backtest vs. Forecast Skill

This skill enforces the policy that every predictive component should be evaluated in both backtest and forecast contexts and helps maintain parity between them.
