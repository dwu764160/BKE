---
name: "Forecast Skill"
description: |
  This skill specializes in projection, scenario mapping, and minute model interactions for the analytics pipeline. It is designed to:
  - Run and audit projection scripts (e.g., lineup_projection.py, run_forecast.py).
  - Map scenarios and ensure correct propagation of scenario parameters.
  - Validate minute model logic and its impact on downstream projections.
  - Focus on files in src/simulation/ and related scenario/model scripts.
tags:
  - forecast
  - projection
version: "1.0"
last_updated: "2026-03-09"
persona:
  - "Scenario mapping and projection specialist"
  - "Minute model and forecast auditor"
preferred_tools:
  - read_file
  - grep_search
  - semantic_search
  - file_search
  - run_in_terminal
  - get_errors
avoid_tools:
  - create_new_workspace
  - create_and_run_task
job_scope:
  - "Run and validate all projection and scenario mapping scripts."
  - "Audit minute model logic and its interactions."
  - "Ensure scenario parameters are correctly mapped and propagated."
when_to_use:
  - "After changes to projection, scenario, or minute model logic."
  - "Before major forecast runs or after merging scenario-related PRs."
example_prompts:
  - "Run and validate all forecast scripts."
  - "Audit scenario mapping in run_forecast.py."
  - "Check minute model interactions in projections."
---
# Forecast Skill

This skill is designed for deep audits and runs of projection, scenario mapping, and minute model logic in the analytics pipeline.
