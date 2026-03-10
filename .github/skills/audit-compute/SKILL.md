---
name: audit-compute
description: |
  Audit suite for the compute stage and pipeline formula integrity. Detects
  silent logic bugs in normalization, aggregation, denominator usage, and
  formula correctness. Focuses on `src/data_compute/`, `src/features/`, and
  any scripts that aggregate or normalize data.
tags:
  - audit
  - compute
  - metrics
version: "1.0"
last_updated: "2026-03-09"
persona:
  - "Compute pipeline auditor"
  - "Formula integrity reviewer"
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
  - "Run formula and merge integrity checks across `src/data_compute/`."
  - "Detect silent logic bugs in normalization, aggregation, and denominator usage."
  - "Validate that all merges and joins are correct and do not introduce data leakage or row duplication."
  - "Ensure all formulas (especially for metrics, normalization, and derived fields) are mathematically sound and match project conventions."
  - "Validate that archetype gating logic and centralized thresholds are present and documented."
  - "Run `tests/validate_rapm.py` and any compute-specific validation scripts."
when_to_use:
  - "After any change to metric, normalization, or aggregation logic."
  - "Before major releases or after merging large PRs."
  - "When silent bugs or unexplained output anomalies are suspected."
example_prompts:
  - "Audit all denominator usage in the pipeline."
  - "Check for merge bugs in src/data_compute scripts."
  - "Validate that all normalization steps match NBA conventions."
  - "Find silent logic bugs in metric aggregation."
---
# Compute Audit Skill

This skill concentrates on correctness of derived metrics and risk points where silent bugs commonly appear: denominator misapplication, bad joins, and unguarded `groupby.apply` usage. It now subsumes the former pipeline-formula integrity skill.
