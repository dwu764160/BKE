---
name: data-audit
description: |
  Repository-wide data audit: normalizes column names across processed artifacts,
  runs data validation tests, and ensures downstream consumers see consistent
  schemas.
tags:
  - data
  - audit
  - schema
version: "1.0"
last_updated: "2026-03-09"
persona:
  - "Data integrity auditor"
  - "Schema normalization specialist"
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
  - "Scan `data/processed/` and `data/features/` for common column name variants and report normalization actions."
  - "Run `tests/validate_data_integrity.py` and other data checks available in `tests/`."
  - "Produce a short remediation plan if columns are inconsistent (suggest `apply_patch` patches to canonical readers or docs)."
when_to_use:
  - "On-demand, before major exports or when schema drift is suspected."
example_prompts:
  - "Run the data audit and normalize common column name variants across processed artifacts."
  - "Run `tests/validate_data_integrity.py --season 2024-25` and summarize failures."
---
# Data Audit Skill

This skill helps catch schema drift early by running targeted checks and producing recommended fixes for maintainers.
