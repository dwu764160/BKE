---
name: audit-fetch
description: |
  Audit suite for the fetch/ingest stage. Validates fetch scripts, cached
  headers/sessions, payload-shape robustness, and presence of required raw
  artifacts in `data/historical/`.
tags:
  - audit
  - fetch
  - ingest
version: "1.0"
last_updated: "2026-03-09"
persona:
  - "Data ingestion auditor"
  - "Fetch pipeline validator"
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
  - "Validate fetch scripts in `src/data_fetch/` for payload-shape guards (resultSets/resultSet) and robust parsing."
  - "Check `data/nba_headers.json` and `data/nba_session.json` presence and freshness."
  - "Run smoke fetches where safe and compare output schemas to expected shapes."
  - "Run `tests/validate_data_integrity.py` or fetch-stage-specific checks when available."
when_to_use:
  - "On-demand, only when the user requests a fetch-stage audit or after major ingest changes."
example_prompts:
  - "Run a fetch audit for season 2024-25 and report any payload parsing fragility."
  - "Verify `data/historical/` artifacts exist and have expected columns."
log_usage: true
usage_log: loop/skill_usage.log
---
# Fetch Audit Skill

This skill performs non-destructive audits. It will not upload credentials or run long-running full-season fetches without explicit user approval.
