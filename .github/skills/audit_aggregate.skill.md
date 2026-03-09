---
name: "Aggregate Audit Skill"
description: |
  Audit suite for profile aggregation (`src/profile_aggregate/`). Ensures all
  player-related stats are normalized and merged into the `player_profile_aggregate`.
tags:
  - audit
  - aggregate
  - profile
version: "1.0"
last_updated: "2026-03-09"
persona:
  - "Aggregate pipeline auditor"
  - "Profile integrity reviewer"
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
  - "Run `src/profile_aggregate/build_profile_aggregate.py` in smoke mode and validate output schema."
  - "Check that new fields from compute/modeling stages appear in the aggregate."
  - "Validate counts (expected 1971 player-season rows) and column normalization rules."
when_to_use:
  - "On-demand, after changes to player-profile sources or when requested by the user."
example_prompts:
  - "Validate the profile aggregate and report missing fields introduced in last commit."
  - "Run aggregation smoke build and produce a column-normalization report."
---
# Aggregate Audit Skill

This skill checks that the single-source-of-truth `player_profile_aggregate` remains complete and normalized.
