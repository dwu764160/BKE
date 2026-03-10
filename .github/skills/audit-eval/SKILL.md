---
name: audit-eval
description: |
  Audit suite for player evaluation (`src/player_eval/`) and minute-model
  pipeline components. Validates player impact profiles, minute-model fits,
  and profile-aggregate integration.
tags:
  - audit
  - player_eval
  - minute_model
version: "1.0"
last_updated: "2026-03-09"
persona:
  - "Player evaluation auditor"
  - "Minute model reviewer"
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
  - "Run `src/player_eval/build_player_impact_profiles.py` in smoke mode and validate shape."
  - "Run minute-model training checks and cross-validate `minute_model_v2.pkl` behavior."
  - "Validate profile aggregate ingestion (profile_aggregate/build_profile_aggregate.py)."
when_to_use:
  - "On-demand, after player-eval logic changes or when requested by the user."
example_prompts:
  - "Audit player impact profile builds and minute-model fits for suspicious distributions."
  - "Validate that the profile aggregate includes newest fields added in the last commit."
---
# Player Eval Audit Skill

This skill runs targeted, non-destructive checks and produces a short diagnostic summary for user review.
