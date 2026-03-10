---
name: simulation
description: |
  Operations and audit skill for the simulation pipeline. Combines schedule
  generation, game-model execution, and validation/report integrity checks.
  Use this skill to run simulations, validate schedule balance, and audit
  Step1/Step2 outputs and constraints.
tags:
  - simulation
  - audit
  - schedule
version: "1.0"
last_updated: "2026-03-09"
persona:
  - "Simulation and schedule validation specialist"
  - "Game model and report consistency auditor"
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
  - "Run and validate all simulation and schedule scripts (`src/simulation/`)."
  - "Audit game model logic and ensure schedule integrity (balanced home/away, per-team totals)."
  - "Check simulation report outputs for consistency, completeness, and expected metrics."
  - "Run `src/simulation/validate_sim.py` and `src/simulation/season_sim.py` in smoke/validation mode."
  - "Validate that Step2 lineup projection constraints (position bands, minutes eligibility, true-big requirement) are enforced."
when_to_use:
  - "After changes to simulation, schedule, or report logic."
  - "Before major simulation runs or after merging game model PRs."
example_prompts:
  - "Run and validate all simulation scripts."
  - "Audit schedule generation in simulation."
  - "Check Step2 lineup constraint enforcement and report consistency."
---
# Simulation Skill (Operations + Audit)

This consolidated skill covers both operational runs (simulate, forecast) and
audit checks (validate schedules, run validation harnesses). Use it for
regular simulation work and for on-demand audits when requested.
