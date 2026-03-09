---
name: "Self-Improvement Skill"
description: |
  Runs at the end of an interactive session to evaluate which skills were used,
  collect simple evaluation signals (user feedback, test / lint results, runtime
  exit codes), and apply minimal, well-scoped updates to skill files to record
  what was learned. Designed to keep skills current and reduce repeated mistakes.
tags:
  - meta
  - self-improvement
version: "1.0"
last_updated: "2026-03-09"
persona:
  - "Reflective automation assistant"
  - "Meta-skill maintainer"
preferred_tools:
  - read_file
  - grep_search
  - semantic_search
  - file_search
  - get_errors
  - apply_patch
  - run_in_terminal
avoid_tools:
  - create_new_workspace
  - create_and_run_task
job_scope:
  - "Detect which skills were invoked during the session or which skill files were edited."
  - "Collect evaluation signals: explicit user feedback, `get_errors` output, test results, and command exit codes where available."
  - "Summarize learnings into a short markdown note and append it to `loop/self_improvement_log.md`."
  - "Apply minimal, focused edits to skill files used in the session: add a `## Learnings` section or append concise bullet(s) into the skill file."
  - "If recurring gaps are found, propose (and optionally scaffold) new skills per the skills-first workflow, but do not modify pipeline source code without explicit user approval."
when_to_use:
  - "At the end of every interactive session (explicit `run self-improvement`), or when requested by the user."
example_prompts:
  - "Run self-improvement for this session and update the used skills."
  - "Summarize last session's skill usage and persist learnings."
---
# Self-Improvement Skill

How this skill works (operational summary):

1. Identify session-relevant skills
   - Preferential sources: `loop/session_skill_usage.log` or `loop/self_improvement_last.json` (if present).
   - Fallback heuristics: grep recent `loop/` files, `data_temp_reprod/logs/`, and recent modified files in `/.github/skills/`.
   - If available, use `git log -n 50 --name-only` to find skill files touched in the session.

2. Collect evaluation signals
   - Run `get_errors` over modified files and capture failing test output (`pytest -q`) when repository tests exist.
   - Collect explicit user feedback (if provided via a short prompt or appended notes).
   - Measure quick runtime health signals (exit codes of last-run commands, presence of error logs in `data_temp_reprod/logs/`).

3. Synthesize a short learning note
   - Produce a concise summary: what worked, what failed, recommended change, confidence (high/medium/low).

4. Update skill files (low-risk edits)
   - For each skill used in the session, append or update a `## Learnings` section near the end of the file with 1–3 bullets.
   - Add or update `example_prompts` if the session revealed a better invocation pattern.
   - All edits are minimal and focused: change only `.github/skills/*.skill.md` and `loop/self_improvement_log.md` automatically.

5. Log and surface changes
   - Append a timestamped entry to `loop/self_improvement_log.md` summarizing changes and pointing to any modified skill files.
   - If the synthesis identifies required changes outside skills (code fixes, new tests), create a brief proposed TODO and ask the user for approval before applying.

Safety & governance
 - This skill will not make bulk code changes in `src/` without explicit user approval. It will only modify skill files and loop logs automatically.
 - When confidence is low or changes are non-trivial, the skill prepares a suggested patch and requests user confirmation.

Example invocation
 - "Run self-improvement and update skills used in the last session."
