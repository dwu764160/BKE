---
name: docs-sync
description: |
  Keeps repository documentation and in-file headers in sync with code changes.
  Verifies `readme.md`, `loop/` notes, and viewer/docs references after edits and
  proposes minimal patches to update file paths and standardized file headers.
tags:
  - docs
  - sync
  - headers
triggers:
  - "sync docs"
  - "update readme"
  - "docs sync"
  - "headers"
version: "1.0"
last_updated: "2026-03-09"
persona:
  - "Documentation maintainer"
  - "Docs and file-header synchronizer"
preferred_tools:
  - read_file
  - grep_search
  - semantic_search
  - file_search
  - apply_patch
  - run_in_terminal
  - get_errors
avoid_tools:
  - create_new_workspace
job_scope:
  - "Detect edits to pipeline scripts and update `readme.md` pipeline order and data layout accordingly."
  - "Verify file headers exist and contain canonical path + separator lines; propose/apply standardized header patches for edited files (user approval required for apply)."
  - "Check viewer/docs references (app/, reference/) for broken or outdated paths after changes and propose updates."
when_to_use:
  - "After code changes that add, remove, or rename pipeline steps or outputs."
  - "Before releases or when PRs touch many files."
example_prompts:
  - "Sync docs after my last commit and patch missing headers."
  - "Check `readme.md` for pipeline order consistency with `src/` and update if needed."
log_usage: true
usage_log: loop/skill_usage.log
---
# Docs Sync Skill

Notes on behavior and safety:

- Preferred workflow: prepare a proposed patch updating docs and headers, show it to the user, and apply only after explicit approval.
- Standard header format: a short top-line path comment and a separator line consistent with other files (see `loop/context_summary.txt` header examples).
- The skill will run `get_errors` after applying header/docs patches to ensure no lint/compiler regressions.
