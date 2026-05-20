# CLAUDE.md — BKE (Basketball KPI Engine)

Guidance for Claude Code sessions working in this repository.

## Repo essentials

- Possession-level NBA player-impact pipeline: fetch → normalize → features →
  compute metrics/archetypes → BKE modeling → simulation/forecast → viewers.
- **`data/`, `reports/`, `models/`, `aggregate/` are gitignored.** A fresh
  clone has code + reference docs only. Most scripts (`validate_sim.py`, app
  viewers) will `FileNotFoundError` until the data pipeline is run or data is
  restored from a `data_backup_*` snapshot.
- Recorded validation numbers live in `reference/**/*.md` and `loop/*.txt`
  (dated, append-only narrative snapshots), not in tracked JSON.
- Do not edit `loop/*DO_NOT_CHANGE*.txt` checkbox formats — automation parses
  them.
- Tests: `pytest -q` (currently only data-free `tests/test_simulation_core.py`,
  3 checks).

## Cross-Repo References

### Sister Repo: Robinhood Trading Bot

This repo's simulation and prediction output feeds Sleeve C
(Prediction Markets) in the Robinhood trading bot's alpha thesis.

Robinhood alpha thesis:   docs/alpha-thesis.md in the Robinhood repo
BKE performance handoff:  docs/integration/for-alpha-thesis.md (this repo)

To fetch this repo's performance doc from a Robinhood CC session:

    gh api repos/Daniel-Wu-Github/BKE/contents/docs/integration/for-alpha-thesis.md?ref=personal \
      | python3 -c "import sys,json,base64; d=json.load(sys.stdin); \
        print(base64.b64decode(d['content']).decode())"

To fetch the Robinhood alpha thesis from a BKE CC session:

    gh api repos/Daniel-Wu-Github/robinhood/contents/docs/alpha-thesis.md?ref=main \
      | python3 -c "import sys,json,base64; d=json.load(sys.stdin); \
        print(base64.b64decode(d['content']).decode())"

### When to Regenerate docs/integration/for-alpha-thesis.md

Re-run the full investigation whenever:
  - A new BKE version is released (new bke_v* report appears in reports/)
  - validate_sim.py is re-run on new season data
  - The alpha thesis in the Robinhood repo updates Sleeve C parameters
  - Any loop/ file marks a major milestone complete

### Current Sleeve C status (as of 2026-05-19)

Per `docs/integration/for-alpha-thesis.md`: **NOT alpha-ready.** Headline Brier ≈ 0.21 is
in-sample/leaky (repo docs admit same-season leakage). Leakage-free forecast is
MAE ≈ 9 wins (~3-4 worse than Vegas) with **no game-level Brier** and **zero
market-price comparison**. Recommended live allocation: **$0 / research-only**
until a walk-forward game-level test vs. Kalshi closing lines shows positive
closing-line value.

## Skill System

This repo has a dual skill library: **portable governance skills** (workflow, safety, quality) and **BKE domain skills** (pipeline-specific). The canonical source of truth is `.github/skills/SKILL_MAP.md`.

### Before Starting Any Task

1. Read `.github/skills/SKILL_MAP.md`.
2. Determine the task domain.
3. Read each applicable skill file before planning or editing.

### Mandatory Skills (every task)

| Skill | Path |
|---|---|
| scope-creep-guard | `.github/skills/scope-creep-guard/SKILL.md` |
| detailed-chat-output | `.github/skills/detailed-chat-output/SKILL.md` |

### Task-Triggered Skills

| Task Domain | Skills to Load |
|---|---|
| Workflow/instruction/skill file changes | `repo-workflow/SKILL.md`, `skill-map-governance/SKILL.md` |
| Documentation authoring or planning doc updates | `documentation-cohesion/SKILL.md` |
| Manual testing guides, runbooks, validation checklists | `manual-testing-guides/SKILL.md` |
| Any task that edits files, config, or process docs | `verification-gate/SKILL.md` |
| Progress log, commit log, or logging surface updates | `workflow-logging/SKILL.md` |
| Commits being pushed to remote | `remote-commit-logging/SKILL.md` |
| Repeated errors, stale docs, or avoidable rework | `self-improvement-loop/SKILL.md`, `skill-improvement-loop/SKILL.md` |
| Archetype, position-band, or simulation decisions | `basketball-knowledge/SKILL.md` |
| Simulation pipeline changes | `simulation/SKILL.md` |
| Forecast/projection/lineup/minute model changes | `forecast/SKILL.md` |
| Modeling/RAPM/BKE metric changes | `audit-model/SKILL.md` |

## Self-Improvement System

### Session Start — Check First

**Before doing anything else**, read `.claude/pending-improvements.md`. If it has unresolved entries:
1. Address each item (run `skill-improvement-loop`, update skills, re-run smoke test).
2. Delete the resolved section from the file.
3. Then proceed with the user's task.

### Automatic Verification (PostToolUse Hook)

`scripts/post-edit-check.sh` runs after every Edit/Write call against modified `.py` files:
- Runs `pytest -q tests/` at the project root
- Writes errors to `.claude/session_errors.tmp`
- Rate-limited to 60 seconds per check/root to avoid excessive runs

### Automatic Verification (Stop Hook)

`scripts/session-end.sh` runs at the end of every session:
- Runs configured checks for modified files
- Appends a structured entry to `.claude/debugging_log.md`
- Calls `scripts/analyze-patterns.sh` to scan for recurring error patterns
- Auto-flags `skill-improvement-loop` if error threshold exceeded (≥2 errors)

### Git Hooks Setup (one-time per local clone)

```bash
git config core.hooksPath .githooks
```

This wires the `pre-commit` (secret scan) and `pre-push` (commit log) hooks.

### Push Notifications

`scripts/notify.sh` sends ntfy.sh push notifications. Set `NTFY_CHANNEL_URL` env var to your topic to activate. Leave unset if not needed — scripts fail silently.

### Running the Smoke Test

After any skill is created, renamed, or reworded:
```bash
bash scripts/implicit-skill-smoke-test.sh
```
A failing test = a skill has a trigger gap. Fix the description or "When to Use" wording, then re-run.
