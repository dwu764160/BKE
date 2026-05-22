# GEMINI.md — BKE (Basketball KPI Engine)

**CRITICAL INSTRUCTION: Always check `CLAUDE.md` for updates and ensure this `GEMINI.md` file remains synchronized with it.**

Guidance for Gemini CLI and Claude Code sessions working in this repository.

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
- **`loop/in_progress_context.txt` is NOT append-only.** It holds ONE task at a time
  (current work description) to save tokens on context. Replace it entirely when
  starting a new task; do NOT append session entries. When a task is complete,
  replace with the next task description.
- **Multiple versions exist** for BKE scoring, archetype scripts, and data artifacts.
  Before touching any versioned file, read `docs/multiple-versions.md` for the canonical
  truth on which version is production.
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

This repo has a comprehensive skill library: **portable governance skills**, **BKE domain skills**, and **context engineering skills**. The canonical single source of truth is `.github/skills/SKILL_MAP.md`.

### MANDATORY: Skill Auto-Loading at Task Start

**Every task MUST follow this sequence before editing or planning:**

1. **Load mandatory skills** via the Skill tool:
   - `scope-creep-guard` — enforce phase boundaries before any edits
   - `detailed-chat-output` — structured output for every task
   
2. **Determine task domain** by reading the user request and current work context (`loop/in_progress_context.txt`).

3. **Load applicable domain skills** by:
   - Consulting `.github/skills/SKILL_MAP.md` **Selection Order** (lines 11–60)
   - Using the Skill tool to invoke each skill that matches your task
   - Reading each skill's guidance before planning or editing

4. **Do not skip or rely on memory:** SKILL_MAP.md is the canonical source; it changes over time. Always consult it before treating a skill as deprecated or adding new trigger conditions.

**Example:** For a modeling change, load: `scope-creep-guard` → `detailed-chat-output` → `audit-model` → then proceed.

For a complete reference of all skills and when to use them, see `.github/skills/SKILL_MAP.md` **Selection Order** and **Skill Registry**.

### Output Format — Mandatory

Every response MUST follow the **detailed-chat-output** structure:

1. **Outcome** — Lead with what was accomplished (1–2 sentences).
2. **Changes** — Concrete file edits and imports (file paths + line numbers).
3. **Verification** — Test results, passes, or "not verified" callout.
4. **Next Steps** — What comes next or what's pending.

Read `loop/in_progress_context.txt` at the start of each session for current task context.

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