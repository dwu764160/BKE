---
name: workflow-logging
description: "Use when recording progress, decisions, verification outcomes, and workflow-impacting changes."
---

# Workflow Logging

## When to Use

Use this skill when a task changes instructions, prompts, skills, process docs, or operating rules.

## When Not to Use

Do not use this skill for trivial edits with no workflow impact and no meaningful verification output.

## Files and Surfaces

- .github/skills/SKILL_MAP.md
- .github/skills/
- docs/
- logging/progress_log.md
- logging/commit_log.md
- .githooks/pre-push

## Deliverables

- concise progress updates during work
- a final change summary that is factual and scoped
- explicit verification results (what was checked and outcome)
- residual risks or follow-up actions when applicable

## Logging Rules

- Log only material decisions, not every keystroke.
- Keep statements testable and specific.
- Record what changed, why it changed, and what was verified.
- If you changed skills, include how the skill map was updated.
- When logging workflow-impacting work, update `logging/progress_log.md` in the same task.
- Every entry in `logging/progress_log.md` must follow the required entry format below.

## Required Entry Format

Use this exact section structure for each new entry.

```markdown
## Entry NNN - YYYY-MM-DD - Quick Title

- Task: One-line statement of the original task/request.
- What the agent did: Concrete outcomes.
- How the agent did it: Brief method used (research, files inspected, checks run).
- Files edited:
	- path/to/file1
	- path/to/file2
- Verification:
	- Specific checks the user can run or inspect.
- Task alignment:
	- Fulfillment: How the result satisfies the request.
	- Deviation: "None" or explicit deviation and reason.
```

Entry rules:

- `NNN` is a zero-padded sequential number (`001`, `002`, ...).
- Date must use ISO format `YYYY-MM-DD`.
- Append new entries in chronological order.

## Verification Checklist

- Progress updates are brief and action-oriented.
- Final summary includes outcome, changes, and verification.
- Any missing verification is clearly stated.
- New `logging/progress_log.md` entries are numbered, dated, titled, and complete.
