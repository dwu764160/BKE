---
name: detailed-chat-output
description: "Mandatory output formatting for every task to keep responses structured and traceable."
---

# Detailed Chat Output

## When to Use

Use this skill on every task, including one-line confirmations, planning-only requests, and implementation work.
This applies to all tasks and is especially relevant when documenting verification steps, outcome reporting, high-risk refactors, or other multi-step changes.

## When Not to Use

N/A — this skill is always required.

## Files and Surfaces

- .github/skills/
- .github/prompts/
- docs/
- README.md

## Deliverables

- response order: outcome, changes, verification, residual risk or next steps
- direct file references for edited files
- concise but complete rationale for non-obvious decisions
- explicit callout of anything not verified

## Output Rules

- Lead with what was achieved.
- Keep sections short and scannable.
- Avoid filler and duplicate explanations.
- Use consistent wording for verification status.

## Verification Checklist

- The answer includes outcome first.
- Changes are mapped to concrete files.
- Verification status is explicit.
- Any risks or follow-up actions are clear.
