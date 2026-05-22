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

- response order: summary, outcome, changes, verification, residual risks, next steps
- direct file references for edited files
- concise but complete rationale for non-obvious decisions
- explicit callout of anything not verified

## Output Rules

- Lead with what was achieved.
- Keep sections short and scannable.
- Avoid filler and duplicate explanations.
- Use consistent wording for verification status.

## Verification Checklist

- The answer includes summary first.
- The answer includes outcome second.
- Changes are mapped to concrete files.
- Verification status is explicit.
- Any risks and follow-up actions are clear.

## Skills Used

Always include a dedicated section listing all skills invoked during the task (mandatory and domain-specific). Format as a numbered list, one per line, with optional brief context if the skill application was non-obvious.

Example:
```
1. scope-creep-guard — locked phase boundaries and file allow-list
2. detailed-chat-output — structured this response
3. audit-eval — verified player eval pipeline correctness
4. documentation-cohesion — updated README for Phase 2 changes
```

This ensures transparency about what governance and domain tools guided the work.
