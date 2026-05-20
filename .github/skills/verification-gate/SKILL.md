---
name: verification-gate
description: "Use before completion to verify file changes, instruction consistency, and task requirement coverage."
---

# Verification Gate

## When to Use

Use this skill for any task that edits files, changes workflow behavior, or updates repository instructions.

## When Not to Use

Do not use this skill for pure brainstorming where no file or process change is made.

## Files and Surfaces

- changed files in the working tree
- .github/skills/SKILL_MAP.md
- related instruction or prompt files touched in the task

## Deliverables

- requirement coverage check
- changed-file validation
- conflict and duplication scan for instruction surfaces
- explicit statement of what could not be validated

## Verification Protocol

1. Confirm each requested deliverable exists.
2. Confirm frontmatter validity for customization files.
3. Confirm links and paths in changed docs still resolve.
4. Confirm no conflicting guidance was introduced.
5. Report residual risk clearly.

## Verification Checklist

- All user-requested outputs are present.
- Changed files are internally consistent.
- Skill map reflects current skill set.
- Unverified items are explicitly listed.
