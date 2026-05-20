---
name: repo-workflow
description: "Use when planning, selecting, or maintaining repository instruction files, prompts, skills, and workflow docs."
---

# Repository Workflow

## When to Use

Use this skill for work that affects the repository's agent workflow surface:

- discovering existing instruction files, prompts, agents, or skills
- selecting the smallest sufficient set of repo-specific skills
- creating or updating customization files
- validating that workflow docs are current and non-conflicting
- logging material workflow changes when the repo already has a progress or changelog file

## When Not to Use

Do not use this skill for feature implementation, runtime debugging, or product design work unless the task specifically includes workflow or instruction maintenance.

## Files and Surfaces

This skill influences:

- `.github/skills/SKILL_MAP.md`
- `.github/prompts/`
- `.github/skills/`
- `.github/instructions/`
- `CLAUDE.md`
- any repo progress or changelog file used to track workflow changes

## Deliverables

When this skill is used, produce the smallest workable set of outputs:

- a clear task classification
- a short plan with ordered steps
- the relevant customization files or edits
- a verification pass against the actual repository state
- a brief note on residual risk or follow-up work

## Operating Rules

- Read `.github/skills/SKILL_MAP.md` first for skill selection and ordering.
- Load `.github/skills/skill-map-governance/SKILL.md` whenever any skill file is added, removed, renamed, or scope-changed.
- Inspect the repo's current instruction surface before making changes.
- Prefer existing workflow docs over inventing new conventions.
- Keep new skill files narrow and specific.
- Update `.github/skills/SKILL_MAP.md` in the same change when any skill is added, removed, renamed, or scope-changed.

## Verification Checklist

- The target files exist at the intended paths.
- Frontmatter parses cleanly and the `description` is meaningful.
- The file scope matches the task and does not pull in unrelated behavior.
- `.github/skills/SKILL_MAP.md` is current if skills were modified.
- No duplicate or conflicting instruction surfaces were introduced.
