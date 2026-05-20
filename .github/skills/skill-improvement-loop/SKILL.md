---
name: skill-improvement-loop
description: "Use when evaluating skill effectiveness, summarizing recurring mistakes, and rewriting SKILL.md files to improve automatic skill loading."
user-invocable: true
---

# Skill Improvement Loop

## When to Use

Use this skill when:

- a task had avoidable mistakes tied to skill selection or skill wording
- a skill did not auto-load when it should have
- multiple skills loaded but guidance quality was weak or conflicting
- the skill catalog changed and effectiveness should be re-evaluated

## When Not to Use

Do not use this skill for first-time one-off errors unless they indicate a broader instruction gap.

## Files and Surfaces

- .github/skills/SKILL_MAP.md
- .github/skills/*/SKILL.md
- .github/prompts/
- memory/

## Deliverables

- a concise mistake summary for the task
- a skill effectiveness scorecard based on observed behavior
- targeted updates to one or more skill files when gaps are found
- map updates if skills were added, removed, renamed, or scope-changed

## Effectiveness Rubric

Score each relevant skill from 0 to 2 on each dimension:

- Trigger quality: did the description attract the right task?
- Scope fit: did the skill guidance match the task boundaries?
- Outcome support: did the skill materially improve execution quality?
- Noise control: did the skill avoid unnecessary instructions?

Interpretation:

- 7 to 8: keep as-is or make minor wording improvements
- 4 to 6: revise description and tighten body instructions
- 0 to 3: rewrite or split the skill

## Improvement Procedure

1. Summarize mistakes linked to skill behavior in plain language.
2. Run `bash scripts/implicit-skill-smoke-test.sh` to test trigger quality.
3. Score relevant skills with the rubric.
4. Update skill descriptions first, then body instructions if needed.
5. Update SKILL_MAP in the same change if the catalog changed.
6. Record the lesson in memory only if reusable across tasks.

## Verification Checklist

- Mistakes are summarized and mapped to specific skills.
- Smoke test was run and results reviewed.
- Effectiveness scores are documented for relevant skills.
- Skill updates are minimal, specific, and non-duplicative.
- SKILL_MAP is updated if the skill catalog changed.
