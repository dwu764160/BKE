---
name: skill-map-governance
description: "Use when adding, removing, renaming, or changing scope of skills to keep SKILL_MAP.md complete, current, and authoritative."
user-invocable: true
---

# Skill Map Governance

## When to Use

Use this skill whenever any file under .github/skills/ is created, removed, renamed, or materially changed in purpose.

## When Not to Use

Do not use this skill when editing unrelated feature code with no skill-catalog impact.

## Files and Surfaces

- .github/skills/SKILL_MAP.md
- .github/skills/*/SKILL.md

## Deliverables

- updated skill registry entries in SKILL_MAP.md
- updated selection order when lifecycle flow changed
- change note describing what changed, why, and expected load conditions

## Governance Rules

- SKILL_MAP.md is the source of truth and must stay in sync with folder contents.
- Every skill registry entry must link to an existing SKILL.md path.
- Skill names must match parent directory names.
- Selection order must reflect actual workflow priority.

## Sync Procedure

1. Compare skill directories against SKILL_MAP registry.
2. Add missing entries and remove stale entries.
3. Validate each entry's purpose and load conditions.
4. Confirm machine-readable index mirrors the human-readable registry.
5. Include a short change note in the final task output.

## Verification Checklist

- No skill directory exists without a registry entry.
- No registry entry points to a missing skill.
- Selection order includes governance and verification before narrower skills.
- Machine-readable index is aligned with the registry table.
