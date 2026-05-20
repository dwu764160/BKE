---
name: manual-testing-guides
description: "Use when writing or revising step-by-step manual testing guides with sunny/rainy paths, recovery drills, and reproducible command output."
user-invocable: false
---

# Manual Testing Guides

## When to Use

Use this skill when you need to write or revise a manual testing guide that a human can execute end to end, especially when the guide combines:

- local service setup
- environment-variable export
- validation of runtime invariants
- direct API, UI, or CLI checks
- failure or recovery drills
- a short log of run outcomes

## When Not to Use

Do not use this skill for:

- unit tests, integration tests, or automation code
- feature implementation
- generic documentation that does not explain how to validate runtime behavior
- one-line run commands that do not need a structured guide

## Required Guide Shape

A strong manual testing guide should usually follow this order:

1. What This Covers
2. Terminal Setup
3. Preflight
4. Start and Reset Local Services
5. Export Local Env Vars
6. Verify Invariants Manually
7. Run the Test Matrix
8. Manual End-to-End Check
9. Optional Rainy Day Drill
10. Personal Notes

## Writing Rules

- Use the target repository's actual commands, paths, endpoints, and outputs.
- Keep the guide copy-pasteable.
- Put commands in fenced code blocks.
- Pair every procedure with clear sunny-day and rainy-day expectations.
- Include exact status codes, headers, response fields, or printed output when the behavior matters.

## Operating Rules

- Load [documentation-cohesion](../documentation-cohesion/SKILL.md) when the task is to draft or refine a guide in a repository doc.
- Load [verification-gate](../verification-gate/SKILL.md) before completion to confirm the guide covers the requested behavior.
- Load [scope-creep-guard](../scope-creep-guard/SKILL.md) before planning or edits.
