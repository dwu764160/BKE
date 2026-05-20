---
name: scope-creep-guard
description: "Mandatory boundary guard for every task. Use to lock phase scope, constrain file touch surfaces, and prevent out-of-scope edits before completion."
user-invocable: false
---

# Scope Creep Guard

## When to Use

Use this skill on every task, including planning-only, documentation, and implementation requests.

## When Not to Use

Never skip this skill. If the task has no phase boundaries, run a lightweight scope check anyway.

## Purpose

Prevent hidden expansion of scope by forcing explicit boundaries on:

- allowed work surface
- allowed file touch set
- allowed behavior changes for the current phase

## Mandatory Procedure

1. Identify the phase and objective before editing any file.
2. Build an explicit allow-list of files and behaviors for the requested phase.
3. Build an explicit deny-list of nearby but out-of-phase behaviors.
4. Before each edit burst, re-check the target files against the allow-list.
5. After edits, run a changed-file audit and map each change to the declared phase objective.
6. If any change does not map to the objective, treat it as scope creep and remove or quarantine it before completion.

## Verification Gate

Before completion, confirm all of the following:

- every changed file is in the declared allow-list
- no out-of-phase behavior was introduced
- contracts, docs, and taskboard language remain consistent
- residual risks and deferred work are called out explicitly

## Escalation Rules

1. If requested work conflicts with phase boundaries, ask for explicit scope change approval.
2. If uncertainty remains, default to smaller scope and document the blocked expansion.
3. If a previous attempt already caused scope creep, tighten the allow-list to exact files and re-run verification before proceeding.

## Minimum Completion Output

Always include:

1. what changed
2. what was intentionally not changed to preserve scope
3. what phase comes next and what remains blocked until then
