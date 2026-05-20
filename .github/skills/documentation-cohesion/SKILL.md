---
name: documentation-cohesion
description: "Use when creating, refining, or maintaining documentation to ensure fixes flow naturally and are readable by both humans and AI agents."
---

# Documentation Cohesion

## When to Use

Use this skill when:

- refining planning or specification documents that guide agent implementation
- integrating corrections, clarifications, or new requirements into existing docs
- ensuring prompt language is explicit enough for AI agents while remaining human-readable
- updating taskboards, decision hierarchies, or workflow docs with new requirements
- reviewing docs to prevent bolted-on sections and maintain natural narrative flow

## When Not to Use

Do not use this skill for:

- simple typo fixes or formatting cleanup with no semantic impact
- feature implementation or runtime code changes
- general writing or communication tasks unrelated to workflow or planning docs
- tasks that do not require both human and AI interpretability

## Core Principles

### 1. Natural Integration Over Bolted Additions

**Rule:** Fixes and new requirements must integrate into existing document structure, not be added at the top or bottom as separate sections.

**Why:** Bolted-on sections break the narrative flow, confuse both humans and AI agents about where requirements actually belong, and make the document harder to navigate during implementation.

### 2. Hierarchy-Aware Placement

**Rule:** Requirements must sit at the correct level in the document hierarchy where they logically belong.

**Why:** AI agents follow hierarchical structure to understand scope and dependencies.

### 3. Dual Readability: Human-First, Agent-Compatible

**Rule:** Write for both human understanding and agent instruction following without creating two separate documents.

### 4. Immutable Source-of-Truth Positioning

**Rule:** Fixes must reference and reinforce existing source-of-truth documents, not replace or override them.

### 5. Progression and Dependency Clarity

**Rule:** Prerequisites, blockers, and dependencies must be explicit in the document structure, not buried in prose.

## Verification Checklist

Before considering a documentation edit complete:

- [ ] Is the fix integrated into the existing doc structure, not bolted on?
- [ ] Does it sit at the correct hierarchy level?
- [ ] Would a human reader understand why and where this requirement matters?
- [ ] Would an AI agent reading this doc follow the requirement correctly?
- [ ] Is mandatory behavior marked with `**Mandatory:**` or explicit "must" language?
- [ ] Does the fix reference or reinforce source-of-truth docs instead of conflicting with them?
- [ ] Are prerequisites and dependencies clear from the document structure?

## Coordination with Other Skills

- **repo-workflow**: Use together when updating instruction surfaces or prompts.
- **skill-map-governance**: Use together if documentation updates require skill changes.
- **verification-gate**: Use together when documentation changes affect execution or safety.
- **workflow-logging**: Use together when documentation changes are material enough to log.
