# Skill Map

This file is the central source of truth for repository skills.

All agents must do these steps before using or editing skills:

1. Read this file first.
2. Load only the smallest sufficient skill set for the task.
3. If any skill is added, removed, renamed, or scope-changed, update this file in the same change.

## Selection Order

### Mandatory (every task)
1. Load [scope-creep-guard](scope-creep-guard/SKILL.md) — enforce phase boundaries before any edits.
2. Load [detailed-chat-output](detailed-chat-output/SKILL.md) — structured output for every task.

### Cross-Cutting (load when applicable)
3. [repo-workflow](repo-workflow/SKILL.md) — any skill/instruction/workflow file changes
4. [skill-map-governance](skill-map-governance/SKILL.md) — skill add/remove/rename/scope change
5. [verification-gate](verification-gate/SKILL.md) — any file edit or config/process change
6. [workflow-logging](workflow-logging/SKILL.md) — material process or instruction updates
7. [documentation-cohesion](documentation-cohesion/SKILL.md) — documentation authoring or planning doc updates
8. [manual-testing-guides](manual-testing-guides/SKILL.md) — manual testing guides, runbooks, validation checklists
9. [remote-commit-logging](remote-commit-logging/SKILL.md) — tasks that add or maintain commit history automation
10. [self-improvement-loop](self-improvement-loop/SKILL.md) — repeated errors, stale docs, avoidable rework
11. [skill-improvement-loop](skill-improvement-loop/SKILL.md) — skill quality issues or missed auto-loading behavior

### BKE Domain (load for BKE-specific tasks)
12. [basketball-knowledge](basketball-knowledge/SKILL.md) — archetype, position-band, or simulation decisions
13. [simulation](simulation/SKILL.md) — simulation pipeline changes, schedule generation, game model, walk-forward validation
14. [audit-model](audit-model/SKILL.md) — modeling/RAPM/BKE layer changes
15. [metric-version-tracker](metric-version-tracker/SKILL.md) — any versioned artifact touched/read/produced (BKE, PTS, archetypes); enforces docs/multiple-versions.md
16. [audit-compute](audit-compute/SKILL.md) — compute stage formula changes
16. [audit-aggregate](audit-aggregate/SKILL.md) — profile aggregation changes
17. [audit-fetch](audit-fetch/SKILL.md) — fetch/ingest changes (user-requested only)
18. [pipeline-integrity](pipeline-integrity/SKILL.md) — after any data fetch/update: verify downstream artifacts, queue re-run chain, sync docs
18b. [forecast-leakage-audit](forecast-leakage-audit/SKILL.md) — before trusting any OOS Brier/CLV or comparing game-level forecast models; rules out temporal leakage vs signal dilution
19. [data-audit](data-audit/SKILL.md) — schema drift or data normalization
20. [docs-sync](docs-sync/SKILL.md) — pipeline step renames or output path changes
21. [self-improvement](self-improvement/SKILL.md) — end-of-session skill quality evaluation (BKE-specific)
22. [skill-curator](skill-curator/SKILL.md) — skill deduplication, overlap detection, catalog maintenance

### Context Engineering (load for context optimization and multi-agent tasks)
> Skills in this section live under `.github/skills/context-engineering/`. Load by full path when needed.

23. [context-fundamentals](context-engineering/context-fundamentals/SKILL.md) — understand context windows, agent design, attention mechanics
24. [context-degradation](context-engineering/context-degradation/SKILL.md) — recognize and avoid context failure patterns
25. [context-compression](context-engineering/context-compression/SKILL.md) — design and evaluate compression strategies
26. [context-optimization](context-engineering/context-optimization/SKILL.md) — apply compaction, masking, caching strategies
27. [memory-systems](context-engineering/memory-systems/SKILL.md) — design short/long-term and graph-based memory
28. [filesystem-context](context-engineering/filesystem-context/SKILL.md) — use filesystems for dynamic context discovery
29. [multi-agent-patterns](context-engineering/multi-agent-patterns/SKILL.md) — orchestrator, peer-to-peer, hierarchical patterns
30. [tool-design](context-engineering/tool-design/SKILL.md) — build tools that agents use effectively
31. [evaluation](context-engineering/evaluation/SKILL.md) — build evaluation frameworks for agent systems
32. [advanced-evaluation](context-engineering/advanced-evaluation/SKILL.md) — LLM-as-a-Judge techniques
33. [project-development](context-engineering/project-development/SKILL.md) — design LLM projects end-to-end
34. [bdi-mental-states](context-engineering/bdi-mental-states/SKILL.md) — transform context into agent mental states (BDI ontology)
35. [latent-briefing](context-engineering/latent-briefing/SKILL.md) — share orchestrator state via KV cache compaction

## Skill Registry

| Skill | Path | Purpose | Load When |
|---|---|---|---|
| scope-creep-guard | [scope-creep-guard/SKILL.md](scope-creep-guard/SKILL.md) | Enforce phase boundaries | Every task |
| detailed-chat-output | [detailed-chat-output/SKILL.md](detailed-chat-output/SKILL.md) | Structured output | Every task |
| repo-workflow | [repo-workflow/SKILL.md](repo-workflow/SKILL.md) | Maintain workflow surfaces | Workflow/instruction maintenance |
| skill-map-governance | [skill-map-governance/SKILL.md](skill-map-governance/SKILL.md) | Keep SKILL_MAP.md current | Any skill catalog change |
| verification-gate | [verification-gate/SKILL.md](verification-gate/SKILL.md) | Verify before completion | Any file edit |
| workflow-logging | [workflow-logging/SKILL.md](workflow-logging/SKILL.md) | Log material changes | Process/instruction updates |
| documentation-cohesion | [documentation-cohesion/SKILL.md](documentation-cohesion/SKILL.md) | Natural doc integration | Creating/refining planning docs |
| manual-testing-guides | [manual-testing-guides/SKILL.md](manual-testing-guides/SKILL.md) | Reproducible test guides | Writing runbooks or test guides |
| remote-commit-logging | [remote-commit-logging/SKILL.md](remote-commit-logging/SKILL.md) | Push-triggered commit log | Commit history automation |
| self-improvement-loop | [self-improvement-loop/SKILL.md](self-improvement-loop/SKILL.md) | Fix recurring mistakes | Repeated errors or stale docs |
| skill-improvement-loop | [skill-improvement-loop/SKILL.md](skill-improvement-loop/SKILL.md) | Improve skill auto-loading | Skill quality issues |
| basketball-knowledge | [basketball-knowledge/SKILL.md](basketball-knowledge/SKILL.md) | Domain: archetype, position, simulation | Archetype/position/forecast decisions |
| simulation | [simulation/SKILL.md](simulation/SKILL.md) | Simulation + walk-forward validation | Sim/schedule/game-model/CLV/forecast changes |
| audit-model | [audit-model/SKILL.md](audit-model/SKILL.md) | Modeling/RAPM/BKE audit | Modeling changes |
| metric-version-tracker | [metric-version-tracker/SKILL.md](metric-version-tracker/SKILL.md) | Enforce canonical versions across all artifacts | Any versioned metric/artifact touched (BKE, PTS, archetypes) |
| audit-compute | [audit-compute/SKILL.md](audit-compute/SKILL.md) | Compute stage audit | Formula/normalization changes |
| audit-aggregate | [audit-aggregate/SKILL.md](audit-aggregate/SKILL.md) | Profile aggregation audit | Aggregation changes |
| audit-fetch | [audit-fetch/SKILL.md](audit-fetch/SKILL.md) | Fetch/ingest audit | Ingest changes (user-requested only) |
| pipeline-integrity | [pipeline-integrity/SKILL.md](pipeline-integrity/SKILL.md) | Post-fetch downstream validation and doc sync | After any data fetch, backfill, or season update |
| forecast-leakage-audit | [forecast-leakage-audit/SKILL.md](forecast-leakage-audit/SKILL.md) | Temporal-leakage / contamination audit for OOS forecast & CLV | Before trusting an OOS Brier/CLV or comparing game models |
| data-audit | [data-audit/SKILL.md](data-audit/SKILL.md) | Schema and data normalization | Schema drift suspected |
| docs-sync | [docs-sync/SKILL.md](docs-sync/SKILL.md) | Sync docs with code changes | Pipeline step renames/output changes |
| self-improvement | [self-improvement/SKILL.md](self-improvement/SKILL.md) | Session-end skill evaluation | End of interactive session |
| skill-curator | [skill-curator/SKILL.md](skill-curator/SKILL.md) | Skill deduplication/curation | Skill catalog maintenance |
| context-fundamentals | [context-engineering/context-fundamentals/SKILL.md](context-engineering/context-fundamentals/SKILL.md) | Understand context windows and agent design | Context or architecture questions |
| context-degradation | [context-engineering/context-degradation/SKILL.md](context-engineering/context-degradation/SKILL.md) | Recognize context failure patterns | Debugging context issues |
| context-compression | [context-engineering/context-compression/SKILL.md](context-engineering/context-compression/SKILL.md) | Design compression strategies | Session optimization |
| context-optimization | [context-engineering/context-optimization/SKILL.md](context-engineering/context-optimization/SKILL.md) | Apply optimization techniques | Context usage optimization |
| memory-systems | [context-engineering/memory-systems/SKILL.md](context-engineering/memory-systems/SKILL.md) | Memory system architecture | Memory or persistence design |
| filesystem-context | [context-engineering/filesystem-context/SKILL.md](context-engineering/filesystem-context/SKILL.md) | Use filesystems for context | File-based context discovery |
| multi-agent-patterns | [context-engineering/multi-agent-patterns/SKILL.md](context-engineering/multi-agent-patterns/SKILL.md) | Multi-agent coordination | Multi-agent or orchestration |
| tool-design | [context-engineering/tool-design/SKILL.md](context-engineering/tool-design/SKILL.md) | Design effective agent tools | Tool creation or improvement |
| evaluation | [context-engineering/evaluation/SKILL.md](context-engineering/evaluation/SKILL.md) | Build evaluation frameworks | Skill or system evaluation |
| advanced-evaluation | [context-engineering/advanced-evaluation/SKILL.md](context-engineering/advanced-evaluation/SKILL.md) | LLM-as-a-Judge techniques | Comparative skill evaluation |
| project-development | [context-engineering/project-development/SKILL.md](context-engineering/project-development/SKILL.md) | Design LLM projects | LLM project architecture |
| bdi-mental-states | [context-engineering/bdi-mental-states/SKILL.md](context-engineering/bdi-mental-states/SKILL.md) | BDI cognitive ontology | Formal reasoning or beliefs |
| latent-briefing | [context-engineering/latent-briefing/SKILL.md](context-engineering/latent-briefing/SKILL.md) | KV cache optimization | Cache-level context tuning |

## Maintenance Rules

- Keep skills non-feature-specific unless implementation code requires otherwise.
- Keep each skill narrow with explicit use and non-use guidance.
- Prefer updating existing skills over creating near-duplicates.
- Keep paths and links in this map valid.
- Context-engineering skills live under `context-engineering/` subdirectory — update paths here when moving.

## Machine-Readable Index

```yaml
skillMap:
  version: 3
  sourceOfTruth: .github/skills/SKILL_MAP.md
  mandatoryReadFirst: true
  requiredOnChange: true
  mandatorySkills:
    - scope-creep-guard
    - detailed-chat-output
  registry:
    - name: scope-creep-guard
      path: .github/skills/scope-creep-guard/SKILL.md
      type: safety-governance
    - name: detailed-chat-output
      path: .github/skills/detailed-chat-output/SKILL.md
      type: communication
    - name: repo-workflow
      path: .github/skills/repo-workflow/SKILL.md
      type: meta-workflow
    - name: skill-map-governance
      path: .github/skills/skill-map-governance/SKILL.md
      type: governance
    - name: verification-gate
      path: .github/skills/verification-gate/SKILL.md
      type: validation
    - name: workflow-logging
      path: .github/skills/workflow-logging/SKILL.md
      type: logging
    - name: documentation-cohesion
      path: .github/skills/documentation-cohesion/SKILL.md
      type: documentation-quality
    - name: manual-testing-guides
      path: .github/skills/manual-testing-guides/SKILL.md
      type: documentation-quality
    - name: remote-commit-logging
      path: .github/skills/remote-commit-logging/SKILL.md
      type: logging-automation
    - name: self-improvement-loop
      path: .github/skills/self-improvement-loop/SKILL.md
      type: maintenance
    - name: skill-improvement-loop
      path: .github/skills/skill-improvement-loop/SKILL.md
      type: evaluation
    - name: basketball-knowledge
      path: .github/skills/basketball-knowledge/SKILL.md
      type: domain
    - name: simulation
      path: .github/skills/simulation/SKILL.md
      type: domain
    - name: audit-model
      path: .github/skills/audit-model/SKILL.md
      type: domain-audit
    - name: metric-version-tracker
      path: .github/skills/metric-version-tracker/SKILL.md
      type: domain-audit
    - name: audit-compute
      path: .github/skills/audit-compute/SKILL.md
      type: domain-audit
    - name: audit-aggregate
      path: .github/skills/audit-aggregate/SKILL.md
      type: domain-audit
    - name: audit-fetch
      path: .github/skills/audit-fetch/SKILL.md
      type: domain-audit
    - name: pipeline-integrity
      path: .github/skills/pipeline-integrity/SKILL.md
      type: domain-audit
    - name: forecast-leakage-audit
      path: .github/skills/forecast-leakage-audit/SKILL.md
      type: domain-audit
    - name: data-audit
      path: .github/skills/data-audit/SKILL.md
      type: domain-audit
    - name: docs-sync
      path: .github/skills/docs-sync/SKILL.md
      type: documentation-quality
    - name: self-improvement
      path: .github/skills/self-improvement/SKILL.md
      type: maintenance
    - name: skill-curator
      path: .github/skills/skill-curator/SKILL.md
      type: governance
    - name: context-fundamentals
      path: .github/skills/context-engineering/context-fundamentals/SKILL.md
      type: context-engineering
    - name: context-degradation
      path: .github/skills/context-engineering/context-degradation/SKILL.md
      type: context-engineering
    - name: context-compression
      path: .github/skills/context-engineering/context-compression/SKILL.md
      type: context-engineering
    - name: context-optimization
      path: .github/skills/context-engineering/context-optimization/SKILL.md
      type: context-engineering
    - name: memory-systems
      path: .github/skills/context-engineering/memory-systems/SKILL.md
      type: context-engineering
    - name: filesystem-context
      path: .github/skills/context-engineering/filesystem-context/SKILL.md
      type: context-engineering
    - name: multi-agent-patterns
      path: .github/skills/context-engineering/multi-agent-patterns/SKILL.md
      type: context-engineering
    - name: tool-design
      path: .github/skills/context-engineering/tool-design/SKILL.md
      type: context-engineering
    - name: evaluation
      path: .github/skills/context-engineering/evaluation/SKILL.md
      type: context-engineering
    - name: advanced-evaluation
      path: .github/skills/context-engineering/advanced-evaluation/SKILL.md
      type: context-engineering
    - name: project-development
      path: .github/skills/context-engineering/project-development/SKILL.md
      type: context-engineering
    - name: bdi-mental-states
      path: .github/skills/context-engineering/bdi-mental-states/SKILL.md
      type: context-engineering
    - name: latent-briefing
      path: .github/skills/context-engineering/latent-briefing/SKILL.md
      type: context-engineering
```
