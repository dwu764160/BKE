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
13. [simulation](simulation/SKILL.md) — simulation pipeline changes, schedule generation, game model
14. [forecast](forecast/SKILL.md) — projection, lineup, or minute model changes
15. [audit-model](audit-model/SKILL.md) — modeling/RAPM/BKE changes
16. [audit-compute](audit-compute/SKILL.md) — compute stage formula changes
17. [audit-eval](audit-eval/SKILL.md) — player eval or minute model changes
18. [audit-aggregate](audit-aggregate/SKILL.md) — profile aggregation changes
19. [audit-fetch](audit-fetch/SKILL.md) — fetch/ingest changes (user-requested only)
20. [pipeline-integrity](pipeline-integrity/SKILL.md) — after any data fetch/update: verify downstream artifacts, queue re-run chain, sync docs
21. [backtest-vs-forecast](backtest-vs-forecast/SKILL.md) — ensuring both backtest and forecast coverage
21. [data-audit](data-audit/SKILL.md) — schema drift or data normalization
22. [docs-sync](docs-sync/SKILL.md) — pipeline step renames or output path changes
23. [frontend-sync](frontend-sync/SKILL.md) — backend schema changes affecting viewers
24. [self-improvement](self-improvement/SKILL.md) — end-of-session skill quality evaluation (BKE-specific)
25. [skill-curator](skill-curator/SKILL.md) — skill deduplication, overlap detection, catalog maintenance

### Context Engineering (load for context optimization and multi-agent tasks)

26. [context-fundamentals](context-fundamentals/SKILL.md) — understand context windows, agent design, attention mechanics
27. [context-degradation](context-degradation/SKILL.md) — recognize and avoid context failure patterns
28. [context-compression](context-compression/SKILL.md) — design and evaluate compression strategies
29. [context-optimization](context-optimization/SKILL.md) — apply compaction, masking, caching strategies
30. [memory-systems](memory-systems/SKILL.md) — design short/long-term and graph-based memory
31. [filesystem-context](filesystem-context/SKILL.md) — use filesystems for dynamic context discovery
32. [multi-agent-patterns](multi-agent-patterns/SKILL.md) — orchestrator, peer-to-peer, hierarchical patterns
33. [tool-design](tool-design/SKILL.md) — build tools that agents use effectively
34. [evaluation](evaluation/SKILL.md) — build evaluation frameworks for agent systems
35. [advanced-evaluation](advanced-evaluation/SKILL.md) — LLM-as-a-Judge techniques
36. [project-development](project-development/SKILL.md) — design LLM projects end-to-end
37. [bdi-mental-states](bdi-mental-states/SKILL.md) — transform context into agent mental states (BDI ontology)
38. [hosted-agents](hosted-agents/SKILL.md) — build background agents with sandboxed VMs
39. [latent-briefing](latent-briefing/SKILL.md) — share orchestrator state via KV cache compaction

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
| simulation | [simulation/SKILL.md](simulation/SKILL.md) | Simulation pipeline operations | Simulation/schedule/game model changes |
| forecast | [forecast/SKILL.md](forecast/SKILL.md) | Projection and minute model | Forecast/lineup/projection changes |
| audit-model | [audit-model/SKILL.md](audit-model/SKILL.md) | Modeling/RAPM/BKE audit | Modeling changes |
| audit-compute | [audit-compute/SKILL.md](audit-compute/SKILL.md) | Compute stage audit | Formula/normalization changes |
| audit-eval | [audit-eval/SKILL.md](audit-eval/SKILL.md) | Player eval/minute model audit | Player eval changes |
| audit-aggregate | [audit-aggregate/SKILL.md](audit-aggregate/SKILL.md) | Profile aggregation audit | Aggregation changes |
| audit-fetch | [audit-fetch/SKILL.md](audit-fetch/SKILL.md) | Fetch/ingest audit | Ingest changes (user-requested only) |
| pipeline-integrity | [pipeline-integrity/SKILL.md](pipeline-integrity/SKILL.md) | Post-fetch downstream validation and doc sync | After any data fetch, backfill, or season update |
| backtest-vs-forecast | [backtest-vs-forecast/SKILL.md](backtest-vs-forecast/SKILL.md) | Backtest/forecast coverage | Preparing forecasts |
| data-audit | [data-audit/SKILL.md](data-audit/SKILL.md) | Schema and data normalization | Schema drift suspected |
| docs-sync | [docs-sync/SKILL.md](docs-sync/SKILL.md) | Sync docs with code changes | Pipeline step renames/output changes |
| frontend-sync | [frontend-sync/SKILL.md](frontend-sync/SKILL.md) | Sync viewers to backend | Backend schema changes |
| self-improvement | [self-improvement/SKILL.md](self-improvement/SKILL.md) | Session-end skill evaluation | End of interactive session |
| skill-curator | [skill-curator/SKILL.md](skill-curator/SKILL.md) | Skill deduplication/curation | Skill catalog maintenance |
| context-fundamentals | [context-fundamentals/SKILL.md](context-fundamentals/SKILL.md) | Understand context windows and agent design | Context or architecture questions |
| context-degradation | [context-degradation/SKILL.md](context-degradation/SKILL.md) | Recognize context failure patterns | Debugging context issues |
| context-compression | [context-compression/SKILL.md](context-compression/SKILL.md) | Design compression strategies | Session optimization |
| context-optimization | [context-optimization/SKILL.md](context-optimization/SKILL.md) | Apply optimization techniques | Context usage optimization |
| memory-systems | [memory-systems/SKILL.md](memory-systems/SKILL.md) | Memory system architecture | Memory or persistence design |
| filesystem-context | [filesystem-context/SKILL.md](filesystem-context/SKILL.md) | Use filesystems for context | File-based context discovery |
| multi-agent-patterns | [multi-agent-patterns/SKILL.md](multi-agent-patterns/SKILL.md) | Multi-agent coordination | Multi-agent or orchestration |
| tool-design | [tool-design/SKILL.md](tool-design/SKILL.md) | Design effective agent tools | Tool creation or improvement |
| evaluation | [evaluation/SKILL.md](evaluation/SKILL.md) | Build evaluation frameworks | Skill or system evaluation |
| advanced-evaluation | [advanced-evaluation/SKILL.md](advanced-evaluation/SKILL.md) | LLM-as-a-Judge techniques | Comparative skill evaluation |
| project-development | [project-development/SKILL.md](project-development/SKILL.md) | Design LLM projects | LLM project architecture |
| bdi-mental-states | [bdi-mental-states/SKILL.md](bdi-mental-states/SKILL.md) | BDI cognitive ontology | Formal reasoning or beliefs |
| hosted-agents | [hosted-agents/SKILL.md](hosted-agents/SKILL.md) | Sandboxed background agents | Background agent implementation |
| latent-briefing | [latent-briefing/SKILL.md](latent-briefing/SKILL.md) | KV cache optimization | Cache-level context tuning |

## Maintenance Rules

- Keep skills non-feature-specific unless implementation code requires otherwise.
- Keep each skill narrow with explicit use and non-use guidance.
- Prefer updating existing skills over creating near-duplicates.
- Keep paths and links in this map valid.

## Machine-Readable Index

```yaml
skillMap:
  version: 2
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
    - name: forecast
      path: .github/skills/forecast/SKILL.md
      type: domain
    - name: audit-model
      path: .github/skills/audit-model/SKILL.md
      type: domain-audit
    - name: audit-compute
      path: .github/skills/audit-compute/SKILL.md
      type: domain-audit
    - name: audit-eval
      path: .github/skills/audit-eval/SKILL.md
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
    - name: backtest-vs-forecast
      path: .github/skills/backtest-vs-forecast/SKILL.md
      type: domain
    - name: data-audit
      path: .github/skills/data-audit/SKILL.md
      type: domain-audit
    - name: docs-sync
      path: .github/skills/docs-sync/SKILL.md
      type: documentation-quality
    - name: frontend-sync
      path: .github/skills/frontend-sync/SKILL.md
      type: domain
    - name: self-improvement
      path: .github/skills/self-improvement/SKILL.md
      type: maintenance
    - name: skill-curator
      path: .github/skills/skill-curator/SKILL.md
      type: governance
    - name: context-fundamentals
      path: .github/skills/context-fundamentals/SKILL.md
      type: context-engineering
    - name: context-degradation
      path: .github/skills/context-degradation/SKILL.md
      type: context-engineering
    - name: context-compression
      path: .github/skills/context-compression/SKILL.md
      type: context-engineering
    - name: context-optimization
      path: .github/skills/context-optimization/SKILL.md
      type: context-engineering
    - name: memory-systems
      path: .github/skills/memory-systems/SKILL.md
      type: context-engineering
    - name: filesystem-context
      path: .github/skills/filesystem-context/SKILL.md
      type: context-engineering
    - name: multi-agent-patterns
      path: .github/skills/multi-agent-patterns/SKILL.md
      type: context-engineering
    - name: tool-design
      path: .github/skills/tool-design/SKILL.md
      type: context-engineering
    - name: evaluation
      path: .github/skills/evaluation/SKILL.md
      type: context-engineering
    - name: advanced-evaluation
      path: .github/skills/advanced-evaluation/SKILL.md
      type: context-engineering
    - name: project-development
      path: .github/skills/project-development/SKILL.md
      type: context-engineering
    - name: bdi-mental-states
      path: .github/skills/bdi-mental-states/SKILL.md
      type: context-engineering
    - name: hosted-agents
      path: .github/skills/hosted-agents/SKILL.md
      type: context-engineering
    - name: latent-briefing
      path: .github/skills/latent-briefing/SKILL.md
      type: context-engineering
```
