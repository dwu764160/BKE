---
name: basketball-knowledge
description: |
  Canonical basketball domain knowledge for the BKE pipeline: role-vs-impact
  separation, position-band policy, archetype gating principles, validation
  practices, and forecast/simulation assumptions. Intended as the single
  reference other skills and agents consult for consistent domain decisions.
tags:
  - domain
  - basketball
  - archetype
version: "1.0"
last_updated: "2026-03-09"
persona:
  - "Basketball domain expert"
  - "Pipeline domain guardrail"
preferred_tools:
  - read_file
  - grep_search
  - semantic_search
  - file_search
avoid_tools:
  - create_new_workspace
job_scope:
  - "Provide canonical guidance for archetype assignments and gating rules."
  - "Document position-band policy and hybrid handling for compute/model stages."
  - "Recommend validation metrics and where to find canonical reports."
  - "Be imported/queried by agents/skills when domain guidance is required."
when_to_use:
  - "When making decisions that affect archetype assignment, position-band handling, or forecast/simulation assumptions."
example_prompts:
  - "What are the canonical position bands and how should hybrids be treated?"
  - "List archetype gating best practices for role assignment."
---
# Basketball Knowledge — Canonical Philosophies

Core principles
- **Role vs Impact:** Archetype (role) = behavior-only. Do not mix role assignment with impact/effectiveness. Apply impact overlays only after roles are assigned.
- **Centralize thresholds:** All gating cutoffs and thresholds must live in a single compute script or config (e.g., `src/data_compute/compute_player_archetypes.py`).
- **Rarity & gating:** Rare roles require multi-criterion gates and minimum minute/possession thresholds. Use hard percentile gates and structural constraints to avoid role inflation.

Position-band policy (canonical)
- Canonical bands: `Guard`, `Guard-Forward`, `Forward`, `Forward-Center`, `Center`.
- Preserve canonical labels in compute/model outputs; do not collapse hybrid bands during core compute stages.
- For structural constraints (lineups, scheduling), derive a coarse role layer (`Guard`, `Wing`, `Big`) while keeping `position_band` unchanged.

Archetype assignment rules
- Use behavior-based features (play-type rates, shot-creation, defensive matchup metrics, tracking signals) as input axes.
- Gate assignment with: (a) percentile-based thresholds, (b) minimum minutes/possessions, (c) multi-axis concurrence for rare roles (e.g., POA/Versatile requires independent high percentiles on multiple axes).
- Centralize gates in one script; document gates and rationale in `reference/archetypes/` and `loop/`.

BKE metric & modeling summary
- BKE is a possession-level composite that synthesizes ORAPM/DRAPM-like model outputs, possession features, and decomposition layers. Inputs: `data/features/`, `data/tracking/`, official stats.
- Modeling notes: apply shrinkage/compression steps and layered decomposition (portable talent, utilization efficiency, archetype elevation, scheme stability). Validate via cross-season correlations and backtesting.

Validation & diagnostics (recommended)
- Cross-season Pearson correlations (BKE stability).  
- Rank-recall / top-N predictive enrichment and mean_abs_rank_shift.  
- RMSE / MAE of team/season predictions vs realized outcomes.  
- Variance explained and compression diagnostics.  
- Archetype prevalence by minute thresholds to verify rarity gates.

Forecast & simulation guidance
- Minutes projection: blend carry-anchor + age/salary/draft multipliers; use roster-adaptive normalization targets and replacement buffers.  
- Scenarios: support `end_of_season` and `preseason_snapshot` mapping; keep forecast scoring leakage-safe (no clutch-minute leakage).  
- For lineup projection, enforce archetype-minute eligibility and position constraints (require at least one true big in starter lineup when applicable).

Where to find canonical docs
- `readme.md` (pipeline order, position-band policy, high-level conventions)
- `reference/bke_repo_context_summary_for_ai.md` (repo-level summary and modeling plan)
- `reference/archetypes/` (archetype variable definitions and gating notes)
- `src/data_compute/compute_player_archetypes.py` (centralized assignment logic)
- `src/profile_aggregate/build_profile_aggregate.py` (profile aggregation guidance)

Agent usage guidance
- When changing archetype or metric logic: (1) update `readme.md` pipeline order, (2) update `loop/` notes, (3) add tests in `tests/`, (4) run backtest validation and publish a short summary in `reports/`.
