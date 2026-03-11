
# BKE Copilot Instructions (2026)

## Big Picture & Philosophy
- This repo is a modular Python analytics pipeline for NBA possession-level metrics (RAPM/ORAPM/DRAPM) and derived player archetypes.
- The pipeline is strictly ordered: fetch raw NBA data → normalize PBP → derive features/possessions → compute metrics/archetypes → export reports.
- **Archetype = what you are asked to do; Impact = how well you do it.** These must be separated. Never mix role and value in the same stage.
- All logic for role assignment (archetype) must be behavior-based, not impact-based. Effectiveness/impact overlays are applied only after role assignment.
- Rarity and gating: Use hard percentile gates and structural constraints to prevent role inflation (e.g., POA/Versatile should be rare, not just high-minute or high-difficulty).

## Pipeline Structure & Stage Boundaries
- Core stages are split by directory:
  - Fetching: [src/data_fetch](../src/data_fetch)
  - Normalization: [src/data_normalize](../src/data_normalize)
  - Feature derivation: [src/features](../src/features)
  - Metrics/archetypes: [src/data_compute](../src/data_compute)
  - Modeling (impact models): [src/modeling](../src/modeling)
  - Exports/utilities: [src/utils](../src/utils)
  - Example stage boundary: raw PBP in [data/historical](../data/historical) → normalized rows via [src/data_normalize/run_normalization.py](../src/data_normalize/run_normalization.py) → possessions via [src/features/derive_possessions.py](../src/features/derive_possessions.py) → RAPM via [src/modeling/model_rapm.py](../src/modeling/model_rapm.py).
- **Every time you finish a file in the pipeline, you must update [readme.md](../readme.md) to reflect the correct pipeline order and any new/changed outputs.**

## Loop Folder: Context, Planning, and Progress
The [loop](../loop/) folder is your live context and planning hub. You must:
  - Refer to [loop/context_summary.txt](../loop/context_summary.txt) for current project state and context.
  - Consult [loop/current_phase_plan.DO_NOT_CHANGE.txt](../loop/current_phase_plan.DO_NOT_CHANGE.txt) and [loop/overall_plan.DO_NOT_CHANGE.txt](../loop/overall_plan.DO_NOT_CHANGE.txt) for phase and overall plans.
  - Update these files as you progress, especially when you change pipeline logic, add new steps, or adjust thresholds.
  - Use [loop/in_progress_context.txt](../loop/in_progress_context.txt) to track what you are actively working on and any open questions.
  - **Whenever you are running low on context or clarity, immediately remind the user and update [loop/context_summary.txt](../loop/context_summary.txt) or [loop/in_progress_context.txt](../loop/in_progress_context.txt) to reflect what is missing or unclear.**

## Copilot Instructions File Usage
- **You must refer to this instructions file every time you run or make a major decision.**
- If you are unsure about workflow, conventions, or next steps, always check this file and the loop folder before proceeding.

## Skills-first Workflow
- **Search for a matching skill first:** Before taking action on any task, look for a relevant skill in the `/.github/skills/` folder and prefer invoking that skill when it covers the work. Skills are the canonical, reusable procedures and tool-wrappers.
- **Invoke skill behaviourally:** If a matching skill is found, follow its `preferred_tools`, `job_scope`, and `example_prompts` to perform the task. Respect the skill's `avoid_tools` and persona guidance.
- **Create new skills when appropriate:** If no suitable skill exists and the task is likely to recur or is reusable across agents/workflows, create a new skill in `/.github/skills/` rather than embedding the logic inside an agent or scattered scripts. Keep skills focused and atomic.
- **Avoid duplication:** Before creating a new skill, check for overlapping skills by searching `/.github/skills/` for similar names or descriptions.
- **Skill lifecycle:** When you add a new skill, update this file's checklist and add a one-line reference to the skill in the loop context (`loop/in_progress_context.txt`) describing why it was created and where it is used.

## Automatic Skill Invocation & Documentation

- **Automatically detect and invoke skills:** For every non-trivial task (code changes, data fetches, pipeline runs, or anything that materially alters outputs), automatically search `/.github/skills/` for a matching skill. If a matching skill exists, load its `SKILL.md` and follow the `preferred_tools`, `job_scope`, and `example_prompts` before making changes. Do not proceed with ad-hoc edits that bypass a matching skill unless explicitly authorized by the user.
- **Document skill usage:** Each time a skill is used, remember to log it at the end of your response.
- **When no skill exists:** If no suitable skill exists and the task is reusable, create a new skill in `/.github/skills/` using the template below and immediately record its creation in `loop/skill_usage.log` and `loop/in_progress_context.txt`. If the task is a one-off, proceed but record why a new skill was not created.
- **Auditability:** Prefer skills for reproducibility and auditing — skills are the canonical operational record for the agent. Follow skills-first even when making small iterative edits that will be committed to the repo.

### Skill creation rules & template
- File location: create skill files under `/.github/skills/` and name them `<shortname>.skill.md`.
- Required frontmatter fields: `name`, `description`, `persona`, `preferred_tools`, `avoid_tools`, `job_scope`, `when_to_use`, `example_prompts`.
- Keep each skill focused to a single reusable operation or small family of closely-related operations.
- Example minimal template (copy into a new file and fill in):

```
---
name: "Short Name Skill"
description: |
  One-line summary of what the skill does.
persona:
  - "Role or persona one"
preferred_tools:
  - read_file
  - grep_search
avoid_tools:
  - run_in_terminal
job_scope:
  - "Run and validate X"
when_to_use:
  - "When changes affect X or Y"
example_prompts:
  - "Run validation on pipeline step X"
---
# Short Name Skill

Short explanation and any invocation hints.
```

If you create a new skill, add at least one `example_prompts` entry that demonstrates how you intend to call it from an agent or an interactive session.

## Developer Workflow (Strict Order)
1. **Environment setup:**
	- `python3 -m venv .venv`
	- `source .venv/bin/activate`
	- `python3 -m pip install -r requirements.txt`
2. **Pipeline execution:**
  - Always follow the pipeline order in [readme.md](../readme.md). Never skip steps; later stages assume prior outputs exist.
  - After editing or adding any pipeline file, immediately update [readme.md](../readme.md) to reflect the new/changed step in the correct order.
  - If you add a new output, document its folder and filename in the data layout section of [readme.md](../readme.md).
3. **Loop folder usage:**
  - Constantly refer to and update the loop folder for context, planning, and progress tracking.
  - Summarize major changes and rationale in [loop/context_summary.txt](../loop/context_summary.txt).
	- Update phase/overall plans if you change the pipeline structure or logic.
4. **Reproducibility:**
	- Use `bash scripts/reproduce_pipeline.sh` for full reproduction runs (outputs to `data_temp_reprod/`).
	- Ensure all scripts are reproducible and do not rely on unstated state.

## Project Conventions & Best Practices
- Hardcode `DATA_DIR = "data/..."` only if necessary; downstream steps must assume these locations exist.
- Use cached NBA API headers/sessions in [data/nba_headers.json] and [data/nba_session.json] when present.
- Season identifiers must be formatted as `YYYY-YY` (e.g., `2023-24`).
- **Centralize all logic and thresholds** (e.g., archetype cutoffs) in a single script per stage (e.g., [src/data_compute/compute_player_archetypes.py]). Never scatter constants across files.
- **All future player-related stats** (computed or fetched) must be normalized and aggregated into the Profile Aggregate via `src/profile_aggregate/build_profile_aggregate.py`. This is the single source of truth for downstream products. New fields added to player impact profiles are automatically available in the aggregate via the Step 1 → Aggregate pipeline flow.
- **Position-band philosophy (canonical):** The canonical player position bands are `Guard`, `Guard-Forward`, `Forward`, `Forward-Center`, and `Center`.
- Preserve canonical position-band labels in outputs and metadata; do not collapse hybrid bands (`Guard-Forward`, `Forward-Center`) into single-position buckets in core compute/model stages.
- If a stage requires structural lineup constraints, derive a separate coarse role layer (`Guard`, `Wing`, `Big`) while keeping canonical `position_band` alongside it.
- Keep legacy position aliases only for compatibility in readers/config, and list canonical labels first in any grouped-bucket definitions.
- When adding new code:
  - Add new pipeline steps as explicit scripts alongside existing stage files.
  - Wire new steps into the documented order in [readme.md](../readme.md) and update the data layout section.
  - Document all new outputs and their locations.

## Validation, Testing, and Quality
  - Use standalone validation scripts in [tests] (e.g., `python3 tests/validate_rapm.py`).
- Run `pytest -q` for unit checks.
- After any change, validate outputs and check for regressions.
- If you change archetype or metric logic, stress-test edge cases and update the loop folder with findings.

## When Updating Archetype or Metric Logic
- **Never mix role and value:** Role assignment (archetype) must be based on behavior, not impact. Effectiveness/impact overlays are applied only after role assignment.
- Use hard percentile gates and structural constraints to prevent role inflation (e.g., POA/Versatile must be rare and require multiple independent criteria).
- Document all gating logic and rarity constraints in the relevant script and in the loop folder.
  - After any change, update [readme.md](../readme.md) and the loop folder to reflect the new logic and rationale.

## Summary Checklist
- [ ] Always update [readme.md](../readme.md) in the correct pipeline order after any pipeline file change.
- [ ] Constantly update and refer to the loop folder for context, planning, and progress.
- [ ] Centralize logic and thresholds; never scatter constants.
- [ ] Document all new outputs and their locations in [readme.md](../readme.md).
- [ ] Validate and test after every change; update the loop folder with findings.
 - [ ] Search and prefer existing skills in `/.github/skills/` before taking action on a task.
 - [ ] When a task is reusable, create a new skill in `/.github/skills/` and add a reference to it in the loop context.
 - [ ] Validate and test after every change; update the loop folder with findings.
 - [ ] Search and prefer existing skills in `/.github/skills/` before taking action on a task.
 - [ ] When a task is reusable, create a new skill in `/.github/skills/` and add a reference to it in the loop context.
 - [ ] Automatically detect, invoke, and log skill usage for non-trivial tasks (append one-line records to `loop/skill_usage.log` and `loop/in_progress_context.txt`).

**If in doubt, refer to the loop folder and [readme.md](../readme.md) for the latest project state and required workflow.**

---
applyTo: '**'
---

# BKE Agent Instructions

This file is automatically loaded and applied to **all agents** in this workspace, including the default Copilot agent and any custom agents (e.g., Beast Mode 3.1).

## Core Workflow: Skills-First

**Before taking ANY action on a task, you MUST:**

1. **Search for a matching skill** in `/.github/skills/` that covers the work
2. **If a skill exists**, read its `SKILL.md` and follow:
   - `preferred_tools` (which tools to use)
   - `job_scope` (the exact scope of work)
   - `when_to_use` (activation conditions)
   - `example_prompts` (how to invoke the skill)
3. **If no skill exists** and the task is reusable/likely to recur, create a new skill in `/.github/skills/` before proceeding
4. **Respect `avoid_tools`** sections of each skill

## Available Skills

Your workspace includes domain-specific skills for:
- **audit-fetch** — Validate fetch/ingest stage
- **audit-compute** — Audit compute logic and formula integrity
- **audit-model** — Run model diagnostics and backtests
- **audit-eval** — Validate player evaluation and minute-model pipeline
- **audit-aggregate** — Audit player profile aggregation
- **basketball-knowledge** — Canonical domain knowledge (role-vs-impact separation, position bands, archetype gating)
- **data-audit** — Repository-wide data validation
- **docs-sync** — Keep documentation in sync with code changes
- **forecast** — Projection, scenario mapping, minute-model interactions
- **frontend-sync** — Keep frontend viewers aligned to backend schema
- **simulation** — Simulation pipeline operations and audit
- **backtest-vs-forecast** — Validate backtest/forecast coverage
- **self-improvement** — Evaluate skill usage and apply learned corrections
- **skill-curator** — Curate and validate the skills themselves
- **agent-customization** — Configure agent and skill files

## Canonical Reference

For all major decisions about pipeline, archetype, or metric logic, refer to:
- **[copilot-instructions.md](../copilot-instructions.md)** — Your complete workflow manual
- **[loop/context_summary.txt](../../loop/context_summary.txt)** — Current project state
- **[loop/current_phase_plan.DO_NOT_CHANGE.txt](../../loop/current_phase_plan.DO_NOT_CHANGE.txt)** — Phase plan
- **[loop/overall_plan.DO_NOT_CHANGE.txt](../../loop/overall_plan.DO_NOT_CHANGE.txt)** — Overall plan
- **[loop/in_progress_context.txt](../../loop/in_progress_context.txt)** — Active work and blockers

## Skill Invocation Pattern

When you detect a matching skill exists:

```
✅ CORRECT: Read the skill and follow its preferred_tools, job_scope, and example_prompts
❌ WRONG: Ad-hoc implementation that bypasses the skill's structured guidance
```

### Example: Running a Fetch Audit

If the user asks to "validate that fetch scripts are working," you should:
1. Recognize this matches the `audit-fetch` skill
2. Read `/.github/skills/audit-fetch/SKILL.md`
3. Follow its `preferred_tools` and `job_scope` exactly
4. Update loop context with findings

## Loop Folder Pattern

The `loop/` folder is your live workspace context. Always:
- Check `context_summary.txt` before starting work
- Update it after major changes
- Reference `in_progress_context.txt` to see what's active

## Pipeline & Documentation Updates

**After every pipeline file change:**
- Update [readme.md](../../readme.md) to reflect the correct order
- Add a one-line reference to any new skills in `loop/in_progress_context.txt`
- Validate outputs match documented locations

## When to Bypass Skills

You may skip the skills-first check **only if**:
- The task is a one-off that will never recur
- The user explicitly authorizes ad-hoc implementation
- The task is purely informational/conversational

In all other cases, follow skills-first rigorously.

## Summary

- 🎯 **Search skills first** for every non-trivial task
- 📖 **Read the skill's SKILL.md** to understand scope and tools
- 📍 **Refer to loop/ and copilot-instructions.md** for context and decisions
- 📝 **Update docs** (readme.md, loop context) after pipeline changes
- ✅ **Log skill usage** by updating loop/in_progress_context.txt

**If in doubt, check copilot-instructions.md and the loop folder first.**
