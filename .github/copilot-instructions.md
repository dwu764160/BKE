
# BKE Copilot Instructions (2026)

## Big Picture & Philosophy
- This repo is a modular Python analytics pipeline for NBA possession-level metrics (RAPM/ORAPM/DRAPM) and derived player archetypes.
- The pipeline is strictly ordered: fetch raw NBA data → normalize PBP → derive features/possessions → compute metrics/archetypes → export reports.
- **Archetype = what you are asked to do; Impact = how well you do it.** These must be separated. Never mix role and value in the same stage.
- All logic for role assignment (archetype) must be behavior-based, not impact-based. Effectiveness/impact overlays are applied only after role assignment.
- Rarity and gating: Use hard percentile gates and structural constraints to prevent role inflation (e.g., POA/Versatile should be rare, not just high-minute or high-difficulty).

## Pipeline Structure & Stage Boundaries
- Core stages are split by directory:
  - Fetching: [src/data_fetch](src/data_fetch)
  - Normalization: [src/data_normalize](src/data_normalize)
  - Feature derivation: [src/features](src/features)
  - Metrics/archetypes: [src/data_compute](src/data_compute)
  - Modeling (impact models): [src/modeling](src/modeling)
  - Exports/utilities: [src/utils](src/utils)
- Example stage boundary: raw PBP in [data/historical] → normalized rows via [src/data_normalize/run_normalization.py] → possessions via [src/features/derive_possessions.py] → RAPM via [src/modeling/compute_rapm.py].
- **Every time you finish a file in the pipeline, you must update [readme.md](readme.md) to reflect the correct pipeline order and any new/changed outputs.**

## Loop Folder: Context, Planning, and Progress
The [loop](loop/) folder is your live context and planning hub. You must:
  - Refer to [loop/context_summary.txt](loop/context_summary.txt) for current project state and context.
  - Consult [loop/current_phase_plan.DO_NOT_CHANGE.txt](loop/current_phase_plan.DO_NOT_CHANGE.txt) and [loop/overall_plan.DO_NOT_CHANGE.txt](loop/overall_plan.DO_NOT_CHANGE.txt) for phase and overall plans.
  - Update these files as you progress, especially when you change pipeline logic, add new steps, or adjust thresholds.
  - Use [loop/in_progress_context.txt](loop/in_progress_context.txt) to track what you are actively working on and any open questions.
  - **Whenever you are running low on context or clarity, immediately remind the user and update [loop/context_summary.txt](loop/context_summary.txt) or [loop/in_progress_context.txt](loop/in_progress_context.txt) to reflect what is missing or unclear.**

## Copilot Instructions File Usage
- **You must refer to this instructions file every time you run or make a major decision.**
- If you are unsure about workflow, conventions, or next steps, always check this file and the loop folder before proceeding.

## Developer Workflow (Strict Order)
1. **Environment setup:**
	- `python3 -m venv .venv`
	- `source .venv/bin/activate`
	- `python3 -m pip install -r requirements.txt`
2. **Pipeline execution:**
	- Always follow the pipeline order in [readme.md](readme.md). Never skip steps; later stages assume prior outputs exist.
	- After editing or adding any pipeline file, immediately update [readme.md](readme.md) to reflect the new/changed step in the correct order.
	- If you add a new output, document its folder and filename in the data layout section of [readme.md](readme.md).
3. **Loop folder usage:**
	- Constantly refer to and update the loop folder for context, planning, and progress tracking.
	- Summarize major changes and rationale in [loop/context_summary.txt](loop/context_summary.txt).
	- Update phase/overall plans if you change the pipeline structure or logic.
4. **Reproducibility:**
	- Use `bash scripts/reproduce_pipeline.sh` for full reproduction runs (outputs to `data_temp_reprod/`).
	- Ensure all scripts are reproducible and do not rely on unstated state.

## Project Conventions & Best Practices
- Hardcode `DATA_DIR = "data/..."` only if necessary; downstream steps must assume these locations exist.
- Use cached NBA API headers/sessions in [data/nba_headers.json] and [data/nba_session.json] when present.
- Season identifiers must be formatted as `YYYY-YY` (e.g., `2023-24`).
- **Centralize all logic and thresholds** (e.g., archetype cutoffs) in a single script per stage (e.g., [src/data_compute/compute_player_archetypes.py]). Never scatter constants across files.
- When adding new code:
  - Add new pipeline steps as explicit scripts alongside existing stage files.
  - Wire new steps into the documented order in [readme.md](readme.md) and update the data layout section.
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
- After any change, update [readme.md](readme.md) and the loop folder to reflect the new logic and rationale.

## Summary Checklist
- [ ] Always update [readme.md](readme.md) in the correct pipeline order after any pipeline file change.
- [ ] Constantly update and refer to the loop folder for context, planning, and progress.
- [ ] Centralize logic and thresholds; never scatter constants.
- [ ] Document all new outputs and their locations in [readme.md](readme.md).
- [ ] Validate and test after every change; update the loop folder with findings.

**If in doubt, refer to the loop folder and [readme.md](readme.md) for the latest project state and required workflow.**
