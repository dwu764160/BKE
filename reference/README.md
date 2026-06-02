# Reference Documentation Map

Legacy deep-dive references and planning docs. For current pipeline docs, see `docs/` instead.

## Canonical root docs

- `BKE_metric_modelling_plan.md` — original BKE modeling plan
- `BKE_metric_modelling_summary.md` — BKE modeling methodology summary
- `BKE_gen1_summary_2026-03-01.md` — Gen 1 BKE state snapshot (2026-03-01)
- `bke_repo_context_summary_for_ai.md` — AI session context summary

## Subfolders

- `archetypes/` — offensive/defensive archetype definitions and role docs
- `data_fetch/` — fetch strategy and fetched-vs-computed source policy
- `modeling/` — modeling notes, trends, and strategy references
- `pipeline/` — reproduction and pipeline operation references (see note below)
- `player_eval/` — player evaluation engine planning docs
- `simulation/` — simulation core planning and summary docs

**Note on `pipeline/data_reproduction.md`:** This is an early-stage planning doc for a manual pipeline reproduction workflow. The comparison manifest uses old uppercase column names (now obsolete after parquet standardization). Treat it as historical context, not a runnable procedure.

## Current schema reference

Column naming and schema rules are now in `docs/reference/data_schemas.md` (canonical, current) and `src/data/schema_contract.py` (implementation). The old `data_schema/` subfolder does not exist.
