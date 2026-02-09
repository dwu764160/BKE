# BKE Copilot Instructions

## Big picture
- This repo is a Python analytics pipeline for NBA possession-level metrics (RAPM/ORAPM/DRAPM) and derived player archetypes. The end-to-end flow is: fetch raw NBA data -> normalize PBP -> derive features/possessions -> compute metrics/archetypes -> export reports.
- Core stages are split by directory: fetching in [src/data_fetch](src/data_fetch), normalization in [src/data_normalize](src/data_normalize), feature derivation in [src/features](src/features), metrics in [src/data_compute](src/data_compute), and exports/utilities in [src/utils](src/utils).
- A typical stage boundary example is: raw PBP in [data/historical](data/historical) -> normalized rows via [src/data_normalize/run_normalization.py](src/data_normalize/run_normalization.py) -> possessions via [src/features/derive_possessions.py](src/features/derive_possessions.py) -> RAPM via [src/data_compute/compute_rapm.py](src/data_compute/compute_rapm.py).

## Developer workflows (from README and scripts)
- Create venv and install deps: `python3 -m venv .venv` then `source .venv/bin/activate` then `python3 -m pip install -r requirements.txt`.
- Full fetch -> normalize -> compute pipeline is documented in [readme.md](readme.md); follow the step order there because later stages assume prior outputs exist.
- Full reproduction run (reroute outputs to a fresh tree): `bash scripts/reproduce_pipeline.sh` which patches scripts to write into `data_temp_reprod/` and logs to `data_temp_reprod/logs/` (see [scripts/reproduce_pipeline.sh](scripts/reproduce_pipeline.sh)).

## Project-specific conventions
- Many scripts hardcode `DATA_DIR = "data/..."` and write directly to repo data folders; downstream steps assume these locations exist.
- NBA API access uses cached headers/sessions in [data/nba_headers.json](data/nba_headers.json) and [data/nba_session.json](data/nba_session.json); fetch scripts depend on them when present.
- Season identifiers are consistently formatted as `YYYY-YY` (example in [src/data_fetch/fetch_historical_data.py](src/data_fetch/fetch_historical_data.py)).

## Key artifacts and outputs
- Raw and normalized PBP live under [data/historical](data/historical); tracking data under [data/tracking](data/tracking); processed outputs under [data/processed](data/processed).
- Archetype logic and thresholds are centralized in [src/data_compute/compute_player_archetypes.py](src/data_compute/compute_player_archetypes.py); update thresholds there rather than scattering constants.

## Tests and validation
- Validation scripts are standalone Python entry points in [tests](tests); a common check is `python3 tests/validate_rapm.py` (see [readme.md](readme.md)).
- Pytest is available for unit checks: `pytest -q`.

## When adding new code
- Prefer adding new pipeline steps as explicit scripts alongside existing stage files and wire them into the documented order in [readme.md](readme.md).
- If your change adds new outputs, document the target folder and filename under the existing data layout in [readme.md](readme.md).
