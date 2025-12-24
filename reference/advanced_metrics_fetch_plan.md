# Advanced Metrics Fetch & Implementation Plan (Integrated)

This document integrates the project's original evaluation plan with a prioritized, executable implementation roadmap for acquiring and computing advanced player/team metrics. The assistant's prioritized plan is authoritative; original notes are preserved where relevant.

## High-level goals
- Acquire high-value tracking/playtype/advanced metrics (drives, playtypes, shot splits, on/off, lineup impacts).
- Compute local proxies for advanced metrics derived from box scores (USG%, TS%, eFG%, AST%, OREB%, per-36, possessions).
- Fetch authoritative season aggregates (BPM, VORP, Win Shares) from Basketball-Reference.
- Build feature pipelines (rolling windows, opponent adjustments, rest/home/back-to-back) and a dataset for modeling.

---

## Prioritized Implementation Plan (short)
1. Foundational (days 0–2)
   - Compute and persist simple situational features: `rest_days`, `home_flag`, `back_to_back` (script: `src/features/compute_rest_home_back2back.py`).
   - Scaffold rolling-window feature builder (3/7/14 games) for player and team: `src/features/build_rolling_features.py`.
2. Data acquisition (days 1–4)
   - Fetch play-by-play for seasons via `nba_api.playbyplayv2` → `data/historical/play_by_play_{season}.parquet` (`src/data_fetch/fetch_play_by_play.py`).
   - Fetch shot logs / shotchartdetail per player/game via `nba_api` → `data/historical/shot_logs.parquet` (`src/data_fetch/fetch_shot_chart.py`).
3. Playtype & shot-type features (days 3–7)
   - Use shot logs and play-by-play to compute: catch-and-shoot, pull-up, rim/mid/3 frequency and PPP; drives per game and drives PPP; transition rates.
   - Output: `data/features/player_shot_playtype_features.parquet`.
4. Lineup / on-off and team ratings (days 4–10)
   - Parse play-by-play substitutions to construct possessions and active lineups; compute on/off per 100 possessions and team ORtg/DRtg/Pace. (`src/features/build_lineups_onoff.py`, `src/features/compute_team_ratings.py`).
5. Advanced modeling (weeks 2–4)
   - RAPM-style ridge regression on possessions × lineup matrix to estimate player offensive/defensive impacts. (`src/models/compute_rapm.py`).
6. External enrichment (parallel)
   - Scrape Basketball-Reference for BPM/VORP/Win Shares and merge into player-season table (`src/data_fetch/fetch_bref_advanced.py`).

---

## Concrete Scripts to Add (first-pass)
- `src/features/compute_rest_home_back2back.py` — compute rest/home/back-to-back flags and write `data/features/game_context.parquet`.
- `src/features/build_rolling_features.py` — compute rolling aggregates for players and teams and write `data/features/player_rolling.parquet` and `team_rolling.parquet`.
- `src/data_fetch/fetch_play_by_play.py` — fetch `playbyplayv2` via `nba_api` and save per-season parquet.
- `src/data_fetch/fetch_shot_chart.py` — fetch `shotchartdetail` / shot logs and defender-distance buckets.
- `src/features/build_playtype_features.py` — map pbp + shot logs to playtype counts and PPPs.
- `src/features/build_lineups_onoff.py` — create lineup-on/off matrices and compute per-100 metrics.
- `src/data_fetch/fetch_bref_advanced.py` — scrape Basketball-Reference (BPM, VORP, WS) and save per-season CSV/parquet.

---

## Priority and Rationale
- Immediate (high ROI, low effort): rest/home/back-to-back, rolling features, team ratings (Pace/Net), Basketball-Reference scrape for BPM/WS.
- Medium (higher effort, high value): shot logs and playtype mapping (unlocks PPP & archetyping).
- Advanced (high effort, highest value): lineup parsing + RAPM (needs pbp and careful validation).

---

## Data storage & outputs
- Persist raw fetched endpoints to `data/historical/` as Parquet.
- Persist features under `data/features/` with descriptive filenames.
- Add simple manifest `data/manifest.yml` (optional) to track file provenance and generation timestamps.

---

## Next actionable choice for me
Pick one to start and I will scaffold it and run quick local validations (I will not execute in your venv; you'll run the scripts locally and paste outputs if you want verification). Options:
- A: Scaffold `compute_rest_home_back2back.py` + `build_rolling_features.py` (recommended first).
- B: Scaffold `fetch_play_by_play.py` + `fetch_shot_chart.py` (foundational data fetch).
- C: Scaffold `fetch_bref_advanced.py` to populate BPM/VORP/WS (quick enrichment).

---

Document updated and saved as `reference/advanced_metrics_fetch_plan.md` (assistant-prioritized plan integrated). If you want the original `.docx` updated too I can produce a `.docx` export from the markdown (requires `pandoc`/`pypandoc` in your environment).
