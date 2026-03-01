# Available Stats Inventory

This document lists the raw and derived statistics currently available in the repository and where to find them. Use this as the canonical reference when planning feature engineering and modeling.

## File locations
- Player-game logs: `data/historical/player_game_logs.parquet`
- Team-game logs: `data/historical/team_game_logs.parquet`
- Per-team padded details: `data/historical/team_game_details.parquet` / `.csv`
- Season summaries (team): `data/historical/team_summaries.parquet` / `.csv`
- Derived player-season metrics: `data/advanced_local_metrics.parquet`

---

## Player-level (box score) — available columns
These are the core per-player-per-game columns present in `player_game_logs.parquet`.

- Identifiers / metadata:
  - `GAME_ID` — unique game identifier
  - `GAME_DATE` — date string for the game
  - `MATCHUP` — matchup string (e.g. "TOR vs MIL" or "TOR @ MIL")
  - `SEASON`, `SEASON_ID` — season identifier
  - `PLAYER_ID` — numeric player identifier

- Box-score counting stats:
  - `MIN` — minutes played
  - `FGM`, `FGA` — field goals made / attempted
  - `FG3M`, `FG3A` — 3pt made / attempted
  - `FTM`, `FTA` — free throws made / attempted
  - `OREB`, `DREB`, `REB` — offensive, defensive, total rebounds
  - `AST` — assists
  - `STL` — steals
  - `BLK` — blocks
  - `TOV` — turnovers
  - `PF` — personal fouls
  - `PTS` — points

- Context / outcome:
  - `PLUS_MINUS` — player plus/minus for the game
  - `WL` — win/loss flag for the player's team in that game (e.g. `W`/`L`)

Notes: column names may appear with different casing (e.g., `Game_ID`, `Game_Id`) — scripts in `src/` apply case-insensitive normalization.

---

## Team-level (box score) — available columns
These are per-team-per-game aggregates in `team_game_logs.parquet` (one row per team per game).

- Identifiers / metadata:
  - `SEASON` — season id
  - `GAME_ID` — unique game id
  - `TEAM_ID` — team identifier (abbreviation or id)

- Aggregated box stats (team totals):
  - `PTS`, `REB`, `OREB`, `DREB`, `AST`, `STL`, `BLK`, `TOV`, `PF`
  - Shooting: `FGM`, `FGA`, `FG3M`, `FG3A`, `FTM`, `FTA`
  - `MIN`, `PLUS_MINUS`
  - Opponent columns: `OPP_TEAM_ID`, `OPP_PTS`
  - `WIN` — team-level win flag (boolean)

- Derived/per-game padded output: `team_game_details.parquet` contains 30 teams × 82 games (padded) per season, with per-game indices and zero-padding for missing games.

---

## Derived player-season metrics (available in `data/advanced_local_metrics.parquet`)
These are computed by `src/data_compute/compute_local_metrics.py`. Column names below reflect the output naming in the script.

- Identity / counts:
  - `PLAYER_ID`, `SEASON`, `GAMES` — counts of games used
  - `MPG` — minutes per game (computed as `MIN_total / GAMES`)

- Raw season sums (renamed in output):
  - `PTS_total`, `AST_total`, `REB_total`, `MIN_total`, `FGM_sum`, `FGA_sum`, `FG3M_sum`, `FG3A_sum`, `FTA_sum`, `FTM_sum`, `TOV_sum`

- Rate / efficiency metrics:
  - `TS_pct` — True Shooting % (PTS / (2*(FGA + 0.44*FTA)))
  - `eFG_pct` — effective FG% ((FGM + 0.5*FG3M) / FGA)
  - `TOV_pct` — turnover percent (approx)
  - `AST_pct` — assist percentage approximation
  - `AST_TOV_ratio` — AST / TOV
  - `OREB_pct`, `DREB_pct` — offensive/defensive rebounding rates (approx)
  - `USG_pct` — approximate usage percent

- Possessions and per-possession metrics:
  - `PLAYER_POSS` — estimated possessions (sum of per-game estimates)
  - `PTS_per75poss` — points per 75 possessions (season estimate)

- Aggregate / responsibility metrics:
  - `PTS_from_assists_est` — estimated points created from assists (0.7 * AST_sum)
  - `points_responsible` — `PTS_sum + PTS_from_assists_est`

- Shooting / shot mix metrics:
  - `three_point_freq` — FG3A / FGA
  - `three_point_pct` — FG3M / FG3A
  - `points_per_shot` — PTS / (FGA + 0.44*FTA)
  - `FT_rate` — free-throw rate = FTA / (FGA + 0.44*FTA)

- Per-36 / per-minute scaled stats:
  - `PTS_per36`, `AST_per36`, `REB_per36`, `OREB_per36`, `DREB_per36`, `TOV_per36`, `FGM_per36`, `FGA_per36`

- Other computed columns / diagnostics:
  - `PLAYER_POSSESSIONS_EST_sum` (season sum of per-game possession estimate)
  - `PROD` — example composite metric (`PTS + 0.7*AST + 0.3*REB`)
  - Team-average columns merged into player-season (`TEAM_TEAM_MIN`, `TEAM_TEAM_FGM`, `TEAM_TEAM_FGA`, `TEAM_TEAM_FTA`, `TEAM_TEAM_TOV`, `TEAM_TEAM_OREB`, `TEAM_TEAM_DREB`) — used for AST%, OREB% calculations

Notes: exact output column names are defined in `src/data_compute/compute_local_metrics.py`. Some intermediate column names use `_sum` suffixes (e.g., `PTS_sum`) and are renamed in the final output (e.g., `PTS_total`).

---

## Missing / optional signals (not present or partially present)
- Minute-level lineup data (per-possession lineup on/off) — not present.
- Advanced tracking stats (distance, touches, shot locations, shot quality) — not present.
- Play-by-play derived features (time-of-possession, scoring runs) — not present in repository.

---

## Recommendations for next-phase feature engineering
- Build rolling-window features (3/7/14 game averages) for: `PTS`, `AST`, `REB`, `TOV`, `FG3A`, `FG3M`, `FTA`, `MIN`, and derived rates (`TS_pct`, `eFG_pct`, `USG_pct`).
- Create opponent-adjusted features: opponent defensive FGA/FG% from `team_game_logs.parquet` and rolling defensive metrics.
- Enrich with rest / home/away / back-to-back using `GAME_DATE` and `MATCHUP`.
- Add CI checks for `GAME_ID` pair counts and WIN consistency (already partly implemented in `data_quality_check.py`).

If you want, I can now scaffold a `src/features/build_rolling_features.py` that reads the player and team tables, computes rolling windows, and writes `data/features/player_rolling_features.parquet` and `data/features/team_rolling_features.parquet`.
