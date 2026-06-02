# BKE Canonical Data Schemas

**Target state after standardization (Options B, 2026-06-01).**  
All files use lowercase column names, `'YYYY-YY'` season format, plain integer string IDs.

---

## Schema Rules

| Rule | Before | After |
|------|--------|-------|
| Column names | `PLAYER_ID`, `SEASON_ID`, `PTS` | `player_id`, `season`, `pts` |
| Season format | `'22024'`, `'22023'` | `'2024-25'`, `'2023-24'` |
| Season column | `SEASON_ID` | `season` (renamed) |
| Team/player IDs | `'1610612744.0'` | `'1610612744'` |
| Case-dup cols | `Player_ID` + `PLAYER_ID` → two cols | `player_id` → one col (first kept) |

---

## Step-2 Critical Path Files

### `data/historical/player_game_logs_{season}.parquet`

| Column | Type | Notes |
|--------|------|-------|
| `season` | str | `'2024-25'` (was `SEASON_ID = '22024'`) |
| `player_id` | str | plain int string (was `Player_ID` + `PLAYER_ID` case-dup) |
| `game_id` | str | |
| `game_date` | str/date | |
| `matchup` | str | |
| `wl` | str | `'W'` or `'L'` |
| `min` | float | |
| `pts`, `reb`, `ast`, `stl`, `blk`, `tov` | float | |
| `fgm`, `fga`, `fg3m`, `fg3a`, `ftm`, `fta` | float | |
| `fg_pct`, `fg3_pct`, `ft_pct` | float | |
| `oreb`, `dreb` | float | |
| `plus_minus` | float | |

**Rows:** ~21k per season. Combined in `final_player_game_logs.parquet` (~69k).

### `data/historical/team_game_logs.parquet`

| Column | Type | Notes |
|--------|------|-------|
| `season` | str | `'2024-25'` (was `SEASON_ID = '22024'` or `SEASON = '2024-25'`) |
| `team_id` | str | plain int string |
| `game_id` | str | |
| `game_date` | str/date | |
| `pts`, `opp_pts` | int | |
| `margin` | int | `pts - opp_pts` |
| `wl` | str | |

**Rows:** ~21k. Combined across all seasons.

### `data/matchup/league_season_matchups.parquet`

| Column | Type | Notes |
|--------|------|-------|
| `season` | str | (was `SEASON_ID`) |
| `off_player_id` | str | |
| `def_player_id` | str | |
| `gp` | int | |
| `matchup_min` | str/float | |
| `partial_poss` | float | |
| `player_pts`, `team_pts` | float | |

**Rows:** ~1.2M.

### `data/matchup/matchups_rollup.parquet`

| Column | Type | Notes |
|--------|------|-------|
| `season` | str | |
| `def_player_id` | str | |
| `position` | str | `'G'`, `'F'`, `'C'`, `'TOTAL'` |
| `percent_of_time` | float | |

### `data/historical/complete_player_season_stats.parquet`

| Column | Type | Notes |
|--------|------|-------|
| `player_id` | str | (was `PLAYER_ID`) |
| `player_name` | str | |
| `team_id` | str | |
| `season` | str | |
| `gp` | int | |
| `min` | float | |
| `pts`, `reb`, `ast` | float | |
| `usg_pct`, `efg_pct`, `ts_pct` | float | |

**Rows:** ~5k (per-season player totals).

### `data/processed/simulation/possession_box_distributions.parquet`

Key output of `scripts/run_possession_engine.py`. Columns are all lowercase already (produced post-Step-2 build). Used by Sleeve C markets layer.

---

## Canonical Column Name Table

| Raw (NBA API / legacy) | Canonical |
|------------------------|-----------|
| `PLAYER_ID`, `Player_ID` | `player_id` |
| `TEAM_ID`, `Team_ID` | `team_id` |
| `GAME_ID`, `Game_ID` | `game_id` |
| `SEASON_ID` | `season` |
| `SEASON` | `season` |
| `PLAYER_NAME` | `player_name` |
| `TEAM_ABBREVIATION` | `team_abbreviation` |
| `GAME_DATE` | `game_date` |
| `MATCHUP` | `matchup` |
| `WL` | `wl` |
| `PTS` | `pts` |
| `REB` | `reb` |
| `AST` | `ast` |
| `STL` | `stl` |
| `BLK` | `blk` |
| `TOV` | `tov` |
| `MIN` | `min` |
| `GP` | `gp` |
| `FGM`, `FGA`, `FG_PCT` | `fgm`, `fga`, `fg_pct` |
| `FG3M`, `FG3A`, `FG3_PCT` | `fg3m`, `fg3a`, `fg3_pct` |
| `FTM`, `FTA`, `FT_PCT` | `ftm`, `fta`, `ft_pct` |
| `OREB`, `DREB` | `oreb`, `dreb` |
| `PLUS_MINUS` | `plus_minus` |
| `USG_PCT` | `usg_pct` |
| `EFG_PCT`, `eFG_pct` | `efg_pct` |
| `TS_PCT` | `ts_pct` |
| `NET_RATING` | `net_rating` |
| `OFF_RATING`, `DEF_RATING` | `off_rating`, `def_rating` |
| `PACE` | `pace` |
| `DEF_PLAYER_ID` | `def_player_id` |
| `OFF_PLAYER_ID` | `off_player_id` |
| `POSITION` | `position` |

---

## Migration Status

| Phase | Scope | Session | Status |
|-------|-------|---------|--------|
| 0 | Foundation (schema_contract.py, validator, docs) | A | ✅ Done |
| 5 | Data fetch layer (12 scripts + parquet rewrites) | A | 🔄 In progress |
| 4 | Data compute layer (8 scripts) | B | — |
| 3 | Modeling layer (10 scripts) | B | — |
| 2 | Player eval + profile aggregate (6 scripts) | B | — |
| 1 | Simulation layer (7 scripts) | C | — |
| 6 | Analysis scripts (24 scripts) | C | — |
| 7 | Historical parquet rewrite | C | — |
