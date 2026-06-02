# src/data_fetch/ — Layer 0: Raw Data Fetch

Hits external APIs (NBA.com, Basketball-Reference, ESPN) and writes canonical parquets to `data/historical/`, `data/tracking/`, `data/matchup/`, `data/official_stats/`.

All write paths use `save_standardized()` from `src.data.schema_contract` — outputs are canonical lowercase from creation.

---

## Scripts

### Game logs + player metadata

| Script | Output | Notes |
|--------|--------|-------|
| `fetch_historical_data.py` | `player_game_logs_{season}.parquet` | Per-game player logs + player ID/name maps |
| `derive_team_game_logs.py` | `team_game_logs.parquet` | Team game logs from PBP; API-first, PBP fallback |
| `summarize_team_logs.py` | `team_summaries.parquet` | Per-team-season summaries; recovers `plus_minus` from `pts - opp_pts` |
| `fetch_players.py` | `players.parquet` | Player metadata (name, DOB, IDs) |
| `fetch_teams.py` | `teams.parquet` | Team metadata |
| `fetch_profiles.py` | `player_team_profiles.db` (profiles table) | Player bio (height, weight) via NBA API + B-Ref scraping |
| `fetch_player_draft_history.py` | `player_draft_history.parquet` | Draft class, round, overall pick |
| `fetch_player_salaries.py` | `player_salaries_{season}.parquet` | Salary data from ESPN |
| `fetch_player_clutch_stats.py` | `player_clutch_stats_{season}.parquet`, `player_clutch_stats_all.parquet` | Last-5-min, ≤7-point differential |
| `fetch_preseason_rosters.py` | `preseason_rosters.parquet` | Preseason roster snapshots for forecast Option B |
| `fetch_box_scores_complete.py` | `complete_player_season_stats.parquet` | League-wide player base + advanced box stats, all seasons |

### Tracking + advanced

| Script | Output | Notes |
|--------|--------|-------|
| `fetch_official_stats.py` | `data/official_stats/official_advanced_{season}.parquet` | NBA.com advanced (ORTG, DRTG, USG%, TS%, etc.) |
| `fetch_tracking_data.py` | `data/tracking/{season}/tracking_*.parquet`, `synergy_*.parquet` | Tracking (drives, passing, etc.) + 11-type synergy playtypes |
| `fetch_shot_zones.py` | `data/tracking/{season}/shot_zones.parquet` | Shot zone data (AT_RIM, MIDRANGE, PAINT, etc.) |
| `fetch_matchup_data.py` | `data/matchup/league_season_matchups.parquet`, etc. | Closest-defender matchup data (Second Spectrum proximity) |
| `fetch_defensive_metrics.py` | legacy hustle stats | Optional; handles both resultSets/resultSet API shapes |

### PBP fetch

| Script | Output | Notes |
|--------|--------|-------|
| `fetch_pbp/fetch_play_by_play.py` | `play_by_play_{season}.parquet` | DOM fallback; handles resultSets + __NEXT_DATA__ payloads |

### Backfill / supplemental

| Script | Purpose |
|--------|---------|
| `fetch_backfill_box_scores.py` | Backfill missing box score seasons |
| `fetch_external_rapm_benchmarks.py` | Fetch external RAPM benchmarks for validation |
| `fetch_darko_manual.py` | Stage manual DARKO CSV exports |
| `fetch_missing_player_profiles.py` | Fill gaps in player profile data |
| `probe_tracking_availability.py` | Check which tracking endpoints are available for a season |

---

## Rate-limiting notes

NBA.com throttles aggressively. All fetch scripts include `time.sleep()` calls. Session cookies in `data/nba_headers.json` and `data/nba_session.json` must be kept current. Bootstrap session with `fetch_pbp/bootstrap_nba_session.py` if you get 403s.

The 2022-23 `CatchShoot` tracking endpoint has a known outage — `fetch_tracking_data.py` includes a proxy fallback. If this starts failing for other seasons, check `docs/findings/network_stats_nba_throttle_2026-05-30.md`.
