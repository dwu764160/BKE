#!/usr/bin/env bash
set -euo pipefail

# fetch_data.sh — Run all fetcher scripts for BKE pipeline
# Usage: bash scripts/fetch_data.sh

# 1. Setup NBA session headers (if needed)
echo "[FETCH] bootstrap_nba_session.py"
python3 src/data_fetch/fetch_pbp/bootstrap_nba_session.py
echo "[FETCH] capture_nba_headers.py"
python3 src/data_fetch/fetch_pbp/capture_nba_headers.py

# 2. Fetch historical and player/team metadata
echo "[FETCH] fetch_historical_data.py"
python3 src/data_fetch/fetch_historical_data.py
echo "[FETCH] fetch_players.py"
python3 src/data_fetch/fetch_players.py
echo "[FETCH] fetch_teams.py"
python3 src/data_fetch/fetch_teams.py
echo "[FETCH] fetch_profiles.py"
python3 src/data_fetch/fetch_profiles.py

# 3. Optionally fetch play-by-play (choose one)
echo "[FETCH] CDN_pbp_fetch.py (fast)"
python3 src/data_fetch/fetch_pbp/CDN_pbp_fetch.py || true
echo "[FETCH] fetch_play_by_play.py (DOM fallback)"
python3 src/data_fetch/fetch_pbp/fetch_play_by_play.py || true

# 4. Fetch official stats / tracking as available
echo "[FETCH] fetch_official_stats.py"
python3 src/data_fetch/fetch_official_stats.py
echo "[FETCH] fetch_tracking_data.py"
python3 src/data_fetch/fetch_tracking_data.py

# 5. Additional fetchers for full pipeline
echo "[FETCH] fetch_box_scores_complete.py"
python3 src/data_fetch/fetch_box_scores_complete.py
echo "[FETCH] fetch_matchup_data.py"
python3 src/data_fetch/fetch_matchup_data.py
echo "[FETCH] fetch_shot_zones.py"
python3 src/data_fetch/fetch_shot_zones.py
