"""
src/data_fetch/fetch_profiles.py
Fetches player and team profile information via NBA endpoints and web scraping.
Input: player/team IDs
Output: writes profile records into the local SQLite DB or parquet
"""

import sqlite3
import pandas as pd
from nba_api.stats.static import teams
from nba_api.stats.endpoints import commonplayerinfo
from bs4 import BeautifulSoup
import requests
import time
from datetime import datetime
from pathlib import Path
import urllib.parse
import sys
import random
import http.client

# Ensure the project root is on sys.path so the `src` package can be imported
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.db_utils import (
    create_tables,
    mark_player_fetched,
    was_player_fetched_recently,
    mark_player_fetched_conn,
    was_player_fetched_recently_conn,
)

BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH = BASE_DIR / "data" / "player_team_profiles.db"


# --- Basketball Reference Scraper for Wingspan ---
def scrape_wingspan(player_name):
    """
    Tries to locate a player's wingspan by searching Basketball-Reference via Google.
    This is a best-effort approach and may return None if not found.
    """
    try:
        query = urllib.parse.quote_plus(f"{player_name} wingspan site:basketball-reference.com")
        url = f"https://www.google.com/search?q={query}"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        text = soup.get_text(separator=" ")
        lowered = text.lower()

        if "wingspan" in lowered:
            # look for patterns like 7'2" or 6'10"
            import re

            m = re.search(r"(\d)\s*'\s*(\d{1,2})\s*\"", text)
            if m:
                feet = int(m.group(1))
                inches = int(m.group(2))
                return feet * 12 + inches
        return None
    except Exception as e:
        # be quiet on failures, return None
        return None


# --- Helper to convert height string like "6-7" → 79 ---
def height_to_inches(height_str):
    try:
        if not height_str:
            return None
        if "-" in height_str:
            feet, inches = height_str.split("-")
            return int(feet) * 12 + int(inches)
        # sometimes NBA API uses format like 6-7 or '6-7'
        return None
    except Exception:
        return None


# --- Fetch player info from nba_api ---
def fetch_player_info(player_id, player_name=None):
    max_attempts = 4
    base_timeout = 30
    for attempt in range(1, max_attempts + 1):
        try:
            timeout = base_timeout + (attempt - 1) * 10
            info = commonplayerinfo.CommonPlayerInfo(player_id=player_id, timeout=timeout)
            data = info.get_normalized_dict().get('CommonPlayerInfo')
            if not data:
                return None
            data = data[0]

            height_inches = height_to_inches(data.get('HEIGHT', None))
            weight = None
            try:
                weight = int(data.get('WEIGHT'))
            except Exception:
                weight = None
            position = data.get('POSITION')
            try:
                exp = int(data.get('SEASON_EXP'))
            except Exception:
                exp = None
            birthdate = data.get('BIRTHDATE')
            full_name = data.get('DISPLAY_FIRST_LAST')
            team_id = data.get('TEAM_ID')

            # compute age if birthdate exists (round down)
            age = None
            if birthdate:
                try:
                    # Normalize and parse common formats
                    bd_str = str(birthdate)
                    # strip time if present
                    if 'T' in bd_str:
                        bd_str = bd_str.split('T')[0]
                    # try ISO first
                    try:
                        bd = datetime.fromisoformat(bd_str).date()
                    except Exception:
                        bd = datetime.strptime(bd_str, "%Y-%m-%d").date()
                    today = datetime.now().date()
                    age = today.year - bd.year - ((today.month, today.day) < (bd.month, bd.day))
                except Exception:
                    age = None

            wingspan = scrape_wingspan(full_name) if full_name else None

            # detect retired / not-active players and clear team_id when appropriate
            try:
                roster_status = None
                for k in ("ROSTERSTATUS", "ROSTER_STATUS", "IS_ACTIVE", "ACTIVE"):
                    if k in data:
                        roster_status = data.get(k)
                        break

                # Some endpoints use 1 for active, 0 for inactive. Treat falsy or 0 as retired/not active.
                retired = False
                if roster_status is not None:
                    try:
                        if str(roster_status).strip() in ("0", "False", "false", "INACTIVE", "Inactive"):
                            retired = True
                    except Exception:
                        retired = False

                if retired or not team_id:
                    team_id = None
            except Exception:
                # keep existing team_id if any unexpected structure
                pass

            return {
                "player_id": int(player_id),
                "full_name": full_name,
                "team_id": int(team_id) if team_id else None,
                "primary_position": position,
                "age": age,
                "height_inches": height_inches,
                "weight_lbs": weight,
                "wingspan_inches": wingspan,
                "experience_years": exp,
                "advanced_metrics": "{}",
                "last_updated": datetime.now().strftime("%Y-%m-%d")
            }
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError, http.client.RemoteDisconnected) as e:
            # transient network errors — retry with exponential backoff
            if attempt < max_attempts:
                wait = 2 ** attempt + random.uniform(0, 1)
                name_display = f" ({player_name})" if player_name else ""
                print(f"[WARN] Network error for player {player_id}{name_display} (attempt {attempt}/{max_attempts}): {e}. Retrying in {wait:.1f}s")
                time.sleep(wait)
                continue
            else:
                name_display = f" ({player_name})" if player_name else ""
                print(f"❌ Error fetching player {player_id}{name_display}: {e}")
                return None
        except Exception as e:
            # non-transient error — log and return
            print(f"❌ Error fetching player {player_id}: {e}")
            return None


# --- Insert or update player record ---
def upsert_player(conn, player_data):
    c = conn.cursor()
    c.execute("""
        INSERT INTO players (
            player_id, full_name, team_id, primary_position, age, height_inches,
            weight_lbs, wingspan_inches, experience_years, advanced_metrics, last_updated
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(player_id) DO UPDATE SET
            full_name=excluded.full_name,
            team_id=excluded.team_id,
            primary_position=excluded.primary_position,
            age=excluded.age,
            height_inches=excluded.height_inches,
            weight_lbs=excluded.weight_lbs,
            wingspan_inches=excluded.wingspan_inches,
            experience_years=excluded.experience_years,
            advanced_metrics=excluded.advanced_metrics,
            last_updated=excluded.last_updated
    """, (
        player_data.get('player_id'),
        player_data.get('full_name'),
        player_data.get('team_id'),
        player_data.get('primary_position'),
        player_data.get('age'),
        player_data.get('height_inches'),
        player_data.get('weight_lbs'),
        player_data.get('wingspan_inches'),
        player_data.get('experience_years'),
        player_data.get('advanced_metrics'),
        player_data.get('last_updated')
    ))
    conn.commit()


# --- Fetch teams ---
def fetch_teams(conn):
    nba_teams = teams.get_teams()
    c = conn.cursor()
    for t in nba_teams:
        team_data = (
            t['id'],
            t['abbreviation'],
            t['full_name'],
            t.get('conference', None),
            t.get('division', None),
            "{}",
            datetime.now().strftime("%Y-%m-%d")
        )
        c.execute("""
            INSERT INTO teams (team_id, abbreviation, full_name, conference, division, advanced_metrics, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(team_id) DO UPDATE SET
                abbreviation=excluded.abbreviation,
                full_name=excluded.full_name,
                conference=excluded.conference,
                division=excluded.division,
                advanced_metrics=excluded.advanced_metrics,
                last_updated=excluded.last_updated
        """, team_data)
    conn.commit()
    print("✅ Teams fetched and updated")


# --- Main runner ---
def main(parquet_file=None, db_path=None):
    if db_path:
        db_path = Path(db_path)
    else:
        db_path = DB_PATH

    create_tables(db_path)
    conn = sqlite3.connect(str(db_path))

    # 1️⃣ Fetch team data
    fetch_teams(conn)

    # 2️⃣ Load player IDs from your parquet (from earlier)
    if parquet_file is None:
        parquet_file = BASE_DIR / "data" / "historical" / "player_game_logs.parquet"

    parquet_file = Path(parquet_file)
    if not parquet_file.exists():
        print(f"Parquet file not found: {parquet_file}. Skipping player fetch.")
        conn.close()
        return

    df = pd.read_parquet(parquet_file)
    if 'PLAYER_ID' not in df.columns:
        print("PLAYER_ID column not found in parquet. Skipping.")
        conn.close()
        return

    player_ids = df['PLAYER_ID'].dropna().unique().tolist()

    print(f"Fetching profiles for {len(player_ids)} players...")

    # 3️⃣ Fetch and insert player info
    for pid in player_ids:
        try:
            # Skip fetching if we recently fetched this player
            if was_player_fetched_recently_conn(conn, pid, days=30):
                print(f"⏭️ Skipping {pid}, recently fetched")
                continue

            player_data = fetch_player_info(pid)
            if player_data:
                upsert_player(conn, player_data)
                # mark in cache using same connection to avoid locking
                mark_player_fetched_conn(conn, pid)
                print(f"✅ {player_data.get('full_name')} added/updated")
        except Exception as e:
            print(f"Error processing player {pid}: {e}")
        time.sleep(1)  # avoid rate limit

    conn.close()
    print("🎯 All player and team profiles saved successfully.")


if __name__ == "__main__":
    main()
