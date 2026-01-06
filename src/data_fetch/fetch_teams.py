"""
src/data_fetch/fetch_teams.py
Fetches NBA team metadata and writes to the local metadata store (DB/parquet).
Input: nba_api teams endpoint
Output: updates local DB / teams table used by downstream scripts
"""

import sys
from pathlib import Path
import sqlite3
from datetime import datetime

# ensure project root
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.db_utils import create_tables
from nba_api.stats.static import teams
from nba_api.stats.endpoints import teaminfocommon
import time

DB_PATH = ROOT / "data" / "player_team_profiles.db"


def run():
    create_tables(DB_PATH)
    conn = sqlite3.connect(str(DB_PATH))

    nba_teams = teams.get_teams()
    c = conn.cursor()
    for t in nba_teams:
        # try to enrich with conference/division via TeamInfoCommon (best-effort)
        conference = None
        division = None
        try:
            info = teaminfocommon.TeamInfoCommon(team_id=t['id'])
            info_dict = info.get_normalized_dict()
            # get the first table and first row if present
            first = None
            for v in info_dict.values():
                if isinstance(v, list) and len(v) > 0:
                    first = v[0]
                    break
            if first:
                # common possible keys
                for k in ("TEAM_CONFERENCE", "CONFERENCE", "CONF_NAME", "CONFERENCE_NAME", "TEAM_CONF"):
                    if k in first:
                        conference = first.get(k)
                        break
                for k in ("TEAM_DIVISION", "DIVISION", "DIVISION_NAME", "TEAM_DIV"):
                    if k in first:
                        division = first.get(k)
                        break
        except Exception:
            # best-effort: ignore if endpoint isn't available or fails
            conference = conference
            division = division

        team_data = (
            t['id'],
            t['abbreviation'],
            t['full_name'],
            conference,
            division,
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
        # small delay to avoid hammering the NBA stats site
        time.sleep(0.6)
    conn.close()
    print("✅ Teams fetched and updated")


if __name__ == "__main__":
    run()
