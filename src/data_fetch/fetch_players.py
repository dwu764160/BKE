import sys
from pathlib import Path
import sqlite3
import pandas as pd
import time
from datetime import datetime

# ensure project root
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.db_utils import (
    create_tables,
    was_player_fetched_recently,
    mark_player_fetched,
    was_player_fetched_recently_conn,
    mark_player_fetched_conn,
)
from src.data_fetch.fetch_profiles import fetch_player_info, upsert_player

DB_PATH = ROOT / "data" / "player_team_profiles.db"


def run(parquet_file=None):
    create_tables(DB_PATH)
    conn = sqlite3.connect(str(DB_PATH))

    if parquet_file is None:
        parquet_file = ROOT / "data" / "historical" / "player_game_logs.parquet"

    parquet_file = Path(parquet_file)
    if not parquet_file.exists():
        print(f"Parquet file not found: {parquet_file}. Exiting.")
        return

    df = pd.read_parquet(parquet_file)
    # Debug: print sample rows to confirm structure
    try:
        print("[DEBUG] Parquet sample rows:")
        print(df.head(5))
    except Exception as e:
        print(f"[DEBUG] Could not print parquet sample: {e}")
    if 'PLAYER_ID' not in df.columns:
        print("PLAYER_ID column not found in parquet. Exiting.")
        return

    # load player id->name mappings from data/historical CSVs (best-effort)
    id_to_name = {}
    try:
        hist_dir = ROOT / "data" / "historical"
        for p in hist_dir.glob("player_id_name_map_*.csv"):
            try:
                tmp = pd.read_csv(p)
                if 'PERSON_ID' in tmp.columns and 'DISPLAY_FIRST_LAST' in tmp.columns:
                    for _, r in tmp.iterrows():
                        try:
                            id_to_name[int(r['PERSON_ID'])] = r['DISPLAY_FIRST_LAST']
                        except Exception:
                            continue
            except Exception:
                continue
    except Exception:
        pass

    # First pass: fetch each unique player via CommonPlayerInfo and store
    unique_ids = pd.Series(df['PLAYER_ID'].dropna().unique()).astype(int).tolist()
    print(f"Fetching common player info for {len(unique_ids)} players...")
    print(f"[DEBUG] First 10 PLAYER_IDs: {unique_ids[:10]}")
    if 'POSITION' in df.columns:
        try:
            print(f"[DEBUG] First 10 player positions: {df['POSITION'].dropna().unique()[:10]}")
        except Exception:
            pass
    if 'HEIGHT' in df.columns:
        try:
            print(f"[DEBUG] First 10 player heights: {df['HEIGHT'].dropna().unique()[:10]}")
        except Exception:
            pass
    if 'WEIGHT' in df.columns:
        try:
            print(f"[DEBUG] First 10 player weights: {df['WEIGHT'].dropna().unique()[:10]}")
        except Exception:
            pass

    for pid in unique_ids:
        player_name = id_to_name.get(int(pid))
        if was_player_fetched_recently_conn(conn, pid, days=30):
            print(f"⏭️ Skipping {pid} ({player_name}), recently fetched")
            continue
        player_data = fetch_player_info(pid, player_name=player_name)
        if player_data:
            upsert_player(conn, player_data)
            # mark using same connection
            mark_player_fetched_conn(conn, pid)
            print(f"✅ {player_data.get('full_name')} added/updated")
        time.sleep(1)

    # Second pass: ensure all player IDs present in players table (insert minimal placeholders if missing)
    c = conn.cursor()
    c.execute("SELECT player_id FROM players")
    existing = {row[0] for row in c.fetchall()}

    missing = [pid for pid in unique_ids if int(pid) not in existing]
    print(f"Ensuring {len(missing)} missing player IDs are recorded in DB...")
    for pid in missing:
        # insert a minimal placeholder row
        placeholder = (
            int(pid),
            None,  # full_name
            None,  # team_id
            None,  # primary_position
            None,  # age
            None,  # height_inches
            None,  # weight_lbs
            None,  # wingspan_inches
            None,  # experience_years
            "{}",
            datetime.now().strftime("%Y-%m-%d")
        )
        c.execute("""
            INSERT INTO players (player_id, full_name, team_id, primary_position, age, height_inches, weight_lbs,
                wingspan_inches, experience_years, advanced_metrics, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(player_id) DO NOTHING
        """, placeholder)
    conn.commit()
    conn.close()
    print("🎯 Player fetch complete.")


if __name__ == "__main__":
    run()
