"""
src/data_fetch/fetch_missing_player_profiles.py
=============================================================================
Fetches NBA player height/weight/metadata for players missing from
data/historical/players.parquet. Merges into the existing file.
Usage: python3 src/data_fetch/fetch_missing_player_profiles.py
=============================================================================
"""

import json
import os
import random
import sys
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import commonplayerinfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PLAYERS_FILE = "data/historical/players.parquet"
MISSING_IDS_FILE = "/tmp/missing_player_ids.json"
SLEEP_BETWEEN = (0.5, 1.2)

def height_to_inches(h):
    if not h or not isinstance(h, str):
        return None
    try:
        if "-" in h:
            ft, ins = h.split("-")
            return int(ft) * 12 + int(ins)
    except Exception:
        pass
    return None

def fetch_player(player_id: str) -> dict | None:
    try:
        info = commonplayerinfo.CommonPlayerInfo(player_id=int(player_id))
        data = info.get_normalized_dict()["CommonPlayerInfo"][0]
        return {
            "player_id": str(player_id),
            "full_name": f"{data.get('FIRST_NAME','')} {data.get('LAST_NAME','')}".strip(),
            "primary_position": data.get("POSITION"),
            "height_inches": height_to_inches(data.get("HEIGHT")),
            "weight_lbs": int(data["WEIGHT"]) if data.get("WEIGHT") and str(data["WEIGHT"]).isdigit() else None,
            "experience_years": data.get("SEASON_EXP"),
        }
    except Exception as e:
        print(f"  Error {player_id}: {e}")
        return None

def main():
    if not os.path.exists(MISSING_IDS_FILE):
        print(f"No missing IDs file at {MISSING_IDS_FILE} — run discovery step first")
        return

    with open(MISSING_IDS_FILE) as f:
        missing_ids = json.load(f)
    print(f"Players to fetch: {len(missing_ids)}")

    existing = pd.read_parquet(PLAYERS_FILE) if os.path.exists(PLAYERS_FILE) else pd.DataFrame()
    existing_ids = set(existing["player_id"].astype(str)) if len(existing) else set()

    to_fetch = [pid for pid in missing_ids if str(pid) not in existing_ids]
    print(f"After dedup: {len(to_fetch)} to fetch")

    records = []
    for i, pid in enumerate(to_fetch, 1):
        print(f"[{i}/{len(to_fetch)}] player_id={pid}", end=" ", flush=True)
        rec = fetch_player(str(pid))
        if rec:
            records.append(rec)
            print(f"✓ {rec['full_name']}")
        else:
            print("✗ skipped")
        time.sleep(random.uniform(*SLEEP_BETWEEN))

        # Checkpoint every 100 players
        if i % 100 == 0 and records:
            _save(existing, records)
            print(f"  → Checkpoint saved ({i} fetched so far)")

    if records:
        _save(existing, records)
        print(f"\nDone. Saved {len(records)} new players.")
    else:
        print("No new players fetched.")

def _save(existing: pd.DataFrame, records: list):
    new_df = pd.DataFrame(records)
    if len(existing):
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["player_id"], keep="last")
    else:
        combined = new_df
    combined.to_parquet(PLAYERS_FILE, index=False)

if __name__ == "__main__":
    main()
