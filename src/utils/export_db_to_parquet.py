"""
src/utils/export_db_to_parquet.py
Exports 'players' and 'teams' tables from SQLite to Parquet.
FIXED: Prevents overwriting player_id with team_id.
"""

import sqlite3
import pandas as pd
import os
import sys
from pathlib import Path

# Adjust path to find src if run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

DB_PATH = Path("data/player_team_profiles.db")
OUTPUT_DIR = Path("data/historical")

def export_table(conn, table_name, output_filename):
    print(f"Exporting '{table_name}'...")
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        
        # Standardization: Create a common 'id' column for merging
        # USE ELIF to prevent overwriting!
        if 'player_id' in df.columns:
            df['id'] = df['player_id'].astype(str)
        elif 'team_id' in df.columns:
            df['id'] = df['team_id'].astype(str)
            
        # Clean the ID (remove .0 just in case)
        if 'id' in df.columns:
            df['id'] = df['id'].str.replace(r'\.0$', '', regex=True)

        output_path = OUTPUT_DIR / output_filename
        df.to_parquet(output_path, index=False)
        print(f"✅ Saved {len(df)} rows to {output_path}")
    except Exception as e:
        print(f"❌ Error exporting {table_name}: {e}")

def main():
    if not DB_PATH.exists():
        print(f"Database not found at {DB_PATH}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    try:
        export_table(conn, "players", "players.parquet")
        export_table(conn, "teams", "teams.parquet")
    finally:
        conn.close()

if __name__ == "__main__":
    main()