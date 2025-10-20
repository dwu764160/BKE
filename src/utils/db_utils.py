import sqlite3
from pathlib import Path
from datetime import datetime


def create_tables(db_path=None):
    """Create SQLite DB and required tables.

    db_path: path to sqlite file. If None, defaults to project-relative data/player_team_profiles.db
    """
    if db_path is None:
        base = Path(__file__).resolve().parents[2]
        db_path = base / "data" / "player_team_profiles.db"
    else:
        db_path = Path(db_path)

    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()

    # Player Table
    c.execute("""
        CREATE TABLE IF NOT EXISTS players (
            player_id INTEGER PRIMARY KEY,
            full_name TEXT,
            team_id INTEGER,
            primary_position TEXT,
            age INTEGER,
            height_inches INTEGER,
            weight_lbs INTEGER,
            wingspan_inches INTEGER,
            experience_years INTEGER,
            advanced_metrics TEXT,
            last_updated TEXT
        )
    """)

    # Team Table
    c.execute("""
        CREATE TABLE IF NOT EXISTS teams (
            team_id INTEGER PRIMARY KEY,
            abbreviation TEXT,
            full_name TEXT,
            conference TEXT,
            division TEXT,
            advanced_metrics TEXT,
            last_updated TEXT
        )
    """)

    # Cache table to track when a player's profile was last fetched
    c.execute("""
        CREATE TABLE IF NOT EXISTS fetch_cache (
            player_id INTEGER PRIMARY KEY,
            last_fetched TEXT
        )
    """)

    conn.commit()
    conn.close()
    print(f"✅ Database and tables ready at {db_path}")


def mark_player_fetched(db_path=None, player_id=None):
    if db_path is None:
        base = Path(__file__).resolve().parents[2]
        db_path = base / "data" / "player_team_profiles.db"
    conn = sqlite3.connect(str(db_path))
    try:
        c = conn.cursor()
        c.execute(
            "INSERT INTO fetch_cache (player_id, last_fetched) VALUES (?, ?)"
            " ON CONFLICT(player_id) DO UPDATE SET last_fetched=excluded.last_fetched",
            (int(player_id), datetime.now().strftime("%Y-%m-%d"))
        )
        conn.commit()
    finally:
        conn.close()


def was_player_fetched_recently(db_path=None, player_id=None, days=30):
    if db_path is None:
        base = Path(__file__).resolve().parents[2]
        db_path = base / "data" / "player_team_profiles.db"
    conn = sqlite3.connect(str(db_path))
    try:
        c = conn.cursor()
        c.execute("SELECT last_fetched FROM fetch_cache WHERE player_id = ?", (int(player_id),))
        row = c.fetchone()
    finally:
        conn.close()
    if not row:
        return False
    try:
        last = datetime.strptime(row[0], "%Y-%m-%d")
        return (datetime.now() - last).days <= days
    except Exception:
        return False


def mark_player_fetched_conn(conn, player_id):
    """Mark player as fetched using an existing sqlite3.Connection."""
    c = conn.cursor()
    c.execute(
        "INSERT INTO fetch_cache (player_id, last_fetched) VALUES (?, ?)"
        " ON CONFLICT(player_id) DO UPDATE SET last_fetched=excluded.last_fetched",
        (int(player_id), datetime.now().strftime("%Y-%m-%d"))
    )
    # do not commit here; caller should manage transaction/commit


def was_player_fetched_recently_conn(conn, player_id, days=30):
    """Check cache using existing sqlite3.Connection."""
    c = conn.cursor()
    c.execute("SELECT last_fetched FROM fetch_cache WHERE player_id = ?", (int(player_id),))
    row = c.fetchone()
    if not row:
        return False
    try:
        last = datetime.strptime(row[0], "%Y-%m-%d")
        return (datetime.now() - last).days <= days
    except Exception:
        return False
