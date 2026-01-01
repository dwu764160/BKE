# tests/pbp_test.py
# Validates and normalizes NBA play-by-play data (parquet or JSON-lines fallback)

import pandas as pd
import re
from pathlib import Path
import json
import warnings
import sys

# ------------------------
# Paths
# ------------------------

PBP_FILE = Path("data/historical/play_by_play_2022-23.parquet")
PBP_JSON_FALLBACK = Path(str(PBP_FILE) + ".as.json")
TEAM_LOGS = Path("data/historical/team_game_logs.parquet")  # optional

# ensure repo root is on sys.path so `import src...` works when running tests directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ------------------------
# Regexes
# ------------------------

TIME_RE = re.compile(r"(\d{1,2}):(\d{2})")
SCORE_RE = re.compile(r"^(\d{1,3})\s*[–-]\s*(\d{1,3})$")

# ------------------------
# Parsing helpers
# ------------------------

def parse_time_to_seconds(s):
    if s is None or pd.isna(s):
        return None
    m = TIME_RE.search(str(s))
    if not m:
        return None
    mm, ss = int(m.group(1)), int(m.group(2))
    return mm * 60 + ss


def parse_score(s):
    if s is None or pd.isna(s):
        return None
    m = SCORE_RE.search(str(s).strip())
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))

# ------------------------
# Load data
# ------------------------

def load():
    if PBP_FILE.exists():
        try:
            return pd.read_parquet(PBP_FILE)
        except Exception as e:
            print("Could not read parquet:", e)

    if PBP_JSON_FALLBACK.exists():
        try:
            return pd.read_json(PBP_JSON_FALLBACK, lines=True)
        except Exception:
            rows = []
            with PBP_JSON_FALLBACK.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        pass
            return pd.DataFrame(rows)

    raise FileNotFoundError("No PBP data found")

# ------------------------
# Normalization / Parser
# ------------------------

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produces canonical columns:
      GAME_ID, RAW, TIME, SCORE, DESCRIPTION
    """

    df = df.copy()

    # Standardize RAW column
    if "RAW_TEXT" in df.columns:
        df["RAW"] = df["RAW_TEXT"]
    elif "RAW" not in df.columns and "DESCRIPTION" in df.columns:
        df["RAW"] = df["DESCRIPTION"]

    def split_raw(raw):
        if raw is None or pd.isna(raw):
            return (None, None, None)

        raw = str(raw)

        parts = raw.split("\n\n", 1)
        header = parts[0].strip()
        desc = parts[1].strip() if len(parts) > 1 else ""

        header_lines = [l.strip() for l in header.splitlines() if l.strip()]

        time = None
        score = None

        if header_lines:
            time = header_lines[0]

        if len(header_lines) > 1:
            if SCORE_RE.search(header_lines[1]):
                score = header_lines[1]
            else:
                m = re.search(
                    r"(\d{1,2}:\d{2})\s+(\d{1,3}\s*[–-]\s*\d{1,3})",
                    header
                )
                if m:
                    time = m.group(1)
                    score = m.group(2)

        return (time, score, desc)

    if ("TIME" not in df.columns) or df["TIME"].isna().all():
        parsed = df.get("RAW", pd.Series([None] * len(df))).apply(split_raw)
        parsed_df = pd.DataFrame(parsed.tolist(), index=df.index)
        parsed_df.columns = ["TIME_parsed", "SCORE_parsed", "DESCRIPTION_parsed"]

        df = pd.concat([df, parsed_df], axis=1)

        df["TIME"] = df.get("TIME", df["TIME_parsed"])
        df["SCORE"] = df.get("SCORE", df["SCORE_parsed"])
        df["DESCRIPTION"] = df.get("DESCRIPTION", df["DESCRIPTION_parsed"])

    return df

# ------------------------
# Validation (fixed)
# ------------------------

def basic_schema_checks(df):
    required = ["GAME_ID"]
    return [c for c in required if c not in df.columns]


def validate_game_rows(df_game: pd.DataFrame):
    errs = []
    df = df_game.copy()

    df["__tsec"] = df["TIME"].apply(parse_time_to_seconds)
    df["__score_parsed"] = df["SCORE"].apply(parse_score)

    # Time sanity
    if df["__tsec"].isna().all():
        errs.append("no-time-parseable")

    # ✅ FIX: Only validate rows that CLAIM to have a score
    has_score = df["SCORE"].notna()
    bad_scores = int(df.loc[has_score, "__score_parsed"].isna().sum())
    if bad_scores > 0:
        errs.append(f"bad-score-format:{bad_scores}")

    # Duplicates (soft warning)
    dupe_subset = [c for c in ["TIME", "SCORE", "DESCRIPTION", "RAW"] if c in df.columns]
    if dupe_subset:
        dupes = int(df.duplicated(subset=dupe_subset).sum())
        if dupes > 0:
            gid = None
            if "GAME_ID" in df.columns:
                try:
                    gid = str(df["GAME_ID"].iloc[0])
                except Exception:
                    gid = None
            msg = f"duplicate-events:{dupes}"
            if gid:
                msg = f"GAME_ID={gid} - {msg}"
            warnings.warn(msg)

    # Size heuristics
    if len(df) < 50:
        errs.append("very-few-events")
    if len(df) > 5000:
        errs.append("very-many-events")

    return errs

# ------------------------
# Entry point
# ------------------------

def validate_all():
    from src.data_normalize.event_typing import add_event_type
    df = load()
    df = normalize(df)

    df = add_event_type(df)

    df.to_parquet(
    "data/historical/pbp_normalized_with_events.parquet",
    index=False
    )

    missing = basic_schema_checks(df)
    if missing:
        print("Missing required columns:", missing)
        return

    results = {}
    for gid, g in df.groupby("GAME_ID"):
        results[gid] = validate_game_rows(g)

    failures = {gid: errs for gid, errs in results.items() if errs}
    print(f"Games checked: {len(results)}, failures: {len(failures)}")

    for gid, errs in list(failures.items())[:25]:
        print(gid, errs)

    if TEAM_LOGS.exists():
        try:
            pd.read_parquet(TEAM_LOGS)
            print("Team logs loaded (optional checks can be added).")
        except Exception:
            print("Could not read team_game_logs.parquet")


if __name__ == "__main__":
    validate_all()
