# tests/pbp_test.py
# Validates play-by-play export (parquet or JSON-lines fallback).
# Each test block is labeled with a comment header.
import pandas as pd
import re
from pathlib import Path
import json

# Primary data file (parquet). If a JSON-lines export exists (parquet.as.json), prefer it.
PBP_FILE = Path("data/historical/play_by_play_2022-23.parquet")
PBP_JSON_FALLBACK = Path(str(PBP_FILE) + ".as.json")
TEAM_LOGS = Path("data/historical/team_game_logs.parquet")  # optional

# regexes
SCORE_RE = re.compile(r"^(\d{1,3})\s*[–-]\s*(\d{1,3})$")


def parse_time_to_seconds(s):
    if s is None: return None
    s = str(s).strip()
    m = re.search(r"(\d{1,2}):(\d{2})", s)
    if not m: return None
    mm, ss = int(m.group(1)), int(m.group(2))
    return mm * 60 + ss


def parse_score(s):
    if s is None: return None
    s = str(s).strip()
    m = SCORE_RE.search(s)
    if not m: return None
    return int(m.group(1)), int(m.group(2))


def load():
    # Try parquet first
    if PBP_FILE.exists():
        try:
            return pd.read_parquet(PBP_FILE)
        except Exception as e:
            print("Could not read parquet file:", e)

    # fallback to json-lines export
    if PBP_JSON_FALLBACK.exists():
        try:
            return pd.read_json(PBP_JSON_FALLBACK, lines=True)
        except Exception:
            rows = []
            with PBP_JSON_FALLBACK.open() as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        pass
            return pd.DataFrame(rows)

    raise FileNotFoundError(f"No data file found at {PBP_FILE} or {PBP_JSON_FALLBACK}")


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure core fields: GAME_ID, RAW (original text), TIME, SCORE, DESCRIPTION
    df = df.copy()

    # RAW text may be in different columns: prefer RAW_TEXT, RAW, DESCRIPTION
    if "RAW_TEXT" in df.columns:
        df["RAW"] = df["RAW_TEXT"]
    elif "RAW" not in df.columns and "DESCRIPTION" in df.columns:
        df["RAW"] = df["DESCRIPTION"]

    # parse TIME/SCORE/DESCRIPTION from RAW if not present
    def split_raw(raw):
        if raw is None:
            return (None, None, None)
        raw = str(raw)
        parts = raw.split("\n\n", 1)
        header = parts[0].strip()
        desc = parts[1].strip() if len(parts) > 1 else ""
        header_lines = [l.strip() for l in header.splitlines() if l.strip()]
        time = header_lines[0] if header_lines else None
        score = None
        if len(header_lines) > 1:
            if SCORE_RE.search(header_lines[1]):
                score = header_lines[1]
            else:
                #!/usr/bin/env python3
                # tests/pbp_test.py
                # Validates play-by-play export (parquet or JSON-lines fallback).
                # Each test block is labeled with a comment header.
                import pandas as pd
                import re
                from pathlib import Path
                import json

                # Primary data file (parquet). If a JSON-lines export exists (parquet.as.json), prefer it.
                PBP_FILE = Path("data/historical/play_by_play_2022-23.parquet")
                PBP_JSON_FALLBACK = Path(str(PBP_FILE) + ".as.json")
                TEAM_LOGS = Path("data/historical/team_game_logs.parquet")  # optional

                # regexes
                SCORE_RE = re.compile(r"^(\d{1,3})\s*[–-]\s*(\d{1,3})$")


                def parse_time_to_seconds(s):
                    if s is None:
                        return None
                    s = str(s).strip()
                    m = re.search(r"(\d{1,2}):(\d{2})", s)
                    if not m:
                        return None
                    mm, ss = int(m.group(1)), int(m.group(2))
                    return mm * 60 + ss


                def parse_score(s):
                    if s is None:
                        return None
                    s = str(s).strip()
                    m = SCORE_RE.search(s)
                    if not m:
                        return None
                    return int(m.group(1)), int(m.group(2))


                def load():
                    # Try parquet first
                    if PBP_FILE.exists():
                        try:
                            return pd.read_parquet(PBP_FILE)
                        except Exception as e:
                            print("Could not read parquet file:", e)

                    # fallback to json-lines export
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

                    raise FileNotFoundError(f"No data file found at {PBP_FILE} or {PBP_JSON_FALLBACK}")


                def normalize(df: pd.DataFrame) -> pd.DataFrame:
                    # Ensure core fields: GAME_ID, RAW (original text), TIME, SCORE, DESCRIPTION
                    df = df.copy()

                    # RAW text may be in different columns: prefer RAW_TEXT, RAW, DESCRIPTION
                    if "RAW_TEXT" in df.columns:
                        df["RAW"] = df["RAW_TEXT"]
                    elif "RAW" not in df.columns and "DESCRIPTION" in df.columns:
                        df["RAW"] = df["DESCRIPTION"]

                    # parse TIME/SCORE/DESCRIPTION from RAW if not present
                    def split_raw(raw):
                        if raw is None:
                            return (None, None, None)
                        raw = str(raw)
                        parts = raw.split("\n\n", 1)
                        header = parts[0].strip()
                        desc = parts[1].strip() if len(parts) > 1 else ""
                        header_lines = [l.strip() for l in header.splitlines() if l.strip()]
                        time = header_lines[0] if header_lines else None
                        score = None
                        if len(header_lines) > 1:
                            if SCORE_RE.search(header_lines[1]):
                                score = header_lines[1]
                            else:
                                m = re.search(r"(\d{1,2}:\d{2})\s+(\d{1,3}\s*[–-]\s*\d{1,3})", header)
                                if m:
                                    time = m.group(1)
                                    score = m.group(2)
                        return (time, score, desc)

                    if ("TIME" not in df.columns) or df["TIME"].isna().all():
                        parsed = df.get("RAW", pd.Series([None] * len(df))).apply(lambda r: split_raw(r))
                        parsed_df = pd.DataFrame(parsed.tolist(), index=df.index)
                        parsed_df.columns = ["TIME_parsed", "SCORE_parsed", "DESCRIPTION_parsed"]
                        df = pd.concat([df, parsed_df], axis=1)
                        if "TIME" not in df.columns:
                            df["TIME"] = df["TIME_parsed"]
                        if "SCORE" not in df.columns:
                            df["SCORE"] = df["SCORE_parsed"]
                        if "DESCRIPTION" not in df.columns:
                            df["DESCRIPTION"] = df["DESCRIPTION_parsed"]

                    return df


                def basic_schema_checks(df: pd.DataFrame):
                    required = ["GAME_ID"]
                    missing = [c for c in required if c not in df.columns]
                    return missing


                def validate_game_rows(df_game: pd.DataFrame):
                    errs = []
                    df = df_game.copy()

                    # --- Test: TIME parseable ---
                    time_col = df.get("TIME")
                    if time_col is None:
                        df["__tsec"] = pd.Series([None] * len(df), index=df.index)
                    else:
                        df["__tsec"] = time_col.apply(lambda v: parse_time_to_seconds(v) if pd.notna(v) else None)
                    if df["__tsec"].isna().all():
                        errs.append("no-time-parseable")

                    # --- Test: SCORE parseable ---
                    score_col = df.get("SCORE")
                    if score_col is None:
                        df["__score_parsed"] = pd.Series([None] * len(df), index=df.index)
                    else:
                        df["__score_parsed"] = score_col.apply(lambda v: parse_score(v) if pd.notna(v) else None)
                    bad_scores = int(df["__score_parsed"].isna().sum())
                    if bad_scores > 0:
                        errs.append(f"score-unparseable-count:{bad_scores}")

                    # --- Test: duplicates ---
                    dupe_subset = [c for c in ["TIME", "SCORE", "DESCRIPTION", "RAW"] if c in df.columns]
                    if dupe_subset:
                        dupe_count = int(df.duplicated(subset=dupe_subset).sum())
                        if dupe_count > 0:
                            errs.append(f"duplicate-events:{dupe_count}")

                    # --- Test: size heuristics ---
                    if len(df) < 50:
                        errs.append("very-few-events")
                    if len(df) > 5000:
                        errs.append("very-many-events")

                    # --- Test: PERIOD checks (only if present) ---
                    if "PERIOD" in df.columns:
                        df["PERIOD"] = pd.to_numeric(df.get("PERIOD"), errors="coerce")
                        if df["PERIOD"].isna().any():
                            errs.append("period-missing-or-non-numeric")
                        if df["PERIOD"].min() < 1:
                            errs.append("period-below-1")
                        for p, g in df.groupby("PERIOD"):
                            times = g["__tsec"].dropna().tolist()
                            if len(times) >= 2:
                                for i in range(1, len(times)):
                                    if times[i] > times[i-1] + 1:
                                        errs.append(f"period-{p}-time-increasing-at-index-{i}")
                                        break

                    return errs


                def validate_all():
                    df = load()
                    df = normalize(df)
                    miss = basic_schema_checks(df)
                    if miss:
                        print("Missing required columns:", miss)
                        return
                    results = {}
                    for gid, g in df.groupby("GAME_ID"):
                        results[gid] = validate_game_rows(g)
                    fails = {gid: errs for gid, errs in results.items() if errs}
                    print(f"Games checked: {len(results)}, failures: {len(fails)}")
                    for gid, errs in list(fails.items())[:50]:
                        print(gid, errs)
                    if TEAM_LOGS.exists():
                        try:
                            tgl = pd.read_parquet(TEAM_LOGS)
                            tgl = tgl.astype({"GAME_ID": str})
                            print("Team logs loaded; you can add final-score comparison checks.")
                        except Exception:
                            print("Could not read team_game_logs.parquet")


                if __name__ == "__main__":
                    validate_all()
                if m:
                    time = m.group(1)
                    score = m.group(2)
        return (time, score, desc)

    if ("TIME" not in df.columns) or df["TIME"].isna().all():
        parsed = df.get("RAW", pd.Series([None]*len(df))).apply(lambda r: split_raw(r))
        parsed_df = pd.DataFrame(parsed.tolist(), index=df.index)
        parsed_df.columns = ["TIME_parsed", "SCORE_parsed", "DESCRIPTION_parsed"]
        df = pd.concat([df, parsed_df], axis=1)
        if "TIME" not in df.columns:
            df["TIME"] = df["TIME_parsed"]
        if "SCORE" not in df.columns:
            df["SCORE"] = df["SCORE_parsed"]
        if "DESCRIPTION" not in df.columns:
            df["DESCRIPTION"] = df["DESCRIPTION_parsed"]

    return df


def basic_schema_checks(df: pd.DataFrame):
    required = ["GAME_ID"]
    missing = [c for c in required if c not in df.columns]
    return missing


def validate_game_rows(df_game: pd.DataFrame):
    errs = []
    df = df_game.copy()

    # --- Test: TIME parseable ---
    df["__tsec"] = df.get("TIME").apply(parse_time_to_seconds)
    if df["__tsec"].isna().all():
        errs.append("no-time-parseable")

    # --- Test: SCORE parseable ---
    df["__score_parsed"] = df.get("SCORE").apply(parse_score)
    bad_scores = int(df["__score_parsed"].isna().sum())
    if bad_scores > 0:
        errs.append(f"score-unparseable-count:{bad_scores}")

    # --- Test: duplicates ---
    dupe_subset = [c for c in ["TIME", "SCORE", "DESCRIPTION", "RAW"] if c in df.columns]
    if dupe_subset:
        dupe_count = int(df.duplicated(subset=dupe_subset).sum())
        if dupe_count > 0:
            errs.append(f"duplicate-events:{dupe_count}")

    # --- Test: size heuristics ---
    if len(df) < 50:
        errs.append("very-few-events")
    if len(df) > 5000:
        errs.append("very-many-events")

    # --- Test: PERIOD checks (only if present) ---
    if "PERIOD" in df.columns:
        df["PERIOD"] = pd.to_numeric(df.get("PERIOD"), errors="coerce")
        if df["PERIOD"].isna().any():
            errs.append("period-missing-or-non-numeric")
        if df["PERIOD"].min() < 1:
            errs.append("period-below-1")
        for p, g in df.groupby("PERIOD"):
            times = g["__tsec"].dropna().tolist()
            if len(times) >= 2:
                for i in range(1, len(times)):
                    if times[i] > times[i-1] + 1:
                        errs.append(f"period-{p}-time-increasing-at-index-{i}")
                        break

    return errs


def validate_all():
    df = load()
    df = normalize(df)
    miss = basic_schema_checks(df)
    if miss:
        print("Missing required columns:", miss)
        return
    results = {}
    for gid, g in df.groupby("GAME_ID"):
        results[gid] = validate_game_rows(g)
    fails = {gid: errs for gid, errs in results.items() if errs}
    print(f"Games checked: {len(results)}, failures: {len(fails)}")
    for gid, errs in list(fails.items())[:50]:
        print(gid, errs)
    if TEAM_LOGS.exists():
        try:
            tgl = pd.read_parquet(TEAM_LOGS)
            tgl = tgl.astype({"GAME_ID": str})
            print("Team logs loaded; you can add final-score comparison checks.")
        except Exception:
            print("Could not read team_game_logs.parquet")


if __name__ == "__main__":
    validate_all()
# tools/validate_pbp.py
import pandas as pd
import re
from pathlib import Path

PBP_FILE = "data/historical/play_by_play_2022-23.parquet"
TEAM_LOGS = "data/historical/team_game_logs.parquet"  # optional

TIME_RE = re.compile(r"(\d{1,2}):(\d{2})")
SCORE_RE = re.compile(r"(\d{1,3})\s*[–-]\s*(\d{1,3})")

def parse_time_to_seconds(s):
    if pd.isna(s): return None
    m = TIME_RE.search(str(s))
    if not m: return None
    mm, ss = int(m.group(1)), int(m.group(2))
    return mm * 60 + ss

def parse_score(s):
    if pd.isna(s): return None
    m = SCORE_RE.search(str(s))
    if not m: return None
    return int(m.group(1)), int(m.group(2))

def load():
    df = pd.read_parquet(PBP_FILE)
    return df

def basic_schema_checks(df):
    required = ["GAME_ID"]
    missing = [c for c in required if c not in df.columns]
    return missing

def validate_game_rows(df_game):
    errs = []
    # parse fields
    df = df_game.copy()
    df["PERIOD"] = pd.to_numeric(df.get("PERIOD", None), errors="coerce")
    df["__tsec"] = df.get("TIME").map(parse_time_to_seconds)
    df["__score_parsed"] = df.get("SCORE").map(parse_score)
    # emptiness
    if len(df) == 0:
        errs.append("empty")
        return errs
    # periods valid
    if df["PERIOD"].isna().any():
        errs.append("period-missing-or-non-numeric")
    if df["PERIOD"].min() < 1:
        errs.append("period-below-1")
    # times parseable
    if df["__tsec"].isna().all():
        errs.append("no-time-parseable")
    # times monotonic per period (non-increasing)
    for p, g in df.groupby("PERIOD"):
        times = g["__tsec"].tolist()
        # drop None
        times = [t for t in times if t is not None]
        if len(times) >= 2:
            # check non-increasing
            for i in range(1, len(times)):
                if times[i] > times[i-1] + 1:  # allow 1s jitter
                    errs.append(f"period-{p}-time-increasing-at-index-{i}")
                    break
    # score parseability
    bad_scores = df["__score_parsed"].isna().sum()
    if bad_scores > 0:
        errs.append(f"score-unparseable-count:{bad_scores}")
    # duplicates
    dupe_count = df.duplicated(subset=["TIME", "SCORE", "DESCRIPTION", "RAW"] if "DESCRIPTION" in df.columns else ["TIME","SCORE"]).sum()
    if dupe_count > 0:
        errs.append(f"duplicate-events:{dupe_count}")
    # size heuristics
    if len(df) < 50:
        errs.append("very-few-events")
    if len(df) > 5000:
        errs.append("very-many-events")
    return errs

def validate_all():
    df = load()
    miss = basic_schema_checks(df)
    if miss:
        print("Missing required columns:", miss)
        return
    results = {}
    for gid, g in df.groupby("GAME_ID"):
        results[gid] = validate_game_rows(g)
    # summarize
    fails = {gid: errs for gid, errs in results.items() if errs}
    print(f"Games checked: {len(results)}, failures: {len(fails)}")
    for gid, errs in list(fails.items())[:50]:
        print(gid, errs)
    # optional: compare final score vs team_game_logs if available
    tgl_path = Path(TEAM_LOGS)
    if tgl_path.exists():
        tgl = pd.read_parquet(tgl_path)
        tgl = tgl.astype({"GAME_ID": str})
        final_scores = {}
        for gid, g in df.groupby("GAME_ID"):
            # find last non-null score
            sp = g["SCORE"].dropna().map(parse_score).dropna().tolist()
            if sp:
                final_scores[gid] = sp[-1]
        # coerce team logs final scores if columns exist like PTS_HOME/PTS_AWAY or SCORE
        # user should adapt this section to their schema
        mismatches = []
        for gid, fs in final_scores.items():
            row = tgl[tgl["GAME_ID"] == str(gid)]
            if row.empty: continue
            # example: look for columns 'PTS_HOME','PTS_AWAY' or 'PTS' per team
            # skip implementation details — adapt to your team_game_logs schema
        print("Done. Inspect mismatches manually if needed.")

if __name__ == "__main__":
    validate_all()