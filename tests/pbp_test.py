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