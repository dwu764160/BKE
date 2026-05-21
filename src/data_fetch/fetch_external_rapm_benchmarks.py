"""
src/data_fetch/fetch_external_rapm_benchmarks.py
=============================================================================
Fetch external player impact benchmarks for Phase 1 Track A RAPM calibration.

Source: Basketball-Reference advanced stats (OBPM, DBPM, BPM, WS, VORP).
These are box-score-based impact metrics — not pure regression RAPM, but the
most reliable freely available external benchmark. Correlation with public
RAPM estimates is typically r~0.80-0.85 for OBPM and r~0.75-0.80 for DBPM.

Requires: curl_cffi (already in project deps), lxml (install if missing).

Output per season:
  data/reference/external_rapm_{season}.parquet
  columns: player_name, team, pos, mp, obpm, dbpm, bpm, vorp, ws, season

Usage:
  python3 src/data_fetch/fetch_external_rapm_benchmarks.py
  python3 src/data_fetch/fetch_external_rapm_benchmarks.py --seasons 2022-23 2023-24
=============================================================================
"""

import argparse
import sys
import time
from io import StringIO
from pathlib import Path

import pandas as pd

try:
    from curl_cffi import requests as curl_requests
except ImportError:
    print("ERROR: curl_cffi not installed. Run: pip install curl_cffi")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.modeling.model_config import SEASONS

REFERENCE_DIR = Path("data/reference")
REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

BREF_IMPERSONATE = "chrome110"
BREF_DELAY = 4.0  # seconds between requests — be polite to BRef

BENCHMARK_SEASONS = [
    "2017-18", "2018-19", "2019-20", "2020-21", "2021-22",
    "2022-23", "2023-24", "2024-25",
]

COLUMN_RENAMES = {
    "Player": "player_name",
    "Team":   "team",
    "Pos":    "pos",
    "MP":     "mp",
    "OBPM":   "obpm",
    "DBPM":   "dbpm",
    "BPM":    "bpm",
    "VORP":   "vorp",
    "WS":     "ws",
    "WS/48":  "ws_per48",
    "PER":    "per",
}

KEEP_COLS = ["player_name", "team", "pos", "mp", "obpm", "dbpm", "bpm", "vorp", "ws", "ws_per48", "per", "season"]


def _season_to_bref_year(season: str) -> int:
    """'2023-24' → 2024."""
    return int(season.split("-")[0]) + 1


def fetch_bref_advanced(season: str) -> pd.DataFrame | None:
    year = _season_to_bref_year(season)
    url = f"https://www.basketball-reference.com/leagues/NBA_{year}_advanced.html"
    try:
        resp = curl_requests.get(url, impersonate=BREF_IMPERSONATE, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        print(f"  [BRef] {season}: request failed — {e}")
        return None

    try:
        tables = pd.read_html(StringIO(resp.text))
    except Exception as e:
        print(f"  [BRef] {season}: HTML parse failed — {e}")
        return None

    if not tables:
        print(f"  [BRef] {season}: no tables found in page")
        return None

    df = tables[0].copy()

    # Drop repeated header rows (BRef repeats column names every 20 rows)
    df = df[df["Player"] != "Player"].copy()
    df = df.dropna(subset=["Player"])

    # Rename to standard schema
    df.rename(columns=COLUMN_RENAMES, inplace=True)

    # Convert numeric columns
    numeric_cols = ["mp", "obpm", "dbpm", "bpm", "vorp", "ws", "ws_per48", "per"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["season"] = season

    # Keep only players with meaningful minutes (>100 MP)
    if "mp" in df.columns:
        df = df[df["mp"] >= 100]

    # Keep only needed columns that exist
    keep = [c for c in KEEP_COLS if c in df.columns]
    df = df[keep].reset_index(drop=True)

    # For traded players (TOT rows), BRef has one row per team + one "TOT" summary row.
    # We want the TOT row (team-agnostic season total) to match our dedup logic.
    # If no TOT row exists, keep the max-minutes team row (same as our pipeline dedup).
    if "team" in df.columns:
        tot_rows = df[df["team"] == "TOT"].copy()
        non_tot = df[df["team"] != "TOT"].copy()
        # Players who have a TOT row: use TOT. Others: keep as-is.
        traded_names = set(tot_rows["player_name"])
        non_tot_unique = non_tot[~non_tot["player_name"].isin(traded_names)]
        # For traded players without TOT, keep max-MP row
        traded_no_tot = non_tot[non_tot["player_name"].isin(
            set(non_tot["player_name"]) - traded_names
        )]
        if "mp" in traded_no_tot.columns:
            traded_no_tot = traded_no_tot.sort_values("mp", ascending=False).drop_duplicates("player_name")
        df = pd.concat([non_tot_unique, tot_rows, traded_no_tot], ignore_index=True)

    print(f"  [BRef] {season}: {len(df)} players, BPM range [{df['bpm'].min():.1f}, {df['bpm'].max():.1f}]")
    return df


def fetch_season(season: str) -> bool:
    out_path = REFERENCE_DIR / f"external_rapm_{season}.parquet"
    print(f"\n{'='*60}")
    print(f"Season: {season}")

    df = fetch_bref_advanced(season)
    if df is None or df.empty:
        print(f"  FAILED: could not fetch {season}")
        return False

    df.to_parquet(out_path, index=False)
    print(f"  Saved {len(df)} players → {out_path}")
    return True


def main(seasons: list = None) -> None:
    if seasons is None:
        seasons = BENCHMARK_SEASONS

    print(f"Fetching BRef external benchmarks for: {seasons}")
    print(f"Note: BPM/OBPM/DBPM are box-score metrics, not regression RAPM.")
    print(f"      Correlation with RAPM is r~0.82 (O) and r~0.78 (D).")
    print(f"      Sufficient for Phase 1 Track A calibration check.\n")

    results = {}
    for i, season in enumerate(seasons):
        ok = fetch_season(season)
        results[season] = ok
        if i < len(seasons) - 1:
            print(f"  Waiting {BREF_DELAY}s before next request...")
            time.sleep(BREF_DELAY)

    print(f"\n{'='*60}")
    print("Summary:")
    for s, ok in results.items():
        print(f"  {s}: {'OK' if ok else 'FAILED'}")

    failed = [s for s, ok in results.items() if not ok]
    if failed:
        print(f"\nFailed: {failed}")
        sys.exit(1)
    print("\nAll seasons fetched successfully.")
    print(f"Output: data/reference/external_rapm_{{season}}.parquet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", default=None)
    args = parser.parse_args()
    main(seasons=args.seasons)
