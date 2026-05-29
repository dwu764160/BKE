"""
scripts/backfill_tracking_missing.py
=============================================================================
Fetch the 8 leaguedashptstats tracking measures that failed for pre-2022
seasons (2017-18 through 2021-22) and 2025-26.

Root cause: the NBA API now requires a full ~30-param query string. The
original fetcher sent only 6 minimal params, causing HTTP 500. fetch_tracking_data.py
is now fixed with _PTSTATS_DEFAULTS; this script re-runs the missing files.

Only fetches what's missing (parquet file absent). Already-present files
are skipped. Runtime: ~6 seasons × 8 measures × ~2s = ~96s total.

Usage:
    python3 scripts/backfill_tracking_missing.py [--seasons 2025-26 2021-22 ...]
=============================================================================
"""

from __future__ import annotations

import argparse
import sys
import time
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data_fetch.fetch_tracking_data import (
    fetch_url_cached,
    _PTSTATS_DEFAULTS,
    TRACKING_MEASURES,
    DATA_DIR,
)

MISSING_SEASONS = [
    "2017-18", "2018-19", "2019-20", "2020-21", "2021-22", "2025-26",
]

PTSTATS_URL = "https://stats.nba.com/stats/leaguedashptstats"


def fetch_missing_for_season(season: str, dry_run: bool = False) -> dict:
    season_dir = DATA_DIR / season
    season_dir.mkdir(parents=True, exist_ok=True)

    results = {"ok": [], "fail": [], "skip": []}

    for measure_name, (api_param, slug) in TRACKING_MEASURES.items():
        outfile = season_dir / f"tracking_{measure_name}.parquet"
        cache_key = f"tracking_{measure_name}_{season}"

        if outfile.exists():
            results["skip"].append(measure_name)
            continue

        if dry_run:
            print(f"    [DRY] would fetch {measure_name}")
            continue

        params = {
            **_PTSTATS_DEFAULTS,
            "LeagueID": "00",
            "PerMode": "PerGame",
            "PlayerOrTeam": "Player",
            "PtMeasureType": api_param,
            "Season": season,
            "SeasonType": "Regular Season",
        }

        df = fetch_url_cached(PTSTATS_URL, params, slug, cache_key)

        if df is not None and not df.empty:
            df.columns = [c.upper() for c in df.columns]
            df.to_parquet(outfile, index=False)
            results["ok"].append(measure_name)
            print(f"    ✓ {measure_name}: {len(df)} players")
        else:
            results["fail"].append(measure_name)
            print(f"    ✗ {measure_name}: empty or failed")

        time.sleep(random.uniform(1.5, 2.5))

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", default=MISSING_SEASONS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("Tracking data backfill — missing leaguedashptstats measures")
    print("=" * 60)
    print(f"Seasons: {args.seasons}")
    print(f"Measures: {list(TRACKING_MEASURES.keys())}")
    print()

    grand_ok = grand_fail = grand_skip = 0

    for season in args.seasons:
        print(f"\n--- {season} ---")
        res = fetch_missing_for_season(season, dry_run=args.dry_run)
        grand_ok += len(res["ok"])
        grand_fail += len(res["fail"])
        grand_skip += len(res["skip"])
        if res["skip"]:
            print(f"  Skipped (already exist): {res['skip']}")

    print(f"\n{'=' * 60}")
    print(f"Total: {grand_ok} fetched, {grand_fail} failed, {grand_skip} skipped")
    if grand_fail:
        print("  Some fetches failed — re-run to retry (cache-first logic skips successes)")


if __name__ == "__main__":
    main()
