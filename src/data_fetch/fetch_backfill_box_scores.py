"""
src/data_fetch/fetch_backfill_box_scores.py
=============================================================================
Backfill box score data for pre-2022 seasons using nba_api.

Fetches base stats, advanced stats, and B-Ref win shares for seasons
not yet in the pipeline. curl_cffi impersonation is unreliable for older
seasons (returns empty rowsets); nba_api handles rate limiting correctly.

Output:
  data/historical/complete_player_season_stats.parquet (merged, all seasons)
  data/official_stats/official_advanced_{season}.parquet

Usage:
  python3 src/data_fetch/fetch_backfill_box_scores.py
  python3 src/data_fetch/fetch_backfill_box_scores.py --seasons 2021-22 2018-19
=============================================================================
"""

import argparse
import sys
import time
import random
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.modeling.model_config import SEASONS

try:
    from nba_api.stats.endpoints import (
        leaguedashplayerstats,
        leaguedashplayerbiostats,
    )
    NBA_API_AVAILABLE = True
except ImportError:
    NBA_API_AVAILABLE = False
    print("ERROR: nba_api not installed. Run: pip install nba_api")
    sys.exit(1)

HISTORICAL_DIR = Path("data/historical")
OFFICIAL_DIR = Path("data/official_stats")
HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)
OFFICIAL_DIR.mkdir(parents=True, exist_ok=True)

BACKFILL_SEASONS = ["2017-18", "2018-19", "2019-20", "2020-21", "2021-22"]


def fetch_base_stats(season: str) -> pd.DataFrame:
    """Fetch per-game base stats for all players in a season."""
    print(f"  Base stats...", end="", flush=True)
    time.sleep(random.uniform(1.0, 2.0))
    df = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        per_mode_detailed="PerGame",
        season_type_all_star="Regular Season",
        timeout=60,
    ).get_data_frames()[0]
    df["season"] = season
    print(f" {len(df)} players")
    return df


def fetch_advanced_stats(season: str) -> pd.DataFrame:
    """Fetch advanced stats for all players in a season."""
    print(f"  Advanced stats...", end="", flush=True)
    time.sleep(random.uniform(1.0, 2.0))
    df = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        measure_type_detailed_defense="Advanced",
        per_mode_detailed="PerGame",
        season_type_all_star="Regular Season",
        timeout=60,
    ).get_data_frames()[0]

    df["season"] = season
    print(f" {len(df)} players")
    return df


def fetch_totals(season: str) -> pd.DataFrame:
    """Fetch season totals for all players."""
    print(f"  Totals...", end="", flush=True)
    time.sleep(random.uniform(1.0, 2.0))
    df = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        per_mode_detailed="Totals",
        season_type_all_star="Regular Season",
        timeout=60,
    ).get_data_frames()[0]
    df["season"] = season
    print(f" {len(df)} players")
    return df


def merge_season(season: str) -> pd.DataFrame:
    """Fetch and merge base + advanced stats for one season."""
    base = fetch_base_stats(season)
    adv = fetch_advanced_stats(season)
    totals = fetch_totals(season)

    # Merge advanced onto base (keep all base players)
    adv_cols = [c for c in adv.columns if c not in base.columns or c in ("PLAYER_ID", "season")]
    merged = base.merge(
        adv[["PLAYER_ID", "season"] + [c for c in adv_cols if c not in ("PLAYER_ID", "season")]],
        on=["PLAYER_ID", "season"],
        how="left",
        suffixes=("", "_adv"),
    )

    # Add MIN totals from totals
    min_cols = ["PLAYER_ID", "MIN"]
    if "MIN" in totals.columns:
        totals_min = totals[["PLAYER_ID", "season", "MIN"]].rename(
            columns={"MIN": "MIN_TOTAL"}
        )
        merged = merged.merge(totals_min, on=["PLAYER_ID", "season"], how="left")

    return merged


def save_official_advanced(season: str, adv_df: pd.DataFrame) -> None:
    """Save official advanced stats per season (matches existing format)."""
    out = OFFICIAL_DIR / f"official_advanced_{season}.parquet"
    adv_df.to_parquet(out, index=False)
    print(f"  Saved: {out}")


def main(seasons: list[str]) -> None:
    existing_path = HISTORICAL_DIR / "complete_player_season_stats.parquet"
    if existing_path.exists():
        existing = pd.read_parquet(existing_path)
        # The existing file may use 'SEASON' (upper) instead of 'season'
        season_col = "season" if "season" in existing.columns else "SEASON"
        existing["season"] = existing[season_col].astype(str)
        existing_seasons = set(existing["season"].unique())
        print(f"Existing data: {sorted(existing_seasons)}")
    else:
        existing = pd.DataFrame()
        existing_seasons = set()

    new_dfs = []
    for season in seasons:
        if season in existing_seasons:
            print(f"\n{season}: already present — skipping")
            continue

        print(f"\n{season}:")
        try:
            df = merge_season(season)
            new_dfs.append(df)

            # Also save per-season official_advanced file
            adv_df = df[[c for c in df.columns if c not in ["MIN_TOTAL"]]].copy()
            save_official_advanced(season, adv_df)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    if not new_dfs:
        print("\nNothing new to add.")
        return


    # Only save new seasons separately — don't merge with existing (different schema)
    new_combined = pd.concat(new_dfs, ignore_index=True)
    new_combined["season"] = new_combined["season"].astype(str)
    # Normalize PLAYER_ID type to match existing parquet
    new_combined["PLAYER_ID"] = new_combined["PLAYER_ID"].astype(str)
    new_out = HISTORICAL_DIR / "complete_player_season_stats_backfill.parquet"
    new_combined.to_parquet(new_out, index=False)
    print(f"\nSaved backfill: {new_out} — {len(new_combined)} rows, seasons: {sorted(new_combined['season'].unique())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", default=BACKFILL_SEASONS)
    args = parser.parse_args()
    main(args.seasons)
