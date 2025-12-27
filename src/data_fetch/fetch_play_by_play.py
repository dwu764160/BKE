"""
Fetch NBA play-by-play using NBA.com __NEXT_DATA__ (stable approach).

Usage:
  python src/data_fetch/fetch_play_by_play.py --seasons 2023-24 2024-25

Requires:
  - team_game_logs.parquet with GAME_ID + SEASON
"""

from playwright.sync_api import sync_playwright
import pandas as pd
import argparse
import time
import os
import json
from pathlib import Path
import re

DATA_DIR = "data/historical"
PBP_CACHE_DIR = f"{DATA_DIR}/pbp_cache"
CACHE_FILE = f"{DATA_DIR}/pbp_fetched.json"

SOURCE_CANDIDATES = [
    "data/historical/team_game_logs.parquet",
    "data/team_game_logs.parquet",
]

PBP_URL = "https://www.nba.com/game/{game_id}/play-by-play"


# -----------------------------
# Utilities
# -----------------------------

def load_team_game_logs():
    for p in SOURCE_CANDIDATES:
        if os.path.exists(p):
            return pd.read_parquet(p)
    raise FileNotFoundError("team_game_logs.parquet not found")


def load_cache() -> set:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return set(json.load(f))
    return set()


def save_cache(cache: set):
    with open(CACHE_FILE, "w") as f:
        json.dump(sorted(list(cache)), f)


def save_game_pbp(game_id: str, df: pd.DataFrame):
    Path(PBP_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    out = f"{PBP_CACHE_DIR}/pbp_{game_id}.parquet"
    df.to_parquet(out, index=False)


# -----------------------------
# Extraction logic
# -----------------------------

def extract_from_next_data(page):
    """Primary method: extract play-by-play from __NEXT_DATA__"""
    try:
        raw = page.evaluate("""
            () => {
                const el = document.querySelector("script#__NEXT_DATA__");
                return el ? el.textContent : null;
            }
        """)
        if not raw:
            return None

        j = json.loads(raw)

        def find_pbp(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if "playbyplay" in k.lower():
                        return v
                    res = find_pbp(v)
                    if res is not None:
                        return res
            elif isinstance(obj, list):
                for item in obj:
                    res = find_pbp(item)
                    if res is not None:
                        return res
            return None

        pbp = find_pbp(j)
        if pbp is None:
            return None

        # NBA stats style
        if isinstance(pbp, dict) and "resultSets" in pbp:
            rs = pbp["resultSets"][0]
            return pd.DataFrame(rs["rowSet"], columns=rs["headers"])

        # List of dicts
        if isinstance(pbp, list) and isinstance(pbp[0], dict):
            return pd.DataFrame(pbp)

    except Exception as e:
        print("NEXT_DATA extraction error:", e)

    return None


def extract_from_dom(page):
    """Last-resort fallback: parse visible DOM text"""
    rows = []
    try:
        items = page.query_selector_all("li, tr, div")
        for it in items:
            try:
                text = it.inner_text().strip()
            except:
                continue
            if not text:
                continue
            if not re.search(r"\b\d{1,2}:\d{2}\b", text):
                continue

            rows.append({"RAW_EVENT": text})

        if rows:
            return pd.DataFrame(rows)
    except Exception:
        pass

    return None


# -----------------------------
# Fetch season
# -----------------------------

def fetch_season(season, game_ids, fetched_cache):
    newly_fetched = 0

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--no-sandbox",
            ],
        )

        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800},
        )

        # 🚀 Block heavy assets
        context.route(
            "**/*",
            lambda route, request: (
                route.abort()
                if request.resource_type in {"image", "media", "font"}
                else route.continue_()
            ),
        )

        page = context.new_page()

        print(f"[{season}] Browser warm-up...")
        page.goto("https://www.nba.com", timeout=60000, wait_until="domcontentloaded")
        time.sleep(5)

        for idx, gid in enumerate(game_ids, 1):
            if gid in fetched_cache:
                continue

            print(f"[{season}] {idx}/{len(game_ids)} → {gid}")

            try:
                url = PBP_URL.format(game_id=gid)
                page.goto(url, timeout=90000, wait_until="domcontentloaded")

                # ✅ CRITICAL: wait for Next.js hydration
                page.wait_for_selector(
                    "script#__NEXT_DATA__", state ="attached", timeout=30000
                )

                df = extract_from_next_data(page)

                if df is None or df.empty:
                    print("  NEXT_DATA missing → DOM fallback")
                    df = extract_from_dom(page)

                if df is not None and not df.empty:
                    df["GAME_ID"] = gid
                    save_game_pbp(gid, df)
                    fetched_cache.add(gid)
                    save_cache(fetched_cache)
                    newly_fetched += 1
                    print(f"  ✓ saved {len(df)} rows")
                else:
                    print("  ✗ no play-by-play found")

            except Exception as e:
                print(f"  ERROR {gid}: {e}")

            time.sleep(1.0)

        browser.close()

    return newly_fetched


# -----------------------------
# Main
# -----------------------------

def main(seasons):
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    Path(PBP_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    fetched_cache = load_cache()
    print(f"Loaded cache: {len(fetched_cache)} games")

    games = load_team_game_logs()
    games["GAME_ID"] = games["GAME_ID"].astype(str)

    for season in seasons:
        print(f"\nProcessing season {season}")
        season_games = games[games["SEASON"] == season]
        game_ids = season_games["GAME_ID"].unique().tolist()

        fetch_season(season, game_ids, fetched_cache)

        # Combine season
        dfs = []
        for gid in game_ids:
            p = f"{PBP_CACHE_DIR}/pbp_{gid}.parquet"
            if os.path.exists(p):
                dfs.append(pd.read_parquet(p))

        if dfs:
            out = pd.concat(dfs, ignore_index=True)
            out_path = f"{DATA_DIR}/play_by_play_{season}.parquet"
            out.to_parquet(out_path, index=False)
            print(f"✓ Season saved → {out_path} ({len(out)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", required=True)
    args = parser.parse_args()
    main(args.seasons)
