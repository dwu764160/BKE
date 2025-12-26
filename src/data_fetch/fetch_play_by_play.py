"""
Fetch NBA play-by-play using Playwright network capture.

Usage:
  python src/data_fetch/fetch_play_by_play.py --seasons 2022-23 2023-24

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

DATA_DIR = "data/historical"
PBP_CACHE_DIR = "data/historical/pbp_cache"
CACHE_FILE = "data/historical/pbp_fetched.json"
SOURCE_CANDIDATES = [
    "data/historical/team_game_logs.parquet",
    "data/team_game_logs.parquet",
]

PBP_URL_TEMPLATE = "https://www.nba.com/game/{game_id}/play-by-play"


def load_game_logs():
    for p in SOURCE_CANDIDATES:
        if os.path.exists(p):
            try:
                df = pd.read_parquet(p)
                return df
            except Exception as e:
                print(f"Warning: could not read {p}: {e}")
    raise FileNotFoundError("No team_game_logs.parquet found in SOURCE_CANDIDATES")


def load_fetched_cache() -> set:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                data = json.load(f)
                return set(data)
        except Exception:
            return set()
    return set()


def save_fetched_cache(s: set):
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(sorted(list(s)), f)
    except Exception as e:
        print(f"Warning: could not write cache file {CACHE_FILE}: {e}")


def save_game_pbp(game_id: str, df: pd.DataFrame):
    Path(PBP_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    out = os.path.join(PBP_CACHE_DIR, f"pbp_{game_id}.parquet")
    try:
        df.to_parquet(out, index=False)
    except Exception as e:
        print(f"  Could not write {out}: {e}")


import re


def extract_pbp_from_next_data(page):
    try:
        next_data = page.evaluate('''() => { const el = document.querySelector("script#__NEXT_DATA__"); return el ? el.textContent : null; }''')
        if not next_data:
            return None
        j = json.loads(next_data)

        def find_key(obj, key_lower='playbyplay'):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if key_lower in k.lower():
                        return v
                    res = find_key(v, key_lower)
                    if res:
                        return res
            elif isinstance(obj, list):
                for item in obj:
                    res = find_key(item, key_lower)
                    if res:
                        return res
            return None

        pbp = find_key(j, 'playbyplay')
        if not pbp:
            return None

        # Handle NBA-stats-style payload
        if isinstance(pbp, dict) and 'resultSets' in pbp:
            rs = pbp.get('resultSets')[0]
            headers_row = rs.get('headers') or rs.get('rowSetHeaders') or []
            rows = rs.get('rowSet') or rs.get('rows') or []
            if headers_row and rows is not None:
                return pd.DataFrame(rows, columns=headers_row)

        # If list of dicts
        if isinstance(pbp, list) and pbp and isinstance(pbp[0], dict):
            return pd.DataFrame(pbp)

    except Exception as e:
        print(f"extract_pbp_from_next_data error: {e}")
    return None


def extract_pbp_from_dom(page):
    try:
        # Try several container selectors known from NBA site layouts
        selectors = [
            'section:has-text("Play-By-Play")',
            'div:has-text("Play-By-Play")',
            'div[class*="play-by-play"]',
            'ul[class*="pbp"]',
            '.pbp',
        ]
        for sel in selectors:
            try:
                conts = page.query_selector_all(sel)
            except Exception:
                conts = []
            for c in conts:
                # look for list items or rows inside the container
                items = c.query_selector_all('li, tr, .pbp-item, .play, .row')
                if not items:
                    items = c.query_selector_all('div')
                rows = []
                for it in items:
                    try:
                        text = it.inner_text().strip()
                    except Exception:
                        continue
                    if not text:
                        continue
                    if not re.search(r'\b\d{1,2}:\d{2}\b', text):
                        continue
                    lines = [ln.strip() for ln in text.split('\n') if ln.strip()]
                    time_val = ''
                    tm = re.search(r'\b\d{1,2}:\d{2}\b', text)
                    if tm:
                        time_val = tm.group(0)
                    score = ''
                    sm = re.search(r'(\d{1,3}\s*[–-]\s*\d{1,3})', text)
                    if sm:
                        score = sm.group(0)
                    rows.append({
                        'TIME': time_val,
                        'SCORE': score,
                        'DESCRIPTION': ' | '.join(lines),
                        'RAW': text,
                    })
                if rows:
                    return pd.DataFrame(rows)
    except Exception as e:
        print(f"extract_pbp_from_dom error: {e}")
    return None


def fetch_season_play_by_play(season: str, game_ids: list, fetched_cache: set):
    remaining_games = [g for g in game_ids if g not in fetched_cache]
    if not remaining_games:
        print(f"No games to fetch for {season}")
        return 0

    newly_fetched = 0
    response_urls: list[str] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        # global listener to record response URLs
        page.on('response', lambda r: response_urls.append(r.url))

        # Warm-up navigation to set cookies and session state
        print(f"[{season}] Warming up browser session...")
        try:
            page.goto("https://www.nba.com", wait_until="domcontentloaded", timeout=30000)
            time.sleep(3)
        except Exception as e:
            print(f"  Warm-up navigation warning: {e}")

        def make_handler(game_id):
            def handle_response(response):
                try:
                    if "playbyplayv2" in response.url:
                        try:
                            j = response.json()
                        except Exception:
                            return
                        if "resultSets" in j and j["resultSets"]:
                            rs = j["resultSets"][0]
                            headers_row = rs.get("headers") or rs.get("rowSetHeaders") or []
                            rows = rs.get("rowSet") or rs.get("rows") or []
                            if headers_row and rows is not None:
                                df = pd.DataFrame(rows, columns=headers_row)
                                df["GAME_ID"] = game_id
                                save_game_pbp(game_id, df)
                                fetched_cache.add(game_id)
                                save_fetched_cache(fetched_cache)
                                print(f"  ✓ Saved {len(df)} rows for {game_id}")
                except Exception as e:
                    print(f"  Parse error for {game_id}: {e}")
            return handle_response

        for idx, current_game_id in enumerate(remaining_games, 1):
            print(f"[{season}] {idx}/{len(remaining_games)} → {current_game_id}")
            response_urls.clear()

            # Register a one-off handler to capture the playbyplayv2 response
            page.once("response", make_handler(current_game_id))

            try:
                # Navigate and capture response so we can detect HTTP errors (502, 503, etc.)
                try:
                    nav_resp = page.goto(
                        PBP_URL_TEMPLATE.format(game_id=current_game_id),
                        timeout=30000,
                        wait_until="domcontentloaded",
                    )
                except Exception as nav_e:
                    nav_resp = None
                    print(f"  Navigation error for {current_game_id}: {nav_e}")

                # If navigation returned a response, check HTTP status
                if nav_resp:
                    try:
                        status = nav_resp.status
                        print(f"    Navigation HTTP status: {status}")
                        if status is None or status >= 400:
                            # Save page snapshot and skip to embedded/dom extraction
                            snippet = page.content()[:4000]
                            debug_path = f"debug_nav_{current_game_id}.html"
                            try:
                                with open(debug_path, "w", encoding="utf-8") as fh:
                                    fh.write(snippet)
                                print(f"    Saved HTML snippet to {debug_path}")
                            except Exception as werr:
                                print(f"    Could not write debug HTML: {werr}")
                    except Exception:
                        pass

                time.sleep(1.0)

                # Click Play-By-Play tab to trigger API call — allow fallback without failing fast
                try:
                    with page.expect_response(lambda r: "playbyplayv2" in r.url, timeout=30000):
                        pbp_tab = page.locator('text="Play-By-Play"').first
                        pbp_tab.click(timeout=8000)
                        time.sleep(1.5)
                except Exception as click_err:
                    # alternative selector
                    try:
                        with page.expect_response(lambda r: "playbyplayv2" in r.url, timeout=30000):
                            page.get_by_role("button", name="Play-By-Play").click(timeout=8000)
                            time.sleep(1.5)
                    except Exception as fallback_err:
                        # do not raise; we'll attempt embedded/dom extraction below
                        print(f"    Play-By-Play click did not trigger API: {click_err}; {fallback_err}")

                newly_fetched += 1
            except Exception as e:
                print(f"  Failed {current_game_id}: {e}")

                # Try extracting embedded JSON first, then DOM
                df = None
                try:
                    df = extract_pbp_from_next_data(page)
                    if df is None:
                        df = extract_pbp_from_dom(page)
                except Exception as ex:
                    print(f"  Embedded/DOM extraction error: {ex}")

                if df is not None and len(df) > 0:
                    df["GAME_ID"] = current_game_id
                    save_game_pbp(current_game_id, df)
                    fetched_cache.add(current_game_id)
                    save_fetched_cache(fetched_cache)
                    print(f"  ✓ Extracted {len(df)} rows from page for {current_game_id}")
                    newly_fetched += 1
                    continue

                # Debug: show what responses were received
                pbp_responses = [url for url in response_urls if "playbyplay" in url.lower()]
                stats_responses = [url for url in response_urls if "stats.nba.com" in url]
                from urllib.parse import urlparse
                domains = set()
                for url in response_urls:
                    try:
                        domains.add(urlparse(url).netloc)
                    except:
                        pass
                print(f"    Total responses: {len(response_urls)}")
                print(f"    Unique domains: {len(domains)}")
                print(f"    Sample domains: {list(domains)[:10]}")
                print(f"    PBP-related: {len(pbp_responses)}")
                print(f"    stats.nba.com: {len(stats_responses)}")
                if pbp_responses:
                    print(f"    PBP URLs: {pbp_responses[:3]}")

                # Dump a small snippet of the Play-By-Play container if present
                try:
                    # Try a tolerant wait for a set of likely selectors (longer timeout)
                    combined = '[class*="PlayByPlay"], [class*="play-by-play"], [data-testid*="play"], article, section:has-text("Play-By-Play"), div:has-text("Play-By-Play")'
                    try:
                        el = page.wait_for_selector(combined, timeout=15000)
                    except Exception:
                        el = None

                    if el:
                        try:
                            html = el.inner_html()
                            text = el.inner_text()
                            print(f"    Play-By-Play container text snippet: {text[:400]!r}")
                            print(f"    Play-By-Play container HTML length: {len(html)}")
                        except Exception:
                            print("    Found container but could not read inner HTML/text")
                    else:
                        # Try searching frames in case content is inside an iframe
                        found = False
                        for fr in page.frames:
                            try:
                                fsel = fr.query_selector('[class*="PlayByPlay"], [class*="play-by-play"], [data-testid*="play"], article')
                                if fsel:
                                    found = True
                                    txt = fsel.inner_text()[:400]
                                    print(f"    Found Play-By-Play inside iframe: {txt!r}")
                                    break
                            except Exception:
                                continue
                        if not found:
                            print("    Play-By-Play container not found by selectors or frames")
                except Exception as dom_err:
                    print(f"    Error inspecting Play-By-Play container: {dom_err}")

                # Take screenshot to debug
                try:
                    screenshot_path = f"debug_fail_{current_game_id}.png"
                    page.screenshot(path=screenshot_path)
                    print(f"    Screenshot saved: {screenshot_path}")
                except Exception as ss_err:
                    print(f"    Screenshot error: {ss_err}")

                # Stop after 3 failures to inspect
                if idx >= 3:
                    print(f"\n  Stopping after 3 failures for debugging. Check screenshots and response logs.")
                    break

            if idx % 25 == 0:
                time.sleep(5)

        browser.close()

    return newly_fetched


def main(seasons: list[str]):
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    Path(PBP_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    # Load cache of already-fetched games
    fetched_cache = load_fetched_cache()
    print(f"Loaded cache: {len(fetched_cache)} games already fetched")

    games = load_game_logs()
    games["GAME_ID"] = games["GAME_ID"].astype(str)

    for season in seasons:
        print(f"\nProcessing season {season}")
        season_games = games[games["SEASON"] == season]
        game_ids = season_games["GAME_ID"].unique().tolist()

        # Fetch games not in cache
        newly_fetched = fetch_season_play_by_play(season, game_ids, fetched_cache)
        
        # Combine all per-game files for this season into one parquet
        season_records = []
        for gid in game_ids:
            game_file = os.path.join(PBP_CACHE_DIR, f"pbp_{gid}.parquet")
            if os.path.exists(game_file):
                try:
                    season_records.append(pd.read_parquet(game_file))
                except Exception as e:
                    print(f"  Warning: could not read {game_file}: {e}")
        
        if season_records:
            combined = pd.concat(season_records, ignore_index=True)
            out_path = f"{DATA_DIR}/play_by_play_{season}.parquet"
            combined.to_parquet(out_path, index=False)
            print(f"✓ Combined {len(season_records)} games ({len(combined)} rows) → {out_path}")
        else:
            print(f"No data for {season}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", required=True)
    args = parser.parse_args()

    main(args.seasons)