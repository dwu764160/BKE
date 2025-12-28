import time
import json
from pathlib import Path
import httpx

BASE_URL = "https://stats.nba.com/stats/playbyplayv2"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Origin": "https://www.nba.com",
    "Referer": "https://www.nba.com/",
    "Connection": "keep-alive",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}

def fetch_play_by_play(game_id: str, retries=3):
    params = {
        "GameID": game_id,
        "StartPeriod": 0,
        "EndPeriod": 14,
    }

    # try to load session headers/cookies captured from a real browser
    session_file = Path("data/nba_session.json")
    headers = HEADERS.copy()
    cookies = None
    if session_file.exists():
        try:
            s = json.load(session_file.open())
            if isinstance(s, dict):
                h = s.get("headers") or {}
                for k, v in h.items():
                    if k.lower() == "user-agent":
                        headers["User-Agent"] = v
                    elif k.lower() == "referer":
                        headers["Referer"] = v
                    else:
                        headers[k] = v

                cookie_list = s.get("cookies") or []
                if cookie_list:
                    cookies = {c["name"]: c.get("value", "") for c in cookie_list if "name" in c}
        except Exception:
            pass

    timeout = httpx.Timeout(connect=20.0, read=120.0, write=20.0, pool=20.0)

    for attempt in range(1, retries + 1):
        try:
            with httpx.Client(
                headers=headers,
                timeout=timeout,
                http2=True,
                cookies=cookies,
            ) as client:
                r = client.get(BASE_URL, params=params)

            r.raise_for_status()
            data = r.json()

            if not data or data == {}:
                raise RuntimeError("Empty JSON returned")

            return data

        except Exception as e:
            print(f"❌ Attempt {attempt} failed: {e}")
            time.sleep(2 * attempt)

    raise RuntimeError("Failed to fetch play-by-play")

if __name__ == "__main__":
    GAME_ID = "0022300001"
    pbp = fetch_play_by_play(GAME_ID)

    rows = pbp["resultSets"][0]["rowSet"]
    print("✅ Play-by-play rows:", len(rows))
