"""
scripts/fetch_kalshi_closing_lines.py

Fetches NBA game pregame closing lines from Kalshi's public API.
Saves to data/external/kalshi_closing_lines.csv

PREGAME vs IN-GAME:
    Kalshi game winner markets are open from days before the game until the game ENDS.
    `close_time` on each market is the resolution time (after the final buzzer).
    In-game prices reflect live score/time-remaining and cannot be used as a "closing line."

    PREGAME FENCE = close_time - 200 minutes.
    This lands 40-80 min before tipoff for regulation games and 20-40 min before
    tipoff for overtime games. The last trade before this fence = pregame closing line.

COVERAGE:
    Kalshi KXNBAGAME markets began April 2025.
    Historical API (settled before 2026-03-27) contains:
        - 2024-25 NBA Playoffs (Apr-Jun 2025)
        - 2025-26 Regular Season through ~Mar 26, 2026

TICKER FORMAT:
    Event: KXNBAGAME-{YY}{MON}{DD}{AWAY3}{HOME3}   e.g. KXNBAGAME-26MAR25TORLAC
    Market: {event_ticker}-{TEAM3}                  e.g. KXNBAGAME-26MAR25TORLAC-LAC
    result="yes" on TEAM3 market means TEAM3 won.

OUTPUT CSV SCHEMA:
    game_date (YYYY-MM-DD), home_team, away_team,
    kalshi_home_win_prob (float 0-1), home_result (int 1=home win 0=away win)
"""

import argparse
import sys
import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, timezone

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

BASE_URL = "https://external-api.kalshi.com/trade-api/v2"

# NBA standard 30-team set — filter out preseason/global-games teams
NBA_TEAMS = {
    "ATL", "BKN", "BOS", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
    "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
    "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS",
}

MONTH_MAP = {
    "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04",
    "MAY": "05", "JUN": "06", "JUL": "07", "AUG": "08",
    "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12",
}

# How far before close_time to place the pregame fence (minutes).
# NBA games last 2h10 (regulation) to 2h40 (triple-OT); 200 min puts us solidly pregame.
PREGAME_FENCE_MINUTES = 200


def parse_event_date(event_ticker: str) -> str:
    """
    Parse game date from event ticker like KXNBAGAME-26MAR25TORLAC.
    Returns YYYY-MM-DD string.
    """
    suffix = event_ticker.split("-", 1)[1]  # "26MAR25TORLAC"
    yy = suffix[:2]
    mon = suffix[2:5].upper()
    dd = suffix[5:7]
    year = f"20{yy}"
    month = MONTH_MAP.get(mon, "00")
    return f"{year}-{month}-{dd.zfill(2)}"


def parse_teams_from_event(event_ticker: str) -> tuple[str, str] | None:
    """
    Extract (away_team, home_team) from event ticker like KXNBAGAME-26MAR25TORLAC.
    Returns None if either team is not in the 30-team NBA set (e.g. preseason games).
    """
    suffix = event_ticker.split("-", 1)[1]  # "26MAR25TORLAC"
    teams_part = suffix[7:]  # "TORLAC" (after YY+MON+DD = 7 chars)
    if len(teams_part) != 6:
        return None
    away = teams_part[:3]
    home = teams_part[3:]
    if away not in NBA_TEAMS or home not in NBA_TEAMS:
        return None
    return away, home


def get_all_historical_markets() -> list[dict]:
    """Paginate through all historical KXNBAGAME settled markets."""
    markets = []
    cursor = ""
    page = 0
    while True:
        params = {"series_ticker": "KXNBAGAME", "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        resp = requests.get(f"{BASE_URL}/historical/markets", params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        batch = data.get("markets", [])
        markets.extend(batch)
        cursor = data.get("cursor", "")
        page += 1
        print(f"  Fetched page {page}: {len(batch)} markets (total {len(markets)})")
        if not cursor or not batch:
            break
        time.sleep(0.4)
    return markets


def _parse_ts(ts_str: str) -> datetime:
    """Parse Kalshi timestamp strings which may have variable sub-second precision."""
    # Normalize: drop sub-second entirely for comparison purposes
    ts_str = ts_str.replace("Z", "+00:00")
    # Truncate to microseconds (6 decimal places max for fromisoformat)
    if "." in ts_str:
        dot_idx = ts_str.index(".")
        tz_idx = next((i for i, c in enumerate(ts_str) if c in "+-" and i > dot_idx), len(ts_str))
        frac = ts_str[dot_idx + 1:tz_idx][:6].ljust(6, "0")
        ts_str = ts_str[:dot_idx + 1] + frac + ts_str[tz_idx:]
    return datetime.fromisoformat(ts_str)


def get_pregame_price(ticker: str, close_time_str: str) -> float | None:
    """
    Fetch historical trades for `ticker` and return the last YES price before
    the pregame fence (close_time - PREGAME_FENCE_MINUTES).

    Returns None if no trades exist before the fence.
    """
    close_dt = _parse_ts(close_time_str)
    fence_dt = close_dt - timedelta(minutes=PREGAME_FENCE_MINUTES)

    all_trades = []
    cursor = ""
    while True:
        params = {"ticker": ticker, "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        resp = requests.get(f"{BASE_URL}/historical/trades", params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        batch = data.get("trades", [])
        all_trades.extend(batch)
        cursor = data.get("cursor", "")
        if not cursor or not batch:
            break
        time.sleep(0.15)

    if not all_trades:
        return None

    # Sort by time, filter to before fence
    all_trades.sort(key=lambda t: t["created_time"])
    pregame = [
        t for t in all_trades
        if _parse_ts(t["created_time"]) < fence_dt
    ]

    if not pregame:
        return None

    return float(pregame[-1]["yes_price_dollars"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=str(REPO / "data/external/kalshi_closing_lines.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max games to process (0 = all, useful for quick tests)",
    )
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Step 1: Fetching all historical KXNBAGAME markets...")
    all_markets = get_all_historical_markets()
    print(f"  Total markets: {len(all_markets)} ({len(all_markets)//2} games)")

    # Group into events (games): each event has 2 markets (one per team)
    # Key: event_ticker (everything except the trailing -TEAM suffix)
    from collections import defaultdict
    events = defaultdict(list)
    for m in all_markets:
        ticker = m["ticker"]
        parts = ticker.rsplit("-", 1)
        if len(parts) == 2:
            event_key = parts[0]  # e.g. KXNBAGAME-26MAR25TORLAC
            events[event_key].append(m)

    print(f"  Unique events (games): {len(events)}")

    rows = []
    errors = 0
    processed = 0

    for event_ticker, mkt_pair in events.items():
        if args.limit and processed >= args.limit:
            break

        team_result = parse_teams_from_event(event_ticker)
        if team_result is None:
            continue  # Skip preseason / non-standard games

        away_team, home_team = team_result
        game_date = parse_event_date(event_ticker)

        # Find the home team market
        home_mkt = next(
            (m for m in mkt_pair if m["ticker"].endswith(f"-{home_team}")), None
        )
        if home_mkt is None:
            errors += 1
            continue

        result = home_mkt.get("result", "")
        if result not in ("yes", "no"):
            continue  # Unresolved or void

        home_result = 1 if result == "yes" else 0
        close_time = home_mkt.get("close_time", "")

        if not close_time:
            errors += 1
            continue

        print(f"  [{processed+1}] {game_date} {away_team}@{home_team} ...", end=" ", flush=True)
        price = get_pregame_price(home_mkt["ticker"], close_time)

        if price is None:
            print("no pregame trades — skipped")
            errors += 1
            continue

        print(f"price={price:.2f} result={'HOME WIN' if home_result else 'AWAY WIN'}")

        rows.append({
            "game_date": game_date,
            "home_team": home_team,
            "away_team": away_team,
            "kalshi_home_win_prob": round(price, 4),
            "home_result": home_result,
        })
        processed += 1
        time.sleep(0.05)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

    print(f"\nDone. {len(df)} games written to {out_path}")
    print(f"Errors/skipped: {errors}")
    if len(df):
        print(f"Date range: {df['game_date'].min()} to {df['game_date'].max()}")
        print(f"Avg Kalshi home win prob: {df['kalshi_home_win_prob'].mean():.3f}")
        print(f"Actual home win rate:     {df['home_result'].mean():.3f}")
        # Quick calibration check
        import numpy as np
        brier = np.mean((df['kalshi_home_win_prob'] - df['home_result'])**2)
        print(f"Kalshi Brier (sanity):    {brier:.4f} (expect ~0.200-0.215)")


if __name__ == "__main__":
    main()
