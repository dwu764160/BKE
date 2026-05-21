"""
src/data_fetch/fetch_player_salaries.py
=============================================================================
Fetches NBA player salary data for all available seasons by scraping Spotrac.
For each player listed on the Spotrac contracts page, visits their contract page and extracts per-season salary data.
Joins with player ID/name mapping from the pipeline, computes years in league, and outputs a parquet file:
    data/historical/player_salaries.parquet
with columns: player_id, player_name, years_in_league, season, salary

Usage:
    python3 src/data_fetch/fetch_player_salaries.py

Notes:
- Be polite to Spotrac: the script sleeps between requests.
- This script fetches all available player-seasons, not just a fixed set.
- Output is joined to your player ID/name map for downstream use.
=============================================================================
"""
import requests
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import time
import re
import unicodedata
from bs4 import BeautifulSoup
import os
from pathlib import Path

OUTPUT_PARQUET = "data/historical/player_salaries.parquet"


def _load_player_id_name_map() -> pd.DataFrame:
    """Load PERSON_ID + DISPLAY_FIRST_LAST from available player-id map files."""
    hist_dir = Path("data/historical")
    candidates = [
        hist_dir / "player_id_name_map_all.csv",
        hist_dir / "player_id_name_map_['2022-23', '2023-24', '2024-25'].csv",
    ]
    candidates.extend(sorted(hist_dir.glob("player_id_name_map_*.csv")))

    seen = set()
    ordered_candidates = []
    for path in candidates:
        norm = str(path)
        if norm not in seen:
            seen.add(norm)
            ordered_candidates.append(path)

    frames = []
    for path in ordered_candidates:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"Warning: could not read player map {path}: {exc}")
            continue

        if {"PERSON_ID", "DISPLAY_FIRST_LAST"}.issubset(df.columns):
            frames.append(df[["PERSON_ID", "DISPLAY_FIRST_LAST"]].copy())

    if not frames:
        raise FileNotFoundError(
            "No usable player_id_name_map_*.csv found in data/historical. "
            "Run src/data_fetch/fetch_historical_data.py first."
        )

    merged = pd.concat(frames, ignore_index=True)
    merged["PERSON_ID"] = pd.to_numeric(merged["PERSON_ID"], errors="coerce").astype("Int64")
    merged = merged.dropna(subset=["PERSON_ID", "DISPLAY_FIRST_LAST"]).copy()
    merged["PERSON_ID"] = merged["PERSON_ID"].astype(int)
    merged = merged.drop_duplicates(subset=["PERSON_ID"], keep="first")
    return merged


def get_espn_player_salaries():
    """
    Scrape ESPN NBA salary pages for 2022-23 to 2025-26, handling pagination.
    Returns a DataFrame with columns: player_name, team, salary, season
    """
    season_map = {
        '2017-18': '2018',
        '2018-19': '2019',
        '2019-20': '2020',
        '2020-21': '2021',
        '2021-22': '2022',
        '2022-23': '2023',
        '2023-24': '2024',
        '2024-25': '2025',
        '2025-26': None  # ESPN default page
    }
    base_url = "https://www.espn.com/nba/salaries"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    data = []
    for season, year in season_map.items():
        print(f"Scraping ESPN salaries for {season}...")
        page = 1
        while True:
            if year is None:
                # 2025-26 (current season): page 1 is base_url, page 2+ is /_/page/{page}
                url = base_url if page == 1 else f"{base_url}/_/page/{page}"
            else:
                url = f"{base_url}/_/year/{year}" if page == 1 else f"{base_url}/_/year/{year}/page/{page}"
            print(f"  Fetching {url}")
            resp = requests.get(url, headers=headers, timeout=20)
            if resp.status_code != 200:
                print(f"    Failed to fetch {url}: {resp.status_code}")
                break
            soup = BeautifulSoup(resp.text, "html.parser")
            table = soup.find("table")
            if not table:
                print(f"    No table found at {url}")
                break
            rows = table.find_all("tr")
            found = 0
            for row in rows[1:]:  # skip header
                tds = row.find_all("td")
                if len(tds) < 4:
                    continue
                player_name = tds[1].get_text(strip=True)
                team = tds[2].get_text(strip=True)
                salary_str = tds[3].get_text(strip=True).replace("$","").replace(",","")
                try:
                    salary = int(salary_str)
                except Exception:
                    continue
                if player_name and team and salary:
                    data.append({"player_name": player_name, "team": team, "salary": salary, "season": season})
                    found += 1
            print(f"    Parsed {found} players from page {page}")
            # ESPN paginates with 40 players per page; stop if fewer than 40 found
            if found < 40:
                break
            page += 1
            time.sleep(0.5)
    return pd.DataFrame(data)

def normalize_name(name):
    # Remove trailing position (", G", ", F", ", C", etc.) if present
    name = re.sub(r",\s*[a-z]$", "", name.strip(), flags=re.IGNORECASE)
    # Decompose diacritics (e.g., č→c, ć→c, ö→o) before lowering
    name = unicodedata.normalize("NFKD", name)
    name = "".join(ch for ch in name if not unicodedata.combining(ch))
    name = name.lower()
    name = re.sub(r"[^a-z0-9 ]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name

def main():
    # Load player ID/name map
    id_map = _load_player_id_name_map()
    id_map["norm_name"] = id_map["DISPLAY_FIRST_LAST"].apply(normalize_name)

    print(f"Fetching all player salary data from ESPN for 2017-18 to 2025-26...")
    salary_df = get_espn_player_salaries()
    if salary_df.empty:
        print("No salary data fetched.")
        return
    salary_df["norm_name"] = salary_df["player_name"].apply(normalize_name)
    print("Sample ESPN normalized player names:")
    print(salary_df[["player_name", "norm_name"]].head(10))
    print("Sample ID map normalized player names:")
    print(id_map[["DISPLAY_FIRST_LAST", "norm_name"]].head(10))

    # Merge with player ID map
    merged = pd.merge(salary_df, id_map, on="norm_name", how="left")
    # Map normalized team names to NBA TEAM_IDs
    TEAM_NAME_TO_ID = {
        "atlanta hawks": "1610612737",
        "boston celtics": "1610612738",
        "brooklyn nets": "1610612751",
        "charlotte hornets": "1610612766",
        "chicago bulls": "1610612741",
        "cleveland cavaliers": "1610612739",
        "dallas mavericks": "1610612742",
        "denver nuggets": "1610612743",
        "detroit pistons": "1610612765",
        "golden state warriors": "1610612744",
        "houston rockets": "1610612745",
        "indiana pacers": "1610612754",
        "la clippers": "1610612746",
        "los angeles clippers": "1610612746",
        "los angeles lakers": "1610612747",
        "memphis grizzlies": "1610612763",
        "miami heat": "1610612748",
        "milwaukee bucks": "1610612749",
        "minnesota timberwolves": "1610612750",
        "new orleans pelicans": "1610612740",
        "new york knicks": "1610612752",
        "oklahoma city thunder": "1610612760",
        "orlando magic": "1610612753",
        "philadelphia 76ers": "1610612755",
        "phoenix suns": "1610612756",
        "portland trail blazers": "1610612757",
        "sacramento kings": "1610612758",
        "san antonio spurs": "1610612759",
        "toronto raptors": "1610612761",
        "utah jazz": "1610612762",
        "washington wizards": "1610612764"
    }
    def get_team_id(team_name):
        norm = re.sub(r"[^a-z0-9 ]", "", str(team_name).lower()) if pd.notnull(team_name) else None
        return TEAM_NAME_TO_ID.get(norm, None)
    merged["team_id"] = merged["team"].apply(get_team_id)

    # Write one parquet file per season
    os.makedirs(os.path.dirname(OUTPUT_PARQUET), exist_ok=True)
    for season, group in merged.groupby("season"):
        out = group.copy()
        # Use ID-map name when matched, fall back to original ESPN name
        out["player_name_final"] = out["DISPLAY_FIRST_LAST"].fillna(out["player_name"])
        out["player_id_final"] = out["PERSON_ID"]
        out = out[["player_id_final", "player_name_final", "team", "team_id", "season", "salary"]].rename(columns={
            "player_id_final": "player_id",
            "player_name_final": "player_name",
        })
        season_file = f"data/historical/player_salaries_{season}.parquet"
        table = pa.Table.from_pandas(out, preserve_index=False)
        pq.write_table(table, season_file)
        print(f"Wrote {season_file} with {len(out)} rows.")

    # Print summary: number of unique players per season
    print("\nSummary: unique players scraped per season:")
    for season, group in merged.groupby("season"):
        n_players = group["PERSON_ID"].nunique()
        print(f"  {season}: {n_players} players")

if __name__ == "__main__":
    main()