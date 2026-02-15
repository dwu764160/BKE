"""
fetch_player_salaries.py
-----------------------------------
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
-----------------------------------
"""
import requests
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import time
import re
from bs4 import BeautifulSoup
import os

PLAYER_MAP_CSV = "data/historical/player_id_name_map_['2022-23', '2023-24', '2024-25'].csv"
OUTPUT_PARQUET = "data/historical/player_salaries.parquet"


def get_espn_player_salaries():
    """
    Scrape ESPN NBA salary pages for 2022-23 to 2025-26, handling pagination.
    Returns a DataFrame with columns: player_name, team, salary, season
    """
    season_map = {
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
    # Remove accents, lower, remove punctuation, join first/last
    name = name.lower()
    name = re.sub(r"[^a-z0-9 ]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name

def main():
    # Load player ID/name map
    id_map = pd.read_csv(PLAYER_MAP_CSV)
    id_map["norm_name"] = id_map["DISPLAY_FIRST_LAST"].apply(normalize_name)

    print(f"Fetching all player salary data from ESPN for 2022-23 to 2025-26...")
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
    # Compute years in league: for each player, min season to current
    merged["season_start"] = merged.groupby("PERSON_ID")["season"].transform("min")
    merged["years_in_league"] = merged.apply(lambda row: int(row["season"].split("-")[0]) - int(row["season_start"].split("-")[0]) + 1 if pd.notnull(row["season_start"]) else None, axis=1)

    # Write one parquet file per season
    os.makedirs(os.path.dirname(OUTPUT_PARQUET), exist_ok=True)
    for season, group in merged.groupby("season"):
        out = group[["PERSON_ID", "DISPLAY_FIRST_LAST", "team", "years_in_league", "season", "salary"]].rename(columns={
            "PERSON_ID": "player_id",
            "DISPLAY_FIRST_LAST": "player_name"
        })
        season_file = f"data/historical/player_salaries_{season}.parquet"
        table = pa.Table.from_pandas(out)
        pq.write_table(table, season_file)
        print(f"Wrote {season_file} with {len(out)} rows.")

    # Print summary: number of unique players per season
    print("\nSummary: unique players scraped per season:")
    for season, group in merged.groupby("season"):
        n_players = group["PERSON_ID"].nunique()
        print(f"  {season}: {n_players} players")

if __name__ == "__main__":
    main()