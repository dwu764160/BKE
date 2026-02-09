#!/usr/bin/env python3
"""Fill Basketball-Reference ground-truth CSV for a given season.
Usage:
  python scripts/fill_bref_ground_truth.py --season 2024-25
This will read bref_ground_truth_2024-25.csv, fetch stats from Basketball-Reference, and populate the bref_* columns. Any players in the CSV that can't be matched to BREF
"""



from __future__ import annotations

import argparse
import re
import unicodedata
from io import StringIO
from typing import Dict, Iterable, List

import cloudscraper
import pandas as pd


NAME_OVERRIDES: Dict[str, str] = {
    "Herb Jones": "Herbert Jones",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Populate bref_ground_truth_<season>.csv from Basketball-Reference"
    )
    parser.add_argument(
        "--season",
        default="2024-25",
        help="Season in YYYY-YY format (default: 2024-25)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output CSV path (default: tests/bref_ground_truth_<season>.csv)",
    )
    return parser.parse_args()


def season_to_bref_year(season: str) -> int:
    try:
        start_year = int(season.split("-")[0])
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Invalid season format: {season}") from exc
    return start_year + 1


def fetch_html(scraper: cloudscraper.CloudScraper, url: str) -> str:
    resp = scraper.get(url)
    resp.raise_for_status()
    html = resp.content.decode("utf-8", errors="replace")
    # Basketball-Reference wraps some tables in HTML comments.
    return html.replace("<!--", "").replace("-->", "")


def read_first_table(scraper: cloudscraper.CloudScraper, url: str) -> pd.DataFrame:
    html = fetch_html(scraper, url)
    tables = pd.read_html(StringIO(html))
    if not tables:
        raise ValueError(f"No tables found in {url}")
    df = tables[0]
    if "Player" in df.columns:
        df = df[df["Player"] != "Player"]
    return df


def norm_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")
    name = name.lower()
    return re.sub(r"[^a-z0-9]+", "", name)


def pick_tot_row(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[pd.Series] = []
    for player_norm, group in df.groupby("_player_norm", sort=False):
        if not player_norm:
            continue
        if (group["Tm"] == "TOT").any():
            row = group[group["Tm"] == "TOT"].iloc[0]
        else:
            row = group.iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


def ensure_columns(df: pd.DataFrame, needed: Iterable[str], label: str) -> None:
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing {label} columns: {missing}")


def fill_bref_csv(season: str, output_path: str) -> List[str]:
    bref_year = season_to_bref_year(season)
    totals_url = f"https://www.basketball-reference.com/leagues/NBA_{bref_year}_totals.html"
    adv_url = f"https://www.basketball-reference.com/leagues/NBA_{bref_year}_advanced.html"
    per_poss_url = (
        f"https://www.basketball-reference.com/leagues/NBA_{bref_year}_per_poss.html"
    )

    scraper = cloudscraper.create_scraper()

    totals = read_first_table(scraper, totals_url)
    adv = read_first_table(scraper, adv_url)
    per_poss = read_first_table(scraper, per_poss_url)

    for df in (totals, adv, per_poss):
        if "Tm" not in df.columns and "Team" in df.columns:
            df.rename(columns={"Team": "Tm"}, inplace=True)

    needed_totals = [
        "Player",
        "Tm",
        "G",
        "MP",
        "FG",
        "FGA",
        "3P",
        "3PA",
        "FT",
        "FTA",
        "ORB",
        "DRB",
        "TRB",
        "AST",
        "STL",
        "BLK",
        "TOV",
        "PF",
        "PTS",
    ]
    needed_adv = ["Player", "Tm", "TS%", "USG%", "BPM", "VORP", "WS", "OWS", "DWS"]
    needed_per_poss = ["Player", "Tm", "ORtg", "DRtg"]

    ensure_columns(totals, needed_totals, "totals")
    ensure_columns(adv, needed_adv, "advanced")
    ensure_columns(per_poss, needed_per_poss, "per-possession")

    totals = totals[needed_totals].copy()
    adv = adv[needed_adv].copy()
    per_poss = per_poss[needed_per_poss].copy()

    for col in needed_totals:
        if col not in ("Player", "Tm"):
            totals[col] = pd.to_numeric(totals[col], errors="coerce")
    for col in needed_adv:
        if col not in ("Player", "Tm"):
            adv[col] = pd.to_numeric(adv[col], errors="coerce")
    for col in needed_per_poss:
        if col not in ("Player", "Tm"):
            per_poss[col] = pd.to_numeric(per_poss[col], errors="coerce")

    for df in (totals, adv, per_poss):
        df["_player_norm"] = df["Player"].map(norm_name)

    totals = pick_tot_row(totals)
    adv = pick_tot_row(adv)
    per_poss = pick_tot_row(per_poss)

    def get_row(df: pd.DataFrame, player_name: str) -> pd.Series | None:
        lookup = NAME_OVERRIDES.get(player_name, player_name)
        key = norm_name(lookup)
        rows = df[df["_player_norm"] == key]
        if rows.empty:
            return None
        return rows.iloc[0]

    bref = pd.read_csv(output_path)
    missing_players: List[str] = []

    for idx, row in bref.iterrows():
        name = row["player_name"]
        t_row = get_row(totals, name)
        a_row = get_row(adv, name)
        p_row = get_row(per_poss, name)
        if t_row is None or a_row is None or p_row is None:
            missing_players.append(name)
            continue

        bref.at[idx, "bref_GP"] = int(t_row["G"]) if pd.notna(t_row["G"]) else ""
        bref.at[idx, "bref_MP"] = int(t_row["MP"]) if pd.notna(t_row["MP"]) else ""
        bref.at[idx, "bref_FGM"] = int(t_row["FG"]) if pd.notna(t_row["FG"]) else ""
        bref.at[idx, "bref_FGA"] = int(t_row["FGA"]) if pd.notna(t_row["FGA"]) else ""
        bref.at[idx, "bref_FG3M"] = int(t_row["3P"]) if pd.notna(t_row["3P"]) else ""
        bref.at[idx, "bref_FG3A"] = int(t_row["3PA"]) if pd.notna(t_row["3PA"]) else ""
        bref.at[idx, "bref_FTM"] = int(t_row["FT"]) if pd.notna(t_row["FT"]) else ""
        bref.at[idx, "bref_FTA"] = int(t_row["FTA"]) if pd.notna(t_row["FTA"]) else ""
        bref.at[idx, "bref_ORB"] = int(t_row["ORB"]) if pd.notna(t_row["ORB"]) else ""
        bref.at[idx, "bref_DRB"] = int(t_row["DRB"]) if pd.notna(t_row["DRB"]) else ""
        bref.at[idx, "bref_TRB"] = int(t_row["TRB"]) if pd.notna(t_row["TRB"]) else ""
        bref.at[idx, "bref_AST"] = int(t_row["AST"]) if pd.notna(t_row["AST"]) else ""
        bref.at[idx, "bref_STL"] = int(t_row["STL"]) if pd.notna(t_row["STL"]) else ""
        bref.at[idx, "bref_BLK"] = int(t_row["BLK"]) if pd.notna(t_row["BLK"]) else ""
        bref.at[idx, "bref_TOV"] = int(t_row["TOV"]) if pd.notna(t_row["TOV"]) else ""
        bref.at[idx, "bref_PF"] = int(t_row["PF"]) if pd.notna(t_row["PF"]) else ""
        bref.at[idx, "bref_PTS"] = int(t_row["PTS"]) if pd.notna(t_row["PTS"]) else ""

        ts = a_row["TS%"]
        usg = a_row["USG%"]
        bpm = a_row["BPM"]
        vorp = a_row["VORP"]
        ws = a_row["WS"]
        ows = a_row["OWS"]
        dws = a_row["DWS"]
        ortg = p_row["ORtg"]
        drtg = p_row["DRtg"]

        if pd.notna(usg) and usg > 1:
            usg = usg / 100.0

        bref.at[idx, "bref_TS_PCT"] = round(float(ts), 3) if pd.notna(ts) else ""
        bref.at[idx, "bref_USG_PCT"] = round(float(usg), 3) if pd.notna(usg) else ""
        bref.at[idx, "bref_ORTG"] = round(float(ortg), 1) if pd.notna(ortg) else ""
        bref.at[idx, "bref_DRTG"] = round(float(drtg), 1) if pd.notna(drtg) else ""
        bref.at[idx, "bref_BPM"] = round(float(bpm), 1) if pd.notna(bpm) else ""
        bref.at[idx, "bref_VORP"] = round(float(vorp), 1) if pd.notna(vorp) else ""
        bref.at[idx, "bref_WS"] = round(float(ws), 1) if pd.notna(ws) else ""
        bref.at[idx, "bref_OWS"] = round(float(ows), 1) if pd.notna(ows) else ""
        bref.at[idx, "bref_DWS"] = round(float(dws), 1) if pd.notna(dws) else ""

    bref.to_csv(output_path, index=False)
    return missing_players


def main() -> None:
    args = parse_args()
    output_path = (
        args.output
        if args.output is not None
        else f"tests/bref_ground_truth_{args.season}.csv"
    )
    missing = fill_bref_csv(args.season, output_path)
    if missing:
        print("Missing players:", missing)
    else:
        print("Filled all players.")


if __name__ == "__main__":
    main()