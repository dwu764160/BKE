"""
src/data_normalize/normalize_darko.py

Normalize staged DARKO CSV exports to a canonical schema.
"""

import glob
import os
import re
from typing import Dict, Optional

import pandas as pd


RAW_GLOB = "data/historical/darko/raw/*.csv"
OUT_DIR = "data/historical/darko/normalized"
PLAYERS_PATH = "data/historical/players.parquet"
os.makedirs(OUT_DIR, exist_ok=True)


def clean_id(val) -> str:
    if pd.isna(val):
        return "0"
    return str(val).replace(".0", "")


def normalize_name(value: str) -> str:
    value = str(value or "").strip().lower()
    value = re.sub(r"[^a-z0-9 ]+", "", value)
    value = re.sub(r"\s+", " ", value)
    return value


def find_col(columns, options) -> Optional[str]:
    normalized = {str(c).strip().lower(): c for c in columns}
    for key in options:
        if key in normalized:
            return normalized[key]
    return None


def season_to_std(val) -> str:
    if pd.isna(val):
        return "unknown"
    value = str(val).strip()
    if re.match(r"^\d{4}-\d{2}$", value):
        return value
    if re.match(r"^\d{4}$", value):
        year = int(value)
        return f"{year}-{str((year + 1) % 100).zfill(2)}"
    return value


def load_player_name_map() -> Dict[str, str]:
    if not os.path.exists(PLAYERS_PATH):
        return {}
    players = pd.read_parquet(PLAYERS_PATH)
    if "full_name" not in players.columns or "id" not in players.columns:
        return {}
    players = players.copy()
    players["_norm_name"] = players["full_name"].map(normalize_name)
    players["_id"] = players["id"].map(clean_id)
    players = players.dropna(subset=["_norm_name"]).drop_duplicates("_norm_name")
    return dict(zip(players["_norm_name"], players["_id"]))


def normalize_one(path: str, name_to_id: Dict[str, str]) -> pd.DataFrame:
    raw = pd.read_csv(path)

    id_col = find_col(raw.columns, ["player_id", "playerid", "nba_player_id", "id"])
    name_col = find_col(raw.columns, ["player_name", "name", "player"])
    season_col = find_col(raw.columns, ["season", "season_id", "year"])

    dpm_col = find_col(raw.columns, ["dpm", "darko_dpm", "daily plus minus", "daily_plus_minus"]) 
    odpm_col = find_col(raw.columns, ["odpm", "darko_odpm", "off_dpm", "offensive_dpm"]) 
    ddpm_col = find_col(raw.columns, ["ddpm", "darko_ddpm", "def_dpm", "defensive_dpm"]) 

    out = pd.DataFrame()
    out["source_file"] = os.path.basename(path)
    out["player_id"] = raw[id_col].map(clean_id) if id_col else "0"
    out["player_name"] = raw[name_col].astype(str) if name_col else "Unknown"
    out["season"] = raw[season_col].map(season_to_std) if season_col else "unknown"
    out["darko_dpm"] = pd.to_numeric(raw[dpm_col], errors="coerce") if dpm_col else pd.NA
    out["darko_odpm"] = pd.to_numeric(raw[odpm_col], errors="coerce") if odpm_col else pd.NA
    out["darko_ddpm"] = pd.to_numeric(raw[ddpm_col], errors="coerce") if ddpm_col else pd.NA

    if not out.empty:
        norm_names = out["player_name"].map(normalize_name)
        out.loc[out["player_id"].eq("0"), "player_id"] = norm_names.map(name_to_id).fillna("0")

    out["has_id_match"] = out["player_id"].ne("0")
    return out


def main() -> None:
    files = sorted(glob.glob(RAW_GLOB))
    if not files:
        print("⚠️ No DARKO raw files found. Run fetch_darko_manual.py first.")
        return

    name_to_id = load_player_name_map()
    parts = []
    for path in files:
        try:
            parts.append(normalize_one(path, name_to_id))
        except Exception as exc:
            print(f"⚠️ Skipping {path}: {exc}")

    if not parts:
        print("❌ No DARKO files normalized.")
        return

    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates(subset=["player_id", "player_name", "season", "source_file"])

    parquet_path = os.path.join(OUT_DIR, "darko_normalized.parquet")
    csv_path = os.path.join(OUT_DIR, "darko_normalized.csv")
    out.to_parquet(parquet_path, index=False)
    out.to_csv(csv_path, index=False)

    quality = {
        "rows": int(len(out)),
        "files": int(len(files)),
        "id_match_rate": float(out["has_id_match"].mean()) if len(out) else 0.0,
        "seasons": sorted([str(x) for x in out["season"].dropna().unique().tolist()]),
    }

    quality_path = os.path.join(OUT_DIR, "darko_normalization_quality.json")
    pd.Series(quality).to_json(quality_path, indent=2)

    print(f"✅ DARKO normalized rows: {len(out):,}")
    print(f"✅ Saved: {parquet_path}")
    print(f"✅ Quality: {quality_path}")


if __name__ == "__main__":
    main()
