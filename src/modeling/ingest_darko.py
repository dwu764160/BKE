"""
src/modeling/ingest_darko.py
=============================================================================
Build modeling input tables that merge DARKO with RAPM and linear metrics.
=============================================================================
"""

import os
from typing import List

import pandas as pd


DARKO_PATH = "data/historical/darko/normalized/darko_normalized.parquet"
RAPM_PATH = "data/processed/player_rapm.parquet"
LINEAR_PATH = "data/advanced_local_metrics.parquet"
OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)


def clean_id(val) -> str:
    if pd.isna(val):
        return "0"
    return str(val).replace(".0", "")


def load_darko() -> pd.DataFrame:
    if not os.path.exists(DARKO_PATH):
        return pd.DataFrame(columns=["player_id", "player_name", "season", "darko_dpm", "darko_odpm", "darko_ddpm"])
    darko = pd.read_parquet(DARKO_PATH)
    darko = darko.copy()
    darko["player_id"] = darko["player_id"].map(clean_id)
    darko["season"] = darko["season"].astype(str)
    keep = ["player_id", "player_name", "season", "darko_dpm", "darko_odpm", "darko_ddpm"]
    for col in keep:
        if col not in darko.columns:
            darko[col] = pd.NA
    darko = darko[keep]
    darko = darko.drop_duplicates(subset=["player_id", "season"], keep="last")
    return darko


def load_rapm() -> pd.DataFrame:
    if not os.path.exists(RAPM_PATH):
        return pd.DataFrame(columns=["player_id", "season", "rapm", "orapm", "drapm", "rapm_type"])

    rapm = pd.read_parquet(RAPM_PATH).copy()
    rapm["player_id"] = rapm["player_id"].map(clean_id)
    rapm["season"] = rapm["season"].astype(str)

    preferred_order = {
        "pooled": 0,
        "pooled_split": 1,
        "single_season": 2,
        "single_season_split": 3,
    }
    rapm["_rank"] = rapm["RAPM_type"].map(preferred_order).fillna(99)
    rapm = rapm.sort_values(["season", "player_id", "_rank"])

    core = (
        rapm.groupby(["season", "player_id"], as_index=False)
        .agg(
            player_name=("player_name", "last"),
            rapm=("RAPM", "last"),
            possessions_played=("possessions_played", "max"),
            rapm_type=("RAPM_type", "last"),
            alpha=("alpha", "last"),
        )
    )

    split = rapm[rapm["RAPM_type"].isin(["pooled_split", "single_season_split"])].copy()
    split = split.sort_values(["season", "player_id", "_rank"]).drop_duplicates(["season", "player_id"], keep="first")
    split = split[["season", "player_id", "ORAPM", "DRAPM"]].rename(columns={"ORAPM": "orapm", "DRAPM": "drapm"})

    out = core.merge(split, on=["season", "player_id"], how="left")
    return out


def load_linear() -> pd.DataFrame:
    if not os.path.exists(LINEAR_PATH):
        return pd.DataFrame(columns=["player_id", "season"])
    linear = pd.read_parquet(LINEAR_PATH).copy()
    if "PLAYER_ID" in linear.columns:
        linear["player_id"] = linear["PLAYER_ID"].map(clean_id)
    elif "player_id" in linear.columns:
        linear["player_id"] = linear["player_id"].map(clean_id)
    else:
        linear["player_id"] = "0"

    if "SEASON" in linear.columns:
        linear["season"] = linear["SEASON"].astype(str)
    elif "season" in linear.columns:
        linear["season"] = linear["season"].astype(str)
    else:
        linear["season"] = "unknown"

    keep_cols = [
        "player_id",
        "season",
        "TS_pct",
        "eFG_pct",
        "USG_pct",
        "AST_pct",
        "TOV_pct",
        "OREB_pct",
        "DREB_pct",
        "PTS_per36",
        "AST_per36",
        "REB_per36",
    ]
    for col in keep_cols:
        if col not in linear.columns:
            linear[col] = pd.NA
    return linear[keep_cols].drop_duplicates(subset=["season", "player_id"], keep="last")


def save_outputs(df: pd.DataFrame, seasons: List[str]) -> None:
    combined_parquet = os.path.join(OUT_DIR, "modeling_inputs_all.parquet")
    combined_csv = os.path.join(OUT_DIR, "modeling_inputs_all.csv")
    df.to_parquet(combined_parquet, index=False)
    df.to_csv(combined_csv, index=False)

    for season in seasons:
        part = df[df["season"] == season].copy()
        part_path = os.path.join(OUT_DIR, f"modeling_inputs_{season}.parquet")
        part.to_parquet(part_path, index=False)


def main() -> None:
    darko = load_darko()
    rapm = load_rapm()
    linear = load_linear()

    merged = rapm.merge(darko, on=["season", "player_id"], how="outer", suffixes=("", "_darko"))
    merged = merged.merge(linear, on=["season", "player_id"], how="left")

    if "player_name" not in merged.columns:
        merged["player_name"] = pd.NA
    if "player_name_darko" in merged.columns:
        merged["player_name"] = merged["player_name"].fillna(merged["player_name_darko"])
        merged = merged.drop(columns=["player_name_darko"])

    merged["has_rapm"] = merged["rapm"].notna()
    merged["has_darko"] = merged["darko_dpm"].notna()
    merged["modeling_row_ready"] = merged["has_rapm"] | merged["has_darko"]

    seasons = sorted([s for s in merged["season"].dropna().unique().tolist() if str(s) != "unknown"])
    merged = merged.sort_values(["season", "player_id"])

    save_outputs(merged, seasons)

    report = {
        "rows": int(len(merged)),
        "seasons": seasons,
        "has_rapm_rate": float(merged["has_rapm"].mean()) if len(merged) else 0.0,
        "has_darko_rate": float(merged["has_darko"].mean()) if len(merged) else 0.0,
    }
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, "modeling_inputs_report.json")
    pd.Series(report).to_json(report_path, indent=2)

    print(f"✅ Modeling inputs rows: {len(merged):,}")
    print(f"✅ Seasons: {seasons}")
    print(f"✅ Saved: {os.path.join(OUT_DIR, 'modeling_inputs_all.parquet')}")


if __name__ == "__main__":
    main()
