"""
src/features/derive_player_team_stints.py
=============================================================================
Derive player-team stints from historical game logs.

Builds one row per player/team stint/season so downstream profile aggregation
can split traded players across the teams they actually played for.

Input:
  data/historical/final_player_game_logs.parquet

Outputs:
  data/processed/player_team_stints.parquet
  reports/player_team_stints_report.json

Usage:
  python3 src/features/derive_player_team_stints.py
=============================================================================
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.player_eval.constants import (  # noqa: E402
    GAME_LOGS_PATH,
    PLAYER_TEAM_STINTS_PATH,
    PLAYER_TEAM_STINTS_REPORT,
)


def _norm_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()


def _parse_minutes(series: pd.Series) -> pd.Series:
    """Parse game-log MIN values that may be numeric or MM:SS strings."""
    mins = pd.to_numeric(series, errors="coerce")
    missing = mins.isna()
    if not missing.any():
        return mins.fillna(0.0)

    text = series.astype(str)
    mmss = text.str.extract(r"^(?P<mm>\d{1,3}):(?P<ss>\d{1,2})$")
    mm = pd.to_numeric(mmss["mm"], errors="coerce")
    ss = pd.to_numeric(mmss["ss"], errors="coerce")
    converted = mm + (ss / 60.0)
    mins = mins.where(~missing, converted)
    return mins.fillna(0.0)


def _extract_team_from_matchup(series: pd.Series) -> pd.Series:
    """Extract team abbreviation from game-log MATCHUP (e.g., 'BKN vs. PHI')."""
    text = series.fillna("").astype(str).str.upper().str.strip()
    token = text.str.extract(r"^([A-Z]{3})\s+(?:VS\.|VS|@)")[0]
    token = token.fillna(text.str.extract(r"^([A-Z]{3})\b")[0])
    return token.fillna("").astype(str).str.strip().str.upper()


def build_player_team_stints(logs: pd.DataFrame) -> pd.DataFrame:
    required = ["PLAYER_ID", "SEASON", "GAME_DATE", "MATCHUP"]
    missing = [c for c in required if c not in logs.columns]
    if missing:
        raise ValueError(f"Missing required game-log columns: {missing}")

    use_cols = [c for c in ["PLAYER_ID", "PLAYER_NAME", "SEASON", "GAME_ID", "GAME_DATE", "MATCHUP", "MIN"] if c in logs.columns]
    df = logs[use_cols].copy()

    df["player_id"] = _norm_id(df["PLAYER_ID"])
    if "PLAYER_NAME" in df.columns:
        df["player_name"] = df["PLAYER_NAME"].astype(str)
    else:
        df["player_name"] = ""
    df["season"] = df["SEASON"].astype(str)
    df["game_date"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    df["team_abbreviation"] = _extract_team_from_matchup(df["MATCHUP"])
    min_series = df["MIN"] if "MIN" in df.columns else pd.Series(0.0, index=df.index)
    df["minutes"] = _parse_minutes(min_series)

    if "GAME_ID" in df.columns:
        df["game_id"] = df["GAME_ID"].astype(str)
    else:
        df["game_id"] = ""

    df = df[
        df["player_id"].notna()
        & (df["player_id"] != "")
        & df["season"].notna()
        & (df["season"] != "")
        & df["game_date"].notna()
        & df["team_abbreviation"].str.fullmatch(r"[A-Z]{3}", na=False)
    ].copy()

    if df.empty:
        return pd.DataFrame(
            columns=[
                "player_id",
                "player_name",
                "season",
                "team_abbreviation",
                "stint_number",
                "games_played",
                "total_minutes",
                "mpg",
                "first_game_date",
                "last_game_date",
                "is_primary_stint",
                "is_final_stint",
                "stint_count",
                "stint_team_count",
            ]
        )

    df = df.sort_values(["player_id", "season", "game_date", "game_id"]).reset_index(drop=True)
    prev_team = df.groupby(["player_id", "season"])["team_abbreviation"].shift(1)
    team_changed = (df["team_abbreviation"] != prev_team).astype(int)
    df["stint_number"] = team_changed.groupby([df["player_id"], df["season"]]).cumsum()

    grouped = (
        df.groupby(["player_id", "player_name", "season", "team_abbreviation", "stint_number"], as_index=False)
        .agg(
            games_played=("game_date", "count"),
            total_minutes=("minutes", "sum"),
            first_game_date=("game_date", "min"),
            last_game_date=("game_date", "max"),
        )
    )

    grouped["mpg"] = grouped["total_minutes"] / grouped["games_played"].replace(0, np.nan)
    grouped["mpg"] = grouped["mpg"].fillna(0.0)

    grouped["stint_count"] = grouped.groupby(["player_id", "season"])["stint_number"].transform("max").astype(int)
    grouped["stint_team_count"] = (
        grouped.groupby(["player_id", "season"])["team_abbreviation"].transform("nunique").astype(int)
    )

    ranked = grouped.sort_values(
        ["player_id", "season", "total_minutes", "stint_number"],
        ascending=[True, True, False, True],
    ).copy()
    ranked["_primary_rank"] = ranked.groupby(["player_id", "season"]).cumcount()
    grouped = grouped.merge(
        ranked[["player_id", "season", "stint_number", "_primary_rank"]],
        on=["player_id", "season", "stint_number"],
        how="left",
    )
    grouped["is_primary_stint"] = grouped["_primary_rank"] == 0

    final_ranked = grouped.sort_values(
        ["player_id", "season", "last_game_date", "stint_number"],
        ascending=[True, True, False, False],
    ).copy()
    final_ranked["_final_rank"] = final_ranked.groupby(["player_id", "season"]).cumcount()
    grouped = grouped.merge(
        final_ranked[["player_id", "season", "stint_number", "_final_rank"]],
        on=["player_id", "season", "stint_number"],
        how="left",
    )
    grouped["is_final_stint"] = grouped["_final_rank"] == 0

    grouped = grouped.drop(columns=["_primary_rank", "_final_rank"], errors="ignore")
    grouped["total_minutes"] = grouped["total_minutes"].round(2)
    grouped["mpg"] = grouped["mpg"].round(3)
    grouped["first_game_date"] = grouped["first_game_date"].dt.strftime("%Y-%m-%d")
    grouped["last_game_date"] = grouped["last_game_date"].dt.strftime("%Y-%m-%d")

    grouped = grouped.sort_values(["season", "player_name", "stint_number"]).reset_index(drop=True)
    return grouped


def main() -> None:
    print("Deriving player-team stints from game logs...")
    if not GAME_LOGS_PATH.exists():
        raise FileNotFoundError(f"Missing game logs: {GAME_LOGS_PATH}")

    logs = pd.read_parquet(GAME_LOGS_PATH)
    stints = build_player_team_stints(logs)

    PLAYER_TEAM_STINTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    stints.to_parquet(PLAYER_TEAM_STINTS_PATH, index=False)

    multi_team = stints[stints["stint_team_count"] > 1]
    report = {
        "rows": int(len(stints)),
        "player_seasons": int(stints[["player_id", "season"]].drop_duplicates().shape[0]),
        "multi_team_player_seasons": int(multi_team[["player_id", "season"]].drop_duplicates().shape[0]),
        "max_stints_single_player_season": int(stints["stint_count"].max()) if len(stints) else 0,
        "seasons": sorted(stints["season"].astype(str).unique().tolist()) if len(stints) else [],
        "examples_multi_team": (
            multi_team.sort_values(["season", "player_name", "stint_number"])
            .head(25)[
                [
                    "player_id",
                    "player_name",
                    "season",
                    "team_abbreviation",
                    "stint_number",
                    "games_played",
                    "total_minutes",
                    "first_game_date",
                    "last_game_date",
                ]
            ]
            .to_dict(orient="records")
        ),
    }
    PLAYER_TEAM_STINTS_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved stints parquet: {PLAYER_TEAM_STINTS_PATH}")
    print(f"Saved stints report: {PLAYER_TEAM_STINTS_REPORT}")
    print(
        "Summary: "
        f"rows={report['rows']}, "
        f"player_seasons={report['player_seasons']}, "
        f"multi_team_player_seasons={report['multi_team_player_seasons']}"
    )


if __name__ == "__main__":
    main()
