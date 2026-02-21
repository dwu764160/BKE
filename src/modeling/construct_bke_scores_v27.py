"""
src/modeling/construct_bke_scores_v27.py
=============================================================================
BKE v2.7 — Final Comprehensive Output Constructor

Builds terminal OBKE / DBKE / BKE outputs from decomposition-layer raw signals,
then writes:

    data/processed/BKE_Scores_v27.json

Rules enforced:
  - No re-percentiling of earlier layers or dimensions.
  - OBKE/DBKE/BKE are built from raw/z-score layer composites only.
  - Final ranking is based ONLY on final league-wide BKE percentile.

Pipeline:
  1) Build dimension score bundle (raw + z)
  2) Build layer composites (raw)
  3) Build OBKE_raw / DBKE_raw
  4) BKE_raw = OBKE_raw + DBKE_raw
  5) Apply monotonic transform to OBKE/DBKE/BKE
  6) Compute final league-wide percentiles and BKE rank
=============================================================================
"""

import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.modeling.model_config import (
    BKE_OUTPUT_PARQUET,
    BKE_SCORES_V27_JSON,
)


REQUIRED_COLUMNS = [
    "player_id",
    "player_name",
    "season",
    "qualified",
    "offensive_portable_z",
    "defensive_portable_z",
    "elevation_orapm",
    "elevation_drapm",
]


DIMENSION_KEYS = [
    "shooting_gravity",
    "driving_gravity",
    "playmaking_creation",
    "extra_possession",
    "defensive_playmaking",
    "defensive_impact",
    "turnover_control",
    "defensive_versatility",
    "self_creation",
]


def _safe_float(value):
    if value is None:
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(val):
        return None
    return round(val, 6)


def _zscore(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    std = vals.std(ddof=0)
    if pd.isna(std) or std < 1e-9:
        return pd.Series(0.0, index=series.index)
    return (vals - vals.mean()) / std


def _signed_log1p(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce").fillna(0.0)
    return np.sign(vals) * np.log1p(np.abs(vals))


def _percentile_0_100(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    return vals.rank(pct=True, method="average") * 100.0


def _group_percentile_0_100(df: pd.DataFrame, value_col: str, group_col: str) -> pd.Series:
    if value_col not in df.columns or group_col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    values = pd.to_numeric(df[value_col], errors="coerce")
    groups = df[group_col].astype(str).replace({"": np.nan, "nan": np.nan, "None": np.nan})
    ranked = values.groupby(groups).rank(method="average", pct=True) * 100.0
    return ranked


def _resolve_rue_z(df: pd.DataFrame) -> pd.Series:
    if "role_utilization_raw_z" in df.columns:
        return pd.to_numeric(df["role_utilization_raw_z"], errors="coerce").fillna(0.0)
    if "role_utilization_raw" in df.columns:
        return _zscore(df["role_utilization_raw"]).fillna(0.0)
    return pd.Series(0.0, index=df.index)


def _build_dimension_bundle(row: pd.Series) -> Dict[str, Dict[str, float]]:
    payload: Dict[str, Dict[str, float]] = {}
    for key in DIMENSION_KEYS:
        z_col = f"dim_{key}_z"
        raw_col = f"dim_{key}_z_raw"
        payload[key] = {
            "raw": _safe_float(row.get(raw_col)),
            "z": _safe_float(row.get(z_col)),
        }
    return payload


def _valid_name(val) -> bool:
    if val is None:
        return False
    text = str(val).strip()
    if not text:
        return False
    return text.lower() not in {"unknown", "nan", "none"}


def _load_name_map_from_parquet(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["player_id", "season", "player_name"])
    df = pd.read_parquet(path)
    if df.empty:
        return pd.DataFrame(columns=["player_id", "season", "player_name"])

    id_col = None
    for cand in ["player_id", "PLAYER_ID", "PERSON_ID"]:
        if cand in df.columns:
            id_col = cand
            break
    if id_col is None:
        return pd.DataFrame(columns=["player_id", "season", "player_name"])

    name_col = None
    for cand in ["player_name", "PLAYER_NAME", "full_name"]:
        if cand in df.columns:
            name_col = cand
            break
    if name_col is None:
        return pd.DataFrame(columns=["player_id", "season", "player_name"])

    season_col = None
    for cand in ["season", "SEASON"]:
        if cand in df.columns:
            season_col = cand
            break

    out = pd.DataFrame()
    out["player_id"] = df[id_col].astype(str).str.replace(r"\\.0$", "", regex=True)
    out["player_name"] = df[name_col].astype(str)
    if season_col is not None:
        out["season"] = df[season_col].astype(str)
    else:
        out["season"] = ""

    out = out[out["player_name"].map(_valid_name)]
    return out[["player_id", "season", "player_name"]].drop_duplicates()


def _build_name_lookup() -> Dict[str, str]:
    candidates = [
        "data/processed/player_archetypes.parquet",
        "data/processed/archetype_embeddings.parquet",
        "data/processed/defensive_archetypes_v2.parquet",
        "data/historical/complete_player_season_stats.parquet",
    ]

    by_id_season: Dict[str, str] = {}
    by_id_latest: Dict[str, str] = {}

    for path in candidates:
        mapped = _load_name_map_from_parquet(path)
        if mapped.empty:
            continue
        for _, row in mapped.iterrows():
            pid = str(row["player_id"])
            season = str(row["season"])
            name = str(row["player_name"]).strip()
            if not _valid_name(name):
                continue
            if season:
                by_id_season[f"{pid}::{season}"] = name
            by_id_latest[pid] = name

    # Merge strategy: season-specific first, then id-only fallback.
    merged = dict(by_id_latest)
    merged.update(by_id_season)
    return merged


def construct_bke_scores_v27(
    input_parquet: str = BKE_OUTPUT_PARQUET,
    output_json: str = BKE_SCORES_V27_JSON,
) -> Dict:
    if not os.path.exists(input_parquet):
        raise FileNotFoundError(
            f"Input decomposition file not found: {input_parquet}. "
            "Run src/modeling/decomposition_engine.py first."
        )

    df = pd.read_parquet(input_parquet)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    eligible = df[df["qualified"] == True].copy()
    if eligible.empty:
        payload = {
            "version": "2.7",
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "input_file": input_parquet,
            "output_file": output_json,
            "eligible_players": 0,
            "players": {},
        }
        with open(output_json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return payload

    name_lookup = _build_name_lookup()

    # Resolve Unknown names using season-aware lookup, then id-only fallback.
    for idx, row in eligible.iterrows():
        current = row.get("player_name")
        if _valid_name(current):
            continue
        pid = str(row.get("player_id", ""))
        season = str(row.get("season", ""))
        resolved = name_lookup.get(f"{pid}::{season}") or name_lookup.get(pid)
        if _valid_name(resolved):
            eligible.at[idx, "player_name"] = resolved

    offensive_portable = pd.to_numeric(eligible["offensive_portable_z"], errors="coerce").fillna(0.0)
    defensive_portable = pd.to_numeric(eligible["defensive_portable_z"], errors="coerce").fillna(0.0)
    rue_z = _resolve_rue_z(eligible)

    off_elevation_z = _zscore(eligible["elevation_orapm"]).fillna(0.0)
    def_elevation_z = _zscore(eligible["elevation_drapm"]).fillna(0.0)

    if "scheme_stability_z" in eligible.columns:
        scheme_bonus_z = pd.to_numeric(eligible["scheme_stability_z"], errors="coerce").fillna(0.0).clip(lower=0.0)
    else:
        scheme_bonus_z = pd.Series(0.0, index=eligible.index)

    # Layer composites (raw) before final transforms.
    eligible["layer1_offensive_raw"] = offensive_portable
    eligible["layer1_defensive_raw"] = defensive_portable
    eligible["layer2_rue_raw"] = rue_z
    eligible["layer3_off_elevation_raw"] = off_elevation_z
    eligible["layer3_def_elevation_raw"] = def_elevation_z
    eligible["layer4_scheme_bonus_raw"] = scheme_bonus_z

    # Weighted offensive and defensive composites.
    # Weights keep Layer 1 as anchor while preserving role/elevation/scheme contributions.
    obke_num = (
        0.55 * eligible["layer1_offensive_raw"] +
        0.25 * eligible["layer2_rue_raw"] +
        0.20 * eligible["layer3_off_elevation_raw"]
    )
    dbke_num = (
        0.60 * eligible["layer1_defensive_raw"] +
        0.25 * eligible["layer3_def_elevation_raw"] +
        0.15 * eligible["layer4_scheme_bonus_raw"]
    )

    eligible["raw_OBKE"] = obke_num
    eligible["raw_DBKE"] = dbke_num
    eligible["raw_BKE"] = eligible["raw_OBKE"] + eligible["raw_DBKE"]

    # Monotonic transforms.
    eligible["transformed_OBKE"] = _signed_log1p(eligible["raw_OBKE"])
    eligible["transformed_DBKE"] = _signed_log1p(eligible["raw_DBKE"])
    eligible["transformed_BKE"] = _signed_log1p(eligible["raw_BKE"])

    # Terminal league-wide percentiles (eligible players only).
    eligible["final_OBKE_percentile"] = _percentile_0_100(eligible["transformed_OBKE"])
    eligible["final_DBKE_percentile"] = _percentile_0_100(eligible["transformed_DBKE"])
    eligible["final_BKE_percentile"] = _percentile_0_100(eligible["transformed_BKE"])

    # Grouped percentiles (transformed metrics) by position band / archetypes.
    if "position_bucket" not in eligible.columns and "primary_position_estimate" in eligible.columns:
        eligible["position_bucket"] = eligible["primary_position_estimate"]

    for metric in ["OBKE", "DBKE", "BKE"]:
        transformed_col = f"transformed_{metric}"
        eligible[f"position_band_{metric}_percentile"] = _group_percentile_0_100(
            eligible,
            transformed_col,
            "position_bucket",
        )
        eligible[f"off_archetype_{metric}_percentile"] = _group_percentile_0_100(
            eligible,
            transformed_col,
            "primary_archetype",
        )
        eligible[f"def_archetype_{metric}_percentile"] = _group_percentile_0_100(
            eligible,
            transformed_col,
            "defensive_archetype",
        )

    eligible["rank"] = (
        eligible["final_BKE_percentile"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    eligible = eligible.sort_values(["rank", "player_name", "season"], ascending=[True, True, True])

    players: Dict[str, Dict] = {}
    for _, row in eligible.iterrows():
        key = (
            f"{row.get('player_name', 'Unknown')} "
            f"({row.get('season', '')}) #{row.get('player_id', '')}"
        )
        players[key] = {
            "raw_OBKE": _safe_float(row.get("raw_OBKE")),
            "raw_DBKE": _safe_float(row.get("raw_DBKE")),
            "transformed_OBKE": _safe_float(row.get("transformed_OBKE")),
            "transformed_DBKE": _safe_float(row.get("transformed_DBKE")),
            "final_OBKE_percentile": _safe_float(row.get("final_OBKE_percentile")),
            "final_DBKE_percentile": _safe_float(row.get("final_DBKE_percentile")),
            "raw_BKE": _safe_float(row.get("raw_BKE")),
            "transformed_BKE": _safe_float(row.get("transformed_BKE")),
            "final_BKE_percentile": _safe_float(row.get("final_BKE_percentile")),
            "position_bucket": str(row.get("position_bucket", "")),
            "primary_archetype": str(row.get("primary_archetype", "")),
            "defensive_archetype": str(row.get("defensive_archetype", "")),
            "position_band_OBKE_percentile": _safe_float(row.get("position_band_OBKE_percentile")),
            "position_band_DBKE_percentile": _safe_float(row.get("position_band_DBKE_percentile")),
            "position_band_BKE_percentile": _safe_float(row.get("position_band_BKE_percentile")),
            "off_archetype_OBKE_percentile": _safe_float(row.get("off_archetype_OBKE_percentile")),
            "off_archetype_DBKE_percentile": _safe_float(row.get("off_archetype_DBKE_percentile")),
            "off_archetype_BKE_percentile": _safe_float(row.get("off_archetype_BKE_percentile")),
            "def_archetype_OBKE_percentile": _safe_float(row.get("def_archetype_OBKE_percentile")),
            "def_archetype_DBKE_percentile": _safe_float(row.get("def_archetype_DBKE_percentile")),
            "def_archetype_BKE_percentile": _safe_float(row.get("def_archetype_BKE_percentile")),
            "rank": int(row.get("rank", 0)),
            "player_id": str(row.get("player_id", "")),
            "season": str(row.get("season", "")),
            "dimension_scores": _build_dimension_bundle(row),
            "layer_scores": {
                "layer1_offensive_raw": _safe_float(row.get("layer1_offensive_raw")),
                "layer1_defensive_raw": _safe_float(row.get("layer1_defensive_raw")),
                "layer2_rue_raw": _safe_float(row.get("layer2_rue_raw")),
                "layer3_off_elevation_raw": _safe_float(row.get("layer3_off_elevation_raw")),
                "layer3_def_elevation_raw": _safe_float(row.get("layer3_def_elevation_raw")),
                "layer4_scheme_bonus_raw": _safe_float(row.get("layer4_scheme_bonus_raw")),
            },
            "obke_dbke_scores": {
                "raw_OBKE": _safe_float(row.get("raw_OBKE")),
                "raw_DBKE": _safe_float(row.get("raw_DBKE")),
                "raw_BKE": _safe_float(row.get("raw_BKE")),
            },
        }

    payload = {
        "version": "2.7",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_file": input_parquet,
        "output_file": output_json,
        "eligible_players": int(len(eligible)),
        "method": {
            "OBKE_raw": "0.55*layer1_offensive_raw + 0.25*layer2_rue_raw + 0.20*layer3_off_elevation_raw",
            "DBKE_raw": "0.60*layer1_defensive_raw + 0.25*layer3_def_elevation_raw + 0.15*layer4_scheme_bonus_raw",
            "BKE_raw": "OBKE_raw + DBKE_raw",
            "transform": "signed_log1p(x) = sign(x)*log(1+abs(x))",
            "terminal_percentile": "league-wide rank percentile over eligible players only",
        },
        "players": players,
    }

    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return payload


if __name__ == "__main__":
    result = construct_bke_scores_v27()
    print(
        f"Saved {result['eligible_players']} player records to "
        f"{result['output_file']}"
    )
