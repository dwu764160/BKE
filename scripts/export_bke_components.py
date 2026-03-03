"""
scripts/export_bke_components.py
=============================================================================
Export per-player BKE components to JSON for the interactive BKE viewer.

Reads the v2.8 decomposition parquet, rebuilds the v3.1 Layer 3+6 baseline,
computes the production proxy z-scores, and outputs a compact JSON file
containing all fields needed to recompute BKE client-side for any lambda.

Output:
  data/processed/bke/bke_v31_components.json

Usage:
  python3 scripts/export_bke_components.py
=============================================================================
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.modeling.bke_v31_experimental_layers import _build_base
from src.modeling.experiment2_production_tilt import (
    _build_layer36_dbke,
    _build_production_proxy,
    _load_layer36_config,
    _zscore,
    _safe_float,
    CFG as EXP2_CFG,
)
from src.modeling.model_config import BKE_V28_OUTPUT_PARQUET, BKE_DIR, V31_EXPERIMENTAL, BKE_V31_COMPONENTS_JSON

OUTPUT_JSON = BKE_V31_COMPONENTS_JSON


def export_components(
    parquet_path: str = BKE_V28_OUTPUT_PARQUET,
    output_json: str = OUTPUT_JSON,
) -> dict:
    """Build and export per-player BKE components for the interactive viewer."""
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Missing decomposition parquet: {parquet_path}")

    raw = pd.read_parquet(parquet_path)
    df = raw[raw["qualified"] == True].copy()
    if df.empty:
        raise ValueError("No qualified rows in decomposition parquet.")

    # ---- Rebuild v3.1 Layer 3+6 baseline ----
    obke_base, dbke_base, _, def_portable = _build_base(df)
    exponent, k = _load_layer36_config()
    dbke_layer36 = _build_layer36_dbke(df, dbke_base, def_portable, exponent=exponent, k=k)

    bke_base_60_40 = V31_EXPERIMENTAL.off_weight_default * obke_base + V31_EXPERIMENTAL.def_weight_default * dbke_layer36
    bke_base_55_45 = V31_EXPERIMENTAL.off_weight_alt * obke_base + V31_EXPERIMENTAL.def_weight_alt * dbke_layer36

    # ---- Build production proxy ----
    prod_proxy, proxy_meta = _build_production_proxy(df)

    # ---- Collect per-player records ----
    work = df[["season", "player_id", "player_name"]].copy()
    work["season"] = work["season"].astype(str)
    work["player_id"] = work["player_id"].astype(str)

    # Optional columns for display
    for col in ["rapm", "orapm", "drapm", "TS_PCT", "PTS", "AST", "MIN",
                 "possessions_played", "USG_RATE"]:
        if col in df.columns:
            work[col] = pd.to_numeric(df[col], errors="coerce")

    # Position bucket (if available)
    if "position_bucket" in df.columns:
        work["position"] = df["position_bucket"].astype(str)
    elif "primary_position" in df.columns:
        work["position"] = df["primary_position"].astype(str)
    else:
        work["position"] = "Unknown"

    # Team (if available)
    if "team_abbreviation" in df.columns:
        work["team"] = df["team_abbreviation"].astype(str)
    elif "TEAM_ABBREVIATION" in df.columns:
        work["team"] = df["TEAM_ABBREVIATION"].astype(str)
    else:
        work["team"] = ""

    # Archetype (if available)
    for col in ["offensive_archetype", "defensive_archetype",
                 "primary_archetype", "defensive_archetype_v2"]:
        if col in df.columns:
            work[col] = df[col].astype(str)

    work["obke"] = pd.to_numeric(obke_base, errors="coerce").fillna(0.0)
    work["dbke_l36"] = pd.to_numeric(dbke_layer36, errors="coerce").fillna(0.0)
    work["bke_base_60_40"] = pd.to_numeric(bke_base_60_40, errors="coerce").fillna(0.0)
    work["bke_base_55_45"] = pd.to_numeric(bke_base_55_45, errors="coerce").fillna(0.0)
    work["prod_proxy_z"] = pd.to_numeric(prod_proxy, errors="coerce").fillna(0.0)

    # Global normalization stats for client-side use
    global_stats = {}
    for split_name, col in [("60_40", "bke_base_60_40"), ("55_45", "bke_base_55_45")]:
        by_season = {}
        for s, grp in work.groupby("season"):
            by_season[str(s)] = {
                "mean": _safe_float(grp[col].mean()),
                "std": _safe_float(grp[col].std(ddof=0)),
                "count": int(len(grp)),
            }
        global_stats[split_name] = by_season

    # Build player list
    players = []
    for _, row in work.iterrows():
        rec = {
            "player_id": str(row["player_id"]),
            "player_name": str(row.get("player_name", "")),
            "season": str(row["season"]),
            "position": str(row.get("position", "")),
            "team": str(row.get("team", "")),
            "obke": _safe_float(row["obke"]),
            "dbke_l36": _safe_float(row["dbke_l36"]),
            "bke_base_60_40": _safe_float(row["bke_base_60_40"]),
            "bke_base_55_45": _safe_float(row["bke_base_55_45"]),
            "prod_proxy_z": _safe_float(row["prod_proxy_z"]),
        }
        # Optional display columns
        for col in ["rapm", "orapm", "drapm", "TS_PCT", "PTS", "AST",
                     "MIN", "possessions_played", "USG_RATE"]:
            if col in row.index:
                rec[col] = _safe_float(row.get(col))

        # Archetypes
        for col in ["offensive_archetype", "defensive_archetype",
                     "primary_archetype", "defensive_archetype_v2"]:
            if col in row.index and pd.notna(row.get(col)):
                rec[col] = str(row[col])

        players.append(rec)

    payload = {
        "version": "v3.1-components",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_file": parquet_path,
        "output_file": output_json,
        "qualified_players": int(len(df)),
        "formula": {
            "description": "bke_final = bke_base + lambda * prod_proxy_z",
            "bke_base_60_40": "0.60 * obke + 0.40 * dbke_l36",
            "bke_base_55_45": "0.55 * obke + 0.45 * dbke_l36",
            "obke": "0.55 * offensive_portable_z + 0.25 * role_utilization_z + 0.20 * elevation_orapm_z",
            "dbke_l36": "Layer6(def_portable, def_elev, scheme; k=layer6_k) -> Layer3(exponent=layer3_exponent)",
        },
        "config": {
            "layer3_exponent": _safe_float(exponent, 4),
            "layer6_k": _safe_float(k, 4),
            "off_weight_default": 0.60,
            "def_weight_default": 0.40,
        },
        "production_proxy": proxy_meta,
        "global_stats": global_stats,
        "players": players,
    }

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[Export] BKE components written: {output_json}")
    print(f"  Players: {len(players)}")
    print(f"  Seasons: {sorted(work['season'].unique().tolist())}")
    return payload


if __name__ == "__main__":
    export_components()
