#!/usr/bin/env python3
"""
scripts/compare_bref_advanced_metrics.py
Compare computed Win Shares/BPM/VORP outputs against Basketball-Reference ground truth.

Usage:
  python3 scripts/compare_bref_advanced_metrics.py --season 2024-25
  python3 scripts/compare_bref_advanced_metrics.py --season 2024-25 --bref tests/bref_ground_truth_2024-25.csv
"""

from __future__ import annotations

import argparse
import os
import unicodedata

import numpy as np
import pandas as pd


PROCESSED_DIR = "data/processed"
DEFAULT_BREF = "tests/bref_ground_truth_{season}.csv"
OUTPUT_TEMPLATE = "data/processed/bref_metric_comparison_{season}.csv"


def normalize_name(name: str) -> str:
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return ""
    name = str(name)
    name = unicodedata.normalize("NFKD", name).encode("ASCII", "ignore").decode("utf-8")
    return " ".join(name.lower().split())


def load_metrics(season: str) -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, "metrics_linear.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"metrics file not found: {path}")
    df = pd.read_parquet(path)
    if "season" not in df.columns:
        raise ValueError("metrics_linear.parquet missing 'season' column")
    return df[df["season"] == season].copy()


def load_bref(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"bref ground truth not found: {path}")
    df = pd.read_csv(path)
    df = df.dropna(subset=["bref_GP"])
    return df


def build_name_index(names: pd.Series) -> dict[str, int]:
    index = {}
    for idx, name in names.items():
        key = normalize_name(name)
        if key and key not in index:
            index[key] = idx
    return index


def compare_metrics(metrics: pd.DataFrame, bref: pd.DataFrame) -> pd.DataFrame:
    if "player_name" not in metrics.columns:
        raise ValueError("metrics_linear.parquet missing 'player_name' column")

    m_index = build_name_index(metrics["player_name"])

    rows = []
    for _, bref_row in bref.iterrows():
        bref_name = bref_row.get("player_name")
        key = normalize_name(bref_name)
        if not key:
            continue
        m_idx = m_index.get(key)
        if m_idx is None:
            rows.append({"player_name": bref_name, "status": "NOT_FOUND"})
            continue

        m = metrics.loc[m_idx]
        out = {
            "player_name": bref_name,
            "status": "OK",
        }

        def add_diff(col_ours: str, col_bref: str, label: str):
            ours = m.get(col_ours, np.nan)
            bref_val = bref_row.get(col_bref, np.nan)
            if pd.notna(ours) and pd.notna(bref_val) and str(bref_val).strip() != "":
                try:
                    ours_f = float(ours)
                    bref_f = float(bref_val)
                except (TypeError, ValueError):
                    return
                out[f"{label}_ours"] = round(ours_f, 3)
                out[f"{label}_bref"] = round(bref_f, 3)
                out[f"{label}_diff"] = round(ours_f - bref_f, 3)

        add_diff("WS", "bref_WS", "WS")
        add_diff("OWS", "bref_OWS", "OWS")
        add_diff("DWS", "bref_DWS", "DWS")
        add_diff("BPM", "bref_BPM", "BPM")
        add_diff("VORP", "bref_VORP", "VORP")

        rows.append(out)

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> None:
    if df.empty:
        print("No rows to summarize.")
        return

    metrics = ["WS", "OWS", "DWS", "BPM", "VORP"]
    for metric in metrics:
        diff_col = f"{metric}_diff"
        if diff_col not in df.columns:
            continue
        diffs = df[diff_col].dropna()
        if diffs.empty:
            continue
        print(
            f"{metric:<4} mean {diffs.mean():+.3f} | med {diffs.median():+.3f} | "
            f"mae {diffs.abs().mean():.3f} | max {diffs.abs().max():.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare computed Win Shares/BPM/VORP vs Basketball-Reference ground truth"
    )
    parser.add_argument("--season", default="2024-25", help="Season to compare (default: 2024-25)")
    parser.add_argument("--bref", default=None, help="Path to bref CSV (defaults to tests/bref_ground_truth_SEASON.csv)")
    parser.add_argument("--output", default=None, help="Optional output CSV path")
    args = parser.parse_args()

    season = args.season
    bref_path = args.bref or DEFAULT_BREF.format(season=season)
    output_path = args.output or OUTPUT_TEMPLATE.format(season=season)

    metrics = load_metrics(season)
    bref = load_bref(bref_path)

    comparison = compare_metrics(metrics, bref)

    print(f"Compared {len(comparison)} rows for season {season}")
    missing = comparison[comparison["status"] == "NOT_FOUND"]
    if not missing.empty:
        print(f"Missing players in metrics_linear: {len(missing)}")

    summarize(comparison)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    comparison.to_csv(output_path, index=False)
    print(f"Saved comparison to {output_path}")


if __name__ == "__main__":
    main()