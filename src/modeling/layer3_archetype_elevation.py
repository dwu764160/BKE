"""
src/modeling/layer3_archetype_elevation.py
=============================================================================
BKE v2.0 — LAYER 3: Archetype Elevation

Estimates:
  "How much better is this player than the average player in their archetype?"

Sub-layers:
  3A. Archetype Baseline Impact — Mean RAPM, playtype surplus, and on/off
      impact for each archetype cohort.
  3B. Elevation Score — Player impact minus archetype baseline, z-scored
      within season (v2.0: already uses z-scores internally).

v2.0: Internal z-score aggregation was already correct in v1.5.
      Added elevation z-score output for decomposition pipeline.

This answers:
  - Are they archetype-replacement level?
  - Or archetype-elite?

Inputs:
  - Layer 1+2 output DataFrame
  - RAPM, playtype surplus, efficiency metrics

Output:
  Columns added:
    - archetype_baseline_rapm / _orapm / _drapm
    - archetype_baseline_surplus
    - elevation_rapm / _surplus / _efficiency
    - elevation_score (composite)
    - elevation_z (v2.0: raw z-score for aggregation)
    - elevation_league_pctl / _archetype_pctl
=============================================================================
"""

import os
import sys
from typing import Dict, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.modeling.model_config import (
    ARCHETYPE_ELEVATION,
    clean_id,
)
from src.modeling.percentile_engine import (
    add_league_percentiles,
    add_league_z_scores,
    add_grouped_percentiles,
    vectorized_percentile_rank,
)


# ---------------------------------------------------------------------------
# 3A. Archetype Baseline Impact
# ---------------------------------------------------------------------------

def compute_archetype_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the baseline impact for each archetype cohort.

    For each archetype × season combination, computes:
      - Mean RAPM, ORAPM, DRAPM
      - Mean playtype surplus total
      - Mean efficiency (TS%)
      - Cohort size

    Returns a DataFrame of archetype baselines.
    """
    if "primary_archetype" not in df.columns:
        return pd.DataFrame()

    agg_cols = {}

    # Impact metrics
    for metric in ["rapm", "orapm", "drapm"]:
        if metric in df.columns:
            agg_cols[f"baseline_{metric}_mean"] = (metric, "mean")
            agg_cols[f"baseline_{metric}_median"] = (metric, "median")
            agg_cols[f"baseline_{metric}_std"] = (metric, "std")

    # Playtype surplus
    if "playtype_surplus_total" in df.columns:
        agg_cols["baseline_surplus_mean"] = ("playtype_surplus_total", "mean")
        agg_cols["baseline_surplus_median"] = ("playtype_surplus_total", "median")

    # Efficiency
    ts_col = "TS_PCT_adj" if "TS_PCT_adj" in df.columns else (
        "TS_PCT" if "TS_PCT" in df.columns else None
    )
    if ts_col:
        agg_cols["baseline_ts_mean"] = (ts_col, "mean")

    # Cohort size
    agg_cols["archetype_cohort_size"] = ("player_id", "count")

    if not agg_cols:
        return pd.DataFrame()

    baselines = df.groupby(["season", "primary_archetype"]).agg(**agg_cols).reset_index()
    return baselines


def merge_archetype_baselines(df: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    """
    Merge archetype baselines onto the player-season DataFrame.
    Preserves the original index (pd.merge resets it by default).
    """
    if baselines.empty or "primary_archetype" not in df.columns:
        return df

    original_index = df.index

    # Merge on season + archetype
    result = df.merge(
        baselines,
        on=["season", "primary_archetype"],
        how="left",
        suffixes=("", "_baseline"),
    )

    # Restore original index (merge resets to RangeIndex)
    result.index = original_index
    return result


# ---------------------------------------------------------------------------
# 3B. Elevation Score
# ---------------------------------------------------------------------------

def compute_elevation_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute elevation scores: player impact minus archetype baseline.

    Elevation captures how much a player exceeds (or falls below) the
    typical member of their archetype.

    Components:
      - elevation_rapm = player_rapm - archetype_mean_rapm
      - elevation_surplus = player_surplus - archetype_mean_surplus
      - elevation_efficiency = player_ts - archetype_mean_ts

    Final elevation_score is a weighted composite of these.
    """
    cfg = ARCHETYPE_ELEVATION
    result = df.copy()

    # RAPM elevation
    if "rapm" in result.columns and "baseline_rapm_mean" in result.columns:
        result["elevation_rapm"] = result["rapm"] - result["baseline_rapm_mean"]
    else:
        result["elevation_rapm"] = np.nan

    # ORAPM elevation
    if "orapm" in result.columns and "baseline_orapm_mean" in result.columns:
        result["elevation_orapm"] = result["orapm"] - result["baseline_orapm_mean"]
    else:
        result["elevation_orapm"] = np.nan

    # DRAPM elevation
    if "drapm" in result.columns and "baseline_drapm_mean" in result.columns:
        result["elevation_drapm"] = result["drapm"] - result["baseline_drapm_mean"]
    else:
        result["elevation_drapm"] = np.nan

    # Playtype surplus elevation
    if "playtype_surplus_total" in result.columns and "baseline_surplus_mean" in result.columns:
        result["elevation_surplus"] = (
            result["playtype_surplus_total"] - result["baseline_surplus_mean"]
        )
    else:
        result["elevation_surplus"] = np.nan

    # Efficiency elevation
    ts_col = "TS_PCT_adj" if "TS_PCT_adj" in result.columns else (
        "TS_PCT" if "TS_PCT" in result.columns else None
    )
    if ts_col and "baseline_ts_mean" in result.columns:
        result["elevation_efficiency"] = result[ts_col] - result["baseline_ts_mean"]
    else:
        result["elevation_efficiency"] = np.nan

    # Composite elevation score (weighted)
    rapm_elev = result["elevation_rapm"].fillna(0)
    surplus_elev = result["elevation_surplus"].fillna(0)
    eff_elev = result["elevation_efficiency"].fillna(0)

    # Normalize each component to z-score within season for fair weighting
    for col in ["elevation_rapm", "elevation_surplus", "elevation_efficiency"]:
        z_col = f"{col}_z"
        if result[col].notna().any():
            result[z_col] = result.groupby("season")[col].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-9)
            )
        else:
            result[z_col] = 0.0

    result["elevation_score_raw"] = (
        cfg.w_rapm_elevation * result["elevation_rapm_z"].fillna(0) +
        cfg.w_playtype_elevation * result["elevation_surplus_z"].fillna(0) +
        cfg.w_efficiency_elevation * result["elevation_efficiency_z"].fillna(0)
    )

    # Convert to percentile
    result["elevation_score"] = result.groupby("season")["elevation_score_raw"].transform(
        vectorized_percentile_rank
    )

    # v2.0: Store the z-scored composite for z-score aggregation pipeline
    result["elevation_z"] = result["elevation_score_raw"]  # already a z-score composite

    # v2.0: Also add z-scores for downstream aggregation
    result = add_league_z_scores(result, ["elevation_score_raw"])

    # Add archetype-level percentile
    if "primary_archetype" in result.columns:
        result = add_grouped_percentiles(
            result,
            ["elevation_score_raw"],
            group_col="primary_archetype",
            suffix="_pctl",
        )

    # Add league percentile
    result = add_league_percentiles(result, ["elevation_score_raw"])

    # Classify elevation tier
    result["elevation_tier"] = pd.cut(
        result["elevation_score"],
        bins=[0, 25, 50, 75, 90, 100],
        labels=["Below Baseline", "Baseline", "Above Average", "Elite", "Archetype-Best"],
        include_lowest=True,
    )

    # Clean up z-score columns
    z_cols = [c for c in result.columns if c.endswith("_z")]
    result = result.drop(columns=z_cols)

    return result


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def build_archetype_elevation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full Layer 3 pipeline: archetype baselines → elevation scores.

    Takes Layer 1+2 output DataFrame and adds Layer 3 columns.
    """
    print("\n" + "=" * 60)
    print("LAYER 3: ARCHETYPE ELEVATION")
    print("=" * 60)

    qualified = df[df.get("qualified", True) == True].copy() if "qualified" in df.columns else df.copy()

    if qualified.empty:
        print("  No qualified players to process")
        return df

    # 3A. Compute archetype baselines
    print("  Computing archetype baselines...")
    baselines = compute_archetype_baselines(qualified)

    if not baselines.empty:
        n_archetypes = baselines["primary_archetype"].nunique()
        print(f"  Baselines computed for {n_archetypes} archetypes")

        # Print summary
        for season in baselines["season"].unique():
            season_base = baselines[baselines["season"] == season]
            print(f"\n  Season {season}:")
            for _, row in season_base.iterrows():
                arch = row["primary_archetype"]
                n = int(row.get("archetype_cohort_size", 0))
                mean_rapm = row.get("baseline_rapm_mean", np.nan)
                print(f"    {arch:35s} | n={n:3d} | "
                      f"mean RAPM: {mean_rapm:+6.2f}" if not np.isnan(mean_rapm) else
                      f"    {arch:35s} | n={n:3d} | mean RAPM: N/A")

    # Merge baselines onto player-season data
    qualified = merge_archetype_baselines(qualified, baselines)

    # 3B. Elevation scores
    print("\n  Computing elevation scores...")
    qualified = compute_elevation_scores(qualified)

    # Merge back — use concat to avoid DataFrame fragmentation
    new_cols = [c for c in qualified.columns if c not in df.columns]
    if new_cols:
        fill = pd.DataFrame(np.nan, index=df.index, columns=new_cols)
        df = pd.concat([df, fill], axis=1)
        df.loc[qualified.index, new_cols] = qualified[new_cols].values

    n = qualified["elevation_score"].notna().sum()
    print(f"\n  Layer 3 complete: {n} players with elevation scores")
    if n > 0:
        print(f"  Elevation range: {qualified['elevation_score'].min():.1f} - "
              f"{qualified['elevation_score'].max():.1f}")

    return df


# ---------------------------------------------------------------------------
# CLI Entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.modeling.layer1_portable_talent import build_portable_talent
    from src.modeling.layer2_role_utilization import build_role_utilization

    df = build_portable_talent()
    if not df.empty:
        df = build_role_utilization(df)
        df = build_archetype_elevation(df)

        qualified = df[df.get("qualified", True) == True].copy()
        if not qualified.empty and "elevation_score" in qualified.columns:
            top = qualified.nlargest(10, "elevation_score")
            name_col = "player_name" if "player_name" in top.columns else "player_id"
            print("\nTop 10 Archetype Elevation Scores:")
            for _, row in top.iterrows():
                print(f"  {row.get(name_col, row['player_id']):25s} | "
                      f"Elev: {row['elevation_score']:5.1f} | "
                      f"Tier: {row.get('elevation_tier', 'N/A')} | "
                      f"Archetype: {row.get('primary_archetype', 'N/A')}")
