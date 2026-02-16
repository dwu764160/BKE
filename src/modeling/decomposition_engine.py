"""
src/modeling/decomposition_engine.py
=============================================================================
BKE v1.5 — FINAL DECOMPOSITION ENGINE

Ties together all four layers into the complete impact decomposition:

  Total Impact = Portable Talent + Role-Dependent Impact
                                   |
                                   +-- Role Utilization Efficiency
                                   +-- Archetype Elevation
                                   +-- Scheme Amplification

For each player, produces:
  1. Portable Talent Score (PTS) — How good they are anywhere
  2. Role-Dependent Impact Score (RDIS) — How much impact depends on environment
  3. Portability Ratio — Portable Talent / Total Impact (0-1 scalar)

All outputs percentile-standardized at three levels:
  - League
  - Position
  - Archetype

Outputs:
  - data/processed/bke_v15_decomposition.parquet / .csv
  - data/processed/bke_v15_report.json
=============================================================================
"""

import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.modeling.model_config import (
    DECOMPOSITION,
    BKE_OUTPUT_PARQUET,
    BKE_OUTPUT_CSV,
    BKE_REPORT_JSON,
    SEASONS,
    clean_id,
)
from src.modeling.percentile_engine import (
    PercentileEngine,
    add_league_percentiles,
    add_grouped_percentiles,
    vectorized_percentile_rank,
)
from src.modeling.layer1_portable_talent import build_portable_talent
from src.modeling.layer2_role_utilization import build_role_utilization
from src.modeling.layer3_archetype_elevation import build_archetype_elevation
from src.modeling.layer4_scheme_amplification import build_scheme_amplification


# ---------------------------------------------------------------------------
# Final Decomposition
# ---------------------------------------------------------------------------

def compute_total_impact(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Total Impact as the sum of Portable Talent and Role-Dependent Impact.

    Total Impact = Portable Talent Score + Role-Dependent Impact Score (RDIS)

    Where RDIS = weighted combination of:
      - Role Utilization Efficiency (RUE)
      - Archetype Elevation
      - Scheme Amplification

    Only qualified players receive scores; unqualified remain NaN.
    """
    cfg = DECOMPOSITION
    result = df.copy()

    # Identify qualified rows
    qualified_mask = (result["qualified"] == True) if "qualified" in result.columns else pd.Series(True, index=result.index)

    # Grab layer scores (NaN for unqualified — that's correct)
    pts = result["portable_talent_score"].copy() if "portable_talent_score" in result.columns else pd.Series(np.nan, index=result.index)
    rue = result["role_utilization_efficiency"].copy() if "role_utilization_efficiency" in result.columns else pd.Series(np.nan, index=result.index)
    elevation = result["elevation_score"].copy() if "elevation_score" in result.columns else pd.Series(np.nan, index=result.index)
    scheme_amp = result["scheme_amplification"].copy() if "scheme_amplification" in result.columns else pd.Series(np.nan, index=result.index)

    # For qualified players only: fill component NaNs with season-median
    # (falling back to overall median if entire season is NaN)
    for col_series, col_name in [(pts, "portable_talent_score"), (rue, "role_utilization_efficiency"),
                                  (elevation, "elevation_score"), (scheme_amp, "scheme_amplification")]:
        if col_name not in result.columns:
            col_series.loc[qualified_mask] = 50.0
            continue
        # Season median for qualified rows
        q_vals = col_series.loc[qualified_mask]
        season_medians = result.loc[qualified_mask].groupby("season")[col_name].transform("median")
        # Fill NaN with season median
        filled = q_vals.fillna(season_medians.reindex(q_vals.index, fill_value=np.nan))
        # If still NaN (entire season was NaN), fill with overall median
        overall_median = col_series.loc[qualified_mask].median()
        overall_median = overall_median if not np.isnan(overall_median) else 50.0
        filled = filled.fillna(overall_median)
        col_series.loc[qualified_mask] = filled

    # Role-Dependent Impact Score (RDIS) — among qualified only
    # Higher RUE = better role fit → more value extracted
    # Higher elevation = more value above archetype baseline
    # Higher scheme_amplification = MORE context-dependent (subtract from portability)
    rdis_raw = pd.Series(np.nan, index=result.index)
    rdis_raw.loc[qualified_mask] = (
        0.40 * rue.loc[qualified_mask] +
        0.40 * elevation.loc[qualified_mask] +
        0.20 * scheme_amp.loc[qualified_mask]  # scheme amp: high = dependent
    )

    result["role_dependent_impact_raw"] = rdis_raw
    # Percentile rank ONLY among qualified players
    result["role_dependent_impact_score"] = np.nan
    for season in result["season"].unique():
        mask = qualified_mask & (result["season"] == season)
        if mask.any():
            result.loc[mask, "role_dependent_impact_score"] = \
                vectorized_percentile_rank(result.loc[mask, "role_dependent_impact_raw"])

    # Total Impact — among qualified only
    total_raw = pd.Series(np.nan, index=result.index)
    total_raw.loc[qualified_mask] = (
        cfg.w_portable_talent * pts.loc[qualified_mask] +
        (cfg.w_role_utilization * rue.loc[qualified_mask] +
         cfg.w_archetype_elevation * elevation.loc[qualified_mask] +
         cfg.w_scheme_amplification * (100 - scheme_amp.loc[qualified_mask]))
    ) / (cfg.w_portable_talent + cfg.w_role_utilization +
         cfg.w_archetype_elevation + cfg.w_scheme_amplification)

    result["total_impact_raw"] = total_raw
    # Percentile rank ONLY among qualified players
    result["total_impact_score"] = np.nan
    for season in result["season"].unique():
        mask = qualified_mask & (result["season"] == season)
        if mask.any():
            result.loc[mask, "total_impact_score"] = \
                vectorized_percentile_rank(result.loc[mask, "total_impact_raw"])

    return result


def compute_portability_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Portability Ratio: Portable Talent / (Portable Talent + RDIS).

    Uses raw additive component scores (not percentile / percentile) to
    capture the true fraction of impact that is context-neutral.

    High (>0.70) → scalable star (talent transfers across contexts)
    Low  (<0.40) → system-amplified player (needs the right environment)

    This is a clean scalar for:
      - Trade valuation
      - Scalability analysis
      - Role projection
    """
    result = df.copy()
    cfg = DECOMPOSITION

    qualified_mask = (result["qualified"] == True) if "qualified" in result.columns else pd.Series(True, index=result.index)

    pts = result.get("portable_talent_score", pd.Series(np.nan, index=result.index))
    rdis = result.get("role_dependent_impact_score", pd.Series(np.nan, index=result.index))

    # Portability ratio = portable fraction of total (among qualified only)
    result["portability_ratio"] = np.nan
    valid = qualified_mask & pts.notna() & rdis.notna()
    result.loc[valid, "portability_ratio"] = np.clip(
        pts.loc[valid] / (pts.loc[valid] + rdis.loc[valid] + 1e-9),
        0.0,
        1.0,
    )

    # Classification
    result["portability_class"] = "Unranked"
    result.loc[valid, "portability_class"] = np.where(
        result.loc[valid, "portability_ratio"] >= cfg.high_portability,
        "Scalable Star",
        np.where(
            result.loc[valid, "portability_ratio"] <= cfg.low_portability,
            "System Player",
            "Context-Moderate"
        )
    )

    return result


def compute_impact_tiers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign impact tiers based on total impact percentile.
    """
    result = df.copy()
    cfg = DECOMPOSITION

    total = result.get("total_impact_score", pd.Series(50.0, index=result.index))

    tiers = []
    for _, row in result.iterrows():
        score = row.get("total_impact_score", 50)
        if np.isnan(score):
            tiers.append("Unranked")
            continue

        assigned = "Unranked"
        for tier_name, (lo, hi) in cfg.tier_boundaries.items():
            if lo <= score <= hi:
                assigned = tier_name
                break
        tiers.append(assigned)

    result["impact_tier"] = tiers

    return result


def add_final_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add three-level percentiles for all final output metrics.
    """
    result = df.copy()

    final_metrics = [
        "portable_talent_score",
        "role_dependent_impact_score",
        "total_impact_score",
        "portability_ratio",
    ]

    # League percentiles
    result = add_league_percentiles(result, final_metrics)

    # Position percentiles
    if "position_bucket" in result.columns:
        result = add_grouped_percentiles(
            result, final_metrics,
            group_col="position_bucket",
            suffix="_pctl",
        )

    # Archetype percentiles
    if "primary_archetype" in result.columns:
        result = add_grouped_percentiles(
            result, final_metrics,
            group_col="primary_archetype",
            suffix="_pctl",
        )

    return result


# ---------------------------------------------------------------------------
# Output Card Generation
# ---------------------------------------------------------------------------

def generate_player_card(row: pd.Series) -> Dict:
    """
    Generate a player impact card for a single player-season.
    
    Example:
        Player X:
        Portable Talent: League 91st, Position 94th, Archetype 88th
        Role-Dependent: League 72nd, Archetype 84th
        Portability Ratio: 0.78
    """
    name_col = "player_name" if "player_name" in row.index else "player_id"

    card = {
        "player_name": str(row.get(name_col, row.get("player_id", "Unknown"))),
        "player_id": str(row.get("player_id", "")),
        "season": str(row.get("season", "")),
        "primary_archetype": str(row.get("primary_archetype", "N/A")),
        "defensive_archetype": str(row.get("defensive_archetype", "N/A")),
        "position_bucket": str(row.get("position_bucket", "N/A")),
        "impact_tier": str(row.get("impact_tier", "Unranked")),
        "portable_talent": {
            "score": _safe_float(row.get("portable_talent_score")),
            "league_pctl": _safe_float(row.get("portable_talent_score_league_pctl")),
            "position_pctl": _safe_float(row.get("portable_talent_score_position_bucket_pctl")),
            "archetype_pctl": _safe_float(row.get("portable_talent_score_primary_archetype_pctl")),
        },
        "role_dependent_impact": {
            "score": _safe_float(row.get("role_dependent_impact_score")),
            "league_pctl": _safe_float(row.get("role_dependent_impact_score_league_pctl")),
            "rue": _safe_float(row.get("role_utilization_efficiency")),
            "elevation": _safe_float(row.get("elevation_score")),
            "scheme_amplification": _safe_float(row.get("scheme_amplification")),
        },
        "total_impact": {
            "score": _safe_float(row.get("total_impact_score")),
            "league_pctl": _safe_float(row.get("total_impact_score_league_pctl")),
        },
        "portability_ratio": _safe_float(row.get("portability_ratio")),
        "portability_class": str(row.get("portability_class", "N/A")),
        "scheme_classification": str(row.get("scheme_classification", "N/A")),
        "rapm": _safe_float(row.get("rapm")),
        "orapm": _safe_float(row.get("orapm")),
        "drapm": _safe_float(row.get("drapm")),
    }
    return card


def _safe_float(val) -> Optional[float]:
    """Convert to float, returning None for NaN."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        return round(float(val), 2)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------

def generate_report(df: pd.DataFrame) -> Dict:
    """
    Generate a summary report of the v1.5 decomposition run.
    """
    qualified = df[df.get("qualified", True) == True] if "qualified" in df.columns else df

    report = {
        "version": "1.5",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_players": int(len(df)),
        "qualified_players": int(len(qualified)),
        "seasons": sorted(df["season"].unique().tolist()),
        "metrics_summary": {},
        "tier_distribution": {},
        "portability_distribution": {},
        "top_10_total_impact": [],
        "top_10_portable_talent": [],
        "top_10_portability_ratio": [],
    }

    # Metrics summary
    for metric in ["portable_talent_score", "role_dependent_impact_score",
                    "total_impact_score", "portability_ratio"]:
        if metric in qualified.columns:
            vals = qualified[metric].dropna()
            report["metrics_summary"][metric] = {
                "mean": round(float(vals.mean()), 2) if len(vals) > 0 else None,
                "std": round(float(vals.std()), 2) if len(vals) > 0 else None,
                "min": round(float(vals.min()), 2) if len(vals) > 0 else None,
                "max": round(float(vals.max()), 2) if len(vals) > 0 else None,
            }

    # Tier distribution
    if "impact_tier" in qualified.columns:
        tier_counts = qualified["impact_tier"].value_counts().to_dict()
        report["tier_distribution"] = {str(k): int(v) for k, v in tier_counts.items()}

    # Portability distribution
    if "portability_class" in qualified.columns:
        port_counts = qualified["portability_class"].value_counts().to_dict()
        report["portability_distribution"] = {str(k): int(v) for k, v in port_counts.items()}

    # Top 10 lists
    name_col = "player_name" if "player_name" in qualified.columns else "player_id"

    if "total_impact_score" in qualified.columns:
        top10_total = qualified.nlargest(10, "total_impact_score")
        for _, row in top10_total.iterrows():
            report["top_10_total_impact"].append({
                "player": str(row.get(name_col, row["player_id"])),
                "season": str(row["season"]),
                "score": round(float(row["total_impact_score"]), 1),
                "archetype": str(row.get("primary_archetype", "N/A")),
            })

    if "portable_talent_score" in qualified.columns:
        top10_pts = qualified.nlargest(10, "portable_talent_score")
        for _, row in top10_pts.iterrows():
            report["top_10_portable_talent"].append({
                "player": str(row.get(name_col, row["player_id"])),
                "season": str(row["season"]),
                "score": round(float(row["portable_talent_score"]), 1),
            })

    if "portability_ratio" in qualified.columns:
        top10_port = qualified.nlargest(10, "portability_ratio")
        for _, row in top10_port.iterrows():
            report["top_10_portability_ratio"].append({
                "player": str(row.get(name_col, row["player_id"])),
                "season": str(row["season"]),
                "ratio": round(float(row["portability_ratio"]), 3),
                "class": str(row.get("portability_class", "N/A")),
            })

    return report


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def run_full_decomposition(
    seasons: Optional[List[str]] = None,
    use_possession_data: bool = False,
    save_output: bool = True,
) -> pd.DataFrame:
    """
    Run the complete v1.5 BKE Impact Decomposition Engine.

    Pipeline order:
      Layer 1: Portable Talent
      Layer 2: Role Utilization Efficiency
      Layer 3: Archetype Elevation
      Layer 4: Scheme Amplification
      Final:   Decomposition + Percentiles + Output

    Parameters:
        seasons: List of seasons to process (default: all)
        use_possession_data: Whether to load raw possessions for Layer 4
                            lineup analysis. Set False for faster runs.
        save_output: Whether to save output files

    Returns:
        Full decomposition DataFrame
    """
    start = time.time()
    seasons = seasons or SEASONS

    print("\n" + "=" * 70)
    print("  BKE v1.5 — PORTABLE TALENT vs ROLE-DEPENDENT IMPACT ENGINE")
    print("=" * 70)
    print(f"  Seasons: {seasons}")
    print(f"  Possession data: {'Yes' if use_possession_data else 'No (fast mode)'}")

    # -----------------------------------------------------------------------
    # Layer 1: Portable Talent
    # -----------------------------------------------------------------------
    df = build_portable_talent(seasons=seasons)
    if df.empty:
        print("\n  FATAL: Layer 1 produced no data. Aborting.")
        return pd.DataFrame()

    # -----------------------------------------------------------------------
    # Layer 2: Role Utilization Efficiency
    # -----------------------------------------------------------------------
    df = build_role_utilization(df)

    # -----------------------------------------------------------------------
    # Layer 3: Archetype Elevation
    # -----------------------------------------------------------------------
    df = build_archetype_elevation(df)

    # -----------------------------------------------------------------------
    # Layer 4: Scheme Amplification
    # -----------------------------------------------------------------------
    df = build_scheme_amplification(df, use_possession_data=use_possession_data)

    # -----------------------------------------------------------------------
    # Final Decomposition
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("FINAL DECOMPOSITION")
    print("=" * 60)

    print("  Computing total impact...")
    df = compute_total_impact(df)

    print("  Computing portability ratio...")
    df = compute_portability_ratio(df)

    print("  Assigning impact tiers...")
    df = compute_impact_tiers(df)

    print("  Adding final three-level percentiles...")
    df = add_final_percentiles(df)

    # -----------------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------------
    elapsed = time.time() - start
    qualified = df[df.get("qualified", True) == True] if "qualified" in df.columns else df

    print("\n" + "=" * 70)
    print("  BKE v1.5 — DECOMPOSITION COMPLETE")
    print("=" * 70)
    print(f"  Total players: {len(df)}")
    print(f"  Qualified players: {len(qualified)}")
    print(f"  Runtime: {elapsed:.1f}s")

    if not qualified.empty:
        name_col = "player_name" if "player_name" in qualified.columns else "player_id"

        print("\n  Top 10 Total Impact:")
        if "total_impact_score" in qualified.columns:
            top = qualified.nlargest(10, "total_impact_score")
            for _, row in top.iterrows():
                port = row.get("portability_ratio", 0)
                print(f"    {row.get(name_col, row['player_id']):25s} | "
                      f"Impact: {row['total_impact_score']:5.1f} | "
                      f"PTS: {row.get('portable_talent_score', 0):5.1f} | "
                      f"Port: {port:.2f} | "
                      f"{row.get('primary_archetype', 'N/A')}")

        print("\n  Top 10 Most Portable:")
        if "portability_ratio" in qualified.columns:
            top_port = qualified.nlargest(10, "portability_ratio")
            for _, row in top_port.iterrows():
                print(f"    {row.get(name_col, row['player_id']):25s} | "
                      f"Portability: {row['portability_ratio']:.3f} | "
                      f"{row.get('portability_class', 'N/A')}")

    # Save outputs
    if save_output:
        print("\n  Saving outputs...")
        df.to_parquet(BKE_OUTPUT_PARQUET, index=False)
        df.to_csv(BKE_OUTPUT_CSV, index=False)
        print(f"  Saved: {BKE_OUTPUT_PARQUET}")
        print(f"  Saved: {BKE_OUTPUT_CSV}")

        report = generate_report(df)
        with open(BKE_REPORT_JSON, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"  Saved: {BKE_REPORT_JSON}")

    return df


# ---------------------------------------------------------------------------
# CLI Entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BKE v1.5 Decomposition Engine")
    parser.add_argument("--seasons", nargs="*", default=None,
                        help="Seasons to process (default: all)")
    parser.add_argument("--possession-data", action="store_true",
                        help="Use raw possession data for lineup analysis (slower)")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save output files")
    args = parser.parse_args()

    result = run_full_decomposition(
        seasons=args.seasons,
        use_possession_data=args.possession_data,
        save_output=not args.no_save,
    )
