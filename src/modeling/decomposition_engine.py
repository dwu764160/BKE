"""
src/modeling/decomposition_engine.py
=============================================================================
BKE v2.0 — FINAL DECOMPOSITION ENGINE

Ties together all four layers into the complete impact decomposition:

  Total Impact = Portable Talent + Role-Dependent Impact
                                   |
                                   +-- Role Utilization Efficiency
                                   +-- Archetype Elevation
                                   +-- Scheme Amplification

v2.0 key fixes:
  - Z-score aggregation: Raw → Z → Weighted Sum → Final Z → Percentile
    (percentiles are presentation-only, not aggregation math)
  - True Portability Index: variance-based stability measurement replaces
    the fake compositional ratio (PTS/Total → 1 - NormalizedVariance)
  - 4-component portability: Lineup Stability, Role Elasticity,
    Archetype Transfer, On/Off Context Sensitivity

For each player, produces:
  1. Portable Talent Score (PTS) — How good they are anywhere
  2. Role-Dependent Impact Score (RDIS) — Environment-dependent contribution
  3. Portability Index — True context-stability measurement (0-1)

All outputs percentile-standardized at three levels:
  - League / Position / Archetype

Outputs:
  - data/processed/bke_v20_decomposition.parquet / .csv
  - data/processed/bke_v20_report.json
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
    PORTABILITY,
    ROLE_UTILIZATION,
    BKE_OUTPUT_PARQUET,
    BKE_OUTPUT_CSV,
    BKE_REPORT_JSON,
    SEASONS,
    clean_id,
)
from src.modeling.percentile_engine import (
    PercentileEngine,
    add_league_percentiles,
    add_league_z_scores,
    add_grouped_percentiles,
    compute_z_score,
    z_to_percentile,
    weighted_z_composite,
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
    Compute Total Impact using v2.0 z-score aggregation.

    v2.0 pipeline:
      1. Grab layer z-scores (portable_talent_z, RUE z, elevation_z, scheme_stability_z)
      2. Weighted z-score composite: Total_z = sum(w_i * z_i) / sum(w_i)
      3. Convert final z-score to percentile (presentation only)

    This fixes the percentile-averaging distortion in v1.5.
    """
    cfg = DECOMPOSITION
    result = df.copy()

    # Identify qualified rows
    qualified_mask = (result["qualified"] == True) if "qualified" in result.columns else pd.Series(True, index=result.index)

    # --- Grab layer z-scores ---
    # PTS z-score (from Layer 1)
    pts_z = result.get("portable_talent_z_adj", result.get("portable_talent_z",
                pd.Series(0.0, index=result.index)))

    # RUE z-score (from Layer 2)
    rue_z = result.get("role_utilization_raw_z",
                pd.Series(0.0, index=result.index))

    # Elevation z-score (from Layer 3)
    elev_z = result.get("elevation_z", result.get("elevation_score_raw",
                pd.Series(0.0, index=result.index)))

    # Scheme stability z-score (from Layer 4) — higher = more portable = better
    scheme_z = result.get("scheme_stability_z",
                pd.Series(0.0, index=result.index))

    # --- Fill NaN for qualified with season median, overall fallback ---
    for z_series_name in ["_pts_z", "_rue_z", "_elev_z", "_scheme_z"]:
        z_map = {"_pts_z": pts_z, "_rue_z": rue_z, "_elev_z": elev_z, "_scheme_z": scheme_z}
        z_s = z_map[z_series_name]
        q_vals = z_s.loc[qualified_mask]
        season_medians = result.loc[qualified_mask].groupby("season").apply(
            lambda g: g.index.map(lambda i: z_s.loc[g.index].median())
        )
        # Flatten season medians
        flat_medians = {}
        for season in result["season"].unique():
            s_mask = qualified_mask & (result["season"] == season)
            med = z_s.loc[s_mask].median()
            flat_medians[season] = med if not np.isnan(med) else 0.0
        fill_vals = result.loc[qualified_mask, "season"].map(flat_medians)
        filled = q_vals.fillna(fill_vals)
        overall_med = z_s.loc[qualified_mask].median()
        overall_med = overall_med if not np.isnan(overall_med) else 0.0
        filled = filled.fillna(overall_med)
        z_map[z_series_name] = z_s.copy()
        z_map[z_series_name].loc[qualified_mask] = filled

        # Write back
        if z_series_name == "_pts_z":
            pts_z = z_map[z_series_name]
        elif z_series_name == "_rue_z":
            rue_z = z_map[z_series_name]
        elif z_series_name == "_elev_z":
            elev_z = z_map[z_series_name]
        elif z_series_name == "_scheme_z":
            scheme_z = z_map[z_series_name]

    # --- Role-Dependent Impact z-score ---
    rdis_z = pd.Series(np.nan, index=result.index)
    rdis_z.loc[qualified_mask] = (
        0.40 * rue_z.loc[qualified_mask] +
        0.40 * elev_z.loc[qualified_mask] +
        0.20 * (-scheme_z.loc[qualified_mask])  # invert: high stability → low role-dependence
    )

    result["role_dependent_impact_z"] = rdis_z
    result["role_dependent_impact_raw"] = rdis_z  # compat

    # Percentile rank ONLY among qualified
    result["role_dependent_impact_score"] = np.nan
    for season in result["season"].unique():
        mask = qualified_mask & (result["season"] == season)
        if mask.any():
            result.loc[mask, "role_dependent_impact_score"] = \
                vectorized_percentile_rank(result.loc[mask, "role_dependent_impact_z"])

    # --- Total Impact z-score ---
    total_z = pd.Series(np.nan, index=result.index)
    total_z.loc[qualified_mask] = (
        cfg.w_portable_talent * pts_z.loc[qualified_mask] +
        cfg.w_role_utilization * rue_z.loc[qualified_mask] +
        cfg.w_archetype_elevation * elev_z.loc[qualified_mask] +
        cfg.w_scheme_amplification * scheme_z.loc[qualified_mask]
    ) / (cfg.w_portable_talent + cfg.w_role_utilization +
         cfg.w_archetype_elevation + cfg.w_scheme_amplification)

    result["total_impact_z"] = total_z
    result["total_impact_raw"] = total_z  # compat

    # Percentile rank ONLY among qualified
    result["total_impact_score"] = np.nan
    for season in result["season"].unique():
        mask = qualified_mask & (result["season"] == season)
        if mask.any():
            result.loc[mask, "total_impact_score"] = \
                vectorized_percentile_rank(result.loc[mask, "total_impact_z"])

    # Also store CDF-based percentile for cross-era comparison
    result["total_impact_cdf_pctl"] = np.nan
    result.loc[qualified_mask, "total_impact_cdf_pctl"] = z_to_percentile(
        total_z.loc[qualified_mask]
    )

    return result


# ---------------------------------------------------------------------------
# v2.0: True Portability Index (Variance-Based)
# ---------------------------------------------------------------------------

def _compute_lineup_stability(df: pd.DataFrame) -> pd.Series:
    """
    Component 1: Lineup Stability Index.

    Measures variance of player impact signals across different contexts.
    Approximated from multi-model agreement (RAPM vs DARKO) and
    dimensional spread in the portable model.

    Low variance → high stability → portable.
    """
    result = pd.Series(0.5, index=df.index)

    # RAPM vs DARKO agreement
    if "rapm" in df.columns and "darko_dpm" in df.columns:
        rapm_z = df.groupby("season")["rapm"].transform(
            lambda x: compute_z_score(x, winsorize=3.5)
        )
        darko_z = df.groupby("season")["darko_dpm"].transform(
            lambda x: compute_z_score(x, winsorize=3.5)
        )
        # Agreement = 1 - abs(diff) / max_range
        agreement = (1.0 - np.abs(rapm_z - darko_z).clip(0, 3) / 3.0).fillna(0.5)
        result = agreement

    # Dimensional consistency: low variance across 8 dims = more stable
    dim_z_cols = [c for c in df.columns if c.startswith("dim_") and c.endswith("_z")
                  and not c.endswith("_poss_offensive_z") and not c.endswith("_poss_defensive_z")]
    if len(dim_z_cols) >= 4:
        dim_values = df[dim_z_cols].fillna(0)
        dim_variance = dim_values.var(axis=1)
        # Invert: low variance = high stability
        dim_stability = df.groupby("season").apply(
            lambda g: 1.0 - (dim_variance.loc[g.index] - dim_variance.loc[g.index].min()) /
                      (dim_variance.loc[g.index].max() - dim_variance.loc[g.index].min() + 1e-9)
        )
        if hasattr(dim_stability, 'droplevel'):
            dim_stability = dim_stability.droplevel(0)
        # Blend with model agreement
        result = 0.5 * result + 0.5 * dim_stability.reindex(df.index, fill_value=0.5)

    return result.clip(0, 1)


def _compute_role_elasticity(df: pd.DataFrame) -> pd.Series:
    """
    Component 2: Role Elasticity Test.

    Simulate ±5% usage shift and estimate impact change.

    Players whose playtype surplus is consistent across volume levels
    are more role-elastic (portable to different usage rates).

    Approximated from surplus stability and usage diversity.
    """
    cfg = PORTABILITY
    result = pd.Series(0.5, index=df.index)

    # Playtype surplus consistency across different playtypes
    surplus_cols = [c for c in df.columns if c.endswith("_surplus") and not c.endswith("_total")]
    if len(surplus_cols) >= 3:
        surplus_vals = df[surplus_cols].fillna(0)
        # Use coefficient of variation: low CoV = consistent surplus across roles
        surplus_mean = surplus_vals.mean(axis=1).abs() + 1e-9
        surplus_std = surplus_vals.std(axis=1)
        cov = surplus_std / surplus_mean
        # Invert and normalize per season
        result = df.groupby("season").apply(
            lambda g: 1.0 - (cov.loc[g.index] - cov.loc[g.index].min()) /
                      (cov.loc[g.index].max() - cov.loc[g.index].min() + 1e-9)
        )
        if hasattr(result, 'droplevel'):
            result = result.droplevel(0)
        result = result.reindex(df.index, fill_value=0.5)

    # PCV entropy: high entropy = more usage diversity = more elastic
    entropy_col = None
    for ec in ["pcv_entropy", "emb_entropy_norm", "emb_entropy"]:
        if ec in df.columns:
            entropy_col = ec
            break

    if entropy_col is not None:
        entropy_norm = df.groupby("season")[entropy_col].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
        ).fillna(0.5)
        result = 0.6 * result + 0.4 * entropy_norm

    return result.clip(0, 1)


def _compute_archetype_transfer(df: pd.DataFrame) -> pd.Series:
    """
    Component 3: Archetype Transfer Simulation.

    Project player into N alternative archetype usage templates.
    Measure projected efficiency change.
    Less dropoff = more portable.

    Approximated: players whose skill profile is balanced across
    multiple dimensions will transfer better across archetype roles.
    """
    result = pd.Series(0.5, index=df.index)

    # Balanced offensive + defensive composite = transfers across roles
    if "offensive_portable_z" in df.columns and "defensive_portable_z" in df.columns:
        off_z = df["offensive_portable_z"].fillna(0)
        def_z = df["defensive_portable_z"].fillna(0)

        # Balance score: 1 when equal magnitude, lower when lopsided
        total_abs = off_z.abs() + def_z.abs() + 1e-9
        diff_abs = (off_z - def_z).abs()
        balance = 1.0 - (diff_abs / total_abs)

        # Also reward having both positive (good at both)
        both_positive = (off_z > 0) & (def_z > 0)
        result = 0.6 * balance + 0.4 * both_positive.astype(float)

    # Role confidence: low confidence = could fit multiple roles = potentially more transferable
    # BUT also could mean unclear skill → less transferable. Use moderate confidence.
    if "role_confidence" in df.columns:
        rc = df["role_confidence"].fillna(0.5)
        # Moderate confidence (0.4-0.7) is most transferable
        # Use inverted distance from 0.55
        transfer_from_rc = 1.0 - np.abs(rc - 0.55).clip(0, 0.45) / 0.45
        result = 0.7 * result + 0.3 * transfer_from_rc

    return result.clip(0, 1)


def _compute_context_sensitivity(df: pd.DataFrame) -> pd.Series:
    """
    Component 4: On/Off Context Sensitivity.

    Measures how much a player's impact varies across different environmental
    conditions (bench-heavy vs starter-heavy, pace differences, etc.).

    Approximated from scheme stability and ORAPM/DRAPM balance.
    """
    result = pd.Series(0.5, index=df.index)

    # Scheme stability (from Layer 4) is a direct context sensitivity measure
    if "scheme_stability_raw" in df.columns:
        result = df.groupby("season")["scheme_stability_raw"].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
        ).fillna(0.5)

    # ORAPM/DRAPM balance as additional signal
    # Two-way players are less context-sensitive
    if "orapm" in df.columns and "drapm" in df.columns:
        orapm_z = df.groupby("season")["orapm"].transform(
            lambda x: compute_z_score(x, winsorize=3.5)
        ).fillna(0)
        drapm_z = df.groupby("season")["drapm"].transform(
            lambda x: compute_z_score(x, winsorize=3.5)
        ).fillna(0)

        # Both positive & balanced = very context-insensitive
        total = orapm_z.abs() + drapm_z.abs() + 1e-9
        imbalance = (orapm_z - drapm_z).abs() / total
        two_way = 1.0 - imbalance
        result = 0.65 * result + 0.35 * two_way.clip(0, 1)

    return result.clip(0, 1)


def compute_portability_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the True Portability Index (v2.0 Fix #2).

    Portability = 1 - Normalized Impact Variance Across Contexts

    Components:
      1. Lineup Stability Index (30%) — variance across teammate contexts
      2. Role Elasticity Test (25%) — impact change under usage shifts
      3. Archetype Transfer Simulation (20%) — efficiency in alt archetypes
      4. On/Off Context Sensitivity (25%) — variance across environments

    This is STRUCTURAL, not COMPOSITIONAL. It measures actual transfer stability,
    not the ratio of one score to another.
    """
    result = df.copy()
    cfg = PORTABILITY

    qualified_mask = (result["qualified"] == True) if "qualified" in result.columns else pd.Series(True, index=result.index)

    # Compute 4 portability components
    print("    Computing lineup stability index...")
    lineup_stability = _compute_lineup_stability(result)

    print("    Computing role elasticity test...")
    role_elasticity = _compute_role_elasticity(result)

    print("    Computing archetype transfer simulation...")
    archetype_transfer = _compute_archetype_transfer(result)

    print("    Computing context sensitivity...")
    context_sensitivity = _compute_context_sensitivity(result)

    # Store individual components
    result["portability_lineup_stability"] = np.nan
    result["portability_role_elasticity"] = np.nan
    result["portability_archetype_transfer"] = np.nan
    result["portability_context_sensitivity"] = np.nan
    result.loc[qualified_mask, "portability_lineup_stability"] = lineup_stability.loc[qualified_mask]
    result.loc[qualified_mask, "portability_role_elasticity"] = role_elasticity.loc[qualified_mask]
    result.loc[qualified_mask, "portability_archetype_transfer"] = archetype_transfer.loc[qualified_mask]
    result.loc[qualified_mask, "portability_context_sensitivity"] = context_sensitivity.loc[qualified_mask]

    # Weighted composite
    portability_raw = pd.Series(np.nan, index=result.index)
    portability_raw.loc[qualified_mask] = (
        cfg.w_lineup_stability * lineup_stability.loc[qualified_mask] +
        cfg.w_role_elasticity * role_elasticity.loc[qualified_mask] +
        cfg.w_archetype_transfer * archetype_transfer.loc[qualified_mask] +
        cfg.w_context_sensitivity * context_sensitivity.loc[qualified_mask]
    )

    result["portability_index_raw"] = portability_raw

    # Percentile-rank within season (qualified only)
    result["portability_index"] = np.nan
    for season in result["season"].unique():
        mask = qualified_mask & (result["season"] == season)
        if mask.any():
            result.loc[mask, "portability_index"] = \
                vectorized_percentile_rank(result.loc[mask, "portability_index_raw"]) / 100.0

    # Classification
    result["portability_class"] = "Unranked"
    valid = qualified_mask & result["portability_index"].notna()
    result.loc[valid, "portability_class"] = np.where(
        result.loc[valid, "portability_index"] >= cfg.high_portability,
        "Scalable Star",
        np.where(
            result.loc[valid, "portability_index"] <= cfg.low_portability,
            "System Player",
            "Context-Moderate"
        )
    )

    # v1.5 compat: portability_ratio alias
    result["portability_ratio"] = result["portability_index"]

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
        "portability_index",
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
    Generate a player impact card for a single player-season (v2.0).

    Includes z-score based Total Impact, True Portability Index,
    and all 4 portability components.
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
            "z_score": _safe_float(row.get("portable_talent_z_adj", row.get("portable_talent_z"))),
            "cdf_pctl": _safe_float(row.get("portable_talent_cdf_pctl")),
            "league_pctl": _safe_float(row.get("portable_talent_score_league_pctl")),
            "position_pctl": _safe_float(row.get("portable_talent_score_position_bucket_pctl")),
            "archetype_pctl": _safe_float(row.get("portable_talent_score_primary_archetype_pctl")),
        },
        "role_dependent_impact": {
            "score": _safe_float(row.get("role_dependent_impact_score")),
            "z_score": _safe_float(row.get("role_dependent_impact_z")),
            "league_pctl": _safe_float(row.get("role_dependent_impact_score_league_pctl")),
            "rue": _safe_float(row.get("role_utilization_efficiency")),
            "elevation": _safe_float(row.get("elevation_score")),
            "scheme_amplification": _safe_float(row.get("scheme_amplification")),
        },
        "total_impact": {
            "score": _safe_float(row.get("total_impact_score")),
            "z_score": _safe_float(row.get("total_impact_z")),
            "cdf_pctl": _safe_float(row.get("total_impact_cdf_pctl")),
            "league_pctl": _safe_float(row.get("total_impact_score_league_pctl")),
        },
        "portability": {
            "index": _safe_float(row.get("portability_index")),
            "raw": _safe_float(row.get("portability_index_raw")),
            "class": str(row.get("portability_class", "N/A")),
            "lineup_stability": _safe_float(row.get("portability_lineup_stability")),
            "role_elasticity": _safe_float(row.get("portability_role_elasticity")),
            "archetype_transfer": _safe_float(row.get("portability_archetype_transfer")),
            "context_sensitivity": _safe_float(row.get("portability_context_sensitivity")),
        },
        "portability_ratio": _safe_float(row.get("portability_ratio")),  # compat
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
    Generate a summary report of the v2.0 decomposition run.
    """
    qualified = df[df.get("qualified", True) == True] if "qualified" in df.columns else df

    report = {
        "version": "2.0",
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
                "index": round(float(row.get("portability_index", row["portability_ratio"])), 3),
                "class": str(row.get("portability_class", "N/A")),
            })

    # v2.0 z-score summary
    for zmetric in ["total_impact_z", "portable_talent_z", "role_dependent_impact_z",
                     "portability_index_raw"]:
        if zmetric in qualified.columns:
            vals = qualified[zmetric].dropna()
            report["metrics_summary"][zmetric] = {
                "mean": round(float(vals.mean()), 3) if len(vals) > 0 else None,
                "std": round(float(vals.std()), 3) if len(vals) > 0 else None,
                "min": round(float(vals.min()), 3) if len(vals) > 0 else None,
                "max": round(float(vals.max()), 3) if len(vals) > 0 else None,
            }

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
    Run the complete v2.0 BKE Impact Decomposition Engine.

    v2.0 fixes:
      - Z-score aggregation (Raw→Z→WeightedSum→FinalZ→Percentile)
      - True Portability Index (variance-based, 4 components)
      - 8-dimension Layer 1C model
      - Bayesian shrinkage for noisy metrics
      - Cross-layer independence enforced

    Pipeline order:
      Layer 1: Portable Talent (8-dimension model, z-score backbone)
      Layer 2: Role Utilization Efficiency (z-scored surplus)
      Layer 3: Archetype Elevation (z-scored elevation)
      Layer 4: Scheme Amplification (z-scored stability)
      Final:   Decomposition + Portability Index + Percentiles + Output

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
    print("  BKE v2.0 — PORTABLE TALENT vs ROLE-DEPENDENT IMPACT ENGINE")
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

    print("  Computing portability index (v2.0 variance-based)...")
    df = compute_portability_index(df)

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
    print("  BKE v2.0 — DECOMPOSITION COMPLETE")
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

    parser = argparse.ArgumentParser(description="BKE v2.0 Decomposition Engine")
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
