"""
src/modeling/decomposition_engine.py
=============================================================================
BKE v2.8 — FINAL DECOMPOSITION ENGINE

Ties together all four layers into the complete impact decomposition:

  Total Impact = Portable Talent + Role-Dependent Impact
                                   |
                                   +-- Role Utilization Efficiency
                                   +-- Archetype Elevation
                                   +-- Scheme Amplification

v2.7 changes from v2.6:
  - Archetype-conditional neutralization in Layer 1C (Fix #3)
  - Three-level z-scores: league, positional, archetype (Fix #1)
  - Percentiles strictly terminal — never fed as inputs (Fix #2)
  - portability_ratio removed from public outputs (Fix #4)
  - Playmaking split: creation + pressure sub-components (MF-5)
  - Enhanced turnover control: TOV_PER_TOUCH, DRIVE_TOV_RATE (MF-1)
  - Expanded defensive playmaking: CHARGES_DRAWN, DEF_LOOSE_BALLS_RECOVERED (MF-4)
  - elevation_z preserved through Layer 3 (bug fix)

For each player, produces:
  1. Portable Talent Score (PTS) — How good they are anywhere
  2. Role-Dependent Impact Score (RDIS) — Environment-dependent contribution
  3. Portability Index — True context-stability measurement (0-1)

All outputs percentile-standardized at three levels:
  - League / Position / Archetype

Outputs:
    - data/processed/bke_v28_decomposition.parquet / .csv
    - data/processed/bke_v28_report.json
    - data/processed/bke_v28_variance_report.json
    - data/processed/bke_v28_compression_report.json
    - data/processed/dimension_scores_v28.json
    - data/processed/layer_scores_v28.json
    - data/processed/obke_dbke_scores_v28.json
=============================================================================
"""

import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import kurtosis

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.modeling.model_config import (
    DECOMPOSITION,
    DISTRIBUTION_INTEGRITY,
    PORTABILITY,
    ROLE_UTILIZATION,
    BKE_V28_OUTPUT_PARQUET,
    BKE_V28_OUTPUT_CSV,
    BKE_V28_REPORT_JSON,
    BKE_V28_VARIANCE_REPORT_JSON,
    BKE_V28_COMPRESSION_REPORT_JSON,
    BKE_V28_DIMENSION_SCORES_JSON,
    BKE_V28_LAYER_SCORES_JSON,
    BKE_V28_OBKE_DBKE_SCORES_JSON,
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


def _logistic_transform(series: pd.Series, alpha: float) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    return pd.Series(expit(alpha * vals) * 100.0, index=series.index)


def _apply_variance_anchor_by_season(
    values: pd.Series,
    reference: pd.Series,
    seasons: pd.Series,
    min_retention: float,
    max_scale: float,
) -> pd.Series:
    anchored = values.copy()
    for season in seasons.dropna().unique():
        mask = seasons == season
        cur = pd.to_numeric(values.loc[mask], errors="coerce")
        ref = pd.to_numeric(reference.loc[mask], errors="coerce")
        cur_std = float(cur.std()) if cur.notna().any() else np.nan
        ref_std = float(ref.std()) if ref.notna().any() else np.nan
        if np.isnan(cur_std) or np.isnan(ref_std) or ref_std < 1e-8:
            continue
        needed = min_retention * ref_std
        if cur_std < needed and cur_std > 1e-10:
            scale = min(max_scale, needed / cur_std)
            anchored.loc[mask] = cur * scale
    return anchored


def _series_distribution_stats(series: pd.Series) -> Dict:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if vals.empty:
        return {"count": 0, "mean": None, "std": None, "iqr": None, "kurtosis": None}
    q1 = float(vals.quantile(0.25))
    q3 = float(vals.quantile(0.75))
    return {
        "count": int(len(vals)),
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "iqr": float(q3 - q1),
        "kurtosis": float(kurtosis(vals, fisher=True, bias=False)) if len(vals) > 3 else None,
    }


def _tail_separation(series: pd.Series) -> Dict:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if vals.empty:
        return {"p90_p95": None, "p95_p99": None, "top5_spread": None}
    p90 = float(vals.quantile(0.90))
    p95 = float(vals.quantile(0.95))
    p99 = float(vals.quantile(0.99))
    top5 = vals.nlargest(min(5, len(vals)))
    return {
        "p90_p95": float(p95 - p90),
        "p95_p99": float(p99 - p95),
        "top5_spread": float(top5.max() - top5.min()) if len(top5) >= 2 else 0.0,
    }


# ---------------------------------------------------------------------------
# Final Decomposition
# ---------------------------------------------------------------------------

def compute_total_impact(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Total Impact using v2.7 z-score aggregation.

    Pipeline:
      1. Grab layer z-scores (portable_talent_z, RUE z, elevation_z, scheme_stability_z)
      2. Weighted z-score composite: Total_z = sum(w_i * z_i) / sum(w_i)
      3. Convert final z-score to percentile (presentation only)

    v2.7: dimension z-scores have soft-membership conditional neutralization upstream.
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

    # --- v2.6: Winsorize layer z-scores to prevent extreme outliers ---
    # RUE_z for players like Jokic can be -3.5; clip to ±2.5σ
    for z_name, z_series in [("_pts_z", pts_z), ("_rue_z", rue_z),
                              ("_elev_z", elev_z), ("_scheme_z", scheme_z)]:
        z_series.loc[qualified_mask] = z_series.loc[qualified_mask].clip(-2.5, 2.5)
        if z_name == "_pts_z": pts_z = z_series
        elif z_name == "_rue_z": rue_z = z_series
        elif z_name == "_elev_z": elev_z = z_series
        elif z_name == "_scheme_z": scheme_z = z_series

    # --- Role-Dependent Impact z-score ---
    rdis_z = pd.Series(np.nan, index=result.index)
    rdis_z.loc[qualified_mask] = (
        0.40 * rue_z.loc[qualified_mask] +
        0.40 * elev_z.loc[qualified_mask] +
        0.20 * (-scheme_z.loc[qualified_mask])  # invert: high stability → low role-dependence
    )

    result["role_dependent_impact_z"] = rdis_z
    result["role_dependent_impact_raw"] = rdis_z  # compat

    # v2.8 variance anchoring for role-dependent layer.
    rdis_reference = (rue_z + elev_z + scheme_z.abs()) / 3.0
    result["role_dependent_impact_z"] = _apply_variance_anchor_by_season(
        result["role_dependent_impact_z"],
        rdis_reference,
        result["season"],
        DISTRIBUTION_INTEGRITY.min_variance_retention,
        DISTRIBUTION_INTEGRITY.max_rescale_factor,
    )

    # Percentile rank ONLY among qualified
    result["role_dependent_impact_transformed"] = np.nan
    result["role_dependent_impact_score"] = np.nan
    for season in result["season"].unique():
        mask = qualified_mask & (result["season"] == season)
        if mask.any():
            transformed = _logistic_transform(
                result.loc[mask, "role_dependent_impact_z"],
                DISTRIBUTION_INTEGRITY.logistic_alpha,
            )
            result.loc[mask, "role_dependent_impact_transformed"] = transformed
            result.loc[mask, "role_dependent_impact_score"] = \
                vectorized_percentile_rank(transformed)

    # --- Total Impact z-score ---
    # v2.6: Scheme stability is bonus-only in TI. Being scheme-dependent
    # (negative scheme_z) should NOT subtract from total impact — stars who
    # define the scheme are still elite. Stability is a bonus, not a gate.
    scheme_z_clipped = scheme_z.copy()
    scheme_z_clipped.loc[qualified_mask] = scheme_z.loc[qualified_mask].clip(lower=0)

    total_z = pd.Series(np.nan, index=result.index)
    total_z.loc[qualified_mask] = (
        cfg.w_portable_talent * pts_z.loc[qualified_mask] +
        cfg.w_role_utilization * rue_z.loc[qualified_mask] +
        cfg.w_archetype_elevation * elev_z.loc[qualified_mask] +
        cfg.w_scheme_amplification * scheme_z_clipped.loc[qualified_mask]
    ) / (cfg.w_portable_talent + cfg.w_role_utilization +
         cfg.w_archetype_elevation + cfg.w_scheme_amplification)

    result["total_impact_z"] = total_z
    result["total_impact_raw"] = total_z  # compat

    # v2.8 variance anchoring for final total impact.
    total_reference = (
        cfg.w_portable_talent * pts_z +
        cfg.w_role_utilization * rue_z +
        cfg.w_archetype_elevation * elev_z +
        cfg.w_scheme_amplification * scheme_z.abs()
    ) / (cfg.w_portable_talent + cfg.w_role_utilization + cfg.w_archetype_elevation + cfg.w_scheme_amplification)
    result["total_impact_z"] = _apply_variance_anchor_by_season(
        result["total_impact_z"],
        total_reference,
        result["season"],
        DISTRIBUTION_INTEGRITY.min_variance_retention,
        DISTRIBUTION_INTEGRITY.max_rescale_factor,
    )

    # Percentile rank ONLY among qualified
    result["total_impact_transformed"] = np.nan
    result["total_impact_score"] = np.nan
    for season in result["season"].unique():
        mask = qualified_mask & (result["season"] == season)
        if mask.any():
            transformed = _logistic_transform(
                result.loc[mask, "total_impact_z"],
                DISTRIBUTION_INTEGRITY.logistic_alpha,
            )
            result.loc[mask, "total_impact_transformed"] = transformed
            result.loc[mask, "total_impact_score"] = \
                vectorized_percentile_rank(transformed)

    # v2.8 transformed portable talent for terminal percentile computation.
    result["portable_talent_transformed"] = np.nan
    if "portable_talent_z_adj" in result.columns:
        for season in result["season"].unique():
            mask = qualified_mask & (result["season"] == season)
            if mask.any():
                transformed = _logistic_transform(
                    result.loc[mask, "portable_talent_z_adj"],
                    DISTRIBUTION_INTEGRITY.logistic_alpha,
                )
                result.loc[mask, "portable_talent_transformed"] = transformed
                result.loc[mask, "portable_talent_score"] = vectorized_percentile_rank(transformed)

    # Also store CDF-based percentile for cross-era comparison
    result["total_impact_cdf_pctl"] = np.nan
    result.loc[qualified_mask, "total_impact_cdf_pctl"] = z_to_percentile(
        total_z.loc[qualified_mask]
    )

    return result


# ---------------------------------------------------------------------------
# v2.7: Structural Portability Index
# ---------------------------------------------------------------------------

def _compute_dimensional_breadth(df: pd.DataFrame) -> pd.Series:
    """
    Component 1: Dimensional Breadth (35%).

    Measures how many skill dimensions a player is above average in.
    Players who are good at many things are more portable than players
    who are elite at one thing.

    Two sub-signals:
      a) Count of positive dimensions / total (breadth)
      b) 1 - Herfindahl concentration of absolute z-scores (diffusion)
    """
    cfg = PORTABILITY
    dims = [c for c in cfg.all_dims if c in df.columns]
    if not dims:
        return pd.Series(0.5, index=df.index)

    dim_vals = df[dims].fillna(0)

    # (a) Fraction of dimensions above zero
    n_positive = (dim_vals > 0).sum(axis=1)
    breadth_frac = n_positive / len(dims)

    # (b) Herfindahl concentration: low = diffuse (more portable)
    abs_vals = dim_vals.abs()
    abs_sum = abs_vals.sum(axis=1) + 1e-9
    abs_sq_sum = (abs_vals ** 2).sum(axis=1)
    herfindahl = abs_sq_sum / (abs_sum ** 2)
    diffusion = 1.0 - herfindahl

    # Combine: 60% breadth, 40% diffusion
    result = 0.6 * breadth_frac + 0.4 * diffusion

    return result.clip(0, 1)


def _compute_universal_skill_presence(df: pd.DataFrame) -> pd.Series:
    """
    Component 2: Universal Skill Presence (25%).

    Measures whether the player has above-average scores in the
    three universally portable skill dimensions:
      - Shooting gravity
      - Playmaking creation
      - Self-creation

    These skills transfer to ANY team context.
    """
    cfg = PORTABILITY
    universal = [c for c in cfg.universal_dims if c in df.columns]
    if not universal:
        return pd.Series(0.5, index=df.index)

    uni_vals = df[universal].fillna(0)

    # (a) Count of universal dims above zero
    n_present = (uni_vals > 0).sum(axis=1)
    presence_frac = n_present / len(universal)

    # (b) Average z-score across universal dims, normalized to [0, 1]
    avg_z = uni_vals.mean(axis=1)
    # Map z-scores: -2σ → 0.0, 0σ → 0.5, +2σ → 1.0
    avg_norm = (avg_z.clip(-2, 2) + 2) / 4.0

    # Combine: 50% presence (did you show up?), 50% quality (how good?)
    result = 0.5 * presence_frac + 0.5 * avg_norm

    return result.clip(0, 1)


def _compute_two_way_balance(df: pd.DataFrame) -> pd.Series:
    """
    Component 3: Two-Way Balance (20%).

    Players who contribute on both offense and defense are more
    portable because they provide value regardless of team needs.

    Score = balance(off_z, def_z) × quality_bonus
    """
    result = pd.Series(0.5, index=df.index)

    if "offensive_portable_z" in df.columns and "defensive_portable_z" in df.columns:
        off_z = df["offensive_portable_z"].fillna(0)
        def_z = df["defensive_portable_z"].fillna(0)

        # Balance: 1 when both sides equal magnitude, 0 when fully one-sided
        total_abs = off_z.abs() + def_z.abs() + 1e-9
        diff_abs = (off_z - def_z).abs()
        balance = 1.0 - (diff_abs / total_abs)

        # Quality bonus: reward being positive on both sides
        both_positive = (off_z > 0) & (def_z > 0)
        one_positive = (off_z > 0) | (def_z > 0)
        quality = both_positive.astype(float) * 0.4 + one_positive.astype(float) * 0.1

        result = 0.5 * balance + 0.5 * quality

    return result.clip(0, 1)


def _compute_archetype_transfer(df: pd.DataFrame) -> pd.Series:
    """
    v2.7 Archetype Transfer signal from soft-membership entropy.

    High entropy indicates role transfer flexibility across adjacent archetypes.
    """
    if not PORTABILITY.use_soft_membership_in_transfer:
        return pd.Series(0.5, index=df.index)

    if "soft_archetype_entropy_norm" in df.columns:
        entropy = df["soft_archetype_entropy_norm"].fillna(0.5).clip(0, 1)
    else:
        prob_cols = [c for c in df.columns if c.startswith("arch_prob_")]
        if not prob_cols:
            return pd.Series(0.5, index=df.index)
        p = df[prob_cols].fillna(0.0).clip(lower=0.0)
        p = p.div(p.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        entropy_raw = -(np.where(p.values > 0, p.values * np.log(p.values + 1e-9), 0.0).sum(axis=1))
        entropy = pd.Series(entropy_raw / np.log(max(len(prob_cols), 2)), index=df.index).clip(0, 1)

    if "role_confidence" in df.columns:
        confidence = pd.to_numeric(df["role_confidence"], errors="coerce").fillna(0.5)
        if confidence.max() > 1.0:
            confidence = (confidence / 100.0).clip(0, 1)
        confidence_term = 1.0 - confidence
        return (0.7 * entropy + 0.3 * confidence_term).clip(0, 1)

    return entropy.clip(0, 1)


def _compute_scheme_independence(df: pd.DataFrame) -> pd.Series:
    """
    Component 4: Scheme Independence (20%).

    Players whose value doesn't depend on specific play calls or
    lineup configurations are more portable.

    Signals:
      a) scheme_stability_raw from Layer 4 (if available)
      b) Low variance in playtype surplus (consistent across play types)
      c) ORAPM/DRAPM balance (two-way players less context-sensitive)
    """
    result = pd.Series(0.5, index=df.index)
    n_signals = 0

    # (a) Scheme stability from Layer 4
    if "scheme_stability_raw" in df.columns:
        stability = df.groupby("season")["scheme_stability_raw"].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
        ).fillna(0.5)
        result = stability
        n_signals = 1

    # (b) Playtype surplus consistency
    surplus_cols = [c for c in df.columns if c.endswith("_surplus") and not c.endswith("_total")]
    if len(surplus_cols) >= 3:
        surplus_vals = df[surplus_cols].fillna(0)
        surplus_std = surplus_vals.std(axis=1)
        surplus_mean = surplus_vals.mean(axis=1).abs() + 1e-9
        cov = surplus_std / surplus_mean
        # Low CoV = consistent across playtypes = more portable
        consistency = df.groupby("season").apply(
            lambda g: 1.0 - (cov.loc[g.index] - cov.loc[g.index].min()) /
                      (cov.loc[g.index].max() - cov.loc[g.index].min() + 1e-9)
        )
        if hasattr(consistency, 'droplevel'):
            consistency = consistency.droplevel(0)
        consistency = consistency.reindex(df.index, fill_value=0.5)

        if n_signals == 0:
            result = consistency
        else:
            result = 0.6 * result + 0.4 * consistency
        n_signals += 1

    # (c) Two-way RAPM balance
    if "orapm" in df.columns and "drapm" in df.columns:
        orapm_z = df.groupby("season")["orapm"].transform(
            lambda x: compute_z_score(x, winsorize=3.5)
        ).fillna(0)
        drapm_z = df.groupby("season")["drapm"].transform(
            lambda x: compute_z_score(x, winsorize=3.5)
        ).fillna(0)

        total = orapm_z.abs() + drapm_z.abs() + 1e-9
        imbalance = (orapm_z - drapm_z).abs() / total
        two_way = 1.0 - imbalance

        if n_signals == 0:
            result = two_way.clip(0, 1)
        else:
            weight = 0.35 / n_signals
            result = (1 - weight) * result + weight * two_way.clip(0, 1)

    transfer = _compute_archetype_transfer(df)
    return (0.8 * result.clip(0, 1) + 0.2 * transfer).clip(0, 1)


def compute_portability_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Structural Portability Index (v2.7).

    v2.7 measures structural skill portability across team contexts:

    Components:
      1. Dimensional Breadth (35%) — How many skills above average
      2. Universal Skill Presence (25%) — Shooting + playmaking + creation
      3. Two-Way Balance (20%) — Offense + defense balance
      4. Scheme Independence (20%) — Low system dependence

    This is STRUCTURAL, not COMPOSITIONAL. It measures breadth and
    transferability of skills, not the magnitude of impact.
    """
    result = df.copy()
    cfg = PORTABILITY

    qualified_mask = (result["qualified"] == True) if "qualified" in result.columns else pd.Series(True, index=result.index)

    # Compute 4 portability components
    print("    Computing dimensional breadth...")
    dimensional_breadth = _compute_dimensional_breadth(result)

    print("    Computing universal skill presence...")
    universal_skill = _compute_universal_skill_presence(result)

    print("    Computing two-way balance...")
    two_way_balance = _compute_two_way_balance(result)

    print("    Computing archetype transfer...")
    archetype_transfer = _compute_archetype_transfer(result)

    print("    Computing scheme independence...")
    scheme_independence = _compute_scheme_independence(result)

    # Store individual components
    result["portability_dimensional_breadth"] = np.nan
    result["portability_universal_skill"] = np.nan
    result["portability_two_way_balance"] = np.nan
    result["portability_scheme_independence"] = np.nan
    result["portability_archetype_transfer"] = np.nan

    result.loc[qualified_mask, "portability_dimensional_breadth"] = dimensional_breadth.loc[qualified_mask]
    result.loc[qualified_mask, "portability_universal_skill"] = universal_skill.loc[qualified_mask]
    result.loc[qualified_mask, "portability_two_way_balance"] = two_way_balance.loc[qualified_mask]
    result.loc[qualified_mask, "portability_scheme_independence"] = scheme_independence.loc[qualified_mask]
    result.loc[qualified_mask, "portability_archetype_transfer"] = archetype_transfer.loc[qualified_mask]

    # Weighted composite
    portability_raw = pd.Series(np.nan, index=result.index)
    portability_raw.loc[qualified_mask] = (
        cfg.w_dimensional_breadth * dimensional_breadth.loc[qualified_mask] +
        cfg.w_universal_skill * universal_skill.loc[qualified_mask] +
        cfg.w_two_way_balance * two_way_balance.loc[qualified_mask] +
        cfg.w_scheme_independence * scheme_independence.loc[qualified_mask]
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
            "dimensional_breadth": _safe_float(row.get("portability_dimensional_breadth")),
            "universal_skill": _safe_float(row.get("portability_universal_skill")),
            "two_way_balance": _safe_float(row.get("portability_two_way_balance")),
            "scheme_independence": _safe_float(row.get("portability_scheme_independence")),
        },
        # v2.5: portability_ratio removed from public card (Fix #4)
        # Retained internally as portability_index alias for diagnostics
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
    Generate a summary report of the v2.7 decomposition run.
    """
    qualified = df[df.get("qualified", True) == True] if "qualified" in df.columns else df

    report = {
        "version": "2.8",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_players": int(len(df)),
        "qualified_players": int(len(qualified)),
        "seasons": sorted(df["season"].unique().tolist()),
        "metrics_summary": {},
        "tier_distribution": {},
        "portability_distribution": {},
        "top_10_total_impact": [],
        "top_10_portable_talent": [],
        "top_10_portability_index": [],
    }

    # Metrics summary
    for metric in [
        "portable_talent_score", "role_dependent_impact_score",
        "total_impact_score", "portability_index",
        "portable_talent_transformed", "role_dependent_impact_transformed",
        "total_impact_transformed",
    ]:
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

    if "portability_index" in qualified.columns:
        top10_port = qualified.nlargest(10, "portability_index")
        for _, row in top10_port.iterrows():
            report["top_10_portability_index"].append({
                "player": str(row.get(name_col, row["player_id"])),
                "season": str(row["season"]),
                "index": round(float(row.get("portability_index", 0)), 3),
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


def generate_distribution_diagnostics(df: pd.DataFrame) -> Dict:
    qualified = df[df.get("qualified", True) == True] if "qualified" in df.columns else df

    layer_cols = [
        "dimension_model_z", "portable_talent_z_adj", "role_utilization_raw_z",
        "elevation_z", "scheme_stability_z", "role_dependent_impact_z", "total_impact_z",
        "portable_talent_transformed", "role_dependent_impact_transformed", "total_impact_transformed",
    ]
    layer_stats = {}
    tail_stats = {}
    for col in layer_cols:
        if col in qualified.columns:
            layer_stats[col] = _series_distribution_stats(qualified[col])
            tail_stats[col] = _tail_separation(qualified[col])

    checks = [
        ("dimension_model_z", "portable_talent_z_adj"),
        ("portable_talent_z_adj", "total_impact_z"),
        ("role_utilization_raw_z", "role_dependent_impact_z"),
    ]
    compression = []
    for src, dst in checks:
        if src in layer_stats and dst in layer_stats:
            src_std = layer_stats[src].get("std")
            dst_std = layer_stats[dst].get("std")
            ratio = None
            flagged = False
            if src_std and dst_std is not None and src_std > 0:
                ratio = float(dst_std / src_std)
                flagged = ratio < DISTRIBUTION_INTEGRITY.compression_warning_ratio
            compression.append({
                "source": src,
                "target": dst,
                "std_ratio": ratio,
                "compression_flag": flagged,
            })

    obke_dbke = {}
    if "offensive_portable_z" in qualified.columns and "defensive_portable_z" in qualified.columns:
        off = pd.to_numeric(qualified["offensive_portable_z"], errors="coerce")
        deff = pd.to_numeric(qualified["defensive_portable_z"], errors="coerce")
        corr = off.corr(deff)
        std_off = float(off.std()) if off.notna().any() else None
        std_def = float(deff.std()) if deff.notna().any() else None
        obke_dbke = {
            "correlation": float(corr) if corr is not None and not np.isnan(corr) else None,
            "off_std": std_off,
            "def_std": std_def,
            "std_ratio_off_over_def": float(std_off / std_def) if std_off and std_def and std_def > 0 else None,
        }

    dim_cols = [c for c in qualified.columns if c.startswith("dim_") and c.endswith("_z") and "_raw" not in c]
    dim_corr = {}
    if dim_cols:
        corr_mat = qualified[dim_cols].corr().fillna(0.0)
        for col in corr_mat.columns:
            dim_corr[col] = {k: float(v) for k, v in corr_mat.loc[col].to_dict().items()}

    return {
        "version": "2.8",
        "layer_distribution_stats": layer_stats,
        "tail_separation": tail_stats,
        "compression_report": compression,
        "obke_dbke_balance": obke_dbke,
        "dimension_correlation_matrix": dim_corr,
    }


def build_v28_score_artifacts(df: pd.DataFrame) -> Dict[str, Dict]:
    qualified = df[df.get("qualified", True) == True] if "qualified" in df.columns else df
    name_col = "player_name" if "player_name" in qualified.columns else "player_id"

    dim_cols = [c for c in qualified.columns if c.startswith("dim_") and c.endswith("_z") and "_raw" not in c]
    layer_cols = [
        "portable_talent_z_adj", "role_utilization_raw_z", "elevation_z",
        "scheme_stability_z", "role_dependent_impact_z", "total_impact_z",
        "portable_talent_transformed", "role_dependent_impact_transformed", "total_impact_transformed",
    ]
    obke_cols = ["offensive_portable_z", "defensive_portable_z", "portable_talent_z_adj"]

    dimension_scores = {"version": "2.8", "players": {}}
    layer_scores = {"version": "2.8", "players": {}}
    obke_dbke_scores = {"version": "2.8", "players": {}}

    for _, row in qualified.iterrows():
        key = f"{row.get(name_col, row.get('player_id', 'Unknown'))} ({row.get('season', '')}) #{row.get('player_id', '')}"
        dimension_scores["players"][key] = {
            "player_id": str(row.get("player_id", "")),
            "season": str(row.get("season", "")),
            **{c: _safe_float(row.get(c)) for c in dim_cols},
        }
        layer_scores["players"][key] = {
            "player_id": str(row.get("player_id", "")),
            "season": str(row.get("season", "")),
            **{c: _safe_float(row.get(c)) for c in layer_cols if c in qualified.columns},
        }
        obke_dbke_scores["players"][key] = {
            "player_id": str(row.get("player_id", "")),
            "season": str(row.get("season", "")),
            **{c: _safe_float(row.get(c)) for c in obke_cols if c in qualified.columns},
            "obke_transformed": _safe_float(_logistic_transform(pd.Series([row.get("offensive_portable_z")]), DISTRIBUTION_INTEGRITY.logistic_alpha).iloc[0]) if "offensive_portable_z" in qualified.columns else None,
            "dbke_transformed": _safe_float(_logistic_transform(pd.Series([row.get("defensive_portable_z")]), DISTRIBUTION_INTEGRITY.logistic_alpha).iloc[0]) if "defensive_portable_z" in qualified.columns else None,
        }

    return {
        "dimension_scores": dimension_scores,
        "layer_scores": layer_scores,
        "obke_dbke_scores": obke_dbke_scores,
    }


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def run_full_decomposition(
    seasons: Optional[List[str]] = None,
    use_possession_data: bool = False,
    save_output: bool = True,
) -> pd.DataFrame:
    """
    Run the complete v2.8 BKE Impact Decomposition Engine.

        v2.7 enhancements over v2.6:
            - Soft archetype membership conditioning across role-dependent layers
            - Full conditional neutralization in Layer 1C (location + scale)
            - RUE optimal-source decoupling from RAPM by config
            - Soft-membership archetype transfer in portability signal
            - Dimension composite variance restoration hook

    Pipeline order:
      Layer 1: Portable Talent (8-dim model, z-score + neutralization)
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
    print("  BKE v2.8 — PORTABLE TALENT vs ROLE-DEPENDENT IMPACT ENGINE")
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

    print("  Computing total impact (v2.8 transformed + anchored)...")
    df = compute_total_impact(df)

    print("  Computing portability index (v2.7 structural)...")
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
    print("  BKE v2.8 — DECOMPOSITION COMPLETE")
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
                port = row.get("portability_index", 0)
                print(f"    {row.get(name_col, row['player_id']):25s} | "
                      f"Impact: {row['total_impact_score']:5.1f} | "
                      f"PTS: {row.get('portable_talent_score', 0):5.1f} | "
                      f"Port: {port:.2f} | "
                      f"{row.get('primary_archetype', 'N/A')}")

        print("\n  Top 10 Most Portable:")
        if "portability_index" in qualified.columns:
            top_port = qualified.nlargest(10, "portability_index")
            for _, row in top_port.iterrows():
                print(f"    {row.get(name_col, row['player_id']):25s} | "
                      f"Portability: {row['portability_index']:.3f} | "
                      f"{row.get('portability_class', 'N/A')}")

    # Save outputs
    if save_output:
        print("\n  Saving outputs...")
        df.to_parquet(BKE_V28_OUTPUT_PARQUET, index=False)
        df.to_csv(BKE_V28_OUTPUT_CSV, index=False)
        print(f"  Saved: {BKE_V28_OUTPUT_PARQUET}")
        print(f"  Saved: {BKE_V28_OUTPUT_CSV}")

        report = generate_report(df)
        with open(BKE_V28_REPORT_JSON, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"  Saved: {BKE_V28_REPORT_JSON}")

        diagnostics = generate_distribution_diagnostics(df)
        with open(BKE_V28_VARIANCE_REPORT_JSON, "w") as f:
            json.dump(diagnostics.get("layer_distribution_stats", {}), f, indent=2, default=str)
        with open(BKE_V28_COMPRESSION_REPORT_JSON, "w") as f:
            json.dump({
                "compression_report": diagnostics.get("compression_report", []),
                "tail_separation": diagnostics.get("tail_separation", {}),
                "obke_dbke_balance": diagnostics.get("obke_dbke_balance", {}),
                "dimension_correlation_matrix": diagnostics.get("dimension_correlation_matrix", {}),
            }, f, indent=2, default=str)
        print(f"  Saved: {BKE_V28_VARIANCE_REPORT_JSON}")
        print(f"  Saved: {BKE_V28_COMPRESSION_REPORT_JSON}")

        artifacts = build_v28_score_artifacts(df)
        with open(BKE_V28_DIMENSION_SCORES_JSON, "w") as f:
            json.dump(artifacts["dimension_scores"], f, indent=2, default=str)
        with open(BKE_V28_LAYER_SCORES_JSON, "w") as f:
            json.dump(artifacts["layer_scores"], f, indent=2, default=str)
        with open(BKE_V28_OBKE_DBKE_SCORES_JSON, "w") as f:
            json.dump(artifacts["obke_dbke_scores"], f, indent=2, default=str)
        print(f"  Saved: {BKE_V28_DIMENSION_SCORES_JSON}")
        print(f"  Saved: {BKE_V28_LAYER_SCORES_JSON}")
        print(f"  Saved: {BKE_V28_OBKE_DBKE_SCORES_JSON}")

    return df


# ---------------------------------------------------------------------------
# CLI Entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BKE v2.8 Decomposition Engine")
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
