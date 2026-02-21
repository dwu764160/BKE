"""
tests/bke_v29_diagnostics.py
=============================================================================
BKE v2.9 — Central Diagnostic Suite

Runs 8 diagnostic domains across the decomposition data:
  1) OBKE/DBKE Variance Asymmetry Audit
  2) Defensive Signal Quality Audit
  3) Counting Stats vs On/Off Dominance
  4) Variance Anchor Stress Test
  5) Offense Weight Bias Experiment
  6) Archetype Coefficient Audit
  7) Portability vs Role-Dependent Drag
  8) Rank Movement Driver Decomposition

Output:
  reports/bke_v29_diagnostic_master.json

No weights or structural changes — measurement only.
=============================================================================
"""

import json
import os
import sys
import time
from collections import OrderedDict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.modeling.model_config import (
    BKE_V28_OUTPUT_PARQUET,
    REPORTS_DIR,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sf(val):
    """Safely convert to JSON-friendly float."""
    if val is None:
        return None
    try:
        v = float(val)
    except (TypeError, ValueError):
        return None
    if np.isnan(v) or np.isinf(v):
        return None
    return round(v, 6)


def _zscore(s: pd.Series) -> pd.Series:
    vals = pd.to_numeric(s, errors="coerce")
    std = vals.std(ddof=0)
    if pd.isna(std) or std < 1e-9:
        return pd.Series(0.0, index=s.index)
    return (vals - vals.mean()) / std


def _signed_log1p(s: pd.Series) -> pd.Series:
    vals = pd.to_numeric(s, errors="coerce").fillna(0.0)
    return np.sign(vals) * np.log1p(np.abs(vals))


def _pct_rank(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").rank(pct=True, method="average") * 100.0


def _partial_corr(x, y, covariates):
    """Partial correlation of x with y, controlling for covariates.
    Returns (r, p-value)."""
    data = pd.DataFrame({"x": x, "y": y})
    for i, cv in enumerate(covariates):
        data[f"cv{i}"] = cv
    data = data.dropna()
    if len(data) < 10:
        return (None, None)
    # Residualize x and y on covariates
    cov_cols = [c for c in data.columns if c.startswith("cv")]
    from numpy.linalg import lstsq
    C = data[cov_cols].values
    C = np.column_stack([C, np.ones(len(C))])
    # Residualize x
    coef_x, _, _, _ = lstsq(C, data["x"].values, rcond=None)
    res_x = data["x"].values - C @ coef_x
    # Residualize y
    coef_y, _, _, _ = lstsq(C, data["y"].values, rcond=None)
    res_y = data["y"].values - C @ coef_y
    r, p = _pearsonr(res_x, res_y)
    return (r, p)


def _pearsonr(x, y):
    """Pure numpy Pearson r with two-sided p-value (t-distribution approx)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 3:
        return (np.nan, np.nan)
    mx, my = x.mean(), y.mean()
    dx, dy = x - mx, y - my
    denom = np.sqrt((dx ** 2).sum() * (dy ** 2).sum())
    if denom < 1e-15:
        return (0.0, 1.0)
    r = (dx * dy).sum() / denom
    r = max(-1.0, min(1.0, r))
    # t-statistic for two-sided test
    if abs(r) == 1.0:
        return (r, 0.0)
    t_stat = r * np.sqrt((n - 2) / (1 - r ** 2))
    # Approximate p-value using normal for large n
    from math import erfc, sqrt
    p = erfc(abs(t_stat) / sqrt(2))
    return (r, p)


def _safe_corr(a, b):
    """Pearson correlation, dropping NaN pairs. Returns (r, p, n)."""
    mask = a.notna() & b.notna()
    n = int(mask.sum())
    if n < 10:
        return (None, None, n)
    r, p = _pearsonr(a[mask].values, b[mask].values)
    return (_sf(r), _sf(p), n)


# ---------------------------------------------------------------------------
# OBKE/DBKE Reconstruction (mirrors construct_bke_scores_v27.py)
# ---------------------------------------------------------------------------

def _reconstruct_obke_dbke(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct OBKE/DBKE/BKE from decomposition columns."""
    out = df.copy()
    off_p = pd.to_numeric(out["offensive_portable_z"], errors="coerce").fillna(0.0)
    def_p = pd.to_numeric(out["defensive_portable_z"], errors="coerce").fillna(0.0)

    if "role_utilization_raw_z" in out.columns:
        rue_z = pd.to_numeric(out["role_utilization_raw_z"], errors="coerce").fillna(0.0)
    elif "role_utilization_raw" in out.columns:
        rue_z = _zscore(out["role_utilization_raw"]).fillna(0.0)
    else:
        rue_z = pd.Series(0.0, index=out.index)

    off_elev_z = _zscore(out["elevation_orapm"]).fillna(0.0)
    def_elev_z = _zscore(out["elevation_drapm"]).fillna(0.0)

    scheme_z = pd.to_numeric(out.get("scheme_stability_z", pd.Series(0.0, index=out.index)),
                             errors="coerce").fillna(0.0).clip(lower=0.0)

    out["OBKE_raw"] = 0.55 * off_p + 0.25 * rue_z + 0.20 * off_elev_z
    out["DBKE_raw"] = 0.60 * def_p + 0.25 * def_elev_z + 0.15 * scheme_z
    out["BKE_raw"] = out["OBKE_raw"] + out["DBKE_raw"]

    # For layer breakdown
    out["layer1_off"] = off_p
    out["layer1_def"] = def_p
    out["layer2_rue"] = rue_z
    out["layer3_off_elev"] = off_elev_z
    out["layer3_def_elev"] = def_elev_z
    out["layer4_scheme"] = scheme_z

    return out


# ===================================================================
# Domain 1: OBKE / DBKE Variance Asymmetry Audit
# ===================================================================

def domain_1_variance_asymmetry(df: pd.DataFrame) -> dict:
    """Tests 1.1, 1.2, 1.3"""
    result = {}

    # --- Test 1.1: Raw Variance Decomposition ---
    var_o = df["OBKE_raw"].var(ddof=0)
    var_d = df["DBKE_raw"].var(ddof=0)
    cov_od = df[["OBKE_raw", "DBKE_raw"]].cov().iloc[0, 1]
    var_bke = df["BKE_raw"].var(ddof=0)
    total_decomp = var_o + var_d + 2 * cov_od

    off_share = var_o / total_decomp if total_decomp > 0 else None
    def_share = var_d / total_decomp if total_decomp > 0 else None
    cov_share = (2 * cov_od) / total_decomp if total_decomp > 0 else None

    result["test_1_1_raw_variance_decomposition"] = {
        "var_OBKE": _sf(var_o),
        "var_DBKE": _sf(var_d),
        "var_BKE": _sf(var_bke),
        "cov_OBKE_DBKE": _sf(cov_od),
        "total_decomp_check": _sf(total_decomp),
        "off_variance_share": _sf(off_share),
        "def_variance_share": _sf(def_share),
        "covariance_share": _sf(cov_share),
        "defense_gt_60pct": bool(def_share is not None and def_share > 0.60),
        "verdict": "ASYMMETRY_CONFIRMED" if (def_share is not None and def_share > 0.60) else "SYMMETRIC_OR_OFFENSE_DOMINANT",
    }

    # --- Test 1.2: Z-Standardized Symmetry Simulation ---
    df["OBKE_z"] = _zscore(df["OBKE_raw"])
    df["DBKE_z"] = _zscore(df["DBKE_raw"])
    df["BKE_equal_var"] = df["OBKE_z"] + df["DBKE_z"]

    df["rank_actual"] = df["BKE_raw"].rank(ascending=False, method="min").astype(int)
    df["rank_equal_var"] = df["BKE_equal_var"].rank(ascending=False, method="min").astype(int)
    df["rank_shift_equal_var"] = df["rank_actual"] - df["rank_equal_var"]

    mean_abs_shift = df["rank_shift_equal_var"].abs().mean()
    median_abs_shift = df["rank_shift_equal_var"].abs().median()

    # Top 20 biggest rank shifts
    top_shifts = (
        df.nlargest(20, "rank_shift_equal_var", keep="first")
        [["player_name", "season", "rank_actual", "rank_equal_var", "rank_shift_equal_var",
          "OBKE_raw", "DBKE_raw"]]
        .to_dict(orient="records")
    )
    for rec in top_shifts:
        for k in rec:
            if isinstance(rec[k], (np.floating, float)):
                rec[k] = _sf(rec[k])
            elif isinstance(rec[k], (np.integer,)):
                rec[k] = int(rec[k])

    # Offensive specialist shifts (top 25% OBKE, bottom 50% DBKE)
    obke_75 = df["OBKE_raw"].quantile(0.75)
    dbke_50 = df["DBKE_raw"].quantile(0.50)
    off_specialists = df[(df["OBKE_raw"] >= obke_75) & (df["DBKE_raw"] <= dbke_50)]
    mean_shift_off_specialists = off_specialists["rank_shift_equal_var"].mean() if len(off_specialists) > 0 else None

    result["test_1_2_z_symmetry_simulation"] = {
        "mean_abs_rank_shift": _sf(mean_abs_shift),
        "median_abs_rank_shift": _sf(median_abs_shift),
        "mean_shift_off_specialists": _sf(mean_shift_off_specialists),
        "off_specialist_count": int(len(off_specialists)),
        "top_20_rank_shifts": top_shifts,
    }

    # --- Test 1.3: Tail Sensitivity ---
    top5_obke = df["OBKE_raw"].quantile(0.95)
    top5_dbke = df["DBKE_raw"].quantile(0.95)
    bot5_dbke = df["DBKE_raw"].quantile(0.05)

    top5_obke_var = df[df["OBKE_raw"] >= top5_obke]["OBKE_raw"].var(ddof=0)
    top5_dbke_var = df[df["DBKE_raw"] >= top5_dbke]["DBKE_raw"].var(ddof=0)
    bot5_dbke_penalty_mag = df[df["DBKE_raw"] <= bot5_dbke]["DBKE_raw"].mean()

    result["test_1_3_tail_sensitivity"] = {
        "top_5pct_OBKE_variance": _sf(top5_obke_var),
        "top_5pct_DBKE_variance": _sf(top5_dbke_var),
        "bottom_5pct_DBKE_mean_penalty": _sf(bot5_dbke_penalty_mag),
        "tail_variance_ratio_def_to_off": _sf(top5_dbke_var / top5_obke_var) if top5_obke_var and top5_obke_var > 0 else None,
    }

    return result


# ===================================================================
# Domain 2: Defensive Signal Quality Audit
# ===================================================================

def domain_2_defensive_signal(df: pd.DataFrame) -> dict:
    result = {}

    drapm = pd.to_numeric(df["drapm"], errors="coerce")
    dbke = df["DBKE_raw"]

    # --- Test 2.1: DBKE vs Ground Truth Signals ---
    corr_dbke_drapm = _safe_corr(dbke, drapm)
    corr_dbke_drtg = _safe_corr(dbke, pd.to_numeric(df.get("DRTG", pd.Series(dtype=float)), errors="coerce"))
    corr_defplay_drapm = _safe_corr(
        pd.to_numeric(df.get("dim_defensive_playmaking_z", pd.Series(dtype=float)), errors="coerce"),
        drapm
    )
    corr_defimp_drapm = _safe_corr(
        pd.to_numeric(df.get("dim_defensive_impact_z", pd.Series(dtype=float)), errors="coerce"),
        drapm
    )
    corr_netrtg = _safe_corr(dbke, pd.to_numeric(df.get("NET_RTG", pd.Series(dtype=float)), errors="coerce"))

    result["test_2_1_dbke_vs_ground_truth"] = {
        "corr_DBKE_DRAPM": {"r": corr_dbke_drapm[0], "p": corr_dbke_drapm[1], "n": corr_dbke_drapm[2]},
        "corr_DBKE_DRTG": {"r": corr_dbke_drtg[0], "p": corr_dbke_drtg[1], "n": corr_dbke_drtg[2]},
        "corr_def_playmaking_DRAPM": {"r": corr_defplay_drapm[0], "p": corr_defplay_drapm[1], "n": corr_defplay_drapm[2]},
        "corr_def_impact_DRAPM": {"r": corr_defimp_drapm[0], "p": corr_defimp_drapm[1], "n": corr_defimp_drapm[2]},
        "corr_DBKE_NET_RTG": {"r": corr_netrtg[0], "p": corr_netrtg[1], "n": corr_netrtg[2]},
        "counting_noise_risk": (
            "HIGH" if (corr_defplay_drapm[0] is not None and corr_defimp_drapm[0] is not None
                       and abs(corr_defplay_drapm[0] - corr_defimp_drapm[0]) < 0.05)
            else "LOW"
        ),
    }

    # --- Test 2.2: Partial Correlation (DBKE ~ DRAPM | STL, BLK) ---
    stl = pd.to_numeric(df.get("STL", pd.Series(dtype=float)), errors="coerce")
    blk = pd.to_numeric(df.get("BLK", pd.Series(dtype=float)), errors="coerce")
    stl_per100 = pd.to_numeric(df.get("STL_PER100_DEF_POSS", pd.Series(dtype=float)), errors="coerce")
    blk_pct = pd.to_numeric(df.get("BLK_PCT", pd.Series(dtype=float)), errors="coerce")

    mask_rd = dbke.notna() & drapm.notna()
    if mask_rd.sum() >= 10:
        raw_r, raw_p = _pearsonr(dbke[mask_rd].values, drapm[mask_rd].values)
    else:
        raw_r, raw_p = None, None

    partial_r, partial_p = _partial_corr(dbke, drapm, [stl, blk])
    partial_r_rate, partial_p_rate = _partial_corr(dbke, drapm, [stl_per100, blk_pct])

    result["test_2_2_partial_correlation"] = {
        "raw_corr_DBKE_DRAPM": _sf(raw_r),
        "partial_corr_DBKE_DRAPM_ctrl_STL_BLK": _sf(partial_r),
        "partial_pvalue": _sf(partial_p),
        "partial_corr_DBKE_DRAPM_ctrl_STLRATE_BLKPCT": _sf(partial_r_rate),
        "partial_pvalue_rate": _sf(partial_p_rate),
        "counting_stats_diluting": (
            "YES" if (partial_r is not None and raw_r is not None and partial_r > raw_r + 0.02)
            else "NO"
        ),
    }

    # --- Test 2.3: Defensive Component Variance Contribution ---
    def_portable = pd.to_numeric(df["defensive_portable_z"], errors="coerce")
    def_elevation = pd.to_numeric(df.get("elevation_drapm", pd.Series(dtype=float)), errors="coerce")
    scheme_bonus = pd.to_numeric(df.get("scheme_stability_z", pd.Series(dtype=float)), errors="coerce").clip(lower=0.0)

    var_def_portable = def_portable.var(ddof=0)
    var_def_elevation = def_elevation.var(ddof=0) if def_elevation.notna().sum() > 0 else 0
    var_scheme_bonus = scheme_bonus.var(ddof=0) if scheme_bonus.notna().sum() > 0 else 0
    total_def_var = var_def_portable + var_def_elevation + var_scheme_bonus

    # Sub-component breakdown of defensive portable
    dim_def_playmaking = pd.to_numeric(df.get("dim_defensive_playmaking_z", pd.Series(dtype=float)), errors="coerce")
    dim_def_impact = pd.to_numeric(df.get("dim_defensive_impact_z", pd.Series(dtype=float)), errors="coerce")
    dim_def_versatility = pd.to_numeric(df.get("dim_defensive_versatility_z", pd.Series(dtype=float)), errors="coerce")
    dim_extra_poss_def = pd.to_numeric(df.get("dim_extra_poss_defensive_z", pd.Series(dtype=float)), errors="coerce")

    var_defplay = dim_def_playmaking.var(ddof=0) if dim_def_playmaking.notna().sum() > 0 else 0
    var_defimp = dim_def_impact.var(ddof=0) if dim_def_impact.notna().sum() > 0 else 0
    var_defvers = dim_def_versatility.var(ddof=0) if dim_def_versatility.notna().sum() > 0 else 0
    var_extposs_d = dim_extra_poss_def.var(ddof=0) if dim_extra_poss_def.notna().sum() > 0 else 0

    result["test_2_3_defensive_component_variance"] = {
        "var_def_portable": _sf(var_def_portable),
        "var_def_elevation": _sf(var_def_elevation),
        "var_scheme_bonus": _sf(var_scheme_bonus),
        "total_def_component_variance": _sf(total_def_var),
        "share_def_portable": _sf(var_def_portable / total_def_var) if total_def_var > 0 else None,
        "share_def_elevation": _sf(var_def_elevation / total_def_var) if total_def_var > 0 else None,
        "share_scheme_bonus": _sf(var_scheme_bonus / total_def_var) if total_def_var > 0 else None,
        "sub_components": {
            "var_def_playmaking": _sf(var_defplay),
            "var_def_impact": _sf(var_defimp),
            "var_def_versatility": _sf(var_defvers),
            "var_extra_poss_def": _sf(var_extposs_d),
        },
        "structural_misalignment": (
            "YES" if (var_defplay > 0 and var_defimp > 0 and var_defplay > var_defimp)
            else "NO"
        ),
    }

    return result


# ===================================================================
# Domain 3: Counting Stats vs On/Off Dominance
# ===================================================================

def domain_3_counting_vs_onoff(df: pd.DataFrame) -> dict:
    result = {}

    obke = df["OBKE_raw"]

    # --- Test 3.1: Offensive Decomposition ---
    pts = pd.to_numeric(df.get("PTS", pd.Series(dtype=float)), errors="coerce")
    pts_per36 = pd.to_numeric(df.get("PTS_per36", pd.Series(dtype=float)), errors="coerce")
    orapm = pd.to_numeric(df.get("orapm", pd.Series(dtype=float)), errors="coerce")
    ts_pct = pd.to_numeric(df.get("TS_PCT", pd.Series(dtype=float)), errors="coerce")
    usg = pd.to_numeric(df.get("USG_RATE", pd.Series(dtype=float)), errors="coerce")
    ortg = pd.to_numeric(df.get("ORTG", pd.Series(dtype=float)), errors="coerce")
    net_rtg = pd.to_numeric(df.get("NET_RTG", pd.Series(dtype=float)), errors="coerce")

    # Volume proxy: PTS * GP (total points in season)
    gp = pd.to_numeric(df.get("GP", pd.Series(dtype=float)), errors="coerce")
    total_pts = pts * gp

    corrs = {}
    for label, series in [
        ("PTS", pts), ("PTS_per36", pts_per36), ("total_PTS_season", total_pts),
        ("orapm", orapm), ("ORTG", ortg), ("TS_PCT", ts_pct),
        ("USG_RATE", usg), ("NET_RTG", net_rtg),
    ]:
        r, p, n = _safe_corr(obke, series)
        corrs[f"corr_OBKE_{label}"] = {"r": r, "p": p, "n": n}

    result["test_3_1_offensive_decomposition"] = {
        **corrs,
        "volume_vs_rate_note": (
            "RATE_DOMINANT" if (
                corrs["corr_OBKE_PTS_per36"]["r"] is not None and
                corrs["corr_OBKE_total_PTS_season"]["r"] is not None and
                abs(corrs["corr_OBKE_PTS_per36"]["r"]) > abs(corrs["corr_OBKE_total_PTS_season"]["r"]) + 0.05
            ) else "BALANCED_OR_VOLUME_LEANING"
        ),
    }

    # --- Test 3.2: Volume Sensitivity Simulation ---
    dim_self = pd.to_numeric(df.get("dim_self_creation_z", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    # Create volume-adjusted self-creation: scale by total points proxy
    total_pts_z = _zscore(total_pts.fillna(0.0))
    ts_z = _zscore(ts_pct.fillna(0.0))
    volume_self_creation = total_pts_z * 0.5 + ts_z * 0.5  # volume-efficiency blend

    # Recompute offensive_portable with the volume-adjusted dimension
    # This is approximate: we replace just self_creation contribution
    off_p_original = pd.to_numeric(df["offensive_portable_z"], errors="coerce").fillna(0.0)
    # Delta = new_dim - old_dim (self creation's approximate weight in the composite)
    delta = (volume_self_creation - dim_self) * 0.10  # approximate weight of self_creation in offensive composite
    off_p_volume = off_p_original + delta

    obke_volume = 0.55 * off_p_volume + 0.25 * df["layer2_rue"] + 0.20 * df["layer3_off_elev"]
    bke_volume = obke_volume + df["DBKE_raw"]

    df["rank_volume_sim"] = bke_volume.rank(ascending=False, method="min").astype(int)
    rank_shift_volume = (df["rank_actual"] - df["rank_volume_sim"]).abs()

    result["test_3_2_volume_sensitivity_simulation"] = {
        "mean_abs_rank_shift": _sf(rank_shift_volume.mean()),
        "median_abs_rank_shift": _sf(rank_shift_volume.median()),
        "max_rank_shift": int(rank_shift_volume.max()) if len(rank_shift_volume) > 0 else None,
        "volume_sensitivity_index": _sf(rank_shift_volume.mean() / len(df) * 100),
    }

    # --- Test 3.3: Counting Stats Sensitivity (Defense) ---
    dbke = df["DBKE_raw"]
    drapm = pd.to_numeric(df["drapm"], errors="coerce")
    stl = pd.to_numeric(df.get("STL", pd.Series(dtype=float)), errors="coerce")
    blk = pd.to_numeric(df.get("BLK", pd.Series(dtype=float)), errors="coerce")
    on_off = pd.to_numeric(df.get("on_off_diff", pd.Series(dtype=float)), errors="coerce")

    reg_data = pd.DataFrame({
        "DBKE": dbke, "STL": stl, "BLK": blk, "DRAPM": drapm, "on_off": on_off
    }).dropna()

    reg_result = {}
    if len(reg_data) >= 30:
        # Standardize all columns
        for col in reg_data.columns:
            reg_data[col] = (reg_data[col] - reg_data[col].mean()) / (reg_data[col].std() + 1e-9)

        X = reg_data[["STL", "BLK", "DRAPM", "on_off"]].values
        y = reg_data["DBKE"].values
        X_aug = np.column_stack([X, np.ones(len(X))])
        try:
            coefs, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
            reg_result = {
                "beta_STL": _sf(coefs[0]),
                "beta_BLK": _sf(coefs[1]),
                "beta_DRAPM": _sf(coefs[2]),
                "beta_on_off": _sf(coefs[3]),
                "intercept": _sf(coefs[4]),
                "stl_overreliance": bool(abs(coefs[0]) >= abs(coefs[2]) * 0.8),
            }
        except Exception:
            reg_result = {"error": "regression_failed"}
    else:
        reg_result = {"error": "insufficient_data", "n": len(reg_data)}

    result["test_3_3_counting_stats_sensitivity_defense"] = reg_result

    return result


# ===================================================================
# Domain 4: Variance Anchor Stress Test
# ===================================================================

def domain_4_variance_anchor(df: pd.DataFrame) -> dict:
    result = {}

    # --- Test 4.1: Per-Season OBKE vs DBKE Std ---
    season_stats = []
    for season in sorted(df["season"].unique()):
        ssub = df[df["season"] == season]
        std_o = ssub["OBKE_raw"].std(ddof=0)
        std_d = ssub["DBKE_raw"].std(ddof=0)
        ratio = std_d / std_o if std_o > 0 else None
        season_stats.append({
            "season": season,
            "n": int(len(ssub)),
            "std_OBKE": _sf(std_o),
            "std_DBKE": _sf(std_d),
            "ratio_def_to_off": _sf(ratio),
        })

    stds = [s["std_DBKE"] for s in season_stats if s["std_DBKE"] is not None]
    max_std = max(stds) if stds else 0
    min_std = min(stds) if stds else 0
    spike = max_std / min_std if min_std > 0 else None

    result["test_4_1_per_season_std"] = {
        "seasons": season_stats,
        "anchor_leakage_ratio": _sf(spike),
        "anchor_leakage": "YES" if (spike is not None and spike > 1.5) else "NO",
    }

    # --- Test 4.2: Cross-Season Stability ---
    seasons = sorted(df["season"].unique())
    yoy_corrs = []
    if len(seasons) >= 2:
        for i in range(len(seasons) - 1):
            s1, s2 = seasons[i], seasons[i + 1]
            df1 = df[df["season"] == s1][["player_id", "OBKE_raw", "DBKE_raw"]].copy()
            df2 = df[df["season"] == s2][["player_id", "OBKE_raw", "DBKE_raw"]].copy()
            df1.columns = ["player_id", "OBKE_y1", "DBKE_y1"]
            df2.columns = ["player_id", "OBKE_y2", "DBKE_y2"]
            merged = df1.merge(df2, on="player_id", how="inner")
            if len(merged) >= 10:
                r_o, p_o = _pearsonr(merged["OBKE_y1"].values, merged["OBKE_y2"].values)
                r_d, p_d = _pearsonr(merged["DBKE_y1"].values, merged["DBKE_y2"].values)
                yoy_corrs.append({
                    "pair": f"{s1} → {s2}",
                    "n_matched": int(len(merged)),
                    "year_to_year_OBKE_corr": _sf(r_o),
                    "year_to_year_DBKE_corr": _sf(r_d),
                    "noise_asymmetry": (
                        "CONFIRMED" if (r_d < 0.5 and r_o > 0.7)
                        else "MODERATE" if (r_d < r_o - 0.1)
                        else "SYMMETRIC"
                    ),
                })

    result["test_4_2_cross_season_stability"] = {
        "pairs": yoy_corrs,
    }

    return result


# ===================================================================
# Domain 5: Offense Weight Bias Experiment
# ===================================================================

def domain_5_offense_bias(df: pd.DataFrame) -> dict:
    result = {}

    obke = df["OBKE_raw"]
    dbke = df["DBKE_raw"]

    # --- Test 5.1: Offense-Biased Composites ---
    bke_55_45 = 0.55 * obke + 0.45 * dbke
    bke_60_40 = 0.60 * obke + 0.40 * dbke
    bke_50_50 = obke + dbke  # actual current = 50/50 since just additive

    df["rank_55_45"] = bke_55_45.rank(ascending=False, method="min").astype(int)
    df["rank_60_40"] = bke_60_40.rank(ascending=False, method="min").astype(int)
    df["rank_shift_55_45"] = df["rank_actual"] - df["rank_55_45"]
    df["rank_shift_60_40"] = df["rank_actual"] - df["rank_60_40"]

    # Top 10 stability
    top10_actual = set(df.nsmallest(10, "rank_actual")["player_id"].tolist())
    top10_55_45 = set(df.nsmallest(10, "rank_55_45")["player_id"].tolist())
    top10_60_40 = set(df.nsmallest(10, "rank_60_40")["player_id"].tolist())

    # Offensive specialists (top 25% OBKE, bottom 50% DBKE)
    obke_75 = obke.quantile(0.75)
    dbke_50 = dbke.quantile(0.50)
    off_specialists = df[(obke >= obke_75) & (dbke <= dbke_50)]

    # Two-way players (top 40% both)
    obke_60 = obke.quantile(0.60)
    dbke_60 = dbke.quantile(0.60)
    two_way = df[(obke >= obke_60) & (dbke >= dbke_60)]

    result["test_5_1_offense_biased_composite"] = {
        "mean_abs_rank_shift_55_45": _sf(df["rank_shift_55_45"].abs().mean()),
        "mean_abs_rank_shift_60_40": _sf(df["rank_shift_60_40"].abs().mean()),
        "top10_overlap_55_45": int(len(top10_actual & top10_55_45)),
        "top10_overlap_60_40": int(len(top10_actual & top10_60_40)),
        "off_specialists_mean_shift_55_45": _sf(off_specialists["rank_shift_55_45"].mean()) if len(off_specialists) > 0 else None,
        "off_specialists_mean_shift_60_40": _sf(off_specialists["rank_shift_60_40"].mean()) if len(off_specialists) > 0 else None,
        "two_way_mean_shift_55_45": _sf(two_way["rank_shift_55_45"].mean()) if len(two_way) > 0 else None,
        "two_way_mean_shift_60_40": _sf(two_way["rank_shift_60_40"].mean()) if len(two_way) > 0 else None,
        "off_specialist_count": int(len(off_specialists)),
        "two_way_count": int(len(two_way)),
        "bias_sensitivity_curve": {
            "50_50_mean_abs_shift": 0.0,
            "55_45_mean_abs_shift": _sf(df["rank_shift_55_45"].abs().mean()),
            "60_40_mean_abs_shift": _sf(df["rank_shift_60_40"].abs().mean()),
        },
    }

    return result


# ===================================================================
# Domain 6: Archetype Coefficient Audit
# ===================================================================

def domain_6_archetype_audit(df: pd.DataFrame) -> dict:
    result = {}

    arch_col = "primary_archetype"
    if arch_col not in df.columns:
        return {"error": "primary_archetype column not found"}

    archetypes = df[arch_col].dropna().unique()

    # --- Test 6.1: Archetype Mean Component Values ---
    arch_means = []
    for arch in sorted(archetypes):
        sub = df[df[arch_col] == arch]
        entry = {
            "archetype": str(arch),
            "n": int(len(sub)),
            "mean_OBKE": _sf(sub["OBKE_raw"].mean()),
            "mean_DBKE": _sf(sub["DBKE_raw"].mean()),
            "mean_BKE": _sf(sub["BKE_raw"].mean()),
        }
        for dim in ["dim_defensive_playmaking_z", "dim_defensive_impact_z",
                     "dim_self_creation_z", "dim_shooting_gravity_z"]:
            if dim in sub.columns:
                entry[f"mean_{dim}"] = _sf(sub[dim].mean())
        arch_means.append(entry)

    # Check for systematic defensive inflation
    dbke_means = [a["mean_DBKE"] for a in arch_means if a["mean_DBKE"] is not None]
    obke_means = [a["mean_OBKE"] for a in arch_means if a["mean_OBKE"] is not None]
    dbke_range = max(dbke_means) - min(dbke_means) if dbke_means else 0
    obke_range = max(obke_means) - min(obke_means) if obke_means else 0

    result["test_6_1_archetype_mean_components"] = {
        "archetypes": arch_means,
        "dbke_range_across_archetypes": _sf(dbke_range),
        "obke_range_across_archetypes": _sf(obke_range),
        "defensive_inflation_risk": "HIGH" if dbke_range > obke_range * 1.5 else "LOW",
    }

    # --- Test 6.2: Weight Detection (report known weight structure) ---
    result["test_6_2_weight_detection"] = {
        "note": "All weights are centralized in model_config.py. No per-archetype multipliers exist in the pipeline.",
        "OBKE_weights": {
            "layer1_offensive": 0.55,
            "layer2_rue": 0.25,
            "layer3_off_elevation": 0.20,
        },
        "DBKE_weights": {
            "layer1_defensive": 0.60,
            "layer3_def_elevation": 0.25,
            "layer4_scheme_bonus": 0.15,
        },
        "conditioning": "soft_archetype_membership (probabilistic, no discrete boosts)",
        "hardcoded_archetype_multipliers_found": False,
    }

    # --- Test 6.3: Archetype Conditional Variance ---
    arch_var = []
    for arch in sorted(archetypes):
        sub = df[df[arch_col] == arch]
        if len(sub) < 5:
            continue
        arch_var.append({
            "archetype": str(arch),
            "n": int(len(sub)),
            "std_OBKE": _sf(sub["OBKE_raw"].std(ddof=0)),
            "std_DBKE": _sf(sub["DBKE_raw"].std(ddof=0)),
            "ratio_DBKE_to_OBKE": _sf(
                sub["DBKE_raw"].std(ddof=0) / sub["OBKE_raw"].std(ddof=0)
            ) if sub["OBKE_raw"].std(ddof=0) > 1e-9 else None,
        })

    # Check for structural artifact (any archetype with 2x DBKE variance)
    artifacts = [a for a in arch_var if a["ratio_DBKE_to_OBKE"] is not None and a["ratio_DBKE_to_OBKE"] >= 2.0]

    result["test_6_3_archetype_conditional_variance"] = {
        "archetypes": arch_var,
        "structural_artifact_count": len(artifacts),
        "structural_artifact_archetypes": [a["archetype"] for a in artifacts],
    }

    return result


# ===================================================================
# Domain 7: Portability vs Role-Dependent Drag
# ===================================================================

def domain_7_portability_drag(df: pd.DataFrame) -> dict:
    result = {}

    pt_col = "portable_talent_score"
    ti_col = "total_impact_score"

    if pt_col not in df.columns or ti_col not in df.columns:
        return {"error": f"Missing columns: {pt_col} or {ti_col}"}

    pt = pd.to_numeric(df[pt_col], errors="coerce")
    ti = pd.to_numeric(df[ti_col], errors="coerce")

    df["rank_portable"] = pt.rank(ascending=False, method="min").astype(int)
    df["rank_total_impact"] = ti.rank(ascending=False, method="min").astype(int)
    df["rank_delta_port_ti"] = df["rank_portable"] - df["rank_total_impact"]

    # Positive delta = player ranks higher on portable than total (role-dependent component drags them down)
    mean_delta = df["rank_delta_port_ti"].mean()
    std_delta = df["rank_delta_port_ti"].std()

    # Top offensive players (top 20% OBKE)
    obke_80 = df["OBKE_raw"].quantile(0.80)
    top_off = df[df["OBKE_raw"] >= obke_80]

    # Decompose drag into offensive / defensive / role components
    # Approximate: compare layer contributions
    df["off_drag"] = df["layer3_off_elev"] + df["layer2_rue"]  # offensive role-dependent components
    df["def_drag"] = df["layer3_def_elev"] + df["layer4_scheme"]  # defensive role-dependent components

    top_off_drag = top_off["rank_delta_port_ti"].mean() if len(top_off) > 0 else None

    # Top dragged players (largest positive delta = portable rank >> total rank)
    top_dragged = (
        df.nlargest(15, "rank_delta_port_ti")
        [["player_name", "season", "rank_portable", "rank_total_impact",
          "rank_delta_port_ti", "OBKE_raw", "DBKE_raw"]]
        .to_dict(orient="records")
    )
    for rec in top_dragged:
        for k in rec:
            if isinstance(rec[k], (np.floating, float)):
                rec[k] = _sf(rec[k])
            elif isinstance(rec[k], (np.integer,)):
                rec[k] = int(rec[k])

    result["test_7_1_portability_vs_total_rank_delta"] = {
        "mean_rank_delta": _sf(mean_delta),
        "std_rank_delta": _sf(std_delta),
        "top_offensive_mean_drag": _sf(top_off_drag),
        "top_15_most_dragged": top_dragged,
        "off_drag_corr_with_delta": _sf(_safe_corr(df["off_drag"], df["rank_delta_port_ti"])[0]),
        "def_drag_corr_with_delta": _sf(_safe_corr(df["def_drag"], df["rank_delta_port_ti"])[0]),
    }

    return result


# ===================================================================
# Domain 8: Rank Movement Driver Decomposition
# ===================================================================

def domain_8_rank_movement(df: pd.DataFrame) -> dict:
    result = {}

    # For each player, decompose BKE_raw into contributions, then see which
    # component drives rank position most.

    # Components:
    # OBKE_raw = 0.55*L1_off + 0.25*L2_rue + 0.20*L3_off_elev
    # DBKE_raw = 0.60*L1_def + 0.25*L3_def_elev + 0.15*L4_scheme
    # BKE = OBKE + DBKE

    # Decompose rank variance by correlating each component with BKE rank
    rank_bke = df["rank_actual"].astype(float)

    components = OrderedDict([
        ("L1_offensive (0.55)", 0.55 * df["layer1_off"]),
        ("L2_RUE (0.25)", 0.25 * df["layer2_rue"]),
        ("L3_off_elevation (0.20)", 0.20 * df["layer3_off_elev"]),
        ("L1_defensive (0.60)", 0.60 * df["layer1_def"]),
        ("L3_def_elevation (0.25)", 0.25 * df["layer3_def_elev"]),
        ("L4_scheme_bonus (0.15)", 0.15 * df["layer4_scheme"]),
    ])

    component_analysis = []
    for name, vals in components.items():
        r, p, n = _safe_corr(vals, -rank_bke)  # negative because rank 1 = highest
        var_contrib = vals.var(ddof=0)
        component_analysis.append({
            "component": name,
            "corr_with_BKE_rank": r,
            "variance": _sf(var_contrib),
            "mean": _sf(vals.mean()),
            "std": _sf(vals.std(ddof=0)),
        })

    # Aggregate offensive vs defensive rank drivers
    off_components = 0.55 * df["layer1_off"] + 0.25 * df["layer2_rue"] + 0.20 * df["layer3_off_elev"]
    def_components = 0.60 * df["layer1_def"] + 0.25 * df["layer3_def_elev"] + 0.15 * df["layer4_scheme"]

    var_off_total = off_components.var(ddof=0)
    var_def_total = def_components.var(ddof=0)
    total_var = var_off_total + var_def_total
    off_driver_share = var_off_total / total_var if total_var > 0 else None
    def_driver_share = var_def_total / total_var if total_var > 0 else None

    # Role compression measure: std of role_dependent_impact vs portable_talent
    rd_col = "role_dependent_impact_score"
    pt_col = "portable_talent_score"
    role_compression = None
    if rd_col in df.columns and pt_col in df.columns:
        std_rd = pd.to_numeric(df[rd_col], errors="coerce").std(ddof=0)
        std_pt = pd.to_numeric(df[pt_col], errors="coerce").std(ddof=0)
        role_compression = _sf(std_rd / std_pt) if std_pt > 0 else None

    result["rank_movement_decomposition"] = {
        "components": component_analysis,
        "off_driver_share": _sf(off_driver_share),
        "def_driver_share": _sf(def_driver_share),
        "role_compression_ratio": role_compression,
        "primary_rank_driver": "DEFENSIVE" if (def_driver_share is not None and def_driver_share > 0.55) else "OFFENSIVE" if (off_driver_share is not None and off_driver_share > 0.55) else "BALANCED",
    }

    return result


# ===================================================================
# Player-Level Diagnostic Table
# ===================================================================

def build_player_level_table(df: pd.DataFrame) -> list:
    """Build per-player diagnostic row for the master file."""
    records = []
    for _, row in df.iterrows():
        rec = {
            "player_name": str(row.get("player_name", "")),
            "player_id": str(row.get("player_id", "")),
            "season": str(row.get("season", "")),
            "primary_archetype": str(row.get("primary_archetype", "")),
            "OBKE_raw": _sf(row.get("OBKE_raw")),
            "DBKE_raw": _sf(row.get("DBKE_raw")),
            "BKE_raw": _sf(row.get("BKE_raw")),
            "OBKE_z": _sf(row.get("OBKE_z")),
            "DBKE_z": _sf(row.get("DBKE_z")),
            "BKE_equal_var": _sf(row.get("BKE_equal_var")),
            "rank_actual": int(row.get("rank_actual", 0)),
            "rank_equal_var": int(row.get("rank_equal_var", 0)),
            "rank_shift_equal_var": int(row.get("rank_shift_equal_var", 0)),
        }
        # Add bias sim ranks if present
        if "rank_55_45" in row.index:
            rec["rank_55_45"] = int(row.get("rank_55_45", 0))
            rec["rank_60_40"] = int(row.get("rank_60_40", 0))
            rec["rank_shift_55_45"] = int(row.get("rank_shift_55_45", 0))
            rec["rank_shift_60_40"] = int(row.get("rank_shift_60_40", 0))
        records.append(rec)
    return records


# ===================================================================
# Main Entry
# ===================================================================

def run_diagnostics(
    parquet_path: str = BKE_V28_OUTPUT_PARQUET,
    output_path: str = None,
) -> dict:
    if output_path is None:
        output_path = os.path.join(REPORTS_DIR, "bke_v29_diagnostic_master.json")

    print(f"[v2.9 Diagnostics] Loading: {parquet_path}")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Decomposition parquet not found: {parquet_path}")

    df_all = pd.read_parquet(parquet_path)
    print(f"[v2.9 Diagnostics] Total rows: {len(df_all)}")

    # Filter to qualified
    df = df_all[df_all["qualified"] == True].copy()
    print(f"[v2.9 Diagnostics] Qualified players: {len(df)}")

    if df.empty:
        raise ValueError("No qualified players found.")

    # Reconstruct OBKE/DBKE/BKE
    df = _reconstruct_obke_dbke(df)
    print("[v2.9 Diagnostics] OBKE/DBKE/BKE reconstructed.")

    t0 = time.time()

    # Run all 8 domains
    print("[v2.9 Diagnostics] Domain 1: Variance Asymmetry...")
    d1 = domain_1_variance_asymmetry(df)

    print("[v2.9 Diagnostics] Domain 2: Defensive Signal Quality...")
    d2 = domain_2_defensive_signal(df)

    print("[v2.9 Diagnostics] Domain 3: Counting Stats vs On/Off...")
    d3 = domain_3_counting_vs_onoff(df)

    print("[v2.9 Diagnostics] Domain 4: Variance Anchor Stress Test...")
    d4 = domain_4_variance_anchor(df)

    print("[v2.9 Diagnostics] Domain 5: Offense Weight Bias...")
    d5 = domain_5_offense_bias(df)

    print("[v2.9 Diagnostics] Domain 6: Archetype Coefficient Audit...")
    d6 = domain_6_archetype_audit(df)

    print("[v2.9 Diagnostics] Domain 7: Portability vs Role Drag...")
    d7 = domain_7_portability_drag(df)

    print("[v2.9 Diagnostics] Domain 8: Rank Movement Decomposition...")
    d8 = domain_8_rank_movement(df)

    elapsed = time.time() - t0

    # Player-level table
    print("[v2.9 Diagnostics] Building player-level table...")
    player_table = build_player_level_table(df)

    # Assemble master diagnostic file
    master = {
        "version": "2.9",
        "type": "diagnostic_suite",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_file": parquet_path,
        "output_file": output_path,
        "qualified_players": int(len(df)),
        "seasons": sorted(df["season"].unique().tolist()),
        "runtime_seconds": _sf(elapsed),
        "domains": {
            "1_variance_asymmetry_audit": d1,
            "2_defensive_signal_quality_audit": d2,
            "3_counting_stats_vs_onoff": d3,
            "4_variance_anchor_stress_test": d4,
            "5_offense_weight_bias_experiment": d5,
            "6_archetype_coefficient_audit": d6,
            "7_portability_vs_role_drag": d7,
            "8_rank_movement_decomposition": d8,
        },
        "player_level_metrics": player_table,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(master, f, indent=2)

    print(f"\n[v2.9 Diagnostics] Complete in {elapsed:.1f}s")
    print(f"[v2.9 Diagnostics] Output: {output_path}")
    print(f"[v2.9 Diagnostics] Domains: 8 / Players: {len(df)}")

    return master


if __name__ == "__main__":
    run_diagnostics()
