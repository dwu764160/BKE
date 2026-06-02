"""
src/modeling/bke_v31_experimental_layers.py
=============================================================================
BKE v3.1 — Stability & Expressiveness Experimental Layer Runner

Runs each v3.1 layer independently against a fixed baseline, then runs a
controlled combined model (max 3 structural changes).

Output:
  reports/bke_v31_experimental_layers.json

This script is experimental-only and does not overwrite production BKE artifacts.
=============================================================================
"""

import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.modeling.model_config import (
from src.data.schema_contract import load_standardized, save_standardized
    BKE_DIR,
    BKE_V28_OUTPUT_PARQUET,
    DEFENSIVE_ARCHETYPES_PATH,
    REPORTS_DIR,
    V31_EXPERIMENTAL,
)


OUTPUT_JSON = os.path.join(REPORTS_DIR, "bke_v31_experimental_layers.json")
OUTPUT_SECOND_PASS_JSON = os.path.join(REPORTS_DIR, "bke_v31_layer36_second_pass.json")
OUTPUT_SCORES_V31_60_40 = os.path.join(BKE_DIR, "BKE_Scores_v31_60_40.json")
OUTPUT_SCORES_V31_55_45 = os.path.join(BKE_DIR, "BKE_Scores_v31_55_45.json")
OUTPUT_SCORES_V31_60_40_L36 = os.path.join(BKE_DIR, "BKE_Scores_v31_60_40_layer36.json")
OUTPUT_SCORES_V31_55_45_L36 = os.path.join(BKE_DIR, "BKE_Scores_v31_55_45_layer36.json")


@dataclass
class V31Config:
    """Local sweep grids; structural weights delegate to V31_EXPERIMENTAL."""
    base_off_weight: float = V31_EXPERIMENTAL.off_weight_default
    base_def_weight: float = V31_EXPERIMENTAL.def_weight_default

    layer1_weight_grid: Tuple[Tuple[float, float], ...] = V31_EXPERIMENTAL.layer1_weight_grid

    layer2_alpha_grid: Tuple[float, ...] = (0.03, 0.05, 0.07)
    layer3_tail_exponents: Tuple[float, ...] = (1.03, 1.05, 1.08)
    layer6_k_grid: Tuple[float, ...] = (0.10, 0.15, 0.20)

    def_driver_share_floor: float = V31_EXPERIMENTAL.def_driver_share_floor
    def_driver_share_ceil: float = V31_EXPERIMENTAL.def_driver_share_ceil


CFG = V31Config()


def _sf(v):
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if np.isnan(x) or np.isinf(x):
        return None
    return round(x, 6)


def _zscore(s: pd.Series) -> pd.Series:
    vals = pd.to_numeric(s, errors="coerce")
    std = vals.std(ddof=0)
    if pd.isna(std) or std < 1e-12:
        return pd.Series(0.0, index=s.index)
    return (vals - vals.mean()) / std


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    aa = pd.to_numeric(a, errors="coerce")
    bb = pd.to_numeric(b, errors="coerce")
    mask = aa.notna() & bb.notna()
    if int(mask.sum()) < 10:
        return np.nan
    return float(np.corrcoef(aa[mask], bb[mask])[0, 1])


def _spearman_corr(a: pd.Series, b: pd.Series) -> float:
    aa = pd.to_numeric(a, errors="coerce")
    bb = pd.to_numeric(b, errors="coerce")
    mask = aa.notna() & bb.notna()
    if int(mask.sum()) < 10:
        return np.nan
    ra = aa[mask].rank(method="average")
    rb = bb[mask].rank(method="average")
    return float(np.corrcoef(ra, rb)[0, 1])


def _with_season_rank(df: pd.DataFrame, score_col: str, rank_col: str = "rank") -> pd.DataFrame:
    out = df.copy()
    out[rank_col] = out.groupby("season")[score_col].rank(ascending=False, method="min")
    return out


def _yoy_score_corr(df: pd.DataFrame, score_col: str) -> float:
    pairs = []
    seasons = sorted(df["season"].dropna().unique())
    for i in range(len(seasons) - 1):
        s1, s2 = seasons[i], seasons[i + 1]
        a = df[df["season"] == s1][["player_id", score_col]].rename(columns={score_col: "x"})
        b = df[df["season"] == s2][["player_id", score_col]].rename(columns={score_col: "y"})
        m = a.merge(b, on="player_id", how="inner")
        if len(m) < 10:
            continue
        pairs.append(_spearman_corr(m["x"], m["y"]))
    if not pairs:
        return np.nan
    return float(np.nanmean(pairs))


def _specialist_yoy(df: pd.DataFrame, dbke_col: str, specialist_ids: set) -> Dict[str, float]:
    seasons = sorted(df["season"].dropna().unique())
    pearsons = []
    spearmans = []
    for i in range(len(seasons) - 1):
        s1, s2 = seasons[i], seasons[i + 1]
        a = df[(df["season"] == s1) & (df["player_id"].isin(specialist_ids))][["player_id", dbke_col]].rename(columns={dbke_col: "x"})
        b = df[(df["season"] == s2) & (df["player_id"].isin(specialist_ids))][["player_id", dbke_col]].rename(columns={dbke_col: "y"})
        m = a.merge(b, on="player_id", how="inner")
        if len(m) < 10:
            continue
        pearsons.append(_safe_corr(m["x"], m["y"]))
        spearmans.append(_spearman_corr(m["x"], m["y"]))
    return {
        "pearson": float(np.nanmean(pearsons)) if pearsons else np.nan,
        "spearman": float(np.nanmean(spearmans)) if spearmans else np.nan,
    }


def _retention(df: pd.DataFrame, score_col: str, top_q: float) -> float:
    vals = []
    seasons = sorted(df["season"].dropna().unique())
    for i in range(len(seasons) - 1):
        s1, s2 = seasons[i], seasons[i + 1]
        a = df[df["season"] == s1][["player_id", score_col]].copy()
        b = df[df["season"] == s2][["player_id", score_col]].copy()
        if len(a) < 20 or len(b) < 20:
            continue
        q1 = float(a[score_col].quantile(top_q))
        q2 = float(b[score_col].quantile(top_q))
        top1 = set(a[a[score_col] >= q1]["player_id"].tolist())
        top2 = set(b[b[score_col] >= q2]["player_id"].tolist())
        if not top1:
            continue
        vals.append(len(top1 & top2) / len(top1))
    if not vals:
        return np.nan
    return float(np.nanmean(vals))


def _predictive_rho(df: pd.DataFrame, score_col: str, target_col: str = "rapm") -> float:
    vals = []
    seasons = sorted(df["season"].dropna().unique())
    for i in range(len(seasons) - 1):
        s1, s2 = seasons[i], seasons[i + 1]
        a = df[df["season"] == s1][["player_id", score_col]].rename(columns={score_col: "x"})
        b = df[df["season"] == s2][["player_id", target_col]].rename(columns={target_col: "y"})
        m = a.merge(b, on="player_id", how="inner")
        if len(m) < 10:
            continue
        vals.append(_safe_corr(m["x"], m["y"]))
    if not vals:
        return np.nan
    return float(np.nanmean(vals))


def _position_split_rho(df: pd.DataFrame, score_col: str, position_col: str = "position_bucket") -> Dict[str, float]:
    out = {}
    for pos, sub in df.groupby(position_col):
        if len(sub) < 40:
            continue
        out[str(pos)] = _predictive_rho(sub, score_col)
    return out


def _mean_abs_rank_shift(df: pd.DataFrame, score_col: str) -> float:
    ranked = _with_season_rank(df, score_col, "rank_temp")
    shifts = []
    seasons = sorted(ranked["season"].dropna().unique())
    for i in range(len(seasons) - 1):
        s1, s2 = seasons[i], seasons[i + 1]
        a = ranked[ranked["season"] == s1][["player_id", "rank_temp"]].rename(columns={"rank_temp": "r1"})
        b = ranked[ranked["season"] == s2][["player_id", "rank_temp"]].rename(columns={"rank_temp": "r2"})
        m = a.merge(b, on="player_id", how="inner")
        if len(m) < 10:
            continue
        shifts.append((m["r1"] - m["r2"]).abs().mean())
    if not shifts:
        return np.nan
    return float(np.nanmean(shifts))


def _def_driver_share(obke: pd.Series, dbke: pd.Series) -> float:
    vo = float(pd.to_numeric(obke, errors="coerce").var(ddof=0))
    vd = float(pd.to_numeric(dbke, errors="coerce").var(ddof=0))
    den = vo + vd
    if den <= 0:
        return np.nan
    return vd / den


def _penalty_asymmetry(dbke: pd.Series) -> float:
    vals = pd.to_numeric(dbke, errors="coerce")
    p5 = float(vals.quantile(0.05))
    p95 = float(vals.quantile(0.95))
    if p95 == 0:
        return np.nan
    return abs(p5) / p95


def _eval_bundle(
    df: pd.DataFrame,
    obke: pd.Series,
    dbke: pd.Series,
    bke: pd.Series,
    specialist_ids: set,
    archetype_conf: pd.Series,
) -> Dict:
    work = df.copy()
    work["exp_obke"] = pd.to_numeric(obke, errors="coerce")
    work["exp_dbke"] = pd.to_numeric(dbke, errors="coerce")
    work["exp_bke"] = pd.to_numeric(bke, errors="coerce")

    specialist = _specialist_yoy(work, "exp_dbke", specialist_ids)
    pos_rho = _position_split_rho(work, "exp_bke")

    center_rho = pos_rho.get("Center", np.nan)

    return {
        "global_yoy_rank_corr": _sf(_yoy_score_corr(work, "exp_bke")),
        "specialist_yoy_pearson": _sf(specialist["pearson"]),
        "specialist_yoy_spearman": _sf(specialist["spearman"]),
        "top10_retention": _sf(_retention(work, "exp_bke", 0.90)),
        "top20_retention": _sf(_retention(work, "exp_bke", 0.80)),
        "predictive_rho": _sf(_predictive_rho(work, "exp_bke", "rapm")),
        "position_split_rho": {k: _sf(v) for k, v in pos_rho.items()},
        "center_next_season_rho": _sf(center_rho),
        "archetype_confidence_mean": _sf(pd.to_numeric(archetype_conf, errors="coerce").mean()),
        "std_dbke_final": _sf(pd.to_numeric(dbke, errors="coerce").std(ddof=0)),
        "penalty_asymmetry": _sf(_penalty_asymmetry(dbke)),
        "mean_abs_rank_shift": _sf(_mean_abs_rank_shift(work, "exp_bke")),
        "def_driver_share": _sf(_def_driver_share(obke, dbke)),
    }


def _delta_metrics(new_metrics: Dict, baseline: Dict) -> Dict:
    delta = {}
    for k, v in new_metrics.items():
        if isinstance(v, dict):
            continue
        b = baseline.get(k)
        if isinstance(v, (int, float)) and isinstance(b, (int, float)):
            delta[f"delta_{k}"] = _sf(v - b)
    return delta


def _score_candidate(metrics: Dict) -> float:
    predictive = metrics.get("predictive_rho") or -1.0
    yoy = metrics.get("global_yoy_rank_corr") or -1.0
    rank_shift = metrics.get("mean_abs_rank_shift")
    rank_term = -rank_shift if isinstance(rank_shift, (int, float)) else -999.0
    specialist = metrics.get("specialist_yoy_pearson") or -1.0
    conf = metrics.get("archetype_confidence_mean") or 0.0

    share = metrics.get("def_driver_share")
    share_penalty = 0.0
    if isinstance(share, (int, float)):
        if share < CFG.def_driver_share_floor:
            share_penalty = (CFG.def_driver_share_floor - share) * 2.0
        elif share > CFG.def_driver_share_ceil:
            share_penalty = (share - CFG.def_driver_share_ceil) * 2.0

    return float((2.5 * predictive) + (2.0 * yoy) + (0.5 * specialist) + (0.2 * conf) + (0.02 * rank_term) - share_penalty)


def _is_viable(metrics: Dict, baseline: Dict) -> bool:
    top10 = metrics.get("top10_retention")
    top20 = metrics.get("top20_retention")
    pred = metrics.get("predictive_rho")
    std_d = metrics.get("std_dbke_final")
    share = metrics.get("def_driver_share")

    b_top10 = baseline.get("top10_retention")
    b_top20 = baseline.get("top20_retention")
    b_pred = baseline.get("predictive_rho")
    b_std = baseline.get("std_dbke_final")
    b_share = baseline.get("def_driver_share")

    if isinstance(top10, (int, float)) and isinstance(b_top10, (int, float)) and top10 < (b_top10 - 0.10):
        return False
    if isinstance(top20, (int, float)) and isinstance(b_top20, (int, float)) and top20 < (b_top20 - 0.10):
        return False
    if isinstance(pred, (int, float)) and isinstance(b_pred, (int, float)) and pred < (b_pred - 0.02):
        return False
    if isinstance(std_d, (int, float)) and isinstance(b_std, (int, float)) and std_d > (b_std * 1.05):
        return False
    if isinstance(share, (int, float)) and isinstance(b_share, (int, float)) and share > (b_share + 0.10):
        return False
    return True


def _export_scores_json(
    df: pd.DataFrame,
    obke: pd.Series,
    dbke: pd.Series,
    bke: pd.Series,
    output_path: str,
    profile_name: str,
    off_weight: float,
    def_weight: float,
) -> Dict:
    out = df[["player_id", "season"]].copy()
    if "player_name" in df.columns:
        out["player_name"] = df["player_name"]
    elif "player_name" in df.columns:
        out["player_name"] = df["player_name"]
    else:
        out["player_name"] = None

    out["obke"] = pd.to_numeric(obke, errors="coerce")
    out["dbke"] = pd.to_numeric(dbke, errors="coerce")
    out["bke"] = pd.to_numeric(bke, errors="coerce")
    out["season"] = out["season"].astype(str)
    out["player_id"] = out["player_id"].astype(str)

    out["rank"] = out.groupby("season")["bke"].rank(ascending=False, method="min")
    n_by_season = out.groupby("season")["bke"].transform("count")
    out["bke_pct"] = (1.0 - (out["rank"] - 1.0) / np.maximum(n_by_season - 1.0, 1.0)) * 100.0

    players = {}
    for i, row in out.iterrows():
        key = f"{row['player_id']}::{row['season']}"
        players[key] = {
            "player_id": row["player_id"],
            "player_name": row.get("player_name"),
            "season": row["season"],
            "obke": _sf(row["obke"]),
            "dbke": _sf(row["dbke"]),
            "bke": _sf(row["bke"]),
            "rank": int(row["rank"]) if pd.notna(row["rank"]) else None,
            "bke_pct": _sf(row["bke_pct"]),
        }

    payload = {
        "version": "3.1",
        "profile": profile_name,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "off_weight": _sf(off_weight),
        "def_weight": _sf(def_weight),
        "players": players,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return {
        "output_file": output_path,
        "rows": int(len(out)),
        "off_weight": _sf(off_weight),
        "def_weight": _sf(def_weight),
        "profile": profile_name,
    }


def _build_base(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    off_p = pd.to_numeric(df["offensive_portable_z"], errors="coerce").fillna(0.0)
    def_p = pd.to_numeric(df["defensive_portable_z"], errors="coerce").fillna(0.0)

    if "role_utilization_raw_z" in df.columns:
        rue = pd.to_numeric(df["role_utilization_raw_z"], errors="coerce").fillna(0.0)
    elif "role_utilization_raw" in df.columns:
        rue = _zscore(df["role_utilization_raw"]).fillna(0.0)
    else:
        rue = pd.Series(0.0, index=df.index)

    off_elev = _zscore(df["elevation_orapm"]).fillna(0.0)
    def_elev = _zscore(df["elevation_drapm"]).fillna(0.0)
    scheme = pd.to_numeric(df.get("scheme_stability_z", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0).clip(lower=0.0)

    obke = V31_EXPERIMENTAL.obke_w_off_portable * off_p + V31_EXPERIMENTAL.obke_w_role_util * rue + V31_EXPERIMENTAL.obke_w_off_elev * off_elev
    dbke = V31_EXPERIMENTAL.dbke_w_def_portable * def_p + V31_EXPERIMENTAL.dbke_w_def_elev * def_elev + V31_EXPERIMENTAL.dbke_w_scheme * scheme
    bke = (CFG.base_off_weight * obke) + (CFG.base_def_weight * dbke)
    return obke, dbke, bke, def_p


def _load_confidence(df: pd.DataFrame) -> pd.Series:
    if not os.path.exists(DEFENSIVE_ARCHETYPES_PATH):
        return pd.Series(0.5, index=df.index)
    conf_df = load_standardized(DEFENSIVE_ARCHETYPES_PATH)
    merge = conf_df[["player_id", "season", "defensive_confidence"]].copy()
    merge = merge.rename(columns={"player_id": "player_id", "season": "season"})
    merge["player_id"] = merge["player_id"].astype(str)
    merge["season"] = merge["season"].astype(str)

    keyed = df[["player_id", "season"]].copy()
    keyed["player_id"] = keyed["player_id"].astype(str)
    keyed["season"] = keyed["season"].astype(str)
    keyed = keyed.merge(merge, on=["player_id", "season"], how="left")
    return pd.to_numeric(keyed["defensive_confidence"], errors="coerce").fillna(0.5).clip(0.0, 1.0)


def _build_d_stabilized(df: pd.DataFrame, def_portable: pd.Series) -> pd.Series:
    out = df[["player_id", "season"]].copy()
    out["D_port"] = pd.to_numeric(def_portable, errors="coerce").fillna(0.0)
    out["D_rapm"] = pd.to_numeric(df.get("drapm", 0.0), errors="coerce").fillna(0.0)
    poss = pd.to_numeric(df.get("possessions_played", df.get("min", 0.0)), errors="coerce").fillna(0.0).clip(lower=0.0)

    shrink_k = 2000.0
    prior_alpha = 0.65
    prior_blend_w = 0.70

    out["lambda"] = poss / (poss + shrink_k)
    out["D_rapm_shrunk"] = out["lambda"] * out["D_rapm"] + (1.0 - out["lambda"]) * out["D_port"]
    out["D_stabilized"] = out["D_rapm_shrunk"]

    def season_key(s):
        try:
            return int(str(s).split("-")[0])
        except Exception:
            return 0

    for pid, grp in out.groupby("player_id"):
        idx = list(grp.sort_values("season", key=lambda x: x.map(season_key)).index)
        vals = out.loc[idx, "D_rapm_shrunk"].astype(float).values
        stab = []
        for i, cur in enumerate(vals):
            if i == 0:
                prior = cur
            elif i == 1:
                prior = vals[i - 1]
            else:
                prior = prior_alpha * vals[i - 1] + (1.0 - prior_alpha) * vals[i - 2]
            stab.append(prior_blend_w * cur + (1.0 - prior_blend_w) * prior)
        out.loc[idx, "D_stabilized"] = stab

    return out["D_stabilized"]


def _pca_orthogonal_confidence(df: pd.DataFrame, archetype_col: str = "defensive_archetype") -> pd.Series:
    dims = [
        "dim_defensive_playmaking_z",
        "dim_defensive_impact_z",
        "dim_defensive_versatility_z",
        "dim_extra_poss_defensive_z",
    ]
    use = [c for c in dims if c in df.columns]
    if len(use) < 2 or archetype_col not in df.columns:
        return pd.Series(0.5, index=df.index)

    x = df[use].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    corr = x.corr().abs()
    drop_cols = set()
    for i, c1 in enumerate(use):
        for c2 in use[i + 1:]:
            if corr.loc[c1, c2] > 0.85:
                v1 = x[c1].var(ddof=0)
                v2 = x[c2].var(ddof=0)
                drop_cols.add(c1 if v1 < v2 else c2)
    keep = [c for c in use if c not in drop_cols]
    if len(keep) < 2:
        keep = use[:2]

    xx = x[keep].values
    xx = xx - xx.mean(axis=0, keepdims=True)
    cov = np.cov(xx, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    eps = 1e-9
    whiten = xx @ eigvecs @ np.diag(1.0 / np.sqrt(np.maximum(eigvals, eps)))

    y = pd.Series(df[archetype_col].astype(str).values).reset_index(drop=True)
    centroids = {}
    for label, idx in y.groupby(y).groups.items():
        if len(idx) < 10:
            continue
        centroids[label] = whiten[list(idx), :].mean(axis=0)
    if not centroids:
        return pd.Series(0.5, index=df.index)

    labels = list(centroids.keys())
    cmat = np.vstack([centroids[k] for k in labels])
    d2 = np.sum((whiten[:, None, :] - cmat[None, :, :]) ** 2, axis=2)
    logits = -d2
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
    conf = probs.max(axis=1)
    return pd.Series(conf, index=df.index)


def run_v31_experiments(
    parquet_path: str = BKE_V28_OUTPUT_PARQUET,
    output_json: str = OUTPUT_JSON,
) -> Dict:
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Missing decomposition parquet: {parquet_path}")

    raw = load_standardized(parquet_path)
    df = raw[raw["qualified"] == True].copy()
    if df.empty:
        raise ValueError("No qualified rows in decomposition parquet.")

    obke_base, dbke_base, bke_base, def_portable = _build_base(df)
    archetype_conf = _load_confidence(df)

    specialist_cut = pd.to_numeric(def_portable, errors="coerce").quantile(0.75)
    specialist_ids = set(df.loc[def_portable >= specialist_cut, "player_id"].tolist())

    baseline_metrics = _eval_bundle(df, obke_base, dbke_base, bke_base, specialist_ids, archetype_conf)

    baseline_export_60 = _export_scores_json(
        df=df,
        obke=obke_base,
        dbke=dbke_base,
        bke=bke_base,
        output_path=OUTPUT_SCORES_V31_60_40,
        profile_name="baseline",
        off_weight=V31_EXPERIMENTAL.off_weight_default,
        def_weight=V31_EXPERIMENTAL.def_weight_default,
    )
    bke_55_45_base = (V31_EXPERIMENTAL.off_weight_alt * obke_base) + (V31_EXPERIMENTAL.def_weight_alt * dbke_base)
    baseline_export_55 = _export_scores_json(
        df=df,
        obke=obke_base,
        dbke=dbke_base,
        bke=bke_55_45_base,
        output_path=OUTPUT_SCORES_V31_55_45,
        profile_name="baseline",
        off_weight=V31_EXPERIMENTAL.off_weight_alt,
        def_weight=V31_EXPERIMENTAL.def_weight_alt,
    )

    layer_results = {
        "layer_1_weight_grid": [],
        "layer_2_specialty_dampening": [],
        "layer_3_def_tail_scaling": [],
        "layer_4_conf_weighted_def_blend": [],
        "layer_5_dimensional_cleanup": [],
        "layer_6_axis_volatility_scaling": [],
    }

    # Layer 1
    best_l1 = None
    best_l1_score = -1e18
    for w_off, w_def in CFG.layer1_weight_grid:
        bke = (w_off * obke_base) + (w_def * dbke_base)
        m = _eval_bundle(df, obke_base, dbke_base, bke, specialist_ids, archetype_conf)
        rec = {
            "config": {"w_off": _sf(w_off), "w_def": _sf(w_def)},
            "metrics": m,
            "delta_vs_baseline": _delta_metrics(m, baseline_metrics),
            "score": _sf(_score_candidate(m)),
        }
        layer_results["layer_1_weight_grid"].append(rec)
        sc = _score_candidate(m)
        if sc > best_l1_score:
            best_l1_score = sc
            best_l1 = rec

    # Layer 2
    best_l2 = None
    best_l2_score = -1e18
    eps = 1e-9
    for alpha in CFG.layer2_alpha_grid:
        sidx = (obke_base - dbke_base).abs() / (obke_base.abs() + dbke_base.abs() + eps)
        sgn = np.sign(obke_base - dbke_base)
        w_o = CFG.base_off_weight * (1.0 + alpha * sidx * sgn)
        w_d = CFG.base_def_weight * (1.0 - alpha * sidx * sgn)
        w_sum = w_o + w_d
        w_o = w_o / np.clip(w_sum, 1e-9, None)
        w_d = w_d / np.clip(w_sum, 1e-9, None)
        bke = (w_o * obke_base) + (w_d * dbke_base)
        m = _eval_bundle(df, obke_base, dbke_base, bke, specialist_ids, archetype_conf)
        rec = {
            "config": {"alpha": _sf(alpha)},
            "metrics": m,
            "delta_vs_baseline": _delta_metrics(m, baseline_metrics),
            "score": _sf(_score_candidate(m)),
        }
        layer_results["layer_2_specialty_dampening"].append(rec)
        sc = _score_candidate(m)
        if sc > best_l2_score:
            best_l2_score = sc
            best_l2 = rec

    # Layer 3
    best_l3 = None
    best_l3_score = -1e18
    mu_d = float(pd.to_numeric(dbke_base, errors="coerce").mean())
    for exp in CFG.layer3_tail_exponents:
        centered = pd.to_numeric(dbke_base, errors="coerce") - mu_d
        dbke_new = mu_d + np.sign(centered) * (np.abs(centered) ** exp)
        bke_new = (CFG.base_off_weight * obke_base) + (CFG.base_def_weight * dbke_new)
        m = _eval_bundle(df, obke_base, dbke_new, bke_new, specialist_ids, archetype_conf)
        rec = {
            "config": {"exponent": _sf(exp)},
            "metrics": m,
            "delta_vs_baseline": _delta_metrics(m, baseline_metrics),
            "score": _sf(_score_candidate(m)),
        }
        layer_results["layer_3_def_tail_scaling"].append(rec)
        sc = _score_candidate(m)
        if sc > best_l3_score:
            best_l3_score = sc
            best_l3 = rec

    # Layer 4
    d_stabilized = _build_d_stabilized(df, def_portable)
    w_rapm = 0.45 * (0.9 + 0.2 * archetype_conf)
    w_rapm = pd.to_numeric(w_rapm, errors="coerce").fillna(0.45).clip(0.40, 0.50)
    dbke_l4 = (1.0 - w_rapm) * def_portable + w_rapm * d_stabilized
    bke_l4 = (CFG.base_off_weight * obke_base) + (CFG.base_def_weight * dbke_l4)
    m4 = _eval_bundle(df, obke_base, dbke_l4, bke_l4, specialist_ids, archetype_conf)
    rec4 = {
        "config": {"rapm_weight_formula": "0.45*(0.9+0.2*conf), clipped to [0.40,0.50]"},
        "metrics": m4,
        "delta_vs_baseline": _delta_metrics(m4, baseline_metrics),
        "score": _sf(_score_candidate(m4)),
    }
    layer_results["layer_4_conf_weighted_def_blend"].append(rec4)
    best_l4 = rec4

    # Layer 5 (ratings unchanged; evaluate confidence geometry effect)
    pca_conf = _pca_orthogonal_confidence(df, "defensive_archetype")
    m5 = _eval_bundle(df, obke_base, dbke_base, bke_base, specialist_ids, pca_conf)
    rec5 = {
        "config": {"pca_cleanup": "drop |r|>0.85 pairs, whitened PCA space"},
        "metrics": m5,
        "delta_vs_baseline": _delta_metrics(m5, baseline_metrics),
        "score": _sf(_score_candidate(m5)),
        "notes": {
            "ratings_changed": False,
            "baseline_conf_mean": _sf(pd.to_numeric(archetype_conf, errors="coerce").mean()),
            "orthogonal_conf_mean": _sf(pd.to_numeric(pca_conf, errors="coerce").mean()),
        },
    }
    layer_results["layer_5_dimensional_cleanup"].append(rec5)
    best_l5 = rec5

    # Layer 6
    best_l6 = None
    best_l6_score = -1e18
    d_parts = pd.DataFrame({
        "def_port": pd.to_numeric(def_portable, errors="coerce").fillna(0.0),
        "def_elev": _zscore(df["elevation_drapm"]).fillna(0.0),
        "scheme": pd.to_numeric(df.get("scheme_stability_z", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0),
        "season": df["season"].astype(str),
    })
    hist_var = {}
    for c in ["def_port", "def_elev", "scheme"]:
        per_season_var = d_parts.groupby("season")[c].var(ddof=0)
        hist_var[c] = float(per_season_var.mean()) if len(per_season_var) else float(d_parts[c].var(ddof=0))

    for k in CFG.layer6_k_grid:
        scaled = d_parts.copy()
        for c in ["def_port", "def_elev", "scheme"]:
            mu = float(scaled[c].mean())
            sc = 1.0 / (1.0 + k * max(hist_var[c], 0.0))
            scaled[c] = mu + sc * (scaled[c] - mu)
        dbke_new = 0.60 * scaled["def_port"] + 0.25 * scaled["def_elev"] + 0.15 * scaled["scheme"]
        bke_new = (CFG.base_off_weight * obke_base) + (CFG.base_def_weight * dbke_new)
        m = _eval_bundle(df, obke_base, dbke_new, bke_new, specialist_ids, archetype_conf)
        rec = {
            "config": {"k": _sf(k)},
            "metrics": m,
            "delta_vs_baseline": _delta_metrics(m, baseline_metrics),
            "score": _sf(_score_candidate(m)),
        }
        layer_results["layer_6_axis_volatility_scaling"].append(rec)
        scv = _score_candidate(m)
        if scv > best_l6_score:
            best_l6_score = scv
            best_l6 = rec

    # Controlled combined model: max 3 structural changes with guardrails
    w_off = float(best_l1["config"]["w_off"])
    w_def = float(best_l1["config"]["w_def"])

    obke_comb = obke_base.copy()
    dbke_comb = dbke_base.copy()
    applied_layers = ["layer1_weight_grid"]

    # Candidate transforms (defensive-side + weighting)
    candidate_transforms = []

    # Layer 2 candidate
    candidate_transforms.append(("layer2_specialty_dampening", best_l2, "weight_only"))

    # Layer 3 candidate
    exp3 = float(best_l3["config"]["exponent"])
    mu_d = float(pd.to_numeric(dbke_base, errors="coerce").mean())
    centered3 = pd.to_numeric(dbke_base, errors="coerce") - mu_d
    dbke_l3_best = mu_d + np.sign(centered3) * (np.abs(centered3) ** exp3)
    candidate_transforms.append(("layer3_def_tail_scaling", best_l3, dbke_l3_best))

    # Layer 4 candidate
    candidate_transforms.append(("layer4_conf_weighted_def_blend", best_l4, dbke_l4))

    # Layer 6 candidate
    k6 = float(best_l6["config"]["k"])
    scaled6 = d_parts.copy()
    for c in ["def_port", "def_elev", "scheme"]:
        mu = float(scaled6[c].mean())
        sc = 1.0 / (1.0 + k6 * max(hist_var[c], 0.0))
        scaled6[c] = mu + sc * (scaled6[c] - mu)
    dbke_l6_best = 0.60 * scaled6["def_port"] + 0.25 * scaled6["def_elev"] + 0.15 * scaled6["scheme"]
    candidate_transforms.append(("layer6_axis_volatility_scaling", best_l6, dbke_l6_best))

    # Pick up to two additional viable transforms by score (after layer1)
    viable = [c for c in candidate_transforms if _is_viable(c[1]["metrics"], baseline_metrics)]
    viable = sorted(viable, key=lambda x: _score_candidate(x[1]["metrics"]), reverse=True)

    for name, rec, payload in viable[:2]:
        if name == "layer2_specialty_dampening":
            applied_layers.append(name)
            continue
        dbke_comb = payload
        applied_layers.append(name)

    # If layer2 selected, apply specialty-aware weights using selected alpha
    if "layer2_specialty_dampening" in applied_layers:
        alpha = float(best_l2["config"]["alpha"])
    else:
        alpha = 0.0

    sidx = (obke_comb - dbke_comb).abs() / (obke_comb.abs() + dbke_comb.abs() + eps)
    sgn = np.sign(obke_comb - dbke_comb)
    w_o = w_off * (1.0 + alpha * sidx * sgn)
    w_d = w_def * (1.0 - alpha * sidx * sgn)
    w_sum = w_o + w_d
    w_o = w_o / np.clip(w_sum, 1e-9, None)
    w_d = w_d / np.clip(w_sum, 1e-9, None)
    bke_comb = (w_o * obke_comb) + (w_d * dbke_comb)

    comb_metrics = _eval_bundle(df, obke_base, dbke_comb, bke_comb, specialist_ids, archetype_conf)

    # Dedicated second-pass: Layer 3 + Layer 6 only, no Layer 1 weight shift
    exp3 = float(best_l3["config"]["exponent"])
    k6 = float(best_l6["config"]["k"])

    scaled6_second = d_parts.copy()
    for c in ["def_port", "def_elev", "scheme"]:
        mu = float(scaled6_second[c].mean())
        sc = 1.0 / (1.0 + k6 * max(hist_var[c], 0.0))
        scaled6_second[c] = mu + sc * (scaled6_second[c] - mu)

    dbke_l6_second = 0.60 * scaled6_second["def_port"] + 0.25 * scaled6_second["def_elev"] + 0.15 * scaled6_second["scheme"]
    mu_l6 = float(pd.to_numeric(dbke_l6_second, errors="coerce").mean())
    centered_l6 = pd.to_numeric(dbke_l6_second, errors="coerce") - mu_l6
    dbke_l36 = mu_l6 + np.sign(centered_l6) * (np.abs(centered_l6) ** exp3)

    bke_l36_60 = (CFG.base_off_weight * obke_base) + (CFG.base_def_weight * dbke_l36)
    metrics_l36 = _eval_bundle(df, obke_base, dbke_l36, bke_l36_60, specialist_ids, archetype_conf)
    delta_l36 = _delta_metrics(metrics_l36, baseline_metrics)

    l3m = best_l3["metrics"]
    l6m = best_l6["metrics"]
    additivity_checks = {
        "global_yoy_not_canceled": bool((metrics_l36.get("global_yoy_rank_corr") or -999) >= max(l3m.get("global_yoy_rank_corr") or -999, l6m.get("global_yoy_rank_corr") or -999)),
        "rank_shift_not_canceled": bool((metrics_l36.get("mean_abs_rank_shift") or 999) <= min(l3m.get("mean_abs_rank_shift") or 999, l6m.get("mean_abs_rank_shift") or 999)),
        "dbke_std_not_canceled": bool((metrics_l36.get("std_dbke_final") or 999) <= min(l3m.get("std_dbke_final") or 999, l6m.get("std_dbke_final") or 999)),
        "global_yoy_vs_baseline": bool((metrics_l36.get("global_yoy_rank_corr") or -999) >= (baseline_metrics.get("global_yoy_rank_corr") or -999)),
        "rank_shift_vs_baseline": bool((metrics_l36.get("mean_abs_rank_shift") or 999) <= (baseline_metrics.get("mean_abs_rank_shift") or 999)),
        "dbke_std_vs_baseline": bool((metrics_l36.get("std_dbke_final") or 999) <= (baseline_metrics.get("std_dbke_final") or 999)),
    }
    additivity_pass = (
        sum(1 for v in additivity_checks.values() if v) >= 5
        and additivity_checks["global_yoy_vs_baseline"]
        and additivity_checks["rank_shift_vs_baseline"]
        and additivity_checks["dbke_std_vs_baseline"]
    )

    second_pass = {
        "applied_layers": ["layer_6_axis_volatility_scaling", "layer_3_def_tail_scaling"],
        "config": {
            "w_off": _sf(CFG.base_off_weight),
            "w_def": _sf(CFG.base_def_weight),
            "layer3_exponent": _sf(exp3),
            "layer6_k": _sf(k6),
            "transform_order": "layer6_then_layer3",
        },
        "metrics": metrics_l36,
        "delta_vs_baseline": delta_l36,
        "additivity_checks": additivity_checks,
        "is_additive": bool(additivity_pass),
        "decision": "promote_to_v31_json_outputs" if additivity_pass else "do_not_promote_yet",
    }

    l36_exports = {}
    if additivity_pass:
        l36_exports["v31_60_40_layer36"] = _export_scores_json(
            df=df,
            obke=obke_base,
            dbke=dbke_l36,
            bke=bke_l36_60,
            output_path=OUTPUT_SCORES_V31_60_40_L36,
            profile_name="layer3_plus_layer6",
            off_weight=V31_EXPERIMENTAL.off_weight_default,
            def_weight=V31_EXPERIMENTAL.def_weight_default,
        )
        bke_l36_55 = (V31_EXPERIMENTAL.off_weight_alt * obke_base) + (V31_EXPERIMENTAL.def_weight_alt * dbke_l36)
        l36_exports["v31_55_45_layer36"] = _export_scores_json(
            df=df,
            obke=obke_base,
            dbke=dbke_l36,
            bke=bke_l36_55,
            output_path=OUTPUT_SCORES_V31_55_45_L36,
            profile_name="layer3_plus_layer6",
            off_weight=V31_EXPERIMENTAL.off_weight_alt,
            def_weight=V31_EXPERIMENTAL.def_weight_alt,
        )

    anchors_v30 = {
        "obke_weight": 0.60,
        "dbke_weight": 0.40,
        "dbke_yoy_corr": 0.709,
        "def_specialist_dbke_yoy_corr": 0.421,
        "def_driver_share": 0.363,
        "center_next_season_rho": 0.409,
    }

    report = {
        "version": "3.1-experimental",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_file": parquet_path,
        "output_file": output_json,
        "qualified_players": int(len(df)),
        "seasons": sorted(df["season"].astype(str).unique().tolist()),
        "baseline": {
            "config": {
                "off_weight": CFG.base_off_weight,
                "def_weight": CFG.base_def_weight,
            },
            "metrics": baseline_metrics,
        },
        "anchors_v30_reference": anchors_v30,
        "layers": layer_results,
        "selected_winners": {
            "layer1": best_l1,
            "layer2": best_l2,
            "layer3": best_l3,
            "layer4": best_l4,
            "layer5": best_l5,
            "layer6": best_l6,
        },
        "combined_model": {
            "applied_layers": applied_layers,
            "config": {
                "layer1_w_off": _sf(w_off),
                "layer1_w_def": _sf(w_def),
                "layer2_alpha": _sf(alpha),
                "layer4_formula": "w_rapm=0.45*(0.9+0.2*conf), clipped [0.40,0.50]",
            },
            "metrics": comb_metrics,
            "delta_vs_baseline": _delta_metrics(comb_metrics, baseline_metrics),
        },
        "dual_split_policy": {
            "default_split": "60/40",
            "secondary_split": "55/45",
            "co_produced": True,
        },
        "dual_split_outputs": {
            "v31_60_40": baseline_export_60,
            "v31_55_45": baseline_export_55,
        },
        "layer36_second_pass": second_pass,
        "layer36_outputs": l36_exports,
    }

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with open(OUTPUT_SECOND_PASS_JSON, "w", encoding="utf-8") as f:
        json.dump(second_pass, f, indent=2)

    return report


if __name__ == "__main__":
    out = run_v31_experiments()
    print("[v3.1] Experimental layer report saved:")
    print(f"  {out['output_file']}")
    print(f"  Qualified players: {out['qualified_players']}")