"""
src/modeling/dbke_v30_defense_shrinkage.py
=============================================================================
BKE v3.0 — Defense Shrinkage / Geometry / Balance (Phase-by-Phase)

Implements 3 sequential phases from the v3.0 defense reconstruction plan:

  Phase A: Defensive Stability & Shrinkage
  Phase B: Defensive Geometry Calibration
  Phase C: Offense/Defense Global Weighting

At the end of each phase, validations are evaluated. If a phase fails validation,
the script stops and writes a single JSON output report.

Output:
  reports/dbke_v30_defense_shrinkage.json
=============================================================================
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.modeling.model_config import BKE_V28_OUTPUT_PARQUET, BKE_V30_DEFENSE_SHRINKAGE_JSON


@dataclass
class V30DefenseConfig:
    # Phase A
    shrink_k_possessions: float = 2000.0
    prior_alpha: float = 0.65
    prior_blend_w: float = 0.70
    dbke_w_portable: float = 0.55
    dbke_w_rapm_shrunk: float = 0.45

    # Phase B
    z_neg_gamma: float = 0.10
    archetype_eta: float = 0.85

    # Phase C
    var_equalize_ratio_threshold: float = 1.50
    offense_delta: float = 0.10

    # Validation thresholds
    phase_a_min_corr_dbke_drapm: float = 0.50
    phase_c_max_def_driver_share: float = 0.65
    phase_c_min_dbke_yoy_corr: float = 0.50
    phase_c_min_def_spec_yoy_corr: float = 0.45
    phase_c_max_arch_ratio: float = 1.60


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


def _season_sort_key(season: str) -> int:
    text = str(season)
    try:
        return int(text.split("-")[0])
    except Exception:
        return 0


def _zscore(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    std = vals.std(ddof=0)
    if pd.isna(std) or std < 1e-12:
        return pd.Series(0.0, index=series.index)
    return (vals - vals.mean()) / std


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    mask = a.notna() & b.notna()
    if int(mask.sum()) < 10:
        return np.nan
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def _compute_yoy_corr(df: pd.DataFrame, value_col: str) -> float:
    seasons = sorted(df["season"].dropna().unique(), key=_season_sort_key)
    corr_vals: List[float] = []
    for i in range(len(seasons) - 1):
        s1, s2 = seasons[i], seasons[i + 1]
        a = df[df["season"] == s1][["player_id", value_col]].rename(columns={value_col: "v1"})
        b = df[df["season"] == s2][["player_id", value_col]].rename(columns={value_col: "v2"})
        merged = a.merge(b, on="player_id", how="inner")
        if len(merged) < 10:
            continue
        corr = _safe_corr(merged["v1"], merged["v2"])
        if np.isfinite(corr):
            corr_vals.append(corr)
    if not corr_vals:
        return np.nan
    return float(np.mean(corr_vals))


def _compute_rank_stability_yoy(
    df: pd.DataFrame,
    value_col: str,
    specialist_mode: str,
    specialist_ids: set | None = None,
    specialist_q: float = 0.75,
) -> Dict:
    seasons = sorted(df["season"].dropna().unique(), key=_season_sort_key)
    pair_rows = []

    for i in range(len(seasons) - 1):
        s1, s2 = seasons[i], seasons[i + 1]
        a = df[df["season"] == s1][["player_id", value_col]].rename(columns={value_col: "v1"})
        b = df[df["season"] == s2][["player_id", value_col]].rename(columns={value_col: "v2"})
        merged = a.merge(b, on="player_id", how="inner")

        if specialist_mode == "global_ids":
            ids = specialist_ids if specialist_ids is not None else set()
        elif specialist_mode == "baseline_value_q":
            qv = float(a["v1"].quantile(specialist_q)) if len(a) else np.nan
            ids = set(a[a["v1"] >= qv]["player_id"].tolist()) if np.isfinite(qv) else set()
        else:
            ids = set()

        merged = merged[merged["player_id"].isin(ids)].copy()
        n = int(len(merged))

        if n < 10:
            pair_rows.append(
                {
                    "pair": f"{s1}->{s2}",
                    "n": n,
                    "spearman": None,
                    "top10_retention": None,
                    "top20_retention": None,
                }
            )
            continue

        spearman = float(merged["v1"].rank(method="average").corr(merged["v2"].rank(method="average")))

        q10_1 = float(merged["v1"].quantile(0.90))
        q10_2 = float(merged["v2"].quantile(0.90))
        top10_1 = set(merged[merged["v1"] >= q10_1]["player_id"].tolist())
        top10_2 = set(merged[merged["v2"] >= q10_2]["player_id"].tolist())
        top10_ret = float(len(top10_1 & top10_2) / len(top10_1)) if len(top10_1) else np.nan

        q20_1 = float(merged["v1"].quantile(0.80))
        q20_2 = float(merged["v2"].quantile(0.80))
        top20_1 = set(merged[merged["v1"] >= q20_1]["player_id"].tolist())
        top20_2 = set(merged[merged["v2"] >= q20_2]["player_id"].tolist())
        top20_ret = float(len(top20_1 & top20_2) / len(top20_1)) if len(top20_1) else np.nan

        pair_rows.append(
            {
                "pair": f"{s1}->{s2}",
                "n": n,
                "spearman": _sf(spearman),
                "top10_retention": _sf(top10_ret),
                "top20_retention": _sf(top20_ret),
            }
        )

    vals_s = [r["spearman"] for r in pair_rows if r["spearman"] is not None]
    vals_r10 = [r["top10_retention"] for r in pair_rows if r["top10_retention"] is not None]
    vals_r20 = [r["top20_retention"] for r in pair_rows if r["top20_retention"] is not None]

    return {
        "avg_spearman": _sf(np.mean(vals_s)) if vals_s else None,
        "avg_top10_retention": _sf(np.mean(vals_r10)) if vals_r10 else None,
        "avg_top20_retention": _sf(np.mean(vals_r20)) if vals_r20 else None,
        "pairs": pair_rows,
    }


def _build_obke(df: pd.DataFrame) -> pd.Series:
    off_port = pd.to_numeric(df.get("offensive_portable_z", 0.0), errors="coerce").fillna(0.0)
    if "role_utilization_raw_z" in df.columns:
        rue = pd.to_numeric(df["role_utilization_raw_z"], errors="coerce").fillna(0.0)
    elif "role_utilization_raw" in df.columns:
        rue = _zscore(df["role_utilization_raw"]).fillna(0.0)
    else:
        rue = pd.Series(0.0, index=df.index)
    off_elev = _zscore(df.get("elevation_orapm", pd.Series(0.0, index=df.index))).fillna(0.0)
    return 0.55 * off_port + 0.25 * rue + 0.20 * off_elev


def run_phase_a(df: pd.DataFrame, cfg: V30DefenseConfig) -> Tuple[pd.DataFrame, Dict]:
    out = df.copy()

    out["D_port"] = pd.to_numeric(out.get("defensive_portable_z", 0.0), errors="coerce").fillna(0.0)
    out["D_rapm"] = pd.to_numeric(out.get("drapm", 0.0), errors="coerce").fillna(0.0)

    n = pd.to_numeric(out.get("possessions_played", out.get("MIN", 0.0)), errors="coerce").fillna(0.0).clip(lower=0.0)
    out["lambda_shrink"] = n / (n + cfg.shrink_k_possessions)
    out["D_rapm_shrunk"] = out["lambda_shrink"] * out["D_rapm"] + (1.0 - out["lambda_shrink"]) * out["D_port"]

    out["D_prior"] = np.nan
    out["D_stabilized"] = np.nan

    for pid, grp in out.groupby("player_id"):
        idx = list(grp.sort_values("season", key=lambda s: s.map(_season_sort_key)).index)
        vals = out.loc[idx, "D_rapm_shrunk"].astype(float).values
        prior_vals = []
        stab_vals = []
        for i, cur in enumerate(vals):
            if i == 0:
                prior = cur
            elif i == 1:
                prior = vals[i - 1]
            else:
                prior = cfg.prior_alpha * vals[i - 1] + (1.0 - cfg.prior_alpha) * vals[i - 2]
            stab = cfg.prior_blend_w * cur + (1.0 - cfg.prior_blend_w) * prior
            prior_vals.append(prior)
            stab_vals.append(stab)
        out.loc[idx, "D_prior"] = prior_vals
        out.loc[idx, "D_stabilized"] = stab_vals

    out["DBKE_raw_v30"] = (
        cfg.dbke_w_portable * out["D_port"] +
        cfg.dbke_w_rapm_shrunk * out["D_stabilized"]
    )

    corr = _safe_corr(out["DBKE_raw_v30"], out["D_rapm"])
    metrics = {
        "corr_DBKEraw_DRAPM": _sf(corr),
        "std_DBKE_raw_v30": _sf(out["DBKE_raw_v30"].std(ddof=0)),
        "nan_DBKE_raw_v30": int(out["DBKE_raw_v30"].isna().sum()),
        "phase_pass": bool(
            np.isfinite(corr) and corr >= cfg.phase_a_min_corr_dbke_drapm and
            out["DBKE_raw_v30"].isna().sum() == 0 and
            out["DBKE_raw_v30"].std(ddof=0) > 0
        ),
    }
    return out, metrics


def run_phase_b(df: pd.DataFrame, cfg: V30DefenseConfig) -> Tuple[pd.DataFrame, Dict]:
    out = df.copy()
    arch_col = "defensive_archetype" if "defensive_archetype" in out.columns else "primary_archetype"
    out[arch_col] = out[arch_col].astype(str)

    sigma_global = float(out["DBKE_raw_v30"].std(ddof=0))
    if not np.isfinite(sigma_global) or sigma_global <= 1e-12:
        sigma_global = 1.0

    out["DBKE_norm_v30"] = out["DBKE_raw_v30"] / sigma_global
    out["DBKE_asym_v30"] = np.where(
        out["DBKE_norm_v30"] >= 0,
        out["DBKE_norm_v30"],
        out["DBKE_norm_v30"] / (1.0 + cfg.z_neg_gamma * np.abs(out["DBKE_norm_v30"])),
    )

    out["arch_mean_z"] = out.groupby(arch_col)["DBKE_asym_v30"].transform("mean")
    out["DBKE_arch_v30"] = (
        cfg.archetype_eta * out["DBKE_asym_v30"] +
        (1.0 - cfg.archetype_eta) * out["arch_mean_z"]
    )
    out["DBKE_final_v30"] = out["DBKE_arch_v30"] * sigma_global
    out["arch_scale"] = np.nan

    post_arch_std = out.groupby(arch_col)["DBKE_final_v30"].std(ddof=0).dropna()
    pre_arch_std = out.groupby(arch_col)["DBKE_raw_v30"].std(ddof=0).dropna()
    pre_spread = float(pre_arch_std.std(ddof=0)) if len(pre_arch_std) else np.nan
    post_spread = float(post_arch_std.std(ddof=0)) if len(post_arch_std) else np.nan

    raw = out["DBKE_raw_v30"]
    final = out["DBKE_final_v30"]
    raw_asym = abs(raw.quantile(0.05)) / raw.quantile(0.95) if raw.quantile(0.95) != 0 else np.nan
    fin_asym = abs(final.quantile(0.05)) / final.quantile(0.95) if final.quantile(0.95) != 0 else np.nan

    metrics = {
        "sigma_global_raw": _sf(sigma_global),
        "std_DBKE_norm_v30": _sf(out["DBKE_norm_v30"].std(ddof=0)),
        "std_DBKE_final_v30": _sf(out["DBKE_final_v30"].std(ddof=0)),
        "archetype_spread_pre": _sf(pre_spread),
        "archetype_spread_post": _sf(post_spread),
        "penalty_asymmetry_pre": _sf(raw_asym),
        "penalty_asymmetry_post": _sf(fin_asym),
        "nan_DBKE_final_v30": int(out["DBKE_final_v30"].isna().sum()),
        "phase_pass": bool(
            out["DBKE_final_v30"].isna().sum() == 0 and
            np.isfinite(fin_asym) and np.isfinite(raw_asym) and fin_asym <= raw_asym and
            (np.isfinite(pre_spread) and np.isfinite(post_spread) and post_spread <= pre_spread)
        ),
    }
    return out, metrics


def run_phase_c(df: pd.DataFrame, cfg: V30DefenseConfig) -> Tuple[pd.DataFrame, Dict]:
    out = df.copy()
    out["OBKE_v30_reference"] = _build_obke(out)

    var_obke = float(out["OBKE_v30_reference"].var(ddof=0))
    var_dbke = float(out["DBKE_final_v30"].var(ddof=0))

    scale_applied = 1.0
    if var_obke > 0 and var_dbke > 0 and (var_dbke > cfg.var_equalize_ratio_threshold * var_obke):
        scale_applied = float(np.sqrt(var_obke / var_dbke))
    mu_dbke = float(out["DBKE_final_v30"].mean())
    reliability = pd.to_numeric(out.get("lambda_shrink", 1.0), errors="coerce").fillna(1.0).clip(lower=0.0, upper=1.0)
    out["DBKE_scaled_v30"] = mu_dbke + scale_applied * reliability * (out["DBKE_final_v30"] - mu_dbke)

    out["BKE_v30"] = (0.5 + cfg.offense_delta) * out["OBKE_v30_reference"] + (0.5 - cfg.offense_delta) * out["DBKE_scaled_v30"]

    var_dbke_scaled = float(out["DBKE_scaled_v30"].var(ddof=0))
    def_driver_share = var_dbke_scaled / (var_obke + var_dbke_scaled) if (var_obke + var_dbke_scaled) > 0 else np.nan

    dbke_yoy = _compute_yoy_corr(out, "DBKE_scaled_v30")

    player_port = out.groupby("player_id")["D_port"].mean().dropna()
    q75 = float(player_port.quantile(0.75)) if len(player_port) else np.nan
    spec_ids = set(player_port[player_port >= q75].index.tolist()) if np.isfinite(q75) else set()
    spec_df = out[out["player_id"].isin(spec_ids)]
    spec_yoy = _compute_yoy_corr(spec_df, "DBKE_scaled_v30") if len(spec_df) else np.nan

    rank_diag_port = _compute_rank_stability_yoy(
        out,
        "DBKE_scaled_v30",
        specialist_mode="global_ids",
        specialist_ids=spec_ids,
    )
    rank_diag_baseline_dbke = _compute_rank_stability_yoy(
        out,
        "DBKE_scaled_v30",
        specialist_mode="baseline_value_q",
        specialist_q=0.75,
    )

    arch_col = "defensive_archetype" if "defensive_archetype" in out.columns else "primary_archetype"
    arch_ratio = []
    for arch, grp in out.groupby(arch_col):
        s_db = float(grp["DBKE_scaled_v30"].std(ddof=0))
        s_ob = float(grp["OBKE_v30_reference"].std(ddof=0))
        if s_ob > 1e-12:
            arch_ratio.append(s_db / s_ob)
    arch_ratio_max = float(np.max(arch_ratio)) if arch_ratio else np.nan

    asym_pre = abs(out["DBKE_raw_v30"].quantile(0.05)) / out["DBKE_raw_v30"].quantile(0.95) if out["DBKE_raw_v30"].quantile(0.95) != 0 else np.nan
    asym_post = abs(out["DBKE_scaled_v30"].quantile(0.05)) / out["DBKE_scaled_v30"].quantile(0.95) if out["DBKE_scaled_v30"].quantile(0.95) != 0 else np.nan

    metrics = {
        "var_OBKE": _sf(var_obke),
        "var_DBKE_final": _sf(var_dbke),
        "var_DBKE_scaled": _sf(var_dbke_scaled),
        "dbke_scale_applied": _sf(scale_applied),
        "def_driver_share": _sf(def_driver_share),
        "dbke_yoy_corr": _sf(dbke_yoy),
        "def_specialist_dbke_yoy_corr": _sf(spec_yoy),
        "penalty_asymmetry_pre": _sf(asym_pre),
        "penalty_asymmetry_post": _sf(asym_post),
        "max_std_ratio_dbke_to_obke_within_archetype": _sf(arch_ratio_max),
        "specialist_rank_stability_top25_by_mean_D_port": rank_diag_port,
        "specialist_rank_stability_top25_by_baseline_DBKE_scaled": rank_diag_baseline_dbke,
        "phase_pass": bool(
            np.isfinite(def_driver_share) and def_driver_share < cfg.phase_c_max_def_driver_share and
            np.isfinite(dbke_yoy) and dbke_yoy > cfg.phase_c_min_dbke_yoy_corr and
            np.isfinite(spec_yoy) and spec_yoy > cfg.phase_c_min_def_spec_yoy_corr and
            np.isfinite(asym_pre) and np.isfinite(asym_post) and asym_post <= asym_pre and
            np.isfinite(arch_ratio_max) and arch_ratio_max < cfg.phase_c_max_arch_ratio
        ),
    }
    return out, metrics


def run_v30_defense_shrinkage(input_parquet: str, output_json: str) -> Dict:
    if not os.path.exists(input_parquet):
        raise FileNotFoundError(f"Input file not found: {input_parquet}")

    cfg = V30DefenseConfig()
    raw = pd.read_parquet(input_parquet)
    df = raw[raw.get("qualified", False) == True].copy()
    if df.empty:
        raise ValueError("No qualified rows found in input parquet.")

    required_cols = ["player_id", "player_name", "season", "drapm", "defensive_portable_z", "offensive_portable_z"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    started = time.time()
    result = {
        "version": "3.0",
        "script": "dbke_v30_defense_shrinkage.py",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_file": input_parquet,
        "output_file": output_json,
        "qualified_rows": int(len(df)),
        "config": {
            "phase_a": {
                "k": cfg.shrink_k_possessions,
                "alpha": cfg.prior_alpha,
                "w": cfg.prior_blend_w,
                "w_portable": cfg.dbke_w_portable,
                "w_rapm_shrunk": cfg.dbke_w_rapm_shrunk,
            },
            "phase_b": {
                "z_neg_gamma": cfg.z_neg_gamma,
                "archetype_eta": cfg.archetype_eta,
            },
            "phase_c": {
                "var_equalize_threshold": cfg.var_equalize_ratio_threshold,
                "offense_delta": cfg.offense_delta,
                "final_weights": {
                    "obke": 0.5 + cfg.offense_delta,
                    "dbke": 0.5 - cfg.offense_delta,
                },
            },
        },
        "phase_results": {},
        "status": "running",
    }

    # Phase A
    phase_a_df, phase_a_metrics = run_phase_a(df, cfg)
    result["phase_results"]["phase_a"] = phase_a_metrics
    if not phase_a_metrics["phase_pass"]:
        result["status"] = "stopped_after_phase_a"
        result["runtime_seconds"] = _sf(time.time() - started)
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        return result

    # Phase B
    phase_b_df, phase_b_metrics = run_phase_b(phase_a_df, cfg)
    result["phase_results"]["phase_b"] = phase_b_metrics
    if not phase_b_metrics["phase_pass"]:
        result["status"] = "stopped_after_phase_b"
        result["runtime_seconds"] = _sf(time.time() - started)
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        return result

    # Phase C
    phase_c_df, phase_c_metrics = run_phase_c(phase_b_df, cfg)
    result["phase_results"]["phase_c"] = phase_c_metrics
    if not phase_c_metrics["phase_pass"]:
        result["status"] = "stopped_after_phase_c"
    else:
        result["status"] = "completed"

    # Single output file: include summary + per-player outputs
    player_cols = [
        "player_id", "player_name", "season",
        "D_port", "D_rapm", "lambda_shrink", "D_rapm_shrunk", "D_prior", "D_stabilized",
        "DBKE_raw_v30", "arch_scale", "DBKE_norm_v30", "DBKE_asym_v30", "DBKE_final_v30", "DBKE_scaled_v30",
        "OBKE_v30_reference", "BKE_v30",
    ]
    keep_cols = [c for c in player_cols if c in phase_c_df.columns]
    players = []
    for _, row in phase_c_df[keep_cols].iterrows():
        rec = {}
        for c in keep_cols:
            val = row.get(c)
            if isinstance(val, (np.floating, float, np.integer, int)):
                rec[c] = _sf(val)
            else:
                rec[c] = None if pd.isna(val) else str(val)
        players.append(rec)

    result["player_outputs"] = players
    result["runtime_seconds"] = _sf(time.time() - started)

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(description="Run BKE v3.0 DBKE defense shrinkage plan phase-by-phase.")
    parser.add_argument("--input", default=BKE_V28_OUTPUT_PARQUET, help="Input decomposition parquet path")
    parser.add_argument("--output", default=BKE_V30_DEFENSE_SHRINKAGE_JSON, help="Single output JSON path")
    args = parser.parse_args()

    result = run_v30_defense_shrinkage(args.input, args.output)
    print(f"[v3.0 Defense] Status: {result['status']}")
    print(f"[v3.0 Defense] Output: {args.output}")
    if "phase_results" in result:
        for phase, metrics in result["phase_results"].items():
            print(f"  - {phase}: {'PASS' if metrics.get('phase_pass') else 'FAIL'}")


if __name__ == "__main__":
    main()
