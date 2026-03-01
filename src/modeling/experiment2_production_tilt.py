"""
src/modeling/experiment2_production_tilt.py
=============================================================================
BKE v3.1 — Experiment 2 rerun (production-importance tilt)

What this script does:
  1) Rebuilds a v3.1 Layer 3 + Layer 6 defensive profile baseline.
  2) Builds an expanded production proxy from:
       - ORAPM, TS_PCT
       - raw offensive box-score stats (PTS, AST, FGM, FGA, FG3M, FG3A, FTM, FTA)
  3) Applies a 15-point lambda sweep to tilt BKE toward production signal.
  4) Reports rank movement and top-100 cohort effects for low/high production.

Output:
  reports/bke_v31_experiment2_production_tilt.json
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

from src.modeling.bke_v31_experimental_layers import _build_base
from src.modeling.model_config import BKE_V28_OUTPUT_PARQUET, REPORTS_DIR


OUTPUT_JSON = os.path.join(REPORTS_DIR, "bke_v31_experiment2_production_tilt.json")
SECOND_PASS_PATH = os.path.join(REPORTS_DIR, "bke_v31_layer36_second_pass.json")


@dataclass
class Experiment2Config:
    lambda_grid: Tuple[float, ...] = tuple(round(0.01 * i, 2) for i in range(1, 16))
    layer3_exponent_fallback: float = 1.08
    layer6_k_fallback: float = 0.20
    low_prod_quantile: float = 0.25
    high_prod_quantile: float = 0.75

    production_weights: Dict[str, float] = None

    def __post_init__(self):
        if self.production_weights is None:
            self.production_weights = {
                "orapm": 0.22,
                "TS_PCT": 0.14,
                "PTS": 0.18,
                "AST": 0.12,
                "FGM": 0.08,
                "FGA": 0.08,
                "FG3M": 0.06,
                "FG3A": 0.04,
                "FTM": 0.04,
                "FTA": 0.04,
            }


CFG = Experiment2Config()


def _safe_float(v, digits=6):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if np.isnan(x) or np.isinf(x):
        return None
    return round(x, digits)


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


def _predictive_rho(df: pd.DataFrame, score_col: str = "bke", target_col: str = "rapm") -> float:
    vals = []
    seasons = sorted(df["season"].astype(str).unique())
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


def _season_rank(df: pd.DataFrame, score_col: str, out_col: str) -> pd.DataFrame:
    out = df.copy()
    out[out_col] = out.groupby("season")[score_col].rank(ascending=False, method="min")
    return out


def _mean_abs_rank_shift(base_ranked: pd.DataFrame, cand_ranked: pd.DataFrame) -> float:
    merged = base_ranked[["season", "player_id", "rank_base"]].merge(
        cand_ranked[["season", "player_id", "rank_cand"]],
        on=["season", "player_id"],
        how="inner",
    )
    if merged.empty:
        return np.nan
    return float((merged["rank_base"] - merged["rank_cand"]).abs().mean())


def _build_layer36_dbke(df: pd.DataFrame, dbke_base: pd.Series, def_portable: pd.Series, exponent: float, k: float) -> pd.Series:
    d_parts = pd.DataFrame(
        {
            "def_port": pd.to_numeric(def_portable, errors="coerce").fillna(0.0),
            "def_elev": _zscore(df["elevation_drapm"]).fillna(0.0),
            "scheme": pd.to_numeric(df.get("scheme_stability_z", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0),
            "season": df["season"].astype(str),
        }
    )

    hist_var = {}
    for col in ["def_port", "def_elev", "scheme"]:
        per_season_var = d_parts.groupby("season")[col].var(ddof=0)
        hist_var[col] = float(per_season_var.mean()) if len(per_season_var) else float(d_parts[col].var(ddof=0))

    scaled = d_parts.copy()
    for col in ["def_port", "def_elev", "scheme"]:
        mu = float(scaled[col].mean())
        sc = 1.0 / (1.0 + k * max(hist_var[col], 0.0))
        scaled[col] = mu + sc * (scaled[col] - mu)

    dbke_l6 = 0.60 * scaled["def_port"] + 0.25 * scaled["def_elev"] + 0.15 * scaled["scheme"]
    mu_l6 = float(pd.to_numeric(dbke_l6, errors="coerce").mean())
    centered = pd.to_numeric(dbke_l6, errors="coerce") - mu_l6
    dbke_l36 = mu_l6 + np.sign(centered) * (np.abs(centered) ** exponent)

    if pd.isna(dbke_l36).all():
        return pd.to_numeric(dbke_base, errors="coerce").fillna(0.0)
    return dbke_l36


def _load_layer36_config() -> Tuple[float, float]:
    if not os.path.exists(SECOND_PASS_PATH):
        return CFG.layer3_exponent_fallback, CFG.layer6_k_fallback
    try:
        payload = json.load(open(SECOND_PASS_PATH, "r", encoding="utf-8"))
        cfg = payload.get("config", {})
        exponent = float(cfg.get("layer3_exponent", CFG.layer3_exponent_fallback))
        k = float(cfg.get("layer6_k", CFG.layer6_k_fallback))
        return exponent, k
    except Exception:
        return CFG.layer3_exponent_fallback, CFG.layer6_k_fallback


def _build_production_proxy(df: pd.DataFrame) -> Tuple[pd.Series, Dict]:
    available = [c for c in CFG.production_weights.keys() if c in df.columns]
    if not available:
        raise ValueError("No configured production proxy columns are available in the input dataframe.")

    weighted_components = []
    effective_weights = {}
    total_weight = 0.0

    for col in available:
        w = float(CFG.production_weights[col])
        if w <= 0:
            continue
        season_z = df.groupby("season", group_keys=False)[col].apply(_zscore).fillna(0.0)
        weighted_components.append(w * season_z)
        effective_weights[col] = w
        total_weight += w

    if total_weight <= 0 or not weighted_components:
        raise ValueError("Production proxy weight configuration is invalid after filtering available columns.")

    prod_proxy = sum(weighted_components) / total_weight

    meta = {
        "available_columns": available,
        "effective_weights": {k: _safe_float(v, 4) for k, v in effective_weights.items()},
        "total_weight": _safe_float(total_weight, 4),
    }
    return prod_proxy, meta


def run_experiment2(
    parquet_path: str = BKE_V28_OUTPUT_PARQUET,
    output_json: str = OUTPUT_JSON,
) -> Dict:
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Missing decomposition parquet: {parquet_path}")

    raw = pd.read_parquet(parquet_path)
    df = raw[raw["qualified"] == True].copy()
    if df.empty:
        raise ValueError("No qualified rows in decomposition parquet.")

    obke_base, dbke_base, _, def_portable = _build_base(df)

    exponent, k = _load_layer36_config()
    dbke_layer36 = _build_layer36_dbke(df, dbke_base, def_portable, exponent=exponent, k=k)
    bke_base = 0.60 * obke_base + 0.40 * dbke_layer36

    work = df[["season", "player_id", "player_name", "rapm"]].copy()
    work["season"] = work["season"].astype(str)
    work["player_id"] = work["player_id"].astype(str)
    work["bke_base"] = pd.to_numeric(bke_base, errors="coerce").fillna(0.0)

    prod_proxy, proxy_meta = _build_production_proxy(df)
    work["prod_proxy_z"] = pd.to_numeric(prod_proxy, errors="coerce").fillna(0.0)

    base_ranked = _season_rank(work, "bke_base", "rank_base")
    base_predictive = _predictive_rho(base_ranked.rename(columns={"bke_base": "bke"}), "bke", "rapm")

    season_q = base_ranked.groupby("season")["prod_proxy_z"].quantile([CFG.low_prod_quantile, CFG.high_prod_quantile]).unstack()
    season_q.columns = ["q_low", "q_high"]
    base_ranked = base_ranked.merge(season_q, on="season", how="left")
    base_ranked["is_low_prod"] = base_ranked["prod_proxy_z"] <= base_ranked["q_low"]
    base_ranked["is_high_prod"] = base_ranked["prod_proxy_z"] >= base_ranked["q_high"]

    base_low_top100 = int(((base_ranked["rank_base"] <= 100) & base_ranked["is_low_prod"]).sum())
    base_high_top100 = int(((base_ranked["rank_base"] <= 100) & base_ranked["is_high_prod"]).sum())

    sweep_rows: List[Dict] = []

    for lam in CFG.lambda_grid:
        candidate = work.copy()
        candidate["bke_cand"] = candidate["bke_base"] + lam * candidate["prod_proxy_z"]
        candidate_ranked = _season_rank(candidate, "bke_cand", "rank_cand")

        merged = candidate_ranked.merge(
            base_ranked[["season", "player_id", "rank_base", "is_low_prod", "is_high_prod"]],
            on=["season", "player_id"],
            how="left",
        )

        pred = _predictive_rho(merged.rename(columns={"bke_cand": "bke"}), "bke", "rapm")
        rank_shift = _mean_abs_rank_shift(base_ranked, candidate_ranked)
        low_top100 = int(((merged["rank_cand"] <= 100) & (merged["is_low_prod"] == True)).sum())
        high_top100 = int(((merged["rank_cand"] <= 100) & (merged["is_high_prod"] == True)).sum())

        top100 = merged[merged["rank_cand"] <= 100]
        sweep_rows.append(
            {
                "lambda": _safe_float(lam, 4),
                "predictive_rho": _safe_float(pred),
                "delta_predictive_vs_base": _safe_float((pred - base_predictive) if pd.notna(pred) and pd.notna(base_predictive) else np.nan),
                "mean_abs_rank_shift_vs_base": _safe_float(rank_shift),
                "low_prod_top100_count": low_top100,
                "delta_low_prod_top100_vs_base": int(low_top100 - base_low_top100),
                "high_prod_top100_count": high_top100,
                "delta_high_prod_top100_vs_base": int(high_top100 - base_high_top100),
                "top100_mean_prod_proxy_z": _safe_float(top100["prod_proxy_z"].mean()),
            }
        )

    sweep_df = pd.DataFrame(sweep_rows)
    feasible = sweep_df[sweep_df["mean_abs_rank_shift_vs_base"] <= 3.5].copy()
    if feasible.empty:
        feasible = sweep_df.copy()
    feasible = feasible.sort_values(["predictive_rho", "lambda"], ascending=[False, True])

    recommended = feasible.iloc[0].to_dict() if not feasible.empty else {}

    report = {
        "version": "v3.1-experiment2-rerun",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_file": parquet_path,
        "output_file": output_json,
        "qualified_players": int(len(df)),
        "base_profile": {
            "name": "layer3_plus_layer6_60_40",
            "layer3_exponent": _safe_float(exponent, 4),
            "layer6_k": _safe_float(k, 4),
            "predictive_rho": _safe_float(base_predictive),
            "low_prod_quantile": CFG.low_prod_quantile,
            "high_prod_quantile": CFG.high_prod_quantile,
            "base_low_prod_top100_count": base_low_top100,
            "base_high_prod_top100_count": base_high_top100,
        },
        "production_proxy": proxy_meta,
        "lambda_sweep": sweep_rows,
        "recommended": recommended,
        "recommendation_note": "Recommended lambda is selected by highest predictive_rho with mean_abs_rank_shift_vs_base <= 3.5.",
    }

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    out = run_experiment2()
    print("[Experiment 2] Production-tilt rerun complete:")
    print(f"  {out['output_file']}")
    print(f"  Qualified players: {out['qualified_players']}")
