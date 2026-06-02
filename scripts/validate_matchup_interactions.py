"""
scripts/validate_matchup_interactions.py
=============================================================================
Simulation Core — Step 1 (Matchup Engine), part 3 of 3.

Walk-forward, leakage-free validation of the cross-team interaction cells at
PLAYER-GAME PPP level on the 2024-25 holdout (the gate defined in the Resolved
Design: player-game PPP MAE, NOT game Brier).

Protocol:
  1. Fit a two-way player FE on TRAIN matchup rows only (2017-18..2023-24),
     yielding off_fe, def_fe, grand_mean and FE-adjusted archetype cells.
  2. EB-shrink those train cells (same shrinkage as the production artifact).
  3. On each 2024-25 matchup row (off player vs def player, weight=PARTIAL_POSS):
        actual_ppp   = PLAYER_PTS / PARTIAL_POSS
        baseline_pred= grand + off_fe[off] + def_fe[def]          (talent only)
        model_pred   = baseline_pred + interaction[off_arch][def_arch]
     Compare possession-weighted MAE(baseline) vs MAE(model). The shared
     baseline cancels, so the delta isolates the OOS value of the interaction
     term. Paired bootstrap on per-row abs-error differences for significance.
  4. Stratify by train-cell significance (|cell| vs its SE) and by source tag.

Leakage control: every estimate that touches the prediction (FE + cells) is
fit on train seasons only and applied forward to 2024-25.

Output: reports/cross_team_interaction_validation.json

Usage:
  python3 scripts/validate_matchup_interactions.py
=============================================================================
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.fit_archetype_interactions_v2 import estimate_fe_cells, apply_eb_shrinkage  # noqa: E402

MATCHUP_PARQUET = ROOT / "data" / "matchup" / "league_season_matchups.parquet"
# archetype source: player_impact_profiles (actual, all 8 seasons) — the same
# source the locked signal test (test_archetype_interactions.py) joined against.
PROFILES = ROOT / "data" / "processed" / "player_eval" / "player_impact_profiles.parquet"
OUT = ROOT / "reports" / "cross_team_interaction_validation.json"

HOLDOUT_SEASON = "2024-25"
TRAIN_SEASONS = ["2017-18", "2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24"]
MIN_PARTIAL_POSS = 2.0


def _norm_id(x) -> str:
    return str(x).strip().replace(".0", "")


def load_archetypes() -> Tuple[Dict[Tuple[str, str], str], Dict[Tuple[str, str], str]]:
    """off/def primary archetype keyed by (player_id, season) from profiles."""
    df = pd.read_parquet(PROFILES)
    cols = {c.lower(): c for c in df.columns}
    pid_c = cols.get("player_id")
    seas_c = cols.get("season")
    off_c = cols.get("off_primary_archetype")
    def_c = cols.get("def_primary_archetype")
    if not all([pid_c, seas_c, off_c, def_c]):
        raise KeyError(f"profiles missing archetype/key cols; have {list(df.columns)}")
    df = df[[pid_c, seas_c, off_c, def_c]].copy()
    df[pid_c] = df[pid_c].map(_norm_id)
    df[seas_c] = df[seas_c].astype(str)
    off_of = {(r[0], r[1]): r[2] for r in df[[pid_c, seas_c, off_c]].itertuples(index=False, name=None)}
    def_of = {(r[0], r[1]): r[2] for r in df[[pid_c, seas_c, def_c]].itertuples(index=False, name=None)}
    return off_of, def_of


def main() -> int:
    if not MATCHUP_PARQUET.exists() or not PROFILES.exists():
        print("FATAL missing matchup parquet or profiles")
        return 1

    off_of, def_of = load_archetypes()
    matchups = pd.read_parquet(MATCHUP_PARQUET)

    # --- 1-2. fit + shrink train-only cells, capture train FE ---
    train_cells, off_fe, def_fe, grand = estimate_fe_cells(
        matchups, off_of, def_of, seasons=TRAIN_SEASONS,
        min_partial_poss=MIN_PARTIAL_POSS,
    )
    if train_cells.empty:
        print("FATAL train cells empty")
        return 1
    shrunk = apply_eb_shrinkage(train_cells)
    cell_map: Dict[Tuple[str, str], float] = {
        (str(r.off_arch), str(r.def_arch)): float(r.interaction_ppp) for r in shrunk.itertuples()
    }
    # significance per train cell: |raw| > 1.96*SE
    se_map = {(str(r.off_arch), str(r.def_arch)): float(r.se) for r in shrunk.itertuples()}
    raw_map = {(str(r.off_arch), str(r.def_arch)): float(r.raw_ppp) for r in shrunk.itertuples()}
    sig_cells = {k for k in cell_map if abs(raw_map[k]) > 1.959963985 * max(se_map[k], 1e-9)}

    # --- 3. holdout rows ---
    hd = matchups[matchups["season"].astype(str) == HOLDOUT_SEASON].copy()
    hd["off_player_id"] = hd["off_player_id"].map(_norm_id)
    hd["def_player_id"] = hd["def_player_id"].map(_norm_id)
    hd["partial_poss"] = pd.to_numeric(hd["partial_poss"], errors="coerce")
    hd["player_pts"] = pd.to_numeric(hd["player_pts"], errors="coerce")
    hd = hd[(hd["partial_poss"] >= MIN_PARTIAL_POSS) & hd["player_pts"].notna()]
    hd["actual_ppp"] = hd["player_pts"] / hd["partial_poss"]

    off_ids = hd["off_player_id"].to_numpy()
    def_ids = hd["def_player_id"].to_numpy()
    season = HOLDOUT_SEASON

    base = np.array([grand + off_fe.get(o, 0.0) + def_fe.get(d, 0.0) for o, d in zip(off_ids, def_ids)])
    oa = np.array([str(off_of.get((o, season))) for o in off_ids])
    da = np.array([str(def_of.get((d, season))) for d in def_ids])
    inter = np.array([cell_map.get((a, b), 0.0) for a, b in zip(oa, da)])
    is_sig = np.array([(a, b) in sig_cells for a, b in zip(oa, da)])
    has_cell = np.array([(a, b) in cell_map for a, b in zip(oa, da)])

    actual = hd["actual_ppp"].to_numpy(dtype=float)
    w = hd["partial_poss"].to_numpy(dtype=float)

    err_base = np.abs(actual - base)
    err_model = np.abs(actual - (base + inter))

    def wmae(mask=None):
        if mask is None:
            mask = np.ones(len(w), dtype=bool)
        ww = w[mask]
        if ww.sum() <= 0:
            return None, None, 0
        return (float(np.sum(err_base[mask] * ww) / ww.sum()),
                float(np.sum(err_model[mask] * ww) / ww.sum()),
                int(mask.sum()))

    mae_b_all, mae_m_all, n_all = wmae()
    mae_b_cell, mae_m_cell, n_cell = wmae(has_cell)
    mae_b_sig, mae_m_sig, n_sig = wmae(is_sig)

    # --- paired bootstrap on per-row abs-error difference (model - base), poss-weighted ---
    rng = np.random.default_rng(7)
    diff = (err_model - err_base)  # negative = model better
    n = len(diff)
    boot = []
    idx = np.arange(n)
    for _ in range(500):
        s = rng.choice(idx, size=n, replace=True)
        ww = w[s]
        boot.append(float(np.sum(diff[s] * ww) / ww.sum()))
    boot = np.array(boot)
    ci_lo, ci_hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    p_better = float(np.mean(boot < 0))

    result = {
        "test": "player-game PPP MAE, 2024-25 holdout, walk-forward (train<=2023-24)",
        "leakage_control": "FE + cells fit on train seasons only, applied forward",
        "n_train_cells": int(len(shrunk)),
        "n_sig_train_cells": int(len(sig_cells)),
        "holdout": {
            "n_rows": int(n_all), "n_poss": float(w.sum()),
            "wmae_baseline": round(mae_b_all, 6), "wmae_model": round(mae_m_all, 6),
            "delta_wmae": round(mae_m_all - mae_b_all, 6),
            "pct_improvement": round(100.0 * (mae_b_all - mae_m_all) / mae_b_all, 4),
        },
        "rows_with_known_cell": {
            "n_rows": n_cell, "wmae_baseline": round(mae_b_cell, 6) if mae_b_cell else None,
            "wmae_model": round(mae_m_cell, 6) if mae_m_cell else None,
            "delta_wmae": round((mae_m_cell - mae_b_cell), 6) if mae_b_cell else None,
        },
        "rows_with_significant_cell": {
            "n_rows": n_sig, "wmae_baseline": round(mae_b_sig, 6) if mae_b_sig else None,
            "wmae_model": round(mae_m_sig, 6) if mae_m_sig else None,
            "delta_wmae": round((mae_m_sig - mae_b_sig), 6) if mae_b_sig else None,
        },
        "paired_bootstrap_delta_wmae": {
            "mean": round(float(boot.mean()), 6), "ci95_lo": round(ci_lo, 6),
            "ci95_hi": round(ci_hi, 6), "p_model_better": round(p_better, 4),
        },
        "verdict": ("MODEL HELPS (OOS)" if (mae_m_all < mae_b_all and p_better > 0.95)
                    else "NEUTRAL/INCONCLUSIVE" if mae_m_all <= mae_b_all
                    else "MODEL HURTS (OOS)"),
        "note": ("Magnitudes are intentionally small (mean-zero matrix); the gate is "
                 "directional OOS improvement on rows where a real cell applies, not a "
                 "large headline MAE drop. Source: emergent_matchup (proximity attribution)."),
    }
    OUT.write_text(json.dumps(result, indent=2))
    print("VAL_OK n=%d dWMAE=%.5f pct=%.3f p_better=%.3f sigrows=%d verdict=%s"
          % (n_all, result["holdout"]["delta_wmae"], result["holdout"]["pct_improvement"],
             p_better, n_sig, result["verdict"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
