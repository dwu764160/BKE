"""
scripts/fit_archetype_interactions_v2.py
=============================================================================
Simulation Core — Step 1 (Matchup Engine), part 1 of 3.

Promote the FE-adjusted cross-team archetype interaction cells (from
scripts/test_archetype_interactions.py, stored in
reports/archetype_interaction_signal_test.json) to a SHRUNK, mean-zero
production matrix artifact.

Two responsibilities:
  1. apply_eb_shrinkage(cells)      — empirical-Bayes shrinkage toward 0 using
     per-cell standard error (derived from the 95% CI, which itself encodes
     cell possessions), then enforce Σ_D interaction[A,D]·poss = 0 per
     offensive archetype A (the double-counting guard from the design note).
  2. estimate_fe_cells(...)         — reusable two-way (off-player + def-player)
     fixed-effects estimator over raw matchup rows, parameterized by a season
     filter. Used by the walk-forward validation (validate_matchup_interactions.py)
     to fit a *train-only* matrix; NOT used to build the production artifact
     (the production artifact shrinks the already-locked 8-season cells, per the
     Resolved Design — "do not re-run the pre-flight").

Output:
  data/processed/bke/cross_team_interactions_matrix.parquet
    one row per (off_arch, def_arch) cell with shrunk value + provenance.

Design ref: docs/plans/cross_team_interaction_model.md  "Resolved Design".
Data caveat: every cell is source='emergent_matchup' (closest-defender
proximity attribution, Second Spectrum), never intentional assignment.

Usage:
  python3 scripts/fit_archetype_interactions_v2.py
=============================================================================
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SIGNAL_JSON = ROOT / "reports" / "archetype_interaction_signal_test.json"
MATRIX_OUT = ROOT / "data" / "processed" / "bke" / "cross_team_interactions_matrix.parquet"
MATCHUP_PARQUET = ROOT / "data" / "matchup" / "league_season_matchups.parquet"

Z95 = 1.959963985
SOURCE_TAG = "emergent_matchup"


# ---------------------------------------------------------------------------
# Part 1 — shrinkage of the locked FE cells
# ---------------------------------------------------------------------------
def load_signal_cells(path: Path = SIGNAL_JSON) -> pd.DataFrame:
    """Load the locked FE-adjusted cells from the signal-test report."""
    payload = json.loads(Path(path).read_text())
    cells = payload["cells"]
    df = pd.DataFrame(cells)
    # Normalize expected columns; tolerate older key spellings.
    rename = {"interaction_ppp": "raw_ppp"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    for col in ("ci_lo", "ci_hi", "poss", "raw_ppp"):
        if col not in df.columns:
            raise KeyError(f"signal cells missing required column '{col}'")
    df["poss"] = pd.to_numeric(df["poss"], errors="coerce").fillna(0.0)
    df["raw_ppp"] = pd.to_numeric(df["raw_ppp"], errors="coerce").fillna(0.0)
    return df


def _se_from_ci(ci_lo: pd.Series, ci_hi: pd.Series, poss: pd.Series) -> pd.Series:
    """Per-cell standard error from the 95% CI half-width.

    The CI half-width already encodes possessions (more poss -> tighter CI),
    so an SE-based EB weight implements 'shrinkage by cell possessions'.
    Falls back to a poss^-0.5 scaled SE if a CI is degenerate/missing.
    """
    se = (pd.to_numeric(ci_hi, errors="coerce") - pd.to_numeric(ci_lo, errors="coerce")) / (2.0 * Z95)
    se = se.abs()
    bad = ~np.isfinite(se) | (se <= 0)
    if bad.any():
        # crude fallback: typical per-possession sd ~1.05 ppp
        fallback = 1.05 / np.sqrt(np.maximum(poss.to_numpy(dtype=float), 1.0))
        se = se.where(~bad, pd.Series(fallback, index=se.index))
    return se


def apply_eb_shrinkage(cells: pd.DataFrame) -> pd.DataFrame:
    """Empirical-Bayes shrink raw FE cells toward 0, then enforce mean-zero
    per offensive archetype (poss-weighted).

    EB weight  w_c = tau2 / (tau2 + se_c^2),  prior mean = 0.
    tau2 estimated by method-of-moments from the poss-weighted spread of raw
    cells minus the mean sampling variance (clamped >= 0).
    """
    df = cells.copy()
    se = _se_from_ci(df["ci_lo"], df["ci_hi"], df["poss"])
    df["se"] = se.to_numpy(dtype=float)

    raw = df["raw_ppp"].to_numpy(dtype=float)
    w_poss = df["poss"].to_numpy(dtype=float)
    w_poss = np.where(np.isfinite(w_poss) & (w_poss > 0), w_poss, 0.0)
    wsum = w_poss.sum()

    if wsum <= 0:
        weighted_var = float(np.var(raw)) if len(raw) else 0.0
    else:
        wmean = float(np.sum(raw * w_poss) / wsum)
        weighted_var = float(np.sum(w_poss * (raw - wmean) ** 2) / wsum)
    mean_samp_var = float(np.mean(df["se"].to_numpy(dtype=float) ** 2))
    tau2 = max(weighted_var - mean_samp_var, 1e-9)

    shrink_factor = tau2 / (tau2 + df["se"].to_numpy(dtype=float) ** 2)
    df["shrink_factor"] = shrink_factor
    df["shrunk_ppp"] = raw * shrink_factor

    # --- enforce mean-zero per offensive archetype (poss-weighted), vectorized ---
    pw = df["poss"].clip(lower=0.0)
    wv = df["shrunk_ppp"] * pw
    denom = df.groupby("off_arch")["poss"].transform(lambda s: s.clip(lower=0.0).sum())
    m = df.assign(_wv=wv).groupby("off_arch")["_wv"].transform("sum") / denom.replace(0.0, np.nan)
    m = m.fillna(0.0)
    df["off_arch_meanzero_shift"] = -m
    df["shrunk_ppp"] = df["shrunk_ppp"] - m

    df["interaction_ppp"] = df["shrunk_ppp"]  # canonical column
    df["source"] = SOURCE_TAG
    df["tau2"] = tau2
    keep = [
        "off_arch", "def_arch", "interaction_ppp", "raw_ppp", "shrunk_ppp",
        "se", "shrink_factor", "off_arch_meanzero_shift", "ci_lo", "ci_hi",
        "poss", "tau2", "source",
    ]
    for opt in ("n_off_players", "n_def_players", "p", "sig_95", "naive_delta"):
        if opt in df.columns:
            keep.append(opt)
    return df[keep].sort_values(["off_arch", "def_arch"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Part 2 — reusable two-way FE estimator (for walk-forward validation only)
# ---------------------------------------------------------------------------
def estimate_fe_cells(
    matchups: pd.DataFrame,
    off_arch_of: Dict[Tuple[str, str], str],
    def_arch_of: Dict[Tuple[str, str], str],
    seasons: Optional[List[str]] = None,
    min_partial_poss: float = 2.0,
    min_cell_poss: float = 200.0,
    demean_iters: int = 12,
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float], float]:
    """Two-way (off-player + def-player) weighted FE on raw matchup rows.

    Returns (cells_df, off_fe, def_fe, grand_mean). cells_df has the same
    schema fields used by apply_eb_shrinkage (off_arch, def_arch, raw_ppp,
    ci_lo, ci_hi, poss). Archetypes are looked up by (player_id, season);
    rows whose off/def player has no archetype are kept for FE estimation
    but excluded from archetype cell aggregation.
    """
    df = matchups.copy()
    df.columns = [c.upper() for c in df.columns]
    df["SEASON"] = df["SEASON"].astype(str)
    if seasons is not None:
        df = df[df["SEASON"].isin(set(seasons))]
    df["OFF_PLAYER_ID"] = df["OFF_PLAYER_ID"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["DEF_PLAYER_ID"] = df["DEF_PLAYER_ID"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["PARTIAL_POSS"] = pd.to_numeric(df["PARTIAL_POSS"], errors="coerce")
    df["PLAYER_PTS"] = pd.to_numeric(df["PLAYER_PTS"], errors="coerce")
    df = df[(df["PARTIAL_POSS"] >= min_partial_poss) & df["PLAYER_PTS"].notna()]
    if df.empty:
        return pd.DataFrame(columns=["off_arch", "def_arch", "raw_ppp", "ci_lo", "ci_hi", "poss"]), {}, {}, 0.0

    df["ppp"] = df["PLAYER_PTS"] / df["PARTIAL_POSS"]
    w = df["PARTIAL_POSS"].to_numpy(dtype=float)
    y = df["ppp"].to_numpy(dtype=float)

    off_key = df["OFF_PLAYER_ID"].to_numpy()
    def_key = df["DEF_PLAYER_ID"].to_numpy()

    grand = float(np.sum(y * w) / np.sum(w))
    resid = y - grand
    off_fe: Dict[str, float] = {}
    def_fe: Dict[str, float] = {}
    off_idx = pd.Series(np.arange(len(df)), index=df.index).groupby(off_key).groups
    def_idx = pd.Series(np.arange(len(df)), index=df.index).groupby(def_key).groups
    off_groups = {k: np.asarray(v) for k, v in pd.DataFrame({"k": off_key}).groupby("k").groups.items()}
    def_groups = {k: np.asarray(v) for k, v in pd.DataFrame({"k": def_key}).groupby("k").groups.items()}

    for _ in range(demean_iters):
        # off effects
        for k, ix in off_groups.items():
            wi = w[ix]
            off_fe[k] = float(np.sum(resid[ix] * wi) / np.sum(wi))
        resid = resid - np.array([off_fe[k] for k in off_key])
        # def effects
        for k, ix in def_groups.items():
            wi = w[ix]
            def_fe[k] = float(np.sum(resid[ix] * wi) / np.sum(wi))
        resid = resid - np.array([def_fe[k] for k in def_key])

    df["_resid"] = resid
    seasons_arr = df["SEASON"].to_numpy()
    df["_off_arch"] = [off_arch_of.get((p, s)) for p, s in zip(off_key, seasons_arr)]
    df["_def_arch"] = [def_arch_of.get((p, s)) for p, s in zip(def_key, seasons_arr)]
    cell_df = df.dropna(subset=["_off_arch", "_def_arch"])

    rows = []
    for (oa, da), g in cell_df.groupby(["_off_arch", "_def_arch"]):
        wi = g["PARTIAL_POSS"].to_numpy(dtype=float)
        ri = g["_resid"].to_numpy(dtype=float)
        poss = float(wi.sum())
        if poss < min_cell_poss:
            continue
        mean = float(np.sum(ri * wi) / poss)
        # weighted SE of the mean
        var = float(np.sum(wi * (ri - mean) ** 2) / poss)
        neff = (wi.sum() ** 2) / np.sum(wi ** 2)
        se = float(np.sqrt(max(var, 0.0) / max(neff, 1.0)))
        rows.append({
            "off_arch": oa, "def_arch": da, "raw_ppp": round(mean, 6),
            "ci_lo": round(mean - Z95 * se, 6), "ci_hi": round(mean + Z95 * se, 6),
            "poss": poss, "n_off_players": int(g["OFF_PLAYER_ID"].nunique()),
            "n_def_players": int(g["DEF_PLAYER_ID"].nunique()),
        })
    return pd.DataFrame(rows), off_fe, def_fe, grand


# ---------------------------------------------------------------------------
# main — build the production (full-data) shrunk matrix
# ---------------------------------------------------------------------------
def main() -> int:
    if not SIGNAL_JSON.exists():
        print(f"FATAL missing {SIGNAL_JSON}")
        return 1
    cells = load_signal_cells()
    shrunk = apply_eb_shrinkage(cells)
    MATRIX_OUT.parent.mkdir(parents=True, exist_ok=True)
    shrunk.to_parquet(MATRIX_OUT, index=False)

    n = len(shrunk)
    raw_std = float(np.std(shrunk["raw_ppp"]))
    shr_std = float(np.std(shrunk["interaction_ppp"]))
    # report mean-zero residual per off arch (should be ~0)
    _tmp = shrunk.assign(wv=shrunk["interaction_ppp"] * shrunk["poss"])
    chk = _tmp.groupby("off_arch")["wv"].sum() / _tmp.groupby("off_arch")["poss"].sum().clip(lower=1.0)
    max_abs_meanzero = float(chk.abs().max())
    summary = {
        "artifact": str(MATRIX_OUT.relative_to(ROOT)),
        "n_cells": n,
        "raw_std_ppp": round(raw_std, 5),
        "shrunk_std_ppp": round(shr_std, 5),
        "mean_shrink_factor": round(float(shrunk["shrink_factor"].mean()), 4),
        "tau2": round(float(shrunk["tau2"].iloc[0]), 8),
        "max_abs_poss_weighted_mean_per_off_arch": round(max_abs_meanzero, 8),
        "source": SOURCE_TAG,
    }
    (ROOT / "reports" / "cross_team_interactions_matrix.json").write_text(json.dumps(summary, indent=2))
    print("FIT_OK cells=%d shr_std=%.4f shrink=%.3f meanzero_resid=%.2e"
          % (n, shr_std, summary["mean_shrink_factor"], max_abs_meanzero))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
