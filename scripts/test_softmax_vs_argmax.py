"""
scripts/test_softmax_vs_argmax.py
=============================================================================
Step 1 SOFTMAX TEST — Does softmax-weighted interaction prediction better
explain FE-residual PPP than argmax archetype lookup?

Method:
  1. Recompute FE residuals on the 8-season matchup data (same as preflight).
  2. For each (OFF_PLAYER × DEF_PLAYER × SEASON) pair, compute:
       argmax_pred  = cell[off_primary_arch][def_primary_arch]
       softmax_off  = Σ_i off_prob_i × cell[off_arch_i][def_primary_arch]
       softmax_both = Σ_i Σ_j off_prob_i × def_prob_j × cell[off_arch_i][def_arch_j]
                      (normalised over the 7 def archs with prob columns; 2 missing handled below)
  3. Aggregate to player-pair level to reduce possession-level noise.
  4. Report: Pearson r, R², mean absolute error, fraction of pairs improved.
  5. Stratify by softmax concentration (flat vs concentrated priors).

Output: reports/softmax_vs_argmax_test.json
=============================================================================
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MATCHUPS  = ROOT / "data/matchup/league_season_matchups.parquet"
PROFILES  = ROOT / "data/processed/player_eval/player_impact_profiles.parquet"
INT_REPORT = ROOT / "reports/archetype_interaction_signal_test.json"
REPORT     = ROOT / "reports/softmax_vs_argmax_test.json"

MIN_POSS   = 2.0
DEMEAN_ITERS = 12
DROP_ARCH  = {"Insufficient Minutes", "Unknown", None}

# ---------- archetype → profile column mappings ----------
OFF_PROB_MAP = {
    "All-Around Scorer":          "off_prob_emb_all_around_scorer",
    "Ball Dominant Creator":      "off_prob_emb_ball_dominant_creator",
    "Ballhandler":                "off_prob_emb_ballhandler",
    "Connector":                  "off_prob_emb_connector",
    "Interior Scorer":            "off_prob_emb_interior_scorer",
    "Off-Ball Finisher":          "off_prob_emb_off_ball_finisher",
    "Off-Ball Movement Shooter":  "off_prob_emb_off_ball_movement_shooter",
    "Off-Ball Stationary Shooter":"off_prob_emb_off_ball_stationary_shooter",
    "Perimeter Scorer":           "off_prob_emb_perimeter_scorer",
    "PnR Popping Big":            "off_prob_emb_pnr_popping_big",
    "PnR Rolling Big":            "off_prob_emb_pnr_rolling_big",
}

DEF_PROB_MAP = {
    "Dropping Big":     "def_prob_drop_big_score",
    "Mobile Big":       "def_prob_mobile_big_score",
    "Off-Ball Chaser":  "def_prob_chaser_score",
    "POA Defender":     "def_prob_poa_score",
    "Rim Protector":    "def_prob_rim_score",
    "Versatile Defender":"def_prob_versatile_score",
    "Wing Stopper":     "def_prob_wing_score",
    # "Low-Activity Defender" and "Rotational Defender" are MISSING from profiles.
    # Their probability mass is captured in the residual below.
}
MISSING_DEF = ["Low-Activity Defender", "Rotational Defender"]

OFF_ARCHS = list(OFF_PROB_MAP.keys())   # 11 — complete
DEF_ARCHS_KNOWN = list(DEF_PROB_MAP.keys())   # 7 of 9


def _norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()


def load_matchups() -> pd.DataFrame:
    m = pd.read_parquet(MATCHUPS, columns=[
        "off_player_id", "def_player_id", "partial_poss", "player_pts", "season"])
    m["off_player_id"] = _norm(m["off_player_id"])
    m["def_player_id"] = _norm(m["def_player_id"])
    m["season"] = m["season"].astype(str)
    m = m[m["partial_poss"] >= MIN_POSS].copy()
    m["ppp"] = m["player_pts"] / m["partial_poss"]
    return m.reset_index(drop=True)


def load_profiles() -> pd.DataFrame:
    off_cols = list(OFF_PROB_MAP.values())
    def_cols = list(DEF_PROB_MAP.values())
    arch_cols = ["off_primary_archetype", "def_primary_archetype", "off_role_confidence"]
    prof = pd.read_parquet(PROFILES, columns=["player_id", "season"] + arch_cols + off_cols + def_cols)
    prof["player_id"] = _norm(prof["player_id"])
    prof["season"] = prof["season"].astype(str)
    return prof


def two_way_demean(df, val, w, g1, g2, iters):
    r = df[val].to_numpy(dtype=float).copy()
    weight = df[w].to_numpy(dtype=float)
    ia = pd.factorize(df[g1].to_numpy())[0]
    ib = pd.factorize(df[g2].to_numpy())[0]
    na, nb = ia.max() + 1, ib.max() + 1
    for _ in range(iters):
        sw = np.bincount(ia, weights=weight, minlength=na)
        swr = np.bincount(ia, weights=weight * r, minlength=na)
        r -= (swr / np.maximum(sw, 1e-12))[ia]
        sw = np.bincount(ib, weights=weight, minlength=nb)
        swr = np.bincount(ib, weights=weight * r, minlength=nb)
        r -= (swr / np.maximum(sw, 1e-12))[ib]
    return r


def build_cell_matrix(report: dict) -> np.ndarray:
    """Return (11 off × 9 def) matrix of interaction_ppp, index order = OFF_ARCHS × all def archs."""
    all_def = DEF_ARCHS_KNOWN + MISSING_DEF   # 9 total
    idx_off = {a: i for i, a in enumerate(OFF_ARCHS)}
    idx_def = {a: i for i, a in enumerate(all_def)}
    mat = np.zeros((len(OFF_ARCHS), len(all_def)))
    for c in report["cells"]:
        i = idx_off.get(c["off_arch"])
        j = idx_def.get(c["def_arch"])
        if i is not None and j is not None:
            mat[i, j] = c["interaction_ppp"]
    return mat, all_def


def compute_predictions(df_pair, prof, mat, all_def_archs):
    """
    df_pair: one row per (OFF_PLAYER_ID, DEF_PLAYER_ID, SEASON).
    Returns df with argmax_pred, softmax_off_pred, softmax_both_pred columns.
    """
    off_cols = [OFF_PROB_MAP[a] for a in OFF_ARCHS]
    def_cols = [DEF_PROB_MAP[a] for a in DEF_ARCHS_KNOWN]

    # Merge off player probs
    off_prof = prof.rename(columns={
        "player_id": "off_player_id",
        "off_primary_archetype": "off_arch",
        "def_primary_archetype": "off_def_arch_unused",
        "off_role_confidence": "off_confidence",
        **{v: f"op_{v}" for v in off_cols},
    })
    # Merge def player probs (use their DEF archetype label + def prob columns)
    def_prof = prof[["player_id", "season", "def_primary_archetype"] + def_cols].rename(columns={
        "player_id": "def_player_id",
        "def_primary_archetype": "def_arch",
    })

    df = df_pair.merge(off_prof[["off_player_id", "season", "off_arch", "off_confidence"] +
                                  [f"op_{v}" for v in off_cols]],
                       on=["off_player_id", "season"], how="left")
    df = df.merge(def_prof[["def_player_id", "season", "def_arch"] + def_cols],
                  on=["def_player_id", "season"], how="left")
    df = df.dropna(subset=["off_arch", "def_arch"]).copy()
    df = df[~df["off_arch"].isin(DROP_ARCH) & ~df["def_arch"].isin(DROP_ARCH)]

    idx_off = {a: i for i, a in enumerate(OFF_ARCHS)}
    idx_def = {a: i for i, a in enumerate(all_def_archs)}

    # argmax prediction
    def argmax_pred(row):
        i = idx_off.get(row["off_arch"])
        j = idx_def.get(row["def_arch"])
        if i is None or j is None:
            return np.nan
        return mat[i, j]

    df["argmax_pred"] = df.apply(argmax_pred, axis=1)

    # softmax_off: Σ_i off_prob_i × mat[i, argmax_def]
    op_cols_renamed = [f"op_{v}" for v in off_cols]
    def_j_arr = df["def_arch"].map(idx_def).values

    off_probs = df[op_cols_renamed].to_numpy(dtype=float)
    # Normalise row-wise (handle players with probs not summing to 1)
    row_sums = off_probs.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-9, 1.0, row_sums)
    off_probs_norm = off_probs / row_sums

    softmax_off = np.array([
        np.dot(off_probs_norm[r], mat[:, def_j_arr[r]])
        if def_j_arr[r] is not None and not np.isnan(def_j_arr[r])
        else np.nan
        for r in range(len(df))
    ])
    df["softmax_off_pred"] = softmax_off

    # softmax_both: Σ_i Σ_j off_prob_i × def_prob_j_norm × mat[i,j]
    # Only over 7 known def archs; normalise def probs to sum=1 over those 7
    def_probs = df[def_cols].to_numpy(dtype=float)
    def_row_sums = def_probs.sum(axis=1, keepdims=True)
    def_row_sums = np.where(def_row_sums < 1e-9, 1.0, def_row_sums)
    def_probs_norm = def_probs / def_row_sums   # (N, 7), normalised over known archs

    # Submatrix: only the 7 known def archs
    known_def_idx = [idx_def[a] for a in DEF_ARCHS_KNOWN]
    mat_sub = mat[:, known_def_idx]   # (11, 7)

    # softmax_both[r] = off_probs_norm[r] @ mat_sub @ def_probs_norm[r]
    softmax_both = np.einsum('ri,ij,rj->r', off_probs_norm, mat_sub, def_probs_norm)
    df["softmax_both_pred"] = softmax_both

    return df


def evaluate(pred_col, actual_col, poss_col, df, label):
    mask = df[pred_col].notna() & df[actual_col].notna()
    d = df[mask]
    act = d[actual_col].to_numpy()
    pred = d[pred_col].to_numpy()
    w = d[poss_col].to_numpy()
    # Pearson r (unweighted for interpretability, weighted as supplemental)
    r = np.corrcoef(pred, act)[0, 1]
    r_w = np.cov(pred, act, aweights=w)[0, 1] / (np.std(pred) * np.std(act) + 1e-12)
    mae = np.mean(np.abs(pred - act))
    mae_w = np.average(np.abs(pred - act), weights=w)
    r2 = 1 - np.sum((act - pred)**2) / (np.sum((act - np.mean(act))**2) + 1e-12)
    return {
        "label": label,
        "n_pairs": int(mask.sum()),
        "pearson_r": round(float(r), 5),
        "pearson_r_wtd": round(float(r_w), 5),
        "r2": round(float(r2), 5),
        "mae": round(float(mae), 6),
        "mae_wtd": round(float(mae_w), 6),
    }


def main():
    report_int = json.loads(INT_REPORT.read_text())
    mat, all_def_archs = build_cell_matrix(report_int)

    print("Loading matchup data...")
    m = load_matchups()
    prof = load_profiles()

    # Join archetypes onto matchup for FE demean (same as preflight)
    arch_join = prof[["player_id", "season", "off_primary_archetype", "def_primary_archetype"]].copy()
    off_a = arch_join.rename(columns={"player_id": "off_player_id",
                                       "off_primary_archetype": "off_arch"}).drop(columns="def_primary_archetype")
    def_a = arch_join.rename(columns={"player_id": "def_player_id",
                                       "def_primary_archetype": "def_arch"}).drop(columns="off_primary_archetype")
    m = m.merge(off_a, on=["off_player_id", "season"], how="left")
    m = m.merge(def_a, on=["def_player_id", "season"], how="left")
    m = m[~m["off_arch"].isin(DROP_ARCH) & ~m["def_arch"].isin(DROP_ARCH)].dropna(subset=["off_arch","def_arch"])
    print(f"Matchup rows after arch join: {len(m):,}")

    print("Computing FE residuals...")
    m["resid"] = two_way_demean(m, "ppp", "partial_poss", "off_player_id", "def_player_id", DEMEAN_ITERS)

    # Aggregate to player-pair level (reduce possession noise)
    pair = (m.groupby(["off_player_id", "def_player_id", "season"])
              .apply(lambda x: pd.Series({
                  "actual_resid": np.average(x["resid"], weights=x["partial_poss"]),
                  "poss": x["partial_poss"].sum(),
              }), include_groups=False)
              .reset_index())
    print(f"Unique player-pairs: {len(pair):,}")

    print("Computing predictions...")
    pair_pred = compute_predictions(pair, prof, mat, all_def_archs)
    print(f"Pairs with complete predictions: {pair_pred['argmax_pred'].notna().sum():,}")

    # --- global evaluation ---
    results = []
    for col, label in [
        ("argmax_pred",      "argmax (baseline)"),
        ("softmax_off_pred", "softmax-off × argmax-def"),
        ("softmax_both_pred","softmax-off × softmax-def (7/9 def archs)"),
    ]:
        r = evaluate(col, "actual_resid", "poss", pair_pred, label)
        results.append(r)
        print(f"  {label:40s}  r={r['pearson_r']:+.5f}  R²={r['r2']:+.5f}  MAE={r['mae']:.5f}")

    # --- stratified by off_role_confidence ---
    pair_pred2 = pair_pred.merge(
        prof[["player_id", "season", "off_role_confidence"]].rename(
            columns={"player_id": "off_player_id"}),
        on=["off_player_id", "season"], how="left")

    print("\n--- By off_role_confidence tier ---")
    strata_results = []
    for label, lo, hi in [("low (<0.4)", 0.0, 0.4), ("mid (0.4-0.7)", 0.4, 0.7), ("high (>0.7)", 0.7, 1.1)]:
        sub = pair_pred2[pair_pred2["off_role_confidence"].between(lo, hi, inclusive="left")]
        if len(sub) < 50:
            continue
        tier = {"tier": label, "n_pairs": len(sub)}
        for col, clabel in [
            ("argmax_pred", "argmax"),
            ("softmax_off_pred", "softmax_off"),
            ("softmax_both_pred", "softmax_both"),
        ]:
            r = np.corrcoef(sub[col].fillna(0), sub["actual_resid"].fillna(0))[0, 1]
            tier[f"r_{clabel}"] = round(float(r), 5)
        strata_results.append(tier)
        print(f"  {label:20s} n={len(sub):6,}  argmax r={tier['r_argmax']:+.5f}  "
              f"smax_off r={tier['r_softmax_off']:+.5f}  smax_both r={tier['r_softmax_both']:+.5f}")

    # --- compare per-pair: which method is closer to actual? ---
    mask = pair_pred[["argmax_pred","softmax_off_pred","softmax_both_pred","actual_resid"]].notna().all(axis=1)
    d = pair_pred[mask].copy()
    d["err_argmax"] = (d["argmax_pred"] - d["actual_resid"]).abs()
    d["err_smax_off"] = (d["softmax_off_pred"] - d["actual_resid"]).abs()
    d["err_smax_both"] = (d["softmax_both_pred"] - d["actual_resid"]).abs()
    frac_smax_off_wins  = (d["err_smax_off"] < d["err_argmax"]).mean()
    frac_smax_both_wins = (d["err_smax_both"] < d["err_argmax"]).mean()
    print(f"\nFraction of pairs where softmax_off < argmax error: {frac_smax_off_wins:.3f}")
    print(f"Fraction of pairs where softmax_both < argmax error: {frac_smax_both_wins:.3f}")

    out = {
        "description": "Softmax vs argmax interaction prediction comparison",
        "n_pairs_total": int(len(pair)),
        "n_pairs_evaluated": int(mask.sum()),
        "global": results,
        "by_off_role_confidence": strata_results,
        "frac_pairs_softmax_off_wins": round(float(frac_smax_off_wins), 4),
        "frac_pairs_softmax_both_wins": round(float(frac_smax_both_wins), 4),
        "notes": {
            "missing_def_archs": MISSING_DEF,
            "softmax_both_normalises": "def probs normalised over 7 known archs only",
            "aggregation": "player-pair level (weighted-mean residual over possessions)",
        },
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {REPORT}")


if __name__ == "__main__":
    main()
