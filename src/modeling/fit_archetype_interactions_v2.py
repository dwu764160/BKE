"""
fit_archetype_interactions_v2.py — Phase 4.6 v2

Lineup-level Lasso fit to identify a SPARSE set of empirically meaningful archetype
interaction pairs (vs. the v1 dense 66-pair matrix that was hand-coded then weakly
identified by Ridge on 240 team-seasons).

Pipeline:
  1. Lineup-level data from data/processed/metrics_lineups.parquet (~29k stints).
     Filter to total_poss >= 100 (~2,390 lineups, ORTG std drops from 37 to 14.7).
  2. Per-lineup features:
       - 66 archetype-pair counts (cross + same), one per unordered pair.
       - lineup_talent = sum of oRAPM over 5 players.
  3. Outcome: lineup ORTG, season-mean centered.
  4. LassoCV with talent-residualized features for sparsity selection.
  5. Forward stepwise verification on candidate set; keep pairs that improve CV R²
     by >= 0.0005.
  6. Bootstrap 95% CI (500 samples) on surviving coefs.
  7. Rescale V_AB = beta / 75 (pipeline math; see docs/plans/archetype_validation_plan.md
     §4.6).

Output: reports/archetype_interaction_fit_v2.json — sparse set ready for review.
Does NOT auto-update INTERACTION_MATRIX. Manual basketball review first.

Run:
    python3 src/modeling/fit_archetype_interactions_v2.py
"""

import argparse
import json
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, Ridge, LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────────────────────
# Paths & constants
# ─────────────────────────────────────────────────────────────────────────────
LINEUPS_PATH = Path("data/processed/metrics_lineups.parquet")
ARCHETYPES_PATH = Path("data/processed/player_archetypes.parquet")
AGGREGATE_PATH = Path("aggregate/player_profile_aggregate.parquet")
OUTPUT_PATH = Path("reports/archetype_interaction_fit_v2.json")

MIN_POSSESSIONS = 100
N_BOOTSTRAP = 500
STEPWISE_DELTA_R2 = 0.0005
RANDOM_SEED = 42

# Pipeline math: V_AB applied as V_AB * S_A * S_B * (1/0.2) * 0.75 * TEAM_SCALE
# 1/0.2 = baseline pair weight inverse (team_feature_aggregation.py:382)
# 0.75  = INTERACTION_LAMBDA  (constants.py:100)
# 20    = DEFAULT_TEAM_SCALE   (constants.py:91)
PIPELINE_SCALE = (1.0 / 0.2) * 0.75 * 20.0   # = 75

OFFENSIVE_ARCHETYPES = [
    "Ball Dominant Creator",
    "All-Around Scorer",
    "Ballhandler",
    "Interior Scorer",
    "Perimeter Scorer",
    "PnR Rolling Big",
    "PnR Popping Big",
    "Off-Ball Finisher",
    "Off-Ball Movement Shooter",
    "Off-Ball Stationary Shooter",
    "Connector",
]

ARCH_KEY = {
    "Ball Dominant Creator":          "ball_dominant_creator",
    "All-Around Scorer":              "all_around_scorer",
    "Ballhandler":                    "ballhandler",
    "Interior Scorer":                "interior_scorer",
    "Perimeter Scorer":               "perimeter_scorer",
    "PnR Rolling Big":                "pnr_rolling_big",
    "PnR Popping Big":                "pnr_popping_big",
    "Off-Ball Finisher":              "off_ball_finisher",
    "Off-Ball Movement Shooter":      "off_ball_movement_shooter",
    "Off-Ball Stationary Shooter":    "off_ball_stationary_shooter",
    "Connector":                      "connector",
}
ARCH_KEYS = [ARCH_KEY[a] for a in OFFENSIVE_ARCHETYPES]

# 66 unordered pairs (canonical order)
ALL_PAIRS = list(combinations_with_replacement(ARCH_KEYS, 2))
PAIR_COLS = [f"{a}__{b}" for a, b in ALL_PAIRS]


# ─────────────────────────────────────────────────────────────────────────────
# Load & build lineup feature matrix
# ─────────────────────────────────────────────────────────────────────────────

def load_player_arch_map() -> pd.DataFrame:
    arche = pd.read_parquet(ARCHETYPES_PATH)
    # Keep player_id, season, primary_archetype only
    cols = {"PLAYER_ID": "player_id", "SEASON": "season",
            "primary_archetype": "primary_archetype"}
    arche = arche.rename(columns=cols)[list(cols.values())]
    arche["player_id"] = arche["player_id"].astype(np.int64)
    return arche[arche["primary_archetype"].isin(OFFENSIVE_ARCHETYPES)]


def load_player_talent() -> pd.DataFrame:
    """Load oRAPM per (player_id, season). Z-score within season to handle inconsistent
    cross-season calibration (2019-20 to 2021-22 use a compressed RAPM scale)."""
    agg = pd.read_parquet(AGGREGATE_PATH)
    talent = agg[["player_id", "season", "orapm"]].copy()
    talent["player_id"] = pd.to_numeric(talent["player_id"], errors="coerce")
    talent = talent.dropna(subset=["player_id"])
    talent["player_id"] = talent["player_id"].astype(np.int64)
    talent["orapm"] = pd.to_numeric(talent["orapm"], errors="coerce")
    # Impute missing with season median, then z-score within season
    season_med = talent.groupby("season")["orapm"].transform("median")
    talent["orapm"] = talent["orapm"].fillna(season_med)
    season_mean = talent.groupby("season")["orapm"].transform("mean")
    season_std = talent.groupby("season")["orapm"].transform("std").replace(0, 1.0)
    talent["orapm"] = (talent["orapm"] - season_mean) / season_std
    return talent.drop_duplicates(subset=["player_id", "season"])


def build_lineup_features(lineups: pd.DataFrame,
                          arche_map: pd.DataFrame,
                          talent_map: pd.DataFrame) -> pd.DataFrame:
    """Build per-lineup feature matrix: 66 pair counts + lineup_talent + ORTG outcome."""

    # Build lookup dicts for fast access
    a_lookup = {(int(r.player_id), r.season): r.primary_archetype
                for r in arche_map.itertuples()}
    t_lookup = {(int(r.player_id), r.season): float(r.orapm) if pd.notna(r.orapm) else np.nan
                for r in talent_map.itertuples()}

    rows = []
    n_drop_unclassified = 0
    n_drop_missing_talent = 0
    n_drop_poss = 0

    for r in lineups.itertuples():
        if r.total_poss < MIN_POSSESSIONS:
            n_drop_poss += 1
            continue
        if not isinstance(r.lineup_ids, (list, np.ndarray)) or len(r.lineup_ids) != 5:
            n_drop_unclassified += 1
            continue

        # Look up archetypes for the 5 players
        season = r.season
        archs = []
        talents = []
        valid = True
        for pid in r.lineup_ids:
            pid_int = int(pid)
            key = (pid_int, season)
            arch = a_lookup.get(key)
            if arch is None or arch not in OFFENSIVE_ARCHETYPES:
                valid = False
                break
            archs.append(ARCH_KEY[arch])
            talents.append(t_lookup.get(key, np.nan))

        if not valid:
            n_drop_unclassified += 1
            continue

        # Count pair occurrences for each of the 66 archetype pairs
        from collections import Counter
        arch_counts = Counter(archs)
        pair_features = {col: 0 for col in PAIR_COLS}
        for a, b in ALL_PAIRS:
            if a == b:
                ca = arch_counts.get(a, 0)
                pair_features[f"{a}__{b}"] = ca * (ca - 1) // 2  # C(n,2)
            else:
                ca = arch_counts.get(a, 0)
                cb = arch_counts.get(b, 0)
                pair_features[f"{a}__{b}"] = ca * cb

        # Lineup talent (sum of 5 oRAPM)
        talents_arr = np.array(talents, dtype=float)
        n_missing = int(np.isnan(talents_arr).sum())
        if n_missing > 0:
            # Impute missing with mean of other 4 in this lineup
            valid_mask = ~np.isnan(talents_arr)
            if valid_mask.sum() == 0:
                n_drop_missing_talent += 1
                continue
            talents_arr[~valid_mask] = talents_arr[valid_mask].mean()
        lineup_talent = float(talents_arr.sum())

        rows.append({
            "season": season,
            "team_name": r.team_name,
            "total_poss": float(r.total_poss),
            "ortg": float(r.ORTG),
            "lineup_talent": lineup_talent,
            "lineup_n_missing_talent": n_missing,
            **pair_features,
        })

    print(f"  Built {len(rows)} usable lineups")
    print(f"  Dropped: {n_drop_poss} (low poss), "
          f"{n_drop_unclassified} (unclassified arch), "
          f"{n_drop_missing_talent} (all 5 missing talent)")
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Fit pipeline
# ─────────────────────────────────────────────────────────────────────────────

def season_center(y: np.ndarray, seasons: np.ndarray) -> np.ndarray:
    """Subtract per-season mean from y."""
    out = y.copy()
    for s in np.unique(seasons):
        mask = seasons == s
        out[mask] -= y[mask].mean()
    return out


def fit_talent_baseline(X_talent: np.ndarray, y: np.ndarray,
                        sample_weight: np.ndarray) -> tuple:
    """Fit y ~ talent only; return (model, residuals, r2)."""
    m = LinearRegression().fit(X_talent, y, sample_weight=sample_weight)
    y_pred = m.predict(X_talent)
    resid = y - y_pred
    r2 = weighted_r2(y, y_pred, sample_weight)
    return m, resid, r2


def weighted_r2(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray) -> float:
    ss_res = float(np.sum(w * (y_true - y_pred) ** 2))
    y_mean = float(np.sum(w * y_true) / np.sum(w))
    ss_tot = float(np.sum(w * (y_true - y_mean) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def lasso_select_candidates(X_pairs_scaled: np.ndarray,
                            resid: np.ndarray,
                            sample_weight: np.ndarray) -> list:
    """LassoCV on talent-residualized pair features. Return indices with |beta|>1e-6."""
    alphas = np.logspace(-4, 1, 50)
    lasso = LassoCV(alphas=alphas, cv=5, max_iter=20000,
                    random_state=RANDOM_SEED).fit(
        X_pairs_scaled, resid, sample_weight=sample_weight
    )
    surviving = np.where(np.abs(lasso.coef_) > 1e-6)[0].tolist()
    print(f"  LassoCV: alpha={lasso.alpha_:.5f}, {len(surviving)} candidate pairs survived")
    return surviving, lasso.alpha_, lasso.coef_


def cv_r2_with_features(X_talent: np.ndarray, X_extra: np.ndarray,
                        y: np.ndarray, w: np.ndarray, n_folds: int = 5) -> float:
    """5-fold CV: return mean weighted R² for [talent + extra] features."""
    if X_extra.shape[1] == 0:
        X_full = X_talent
    else:
        X_full = np.hstack([X_talent, X_extra])
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    r2s = []
    for tr, te in kf.split(X_full):
        m = LinearRegression().fit(X_full[tr], y[tr], sample_weight=w[tr])
        y_pred = m.predict(X_full[te])
        r2s.append(weighted_r2(y[te], y_pred, w[te]))
    return float(np.mean(r2s))


def forward_stepwise_verify(X_talent: np.ndarray, X_pairs_scaled: np.ndarray,
                            y: np.ndarray, w: np.ndarray,
                            candidate_indices: list) -> list:
    """Greedy add candidates by largest CV R² gain; stop when gain < threshold."""
    baseline_r2 = cv_r2_with_features(X_talent, np.zeros((len(y), 0)), y, w)
    print(f"  Baseline CV R² (talent only): {baseline_r2:.4f}")

    kept = []
    remaining = list(candidate_indices)
    current_r2 = baseline_r2
    while remaining:
        best_gain = -np.inf
        best_idx = None
        for cand in remaining:
            test_set = kept + [cand]
            X_test = X_pairs_scaled[:, test_set]
            r2 = cv_r2_with_features(X_talent, X_test, y, w)
            gain = r2 - current_r2
            if gain > best_gain:
                best_gain = gain
                best_idx = cand
                best_r2 = r2
        if best_gain < STEPWISE_DELTA_R2:
            print(f"  Stopping: next-best gain {best_gain:.5f} < threshold {STEPWISE_DELTA_R2}")
            break
        kept.append(best_idx)
        remaining.remove(best_idx)
        a, b = ALL_PAIRS[best_idx]
        current_r2 = best_r2
        print(f"  +{a:<28} × {b:<28} (ΔR²={best_gain:+.5f}, cumR²={current_r2:.4f})")

    return kept, current_r2


def bootstrap_ci(X_talent: np.ndarray, X_kept_scaled: np.ndarray,
                 y: np.ndarray, w: np.ndarray, n_boot: int = N_BOOTSTRAP) -> tuple:
    """Bootstrap 95% CI for coefficients of [talent + kept_pairs]."""
    n = len(y)
    rng = np.random.default_rng(RANDOM_SEED)
    n_features = 1 + X_kept_scaled.shape[1]
    boot_coefs = np.zeros((n_boot, n_features))
    X_full = np.hstack([X_talent, X_kept_scaled])
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        m = LinearRegression().fit(X_full[idx], y[idx], sample_weight=w[idx])
        boot_coefs[b] = m.coef_
    ci_lo = np.percentile(boot_coefs, 2.5, axis=0)
    ci_hi = np.percentile(boot_coefs, 97.5, axis=0)
    return ci_lo, ci_hi


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global MIN_POSSESSIONS
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--min-poss", type=int, default=MIN_POSSESSIONS)
    args = parser.parse_args()
    MIN_POSSESSIONS = args.min_poss

    print("Loading inputs...")
    lineups = pd.read_parquet(LINEUPS_PATH)
    print(f"  Lineups: {len(lineups)} stints, {lineups['season'].nunique()} seasons")
    arche_map = load_player_arch_map()
    print(f"  Archetypes: {len(arche_map)} player-seasons classified")
    talent_map = load_player_talent()
    print(f"  Talent: {len(talent_map)} player-seasons with oRAPM")

    print(f"\nBuilding lineup feature matrix (min_poss={MIN_POSSESSIONS})...")
    data = build_lineup_features(lineups, arche_map, talent_map)

    # Outcome: season-mean-centered ORTG
    y_raw = data["ortg"].values
    seasons = data["season"].values
    y = season_center(y_raw, seasons)

    # Sample weights: sqrt(possessions)
    w = np.sqrt(data["total_poss"].values)

    # Talent feature
    X_talent = data[["lineup_talent"]].values

    # 66 pair features
    X_pairs = data[PAIR_COLS].values
    pair_scaler = StandardScaler()
    X_pairs_scaled = pair_scaler.fit_transform(X_pairs)

    print(f"\nFitting talent baseline...")
    _, resid, r2_talent = fit_talent_baseline(X_talent, y, w)
    print(f"  R² (talent only): {r2_talent:.4f}")

    print(f"\nLassoCV candidate selection...")
    candidates, lasso_alpha, lasso_coefs = lasso_select_candidates(
        X_pairs_scaled, resid, w
    )

    print(f"\nForward stepwise verification (ΔR² threshold = {STEPWISE_DELTA_R2})...")
    if not candidates:
        print("  No candidates from Lasso — no pairs survive.")
        kept, r2_final = [], r2_talent
    else:
        kept, r2_final = forward_stepwise_verify(
            X_talent, X_pairs_scaled, y, w, candidates
        )

    print(f"\nBootstrap CI ({N_BOOTSTRAP} samples) on surviving pairs...")
    if kept:
        X_kept_scaled = X_pairs_scaled[:, kept]
        # Re-fit final model on full data to get final coefs in scaled space
        X_full = np.hstack([X_talent, X_kept_scaled])
        final_m = LinearRegression().fit(X_full, y, sample_weight=w)
        # Unscale pair coefs back to raw feature units
        kept_scales = pair_scaler.scale_[kept]
        pair_betas_raw = final_m.coef_[1:] / kept_scales
        talent_coef = float(final_m.coef_[0])

        ci_lo, ci_hi = bootstrap_ci(X_talent, X_kept_scaled, y, w)
        # ci_lo[0]/ci_hi[0] = talent coef CI (in scaled space, but we don't need it raw)
        pair_ci_lo_raw = ci_lo[1:] / kept_scales
        pair_ci_hi_raw = ci_hi[1:] / kept_scales
    else:
        pair_betas_raw = np.array([])
        talent_coef = 0.0
        pair_ci_lo_raw = np.array([])
        pair_ci_hi_raw = np.array([])

    # Build surviving pairs list
    surviving = []
    for i, idx in enumerate(kept):
        a, b = ALL_PAIRS[idx]
        beta = float(pair_betas_raw[i])
        v_ab = beta / PIPELINE_SCALE
        ci_lo_v = float(pair_ci_lo_raw[i]) / PIPELINE_SCALE
        ci_hi_v = float(pair_ci_hi_raw[i]) / PIPELINE_SCALE
        n_with_pair = int((data[f"{a}__{b}"] > 0).sum())
        ci_spans_zero = (ci_lo_v < 0 < ci_hi_v) or (ci_lo_v > 0 > ci_hi_v)
        surviving.append({
            "a": a,
            "b": b,
            "beta_raw": round(beta, 5),
            "V_AB": round(v_ab, 5),
            "ci_lo_V": round(ci_lo_v, 5),
            "ci_hi_V": round(ci_hi_v, 5),
            "ci_spans_zero": bool(ci_spans_zero),
            "n_lineups_with_pair": n_with_pair,
            "involves_pnr_popping_big": (a == "pnr_popping_big" or b == "pnr_popping_big"),
        })

    results = {
        "method": "lineup_lasso_forward_stepwise",
        "n_lineups": int(len(data)),
        "min_possessions_filter": MIN_POSSESSIONS,
        "seasons": sorted(data["season"].unique().tolist()),
        "talent_control": "sum_orapm_over_5_players",
        "sample_weight": "sqrt(total_poss)",
        "pipeline_scale_factor": PIPELINE_SCALE,
        "lasso_alpha": float(lasso_alpha),
        "lasso_candidate_count": len(candidates),
        "stepwise_delta_r2_threshold": STEPWISE_DELTA_R2,
        "r2_talent_only": round(r2_talent, 4),
        "r2_final": round(r2_final, 4),
        "r2_incremental_from_pairs": round(r2_final - r2_talent, 4),
        "talent_coefficient": round(talent_coef, 5),
        "n_surviving_pairs": len(surviving),
        "n_pairs_pruned": 66 - len(surviving),
        "surviving_pairs": surviving,
    }

    args.save_output.parent.mkdir(parents=True, exist_ok=True)
    args.save_output.write_text(json.dumps(results, indent=2))

    # Print review table
    print()
    print("=" * 92)
    print("LINEUP LASSO FIT — SURVIVING PAIRS (READY FOR REVIEW)")
    print("=" * 92)
    print(f"  n={results['n_lineups']} lineups | seasons={results['seasons']}")
    print(f"  R²: talent={r2_talent:.3f} → full={r2_final:.3f} "
          f"(+{r2_final - r2_talent:.3f} from {len(surviving)} pairs)")
    print(f"  Lasso α={lasso_alpha:.4f} | Candidates={len(candidates)} | Surviving={len(surviving)}")
    print()
    if not surviving:
        print("  NO PAIRS SURVIVED. Defense + talent control alone explains team variance.")
    else:
        print(f"  {'Pair':<60} {'V_AB':>8} {'CI-lo':>7} {'CI-hi':>7}  {'N-with':>7}  Notes")
        print(f"  {'-'*60} {'-'*8} {'-'*7} {'-'*7}  {'-'*7}  -----")
        for s in surviving:
            pnr = " ⚠pnr_pop" if s["involves_pnr_popping_big"] else ""
            ci_flag = " (CI spans 0)" if s["ci_spans_zero"] else ""
            label = f"{s['a']} × {s['b']}"
            print(f"  {label:<60} {s['V_AB']:>+8.4f} {s['ci_lo_V']:>+7.4f} {s['ci_hi_V']:>+7.4f}  "
                  f"{s['n_lineups_with_pair']:>7}{pnr}{ci_flag}")
        print()
        print(f"  Total pairs pruned to zero: {66 - len(surviving)} of 66")
    print()
    print(f"  Results: {args.save_output}")
    print(f"  Switch to Opus for basketball-review of these pairs before merging.")


if __name__ == "__main__":
    main()
