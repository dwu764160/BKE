"""
fit_archetype_interactions.py — Phase 4.6

Fits Ridge regression to empirically estimate the 66 archetype pair interaction
values currently hand-coded in team_feature_aggregation.INTERACTION_MATRIX.

Model:
    team_NET_RATING (season-centered) = talent_control + Σ β_ij * X_ij + ε

Where X_ij is the minute-weighted pair exposure on the team:
    X_ij = S_i * S_j   (S_k = fraction of qualifying-player minutes for archetype k)
    X_ii = S_i²         (same-archetype pair)

Stopping point: outputs a formatted coefficient table for Opus review.
DO NOT auto-update INTERACTION_MATRIX — human basketball review required first.

Run:
    python3 src/modeling/fit_archetype_interactions.py
    python3 src/modeling/fit_archetype_interactions.py --save-output path/to/results.json
"""

import argparse
import json
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from src.data.schema_contract import load_standardized, save_standardized

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
AGGREGATE_PATH = Path("aggregate/player_profile_aggregate.parquet")
MODELING_INPUTS_PATH = Path("data/processed/modeling_inputs_all.parquet")
TEAM_OUTCOMES_PATH = Path("data/processed/team_season_outcomes.parquet")
DEFAULT_OUTPUT = Path("reports/archetype_interaction_fit.json")

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

# Snake-case keys (matches INTERACTION_MATRIX keys in team_feature_aggregation.py)
ARCH_KEYS = {
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

# Current hand-coded values for comparison
V_PP = +0.10; V_P = +0.06; V_MP = +0.03; V_M = -0.06; V_MM = -0.03
V_N  = -0.10; V_0 = 0.00

CURRENT_MATRIX = {
    ("ball_dominant_creator",  "ball_dominant_creator"):           V_M,
    ("ball_dominant_creator",  "all_around_scorer"):               V_MM,
    ("ball_dominant_creator",  "ballhandler"):                     V_MM,
    ("ball_dominant_creator",  "interior_scorer"):                 V_P,
    ("ball_dominant_creator",  "perimeter_scorer"):                V_MP,
    ("ball_dominant_creator",  "connector"):                       V_P,
    ("ball_dominant_creator",  "pnr_rolling_big"):                 V_PP,
    ("ball_dominant_creator",  "pnr_popping_big"):                 V_P,
    ("ball_dominant_creator",  "off_ball_finisher"):               V_P,
    ("ball_dominant_creator",  "off_ball_movement_shooter"):       V_PP,
    ("ball_dominant_creator",  "off_ball_stationary_shooter"):     V_P,
    ("all_around_scorer",      "all_around_scorer"):               V_MM,
    ("all_around_scorer",      "ballhandler"):                     V_0,
    ("all_around_scorer",      "interior_scorer"):                 V_MP,
    ("all_around_scorer",      "perimeter_scorer"):                V_MP,
    ("all_around_scorer",      "connector"):                       V_P,
    ("all_around_scorer",      "pnr_rolling_big"):                 V_MP,
    ("all_around_scorer",      "pnr_popping_big"):                 V_MP,
    ("all_around_scorer",      "off_ball_finisher"):               V_MP,
    ("all_around_scorer",      "off_ball_movement_shooter"):       V_P,
    ("all_around_scorer",      "off_ball_stationary_shooter"):     V_MP,
    ("ballhandler",            "ballhandler"):                     V_M,
    ("ballhandler",            "interior_scorer"):                 V_MP,
    ("ballhandler",            "perimeter_scorer"):                V_MP,
    ("ballhandler",            "connector"):                       V_P,
    ("ballhandler",            "pnr_rolling_big"):                 V_PP,
    ("ballhandler",            "pnr_popping_big"):                 V_P,
    ("ballhandler",            "off_ball_finisher"):               V_MP,
    ("ballhandler",            "off_ball_movement_shooter"):       V_P,
    ("ballhandler",            "off_ball_stationary_shooter"):     V_MP,
    ("interior_scorer",        "interior_scorer"):                 V_M,
    ("interior_scorer",        "perimeter_scorer"):                V_MP,
    ("interior_scorer",        "connector"):                       V_MP,
    ("interior_scorer",        "pnr_rolling_big"):                 V_MM,
    ("interior_scorer",        "pnr_popping_big"):                 V_MP,
    ("interior_scorer",        "off_ball_finisher"):               V_N,
    ("interior_scorer",        "off_ball_movement_shooter"):       V_MP,
    ("interior_scorer",        "off_ball_stationary_shooter"):     V_0,
    ("perimeter_scorer",       "perimeter_scorer"):                V_MM,
    ("perimeter_scorer",       "connector"):                       V_P,
    ("perimeter_scorer",       "pnr_rolling_big"):                 V_MP,
    ("perimeter_scorer",       "pnr_popping_big"):                 V_MP,
    ("perimeter_scorer",       "off_ball_finisher"):               V_0,
    ("perimeter_scorer",       "off_ball_movement_shooter"):       V_M,
    ("perimeter_scorer",       "off_ball_stationary_shooter"):     V_MM,
    ("connector",              "connector"):                       V_MP,
    ("connector",              "pnr_rolling_big"):                 V_P,
    ("connector",              "pnr_popping_big"):                 V_P,
    ("connector",              "off_ball_finisher"):               V_MP,
    ("connector",              "off_ball_movement_shooter"):       V_P,
    ("connector",              "off_ball_stationary_shooter"):     V_MP,
    ("pnr_rolling_big",        "pnr_rolling_big"):                 V_N,
    ("pnr_rolling_big",        "pnr_popping_big"):                 V_MP,
    ("pnr_rolling_big",        "off_ball_finisher"):               V_MM,
    ("pnr_rolling_big",        "off_ball_movement_shooter"):       V_P,
    ("pnr_rolling_big",        "off_ball_stationary_shooter"):     V_MP,
    ("pnr_popping_big",        "pnr_popping_big"):                 V_MM,
    ("pnr_popping_big",        "off_ball_finisher"):               V_0,
    ("pnr_popping_big",        "off_ball_movement_shooter"):       V_MP,
    ("pnr_popping_big",        "off_ball_stationary_shooter"):     V_MP,
    ("off_ball_finisher",      "off_ball_finisher"):               V_N,
    ("off_ball_finisher",      "off_ball_movement_shooter"):       V_MP,
    ("off_ball_finisher",      "off_ball_stationary_shooter"):     V_0,
    ("off_ball_movement_shooter", "off_ball_movement_shooter"):    V_MM,
    ("off_ball_movement_shooter", "off_ball_stationary_shooter"):  V_0,
    ("off_ball_stationary_shooter", "off_ball_stationary_shooter"): V_MM,
}

# All 66 pairs in canonical order
ALL_PAIRS = list(combinations_with_replacement(
    [ARCH_KEYS[a] for a in OFFENSIVE_ARCHETYPES], 2
))

MIN_THRESHOLD = 200   # minimum season minutes to include a player
PNR_POP_KEY = "pnr_popping_big"


# ─────────────────────────────────────────────────────────────────────────────
# Build team-season feature matrix
# ─────────────────────────────────────────────────────────────────────────────

def build_team_features(agg: pd.DataFrame, rapm: pd.DataFrame) -> pd.DataFrame:
    """Compute archetype pair features and talent control for each team-season."""
    # RAPM is already in the aggregate — use it directly
    merged = agg.copy()

    # Filter to qualifying players with valid archetype
    qual = merged[
        merged["primary_archetype"].isin(OFFENSIVE_ARCHETYPES) &
        (merged["_min"] >= MIN_THRESHOLD) &
        merged["team_id"].notna()
    ].copy()
    qual["team_id"] = qual["team_id"].astype(np.int64)

    rows = []
    for (season, team_id), grp in qual.groupby(["season", "team_id"]):
        total_min = grp["_min"].sum()
        if total_min == 0:
            continue
        grp = grp.copy()
        grp["min_share"] = grp["_min"] / total_min

        # S_k for each archetype
        S = {}
        for arch in OFFENSIVE_ARCHETYPES:
            key = ARCH_KEYS[arch]
            S[key] = grp.loc[grp["primary_archetype"] == arch, "min_share"].sum()

        # 66 pair features: X_ij = S_i * S_j
        pair_features = {}
        for a, b in ALL_PAIRS:
            pair_features[f"{a}__{b}"] = S[a] * S[b]

        # Talent control: minute-weighted mean RAPM
        has_rapm = grp["rapm"].notna()
        if has_rapm.any():
            rapm_talent = (grp.loc[has_rapm, "rapm"] * grp.loc[has_rapm, "min_share"]).sum()
            rapm_talent /= grp.loc[has_rapm, "min_share"].sum()
        else:
            rapm_talent = np.nan

        # Archetype coverage stats
        n_arch = (grp["primary_archetype"].value_counts()).to_dict()
        n_pnr_pop = n_arch.get("PnR Popping Big", 0)

        row = {
            "season": season,
            "team_id": team_id,
            "n_players": len(grp),
            "rapm_talent": rapm_talent,
            "n_pnr_popping_big": n_pnr_pop,
            **pair_features,
        }
        rows.append(row)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Ridge regression
# ─────────────────────────────────────────────────────────────────────────────

def fit_ridge(team_df: pd.DataFrame, outcomes: pd.DataFrame) -> dict:
    """Fit Ridge regression and return all results."""

    # Merge team features with outcomes
    outcomes = outcomes.copy()
    outcomes["team_id"] = outcomes["team_id"].astype(np.int64)
    data = team_df.merge(
        outcomes[["season", "team_id", "NET_RATING"]],
        left_on=["season", "team_id"],
        right_on=["season", "team_id"],
        how="inner",
    )
    print(f"  Matched {len(data)} team-seasons (of {len(team_df)} built, {len(outcomes)} in outcomes)")

    # Season-center NET_RATING to remove cross-season level shifts
    season_means = data.groupby("season")["NET_RATING"].transform("mean")
    data["NET_RATING_centered"] = data["NET_RATING"] - season_means

    # Feature columns
    pair_cols = [f"{a}__{b}" for a, b in ALL_PAIRS]

    # Drop rows missing talent control (impute with season mean)
    data["rapm_talent"] = data.groupby("season")["rapm_talent"].transform(
        lambda x: x.fillna(x.mean())
    )

    y = data["NET_RATING_centered"].values
    X_talent = data["rapm_talent"].values.reshape(-1, 1)
    X_pairs = data[pair_cols].values

    # --- Baseline: talent-only model ---
    talent_scaler = StandardScaler()
    X_tal_s = talent_scaler.fit_transform(X_talent)
    alpha_candidates = np.logspace(-2, 4, 50)
    baseline_cv = RidgeCV(alphas=alpha_candidates, cv=5).fit(X_tal_s, y)
    y_pred_talent = baseline_cv.predict(X_tal_s)
    ss_res_talent = np.sum((y - y_pred_talent) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2_talent = 1 - ss_res_talent / ss_tot

    # --- Full model: talent + archetype pairs ---
    X_full = np.hstack([X_talent, X_pairs])
    full_scaler = StandardScaler()
    X_full_s = full_scaler.fit_transform(X_full)
    full_cv = RidgeCV(alphas=alpha_candidates, cv=5).fit(X_full_s, y)
    y_pred_full = full_cv.predict(X_full_s)
    ss_res_full = np.sum((y - y_pred_full) ** 2)
    r2_full = 1 - ss_res_full / ss_tot

    # Unscale coefficients back to raw feature units
    # (coef[i] in original units = coef_scaled[i] / scale[i])
    coef_scaled = full_cv.coef_
    scales = full_scaler.scale_
    coef_raw = coef_scaled / scales

    # Talent coef and pair coefs
    talent_coef = coef_raw[0]
    pair_coefs = coef_raw[1:]  # one per pair

    # Build result dict
    pair_results = {}
    for i, (a, b) in enumerate(ALL_PAIRS):
        key = (a, b)
        current = CURRENT_MATRIX.get(key, CURRENT_MATRIX.get((b, a), 0.0))
        involves_pnr_pop = (a == PNR_POP_KEY or b == PNR_POP_KEY)
        pair_results[f"{a}__{b}"] = {
            "archetype_a": a,
            "archetype_b": b,
            "fitted": round(float(pair_coefs[i]), 5),
            "current": current,
            "delta": round(float(pair_coefs[i]) - current, 5),
            "involves_pnr_popping_big": involves_pnr_pop,
        }

    # Mean pair exposure per team (for coverage reporting)
    pair_means = data[pair_cols].mean()
    pair_stds = data[pair_cols].std()
    for i, (a, b) in enumerate(ALL_PAIRS):
        col = f"{a}__{b}"
        pair_results[col]["mean_exposure"] = round(float(pair_means[col]), 5)
        pair_results[col]["std_exposure"] = round(float(pair_stds[col]), 5)

    # Bootstrap confidence intervals (500 samples)
    print("  Computing bootstrap CIs (500 samples)...")
    n = len(y)
    boot_coefs = np.zeros((500, len(pair_cols)))
    rng = np.random.default_rng(42)
    for b_iter in range(500):
        idx = rng.integers(0, n, size=n)
        X_b = X_full_s[idx]
        y_b = y[idx]
        m = Ridge(alpha=full_cv.alpha_).fit(X_b, y_b)
        boot_coefs[b_iter] = m.coef_[1:] / scales[1:]  # unscaled pair coefs

    ci_lo = np.percentile(boot_coefs, 2.5, axis=0)
    ci_hi = np.percentile(boot_coefs, 97.5, axis=0)
    for i, (a, b) in enumerate(ALL_PAIRS):
        col = f"{a}__{b}"
        pair_results[col]["ci_lo_95"] = round(float(ci_lo[i]), 5)
        pair_results[col]["ci_hi_95"] = round(float(ci_hi[i]), 5)

    return {
        "n_observations": int(len(data)),
        "ridge_alpha_selected": float(full_cv.alpha_),
        "r2_talent_only": round(float(r2_talent), 4),
        "r2_full_model": round(float(r2_full), 4),
        "r2_incremental_from_pairs": round(float(r2_full - r2_talent), 4),
        "talent_coefficient": round(float(talent_coef), 5),
        "pairs": pair_results,
        "seasons_used": sorted(data["season"].unique().tolist()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Print review table
# ─────────────────────────────────────────────────────────────────────────────

def print_review_table(results: dict) -> None:
    print()
    print("=" * 90)
    print("ARCHETYPE INTERACTION FIT — READY FOR BASKETBALL REVIEW")
    print("=" * 90)
    print(f"  n={results['n_observations']} team-seasons | "
          f"Ridge α={results['ridge_alpha_selected']:.2f} | "
          f"R²: talent={results['r2_talent_only']:.3f} → full={results['r2_full_model']:.3f} "
          f"(+{results['r2_incremental_from_pairs']:.3f} from pairs)")
    print()
    print(f"  {'Pair':<55} {'Fitted':>8} {'Current':>8} {'Delta':>7} {'CI-lo':>7} {'CI-hi':>7}  {'Exp':>6}  Notes")
    print(f"  {'-'*55} {'-'*8} {'-'*8} {'-'*7} {'-'*7} {'-'*7}  {'-'*6}  -----")

    pairs = results["pairs"]
    for (a, b) in ALL_PAIRS:
        col = f"{a}__{b}"
        p = pairs[col]
        pnr_flag = " ⚠ pnr_pop" if p["involves_pnr_popping_big"] else ""
        label = f"{a} × {b}" if a != b else f"{a} × same"
        # Signal / noise check: CI spans zero?
        ci_lo, ci_hi = p["ci_lo_95"], p["ci_hi_95"]
        spans_zero = (ci_lo < 0 < ci_hi) or (ci_lo > 0 > ci_hi)
        noise_flag = " (wide CI)" if spans_zero else ""
        print(f"  {label:<55} {p['fitted']:>+8.4f} {p['current']:>+8.4f} {p['delta']:>+7.4f} "
              f"{ci_lo:>+7.4f} {ci_hi:>+7.4f}  {p['mean_exposure']:>6.4f}{pnr_flag}{noise_flag}")

    print()
    print("  ⚠ = involves PnR Popping Big (low stability, high shrinkage expected)")
    print("  'wide CI' = 95% bootstrap CI spans zero — treat as unreliable")
    print()
    print("  STOP: switch to Opus for basketball sense review of these coefficients.")
    print("  Do NOT auto-update INTERACTION_MATRIX until review is complete.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    print("Loading data...")
    agg = load_standardized(AGGREGATE_PATH)
    # Normalize key columns
    min_col = "min" if "min" in agg.columns else "min"
    agg["_min"] = pd.to_numeric(agg[min_col], errors="coerce").fillna(0)
    arch_col = "primary_archetype"
    print(f"  Aggregate: {len(agg)} rows, {agg['season'].nunique()} seasons")

    rapm = load_standardized(MODELING_INPUTS_PATH)
    print(f"  RAPM inputs: {len(rapm)} rows")

    outcomes = load_standardized(TEAM_OUTCOMES_PATH)
    print(f"  Team outcomes: {len(outcomes)} rows, {outcomes['season'].nunique()} seasons")

    print("Building team-season feature matrix...")
    team_df = build_team_features(agg, rapm)
    print(f"  Built {len(team_df)} team-seasons")
    print(f"  PnR Popping Big counts per season:")
    for s, g in team_df.groupby("season"):
        print(f"    {s}: {g['n_pnr_popping_big'].sum():.0f} player-slots across {len(g)} teams")

    print("Fitting Ridge regression...")
    results = fit_ridge(team_df, outcomes)

    print_review_table(results)

    args.save_output.parent.mkdir(parents=True, exist_ok=True)
    args.save_output.write_text(json.dumps(results, indent=2))
    print(f"\n  Results saved to {args.save_output}")
    print("  Share this file with Opus for basketball sense review.")


if __name__ == "__main__":
    main()
