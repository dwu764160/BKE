"""
src/modeling/backtesting.py
=============================================================================
BKE v2.6 — Backtesting Framework

Tests: Do BKE scores from season N predict performance in season N+1?

Design:
    Split: Train on 2022-23 + 2023-24, test predictions vs 2024-25 actuals.
    For players appearing in consecutive seasons, measure:
        1. Rank correlation (Spearman) of BKE scores
        2. Tier classification accuracy
        3. Root Mean Square Error of z-scores
        4. Stability of archetype assignments

Integration:
    Called standalone or from run_full_decomposition as a validation step.
    Does NOT affect the main pipeline outputs.

Outputs:
    - Prints summary to console
    - Returns a diagnostics DataFrame
    - Optionally saves to data/processed/bke_v26_backtest.json
=============================================================================
"""

import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.modeling.model_config import (
    PROCESSED_DIR,
    SEASONS,
    DECOMPOSITION,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Metrics to test prediction quality for
PREDICTION_METRICS = [
    "portable_talent_score",
    "portable_talent_z",
    "total_impact_score",
    "total_impact_z",
    "role_utilization_efficiency",
    "portability_index",
]

# Dimension z-scores to test stability
DIMENSION_METRICS = [
    "dim_shooting_gravity_z",
    "dim_driving_gravity_z",
    "dim_playmaking_creation_z",
    "dim_extra_possession_z",
    "dim_defensive_playmaking_z",
    "dim_defensive_impact_z",
    "dim_turnover_control_z",
    "dim_defensive_versatility_z",
    "dim_self_creation_z",
]

# Tier boundaries for classification accuracy
TIER_MAP = {
    "Elite": (90, 100),
    "All-Star": (75, 90),
    "Starter": (50, 75),
    "Rotation": (25, 50),
    "Fringe": (0, 25),
}


# ---------------------------------------------------------------------------
# Core: Year-over-Year Prediction Analysis
# ---------------------------------------------------------------------------

def _assign_tier(score: float) -> str:
    """Assign tier based on percentile score."""
    if pd.isna(score):
        return "Unknown"
    for tier, (lo, hi) in TIER_MAP.items():
        if lo <= score <= hi:
            return tier
    return "Unknown"


def find_returning_players(
    df: pd.DataFrame,
    train_seasons: List[str],
    test_season: str,
    id_col: str = "player_id",
) -> pd.DataFrame:
    """
    Find players who appear in both training and test seasons.

    Returns a merged DataFrame with columns suffixed _train and _test.
    """
    train_df = df[df["season"].isin(train_seasons)].copy()
    test_df = df[df["season"] == test_season].copy()

    # For players with multiple training seasons, use the most recent
    if len(train_seasons) > 1:
        # Sort seasons and take the latest one per player
        season_order = {s: i for i, s in enumerate(SEASONS)}
        train_df["_season_order"] = train_df["season"].map(season_order)
        train_df = train_df.sort_values("_season_order", ascending=False)
        train_df = train_df.drop_duplicates(subset=[id_col], keep="first")
        train_df = train_df.drop(columns=["_season_order"])

    # Merge on player_id
    merged = train_df.merge(
        test_df,
        on=id_col,
        suffixes=("_train", "_test"),
        how="inner",
    )

    return merged


def compute_rank_correlation(
    merged: pd.DataFrame,
    metric: str,
) -> Dict[str, float]:
    """
    Compute Spearman rank correlation for a metric between train and test.
    """
    train_col = f"{metric}_train"
    test_col = f"{metric}_test"

    if train_col not in merged.columns or test_col not in merged.columns:
        return {"spearman_rho": np.nan, "p_value": np.nan, "n": 0}

    valid = merged[[train_col, test_col]].dropna()
    n = len(valid)

    if n < 5:
        return {"spearman_rho": np.nan, "p_value": np.nan, "n": n}

    rho, p = scipy_stats.spearmanr(valid[train_col], valid[test_col])

    return {
        "spearman_rho": float(rho),
        "p_value": float(p),
        "n": n,
    }


def compute_tier_accuracy(
    merged: pd.DataFrame,
    metric: str = "total_impact_score",
) -> Dict[str, float]:
    """
    Compute tier classification accuracy.

    Measures:
      - Exact tier match rate
      - Within-1-tier accuracy (e.g., Starter predicted as All-Star = hit)
    """
    train_col = f"{metric}_train"
    test_col = f"{metric}_test"

    if train_col not in merged.columns or test_col not in merged.columns:
        return {"exact_accuracy": np.nan, "within_1_accuracy": np.nan, "n": 0}

    valid = merged[[train_col, test_col]].dropna()
    n = len(valid)

    if n < 5:
        return {"exact_accuracy": np.nan, "within_1_accuracy": np.nan, "n": n}

    tier_order = list(TIER_MAP.keys())

    train_tiers = valid[train_col].apply(_assign_tier)
    test_tiers = valid[test_col].apply(_assign_tier)

    # Exact match
    exact = (train_tiers == test_tiers).mean()

    # Within-1 tier
    def tier_distance(t1, t2):
        if t1 not in tier_order or t2 not in tier_order:
            return 99
        return abs(tier_order.index(t1) - tier_order.index(t2))

    within_1 = pd.Series([
        tier_distance(t1, t2) <= 1
        for t1, t2 in zip(train_tiers, test_tiers)
    ]).mean()

    return {
        "exact_accuracy": float(exact),
        "within_1_accuracy": float(within_1),
        "n": n,
    }


def compute_rmse(
    merged: pd.DataFrame,
    metric: str,
) -> Dict[str, float]:
    """Compute RMSE between train and test values."""
    train_col = f"{metric}_train"
    test_col = f"{metric}_test"

    if train_col not in merged.columns or test_col not in merged.columns:
        return {"rmse": np.nan, "n": 0}

    valid = merged[[train_col, test_col]].dropna()
    n = len(valid)

    if n < 5:
        return {"rmse": np.nan, "n": n}

    rmse = np.sqrt(((valid[train_col] - valid[test_col]) ** 2).mean())

    return {"rmse": float(rmse), "n": n}


def compute_archetype_stability(
    merged: pd.DataFrame,
    archetype_col: str = "primary_archetype",
) -> Dict[str, float]:
    """
    Measure archetype assignment stability across seasons.

    A good system should assign consistent archetypes to the same player
    across consecutive seasons (barring genuine role changes).
    """
    train_col = f"{archetype_col}_train"
    test_col = f"{archetype_col}_test"

    if train_col not in merged.columns or test_col not in merged.columns:
        return {"archetype_stability": np.nan, "n": 0}

    valid = merged[[train_col, test_col]].dropna()
    n = len(valid)

    if n < 5:
        return {"archetype_stability": np.nan, "n": n}

    stability = (valid[train_col] == valid[test_col]).mean()

    return {"archetype_stability": float(stability), "n": n}


# ---------------------------------------------------------------------------
# Big Movers Analysis
# ---------------------------------------------------------------------------

def find_big_movers(
    merged: pd.DataFrame,
    metric: str = "total_impact_score",
    n_top: int = 10,
) -> pd.DataFrame:
    """
    Find players with the biggest positive and negative changes.

    These represent cases where BKE was most wrong (or where genuine
    breakout/decline occurred).
    """
    train_col = f"{metric}_train"
    test_col = f"{metric}_test"
    name_col = "player_name_test" if "player_name_test" in merged.columns else "player_id"

    if train_col not in merged.columns or test_col not in merged.columns:
        return pd.DataFrame()

    valid = merged.copy()
    valid["_delta"] = valid[test_col] - valid[train_col]

    # Top risers
    risers = valid.nlargest(n_top, "_delta")[[name_col, train_col, test_col, "_delta"]].copy()
    risers["direction"] = "⬆️ Riser"

    # Top fallers
    fallers = valid.nsmallest(n_top, "_delta")[[name_col, train_col, test_col, "_delta"]].copy()
    fallers["direction"] = "⬇️ Faller"

    return pd.concat([risers, fallers], ignore_index=True)


# ---------------------------------------------------------------------------
# Position-Split Analysis
# ---------------------------------------------------------------------------

def position_split_analysis(
    merged: pd.DataFrame,
    metric: str = "total_impact_score",
    position_col: str = "position_bucket",
) -> Dict[str, Dict[str, float]]:
    """
    Compute prediction quality split by position.

    Identifies if the model predicts better for certain positions.
    """
    pos_train = f"{position_col}_train"
    if pos_train not in merged.columns:
        return {}

    results = {}
    for pos in merged[pos_train].dropna().unique():
        pos_mask = merged[pos_train] == pos
        pos_merged = merged[pos_mask]

        rho_result = compute_rank_correlation(pos_merged, metric)
        tier_result = compute_tier_accuracy(pos_merged, metric)

        results[pos] = {
            "n": rho_result["n"],
            "spearman_rho": rho_result["spearman_rho"],
            "exact_tier_accuracy": tier_result["exact_accuracy"],
        }

    return results


# ---------------------------------------------------------------------------
# Main: Run Full Backtest
# ---------------------------------------------------------------------------

def run_backtest(
    df: Optional[pd.DataFrame] = None,
    train_seasons: Optional[List[str]] = None,
    test_season: Optional[str] = None,
    save_output: bool = True,
) -> Dict:
    """
    Run the full BKE backtesting suite.

    Default split: Train on 2022-23 + 2023-24, test on 2024-25.

    Parameters
    ----------
    df : Pre-computed BKE decomposition DataFrame. If None, loads from disk.
    train_seasons : Seasons for training. Default: ["2022-23", "2023-24"].
    test_season : Season for testing. Default: "2024-25".
    save_output : Whether to save results JSON.

    Returns
    -------
    Dict with all backtest results.
    """
    # Load data if not provided
    if df is None:
        parquet_path = os.path.join(PROCESSED_DIR, "bke_v26_decomposition.parquet")
        if not os.path.exists(parquet_path):
            print(f"ERROR: {parquet_path} not found. Run decomposition first.")
            return {}
        df = pd.read_parquet(parquet_path)

    # Default seasons
    train_seasons = train_seasons or ["2022-23", "2023-24"]
    test_season = test_season or "2024-25"

    # Filter to qualified
    if "qualified" in df.columns:
        df = df[df["qualified"] == True].copy()

    print("\n" + "=" * 70)
    print("  BKE v2.6 — BACKTESTING FRAMEWORK")
    print("=" * 70)
    print(f"  Train seasons: {train_seasons}")
    print(f"  Test season:   {test_season}")
    print(f"  Total qualified: {len(df)}")

    # Find returning players
    merged = find_returning_players(df, train_seasons, test_season)
    print(f"  Returning players: {len(merged)}")

    if len(merged) < 20:
        print("  WARNING: Too few returning players for reliable backtest")
        return {"error": "insufficient_returning_players", "n": len(merged)}

    # -----------------------------------------------------------------------
    # 1. Rank Correlations
    # -----------------------------------------------------------------------
    print("\n  --- Rank Correlations (Spearman ρ) ---")
    rank_results = {}
    for metric in PREDICTION_METRICS:
        result = compute_rank_correlation(merged, metric)
        rank_results[metric] = result
        if not np.isnan(result["spearman_rho"]):
            sig = "***" if result["p_value"] < 0.001 else "**" if result["p_value"] < 0.01 else "*" if result["p_value"] < 0.05 else ""
            print(f"    {metric:40s} ρ={result['spearman_rho']:+.3f} {sig:3s} (n={result['n']})")

    # Dimension stability
    print("\n  --- Dimension Stability (Spearman ρ) ---")
    dim_results = {}
    for metric in DIMENSION_METRICS:
        result = compute_rank_correlation(merged, metric)
        dim_results[metric] = result
        if not np.isnan(result["spearman_rho"]):
            print(f"    {metric:40s} ρ={result['spearman_rho']:+.3f} (n={result['n']})")

    # -----------------------------------------------------------------------
    # 2. Tier Accuracy
    # -----------------------------------------------------------------------
    print("\n  --- Tier Classification Accuracy ---")
    tier_results = {}
    for metric in ["total_impact_score", "portable_talent_score"]:
        result = compute_tier_accuracy(merged, metric)
        tier_results[metric] = result
        if not np.isnan(result["exact_accuracy"]):
            print(f"    {metric:40s} Exact: {result['exact_accuracy']:.1%}  "
                  f"Within-1: {result['within_1_accuracy']:.1%}  (n={result['n']})")

    # -----------------------------------------------------------------------
    # 3. RMSE
    # -----------------------------------------------------------------------
    print("\n  --- Root Mean Square Error ---")
    rmse_results = {}
    for metric in PREDICTION_METRICS:
        result = compute_rmse(merged, metric)
        rmse_results[metric] = result
        if not np.isnan(result["rmse"]):
            print(f"    {metric:40s} RMSE={result['rmse']:.3f} (n={result['n']})")

    # -----------------------------------------------------------------------
    # 4. Archetype Stability
    # -----------------------------------------------------------------------
    print("\n  --- Archetype Stability ---")
    arch_stab = compute_archetype_stability(merged)
    if not np.isnan(arch_stab.get("archetype_stability", np.nan)):
        print(f"    Offensive archetype stability: {arch_stab['archetype_stability']:.1%} "
              f"(n={arch_stab['n']})")

    def_arch_stab = compute_archetype_stability(merged, "defensive_archetype")
    if not np.isnan(def_arch_stab.get("archetype_stability", np.nan)):
        print(f"    Defensive archetype stability: {def_arch_stab['archetype_stability']:.1%} "
              f"(n={def_arch_stab['n']})")

    # -----------------------------------------------------------------------
    # 5. Position-Split Analysis
    # -----------------------------------------------------------------------
    print("\n  --- Position-Split Prediction Quality ---")
    pos_results = position_split_analysis(merged, "total_impact_score")
    for pos, result in sorted(pos_results.items()):
        if result["n"] >= 5:
            print(f"    {pos:15s} ρ={result['spearman_rho']:+.3f}  "
                  f"Tier Acc: {result['exact_tier_accuracy']:.1%}  (n={result['n']})")

    # -----------------------------------------------------------------------
    # 6. Big Movers
    # -----------------------------------------------------------------------
    print("\n  --- Biggest Year-over-Year Movers (Total Impact) ---")
    name_col = "player_name_test" if "player_name_test" in merged.columns else "player_id"
    movers = find_big_movers(merged, "total_impact_score", n_top=5)
    if not movers.empty:
        for _, row in movers.iterrows():
            train_val = row.get("total_impact_score_train", 0)
            test_val = row.get("total_impact_score_test", 0)
            delta = row.get("_delta", 0)
            print(f"    {row[name_col]:25s} {train_val:5.1f} → {test_val:5.1f} "
                  f"({delta:+.1f}) {row['direction']}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    # Compute overall quality score
    key_rho = rank_results.get("portable_talent_score", {}).get("spearman_rho", 0)
    key_tier = tier_results.get("total_impact_score", {}).get("within_1_accuracy", 0)

    print("\n  === BACKTEST SUMMARY ===")
    print(f"  PTS Rank Correlation:     ρ={key_rho:+.3f}")
    print(f"  TI Within-1-Tier Accuracy: {key_tier:.1%}")

    if abs(key_rho) >= 0.6:
        print("  ✅ Strong year-over-year predictive signal")
    elif abs(key_rho) >= 0.4:
        print("  ⚠️ Moderate year-over-year signal (room for improvement)")
    else:
        print("  ❌ Weak year-over-year signal (model needs work)")

    # Build output dict
    output = {
        "version": "2.6",
        "train_seasons": train_seasons,
        "test_season": test_season,
        "n_returning_players": len(merged),
        "rank_correlations": rank_results,
        "dimension_stability": dim_results,
        "tier_accuracy": tier_results,
        "rmse": rmse_results,
        "archetype_stability": {
            "offensive": arch_stab,
            "defensive": def_arch_stab,
        },
        "position_split": pos_results,
        "summary": {
            "pts_rho": key_rho,
            "ti_within_1_accuracy": key_tier,
        },
    }

    if save_output:
        out_path = os.path.join(PROCESSED_DIR, "bke_v26_backtest.json")
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))
        print(f"\n  Saved: {out_path}")

    return output


# ---------------------------------------------------------------------------
# Pair-level backtest (consecutive seasons)
# ---------------------------------------------------------------------------

def run_consecutive_season_backtest(
    df: Optional[pd.DataFrame] = None,
) -> Dict:
    """
    Run backtests for all consecutive season pairs.

    Tests:
      - 2022-23 → 2023-24
      - 2023-24 → 2024-25
      - Combined: 2022-23 + 2023-24 → 2024-25
    """
    if df is None:
        parquet_path = os.path.join(PROCESSED_DIR, "bke_v26_decomposition.parquet")
        if not os.path.exists(parquet_path):
            return {}
        df = pd.read_parquet(parquet_path)

    results = {}

    # Single-season pairs
    for i in range(len(SEASONS) - 1):
        train = [SEASONS[i]]
        test = SEASONS[i + 1]
        print(f"\n{'='*70}")
        print(f"  Backtest: {train[0]} → {test}")
        print(f"{'='*70}")
        results[f"{train[0]}_to_{test}"] = run_backtest(
            df, train_seasons=train, test_season=test, save_output=False
        )

    # Combined training
    if len(SEASONS) >= 3:
        print(f"\n{'='*70}")
        print(f"  Combined Backtest: {SEASONS[:2]} → {SEASONS[2]}")
        print(f"{'='*70}")
        results["combined"] = run_backtest(
            df, train_seasons=SEASONS[:2], test_season=SEASONS[2], save_output=True
        )

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BKE v2.6 Backtesting")
    parser.add_argument("--all-pairs", action="store_true",
                        help="Run all consecutive season pairs")
    args = parser.parse_args()

    if args.all_pairs:
        run_consecutive_season_backtest()
    else:
        run_backtest()
