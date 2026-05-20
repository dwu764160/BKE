"""
src/player_eval/year_to_year_bke_deltas.py
=============================================================================
Age Curve Empirical Calibration Script

For each player appearing in consecutive seasons, computes:
  delta_bke = BKE(season_N+1) - BKE(season_N)

Groups results by age bracket and compares empirical means to the
hardcoded AGE_CURVE_DELTAS in src/player_eval/constants.py.

Outputs:
  reports/age_curve_calibration.json

Usage:
  python3 src/player_eval/year_to_year_bke_deltas.py
=============================================================================
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.player_eval.constants import AGE_CURVE_BREAKPOINTS, AGE_CURVE_DELTAS
from src.simulation.simulation_config import REPORTS_DIR

PROFILE_AGGREGATE_PATH = Path("aggregate/player_profile_aggregate.parquet")

BKE_COL = "bke_transformed_bke"
AGE_COL = "age"
PLAYER_ID_COL = "player_id"
SEASON_COL = "season"

MIN_MINUTES = 500
MIN_GP = 20


def load_profiles() -> pd.DataFrame:
    """Load player profile aggregate and filter to qualified players."""
    df = pd.read_parquet(PROFILE_AGGREGATE_PATH)

    # Normalize column names
    df.columns = [str(c).lower() for c in df.columns]

    bke_col = BKE_COL.lower()
    age_col = AGE_COL.lower()
    pid_col = PLAYER_ID_COL.lower()
    season_col = SEASON_COL.lower()

    required = [bke_col, age_col, pid_col, season_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in aggregate: {missing}")

    # Filter to qualified players
    min_col = next((c for c in df.columns if c in ("min", "total_minutes", "mpg")), None)
    gp_col = next((c for c in df.columns if c == "gp"), None)

    if min_col:
        df = df[df[min_col].fillna(0) >= MIN_MINUTES]
    if gp_col:
        df = df[df[gp_col].fillna(0) >= MIN_GP]

    df[season_col] = df[season_col].astype(str)
    df[bke_col] = pd.to_numeric(df[bke_col], errors="coerce")
    df[age_col] = pd.to_numeric(df[age_col], errors="coerce")

    return df[[pid_col, season_col, bke_col, age_col]].dropna()


def sort_seasons(seasons: list[str]) -> list[str]:
    """Sort NBA seasons chronologically ('2022-23', '2023-24', ...)."""
    return sorted(seasons, key=lambda s: int(s.split("-")[0]))


def build_yoy_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Build year-over-year pairs for consecutive seasons."""
    seasons = sort_seasons(df["season"].unique().tolist())
    pairs = []

    for i in range(len(seasons) - 1):
        s_n = seasons[i]
        s_n1 = seasons[i + 1]

        # Check seasons are truly consecutive
        yr_n = int(s_n.split("-")[0])
        yr_n1 = int(s_n1.split("-")[0])
        if yr_n1 - yr_n != 1:
            print(f"  Skipping non-consecutive pair: {s_n} → {s_n1}")
            continue

        df_n = df[df["season"] == s_n].rename(
            columns={"bke_transformed_bke": "bke_n", "age": "age_n"}
        )
        df_n1 = df[df["season"] == s_n1].rename(
            columns={"bke_transformed_bke": "bke_n1", "age": "age_n1"}
        )

        merged = df_n.merge(df_n1, on="player_id", suffixes=("", "_next"))
        merged["delta_bke"] = merged["bke_n1"] - merged["bke_n"]
        merged["age_at_n1"] = merged["age_n1"]
        merged["transition"] = f"{s_n}→{s_n1}"

        pairs.append(merged[["player_id", "transition", "age_at_n1", "bke_n", "bke_n1", "delta_bke"]])

    if not pairs:
        return pd.DataFrame(columns=["player_id", "transition", "age_at_n1", "bke_n", "bke_n1", "delta_bke"])
    return pd.concat(pairs, ignore_index=True)


def age_bracket(age: float, breakpoints: list[float]) -> str:
    """Return the age bracket label for a given age."""
    if age < breakpoints[0]:
        return f"<{breakpoints[0]}"
    for i in range(len(breakpoints) - 1):
        if breakpoints[i] <= age < breakpoints[i + 1]:
            return f"{breakpoints[i]}-{breakpoints[i+1]}"
    return f"{breakpoints[-1]}+"


def analyze(pairs: pd.DataFrame) -> dict:
    """Compute empirical age-curve deltas and compare to hardcoded constants."""
    breakpoints = AGE_CURVE_BREAKPOINTS

    bracket_labels = [f"<{breakpoints[0]}"] + [
        f"{breakpoints[i]}-{breakpoints[i+1]}" for i in range(len(breakpoints) - 1)
    ] + [f"{breakpoints[-1]}+"]

    # Map each player-year to a bracket
    pairs["bracket"] = pairs["age_at_n1"].apply(lambda a: age_bracket(a, breakpoints))

    results = []
    for label in bracket_labels:
        subset = pairs[pairs["bracket"] == label]
        if len(subset) < 5:
            results.append({
                "bracket": label,
                "n_pairs": len(subset),
                "empirical_mean_delta": None,
                "empirical_median_delta": None,
                "empirical_std": None,
                "hardcoded_delta": None,
                "drift_from_hardcoded": None,
                "flag_significant": False,
                "note": "insufficient data (< 5 pairs)",
            })
            continue

        # Map label to hardcoded constant
        if label == f"<{breakpoints[0]}":
            hardcoded = AGE_CURVE_DELTAS[0]
        elif label == f"{breakpoints[-1]}+":
            hardcoded = AGE_CURVE_DELTAS[-1]
        else:
            lo = int(label.split("-")[0])
            idx = next(
                (i for i, bp in enumerate(breakpoints) if bp == lo),
                None,
            )
            hardcoded = AGE_CURVE_DELTAS[idx] if idx is not None else None

        mean_delta = float(np.mean(subset["delta_bke"]))
        median_delta = float(np.median(subset["delta_bke"]))
        std_delta = float(np.std(subset["delta_bke"]))
        drift = (mean_delta - hardcoded) if hardcoded is not None else None

        results.append({
            "bracket": label,
            "n_pairs": len(subset),
            "empirical_mean_delta": round(mean_delta, 4),
            "empirical_median_delta": round(median_delta, 4),
            "empirical_std": round(std_delta, 4),
            "hardcoded_delta": hardcoded,
            "drift_from_hardcoded": round(drift, 4) if drift is not None else None,
            "flag_significant": abs(drift) >= 0.02 if drift is not None else False,
        })

    return results


def main() -> None:
    print("Age Curve Empirical Calibration")
    print("=" * 50)
    print(f"BKE column: {BKE_COL}")
    print(f"Hardcoded breakpoints: {AGE_CURVE_BREAKPOINTS}")
    print(f"Hardcoded deltas: {AGE_CURVE_DELTAS}")
    print()

    df = load_profiles()
    print(f"Loaded {len(df)} qualified player-seasons across: {sorted(df['season'].unique())}")

    pairs = build_yoy_pairs(df)
    print(f"Built {len(pairs)} year-over-year pairs across {pairs['transition'].nunique()} transition(s)")

    if pairs.empty:
        print("No consecutive season pairs found — need at least 2 seasons of data.")
        return

    bracket_results = analyze(pairs)

    print("\nAge Bracket Analysis:")
    print(f"{'Bracket':<15} {'N':<6} {'Empirical':>10} {'Hardcoded':>10} {'Drift':>8} {'Flag'}")
    print("-" * 60)
    for r in bracket_results:
        emp = f"{r['empirical_mean_delta']:.4f}" if r["empirical_mean_delta"] is not None else "N/A"
        hrd = f"{r['hardcoded_delta']:.4f}" if r["hardcoded_delta"] is not None else "N/A"
        drift = f"{r['drift_from_hardcoded']:+.4f}" if r["drift_from_hardcoded"] is not None else "N/A"
        flag = "⚠ UPDATE" if r.get("flag_significant") else ""
        print(f"{r['bracket']:<15} {r['n_pairs']:<6} {emp:>10} {hrd:>10} {drift:>8}  {flag}")

    flagged = [r for r in bracket_results if r.get("flag_significant")]
    if flagged:
        print(f"\n{len(flagged)} bracket(s) have drift ≥ 0.02 from hardcoded constants.")
        print("Consider updating AGE_CURVE_DELTAS in src/player_eval/constants.py.")
    else:
        print("\nAll brackets within ±0.02 of hardcoded constants. No update needed.")

    output = {
        "note": (
            f"Based on {len(pairs)} player year-over-year pairs across "
            f"{sorted(df['season'].unique())} seasons. "
            f"Only {MIN_MINUTES}+ minute, {MIN_GP}+ game players included. "
            f"With only {df['season'].nunique()} season(s), estimates for edge brackets are thin."
        ),
        "breakpoints": AGE_CURVE_BREAKPOINTS,
        "current_hardcoded_deltas": list(AGE_CURVE_DELTAS),
        "bracket_analysis": bracket_results,
        "n_pairs_total": len(pairs),
        "transitions": pairs["transition"].unique().tolist(),
    }

    out_path = REPORTS_DIR / "age_curve_calibration.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
