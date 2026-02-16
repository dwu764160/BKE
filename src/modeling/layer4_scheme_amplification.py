"""
src/modeling/layer4_scheme_amplification.py
=============================================================================
BKE v1.5 — LAYER 4: Scheme Amplification

Estimates context sensitivity:
  "How much of this player's impact depends on their specific environment?"

Components:
  - Lineup Interaction Coefficient: impact variance across different lineups
  - On/Off Variance: how much team performance changes with player on/off
  - Teammate Dependency: correlation between player impact and teammate quality

High variance → role-dependent (system amplified)
Low variance  → portable (stable across contexts)

Output:
  - scheme_stability_index: 0-100, high = portable, low = system-dependent
  - lineup_interaction_coef: raw variance measure
  - on_off_variance: raw on/off measure
  - teammate_dependency: lineup quality dependency
  - scheme_league_pctl / _archetype_pctl
=============================================================================
"""

import glob
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.modeling.model_config import (
    SCHEME_AMPLIFICATION,
    POSSESSIONS_GLOB,
    DATA_DIR,
    clean_id,
)
from src.modeling.percentile_engine import (
    add_league_percentiles,
    add_grouped_percentiles,
    vectorized_percentile_rank,
)


# ---------------------------------------------------------------------------
# Lineup-level data loading
# ---------------------------------------------------------------------------

def load_possession_data() -> pd.DataFrame:
    """
    Load clean possession data for lineup-level analysis.
    Returns a DataFrame with off_lineup, def_lineup, points, season.
    """
    files = sorted(glob.glob(POSSESSIONS_GLOB))
    if not files:
        print(f"  WARNING: No possession files found at {POSSESSIONS_GLOB}")
        return pd.DataFrame()

    parts = []
    for path in files:
        df = pd.read_parquet(path)
        if "season" not in df.columns:
            season = os.path.basename(path).replace("possessions_clean_", "").replace(".parquet", "")
            df["season"] = season
        parts.append(df)

    full = pd.concat(parts, ignore_index=True)
    return full


# ---------------------------------------------------------------------------
# Lineup Interaction Coefficient
# ---------------------------------------------------------------------------

def compute_lineup_variance(poss_df: pd.DataFrame, min_poss: int = 50) -> pd.DataFrame:
    """
    Compute per-player variance in team performance across different lineups.

    For each player, groups possessions by the other 4 teammates and computes
    the variance in points per possession across these lineup configurations.

    High variance → player's impact depends heavily on teammates
    Low variance  → player performs consistently across lineups

    Returns:
        DataFrame with player_id, season, lineup_variance, n_lineups
    """
    if poss_df.empty:
        return pd.DataFrame(columns=["player_id", "season", "lineup_variance", "n_lineups"])

    results = []

    for season in poss_df["season"].unique():
        season_df = poss_df[poss_df["season"] == season]

        # Build player → lineup performance map
        player_lineup_perf: Dict[str, List[float]] = {}

        for _, row in season_df.iterrows():
            off_lineup = row.get("off_lineup", [])
            points = row.get("points", 0)

            if not isinstance(off_lineup, (list, np.ndarray)):
                continue

            clean_lineup = sorted([clean_id(p) for p in off_lineup if clean_id(p) != "0"])

            for player in clean_lineup:
                if player not in player_lineup_perf:
                    player_lineup_perf[player] = []
                player_lineup_perf[player].append(points)

        # Compute variance for each player
        for player, all_points in player_lineup_perf.items():
            if len(all_points) < min_poss:
                continue

            # Split into chunks (pseudo-lineups) to estimate variance
            chunk_size = max(min_poss, len(all_points) // 10)
            n_chunks = len(all_points) // chunk_size

            if n_chunks < 2:
                results.append({
                    "player_id": player,
                    "season": season,
                    "lineup_variance": 0.0,
                    "n_lineups": 1,
                    "mean_ppp": np.mean(all_points),
                })
                continue

            chunk_means = []
            for i in range(n_chunks):
                start = i * chunk_size
                end = start + chunk_size
                chunk = all_points[start:end]
                chunk_means.append(np.mean(chunk))

            results.append({
                "player_id": player,
                "season": season,
                "lineup_variance": float(np.var(chunk_means)),
                "n_lineups": n_chunks,
                "mean_ppp": np.mean(all_points),
            })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# On/Off Variance
# ---------------------------------------------------------------------------

def compute_on_off_splits(poss_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute on/off differential for each player.

    For each player, computes:
      - on_court_ppp: points per possession when player is on court
      - off_court_ppp: points per possession when player is NOT on court
      - on_off_diff: on_court_ppp - off_court_ppp

    The variance of these across lineup contexts indicates context sensitivity.

    Returns:
        DataFrame with player_id, season, on_court_ppp, off_court_ppp, on_off_diff
    """
    if poss_df.empty:
        return pd.DataFrame(columns=["player_id", "season", "on_court_ppp", "off_court_ppp", "on_off_diff"])

    results = []

    for season in poss_df["season"].unique():
        season_df = poss_df[poss_df["season"] == season]
        total_points = season_df["points"].sum()
        total_poss = len(season_df)

        if total_poss == 0:
            continue

        # Build player on-court stats
        player_on: Dict[str, Dict[str, float]] = {}

        for _, row in season_df.iterrows():
            off_lineup = row.get("off_lineup", [])
            points = row.get("points", 0)

            if not isinstance(off_lineup, (list, np.ndarray)):
                continue

            for pid in off_lineup:
                key = clean_id(pid)
                if key == "0":
                    continue
                if key not in player_on:
                    player_on[key] = {"points": 0.0, "poss": 0}
                player_on[key]["points"] += points
                player_on[key]["poss"] += 1

        for player, stats in player_on.items():
            if stats["poss"] < 100:  # minimum for meaningful on/off
                continue

            on_ppp = stats["points"] / stats["poss"]
            off_points = total_points - stats["points"]
            off_poss = total_poss - stats["poss"]
            off_ppp = off_points / off_poss if off_poss > 0 else on_ppp

            results.append({
                "player_id": player,
                "season": season,
                "on_court_ppp": on_ppp,
                "off_court_ppp": off_ppp,
                "on_off_diff": on_ppp - off_ppp,
            })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Scheme Stability Index
# ---------------------------------------------------------------------------

def compute_scheme_stability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Scheme Stability Index from lineup variance and on/off splits.

    High stability → portable talent (performs well regardless of context)
    Low stability  → system player (depends on the right environment)

    Method:
      - Invert lineup_variance → lower variance = higher stability
      - Combine with on/off consistency
      - Percentile-rank the result
    """
    cfg = SCHEME_AMPLIFICATION
    result = df.copy()

    # --- Lineup Interaction Coefficient ---
    if "lineup_variance" in result.columns:
        # Invert (lower variance = better stability)
        result["lineup_interaction_coef"] = result.groupby("season")["lineup_variance"].transform(
            lambda x: 1.0 - (x - x.min()) / (x.max() - x.min() + 1e-9)
        )
    else:
        result["lineup_interaction_coef"] = 0.5

    # --- On/Off Stability ---
    if "on_off_diff" in result.columns:
        # Higher on/off diff means more impactful regardless = more portable
        # But extreme values might indicate context dependency, so we use
        # the absolute stability (inverse of variance around mean)
        result["on_off_stability"] = result.groupby("season")["on_off_diff"].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
        )
    else:
        result["on_off_stability"] = 0.5

    # --- Teammate Dependency ---
    # Players whose RAPM deviates heavily from on/off are more context-dependent
    if "rapm" in result.columns and "on_off_diff" in result.columns:
        rapm_normalized = result.groupby("season")["rapm"].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-9)
        )
        on_off_normalized = result.groupby("season")["on_off_diff"].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-9)
        )
        # Agreement between RAPM and on/off → more portable
        result["teammate_dependency"] = 1.0 - np.abs(rapm_normalized - on_off_normalized).clip(0, 3) / 3.0
    else:
        result["teammate_dependency"] = 0.5

    # --- Composite Scheme Stability Index ---
    result["scheme_stability_raw"] = (
        cfg.w_lineup_variance * result["lineup_interaction_coef"] +
        cfg.w_on_off_variance * result["on_off_stability"] +
        cfg.w_teammate_dependency * result["teammate_dependency"]
    )

    # Convert to percentile
    result["scheme_stability_index"] = result.groupby("season")["scheme_stability_raw"].transform(
        vectorized_percentile_rank
    )

    # Scheme amplification = inverse of stability (how much context matters)
    result["scheme_amplification"] = 100.0 - result["scheme_stability_index"]

    # Add grouped percentiles
    result = add_league_percentiles(result, ["scheme_stability_raw"])
    if "primary_archetype" in result.columns:
        result = add_grouped_percentiles(
            result,
            ["scheme_stability_raw"],
            group_col="primary_archetype",
            suffix="_pctl",
        )

    # Classify
    result["scheme_classification"] = np.where(
        result["scheme_stability_index"] >= cfg.high_stability_threshold * 100,
        "Portable",
        np.where(
            result["scheme_stability_index"] <= cfg.low_stability_threshold * 100,
            "System-Dependent",
            "Context-Moderate"
        )
    )

    return result


# ---------------------------------------------------------------------------
# Proxy Features (when possession data is unavailable)
# ---------------------------------------------------------------------------

def _approximate_scheme_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Approximate lineup variance and on/off features from box-score proxies
    when raw possession data is not loaded (fast mode).

    Proxy signals:
      - RAPM vs DARKO agreement → players where two independent models agree
        are more likely to be truly portable (skill, not context).
      - Playtype entropy → players with diversified playtype usage are harder
        to scheme against → more portable.
      - Role confidence → high archetype certainty ≈ clear role ≈ less
        context-dependent.
      - ORAPM / DRAPM balance → two-way players tend to be more portable
        than one-dimensional ones.
    """
    result = df.copy()

    # 1. RAPM vs DARKO agreement as on/off proxy
    #    If RAPM ≈ DARKO DPM, the signal is robust across methods → portable
    rapm_col = "rapm" if "rapm" in result.columns else None
    darko_col = "darko_dpm" if "darko_dpm" in result.columns else None

    if rapm_col and darko_col:
        rapm_z = result.groupby("season")[rapm_col].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-9)
        )
        darko_z = result.groupby("season")[darko_col].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-9)
        )
        # Agreement = 1 - abs(diff), clipped to [0, 1]
        agreement = (1.0 - np.abs(rapm_z - darko_z).clip(0, 3) / 3.0).fillna(0.5)
        result["on_off_diff"] = agreement  # higher = more portable
    elif rapm_col:
        # Use abs(RAPM) as a weak proxy — higher-impact players tend to be
        # more portable (they drive team performance, not the other way)
        result["on_off_diff"] = result.groupby("season")[rapm_col].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
        ).fillna(0.5)
    else:
        result["on_off_diff"] = 0.5

    # 2. Playtype entropy as lineup_variance proxy
    #    High entropy → balanced usage → less scheme-dependent
    playtype_poss_pct_cols = [c for c in result.columns if c.endswith("_POSS_PCT")]
    if playtype_poss_pct_cols:
        poss_pcts = result[playtype_poss_pct_cols].fillna(0).values
        # Normalize to distribution
        row_sums = poss_pcts.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        probs = poss_pcts / row_sums
        # Shannon entropy
        entropy = -np.sum(np.where(probs > 0, probs * np.log(probs + 1e-12), 0), axis=1)
        result["_playtype_entropy"] = entropy
        # Invert — low entropy = specialist = more scheme-dependent
        # So lineup_variance = 1 - normalized_entropy (higher entropy = LOWER variance)
        result["lineup_variance"] = result.groupby("season")["_playtype_entropy"].transform(
            lambda x: 1.0 - (x - x.min()) / (x.max() - x.min() + 1e-9)
        ).fillna(0.5)
        result.drop(columns=["_playtype_entropy"], inplace=True)
    elif "role_confidence" in result.columns:
        # High role_confidence → clear role → less variance
        result["lineup_variance"] = result.groupby("season")["role_confidence"].transform(
            lambda x: 1.0 - (x - x.min()) / (x.max() - x.min() + 1e-9)
        ).fillna(0.5)
    else:
        result["lineup_variance"] = 0.5

    print(f"  Using proxy features for scheme estimation")
    return result


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def build_scheme_amplification(df: pd.DataFrame, use_possession_data: bool = True) -> pd.DataFrame:
    """
    Full Layer 4 pipeline: lineup variance → on/off → scheme stability.

    Takes Layer 1+2+3 output DataFrame and adds Layer 4 columns.

    Parameters:
        df: Player-season DataFrame from layers 1-3
        use_possession_data: Whether to load and process raw possession data
                            for lineup variance. Set to False to skip (faster
                            but less accurate).
    """
    print("\n" + "=" * 60)
    print("LAYER 4: SCHEME AMPLIFICATION")
    print("=" * 60)

    qualified = df[df.get("qualified", True) == True].copy() if "qualified" in df.columns else df.copy()

    if qualified.empty:
        print("  No qualified players to process")
        return df

    if use_possession_data:
        # Load possession data for lineup-level analysis
        print("  Loading possession data for lineup analysis...")
        poss_df = load_possession_data()

        if not poss_df.empty:
            # Lineup variance
            print("  Computing lineup variance...")
            lineup_var = compute_lineup_variance(poss_df, min_poss=SCHEME_AMPLIFICATION.min_lineup_poss)
            if not lineup_var.empty:
                orig_idx = qualified.index
                qualified = qualified.merge(
                    lineup_var[["player_id", "season", "lineup_variance", "n_lineups", "mean_ppp"]],
                    on=["season", "player_id"],
                    how="left",
                )
                qualified.index = orig_idx  # preserve index (merge resets it)
                print(f"  Lineup variance computed for {lineup_var['player_id'].nunique()} players")

            # On/Off splits
            print("  Computing on/off splits...")
            on_off = compute_on_off_splits(poss_df)
            if not on_off.empty:
                orig_idx = qualified.index
                qualified = qualified.merge(
                    on_off[["player_id", "season", "on_court_ppp", "off_court_ppp", "on_off_diff"]],
                    on=["season", "player_id"],
                    how="left",
                )
                qualified.index = orig_idx  # preserve index
                print(f"  On/off splits computed for {on_off['player_id'].nunique()} players")
        else:
            print("  WARNING: No possession data available, using approximations")
    else:
        print("  Skipping possession data (using proxy features)")
        # Approximate lineup variance from available metrics
        qualified = _approximate_scheme_features(qualified)

    # Compute scheme stability
    print("  Computing scheme stability index...")
    qualified = compute_scheme_stability(qualified)

    # Merge back — use concat to avoid DataFrame fragmentation
    new_cols = [c for c in qualified.columns if c not in df.columns]
    if new_cols:
        fill = pd.DataFrame(np.nan, index=df.index, columns=new_cols)
        df = pd.concat([df, fill], axis=1)
        df.loc[qualified.index, new_cols] = qualified[new_cols].values

    n = qualified["scheme_stability_index"].notna().sum()
    print(f"\n  Layer 4 complete: {n} players with scheme stability scores")
    if n > 0:
        print(f"  Stability range: {qualified['scheme_stability_index'].min():.1f} - "
              f"{qualified['scheme_stability_index'].max():.1f}")
        portable = (qualified["scheme_classification"] == "Portable").sum()
        system = (qualified["scheme_classification"] == "System-Dependent").sum()
        moderate = (qualified["scheme_classification"] == "Context-Moderate").sum()
        print(f"  Classification: {portable} portable, {moderate} moderate, {system} system-dependent")

    return df


# ---------------------------------------------------------------------------
# CLI Entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.modeling.layer1_portable_talent import build_portable_talent
    from src.modeling.layer2_role_utilization import build_role_utilization
    from src.modeling.layer3_archetype_elevation import build_archetype_elevation

    df = build_portable_talent()
    if not df.empty:
        df = build_role_utilization(df)
        df = build_archetype_elevation(df)
        df = build_scheme_amplification(df, use_possession_data=False)  # fast mode

        qualified = df[df.get("qualified", True) == True].copy()
        if not qualified.empty and "scheme_stability_index" in qualified.columns:
            top = qualified.nlargest(10, "scheme_stability_index")
            name_col = "player_name" if "player_name" in top.columns else "player_id"
            print("\nTop 10 Most Portable Players (Scheme Stability):")
            for _, row in top.iterrows():
                print(f"  {row.get(name_col, row['player_id']):25s} | "
                      f"Stability: {row['scheme_stability_index']:5.1f} | "
                      f"Class: {row.get('scheme_classification', 'N/A')}")
