"""
src/modeling/layer2_role_utilization.py
=============================================================================
BKE v2.0 — LAYER 2: Role Utilization Efficiency

Estimates:
  "How much of their talent is actually being expressed in their current role?"

Sub-layers:
  2A. Usage-Conditioned Efficiency — Playtype surplus over league avg at same
      usage bucket → Playtype Contribution Vector (PCV)
  2B. Role Utilization Efficiency (RUE) — Alignment between observed playtype
      distribution and archetype-optimal distribution

v2.0: z-score aggregation for surplus metrics. RUE computation unchanged
(cosine similarity is already interval-scaled).

Inputs:
  - Layer 1 output (player-season table with archetypes + playtypes)
  - player_archetypes.parquet (playtype frequency + PPP)

Output:
  Columns added to the player-season DataFrame:
    - Per-playtype surplus percentiles
    - playtype_surplus_total / z-score
    - role_utilization_efficiency (RUE)
    - RUE league/archetype percentiles
=============================================================================
"""

import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.modeling.model_config import (
    ROLE_UTILIZATION,
    PLAYER_ARCHETYPES_PATH,
    clean_id,
)
from src.modeling.percentile_engine import (
    add_league_percentiles,
    add_league_z_scores,
    add_grouped_percentiles,
    vectorized_percentile_rank,
)


# ---------------------------------------------------------------------------
# 2A. Usage-Conditioned Efficiency / Playtype Contribution Vector
# ---------------------------------------------------------------------------

def compute_league_playtype_benchmarks(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Compute league-average PPP by usage bucket for each playtype, per season.

    For each playtype, we bucket players by their possession share (how much
    they use that playtype) and compute the mean PPP within each bucket.
    This provides usage-conditioned benchmarks: "How efficient is the average
    player who uses this playtype at the same frequency?"

    Returns:
        Dict[season, DataFrame] with columns [playtype, usage_bucket, league_avg_ppp, n_players]
    """
    cfg = ROLE_UTILIZATION
    benchmarks = {}

    for season in df["season"].unique():
        season_df = df[df["season"] == season]
        rows = []

        for playtype in cfg.playtypes:
            poss_pct_col = f"{playtype}{cfg.poss_pct_suffix}"
            ppp_col = f"{playtype}{cfg.ppp_suffix}"
            poss_col = f"{playtype}{cfg.poss_suffix}"

            if poss_pct_col not in season_df.columns or ppp_col not in season_df.columns:
                continue

            # Filter to players with enough possessions in this playtype
            valid = season_df[
                (season_df[poss_col].fillna(0) >= cfg.min_playtype_poss) &
                (season_df[ppp_col].notna())
            ].copy() if poss_col in season_df.columns else season_df[season_df[ppp_col].notna()].copy()

            if valid.empty:
                continue

            # Bucket by usage (possession share)
            for i in range(len(cfg.usage_buckets) - 1):
                lo = cfg.usage_buckets[i]
                hi = cfg.usage_buckets[i + 1]
                bucket_mask = (valid[poss_pct_col] >= lo) & (valid[poss_pct_col] < hi)
                bucket_players = valid[bucket_mask]

                if len(bucket_players) >= 3:
                    avg_ppp = bucket_players[ppp_col].mean()
                else:
                    avg_ppp = valid[ppp_col].mean()  # fallback to overall

                rows.append({
                    "playtype": playtype,
                    "usage_bucket_lo": lo,
                    "usage_bucket_hi": hi,
                    "league_avg_ppp": avg_ppp,
                    "n_players": len(bucket_players),
                })

        benchmarks[season] = pd.DataFrame(rows)

    return benchmarks


def _find_usage_bucket(poss_pct: float, buckets: List[float]) -> Tuple[float, float]:
    """Find which usage bucket a possession share falls into."""
    for i in range(len(buckets) - 1):
        if poss_pct < buckets[i + 1]:
            return (buckets[i], buckets[i + 1])
    return (buckets[-2], buckets[-1])


def compute_playtype_surplus(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute playtype surplus: player PPP minus league avg PPP at same usage bucket.

    For each playtype, this measures:
      "How much more (or less) efficient is this player compared to the average
       player who uses this playtype at the same frequency?"

    Produces:
      - {PLAYTYPE}_surplus: raw surplus (PPP - league_avg_PPP)
      - {PLAYTYPE}_surplus_pctl: league percentile of surplus
      - playtype_surplus_total: possession-weighted total surplus
    """
    cfg = ROLE_UTILIZATION
    result = df.copy()

    # Pre-compute benchmarks
    benchmarks = compute_league_playtype_benchmarks(df)

    surplus_cols = []

    for playtype in cfg.playtypes:
        poss_pct_col = f"{playtype}{cfg.poss_pct_suffix}"
        ppp_col = f"{playtype}{cfg.ppp_suffix}"
        poss_col = f"{playtype}{cfg.poss_suffix}"
        surplus_col = f"{playtype}_surplus"

        if ppp_col not in result.columns:
            result[surplus_col] = np.nan
            surplus_cols.append(surplus_col)
            continue

        surpluses = []
        for idx, row in result.iterrows():
            season = row["season"]
            ppp = row.get(ppp_col, np.nan)
            poss_pct = row.get(poss_pct_col, 0)
            poss = row.get(poss_col, 0) if poss_col in result.columns else 0

            if pd.isna(ppp) or poss < cfg.min_playtype_poss:
                surpluses.append(np.nan)
                continue

            # Find usage bucket and league avg
            bench = benchmarks.get(season, pd.DataFrame())
            if bench.empty:
                surpluses.append(np.nan)
                continue

            bucket = _find_usage_bucket(poss_pct, cfg.usage_buckets)
            match = bench[
                (bench["playtype"] == playtype) &
                (bench["usage_bucket_lo"] == bucket[0]) &
                (bench["usage_bucket_hi"] == bucket[1])
            ]

            if not match.empty:
                league_avg = match.iloc[0]["league_avg_ppp"]
            else:
                # Fallback to overall playtype avg
                pt_bench = bench[bench["playtype"] == playtype]
                league_avg = pt_bench["league_avg_ppp"].mean() if not pt_bench.empty else ppp

            surpluses.append(ppp - league_avg)

        result[surplus_col] = surpluses
        surplus_cols.append(surplus_col)

    # Possession-weighted total surplus
    total_surplus = pd.Series(0.0, index=result.index)
    total_weight = pd.Series(0.0, index=result.index)

    for playtype in cfg.playtypes:
        surplus_col = f"{playtype}_surplus"
        poss_pct_col = f"{playtype}{cfg.poss_pct_suffix}"

        if surplus_col in result.columns and poss_pct_col in result.columns:
            valid = result[surplus_col].notna() & result[poss_pct_col].notna()
            weight = result[poss_pct_col].fillna(0)
            total_surplus += np.where(valid, result[surplus_col].fillna(0) * weight, 0)
            total_weight += np.where(valid, weight, 0)

    result["playtype_surplus_total"] = np.where(
        total_weight > 0,
        total_surplus / total_weight,
        np.nan
    )

    # Add percentiles for surpluses
    surplus_pctl_metrics = surplus_cols + ["playtype_surplus_total"]
    result = add_league_percentiles(result, surplus_pctl_metrics)

    # v2.0: Also add z-scores for surplus metrics (for z-score aggregation pipeline)
    result = add_league_z_scores(result, surplus_pctl_metrics)

    return result


# ---------------------------------------------------------------------------
# Playtype Contribution Vector (PCV)
# ---------------------------------------------------------------------------

def compute_playtype_contribution_vector(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the Playtype Contribution Vector (PCV) for each player.

    The PCV is the player's playtype surplus profile, percentile-ranked.
    It defines *how* impact is generated and *how* it maps to role.

    Output columns:
      - pcv_{playtype}: percentile rank of that playtype's surplus
      - pcv_dominant_playtype: the playtype with highest surplus
      - pcv_entropy: how evenly distributed surplus is across playtypes
    """
    cfg = ROLE_UTILIZATION
    result = df.copy()

    pcv_cols = []
    for playtype in cfg.playtypes:
        surplus_pctl = f"{playtype}_surplus_league_pctl"
        pcv_col = f"pcv_{playtype.lower()}"

        if surplus_pctl in result.columns:
            result[pcv_col] = result[surplus_pctl]
        else:
            # Compute from raw surplus
            surplus_col = f"{playtype}_surplus"
            if surplus_col in result.columns:
                result[pcv_col] = result.groupby("season")[surplus_col].transform(
                    vectorized_percentile_rank
                )
            else:
                result[pcv_col] = np.nan

        pcv_cols.append(pcv_col)

    # Dominant playtype (highest surplus percentile)
    pcv_data = result[pcv_cols].copy()
    # Handle all-NaN rows: idxmax with skipna=True, then fill NaN results
    has_any = pcv_data.notna().any(axis=1)
    dom_playtype = pd.Series("UNKNOWN", index=result.index)
    if has_any.any():
        dom_playtype[has_any] = (
            pcv_data.loc[has_any]
            .idxmax(axis=1)
            .str.replace("pcv_", "")
            .str.upper()
        )
    result["pcv_dominant_playtype"] = dom_playtype

    # PCV entropy (how evenly distributed)
    def _entropy(row):
        vals = row.dropna().values
        if len(vals) == 0:
            return np.nan
        vals = np.clip(vals, 1, 100)  # avoid log(0)
        probs = vals / vals.sum()
        return -np.sum(probs * np.log(probs + 1e-12))

    result["pcv_entropy"] = pcv_data.apply(_entropy, axis=1)

    return result


# ---------------------------------------------------------------------------
# 2B. Role Utilization Efficiency (RUE)
# ---------------------------------------------------------------------------

def compute_archetype_optimal_distributions(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Compute the "optimal" playtype distribution for each archetype.

    This is the mean playtype possession share for the top performers
    (above-median impact) within each archetype.

    Returns:
        {archetype: {playtype: mean_poss_pct}}
    """
    cfg = ROLE_UTILIZATION
    optimal = {}

    if "primary_archetype" not in df.columns:
        return optimal

    source_pref = (cfg.rue_optimal_source or "portable_talent").lower()
    source_candidates = {
        "portable_talent": ["portable_talent_z_adj", "portable_talent_z", "dimension_model_z"],
        "dimension_model": ["dimension_model_z", "portable_talent_z_adj", "portable_talent_z"],
        "rapm": ["rapm", "portable_talent_z_adj", "dimension_model_z"],
    }
    source_col = None
    for candidate in source_candidates.get(source_pref, source_candidates["portable_talent"]):
        if candidate in df.columns:
            source_col = candidate
            break

    for archetype in df["primary_archetype"].dropna().unique():
        arch_df = df[df["primary_archetype"] == archetype]

        # v2.7: decouple from RAPM by config source
        if source_col and source_col in arch_df.columns:
            median_val = arch_df[source_col].median()
            top_half = arch_df[arch_df[source_col] >= median_val]
        else:
            top_half = arch_df

        dist = {}
        for playtype in cfg.playtypes:
            poss_pct_col = f"{playtype}{cfg.poss_pct_suffix}"
            if poss_pct_col in top_half.columns:
                dist[playtype] = top_half[poss_pct_col].mean()
            else:
                dist[playtype] = 0.0

        optimal[archetype] = dist

    return optimal


def compute_role_utilization_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Role Utilization Efficiency (RUE).

    v2.6 FIX: Blended measure that combines:
      1. Cosine similarity to archetype optimal (40%) — role alignment
      2. Efficiency surplus quality (30%) — how well they perform across playtypes
      3. Volume-weighted utilization (30%) — reward high-usage players who create

    This prevents stars like Jokic from being penalized for having unique
    but highly effective role distributions. Pure cosine similarity treated
    unique distributions as "misalignment" when they're actually elite adaptation.

    High RUE → player is effective in their current role
    Low RUE → player may be miscast, underutilized, or inefficient
    """
    cfg = ROLE_UTILIZATION
    result = df.copy()

    # Compute archetype optimal distributions
    optimal_dists = compute_archetype_optimal_distributions(df)
    prob_cols = [c for c in result.columns if c.startswith("arch_prob_")]

    def _arch_to_prob_col(arch: str) -> str:
        return f"arch_prob_{str(arch).lower().replace('-', '_').replace(' ', '_')}"

    rue_scores = []

    for idx, row in result.iterrows():
        archetype = row.get("primary_archetype", None)

        # --- Component 1: Cosine similarity (role alignment) ---
        observed = []
        optimal = []

        # v2.7: weighted optimal vector from soft memberships when available
        weighted_opt = {p: 0.0 for p in cfg.playtypes}
        weight_sum = 0.0
        if prob_cols:
            for arch_name, dist in optimal_dists.items():
                p_col = _arch_to_prob_col(arch_name)
                if p_col not in result.columns:
                    continue
                w = row.get(p_col, 0.0) or 0.0
                if w <= 0:
                    continue
                weight_sum += w
                for playtype in cfg.playtypes:
                    weighted_opt[playtype] += w * (dist.get(playtype, 0.0) or 0.0)

        if weight_sum <= 0:
            if pd.isna(archetype) or archetype not in optimal_dists:
                rue_scores.append(np.nan)
                continue
            base_opt = optimal_dists[archetype]
        else:
            base_opt = {k: v / weight_sum for k, v in weighted_opt.items()}

        for playtype in cfg.playtypes:
            poss_pct_col = f"{playtype}{cfg.poss_pct_suffix}"
            obs_val = row.get(poss_pct_col, 0) or 0
            opt_val = base_opt.get(playtype, 0) or 0
            observed.append(obs_val)
            optimal.append(opt_val)

        observed = np.array(observed, dtype=float)
        optimal = np.array(optimal, dtype=float)

        dot = np.dot(observed, optimal)
        norm_obs = np.linalg.norm(observed)
        norm_opt = np.linalg.norm(optimal)

        if norm_obs > 0 and norm_opt > 0:
            cosine_sim = dot / (norm_obs * norm_opt)
            cosine_score = float(np.clip(cosine_sim, 0, 1))
        else:
            cosine_score = 0.5

        # --- Component 2: Efficiency surplus quality ---
        # Average surplus across playtypes the player actually uses
        surplus_sum = 0.0
        surplus_count = 0
        for playtype in cfg.playtypes:
            surplus_col = f"{playtype}_surplus"
            poss_col = f"{playtype}{cfg.poss_suffix}"
            if surplus_col in result.columns:
                surplus_val = row.get(surplus_col, np.nan)
                poss_val = row.get(poss_col, 0) if poss_col in result.columns else 0
                if not pd.isna(surplus_val) and poss_val >= cfg.min_playtype_poss:
                    surplus_sum += surplus_val
                    surplus_count += 1

        if surplus_count > 0:
            avg_surplus = surplus_sum / surplus_count
            # Map [-0.15, +0.15] PPP surplus → [0, 1]
            efficiency_score = float(np.clip((avg_surplus + 0.15) / 0.30, 0, 1))
        else:
            efficiency_score = 0.5

        # --- Component 3: Volume-weighted utilization ---
        # Players who use more playtypes at volume get a creation bonus
        n_active_playtypes = 0
        total_poss_share = 0.0
        for playtype in cfg.playtypes:
            poss_pct_col = f"{playtype}{cfg.poss_pct_suffix}"
            poss_col = f"{playtype}{cfg.poss_suffix}"
            poss_pct = row.get(poss_pct_col, 0) or 0
            poss = row.get(poss_col, 0) if poss_col in result.columns else 0
            if poss >= cfg.min_playtype_poss and poss_pct > 0.02:
                n_active_playtypes += 1
                total_poss_share += poss_pct

        # Reward breadth: more active playtypes = more versatile role usage
        # 1-2 playtypes = low (0.2), 3-4 = moderate (0.5-0.7), 5+ = high (0.8-1.0)
        breadth_score = float(np.clip(n_active_playtypes / 6.0, 0, 1))
        volume_score = float(np.clip(total_poss_share, 0, 1))
        utilization_score = 0.6 * breadth_score + 0.4 * volume_score

        # --- Blended RUE ---
        rue = (
            0.40 * cosine_score +
            0.30 * efficiency_score +
            0.30 * utilization_score
        )
        rue_scores.append(float(np.clip(rue, 0, 1)))

    result["role_utilization_raw"] = rue_scores

    # Convert to percentile
    result["role_utilization_efficiency"] = result.groupby("season")["role_utilization_raw"].transform(
        vectorized_percentile_rank
    )

    # v2.0: Also compute RUE z-score for z-score aggregation pipeline
    result = add_league_z_scores(result, ["role_utilization_raw"])

    # Add archetype-grouped percentile
    if "primary_archetype" in result.columns:
        result = add_grouped_percentiles(
            result,
            ["role_utilization_raw"],
            group_col="primary_archetype",
            suffix="_pctl",
        )

    return result


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def build_role_utilization(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full Layer 2 pipeline: playtype surplus → PCV → RUE.

    Takes Layer 1 output DataFrame and adds Layer 2 columns.
    """
    print("\n" + "=" * 60)
    print("LAYER 2: ROLE UTILIZATION EFFICIENCY")
    print("=" * 60)

    qualified = df[df.get("qualified", True) == True].copy() if "qualified" in df.columns else df.copy()

    if qualified.empty:
        print("  No qualified players to process")
        return df

    # 2A. Playtype surplus
    print("  Computing playtype surplus...")
    qualified = compute_playtype_surplus(qualified)

    # 2A+. Playtype Contribution Vector
    print("  Building Playtype Contribution Vectors...")
    qualified = compute_playtype_contribution_vector(qualified)

    # 2B. Role Utilization Efficiency
    print("  Computing Role Utilization Efficiency...")
    qualified = compute_role_utilization_efficiency(qualified)

    # Merge back — use concat to avoid DataFrame fragmentation
    new_cols = [c for c in qualified.columns if c not in df.columns]
    if new_cols:
        fill = pd.DataFrame(np.nan, index=df.index, columns=new_cols)
        df = pd.concat([df, fill], axis=1)
        df.loc[qualified.index, new_cols] = qualified[new_cols].values

    n = qualified["role_utilization_efficiency"].notna().sum()
    print(f"\n  Layer 2 complete: {n} players with RUE scores")
    if n > 0:
        print(f"  RUE range: {qualified['role_utilization_efficiency'].min():.1f} - "
              f"{qualified['role_utilization_efficiency'].max():.1f}")

    return df


# ---------------------------------------------------------------------------
# CLI Entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.modeling.layer1_portable_talent import build_portable_talent
    df = build_portable_talent()
    if not df.empty:
        df = build_role_utilization(df)
        qualified = df[df.get("qualified", True) == True].copy()
        if not qualified.empty:
            top = qualified.nlargest(10, "role_utilization_efficiency")
            name_col = "player_name" if "player_name" in top.columns else "player_id"
            print("\nTop 10 Role Utilization Efficiency:")
            for _, row in top.iterrows():
                print(f"  {row.get(name_col, row['player_id']):25s} | "
                      f"RUE: {row['role_utilization_efficiency']:5.1f} | "
                      f"Archetype: {row.get('primary_archetype', 'N/A')}")
