"""
src/modeling/layer1_portable_talent.py
=============================================================================
BKE v1.5 — LAYER 1: True Portable Talent (Context-Neutral Layer)

Estimates:
  "How good is this player independent of role volume and scheme?"

Sub-layers:
  1A. Multi-Year Bayesian RAPM (ridge, prior-shrunk)
  1B. Luck Adjustment (shooting, opponent 3PT, FT variance)
  1C. Portable Skill Components (gravity, rim protection, passing, etc.)
  1D. Portable Talent Score (PTS) — combined percentile-weighted score

Inputs:
  - player_rapm.parquet (RAPM / ORAPM / DRAPM)
  - modeling_inputs_all.parquet (DARKO, linear stats)
  - player_profiles_advanced.parquet (box score + four factors)
  - player_archetypes.parquet (offensive archetype + playtypes)
  - defensive_archetypes_v2.parquet (defensive metrics)
  - player_position_estimates.parquet (positional percentages)

Output:
  DataFrame with per-player portable talent decomposition and
  three-level percentile standardization.
=============================================================================
"""

import os
import sys
from typing import Dict, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.modeling.model_config import (
    PORTABLE_TALENT,
    PORTABLE_SKILL_COMPONENTS,
    RAPM_PATH,
    MODELING_INPUTS_PATH,
    PLAYER_PROFILES_PATH,
    PLAYER_ARCHETYPES_PATH,
    DEFENSIVE_ARCHETYPES_PATH,
    POSITION_ESTIMATES_PATH,
    RAPM_PREFERRED_TYPE,
    RAPM_FALLBACK_TYPE,
    MIN_MINUTES,
    MIN_GP,
    MIN_MPG,
    MIN_POSSESSIONS,
    SEASONS,
    clean_id,
)
from src.modeling.percentile_engine import (
    add_league_percentiles,
    add_grouped_percentiles,
    vectorized_percentile_rank,
)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_rapm() -> pd.DataFrame:
    """Load RAPM data, preferring pooled split for O/D separation."""
    if not os.path.exists(RAPM_PATH):
        print(f"  WARNING: {RAPM_PATH} not found")
        return pd.DataFrame()

    rapm = pd.read_parquet(RAPM_PATH)
    rapm["player_id"] = rapm["player_id"].astype(str).apply(clean_id)
    rapm["season"] = rapm["season"].astype(str)

    # Prefer pooled_split, then fall back to single_season_split
    preferred = rapm[rapm["RAPM_type"] == RAPM_PREFERRED_TYPE].copy()
    fallback = rapm[rapm["RAPM_type"] == RAPM_FALLBACK_TYPE].copy()

    # Merge: prefer pooled, fill with fallback where missing
    combined = preferred.copy()
    missing_keys = set(
        zip(fallback["season"], fallback["player_id"])
    ) - set(
        zip(combined["season"], combined["player_id"])
    )
    if missing_keys:
        fallback_rows = fallback[
            fallback.apply(lambda r: (r["season"], r["player_id"]) in missing_keys, axis=1)
        ]
        combined = pd.concat([combined, fallback_rows], ignore_index=True)

    keep = ["season", "player_id", "player_name", "RAPM", "ORAPM", "DRAPM",
            "possessions_played", "RAPM_type"]
    for col in keep:
        if col not in combined.columns:
            combined[col] = np.nan

    return combined[keep].rename(columns={
        "RAPM": "rapm", "ORAPM": "orapm", "DRAPM": "drapm",
    })


def load_modeling_inputs() -> pd.DataFrame:
    """Load DARKO + linear stats from modeling inputs."""
    if not os.path.exists(MODELING_INPUTS_PATH):
        print(f"  WARNING: {MODELING_INPUTS_PATH} not found")
        return pd.DataFrame()

    df = pd.read_parquet(MODELING_INPUTS_PATH)
    df["player_id"] = df["player_id"].astype(str).apply(clean_id)
    df["season"] = df["season"].astype(str)

    keep = ["season", "player_id", "darko_dpm", "darko_odpm", "darko_ddpm",
            "TS_pct", "eFG_pct", "USG_pct", "AST_pct", "TOV_pct",
            "OREB_pct", "DREB_pct", "PTS_per36", "AST_per36", "REB_per36"]
    available = [c for c in keep if c in df.columns]
    return df[available].drop_duplicates(subset=["season", "player_id"], keep="last")


def load_player_profiles() -> pd.DataFrame:
    """Load advanced player profiles (box score + four factors)."""
    if not os.path.exists(PLAYER_PROFILES_PATH):
        print(f"  WARNING: {PLAYER_PROFILES_PATH} not found")
        return pd.DataFrame()

    df = pd.read_parquet(PLAYER_PROFILES_PATH)
    df["player_id"] = df["player_id"].astype(str).apply(clean_id)
    df["season"] = df["season"].astype(str)

    keep = ["season", "player_id", "player_name", "GP", "MIN", "MPG",
            "PTS", "AST", "REB", "ORB", "DRB", "STL", "BLK", "TOV",
            "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
            "TS_PCT", "EFG_PCT", "USG_RATE", "AST_PCT", "TOV_PCT",
            "ORTG", "DRTG", "NET_RTG",
            "POSS_OFF", "POSS_DEF"]
    available = [c for c in keep if c in df.columns]
    return df[available].drop_duplicates(subset=["season", "player_id"], keep="last")


def load_archetypes() -> pd.DataFrame:
    """Load offensive archetypes with playtype data."""
    if not os.path.exists(PLAYER_ARCHETYPES_PATH):
        print(f"  WARNING: {PLAYER_ARCHETYPES_PATH} not found")
        return pd.DataFrame()

    df = pd.read_parquet(PLAYER_ARCHETYPES_PATH)

    # Normalize ID column
    id_col = "PLAYER_ID" if "PLAYER_ID" in df.columns else "player_id"
    df["player_id"] = df[id_col].astype(str).apply(clean_id)

    season_col = "SEASON" if "SEASON" in df.columns else "season"
    df["season"] = df[season_col].astype(str)

    # Core archetype columns
    core_cols = ["season", "player_id", "primary_archetype", "secondary_archetype",
                 "role_confidence", "role_effectiveness",
                 "BALL_DOMINANT_PCT", "PLAYMAKING_SCORE",
                 "FG3_PCT", "FG3A_PER36", "TS_ZSCORE"]

    # Dynamically include ALL playtype columns (PPP, POSS_PCT, POSS, etc.)
    # These are critical for Layer 2 (Role Utilization Efficiency)
    playtype_prefixes = [
        "ISOLATION", "PRBALLHANDLER", "POSTUP", "CUT", "PRROLLMAN",
        "HANDOFF", "OFFSCREEN", "SPOTUP", "TRANSITION",
    ]
    playtype_cols = [c for c in df.columns
                     if any(c.startswith(prefix) for prefix in playtype_prefixes)]

    keep = core_cols + playtype_cols
    available = [c for c in keep if c in df.columns]
    return df[available].drop_duplicates(subset=["season", "player_id"], keep="last")


def load_defensive_archetypes() -> pd.DataFrame:
    """Load defensive archetypes with defensive metrics."""
    if not os.path.exists(DEFENSIVE_ARCHETYPES_PATH):
        print(f"  WARNING: {DEFENSIVE_ARCHETYPES_PATH} not found")
        return pd.DataFrame()

    df = pd.read_parquet(DEFENSIVE_ARCHETYPES_PATH)
    id_col = "PLAYER_ID" if "PLAYER_ID" in df.columns else "player_id"
    df["player_id"] = df[id_col].astype(str).apply(clean_id)

    season_col = "SEASON" if "SEASON" in df.columns else "season"
    df["season"] = df[season_col].astype(str)

    keep = ["season", "player_id", "defensive_archetype",
            "switch_score", "versatility_pctl", "assignment_difficulty",
            "rim_protection_index_pctl", "engagement_pctl",
            "BLK_PCT", "STL_PER100_DEF_POSS"]
    available = [c for c in keep if c in df.columns]
    return df[available].drop_duplicates(subset=["season", "player_id"], keep="last")


def load_position_estimates() -> pd.DataFrame:
    """Load position estimates for positional bucketing."""
    if not os.path.exists(POSITION_ESTIMATES_PATH):
        print(f"  WARNING: {POSITION_ESTIMATES_PATH} not found")
        return pd.DataFrame()

    df = pd.read_parquet(POSITION_ESTIMATES_PATH)
    id_col = "PLAYER_ID" if "PLAYER_ID" in df.columns else "player_id"
    df["player_id"] = df[id_col].astype(str).apply(clean_id)

    season_col = "SEASON" if "SEASON" in df.columns else "season"
    df["season"] = df[season_col].astype(str)

    # Find position percentage columns
    pos_cols = [c for c in df.columns if c.startswith("pct_")]
    keep = ["season", "player_id"] + pos_cols
    if "primary_position_estimate" in df.columns:
        keep.append("primary_position_estimate")
    available = [c for c in keep if c in df.columns]
    return df[available].drop_duplicates(subset=["season", "player_id"], keep="last")


# ---------------------------------------------------------------------------
# Position Bucketing
# ---------------------------------------------------------------------------

def assign_position_bucket(row: pd.Series) -> str:
    """
    Assign a player to a position bucket based on position estimates.

    Uses the primary_position_estimate if available, otherwise infers
    from pct_ columns.
    """
    if "primary_position_estimate" in row.index and pd.notna(row.get("primary_position_estimate")):
        pos = str(row["primary_position_estimate"]).upper()
        if pos in ("PG", "SG"):
            return "Guard"
        elif pos in ("SG-SF", "SF-SG", "G-F"):
            return "Guard-Forward"
        elif pos in ("SF", "PF"):
            return "Forward"
        elif pos in ("PF-C", "C-PF"):
            return "Forward-Center"
        elif pos == "C":
            return "Center"

    # Fallback: use pct_ columns to infer
    pct_cols = {
        "pct_pg": "Guard", "pct_sg": "Guard",
        "pct_sf": "Forward", "pct_pf": "Forward",
        "pct_c": "Center",
    }
    best_pos = "Forward"  # default
    best_val = -1
    for col, bucket in pct_cols.items():
        val = row.get(col, 0) or 0
        if val > best_val:
            best_val = val
            best_pos = bucket
    return best_pos


# ---------------------------------------------------------------------------
# 1B. Luck Adjustment
# ---------------------------------------------------------------------------

def apply_luck_adjustment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adjust shooting-based metrics for variance/luck.

    Regresses extreme shooting performances toward league mean to
    isolate true skill from single-season variance.
    """
    result = df.copy()
    cfg = PORTABLE_TALENT

    for season in result["season"].unique():
        mask = result["season"] == season

        # 3PT% luck adjustment
        if "FG3_PCT" in result.columns:
            league_mean_3pt = result.loc[mask, "FG3_PCT"].mean()
            if not np.isnan(league_mean_3pt):
                raw_3pt = result.loc[mask, "FG3_PCT"]
                adjusted = league_mean_3pt + (1 - cfg.shooting_regression_rate) * (raw_3pt - league_mean_3pt)
                result.loc[mask, "FG3_PCT_adj"] = adjusted
            else:
                result.loc[mask, "FG3_PCT_adj"] = result.loc[mask, "FG3_PCT"]
        
        # TS% luck adjustment (lighter touch)
        if "TS_PCT" in result.columns:
            league_mean_ts = result.loc[mask, "TS_PCT"].mean()
            if not np.isnan(league_mean_ts):
                raw_ts = result.loc[mask, "TS_PCT"]
                # Less regression for TS since it includes FT
                adjusted = league_mean_ts + (1 - cfg.ft_regression_rate) * (raw_ts - league_mean_ts)
                result.loc[mask, "TS_PCT_adj"] = adjusted
            else:
                result.loc[mask, "TS_PCT_adj"] = result.loc[mask, "TS_PCT"]

    return result


# ---------------------------------------------------------------------------
# 1C. Portable Skill Components
# ---------------------------------------------------------------------------

def compute_skill_components(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute portable skill component scores.

    Each component captures a dimensionally-independent skill that
    scales across team contexts and schemes.
    """
    result = df.copy()

    # --- Shooting Gravity ---
    # Combines 3PT volume, efficiency, and true shooting
    ts = result.get("TS_PCT_adj", result.get("TS_PCT", pd.Series(dtype=float)))
    fg3a = result.get("FG3A_PER36", pd.Series(0, index=result.index))
    fg3_pct = result.get("FG3_PCT_adj", result.get("FG3_PCT", pd.Series(dtype=float)))

    # Normalize each sub-component to [0, 1] within season
    result["_ts_norm"] = result.groupby("season")[ts.name if hasattr(ts, 'name') and ts.name else "TS_PCT"].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
    ) if "TS_PCT" in result.columns or "TS_PCT_adj" in result.columns else 0.5

    if "FG3A_PER36" in result.columns:
        result["_fg3a_norm"] = result.groupby("season")["FG3A_PER36"].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
        )
    else:
        result["_fg3a_norm"] = 0.5

    if "FG3_PCT" in result.columns or "FG3_PCT_adj" in result.columns:
        fg3_col = "FG3_PCT_adj" if "FG3_PCT_adj" in result.columns else "FG3_PCT"
        result["_fg3pct_norm"] = result.groupby("season")[fg3_col].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
        )
    else:
        result["_fg3pct_norm"] = 0.5

    result["shooting_gravity"] = (
        0.40 * result["_ts_norm"] +
        0.30 * result["_fg3a_norm"] +
        0.30 * result["_fg3pct_norm"]
    )

    # --- Passing Efficiency ---
    if "AST_per36" in result.columns or "AST_PER36" in result.columns:
        ast_col = "AST_per36" if "AST_per36" in result.columns else "AST_PER36"
    else:
        ast_col = None

    if ast_col:
        result["_ast_norm"] = result.groupby("season")[ast_col].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
        )
    else:
        result["_ast_norm"] = 0.5

    if "PLAYMAKING_SCORE" in result.columns:
        result["_playmaking_norm"] = result.groupby("season")["PLAYMAKING_SCORE"].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
        )
    else:
        result["_playmaking_norm"] = 0.5

    # Invert TOV for passing (lower is better)
    tov_col = "TOV_pct" if "TOV_pct" in result.columns else ("TOV_PCT" if "TOV_PCT" in result.columns else None)
    if tov_col:
        result["_tov_inv_norm"] = result.groupby("season")[tov_col].transform(
            lambda x: 1.0 - (x - x.min()) / (x.max() - x.min() + 1e-9)
        )
    else:
        result["_tov_inv_norm"] = 0.5

    result["passing_efficiency"] = (
        0.45 * result["_ast_norm"] +
        0.30 * result["_playmaking_norm"] +
        0.25 * result["_tov_inv_norm"]
    )

    # --- Rim Protection ---
    if "rim_protection_index_pctl" in result.columns:
        result["rim_protection"] = result["rim_protection_index_pctl"] / 100.0
    elif "BLK_PCT" in result.columns:
        result["rim_protection"] = result.groupby("season")["BLK_PCT"].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
        )
    else:
        result["rim_protection"] = 0.0

    # --- Defensive Versatility ---
    if "versatility_pctl" in result.columns:
        result["defensive_versatility"] = result["versatility_pctl"] / 100.0
    elif "switch_score" in result.columns:
        result["defensive_versatility"] = result.groupby("season")["switch_score"].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
        )
    else:
        result["defensive_versatility"] = 0.0

    # --- Turnover Control ---
    if tov_col:
        result["turnover_control"] = result["_tov_inv_norm"]
    else:
        result["turnover_control"] = 0.5

    # --- Rebounding ---
    reb_col = "REB_per36" if "REB_per36" in result.columns else ("REB_PER36" if "REB_PER36" in result.columns else None)
    if reb_col:
        result["rebounding"] = result.groupby("season")[reb_col].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
        )
    else:
        result["rebounding"] = 0.0

    # Clean up temp columns
    temp_cols = [c for c in result.columns if c.startswith("_")]
    result = result.drop(columns=temp_cols)

    return result


# ---------------------------------------------------------------------------
# 1D. Portable Talent Score
# ---------------------------------------------------------------------------

def compute_portable_talent_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Portable Talent Score (PTS) as a weighted combination of:
      - Adjusted RAPM percentile
      - Portable skill percentiles
      - Stability-adjusted impact

    Percentile-based computation ensures cross-era comparability.
    """
    result = df.copy()
    cfg = PORTABLE_TALENT

    # RAPM component (league percentile)
    rapm_col = "rapm_league_pctl" if "rapm_league_pctl" in result.columns else None
    if rapm_col is None and "rapm" in result.columns:
        # Compute inline if not yet percentiled
        result["rapm_league_pctl"] = result.groupby("season")["rapm"].transform(
            vectorized_percentile_rank
        )
        rapm_col = "rapm_league_pctl"

    rapm_score = result[rapm_col] / 100.0 if rapm_col else 0.5

    # Stability weight: players with more possessions get more trust
    if "possessions_played" in result.columns:
        stability = np.clip(
            result["possessions_played"] / cfg.min_poss_full_weight,
            0.0, 1.0
        )
    else:
        stability = 1.0

    # Skill components
    skill_cols = {
        "shooting_gravity": cfg.w_shooting,
        "passing_efficiency": cfg.w_passing,
        "rim_protection": cfg.w_rim_protection,
        "defensive_versatility": cfg.w_defensive_versatility,
        "turnover_control": cfg.w_turnover_control,
        "rebounding": cfg.w_rebounding,
    }

    skill_score = pd.Series(0.0, index=result.index)
    total_skill_weight = 0.0
    for col, weight in skill_cols.items():
        if col in result.columns:
            skill_score += weight * result[col].fillna(0.5)
            total_skill_weight += weight

    # Normalize skill score
    if total_skill_weight > 0:
        skill_score = skill_score / total_skill_weight
    else:
        skill_score = 0.5

    # Combine: RAPM-weighted portable talent
    pts = (
        cfg.w_rapm * rapm_score * stability +
        (1 - cfg.w_rapm) * skill_score
    )

    # Scale to interpretable range (0-100 percentile-like)
    result["portable_talent_raw"] = pts
    result["portable_talent_score"] = result.groupby("season")["portable_talent_raw"].transform(
        vectorized_percentile_rank
    )

    # Stability-adjusted score
    result["stability_weight"] = stability if isinstance(stability, pd.Series) else stability
    result["portable_talent_stability_adj"] = result["portable_talent_score"] * (
        0.8 + 0.2 * result["stability_weight"]
    )

    return result


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def build_portable_talent(seasons: Optional[list] = None) -> pd.DataFrame:
    """
    Full Layer 1 pipeline: load data → merge → adjust → score → percentile.

    Returns DataFrame with portable talent decomposition for all qualified
    player-seasons.
    """
    seasons = seasons or SEASONS
    print("\n" + "=" * 60)
    print("LAYER 1: PORTABLE TALENT")
    print("=" * 60)

    # 1. Load all data sources
    print("  Loading data sources...")
    rapm = load_rapm()
    modeling = load_modeling_inputs()
    profiles = load_player_profiles()
    archetypes = load_archetypes()
    def_archetypes = load_defensive_archetypes()
    positions = load_position_estimates()

    if rapm.empty and profiles.empty:
        print("  ERROR: No RAPM or profile data available")
        return pd.DataFrame()

    # 2. Merge into unified player-season table
    print("  Merging data...")
    # Start with RAPM as the base if available, otherwise profiles
    if not rapm.empty:
        base = rapm.copy()
    else:
        base = profiles[["season", "player_id"]].drop_duplicates()
        base["rapm"] = np.nan
        base["orapm"] = np.nan
        base["drapm"] = np.nan
        base["possessions_played"] = np.nan

    # Merge in all data sources
    if not profiles.empty:
        base = base.merge(profiles, on=["season", "player_id"], how="left", suffixes=("", "_prof"))
        # Resolve name conflicts
        for col in base.columns:
            if col.endswith("_prof"):
                primary = col.replace("_prof", "")
                if primary in base.columns:
                    base[primary] = base[primary].fillna(base[col])
                else:
                    base[primary] = base[col]
                base = base.drop(columns=[col])

    if not modeling.empty:
        base = base.merge(modeling, on=["season", "player_id"], how="left", suffixes=("", "_mod"))
        for col in base.columns:
            if col.endswith("_mod"):
                primary = col.replace("_mod", "")
                if primary in base.columns:
                    base[primary] = base[primary].fillna(base[col])
                else:
                    base[primary] = base[col]
                base = base.drop(columns=[col])

    if not archetypes.empty:
        base = base.merge(archetypes, on=["season", "player_id"], how="left", suffixes=("", "_arch"))
        for col in base.columns:
            if col.endswith("_arch"):
                primary = col.replace("_arch", "")
                if primary in base.columns:
                    base[primary] = base[primary].fillna(base[col])
                else:
                    base[primary] = base[col]
                base = base.drop(columns=[col])

    if not def_archetypes.empty:
        base = base.merge(def_archetypes, on=["season", "player_id"], how="left", suffixes=("", "_def"))
        for col in base.columns:
            if col.endswith("_def"):
                primary = col.replace("_def", "")
                if primary in base.columns:
                    base[primary] = base[primary].fillna(base[col])
                else:
                    base[primary] = base[col]
                base = base.drop(columns=[col])

    if not positions.empty:
        base = base.merge(positions, on=["season", "player_id"], how="left", suffixes=("", "_pos"))
        for col in base.columns:
            if col.endswith("_pos"):
                base = base.drop(columns=[col])

    # 3. Assign position buckets
    print("  Assigning position buckets...")
    base["position_bucket"] = base.apply(assign_position_bucket, axis=1)

    # 4. Qualify players
    print("  Qualifying players...")
    qualified_mask = pd.Series(True, index=base.index)
    if "MIN" in base.columns:
        qualified_mask &= base["MIN"] >= MIN_MINUTES
    if "GP" in base.columns:
        qualified_mask &= base["GP"] >= MIN_GP
    if "MPG" in base.columns:
        qualified_mask &= base["MPG"] >= MIN_MPG

    base["qualified"] = qualified_mask
    qualified = base[qualified_mask].copy()
    print(f"  Qualified players: {len(qualified)}")

    if qualified.empty:
        print("  WARNING: No qualified players found")
        return base

    # 5. Apply luck adjustment
    print("  Applying luck adjustment...")
    qualified = apply_luck_adjustment(qualified)

    # 6. Compute skill components
    print("  Computing portable skill components...")
    qualified = compute_skill_components(qualified)

    # 7. Add league percentiles for key metrics
    print("  Computing percentile standardization...")
    pctl_metrics = ["rapm", "orapm", "drapm",
                    "shooting_gravity", "passing_efficiency",
                    "rim_protection", "defensive_versatility",
                    "turnover_control", "rebounding"]
    qualified = add_league_percentiles(qualified, pctl_metrics)

    # 8. Add positional percentiles
    qualified = add_grouped_percentiles(
        qualified, pctl_metrics, group_col="position_bucket",
        suffix="_pctl",
    )

    # 9. Add archetype percentiles
    if "primary_archetype" in qualified.columns:
        qualified = add_grouped_percentiles(
            qualified, pctl_metrics, group_col="primary_archetype",
            suffix="_pctl",
        )

    # 10. Compute Portable Talent Score
    print("  Computing Portable Talent Score...")
    qualified = compute_portable_talent_score(qualified)

    # 11. Merge back with unqualified players
    # Unqualified players get NaN for all computed columns
    new_cols = [c for c in qualified.columns if c not in base.columns]
    for col in new_cols:
        base[col] = np.nan

    base.loc[qualified.index, new_cols] = qualified[new_cols]

    n_seasons = base["season"].nunique()
    print(f"\n  Layer 1 complete: {len(qualified)} qualified player-seasons across {n_seasons} seasons")
    print(f"  Portable Talent Score range: "
          f"{qualified['portable_talent_score'].min():.1f} - {qualified['portable_talent_score'].max():.1f}")

    return base


# ---------------------------------------------------------------------------
# CLI Entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = build_portable_talent()
    if not result.empty:
        # Show top 10 by portable talent
        qualified = result[result["qualified"] == True].copy()
        if not qualified.empty:
            top = qualified.nlargest(10, "portable_talent_score")
            name_col = "player_name" if "player_name" in top.columns else "player_id"
            print("\nTop 10 Portable Talent Scores:")
            for _, row in top.iterrows():
                print(f"  {row.get(name_col, row['player_id']):25s} | "
                      f"PTS: {row['portable_talent_score']:5.1f} | "
                      f"RAPM: {row.get('rapm', 0):+6.2f} | "
                      f"Season: {row['season']}")
