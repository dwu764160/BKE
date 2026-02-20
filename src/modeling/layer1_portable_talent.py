"""
src/modeling/layer1_portable_talent.py
=============================================================================
BKE v2.0 — LAYER 1: True Portable Talent (Context-Neutral Layer)

Estimates:
  "How good is this player independent of role volume and scheme?"

v2.0 structure:
  1A. RAPM Backbone (25%) — Multi-source impact z-score
  1B. Playtype Efficiency (20%) — Usage-adjusted playtype composite
  1C. Portable Dimension Model (55%) — 8 logically independent dimensions:
      OFFENSIVE: Shooting Gravity, Driving Gravity, Playmaking,
                 Extra Possession Creation, Turnover Control
      DEFENSIVE: Defensive Playmaking, Defensive Impact, Defensive Versatility

v2.0 fixes:
  - Z-score aggregation replaces percentile averaging (Fix #1)
  - Raw → Z → Weighted Sum → Final Z → Final Percentile
  - Driving Gravity: removed FT%, focus on rim pressure creation
  - Turnover Control: restored as standalone offensive dimension
  - Extra Possession Creation: cross-domain (40% off, 60% def)
  - Bayesian shrinkage for noisy defensive metrics
  - 8 equally-weighted dimensions in Layer 1C (initially)

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
    PORTABLE_DIMENSIONS,
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
    BAYESIAN_SHRINKAGE,
    SEASONS,
    clean_id,
)
from src.modeling.percentile_engine import (
    add_league_percentiles,
    add_grouped_percentiles,
    add_league_z_scores,
    add_grouped_z_scores,
    compute_z_score,
    z_to_percentile,
    weighted_z_composite,
    apply_bayesian_shrinkage,
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

    # v2.0: Additional tracking/feature columns needed for 8-dimension model
    tracking_cols = [
        # Driving Gravity (Dim 2)
        "DRIVES", "DRIVES_PER36", "DRIVE_PTS", "DRIVE_FG_PCT", "DRIVE_AST", "DRIVE_TOV",
        "AT_RIM_FREQ", "AT_RIM_FG_PCT", "AT_RIM_PLUS_PAINT_FREQ",
        "PAINT_FREQ", "PAINT_FGA", "FT_RATE",
        # Playmaking (Dim 3)
        "POTENTIAL_AST", "POTENTIAL_AST_PER36",
        "SECONDARY_AST", "SECONDARY_AST_PER36",
        # Turnover Control (Dim 7)
        "TOV", "TOV_PER36", "TOV_PCT",
        # Portability: PCV entropy (emb_entropy_norm as proxy)
        "pcv_entropy", "emb_entropy", "emb_entropy_norm",
    ]

    # Dynamically include ALL playtype columns (PPP, POSS_PCT, POSS, etc.)
    # These are critical for Layer 2 (Role Utilization Efficiency)
    playtype_prefixes = [
        "ISOLATION", "PRBALLHANDLER", "POSTUP", "CUT", "PRROLLMAN",
        "HANDOFF", "OFFSCREEN", "SPOTUP", "TRANSITION",
    ]
    playtype_cols = [c for c in df.columns
                     if any(c.startswith(prefix) for prefix in playtype_prefixes)]

    keep = core_cols + tracking_cols + playtype_cols
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
            "BLK_PCT", "STL_PER100_DEF_POSS",
            # v2.0: Additional columns for 8-dimension model
            "hustle_score", "engagement_score", "DEFLECTIONS",
            "matchup_diversity_pctl", "d_results_pctl",
            "defensive_fit", "size_band"]
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

    v2.0: Renamed ft_regression_rate → ts_regression_rate for clarity.
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
        
        # TS% luck adjustment (lighter touch — includes FT component)
        if "TS_PCT" in result.columns:
            league_mean_ts = result.loc[mask, "TS_PCT"].mean()
            if not np.isnan(league_mean_ts):
                raw_ts = result.loc[mask, "TS_PCT"]
                adjusted = league_mean_ts + (1 - cfg.ts_regression_rate) * (raw_ts - league_mean_ts)
                result.loc[mask, "TS_PCT_adj"] = adjusted
            else:
                result.loc[mask, "TS_PCT_adj"] = result.loc[mask, "TS_PCT"]

    return result


# ---------------------------------------------------------------------------
# 1C. Portable Dimension Model — 8 Independent Dimensions (v2.0)
# ---------------------------------------------------------------------------

def _zscore_within_season(df: pd.DataFrame, col: str, invert: bool = False) -> pd.Series:
    """
    Compute z-score of a column within each season.

    Parameters
    ----------
    invert : If True, negate z-score (lower raw = higher score, e.g. TOV%).
    """
    z = df.groupby("season")[col].transform(
        lambda x: compute_z_score(x, winsorize=3.5)
    )
    if invert:
        z = -z
    return z


def _avg_z_components(z_components: dict) -> pd.Series:
    """Average z-score components, filling individual NaN with 0 (league avg)."""
    if not z_components:
        raise ValueError("No z-components to average")
    filled = [v.fillna(0) for v in z_components.values()]
    return sum(filled) / len(filled)


def compute_dimension_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the 8-dimension portable talent model (Layer 1C).

    v2.0 uses z-score aggregation within each dimension:
      1. Per sub-metric: compute z-score within season
      2. Within dimension: weighted average of z-scores
      3. Cross-dimension: equal-weighted z-score composite

    Returns DataFrame with dimension z-scores and composite.
    """
    result = df.copy()
    cfg = PORTABLE_TALENT

    # ===== DIMENSION 1: SHOOTING GRAVITY =====
    # TS% (luck-adjusted), 3PT%, 3PA volume, catch-and-shoot
    z_components = {}
    ts_col = "TS_PCT_adj" if "TS_PCT_adj" in result.columns else "TS_PCT"
    if ts_col in result.columns:
        z_components["ts"] = _zscore_within_season(result, ts_col)

    fg3_col = "FG3_PCT_adj" if "FG3_PCT_adj" in result.columns else "FG3_PCT"
    if fg3_col in result.columns:
        z_components["fg3_pct"] = _zscore_within_season(result, fg3_col)

    if "FG3A_PER36" in result.columns:
        z_components["fg3a_vol"] = _zscore_within_season(result, "FG3A_PER36")

    if "CATCH_SHOOT_FG3_PCT" in result.columns:
        z_components["cs_fg3"] = _zscore_within_season(result, "CATCH_SHOOT_FG3_PCT")

    if z_components:
        result["dim_shooting_gravity_z"] = _avg_z_components(z_components)
    else:
        result["dim_shooting_gravity_z"] = 0.0

    # ===== DIMENSION 2: DRIVING GRAVITY (v2.0 REVISED) =====
    # Focus on rim pressure: drives, rim FGA, fouls drawn (FT_RATE), paint freq
    # REMOVED: FT%, perimeter initiation filtering
    z_components = {}
    if "DRIVES_PER36" in result.columns:
        z_components["drives"] = _zscore_within_season(result, "DRIVES_PER36")

    if "AT_RIM_FREQ" in result.columns:
        z_components["rim_fga"] = _zscore_within_season(result, "AT_RIM_FREQ")

    if "FT_RATE" in result.columns:
        z_components["fouls_drawn"] = _zscore_within_season(result, "FT_RATE")

    if "PAINT_FREQ" in result.columns:
        z_components["paint"] = _zscore_within_season(result, "PAINT_FREQ")

    if z_components:
        result["dim_driving_gravity_z"] = _avg_z_components(z_components)
    else:
        result["dim_driving_gravity_z"] = 0.0

    # ===== DIMENSION 3: PLAYMAKING =====
    # Adjusted AST%, potential assists, playmaking score, secondary assists
    z_components = {}
    ast_col = "AST_PER36" if "AST_PER36" in result.columns else (
        "AST_per36" if "AST_per36" in result.columns else None
    )
    if ast_col:
        z_components["ast"] = _zscore_within_season(result, ast_col)

    if "PLAYMAKING_SCORE" in result.columns:
        z_components["playmaking"] = _zscore_within_season(result, "PLAYMAKING_SCORE")

    if "POTENTIAL_AST_PER36" in result.columns:
        z_components["pot_ast"] = _zscore_within_season(result, "POTENTIAL_AST_PER36")

    if "SECONDARY_AST_PER36" in result.columns:
        z_components["sec_ast"] = _zscore_within_season(result, "SECONDARY_AST_PER36")

    if z_components:
        result["dim_playmaking_z"] = _avg_z_components(z_components)
    else:
        result["dim_playmaking_z"] = 0.0

    # ===== DIMENSION 4: EXTRA POSSESSION CREATION (cross-domain) =====
    # OREB%, DREB%, REB/36 — stored raw, then split 40/60 into O/D composites
    z_components = {}
    if "OREB_pct" in result.columns:
        z_components["oreb"] = _zscore_within_season(result, "OREB_pct")
    if "DREB_pct" in result.columns:
        z_components["dreb"] = _zscore_within_season(result, "DREB_pct")

    reb_col = "REB_PER36" if "REB_PER36" in result.columns else (
        "REB_per36" if "REB_per36" in result.columns else None
    )
    if reb_col:
        z_components["reb_rate"] = _zscore_within_season(result, reb_col)

    if z_components:
        result["dim_extra_possession_z"] = _avg_z_components(z_components)
    else:
        result["dim_extra_possession_z"] = 0.0

    # ===== DIMENSION 5: DEFENSIVE PLAYMAKING (expanded) =====
    # STL%, BLK%, Deflections, hustle_score, engagement_score
    # Apply Bayesian shrinkage to reduce noise
    z_components = {}
    shrinkage = BAYESIAN_SHRINKAGE.defensive_playmaking_shrinkage
    gp_series = result.get("GP") if "GP" in result.columns else None

    for col in ["STL_PER100_DEF_POSS", "BLK_PCT", "DEFLECTIONS",
                "hustle_score", "engagement_score"]:
        if col in result.columns:
            # Shrink toward league mean per season
            shrunk = result.groupby("season")[col].transform(
                lambda x: apply_bayesian_shrinkage(
                    x, prior=x.mean(), shrinkage_strength=shrinkage,
                    gp=gp_series.loc[x.index] if gp_series is not None else None,
                    min_gp_full=BAYESIAN_SHRINKAGE.min_gp_full_weight,
                )
            )
            z_components[col] = _zscore_within_season(
                result.assign(**{f"_shrunk_{col}": shrunk}), f"_shrunk_{col}"
            )

    if z_components:
        result["dim_defensive_playmaking_z"] = _avg_z_components(z_components)
    else:
        result["dim_defensive_playmaking_z"] = 0.0

    # ===== DIMENSION 6: DEFENSIVE IMPACT (RAPM-informed) =====
    # DRAPM, DRTG (inverted), d_results_pctl — NO rim protection (avoid redundancy)
    z_components = {}
    if "drapm" in result.columns:
        z_components["drapm"] = _zscore_within_season(result, "drapm")

    if "DRTG" in result.columns:
        z_components["drtg"] = _zscore_within_season(result, "DRTG", invert=True)

    if "d_results_pctl" in result.columns:
        z_components["d_results"] = _zscore_within_season(result, "d_results_pctl")

    if z_components:
        result["dim_defensive_impact_z"] = _avg_z_components(z_components)
    else:
        result["dim_defensive_impact_z"] = 0.0

    # ===== DIMENSION 7: TURNOVER CONTROL (offensive, v2.0 restored) =====
    # TOV%, TOV/36 — INVERTED (lower is better)
    z_components = {}
    tov_col = "TOV_pct" if "TOV_pct" in result.columns else (
        "TOV_PCT" if "TOV_PCT" in result.columns else None
    )
    if tov_col:
        z_components["tov_pct"] = _zscore_within_season(result, tov_col, invert=True)

    tov_per36 = "TOV_PER36" if "TOV_PER36" in result.columns else (
        "TOV_per36" if "TOV_per36" in result.columns else None
    )
    if tov_per36:
        z_components["tov_rate"] = _zscore_within_season(result, tov_per36, invert=True)

    if z_components:
        result["dim_turnover_control_z"] = _avg_z_components(z_components)
    else:
        result["dim_turnover_control_z"] = 0.0

    # ===== DIMENSION 8: DEFENSIVE VERSATILITY =====
    # Switch score, versatility, assignment difficulty, matchup diversity
    z_components = {}

    # assignment_difficulty is categorical ("Low"/"Medium"/"High"/"Unknown") — encode numerically
    if "assignment_difficulty" in result.columns:
        ad_map = {"Unknown": np.nan, "Low": 0.0, "Medium": 0.5, "High": 1.0}
        result["_assignment_difficulty_num"] = result["assignment_difficulty"].map(ad_map)

    for col in ["switch_score", "versatility_pctl", "_assignment_difficulty_num",
                "matchup_diversity_pctl"]:
        if col in result.columns:
            z_components[col] = _zscore_within_season(result, col)

    if z_components:
        result["dim_defensive_versatility_z"] = _avg_z_components(z_components)
    else:
        result["dim_defensive_versatility_z"] = 0.0

    # ===== CROSS-DOMAIN SPLIT: Extra Possession Creation =====
    # 40% to offensive composite, 60% to defensive composite
    result["dim_extra_poss_offensive_z"] = cfg.extra_poss_offensive_share * result["dim_extra_possession_z"]
    result["dim_extra_poss_defensive_z"] = cfg.extra_poss_defensive_share * result["dim_extra_possession_z"]

    # ===== DIMENSION COMPOSITE (1C) =====
    # Equal-weighted z-score composite across all 8 dimensions
    dim_z_cols = [
        "dim_shooting_gravity_z",
        "dim_driving_gravity_z",
        "dim_playmaking_z",
        "dim_extra_possession_z",
        "dim_turnover_control_z",
        "dim_defensive_playmaking_z",
        "dim_defensive_impact_z",
        "dim_defensive_versatility_z",
    ]
    dim_weights = [
        cfg.w_dim_shooting_gravity,
        cfg.w_dim_driving_gravity,
        cfg.w_dim_playmaking,
        cfg.w_dim_extra_possession,
        cfg.w_dim_turnover_control,
        cfg.w_dim_defensive_playmaking,
        cfg.w_dim_defensive_impact,
        cfg.w_dim_defensive_versatility,
    ]

    result = weighted_z_composite(result, dim_z_cols, dim_weights, "dimension_model_z")

    # ===== OFFENSIVE / DEFENSIVE SUB-COMPOSITES =====
    off_cols = ["dim_shooting_gravity_z", "dim_driving_gravity_z",
                "dim_playmaking_z", "dim_extra_poss_offensive_z",
                "dim_turnover_control_z"]
    off_weights = [cfg.w_dim_shooting_gravity, cfg.w_dim_driving_gravity,
                   cfg.w_dim_playmaking,
                   cfg.w_dim_extra_possession * cfg.extra_poss_offensive_share,
                   cfg.w_dim_turnover_control]
    result = weighted_z_composite(result, off_cols, off_weights, "offensive_portable_z")

    def_cols = ["dim_defensive_playmaking_z", "dim_defensive_impact_z",
                "dim_defensive_versatility_z", "dim_extra_poss_defensive_z"]
    def_weights = [cfg.w_dim_defensive_playmaking, cfg.w_dim_defensive_impact,
                   cfg.w_dim_defensive_versatility,
                   cfg.w_dim_extra_possession * cfg.extra_poss_defensive_share]
    result = weighted_z_composite(result, def_cols, def_weights, "defensive_portable_z")

    # Clean up internal temp columns
    temp_cols = [c for c in result.columns if c.startswith("_shrunk_")]
    if temp_cols:
        result = result.drop(columns=temp_cols)

    return result


# ---------------------------------------------------------------------------
# 1D. Portable Talent Score (v2.0: z-score aggregation)
# ---------------------------------------------------------------------------
    # --- Turnover Control ---
    # Kept as v1.5 backward-compat scalar (0-1)
    tov_col = "TOV_pct" if "TOV_pct" in result.columns else ("TOV_PCT" if "TOV_PCT" in result.columns else None)
    if tov_col:
        result["turnover_control"] = result.groupby("season")[tov_col].transform(
            lambda x: 1.0 - (x - x.min()) / (x.max() - x.min() + 1e-9)
        )
    else:
        result["turnover_control"] = 0.5

    # --- Rebounding (v1.5 compat scalar) ---
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
# 1D. Portable Talent Score (v2.0: z-score aggregation)
# ---------------------------------------------------------------------------

def compute_portable_talent_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Portable Talent Score (PTS) using v2.0 z-score aggregation:

      PTS_z = 25% RAPM_z + 20% Playtype_z + 55% Dimension_z

    Then: PTS_z → Percentile (for presentation only).

    v2.0 key change: z-scores preserve interval meaning during aggregation,
    preventing the percentile-averaging distortion identified in v1.5.
    """
    result = df.copy()
    cfg = PORTABLE_TALENT

    # --- 1A. RAPM Backbone z-score ---
    if "rapm" in result.columns:
        result["rapm_z"] = result.groupby("season")["rapm"].transform(
            lambda x: compute_z_score(x, winsorize=3.5)
        )
    else:
        result["rapm_z"] = 0.0

    # --- 1B. Playtype Efficiency z-score ---
    # Use playtype surplus total if available (computed in Layer 2), else TS adj
    ts_col = "TS_PCT_adj" if "TS_PCT_adj" in result.columns else "TS_PCT"
    if ts_col in result.columns:
        result["playtype_efficiency_z"] = result.groupby("season")[ts_col].transform(
            lambda x: compute_z_score(x, winsorize=3.5)
        )
    else:
        result["playtype_efficiency_z"] = 0.0

    # --- 1C. Dimension Model z-score (already computed) ---
    if "dimension_model_z" not in result.columns:
        result["dimension_model_z"] = 0.0

    # --- COMBINED PTS z-score ---
    # PTS_z = 25% RAPM + 20% Playtype + 55% Dimension
    result["portable_talent_z"] = (
        cfg.w_rapm_backbone * result["rapm_z"] +
        cfg.w_playtype_efficiency * result["playtype_efficiency_z"] +
        cfg.w_dimension_model * result["dimension_model_z"]
    )

    # Stability weight: players with more possessions get slightly more trust
    if "possessions_played" in result.columns:
        stability = np.clip(
            result["possessions_played"] / cfg.min_poss_full_weight,
            0.0, 1.0
        )
    else:
        stability = 1.0

    result["stability_weight"] = stability if isinstance(stability, pd.Series) else stability

    # Apply light stability adjustment to z-score
    result["portable_talent_z_adj"] = result["portable_talent_z"] * (
        0.85 + 0.15 * result["stability_weight"]
    )

    # --- Convert to percentile (presentation only) ---
    # Use within-season percentile rank for comparability
    result["portable_talent_raw"] = result["portable_talent_z_adj"]
    result["portable_talent_score"] = result.groupby("season")["portable_talent_z_adj"].transform(
        vectorized_percentile_rank
    )

    # Also store the z-score based percentile for cross-era comparison
    result["portable_talent_cdf_pctl"] = z_to_percentile(result["portable_talent_z_adj"])

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

    # 6. Compute 8-dimension portable model (v2.0 Layer 1C)
    print("  Computing 8-dimension portable model (z-score aggregation)...")
    qualified = compute_dimension_model(qualified)

    # 7. Add league z-scores for key metrics
    print("  Computing z-score standardization...")
    z_metrics = ["rapm", "orapm", "drapm",
                 "dim_shooting_gravity_z", "dim_driving_gravity_z",
                 "dim_playmaking_z", "dim_extra_possession_z",
                 "dim_turnover_control_z", "dim_defensive_playmaking_z",
                 "dim_defensive_impact_z", "dim_defensive_versatility_z",
                 "dimension_model_z", "offensive_portable_z", "defensive_portable_z"]
    qualified = add_league_z_scores(qualified, ["rapm", "orapm", "drapm"])

    # 8. Also add league percentiles for output/presentation
    pctl_metrics = ["rapm", "orapm", "drapm"]
    qualified = add_league_percentiles(qualified, pctl_metrics)

    # 9. Add positional percentiles
    qualified = add_grouped_percentiles(
        qualified, pctl_metrics, group_col="position_bucket",
        suffix="_pctl",
    )

    # 10. Add archetype percentiles
    if "primary_archetype" in qualified.columns:
        qualified = add_grouped_percentiles(
            qualified, pctl_metrics, group_col="primary_archetype",
            suffix="_pctl",
        )

    # 11. Compute Portable Talent Score (v2.0: z-score aggregation)
    print("  Computing Portable Talent Score (25% RAPM + 20% Playtype + 55% Dimensions)...")
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
