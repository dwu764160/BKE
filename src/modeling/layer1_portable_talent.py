"""
src/modeling/layer1_portable_talent.py
=============================================================================
BKE v2.6 — LAYER 1: True Portable Talent (Context-Neutral Layer)

Estimates:
  "How good is this player independent of role volume and scheme?"

v2.6 structure:
  1A. RAPM Backbone (25%) — Multi-source impact z-score
  1B. Playtype Efficiency (20%) — Usage-adjusted playtype composite
  1C. Portable Dimension Model (55%) — 9 basketball-informed dimensions:
      UNIVERSALLY PORTABLE (league z + neutralization):
        Shooting Gravity, Playmaking Creation, Self-Creation, Turnover Control
      POSITION-CONDITIONAL (position z):
        Driving Gravity, Extra Possession, Def Playmaking, Def Impact, Def Versatility

v2.6 fixes over v2.5:
  - Position-conditional z-scores for biased dimensions (Fix #4)
    Dimensions dominated by position (rebounding, rim protection, blocks)
    now compute z-scores within position group, not league-wide.
  - Self-Creation dimension added (Dim 9): pull-up shooting, off-dribble
    creation, ball dominance — the key differentiator for portable vs. system players
  - Basketball-informed weights: shooting/playmaking/creation at 14% each,
    rebounding/def-playmaking at 8% each
  - Big-man bias eliminated: Centers no longer dominate top 10

Inputs:
  - player_rapm.parquet (RAPM / ORAPM / DRAPM)
  - modeling_inputs_all.parquet (DARKO, linear stats)
  - player_profiles_advanced.parquet (box score + four factors)
  - player_archetypes.parquet (offensive archetype + playtypes)
  - defensive_archetypes_v2.parquet (defensive metrics)
  - player_position_estimates.parquet (positional percentages)
  - tracking/{season}/tracking_PullUpShot.parquet (pull-up shot data)

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
    TRACKING_DIR,
    RAPM_PREFERRED_TYPE,
    RAPM_FALLBACK_TYPE,
    MIN_MINUTES,
    MIN_GP,
    MIN_MPG,
    MIN_POSSESSIONS,
    BAYESIAN_SHRINKAGE,
    NEUTRALIZATION,
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
    neutralize_dimensions,
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
            # v2.0: Additional columns for dimension model
            "hustle_score", "engagement_score", "DEFLECTIONS",
            "matchup_diversity_pctl", "d_results_pctl",
            "defensive_fit", "size_band"]
    available = [c for c in keep if c in df.columns]
    return df[available].drop_duplicates(subset=["season", "player_id"], keep="last")


def load_hustle_stats(seasons: Optional[list] = None) -> pd.DataFrame:
    """
    Load raw hustle stats from tracking data for MF-4 (v2.5).

    Provides: CHARGES_DRAWN, DEF_LOOSE_BALLS_RECOVERED, LOOSE_BALLS_RECOVERED
    """
    parts = []
    for season in (seasons or SEASONS):
        path = os.path.join(TRACKING_DIR, season, "hustle_stats.parquet")
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found")
            continue
        df = pd.read_parquet(path)
        id_col = "PLAYER_ID" if "PLAYER_ID" in df.columns else "player_id"
        df["player_id"] = df[id_col].astype(str).apply(clean_id)
        season_col = "SEASON" if "SEASON" in df.columns else "season"
        df["season"] = df[season_col].astype(str)

        keep = ["season", "player_id",
                "CHARGES_DRAWN", "DEF_LOOSE_BALLS_RECOVERED",
                "LOOSE_BALLS_RECOVERED", "SCREEN_ASSISTS"]
        available = [c for c in keep if c in df.columns]
        parts.append(df[available])

    if not parts:
        print("  WARNING: No hustle stats found")
        return pd.DataFrame()

    combined = pd.concat(parts, ignore_index=True)
    return combined.drop_duplicates(subset=["season", "player_id"], keep="last")


def load_pullup_tracking(seasons: Optional[list] = None) -> pd.DataFrame:
    """
    Load pull-up shot tracking data for v2.6 Self-Creation dimension (Dim 9).

    Provides: PULL_UP_FGA, PULL_UP_FG_PCT, PULL_UP_EFG_PCT, PULL_UP_FG3A, etc.
    """
    parts = []
    for season in (seasons or SEASONS):
        path = os.path.join(TRACKING_DIR, season, "tracking_PullUpShot.parquet")
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found")
            continue
        df = pd.read_parquet(path)
        id_col = "PLAYER_ID" if "PLAYER_ID" in df.columns else "player_id"
        df["player_id"] = df[id_col].astype(str).apply(clean_id)
        df["season"] = season

        keep = ["season", "player_id",
                "PULL_UP_FGM", "PULL_UP_FGA", "PULL_UP_FG_PCT",
                "PULL_UP_PTS", "PULL_UP_FG3M", "PULL_UP_FG3A",
                "PULL_UP_FG3_PCT", "PULL_UP_EFG_PCT", "GP", "MIN"]
        available = [c for c in keep if c in df.columns]
        parts.append(df[available])

    if not parts:
        print("  WARNING: No pull-up tracking data found")
        return pd.DataFrame()

    combined = pd.concat(parts, ignore_index=True)
    return combined.drop_duplicates(subset=["season", "player_id"], keep="last")


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
# 1C. Portable Dimension Model — 8 Independent Dimensions (v2.5 Neutralized)
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


def _zscore_within_position_season(df: pd.DataFrame, col: str, invert: bool = False) -> pd.Series:
    """
    Compute z-score of a column within each (season, position_bucket) group.

    v2.6: Position-conditional z-scoring for dimensions that heavily correlate
    with position (rebounding, blocks, rim protection, driving, versatility).

    This ensures a center's rebounding is compared to other centers, not guards.
    A center at +0.5σ among centers reflects genuine above-position talent;
    a center at +2.0σ league-wide just reflects that centers rebound more.

    Falls back to season-level z-score if position_bucket not available
    or cohort is too small (< 15 players).
    """
    min_cohort = 15  # need at least 15 players in position group

    if "position_bucket" not in df.columns:
        return _zscore_within_season(df, col, invert=invert)

    z = pd.Series(0.0, index=df.index)

    for (season, pos), group in df.groupby(["season", "position_bucket"]):
        mask = group.index
        vals = df.loc[mask, col]

        if len(vals.dropna()) < min_cohort:
            # Fall back to season-level z-score for small cohorts
            season_mask = df["season"] == season
            season_vals = df.loc[season_mask, col]
            season_z = compute_z_score(season_vals, winsorize=3.5)
            z.loc[mask] = season_z.loc[mask]
        else:
            pos_z = compute_z_score(vals, winsorize=3.5)
            z.loc[mask] = pos_z

    if invert:
        z = -z
    return z


def compute_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived metrics needed for v2.6 dimensions.

    v2.6 new derived columns:
      - TOV_PER_TOUCH: turnovers per touch (MF-1)
      - DRIVE_TOV_RATE: drive turnovers per drive (MF-1)
      - DRIVE_AST_RATIO: drive assists per drive (MF-5 pressure proxy)
      - PASSES_MADE_PER36: passing volume per 36 min
      - PULL_UP_FGA_PER36: pull-up shot volume per 36 min (Dim 9)
    """
    result = df.copy()

    # MF-1: TOV per touch
    if "TOV" in result.columns and "TOUCHES" in result.columns:
        touches = result["TOUCHES"].replace(0, np.nan)
        result["TOV_PER_TOUCH"] = result["TOV"] / touches
    else:
        result["TOV_PER_TOUCH"] = np.nan

    # MF-1: Drive TOV rate (drive turnovers per drive)
    if "DRIVE_TOV" in result.columns and "DRIVES" in result.columns:
        drives = result["DRIVES"].replace(0, np.nan)
        result["DRIVE_TOV_RATE"] = result["DRIVE_TOV"] / drives
    else:
        result["DRIVE_TOV_RATE"] = np.nan

    # MF-5: Drive-and-kick frequency proxy (drive assists per drive)
    if "DRIVE_AST" in result.columns and "DRIVES" in result.columns:
        drives = result["DRIVES"].replace(0, np.nan)
        result["DRIVE_AST_RATIO"] = result["DRIVE_AST"] / drives
    else:
        result["DRIVE_AST_RATIO"] = np.nan

    # Passes made per 36
    if "PASSES_MADE" in result.columns and "MIN" in result.columns:
        minutes = result["MIN"].replace(0, np.nan)
        result["PASSES_MADE_PER36"] = result["PASSES_MADE"] / minutes * 36.0 * result.get("GP", 1)
        # Correct: PASSES_MADE is already per-game in the tracking data
        # Just normalize to per-36
        if "MPG" in result.columns:
            mpg = result["MPG"].replace(0, np.nan)
            result["PASSES_MADE_PER36"] = result["PASSES_MADE"] / mpg * 36.0
    else:
        result["PASSES_MADE_PER36"] = np.nan

    # v2.6: Pull-up FGA per 36 min (Self-Creation dimension)
    if "PULL_UP_FGA" in result.columns:
        if "MPG" in result.columns:
            mpg = result["MPG"].replace(0, np.nan)
            result["PULL_UP_FGA_PER36"] = result["PULL_UP_FGA"] / mpg * 36.0
        elif "MIN" in result.columns and "GP" in result.columns:
            mpg = (result["MIN"] / result["GP"].replace(0, np.nan)).replace(0, np.nan)
            result["PULL_UP_FGA_PER36"] = result["PULL_UP_FGA"] / mpg * 36.0
        else:
            result["PULL_UP_FGA_PER36"] = np.nan
    else:
        result["PULL_UP_FGA_PER36"] = np.nan

    return result


def compute_dimension_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the 9-dimension portable talent model (Layer 1C).

    v2.6 uses dual z-score mode + archetype-conditional neutralization:
      1. Per sub-metric: compute z-score (league OR position-conditional)
      2. Within dimension: average of z-scores
      3. NEUTRALIZE: subtract archetype expected z-score (league-z dims only)
      4. Cross-dimension: basketball-weighted z-score composite

    v2.6 key changes:
      - Position-conditional z-scores for position-dominated dimensions
        (driving, rebounding, blocks, DRAPM, versatility)
      - Self-Creation dimension added (pull-up shooting, ball dominance)
      - Basketball-informed weighting (shooting/playmaking/creation at 14%)
      - Big-man bias eliminated

    Returns DataFrame with dimension z-scores and composite.
    """
    result = df.copy()
    cfg = PORTABLE_TALENT

    # ===== DIMENSION 1: SHOOTING GRAVITY (league z — universally portable) =====
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

    # ===== DIMENSION 2: DRIVING GRAVITY (position z — rim finishing is positional) =====
    z_components = {}
    if "DRIVES_PER36" in result.columns:
        z_components["drives"] = _zscore_within_position_season(result, "DRIVES_PER36")

    if "AT_RIM_FREQ" in result.columns:
        z_components["rim_fga"] = _zscore_within_position_season(result, "AT_RIM_FREQ")

    if "FT_RATE" in result.columns:
        z_components["fouls_drawn"] = _zscore_within_position_season(result, "FT_RATE")

    if "PAINT_FREQ" in result.columns:
        z_components["paint"] = _zscore_within_position_season(result, "PAINT_FREQ")

    if z_components:
        result["dim_driving_gravity_z"] = _avg_z_components(z_components)
    else:
        result["dim_driving_gravity_z"] = 0.0

    # ===== DIMENSION 3: PLAYMAKING CREATION (league z — universally portable) =====
    creation_z = {}
    ast_col = "AST_PER36" if "AST_PER36" in result.columns else (
        "AST_per36" if "AST_per36" in result.columns else None
    )
    if ast_col:
        creation_z["ast"] = _zscore_within_season(result, ast_col)

    if "PLAYMAKING_SCORE" in result.columns:
        creation_z["playmaking"] = _zscore_within_season(result, "PLAYMAKING_SCORE")

    # Pressure sub-component: collapse + rotation forcing
    pressure_z = {}
    if "POTENTIAL_AST_PER36" in result.columns:
        pressure_z["pot_ast"] = _zscore_within_season(result, "POTENTIAL_AST_PER36")

    if "SECONDARY_AST_PER36" in result.columns:
        pressure_z["sec_ast"] = _zscore_within_season(result, "SECONDARY_AST_PER36")

    if "DRIVE_AST_RATIO" in result.columns:
        pressure_z["drv_kick"] = _zscore_within_season(result, "DRIVE_AST_RATIO")

    if "PASSES_MADE_PER36" in result.columns:
        pressure_z["passes"] = _zscore_within_season(result, "PASSES_MADE_PER36")

    # Combine creation + pressure (50/50 blend)
    creation_score = _avg_z_components(creation_z) if creation_z else pd.Series(0.0, index=result.index)
    pressure_score = _avg_z_components(pressure_z) if pressure_z else pd.Series(0.0, index=result.index)

    result["dim_playmaking_creation_sub_z"] = creation_score
    result["dim_playmaking_pressure_sub_z"] = pressure_score
    result["dim_playmaking_creation_z"] = 0.5 * creation_score + 0.5 * pressure_score

    # ===== DIMENSION 4: EXTRA POSSESSION (position z — rebounding is heavily positional) =====
    z_components = {}
    if "OREB_pct" in result.columns:
        z_components["oreb"] = _zscore_within_position_season(result, "OREB_pct")
    if "DREB_pct" in result.columns:
        z_components["dreb"] = _zscore_within_position_season(result, "DREB_pct")

    reb_col = "REB_PER36" if "REB_PER36" in result.columns else (
        "REB_per36" if "REB_per36" in result.columns else None
    )
    if reb_col:
        z_components["reb_rate"] = _zscore_within_position_season(result, reb_col)

    if z_components:
        result["dim_extra_possession_z"] = _avg_z_components(z_components)
    else:
        result["dim_extra_possession_z"] = 0.0

    # ===== DIMENSION 5: DEFENSIVE PLAYMAKING (position z — BLK% dominated by centers) =====
    z_components = {}
    shrinkage = BAYESIAN_SHRINKAGE.defensive_playmaking_shrinkage
    gp_series = result.get("GP") if "GP" in result.columns else None

    for col in ["STL_PER100_DEF_POSS", "BLK_PCT", "DEFLECTIONS",
                "hustle_score", "engagement_score",
                "CHARGES_DRAWN", "DEF_LOOSE_BALLS_RECOVERED"]:
        if col in result.columns:
            # Shrink toward league mean per season
            shrunk = result.groupby("season")[col].transform(
                lambda x: apply_bayesian_shrinkage(
                    x, prior=x.mean(), shrinkage_strength=shrinkage,
                    gp=gp_series.loc[x.index] if gp_series is not None else None,
                    min_gp_full=BAYESIAN_SHRINKAGE.min_gp_full_weight,
                )
            )
            # v2.6: Use position-conditional z-scores for this dimension
            z_components[col] = _zscore_within_position_season(
                result.assign(**{f"_shrunk_{col}": shrunk}), f"_shrunk_{col}"
            )

    if z_components:
        result["dim_defensive_playmaking_z"] = _avg_z_components(z_components)
    else:
        result["dim_defensive_playmaking_z"] = 0.0

    # ===== DIMENSION 6: DEFENSIVE IMPACT (position z — DRAPM favors rim protectors) =====
    z_components = {}
    if "drapm" in result.columns:
        z_components["drapm"] = _zscore_within_position_season(result, "drapm")

    if "DRTG" in result.columns:
        z_components["drtg"] = _zscore_within_position_season(result, "DRTG", invert=True)

    if "d_results_pctl" in result.columns:
        z_components["d_results"] = _zscore_within_position_season(result, "d_results_pctl")

    if z_components:
        result["dim_defensive_impact_z"] = _avg_z_components(z_components)
    else:
        result["dim_defensive_impact_z"] = 0.0

    # ===== DIMENSION 7: TURNOVER CONTROL (league z — usage-adjusted metrics) =====
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

    # MF-1: TOV per touch — THIS is the key usage-adjusted metric
    # Low-usage bigs will NOT get free credit because it's per-touch, not per-36
    if "TOV_PER_TOUCH" in result.columns:
        z_components["tov_per_touch"] = _zscore_within_season(result, "TOV_PER_TOUCH", invert=True)

    if "DRIVE_TOV_RATE" in result.columns:
        z_components["drive_tov"] = _zscore_within_season(result, "DRIVE_TOV_RATE", invert=True)

    if z_components:
        result["dim_turnover_control_z"] = _avg_z_components(z_components)
    else:
        result["dim_turnover_control_z"] = 0.0

    # ===== DIMENSION 8: DEFENSIVE VERSATILITY (position z — fix Centers +0.86 bias) =====
    z_components = {}

    # assignment_difficulty is categorical — encode numerically
    if "assignment_difficulty" in result.columns:
        ad_map = {"Unknown": np.nan, "Low": 0.0, "Medium": 0.5, "High": 1.0}
        result["_assignment_difficulty_num"] = result["assignment_difficulty"].map(ad_map)

    for col in ["switch_score", "versatility_pctl", "_assignment_difficulty_num",
                "matchup_diversity_pctl"]:
        if col in result.columns:
            # v2.6: Position-conditional z-scores to fix the center bias
            z_components[col] = _zscore_within_position_season(result, col)

    if z_components:
        result["dim_defensive_versatility_z"] = _avg_z_components(z_components)
    else:
        result["dim_defensive_versatility_z"] = 0.0

    # ===== DIMENSION 9: SELF-CREATION (league z — NEW v2.6) =====
    # Off-dribble shot creation, pull-up shooting, ball dominance
    # This dimension naturally favors guards/wings who create their own offense
    # and is the KEY counterbalance to position-dependent dimensions
    z_components = {}

    if "PULL_UP_FGA_PER36" in result.columns:
        z_components["pullup_vol"] = _zscore_within_season(result, "PULL_UP_FGA_PER36")

    if "PULL_UP_EFG_PCT" in result.columns:
        z_components["pullup_eff"] = _zscore_within_season(result, "PULL_UP_EFG_PCT")

    if "AVG_DRIB_PER_TOUCH" in result.columns:
        z_components["drib_per_touch"] = _zscore_within_season(result, "AVG_DRIB_PER_TOUCH")

    if "ON_BALL_CREATION" in result.columns:
        z_components["on_ball_creation"] = _zscore_within_season(result, "ON_BALL_CREATION")

    if "BALL_DOMINANT_PCT" in result.columns:
        z_components["ball_dominance"] = _zscore_within_season(result, "BALL_DOMINANT_PCT")

    if z_components:
        result["dim_self_creation_z"] = _avg_z_components(z_components)
    else:
        result["dim_self_creation_z"] = 0.0

    # ===== v2.6 ARCHETYPE-CONDITIONAL NEUTRALIZATION =====
    # Only neutralize LEAGUE-z dimensions. Position-z dimensions already
    # account for positional expectations via position-conditional z-scoring.
    neutralize_cols = NEUTRALIZATION.neutralize_dimensions
    # Filter to only cols that exist
    neutralize_cols = [c for c in neutralize_cols if c in result.columns]
    if neutralize_cols:
        result = neutralize_dimensions(
            result, neutralize_cols,
            archetype_col="primary_archetype",
            season_col="season",
        )

    # ===== CROSS-DOMAIN SPLIT: Extra Possession Creation =====
    result["dim_extra_poss_offensive_z"] = cfg.extra_poss_offensive_share * result["dim_extra_possession_z"]
    result["dim_extra_poss_defensive_z"] = cfg.extra_poss_defensive_share * result["dim_extra_possession_z"]

    # ===== DIMENSION COMPOSITE (1C) — v2.6: 9 dimensions, basketball-weighted =====
    dim_z_cols = [
        "dim_shooting_gravity_z",
        "dim_driving_gravity_z",
        "dim_playmaking_creation_z",
        "dim_extra_possession_z",
        "dim_turnover_control_z",
        "dim_defensive_playmaking_z",
        "dim_defensive_impact_z",
        "dim_defensive_versatility_z",
        "dim_self_creation_z",
    ]
    dim_weights = [
        cfg.w_dim_shooting_gravity,       # 0.14
        cfg.w_dim_driving_gravity,        # 0.10
        cfg.w_dim_playmaking,             # 0.14
        cfg.w_dim_extra_possession,       # 0.08
        cfg.w_dim_turnover_control,       # 0.10
        cfg.w_dim_defensive_playmaking,   # 0.08
        cfg.w_dim_defensive_impact,       # 0.10
        cfg.w_dim_defensive_versatility,  # 0.12
        cfg.w_dim_self_creation,          # 0.14
    ]

    result = weighted_z_composite(result, dim_z_cols, dim_weights, "dimension_model_z")

    # ===== OFFENSIVE / DEFENSIVE SUB-COMPOSITES =====
    off_cols = ["dim_shooting_gravity_z", "dim_driving_gravity_z",
                "dim_playmaking_creation_z", "dim_extra_poss_offensive_z",
                "dim_turnover_control_z", "dim_self_creation_z"]
    off_weights = [cfg.w_dim_shooting_gravity, cfg.w_dim_driving_gravity,
                   cfg.w_dim_playmaking,
                   cfg.w_dim_extra_possession * cfg.extra_poss_offensive_share,
                   cfg.w_dim_turnover_control,
                   cfg.w_dim_self_creation]
    result = weighted_z_composite(result, off_cols, off_weights, "offensive_portable_z")

    def_cols = ["dim_defensive_playmaking_z", "dim_defensive_impact_z",
                "dim_defensive_versatility_z", "dim_extra_poss_defensive_z"]
    def_weights = [cfg.w_dim_defensive_playmaking, cfg.w_dim_defensive_impact,
                   cfg.w_dim_defensive_versatility,
                   cfg.w_dim_extra_possession * cfg.extra_poss_defensive_share]
    result = weighted_z_composite(result, def_cols, def_weights, "defensive_portable_z")

    # Clean up internal temp columns
    temp_cols = [c for c in result.columns if c.startswith("_shrunk_") or c == "_assignment_difficulty_num"]
    if temp_cols:
        result = result.drop(columns=temp_cols)

    return result


# ---------------------------------------------------------------------------
# 1D. Portable Talent Score (v2.6: z-score aggregation)
# ---------------------------------------------------------------------------

def compute_portable_talent_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Portable Talent Score (PTS) using v2.5 z-score aggregation:

      PTS_z = 25% RAPM_z + 20% Playtype_z + 55% Dimension_z

    Then: PTS_z → Percentile (for presentation only).

    v2.5: z-scores preserve interval meaning; archetype-conditional
    neutralization is applied within dimension model before this stage.
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
    Full Layer 1 pipeline (v2.6): load data → merge → derive → neutralize → score.

    v2.6 changes:
      - Loads pull-up tracking data (PULL_UP_FGA, PULL_UP_EFG_PCT, etc.)
      - 9-dimension model with position-conditional z-scores
      - Self-creation dimension (pull-up shooting, ball dominance)
      - Basketball-informed dimension weighting
      - Fixes big-man bias in defensive versatility + rebounding

    Returns DataFrame with portable talent decomposition for all qualified
    player-seasons.
    """
    seasons = seasons or SEASONS
    print("\n" + "=" * 60)
    print("LAYER 1: PORTABLE TALENT (v2.6)")
    print("=" * 60)

    # 1. Load all data sources
    print("  Loading data sources...")
    rapm = load_rapm()
    modeling = load_modeling_inputs()
    profiles = load_player_profiles()
    archetypes = load_archetypes()
    def_archetypes = load_defensive_archetypes()
    positions = load_position_estimates()
    hustle = load_hustle_stats(seasons)
    pullup = load_pullup_tracking(seasons)

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

    # v2.5: Merge hustle stats (CHARGES_DRAWN, DEF_LOOSE_BALLS_RECOVERED, etc.)
    if not hustle.empty:
        base = base.merge(hustle, on=["season", "player_id"], how="left", suffixes=("", "_hst"))
        for col in list(base.columns):
            if col.endswith("_hst"):
                primary = col.replace("_hst", "")
                if primary in base.columns:
                    base[primary] = base[primary].fillna(base[col])
                else:
                    base[primary] = base[col]
                base = base.drop(columns=[col])
        print(f"  Hustle stats merged: {hustle.shape[0]} rows, cols: {list(hustle.columns)}")

    # v2.6: Merge pull-up tracking data (PULL_UP_FGA, PULL_UP_EFG_PCT, etc.)
    if not pullup.empty:
        base = base.merge(pullup, on=["season", "player_id"], how="left", suffixes=("", "_pu"))
        for col in list(base.columns):
            if col.endswith("_pu"):
                primary = col.replace("_pu", "")
                if primary in base.columns:
                    base[primary] = base[primary].fillna(base[col])
                else:
                    base[primary] = base[col]
                base = base.drop(columns=[col])
        print(f"  Pull-up tracking merged: {pullup.shape[0]} rows, cols: {list(pullup.columns)}")

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

    # v2.6: Compute derived metrics (TOV_PER_TOUCH, DRIVE_TOV_RATE, PULL_UP_FGA_PER36, etc.)
    print("  Computing v2.6 derived metrics...")
    qualified = compute_derived_metrics(qualified)

    # 6. Compute 9-dimension portable model (v2.6 Layer 1C with position-z + neutralization)
    print("  Computing 9-dimension portable model (v2.6 position-z + neutralization)...")
    qualified = compute_dimension_model(qualified)

    # 6.5. Apply hierarchical Bayesian shrinkage to dimension z-scores (v2.6)
    print("  Applying hierarchical Bayesian shrinkage (v2.6)...")
    from src.modeling.bayesian_hierarchical import apply_and_replace
    qualified = apply_and_replace(qualified)

    # 7. Add league z-scores for key metrics
    print("  Computing z-score standardization...")
    qualified = add_league_z_scores(qualified, ["rapm", "orapm", "drapm"])

    # 8. Also add league percentiles for output/presentation (TERMINAL only)
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

    # v2.6: Three-level z-scores for dimension columns
    dim_z_cols = [
        "dim_shooting_gravity_z", "dim_driving_gravity_z",
        "dim_playmaking_creation_z", "dim_extra_possession_z",
        "dim_turnover_control_z", "dim_defensive_playmaking_z",
        "dim_defensive_impact_z", "dim_defensive_versatility_z",
        "dim_self_creation_z",
        "dimension_model_z", "offensive_portable_z", "defensive_portable_z",
    ]
    # Positional z-scores: z-score of each dimension WITHIN position bucket
    print("  Computing three-level z-scores (league, positional, archetype)...")
    if "position_bucket" in qualified.columns:
        for col in dim_z_cols:
            if col in qualified.columns:
                qualified[f"{col}_positional"] = qualified.groupby(
                    ["season", "position_bucket"]
                )[col].transform(
                    lambda x: (x - x.mean()) / max(x.std(), 0.01)
                )
    # Archetype z-scores: z-score of each dimension WITHIN archetype
    if "primary_archetype" in qualified.columns:
        for col in dim_z_cols:
            if col in qualified.columns:
                qualified[f"{col}_archetype"] = qualified.groupby(
                    ["season", "primary_archetype"]
                )[col].transform(
                    lambda x: (x - x.mean()) / max(x.std(), 0.01) if len(x) >= 5 else 0.0
                )

    # 11. Compute Portable Talent Score (v2.5: z-score aggregation)
    print("  Computing Portable Talent Score (25% RAPM + 20% Playtype + 55% Dimensions)...")
    qualified = compute_portable_talent_score(qualified)

    # 12. Merge back with unqualified players
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
