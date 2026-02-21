"""
src/modeling/percentile_engine.py
=============================================================================
BKE v2.5 — Three-Level Percentile Standardization + Z-Score Engine.

Computes percentiles and z-scores at three structural levels:
  1. League-wide     — macro comparison across all qualified players
  2. Positional      — role-fairness within positional cohort (G, GF, F, FC, C)
  3. Archetype       — micro peer comparison within assigned archetype

v2.5 additions (over v2.0):
  - Archetype-conditional neutralization: subtract archetype expected z
  - Three-level z-scores (league, position, archetype) for dimensions
  - Strict terminal-only percentile enforcement

v2.0 retained:
  - Z-score aggregation: Raw → Z → Weighted Sum → Final Z → Final Percentile
    (percentiles are presentation-only, NOT aggregation math)
  - Bayesian shrinkage for noisy metrics (defensive playmaking, on/off)
  - Winsorized z-scores to prevent outlier distortion

Design:
  - Percentile computations use pandas rank method (vectorized, fast).
  - Z-score computations use scipy.stats.zscore with NaN handling.
  - Small cohorts are Bayesian-shrunk toward the league values.
  - Results cached per season for efficient downstream use.
=============================================================================
"""

import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.modeling.model_config import (
    PERCENTILE,
    ZSCORE,
    BAYESIAN_SHRINKAGE,
    NEUTRALIZATION,
    clean_id,
)


# ---------------------------------------------------------------------------
# Core: Percentile computation
# ---------------------------------------------------------------------------

def compute_percentile(value: float, distribution: np.ndarray) -> float:
    """
    Compute the percentile rank of a value within a distribution.

    Uses 'rank' method (average of 'weak' and 'strict') for robust handling
    of ties and sparse distributions.

    Returns:
        Percentile rank in [0, 100].
    """
    if len(distribution) == 0 or np.isnan(value):
        return np.nan
    return float(stats.percentileofscore(distribution, value, kind="rank"))


def compute_percentiles_for_column(
    df: pd.DataFrame,
    column: str,
    output_col: Optional[str] = None,
) -> pd.Series:
    """
    Compute league-wide percentile for every row in ``column``.

    Parameters
    ----------
    df : DataFrame with the column to percentile-rank.
    column : name of the numeric column.
    output_col : name of output column (defaults to ``{column}_pctl``).

    Returns
    -------
    Series of percentile ranks [0, 100].
    """
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, name=output_col or f"{column}_pctl")

    values = df[column].dropna().values
    result = df[column].apply(lambda v: compute_percentile(v, values))
    result.name = output_col or f"{column}_pctl"
    return result


# ---------------------------------------------------------------------------
# Three-level percentile engine
# ---------------------------------------------------------------------------

class PercentileEngine:
    """
    Computes and stores percentile standardization at three cohort levels.

    Levels:
        1. League    — all qualified players in a season
        2. Position  — positional bucket within a season
        3. Archetype — assigned offensive archetype within a season

    Usage:
        engine = PercentileEngine()
        engine.fit(df, metrics=["rapm", "orapm", "drapm", ...])
        result = engine.transform(df)
    """

    def __init__(self, config=None):
        self.config = config or PERCENTILE
        # Internal caches: {season: {metric: distribution_array}}
        self._league_dists: Dict[str, Dict[str, np.ndarray]] = {}
        self._position_dists: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
        self._archetype_dists: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}

    # -----------------------------------------------------------------------
    # Fit: build distributions from data
    # -----------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        metrics: List[str],
        season_col: str = "season",
        position_col: str = "position_bucket",
        archetype_col: str = "primary_archetype",
    ) -> "PercentileEngine":
        """
        Build percentile reference distributions for all three levels.

        Parameters
        ----------
        df : DataFrame with player-season rows.
        metrics : list of numeric column names to standardize.
        season_col : column containing season identifier.
        position_col : column with position bucket (Guard/Forward/Center etc).
        archetype_col : column with primary archetype assignment.
        """
        available_metrics = [m for m in metrics if m in df.columns]

        for season in df[season_col].unique():
            season_df = df[df[season_col] == season]

            # 1. League distributions
            self._league_dists[season] = {}
            for metric in available_metrics:
                vals = season_df[metric].dropna().values
                self._league_dists[season][metric] = vals

            # 2. Position distributions
            self._position_dists[season] = {}
            if position_col in df.columns:
                for pos_bucket in season_df[position_col].dropna().unique():
                    pos_df = season_df[season_df[position_col] == pos_bucket]
                    self._position_dists[season][pos_bucket] = {}
                    for metric in available_metrics:
                        vals = pos_df[metric].dropna().values
                        self._position_dists[season][pos_bucket][metric] = vals

            # 3. Archetype distributions
            self._archetype_dists[season] = {}
            if archetype_col in df.columns:
                for archetype in season_df[archetype_col].dropna().unique():
                    arch_df = season_df[season_df[archetype_col] == archetype]
                    self._archetype_dists[season][archetype] = {}
                    for metric in available_metrics:
                        vals = arch_df[metric].dropna().values
                        self._archetype_dists[season][archetype][metric] = vals

        return self

    # -----------------------------------------------------------------------
    # Transform: compute percentiles for each player row
    # -----------------------------------------------------------------------

    def transform(
        self,
        df: pd.DataFrame,
        metrics: List[str],
        season_col: str = "season",
        position_col: str = "position_bucket",
        archetype_col: str = "primary_archetype",
        prefix: str = "",
    ) -> pd.DataFrame:
        """
        Add three-level percentile columns for each metric.

        For each ``metric``, adds:
            - ``{prefix}{metric}_league_pctl``
            - ``{prefix}{metric}_position_pctl``
            - ``{prefix}{metric}_archetype_pctl``

        Small archetype/position cohorts are Bayesian-shrunk toward league.
        """
        result = df.copy()
        available_metrics = [m for m in metrics if m in df.columns]

        for metric in available_metrics:
            league_pctls = []
            position_pctls = []
            archetype_pctls = []

            for idx, row in df.iterrows():
                season = row.get(season_col, None)
                value = row.get(metric, np.nan)

                # League percentile
                league_dist = self._league_dists.get(season, {}).get(metric, np.array([]))
                league_p = compute_percentile(value, league_dist)
                league_pctls.append(league_p)

                # Position percentile (with shrinkage)
                pos_bucket = row.get(position_col, None)
                pos_dist = (
                    self._position_dists.get(season, {})
                    .get(pos_bucket, {})
                    .get(metric, np.array([]))
                )
                if len(pos_dist) >= self.config.min_cohort_size:
                    position_p = compute_percentile(value, pos_dist)
                else:
                    # Shrink toward league
                    raw_pos = compute_percentile(value, pos_dist) if len(pos_dist) > 0 else league_p
                    shrink = self.config.small_cohort_shrinkage
                    position_p = shrink * league_p + (1 - shrink) * raw_pos if not np.isnan(raw_pos) else league_p
                position_pctls.append(position_p)

                # Archetype percentile (with shrinkage)
                archetype = row.get(archetype_col, None)
                arch_dist = (
                    self._archetype_dists.get(season, {})
                    .get(archetype, {})
                    .get(metric, np.array([]))
                )
                if len(arch_dist) >= self.config.min_cohort_size:
                    archetype_p = compute_percentile(value, arch_dist)
                else:
                    raw_arch = compute_percentile(value, arch_dist) if len(arch_dist) > 0 else league_p
                    shrink = self.config.small_cohort_shrinkage
                    archetype_p = shrink * league_p + (1 - shrink) * raw_arch if not np.isnan(raw_arch) else league_p
                archetype_pctls.append(archetype_p)

            result[f"{prefix}{metric}_league_pctl"] = league_pctls
            result[f"{prefix}{metric}_position_pctl"] = position_pctls
            result[f"{prefix}{metric}_archetype_pctl"] = archetype_pctls

        return result

    # -----------------------------------------------------------------------
    # Convenience: get a single player's percentile card
    # -----------------------------------------------------------------------

    def get_player_card(
        self,
        df: pd.DataFrame,
        player_id: str,
        season: str,
        metrics: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """
        Return a percentile card for a single player-season.

        Returns dict of {metric: {"league": X, "position": Y, "archetype": Z}}.
        """
        row = df[(df["player_id"] == player_id) & (df["season"] == season)]
        if row.empty:
            return {}

        card = {}
        for metric in metrics:
            league_col = f"{metric}_league_pctl"
            pos_col = f"{metric}_position_pctl"
            arch_col = f"{metric}_archetype_pctl"
            card[metric] = {
                "league": float(row[league_col].iloc[0]) if league_col in row.columns else np.nan,
                "position": float(row[pos_col].iloc[0]) if pos_col in row.columns else np.nan,
                "archetype": float(row[arch_col].iloc[0]) if arch_col in row.columns else np.nan,
            }
        return card


# ---------------------------------------------------------------------------
# Vectorized percentile (faster for large DataFrames)
# ---------------------------------------------------------------------------

def vectorized_percentile_rank(series: pd.Series) -> pd.Series:
    """
    Compute percentile rank for every element in a Series.

    Uses pandas rank method, much faster than row-by-row scipy calls
    for large DataFrames.

    Returns:
        Series of percentile ranks in [0, 100].
    """
    if series.empty:
        return series
    return series.rank(pct=True, method="average") * 100


def add_league_percentiles(
    df: pd.DataFrame,
    metrics: List[str],
    season_col: str = "season",
    suffix: str = "_league_pctl",
) -> pd.DataFrame:
    """
    Fast vectorized league-wide percentile computation (per season).

    This is the fast path used when only league percentiles are needed.
    """
    result = df.copy()
    available = [m for m in metrics if m in df.columns]

    for metric in available:
        result[f"{metric}{suffix}"] = (
            result.groupby(season_col)[metric]
            .transform(vectorized_percentile_rank)
        )
    return result


def add_grouped_percentiles(
    df: pd.DataFrame,
    metrics: List[str],
    group_col: str,
    season_col: str = "season",
    suffix: str = "_pctl",
    min_group_size: int = 10,
    shrink_to_league: float = 0.30,
) -> pd.DataFrame:
    """
    Compute percentiles within groups, with small-group shrinkage to league.

    Parameters
    ----------
    df : DataFrame
    metrics : columns to percentile-rank
    group_col : column defining groups (position_bucket, primary_archetype, etc.)
    season_col : season column
    suffix : appended to metric name for output column
    min_group_size : below this, apply Bayesian shrinkage to league
    shrink_to_league : shrinkage weight toward league percentile for small groups
    """
    result = df.copy()
    available = [m for m in metrics if m in df.columns]

    for metric in available:
        col_name = f"{metric}_{group_col}{suffix}"
        league_pctl_col = f"{metric}_league_pctl"

        # Compute group percentiles
        group_pctl = (
            result.groupby([season_col, group_col])[metric]
            .transform(vectorized_percentile_rank)
        )

        # Compute group sizes
        group_sizes = result.groupby([season_col, group_col])[metric].transform("count")

        # If league percentile exists, shrink small groups
        if league_pctl_col in result.columns:
            is_small = group_sizes < min_group_size
            league_pctl = result[league_pctl_col]
            group_pctl = np.where(
                is_small,
                shrink_to_league * league_pctl + (1 - shrink_to_league) * group_pctl,
                group_pctl,
            )

        result[col_name] = group_pctl

    return result


# ---------------------------------------------------------------------------
# v2.0: Z-Score Aggregation Engine
# ---------------------------------------------------------------------------

def compute_z_score(series: pd.Series, winsorize: float = 3.5) -> pd.Series:
    """
    Compute z-score for a Series, with optional winsorization.

    Z-scores preserve interval meaning for aggregation, unlike percentiles.

    Parameters
    ----------
    series : Raw metric values.
    winsorize : Clip z-scores to [-limit, +limit] to prevent outlier distortion.

    Returns
    -------
    Series of z-scores (mean=0, std=1), NaN preserved.
    """
    if series.dropna().empty or series.std() < 1e-12:
        return pd.Series(0.0, index=series.index)

    z = (series - series.mean()) / (series.std() + 1e-12)
    if winsorize > 0:
        z = z.clip(-winsorize, winsorize)
    return z


def add_league_z_scores(
    df: pd.DataFrame,
    metrics: List[str],
    season_col: str = "season",
    suffix: str = "_z",
    winsorize: float = 3.5,
) -> pd.DataFrame:
    """
    Compute league-wide z-scores per season for each metric.

    This is the primary aggregation input for v2.0.
    """
    result = df.copy()
    available = [m for m in metrics if m in df.columns]

    for metric in available:
        result[f"{metric}{suffix}"] = (
            result.groupby(season_col)[metric]
            .transform(lambda x: compute_z_score(x, winsorize))
        )
    return result


def add_grouped_z_scores(
    df: pd.DataFrame,
    metrics: List[str],
    group_col: str,
    season_col: str = "season",
    suffix: str = "_z",
    min_group_size: int = 10,
    shrink_to_league: float = 0.30,
    winsorize: float = 3.5,
) -> pd.DataFrame:
    """
    Compute z-scores within groups, with small-group shrinkage to league z-score.

    Parameters
    ----------
    df : DataFrame
    metrics : columns to z-score
    group_col : column defining groups (position_bucket, primary_archetype, etc.)
    suffix : appended to metric name for output column
    min_group_size : below this, apply Bayesian shrinkage to league z
    shrink_to_league : shrinkage weight toward league z for small groups
    """
    result = df.copy()
    available = [m for m in metrics if m in df.columns]

    for metric in available:
        col_name = f"{metric}_{group_col}{suffix}"
        league_z_col = f"{metric}_z"

        # Compute group z-scores
        group_z = (
            result.groupby([season_col, group_col])[metric]
            .transform(lambda x: compute_z_score(x, winsorize))
        )

        # Compute group sizes
        group_sizes = result.groupby([season_col, group_col])[metric].transform("count")

        # If league z exists, shrink small groups
        if league_z_col in result.columns:
            is_small = group_sizes < min_group_size
            league_z = result[league_z_col]
            group_z = np.where(
                is_small,
                shrink_to_league * league_z + (1 - shrink_to_league) * group_z,
                group_z,
            )

        result[col_name] = group_z

    return result


def z_to_percentile(z_scores: pd.Series) -> pd.Series:
    """
    Convert z-scores to percentiles using the normal CDF.

    This is the final presentation step in v2.0:
      Raw → Z → Weighted Sum → Final Z → Final Percentile (this step)
    """
    from scipy.stats import norm
    return pd.Series(
        norm.cdf(z_scores.fillna(0)) * 100,
        index=z_scores.index,
    )


def weighted_z_composite(
    df: pd.DataFrame,
    z_columns: List[str],
    weights: List[float],
    output_col: str = "composite_z",
) -> pd.DataFrame:
    """
    Compute a weighted sum of z-score columns.

    This replaces percentile averaging in v2.0:
      composite_z = sum(w_i * z_i) / sum(w_i)

    All z-scores are interval-scaled, so weighted sums are meaningful.
    """
    result = df.copy()
    available_cols = [c for c in z_columns if c in df.columns]
    available_weights = [w for c, w in zip(z_columns, weights) if c in df.columns]

    if not available_cols:
        result[output_col] = 0.0
        return result

    total_weight = sum(available_weights)
    if total_weight <= 0:
        result[output_col] = 0.0
        return result

    composite = pd.Series(0.0, index=df.index)
    for col, w in zip(available_cols, available_weights):
        composite += w * df[col].fillna(0)

    result[output_col] = composite / total_weight
    return result


# ---------------------------------------------------------------------------
# v2.0: Bayesian Shrinkage
# ---------------------------------------------------------------------------

def apply_bayesian_shrinkage(
    series: pd.Series,
    prior: float,
    shrinkage_strength: float = 0.20,
    gp: Optional[pd.Series] = None,
    min_gp_full: int = 50,
) -> pd.Series:
    """
    Apply empirical Bayes shrinkage to a metric series.

    Shrinks toward a prior (league mean or positional mean) based on
    sample size and configured strength.

    Parameters
    ----------
    series : Raw metric values.
    prior : The prior to shrink toward (e.g., league mean).
    shrinkage_strength : Base shrinkage strength (0 = no shrinkage, 1 = full).
    gp : Games played per player (used to scale shrinkage).
    min_gp_full : Games threshold for full-weight (no extra shrinkage).

    Returns
    -------
    Shrunk series.
    """
    if gp is not None:
        # More shrinkage for low-GP players
        gp_factor = np.clip(gp / min_gp_full, 0.0, 1.0)
        effective_shrinkage = shrinkage_strength * (1 + (1 - gp_factor))
        effective_shrinkage = np.clip(effective_shrinkage, 0.0, 0.8)
    else:
        effective_shrinkage = shrinkage_strength

    shrunk = (1 - effective_shrinkage) * series + effective_shrinkage * prior
    return shrunk


# ---------------------------------------------------------------------------
# v2.5: Archetype-Conditional Neutralization
# ---------------------------------------------------------------------------

def neutralize_by_archetype(
    df: pd.DataFrame,
    z_col: str,
    archetype_col: str = "primary_archetype",
    season_col: str = "season",
    min_cohort: int = 8,
    small_cohort_shrinkage: float = 0.50,
) -> pd.Series:
    """
    Archetype-conditional neutralization for a dimension z-score.

    Subtracts the archetype expected z-score so that the neutralized score
    measures ability ABOVE what's expected for the player's role.

    Neutralized_z = Observed_z - E[z | archetype, season]

    For small archetype cohorts (< min_cohort), shrinks the expected value
    toward the league mean (0) to avoid noisy estimates.

    Parameters
    ----------
    df : DataFrame with z-score column, archetype, and season.
    z_col : Name of the z-score column to neutralize.
    archetype_col : Column with archetype assignment.
    season_col : Season column.
    min_cohort : Minimum archetype cohort size for reliable expected value.
    small_cohort_shrinkage : Shrinkage toward league mean for small cohorts.

    Returns
    -------
    Series of neutralized z-scores.
    """
    if z_col not in df.columns:
        return pd.Series(0.0, index=df.index)

    result = df[z_col].copy()

    if archetype_col not in df.columns:
        # No archetype → return raw z (league-relative already)
        return result

    for season in df[season_col].unique():
        s_mask = df[season_col] == season
        for arch in df.loc[s_mask, archetype_col].dropna().unique():
            a_mask = s_mask & (df[archetype_col] == arch)
            cohort_z = df.loc[a_mask, z_col]
            cohort_size = cohort_z.notna().sum()

            if cohort_size < 2:
                # Too small to estimate — assume expected = 0 (league avg)
                expected = 0.0
            elif cohort_size < min_cohort:
                # Small cohort: shrink toward league mean (0)
                raw_expected = cohort_z.mean()
                expected = (1 - small_cohort_shrinkage) * raw_expected
            else:
                expected = cohort_z.mean()

            result.loc[a_mask] = df.loc[a_mask, z_col] - expected

    return result


def neutralize_dimensions(
    df: pd.DataFrame,
    dim_z_cols: List[str],
    archetype_col: str = "primary_archetype",
    season_col: str = "season",
) -> pd.DataFrame:
    """
    Apply archetype-conditional neutralization to a list of dimension z-scores.

    Modifies the DataFrame in-place, replacing raw z-scores with neutralized ones.
    Stores the raw (pre-neutralization) values in {col}_raw columns for diagnostics.

    Parameters
    ----------
    df : DataFrame with dimension z-scores.
    dim_z_cols : List of z-score column names to neutralize.
    archetype_col : Column with archetype assignment.
    season_col : Season column.

    Returns
    -------
    DataFrame with neutralized z-scores (original columns overwritten).
    """
    cfg = NEUTRALIZATION
    result = df.copy()

    for z_col in dim_z_cols:
        if z_col not in result.columns:
            continue

        # Skip dimensions that are already context-neutral
        if z_col in cfg.skip_neutralization:
            continue

        # Store raw for diagnostics
        result[f"{z_col}_raw"] = result[z_col].copy()

        # Neutralize
        result[z_col] = neutralize_by_archetype(
            result, z_col,
            archetype_col=archetype_col,
            season_col=season_col,
            min_cohort=cfg.min_cohort_for_neutralization,
            small_cohort_shrinkage=cfg.small_cohort_shrinkage,
        )

    return result
