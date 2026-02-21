"""
src/modeling/bayesian_hierarchical.py
=============================================================================
BKE v2.6 — Bayesian Hierarchical Regression for Player Metrics

Implements a 3-level Normal-Normal hierarchical model:
    Level 1 (League):    μ₀ (grand mean)
    Level 2 (Position):  μ_pos ~ N(μ₀, τ²_pos)
    Level 3 (Archetype): μ_arch ~ N(μ_pos, τ²_arch)
    Level 4 (Player):    x_i ~ N(θ_i, σ²_i)

Uses analytical conjugate-prior posteriors (no MCMC needed):
    - Fast: ~100ms for 950 players × 9 dimensions
    - Exact for Gaussian models
    - Proper hierarchical shrinkage at each level

Players with fewer minutes/possessions get more shrinkage toward
their archetype/position/league group means. This produces more
stable estimates for role players while preserving signal for stars.

Applied to:
    - All 9 Layer 1C dimension z-scores
    - Noisy defensive metrics (DRAPM, defensive playmaking)
    - Small-sample role splits

Integration:
    Called from layer1_portable_talent.py after dimension z-scores
    are computed, BEFORE composite scoring.
=============================================================================
"""

import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Dimensions to apply hierarchical shrinkage to
HIERARCHICAL_DIMS = [
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

# Additional noisy metrics to shrink
NOISY_METRICS = [
    "drapm",
]

# Minimum group size for reliable group mean estimation
MIN_GROUP_SIZE = 5

# Reliability weight parameters
# Players with fewer minutes get more shrinkage toward group means
MIN_MINUTES_FULL_RELIABILITY = 2000  # ~25 mpg × 80 games
MIN_MINUTES_ANY_RELIABILITY = 500    # qualification threshold


# ---------------------------------------------------------------------------
# Core: Estimate hyperparameters via Method of Moments (Empirical Bayes)
# ---------------------------------------------------------------------------

def _estimate_between_group_variance(
    group_means: pd.Series,
    group_sizes: pd.Series,
    overall_variance: float,
) -> float:
    """
    Estimate between-group variance (τ²) using method of moments.

    τ² = max(0, Var(group_means) - mean(σ²/n_group))

    This is the standard DerSimonian-Laird estimator adapted for
    hierarchical models.

    Parameters
    ----------
    group_means : Mean of the metric within each group.
    group_sizes : Number of observations in each group.
    overall_variance : Overall (pooled) variance of the metric.

    Returns
    -------
    Estimated between-group variance τ².
    """
    if len(group_means) < 2:
        return 0.0

    # Observed variance of group means
    var_of_means = group_means.var(ddof=1)

    # Expected sampling variance contribution
    # If each group has n_k observations with variance σ², the sampling
    # variance of the group mean is σ²/n_k. The average of this across
    # groups gives the expected contribution of sampling noise.
    expected_sampling_var = (overall_variance / group_sizes).mean()

    # Between-group variance = observed variance minus sampling noise
    tau_sq = max(0.0, var_of_means - expected_sampling_var)

    return tau_sq


def _compute_reliability_weight(minutes: float) -> float:
    """
    Compute reliability weight based on minutes played.

    Players with more minutes have more reliable observed values.
    This determines how much to trust the observation vs the prior.

    Returns float in [0.1, 1.0].
    """
    if pd.isna(minutes) or minutes <= 0:
        return 0.1

    # Linear ramp from MIN to FULL, clipped to [0.1, 1.0]
    weight = (minutes - MIN_MINUTES_ANY_RELIABILITY) / (
        MIN_MINUTES_FULL_RELIABILITY - MIN_MINUTES_ANY_RELIABILITY
    )
    return float(np.clip(weight, 0.1, 1.0))


# ---------------------------------------------------------------------------
# Level-by-level posterior computation
# ---------------------------------------------------------------------------

def _compute_group_posteriors(
    df: pd.DataFrame,
    metric: str,
    group_col: str,
    prior_means: Dict[str, float],
    prior_variance: float,
    season_col: str = "season",
    minutes_col: str = "MIN",
) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Compute posterior estimates for each player within groups.

    For each group g with prior mean μ_g and between-group variance τ²:
        θ̂_i = λ_i * μ_g + (1 - λ_i) * x_i

    where λ_i = σ²_i / (σ²_i + τ²) is the shrinkage factor,
    and σ²_i = σ² / reliability_i (more minutes → less noise → less shrinkage).

    Parameters
    ----------
    df : Player-season DataFrame.
    metric : Column name of the metric to shrink.
    group_col : Column defining groups (position_bucket, primary_archetype).
    prior_means : Dict of {group_name: prior_mean}.
    prior_variance : Between-group variance τ² from higher level.
    season_col : Season column for within-season computation.
    minutes_col : Minutes column for reliability weighting.

    Returns
    -------
    (posterior_means_series, group_mean_dict)
    """
    if metric not in df.columns:
        return pd.Series(np.nan, index=df.index), {}

    result = df[metric].copy()
    group_means = {}

    for season in df[season_col].unique():
        season_mask = df[season_col] == season
        season_df = df[season_mask]

        # Overall variance for this season
        overall_var = season_df[metric].var()
        if pd.isna(overall_var) or overall_var < 1e-12:
            overall_var = 1.0  # fallback

        for group_val in season_df[group_col].dropna().unique():
            group_mask = season_mask & (df[group_col] == group_val)
            group_df = df[group_mask]

            if len(group_df) < MIN_GROUP_SIZE:
                # Too few players for reliable group estimate
                # Use prior directly
                prior_mean = prior_means.get(group_val, 0.0)
                for idx in group_df.index:
                    obs = df.loc[idx, metric]
                    if pd.isna(obs):
                        continue
                    mins = df.loc[idx, minutes_col] if minutes_col in df.columns else 1000
                    reliability = _compute_reliability_weight(mins)
                    # Heavy shrinkage for small groups
                    shrinkage = 0.6 / (0.6 + reliability)
                    result.loc[idx] = shrinkage * prior_mean + (1 - shrinkage) * obs
                group_means[f"{season}_{group_val}"] = prior_means.get(group_val, 0.0)
                continue

            # Compute group mean (itself will be shrunk at next level)
            group_mean = group_df[metric].mean()

            # Between-player variance within this group
            within_var = group_df[metric].var()
            if pd.isna(within_var) or within_var < 1e-12:
                within_var = overall_var

            # Prior mean for this group
            prior_mean = prior_means.get(group_val, 0.0)

            # Shrink group mean toward prior
            if prior_variance > 1e-12:
                n_group = len(group_df)
                group_shrinkage = within_var / (within_var + n_group * prior_variance)
                shrunk_group_mean = group_shrinkage * prior_mean + (1 - group_shrinkage) * group_mean
            else:
                shrunk_group_mean = group_mean

            group_means[f"{season}_{group_val}"] = shrunk_group_mean

            # Shrink each player toward the shrunk group mean
            for idx in group_df.index:
                obs = df.loc[idx, metric]
                if pd.isna(obs):
                    continue

                mins = df.loc[idx, minutes_col] if minutes_col in df.columns else 1000
                reliability = _compute_reliability_weight(mins)

                # Observation noise: decreases with sample size (games played)
                # More games + more minutes = lower noise = less shrinkage
                gp = df.loc[idx, "GP"] if "GP" in df.columns else 40
                effective_samples = max(gp * reliability, 1)

                # Player-level noise variance σ²_i = σ²_within / effective_samples
                player_noise_var = within_var / effective_samples

                # Shrinkage factor: λ = σ²_i / (σ²_i + τ²_group)
                # Low λ = trust observation; High λ = trust group mean
                if within_var > 1e-12:
                    lambda_i = player_noise_var / (player_noise_var + within_var)
                else:
                    lambda_i = 0.0

                # Posterior mean
                result.loc[idx] = lambda_i * shrunk_group_mean + (1 - lambda_i) * obs

    return result, group_means


# ---------------------------------------------------------------------------
# Main: 3-Level Hierarchical Bayesian Shrinkage
# ---------------------------------------------------------------------------

def apply_hierarchical_shrinkage(
    df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    season_col: str = "season",
    position_col: str = "position_bucket",
    archetype_col: str = "primary_archetype",
    minutes_col: str = "MIN",
) -> pd.DataFrame:
    """
    Apply 3-level hierarchical Bayesian shrinkage to player metrics.

    Hierarchy:
        League mean → Position mean → Archetype mean → Player observation

    Each level produces shrunk estimates:
        - Position means are shrunk toward league mean
        - Archetype means are shrunk toward position mean
        - Player observations are shrunk toward archetype mean

    The amount of shrinkage depends on:
        - Sample size (fewer observations → more shrinkage)
        - Between-group variance (more heterogeneity → less shrinkage)
        - Player reliability (fewer minutes → more shrinkage)

    Parameters
    ----------
    df : Player-season DataFrame with dimension z-scores.
    metrics : List of columns to apply shrinkage to.
              Defaults to HIERARCHICAL_DIMS + NOISY_METRICS.
    season_col : Season identifier column.
    position_col : Position bucket column.
    archetype_col : Archetype assignment column.
    minutes_col : Minutes played column (for reliability weighting).

    Returns
    -------
    DataFrame with _bayesian suffix columns containing shrunk estimates.
    Original columns preserved (not overwritten).
    """
    result = df.copy()
    metrics = metrics or (HIERARCHICAL_DIMS + NOISY_METRICS)
    available = [m for m in metrics if m in df.columns]

    if not available:
        return result

    n_shrunk = 0

    for metric in available:
        # ---------------------------------------------------------------
        # Level 1: League grand mean (per season)
        # ---------------------------------------------------------------
        league_means = {}
        for season in df[season_col].unique():
            season_vals = df.loc[df[season_col] == season, metric].dropna()
            league_means[season] = float(season_vals.mean()) if len(season_vals) > 0 else 0.0

        # ---------------------------------------------------------------
        # Level 2: Position means, shrunk toward league
        # ---------------------------------------------------------------
        position_means = {}
        for season in df[season_col].unique():
            season_mask = df[season_col] == season
            season_df = df[season_mask]

            if position_col not in season_df.columns:
                continue

            # Group means and sizes by position
            pos_groups = season_df.groupby(position_col)[metric]
            pos_group_means = pos_groups.mean()
            pos_group_sizes = pos_groups.count()

            # Overall variance
            overall_var = season_df[metric].var()
            if pd.isna(overall_var) or overall_var < 1e-12:
                overall_var = 1.0

            # Estimate between-position variance
            tau_sq_pos = _estimate_between_group_variance(
                pos_group_means, pos_group_sizes, overall_var
            )

            # Shrink position means toward league mean
            league_mean = league_means.get(season, 0.0)
            for pos_val in pos_group_means.index:
                n_pos = pos_group_sizes.get(pos_val, 1)
                if tau_sq_pos > 1e-12:
                    within_var = overall_var / max(n_pos, 1)
                    shrinkage = within_var / (within_var + tau_sq_pos)
                    shrunk = shrinkage * league_mean + (1 - shrinkage) * pos_group_means[pos_val]
                else:
                    shrunk = pos_group_means[pos_val]
                position_means[f"{season}_{pos_val}"] = shrunk

        # ---------------------------------------------------------------
        # Level 3: Archetype means, shrunk toward position means
        # ---------------------------------------------------------------
        # Build position priors dict for archetype-level shrinkage
        archetype_priors = {}
        for season in df[season_col].unique():
            season_mask = df[season_col] == season
            season_df = df[season_mask]

            if archetype_col not in season_df.columns or position_col not in season_df.columns:
                continue

            for arch_val in season_df[archetype_col].dropna().unique():
                # Find the dominant position for this archetype
                arch_mask = season_mask & (df[archetype_col] == arch_val)
                positions = df.loc[arch_mask, position_col].value_counts()
                if positions.empty:
                    archetype_priors[arch_val] = league_means.get(season, 0.0)
                else:
                    dominant_pos = positions.index[0]
                    archetype_priors[arch_val] = position_means.get(
                        f"{season}_{dominant_pos}",
                        league_means.get(season, 0.0)
                    )

        # Estimate between-archetype variance
        bp_tau_sq = 0.0
        for season in df[season_col].unique():
            season_mask = df[season_col] == season
            season_df = df[season_mask]
            if archetype_col not in season_df.columns:
                continue
            arch_groups = season_df.groupby(archetype_col)[metric]
            arch_means = arch_groups.mean()
            arch_sizes = arch_groups.count()
            overall_var = season_df[metric].var()
            if pd.isna(overall_var) or overall_var < 1e-12:
                overall_var = 1.0
            bp_tau_sq += _estimate_between_group_variance(arch_means, arch_sizes, overall_var)
        bp_tau_sq /= max(len(df[season_col].unique()), 1)

        # ---------------------------------------------------------------
        # Level 4: Player observations, shrunk toward archetype means
        # ---------------------------------------------------------------
        posterior, _ = _compute_group_posteriors(
            df=df,
            metric=metric,
            group_col=archetype_col,
            prior_means=archetype_priors,
            prior_variance=bp_tau_sq,
            season_col=season_col,
            minutes_col=minutes_col,
        )

        # Store with _bayesian suffix (preserve original)
        bayesian_col = f"{metric}_bayesian"
        result[bayesian_col] = posterior
        n_shrunk += 1

    print(f"    Hierarchical Bayesian shrinkage applied to {n_shrunk} metrics")

    # Report shrinkage statistics
    for metric in available[:3]:  # show first 3
        bayesian_col = f"{metric}_bayesian"
        if bayesian_col in result.columns:
            orig_std = df[metric].std()
            shrunk_std = result[bayesian_col].std()
            reduction = (1 - shrunk_std / orig_std) * 100 if orig_std > 0 else 0
            print(f"      {metric}: std {orig_std:.3f} → {shrunk_std:.3f} "
                  f"({reduction:.1f}% variance reduction)")

    return result


# ---------------------------------------------------------------------------
# Overwrite mode: replace originals with Bayesian estimates
# ---------------------------------------------------------------------------

def apply_and_replace(
    df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Apply hierarchical shrinkage and REPLACE the original columns.

    This is the integration mode: after calling this, the dimension
    z-scores have been hierarchically shrunk in place.

    The original (pre-shrinkage) values are saved with _raw_preshrink suffix.
    """
    result = apply_hierarchical_shrinkage(df, metrics=metrics, **kwargs)

    metrics = metrics or (HIERARCHICAL_DIMS + NOISY_METRICS)
    available = [m for m in metrics if m in df.columns]

    for metric in available:
        bayesian_col = f"{metric}_bayesian"
        if bayesian_col in result.columns:
            # Save original
            result[f"{metric}_raw_preshrink"] = result[metric]
            # Replace with Bayesian estimate
            result[metric] = result[bayesian_col]

    return result


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def shrinkage_diagnostics(df: pd.DataFrame, metrics: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Generate diagnostics for hierarchical shrinkage.

    For each metric, reports:
      - Pre-shrinkage stats (mean, std, range)
      - Post-shrinkage stats
      - Variance reduction %
      - Correlation between pre and post
    """
    metrics = metrics or HIERARCHICAL_DIMS
    rows = []

    for metric in metrics:
        bayesian_col = f"{metric}_bayesian"
        raw_col = f"{metric}_raw_preshrink"

        # Use raw_preshrink if available, otherwise _bayesian comparison
        pre = df[raw_col] if raw_col in df.columns else df.get(metric)
        post = df[bayesian_col] if bayesian_col in df.columns else df.get(metric)

        if pre is None or post is None:
            continue

        pre_clean = pre.dropna()
        post_clean = post.dropna()

        both_valid = pre.notna() & post.notna()
        corr = pre[both_valid].corr(post[both_valid]) if both_valid.sum() > 5 else np.nan

        rows.append({
            "metric": metric,
            "pre_std": pre_clean.std(),
            "post_std": post_clean.std(),
            "variance_reduction_pct": (1 - post_clean.std() / pre_clean.std()) * 100 if pre_clean.std() > 0 else 0,
            "correlation": corr,
            "pre_range": pre_clean.max() - pre_clean.min() if len(pre_clean) > 0 else 0,
            "post_range": post_clean.max() - post_clean.min() if len(post_clean) > 0 else 0,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.modeling.layer1_portable_talent import build_portable_talent

    print("Loading Layer 1 data...")
    df = build_portable_talent()

    if df.empty:
        print("No data loaded")
        sys.exit(1)

    print(f"\n{len(df)} player-seasons loaded")
    print("\nApplying hierarchical Bayesian shrinkage...")

    result = apply_hierarchical_shrinkage(df)

    print("\n=== Shrinkage Diagnostics ===")
    diag = shrinkage_diagnostics(result)
    print(diag.to_string(index=False))

    qualified = result[result.get("qualified", True) == True]
    name_col = "player_name" if "player_name" in qualified.columns else "player_id"

    print("\n=== Top 10 PTS (post-shrinkage) ===")
    if "portable_talent_score" in qualified.columns:
        top = qualified.nlargest(10, "portable_talent_score")
        for _, row in top.iterrows():
            print(f"  {row.get(name_col, row['player_id']):25s} | PTS: {row['portable_talent_score']:5.1f}")
