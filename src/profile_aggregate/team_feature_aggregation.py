"""
src/profile_aggregate/team_feature_aggregation.py
=============================================================================
PEC Step 3 — Team Feature Aggregation (Offensive & Defensive Structure Layer)

Computes team-level structural metrics from player impact profiles and
minute predictions. Generates:
  - Offensive Mean Model: TalentBase + InteractionTerm + StructureTerm
  - Defensive Mean Model: DefTalent + Essentials + Diversity - Liability
  - Team Net Rating (Mean)
  - Volatility Model: base + 3PA + creation concentration + transition
  - Transition as BOTH structure term AND volatility component

Inputs:
  data/processed/player_eval/player_impact_profiles.parquet
  data/processed/player_eval/minute_model_predictions_v2.parquet
  data/historical/team_summaries.parquet  (actual wins/net rating for validation)

Output:
  data/processed/player_eval/team_feature_aggregation.parquet
  reports/player_eval_step3_team_features_validation.json

Usage:
  python3 src/player_eval/team_feature_aggregation.py

Architecture:
  Each component is toggleable via flags (use_interaction, use_structure,
  use_defense_architecture, use_volatility) for ablation testing.
=============================================================================
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.player_eval.constants import (
    PLAYER_PROFILES_PARQUET,
    MINUTE_PREDICTIONS_PATH,
    TEAM_FEATURES_PATH,
    STEP3_VALIDATION_REPORT,
    HISTORICAL_DIR,
    REPORTS_DIR,
    # Team scaling
    DEFAULT_TEAM_SCALE,
    TEAM_SCALE_MIN,
    TEAM_SCALE_MAX,
    TEAM_SCALE_FIT_R_THRESHOLD,
    MODIFIER_MAX_FRACTION,
    # Interaction
    INTERACTION_LAMBDA,
    INTERACTION_CAP,
    # Structure
    BETA_TOV,
    BETA_FTR,
    BETA_TRANSITION,
    PLAYMAKING_MPG_THRESHOLD,
    PLAYMAKING_AST_THRESHOLD,
    SPACING_MPG_THRESHOLD,
    SPACING_3PA_RATE_THRESHOLD,
    SPACING_3P_PCT_THRESHOLD,
    STRUCTURE_CAP,
    TRANSITION_PPP_LEAGUE_AVG,
    # Structure magnitudes (v2)
    PLAYMAKING_SOLO_PENALTY,
    PLAYMAKING_DEEP_BONUS,
    SPACING_POOR_PENALTY,
    SPACING_ELITE_BONUS,
    # Defense
    DEFENSE_CAP,
    RP_MPG_THRESHOLD,
    POA_MPG_THRESHOLD,
    DIVERSITY_MPG_THRESHOLD,
    LIABILITY_DBKE_THRESHOLD,
    LIABILITY_MPG_THRESHOLD,
    # Defense magnitudes (v2)
    RP_MISSING_PENALTY,
    POA_MISSING_PENALTY,
    BOTH_MISSING_PENALTY,
    ANCHOR_RP_WEIGHT,
    ANCHOR_POA_WEIGHT,
    DIVERSITY_BONUS_PER_ARCH,
    DIVERSITY_BONUS_CAP,
    LIABILITY_PER_PLAYER,
    LIABILITY_STACKING_PENALTY,
    # Volatility
    VOL_FLOOR,
    VOL_CEILING,
    ALPHA_3PA,
    ALPHA_CREATION,
    ALPHA_TRANSITION,
    CREATION_CONCENTRATION_THRESHOLD,
    # Star concentration
    ENABLE_FORECAST_STAR_CONCENTRATION,
    FORECAST_STAR_TOP1_SHARE_THRESHOLD,
    FORECAST_STAR_TOP2_SHARE_THRESHOLD,
    FORECAST_STAR_CONCENTRATION_PENALTY_MAX,
    FORECAST_STAR_CONCENTRATION_PENALTY_SLOPE,
)


def _replacement_pool_mask(df: pd.DataFrame) -> pd.Series:
    """Identify synthetic replacement-pool rows across old/new schemas."""
    if df.empty:
        return pd.Series(False, index=df.index, dtype=bool)

    if "player_id" in df.columns:
        pid = df["player_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    else:
        pid = pd.Series("", index=df.index, dtype=str)
    id_mask = pid.str.lower().str.startswith("repl_")

    if "player_name" in df.columns:
        names = df["player_name"].astype(str)
    else:
        names = pd.Series("", index=df.index, dtype=str)
    name_mask = names.str.contains("replacement pool", case=False, na=False)

    flag_mask = pd.Series(False, index=df.index, dtype=bool)
    if "is_replacement_pool" in df.columns:
        flag_mask = pd.to_numeric(df["is_replacement_pool"], errors="coerce").fillna(0).astype(int) == 1

    return id_mask | name_mask | flag_mask


# ═════════════════════════════════════════════════════════════════════
# Offensive Archetype Interaction Matrix
# ═════════════════════════════════════════════════════════════════════

# Short labels for the 11 offensive archetypes
OFF_ARCHETYPES = [
    "ball_dominant_creator",   # BDC
    "all_around_scorer",       # AAS
    "ballhandler",             # BH
    "interior_scorer",         # IS
    "perimeter_scorer",        # PS
    "connector",               # CON
    "pnr_rolling_big",         # PRB
    "pnr_popping_big",         # PPB
    "off_ball_finisher",       # OBF
    "off_ball_movement_shooter",  # OBM
    "off_ball_stationary_shooter", # OBS
]

# Numeric interaction values — RETAINED for historical reference only.
# 4.6 v2 (2026-05-21) eliminated the hand-coded INTERACTION_MATRIX after empirical
# Lasso fit on 4,914 lineup-stints (8 seasons, talent-controlled) found ZERO pair
# effects above the cross-validated significance threshold. Constants left in place
# in case future research wants to re-introduce specific pairs.
V_PP = +0.10   # ++
V_P  = +0.06   # +
V_MP = +0.03   # mild +
V_M  = -0.06   # -
V_MM = -0.03   # mild -
V_N  = -0.10   # --
V_0  = 0.00    # n/a

# INTERACTION_MATRIX — EMPTY by empirical decision (Phase 4.6 v2, 2026-05-21).
#
# The original 66-entry hand-coded matrix was tested against 8 seasons of NBA lineup
# data (4,914 lineup-stints, 2017-25, talent-controlled via season-z-scored oRAPM).
# Lasso regression with cross-validated alpha returned ZERO surviving pairs above the
# significance threshold. Strict-data sensitivity check at ≥200 possessions
# (1,895 lineups) and an alternative oBPM talent control confirmed the result.
#
# Interpretation: after controlling for individual player talent and the existing
# defensive composition penalties, archetype-pair composition contributes negligible
# predictive value for offensive efficiency. Pair-based interaction effects either
# do not exist at the population level, or are already absorbed by context-aware
# talent metrics like oRAPM.
#
# The matrix is left as an empty dict so that _get_interaction_value() returns
# (V_0=0.0, None) for any pair via its fallback path. Constants V_PP..V_N retained
# above for historical/research reference. The use_interaction flag in main() now
# defaults to False (the interaction loop is computationally wasteful when all
# values are 0).
#
# Findings: reports/archetype_interaction_fit_v2.json
# Fit script: src/modeling/fit_archetype_interactions_v2.py
INTERACTION_MATRIX: dict = {}


def _get_interaction_value(arch_i: str, arch_j: str) -> Tuple[float, Optional[str]]:
    """Look up interaction value for an archetype pair (order-independent)."""
    key = (arch_i, arch_j)
    if key in INTERACTION_MATRIX:
        return INTERACTION_MATRIX[key]
    key = (arch_j, arch_i)
    if key in INTERACTION_MATRIX:
        return INTERACTION_MATRIX[key]
    return (V_0, None)


# ═════════════════════════════════════════════════════════════════════
# Core computation functions
# ═════════════════════════════════════════════════════════════════════


def compute_star_concentration(
    players: pd.DataFrame,
    team_scale: float,
    forecast_mode: bool = False,
) -> Dict[str, float]:
    """Compute star concentration metrics and fragility penalty.

    Measures how much of a team's projected value is concentrated in the
    top 1-2 players.  Teams with extreme concentration historically
    underperform vs. balanced rosters of similar total talent because:
      - Injury/trade to star = catastrophic drop
      - Opponents scheme aggressively against lone stars
      - Supporting cast quality matters more than raw accumulation

    Returns dict with concentration metrics and a net-rating penalty.
    """
    result = {}
    tp = players.copy()
    tp["impact_bke"] = pd.to_numeric(tp.get("impact_bke"), errors="coerce").fillna(0.0)
    tp["minute_share"] = pd.to_numeric(tp.get("minute_share"), errors="coerce").fillna(0.0)

    # Weighted impact contribution per player
    tp["weighted_impact"] = tp["minute_share"] * tp["impact_bke"]
    team_net = tp["weighted_impact"].sum()

    # Sort by weighted impact descending
    tp = tp.sort_values("weighted_impact", ascending=False)
    impacts = tp["weighted_impact"].values

    if len(impacts) == 0 or abs(team_net) < 1e-9:
        result["star_top1_impact"] = 0.0
        result["star_top2_impact"] = 0.0
        result["star_top3_impact"] = 0.0
        result["star_top1_share"] = 0.0
        result["star_top2_share"] = 0.0
        result["star_concentration_penalty"] = 0.0
        return result

    top1 = float(impacts[0]) if len(impacts) >= 1 else 0.0
    top2 = float(impacts[:2].sum()) if len(impacts) >= 2 else top1
    top3 = float(impacts[:3].sum()) if len(impacts) >= 3 else top2

    # Share of positive team impact from top players
    positive_sum = float(tp[tp["weighted_impact"] > 0]["weighted_impact"].sum())
    if positive_sum < 1e-9:
        positive_sum = abs(team_net) + 1e-9

    top1_share = abs(top1) / positive_sum
    top2_share = abs(top2) / positive_sum

    result["star_top1_impact"] = round(top1 * team_scale, 4)
    result["star_top2_impact"] = round(top2 * team_scale, 4)
    result["star_top3_impact"] = round(top3 * team_scale, 4)
    result["star_top1_share"] = round(top1_share, 4)
    result["star_top2_share"] = round(top2_share, 4)

    # Concentration penalty (applied in forecast mode)
    penalty = 0.0
    if forecast_mode and ENABLE_FORECAST_STAR_CONCENTRATION:
        # Penalty for top-1 share exceeding threshold
        if top1_share > FORECAST_STAR_TOP1_SHARE_THRESHOLD:
            excess = top1_share - FORECAST_STAR_TOP1_SHARE_THRESHOLD
            penalty += excess * FORECAST_STAR_CONCENTRATION_PENALTY_SLOPE

        # Additional penalty for top-2 share exceeding threshold
        if top2_share > FORECAST_STAR_TOP2_SHARE_THRESHOLD:
            excess = top2_share - FORECAST_STAR_TOP2_SHARE_THRESHOLD
            penalty += excess * FORECAST_STAR_CONCENTRATION_PENALTY_SLOPE * 0.5

        penalty = min(penalty, FORECAST_STAR_CONCENTRATION_PENALTY_MAX)

    result["star_concentration_penalty"] = round(penalty, 4)
    return result

def _get_primary_archetype(off_probs: Dict[str, float]) -> str:
    """Return the archetype with highest probability."""
    if not off_probs:
        return "connector"
    return max(off_probs, key=lambda k: off_probs.get(k, 0.0))


def compute_offensive_mean(
    players: pd.DataFrame,
    league_avg_tov: float,
    league_avg_ftr: float,
    league_avg_transition_freq: float,
    league_avg_transition_ppp: float,
    archetype_percentiles: Dict[str, Dict[str, float]],
    use_interaction: bool = True,
    use_structure: bool = True,
    spacing_3pa_threshold: Optional[float] = None,
    spacing_efg_threshold: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute offensive mean model for one team-season.

    players: DataFrame with columns:
        minute_share, impact_obke, off_primary_archetype,
        behavioral_turnover_rate, behavioral_free_throw_rate,
        behavioral_assist_rate, behavioral_three_point_rate,
        behavioral_efg, mpg, off_prob_*, playtype_8 (transition)
    """
    result = {}

    # 2.1 TALENT BASE
    talent_base = (players["minute_share"] * players["impact_obke"]).sum()
    result["off_talent_base"] = talent_base

    # 2.2 INTERACTION TERM
    interaction_term = 0.0
    interaction_details = []
    if use_interaction and len(players) > 1:
        n = len(players)
        for i in range(n):
            for j in range(i + 1, n):
                pi = players.iloc[i]
                pj = players.iloc[j]
                arch_i = pi["off_primary_archetype"]
                arch_j = pj["off_primary_archetype"]
                # Scale pair weight to avoid vanishing interaction magnitudes.
                # Baseline 0.2 corresponds to two rotation players (~20% each).
                pair_weight = (pi["minute_share"] * pj["minute_share"]) / 0.2

                base_val, cond_flag = _get_interaction_value(arch_i, arch_j)

                # Apply conditional logic
                if cond_flag == "both_low":
                    pctls_i = archetype_percentiles.get(arch_i, {})
                    pctls_j = archetype_percentiles.get(arch_j, {})
                    p25_i = pctls_i.get("p25", 0.0)
                    p25_j = pctls_j.get("p25", 0.0)
                    if pi["impact_obke"] >= p25_i or pj["impact_obke"] >= p25_j:
                        base_val = 0.0  # Only penalize if BOTH are below p25
                elif cond_flag == "j_low":
                    pctls_j = archetype_percentiles.get(arch_j, {})
                    p25_j = pctls_j.get("p25", 0.0)
                    if pj["impact_obke"] >= p25_j:
                        base_val = 0.0  # Only penalize if j is below threshold

                contribution = pair_weight * base_val
                if abs(contribution) > 1e-6:
                    interaction_details.append({
                        "player_i": pi.get("player_name", ""),
                        "player_j": pj.get("player_name", ""),
                        "arch_i": arch_i,
                        "arch_j": arch_j,
                        "base_val": base_val,
                        "pair_weight": round(pair_weight, 4),
                        "contribution": round(contribution, 4),
                    })
                interaction_term += contribution

        interaction_term = np.clip(
            INTERACTION_LAMBDA * interaction_term,
            -INTERACTION_CAP, INTERACTION_CAP
        )

    result["off_interaction_raw"] = interaction_term / INTERACTION_LAMBDA if INTERACTION_LAMBDA else 0.0
    result["off_interaction_term"] = interaction_term
    result["off_interaction_details"] = interaction_details

    # 2.3 STRUCTURE TERM
    structure_term = 0.0
    structure_details = {}

    if use_structure:
        # 2.3.1 Turnover Control
        team_tov = (players["minute_share"] * players["behavioral_turnover_rate"]).sum()
        tov_penalty = -BETA_TOV * (team_tov - league_avg_tov)
        structure_details["team_tov_rate"] = round(team_tov, 4)
        structure_details["tov_penalty"] = round(tov_penalty, 4)
        structure_term += tov_penalty

        # 2.3.2 Free Throw Rate
        team_ftr = (players["minute_share"] * players["behavioral_free_throw_rate"]).sum()
        ftr_bonus = BETA_FTR * (team_ftr - league_avg_ftr)
        structure_details["team_ftr"] = round(team_ftr, 4)
        structure_details["ftr_bonus"] = round(ftr_bonus, 4)
        structure_term += ftr_bonus

        # 2.3.3 Playmaking Diversity
        playmakers = players[
            (players["behavioral_assist_rate"] > PLAYMAKING_AST_THRESHOLD) &
            (players["mpg"] > PLAYMAKING_MPG_THRESHOLD)
        ]
        n_playmakers = len(playmakers)
        if n_playmakers == 1:
            playmaking_adj = PLAYMAKING_SOLO_PENALTY
        elif n_playmakers >= 3:
            playmaking_adj = PLAYMAKING_DEEP_BONUS
        else:
            playmaking_adj = 0.0
        structure_details["n_playmakers"] = n_playmakers
        structure_details["playmaking_adj"] = playmaking_adj
        structure_term += playmaking_adj

        # 2.3.4 Spacing Credibility — season-relative when override provided
        # Fixed-value fallback preserves historical behavior; passing season percentiles
        # auto-adjusts to league 3PT evolution year-over-year.
        _3pa_thresh = spacing_3pa_threshold if spacing_3pa_threshold is not None else SPACING_3PA_RATE_THRESHOLD
        _efg_thresh = spacing_efg_threshold if spacing_efg_threshold is not None else max(SPACING_3P_PCT_THRESHOLD, 0.52)
        shooters = players[
            (players["behavioral_three_point_rate"] > _3pa_thresh) &
            (players["behavioral_efg"] > _efg_thresh) &
            (players["mpg"] > SPACING_MPG_THRESHOLD) &
            (players["minute_share"] > 0.08)
        ]
        n_shooters = len(shooters)
        if n_shooters < 2:
            spacing_adj = SPACING_POOR_PENALTY
        elif n_shooters >= 4:
            spacing_adj = SPACING_ELITE_BONUS
        else:
            spacing_adj = 0.0
        structure_details["n_shooters"] = n_shooters
        structure_details["spacing_adj"] = spacing_adj
        structure_term += spacing_adj

        # 2.3.5 Transition as Structure Term (NEW — success-weighted)
        # Transition frequency from playtype vector (index 8 = TRANSITION_POSS_PCT)
        team_transition_freq = (players["minute_share"] * players["playtype_8"]).sum()
        # Estimate transition success using transition-heavy players' eFG as proxy
        trans_players = players[players["playtype_8"] > 0.05]
        if len(trans_players) > 0:
            team_transition_success = (
                trans_players["minute_share"] * trans_players["behavioral_efg"] * trans_players["playtype_8"]
            ).sum() / max(
                (trans_players["minute_share"] * trans_players["playtype_8"]).sum(), 1e-6
            )
        else:
            team_transition_success = league_avg_transition_ppp / 2.0
        # Reward teams with BOTH high transition frequency AND good transition success
        transition_bonus = BETA_TRANSITION * (team_transition_freq - league_avg_transition_freq) * (
            team_transition_success / max(league_avg_transition_ppp / 2.0, 0.3)
        )
        structure_details["team_transition_freq"] = round(team_transition_freq, 4)
        structure_details["team_transition_success"] = round(team_transition_success, 4)
        structure_details["transition_bonus"] = round(transition_bonus, 4)
        structure_term += transition_bonus

        structure_term = np.clip(structure_term, -STRUCTURE_CAP, STRUCTURE_CAP)

    result["off_structure_term"] = structure_term
    result["off_structure_details"] = structure_details

    # FINAL OFFENSIVE MEAN
    result["off_mean"] = talent_base + interaction_term + structure_term
    return result


def compute_defensive_mean(
    players: pd.DataFrame,
    use_defense: bool = True,
) -> Dict[str, float]:
    """Compute defensive mean model for one team-season."""
    result = {}

    # 3.1 Defensive Talent Base
    def_talent = (players["minute_share"] * players["impact_dbke"]).sum()
    result["def_talent_base"] = def_talent

    if not use_defense:
        result["def_mean"] = def_talent
        return result

    adj = 0.0
    defense_details = {}

    # 3.2 Essentials — Rim Presence
    #   "Rim Protector", "Dropping Big", and "Mobile Big" all provide paint protection.
    #   Penalize only teams with NO big archetype in significant minutes.
    _rim_labels = ["rim protector", "dropping big", "mobile big"]
    rim_protectors = players[
        (players["def_primary_archetype"].str.lower().isin(_rim_labels)) &
        (players["mpg"] > RP_MPG_THRESHOLD)
    ]
    has_rp = len(rim_protectors) > 0
    if not has_rp:
        adj += RP_MISSING_PENALTY
        defense_details["rim_protector_penalty"] = RP_MISSING_PENALTY
    else:
        defense_details["rim_protector_penalty"] = 0.0

    # POA Defender
    poa_defenders = players[
        (players["def_primary_archetype"].str.lower().str.contains("poa", na=False)) &
        (players["mpg"] > POA_MPG_THRESHOLD)
    ]
    has_poa = len(poa_defenders) > 0
    if not has_poa:
        adj += POA_MISSING_PENALTY
        defense_details["poa_defender_penalty"] = POA_MISSING_PENALTY
    else:
        defense_details["poa_defender_penalty"] = 0.0

    # Both missing
    if not has_rp and not has_poa:
        adj += BOTH_MISSING_PENALTY
        defense_details["both_missing_penalty"] = BOTH_MISSING_PENALTY
    else:
        defense_details["both_missing_penalty"] = 0.0

    # 3.3 Anchor Quality Scaling
    anchor_bonus = 0.0
    if has_rp:
        top_rp_dbke = rim_protectors["impact_dbke"].max()
        anchor_bonus += ANCHOR_RP_WEIGHT * top_rp_dbke
        defense_details["top_rp_dbke"] = round(top_rp_dbke, 3)
    if has_poa:
        top_poa_dbke = poa_defenders["impact_dbke"].max()
        anchor_bonus += ANCHOR_POA_WEIGHT * top_poa_dbke
        defense_details["top_poa_dbke"] = round(top_poa_dbke, 3)
    adj += anchor_bonus
    defense_details["anchor_bonus"] = round(anchor_bonus, 3)

    # 3.4 Diversity Bonus
    sig_defenders = players[players["mpg"] > DIVERSITY_MPG_THRESHOLD]
    unique_archetypes = sig_defenders["def_primary_archetype"].nunique()
    if unique_archetypes > 3:
        diversity_bonus = min(DIVERSITY_BONUS_PER_ARCH * (unique_archetypes - 3), DIVERSITY_BONUS_CAP)
    else:
        diversity_bonus = 0.0
    adj += diversity_bonus
    defense_details["unique_def_archetypes"] = unique_archetypes
    defense_details["diversity_bonus"] = round(diversity_bonus, 3)

    # 3.5 Liability Penalty
    liabilities = players[
        (players["impact_dbke"] < LIABILITY_DBKE_THRESHOLD) &
        (players["mpg"] > LIABILITY_MPG_THRESHOLD)
    ]
    n_liabilities = len(liabilities)
    liability_penalty = LIABILITY_PER_PLAYER * n_liabilities
    if n_liabilities >= 2:
        liability_penalty += LIABILITY_STACKING_PENALTY
    adj += liability_penalty
    defense_details["n_liabilities"] = n_liabilities
    defense_details["liability_penalty"] = round(liability_penalty, 3)

    adj = np.clip(adj, -DEFENSE_CAP, DEFENSE_CAP)
    result["def_adjustments"] = adj
    result["def_details"] = defense_details
    result["def_mean"] = def_talent + adj
    return result


def compute_volatility(
    players: pd.DataFrame,
    league_avg_3pa_rate: float,
    league_avg_transition_freq: float,
    league_std_net: float,
    use_volatility: bool = True,
) -> Dict[str, float]:
    """Compute team volatility model (separate from mean)."""
    result = {}

    sigma_base = league_std_net
    result["vol_base"] = sigma_base

    # Dynamic volatility bounds — range allows meaningful team differentiation
    # Target: vol_total should span ~1-4 points across teams
    vol_floor_used = max(VOL_FLOOR, 0.3 * league_std_net)
    vol_ceiling_used = max(vol_floor_used + 1.0, min(VOL_CEILING, 4.0 * league_std_net))
    result["vol_floor_used"] = round(vol_floor_used, 4)
    result["vol_ceiling_used"] = round(vol_ceiling_used, 4)

    if not use_volatility:
        result["vol_total"] = np.clip(sigma_base, vol_floor_used, vol_ceiling_used)
        return result

    # 5.2 3PT Frequency Volatility — heavy three-point teams are more volatile
    team_3pa_rate = (players["minute_share"] * players["behavioral_three_point_rate"]).sum()
    sigma_3pa = ALPHA_3PA * (team_3pa_rate - league_avg_3pa_rate)
    result["vol_3pa"] = round(sigma_3pa, 4)
    result["team_3pa_rate"] = round(team_3pa_rate, 4)

    # 5.3 Creation Concentration — star-dependent teams are more volatile
    max_usage = players["behavioral_usage"].max()
    if max_usage > CREATION_CONCENTRATION_THRESHOLD:
        sigma_creation = ALPHA_CREATION * (max_usage - CREATION_CONCENTRATION_THRESHOLD)
    else:
        sigma_creation = 0.0
    result["vol_creation"] = round(sigma_creation, 4)
    result["max_usage"] = round(max_usage, 4)

    # 5.4 Transition Frequency — transition-heavy teams are more volatile
    team_transition_freq = (players["minute_share"] * players["playtype_8"]).sum()
    sigma_transition = ALPHA_TRANSITION * (team_transition_freq - league_avg_transition_freq)
    result["vol_transition"] = round(sigma_transition, 4)
    result["team_transition_freq"] = round(team_transition_freq, 4)

    # 5.5 Roster depth volatility — thin rotations are more volatile
    sig_players = players[players["minute_share"] > 0.08]
    n_sig = len(sig_players)
    if n_sig <= 6:
        sigma_depth = 0.3 * (7 - n_sig)  # +0.3 per missing rotation player below 7
    elif n_sig >= 10:
        sigma_depth = -0.2  # deep teams are slightly less volatile
    else:
        sigma_depth = 0.0
    result["vol_depth"] = round(sigma_depth, 4)
    result["n_sig_players"] = n_sig

    # 5.6 Final Volatility
    sigma_team = sigma_base + sigma_3pa + sigma_creation + sigma_transition + sigma_depth
    sigma_team = np.clip(sigma_team, vol_floor_used, vol_ceiling_used)
    result["vol_total"] = round(sigma_team, 4)

    return result


def _compute_defense_sign(
    profiles: pd.DataFrame,
    actual_team_data: pd.DataFrame,
) -> Dict[str, float]:
    """
    Infer defensive sign convention from data.

    Returns:
      {
        "defense_sign": +1 or -1,
        "corr_def_talent_vs_actual": ...,
        "corr_plus": corr(off + def, actual),
        "corr_minus": corr(off - def, actual),
      }
    """
    out = {
        "defense_sign": 1.0,
        "corr_def_talent_vs_actual": np.nan,
        "corr_plus": np.nan,
        "corr_minus": np.nan,
    }

    if actual_team_data.empty:
        return out

    team_key = ["season", "team_abbreviation"]
    tmp = profiles[["season", "team_abbreviation", "minute_share", "impact_obke", "impact_dbke"]].copy()
    tmp["w_obke"] = tmp["minute_share"] * tmp["impact_obke"]
    tmp["w_dbke"] = tmp["minute_share"] * tmp["impact_dbke"]
    talent = tmp.groupby(team_key, as_index=False).agg(
        off_talent_base=("w_obke", "sum"),
        def_talent_base=("w_dbke", "sum"),
    )

    td = actual_team_data[["season", "team_abbreviation", "actual_net_rating"]].copy()
    td["season"] = td["season"].astype(str)
    td["team_abbreviation"] = td["team_abbreviation"].astype(str).str.upper()

    merged = talent.merge(td, on=["season", "team_abbreviation"], how="inner")
    if len(merged) < 5:
        return out

    corr_def = merged["def_talent_base"].corr(merged["actual_net_rating"])
    corr_plus = (merged["off_talent_base"] + merged["def_talent_base"]).corr(merged["actual_net_rating"])
    corr_minus = (merged["off_talent_base"] - merged["def_talent_base"]).corr(merged["actual_net_rating"])

    out["corr_def_talent_vs_actual"] = float(corr_def)
    out["corr_plus"] = float(corr_plus)
    out["corr_minus"] = float(corr_minus)
    out["defense_sign"] = 1.0 if (corr_plus >= corr_minus) else -1.0
    return out


def _linear_fit(x: pd.Series, y: pd.Series) -> Tuple[float, float]:
    """Fit y = a + b*x with robust fallback."""
    xv = pd.to_numeric(x, errors="coerce")
    yv = pd.to_numeric(y, errors="coerce")
    mask = xv.notna() & yv.notna()
    xv = xv[mask]
    yv = yv[mask]
    if len(xv) < 2 or float(xv.std()) == 0.0:
        return 0.0, 1.0
    b = float(np.cov(xv, yv, ddof=0)[0, 1] / np.var(xv, ddof=0))
    a = float(yv.mean() - b * xv.mean())
    return a, b


def _metrics(pred: pd.Series, actual: pd.Series) -> Dict[str, float]:
    """Correlation + error + spread metrics."""
    p = pd.to_numeric(pred, errors="coerce")
    a = pd.to_numeric(actual, errors="coerce")
    mask = p.notna() & a.notna()
    p = p[mask]
    a = a[mask]
    if len(p) == 0:
        return {
            "n": 0,
            "r": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
            "std_pred": np.nan,
            "std_actual": np.nan,
        }
    residual = p - a
    return {
        "n": int(len(p)),
        "r": float(p.corr(a)),
        "mae": float(residual.abs().mean()),
        "rmse": float(np.sqrt((residual ** 2).mean())),
        "std_pred": float(p.std()),
        "std_actual": float(a.std()),
    }


def _ablation_report(
    result_df: pd.DataFrame,
    actual_team_data: pd.DataFrame,
    defense_sign: float,
    team_scale: float = 5.0,
    holdout_season: str = "2024-25",
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float], pd.DataFrame]:
    """
    Run requested ablations and calibration diagnostics.

    Returns:
      ablation_report, calibration_params, merged_frame_with_predictions
    """
    td = actual_team_data[["season", "team_abbreviation", "actual_net_rating"]].copy()
    td["season"] = td["season"].astype(str)
    td["team_abbreviation"] = td["team_abbreviation"].astype(str).str.upper()

    merged = result_df.merge(td, on=["season", "team_abbreviation"], how="inner")
    if merged.empty:
        return {}, {"intercept": 0.0, "slope": 1.0, "holdout_season": holdout_season}, merged

    merged["ab_talent_only_scaled"] = team_scale * merged["off_talent_base"]
    merged["ab_talent_plus_defense"] = (
        team_scale * merged["off_talent_base"]
        + defense_sign * team_scale * merged["def_talent_base"]
    )
    merged["ab_talent_defense_structure"] = (
        team_scale * merged["off_talent_base"]
        + defense_sign * team_scale * merged["def_talent_base"]
        + merged["off_structure_term"]
        + defense_sign * merged["def_adjustments"]
    )
    merged["ab_full_raw"] = (
        team_scale * merged["off_talent_base"]
        + defense_sign * team_scale * merged["def_talent_base"]
        + merged["off_interaction_term"]
        + merged["off_structure_term"]
        + defense_sign * merged["def_adjustments"]
    )

    train_mask = merged["season"].astype(str) != str(holdout_season)
    test_mask = merged["season"].astype(str) == str(holdout_season)

    # Calibration on talent-only scaled (requested step 2)
    a_t, b_t = _linear_fit(
        merged.loc[train_mask, "ab_talent_only_scaled"],
        merged.loc[train_mask, "actual_net_rating"],
    )
    merged["ab_talent_only_scaled_calibrated"] = a_t + b_t * merged["ab_talent_only_scaled"]

    # Calibration on full model (used for final projected output)
    a_f, b_f = _linear_fit(
        merged.loc[train_mask, "ab_full_raw"],
        merged.loc[train_mask, "actual_net_rating"],
    )
    merged["ab_full_calibrated"] = a_f + b_f * merged["ab_full_raw"]

    def _split_metrics(col: str) -> Dict[str, Dict[str, float]]:
        return {
            "all": _metrics(merged[col], merged["actual_net_rating"]),
            "train": _metrics(merged.loc[train_mask, col], merged.loc[train_mask, "actual_net_rating"]),
            "holdout": _metrics(merged.loc[test_mask, col], merged.loc[test_mask, "actual_net_rating"]),
        }

    report = {
        "1_talent_only_scaled": _split_metrics("ab_talent_only_scaled"),
        "2_talent_only_plus_calibration": _split_metrics("ab_talent_only_scaled_calibrated"),
        "3_talent_plus_defense": _split_metrics("ab_talent_plus_defense"),
        "4_talent_plus_defense_plus_structure": _split_metrics("ab_talent_defense_structure"),
        "5_full_model_raw": _split_metrics("ab_full_raw"),
        "6_full_model_calibrated": _split_metrics("ab_full_calibrated"),
    }

    calibration_params = {
        "holdout_season": holdout_season,
        "team_scale": float(team_scale),
        "talent_intercept": float(a_t),
        "talent_slope": float(b_t),
        "full_intercept": float(a_f),
        "full_slope": float(b_f),
    }
    return report, calibration_params, merged


# ═════════════════════════════════════════════════════════════════════
# Main pipeline
# ═════════════════════════════════════════════════════════════════════

def main(
    use_interaction: bool = False,   # Empirically eliminated in 4.6 v2 — see INTERACTION_MATRIX comment
    use_structure: bool = True,
    use_defense: bool = True,
    use_volatility: bool = True,
    forecast_mode: bool = False,
    profiles_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Build team feature aggregation for all team-seasons.

    Args:
        forecast_mode: When True, reads projected profiles, skips calibration
            against actuals, and uses DEFAULT_TEAM_SCALE.
        profiles_path: Override path for input profiles parquet.
        output_path: Override path for output team features parquet.
    """
    mode_label = "FORECAST" if forecast_mode else "BACKTEST"
    print(f"Step 3 — Team Feature Aggregation ({mode_label})")
    print(f"  Flags: interaction={use_interaction}, structure={use_structure}, "
          f"defense={use_defense}, volatility={use_volatility}")

    # ── Load profiles and predictions ──────────────────────────────
    from src.player_eval.constants import PROJECTED_PROFILES_PATH, PROJECTED_TEAM_FEATURES_PATH

    src_path = profiles_path or (PROJECTED_PROFILES_PATH if forecast_mode else PLAYER_PROFILES_PARQUET)
    if not src_path.exists():
        raise FileNotFoundError(f"Profiles not found: {src_path}")

    profiles = load_standardized(src_path)
    print(f"  Loaded {len(profiles)} player-season profiles from {src_path.name}")

    profiles["player_id"] = profiles["player_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    profiles["season"] = profiles["season"].astype(str)
    profiles["team_abbreviation"] = profiles["team_abbreviation"].astype(str).str.upper()

    repl_mask = _replacement_pool_mask(profiles)
    repl_count = int(repl_mask.sum())
    if repl_count:
        profiles = profiles.loc[~repl_mask].copy()
        print(f"  Excluded {repl_count} replacement-pool rows before aggregation")

    # Load minute predictions (only in backtest mode — forecast profiles already have projected MPG)
    if not forecast_mode:
        predictions = pd.DataFrame()
        if MINUTE_PREDICTIONS_PATH.exists():
            predictions = load_standardized(MINUTE_PREDICTIONS_PATH)
            predictions["player_id"] = predictions["player_id"].astype(str).str.replace(r"\.0$", "", regex=True)
            predictions["season"] = predictions["season"].astype(str)

            merge_keys = ["player_id", "season"]
            if "team_abbreviation" in predictions.columns:
                predictions["team_abbreviation"] = predictions["team_abbreviation"].astype(str).str.upper()
                merge_keys.append("team_abbreviation")

            pred_cols = [c for c in predictions.columns if c not in ["player_name"]]
            predictions = predictions[pred_cols].drop_duplicates(subset=merge_keys, keep="first")
            profiles = profiles.merge(predictions, on=merge_keys, how="left")

            pred_mpg_col = None
            for candidate in ["pred_mpg_raw", "pred_mpg_team_norm"]:
                if candidate in profiles.columns:
                    pred_mpg_col = candidate
                    break

            if pred_mpg_col:
                base_mpg = pd.to_numeric(profiles.get("mpg"), errors="coerce")
                pred_mpg = pd.to_numeric(profiles[pred_mpg_col], errors="coerce")
                profiles["mpg"] = pred_mpg.fillna(base_mpg).fillna(0.0)

                games = pd.to_numeric(profiles.get("games"), errors="coerce").fillna(72.0).clip(lower=10.0)
                profiles["minutes"] = profiles["mpg"] * games
                print(f"  Backtest minute source: {pred_mpg_col} (fallback=profile mpg)")
            else:
                profiles["mpg"] = pd.to_numeric(profiles.get("mpg"), errors="coerce").fillna(0.0)
                print("  Backtest minute source: profile mpg (minute predictions missing columns)")
        else:
            profiles["mpg"] = pd.to_numeric(profiles.get("mpg"), errors="coerce").fillna(0.0)
            print("  Backtest minute source: profile mpg (minute predictions file missing)")
    else:
        profiles["mpg"] = pd.to_numeric(profiles.get("mpg"), errors="coerce").fillna(0.0)

    profiles["minutes"] = pd.to_numeric(profiles.get("minutes"), errors="coerce").fillna(0.0)
    profiles["impact_obke"] = pd.to_numeric(profiles["impact_obke"], errors="coerce").fillna(0.0)
    profiles["impact_dbke"] = pd.to_numeric(profiles["impact_dbke"], errors="coerce").fillna(0.0)
    profiles["behavioral_usage"] = pd.to_numeric(profiles["behavioral_usage"], errors="coerce").fillna(0.0)
    profiles["behavioral_turnover_rate"] = pd.to_numeric(profiles["behavioral_turnover_rate"], errors="coerce").fillna(0.0)
    profiles["behavioral_free_throw_rate"] = pd.to_numeric(profiles["behavioral_free_throw_rate"], errors="coerce").fillna(0.0)
    profiles["behavioral_assist_rate"] = pd.to_numeric(profiles["behavioral_assist_rate"], errors="coerce").fillna(0.0)
    profiles["behavioral_three_point_rate"] = pd.to_numeric(profiles["behavioral_three_point_rate"], errors="coerce").fillna(0.0)
    profiles["behavioral_efg"] = pd.to_numeric(profiles["behavioral_efg"], errors="coerce").fillna(0.0)

    # Playtype columns
    for i in range(11):
        col = f"playtype_{i}"
        if col in profiles.columns:
            profiles[col] = pd.to_numeric(profiles[col], errors="coerce").fillna(0.0)
        else:
            profiles[col] = 0.0

    # ── Compute minute shares per team-season ──────────────────────
    team_key = ["season", "team_abbreviation"]
    team_total_min = profiles.groupby(team_key)["minutes"].sum().rename("team_total_minutes").reset_index()
    profiles = profiles.merge(team_total_min, on=team_key, how="left")
    profiles["minute_share"] = profiles["minutes"] / profiles["team_total_minutes"].replace(0, np.nan)
    profiles["minute_share"] = profiles["minute_share"].fillna(0.0)

    # ── Compute league averages ───────────────────────────────────
    # Weight by minutes for league averages
    total_min = profiles["minutes"].sum()
    if total_min > 0:
        league_avg_tov = (profiles["minutes"] * profiles["behavioral_turnover_rate"]).sum() / total_min
        league_avg_ftr = (profiles["minutes"] * profiles["behavioral_free_throw_rate"]).sum() / total_min
        league_avg_3pa = (profiles["minutes"] * profiles["behavioral_three_point_rate"]).sum() / total_min
        league_avg_trans = (profiles["minutes"] * profiles["playtype_8"]).sum() / total_min
        league_avg_trans_ppp = (profiles["minutes"] * profiles["behavioral_efg"]).sum() / total_min
    else:
        league_avg_tov = 0.14
        league_avg_ftr = 0.27
        league_avg_3pa = 0.38
        league_avg_trans = 0.15
        league_avg_trans_ppp = 0.50

    # ── Compute archetype percentiles (league-wide) ───────────────
    archetype_percentiles = {}
    for arch in OFF_ARCHETYPES:
        # Players whose primary archetype is this one
        mask = profiles["off_primary_archetype"].str.lower().str.replace(" ", "_").str.replace("-", "_") == arch.lower()
        if mask.sum() > 0:
            vals = profiles.loc[mask, "impact_obke"]
            archetype_percentiles[arch] = {
                "p25": float(vals.quantile(0.25)),
                "p35": float(vals.quantile(0.35)),
                "p50": float(vals.quantile(0.50)),
            }
        else:
            archetype_percentiles[arch] = {"p25": 0.0, "p35": 0.0, "p50": 0.0}

    # Map primary archetype to canonical key
    arch_map = {}
    for arch in OFF_ARCHETYPES:
        arch_map[arch.lower()] = arch
        # Space-separated version
        arch_map[arch.lower().replace("_", " ")] = arch
        # Capitalize version
        arch_map[arch.replace("_", " ").title().replace(" ", "_").lower()] = arch

    def _map_archetype(name):
        if pd.isna(name):
            return "connector"
        key = str(name).strip().lower().replace(" ", "_").replace("-", "_")
        return arch_map.get(key, key)

    profiles["off_primary_archetype_mapped"] = profiles["off_primary_archetype"].map(_map_archetype)

    # ── League std of team net rating (for volatility base) ───────
    league_std_net = 6.0  # sensible default
    actual_team_data = pd.DataFrame()

    if not forecast_mode:
        # In backtest mode, load actual team data for calibration/validation
        game_logs_path = HISTORICAL_DIR / "team_game_logs.parquet"
        teams_path = HISTORICAL_DIR / "teams.parquet"

        # Build team_id → abbreviation mapping
        team_id_to_abbr = {}
        if teams_path.exists():
            teams_meta = load_standardized(teams_path)
            for _, t in teams_meta.iterrows():
                tid = str(int(float(t["team_id"])))
                team_id_to_abbr[tid] = str(t["abbreviation"]).upper()

        if game_logs_path.exists():
            game_logs = load_standardized(game_logs_path)
            game_logs["team_id"] = game_logs["team_id"].astype(str).str.replace(r"\.0$", "", regex=True)
            game_logs["season"] = game_logs["season"].astype(str)
            game_logs["pts"] = pd.to_numeric(game_logs["pts"], errors="coerce")
            game_logs["OPP_PTS"] = pd.to_numeric(game_logs["OPP_PTS"], errors="coerce")
            game_logs["margin"] = game_logs["pts"] - game_logs["OPP_PTS"]
            team_net = game_logs.groupby(["season", "team_id"]).agg(
                games=("margin", "count"),
                avg_margin=("margin", "mean"),
            ).reset_index()
            team_net["team_abbreviation"] = team_net["team_id"].map(team_id_to_abbr)
            team_net = team_net.rename(columns={"season": "season", "avg_margin": "actual_net_rating"})
            actual_team_data = team_net[["season", "team_abbreviation", "actual_net_rating", "games"]].dropna()
            if len(actual_team_data) > 0:
                league_std_net = float(actual_team_data["actual_net_rating"].std())
                print(f"  League std net rating: {league_std_net:.2f} (from {len(actual_team_data)} team-seasons)")

    # ── Infer defense sign convention from data ──────────────────
    if not forecast_mode:
        defense_sign_info = _compute_defense_sign(profiles, actual_team_data)
        defense_sign = float(defense_sign_info.get("defense_sign", 1.0))
        print(
            "  Defense sign inference: "
            f"sign={defense_sign:+.0f}, "
            f"corr(def_talent,actual)={defense_sign_info.get('corr_def_talent_vs_actual', np.nan):.3f}, "
            f"corr(off+def,actual)={defense_sign_info.get('corr_plus', np.nan):.3f}, "
            f"corr(off-def,actual)={defense_sign_info.get('corr_minus', np.nan):.3f}"
        )
    else:
        defense_sign = 1.0  # Use positive convention for forecast
        print(f"  Forecast mode: defense_sign=+1, league_std_net={league_std_net:.2f}")

    TEAM_SCALE = DEFAULT_TEAM_SCALE

    # ── Data-driven TEAM_SCALE (backtest only) ────────────────────
    if not forecast_mode and not actual_team_data.empty:
        tmp_talent = profiles[["season", "team_abbreviation", "minute_share",
                               "impact_obke", "impact_dbke"]].copy()
        tmp_talent["w_off"] = tmp_talent["minute_share"] * tmp_talent["impact_obke"]
        tmp_talent["w_def"] = tmp_talent["minute_share"] * tmp_talent["impact_dbke"]
        team_talent_agg = tmp_talent.groupby(["season", "team_abbreviation"], as_index=False).agg(
            off_sum=("w_off", "sum"), def_sum=("w_def", "sum"),
        )
        team_talent_agg["combined"] = (
            team_talent_agg["off_sum"] + defense_sign * team_talent_agg["def_sum"]
        )
        td_fit = actual_team_data[["season", "team_abbreviation", "actual_net_rating"]].copy()
        td_fit["season"] = td_fit["season"].astype(str)
        td_fit["team_abbreviation"] = td_fit["team_abbreviation"].astype(str).str.upper()
        fit_df = team_talent_agg.merge(td_fit, on=["season", "team_abbreviation"], how="inner")
        if len(fit_df) >= 10:
            _a, _b = _linear_fit(fit_df["combined"], fit_df["actual_net_rating"])
            fitted_r = fit_df["combined"].corr(fit_df["actual_net_rating"])
            # Only use data-driven scale if the fit has meaningful signal
            if abs(fitted_r) > TEAM_SCALE_FIT_R_THRESHOLD and abs(_b) > TEAM_SCALE_MIN:
                TEAM_SCALE = float(np.clip(_b, TEAM_SCALE_MIN, TEAM_SCALE_MAX))
                print(f"  Data-driven TEAM_SCALE: {TEAM_SCALE:.2f} (r={fitted_r:.4f}, raw_slope={_b:.4f})")
            else:
                print(f"  Talent-actual r too weak ({fitted_r:.4f}, threshold={TEAM_SCALE_FIT_R_THRESHOLD}); using DEFAULT_TEAM_SCALE={DEFAULT_TEAM_SCALE}")
        else:
            print(f"  Insufficient merge ({len(fit_df)} rows); using DEFAULT_TEAM_SCALE={DEFAULT_TEAM_SCALE}")

    # ── Compute per-season spacing thresholds (season-relative percentiles) ──
    # SPACING_3PA_RATE_THRESHOLD (fixed 0.30) and SPACING_3P_PCT_THRESHOLD (fixed 0.35)
    # are replaced per-season by the P40 of 3PA rate and P35 of EFG among players
    # with non-trivial minute share (>0.05). This auto-calibrates as league 3PT volume
    # evolves: a "spacer" stays in roughly the top 60% of 3PA rate every season instead
    # of being defined against a static fixed value.
    season_spacing_thresholds: Dict[str, Dict[str, float]] = {}
    for season_id, season_df in profiles.groupby("season"):
        qualifiers = season_df[season_df.get("minute_share", 0.0).fillna(0.0) > 0.05]
        if len(qualifiers) >= 30:
            p40_3pa = float(qualifiers["behavioral_three_point_rate"].quantile(0.40))
            p35_efg = float(qualifiers["behavioral_efg"].quantile(0.35))
            # Floor at sensible minimums so a weak-shooting season doesn't drop the bar too low
            p40_3pa = max(p40_3pa, 0.20)
            p35_efg = max(p35_efg, 0.48)
        else:
            p40_3pa = SPACING_3PA_RATE_THRESHOLD
            p35_efg = max(SPACING_3P_PCT_THRESHOLD, 0.52)
        season_spacing_thresholds[str(season_id)] = {
            "spacing_3pa_threshold": p40_3pa,
            "spacing_efg_threshold": p35_efg,
        }
        print(f"  Season {season_id}: spacing thresholds 3PA={p40_3pa:.3f}, EFG={p35_efg:.3f} (n_qual={len(qualifiers)})")

    # ── Process each team-season ──────────────────────────────────
    teams = profiles.groupby(team_key)
    team_rows = []

    for (season, team_abbr), team_players in teams:
        if len(team_players) < 3:
            continue

        tp = team_players.copy()
        tp["off_primary_archetype"] = tp["off_primary_archetype_mapped"]

        # ── Offense ──
        season_thresh = season_spacing_thresholds.get(str(season), {})
        off_result = compute_offensive_mean(
            tp, league_avg_tov, league_avg_ftr,
            league_avg_trans, league_avg_trans_ppp,
            archetype_percentiles,
            use_interaction=use_interaction,
            use_structure=use_structure,
            spacing_3pa_threshold=season_thresh.get("spacing_3pa_threshold"),
            spacing_efg_threshold=season_thresh.get("spacing_efg_threshold"),
        )

        # ── Defense ──
        def_result = compute_defensive_mean(tp, use_defense=use_defense)

        # ── Volatility ──
        vol_result = compute_volatility(
            tp, league_avg_3pa, league_avg_trans,
            league_std_net, use_volatility=use_volatility,
        )

        # ── Team Net Rating ──
        # Star concentration penalty (forecast mode only)
        star_result = compute_star_concentration(tp, TEAM_SCALE, forecast_mode=forecast_mode)

        team_net_raw = (
            TEAM_SCALE * off_result["off_talent_base"]
            + defense_sign * TEAM_SCALE * def_result["def_talent_base"]
            + off_result["off_interaction_term"]
            + off_result["off_structure_term"]
            + defense_sign * def_result.get("def_adjustments", 0.0)
            - star_result.get("star_concentration_penalty", 0.0)
        )

        # ── Roster composition summary ──
        n_players = len(tp)
        top_player = tp.loc[tp["impact_obke"].idxmax()] if len(tp) > 0 else None
        off_arch_dist = tp["off_primary_archetype"].value_counts().to_dict()
        def_arch_dist = tp["def_primary_archetype"].value_counts().to_dict()

        row = {
            "season": season,
            "team_abbreviation": team_abbr,
            "n_players": n_players,
            "team_total_minutes": float(tp["team_total_minutes"].iloc[0]) if len(tp) > 0 else 0.0,
            # Offense
            "off_talent_base": round(off_result["off_talent_base"], 4),
            "off_interaction_term": round(off_result["off_interaction_term"], 4),
            "off_structure_term": round(off_result["off_structure_term"], 4),
            "off_mean": round(off_result["off_mean"], 4),
            # Structure details
            "off_tov_penalty": off_result.get("off_structure_details", {}).get("tov_penalty", 0.0),
            "off_ftr_bonus": off_result.get("off_structure_details", {}).get("ftr_bonus", 0.0),
            "off_playmaking_adj": off_result.get("off_structure_details", {}).get("playmaking_adj", 0.0),
            "off_spacing_adj": off_result.get("off_structure_details", {}).get("spacing_adj", 0.0),
            "off_transition_bonus": off_result.get("off_structure_details", {}).get("transition_bonus", 0.0),
            "off_n_playmakers": off_result.get("off_structure_details", {}).get("n_playmakers", 0),
            "off_n_shooters": off_result.get("off_structure_details", {}).get("n_shooters", 0),
            "off_team_transition_freq": off_result.get("off_structure_details", {}).get("team_transition_freq", 0.0),
            "off_team_transition_success": off_result.get("off_structure_details", {}).get("team_transition_success", 0.0),
            # Defense
            "def_talent_base": round(def_result["def_talent_base"], 4),
            "def_adjustments": round(def_result.get("def_adjustments", 0.0), 4),
            "def_mean": round(def_result["def_mean"], 4),
            # Defense details
            "def_rim_protector_penalty": def_result.get("def_details", {}).get("rim_protector_penalty", 0.0),
            "def_poa_defender_penalty": def_result.get("def_details", {}).get("poa_defender_penalty", 0.0),
            "def_both_missing_penalty": def_result.get("def_details", {}).get("both_missing_penalty", 0.0),
            "def_anchor_bonus": def_result.get("def_details", {}).get("anchor_bonus", 0.0),
            "def_diversity_bonus": def_result.get("def_details", {}).get("diversity_bonus", 0.0),
            "def_liability_penalty": def_result.get("def_details", {}).get("liability_penalty", 0.0),
            "def_n_liabilities": def_result.get("def_details", {}).get("n_liabilities", 0),
            "def_unique_archetypes": def_result.get("def_details", {}).get("unique_def_archetypes", 0),
            # Net
            "team_net_rating_raw": round(team_net_raw, 4),
            "team_net_rating_projected": round(team_net_raw, 4),
            # Star concentration
            "star_top1_impact": star_result.get("star_top1_impact", 0.0),
            "star_top2_impact": star_result.get("star_top2_impact", 0.0),
            "star_top3_impact": star_result.get("star_top3_impact", 0.0),
            "star_top1_share": star_result.get("star_top1_share", 0.0),
            "star_top2_share": star_result.get("star_top2_share", 0.0),
            "star_concentration_penalty": star_result.get("star_concentration_penalty", 0.0),
            # Volatility
            "vol_base": round(vol_result["vol_base"], 4),
            "vol_3pa": vol_result.get("vol_3pa", 0.0),
            "vol_creation": vol_result.get("vol_creation", 0.0),
            "vol_transition": vol_result.get("vol_transition", 0.0),
            "vol_depth": vol_result.get("vol_depth", 0.0),
            "n_sig_players": vol_result.get("n_sig_players", 0),
            "vol_floor_used": vol_result.get("vol_floor_used", np.nan),
            "vol_ceiling_used": vol_result.get("vol_ceiling_used", np.nan),
            "vol_total": round(vol_result["vol_total"], 4),
            # Roster composition
            "off_archetype_distribution": json.dumps(off_arch_dist),
            "def_archetype_distribution": json.dumps(def_arch_dist),
            # Interaction details (stored as JSON for frontend)
            "interaction_details": json.dumps(off_result.get("off_interaction_details", [])),
        }

        # Add per-player data summary for frontend
        player_summaries = []
        for _, p in tp.iterrows():
            player_summaries.append({
                "player_id": str(p.get("player_id", "")),
                "player_name": str(p.get("player_name", "")),
                "mpg": round(float(p.get("mpg", 0)), 1),
                "minute_share": round(float(p.get("minute_share", 0)), 4),
                "impact_obke": round(float(p.get("impact_obke", 0)), 3),
                "impact_dbke": round(float(p.get("impact_dbke", 0)), 3),
                "off_archetype": str(p.get("off_primary_archetype", "")),
                "def_archetype": str(p.get("def_primary_archetype", "")),
                "usage": round(float(p.get("behavioral_usage", 0)), 3),
                "ast_rate": round(float(p.get("behavioral_assist_rate", 0)), 3),
                "tov_rate": round(float(p.get("behavioral_turnover_rate", 0)), 3),
                "three_rate": round(float(p.get("behavioral_three_point_rate", 0)), 3),
                "efg": round(float(p.get("behavioral_efg", 0)), 3),
                "transition_freq": round(float(p.get("playtype_8", 0)), 3),
            })
        row["player_summaries"] = json.dumps(player_summaries)

        team_rows.append(row)

    result_df = pd.DataFrame(team_rows)
    print(f"  Computed team features for {len(result_df)} team-seasons")

    # ── Component proportion diagnostics ─────────────────────────
    if len(result_df) > 0:
        off_t_std = (TEAM_SCALE * result_df["off_talent_base"]).std()
        def_t_std = (TEAM_SCALE * result_df["def_talent_base"]).std()
        talent_std = off_t_std + def_t_std
        struct_std = result_df["off_structure_term"].std()
        defadj_std = result_df["def_adjustments"].std()
        inter_std = result_df["off_interaction_term"].std()
        raw_std = result_df["team_net_rating_raw"].std()

        print("  ─── Component Proportion Diagnostics ───")
        print(f"    TEAM_SCALE = {TEAM_SCALE:.2f}")
        print(f"    {TEAM_SCALE:.0f}×off_talent std:   {off_t_std:.4f}  ({off_t_std/raw_std:.0%} of raw)")
        print(f"    {TEAM_SCALE:.0f}×def_talent std:   {def_t_std:.4f}  ({def_t_std/raw_std:.0%} of raw)")
        print(f"    structure_term std:  {struct_std:.4f}  ({struct_std/raw_std:.0%} of raw)")
        print(f"    def_adjustments std: {defadj_std:.4f}  ({defadj_std/raw_std:.0%} of raw)")
        print(f"    interaction std:     {inter_std:.4f}  ({inter_std/raw_std:.0%} of raw)")
        print(f"    team_net_raw std:    {raw_std:.4f}")

        # Modifier proportion check
        modifier_std = struct_std + defadj_std + inter_std
        modifier_ratio = modifier_std / max(talent_std, 1e-6)
        print(f"    modifier_total_std:  {modifier_std:.4f}")
        print(f"    modifier/talent:     {modifier_ratio:.2%} (target <{MODIFIER_MAX_FRACTION:.0%})")
        if modifier_ratio > MODIFIER_MAX_FRACTION:
            print(f"    ⚠️  Modifiers exceed {MODIFIER_MAX_FRACTION:.0%} of talent — consider further reduction")

    # ── Ablations + calibration (backtest only) ────────────────────
    ablations = {}
    calibration_params = {}

    if not forecast_mode:
        ablations, calibration_params, merged_diag = _ablation_report(
            result_df=result_df,
            actual_team_data=actual_team_data,
            defense_sign=defense_sign,
            team_scale=TEAM_SCALE,
            holdout_season="2024-25",
        )

        if not merged_diag.empty:
            holdout_r = ablations.get("6_full_model_calibrated", {}).get("holdout", {}).get("r", 0)
            train_r = ablations.get("6_full_model_calibrated", {}).get("train", {}).get("r", 0)
            cal_slope = calibration_params.get("full_slope", 1.0)

            if holdout_r < 0 or abs(train_r) < 0.20 or abs(cal_slope) < 0.10:
                print(f"  ⚠️  Skipping calibration (holdout_r={holdout_r:.3f}, train_r={train_r:.3f}, "
                      f"slope={cal_slope:.3f}) — using raw predictions with mean centering")
                actual_mean = merged_diag["actual_net_rating"].mean() if "actual_net_rating" in merged_diag else 0.0
                raw_mean = result_df["team_net_rating_raw"].mean()
                result_df["team_net_rating_projected"] = result_df["team_net_rating_raw"] - raw_mean + actual_mean
                calibration_params["calibration_applied"] = False
                calibration_params["calibration_skip_reason"] = (
                    f"holdout_r={holdout_r:.3f}, train_r={train_r:.3f}, slope={cal_slope:.3f}"
                )
            else:
                calibrated = merged_diag[
                    ["season", "team_abbreviation", "ab_full_calibrated"]
                ].rename(columns={"ab_full_calibrated": "team_net_rating_projected_calibrated"})
                result_df = result_df.merge(
                    calibrated,
                    on=["season", "team_abbreviation"],
                    how="left",
                )
                result_df["team_net_rating_projected"] = result_df["team_net_rating_projected_calibrated"].fillna(
                    result_df["team_net_rating_raw"]
                )
                result_df = result_df.drop(columns=["team_net_rating_projected_calibrated"], errors="ignore")
                calibration_params["calibration_applied"] = True

        print("  Ablation summary (r on holdout 2024-25):")
        for key in [
            "1_talent_only_scaled",
            "2_talent_only_plus_calibration",
            "3_talent_plus_defense",
            "4_talent_plus_defense_plus_structure",
            "5_full_model_raw",
            "6_full_model_calibrated",
        ]:
            holdout = ablations.get(key, {}).get("holdout", {})
            print(
                f"    {key}: r={holdout.get('r', np.nan):.4f}, "
                f"std_pred={holdout.get('std_pred', np.nan):.4f}, "
                f"std_actual={holdout.get('std_actual', np.nan):.4f}"
            )

    # ── Validation ───────────────────────────────────────────────
    validation = _validate(
        result_df,
        actual_team_data,
        model_flags={
            "interaction": bool(use_interaction),
            "structure": bool(use_structure),
            "defense": bool(use_defense),
            "volatility": bool(use_volatility),
        },
    )
    if not forecast_mode:
        validation["defense_sign_inference"] = {
            "defense_sign": defense_sign,
            "corr_def_talent_vs_actual": defense_sign_info.get("corr_def_talent_vs_actual", np.nan),
            "corr_off_plus_def_vs_actual": defense_sign_info.get("corr_plus", np.nan),
            "corr_off_minus_def_vs_actual": defense_sign_info.get("corr_minus", np.nan),
        }
    else:
        validation["mode"] = "forecast"
        validation["defense_sign"] = defense_sign
    validation["calibration"] = calibration_params
    validation["ablation_stack"] = ablations

    # ── Save outputs ──────────────────────────────────────────────
    dst_path = output_path or (PROJECTED_TEAM_FEATURES_PATH if forecast_mode else TEAM_FEATURES_PATH)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    save_standardized(result_df, dst_path)

    report_path = STEP3_VALIDATION_REPORT
    report_path.parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )

    print(f"  Saved: {dst_path}")
    print(f"  Saved: {report_path}")

    return result_df


def _validate(
    result_df: pd.DataFrame,
    actual_team_data: pd.DataFrame,
    model_flags: Optional[Dict[str, bool]] = None,
) -> dict:
    """Validate team projections against actual data."""
    validation = {
        "step": "Step 3 — Team Feature Aggregation",
        "n_team_seasons": len(result_df),
        "seasons": sorted(result_df["season"].unique().tolist()),
        "summary": {},
        "model_flags": model_flags or {
            "interaction": True,
            "structure": True,
            "defense": True,
            "volatility": True,
        },
    }

    # Summary statistics
    for col in ["off_talent_base", "off_interaction_term", "off_structure_term",
                 "off_mean", "def_talent_base", "def_adjustments", "def_mean",
                 "team_net_rating_projected", "vol_total"]:
        if col in result_df.columns:
            vals = result_df[col]
            validation["summary"][col] = {
                "mean": round(float(vals.mean()), 4),
                "std": round(float(vals.std()), 4),
                "min": round(float(vals.min()), 4),
                "max": round(float(vals.max()), 4),
                "median": round(float(vals.median()), 4),
            }

    # Structural detail summaries
    validation["structure_details"] = {
        "avg_n_playmakers": round(float(result_df.get("off_n_playmakers", pd.Series([0])).mean()), 2),
        "avg_n_shooters": round(float(result_df.get("off_n_shooters", pd.Series([0])).mean()), 2),
        "avg_transition_freq": round(float(result_df.get("off_team_transition_freq", pd.Series([0])).mean()), 4),
        "avg_transition_success": round(float(result_df.get("off_team_transition_success", pd.Series([0])).mean()), 4),
        "pct_teams_no_rp": round(float((result_df.get("def_rim_protector_penalty", pd.Series([0])) < 0).mean()), 3),
        "pct_teams_no_poa": round(float((result_df.get("def_poa_defender_penalty", pd.Series([0])) < 0).mean()), 3),
        "avg_def_liabilities": round(float(result_df.get("def_n_liabilities", pd.Series([0])).mean()), 2),
    }

    # ── Compare with actual team data ─────────────────────────────
    if not actual_team_data.empty:
        # actual_team_data already has: season, team_abbreviation, actual_net_rating
        td = actual_team_data.copy()
        td["season"] = td["season"].astype(str)
        td["team_abbreviation"] = td["team_abbreviation"].astype(str).str.upper()

        merged = result_df.merge(
            td[["season", "team_abbreviation", "actual_net_rating"]],
            on=["season", "team_abbreviation"], how="inner"
        )

        if len(merged) > 0:
            merged["actual_net_rating"] = pd.to_numeric(merged["actual_net_rating"], errors="coerce")
            merged = merged.dropna(subset=["actual_net_rating"])

            if len(merged) > 0:
                residuals = merged["team_net_rating_projected"] - merged["actual_net_rating"]
                validation["vs_actual"] = {
                    "n_matched": len(merged),
                    "correlation": round(float(
                        merged["team_net_rating_projected"].corr(merged["actual_net_rating"])
                    ), 4),
                    "mae": round(float(residuals.abs().mean()), 4),
                    "rmse": round(float(np.sqrt((residuals ** 2).mean())), 4),
                    "mean_residual": round(float(residuals.mean()), 4),
                    "std_residual": round(float(residuals.std()), 4),
                }

                # Rank correlation
                from scipy.stats import spearmanr
from src.data.schema_contract import load_standardized, save_standardized
                spearman_r, spearman_p = spearmanr(
                    merged["team_net_rating_projected"],
                    merged["actual_net_rating"]
                )
                validation["vs_actual"]["spearman_r"] = round(float(spearman_r), 4)
                validation["vs_actual"]["spearman_p"] = round(float(spearman_p), 6)

                # Also compute wins correlation using team_summaries if available
                ts_path = HISTORICAL_DIR / "team_summaries.parquet"
                if ts_path.exists():
                    ts = load_standardized(ts_path)
                    if "WINS" in ts.columns and "team_id" in ts.columns:
                        ts["team_id"] = ts["team_id"].astype(str).str.replace(r"\.0$", "", regex=True)
                        ts["season"] = ts["season"].astype(str)
                        # Map team_id to abbreviation
                        ts["team_abbreviation"] = ts["team_id"].map(
                            {str(int(float(k))): v for k, v in
                             (load_standardized(HISTORICAL_DIR / "teams.parquet")[["team_id", "abbreviation"]]
                              .itertuples(index=False))} if (HISTORICAL_DIR / "teams.parquet").exists() else {}
                        )
                        ts = ts.rename(columns={"season": "season"})
                        merged_w = result_df.merge(
                            ts[["season", "team_abbreviation", "WINS"]].dropna(),
                            on=["season", "team_abbreviation"], how="inner"
                        )
                        if len(merged_w) > 0:
                            merged_w["WINS"] = pd.to_numeric(merged_w["WINS"], errors="coerce")
                            validation["vs_actual_wins"] = {
                                "n_matched": len(merged_w),
                                "net_rating_wins_corr": round(float(
                                    merged_w["team_net_rating_projected"].corr(merged_w["WINS"])
                                ), 4),
                            }

    return validation


if __name__ == "__main__":
    main()
