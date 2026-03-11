"""
src/simulation/player_stats_sim.py
=============================================================================
Detailed player stat simulation for the simulation core.

This module provides three layers that sit underneath season_sim.py:
  1. Lineup-context normalization (starter, rotation, clutch marginal bonuses)
  2. Archetype-based matchup adjustments for main rotation players
  3. Detailed single-game and sample-season player box-score simulation

The detailed season path is intentionally lightweight: season_sim.py still owns
large-scale Monte Carlo win distributions, while this module produces one
deterministic sample season per run for realistic player counting stats and
single-game inspection.
=============================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.simulation.simulation_config import (
    CLUTCH_MARGIN_TRIGGER,
    LINEUP_CLUTCH_BONUS_SCALE,
    LINEUP_CONTINUITY_BONUS_SCALE,
    LINEUP_FIT_BONUS_SCALE,
    LINEUP_ROTATION_BONUS_SCALE,
    LINEUP_STARTER_BONUS_SCALE,
    LINEUP_TEAM_BONUS_CLIP,
    MATCHUP_TEAM_INTERACTION_SCALE,
    MATCHUP_SPREAD_BONUS_CLIP,
    PLAYER_GAME_MIN_ACTIVE,
    PLAYER_GAME_MINUTES_BENCH_PENALTY,
    PLAYER_GAME_MINUTES_CLUTCH_BONUS,
    PLAYER_GAME_MINUTES_STARTER_BONUS,
    PLAYER_GAME_ROTATION_SIZE,
    PLAYER_GAME_TOTAL_MINUTES,
    PROFILE_AGGREGATE_PATH,
)


CREATOR_ARCHETYPES = {
    "Ball Dominant Creator",
    "Ballhandler",
    "All-Around Scorer",
    "Perimeter Scorer",
}
SPACER_ARCHETYPES = {
    "Off-Ball Movement Shooter",
    "Off-Ball Stationary Shooter",
    "Perimeter Scorer",
    "PnR Popping Big",
}
RIM_ARCHETYPES = {
    "Interior Scorer",
    "Off-Ball Finisher",
    "PnR Rolling Big",
}
POA_DEF_ARCHETYPES = {"POA Defender", "Wing Stopper", "Off-Ball Chaser"}
RIM_DEF_ARCHETYPES = {"Rim Protector", "Dropping Big", "Mobile Big"}
VERSATILE_DEF_ARCHETYPES = {"Versatile Defender", "Wing Stopper", "Off-Ball Chaser"}
LOW_ACTIVITY_DEF_ARCHETYPES = {"Low-Activity Defender", "Rotational Defender", "Unknown", "Insufficient Minutes"}


def _norm_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True)


def _safe_series(df: pd.DataFrame, names: Iterable[str], default: float = 0.0) -> pd.Series:
    for name in names:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(default, index=df.index, dtype=float)


def _safe_text_series(df: pd.DataFrame, names: Iterable[str], default: str = "") -> pd.Series:
    for name in names:
        if name in df.columns:
            return df[name].fillna(default).astype(str)
    return pd.Series(default, index=df.index, dtype=object)


def _season_start_year(season: str) -> int:
    return int(str(season)[:4])


def _role_from_text(text: str) -> str:
    value = str(text or "").strip().lower().replace("_", "-").replace("/", "-")
    if value in {"guard-forward", "forward-guard", "gf", "fg", "g-f", "f-g"}:
        return "Wing"
    if value in {"forward-center", "center-forward", "fc", "cf", "f-c", "c-f"}:
        return "Big"
    if "center" in value or value == "c":
        return "Big"
    if "forward" in value or value in {"wing", "f"}:
        return "Wing"
    if "guard" in value or value == "g":
        return "Guard"
    return "Wing"


def _clip(value: float, lo: float, hi: float) -> float:
    return float(np.clip(value, lo, hi))


def _weighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    weight_sum = float(weights.sum())
    if weight_sum <= 1e-9:
        return float(np.mean(values))
    return float(np.dot(values, weights) / weight_sum)


def build_lineup_bonus_map(lineup_rows: List[Dict]) -> Dict[str, Dict[str, Dict[str, float]]]:
    if not lineup_rows:
        return {}

    frame = pd.DataFrame(lineup_rows)
    if frame.empty:
        return {}

    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for season, season_df in frame.groupby("season", sort=True):
        working = season_df.copy()
        for col in [
            "starter_delta_impact",
            "rotation_delta_impact",
            "continuity_score",
            "starter_fit_score",
            "clutch_delta_impact",
            "clutch_fit_score",
        ]:
            series = pd.to_numeric(working.get(col), errors="coerce").fillna(0.0)
            std = float(series.std(ddof=0))
            if std <= 1e-9:
                working[f"{col}_z"] = 0.0
            else:
                working[f"{col}_z"] = (series - float(series.mean())) / std

        out[str(season)] = {}
        for _, row in working.iterrows():
            starter_component = LINEUP_STARTER_BONUS_SCALE * float(row.get("starter_delta_impact_z", 0.0))
            rotation_component = LINEUP_ROTATION_BONUS_SCALE * float(row.get("rotation_delta_impact_z", 0.0))
            continuity_component = LINEUP_CONTINUITY_BONUS_SCALE * float(row.get("continuity_score_z", 0.0))
            fit_component = LINEUP_FIT_BONUS_SCALE * float(row.get("starter_fit_score_z", 0.0))
            clutch_component = (
                LINEUP_CLUTCH_BONUS_SCALE * float(row.get("clutch_delta_impact_z", 0.0))
                + 0.25 * float(row.get("clutch_fit_score_z", 0.0))
            )
            out[str(season)][str(row["team_abbreviation"]).upper()] = {
                "team_bonus": _clip(
                    starter_component + rotation_component + continuity_component + fit_component,
                    -LINEUP_TEAM_BONUS_CLIP,
                    LINEUP_TEAM_BONUS_CLIP,
                ),
                "clutch_bonus": _clip(clutch_component, -0.9, 0.9),
                "continuity_score": float(pd.to_numeric(pd.Series([row.get("continuity_score")]), errors="coerce").fillna(0.0).iloc[0]),
                "starter_fit_score": float(pd.to_numeric(pd.Series([row.get("starter_fit_score")]), errors="coerce").fillna(0.0).iloc[0]),
                "starter_ids": [str(x) for x in row.get("starter_player_ids", [])],
                "clutch_ids": [str(x) for x in row.get("clutch_player_ids", [])],
            }
    return out


def _prepare_historical_stat_templates() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not PROFILE_AGGREGATE_PATH.exists():
        empty = pd.DataFrame()
        return empty, empty, empty

    aggregate = pd.read_parquet(PROFILE_AGGREGATE_PATH)
    aggregate["player_id"] = _norm_id(_safe_text_series(aggregate, ["player_id"]))
    aggregate["season"] = _safe_text_series(aggregate, ["season"])
    aggregate["role"] = _safe_text_series(aggregate, ["primary_position_estimate", "position_proxy"]).map(_role_from_text)
    aggregate["off_archetype"] = _safe_text_series(aggregate, ["primary_archetype", "off_primary_archetype"], default="Unknown")

    minutes_total = _safe_series(aggregate, ["minutes", "total_minutes"], default=np.nan)
    mpg = _safe_series(aggregate, ["mpg"], default=0.0).fillna(0.0)
    games = _safe_series(aggregate, ["games", "gp", "g"], default=82.0).fillna(82.0).replace(0.0, 82.0)
    minutes_total = minutes_total.fillna(mpg * games).replace(0.0, np.nan).fillna(np.maximum(mpg, 1.0) * games)

    pts = _safe_series(aggregate, ["box_pts", "pts"])
    ast = _safe_series(aggregate, ["box_ast", "ast"])
    reb = _safe_series(aggregate, ["box_reb", "reb"])
    stl = _safe_series(aggregate, ["box_stl", "stl"])
    blk = _safe_series(aggregate, ["box_blk", "blk"])
    tov = _safe_series(aggregate, ["box_tov", "tov"])
    fgm = _safe_series(aggregate, ["box_fgm", "fgm"])
    fga = _safe_series(aggregate, ["box_fga", "fga"])
    ftm = _safe_series(aggregate, ["box_ftm", "ftm"])
    fta = _safe_series(aggregate, ["box_fta", "fta"])
    fg3m = _safe_series(aggregate, ["box_fg3_m", "fg3_m"], default=0.0)
    fg3a = _safe_series(aggregate, ["box_fg3_a", "fg3_a"], default=0.0)
    oreb = _safe_series(aggregate, ["oreb", "box_oreb"], default=0.0)
    dreb = _safe_series(aggregate, ["dreb", "box_dreb"], default=0.0)
    pf = _safe_series(aggregate, ["pf", "box_pf"], default=2.5)

    template = aggregate[["player_id", "season", "role", "off_archetype"]].copy()
    for name, values in {
        "pts_per36": pts,
        "ast_per36": ast,
        "reb_per36": reb,
        "stl_per36": stl,
        "blk_per36": blk,
        "tov_per36": tov,
        "fga_per36": fga,
        "fta_per36": fta,
        "fg3a_per36": fg3a,
        "oreb_per36": oreb,
        "dreb_per36": dreb,
        "pf_per36": pf,
    }.items():
        template[name] = 36.0 * values / minutes_total.replace(0.0, np.nan)

    fg2a = (fga - fg3a).clip(lower=0.0)
    fg2m = (fgm - fg3m).clip(lower=0.0)
    template["fg2_pct"] = (fg2m / fg2a.replace(0.0, np.nan)).clip(lower=0.30, upper=0.80)
    template["fg3_pct"] = (fg3m / fg3a.replace(0.0, np.nan)).clip(lower=0.18, upper=0.50)
    template["ft_pct"] = (ftm / fta.replace(0.0, np.nan)).clip(lower=0.45, upper=0.95)
    template = template.replace([np.inf, -np.inf], np.nan)

    group_medians = (
        template.groupby(["off_archetype", "role"], as_index=False)
        .median(numeric_only=True)
        .rename(columns={"off_archetype": "group_off_archetype", "role": "group_role"})
    )
    role_medians = (
        template.groupby("role", as_index=False)
        .median(numeric_only=True)
        .rename(columns={"role": "group_role"})
    )
    return template, group_medians, role_medians


def load_player_stat_profiles(player_profiles_path: Path, forecast_mode: bool = False) -> pd.DataFrame:
    profiles = pd.read_parquet(player_profiles_path)
    profiles["player_id"] = _norm_id(_safe_text_series(profiles, ["player_id"]))
    profiles["season"] = _safe_text_series(profiles, ["season"])
    profiles["team_abbreviation"] = _safe_text_series(profiles, ["team_abbreviation"]).str.upper()
    profiles["player_name"] = _safe_text_series(profiles, ["player_name"])
    profiles["off_archetype"] = _safe_text_series(profiles, ["off_primary_archetype"], default="Unknown")
    profiles["def_archetype"] = _safe_text_series(profiles, ["def_primary_archetype"], default="Unknown")
    profiles["role"] = _safe_text_series(profiles, ["position_proxy"], default="Wing").map(_role_from_text)
    profiles["base_mpg"] = _safe_series(profiles, ["projected_mpg", "adjusted_mpg", "mpg"], default=0.0).fillna(0.0)
    profiles["games_target"] = _safe_series(profiles, ["projected_games", "games", "gp"], default=np.nan)
    default_games = pd.Series(
        np.where(profiles["base_mpg"] >= 28.0, 72.0, np.where(profiles["base_mpg"] >= 20.0, 74.0, 76.0)),
        index=profiles.index,
        dtype=float,
    )
    profiles["games_target"] = profiles["games_target"].fillna(default_games)
    profiles["games_target"] = profiles["games_target"].clip(lower=0.0, upper=82.0)
    for col in [
        "impact_total_impact",
        "impact_obke",
        "impact_dbke",
        "behavioral_usage",
        "behavioral_assist_rate",
        "behavioral_turnover_rate",
        "behavioral_three_point_rate",
    ]:
        profiles[col] = _safe_series(profiles, [col], default=0.0).fillna(0.0)

    template, group_medians, role_medians = _prepare_historical_stat_templates()
    merged_frames: List[pd.DataFrame] = []
    needed_cols = [
        "player_id",
        "season",
        "pts_per36",
        "ast_per36",
        "reb_per36",
        "stl_per36",
        "blk_per36",
        "tov_per36",
        "fga_per36",
        "fta_per36",
        "fg3a_per36",
        "oreb_per36",
        "dreb_per36",
        "pf_per36",
        "fg2_pct",
        "fg3_pct",
        "ft_pct",
    ]

    if template.empty:
        for col in needed_cols[2:]:
            profiles[col] = np.nan
        merged = profiles
    else:
        template["season_year"] = template["season"].map(_season_start_year)
        for season, season_profiles in profiles.groupby("season", sort=False):
            target_year = _season_start_year(season)
            if forecast_mode:
                eligible = template[template["season_year"] < target_year].copy()
            else:
                eligible = template[template["season_year"] <= target_year].copy()
            if eligible.empty:
                merged_frames.append(season_profiles.copy())
                continue
            latest = eligible.sort_values(["player_id", "season_year"]).groupby("player_id", as_index=False).tail(1)
            merged_frames.append(season_profiles.merge(latest[needed_cols], on="player_id", how="left", suffixes=("", "_hist")))
        merged = pd.concat(merged_frames, ignore_index=True)

    merged = merged.merge(
        group_medians,
        left_on=["off_archetype", "role"],
        right_on=["group_off_archetype", "group_role"],
        how="left",
        suffixes=("", "_group"),
    )
    merged = merged.merge(role_medians, left_on="role", right_on="group_role", how="left", suffixes=("", "_role"))

    fill_cols = [
        "pts_per36",
        "ast_per36",
        "reb_per36",
        "stl_per36",
        "blk_per36",
        "tov_per36",
        "fga_per36",
        "fta_per36",
        "fg3a_per36",
        "oreb_per36",
        "dreb_per36",
        "pf_per36",
        "fg2_pct",
        "fg3_pct",
        "ft_pct",
    ]
    for col in fill_cols:
        group_col = f"{col}_group"
        role_col = f"{col}_role"
        merged[col] = pd.to_numeric(merged.get(col), errors="coerce")
        if group_col in merged.columns:
            merged[col] = merged[col].fillna(pd.to_numeric(merged[group_col], errors="coerce"))
        if role_col in merged.columns:
            merged[col] = merged[col].fillna(pd.to_numeric(merged[role_col], errors="coerce"))

    defaults = {
        "pts_per36": 14.0,
        "ast_per36": 3.5,
        "reb_per36": 5.5,
        "stl_per36": 1.0,
        "blk_per36": 0.7,
        "tov_per36": 1.8,
        "fga_per36": 12.0,
        "fta_per36": 3.0,
        "fg3a_per36": 4.0,
        "oreb_per36": 1.2,
        "dreb_per36": 4.0,
        "pf_per36": 2.8,
        "fg2_pct": 0.51,
        "fg3_pct": 0.35,
        "ft_pct": 0.77,
    }
    for col, default in defaults.items():
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(default)

    merged["fg3a_per36"] = np.minimum(merged["fg3a_per36"], merged["fga_per36"])
    merged["base_fga_per36"] = merged["fga_per36"] * (0.80 + merged["behavioral_usage"].clip(0.05, 0.40))
    merged["base_fta_per36"] = merged["fta_per36"] * (0.85 + 0.75 * merged["behavioral_usage"].clip(0.05, 0.40))
    merged["availability_rate"] = (merged["games_target"] / 82.0).clip(lower=0.25, upper=1.0)
    keep_cols = [
        "player_id",
        "player_name",
        "season",
        "team_abbreviation",
        "role",
        "off_archetype",
        "def_archetype",
        "base_mpg",
        "games_target",
        "availability_rate",
        "impact_total_impact",
        "impact_obke",
        "impact_dbke",
        "behavioral_usage",
        "behavioral_assist_rate",
        "behavioral_turnover_rate",
        "behavioral_three_point_rate",
        "pts_per36",
        "ast_per36",
        "reb_per36",
        "stl_per36",
        "blk_per36",
        "tov_per36",
        "base_fga_per36",
        "base_fta_per36",
        "fg3a_per36",
        "oreb_per36",
        "dreb_per36",
        "pf_per36",
        "fg2_pct",
        "fg3_pct",
        "ft_pct",
    ]
    return merged[keep_cols].copy()


def build_team_descriptor_map(rosters: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, float]]]:
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for (season, team), group in rosters.groupby(["season", "team_abbreviation"], sort=False):
        core = group.sort_values("base_mpg", ascending=False).head(PLAYER_GAME_ROTATION_SIZE).copy()
        weights = np.maximum(core["base_mpg"].to_numpy(dtype=float), 1e-3)
        weights = weights / weights.sum()
        off = core["off_archetype"].fillna("").astype(str)
        deff = core["def_archetype"].fillna("").astype(str)
        role = core["role"].fillna("Wing").astype(str)

        creator_mask = off.isin(CREATOR_ARCHETYPES) & role.isin({"Guard", "Wing"})
        spacing_mask = off.isin(SPACER_ARCHETYPES)
        rim_mask = off.isin(RIM_ARCHETYPES)
        poa_mask = deff.isin(POA_DEF_ARCHETYPES) & role.isin({"Guard", "Wing"})
        rim_def_mask = deff.isin(RIM_DEF_ARCHETYPES)
        versatile_mask = deff.isin(VERSATILE_DEF_ARCHETYPES)
        low_activity_mask = deff.isin(LOW_ACTIVITY_DEF_ARCHETYPES)
        drop_mask = deff.eq("Dropping Big")

        out.setdefault(str(season), {})[str(team).upper()] = {
            "creator_share": float(np.dot(creator_mask.astype(float).to_numpy(dtype=float), weights)),
            "spacing_share": float(np.dot(spacing_mask.astype(float).to_numpy(dtype=float), weights)),
            "rim_pressure": float(np.dot(rim_mask.astype(float).to_numpy(dtype=float), weights)),
            "poa_pressure": float(np.dot(poa_mask.astype(float).to_numpy(dtype=float), weights)),
            "rim_defense": float(np.dot(rim_def_mask.astype(float).to_numpy(dtype=float), weights)),
            "versatile_pressure": float(np.dot(versatile_mask.astype(float).to_numpy(dtype=float), weights)),
            "low_activity_share": float(np.dot(low_activity_mask.astype(float).to_numpy(dtype=float), weights)),
            "drop_exposure": float(np.dot(drop_mask.astype(float).to_numpy(dtype=float), weights)),
            "turnover_pressure": float(
                np.dot((poa_mask | versatile_mask).astype(float).to_numpy(dtype=float), weights)
            ),
            "off_usage_load": _weighted_average(core["behavioral_usage"].to_numpy(dtype=float), weights),
        }
    return out


def build_pairwise_matchup_map(rosters: pd.DataFrame) -> Dict[str, Dict[Tuple[str, str], Dict[str, float]]]:
    descriptors = build_team_descriptor_map(rosters)
    out: Dict[str, Dict[Tuple[str, str], Dict[str, float]]] = {}
    for season, teams in descriptors.items():
        out[season] = {}
        team_keys = sorted(teams.keys())
        for home in team_keys:
            for away in team_keys:
                if home == away:
                    continue
                h = teams[home]
                a = teams[away]
                home_off = (
                    1.20 * h["creator_share"] * (0.45 - a["poa_pressure"])
                    + 1.00 * h["spacing_share"] * (0.45 - a["versatile_pressure"])
                    + 1.10 * h["rim_pressure"] * (0.50 - a["rim_defense"])
                    - 0.85 * h["off_usage_load"] * a["turnover_pressure"]
                    + 0.55 * h["spacing_share"] * a["drop_exposure"]
                    + 0.40 * a["low_activity_share"]
                )
                away_off = (
                    1.20 * a["creator_share"] * (0.45 - h["poa_pressure"])
                    + 1.00 * a["spacing_share"] * (0.45 - h["versatile_pressure"])
                    + 1.10 * a["rim_pressure"] * (0.50 - h["rim_defense"])
                    - 0.85 * a["off_usage_load"] * h["turnover_pressure"]
                    + 0.55 * a["spacing_share"] * h["drop_exposure"]
                    + 0.40 * h["low_activity_share"]
                )
                home_off *= MATCHUP_TEAM_INTERACTION_SCALE
                away_off *= MATCHUP_TEAM_INTERACTION_SCALE
                spread_bonus = _clip(home_off - away_off, -MATCHUP_SPREAD_BONUS_CLIP, MATCHUP_SPREAD_BONUS_CLIP)
                out[season][(home, away)] = {
                    "spread_bonus": float(spread_bonus),
                    "home_offense_bonus": float(home_off),
                    "away_offense_bonus": float(away_off),
                }
    return out


def _player_matchup_adjustments(player: pd.Series, opponent_descriptor: Dict[str, float]) -> Dict[str, float]:
    off = str(player.get("off_archetype", "Unknown"))
    scoring_eff = 1.0
    three_eff = 1.0
    turnover_mult = 1.0
    ast_mult = 1.0
    reb_mult = 1.0
    stocks_mult = 1.0

    if off in {"Ball Dominant Creator", "Ballhandler", "Perimeter Scorer", "All-Around Scorer"}:
        scoring_eff -= 0.08 * opponent_descriptor.get("poa_pressure", 0.0)
        turnover_mult += 0.16 * opponent_descriptor.get("turnover_pressure", 0.0)
        ast_mult -= 0.07 * opponent_descriptor.get("poa_pressure", 0.0)

    if off in {"Off-Ball Movement Shooter", "Off-Ball Stationary Shooter", "PnR Popping Big"}:
        three_eff -= 0.09 * opponent_descriptor.get("versatile_pressure", 0.0)
        three_eff += 0.06 * opponent_descriptor.get("drop_exposure", 0.0)

    if off in {"Interior Scorer", "Off-Ball Finisher", "PnR Rolling Big"}:
        scoring_eff -= 0.10 * opponent_descriptor.get("rim_defense", 0.0)
        reb_mult += 0.05 * (1.0 - opponent_descriptor.get("rim_defense", 0.0))

    if str(player.get("def_archetype", "")) in POA_DEF_ARCHETYPES:
        stocks_mult += 0.12 * opponent_descriptor.get("creator_share", 0.0)
    if str(player.get("def_archetype", "")) in RIM_DEF_ARCHETYPES:
        stocks_mult += 0.10 * opponent_descriptor.get("rim_pressure", 0.0)
        reb_mult += 0.05

    return {
        "scoring_eff": _clip(scoring_eff, 0.78, 1.15),
        "three_eff": _clip(three_eff, 0.78, 1.18),
        "turnover_mult": _clip(turnover_mult, 0.88, 1.25),
        "ast_mult": _clip(ast_mult, 0.84, 1.15),
        "reb_mult": _clip(reb_mult, 0.88, 1.15),
        "stocks_mult": _clip(stocks_mult, 0.90, 1.25),
    }


def _build_team_game_index_map(schedule: List[object]) -> Dict[Tuple[str, str], List[int]]:
    out: Dict[Tuple[str, str], List[int]] = {}
    for idx, game in enumerate(schedule):
        out.setdefault((str(game.season), str(game.home_team).upper()), []).append(idx)
        out.setdefault((str(game.season), str(game.away_team).upper()), []).append(idx)
    return out


def build_availability_plan(schedule: List[object], rosters: pd.DataFrame, rng: np.random.Generator) -> Dict[Tuple[str, str, str], set]:
    team_games = _build_team_game_index_map(schedule)
    out: Dict[Tuple[str, str, str], set] = {}
    for (season, team), group in rosters.groupby(["season", "team_abbreviation"], sort=False):
        indices = team_games.get((str(season), str(team).upper()), [])
        total_games = len(indices)
        for _, row in group.iterrows():
            key = (str(season), str(team).upper(), str(row["player_id"]))
            target = int(round(min(float(row.get("games_target", total_games)), float(total_games))))
            if target >= total_games:
                out[key] = set(indices)
                continue
            if target <= 0 or total_games == 0:
                out[key] = set()
                continue
            sampled = rng.choice(indices, size=target, replace=False)
            out[key] = set(int(x) for x in sampled.tolist())
    return out


def _ensure_minimum_active(team_df: pd.DataFrame, active_mask: pd.Series) -> pd.Series:
    if int(active_mask.sum()) >= PLAYER_GAME_MIN_ACTIVE:
        return active_mask
    ranked = team_df.sort_values("base_mpg", ascending=False).index.tolist()
    fixed = active_mask.copy()
    for idx in ranked:
        fixed.loc[idx] = True
        if int(fixed.sum()) >= PLAYER_GAME_MIN_ACTIVE:
            break
    return fixed


def _allocate_minutes(team_df: pd.DataFrame, lineup_context: Dict[str, float], active_mask: pd.Series) -> pd.DataFrame:
    active = team_df.loc[active_mask].copy()
    if active.empty:
        return active

    starter_ids = set(lineup_context.get("starter_ids", []))
    clutch_ids = set(lineup_context.get("clutch_ids", []))

    # Compute raw importance weights for rotation selection
    raw_weights = active["base_mpg"].clip(lower=1.0).to_numpy(dtype=float)
    starter_boost = active["player_id"].astype(str).isin(starter_ids).astype(float).to_numpy(dtype=float)
    clutch_boost = active["player_id"].astype(str).isin(clutch_ids).astype(float).to_numpy(dtype=float)
    importance = raw_weights * (1.0 + 0.20 * starter_boost) * (1.0 + 0.10 * clutch_boost)

    # Limit rotation to PLAYER_GAME_ROTATION_SIZE (typically 10).
    # Always include starters; fill remaining slots by importance.
    rotation_size = min(PLAYER_GAME_ROTATION_SIZE, len(active))
    order = np.argsort(-importance)
    selected = set()
    for i, pid in enumerate(active["player_id"].astype(str)):
        if pid in starter_ids:
            selected.add(i)
    for idx in order:
        if len(selected) >= rotation_size:
            break
        selected.add(int(idx))

    rotation_mask = pd.Series(False, index=active.index)
    for i, orig_idx in enumerate(active.index):
        if i in selected:
            rotation_mask.loc[orig_idx] = True
    active = active.loc[rotation_mask].copy()
    if active.empty:
        return active

    # Anchored minute allocation: start from base_mpg, apply small
    # context bonuses, then redistribute surplus/deficit so total == 240.
    base = active["base_mpg"].clip(lower=6.0).to_numpy(dtype=float).copy()
    is_starter = active["player_id"].astype(str).isin(starter_ids).to_numpy(dtype=bool)
    is_clutch = active["player_id"].astype(str).isin(clutch_ids).to_numpy(dtype=bool)

    # Small context bonuses (additive, not multiplicative, to avoid inflation)
    base[is_starter] += PLAYER_GAME_MINUTES_STARTER_BONUS * 10.0  # ~+1.4 min
    base[is_clutch] += PLAYER_GAME_MINUTES_CLUTCH_BONUS * 10.0    # ~+0.5 min
    base[~is_starter] -= PLAYER_GAME_MINUTES_BENCH_PENALTY * 10.0 # ~-0.8 min
    base = np.maximum(base, 4.0)

    # Enforce 240 total: proportionally scale, but cap individual at base_mpg + 4
    total = float(base.sum())
    if total > 1e-6:
        minutes = PLAYER_GAME_TOTAL_MINUTES * (base / total)
    else:
        minutes = np.full(len(base), PLAYER_GAME_TOTAL_MINUTES / len(base))

    # Cap: no player plays more than base_mpg + 4 in a single game
    original_mpg = active["base_mpg"].clip(lower=6.0).to_numpy(dtype=float)
    cap = np.minimum(original_mpg + 4.0, 40.0)
    excess = np.maximum(minutes - cap, 0.0)
    minutes = np.minimum(minutes, cap)
    # Redistribute excess proportionally to players under their cap
    headroom = cap - minutes
    headroom_total = float(headroom.sum())
    if float(excess.sum()) > 0.5 and headroom_total > 0.5:
        minutes += headroom * (float(excess.sum()) / headroom_total)
        minutes = np.minimum(minutes, cap)

    minutes = np.clip(minutes, 4.0, 40.0)
    # Final normalization pass to ensure exact 240
    minutes = PLAYER_GAME_TOTAL_MINUTES * (minutes / max(float(minutes.sum()), 1.0))

    active["sim_minutes"] = minutes
    active["is_starter"] = is_starter
    active["is_clutch_core"] = is_clutch
    return active


def _reconcile_team_points(team_df: pd.DataFrame, team_score: int) -> pd.DataFrame:
    if team_df.empty:
        return team_df
    current = int(team_df["pts"].sum())
    if current == team_score:
        return team_df

    adjusted = team_df.sort_values(["usage_proxy", "sim_minutes"], ascending=False).copy()
    idx_cycle = list(adjusted.index)
    cursor = 0
    while current < team_score and idx_cycle:
        idx = idx_cycle[cursor % len(idx_cycle)]
        adjusted.loc[idx, "ftm"] += 1
        adjusted.loc[idx, "fta"] += 1
        adjusted.loc[idx, "pts"] += 1
        current += 1
        cursor += 1
    while current > team_score and idx_cycle:
        idx = idx_cycle[cursor % len(idx_cycle)]
        if adjusted.loc[idx, "ftm"] > 0:
            adjusted.loc[idx, "ftm"] -= 1
            adjusted.loc[idx, "fta"] = max(int(adjusted.loc[idx, "fta"]), int(adjusted.loc[idx, "ftm"]))
            adjusted.loc[idx, "pts"] -= 1
            current -= 1
        cursor += 1
        if cursor > 500:
            break
    return adjusted.sort_index()


def simulate_single_game_player_stats(
    season: str,
    team_abbreviation: str,
    opponent_abbreviation: str,
    team_score: int,
    opponent_score: int,
    possessions: float,
    team_roster: pd.DataFrame,
    opponent_descriptor: Dict[str, float],
    lineup_context: Dict[str, float],
    rng: np.random.Generator,
    close_game: bool,
    is_home: bool,
    game_id: str,
    date: str,
    model_key: str,
) -> pd.DataFrame:
    active = _allocate_minutes(team_roster, lineup_context, pd.Series(True, index=team_roster.index))
    if active.empty:
        return active

    adjustment_rows = []
    for _, row in active.iterrows():
        adj = _player_matchup_adjustments(row, opponent_descriptor)
        adjustment_rows.append(adj)
    adj_df = pd.DataFrame(adjustment_rows, index=active.index)
    active = pd.concat([active, adj_df], axis=1)

    active["usage_proxy"] = (
        active["behavioral_usage"].clip(lower=0.08, upper=0.40)
        * (0.85 + 0.15 * active["is_starter"].astype(float))
    )
    shot_volume = active["base_fga_per36"] * (active["sim_minutes"] / 36.0) * active["scoring_eff"]
    ft_volume = active["base_fta_per36"] * (active["sim_minutes"] / 36.0) * active["scoring_eff"]
    ast_volume = active["ast_per36"] * (active["sim_minutes"] / 36.0) * active["ast_mult"]
    tov_volume = active["tov_per36"] * (active["sim_minutes"] / 36.0) * active["turnover_mult"]
    oreb_volume = active["oreb_per36"] * (active["sim_minutes"] / 36.0) * active["reb_mult"]
    dreb_volume = active["dreb_per36"] * (active["sim_minutes"] / 36.0) * active["reb_mult"]
    stl_volume = active["stl_per36"] * (active["sim_minutes"] / 36.0) * active["stocks_mult"]
    blk_volume = active["blk_per36"] * (active["sim_minutes"] / 36.0) * active["stocks_mult"]
    pf_volume = active["pf_per36"] * (active["sim_minutes"] / 36.0)

    fg3_share = (active["fg3a_per36"] / active["base_fga_per36"].replace(0.0, np.nan)).fillna(0.32).clip(0.05, 0.85)
    expected_points = (
        shot_volume * ((1.0 - fg3_share) * 2.0 * active["fg2_pct"] + fg3_share * 3.0 * active["fg3_pct"] * active["three_eff"])
        + ft_volume * active["ft_pct"]
    )
    raw_points = float(expected_points.sum())
    scale = team_score / raw_points if raw_points > 1e-6 else 1.0
    scale = _clip(scale, 0.70, 1.30)

    shot_volume *= scale
    ft_volume *= scale
    ast_volume *= np.sqrt(scale)

    fga = rng.poisson(np.maximum(shot_volume, 0.1))
    fg3a = np.minimum(fga, rng.poisson(np.maximum(shot_volume * fg3_share, 0.05)))
    fta = rng.poisson(np.maximum(ft_volume, 0.05))
    tov = rng.poisson(np.maximum(tov_volume, 0.02))
    ast = rng.poisson(np.maximum(ast_volume, 0.02))
    oreb = rng.poisson(np.maximum(oreb_volume, 0.02))
    dreb = rng.poisson(np.maximum(dreb_volume, 0.02))
    stl = rng.poisson(np.maximum(stl_volume, 0.01))
    blk = rng.poisson(np.maximum(blk_volume, 0.01))
    pf = rng.poisson(np.maximum(pf_volume, 0.05))

    fg2a = np.maximum(fga - fg3a, 0)
    fg3m = np.array([rng.binomial(int(a), _clip(float(p), 0.15, 0.60)) for a, p in zip(fg3a, active["fg3_pct"] * active["three_eff"])])
    fg2m = np.array([rng.binomial(int(a), _clip(float(p), 0.30, 0.78)) for a, p in zip(fg2a, active["fg2_pct"] * active["scoring_eff"])])
    ftm = np.array([rng.binomial(int(a), _clip(float(p), 0.45, 0.95)) for a, p in zip(fta, active["ft_pct"])])
    pts = (2 * fg2m + 3 * fg3m + ftm).astype(int)

    team_df = active[["player_id", "player_name", "season", "team_abbreviation", "role", "off_archetype", "def_archetype", "sim_minutes", "is_starter", "is_clutch_core", "impact_total_impact", "impact_obke", "impact_dbke", "usage_proxy"]].copy()
    team_df["fga"] = fga.astype(int)
    team_df["fgm"] = (fg2m + fg3m).astype(int)
    team_df["fg3a"] = fg3a.astype(int)
    team_df["fg3m"] = fg3m.astype(int)
    team_df["fta"] = fta.astype(int)
    team_df["ftm"] = ftm.astype(int)
    team_df["oreb"] = oreb.astype(int)
    team_df["dreb"] = dreb.astype(int)
    team_df["reb"] = (oreb + dreb).astype(int)
    team_df["ast"] = ast.astype(int)
    team_df["stl"] = stl.astype(int)
    team_df["blk"] = blk.astype(int)
    team_df["tov"] = tov.astype(int)
    team_df["pf"] = pf.astype(int)
    team_df["pts"] = pts.astype(int)
    team_df["game_id"] = str(game_id)
    team_df["date"] = str(date)
    team_df["opponent_abbreviation"] = str(opponent_abbreviation).upper()
    team_df["is_home"] = bool(is_home)
    team_df["model"] = str(model_key)
    team_df["team_score"] = int(team_score)
    team_df["opponent_score"] = int(opponent_score)
    team_df["possessions"] = float(possessions)
    team_df["close_game_triggered"] = bool(close_game)
    team_df["lineup_team_bonus"] = float(lineup_context.get("team_bonus", 0.0))
    team_df["lineup_clutch_bonus"] = float(lineup_context.get("clutch_bonus", 0.0)) if close_game else 0.0
    team_df = _reconcile_team_points(team_df, int(team_score))
    team_df["ts_pct"] = team_df["pts"] / (2.0 * (team_df["fga"] + 0.44 * team_df["fta"]).replace(0.0, np.nan))
    team_df["shot_share"] = team_df["fga"] / max(float(team_df["fga"].sum()), 1.0)
    team_df["assist_share"] = team_df["ast"] / max(float(team_df["ast"].sum()), 1.0)
    team_df["rebound_share"] = team_df["reb"] / max(float(team_df["reb"].sum()), 1.0)
    team_df["game_impact_proxy"] = (
        team_df["pts"]
        + 1.5 * team_df["ast"]
        + 1.2 * team_df["reb"]
        + 2.5 * team_df["stl"]
        + 2.5 * team_df["blk"]
        - 1.2 * team_df["tov"]
        - 0.7 * (team_df["fga"] - team_df["fgm"])
        - 0.4 * (team_df["fta"] - team_df["ftm"])
    )
    return team_df.reset_index(drop=True)


def simulate_detailed_season(
    schedule: List[object],
    rosters: pd.DataFrame,
    lineup_bonus_map: Dict[str, Dict[str, Dict[str, float]]],
    matchup_map: Dict[str, Dict[Tuple[str, str], Dict[str, float]]],
    game_environment_rows: List[Dict[str, float]],
    rng: np.random.Generator,
    model_key: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    env_by_game = {str(row["game_id"]): row for row in game_environment_rows}
    descriptors = build_team_descriptor_map(rosters)
    availability_plan = build_availability_plan(schedule, rosters, rng)
    game_rows: List[pd.DataFrame] = []

    for game_index, game in enumerate(schedule):
        env = env_by_game.get(str(game.game_id))
        if not env:
            continue
        season = str(game.season)
        home = str(game.home_team).upper()
        away = str(game.away_team).upper()
        home_roster = rosters[(rosters["season"] == season) & (rosters["team_abbreviation"] == home)].copy()
        away_roster = rosters[(rosters["season"] == season) & (rosters["team_abbreviation"] == away)].copy()
        if home_roster.empty or away_roster.empty:
            continue

        home_active_mask = home_roster["player_id"].astype(str).map(lambda pid: game_index in availability_plan.get((season, home, pid), set()))
        away_active_mask = away_roster["player_id"].astype(str).map(lambda pid: game_index in availability_plan.get((season, away, pid), set()))
        home_active_mask = _ensure_minimum_active(home_roster, home_active_mask)
        away_active_mask = _ensure_minimum_active(away_roster, away_active_mask)
        home_roster = home_roster.loc[home_active_mask].copy()
        away_roster = away_roster.loc[away_active_mask].copy()

        home_ctx = lineup_bonus_map.get(season, {}).get(home, {})
        away_ctx = lineup_bonus_map.get(season, {}).get(away, {})
        home_desc = descriptors.get(season, {}).get(home, {})
        away_desc = descriptors.get(season, {}).get(away, {})
        close_game = abs(float(env.get("preclutch_margin", env.get("margin_mean", 0.0)))) <= CLUTCH_MARGIN_TRIGGER

        home_rows = simulate_single_game_player_stats(
            season=season,
            team_abbreviation=home,
            opponent_abbreviation=away,
            team_score=int(env["home_score"]),
            opponent_score=int(env["away_score"]),
            possessions=float(env["possessions"]),
            team_roster=home_roster,
            opponent_descriptor=away_desc,
            lineup_context=home_ctx,
            rng=rng,
            close_game=close_game,
            is_home=True,
            game_id=str(game.game_id),
            date=str(game.date),
            model_key=model_key,
        )
        away_rows = simulate_single_game_player_stats(
            season=season,
            team_abbreviation=away,
            opponent_abbreviation=home,
            team_score=int(env["away_score"]),
            opponent_score=int(env["home_score"]),
            possessions=float(env["possessions"]),
            team_roster=away_roster,
            opponent_descriptor=home_desc,
            lineup_context=away_ctx,
            rng=rng,
            close_game=close_game,
            is_home=False,
            game_id=str(game.game_id),
            date=str(game.date),
            model_key=model_key,
        )
        game_rows.extend([home_rows, away_rows])

    if not game_rows:
        empty = pd.DataFrame()
        return empty, empty

    games_df = pd.concat(game_rows, ignore_index=True)
    season_df = (
        games_df.groupby(["season", "model", "team_abbreviation", "player_id", "player_name"], as_index=False)
        .agg(
            games_played=("game_id", "nunique"),
            games_started=("is_starter", "sum"),
            minutes=("sim_minutes", "sum"),
            pts=("pts", "sum"),
            reb=("reb", "sum"),
            ast=("ast", "sum"),
            stl=("stl", "sum"),
            blk=("blk", "sum"),
            tov=("tov", "sum"),
            pf=("pf", "sum"),
            fgm=("fgm", "sum"),
            fga=("fga", "sum"),
            fg3m=("fg3m", "sum"),
            fg3a=("fg3a", "sum"),
            ftm=("ftm", "sum"),
            fta=("fta", "sum"),
            oreb=("oreb", "sum"),
            dreb=("dreb", "sum"),
            impact_total_impact=("impact_total_impact", "first"),
            impact_obke=("impact_obke", "first"),
            impact_dbke=("impact_dbke", "first"),
            total_game_impact_proxy=("game_impact_proxy", "sum"),
        )
    )
    season_df["mpg"] = season_df["minutes"] / season_df["games_played"].replace(0, np.nan)
    season_df["ts_pct"] = season_df["pts"] / (2.0 * (season_df["fga"] + 0.44 * season_df["fta"]).replace(0.0, np.nan))
    season_df["usage_proxy"] = (season_df["fga"] + 0.44 * season_df["fta"] + season_df["tov"]) / season_df["minutes"].replace(0.0, np.nan)
    season_df["ast_to_ratio"] = season_df["ast"] / season_df["tov"].replace(0.0, np.nan)
    season_df["off_rating_proxy"] = 100.0 * season_df["pts"] / (season_df["fga"] + 0.44 * season_df["fta"] + season_df["tov"]).replace(0.0, np.nan)
    season_df["reb_per36"] = 36.0 * season_df["reb"] / season_df["minutes"].replace(0.0, np.nan)
    season_df["ast_per36"] = 36.0 * season_df["ast"] / season_df["minutes"].replace(0.0, np.nan)
    season_df["pts_per36"] = 36.0 * season_df["pts"] / season_df["minutes"].replace(0.0, np.nan)
    return season_df.reset_index(drop=True), games_df.reset_index(drop=True)