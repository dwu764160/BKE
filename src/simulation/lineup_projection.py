"""
src/simulation/lineup_projection.py
=============================================================================
Simulation Core — Step 2: Lineup Projection Model

Builds three team-phase strength profiles from player-level PEC outputs:
  1) Starters
  2) Rotation
  3) Clutch lineup

Validation outputs:
  - Starter overlap vs observed high-possession lineup
  - Clutch overlap vs top-5 clutch-minute players
  - Rotation strength correlation vs bench-heavy lineup NET_RTG

Inputs:
  - data/processed/player_eval/player_impact_profiles.parquet
  - data/processed/player_position_estimates.parquet
  - data/historical/player_clutch_stats_all.parquet
  - data/processed/metrics_lineups.parquet
  - data/historical/teams.parquet

Outputs:
  - data/processed/simulation/simulation_step2_lineup_profiles.parquet
  - reports/simulation_step2_lineup_profiles.json
  - reports/simulation_step2_validation.json

Usage:
  python3 src/simulation/lineup_projection.py
=============================================================================
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.simulation.simulation_config import (
    CLUTCH_STATS_ALL_PATH,
    CLUTCH_WEIGHT_C,
    CLUTCH_WEIGHT_I,
    CLUTCH_WEIGHT_M,
    CLUTCH_WEIGHT_TALENT,
    HISTORICAL_DIR,
    LINEUP_SIZE,
    METRICS_LINEUPS_PATH,
    MIN_MPG_FOR_POOL,
    PLAYER_PROFILES_PATH,
    PLAYER_VOL_BASE,
    PLAYER_VOL_CEILING,
    PLAYER_VOL_FLOOR,
    PLAYER_VOL_STABILITY_CENTER,
    PLAYER_VOL_STABILITY_SCALE,
    POSITION_ESTIMATES_PATH,
    ROTATION_ALPHA,
    STAGGER_MINUTES_THRESHOLD,
    STARTER_WEIGHT_C,
    STARTER_WEIGHT_I,
    STARTER_WEIGHT_M,
    STARTER_WEIGHT_POS,
    STEP2_CLUTCH_CANDIDATE_SIZE,
    STEP2_CLUTCH_MAX_SWAPS,
    STEP2_CLUTCH_MIN_STARTERS,
    STEP2_CANDIDATE_LIMIT,
    STEP2_CANDIDATE_PER_BAND,
    STEP2_CANDIDATE_PER_ROLE,
    STEP2_CONTINUITY_DEFAULT_RETURNING,
    STEP2_CONTINUITY_KAPPA,
    STEP2_CONTINUITY_ROTATION_BONUS,
    STEP2_CONTINUITY_STARTER_BONUS,
    STEP2_ENABLE_CLUTCH_CORE_CONSTRAINT,
    STEP2_ENABLE_CONTINUITY_PRIOR,
    STEP2_ENABLE_ROTATION_REGIME_MODEL,
    STEP2_FIT_CREATOR_BONUS,
    STEP2_FIT_POA_BONUS,
    STEP2_FIT_RIM_BONUS,
    STEP2_FIT_SPACING_BONUS,
    STEP2_FIT_STAGGER_BONUS,
    STEP2_IMPACT_COLUMN,
    STEP2_LINEUP_PROFILES_PATH,
    STEP2_LINEUP_REPORT_PATH,
    STEP2_MINUTES_COLUMN,
    STEP2_REGIME_BENCH_COEF,
    STEP2_REGIME_STAR_COEF,
    STEP2_ROT_BENCH_BENCH_W,
    STEP2_ROT_BENCH_START_W,
    STEP2_ROT_SIGMA_BASE,
    STEP2_ROT_SIGMA_BENCH_W,
    STEP2_ROT_STAGGER_BENCH_W,
    STEP2_ROT_STAGGER_START_W,
    STEP2_STABILITY_COLUMN,
    STEP2_STARTER_LOW_MINUTES_PENALTY,
    STEP2_STARTER_LOW_MINUTES_THRESHOLD,
    STEP2_STARTER_REQUIRE_TRUE_BIG,
    STEP2_STARTER_TRUE_BIG_BANDS,
    STEP2_VALIDATION_PATH,
    TEAMS_PATH,
)


def _norm_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True)


def _safe_float(value: float, decimals: int = 4) -> Optional[float]:
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), decimals)


def _minmax(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    lo = float(s.min())
    hi = float(s.max())
    if hi - lo <= 1e-12:
        return pd.Series(np.ones(len(s)), index=s.index, dtype=float)
    return (s - lo) / (hi - lo)


def _role_from_text(text: str) -> str:
    t = str(text or "").strip().lower()
    if not t or t in {"nan", "none", "unknown"}:
        return "Wing"

    # Normalize separators so legacy and canonical hybrid labels map consistently.
    normalized = t.replace("_", "-").replace("/", "-").replace(" ", "-")

    # Explicit hybrid handling avoids accidental guard-first routing for Guard-Forward.
    if normalized in {"guard-forward", "forward-guard", "gf", "fg", "g-f", "f-g"}:
        return "Wing"
    if normalized in {"forward-center", "center-forward", "fc", "cf", "f-c", "c-f"}:
        return "Big"

    if "center" in normalized or normalized == "c":
        return "Big"
    if "forward" in normalized or normalized in {"f", "wing"}:
        return "Wing"
    if "guard" in normalized or normalized == "g":
        return "Guard"
    return "Wing"


def _clean_text(value) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.lower() in {"nan", "none", "null", "unknown", "n/a", "na"}:
        return None
    return text


def _clean_name(value) -> Optional[str]:
    return _clean_text(value)


def _weighted_avg(values: np.ndarray, weights: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    w = np.where(np.isfinite(w), w, 0.0)
    v = np.where(np.isfinite(v), v, 0.0)
    if w.sum() <= 1e-9:
        return float(np.mean(v)) if len(v) else 0.0
    return float(np.sum(v * w) / np.sum(w))


def _to_id_list(value) -> List[str]:
    if isinstance(value, np.ndarray):
        return [str(v).strip().replace(".0", "") for v in value.tolist() if str(v).strip()]
    if isinstance(value, (list, tuple)):
        return [str(v).strip().replace(".0", "") for v in value if str(v).strip()]
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            text = text.strip("[]")
            if not text:
                return []
            parts = [p.strip().strip("'\"") for p in text.split(",")]
            return [p.replace(".0", "") for p in parts if p]
        return [text.replace(".0", "")]
    return []


def _lineup_from_value(value) -> List[str]:
    ids = []
    seen = set()
    for pid in _to_id_list(value):
        clean = str(pid).strip().replace(".0", "")
        if not clean or clean == "0" or clean in seen:
            continue
        seen.add(clean)
        ids.append(clean)
    return ids


def _replacement_pool_mask(df: pd.DataFrame) -> pd.Series:
    """Identify synthetic replacement rows across forecast schema versions."""
    if df.empty:
        return pd.Series(False, index=df.index, dtype=bool)

    if "player_id" in df.columns:
        pid = _norm_id(df["player_id"])
    else:
        pid = pd.Series("", index=df.index, dtype=str)
    id_mask = pid.astype(str).str.lower().str.startswith("repl_")

    if "player_name" in df.columns:
        names = df["player_name"].astype(str)
    else:
        names = pd.Series("", index=df.index, dtype=str)
    name_mask = names.str.contains("replacement pool", case=False, na=False)

    flag_mask = pd.Series(False, index=df.index, dtype=bool)
    if "is_replacement_pool" in df.columns:
        flag_mask = pd.to_numeric(df["is_replacement_pool"], errors="coerce").fillna(0).astype(int) == 1

    return id_mask | name_mask | flag_mask


def _is_valid_five_lineup(value) -> bool:
    return len(_lineup_from_value(value)) == LINEUP_SIZE


def _extract_season_from_path(path: Path) -> Optional[str]:
    match = re.search(r"pbp_with_lineups_(\d{4}-\d{2})\.parquet$", path.name)
    if not match:
        return None
    return match.group(1)


def _verify_q1_substitution_patterns(game_q1: pd.DataFrame, lineup_col: str) -> Tuple[bool, int, int]:
    """
    Validate first-quarter starter extraction using substitution clusters.

    Sub events are often emitted as a short burst (multiple SUB OUT / SUB IN rows).
    We verify cluster-level transitions from the last valid 5-man lineup before the
    cluster to the first valid 5-man lineup after the cluster.
    """
    if lineup_col not in game_q1.columns:
        return False, 0, 0

    subs_idx = game_q1.index[game_q1["event_type"] == "SUBSTITUTION"].tolist()
    if not subs_idx:
        return True, 0, 0

    clusters: List[Tuple[int, int]] = []
    start = subs_idx[0]
    prev = subs_idx[0]
    for idx in subs_idx[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        clusters.append((start, prev))
        start = idx
        prev = idx
    clusters.append((start, prev))

    lineups = game_q1[lineup_col].apply(_lineup_from_value)
    valid_mask = lineups.apply(lambda ids: len(ids) == LINEUP_SIZE)

    valid_clusters = 0
    considered_clusters = 0
    for start_idx, end_idx in clusters:
        before_mask = valid_mask.loc[: start_idx - 1] if start_idx > 0 else pd.Series(dtype=bool)
        after_mask = valid_mask.loc[end_idx + 1 :]
        if before_mask.empty or after_mask.empty:
            continue
        if not before_mask.any() or not after_mask.any():
            continue

        before_idx = before_mask[before_mask].index[-1]
        after_idx = after_mask[after_mask].index[0]

        before = set(lineups.loc[before_idx])
        after = set(lineups.loc[after_idx])
        diff_out = before - after
        diff_in = after - before

        considered_clusters += 1
        if 1 <= len(diff_out) <= LINEUP_SIZE and len(diff_out) == len(diff_in):
            valid_clusters += 1

    if considered_clusters == 0:
        return False, len(clusters), 0

    return (valid_clusters / float(considered_clusters)) >= 0.50, len(clusters), valid_clusters


def load_pbp_q1_starter_games(team_id_to_abbr: Dict[str, str]) -> pd.DataFrame:
    """
    Build per-game starter targets from first-quarter PBP lineups.

    Extraction rule:
      - use players on court at the first valid Q1 event (5-man lineup)
    Verification rule:
      - validate against Q1 substitution cluster transitions
    """
    records: List[Dict] = []
    pbp_files = sorted(Path(HISTORICAL_DIR).glob("pbp_with_lineups_*.parquet"))
    if not pbp_files or not team_id_to_abbr:
        return pd.DataFrame(
            columns=[
                "season",
                "game_id",
                "team_abbreviation",
                "starter_lineup_ids",
                "starter_verified",
                "sub_clusters",
                "sub_clusters_valid",
            ]
        )

    team_ids = sorted(team_id_to_abbr.keys())
    lineup_cols = [f"lineup_{tid}" for tid in team_ids]

    for path in pbp_files:
        season = _extract_season_from_path(path)
        if not season:
            continue

        base_cols = ["game_id", "period", "event_type", "clock"]
        read_cols = base_cols + lineup_cols
        try:
            df = pd.read_parquet(path, columns=read_cols)
        except Exception:
            # Fallback in case parquet column projection fails in some environments.
            df = pd.read_parquet(path)
            keep_cols = [c for c in read_cols if c in df.columns]
            df = df[keep_cols]

        if "period" not in df.columns or "game_id" not in df.columns:
            continue

        q1 = df[df["period"] == 1].copy()
        if q1.empty:
            continue

        for game_id, gdf in q1.groupby("game_id", sort=False):
            gdf = gdf.reset_index(drop=True)
            for team_id in team_ids:
                lineup_col = f"lineup_{team_id}"
                if lineup_col not in gdf.columns:
                    continue

                lineups = gdf[lineup_col].apply(_lineup_from_value)
                valid_mask = lineups.apply(lambda ids: len(ids) == LINEUP_SIZE)
                if not valid_mask.any():
                    continue

                first_valid_idx = valid_mask[valid_mask].index[0]
                first_lineup = lineups.loc[first_valid_idx]
                if len(first_lineup) != LINEUP_SIZE:
                    continue

                verified, n_clusters, n_valid = _verify_q1_substitution_patterns(gdf, lineup_col)

                records.append(
                    {
                        "season": season,
                        "game_id": str(game_id),
                        "team_abbreviation": team_id_to_abbr.get(team_id),
                        "starter_lineup_ids": first_lineup,
                        "starter_verified": bool(verified),
                        "sub_clusters": int(n_clusters),
                        "sub_clusters_valid": int(n_valid),
                    }
                )

    out = pd.DataFrame(records)
    if out.empty:
        return out
    out = out.dropna(subset=["team_abbreviation"])
    out["season"] = out["season"].astype(str)
    out["team_abbreviation"] = out["team_abbreviation"].astype(str).str.upper()
    return out


def build_observed_starters_from_pbp(
    starter_games: pd.DataFrame,
    predicted_rows: List[Dict],
) -> Tuple[Dict[Tuple[str, str], Set[str]], Dict[Tuple[str, str], Dict]]:
    """
    Aggregate per-game first-quarter starters to a season-level observed starter set.

    Priority:
      1) Most frequent verified full lineup (mode lineup)
      2) Top-5 player frequency across verified/all games
      3) Predicted starter fallback to guarantee exactly 5 IDs
    """
    observed: Dict[Tuple[str, str], Set[str]] = {}
    meta: Dict[Tuple[str, str], Dict] = {}

    predicted_map = {(r["season"], r["team_abbreviation"]): r for r in predicted_rows}

    if not starter_games.empty:
        for (season, team), g in starter_games.groupby(["season", "team_abbreviation"], sort=False):
            g_all = g.copy()
            g_verified = g_all[g_all["starter_verified"] == True]
            source = g_verified if len(g_verified) >= 5 else g_all

            lineups = []
            for lineup in source["starter_lineup_ids"].tolist():
                ids = _lineup_from_value(lineup)
                if len(ids) == LINEUP_SIZE:
                    lineups.append(ids)

            if not lineups:
                continue

            lineup_counter = Counter(tuple(sorted(ids)) for ids in lineups)
            player_counter = Counter(pid for ids in lineups for pid in ids)

            selection_method = "mode_lineup"
            target_ids: List[str] = []
            if lineup_counter:
                target_ids = list(max(lineup_counter.items(), key=lambda kv: (kv[1], kv[0]))[0])

            if len(target_ids) < LINEUP_SIZE:
                selection_method = "frequency_top5"
                for pid, _ in player_counter.most_common():
                    if pid in target_ids:
                        continue
                    target_ids.append(pid)
                    if len(target_ids) >= LINEUP_SIZE:
                        break

            pred = predicted_map.get((season, team), {})
            if len(target_ids) < LINEUP_SIZE:
                selection_method = "predicted_fallback"
                for pid in [str(x) for x in pred.get("starter_player_ids", [])]:
                    if pid in target_ids:
                        continue
                    target_ids.append(pid)
                    if len(target_ids) >= LINEUP_SIZE:
                        break

            target_ids = target_ids[:LINEUP_SIZE]
            if len(target_ids) != LINEUP_SIZE:
                continue

            key = (str(season), str(team).upper())
            observed[key] = set(target_ids)
            meta[key] = {
                "selection_method": selection_method,
                "n_games_total": int(len(g_all)),
                "n_games_verified": int(len(g_verified)),
                "starter_ids_ranked": target_ids,
            }

    # Guarantee every predicted team-season has a 5-player observed set.
    for row in predicted_rows:
        key = (str(row["season"]), str(row["team_abbreviation"]).upper())
        if key in observed:
            continue
        fallback = []
        for pid in [str(x) for x in row.get("starter_player_ids", [])]:
            if pid and pid not in fallback:
                fallback.append(pid)
            if len(fallback) >= LINEUP_SIZE:
                break
        if len(fallback) == LINEUP_SIZE:
            observed[key] = set(fallback)
            meta[key] = {
                "selection_method": "predicted_fallback_only",
                "n_games_total": 0,
                "n_games_verified": 0,
                "starter_ids_ranked": fallback,
            }

    return observed, meta


@dataclass
class Step2Config:
    lineup_size: int = LINEUP_SIZE
    min_mpg_for_pool: float = MIN_MPG_FOR_POOL
    clutch_weight_c: float = CLUTCH_WEIGHT_C
    clutch_weight_talent: float = CLUTCH_WEIGHT_TALENT
    clutch_weight_i: float = CLUTCH_WEIGHT_I
    clutch_weight_m: float = CLUTCH_WEIGHT_M
    starter_weight_m: float = STARTER_WEIGHT_M
    starter_weight_i: float = STARTER_WEIGHT_I
    starter_weight_pos: float = STARTER_WEIGHT_POS
    starter_weight_c: float = STARTER_WEIGHT_C
    rotation_alpha: float = ROTATION_ALPHA
    stagger_minutes_threshold: float = STAGGER_MINUTES_THRESHOLD
    impact_column: str = STEP2_IMPACT_COLUMN
    minutes_column: str = STEP2_MINUTES_COLUMN
    stability_column: str = STEP2_STABILITY_COLUMN
    player_vol_base: float = PLAYER_VOL_BASE
    player_vol_stability_scale: float = PLAYER_VOL_STABILITY_SCALE
    player_vol_stability_center: float = PLAYER_VOL_STABILITY_CENTER
    player_vol_floor: float = PLAYER_VOL_FLOOR
    player_vol_ceiling: float = PLAYER_VOL_CEILING
    forecast_mode: bool = False
    # v1 optimization toggles
    enable_clutch_core: bool = STEP2_ENABLE_CLUTCH_CORE_CONSTRAINT
    enable_rotation_regime: bool = STEP2_ENABLE_ROTATION_REGIME_MODEL
    clutch_candidate_size: int = STEP2_CLUTCH_CANDIDATE_SIZE
    clutch_min_starters: int = STEP2_CLUTCH_MIN_STARTERS
    clutch_max_swaps: int = STEP2_CLUTCH_MAX_SWAPS
    regime_star_coef: float = STEP2_REGIME_STAR_COEF
    regime_bench_coef: float = STEP2_REGIME_BENCH_COEF
    rot_stagger_start_w: float = STEP2_ROT_STAGGER_START_W
    rot_stagger_bench_w: float = STEP2_ROT_STAGGER_BENCH_W
    rot_bench_start_w: float = STEP2_ROT_BENCH_START_W
    rot_bench_bench_w: float = STEP2_ROT_BENCH_BENCH_W
    rot_sigma_base: float = STEP2_ROT_SIGMA_BASE
    rot_sigma_bench_w: float = STEP2_ROT_SIGMA_BENCH_W
    # Constraint tuning
    require_true_big: bool = STEP2_STARTER_REQUIRE_TRUE_BIG
    true_big_bands: Tuple[str, ...] = STEP2_STARTER_TRUE_BIG_BANDS
    low_minutes_threshold: float = STEP2_STARTER_LOW_MINUTES_THRESHOLD
    low_minutes_penalty: float = STEP2_STARTER_LOW_MINUTES_PENALTY
    enable_continuity_prior: bool = STEP2_ENABLE_CONTINUITY_PRIOR
    continuity_kappa: float = STEP2_CONTINUITY_KAPPA
    continuity_default_returning: float = STEP2_CONTINUITY_DEFAULT_RETURNING
    continuity_starter_bonus: float = STEP2_CONTINUITY_STARTER_BONUS
    continuity_rotation_bonus: float = STEP2_CONTINUITY_ROTATION_BONUS
    candidate_limit: int = STEP2_CANDIDATE_LIMIT
    candidate_per_role: int = STEP2_CANDIDATE_PER_ROLE
    candidate_per_band: int = STEP2_CANDIDATE_PER_BAND
    fit_creator_bonus: float = STEP2_FIT_CREATOR_BONUS
    fit_spacing_bonus: float = STEP2_FIT_SPACING_BONUS
    fit_poa_bonus: float = STEP2_FIT_POA_BONUS
    fit_rim_bonus: float = STEP2_FIT_RIM_BONUS
    fit_stagger_bonus: float = STEP2_FIT_STAGGER_BONUS


def _load_team_map() -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    if not TEAMS_PATH.exists():
        return {}, {}, {}
    teams = pd.read_parquet(TEAMS_PATH)
    teams.columns = [str(c).lower() for c in teams.columns]

    full_to_abbr = {}
    abbr_to_conf = {}
    team_id_to_abbr = {}
    if "full_name" in teams.columns and "abbreviation" in teams.columns:
        for _, row in teams.iterrows():
            full = str(row.get("full_name", "")).strip()
            abbr = str(row.get("abbreviation", "")).strip().upper()
            if full and abbr:
                full_to_abbr[full] = abbr
    if "abbreviation" in teams.columns and "conference" in teams.columns:
        for _, row in teams.iterrows():
            abbr = str(row.get("abbreviation", "")).strip().upper()
            conf = str(row.get("conference", "")).strip().title()
            if abbr:
                abbr_to_conf[abbr] = conf or "Unknown"
    if "abbreviation" in teams.columns:
        for _, row in teams.iterrows():
            abbr = str(row.get("abbreviation", "")).strip().upper()
            if not abbr:
                continue
            for key in ["team_id", "id"]:
                if key in teams.columns:
                    raw_id = row.get(key)
                    clean_id = str(raw_id).strip().replace(".0", "")
                    if clean_id and clean_id.lower() not in {"nan", "none"}:
                        team_id_to_abbr[clean_id] = abbr
    return full_to_abbr, abbr_to_conf, team_id_to_abbr


def load_player_pool(cfg: Step2Config, source_path: Optional[Path] = None) -> pd.DataFrame:
    src = source_path or PLAYER_PROFILES_PATH
    df = pd.read_parquet(src)
    repl_mask = _replacement_pool_mask(df)
    if int(repl_mask.sum()) > 0:
        df = df.loc[~repl_mask].copy()

    df["player_id"] = _norm_id(df["player_id"])
    df["season"] = df["season"].astype(str)
    df["team_abbreviation"] = df["team_abbreviation"].astype(str).str.upper()

    for col in [cfg.impact_column, cfg.minutes_column, cfg.stability_column]:
        if col not in df.columns:
            df[col] = np.nan
    for col in ["off_primary_archetype", "def_primary_archetype"]:
        if col not in df.columns:
            df[col] = None

    out = df[
        [
            "player_id",
            "player_name",
            "season",
            "team_abbreviation",
            "team_id",
            "position_proxy",
            "off_primary_archetype",
            "def_primary_archetype",
            cfg.impact_column,
            cfg.minutes_column,
            cfg.stability_column,
        ]
    ].copy()
    out = out.rename(
        columns={
            cfg.impact_column: "impact_value",
            cfg.minutes_column: "minutes",
            cfg.stability_column: "impact_stability",
            "off_primary_archetype": "off_archetype",
            "def_primary_archetype": "def_archetype",
        }
    )

    out["impact_value"] = pd.to_numeric(out["impact_value"], errors="coerce").fillna(0.0)
    out["minutes"] = pd.to_numeric(out["minutes"], errors="coerce").fillna(0.0)
    out["impact_stability"] = pd.to_numeric(out["impact_stability"], errors="coerce").fillna(
        cfg.player_vol_stability_center
    )
    out["player_name"] = out["player_name"].map(_clean_name)
    out["off_archetype"] = out["off_archetype"].map(_clean_text)
    out["def_archetype"] = out["def_archetype"].map(_clean_text)

    out = out[(out["team_abbreviation"].notna()) & (out["team_abbreviation"] != "NAN")]
    out = out[out["minutes"] > 0]
    # Forecast-mode heuristic: projected profiles may carry historic
    # archetype labels like 'Insufficient Minutes' from prior seasons.
    # If a player has a forecasted minute projection that places them
    # in the Step-2 pool, treat the historic 'Insufficient Minutes'
    # label as unknown so the lineup model can consider them normally.
    try:
        if getattr(cfg, "forecast_mode", False):
            ins_mask = out["off_archetype"].fillna("").astype(str).str.strip() == "Insufficient Minutes"
            override_mask = ins_mask & (out["minutes"] >= float(cfg.min_mpg_for_pool))
            if override_mask.any():
                out.loc[override_mask, "off_archetype"] = None
            dins_mask = out["def_archetype"].fillna("").astype(str).str.strip() == "Insufficient Minutes"
            doverride_mask = dins_mask & (out["minutes"] >= float(cfg.min_mpg_for_pool))
            if doverride_mask.any():
                out.loc[doverride_mask, "def_archetype"] = None
    except Exception:
        # Be conservative: don't fail load on unexpected schema
        pass
    out = out.sort_values("minutes", ascending=False).drop_duplicates(
        subset=["season", "team_abbreviation", "player_id"], keep="first"
    )
    return out.reset_index(drop=True)


def load_positions() -> pd.DataFrame:
    if not POSITION_ESTIMATES_PATH.exists():
        return pd.DataFrame(columns=["season", "player_id"])

    pos = pd.read_parquet(POSITION_ESTIMATES_PATH)
    rename = {
        "PLAYER_ID": "player_id",
        "SEASON": "season",
    }
    pos = pos.rename(columns=rename)
    pos["player_id"] = _norm_id(pos["player_id"])
    pos["season"] = pos["season"].astype(str)

    keep = ["player_id", "season", "primary_position_estimate", "pct_pg", "pct_sg", "pct_sf", "pct_pf", "pct_c"]
    keep = [c for c in keep if c in pos.columns]
    pos = pos[keep].copy()

    for c in ["pct_pg", "pct_sg", "pct_sf", "pct_pf", "pct_c"]:
        if c not in pos.columns:
            pos[c] = 0.0
        pos[c] = pd.to_numeric(pos[c], errors="coerce").fillna(0.0)

    pos["guard_score"] = pos["pct_pg"] + pos["pct_sg"]
    pos["wing_score"] = pos["pct_sf"] + 0.25 * (pos["pct_sg"] + pos["pct_pf"])
    pos["big_score"] = pos["pct_pf"] + pos["pct_c"]

    role_cols = ["guard_score", "wing_score", "big_score"]
    role_names = {"guard_score": "Guard", "wing_score": "Wing", "big_score": "Big"}
    pos["role_from_position"] = pos[role_cols].idxmax(axis=1).map(role_names)

    # Fallback for flat score rows.
    flat = (pos[role_cols].sum(axis=1) <= 1e-9)
    if "primary_position_estimate" in pos.columns:
        pos.loc[flat, "role_from_position"] = pos.loc[flat, "primary_position_estimate"].map(_role_from_text)
    else:
        pos.loc[flat, "role_from_position"] = "Wing"

    if "primary_position_estimate" in pos.columns:
        pos["position_band"] = pos["primary_position_estimate"].map(_clean_text).fillna("Unknown")
    else:
        pos["position_band"] = pos["role_from_position"].map(
            {"Guard": "Guard", "Wing": "Forward", "Big": "Center"}
        ).fillna("Unknown")

    return pos[
        [
            "player_id",
            "season",
            "role_from_position",
            "position_band",
            "guard_score",
            "wing_score",
            "big_score",
        ]
    ]


def _previous_season(season: str) -> str:
    start_year = int(str(season)[:4]) - 1
    end_suffix = str(int(str(season)[:4]))[-2:]
    return f"{start_year}-{end_suffix}"


def _load_returning_flags(players: pd.DataFrame) -> pd.DataFrame:
    if players.empty or not PLAYER_PROFILES_PATH.exists():
        players = players.copy()
        players["is_returning_to_team"] = float(STEP2_CONTINUITY_DEFAULT_RETURNING)
        players["continuity_factor"] = 1.0
        return players

    hist = pd.read_parquet(PLAYER_PROFILES_PATH, columns=["player_id", "season", "team_abbreviation"])
    hist["player_id"] = _norm_id(hist["player_id"])
    hist["season"] = hist["season"].astype(str)
    hist["team_abbreviation"] = hist["team_abbreviation"].astype(str).str.upper()
    prev_map = {
        (str(row["player_id"]), str(row["season"])): str(row["team_abbreviation"])
        for _, row in hist.iterrows()
        if str(row["team_abbreviation"]).strip()
    }

    enriched = players.copy()
    prev_teams = []
    returning = []
    for _, row in enriched[["player_id", "season", "team_abbreviation"]].iterrows():
        prev_season = _previous_season(str(row["season"]))
        prev_team = prev_map.get((str(row["player_id"]), prev_season))
        prev_teams.append(prev_team)
        returning.append(float(prev_team == str(row["team_abbreviation"])))

    enriched["previous_team_abbreviation"] = prev_teams
    enriched["is_returning_to_team"] = pd.Series(returning, index=enriched.index, dtype=float).fillna(
        float(STEP2_CONTINUITY_DEFAULT_RETURNING)
    )
    enriched["continuity_factor"] = 1.0 - STEP2_CONTINUITY_KAPPA * (1.0 - enriched["is_returning_to_team"])
    return enriched


def _off_archetype_flags(frame: pd.DataFrame) -> Dict[str, pd.Series]:
    off = frame.get("off_archetype", pd.Series("", index=frame.index)).fillna("").astype(str)
    return {
        "creator": off.isin({"Ball Dominant Creator", "Ballhandler", "All-Around Scorer", "Perimeter Scorer"}),
        "spacer": off.isin({"Off-Ball Movement Shooter", "Off-Ball Stationary Shooter", "Perimeter Scorer", "PnR Popping Big"}),
        "rim": off.isin({"Interior Scorer", "Off-Ball Finisher", "PnR Rolling Big"}),
    }


def _def_archetype_flags(frame: pd.DataFrame) -> Dict[str, pd.Series]:
    deff = frame.get("def_archetype", pd.Series("", index=frame.index)).fillna("").astype(str)
    return {
        "poa": deff.isin({"POA Defender", "Wing Stopper", "Off-Ball Chaser"}),
        "rim": deff.isin({"Rim Protector", "Dropping Big", "Mobile Big"}),
    }


def _lineup_fit_components(lineup_df: pd.DataFrame, cfg: Step2Config) -> Dict[str, float]:
    if lineup_df.empty:
        return {
            "fit_score": 0.0,
            "creator_share": 0.0,
            "spacing_share": 0.0,
            "poa_share": 0.0,
            "rim_share": 0.0,
            "continuity_share": 0.0,
        }

    minutes = np.maximum(lineup_df["minutes"].to_numpy(dtype=float), 1e-3)
    weights = minutes / minutes.sum()
    off_flags = _off_archetype_flags(lineup_df)
    def_flags = _def_archetype_flags(lineup_df)

    creator_share = float(np.dot(off_flags["creator"].astype(float).to_numpy(dtype=float), weights))
    spacing_share = float(np.dot(off_flags["spacer"].astype(float).to_numpy(dtype=float), weights))
    poa_share = float(np.dot(def_flags["poa"].astype(float).to_numpy(dtype=float), weights))
    rim_share = float(np.dot(def_flags["rim"].astype(float).to_numpy(dtype=float), weights))
    continuity_share = float(
        np.dot(
            pd.to_numeric(lineup_df.get("is_returning_to_team"), errors="coerce").fillna(0.0).to_numpy(dtype=float),
            weights,
        )
    )
    staggerable = float(np.mean(lineup_df["minutes"].to_numpy(dtype=float) >= cfg.stagger_minutes_threshold))

    fit_score = (
        cfg.fit_creator_bonus * creator_share
        + cfg.fit_spacing_bonus * spacing_share
        + cfg.fit_poa_bonus * poa_share
        + cfg.fit_rim_bonus * rim_share
        + cfg.fit_stagger_bonus * staggerable
        + cfg.continuity_starter_bonus * continuity_share
    )
    return {
        "fit_score": float(fit_score),
        "creator_share": round(creator_share, 4),
        "spacing_share": round(spacing_share, 4),
        "poa_share": round(poa_share, 4),
        "rim_share": round(rim_share, 4),
        "continuity_share": round(continuity_share, 4),
    }


def _candidate_pool(
    players: pd.DataFrame,
    score_col: str,
    cfg: Step2Config,
    require_position_bands: Optional[Tuple[str, ...]] = None,
) -> pd.DataFrame:
    ranked = players.sort_values(score_col, ascending=False).reset_index(drop=True)
    if len(ranked) <= cfg.candidate_limit:
        return ranked.copy()

    pieces = [ranked.head(cfg.candidate_limit)]
    for _, role_frame in ranked.groupby("role", sort=False):
        pieces.append(role_frame.head(cfg.candidate_per_role))

    if require_position_bands:
        band_mask = ranked["position_band"].astype(str).isin(set(require_position_bands))
        if band_mask.any():
            pieces.append(ranked.loc[band_mask].head(cfg.candidate_per_band))

    combined = pd.concat(pieces, ignore_index=True)
    combined = combined.drop_duplicates(subset=["player_id"], keep="first")
    return combined.sort_values(score_col, ascending=False).reset_index(drop=True)


def load_clutch_stats() -> pd.DataFrame:
    if not CLUTCH_STATS_ALL_PATH.exists():
        return pd.DataFrame(columns=["player_id", "season", "team_abbreviation", "clutch_minutes", "clutch_gp"])
    clutch = pd.read_parquet(CLUTCH_STATS_ALL_PATH)
    clutch["player_id"] = _norm_id(clutch["player_id"])
    clutch["season"] = clutch["season"].astype(str)
    clutch["team_abbreviation"] = clutch["team_abbreviation"].astype(str).str.upper()
    for c in ["clutch_minutes", "clutch_gp"]:
        if c not in clutch.columns:
            clutch[c] = 0.0
        clutch[c] = pd.to_numeric(clutch[c], errors="coerce").fillna(0.0)

    clutch = clutch.sort_values("clutch_minutes", ascending=False).drop_duplicates(
        subset=["season", "team_abbreviation", "player_id"],
        keep="first",
    )
    return clutch[["player_id", "season", "team_abbreviation", "clutch_minutes", "clutch_gp"]]


def load_metrics_lineups(full_to_abbr: Dict[str, str]) -> pd.DataFrame:
    if not METRICS_LINEUPS_PATH.exists():
        return pd.DataFrame(columns=["season", "team_abbreviation", "NET_RTG", "total_poss", "lineup_ids"])
    lineups = pd.read_parquet(METRICS_LINEUPS_PATH)
    lineups["season"] = lineups["season"].astype(str)
    lineups["team_name"] = lineups["team_name"].astype(str)
    lineups["team_abbreviation"] = lineups["team_name"].map(full_to_abbr)
    lineups["NET_RTG"] = pd.to_numeric(lineups["NET_RTG"], errors="coerce")
    lineups["total_poss"] = pd.to_numeric(lineups["total_poss"], errors="coerce").fillna(0.0)
    lineups["lineup_ids"] = lineups["lineup_ids"].apply(_to_id_list)
    lineups = lineups.dropna(subset=["team_abbreviation", "NET_RTG"])
    return lineups[["season", "team_abbreviation", "NET_RTG", "total_poss", "lineup_ids"]]


def _select_best_lineup(
    pool: pd.DataFrame,
    score_col: str,
    lineup_size: int,
    cfg: Step2Config,
    require_position_bands: Optional[Tuple[str, ...]] = None,
) -> pd.DataFrame:
    players = _candidate_pool(pool, score_col, cfg, require_position_bands=require_position_bands)
    if len(players) <= lineup_size:
        return players.head(lineup_size).copy()

    best_idx = None
    best_score = -np.inf

    for combo in combinations(range(len(players)), lineup_size):
        chunk = players.iloc[list(combo)]
        roles = set(chunk["role"].tolist())
        if not {"Guard", "Wing", "Big"}.issubset(roles):
            continue
        if require_position_bands:
            bands = set(chunk["position_band"].astype(str).tolist())
            if not bands.intersection(set(require_position_bands)):
                continue
        fit = _lineup_fit_components(chunk, cfg)
        score = float(chunk[score_col].sum()) + float(fit["fit_score"])
        if score > best_score:
            best_score = score
            best_idx = combo

    if best_idx is None:
        return players.head(lineup_size).copy()
    return players.iloc[list(best_idx)].copy()


def _select_clutch_constrained(
    candidates: pd.DataFrame,
    starter_ids: Set[str],
    lineup_size: int,
    min_starters: int,
    max_swaps: int,
    cfg: Step2Config,
) -> pd.DataFrame:
    """Select clutch lineup from candidates with starter carryover constraint.

    Requires at least `min_starters` from the starter lineup and at most
    `max_swaps` non-starters.  Falls back to best-score if no valid combo
    satisfies the G/W/B coverage requirement.
    """
    players = _candidate_pool(candidates, "clutch_score", cfg)
    if len(players) <= lineup_size:
        return players.head(lineup_size).copy()

    best_idx = None
    best_score = -np.inf

    for combo in combinations(range(len(players)), lineup_size):
        chunk = players.iloc[list(combo)]
        roles = set(chunk["role"].tolist())
        if not {"Guard", "Wing", "Big"}.issubset(roles):
            continue
        n_starters = sum(1 for pid in chunk["player_id"].astype(str) if pid in starter_ids)
        if n_starters < min_starters:
            continue
        n_swaps = lineup_size - n_starters
        if n_swaps > max_swaps:
            continue
        fit = _lineup_fit_components(chunk, cfg)
        score = float(chunk["clutch_score"].sum()) + float(fit["fit_score"])
        if score > best_score:
            best_score = score
            best_idx = combo

    if best_idx is None:
        # Fallback: relax constraint, just use best score with G/W/B coverage
        return _select_best_lineup(candidates, "clutch_score", lineup_size, cfg)
    return players.iloc[list(best_idx)].copy()


def _lineup_strength(lineup_df: pd.DataFrame) -> Tuple[float, float]:
    if lineup_df.empty:
        return 0.0, 0.0
    weights = np.maximum(lineup_df["minutes"].to_numpy(dtype=float), 1e-3)
    impact = lineup_df["impact_value"].to_numpy(dtype=float)
    mu = _weighted_avg(impact, weights)
    sigma_sq = float(np.mean(np.square(lineup_df["player_volatility"].to_numpy(dtype=float))))
    sigma = math.sqrt(max(sigma_sq, 0.0))
    return mu, sigma


def build_team_profiles(pool: pd.DataFrame, cfg: Step2Config, abbr_to_conf: Dict[str, str]) -> List[Dict]:
    rows: List[Dict] = []

    grouped = pool.groupby(["season", "team_abbreviation"], sort=True)
    for (season, team), g in grouped:
        team_pool = g.copy()
        team_pool = team_pool.sort_values("minutes", ascending=False)

        lineup_pool = team_pool[team_pool["minutes"] >= cfg.min_mpg_for_pool].copy()
        if len(lineup_pool) < cfg.lineup_size:
            lineup_pool = team_pool.head(cfg.lineup_size).copy()
        if len(lineup_pool) < cfg.lineup_size:
            continue

        lineup_pool["minutes_norm"] = lineup_pool["minutes"] / max(float(lineup_pool["minutes"].max()), 1e-6)
        lineup_pool["impact_norm"] = _minmax(lineup_pool["impact_value"])
        team_mean_impact = _weighted_avg(
            lineup_pool["impact_value"].to_numpy(dtype=float),
            np.maximum(lineup_pool["minutes"].to_numpy(dtype=float), 1e-3),
        )
        lineup_pool["is_returning_to_team"] = pd.to_numeric(
            lineup_pool.get("is_returning_to_team"), errors="coerce"
        ).fillna(cfg.continuity_default_returning)
        lineup_pool["continuity_factor"] = pd.to_numeric(
            lineup_pool.get("continuity_factor"), errors="coerce"
        ).fillna(1.0)

        team_clutch_total = float(lineup_pool["clutch_minutes"].sum())
        if team_clutch_total <= 1e-9:
            lineup_pool["clutch_share"] = 0.0
        else:
            lineup_pool["clutch_share"] = lineup_pool["clutch_minutes"] / team_clutch_total

        role_counts = lineup_pool["role"].value_counts()
        lineup_pool["pos_scarcity_raw"] = lineup_pool["role"].map(lambda r: 1.0 / max(int(role_counts.get(r, 1)), 1))
        lineup_pool["pos_scarcity"] = _minmax(lineup_pool["pos_scarcity_raw"])

        lineup_pool["player_volatility"] = np.clip(
            cfg.player_vol_base + cfg.player_vol_stability_scale * (cfg.player_vol_stability_center - lineup_pool["impact_stability"]),
            cfg.player_vol_floor,
            cfg.player_vol_ceiling,
        )

        # --- Scoring formulas differ between backtest and forecast modes ---
        if cfg.forecast_mode:
            # Forecast: no clutch_minutes available — score purely by impact + minutes
            # Starter: redistribute clutch weight to minutes + impact
            lineup_pool["starter_score"] = (
                0.55 * lineup_pool["minutes_norm"]
                + 0.30 * lineup_pool["impact_norm"]
                + 0.15 * lineup_pool["pos_scarcity"]
            )
        else:
            # Backtest: original formula with clutch_share contribution
            lineup_pool["clutch_score"] = (
                cfg.clutch_weight_c * lineup_pool["clutch_share"]
                + cfg.clutch_weight_talent
                * (
                    cfg.clutch_weight_i * lineup_pool["impact_norm"]
                    + cfg.clutch_weight_m * lineup_pool["minutes_norm"]
                )
            )

            lineup_pool["starter_score"] = (
                cfg.starter_weight_m * lineup_pool["minutes_norm"]
                + cfg.starter_weight_i * lineup_pool["impact_norm"]
                + cfg.starter_weight_pos * lineup_pool["pos_scarcity"]
                + cfg.starter_weight_c * lineup_pool["clutch_share"]
            )

        if cfg.enable_continuity_prior:
            lineup_pool["starter_score"] = lineup_pool["starter_score"] + (
                cfg.continuity_starter_bonus * lineup_pool["is_returning_to_team"]
            )
            lineup_pool["clutch_score"] = lineup_pool.get("clutch_score", pd.Series(0.0, index=lineup_pool.index)) + (
                0.5 * cfg.continuity_starter_bonus * lineup_pool["is_returning_to_team"]
            )

        # --- Constraint: penalize low-minutes starter candidates ---
        if cfg.low_minutes_penalty > 1.0:
            low_min_mask = lineup_pool["minutes"] < cfg.low_minutes_threshold
            lineup_pool.loc[low_min_mask, "starter_score"] = (
                lineup_pool.loc[low_min_mask, "starter_score"] / cfg.low_minutes_penalty
            )

        # --- Select starters ---
        big_bands = cfg.true_big_bands if cfg.require_true_big else None
        starter_lineup = _select_best_lineup(
            lineup_pool, "starter_score", cfg.lineup_size, cfg,
            require_position_bands=big_bands,
        )
        starter_mu, starter_sigma = _lineup_strength(starter_lineup)
        starter_ids = set(starter_lineup["player_id"].astype(str))
        starter_fit = _lineup_fit_components(starter_lineup, cfg)

        # --- Select clutch lineup ---
        if cfg.forecast_mode:
            # Forecast clutch: starters + high-impact players are clutch candidates.
            # Score by impact with a starter-indicator bonus.
            lineup_pool["_is_starter"] = lineup_pool["player_id"].astype(str).isin(starter_ids).astype(float)
            lineup_pool["clutch_score"] = (
                0.60 * lineup_pool["impact_norm"]
                + 0.20 * lineup_pool["minutes_norm"]
                + 0.20 * lineup_pool["_is_starter"]
            )
            if cfg.enable_continuity_prior:
                lineup_pool["clutch_score"] = lineup_pool["clutch_score"] + (
                    0.5 * cfg.continuity_starter_bonus * lineup_pool["is_returning_to_team"]
                )
            clutch_lineup = _select_best_lineup(lineup_pool, "clutch_score", cfg.lineup_size, cfg)
        elif cfg.enable_clutch_core and not cfg.forecast_mode:
            # v1 clutch core constraint: select from top-N candidates,
            # requiring >=min_starters from the starter lineup, with max swaps.
            candidates = lineup_pool.sort_values("clutch_score", ascending=False).head(cfg.clutch_candidate_size).copy()
            # Ensure all starters are in the candidate pool
            for sid in starter_ids:
                if sid not in candidates["player_id"].astype(str).values:
                    starter_row = lineup_pool[lineup_pool["player_id"].astype(str) == sid]
                    if not starter_row.empty:
                        candidates = pd.concat([candidates, starter_row.head(1)], ignore_index=True)
            candidates = candidates.drop_duplicates(subset=["player_id"], keep="first")
            clutch_lineup = _select_clutch_constrained(
                candidates, starter_ids, cfg.lineup_size,
                cfg.clutch_min_starters, cfg.clutch_max_swaps, cfg,
            )
        else:
            clutch_lineup = _select_best_lineup(lineup_pool, "clutch_score", cfg.lineup_size, cfg)

        clutch_mu, clutch_sigma = _lineup_strength(clutch_lineup)
        clutch_fit = _lineup_fit_components(clutch_lineup, cfg)

        # --- Rotation model ---
        bench = lineup_pool[~lineup_pool["player_id"].astype(str).isin(starter_ids)].copy()

        bench_mu = _weighted_avg(bench["impact_value"].to_numpy(dtype=float), np.maximum(bench["minutes"].to_numpy(dtype=float), 1e-3))
        if bench.empty:
            bench_mu = starter_mu
            bench_sigma_sq = float(np.mean(np.square(starter_lineup["player_volatility"].to_numpy(dtype=float))))
        else:
            bench_sigma_sq = _weighted_avg(
                np.square(bench["player_volatility"].to_numpy(dtype=float)),
                np.maximum(bench["minutes"].to_numpy(dtype=float), 1e-3),
            )

        stagger_weights = np.maximum(starter_lineup["minutes"].to_numpy(dtype=float) - cfg.stagger_minutes_threshold, 0.0)
        if float(stagger_weights.sum()) <= 1e-9:
            stagger_mu = starter_mu
            stagger_sigma_sq = float(np.mean(np.square(starter_lineup["player_volatility"].to_numpy(dtype=float))))
        else:
            stagger_mu = _weighted_avg(starter_lineup["impact_value"].to_numpy(dtype=float), stagger_weights)
            stagger_sigma_sq = _weighted_avg(
                np.square(starter_lineup["player_volatility"].to_numpy(dtype=float)),
                stagger_weights,
            )

        if cfg.enable_rotation_regime:
            # Rotation regime model: sigmoid-based regime detection
            # p_stagger = sigmoid(star_coef * S - bench_coef * B)
            # where S = max starter impact_norm, B = mean bench impact_norm
            star_max = float(starter_lineup["impact_norm"].max()) if not starter_lineup.empty else 0.5
            bench_mean = float(bench["impact_norm"].mean()) if not bench.empty else 0.5
            regime_logit = cfg.regime_star_coef * star_max - cfg.regime_bench_coef * bench_mean
            p_stagger = 1.0 / (1.0 + math.exp(-regime_logit))

            # Regime-blended rotation:
            # In star-heavy regime (high p_stagger): stagger dominates
            # In bench-deep regime (low p_stagger): bench dominates
            stagger_blend_mu = cfg.rot_stagger_start_w * stagger_mu + cfg.rot_stagger_bench_w * bench_mu
            bench_blend_mu = cfg.rot_bench_start_w * stagger_mu + cfg.rot_bench_bench_w * bench_mu
            rotation_mu = p_stagger * stagger_blend_mu + (1.0 - p_stagger) * bench_blend_mu
            rotation_sigma = math.sqrt(max(
                cfg.rot_sigma_base * stagger_sigma_sq + cfg.rot_sigma_bench_w * bench_sigma_sq,
                0.0,
            ))
        else:
            rotation_mu = cfg.rotation_alpha * stagger_mu + (1.0 - cfg.rotation_alpha) * bench_mu
            rotation_sigma = math.sqrt(max(bench_sigma_sq + 0.5 * stagger_sigma_sq, 0.0))

        if cfg.enable_continuity_prior and not bench.empty:
            bench_continuity = _weighted_avg(
                pd.to_numeric(bench.get("is_returning_to_team"), errors="coerce").fillna(0.0).to_numpy(dtype=float),
                np.maximum(bench["minutes"].to_numpy(dtype=float), 1e-3),
            )
            rotation_mu += cfg.continuity_rotation_bonus * bench_continuity
        else:
            bench_continuity = float(starter_fit["continuity_share"])

        def _pack_players(df: pd.DataFrame, score_col: str) -> List[Dict]:
            frame = df.sort_values(score_col, ascending=False)
            packed = []
            for _, row in frame.iterrows():
                player_name = _clean_name(row["player_name"])
                packed.append(
                    {
                        "player_id": str(row["player_id"]),
                        "player_name": player_name,
                        "role": str(row["role"]),
                        "position_band": _clean_text(row.get("position_band")),
                        "off_archetype": _clean_text(row.get("off_archetype")),
                        "def_archetype": _clean_text(row.get("def_archetype")),
                        "minutes": _safe_float(row["minutes"], 2),
                        "impact": _safe_float(row["impact_value"], 4),
                        "clutch_minutes": _safe_float(row.get("clutch_minutes", 0), 2),
                        "score": _safe_float(row[score_col], 4),
                    }
                )
            return packed

        def _pack_pool(df: pd.DataFrame) -> List[Dict]:
            frame = df.sort_values("minutes", ascending=False)
            packed: List[Dict] = []
            seen_ids: Set[str] = set()
            for _, row in frame.iterrows():
                player_id = str(row["player_id"])
                if player_id in seen_ids:
                    continue
                seen_ids.add(player_id)
                player_name = _clean_name(row["player_name"])
                if not player_name:
                    continue
                packed.append(
                    {
                        "player_id": player_id,
                        "player_name": player_name,
                        "position_band": _clean_text(row.get("position_band")),
                        "off_archetype": _clean_text(row.get("off_archetype")),
                        "def_archetype": _clean_text(row.get("def_archetype")),
                        "minutes": _safe_float(row["minutes"], 2),
                    }
                )
            return packed

        rows.append(
            {
                "season": season,
                "team_abbreviation": team,
                "conference": abbr_to_conf.get(team, "Unknown"),
                "n_players_pool": int(len(lineup_pool)),
                "team_clutch_minutes_total": _safe_float(team_clutch_total, 2),
                "mu_start": _safe_float(starter_mu),
                "sigma_start": _safe_float(starter_sigma),
                "mu_rotation": _safe_float(rotation_mu),
                "sigma_rotation": _safe_float(rotation_sigma),
                "mu_clutch": _safe_float(clutch_mu),
                "sigma_clutch": _safe_float(clutch_sigma),
                "team_mean_impact": _safe_float(team_mean_impact),
                "starter_delta_impact": _safe_float(starter_mu - team_mean_impact),
                "rotation_delta_impact": _safe_float(rotation_mu - team_mean_impact),
                "clutch_delta_impact": _safe_float(clutch_mu - starter_mu),
                "starter_fit_score": _safe_float(starter_fit["fit_score"]),
                "clutch_fit_score": _safe_float(clutch_fit["fit_score"]),
                "continuity_score": _safe_float(starter_fit["continuity_share"]),
                "bench_continuity_score": _safe_float(bench_continuity),
                "starter_player_ids": [str(x) for x in starter_lineup["player_id"].tolist()],
                "clutch_player_ids": [str(x) for x in clutch_lineup["player_id"].tolist()],
                "starter_players": _pack_players(starter_lineup, "starter_score"),
                "clutch_players": _pack_players(clutch_lineup, "clutch_score"),
                "pool_players": _pack_pool(lineup_pool),
                "fit_components": {
                    "starter": starter_fit,
                    "clutch": clutch_fit,
                },
                "rotation_breakdown": {
                    "bench_impact": _safe_float(bench_mu),
                    "stagger_impact": _safe_float(stagger_mu),
                    "bench_sigma_sq": _safe_float(bench_sigma_sq),
                    "stagger_sigma_sq": _safe_float(stagger_sigma_sq),
                    "alpha": cfg.rotation_alpha,
                },
            }
        )

    return rows


def build_actual_maps(
    lineups: pd.DataFrame,
    clutch: pd.DataFrame,
    predicted_rows: List[Dict],
    starter_games: pd.DataFrame,
) -> Tuple[
    Dict[Tuple[str, str], Set[str]],
    Dict[Tuple[str, str], Set[str]],
    Dict[Tuple[str, str], float],
    Dict[Tuple[str, str], Dict],
]:
    actual_starters, starter_meta = build_observed_starters_from_pbp(
        starter_games=starter_games,
        predicted_rows=predicted_rows,
    )
    actual_clutch: Dict[Tuple[str, str], Set[str]] = {}
    actual_rotation_net: Dict[Tuple[str, str], float] = {}

    # Clutch proxy: top-5 clutch-minute players per team-season.
    if not clutch.empty:
        grouped = clutch.groupby(["season", "team_abbreviation"], sort=False)
        for (season, team), g in grouped:
            top5 = g.sort_values("clutch_minutes", ascending=False).head(LINEUP_SIZE)
            ids = set(top5["player_id"].astype(str).tolist())
            if ids:
                actual_clutch[(str(season), str(team))] = ids

    # Rotation proxy: weighted average impact_total_impact of bench (non-starter)
    # players from actual profiles.  This measures how good the teams' rotational
    # minutes actually were, on the same BKE scale as the predicted mu_rotation.
    # Using player-level profiles (deduced from game logs) rather than lineup-level
    # NET_RTG avoids noise from specific 5-man combination sample sizes.
    try:
        actual_profiles = pd.read_parquet(
            PLAYER_PROFILES_PATH,
            columns=["player_id", "season", "team_abbreviation",
                      "minutes", "impact_total_impact"],
        )
        actual_profiles["player_id"] = actual_profiles["player_id"].astype(str).str.strip()
        actual_profiles["impact_total_impact"] = pd.to_numeric(
            actual_profiles["impact_total_impact"], errors="coerce"
        ).fillna(0.0)
        # Normalize actual minutes into MPG for consistent weighting with
        # forecasted profiles (which use MPG). If the parquet contains a
        # season-total minutes column, convert to MPG using available games
        # information or a safe heuristic divisor.
        actual_profiles["minutes"] = pd.to_numeric(
            actual_profiles.get("minutes"), errors="coerce"
        ).fillna(0.0)
        if "mpg" in actual_profiles.columns:
            actual_profiles["mpg"] = pd.to_numeric(actual_profiles.get("mpg"), errors="coerce").fillna(0.0)
        else:
            # Prefer an explicit games column if present; otherwise infer.
            if "games" in actual_profiles.columns:
                games = pd.to_numeric(actual_profiles.get("games"), errors="coerce").fillna(82.0)
                # avoid division by zero
                games = games.replace(0.0, 82.0)
                actual_profiles["mpg"] = actual_profiles["minutes"] / games
            else:
                # Heuristic: values > 200 look like season-total minutes; convert
                # to per-game by dividing by 82. Otherwise assume value already MPG.
                actual_profiles["mpg"] = actual_profiles["minutes"].where(
                    actual_profiles["minutes"] <= 200, actual_profiles["minutes"] / 82.0
                )

        for (season, team), g in actual_profiles.groupby(
            ["season", "team_abbreviation"], sort=False
        ):
            key = (str(season), str(team))
            starters = actual_starters.get(key)
            if not starters:
                continue
            bench = g[~g["player_id"].isin(starters)].copy()
            if bench.empty:
                continue
            # Use MPG for weighting so predicted and actual are unit-consistent.
            mins = bench.get("mpg", bench["minutes"]).to_numpy(dtype=float)
            if mins.sum() <= 0:
                continue
            rating = _weighted_avg(
                bench["impact_total_impact"].to_numpy(dtype=float),
                np.maximum(mins, 1e-3),
            )
            actual_rotation_net[key] = rating
    except Exception:
        # Fallback to lineup-based rotation if profiles unavailable
        predicted_map = {(r["season"], r["team_abbreviation"]): r for r in predicted_rows}
        for (season, team), g in lineups.groupby(["season", "team_abbreviation"], sort=False):
            key = (str(season), str(team))
            starters = actual_starters.get(key)
            if starters is None:
                pred = predicted_map.get(key, {})
                starters = set(pred.get("starter_player_ids", []))
            if not starters:
                continue
            tmp = g.copy()
            tmp["overlap"] = tmp["lineup_ids"].apply(lambda ids: len(set(_to_id_list(ids)) & starters))
            candidates = tmp[(tmp["overlap"] <= 2) & (tmp["total_poss"] >= 50)]
            if candidates.empty:
                continue
            rating = _weighted_avg(candidates["NET_RTG"].to_numpy(dtype=float), candidates["total_poss"].to_numpy(dtype=float))
            actual_rotation_net[key] = rating

    return actual_starters, actual_clutch, actual_rotation_net, starter_meta


def attach_validation(
    predicted_rows: List[Dict],
    actual_starters: Dict[Tuple[str, str], Set[str]],
    actual_clutch: Dict[Tuple[str, str], Set[str]],
    actual_rotation_net: Dict[Tuple[str, str], float],
    starter_meta: Optional[Dict[Tuple[str, str], Dict]] = None,
    player_name_lookup: Optional[Dict[Tuple[str, str, str], str]] = None,
) -> Dict:
    team_metrics = []

    for row in predicted_rows:
        key = (row["season"], row["team_abbreviation"])
        pred_starters = set([str(x) for x in row.get("starter_player_ids", [])])
        pred_clutch = set([str(x) for x in row.get("clutch_player_ids", [])])
        pool_name_map = {}
        for p in row.get("pool_players", []):
            pid = str(p.get("player_id", "")).strip()
            name = _clean_name(p.get("player_name"))
            if pid and name:
                pool_name_map[pid] = name

        starter_true = actual_starters.get(key)
        clutch_true = actual_clutch.get(key)
        starter_info = starter_meta.get(key, {}) if starter_meta else {}

        def _name_list(id_set: Optional[Set[str]]) -> List[str]:
            if not id_set:
                return []
            names = []
            for pid in sorted(list(id_set)):
                name = pool_name_map.get(str(pid))
                if not name and player_name_lookup:
                    name = _clean_name(player_name_lookup.get((str(row["season"]), str(row["team_abbreviation"]), str(pid))))
                if not name:
                    name = f"ID {pid}"
                names.append(name)
            return names

        actual_starter_names = _name_list(starter_true)
        actual_clutch_names = _name_list(clutch_true)

        starter_overlap = None
        starter_overlap_rate = None
        if starter_true:
            starter_overlap = len(pred_starters & starter_true)
            starter_overlap_rate = starter_overlap / float(LINEUP_SIZE)

        clutch_overlap = None
        clutch_overlap_rate = None
        if clutch_true:
            clutch_overlap = len(pred_clutch & clutch_true)
            clutch_overlap_rate = clutch_overlap / float(LINEUP_SIZE)

        rot_actual = actual_rotation_net.get(key)

        row["validation"] = {
            "starter_overlap": starter_overlap,
            "starter_overlap_rate": _safe_float(starter_overlap_rate),
            "actual_starter_ids": sorted(list(starter_true)) if starter_true else [],
            "actual_starter_names": actual_starter_names,
            "actual_starter_count": int(len(starter_true)) if starter_true else 0,
            "starter_target_source": starter_info.get("selection_method"),
            "starter_games_total": starter_info.get("n_games_total"),
            "starter_games_verified": starter_info.get("n_games_verified"),
            "clutch_overlap": clutch_overlap,
            "clutch_overlap_rate": _safe_float(clutch_overlap_rate),
            "actual_clutch_ids": sorted(list(clutch_true)) if clutch_true else [],
            "actual_clutch_names": actual_clutch_names,
            "rotation_actual_net": _safe_float(rot_actual),
            "rotation_predicted": row.get("mu_rotation"),
        }

        team_metrics.append(
            {
                "season": row["season"],
                "team_abbreviation": row["team_abbreviation"],
                "starter_overlap": starter_overlap,
                "starter_overlap_rate": starter_overlap_rate,
                "actual_starter_count": int(len(starter_true)) if starter_true else None,
                "clutch_overlap": clutch_overlap,
                "clutch_overlap_rate": clutch_overlap_rate,
                "rotation_predicted": row.get("mu_rotation"),
                "rotation_actual_net": rot_actual,
            }
        )

    metrics_df = pd.DataFrame(team_metrics)

    def _corr(a: pd.Series, b: pd.Series) -> Optional[float]:
        m = a.notna() & b.notna()
        if int(m.sum()) < 3:
            return None
        value = np.corrcoef(a[m].astype(float), b[m].astype(float))[0, 1]
        return _safe_float(value)

    def _season_summary(df: pd.DataFrame) -> Dict:
        starter_count = pd.to_numeric(df.get("actual_starter_count"), errors="coerce")
        exact5 = (starter_count == LINEUP_SIZE)
        exact5_rate = float(exact5.mean()) if len(starter_count) else np.nan
        return {
            "n_teams": int(len(df)),
            "starter_overlap_rate_mean": _safe_float(df["starter_overlap_rate"].mean()),
            "starter_overlap_count_mean": _safe_float(df["starter_overlap"].mean(), 3),
            "observed_starter_size_mean": _safe_float(starter_count.mean(), 3),
            "observed_starter_exact5_rate": _safe_float(exact5_rate),
            "observed_starter_exact5_count": int(exact5.sum()),
            "clutch_overlap_rate_mean": _safe_float(df["clutch_overlap_rate"].mean()),
            "clutch_overlap_count_mean": _safe_float(df["clutch_overlap"].mean(), 3),
            "rotation_corr": _corr(df["rotation_predicted"], df["rotation_actual_net"]),
            "starter_target_met": bool((df["starter_overlap_rate"].mean(skipna=True) or 0) >= 0.80),
            "clutch_target_met": bool((df["clutch_overlap"].mean(skipna=True) or 0) >= 3.5),
            "rotation_target_met": bool(((_corr(df["rotation_predicted"], df["rotation_actual_net"]) or -1.0) >= 0.70)),
        }

    per_season = {}
    for season, sdf in metrics_df.groupby("season", sort=True):
        per_season[str(season)] = _season_summary(sdf)

    overall = _season_summary(metrics_df)
    overall.update(
        {
            "n_team_seasons": int(len(metrics_df)),
            "starter_overlap_available": int(metrics_df["starter_overlap"].notna().sum()),
            "starter_exact5_available": int(metrics_df["actual_starter_count"].notna().sum()),
            "starter_exact5_count": int((pd.to_numeric(metrics_df["actual_starter_count"], errors="coerce") == LINEUP_SIZE).sum()),
            "clutch_overlap_available": int(metrics_df["clutch_overlap"].notna().sum()),
            "rotation_actual_available": int(metrics_df["rotation_actual_net"].notna().sum()),
        }
    )

    return {
        "overall": overall,
        "per_season": per_season,
        "team_metrics": metrics_df.sort_values(["season", "team_abbreviation"]).to_dict(orient="records"),
    }


def build_projected_lineup_rows(
    forecast_mode: bool = False,
    profiles_path: Optional[Path] = None,
) -> Tuple[List[Dict], pd.DataFrame, Dict[str, str], Dict[str, str], Dict[str, str]]:
    cfg = Step2Config(forecast_mode=forecast_mode)
    full_to_abbr, abbr_to_conf, team_id_to_abbr = _load_team_map()

    from src.player_eval.constants import PROJECTED_PROFILES_PATH

    if forecast_mode:
        src = profiles_path or PROJECTED_PROFILES_PATH
        if not src.exists():
            raise FileNotFoundError(f"Projected profiles not found: {src}")
        players = load_player_pool(cfg, source_path=src)
    else:
        players = load_player_pool(cfg)

    positions = load_positions()
    if forecast_mode:
        clutch_model = pd.DataFrame(columns=["player_id", "season", "team_abbreviation", "clutch_minutes", "clutch_gp"])
    else:
        clutch_model = load_clutch_stats()

    players = players.merge(positions, on=["player_id", "season"], how="left")
    players = players.merge(
        clutch_model,
        on=["player_id", "season", "team_abbreviation"],
        how="left",
    )
    players["clutch_minutes"] = pd.to_numeric(players.get("clutch_minutes"), errors="coerce").fillna(0.0)
    players["clutch_gp"] = pd.to_numeric(players.get("clutch_gp"), errors="coerce").fillna(0.0)
    players = _load_returning_flags(players)

    players["role"] = players["role_from_position"]
    role_missing = players["role"].isna() | (players["role"].astype(str).str.strip() == "")
    players.loc[role_missing, "role"] = players.loc[role_missing, "position_proxy"].map(_role_from_text)
    players["role"] = players["role"].fillna("Wing")

    predicted_rows = build_team_profiles(players, cfg, abbr_to_conf)
    return predicted_rows, players, full_to_abbr, abbr_to_conf, team_id_to_abbr


def main(
    forecast_mode: bool = False,
    profiles_path: Path = None,
    profiles_output_path: Path = None,
    report_output_path: Path = None,
    validation_output_path: Path = None,
) -> None:
    mode_label = "FORECAST" if forecast_mode else "BACKTEST"
    print(f"Simulation Core — Step 2: Lineup Projection [{mode_label}]")
    cfg = Step2Config(forecast_mode=forecast_mode)

    from src.simulation.simulation_config import (
        FORECAST_LINEUP_PROFILES_PATH,
        FORECAST_LINEUP_REPORT_PATH,
        FORECAST_VALIDATION_PATH,
    )
    predicted_rows, players, full_to_abbr, abbr_to_conf, team_id_to_abbr = build_projected_lineup_rows(
        forecast_mode=forecast_mode,
        profiles_path=profiles_path,
    )

    if forecast_mode:
        clutch_validation = load_clutch_stats()
    else:
        clutch_validation = load_clutch_stats()

    lineups = load_metrics_lineups(full_to_abbr)
    starter_games = load_pbp_q1_starter_games(team_id_to_abbr)

    print(f"  Players loaded: {len(players)}")
    print(f"  Positions loaded: {players[['player_id', 'season']].drop_duplicates().shape[0]}")
    if forecast_mode:
        print("  Forecast scoring clutch source: disabled (leakage-safe)")
    print(f"  Clutch rows loaded for validation: {len(clutch_validation)}")
    print(f"  Lineup rows loaded for validation: {len(lineups)}")
    print(f"  Q1 starter game rows loaded for validation: {len(starter_games)}")

    player_name_lookup: Dict[Tuple[str, str, str], str] = {}
    for _, p in players[["season", "team_abbreviation", "player_id", "player_name"]].iterrows():
        pid = str(p.get("player_id", "")).strip()
        if not pid:
            continue
        name = _clean_name(p.get("player_name"))
        if not name:
            continue
        key = (str(p.get("season", "")), str(p.get("team_abbreviation", "")).upper(), pid)
        player_name_lookup[key] = name

    print(f"  Team-season profiles built: {len(predicted_rows)}")

    actual_starters, actual_clutch, actual_rotation_net, starter_meta = build_actual_maps(
        lineups=lineups,
        clutch=clutch_validation,
        predicted_rows=predicted_rows,
        starter_games=starter_games,
    )

    validation = attach_validation(
        predicted_rows=predicted_rows,
        actual_starters=actual_starters,
        actual_clutch=actual_clutch,
        actual_rotation_net=actual_rotation_net,
        starter_meta=starter_meta,
        player_name_lookup=player_name_lookup,
    )

    # Flatten for parquet convenience.
    flat_rows = []
    for row in predicted_rows:
        flat_rows.append(
            {
                "season": row["season"],
                "team_abbreviation": row["team_abbreviation"],
                "conference": row["conference"],
                "n_players_pool": row["n_players_pool"],
                "team_clutch_minutes_total": row["team_clutch_minutes_total"],
                "mu_start": row["mu_start"],
                "sigma_start": row["sigma_start"],
                "mu_rotation": row["mu_rotation"],
                "sigma_rotation": row["sigma_rotation"],
                "mu_clutch": row["mu_clutch"],
                "sigma_clutch": row["sigma_clutch"],
                "team_mean_impact": row.get("team_mean_impact"),
                "starter_delta_impact": row.get("starter_delta_impact"),
                "rotation_delta_impact": row.get("rotation_delta_impact"),
                "clutch_delta_impact": row.get("clutch_delta_impact"),
                "starter_fit_score": row.get("starter_fit_score"),
                "clutch_fit_score": row.get("clutch_fit_score"),
                "continuity_score": row.get("continuity_score"),
                "bench_continuity_score": row.get("bench_continuity_score"),
                "starter_overlap": row.get("validation", {}).get("starter_overlap"),
                "starter_overlap_rate": row.get("validation", {}).get("starter_overlap_rate"),
                "clutch_overlap": row.get("validation", {}).get("clutch_overlap"),
                "clutch_overlap_rate": row.get("validation", {}).get("clutch_overlap_rate"),
                "rotation_actual_net": row.get("validation", {}).get("rotation_actual_net"),
                "starter_player_ids": json.dumps(row.get("starter_player_ids", [])),
                "clutch_player_ids": json.dumps(row.get("clutch_player_ids", [])),
            }
        )

    flat_df = pd.DataFrame(flat_rows)

    # Choose output paths based on mode
    profiles_dst = profiles_output_path or (
        FORECAST_LINEUP_PROFILES_PATH if forecast_mode else STEP2_LINEUP_PROFILES_PATH
    )
    report_dst = report_output_path or (
        FORECAST_LINEUP_REPORT_PATH if forecast_mode else STEP2_LINEUP_REPORT_PATH
    )
    validation_dst = validation_output_path or (
        FORECAST_VALIDATION_PATH if forecast_mode else STEP2_VALIDATION_PATH
    )

    profiles_dst.parent.mkdir(parents=True, exist_ok=True)
    flat_df.to_parquet(profiles_dst, index=False)

    # Build season-keyed JSON for frontend rendering.
    seasons_blob: Dict[str, Dict] = {}
    for season in sorted({r["season"] for r in predicted_rows}):
        teams = [r for r in predicted_rows if r["season"] == season]
        season_summary = validation["per_season"].get(season, {})
        seasons_blob[season] = {
            "summary": season_summary,
            "teams": sorted(teams, key=lambda x: x["team_abbreviation"]),
        }

    report_blob = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "config": asdict(cfg),
        "validation_targets": {
            "starter": "First-quarter starters from PBP (players on court at first valid Q1 event), verified with Q1 substitution cluster transitions.",
            "rotation": "Minute-weighted impact_total_impact of non-starter players from profile data (lineup NET_RTG fallback only if profile source is unavailable).",
            "clutch": "Top-5 clutch-minute players per team-season.",
        },
        "seasons": seasons_blob,
        "validation_overall": validation["overall"],
    }

    report_dst.parent.mkdir(parents=True, exist_ok=True)
    report_dst.write_text(json.dumps(report_blob, indent=2), encoding="utf-8")
    validation_dst.parent.mkdir(parents=True, exist_ok=True)
    validation_dst.write_text(json.dumps(validation, indent=2), encoding="utf-8")

    print(f"Saved profiles parquet: {profiles_dst}")
    print(f"Saved step2 lineup report: {report_dst}")
    print(f"Saved step2 validation: {validation_dst}")
    exact5_count = validation["overall"].get("starter_exact5_count", 0)
    exact5_avail = validation["overall"].get("starter_exact5_available", 0)
    print(f"Observed starters exact-5 check: {exact5_count}/{exact5_avail} team-seasons")
    print("\nValidation snapshot:")
    print(json.dumps(validation["overall"], indent=2))


if __name__ == "__main__":
    main()
