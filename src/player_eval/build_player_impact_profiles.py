"""
src/player_eval/build_player_impact_profiles.py
=============================================================================
PEC Step 1 — Build Player Impact Profiles

Merges BKE decomposition scores, offensive/defensive archetypes, official
stats, salaries, and behavioral metrics into a unified player-season profile.

Inputs:
  data/processed/bke/bke_v30_decomposition.parquet
  data/processed/player_archetypes.parquet
  data/processed/defensive_archetypes_v2.parquet
  data/historical/complete_player_season_stats.parquet
  data/historical/player_salaries.parquet

Output:
  data/processed/player_eval/player_impact_profiles.parquet
  reports/player_eval_step1_profiles_report.json

Usage:
  python3 src/player_eval/build_player_impact_profiles.py
=============================================================================
"""
import json
import pickle
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.player_eval.constants import (
    BKE_DECOMP_PATH,
    BKE_SCORES_PATH,
    BKE_V29_PLAYER_DIAGNOSTIC_PATH,
    COMPLETE_STATS_PATH,
    DBKE_V30_SHRINKAGE_PATH,
    DEF_ARCHETYPES_PATH,
    METRICS_LINEAR_PATH,
    PLAYER_ARCHETYPES_PATH,
    PLAYER_PROFILES_PARQUET,
    PLAYER_PROFILES_PKL,
    PLAYERS_META_PATH,
    POSITION_ESTIMATES_PATH,
    STEP1_VALIDATION_REPORT,
    STEP4_VALIDATION_REPORT,
    HISTORICAL_DIR,
    AVAIL_AGE_BREAKPOINTS,
    AVAIL_AGE_FACTORS,
    AVAIL_MAJOR_ROTATION_MPG,
    AVAIL_HIGH_SALARY_THRESHOLD,
    AVAIL_FULL_SEASON_GAMES,
    AVAIL_MIN_EXPECTED_GAMES,
)
from src.utils.player_name_normalizer import (
    apply_player_name_normalization,
    build_player_name_maps,
)


def _safe_float(value, default=np.nan):
    try:
        result = float(value)
        if np.isnan(result):
            return default
        return result
    except (TypeError, ValueError):
        return default


def _safe_str(value, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return default
    text = str(value).strip()
    if text.lower() in {"", "nan", "none"}:
        return default
    return text


def _norm_player_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True)


def _season_to_int(series: pd.Series) -> pd.Series:
    return series.astype(str).str.slice(0, 4).astype(int)


def _pct_to_rate(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    return np.where(vals > 1.0, vals / 100.0, vals)


def _compute_age_factor(age: float) -> float:
    """Piecewise linear age penalty. Returns factor in [AVAIL_AGE_FACTORS[-1], 1.0]."""
    if np.isnan(age) or age < AVAIL_AGE_BREAKPOINTS[0]:
        return AVAIL_AGE_FACTORS[0]  # 1.0 — young, no penalty
    for i, bp in enumerate(AVAIL_AGE_BREAKPOINTS):
        if age < bp:
            # interpolate between previous factor and this factor
            prev_bp = AVAIL_AGE_BREAKPOINTS[i - 1] if i > 0 else 0
            prev_f = AVAIL_AGE_FACTORS[i]
            next_f = AVAIL_AGE_FACTORS[i]
            return next_f
        if i < len(AVAIL_AGE_BREAKPOINTS) - 1:
            next_bp = AVAIL_AGE_BREAKPOINTS[i + 1]
            if age <= next_bp:
                # linear interpolation between breakpoints
                t = (age - bp) / (next_bp - bp)
                return AVAIL_AGE_FACTORS[i + 1] * (1 - t) + AVAIL_AGE_FACTORS[i + 2] * t
    # Beyond last breakpoint
    return AVAIL_AGE_FACTORS[-1]


def _compute_availability_score(
    games: float, age: float, mpg: float, salary: float,
    predicted_mpg: float = 0.0,
) -> float:
    """Compute player availability score (0-1).

    Higher = more available/reliable.  Factors:
      1. games_ratio: games played / expected games
      2. age_factor: piecewise linear age penalty
      3. Role expectation: major rotation / high-salary players are expected to
         play 82 games; lower-minute players have a softer expectation.
    """
    age_factor = _compute_age_factor(age)

    # Determine expected games based on role
    effective_mpg = max(mpg, predicted_mpg) if not np.isnan(predicted_mpg) else mpg
    is_major = (
        effective_mpg >= AVAIL_MAJOR_ROTATION_MPG
        or (not np.isnan(salary) and salary >= AVAIL_HIGH_SALARY_THRESHOLD)
    )
    if is_major:
        expected_games = AVAIL_FULL_SEASON_GAMES
    else:
        # Scale expected games: bench players expected fewer games
        # Linear scale from MIN_EXPECTED to 82 based on MPG fraction
        mpg_frac = min(effective_mpg / AVAIL_MAJOR_ROTATION_MPG, 1.0)
        expected_games = (
            AVAIL_MIN_EXPECTED_GAMES
            + (AVAIL_FULL_SEASON_GAMES - AVAIL_MIN_EXPECTED_GAMES) * mpg_frac
        )

    games_ratio = min(games / max(expected_games, 1.0), 1.0)
    availability = games_ratio * age_factor
    return float(np.clip(availability, 0.0, 1.0))


def _softmax_df(df: pd.DataFrame) -> pd.DataFrame:
    values = df.to_numpy(dtype=float)
    values = np.nan_to_num(values, nan=0.0)
    max_vals = np.max(values, axis=1, keepdims=True)
    shifted = values - max_vals
    exp_vals = np.exp(shifted)
    sums = exp_vals.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    return pd.DataFrame(exp_vals / sums, columns=df.columns, index=df.index)


def _first_existing(df: pd.DataFrame, candidates: List[str]) -> pd.Series:
    for col in candidates:
        if col in df.columns:
            return df[col]
    return pd.Series(np.nan, index=df.index)


def _coalesce_columns(df: pd.DataFrame, target_col: str, candidates: List[str]) -> None:
    out = pd.Series(np.nan, index=df.index)
    for col in candidates:
        if col in df.columns:
            out = out.fillna(df[col])
    df[target_col] = out


@dataclass
class PlayerImpactProfile:
    # --- Identity ---
    player_id: str
    player_name: str
    season: str
    team_abbreviation: str
    team_id: str
    # --- Impact metrics ---
    impact_orapm: float
    impact_drapm: float
    impact_bke: float
    impact_obke: float
    impact_dbke: float
    impact_stability: float
    impact_ws: float
    impact_bpm: float
    impact_vorp: float
    impact_portable_talent: float
    impact_total_impact: float
    # --- Behavioral rates ---
    behavioral_usage: float                # USG% (0‑1)
    behavioral_assist_rate: float          # AST% (0‑1)
    behavioral_turnover_rate: float        # TOV% (0‑1)
    behavioral_three_point_rate: float     # FG3A / FGA (0‑1)
    behavioral_rim_rate: float             # At‑rim frequency (0‑1)
    behavioral_efg: float                  # eFG% (0‑1)
    behavioral_orb_rate: float             # OREB% (0‑1)
    behavioral_drb_rate: float             # DREB% (0‑1)
    behavioral_free_throw_rate: float      # FTA / FGA (≥0, standard FT rate)
    behavioral_hustle_pctl: float          # hustle percentile (0‑1)
    behavioral_foul_rate: float            # personal fouls per 36 min
    # --- Vectors ---
    playtype_vector: List[float]
    offensive_archetype_probs: Dict[str, float]   # all 11 offensive archetypes
    defensive_archetype_probs: Dict[str, float]   # 7 defensive roles
    # --- Canonical archetype labels (same source as player_data_viewer) ---
    off_primary_archetype: str
    off_secondary_archetype: str
    off_role_confidence: float
    off_role_effectiveness: float
    def_primary_archetype: str
    def_secondary_archetype: str
    def_role_confidence: float
    # --- Context ---
    position_proxy: str
    age: float
    height_inches: float
    weight_lbs: float
    experience_years: float
    # --- Volume ---
    minutes: float
    mpg: float
    minute_share_potential: float
    possessions: float
    games: float
    # --- Team / Scheme ---
    on_off_diff: float                     # team on/off differential
    scheme_stability_index: float
    compression_flag: float
    defensive_shrinkage_lambda: float
    # --- Financial ---
    salary: float
    # --- Availability ---
    availability_score: float  # 0-1, combines games played ratio, age factor, role expectation


def _load_bke_json(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["player_id", "season", "impact_bke", "impact_obke", "impact_dbke"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for _, info in payload.get("players", {}).items():
        rows.append(
            {
                "player_id": str(info.get("player_id", "")),
                "season": str(info.get("season", "")),
                "impact_bke": _safe_float(info.get("raw_BKE"), default=np.nan),
                "impact_obke": _safe_float(info.get("raw_OBKE"), default=np.nan),
                "impact_dbke": _safe_float(info.get("raw_DBKE"), default=np.nan),
            }
        )
    return pd.DataFrame(rows)


def _load_v29_stability(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["player_id", "season"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("player_level_metrics", [])
    if not rows:
        return pd.DataFrame(columns=["player_id", "season"])
    df = pd.DataFrame(rows)
    return df.rename(
        columns={
            "d_stability": "diag_d_stability",
            "o_stability": "diag_o_stability",
            "vol_ratio_dbke_to_drapm": "diag_vol_ratio",
        }
    )[
        [
            "player_id",
            "season",
            "diag_d_stability",
            "diag_o_stability",
            "diag_vol_ratio",
        ]
    ]


def _load_v30_shrinkage(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["player_id", "season", "lambda_shrink"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("player_outputs", [])
    if not rows:
        return pd.DataFrame(columns=["player_id", "season", "lambda_shrink"])
    df = pd.DataFrame(rows)
    return df[["player_id", "season", "lambda_shrink"]]


def _build_stability(df: pd.DataFrame) -> pd.Series:
    d_stab = pd.to_numeric(df.get("diag_d_stability"), errors="coerce")
    o_stab = pd.to_numeric(df.get("diag_o_stability"), errors="coerce")
    vol_ratio = pd.to_numeric(df.get("diag_vol_ratio"), errors="coerce")
    lambda_shrink = pd.to_numeric(df.get("lambda_shrink"), errors="coerce")

    base_stability = np.nanmean(np.vstack([d_stab, o_stab]), axis=0)
    base_stability = pd.Series(base_stability, index=df.index)

    vol_penalty = np.clip((vol_ratio - 1.0).fillna(0.0), 0.0, 1.0)
    shrink_bonus = lambda_shrink.fillna(lambda_shrink.median())

    stability = 0.50 * base_stability + 0.30 * shrink_bonus + 0.20 * (1.0 - vol_penalty)
    stability = stability.fillna(stability.median())
    return stability.clip(lower=0.0, upper=1.0)


def main() -> None:
    if not BKE_DECOMP_PATH.exists():
        raise FileNotFoundError(f"Missing required file: {BKE_DECOMP_PATH}")

    # ── 1. Load BKE decomposition (spine) ──────────────────────────────
    base = pd.read_parquet(BKE_DECOMP_PATH).copy()
    base["player_id"] = _norm_player_id(base["player_id"])
    base["season"] = base["season"].astype(str)

    # ── 2. Load BKE scores JSON (OBKE / DBKE) ─────────────────────────
    bke_df = _load_bke_json(BKE_SCORES_PATH)
    if not bke_df.empty:
        bke_df["player_id"] = _norm_player_id(bke_df["player_id"])
        bke_df["season"] = bke_df["season"].astype(str)

    # ── 3. Load stability / diagnostics ────────────────────────────────
    v29 = _load_v29_stability(BKE_V29_PLAYER_DIAGNOSTIC_PATH)
    if not v29.empty:
        v29["player_id"] = _norm_player_id(v29["player_id"])
        v29["season"] = v29["season"].astype(str)

    v30 = _load_v30_shrinkage(DBKE_V30_SHRINKAGE_PATH)
    if not v30.empty:
        v30["player_id"] = _norm_player_id(v30["player_id"])
        v30["season"] = v30["season"].astype(str)

    # ── 4. Load box-score stats (keep only TOT rows for traded players) ─
    stats = pd.read_parquet(COMPLETE_STATS_PATH)
    stats = stats.rename(
        columns={
            "PLAYER_ID": "player_id",
            "SEASON": "season",
            "TEAM_ID": "team_id",
            "TEAM_ABBREVIATION": "team_abbreviation",
            "AGE": "age",
            "GP": "games",
            "MIN": "minutes_box",
            "PF": "pf_box",
        }
    )
    stats["player_id"] = _norm_player_id(stats["player_id"])
    stats["season"] = stats["season"].astype(str)
    # For traded players (TEAM_COUNT > 1), keep only the TOT or highest-minute row
    stats["_tc"] = pd.to_numeric(stats.get("TEAM_COUNT"), errors="coerce").fillna(1)
    stats["minutes_box"] = pd.to_numeric(stats["minutes_box"], errors="coerce")
    stats = stats.sort_values("minutes_box", ascending=False).drop_duplicates(
        subset=["player_id", "season"], keep="first"
    )
    stats_keep = ["player_id", "season", "team_id", "team_abbreviation", "age", "games",
                  "minutes_box", "pf_box", "FGA", "FG3A", "FTA"]
    stats_keep = [c for c in stats_keep if c in stats.columns]
    stats = stats[stats_keep]

    # ── 5. Load offensive archetypes (all 11 embedding dims) ───────────
    arche = pd.read_parquet(PLAYER_ARCHETYPES_PATH)
    arche = arche.rename(
        columns={
            "PLAYER_ID": "player_id",
            "SEASON": "season",
            "TEAM_ABBREVIATION": "team_abbreviation_arche",
            "USG_PCT": "usage_pct",
            "TS_PCT": "ts_pct",
            "EFG_PCT": "efg_pct",
            "FG3A_PER36": "fg3a_per36",
            "FG3_PCT": "fg3_pct",
            "TOV_PCT": "tov_pct",
            "FT_RATE": "ft_rate",
            "AT_RIM_FREQ": "rim_rate",
        }
    )
    arche["player_id"] = _norm_player_id(arche["player_id"])
    arche["season"] = arche["season"].astype(str)

    # ── 6. Load defensive archetypes ───────────────────────────────────
    def_arche = pd.read_parquet(DEF_ARCHETYPES_PATH)
    def_arche = def_arche.rename(columns={"PLAYER_ID": "player_id", "SEASON": "season"})
    def_arche["player_id"] = _norm_player_id(def_arche["player_id"])
    def_arche["season"] = def_arche["season"].astype(str)

    # ── 7. Load position estimates ─────────────────────────────────────
    pos = pd.read_parquet(POSITION_ESTIMATES_PATH)
    pos = pos.rename(columns={"PLAYER_ID": "player_id", "SEASON": "season", "GP": "games_pos", "MIN": "minutes_pos"})
    pos["player_id"] = _norm_player_id(pos["player_id"])
    pos["season"] = pos["season"].astype(str)

    # ── 8. Load linear metrics (WS, BPM, VORP) ────────────────────────
    metrics_lin = pd.DataFrame()
    if METRICS_LINEAR_PATH.exists():
        metrics_lin = pd.read_parquet(METRICS_LINEAR_PATH)
        metrics_lin["player_id"] = _norm_player_id(metrics_lin["player_id"])
        metrics_lin["season"] = metrics_lin["season"].astype(str)

    # ── 9. Load player bio metadata ────────────────────────────────────
    players_meta = pd.DataFrame()
    if PLAYERS_META_PATH.exists():
        players_meta = pd.read_parquet(PLAYERS_META_PATH)
        players_meta = players_meta.rename(columns={"player_id": "player_id_meta"})
        players_meta["player_id_meta"] = _norm_player_id(players_meta["player_id_meta"])

    # ── 10. Load salary data (all seasons) ─────────────────────────────
    import glob
    salary_files = sorted(glob.glob(str(HISTORICAL_DIR / "player_salaries_*.parquet")))
    salary_frames = []
    salary_name_frames = []  # for name-based fallback
    for sf in salary_files:
        try:
            sdf = pd.read_parquet(sf)
            # Rows with valid player_id
            has_id = sdf["player_id"].notna()
            if has_id.any():
                matched = sdf.loc[has_id].copy()
                matched["player_id"] = _norm_player_id(matched["player_id"])
                matched["season"] = matched["season"].astype(str)
                salary_frames.append(matched[["player_id", "season", "salary"]])
            # Rows without player_id but with player_name (for name-based fallback)
            has_name = (~has_id) & sdf["player_name"].notna()
            if has_name.any():
                unmatched = sdf.loc[has_name].copy()
                unmatched["season"] = unmatched["season"].astype(str)
                salary_name_frames.append(unmatched[["player_name", "season", "salary"]])
        except Exception:
            pass
    salary_df = pd.concat(salary_frames, ignore_index=True) if salary_frames else pd.DataFrame(columns=["player_id", "season", "salary"])
    salary_name_df = pd.concat(salary_name_frames, ignore_index=True) if salary_name_frames else pd.DataFrame(columns=["player_name", "season", "salary"])

    # ═══════════════════ MERGE ═══════════════════════════════════════
    data = base.merge(bke_df, on=["player_id", "season"], how="left")
    data = data.merge(v29, on=["player_id", "season"], how="left")
    data = data.merge(v30, on=["player_id", "season"], how="left")
    data = data.merge(stats, on=["player_id", "season"], how="left")

    # Archetype columns to keep (all 11 embeddings + behavioral inputs)
    arche_keep = [
        "player_id", "season",
        "usage_pct", "tov_pct", "fg3_pct", "fg3a_per36", "rim_rate", "efg_pct", "ft_rate",
        "primary_archetype", "secondary_archetype", "role_confidence", "archetype_confidence", "role_effectiveness",
        "ISOLATION_POSS_PCT", "PRBALLHANDLER_POSS_PCT", "POSTUP_POSS_PCT",
        "CUT_POSS_PCT", "PRROLLMAN_POSS_PCT", "HANDOFF_POSS_PCT",
        "OFFSCREEN_POSS_PCT", "SPOTUP_POSS_PCT", "TRANSITION_POSS_PCT",
        "OFFREBOUND_POSS_PCT", "MISC_POSS_PCT",
        # All 11 offensive archetype embeddings
        "emb_ball_dominant_creator", "emb_all_around_scorer", "emb_ballhandler",
        "emb_interior_scorer", "emb_perimeter_scorer", "emb_connector",
        "emb_pnr_rolling_big", "emb_pnr_popping_big", "emb_off_ball_finisher",
        "emb_off_ball_movement_shooter", "emb_off_ball_stationary_shooter",
    ]
    arche_keep = [c for c in arche_keep if c in arche.columns]
    data = data.merge(arche[arche_keep], on=["player_id", "season"], how="left", suffixes=("", "_arche"))

    # Resolve any overlap from base vs archetype sources (prefer archetype source when available)
    arche_overlap_cols = [
        "ISOLATION_POSS_PCT", "PRBALLHANDLER_POSS_PCT", "POSTUP_POSS_PCT",
        "CUT_POSS_PCT", "PRROLLMAN_POSS_PCT", "HANDOFF_POSS_PCT",
        "OFFSCREEN_POSS_PCT", "SPOTUP_POSS_PCT", "TRANSITION_POSS_PCT",
        "emb_ball_dominant_creator", "emb_all_around_scorer", "emb_ballhandler",
        "emb_interior_scorer", "emb_perimeter_scorer", "emb_connector",
        "emb_pnr_rolling_big", "emb_pnr_popping_big", "emb_off_ball_finisher",
        "emb_off_ball_movement_shooter", "emb_off_ball_stationary_shooter",
    ]
    for col in arche_overlap_cols:
        _coalesce_columns(data, col, [f"{col}_arche", col])

    # Canonical offensive archetype labels/confidence from player_archetypes.parquet
    _coalesce_columns(data, "off_primary_archetype", ["primary_archetype_arche", "primary_archetype"])
    _coalesce_columns(data, "off_secondary_archetype", ["secondary_archetype_arche", "secondary_archetype"])
    _coalesce_columns(data, "off_role_confidence", ["role_confidence_arche", "archetype_confidence_arche", "role_confidence", "archetype_confidence"])
    _coalesce_columns(data, "off_role_effectiveness", ["role_effectiveness_arche", "role_effectiveness"])

    data = data.merge(
        def_arche[
            [
                "player_id", "season",
                "hustle_score", "hustle_pctl", "defensive_confidence",
                "defensive_secondary",
                "poa_score", "wing_score", "chaser_score", "versatile_score",
                "rim_score", "drop_big_score", "mobile_big_score",
                "defensive_archetype",
            ]
        ].rename(columns={
            "defensive_archetype": "def_primary_archetype_src",
            "defensive_secondary": "def_secondary_archetype_src",
            "defensive_confidence": "def_role_confidence_src",
        }).drop_duplicates(subset=["player_id", "season"], keep="first"),
        on=["player_id", "season"],
        how="left",
    )
    data = data.merge(
        pos[["player_id", "season", "primary_position_estimate", "primary_position",
             "pct_pg", "pct_sg", "pct_sf", "pct_pf", "pct_c"]],
        on=["player_id", "season"],
        how="left",
    )

    # Metrics linear → WS, BPM, VORP
    if not metrics_lin.empty:
        ml_keep = ["player_id", "season"]
        for c in ["WS", "BPM", "VORP"]:
            if c in metrics_lin.columns:
                ml_keep.append(c)
        data = data.merge(metrics_lin[ml_keep].drop_duplicates(subset=["player_id", "season"]),
                          on=["player_id", "season"], how="left")

    # Salary — merge by ID first, then name-based fallback
    if not salary_df.empty:
        data = data.merge(salary_df.drop_duplicates(subset=["player_id", "season"]),
                          on=["player_id", "season"], how="left")
    else:
        data["salary"] = np.nan

    # Name-based salary fallback for players without salary after ID merge
    if not salary_name_df.empty:
        from src.utils.player_name_normalizer import canonical_name_key
        salary_missing = data["salary"].isna()
        if salary_missing.any():
            data["_canon_key"] = data["player_name"].astype(str).map(canonical_name_key)
            salary_name_df["_canon_key"] = salary_name_df["player_name"].astype(str).map(canonical_name_key)
            name_salary_map = salary_name_df.drop_duplicates(subset=["_canon_key", "season"]).set_index(["_canon_key", "season"])["salary"]
            for idx in data.index[salary_missing]:
                key = (data.at[idx, "_canon_key"], str(data.at[idx, "season"]))
                if key in name_salary_map.index:
                    data.at[idx, "salary"] = name_salary_map[key]
            data.drop(columns=["_canon_key"], inplace=True)

    # Player bio (join on player_id only — bio is not season-specific)
    if not players_meta.empty:
        bio_keep = ["player_id_meta"]
        for c in ["height_inches", "weight_lbs", "experience_years"]:
            if c in players_meta.columns:
                bio_keep.append(c)
        bio = players_meta[bio_keep].drop_duplicates(subset=["player_id_meta"])
        data = data.merge(bio, left_on="player_id", right_on="player_id_meta", how="left")
    else:
        data["height_inches"] = np.nan
        data["weight_lbs"] = np.nan
        data["experience_years"] = np.nan

    # ═══════════════════ FEATURE ENGINEERING ═════════════════════════
    # --- Impact ---
    data["impact_bke"] = pd.to_numeric(data["impact_bke"], errors="coerce")
    data["impact_bke"] = data["impact_bke"].fillna(pd.to_numeric(data.get("raw_BKE"), errors="coerce"))
    data["impact_obke"] = pd.to_numeric(data.get("impact_obke"), errors="coerce")
    data["impact_dbke"] = pd.to_numeric(data.get("impact_dbke"), errors="coerce")
    data["impact_orapm"] = pd.to_numeric(data.get("orapm"), errors="coerce").fillna(
        pd.to_numeric(data.get("ORAPM"), errors="coerce")
    )
    data["impact_drapm"] = pd.to_numeric(data.get("drapm"), errors="coerce").fillna(
        pd.to_numeric(data.get("DRAPM"), errors="coerce")
    )
    data["impact_stability"] = _build_stability(data)
    data["impact_ws"] = pd.to_numeric(data.get("WS"), errors="coerce")
    data["impact_bpm"] = pd.to_numeric(data.get("BPM"), errors="coerce")
    data["impact_vorp"] = pd.to_numeric(data.get("VORP"), errors="coerce")
    data["impact_portable_talent"] = pd.to_numeric(data.get("portable_talent_score"), errors="coerce")
    data["impact_total_impact"] = pd.to_numeric(data.get("total_impact_score"), errors="coerce")

    # --- Behavioral rates (basketball-correct) ---
    data["usage_rate"] = _pct_to_rate(_first_existing(data, ["usage_pct", "USG_RATE", "USG_pct", "USG_PCT"]))
    data["assist_rate"] = _pct_to_rate(_first_existing(data, ["AST_PCT", "AST_pct"]))
    data["turnover_rate"] = _pct_to_rate(_first_existing(data, ["tov_pct", "TOV_PCT", "TOV_pct"]))

    # FIX: three_point_rate = FG3A / FGA (share of shots from three, 0‑1)
    # Note: base (BKE decomp) AND stats both have FGA/FG3A/FTA, causing _x/_y suffixes
    fg3a = pd.to_numeric(_first_existing(data, ["FG3A", "FG3A_y", "FG3A_x", "fg3a"]), errors="coerce")
    fga = pd.to_numeric(_first_existing(data, ["FGA", "FGA_y", "FGA_x", "fga", "TOTAL_FGA"]), errors="coerce").replace(0, np.nan)
    data["three_point_rate"] = (fg3a / fga).clip(0.0, 1.0)

    data["rim_rate"] = _pct_to_rate(_first_existing(data, ["rim_rate", "AT_RIM_FREQ"]))
    data["efg"] = _pct_to_rate(_first_existing(data, ["efg_pct", "EFG_PCT", "eFG_pct", "EFG_PCT_y", "EFG_PCT_x"]))
    data["orb_rate"] = _pct_to_rate(_first_existing(data, ["OREB_PCT", "OREB_pct", "OREB_PCT_y"])).clip(0.0, 1.0)
    data["drb_rate"] = _pct_to_rate(_first_existing(data, ["DREB_PCT", "DREB_pct", "DREB_PCT_y"])).clip(0.0, 1.0)

    # FIX: free_throw_rate = FTA / FGA (standard FT rate, can be > 1.0, that's valid)
    fta = pd.to_numeric(_first_existing(data, ["FTA", "FTA_y", "FTA_x", "fta"]), errors="coerce")
    data["free_throw_rate"] = (fta / fga).clip(lower=0.0)

    # FIX: hustle = hustle_pctl from defensive archetypes (already 0‑1 normalized)
    data["hustle_pctl"] = pd.to_numeric(data.get("hustle_pctl"), errors="coerce").clip(0.0, 1.0)

    # FIX: foul_rate = personal fouls per 36 minutes
    pf = pd.to_numeric(_first_existing(data, ["pf_box", "PF"]), errors="coerce")
    mins = pd.to_numeric(data.get("MIN"), errors="coerce").fillna(
        pd.to_numeric(data.get("minutes_box"), errors="coerce")
    )
    data["foul_rate"] = (pf * 36.0 / mins.replace(0, np.nan)).clip(lower=0.0)

    # FIX: on_off_diff = actual team on/off differential from BKE decomposition
    data["on_off_diff"] = pd.to_numeric(data.get("on_off_diff"), errors="coerce")

    # Scheme stability from BKE decomposition
    data["scheme_stability_index"] = pd.to_numeric(data.get("scheme_stability_index"), errors="coerce")

    data["defensive_shrinkage_lambda"] = pd.to_numeric(data.get("lambda_shrink"), errors="coerce")
    data["compression_flag"] = (pd.to_numeric(data.get("diag_vol_ratio"), errors="coerce") > 1.15).astype(float)

    # Canonical defensive archetype labels/confidence (same fields as player_data_viewer)
    _coalesce_columns(data, "def_primary_archetype", ["def_primary_archetype_src", "defensive_archetype"])
    _coalesce_columns(data, "def_secondary_archetype", ["def_secondary_archetype_src", "defensive_secondary"])
    _coalesce_columns(data, "def_role_confidence", ["def_role_confidence_src", "defensive_confidence"])
    data["def_role_confidence"] = pd.to_numeric(data["def_role_confidence"], errors="coerce")

    # --- Playtype vector (possession shares) ---
    playtype_cols = [
        "ISOLATION_POSS_PCT", "PRBALLHANDLER_POSS_PCT", "POSTUP_POSS_PCT",
        "CUT_POSS_PCT", "PRROLLMAN_POSS_PCT", "HANDOFF_POSS_PCT",
        "OFFSCREEN_POSS_PCT", "SPOTUP_POSS_PCT", "TRANSITION_POSS_PCT",
        "OFFREBOUND_POSS_PCT", "MISC_POSS_PCT",
    ]
    for col in playtype_cols:
        data[col] = _pct_to_rate(data.get(col)).astype(float)

    # --- Offensive archetype probabilities (all 11, softmax-normalized) ---
    off_prob_cols = [
        "emb_ball_dominant_creator", "emb_all_around_scorer", "emb_ballhandler",
        "emb_interior_scorer", "emb_perimeter_scorer", "emb_connector",
        "emb_pnr_rolling_big", "emb_pnr_popping_big", "emb_off_ball_finisher",
        "emb_off_ball_movement_shooter", "emb_off_ball_stationary_shooter",
    ]
    for col in off_prob_cols:
        data[col] = pd.to_numeric(data.get(col), errors="coerce")
    # Use archetype-provided embeddings when available (coalesced above).
    # Many archetype outputs are already L1-normalized; only normalize rows
    # where the sum is 0 or meaningfully different from 1 to avoid altering
    # authoritative embeddings unnecessarily.
    off_raw = data[off_prob_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(lower=0.0)
    off_row_sums = off_raw.sum(axis=1)
    eps = 1e-6
    needs_norm = (off_row_sums == 0.0) | ((off_row_sums - 1.0).abs() > eps)
    # Default: copy raw values
    data.loc[:, off_prob_cols] = off_raw
    if needs_norm.any():
        # Normalize only the rows that need it
        to_norm_idx = needs_norm[needs_norm].index
        sub = off_raw.loc[to_norm_idx]
        sub_sums = sub.sum(axis=1).replace(0.0, 1.0)
        data.loc[to_norm_idx, off_prob_cols] = sub.div(sub_sums, axis=0)

    # --- Defensive archetype probabilities (7 roles, softmax-normalized) ---
    def_prob_cols = [
        "poa_score", "wing_score", "chaser_score", "versatile_score",
        "rim_score", "drop_big_score", "mobile_big_score",
    ]
    for col in def_prob_cols:
        data[col] = pd.to_numeric(data.get(col), errors="coerce")
    data[def_prob_cols] = _softmax_df(data[def_prob_cols].fillna(0.0))

    # --- Volume ---
    data["minutes"] = pd.to_numeric(data.get("MIN"), errors="coerce").fillna(
        pd.to_numeric(data.get("minutes_box"), errors="coerce")
    )
    data["games"] = pd.to_numeric(data.get("GP"), errors="coerce").fillna(
        pd.to_numeric(data.get("games"), errors="coerce")
    )
    data["possessions"] = pd.to_numeric(data.get("possessions_played"), errors="coerce")
    data["age"] = pd.to_numeric(data.get("age"), errors="coerce")
    data["mpg"] = data["minutes"] / data["games"].replace(0, np.nan)
    data["height_inches"] = pd.to_numeric(data.get("height_inches"), errors="coerce")
    data["weight_lbs"] = pd.to_numeric(data.get("weight_lbs"), errors="coerce")
    data["experience_years"] = pd.to_numeric(data.get("experience_years"), errors="coerce")
    data["salary"] = pd.to_numeric(data.get("salary"), errors="coerce")

    data["team_abbreviation"] = data["team_abbreviation"].astype(str)
    data["team_id"] = data["team_id"].astype(str)

    # Minute share = player's share of team total minutes
    team_minutes = (
        data.groupby(["season", "team_abbreviation"], dropna=False)["minutes"]
        .sum()
        .rename("team_minutes_sum")
        .reset_index()
    )
    data = data.merge(team_minutes, on=["season", "team_abbreviation"], how="left")
    data["minute_share_potential"] = data["minutes"] / data["team_minutes_sum"].replace(0, np.nan)

    # ── Name normalization (ID-first, alias-aware) ────────────────────
    name_sources = [
        (PLAYERS_META_PATH, ["id", "player_id"], ["full_name", "player_name"], 1),
        (PLAYER_ARCHETYPES_PATH, ["PLAYER_ID", "player_id"], ["PLAYER_NAME", "player_name"], 2),
        (COMPLETE_STATS_PATH, ["PLAYER_ID", "player_id"], ["PLAYER_NAME", "player_name"], 2),
        (BKE_DECOMP_PATH, ["player_id", "PLAYER_ID"], ["player_name", "PLAYER_NAME"], 3),
    ]
    id_to_name, key_to_name = build_player_name_maps(name_sources)
    data = apply_player_name_normalization(
        df=data,
        player_id_col="player_id",
        player_name_col="player_name",
        id_to_name=id_to_name,
        key_to_name=key_to_name,
    )

    # ── Median-fill numeric columns ────────────────────────────────────
    numeric_fill_cols = [
        "impact_orapm", "impact_drapm", "impact_bke", "impact_obke", "impact_dbke",
        "impact_stability", "impact_ws", "impact_bpm", "impact_vorp",
        "impact_portable_talent", "impact_total_impact",
        "usage_rate", "assist_rate", "turnover_rate", "three_point_rate",
        "rim_rate", "efg", "orb_rate", "drb_rate", "free_throw_rate",
        "hustle_pctl", "foul_rate",
        "age", "minutes", "mpg", "minute_share_potential", "possessions", "games",
        "on_off_diff", "scheme_stability_index",
        "compression_flag", "defensive_shrinkage_lambda",
        "off_role_confidence", "off_role_effectiveness", "def_role_confidence",
    ]
    for col in numeric_fill_cols:
        median = pd.to_numeric(data[col], errors="coerce").median()
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(median if not np.isnan(median) else 0.0)

    # ═══════════════════ BUILD PROFILES ══════════════════════════════
    profile_rows = []
    flat_rows = []

    for _, row in data.iterrows():
        off_probs = {k: _safe_float(row[k], default=0.0) for k in off_prob_cols}
        def_probs = {k: _safe_float(row[k], default=0.0) for k in def_prob_cols}
        play_vec = [_safe_float(row[c], default=0.0) for c in playtype_cols]

        profile = PlayerImpactProfile(
            player_id=str(row["player_id"]),
            player_name=str(row.get("player_name", "Unknown")),
            season=str(row["season"]),
            team_abbreviation=str(row.get("team_abbreviation", "")),
            team_id=str(row.get("team_id", "")),
            # Impact
            impact_orapm=_safe_float(row["impact_orapm"], default=0.0),
            impact_drapm=_safe_float(row["impact_drapm"], default=0.0),
            impact_bke=_safe_float(row["impact_bke"], default=0.0),
            impact_obke=_safe_float(row["impact_obke"], default=0.0),
            impact_dbke=_safe_float(row["impact_dbke"], default=0.0),
            impact_stability=_safe_float(row["impact_stability"], default=0.5),
            impact_ws=_safe_float(row["impact_ws"], default=0.0),
            impact_bpm=_safe_float(row["impact_bpm"], default=0.0),
            impact_vorp=_safe_float(row["impact_vorp"], default=0.0),
            impact_portable_talent=_safe_float(row["impact_portable_talent"], default=0.0),
            impact_total_impact=_safe_float(row["impact_total_impact"], default=0.0),
            # Behavioral
            behavioral_usage=_safe_float(row["usage_rate"], default=0.0),
            behavioral_assist_rate=_safe_float(row["assist_rate"], default=0.0),
            behavioral_turnover_rate=_safe_float(row["turnover_rate"], default=0.0),
            behavioral_three_point_rate=_safe_float(row["three_point_rate"], default=0.0),
            behavioral_rim_rate=_safe_float(row["rim_rate"], default=0.0),
            behavioral_efg=_safe_float(row["efg"], default=0.0),
            behavioral_orb_rate=_safe_float(row["orb_rate"], default=0.0),
            behavioral_drb_rate=_safe_float(row["drb_rate"], default=0.0),
            behavioral_free_throw_rate=_safe_float(row["free_throw_rate"], default=0.0),
            behavioral_hustle_pctl=_safe_float(row["hustle_pctl"], default=0.0),
            behavioral_foul_rate=_safe_float(row["foul_rate"], default=0.0),
            # Vectors
            playtype_vector=play_vec,
            offensive_archetype_probs=off_probs,
            defensive_archetype_probs=def_probs,
            off_primary_archetype=_safe_str(row.get("off_primary_archetype"), default="Unknown"),
            off_secondary_archetype=_safe_str(row.get("off_secondary_archetype"), default=""),
            off_role_confidence=_safe_float(row.get("off_role_confidence"), default=0.0),
            off_role_effectiveness=_safe_float(row.get("off_role_effectiveness"), default=0.0),
            def_primary_archetype=_safe_str(row.get("def_primary_archetype"), default="Unknown"),
            def_secondary_archetype=_safe_str(row.get("def_secondary_archetype"), default=""),
            def_role_confidence=_safe_float(row.get("def_role_confidence"), default=0.0),
            # Context
            position_proxy=str(row.get("primary_position_estimate", row.get("primary_position", ""))),
            age=_safe_float(row["age"], default=0.0),
            height_inches=_safe_float(row["height_inches"], default=0.0),
            weight_lbs=_safe_float(row["weight_lbs"], default=0.0),
            experience_years=_safe_float(row["experience_years"], default=0.0),
            # Volume
            minutes=_safe_float(row["minutes"], default=0.0),
            mpg=_safe_float(row["mpg"], default=0.0),
            minute_share_potential=_safe_float(row["minute_share_potential"], default=0.0),
            possessions=_safe_float(row["possessions"], default=0.0),
            games=_safe_float(row["games"], default=0.0),
            # Team/scheme
            on_off_diff=_safe_float(row["on_off_diff"], default=0.0),
            scheme_stability_index=_safe_float(row["scheme_stability_index"], default=0.0),
            compression_flag=_safe_float(row["compression_flag"], default=0.0),
            defensive_shrinkage_lambda=_safe_float(row["defensive_shrinkage_lambda"], default=0.0),
            # Financial
            salary=_safe_float(row["salary"], default=np.nan),
            # Availability
            availability_score=_compute_availability_score(
                games=_safe_float(row["games"], default=0.0),
                age=_safe_float(row["age"], default=25.0),
                mpg=_safe_float(row["mpg"], default=0.0),
                salary=_safe_float(row["salary"], default=np.nan),
            ),
        )

        profile_rows.append(profile)
        flat = asdict(profile)
        for key, val in profile.offensive_archetype_probs.items():
            flat[f"off_prob_{key}"] = val
        for key, val in profile.defensive_archetype_probs.items():
            flat[f"def_prob_{key}"] = val
        for idx, val in enumerate(profile.playtype_vector):
            flat[f"playtype_{idx}"] = val
        flat_rows.append(flat)

    flat_df = pd.DataFrame(flat_rows)

    # ═══════════════════ VALIDATION ══════════════════════════════════
    rate_cols = [
        "behavioral_usage", "behavioral_assist_rate", "behavioral_turnover_rate",
        "behavioral_three_point_rate", "behavioral_rim_rate", "behavioral_efg",
        "behavioral_orb_rate", "behavioral_drb_rate", "behavioral_hustle_pctl",
    ]
    bounds_violations = {}
    for col in rate_cols:
        vals = pd.to_numeric(flat_df[col], errors="coerce")
        bounds_violations[col] = int(((vals < 0) | (vals > 1)).sum())

    off_prob_sum = flat_df[[c for c in flat_df.columns if c.startswith("off_prob_")]].sum(axis=1)
    def_prob_sum = flat_df[[c for c in flat_df.columns if c.startswith("def_prob_")]].sum(axis=1)

    consistency_delta = (flat_df["impact_orapm"] + flat_df["impact_drapm"] - flat_df["impact_bke"]).abs()

    validation_report = {
        "rows": int(len(flat_df)),
        "seasons": sorted(flat_df["season"].astype(str).unique().tolist(), key=lambda s: int(s[:4])),
        "offensive_archetype_count": int(len([c for c in flat_df.columns if c.startswith("off_prob_")])),
        "defensive_archetype_count": int(len([c for c in flat_df.columns if c.startswith("def_prob_")])),
        "offensive_prob_sum_mean": float(off_prob_sum.mean()),
        "defensive_prob_sum_mean": float(def_prob_sum.mean()),
        "offensive_prob_sum_max_abs_error": float((off_prob_sum - 1.0).abs().max()),
        "defensive_prob_sum_max_abs_error": float((def_prob_sum - 1.0).abs().max()),
        "rate_bounds_violations": bounds_violations,
        "orapm_drapm_bke_abs_delta_mean": float(consistency_delta.mean()),
        "orapm_drapm_bke_abs_delta_p95": float(consistency_delta.quantile(0.95)),
        "minute_share_potential_sum_by_team_season": float(
            flat_df.groupby(["season", "team_abbreviation"])["minute_share_potential"].sum().mean()
        ),
        "stability_summary": {
            "mean": float(flat_df["impact_stability"].mean()),
            "std": float(flat_df["impact_stability"].std(ddof=0)),
            "min": float(flat_df["impact_stability"].min()),
            "max": float(flat_df["impact_stability"].max()),
        },
        "mpg_summary": {
            "mean": float(flat_df["mpg"].mean()),
            "median": float(flat_df["mpg"].median()),
            "max": float(flat_df["mpg"].max()),
        },
        "salary_coverage": float((flat_df["salary"] > 0).mean()),
        "bio_coverage": {
            "height": float((flat_df["height_inches"] > 0).mean()),
            "weight": float((flat_df["weight_lbs"] > 0).mean()),
            "experience": float((flat_df["experience_years"] > 0).mean()),
        },
        "new_impact_coverage": {
            "obke": float(flat_df["impact_obke"].notna().mean()),
            "dbke": float(flat_df["impact_dbke"].notna().mean()),
            "ws": float((flat_df["impact_ws"] != 0).mean()),
            "bpm": float((flat_df["impact_bpm"] != 0).mean()),
            "vorp": float((flat_df["impact_vorp"] != 0).mean()),
            "portable_talent": float((flat_df["impact_portable_talent"] != 0).mean()),
            "total_impact": float((flat_df["impact_total_impact"] != 0).mean()),
        },
        "name_quality": {
            "unknown_count": int(flat_df["player_name"].astype(str).str.lower().eq("unknown").sum()),
            "unique_names": int(flat_df["player_name"].astype(str).nunique()),
        },
        "availability_summary": {
            "mean": float(flat_df["availability_score"].mean()),
            "std": float(flat_df["availability_score"].std(ddof=0)),
            "min": float(flat_df["availability_score"].min()),
            "max": float(flat_df["availability_score"].max()),
            "median": float(flat_df["availability_score"].median()),
            "bounds_violations": int(((flat_df["availability_score"] < 0) | (flat_df["availability_score"] > 1)).sum()),
        },
        "notes": [
            "Offensive archetypes: all 11 embedding dimensions L1-normalized to preserve source role mix proportions.",
            "Defensive archetypes: 7 role scores softmax-normalized (poa through mobile_big).",
            "three_point_rate = FG3A/FGA (share of shots from three); free_throw_rate = FTA/FGA (standard FT rate, can exceed 1.0).",
            "hustle_pctl sourced from defensive_archetypes_v2 percentile; foul_rate = PF per 36 minutes.",
            "on_off_diff = actual team on/off net rating differential from BKE decomposition (not individual NET_RTG).",
            "Stability integrates v2.9 diagnostics + v3.0 shrinkage lambda.",
            "availability_score = games_ratio * age_factor; role-adjusted (major rotation / high salary expected 82 GP, bench softer).",
        ],
    }

    # Step 4 specific validation report
    avail_report = {
        "rows": int(len(flat_df)),
        "seasons": sorted(flat_df["season"].astype(str).unique().tolist(), key=lambda s: int(s[:4])),
        "availability_summary": validation_report["availability_summary"],
        "by_age_band": {},
        "by_role": {},
        "constants_used": {
            "age_breakpoints": AVAIL_AGE_BREAKPOINTS,
            "age_factors": AVAIL_AGE_FACTORS,
            "major_rotation_mpg": AVAIL_MAJOR_ROTATION_MPG,
            "high_salary_threshold": AVAIL_HIGH_SALARY_THRESHOLD,
            "full_season_games": AVAIL_FULL_SEASON_GAMES,
            "min_expected_games": AVAIL_MIN_EXPECTED_GAMES,
        },
    }
    # Availability by age band
    for label, lo, hi in [("<25", 0, 25), ("25-29", 25, 30), ("30-34", 30, 35), ("35+", 35, 50)]:
        mask = (flat_df["age"] >= lo) & (flat_df["age"] < hi)
        if mask.sum() > 0:
            avail_report["by_age_band"][label] = {
                "count": int(mask.sum()),
                "mean": float(flat_df.loc[mask, "availability_score"].mean()),
                "median": float(flat_df.loc[mask, "availability_score"].median()),
            }
    # Availability by role (major rotation vs bench)
    major_mask = (flat_df["mpg"] >= AVAIL_MAJOR_ROTATION_MPG) | (flat_df["salary"] >= AVAIL_HIGH_SALARY_THRESHOLD)
    for label, mask in [("major_rotation", major_mask), ("bench", ~major_mask)]:
        if mask.sum() > 0:
            avail_report["by_role"][label] = {
                "count": int(mask.sum()),
                "mean": float(flat_df.loc[mask, "availability_score"].mean()),
                "median": float(flat_df.loc[mask, "availability_score"].median()),
            }
    STEP4_VALIDATION_REPORT.write_text(json.dumps(avail_report, indent=2), encoding="utf-8")

    PLAYER_PROFILES_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    flat_df.to_parquet(PLAYER_PROFILES_PARQUET, index=False)
    with open(PLAYER_PROFILES_PKL, "wb") as handle:
        pickle.dump(profile_rows, handle)
    STEP1_VALIDATION_REPORT.write_text(json.dumps(validation_report, indent=2), encoding="utf-8")

    print(f"Saved Step 1 profiles parquet: {PLAYER_PROFILES_PARQUET}")
    print(f"Saved Step 1 profiles pickle: {PLAYER_PROFILES_PKL}")
    print(f"Saved Step 1 validation report: {STEP1_VALIDATION_REPORT}")
    print(f"Saved Step 4 availability report: {STEP4_VALIDATION_REPORT}")
    print(json.dumps(validation_report, indent=2))


if __name__ == "__main__":
    main()
