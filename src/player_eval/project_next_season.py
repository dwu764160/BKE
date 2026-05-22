"""
src/player_eval/project_next_season.py
=============================================================================
Forward Projection Pipeline — Project Player Impact Profiles to Next Season

Takes prior season player impact profiles and produces projected profiles
for the upcoming season. Handles:
    1. Age-based impact projection with conditional regression-to-mean
    2. Team mapping (carry forward or from roster file)
    3. Draft-aware rookie/newcomer projection (no CSV dependency by default)
    4. Minute projection (age + impact + salary + depth, then team normalization)

Modes:
  - Backtest: Project season N from season N-1 using actual N data for
        team mappings and rookie identification. Enables validation.
  - Forecast: Project next season from latest available season using
        draft history plus optional roster overrides. No target-season actuals.

Inputs:
  data/processed/player_eval/player_impact_profiles.parquet
    data/historical/player_draft_history.parquet
    data/historical/player_salaries_{season}.parquet
    aggregate/player_profile_aggregate.parquet (fallback draft source)

Output:
  data/processed/forecast/projected_player_profiles.parquet

Usage:
  python3 src/player_eval/project_next_season.py                    # Backtest all available seasons
  python3 src/player_eval/project_next_season.py --forecast 2025-26 # True forecast
=============================================================================
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.player_eval.constants import (
    AGE_CURVE_BREAKPOINTS,
    AGE_CURVE_DELTAS,
    MINUTE_PREDICTIONS_PATH,
    MINUTE_MODEL_PATH,
    ENABLE_IMPACT_REGRESSION_TO_MEAN,
    ENABLE_MINUTES_IMPACT_ADJUSTMENT,
    ENABLE_MINUTES_SALARY_ADJUSTMENT,
    ENABLE_MINUTES_TEAM_COMPETITION_ADJUSTMENT,
    IMPACT_REGRESSION_BASE_WEIGHT,
    IMPACT_REGRESSION_LOW_MPG_THRESHOLD,
    IMPACT_REGRESSION_MAX_WEIGHT,
    IMPACT_REGRESSION_MINUTES_WEIGHT,
    IMPACT_REGRESSION_MIN_WEIGHT,
    IMPACT_REGRESSION_STABILITY_WEIGHT,
    HISTORICAL_DIR,
    MINUTES_COMPETITION_MAX_PENALTY,
    MINUTES_COMPETITION_PENALTY,
    MINUTES_IMPACT_ADJUST_CLIP,
    MINUTES_IMPACT_ADJUST_SLOPE,
    MINUTES_SALARY_ADJUST_CLIP,
    MINUTES_SALARY_CHANGE_SCALE_M,
    MINUTES_SALARY_CHANGE_WEIGHT,
    MINUTES_SALARY_LEVEL_SCALE_M,
    MINUTES_SALARY_LEVEL_WEIGHT,
    PLAYER_PROFILES_PARQUET,
    PLAYER_DRAFT_HISTORY_PATH,
    PROFILE_AGGREGATE_PATH,
    PROJECTED_PROFILES_PATH,
    FORECAST_VALIDATION_REPORT,
    PRESEASON_ROSTERS_PATH,
    PRESEASON_ROSTERS_DIR,
    ROOKIE_IMPACT_SCALE_DEFAULT,
    ROOKIE_IMPACT_SCALE_GRID,
    ROOKIE_DEFAULT_3PT_RATE,
    ROOKIE_DEFAULT_AST_RATE,
    ROOKIE_DEFAULT_EFG,
    ROOKIE_DEFAULT_FTR,
    ROOKIE_DEFAULT_TOV_RATE,
    ROOKIE_DEFAULT_USAGE,
    ROOKIE_IMPACT_BY_TIER,
    ROOKIE_MPG_BY_TIER,
    ROOKIES_INPUT_PATH,
    # Forecast structural improvements
    ENABLE_FORECAST_AVAILABILITY_DISCOUNT,
    FORECAST_AVAIL_STAR_GAMES_FRACTION,
    FORECAST_AVAIL_STARTER_GAMES_FRACTION,
    FORECAST_AVAIL_ROTATION_GAMES_FRACTION,
    FORECAST_AVAIL_BENCH_GAMES_FRACTION,
    FORECAST_AVAIL_AGE_DISCOUNT_START,
    FORECAST_AVAIL_AGE_DISCOUNT_PER_YEAR,
    FORECAST_AVAIL_AGE_DISCOUNT_MAX,
    FORECAST_AVAIL_PRIOR_GP_DISCOUNT_THRESHOLD,
    FORECAST_AVAIL_PRIOR_GP_DISCOUNT_SLOPE,
    ENABLE_FORECAST_REPLACEMENT_BUFFER,
    FORECAST_REPLACEMENT_BUFFER_FRACTION,
    FORECAST_REPLACEMENT_IMPACT_BKE,
    FORECAST_REPLACEMENT_IMPACT_OBKE,
    FORECAST_REPLACEMENT_IMPACT_DBKE,
)


# ═════════════════════════════════════════════════════════════════════
# Forecast Structural Improvements
# ═════════════════════════════════════════════════════════════════════


def _availability_games_fraction(mpg: float, age: float, prior_games: float) -> float:
    """Compute expected games-played fraction for a player in the projected season.

    Combines role-based baseline, age penalty, and prior-season GP history.
    Returns a value in (0, 1] that scales projected games.
    """
    # Role-based baseline
    if mpg >= 28.0:
        base = FORECAST_AVAIL_STAR_GAMES_FRACTION
    elif mpg >= 20.0:
        base = FORECAST_AVAIL_STARTER_GAMES_FRACTION
    elif mpg >= 10.0:
        base = FORECAST_AVAIL_ROTATION_GAMES_FRACTION
    else:
        base = FORECAST_AVAIL_BENCH_GAMES_FRACTION

    # Age penalty (multiplicative)
    age_penalty = 0.0
    if np.isfinite(age) and age >= FORECAST_AVAIL_AGE_DISCOUNT_START:
        years_over = age - FORECAST_AVAIL_AGE_DISCOUNT_START
        age_penalty = min(
            years_over * FORECAST_AVAIL_AGE_DISCOUNT_PER_YEAR,
            FORECAST_AVAIL_AGE_DISCOUNT_MAX,
        )

    # Prior-season GP: players who missed games last year are likelier to miss again
    gp_penalty = 0.0
    if np.isfinite(prior_games) and prior_games < FORECAST_AVAIL_PRIOR_GP_DISCOUNT_THRESHOLD:
        shortfall = FORECAST_AVAIL_PRIOR_GP_DISCOUNT_THRESHOLD - prior_games
        gp_penalty = shortfall * FORECAST_AVAIL_PRIOR_GP_DISCOUNT_SLOPE

    fraction = base * (1.0 - age_penalty) * (1.0 - min(gp_penalty, 0.20))
    return float(np.clip(fraction, 0.30, 1.0))


def apply_availability_discount(df: pd.DataFrame) -> pd.DataFrame:
    """Scale projected games and derived minutes by availability factor."""
    if not ENABLE_FORECAST_AVAILABILITY_DISCOUNT:
        return df

    df = df.copy()
    df["age"] = pd.to_numeric(df.get("age"), errors="coerce").fillna(25.0)
    df["mpg"] = pd.to_numeric(df.get("mpg"), errors="coerce").fillna(0.0)
    df["games"] = pd.to_numeric(df.get("games"), errors="coerce").fillna(72.0)

    fractions = df.apply(
        lambda r: _availability_games_fraction(
            float(r.get("mpg", 0)),
            float(r.get("age", 25)),
            float(r.get("games", 72)),
        ),
        axis=1,
    )
    df["availability_discount"] = fractions
    df["games"] = (df["games"] * fractions).round(0).clip(lower=10)
    df["minutes"] = df["mpg"] * df["games"]
    df["possessions"] = df["minutes"] * 2.0

    n_heavy = int((fractions < 0.90).sum())
    mean_frac = float(fractions.mean())
    print(f"    Availability discount applied: mean={mean_frac:.3f}, heavy_discount={n_heavy}")
    return df


def apply_replacement_buffer(df: pd.DataFrame) -> pd.DataFrame:
    """Reserve a fraction of team minutes for replacement-level contributors.

    Scales down each player's minutes proportionally and adds a synthetic
    replacement-pool row per team-season with near-zero impact.
    """
    if not ENABLE_FORECAST_REPLACEMENT_BUFFER:
        return df

    df = df.copy()
    if "is_replacement_pool" not in df.columns:
        df["is_replacement_pool"] = 0

    buffer_frac = FORECAST_REPLACEMENT_BUFFER_FRACTION

    replacement_rows = []
    for (season, team), idx in df.groupby(["season", "team_abbreviation"]).groups.items():
        team_mins = df.loc[idx, "minutes"].sum()
        if team_mins <= 0:
            continue
        # Scale down existing player minutes
        scale = 1.0 - buffer_frac
        df.loc[idx, "mpg"] = df.loc[idx, "mpg"] * scale
        df.loc[idx, "minutes"] = df.loc[idx, "minutes"] * scale

        # Create replacement pool entry
        repl_mins = team_mins * buffer_frac
        repl_games = 82.0
        repl_mpg = repl_mins / repl_games
        replacement_rows.append({
            "player_id": f"repl_{team}_{season}",
            "player_name": f"Replacement Pool ({team})",
            "season": season,
            "team_abbreviation": team,
            "team_id": "",
            "impact_bke": FORECAST_REPLACEMENT_IMPACT_BKE,
            "impact_obke": FORECAST_REPLACEMENT_IMPACT_OBKE,
            "impact_dbke": FORECAST_REPLACEMENT_IMPACT_DBKE,
            "impact_orapm": FORECAST_REPLACEMENT_IMPACT_BKE * 6.0,
            "impact_drapm": FORECAST_REPLACEMENT_IMPACT_DBKE * 6.0,
            "impact_total_impact": 40.0,
            "impact_bpm": -2.0,
            "impact_stability": 0.30,
            "impact_ws": 0.0,
            "impact_vorp": 0.0,
            "impact_portable_talent": 40.0,
            "behavioral_usage": 0.16,
            "behavioral_assist_rate": 0.10,
            "behavioral_turnover_rate": 0.14,
            "behavioral_three_point_rate": 0.35,
            "behavioral_efg": 0.48,
            "behavioral_free_throw_rate": 0.25,
            "behavioral_rim_rate": 0.12,
            "behavioral_orb_rate": 0.04,
            "behavioral_drb_rate": 0.10,
            "behavioral_hustle_pctl": 0.30,
            "behavioral_foul_rate": 3.5,
            "mpg": repl_mpg,
            "minutes": repl_mins,
            "games": repl_games,
            "possessions": repl_mins * 2.0,
            "age": 24.0,
            "experience_years": 2.0,
            "off_primary_archetype": "Connector",
            "def_primary_archetype": "Wing Defender",
            "off_role_confidence": 0.2,
            "def_role_confidence": 0.2,
            "salary": 1_500_000,
            "availability_score": 0.5,
            "is_replacement_pool": 1,
        })

    if replacement_rows:
        repl_df = pd.DataFrame(replacement_rows)
        # Ensure columns align
        for col in df.columns:
            if col not in repl_df.columns:
                if col == "playtype_vector":
                    repl_df[col] = [np.zeros(11).tolist()] * len(repl_df)
                elif col == "offensive_archetype_probs":
                    repl_df[col] = [{}] * len(repl_df)
                elif col == "defensive_archetype_probs":
                    repl_df[col] = [{}] * len(repl_df)
                elif df[col].dtype in [float, np.float64]:
                    repl_df[col] = np.nan
                else:
                    repl_df[col] = ""
        repl_df = repl_df[[c for c in df.columns if c in repl_df.columns]]
        df = pd.concat([df, repl_df], ignore_index=True)
        print(f"    Replacement buffer: {len(replacement_rows)} pools, "
              f"{buffer_frac:.0%} of minutes reserved")

    return df


def _replacement_pool_mask(df: pd.DataFrame) -> pd.Series:
    """Identify synthetic replacement-pool rows robustly across old/new schemas."""
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


def _exclude_replacement_pool_rows(df: pd.DataFrame, context: str) -> pd.DataFrame:
    """Drop synthetic replacement rows so downstream calculations use real players only."""
    mask = _replacement_pool_mask(df)
    removed = int(mask.sum())
    if removed:
        print(f"    Excluding {removed} replacement-pool rows from {context}")
        return df.loc[~mask].copy()
    return df


# ═════════════════════════════════════════════════════════════════════
# Age Curve
# ═════════════════════════════════════════════════════════════════════


def _norm_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()


def season_start_year(season: str) -> int:
    return int(str(season)[:4])

def age_delta(age: float) -> float:
    """Return per-year impact delta for a player at given age.

    Uses piecewise linear interpolation between AGE_CURVE_BREAKPOINTS.
    Returns the expected change in BKE-scale impact for one year of aging.
    """
    if np.isnan(age):
        return 0.0

    # Below first breakpoint
    if age < AGE_CURVE_BREAKPOINTS[0]:
        return AGE_CURVE_DELTAS[0]

    # Between breakpoints: use the delta for the bracket the age falls in
    for i, bp in enumerate(AGE_CURVE_BREAKPOINTS):
        if age < bp:
            return AGE_CURVE_DELTAS[i]

    # Above last breakpoint
    return AGE_CURVE_DELTAS[-1]


def _regression_weight(row: pd.Series) -> float:
    """Compute conditional regression weight for impact projection."""
    if not ENABLE_IMPACT_REGRESSION_TO_MEAN:
        return 0.0

    stability = float(pd.to_numeric(pd.Series([row.get("impact_stability")]), errors="coerce").iloc[0])
    mpg = float(pd.to_numeric(pd.Series([row.get("mpg")]), errors="coerce").iloc[0])

    w = IMPACT_REGRESSION_BASE_WEIGHT
    if np.isfinite(stability):
        w += IMPACT_REGRESSION_STABILITY_WEIGHT * max(0.0, 1.0 - float(np.clip(stability, 0.0, 1.0)))
    if np.isfinite(mpg) and mpg < IMPACT_REGRESSION_LOW_MPG_THRESHOLD:
        shortfall = (IMPACT_REGRESSION_LOW_MPG_THRESHOLD - mpg) / max(IMPACT_REGRESSION_LOW_MPG_THRESHOLD, 1e-6)
        w += IMPACT_REGRESSION_MINUTES_WEIGHT * max(0.0, shortfall)

    return float(np.clip(w, IMPACT_REGRESSION_MIN_WEIGHT, IMPACT_REGRESSION_MAX_WEIGHT))


def apply_impact_projection(row: pd.Series, base_means: Dict[str, float]) -> pd.Series:
    """Apply impact projection with regression-to-mean + age adjustment."""
    row = row.copy()
    age = float(row.get("age", 25))
    delta = age_delta(age)
    reg_w = _regression_weight(row)

    def _project(metric: str, default: float, age_scale: float) -> float:
        mean_val = float(base_means.get(metric, default))
        current_val = float(pd.to_numeric(pd.Series([row.get(metric, mean_val)]), errors="coerce").iloc[0])
        if not np.isfinite(current_val):
            current_val = mean_val
        return (1.0 - reg_w) * current_val + reg_w * mean_val + delta * age_scale

    row["impact_bke"] = _project("impact_bke", 0.0, 1.0)
    row["impact_obke"] = _project("impact_obke", 0.0, 0.5)
    row["impact_dbke"] = _project("impact_dbke", 0.0, 0.5)
    row["impact_orapm"] = _project("impact_orapm", 0.0, 6.0)
    row["impact_drapm"] = _project("impact_drapm", 0.0, 6.0)
    row["impact_total_impact"] = _project("impact_total_impact", 50.0, 15.0)
    row["impact_bpm"] = _project("impact_bpm", 0.0, 3.6)

    ws_val = _project("impact_ws", 0.0, 0.0)
    vorp_val = _project("impact_vorp", 0.0, 0.0)
    row["impact_ws"] = max(0.0, ws_val * (1.0 + delta * 0.50))
    row["impact_vorp"] = max(0.0, vorp_val * (1.0 + delta * 0.50))

    # Age the player
    row["age"] = age + 1.0
    row["experience_years"] = float(row.get("experience_years", 0)) + 1.0

    return row


# ═════════════════════════════════════════════════════════════════════
# Draft + Salary + Team metadata helpers
# ═════════════════════════════════════════════════════════════════════


def load_target_salary_data(target_season: str) -> Tuple[Dict[str, float], Dict[str, str]]:
    """Load known target-season salary and team maps for minute projection."""
    salary_path = HISTORICAL_DIR / f"player_salaries_{target_season}.parquet"
    if not salary_path.exists():
        return {}, {}

    df = pd.read_parquet(salary_path)
    if df.empty or "player_id" not in df.columns:
        return {}, {}

    df["player_id"] = _norm_id(df["player_id"])
    df["salary"] = pd.to_numeric(df.get("salary"), errors="coerce")
    salary_map = (
        df.dropna(subset=["salary"])      # keep only known salary rows
        .drop_duplicates(subset=["player_id"], keep="first")
        .set_index("player_id")["salary"]
        .to_dict()
    )

    # Team field can be full team name in salary files.
    team_name_to_abbr = {}
    teams_path = HISTORICAL_DIR / "teams.parquet"
    if teams_path.exists():
        teams = pd.read_parquet(teams_path)
        for _, r in teams.iterrows():
            abbr = str(r.get("abbreviation", "")).upper()
            full_name = str(r.get("full_name", "")).lower().strip()
            team_name_to_abbr[full_name] = abbr

    team_map = {}
    if "team" in df.columns:
        for _, r in df.drop_duplicates(subset=["player_id"], keep="first").iterrows():
            pid = str(r.get("player_id", ""))
            team_raw = str(r.get("team", "")).strip()
            team_abbr = ""
            if len(team_raw) <= 4 and team_raw.isupper():
                team_abbr = team_raw
            else:
                team_abbr = team_name_to_abbr.get(team_raw.lower().strip(), "")
            if pid:
                team_map[pid] = team_abbr

    return salary_map, team_map


def load_draft_data() -> pd.DataFrame:
    """Load draft metadata using aggregate + fetched history union."""
    parts = []

    if PROFILE_AGGREGATE_PATH.exists():
        agg = pd.read_parquet(PROFILE_AGGREGATE_PATH)
        wanted = [
            "player_id",
            "player_name",
            "draft_class_year",
            "draft_round",
            "draft_pick_in_round",
            "draft_pick_overall",
            "draft_tier",
            "draft_team_abbreviation",
            "draft_source",
        ]
        keep = [c for c in wanted if c in agg.columns]
        if keep and "player_id" in keep:
            draft = agg[keep].copy()
            draft["player_id"] = _norm_id(draft["player_id"])
            parts.append(draft.drop_duplicates(subset=["player_id"], keep="first"))

    if PLAYER_DRAFT_HISTORY_PATH.exists():
        draft = pd.read_parquet(PLAYER_DRAFT_HISTORY_PATH)
        if not draft.empty and "player_id" in draft.columns:
            draft = draft.copy()
            draft["player_id"] = _norm_id(draft["player_id"])
            parts.append(draft.drop_duplicates(subset=["player_id"], keep="first"))

    if parts:
        combined = pd.concat(parts, ignore_index=True, sort=False)
        # Aggregate-first precedence for overlapping IDs/columns.
        combined = combined.drop_duplicates(subset=["player_id"], keep="first")
        return combined

    return pd.DataFrame()


def _draft_lookup(draft_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    if draft_df.empty or "player_id" not in draft_df.columns:
        return {}
    out: Dict[str, Dict[str, float]] = {}
    for _, r in draft_df.iterrows():
        pid = str(r.get("player_id", "")).strip()
        if not pid:
            continue
        pick = pd.to_numeric(pd.Series([r.get("draft_pick_overall")]), errors="coerce").iloc[0]
        cls_year = pd.to_numeric(pd.Series([r.get("draft_class_year")]), errors="coerce").iloc[0]
        tier = r.get("draft_tier")
        if pd.isna(tier) or str(tier).strip() == "":
            tier = _draft_tier(int(pick) if pd.notna(pick) else 999)
        out[pid] = {
            "draft_pick_overall": float(pick) if pd.notna(pick) else np.nan,
            "draft_class_year": float(cls_year) if pd.notna(cls_year) else np.nan,
            "draft_tier": str(tier),
            "draft_team_abbreviation": str(r.get("draft_team_abbreviation", "")).upper(),
        }
    return out


def _rookie_tier_for_player(player_id: str, exp: float, draft_map: Dict[str, Dict[str, float]]) -> str:
    info = draft_map.get(str(player_id), {})
    pick = info.get("draft_pick_overall", np.nan)
    if pd.notna(pick):
        return _draft_tier(int(pick))
    if exp <= 0:
        return "late_first"
    if exp <= 1:
        return "second_round"
    return "undrafted"


def tune_rookie_impact_scale(all_profiles: pd.DataFrame, draft_map: Dict[str, Dict[str, float]]) -> Tuple[float, Dict]:
    """Tune rookie impact multiplier using historical backtest rookie MAE."""
    seasons = sorted(all_profiles["season"].astype(str).unique())
    if len(seasons) < 2:
        return ROOKIE_IMPACT_SCALE_DEFAULT, {"tuned": False, "reason": "insufficient seasons"}

    evaluations = []
    for scale in ROOKIE_IMPACT_SCALE_GRID:
        errors = []
        n_rookies = 0

        for i in range(len(seasons) - 1):
            base_season = seasons[i]
            target_season = seasons[i + 1]

            base_ids = set(_norm_id(all_profiles[all_profiles["season"] == base_season]["player_id"]))
            target = all_profiles[all_profiles["season"] == target_season].copy()
            target["player_id"] = _norm_id(target["player_id"])
            rookies = target[~target["player_id"].isin(base_ids)].copy()
            if rookies.empty:
                continue

            for _, r in rookies.iterrows():
                pid = str(r.get("player_id", ""))
                exp = float(pd.to_numeric(pd.Series([r.get("experience_years")]), errors="coerce").fillna(0.0).iloc[0])
                tier = _rookie_tier_for_player(pid, exp, draft_map)
                pred = ROOKIE_IMPACT_BY_TIER.get(tier, ROOKIE_IMPACT_BY_TIER["undrafted"]) * scale
                actual = float(pd.to_numeric(pd.Series([r.get("impact_bke")]), errors="coerce").fillna(0.0).iloc[0])
                errors.append(abs(pred - actual))
                n_rookies += 1

        if errors:
            mae = float(np.mean(errors))
            evaluations.append({"scale": float(scale), "rookie_bke_mae": round(mae, 4), "n_rookies": int(n_rookies)})

    if not evaluations:
        return ROOKIE_IMPACT_SCALE_DEFAULT, {"tuned": False, "reason": "no rookie rows in backtest"}

    best = min(evaluations, key=lambda x: x["rookie_bke_mae"])
    return float(best["scale"]), {
        "tuned": True,
        "selected_scale": float(best["scale"]),
        "evaluations": evaluations,
    }


# ═════════════════════════════════════════════════════════════════════
# Minute Projection (forecast-safe, no leakage)
# ═════════════════════════════════════════════════════════════════════


def _coarse_role(value: str) -> str:
    text = str(value or "").lower()
    if any(k in text for k in ["center", "forward-center", "pf", "c"]):
        return "big"
    if any(k in text for k in ["guard", "pg", "sg"]):
        return "guard"
    return "wing"


def _mpg_age_delta(age: float) -> float:
    if np.isnan(age):
        return 0.0
    if age < 22:
        return +2.0
    if age < 24:
        return +1.0
    if age < 27:
        return +0.5
    if age < 30:
        return 0.0
    if age < 34:
        return -1.0
    return -2.0


def _collect_minutes_training_rows(
    all_profiles: pd.DataFrame,
    max_base_season: Optional[str] = None,
) -> pd.DataFrame:
    """Build returning-player minute carry rows from adjacent historical seasons."""
    seasons = sorted(all_profiles["season"].astype(str).unique())
    if len(seasons) < 2:
        return pd.DataFrame()

    cutoff_year = season_start_year(max_base_season) if max_base_season else None
    rows = []

    for i in range(len(seasons) - 1):
        base_season = seasons[i]
        target_season = seasons[i + 1]
        if cutoff_year is not None and season_start_year(base_season) > cutoff_year:
            continue

        base = collapse_profiles_for_projection(all_profiles[all_profiles["season"] == base_season].copy())
        target = collapse_profiles_for_projection(all_profiles[all_profiles["season"] == target_season].copy())
        if base.empty or target.empty:
            continue

        base = base[["player_id", "mpg", "impact_bke", "age", "salary"]].copy()
        target = target[["player_id", "mpg"]].copy().rename(columns={"mpg": "target_mpg"})
        base["player_id"] = _norm_id(base["player_id"])
        target["player_id"] = _norm_id(target["player_id"])

        merged = base.merge(target, on="player_id", how="inner")
        if merged.empty:
            continue

        merged["base_mpg"] = pd.to_numeric(merged["mpg"], errors="coerce")
        merged["base_impact"] = pd.to_numeric(merged["impact_bke"], errors="coerce")
        merged["base_age"] = pd.to_numeric(merged["age"], errors="coerce")
        merged["base_salary"] = pd.to_numeric(merged["salary"], errors="coerce")
        merged["target_mpg"] = pd.to_numeric(merged["target_mpg"], errors="coerce")
        merged = merged.dropna(subset=["base_mpg", "target_mpg"]).copy()
        if merged.empty:
            continue

        rows.append(merged[["base_mpg", "base_impact", "base_age", "base_salary", "target_mpg"]])

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def fit_minutes_carry_model(
    all_profiles: pd.DataFrame,
    max_base_season: Optional[str] = None,
) -> Dict[str, float]:
    """Fit a lightweight linear carry model for MPG using only prior transitions."""
    default_model = {
        "intercept": 0.8,
        "w_mpg": 0.90,
        "w_impact": 0.15,
        "w_age": -0.05,
        "w_salary": 0.10,
        "fitted": False,
        "n_samples": 0,
        "fit_corr": None,
    }

    train = _collect_minutes_training_rows(all_profiles, max_base_season=max_base_season)
    if train.empty or len(train) < 150:
        return default_model

    base_mpg = train["base_mpg"].to_numpy(dtype=float)
    base_impact = train["base_impact"].fillna(0.0).to_numpy(dtype=float)
    base_age_excess = np.maximum(train["base_age"].fillna(27.0).to_numpy(dtype=float) - 27.0, 0.0)
    salary_m = (train["base_salary"].fillna(8_000_000.0).to_numpy(dtype=float) / 1_000_000.0)
    salary_signal = np.tanh((salary_m - 8.0) / 10.0)
    y = train["target_mpg"].to_numpy(dtype=float)

    X = np.column_stack([
        np.ones(len(train), dtype=float),
        base_mpg,
        base_impact,
        base_age_excess,
        salary_signal,
    ])

    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return default_model

    pred = X @ beta
    corr = float(np.corrcoef(pred, y)[0, 1]) if len(y) > 1 else np.nan

    model = {
        "intercept": float(np.clip(beta[0], -2.0, 4.0)),
        "w_mpg": float(np.clip(beta[1], 0.85, 0.98)),
        "w_impact": float(np.clip(beta[2], -0.50, 0.50)),
        "w_age": float(np.clip(beta[3], -0.20, 0.05)),
        "w_salary": float(np.clip(beta[4], -0.30, 0.30)),
        "fitted": True,
        "n_samples": int(len(train)),
        "fit_corr": float(round(corr, 4)) if np.isfinite(corr) else None,
    }

    # Guardrail: prefer fallback only when fit is genuinely weak.
    # A 0.75 cutoff keeps noisy fits out while allowing useful carry signal.
    if not np.isfinite(corr) or corr < 0.75:
        fallback = default_model.copy()
        fallback["n_samples"] = int(len(train))
        fallback["fit_corr"] = float(round(corr, 4)) if np.isfinite(corr) else None
        return fallback

    return model

def project_minutes(
    players: pd.DataFrame,
    target_salary_map: Optional[Dict[str, float]] = None,
    draft_map: Optional[Dict[str, Dict[str, float]]] = None,
    minutes_carry_model: Optional[Dict[str, float]] = None,
    team_mpg_target: float = 350.0,
    mpg_cap: float = 38.0,
) -> pd.DataFrame:
    """Project MPG using the temporal Ridge minute model predictions.

     Steps:
        1. Load pre-computed predictions for returning players.
        2. Apply rookie lookup table logic for new players.
        3. Normalize within each team so totals match target.
        4. Cap individual MPG and redistribute excess.
    """
    df = players.copy()
    df["player_id"] = _norm_id(df["player_id"])

    # Load Ridge predictions
    try:
        preds = pd.read_parquet(MINUTE_PREDICTIONS_PATH)
        # We need to match on player_id and season
        preds["player_id"] = _norm_id(preds["player_id"])
        preds["season"] = preds["season"].astype(str)
        # Map predictions to df
        merged = df.merge(preds[["player_id", "season", "pred_mpg_raw"]], on=["player_id", "season"], how="left")
        df["adjusted_mpg"] = merged["pred_mpg_raw"]
    except Exception as e:
        print(f"Warning: Could not load minute model predictions: {e}")
        df["adjusted_mpg"] = np.nan

    # For rookies and missing players, use fallback logic
    missing_mask = df["adjusted_mpg"].isna()
    if missing_mask.any():
        print(f"    Falling back to rookie lookup for {missing_mask.sum()} players")
        try:
            with open(MINUTE_MODEL_PATH, "rb") as f:
                model_data = pickle.load(f)
                rookie_lookup = model_data.get("rookie_lookup", {})
        except Exception:
            rookie_lookup = {"default": 12.0}

        def get_rookie_mpg(pid):
            if not draft_map or pid not in draft_map:
                return rookie_lookup.get("undrafted", rookie_lookup.get("default", 12.0))
            info = draft_map[pid]
            r = info.get("draft_round", np.nan)
            p = info.get("draft_pick_overall", np.nan)
            if pd.isna(r) or pd.isna(p):
                return rookie_lookup.get("undrafted", 12.0)
            r = int(r)
            p = int(p)
            if r == 1:
                if p <= 5: return rookie_lookup.get("r1_1_5", 24.0)
                elif p <= 15: return rookie_lookup.get("r1_6_15", 20.0)
                else: return rookie_lookup.get("r1_16_30", 16.0)
            elif r == 2:
                return rookie_lookup.get("r2", 8.0)
            return rookie_lookup.get("undrafted", 12.0)

        fallback_vals = df.loc[missing_mask, "player_id"].apply(get_rookie_mpg)
        df.loc[missing_mask, "adjusted_mpg"] = fallback_vals

    df["adjusted_mpg"] = df["adjusted_mpg"].clip(lower=0.0)

    # ── Team normalization: proportional scaling to roster-size-adaptive target ──
    per_player_target = max(5.0, float(team_mpg_target) / 19.0)

    for (season, team), idx in df.groupby(["season", "team_abbreviation"]).groups.items():
        vals = df.loc[idx, "adjusted_mpg"].copy()
        total = vals.sum()
        n = len(idx)
        adaptive_target = n * per_player_target

        if total <= 0:
            df.loc[idx, "projected_mpg"] = adaptive_target / max(n, 1)
            continue

        scale = adaptive_target / total
        mpg_vals = vals * scale

        for _ in range(5):
            capped = mpg_vals.clip(upper=mpg_cap)
            excess = mpg_vals.sum() - capped.sum()
            if excess <= 0.1:
                mpg_vals = capped
                break
            free_mask = capped < mpg_cap
            if free_mask.sum() == 0:
                mpg_vals = capped
                break
            free_total = capped[free_mask].sum()
            if free_total <= 0:
                mpg_vals = capped
                break
            capped.loc[free_mask] = capped[free_mask] + excess * (capped[free_mask] / free_total)
            mpg_vals = capped
        df.loc[idx, "projected_mpg"] = mpg_vals

    df["mpg"] = df["projected_mpg"]
    return df


def collapse_profiles_for_projection(profiles: pd.DataFrame) -> pd.DataFrame:
    """Collapse potential stint-split rows to one row per player-season for projection.

    Impact/behavioral fields are portable and shared across stints; volume fields are
    summed across stints so the projection starts from full-season context.
    """
    if profiles.empty:
        return profiles.copy()

    df = profiles.copy()
    df["player_id"] = _norm_id(df["player_id"])
    df["season"] = df["season"].astype(str)

    # Prefer final stint row for base metadata/team carry-forward tie-breaks.
    sort_cols = ["player_id"]
    asc = [True]
    if "is_final_stint" in df.columns:
        sort_cols.append("is_final_stint")
        asc.append(False)
    if "stint_number" in df.columns:
        sort_cols.append("stint_number")
        asc.append(False)
    df = df.sort_values(sort_cols, ascending=asc)

    base = df.drop_duplicates(subset=["player_id", "season"], keep="first").copy()

    for col in ["minutes", "games", "possessions"]:
        if col in df.columns:
            sums = (
                pd.to_numeric(df[col], errors="coerce")
                .groupby([df["player_id"], df["season"]])
                .sum(min_count=1)
                .reset_index(name=f"{col}_sum")
            )
            base = base.merge(sums, on=["player_id", "season"], how="left")
            base[col] = pd.to_numeric(base[f"{col}_sum"], errors="coerce").fillna(
                pd.to_numeric(base.get(col), errors="coerce")
            )
            base = base.drop(columns=[f"{col}_sum"], errors="ignore")

    base["games"] = pd.to_numeric(base.get("games"), errors="coerce").fillna(0.0)
    base["minutes"] = pd.to_numeric(base.get("minutes"), errors="coerce").fillna(0.0)
    base["mpg"] = np.where(base["games"] > 0, base["minutes"] / base["games"], base.get("mpg", 0.0))
    return base.reset_index(drop=True)


def _load_manual_roster_map(roster_path: Optional[Path]) -> Dict[str, str]:
    if not roster_path or not roster_path.exists():
        return {}
    roster = pd.read_csv(roster_path)
    if "player_id" not in roster.columns:
        return {}
    roster["player_id"] = _norm_id(roster["player_id"])
    team_col = None
    for c in ["new_team", "team", "team_abbreviation"]:
        if c in roster.columns:
            team_col = c
            break
    if not team_col:
        return {}

    out = {}
    for pid, team in zip(roster["player_id"], roster[team_col]):
        pid_s = str(pid).strip()
        team_s = str(team).strip().upper()
        if pid_s and team_s and team_s not in {"NAN", "NONE"}:
            out[pid_s] = team_s
    return out


def _resolve_preseason_roster_path(target_season: str, preseason_rosters_path: Optional[Path]) -> Optional[Path]:
    if preseason_rosters_path and preseason_rosters_path.exists():
        if preseason_rosters_path.is_dir():
            candidate = preseason_rosters_path / f"preseason_rosters_{target_season}.parquet"
            if candidate.exists():
                return candidate
        return preseason_rosters_path

    season_path = PRESEASON_ROSTERS_DIR / f"preseason_rosters_{target_season}.parquet"
    if season_path.exists():
        return season_path
    if PRESEASON_ROSTERS_PATH.exists():
        return PRESEASON_ROSTERS_PATH
    return None


def load_preseason_roster_maps(
    target_season: str,
    preseason_rosters_path: Optional[Path],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Load player->team and player->team_id maps from preseason roster snapshot."""
    src = _resolve_preseason_roster_path(target_season, preseason_rosters_path)
    if not src:
        return {}, {}

    df = pd.read_parquet(src)
    if df.empty or "player_id" not in df.columns:
        return {}, {}

    df = df.copy()
    df["player_id"] = _norm_id(df["player_id"])
    if "season" in df.columns:
        df["season"] = df["season"].astype(str)
        df = df[df["season"] == target_season].copy()
    if df.empty:
        return {}, {}

    team_col = "team_abbreviation" if "team_abbreviation" in df.columns else None
    team_id_col = "team_id" if "team_id" in df.columns else None
    if not team_col:
        return {}, {}

    team_map = {}
    team_id_map = {}
    for _, row in df.iterrows():
        pid = str(row.get("player_id", "")).strip()
        team = str(row.get(team_col, "")).strip().upper()
        if not pid or not team or team in {"NAN", "NONE"}:
            continue
        team_map[pid] = team
        if team_id_col:
            tid = str(row.get(team_id_col, "")).strip()
            if tid and tid not in {"NAN", "NONE"}:
                team_id_map[pid] = tid
    return team_map, team_id_map


def _select_end_of_season_target_rows(target_profiles: pd.DataFrame) -> pd.DataFrame:
    """Select one target-season row per player using final stint preference."""
    target = target_profiles.copy()
    target["player_id"] = _norm_id(target["player_id"])

    sort_cols = ["player_id"]
    asc = [True]
    if "is_final_stint" in target.columns:
        sort_cols.append("is_final_stint")
        asc.append(False)
    if "stint_last_game_date" in target.columns:
        target["_stint_last_game_date"] = pd.to_datetime(target["stint_last_game_date"], errors="coerce")
        sort_cols.append("_stint_last_game_date")
        asc.append(False)
    if "stint_number" in target.columns:
        sort_cols.append("stint_number")
        asc.append(False)

    target = target.sort_values(sort_cols, ascending=asc)
    target = target.drop_duplicates(subset=["player_id"], keep="first")
    return target


# ═════════════════════════════════════════════════════════════════════
# Team Mapping
# ═════════════════════════════════════════════════════════════════════

def map_players_to_teams_backtest(
    prior_profiles: pd.DataFrame,
    target_profiles: pd.DataFrame,
    team_mapping_mode: str = "end_of_season",
    preseason_team_map: Optional[Dict[str, str]] = None,
    preseason_team_id_map: Optional[Dict[str, str]] = None,
    roster_override_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Map backtest players to target-season teams under selected scenario."""
    preseason_team_map = preseason_team_map or {}
    preseason_team_id_map = preseason_team_id_map or {}
    roster_override_map = roster_override_map or {}

    prior = prior_profiles.copy()
    prior["player_id"] = _norm_id(prior["player_id"])

    if team_mapping_mode == "preseason_snapshot":
        # If preseason snapshot exists, apply mapped teams and keep carry-forward
        # team assignment for players missing from the snapshot.
        if preseason_team_map:
            mapped_team = prior["player_id"].map(preseason_team_map)
            prior["team_abbreviation"] = mapped_team.fillna(prior.get("team_abbreviation"))
            if "team_id" in prior.columns:
                mapped_team_id = prior["player_id"].map(preseason_team_id_map)
                prior["team_id"] = mapped_team_id.fillna(prior.get("team_id"))
        merged = prior
    else:
        # End-of-season (peek) scenario uses final target-season stint/team.
        target_teams = _select_end_of_season_target_rows(target_profiles)
        target_teams = target_teams[["player_id", "team_abbreviation", "team_id"]].rename(
            columns={"team_abbreviation": "target_team", "team_id": "target_team_id"}
        )
        target_teams["player_id"] = _norm_id(target_teams["player_id"])

        merged = prior.merge(target_teams, on="player_id", how="inner")
        merged["team_abbreviation"] = merged["target_team"]
        merged["team_id"] = merged["target_team_id"]
        merged = merged.drop(columns=["target_team", "target_team_id"], errors="ignore")

    if roster_override_map:
        manual = merged["player_id"].map(roster_override_map)
        merged["team_abbreviation"] = manual.fillna(merged["team_abbreviation"])

    merged["team_assignment_scenario"] = team_mapping_mode
    return merged


def map_players_to_teams_forecast(
    prior_profiles: pd.DataFrame,
    team_mapping_mode: str = "end_of_season",
    preseason_team_map: Optional[Dict[str, str]] = None,
    preseason_team_id_map: Optional[Dict[str, str]] = None,
    roster_override_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Map forecast players to next-season teams under selected scenario."""
    preseason_team_map = preseason_team_map or {}
    preseason_team_id_map = preseason_team_id_map or {}
    roster_override_map = roster_override_map or {}

    df = prior_profiles.copy()
    df["player_id"] = _norm_id(df["player_id"])

    # End-of-season scenario = carry-forward prior team. Preseason scenario overrides.
    if team_mapping_mode == "preseason_snapshot":
        if preseason_team_map:
            mapped_team = df["player_id"].map(preseason_team_map)
            df["team_abbreviation"] = mapped_team.fillna(df.get("team_abbreviation"))
            if "team_id" in df.columns:
                mapped_team_id = df["player_id"].map(preseason_team_id_map)
                df["team_id"] = mapped_team_id.fillna(df.get("team_id"))

    if roster_override_map:
        manual = df["player_id"].map(roster_override_map)
        df["team_abbreviation"] = manual.fillna(df["team_abbreviation"])

    df["team_assignment_scenario"] = team_mapping_mode
    return df


# ═════════════════════════════════════════════════════════════════════
# Rookie Projection
# ═════════════════════════════════════════════════════════════════════

def _draft_tier(pick: int) -> str:
    if pick <= 14:
        return "lottery"
    if pick <= 25:
        return "mid_first"
    if pick <= 30:
        return "late_first"
    if pick <= 60:
        return "second_round"
    return "undrafted"


def _position_from_height(height_inches: float) -> str:
    """Estimate position from height."""
    if np.isnan(height_inches):
        return "Forward"
    if height_inches <= 75:  # <= 6'3"
        return "Guard"
    if height_inches <= 79:  # <= 6'7"
        return "Guard-Forward"
    if height_inches <= 81:  # <= 6'9"
        return "Forward"
    if height_inches <= 83:  # <= 6'11"
        return "Forward-Center"
    return "Center"


def build_rookie_profiles_backtest(
    target_profiles: pd.DataFrame,
    prior_player_ids: set,
    target_season: str,
    draft_map: Dict[str, Dict[str, float]],
    rookie_impact_scale: float,
) -> pd.DataFrame:
    """Build rookie profiles from actual target-season data (backtest).

    Rookies are players in the target season who were NOT in the prior season.
    We use their actual data but at REPLACEMENT level impact to simulate
    what a pre-season projection would look like.
    """
    target = target_profiles.copy()
    target["player_id"] = target["player_id"].astype(str)
    rookies = target[~target["player_id"].isin(prior_player_ids)].copy()

    # Filter out rookies with no valid team
    rookies = rookies[rookies["team_abbreviation"].notna() &
                      (rookies["team_abbreviation"].astype(str) != "nan")].copy()

    if rookies.empty:
        return pd.DataFrame(columns=target.columns)

    for idx, row in rookies.iterrows():
        pid = str(row.get("player_id", ""))
        exp = float(row.get("experience_years", 0))
        tier = _rookie_tier_for_player(pid, exp, draft_map)
        impact = ROOKIE_IMPACT_BY_TIER.get(tier, ROOKIE_IMPACT_BY_TIER["undrafted"]) * rookie_impact_scale

        rookies.at[idx, "impact_bke"] = impact
        rookies.at[idx, "impact_obke"] = impact * 0.5
        rookies.at[idx, "impact_dbke"] = impact * 0.5
        rookies.at[idx, "impact_orapm"] = impact * 6.0
        rookies.at[idx, "impact_drapm"] = impact * 6.0
        rookies.at[idx, "impact_total_impact"] = 50.0 + impact * 20.0
        rookies.at[idx, "impact_bpm"] = impact * 3.0
        rookies.at[idx, "impact_stability"] = 0.35  # Low stability for projections
        rookies.at[idx, "draft_tier"] = tier

        # Use actual minutes as upper guide, still cap to tier-driven rookie envelope.
        actual_mpg = float(row.get("mpg", 0))
        tier_mpg = ROOKIE_MPG_BY_TIER[tier]
        rookies.at[idx, "mpg"] = min(actual_mpg, tier_mpg * 1.25)

    rookies["season"] = target_season
    return rookies


def build_rookie_profiles_from_draft(
    target_season: str,
    prior_player_ids: set,
    profile_columns: list,
    draft_df: pd.DataFrame,
    draft_map: Dict[str, Dict[str, float]],
    roster_path: Optional[Path],
    salary_map: Dict[str, float],
    salary_team_map: Dict[str, str],
    rookie_impact_scale: float,
    allow_salary_team_fallback: bool = True,
) -> pd.DataFrame:
    """Build rookie profiles from fetched draft history (forecast mode)."""
    if draft_df.empty:
        return pd.DataFrame(columns=profile_columns)

    target_year = season_start_year(target_season)
    draft = draft_df.copy()
    draft["player_id"] = _norm_id(draft["player_id"])
    draft["draft_class_year"] = pd.to_numeric(draft.get("draft_class_year"), errors="coerce")

    rookies = draft[draft["draft_class_year"] == target_year].copy()
    rookies = rookies[~rookies["player_id"].isin({str(x) for x in prior_player_ids})].copy()
    if rookies.empty:
        return pd.DataFrame(columns=profile_columns)

    roster_map = {}
    if roster_path and roster_path.exists():
        roster = pd.read_csv(roster_path)
        if "player_id" in roster.columns:
            roster["player_id"] = _norm_id(roster["player_id"])
            team_col = None
            for c in ["new_team", "team", "team_abbreviation"]:
                if c in roster.columns:
                    team_col = c
                    break
            if team_col:
                roster_map = {
                    str(pid): str(team).upper()
                    for pid, team in zip(roster["player_id"], roster[team_col])
                }

    meta_map = {}
    if (HISTORICAL_DIR / "players.parquet").exists():
        meta = pd.read_parquet(HISTORICAL_DIR / "players.parquet")
        if "player_id" in meta.columns:
            meta["player_id"] = _norm_id(meta["player_id"])
            meta = meta.drop_duplicates(subset=["player_id"], keep="first")
            meta_map = meta.set_index("player_id").to_dict(orient="index")

    rows = []

    for _, r in rookies.iterrows():
        pid = str(r.get("player_id", ""))
        name = str(r.get("player_name", f"Rookie {pid}"))
        info = draft_map.get(pid, {})

        pick = pd.to_numeric(pd.Series([info.get("draft_pick_overall", r.get("draft_pick_overall"))]), errors="coerce").iloc[0]
        tier = str(info.get("draft_tier", _draft_tier(int(pick) if pd.notna(pick) else 999)))

        team = roster_map.get(pid)
        if not team and allow_salary_team_fallback:
            team = salary_team_map.get(pid)
        if not team:
            team = str(info.get("draft_team_abbreviation", "")).upper() or "FA"
        if not team or team == "NAN":
            team = "FA"

        meta = meta_map.get(pid, {})
        height = float(pd.to_numeric(pd.Series([meta.get("height_inches", 78)]), errors="coerce").fillna(78).iloc[0])
        weight = float(pd.to_numeric(pd.Series([meta.get("weight_lbs", 210)]), errors="coerce").fillna(210).iloc[0])
        position = str(meta.get("primary_position", _position_from_height(height)))

        impact = ROOKIE_IMPACT_BY_TIER.get(tier, ROOKIE_IMPACT_BY_TIER["undrafted"]) * rookie_impact_scale
        mpg = ROOKIE_MPG_BY_TIER.get(tier, ROOKIE_MPG_BY_TIER["undrafted"])

        profile = {col: 0.0 for col in profile_columns}
        profile.update({
            "player_id": pid,
            "player_name": name,
            "season": target_season,
            "team_abbreviation": team,
            "team_id": "",
            "impact_bke": impact,
            "impact_obke": impact * 0.5,
            "impact_dbke": impact * 0.5,
            "impact_orapm": impact * 6.0,
            "impact_drapm": impact * 6.0,
            "impact_total_impact": 50.0 + impact * 20.0,
            "impact_bpm": impact * 3.0,
            "impact_stability": 0.35,
            "impact_ws": max(0, mpg * 0.04 * 82),
            "impact_vorp": max(0, impact * 3.0 * 0.05),
            "impact_portable_talent": 50.0,
            "behavioral_usage": ROOKIE_DEFAULT_USAGE,
            "behavioral_assist_rate": ROOKIE_DEFAULT_AST_RATE,
            "behavioral_turnover_rate": ROOKIE_DEFAULT_TOV_RATE,
            "behavioral_three_point_rate": ROOKIE_DEFAULT_3PT_RATE,
            "behavioral_efg": ROOKIE_DEFAULT_EFG,
            "behavioral_free_throw_rate": ROOKIE_DEFAULT_FTR,
            "behavioral_rim_rate": 0.15,
            "behavioral_orb_rate": 0.05,
            "behavioral_drb_rate": 0.12,
            "behavioral_hustle_pctl": 0.5,
            "behavioral_foul_rate": 3.0,
            "position_proxy": position,
            "age": float(pd.to_numeric(pd.Series([meta.get("age", 20)]), errors="coerce").fillna(20).iloc[0]),
            "height_inches": height,
            "weight_lbs": weight,
            "experience_years": 0.0,
            "minutes": mpg * 72,  # Assume 72 games for rookies
            "mpg": mpg,
            "minute_share_potential": 0.0,
            "possessions": mpg * 72 * 2.0,
            "games": 72.0,
            "on_off_diff": 0.0,
            "scheme_stability_index": 0.0,
            "compression_flag": 0.0,
            "defensive_shrinkage_lambda": 0.5,
            "salary": float(salary_map.get(pid, 3_000_000)),
            "availability_score": 0.85,
            "off_primary_archetype": "Connector",
            "off_secondary_archetype": "",
            "off_role_confidence": 0.3,
            "off_role_effectiveness": 0.4,
            "def_primary_archetype": "Wing Defender",
            "def_secondary_archetype": "",
            "def_role_confidence": 0.3,
            "draft_pick_overall": pick if pd.notna(pick) else np.nan,
            "draft_class_year": float(target_year),
            "draft_tier": tier,
        })

        # Set default archetype probabilities
        for arch_col in [c for c in profile_columns if c.startswith("off_prob_")]:
            profile[arch_col] = 1.0 / 11.0  # uniform prior
        for arch_col in [c for c in profile_columns if c.startswith("def_prob_")]:
            profile[arch_col] = 1.0 / 7.0   # uniform prior
        for pt_col in [c for c in profile_columns if c.startswith("playtype_")]:
            profile[pt_col] = 1.0 / 11.0    # uniform prior

        rows.append(profile)

    if not rows:
        return pd.DataFrame(columns=profile_columns)

    result = pd.DataFrame(rows)
    # Ensure proper types for non-numeric columns
    for col in ["player_id", "player_name", "season", "team_abbreviation", "team_id",
                 "position_proxy", "off_primary_archetype", "off_secondary_archetype",
                 "def_primary_archetype", "def_secondary_archetype"]:
        if col in result.columns:
            result[col] = result[col].astype(str)

    # Handle dict/list columns
    if "offensive_archetype_probs" in profile_columns:
        result["offensive_archetype_probs"] = [{}] * len(result)
    if "defensive_archetype_probs" in profile_columns:
        result["defensive_archetype_probs"] = [{}] * len(result)
    if "playtype_vector" in profile_columns:
        result["playtype_vector"] = [np.zeros(11).tolist()] * len(result)

    return result


def build_rookie_profiles_from_csv(
    rookies_path: Path,
    target_season: str,
    profile_columns: list,
    rookie_impact_scale: float,
) -> pd.DataFrame:
    """Manual fallback for ad-hoc runs when draft history is unavailable."""
    if not rookies_path.exists():
        return pd.DataFrame(columns=profile_columns)

    rookies_csv = pd.read_csv(rookies_path)
    rows = []
    for _, r in rookies_csv.iterrows():
        pick = int(r.get("draft_position", 60))
        tier = _draft_tier(pick)
        impact = ROOKIE_IMPACT_BY_TIER.get(tier, ROOKIE_IMPACT_BY_TIER["undrafted"]) * rookie_impact_scale
        mpg = ROOKIE_MPG_BY_TIER.get(tier, ROOKIE_MPG_BY_TIER["undrafted"])

        profile = {col: np.nan for col in profile_columns}
        profile.update({
            "player_id": str(r.get("player_id", f"rookie_{pick}")),
            "player_name": str(r.get("player_name", f"Rookie #{pick}")),
            "season": target_season,
            "team_abbreviation": str(r.get("team", "FA")).upper(),
            "impact_bke": impact,
            "impact_obke": impact * 0.5,
            "impact_dbke": impact * 0.5,
            "impact_orapm": impact * 6.0,
            "impact_drapm": impact * 6.0,
            "impact_total_impact": 50.0 + impact * 20.0,
            "impact_bpm": impact * 3.0,
            "impact_stability": 0.35,
            "mpg": mpg,
            "minutes": mpg * 72,
            "games": 72.0,
            "age": float(r.get("age", 20)),
            "experience_years": 0.0,
            "position_proxy": str(r.get("position", "Forward")),
            "salary": float(r.get("salary", 3_000_000)),
            "draft_pick_overall": float(pick),
            "draft_class_year": float(season_start_year(target_season)),
            "draft_tier": tier,
            "behavioral_usage": ROOKIE_DEFAULT_USAGE,
            "behavioral_assist_rate": ROOKIE_DEFAULT_AST_RATE,
            "behavioral_turnover_rate": ROOKIE_DEFAULT_TOV_RATE,
            "behavioral_three_point_rate": ROOKIE_DEFAULT_3PT_RATE,
            "behavioral_efg": ROOKIE_DEFAULT_EFG,
            "behavioral_free_throw_rate": ROOKIE_DEFAULT_FTR,
            "off_primary_archetype": "Connector",
            "def_primary_archetype": "Wing Defender",
        })
        rows.append(profile)

    if not rows:
        return pd.DataFrame(columns=profile_columns)
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════
# Season Helpers
# ═════════════════════════════════════════════════════════════════════

def next_season_str(season: str) -> str:
    """Convert '2023-24' to '2024-25'."""
    start = int(season[:4])
    return f"{start + 1}-{str(start + 2)[-2:]}"


def prior_season_str(season: str) -> str:
    """Convert '2024-25' to '2023-24'."""
    start = int(season[:4])
    return f"{start - 1}-{str(start)[-2:]}"


# ═════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═════════════════════════════════════════════════════════════════════

def project_season(
    base_season: str,
    target_season: str,
    all_profiles: pd.DataFrame,
    draft_df: pd.DataFrame,
    draft_map: Dict[str, Dict[str, float]],
    rookie_impact_scale: float,
    mode: str = "backtest",
    roster_path: Optional[Path] = None,
    preseason_rosters_path: Optional[Path] = None,
    team_mapping_mode: str = "end_of_season",
    rookies_path: Optional[Path] = None,
    target_salary_map: Optional[Dict[str, float]] = None,
    target_salary_team_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Project players from base_season to target_season.

    Args:
        base_season: Source season (e.g., '2023-24')
        target_season: Target season to project into (e.g., '2024-25')
        all_profiles: Full impact profiles DataFrame (all seasons)
        mode: 'backtest' or 'forecast'
        roster_path: Optional CSV with manual team overrides
        preseason_rosters_path: Optional preseason roster parquet path
        team_mapping_mode: 'end_of_season' or 'preseason_snapshot'
        rookies_path: Optional CSV with rookie data for forecast mode

    Returns:
        DataFrame with projected player profiles for target_season
    """
    print(
        f"\n  Projecting {base_season} → {target_season} "
        f"(mode={mode}, team_mapping={team_mapping_mode})"
    )

    # 1. Get base season profiles
    base_raw = all_profiles[all_profiles["season"] == base_season].copy()
    base_raw["player_id"] = base_raw["player_id"].astype(str)
    base = collapse_profiles_for_projection(base_raw)
    if base.empty:
        print(f"    WARNING: No profiles for base season {base_season}")
        return pd.DataFrame(columns=all_profiles.columns)

    print(f"    Base season players: {len(base)}")

    minutes_carry_model = fit_minutes_carry_model(all_profiles, max_base_season=base_season)
    if minutes_carry_model.get("fitted"):
        print(
            "    Minutes carry model: "
            f"n={minutes_carry_model.get('n_samples')} "
            f"r={minutes_carry_model.get('fit_corr')} "
            f"w_mpg={minutes_carry_model.get('w_mpg'):.3f}"
        )
    else:
        print("    Minutes carry model: using fallback coefficients")

    # 2. Map players to new teams
    roster_override_map = _load_manual_roster_map(roster_path)
    preseason_team_map: Dict[str, str] = {}
    preseason_team_id_map: Dict[str, str] = {}
    mapping_source = "none"
    if team_mapping_mode == "preseason_snapshot":
        preseason_team_map, preseason_team_id_map = load_preseason_roster_maps(
            target_season=target_season,
            preseason_rosters_path=preseason_rosters_path,
        )
        # Build a name->team fallback map from the preseason snapshot when available.
        name_map = {}
        try:
            roster_src = _resolve_preseason_roster_path(target_season, preseason_rosters_path)
            if roster_src and roster_src.exists():
                rdf = pd.read_parquet(roster_src)
                if "player_name" in rdf.columns and ("team_abbreviation" in rdf.columns or "team" in rdf.columns):
                    team_col = "team_abbreviation" if "team_abbreviation" in rdf.columns else "team"
                    rdf = rdf.copy()
                    rdf["player_name_clean"] = rdf["player_name"].astype(str).str.strip().str.lower()
                    rdf[team_col] = rdf[team_col].astype(str).str.upper().str.strip()
                    # Only keep unique name -> single team mappings to reduce false positive matches.
                    counts = rdf.groupby("player_name_clean")[team_col].nunique()
                    uniques = set(counts[counts == 1].index.tolist())
                    if uniques:
                        subset = rdf[rdf["player_name_clean"].isin(uniques)]
                        name_map = (
                            subset.drop_duplicates(subset=["player_name_clean"]).set_index("player_name_clean")
                            [team_col].to_dict()
                        )
        except Exception:
            name_map = {}
        if preseason_team_map:
            print(f"    Preseason snapshot rows mapped: {len(preseason_team_map)} players")
            mapping_source = "preseason_rosters"
        else:
            mapping_source = "carry_forward_fallback"
            print(
                "    WARNING: preseason snapshot unavailable; "
                "falling back to carry-forward teams (salary-team fallback disabled)"
            )

    if mode == "backtest":
        target = all_profiles[all_profiles["season"] == target_season].copy()
        target["player_id"] = target["player_id"].astype(str)
        if target.empty:
            print(f"    WARNING: No target season data for {target_season}")
            return pd.DataFrame(columns=all_profiles.columns)
        projected = map_players_to_teams_backtest(
            prior_profiles=base,
            target_profiles=target,
            team_mapping_mode=team_mapping_mode,
            preseason_team_map=preseason_team_map,
            preseason_team_id_map=preseason_team_id_map,
            roster_override_map=roster_override_map,
        )
        print(f"    Returning players mapped: {len(projected)}")
    else:
        projected = map_players_to_teams_forecast(
            prior_profiles=base,
            team_mapping_mode=team_mapping_mode,
            preseason_team_map=preseason_team_map,
            preseason_team_id_map=preseason_team_id_map,
            roster_override_map=roster_override_map,
        )
        print(f"    Players carried forward: {len(projected)}")

    # Apply name-based team fallback for forecast/preseason scenario when player_id mapping
    # left some players unmapped. This helps when IDs differ between sources but names
    # match uniquely in the preseason snapshot.
    if team_mapping_mode == "preseason_snapshot" and name_map:
        try:
            projected["player_name_clean"] = projected["player_name"].astype(str).str.strip().str.lower()
            missing_mask = projected["team_abbreviation"].isna() | projected["team_abbreviation"].astype(str).str.strip().isin({"", "NAN", "NONE", "FA"})
            if missing_mask.any():
                before_missing = int(missing_mask.sum())
                projected.loc[missing_mask, "team_abbreviation"] = (
                    projected.loc[missing_mask, "player_name_clean"].map(name_map).fillna(projected.loc[missing_mask, "team_abbreviation"])  # type: ignore[arg-type]
                )
                after_missing = int(projected["team_abbreviation"].isna().sum())
                mapped = before_missing - after_missing
                if mapped > 0:
                    print(f"    Name-based team mapping fallback matched {mapped} players from preseason snapshot")
        except Exception:
            pass

    if team_mapping_mode == "preseason_snapshot" and mapping_source != "none":
        projected["team_mapping_source"] = mapping_source

    # Drop players with no valid team (suspended, waived, etc.)
    before = len(projected)
    team_text = projected["team_abbreviation"].astype(str).str.upper().str.strip()
    projected = projected[projected["team_abbreviation"].notna() & ~team_text.isin({"", "NAN", "NONE", "FA"})].copy()
    dropped = before - len(projected)
    if dropped > 0:
        print(f"    Dropped {dropped} players with no valid team assignment")

    # 3. Apply impact projection (regression-to-mean + age curve)
    impact_cols = [
        "impact_bke", "impact_obke", "impact_dbke", "impact_orapm", "impact_drapm",
        "impact_total_impact", "impact_bpm", "impact_ws", "impact_vorp",
    ]
    base_means = {}
    for col in impact_cols:
        if col in base.columns:
            base_means[col] = float(pd.to_numeric(base[col], errors="coerce").mean())
    projected = projected.apply(lambda r: apply_impact_projection(r, base_means), axis=1)

    # 4. Update season identifier
    projected["season"] = target_season

    # Update salary if known for target season.
    if target_salary_map:
        projected["salary_target"] = _norm_id(projected["player_id"]).map(target_salary_map)
        if "salary" in projected.columns:
            projected["salary"] = pd.to_numeric(projected["salary_target"], errors="coerce").fillna(
                pd.to_numeric(projected["salary"], errors="coerce")
            )

    # 5. Add rookies BEFORE minute projection so normalization includes them.
    prior_ids = set(projected["player_id"].astype(str))
    if mode == "backtest":
        target = all_profiles[all_profiles["season"] == target_season].copy()
        rookie_df = build_rookie_profiles_backtest(
            target,
            prior_ids,
            target_season,
            draft_map=draft_map,
            rookie_impact_scale=rookie_impact_scale,
        )
    else:
        rookie_df = build_rookie_profiles_from_draft(
            target_season=target_season,
            prior_player_ids=prior_ids,
            profile_columns=list(all_profiles.columns),
            draft_df=draft_df,
            draft_map=draft_map,
            roster_path=roster_path,
            salary_map=target_salary_map or {},
            salary_team_map=target_salary_team_map or {},
            rookie_impact_scale=rookie_impact_scale,
            allow_salary_team_fallback=(team_mapping_mode != "preseason_snapshot"),
        )

        # Manual CSV fallback only if draft pipeline is unavailable.
        if rookie_df.empty:
            rp = rookies_path if rookies_path else ROOKIES_INPUT_PATH
            if rp and rp.exists():
                print("    Draft-driven rookies unavailable; falling back to rookies CSV override")
                rookie_df = build_rookie_profiles_from_csv(
                    rp,
                    target_season,
                    list(all_profiles.columns),
                    rookie_impact_scale=rookie_impact_scale,
                )

    if not rookie_df.empty:
        rookie_df["player_id"] = _norm_id(rookie_df["player_id"])

        if team_mapping_mode == "preseason_snapshot" and preseason_team_map:
            rookie_df["team_abbreviation"] = rookie_df["player_id"].map(preseason_team_map).fillna(
                rookie_df["team_abbreviation"]
            )
            if "team_id" in rookie_df.columns:
                rookie_df["team_id"] = rookie_df["player_id"].map(preseason_team_id_map).fillna(
                    rookie_df["team_id"]
                )

        if roster_override_map:
            rookie_df["team_abbreviation"] = rookie_df["player_id"].map(roster_override_map).fillna(
                rookie_df["team_abbreviation"]
            )

        rookie_team_text = rookie_df["team_abbreviation"].astype(str).str.upper().str.strip()
        rookie_df = rookie_df[~rookie_team_text.isin({"", "NAN", "NONE", "FA"})].copy()

        # Ensure rookie_df has same columns
        for col in projected.columns:
            if col not in rookie_df.columns:
                rookie_df[col] = np.nan if projected[col].dtype in [float, np.float64] else ""
        rookie_df = rookie_df[[c for c in projected.columns if c in rookie_df.columns]]
        projected = pd.concat([projected, rookie_df], ignore_index=True)
        print(f"    Rookies/newcomers added: {len(rookie_df)}")

    print(f"    Total projected roster: {len(projected)}")

    # 6. Project minutes (single pass, after rookies are added).
    # Uses minute-share approach: multiplicative adjustments + team normalization.
    projected = project_minutes(
        projected,
        target_salary_map=target_salary_map,
        draft_map=draft_map,
        minutes_carry_model=minutes_carry_model,
    )

    # 6a. Apply availability discount (expected missed games — applied ONCE).
    projected = apply_availability_discount(projected)

    # Recompute derived minute fields.
    projected["minutes"] = projected["mpg"] * projected["games"].clip(lower=10)
    projected["possessions"] = projected["minutes"] * 2.0

    # 8. Add replacement-level minutes buffer
    projected = apply_replacement_buffer(projected)
    projected = _exclude_replacement_pool_rows(projected, context=f"{target_season} projected output")

    return projected


def main() -> None:
    parser = argparse.ArgumentParser(description="Project player profiles to next season")
    parser.add_argument("--forecast", type=str, default=None,
                        help="Target season for true forecast (e.g., 2025-26)")
    parser.add_argument(
        "--team-mapping-mode",
        type=str,
        default="end_of_season",
        choices=["end_of_season", "preseason_snapshot"],
        help="Team assignment scenario for projected players",
    )
    parser.add_argument(
        "--preseason-rosters-path",
        type=str,
        default=None,
        help="Optional preseason roster parquet (combined or season-specific)",
    )
    parser.add_argument("--roster", type=str, default=None,
                        help="Path to roster CSV manual overrides")
    parser.add_argument("--rookies", type=str, default=None,
                        help="Path to manual rookies CSV fallback")
    parser.add_argument("--rookie-impact-scale", type=float, default=None,
                        help="Override rookie impact scale multiplier")
    parser.add_argument("--no-rookie-scale-tune", action="store_true",
                        help="Disable historical tuning for rookie impact scale")
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Optional output parquet path override",
    )
    parser.add_argument(
        "--validation-output-path",
        type=str,
        default=None,
        help="Optional validation report path override",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Forward Projection Pipeline")
    print("=" * 60)

    output_path = Path(args.output_path) if args.output_path else PROJECTED_PROFILES_PATH
    validation_output_path = Path(args.validation_output_path) if args.validation_output_path else FORECAST_VALIDATION_REPORT
    preseason_rosters_path = Path(args.preseason_rosters_path) if args.preseason_rosters_path else None
    print(f"Team mapping mode: {args.team_mapping_mode}")

    if not PLAYER_PROFILES_PARQUET.exists():
        raise FileNotFoundError(f"Missing: {PLAYER_PROFILES_PARQUET}")

    all_profiles = pd.read_parquet(PLAYER_PROFILES_PARQUET)
    all_profiles["player_id"] = _norm_id(all_profiles["player_id"])
    all_profiles["season"] = all_profiles["season"].astype(str)
    seasons = sorted(all_profiles["season"].unique())
    print(f"Available seasons: {seasons}")

    draft_df = load_draft_data()
    draft_map = _draft_lookup(draft_df)
    print(f"Draft rows loaded: {len(draft_df)}")

    if args.rookie_impact_scale is not None:
        rookie_impact_scale = float(args.rookie_impact_scale)
        rookie_scale_report = {
            "tuned": False,
            "selected_scale": rookie_impact_scale,
            "reason": "manual override",
        }
    elif args.no_rookie_scale_tune:
        rookie_impact_scale = ROOKIE_IMPACT_SCALE_DEFAULT
        rookie_scale_report = {
            "tuned": False,
            "selected_scale": rookie_impact_scale,
            "reason": "tuning disabled",
        }
    else:
        rookie_impact_scale, rookie_scale_report = tune_rookie_impact_scale(all_profiles, draft_map)

    print(f"Rookie impact scale: {rookie_impact_scale:.3f}")

    all_projected = []
    validation_results = {}

    if args.forecast:
        # True forecast mode: project from latest available season
        base_season = seasons[-1]
        target_season = args.forecast
        roster_path = Path(args.roster) if args.roster else None
        rookies_path = Path(args.rookies) if args.rookies else ROOKIES_INPUT_PATH
        salary_map, salary_team_map = load_target_salary_data(target_season)

        projected = project_season(
            base_season=base_season,
            target_season=target_season,
            all_profiles=all_profiles,
            draft_df=draft_df,
            draft_map=draft_map,
            rookie_impact_scale=rookie_impact_scale,
            mode="forecast",
            roster_path=roster_path,
            preseason_rosters_path=preseason_rosters_path,
            team_mapping_mode=args.team_mapping_mode,
            rookies_path=rookies_path,
            target_salary_map=salary_map,
            target_salary_team_map=salary_team_map,
        )
        all_projected.append(projected)
    else:
        # Backtest mode: project each season pair
        for i in range(len(seasons) - 1):
            base = seasons[i]
            target = seasons[i + 1]
            salary_map, salary_team_map = load_target_salary_data(target)

            projected = project_season(
                base_season=base,
                target_season=target,
                all_profiles=all_profiles,
                draft_df=draft_df,
                draft_map=draft_map,
                rookie_impact_scale=rookie_impact_scale,
                mode="backtest",
                roster_path=Path(args.roster) if args.roster else None,
                preseason_rosters_path=preseason_rosters_path,
                team_mapping_mode=args.team_mapping_mode,
                target_salary_map=salary_map,
                target_salary_team_map=salary_team_map,
            )
            all_projected.append(projected)

            # Validate against actuals
            actual_raw = all_profiles[all_profiles["season"] == target].copy()
            actual = collapse_profiles_for_projection(actual_raw)
            actual["player_id"] = _norm_id(actual["player_id"])

            # Team-match baseline depends on scenario.
            if args.team_mapping_mode == "preseason_snapshot":
                preseason_team_map_target, _ = load_preseason_roster_maps(
                    target_season=target,
                    preseason_rosters_path=preseason_rosters_path,
                )
                if preseason_team_map_target:
                    actual["team_abbreviation"] = actual["player_id"].map(preseason_team_map_target).fillna(
                        actual["team_abbreviation"]
                    )

            if not actual.empty:
                merged = projected.merge(
                    actual[[
                        "player_id",
                        "impact_bke",
                        "impact_orapm",
                        "impact_drapm",
                        "impact_total_impact",
                        "mpg",
                        "team_abbreviation",
                    ]].rename(
                        columns={"impact_bke": "actual_bke", "mpg": "actual_mpg",
                                 "team_abbreviation": "actual_team",
                                 "impact_orapm": "actual_orapm",
                                 "impact_drapm": "actual_drapm",
                                 "impact_total_impact": "actual_total_impact"}
                    ),
                    on="player_id", how="inner",
                )
                if len(merged) > 5:
                    base_ids = set(_norm_id(all_profiles[all_profiles["season"] == base]["player_id"]))
                    n_rookies = int((~_norm_id(projected["player_id"]).isin(base_ids)).sum())

                    bke_corr = float(merged["impact_bke"].corr(merged["actual_bke"]))
                    bke_mae = float((merged["impact_bke"] - merged["actual_bke"]).abs().mean())
                    mpg_corr = float(merged["mpg"].corr(merged["actual_mpg"]))
                    mpg_mae = float((merged["mpg"] - merged["actual_mpg"]).abs().mean())
                    orapm_corr = float(merged["impact_orapm"].corr(merged["actual_orapm"]))
                    orapm_mae = float((merged["impact_orapm"] - merged["actual_orapm"]).abs().mean())
                    drapm_corr = float(merged["impact_drapm"].corr(merged["actual_drapm"]))
                    drapm_mae = float((merged["impact_drapm"] - merged["actual_drapm"]).abs().mean())
                    total_corr = float(merged["impact_total_impact"].corr(merged["actual_total_impact"]))
                    total_mae = float((merged["impact_total_impact"] - merged["actual_total_impact"]).abs().mean())
                    team_match = float((merged["team_abbreviation"] == merged["actual_team"]).mean())

                    rookie_rows = merged[~_norm_id(merged["player_id"]).isin(base_ids)].copy()
                    rookie_bke_mae = float((rookie_rows["impact_bke"] - rookie_rows["actual_bke"]).abs().mean()) if len(rookie_rows) else np.nan
                    rookie_mpg_mae = float((rookie_rows["mpg"] - rookie_rows["actual_mpg"]).abs().mean()) if len(rookie_rows) else np.nan

                    validation_results[f"{base}→{target}"] = {
                        "n_returning": len(merged),
                        "n_rookies": n_rookies,
                        "bke_correlation": round(bke_corr, 4),
                        "bke_mae": round(bke_mae, 4),
                        "orapm_correlation": round(orapm_corr, 4),
                        "orapm_mae": round(orapm_mae, 4),
                        "drapm_correlation": round(drapm_corr, 4),
                        "drapm_mae": round(drapm_mae, 4),
                        "impact_total_correlation": round(total_corr, 4),
                        "impact_total_mae": round(total_mae, 4),
                        "mpg_correlation": round(mpg_corr, 4),
                        "mpg_mae": round(mpg_mae, 2),
                        "rookie_bke_mae": round(rookie_bke_mae, 4) if np.isfinite(rookie_bke_mae) else None,
                        "rookie_mpg_mae": round(rookie_mpg_mae, 2) if np.isfinite(rookie_mpg_mae) else None,
                        "team_match_rate": round(team_match, 4),
                    }
                    print(f"\n    Validation {base}→{target}:")
                    print(f"      BKE: r={bke_corr:.4f}, MAE={bke_mae:.4f}")
                    print(f"      ORAPM: r={orapm_corr:.4f}, MAE={orapm_mae:.4f}")
                    print(f"      DRAPM: r={drapm_corr:.4f}, MAE={drapm_mae:.4f}")
                    print(f"      Impact Total: r={total_corr:.4f}, MAE={total_mae:.4f}")
                    print(f"      MPG: r={mpg_corr:.4f}, MAE={mpg_mae:.2f}")
                    print(f"      Team match: {team_match:.1%}")

    if not all_projected:
        print("No projections generated.")
        return

    result = pd.concat(all_projected, ignore_index=True)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    print(f"\nSaved projected profiles: {output_path}")
    print(f"  Shape: {result.shape}")
    print(f"  Seasons: {sorted(result['season'].unique())}")

    # Save validation report
    if validation_results:
        report = {
            "pipeline": "Forward Projection",
            "mode": "backtest",
            "team_mapping_mode": args.team_mapping_mode,
            "rookie_model": {
                "rookie_impact_scale": rookie_impact_scale,
                "tuning": rookie_scale_report,
            },
            "projections": validation_results,
        }
        validation_output_path.parent.mkdir(parents=True, exist_ok=True)
        validation_output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved validation: {validation_output_path}")
    else:
        report = {
            "pipeline": "Forward Projection",
            "mode": "forecast",
            "team_mapping_mode": args.team_mapping_mode,
            "rookie_model": {
                "rookie_impact_scale": rookie_impact_scale,
                "tuning": rookie_scale_report,
            },
            "projections": {},
        }
        validation_output_path.parent.mkdir(parents=True, exist_ok=True)
        validation_output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved forecast metadata: {validation_output_path}")


if __name__ == "__main__":
    main()
