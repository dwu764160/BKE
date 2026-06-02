"""
src/data_compute/compute_defensive_archetypes_v2.py
=============================================================================
Role-based defensive archetype classification (v3.4).

v3.4 Key Changes from v3.3:
    - Added hard primary-position eligibility gates for each defensive archetype
    - Center is restricted to big roles; Guard is restricted to perimeter roles
    - Forward-Center can access forward + big roles with forward-role preference coefficients
    - Tuned distribution by coefficients: Rim Protector slightly rarer, Versatile Defender more common

v3.3 Key Changes from v3.2:
    - Removed Rotational Big archetype from assignment path
    - Big-band players now route to Rim Protector, Dropping Big, or Mobile Big
    - Margin fallback is retained for wing/guard rotational role only

v3.2 Key Changes from v3.1:
    - Replaced hard-threshold routing with confidence-scored role selection
    - Role is selected via argmax across eligible role confidence scores within size band
    - Added margin rule for stability: if top-second < 0.05, assign rotational role
    - Versatile Defender and Mobile Big now selected by confidence competition, not branch order
    - Rotational Defender / Rotational Big now positive identity roles (not pure leftovers)
    - Defensive effectiveness/fit retained as post-classification overlay only

Defensive Archetype Definitions (v3.4 — Confidence-Scored Decision Flow):
=============================================================
Primary Roles:
    1. POA Defender       — High ball pressure + screen navigation guard specialist
    2. Wing Stopper       — Point-of-attack wing with switch support
    3. Off-Ball Chaser    — Event-driven off-ball disruptor
    4. Versatile Defender — Rare high-diversity switch defender
    5. Rim Protector      — Paint deterrence / rim denial anchor
    6. Dropping Big       — Deep-coverage interior big
    7. Mobile Big         — Switch-capable big without elite rim profile
    8. Rotational Defender — Help-side / generalist wing-guard role
    9. Low-Activity Defender — Low-engagement behavior profile
=============================================================================
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = Path("data")
MATCHUP_DIR = DATA_DIR / "matchup"
TRACKING_DIR = DATA_DIR / "tracking"
HISTORICAL_DIR = DATA_DIR / "historical"
OUTPUT_DIR = DATA_DIR / "processed"
OUTPUT_DIR.mkdir(exist_ok=True)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.modeling.model_config import SEASONS
from src.data.schema_contract import load_standardized, save_standardized

# Minimum requirements
MIN_MINUTES = 200
MIN_GP = 10


# =============================================================================
# DATA LOADING
# =============================================================================

def load_matchup_versatility() -> pd.DataFrame:
    path = MATCHUP_DIR / "matchup_versatility.parquet"
    if path.exists():
        return load_standardized(path)
    return pd.DataFrame()


def load_matchup_difficulty() -> pd.DataFrame:
    path = MATCHUP_DIR / "matchup_difficulty.parquet"
    if path.exists():
        return load_standardized(path)
    return pd.DataFrame()


def load_defense_tracking(season: str) -> pd.DataFrame:
    """Load defense dashboard data (Overall, LessThan6Ft, 3Pointers).
    
    NOTE: Each category has DIFFERENT column names:
      - Overall: D_FG_PCT, PCT_PLUSMINUS, D_FGM, D_FGA, FREQ
      - LessThan6Ft: LT_06_PCT, PLUSMINUS, FGM_LT_06, FGA_LT_06, FREQ
      - 3Pointers: FG3_PCT, PLUSMINUS, FG3M, FG3A, FREQ
    """
    season_dir = TRACKING_DIR / season

    # --- Overall ---
    merged = pd.DataFrame()
    path_ov = season_dir / "defense_Overall.parquet"
    if path_ov.exists():
        ov = load_standardized(path_ov)
        ov = ov.rename(columns={
            "d_fg_pct": "D_FG_PCT_Overall",
            "pct_plusminus": "PCT_PLUSMINUS_Overall",
            "d_fgm": "D_FGM_Overall",
            "d_fga": "D_FGA_Overall",
            "freq": "FREQ_Overall",
        })
        keep = ["close_def_person_id", "player_name"] + [c for c in ov.columns if "Overall" in c]
        merged = ov[keep].copy()

    # --- LessThan6Ft ---
    path_lt = season_dir / "defense_LessThan6Ft.parquet"
    if path_lt.exists():
        lt = load_standardized(path_lt)
        lt = lt.rename(columns={
            "lt_06_pct": "D_FG_PCT_LessThan6Ft",
            "plusminus": "PCT_PLUSMINUS_LessThan6Ft",
            "fgm_lt_06": "D_FGM_LessThan6Ft",
            "fga_lt_06": "D_FGA_LessThan6Ft",
            "freq": "FREQ_LessThan6Ft",
        })
        keep = ["close_def_person_id", "player_name"] + [c for c in lt.columns if "LessThan6Ft" in c]
        lt = lt[keep]
        if len(merged) > 0:
            merged = merged.merge(lt, on=["close_def_person_id", "player_name"], how="outer")
        else:
            merged = lt.copy()

    # --- 3Pointers ---
    path_3p = season_dir / "defense_3Pointers.parquet"
    if path_3p.exists():
        tp = load_standardized(path_3p)
        tp = tp.rename(columns={
            "fg3_pct": "D_FG_PCT_3Pointers",
            "plusminus": "PCT_PLUSMINUS_3Pointers",
            "fg3m": "D_FGM_3Pointers",
            "fg3a": "D_FGA_3Pointers",
            "freq": "FREQ_3Pointers",
        })
        keep = ["close_def_person_id", "player_name"] + [c for c in tp.columns if "3Pointers" in c]
        tp = tp[keep]
        if len(merged) > 0:
            merged = merged.merge(tp, on=["close_def_person_id", "player_name"], how="outer")
        else:
            merged = tp.copy()

    if len(merged) == 0:
        return pd.DataFrame()

    merged["season"] = season
    merged = merged.rename(columns={"close_def_person_id": "player_id"})
    return merged


def load_tracking_defense(season: str) -> pd.DataFrame:
    """Load tracking_Defense.parquet -- NOTE: all stats here are PER GAME."""
    path = TRACKING_DIR / season / "tracking_Defense.parquet"
    if path.exists():
        df = load_standardized(path)
        df["season"] = season
        return df
    return pd.DataFrame()


def load_box_score_data() -> pd.DataFrame:
    path = HISTORICAL_DIR / "complete_player_season_stats.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = load_standardized(path)
    cols = ["player_id", "player_name", "season", "gp", "min", "stl", "blk",
            "dreb", "tov", "pf", "reb_pct"]
    available = [c for c in cols if c in df.columns]
    return df[available]


def load_player_profiles() -> pd.DataFrame:
    path = OUTPUT_DIR / "player_profiles_advanced.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = load_standardized(path)
    df = df.rename(columns={
        "player_id": "player_id",
        "season": "season",
        "player_name": "player_name",
    })
    return df


def load_speed_distance() -> pd.DataFrame:
    all_data = []
    for season in SEASONS:
        path = TRACKING_DIR / season / "tracking_SpeedDistance.parquet"
        if path.exists():
            df = load_standardized(path)
            df["season"] = season
            all_data.append(df)
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


def load_tracking_rebounding() -> pd.DataFrame:
    all_data = []
    for season in SEASONS:
        path = TRACKING_DIR / season / "tracking_Rebounding.parquet"
        if path.exists():
            df = load_standardized(path)
            df["season"] = season
            all_data.append(df)
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


def load_hustle_stats() -> pd.DataFrame:
    """Load per-game hustle stats (deflections, contested shots, etc.)."""
    all_data = []
    for season in SEASONS:
        path = TRACKING_DIR / season / "hustle_stats.parquet"
        if path.exists():
            df = load_standardized(path)
            if "season" not in df.columns:
                df["season"] = season
            all_data.append(df)
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


def load_position_estimates() -> pd.DataFrame:
    """Load season-level position estimate profiles (PG/SG/SF/PF/C shares)."""
    season_files = sorted(OUTPUT_DIR.glob("player_position_estimates_*.parquet"))
    frames = []

    for path in season_files:
        df = load_standardized(path)
        if "season" not in df.columns:
            season = path.stem.replace("player_position_estimates_", "")
            df["season"] = season
        frames.append(df)

    if frames:
        merged = pd.concat(frames, ignore_index=True)
        merged = merged.rename(columns={"player_id": "player_id"})
        return merged

    # Backward compatibility with legacy single-file output
    path = OUTPUT_DIR / "player_position_estimates.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = load_standardized(path)
    df = df.rename(columns={"player_id": "player_id"})
    return df


# =============================================================================
# FEATURE COMPUTATION
# =============================================================================

def safe_col(df, col, default=0):
    if col in df.columns:
        return df[col].fillna(default)
    return pd.Series([default] * len(df), index=df.index)


def normalize_player_id(df: pd.DataFrame) -> pd.DataFrame:
    if "player_id" in df.columns:
        df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce")
    return df


def compute_features(
    versatility: pd.DataFrame,
    difficulty: pd.DataFrame,
    defense_tracking: pd.DataFrame,
    tracking_defense: pd.DataFrame,
    box: pd.DataFrame,
    profiles: pd.DataFrame,
    speed_dist: pd.DataFrame,
    reb_tracking: pd.DataFrame,
    hustle: pd.DataFrame,
    position_estimates: pd.DataFrame,
    season: str,
) -> pd.DataFrame:
    vers = versatility[versatility["season"] == season].copy() if len(versatility) > 0 else pd.DataFrame()
    diff = difficulty[difficulty["season"] == season].copy() if len(difficulty) > 0 else pd.DataFrame()
    def_trk = defense_tracking.copy() if len(defense_tracking) > 0 else pd.DataFrame()
    trk_def = tracking_defense.copy() if len(tracking_defense) > 0 else pd.DataFrame()
    bx = box[box["season"] == season].copy() if len(box) > 0 else pd.DataFrame()
    prof = profiles[profiles["season"] == season].copy() if len(profiles) > 0 else pd.DataFrame()
    spd = speed_dist[speed_dist["season"] == season].copy() if len(speed_dist) > 0 else pd.DataFrame()
    reb = reb_tracking[reb_tracking["season"] == season].copy() if len(reb_tracking) > 0 else pd.DataFrame()
    hst = hustle[hustle["season"] == season].copy() if len(hustle) > 0 else pd.DataFrame()
    pos_est = (
        position_estimates[position_estimates["season"] == season].copy()
        if len(position_estimates) > 0 and "season" in position_estimates.columns
        else pd.DataFrame()
    )

    for frame in [vers, diff, def_trk, trk_def, bx, prof, spd, reb, hst, pos_est]:
        normalize_player_id(frame)

    if bx.empty:
        return pd.DataFrame()

    features = bx.copy()

    # Pre-2022 complete_player_season_stats stores per-game values (MIN=32.9 MPG).
    # Post-2022 stores season totals. Detect by median MIN < 50 → per-game format.
    _per_game_cols = ['min', 'pts', 'ast', 'reb', 'oreb', 'dreb', 'stl', 'blk', 'tov',
                      'fga', 'fgm', 'fg3a', 'fg3m', 'fta', 'ftm']
    if features['min'].median() < 50 and 'gp' in features.columns:
        for _col in _per_game_cols:
            if _col in features.columns:
                features[_col] = features[_col] * features['gp']

    # Merge matchup versatility
    if not vers.empty:
        vers_r = vers.rename(columns={"def_player_id": "player_id"})
        normalize_player_id(vers_r)
        features = features.merge(
            vers_r.drop(columns=["def_player_name", "season"], errors="ignore"),
            on="player_id", how="left",
        )

    # Merge matchup difficulty
    if not diff.empty:
        diff_r = diff.rename(columns={"def_player_id": "player_id"})
        normalize_player_id(diff_r)
        features = features.merge(
            diff_r.drop(columns=["def_player_name", "season"], errors="ignore"),
            on="player_id", how="left",
        )

    # Merge defense tracking (leaguedashptdefend — D_FG_PCT_Overall, etc.)
    if not def_trk.empty:
        features = features.merge(
            def_trk.drop(columns=["player_name", "season"], errors="ignore"),
            on="player_id", how="left",
        )

    # Merge tracking defense (per-game: STL, BLK, DREB, DEF_RIM_FGM/FGA/FG_PCT)
    if not trk_def.empty:
        trk_def_r = trk_def.rename(columns={
            "stl": "STL_PG_TRK", "blk": "BLK_PG_TRK", "dreb": "DREB_PG_TRK",
        })
        drop_cols = ["player_name", "season", "gp", "W", "L", "min",
                     "team_id", "team_abbreviation"]
        features = features.merge(
            trk_def_r.drop(columns=drop_cols, errors="ignore"),
            on="player_id", how="left",
        )

    # Merge profiles (defensive workload)
    if not prof.empty:
        features = features.merge(
            prof.drop(columns=["player_name", "season"], errors="ignore"),
            on="player_id", how="left", suffixes=("", "_prof"),
        )

    # Merge speed/distance
    if not spd.empty and "player_id" in spd.columns:
        spd_cols = ["player_id", "dist_miles", "dist_miles_def", "avg_speed",
                    "avg_speed_def"]
        spd_cols = [c for c in spd_cols if c in spd.columns]
        features = features.merge(spd[spd_cols], on="player_id", how="left")

    # Merge rebounding tracking
    if not reb.empty and "player_id" in reb.columns:
        reb_cols = ["player_id", "dreb_chances", "dreb_chance_pct",
                    "dreb_contest", "DREB_CONTEST_PCT"]
        reb_cols = [c for c in reb_cols if c in reb.columns]
        features = features.merge(reb[reb_cols], on="player_id", how="left")

    # Merge hustle stats
    if not hst.empty and "player_id" in hst.columns:
        hustle_cols = ["player_id", "deflections", "contested_shots",
                       "contested_shots_2pt", "contested_shots_3pt",
                       "charges_drawn", "screen_assists",
                       "def_loose_balls_recovered", "loose_balls_recovered",
                       "def_boxouts", "box_outs"]
        hustle_cols = [c for c in hustle_cols if c in hst.columns]
        features = features.merge(hst[hustle_cols], on="player_id", how="left")

    # Merge season-level position estimates (own-position usage and height)
    if not pos_est.empty:
        est_cols = [
            "player_id",
            "primary_position",
            "primary_position_estimate",
            "height_inches",
            "pct_pg", "pct_sg", "pct_sf", "pct_pf", "pct_c",
            "pct_guards_own", "pct_forwards_own", "pct_centers_own",
            "position_estimate_method",
        ]
        est_cols = [c for c in est_cols if c in pos_est.columns]
        if est_cols:
            features = features.merge(
                pos_est[est_cols].drop_duplicates(subset=["player_id"]),
                on="player_id", how="left",
            )

    # ---- Compute derived features ----
    poss_def = safe_col(features, "poss_def", np.nan).fillna(features["min"] * 2.0)
    seconds_def = safe_col(features, "seconds_def", np.nan).fillna(features["min"] * 60.0)
    features["poss_def"] = poss_def
    features["seconds_def"] = seconds_def

    features["stl_per100_def_poss"] = (
        safe_col(features, "stl", 0)
        / poss_def.replace(0, np.nan)
        * 100
    )
    features["dreb_per100_def_poss"] = (
        safe_col(features, "dreb", 0)
        / poss_def.replace(0, np.nan)
        * 100
    )

    # ---- FIX BLK_PCT: Use per-game BLK from tracking_Defense / per-game DEF_RIM_FGA ----
    blk_pg = safe_col(features, "BLK_PG_TRK", np.nan)
    rim_fga_pg = safe_col(features, "def_rim_fga", np.nan)

    # Fallback: compute per-game BLK from box totals
    blk_pg = blk_pg.fillna(
        safe_col(features, "blk", 0) / features["gp"].replace(0, np.nan)
    )
    # Fallback rim FGA: use LessThan6Ft D_FGA if available
    rim_fga_pg = rim_fga_pg.fillna(
        safe_col(features, "D_FGA_LessThan6Ft", np.nan)
    )
    rim_fga_pg = rim_fga_pg.fillna(3.0)  # league average fallback

    features["blk_pg"] = blk_pg
    features["def_rim_fga_pg"] = rim_fga_pg
    features["blk_pct"] = blk_pg / rim_fga_pg.replace(0, np.nan)
    features["blk_pct"] = features["blk_pct"].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0)
    features["blk_pct"] = features["blk_pct"].clip(0, 1.0)

    features["rim_fga_rate"] = rim_fga_pg
    features["DEF_SECONDS_PER_GAME"] = (
        seconds_def / features["gp"].replace(0, np.nan)
    )
    features["def_rim_fg_pct"] = safe_col(features, "def_rim_fg_pct", 0.62)
    features["d_fg_diff"] = safe_col(features, "PCT_PLUSMINUS_Overall", 0)

    features["season"] = season
    return features


# =============================================================================
# CLASSIFICATION (v3.4 — Confidence-Scored Decision Framework)
# =============================================================================

SCORE_COLS = [
    "poa_score", "wing_score", "chaser_score", "versatile_score",
    "rim_score", "drop_big_score", "mobile_big_score",
]

MARGIN_RULE_GAP = 0.03

# v3.4 tuning coefficients (distribution shaping only)
RIM_PROTECTOR_SCORE_COEF = 0.45  # v3.5: Make Rim Protectors extremely rare
VERSATILE_DEFENDER_SCORE_COEF = 1.12
FORWARD_CENTER_FORWARD_ROLE_BOOST = 1.08
FORWARD_CENTER_BIG_ROLE_PENALTY = 0.94

# v3.4 anchor tuning
RIM_PROTECTOR_ANCHOR_MIN = 0.95  # v3.5: Only the very top rim protectors qualify
VERSATILE_SWITCH_ANCHOR_MIN = 0.62
VERSATILE_MAX_POSITION_SHARE = 0.66


def _own_position_group(pos_str):
    """Map bio position to a group for feasibility masking."""
    if pd.isna(pos_str) or not pos_str:
        return "unknown"
    pos = str(pos_str).strip()
    if pos in ("Guard",):
        return "guard"
    elif pos in ("Guard-Forward", "Forward-Guard"):
        return "guard-wing"
    elif pos in ("Forward",):
        return "wing"
    elif pos in ("Forward-Center", "Center-Forward"):
        return "wing-big"
    elif pos in ("Center",):
        return "big"
    return "unknown"


def _classify_size_band(row):
    """Size-band routing using own-position distributions + height priors."""
    height = row.get("height_inches", 78)
    guard_share = row.get("pct_guards_own", row.get("pct_guards", 0.33))
    forward_share = row.get("pct_forwards_own", row.get("pct_forwards", 0.33))
    center_share = row.get("pct_centers_own", row.get("pct_centers", 0.33))
    primary_est = str(row.get("primary_position_estimate", row.get("primary_position", ""))).strip()

    if primary_est in ("Center", "Forward-Center", "Center-Forward") and (center_share >= 0.30 or height >= 80):
        return "Big"
    if center_share >= 0.55:
        return "Big"
    if center_share >= 0.40 and height >= 80:
        return "Big"

    if primary_est in ("Guard", "Guard-Forward", "Forward-Guard") and (guard_share >= 0.35 or height <= 78):
        return "Guard"
    if guard_share >= 0.62:
        return "Guard"
    if guard_share >= 0.55 and center_share <= 0.20 and height <= 79:
        return "Guard"

    if forward_share >= 0.45:
        return "Wing"
    if height >= 82:
        return "Big"
    if height <= 76:
        return "Guard"
    if center_share > guard_share and center_share >= 0.33:
        return "Big"
    if guard_share > center_share and guard_share >= 0.33:
        return "Guard"

    return "Wing"


def _resolve_primary_position_bucket(row) -> str:
    """Resolve canonical primary position bucket for hard eligibility gating."""
    primary_est = str(row.get("primary_position_estimate", "")).strip()
    primary_bio = str(row.get("primary_position", "")).strip()
    source = primary_est if primary_est and primary_est != "Unknown" else primary_bio

    if source in ("Center",):
        return "Center"
    if source in ("Forward-Center", "Center-Forward"):
        return "Forward-Center"
    if source in ("Forward",):
        return "Forward"
    if source in ("Guard-Forward", "Forward-Guard"):
        return "Guard-Forward"
    if source in ("Guard",):
        return "Guard"

    return "Unknown"


def _allowed_roles_for_position(primary_bucket: str, size_band: str) -> set[str]:
    """Hard role eligibility gates by primary position bucket."""
    if primary_bucket == "Center":
        return {
            "Rim Protector",
            "Dropping Big",
            "Mobile Big",
            "Low-Activity Defender",
        }
    if primary_bucket == "Forward-Center":
        return {
            "Rim Protector",
            "Dropping Big",
            "Mobile Big",
            "Wing Stopper",
            "Versatile Defender",
            "Low-Activity Defender",
        }
    if primary_bucket in ("Forward", "Guard-Forward"):
        return {
            "POA Defender",
            "Off-Ball Chaser",
            "Rotational Defender",
            "Wing Stopper",
            "Versatile Defender",
            "Low-Activity Defender",
        }
    if primary_bucket == "Guard":
        return {
            "POA Defender",
            "Off-Ball Chaser",
            "Rotational Defender",
            "Low-Activity Defender",
        }

    # Unknown fallback mirrors current size band behavior while preserving hard role families.
    if size_band == "Big":
        return {"Rim Protector", "Dropping Big", "Mobile Big", "Low-Activity Defender"}
    if size_band == "Guard":
        return {"POA Defender", "Off-Ball Chaser", "Rotational Defender", "Low-Activity Defender"}
    return {
        "POA Defender",
        "Off-Ball Chaser",
        "Rotational Defender",
        "Wing Stopper",
        "Versatile Defender",
        "Low-Activity Defender",
    }


def _apply_position_and_distribution_coefficients(primary_bucket: str, scores: dict[str, float]) -> dict[str, float]:
    """Apply v3.5 distribution and Forward-Center preference coefficients."""
    adjusted = dict(scores)

    if "Rim Protector" in adjusted:
        adjusted["Rim Protector"] *= RIM_PROTECTOR_SCORE_COEF
    if "Versatile Defender" in adjusted:
        adjusted["Versatile Defender"] *= VERSATILE_DEFENDER_SCORE_COEF
    if "Wing Stopper" in adjusted:
        adjusted["Wing Stopper"] *= 1.12  # v3.5: Further boost to Wing Stopper

    if primary_bucket == "Forward-Center":
        for role in ("Wing Stopper", "Versatile Defender"):
            if role in adjusted:
                adjusted[role] *= FORWARD_CENTER_FORWARD_ROLE_BOOST
        for role in ("Rim Protector", "Dropping Big", "Mobile Big"):
            if role in adjusted:
                adjusted[role] *= FORWARD_CENTER_BIG_ROLE_PENALTY

    return adjusted


def _pick_with_margin(scores: dict[str, float], rotational_role: str | None):
    """Pick role by max score; if separation is small, optionally route to rotational role."""
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_role, top_score = ordered[0]
    second_score = ordered[1][1] if len(ordered) > 1 else 0.0
    margin = top_score - second_score
    if rotational_role and margin < MARGIN_RULE_GAP:
        return rotational_role, top_score, second_score, margin
    return top_role, top_score, second_score, margin


def _pick_secondary(archetype, row):
    """Defensive secondary tags eliminated — 9 primary archetypes are sufficient."""
    return None


def compute_defensive_effectiveness(row, archetype):
    """Archetype-specific effectiveness (0-1). Results quality."""
    d_res = row.get("d_results_pctl", 0.5)
    contest_pctl = row.get("contested_shots_pctl", 0.5)
    defl_pctl = row.get("deflections_pctl", 0.5)
    hustle_pctl = row.get("hustle_pctl", 0.5)

    if archetype == "POA Defender":
        return (0.40 * d_res + 0.20 * row.get("stl_pctl", 0.5)
                + 0.20 * defl_pctl + 0.20 * contest_pctl)
    elif archetype == "Wing Stopper":
        return (0.50 * d_res + 0.20 * contest_pctl
                + 0.15 * defl_pctl + 0.15 * hustle_pctl)
    elif archetype == "Off-Ball Chaser":
        return (0.25 * d_res + 0.35 * row.get("stl_pctl", 0.5)
                + 0.25 * defl_pctl + 0.15 * hustle_pctl)
    elif archetype == "Versatile Defender":
        return (0.40 * d_res + 0.20 * row.get("versatility_pctl", 0.5)
                + 0.20 * contest_pctl + 0.20 * defl_pctl)
    elif archetype == "Rim Protector":
        return (0.30 * d_res + 0.35 * row.get("blk_pctl", 0.5)
                + 0.20 * (1 - row.get("rim_fg_pctl", 0.5))
                + 0.15 * contest_pctl)
    elif archetype == "Dropping Big":
        return (0.35 * d_res + 0.25 * row.get("blk_pctl", 0.5)
                + 0.20 * row.get("reb_pctl", 0.5) + 0.20 * contest_pctl)
    elif archetype == "Mobile Big":
        return (0.35 * d_res + 0.25 * row.get("versatility_pctl", 0.5)
                + 0.20 * contest_pctl + 0.20 * hustle_pctl)
    return 0.50 * d_res + 0.50 * hustle_pctl


def compute_defensive_fit(row, archetype):
    """Evaluate scheme fit: Elite / Good / Average / Poor."""
    eff = compute_defensive_effectiveness(row, archetype)
    if eff >= 0.75:
        return "Elite"
    elif eff >= 0.55:
        return "Good"
    elif eff >= 0.35:
        return "Average"
    return "Poor"


def classify_defenders(features: pd.DataFrame) -> pd.DataFrame:
    """Role-only defensive classification (v3.4)."""
    df = features.copy()
    qualified = df[(df["min"] >= MIN_MINUTES) & (df["gp"] >= MIN_GP)].copy()
    unqualified = df[(df["min"] < MIN_MINUTES) | (df["gp"] < MIN_GP)].copy()

    if len(qualified) == 0:
        return pd.DataFrame()

    # -- Column defaults --
    qualified["switch_score"] = safe_col(qualified, "switch_score", 0.5)
    qualified["avg_opponent_ppg"] = safe_col(qualified, "avg_opponent_ppg", 11.5)
    qualified["elite_matchup_pct"] = safe_col(qualified, "elite_matchup_pct", 0.15)
    qualified["stl_per100_def_poss"] = safe_col(qualified, "stl_per100_def_poss", 0.8)
    qualified["blk_pct"] = safe_col(qualified, "blk_pct", 0.03)
    qualified["def_rim_fg_pct"] = safe_col(qualified, "def_rim_fg_pct", 0.62)
    qualified["rim_fga_rate"] = safe_col(qualified, "rim_fga_rate", 3.0)
    qualified["d_fg_diff"] = safe_col(qualified, "d_fg_diff", 0)
    qualified["poss_def"] = safe_col(qualified, "poss_def", qualified["min"] * 2.0)
    qualified["seconds_def"] = safe_col(qualified, "seconds_def", qualified["min"] * 60.0)
    qualified["dist_miles_def"] = safe_col(qualified, "dist_miles_def", 0)
    qualified["avg_speed_def"] = safe_col(qualified, "avg_speed_def", 0)
    qualified["dreb_chances"] = safe_col(qualified, "dreb_chances", 0)
    qualified["dreb_chance_pct"] = safe_col(qualified, "dreb_chance_pct", 0)
    qualified["reb_pct"] = safe_col(qualified, "reb_pct", 0)
    qualified["dreb_chance_pct"] = safe_col(qualified, "dreb_chance_pct", 0)
    qualified["pct_guards"] = safe_col(qualified, "pct_guards", 0.33)
    qualified["pct_forwards"] = safe_col(qualified, "pct_forwards", 0.33)
    qualified["pct_centers"] = safe_col(qualified, "pct_centers", 0.33)
    qualified["pct_guards_own"] = safe_col(qualified, "pct_guards_own", np.nan).fillna(qualified["pct_guards"])
    qualified["pct_forwards_own"] = safe_col(qualified, "pct_forwards_own", np.nan).fillna(qualified["pct_forwards"])
    qualified["pct_centers_own"] = safe_col(qualified, "pct_centers_own", np.nan).fillna(qualified["pct_centers"])
    if "primary_position" not in qualified.columns:
        qualified["primary_position"] = "Unknown"
    else:
        qualified["primary_position"] = qualified["primary_position"].fillna("Unknown")
    if "primary_position_estimate" not in qualified.columns:
        qualified["primary_position_estimate"] = qualified["primary_position"]
    else:
        qualified["primary_position_estimate"] = qualified["primary_position_estimate"].fillna(qualified["primary_position"])
    qualified["height_inches"] = safe_col(qualified, "height_inches", 78)

    # Hustle stat defaults
    for hcol in ["deflections", "contested_shots", "contested_shots_3pt",
                 "charges_drawn", "def_loose_balls_recovered", "def_boxouts"]:
        qualified[hcol] = safe_col(qualified, hcol, 0)

    # -- Percentiles --
    qualified["versatility_pctl"] = qualified["switch_score"].rank(pct=True)
    qualified["difficulty_pctl"] = qualified["avg_opponent_ppg"].rank(pct=True)
    qualified["elite_matchup_pctl"] = qualified["elite_matchup_pct"].rank(pct=True)
    qualified["stl_pctl"] = qualified["stl_per100_def_poss"].rank(pct=True)
    qualified["blk_pctl"] = qualified["blk_pct"].rank(pct=True)
    qualified["rim_fg_pctl"] = qualified["def_rim_fg_pct"].rank(pct=True)
    qualified["rim_fga_pctl"] = qualified["rim_fga_rate"].rank(pct=True)
    qualified["d_results_pctl"] = 1 - qualified["d_fg_diff"].rank(pct=True)
    qualified["reb_pctl"] = qualified["reb_pct"].rank(pct=True)
    qualified["guard_pctl"] = qualified["pct_guards"].rank(pct=True)
    qualified["forward_pctl"] = qualified["pct_forwards"].rank(pct=True)
    qualified["center_pctl"] = qualified["pct_centers"].rank(pct=True)
    qualified["speed_pctl"] = qualified["avg_speed_def"].rank(pct=True)
    qualified["height_pctl"] = qualified["height_inches"].rank(pct=True)
    qualified["rim_presence_pctl"] = qualified["rim_fga_rate"].rank(pct=True)
    qualified["dreb_chance_pctl"] = qualified["dreb_chance_pct"].rank(pct=True)

    # Hustle percentiles
    qualified["deflections_pctl"] = qualified["deflections"].rank(pct=True)
    qualified["contested_shots_pctl"] = qualified["contested_shots"].rank(pct=True)
    qualified["contested_3pt_pctl"] = qualified["contested_shots_3pt"].rank(pct=True)
    qualified["charges_pctl"] = qualified["charges_drawn"].rank(pct=True)
    qualified["loose_balls_pctl"] = qualified["def_loose_balls_recovered"].rank(pct=True)
    qualified["boxouts_pctl"] = qualified["def_boxouts"].rank(pct=True)

    # -- Engagement & Hustle composite --
    poss_def_pg = qualified["poss_def"] / qualified["gp"].replace(0, np.nan)
    seconds_def_pg = qualified["seconds_def"] / qualified["gp"].replace(0, np.nan)
    dist_def_pg = qualified["dist_miles_def"] / qualified["gp"].replace(0, np.nan)

    def zscore(s):
        m, sd = s.mean(skipna=True), s.std(skipna=True)
        if sd == 0 or np.isnan(sd):
            return s * 0
        return (s - m) / sd

    engagement_score = (
        zscore(poss_def_pg.fillna(0))
        + zscore(seconds_def_pg.fillna(0))
        + zscore(dist_def_pg.fillna(0))
        + zscore(qualified["avg_speed_def"].fillna(0))
    )
    hustle_score = (
        zscore(qualified["dreb_chances"].fillna(0))
        + zscore(qualified["dreb_chance_pct"].fillna(0))
        + zscore(qualified["deflections"].fillna(0)) * 0.5
        + zscore(qualified["contested_shots"].fillna(0)) * 0.5
    )

    qualified["engagement_score"] = engagement_score
    qualified["hustle_score"] = hustle_score
    qualified["engagement_pctl"] = qualified["engagement_score"].rank(pct=True)
    qualified["hustle_pctl"] = qualified["hustle_score"].rank(pct=True)

    # -- On-ball exclusivity signal for POA --
    qualified["onball_exclusivity"] = (
        qualified["pct_guards"] * 2.0
        - qualified["pct_forwards"] * 0.5
        - qualified["pct_centers"] * 1.5
    ).clip(0, 2)
    qualified["onball_pctl"] = qualified["onball_exclusivity"].rank(pct=True)

    # ==== Role feature axes (behavior only; no impact in assignment) ====
    max_pos_share = qualified[["pct_guards", "pct_forwards", "pct_centers"]].max(axis=1)
    qualified["pos_balance_pctl"] = (1 - max_pos_share).rank(pct=True)

    qualified["ball_pressure_index"] = (
        qualified["elite_matchup_pctl"] * 0.35
        + qualified["onball_pctl"] * 0.25
        + qualified["guard_pctl"] * 0.20
        + qualified["deflections_pctl"] * 0.10
        + qualified["contested_3pt_pctl"] * 0.10
    )
    qualified["screen_navigation_index"] = (
        qualified["contested_3pt_pctl"] * 0.45
        + qualified["deflections_pctl"] * 0.25
        + qualified["speed_pctl"] * 0.15
        + qualified["onball_pctl"] * 0.15
    )
    qualified["offball_navigation_index"] = (
        qualified["stl_pctl"] * 0.25
        + qualified["deflections_pctl"] * 0.25
        + qualified["speed_pctl"] * 0.20
        + qualified["loose_balls_pctl"] * 0.15
        + qualified["contested_3pt_pctl"] * 0.15
    )
    qualified["drop_coverage_index"] = (
        qualified["center_pctl"] * 0.30
        + qualified["rim_fga_pctl"] * 0.20
        + (1 - qualified["versatility_pctl"]) * 0.20
        + qualified["boxouts_pctl"] * 0.15
        + (1 - qualified["guard_pctl"]) * 0.15
    )
    qualified["switch_index"] = (
        qualified["versatility_pctl"] * 0.45
        + qualified["pos_balance_pctl"] * 0.25
        + qualified["forward_pctl"] * 0.15
        + qualified["center_pctl"] * 0.15
    )
    qualified["rim_protection_index"] = (
        qualified["blk_pctl"] * 0.35
        + (1 - qualified["rim_fg_pctl"]) * 0.20
        + qualified["rim_fga_pctl"] * 0.15
        + qualified["center_pctl"] * 0.15
        + qualified["height_pctl"] * 0.10
        + qualified["contested_shots_pctl"] * 0.05
    )
    qualified["help_activity_index"] = (
        qualified["boxouts_pctl"] * 0.25
        + qualified["contested_shots_pctl"] * 0.25
        + qualified["charges_pctl"] * 0.20
        + qualified["dreb_chance_pctl"] * 0.15
        + (1 - qualified["onball_pctl"]) * 0.15
    )

    for base_col in [
        "ball_pressure_index", "screen_navigation_index", "offball_navigation_index",
        "drop_coverage_index", "switch_index", "rim_protection_index", "help_activity_index",
    ]:
        qualified[f"{base_col}_pctl"] = qualified[base_col].rank(pct=True)

    qualified["matchup_diversity_pctl"] = qualified["pos_balance_pctl"]
    qualified["liability_index"] = (
        (1 - qualified["ball_pressure_index_pctl"]) * 0.25
        + (1 - qualified["help_activity_index_pctl"]) * 0.20
        + (1 - qualified["rim_protection_index_pctl"]) * 0.20
        + (1 - qualified["engagement_pctl"]) * 0.20
        + (1 - qualified["hustle_pctl"]) * 0.15
    )
    qualified["liability_index_pctl"] = qualified["liability_index"].rank(pct=True)

    # Schema compatibility role scores (still behavior-only)
    qualified["poa_score"] = (
        qualified["ball_pressure_index_pctl"] * 0.70
        + qualified["screen_navigation_index_pctl"] * 0.30
    )
    qualified["wing_score"] = (
        qualified["ball_pressure_index_pctl"] * 0.55
        + qualified["switch_index_pctl"] * 0.45
    )
    qualified["chaser_score"] = qualified["offball_navigation_index_pctl"]
    qualified["versatile_score"] = (
        qualified["switch_index_pctl"] * 0.55
        + qualified["matchup_diversity_pctl"] * 0.45
    )
    qualified["rim_score"] = qualified["rim_protection_index_pctl"]
    qualified["drop_big_score"] = qualified["drop_coverage_index_pctl"]
    qualified["mobile_big_score"] = (
        qualified["switch_index_pctl"] * 0.60
        + qualified["rim_protection_index_pctl"] * 0.40
    )

    qualified["size_band"] = qualified.apply(_classify_size_band, axis=1)

    # ==== Confidence-scored role assignment (role only) ====
    qualified["defensive_archetype"] = "Rotational Defender"

    qualified["poa_candidate_score"] = (
        qualified["ball_pressure_index_pctl"] * 0.40
        + qualified["screen_navigation_index_pctl"] * 0.30
        + qualified["onball_pctl"] * 0.20
        + qualified["difficulty_pctl"] * 0.10
    )
    qualified["offball_candidate_score"] = (
        qualified["offball_navigation_index_pctl"] * 0.50
        + qualified["deflections_pctl"] * 0.25
        + qualified["contested_3pt_pctl"] * 0.15
        + qualified["help_activity_index_pctl"] * 0.10
    )
    qualified["versatile_candidate_score"] = (
        qualified["switch_index_pctl"] * 0.35
        + qualified["matchup_diversity_pctl"] * 0.25
        + np.minimum(
            qualified["ball_pressure_index_pctl"],
            qualified["rim_protection_index_pctl"],
        ) * 0.20
        + qualified["help_activity_index_pctl"] * 0.20
    )
    qualified["rim_candidate_score"] = (
        qualified["rim_protection_index_pctl"] * 0.55
        + qualified["rim_fga_pctl"] * 0.15
        + qualified["height_pctl"] * 0.15
        + qualified["help_activity_index_pctl"] * 0.15
    )
    qualified["drop_candidate_score"] = (
        qualified["drop_coverage_index_pctl"] * 0.50
        + (1 - qualified["switch_index_pctl"]) * 0.20
        + qualified["center_pctl"] * 0.15
        + qualified["boxouts_pctl"] * 0.15
    )
    qualified["mobile_candidate_score"] = (
        qualified["switch_index_pctl"] * 0.45
        + qualified["speed_pctl"] * 0.25
        + qualified["help_activity_index_pctl"] * 0.20
        + qualified["rim_protection_index_pctl"] * 0.10
    )
    role_stack = qualified[[
        "ball_pressure_index_pctl", "switch_index_pctl",
        "offball_navigation_index_pctl", "help_activity_index_pctl"
    ]].copy()
    balanced_identity = 1 - (role_stack.std(axis=1).clip(0, 0.5) / 0.5)
    qualified["rotational_def_candidate_score"] = (
        balanced_identity * 0.60 + qualified["help_activity_index_pctl"] * 0.40
    )
    big_mid = (
        (1 - np.abs(qualified["rim_protection_index_pctl"] - 0.575) / 0.575).clip(0, 1)
        + (1 - np.abs(qualified["switch_index_pctl"] - 0.575) / 0.575).clip(0, 1)
    ) / 2
    qualified["rotational_big_candidate_score"] = (
        big_mid * 0.65 + qualified["help_activity_index_pctl"] * 0.35
    )

    for idx, row in qualified.iterrows():
        size_band = row["size_band"]

        ball_pressure = row["ball_pressure_index_pctl"]
        screen_navigation = row["screen_navigation_index_pctl"]
        offball_navigation = row["offball_navigation_index_pctl"]
        deflections = row["deflections_pctl"]
        help_activity = row["help_activity_index_pctl"]
        switch_index = row["switch_index_pctl"]
        matchup_diversity = row["matchup_diversity_pctl"]
        rim_protection = row["rim_protection_index_pctl"]
        drop_coverage = row["drop_coverage_index_pctl"]
        mobility_metric = row["speed_pctl"]
        engagement = row["engagement_pctl"]
        difficulty = row["difficulty_pctl"]
        max_position_share = max(row["pct_guards"], row["pct_forwards"], row["pct_centers"])
        contest_3pt = row["contested_3pt_pctl"]
        contest_2pt = row["contested_shots_pctl"]

        low_activity_score = 1 - engagement

        primary_bucket = _resolve_primary_position_bucket(row)
        allowed_roles = _allowed_roles_for_position(primary_bucket, size_band)
        rotational_role = "Rotational Defender" if "Rotational Defender" in allowed_roles else None

        poa_score = (
            0.35 * ball_pressure
            + 0.25 * screen_navigation
            + 0.20 * difficulty
            + 0.10 * matchup_diversity
            + 0.10 * engagement
        )
        wing_score = (
            0.35 * difficulty
            + 0.25 * ball_pressure
            + 0.20 * contest_2pt
            + 0.10 * matchup_diversity
            + 0.10 * engagement
        )
        versatile_score = (
            0.30 * switch_index
            + 0.25 * matchup_diversity
            + 0.20 * help_activity
            + 0.15 * min(ball_pressure, rim_protection)
            + 0.10 * engagement
        )
        offball_score = (
            0.35 * offball_navigation
            + 0.25 * deflections
            + 0.20 * contest_3pt
            + 0.10 * help_activity
            + 0.10 * engagement
        )
        rotational_score = 1 - np.std([ball_pressure, offball_navigation, switch_index])
        rim_score = (
            0.45 * rim_protection
            + 0.20 * contest_2pt
            + 0.15 * help_activity
            + 0.10 * drop_coverage
            + 0.10 * engagement
        )
        drop_score = (
            0.40 * drop_coverage
            + 0.25 * rim_protection
            + 0.15 * help_activity
            + 0.10 * difficulty
            + 0.10 * engagement
        )
        mobile_score = (
            0.40 * switch_index
            + 0.20 * mobility_metric
            + 0.15 * matchup_diversity
            + 0.15 * help_activity
            + 0.10 * rim_protection
        )

        all_scores = {
            "POA Defender": poa_score,
            "Wing Stopper": wing_score,
            "Versatile Defender": versatile_score,
            "Off-Ball Chaser": offball_score,
            "Rotational Defender": rotational_score,
            "Rim Protector": rim_score,
            "Dropping Big": drop_score,
            "Mobile Big": mobile_score,
            "Low-Activity Defender": low_activity_score,
        }
        all_scores = _apply_position_and_distribution_coefficients(primary_bucket, all_scores)

        eligible_scores = {}
        if "POA Defender" in allowed_roles and ball_pressure >= 0.60:
            eligible_scores["POA Defender"] = all_scores["POA Defender"]        
            # v3.5: Lower Wing Stopper difficulty threshold to 0.43 to increase prevalence, mainly pulling from Rotational Defender
        if "Wing Stopper" in allowed_roles and difficulty >= 0.43:
            eligible_scores["Wing Stopper"] = all_scores["Wing Stopper"]
        if (
            "Versatile Defender" in allowed_roles
            and switch_index >= VERSATILE_SWITCH_ANCHOR_MIN
            and max_position_share <= VERSATILE_MAX_POSITION_SHARE
        ):
            eligible_scores["Versatile Defender"] = all_scores["Versatile Defender"]
        if "Off-Ball Chaser" in allowed_roles:
            eligible_scores["Off-Ball Chaser"] = all_scores["Off-Ball Chaser"]
        if (
            "Rotational Defender" in allowed_roles
            and 0.30 <= ball_pressure <= 0.75
            and 0.30 <= offball_navigation <= 0.75
        ):
            eligible_scores["Rotational Defender"] = all_scores["Rotational Defender"]
        if "Rim Protector" in allowed_roles and rim_protection >= RIM_PROTECTOR_ANCHOR_MIN:
            eligible_scores["Rim Protector"] = all_scores["Rim Protector"]
        if "Dropping Big" in allowed_roles and drop_coverage >= 0.65 and switch_index <= 0.60:
            eligible_scores["Dropping Big"] = all_scores["Dropping Big"]
        if "Mobile Big" in allowed_roles and switch_index >= 0.65 and drop_coverage <= 0.75:
            eligible_scores["Mobile Big"] = all_scores["Mobile Big"]
        if "Low-Activity Defender" in allowed_roles and engagement <= 0.30:
            eligible_scores["Low-Activity Defender"] = all_scores["Low-Activity Defender"]

        if not eligible_scores:
            fallback_roles = [
                role for role in allowed_roles
                if role != "Low-Activity Defender" and role in all_scores
            ]
            if not fallback_roles and "Low-Activity Defender" in allowed_roles:
                fallback_roles = ["Low-Activity Defender"]
            for role in fallback_roles:
                eligible_scores[role] = all_scores[role]

        archetype, top_score, second_score, margin = _pick_with_margin(
            eligible_scores, rotational_role
        )

        # Low-activity override
        if engagement < 0.25 and top_score < 0.65:
            archetype = "Low-Activity Defender"
            if "Low-Activity Defender" in eligible_scores:
                top_score = max(top_score, eligible_scores["Low-Activity Defender"])
                margin = top_score - second_score

        qualified.at[idx, "defensive_archetype"] = archetype
        qualified.at[idx, "top_role_score"] = top_score
        qualified.at[idx, "second_role_score"] = second_score
        qualified.at[idx, "role_margin"] = margin

    # ==== Build output rows ====
    results = []

    for _, row in qualified.iterrows():
        archetype = row["defensive_archetype"]
        eng_pctl = row["engagement_pctl"]
        d_res = row["d_results_pctl"]
        secondary = _pick_secondary(archetype, row)

        top_role_score = row.get("top_role_score", 0.5)
        role_margin = row.get("role_margin", 0.0)
        _top_c = max(0.0, min(1.0, top_role_score))
        _margin_c = max(0.0, min(1.0, role_margin))
        confidence = (
            0.50 * _top_c
            + 0.50 * (0.5 + 0.5 * (_margin_c ** 0.4))
            + 0.03 * max(0.0, _top_c - 0.75)
        )
        if archetype == "Low-Activity Defender":
            secondary = "Liability"

        # Impact overlay is retained for downstream reporting, but not used for assignment
        effectiveness = compute_defensive_effectiveness(row, archetype)
        fit = compute_defensive_fit(row, archetype)

        diff_pctl = row["difficulty_pctl"]
        difficulty_level = (
            "High" if diff_pctl >= 0.75
            else ("Low" if diff_pctl <= 0.25 else "Medium")
        )

        results.append({
            "player_id": row["player_id"],
            "player_name": row["player_name"],
            "season": row["season"],
            "gp": row["gp"],
            "min": row["min"],
            "defensive_archetype": archetype,
            "defensive_secondary": secondary,
            "defensive_confidence": min(1.0, confidence),
            "defensive_effectiveness": round(effectiveness, 3),
            "defensive_fit": fit,
            "assignment_difficulty": difficulty_level,
            "size_band": row["size_band"],
            "switch_score": row["switch_score"],
            "versatility_pctl": row["versatility_pctl"],
            "avg_opponent_ppg": row["avg_opponent_ppg"],
            "elite_matchup_pct": row["elite_matchup_pct"],
            "difficulty_pctl": diff_pctl,
            "stl_per100_def_poss": row["stl_per100_def_poss"],
            "stl_pctl": row["stl_pctl"],
            "blk_pct": row["blk_pct"],
            "blk_pctl": row["blk_pctl"],
            "def_rim_fg_pct": row["def_rim_fg_pct"],
            "rim_fga_rate": row["rim_fga_rate"],
            "engagement_score": row["engagement_score"],
            "engagement_pctl": eng_pctl,
            "hustle_score": row["hustle_score"],
            "hustle_pctl": row["hustle_pctl"],
            "d_fg_diff": row["d_fg_diff"],
            "d_results_pctl": d_res,
            "deflections": row.get("deflections", 0),
            "deflections_pctl": row.get("deflections_pctl", 0),
            "contested_shots": row.get("contested_shots", 0),
            "contested_shots_pctl": row.get("contested_shots_pctl", 0),
            "ball_pressure_index_pctl": row["ball_pressure_index_pctl"],
            "screen_navigation_index_pctl": row["screen_navigation_index_pctl"],
            "offball_navigation_index_pctl": row["offball_navigation_index_pctl"],
            "drop_coverage_index_pctl": row["drop_coverage_index_pctl"],
            "switch_index_pctl": row["switch_index_pctl"],
            "rim_protection_index_pctl": row["rim_protection_index_pctl"],
            "help_activity_index_pctl": row["help_activity_index_pctl"],
            "liability_index_pctl": row["liability_index_pctl"],
            "matchup_diversity_pctl": row["matchup_diversity_pctl"],
            "top_role_score": top_role_score,
            "second_role_score": row.get("second_role_score", 0.0),
            "role_margin": role_margin,
            "poa_score": row["poa_score"],
            "wing_score": row["wing_score"],
            "chaser_score": row["chaser_score"],
            "versatile_score": row["versatile_score"],
            "rim_score": row["rim_score"],
            "drop_big_score": row["drop_big_score"],
            "mobile_big_score": row["mobile_big_score"],
        })

    _zero = {c: 0 for c in SCORE_COLS}
    for _, row in unqualified.iterrows():
        results.append({
            "player_id": row["player_id"],
            "player_name": row["player_name"],
            "season": row["season"],
            "gp": row["gp"],
            "min": row["min"],
            "defensive_archetype": "Insufficient Minutes",
            "defensive_secondary": None,
            "defensive_confidence": 0.0,
            "defensive_effectiveness": 0.0,
            "defensive_fit": "N/A",
            "assignment_difficulty": "Unknown",
            "size_band": "Unknown",
            "switch_score": 0, "versatility_pctl": 0,
            "avg_opponent_ppg": 0, "elite_matchup_pct": 0,
            "difficulty_pctl": 0,
            "stl_per100_def_poss": 0, "stl_pctl": 0,
            "blk_pct": 0, "blk_pctl": 0,
            "def_rim_fg_pct": 0, "rim_fga_rate": 0,
            "engagement_score": 0, "engagement_pctl": 0,
            "hustle_score": 0, "hustle_pctl": 0,
            "d_fg_diff": 0, "d_results_pctl": 0,
            "deflections": 0, "deflections_pctl": 0,
            "contested_shots": 0, "contested_shots_pctl": 0,
            "ball_pressure_index_pctl": 0,
            "screen_navigation_index_pctl": 0,
            "offball_navigation_index_pctl": 0,
            "drop_coverage_index_pctl": 0,
            "switch_index_pctl": 0,
            "rim_protection_index_pctl": 0,
            "help_activity_index_pctl": 0,
            "liability_index_pctl": 0,
            "matchup_diversity_pctl": 0,
            "top_role_score": 0,
            "second_role_score": 0,
            "role_margin": 0,
            **_zero,
        })

    return pd.DataFrame(results)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("DEFENSIVE ARCHETYPE CLASSIFICATION v3.4")
    print("  Confidence-Scored Decision Framework + Margin Rule")
    print("=" * 70)

    print("\nLoading data...")
    versatility = load_matchup_versatility()
    difficulty = load_matchup_difficulty()
    box = load_box_score_data()
    profiles = load_player_profiles()
    speed_dist = load_speed_distance()
    reb_tracking = load_tracking_rebounding()
    hustle = load_hustle_stats()
    position_estimates = load_position_estimates()

    print(f"  Versatility: {len(versatility)} records")
    print(f"  Difficulty:  {len(difficulty)} records")
    print(f"  Box scores:  {len(box)} records")
    print(f"  Profiles:    {len(profiles)} records")
    print(f"  Hustle:      {len(hustle)} records")
    print(f"  Position est:{len(position_estimates)} records")

    all_results = []

    for season in SEASONS:
        print(f"\nProcessing {season}...")
        def_tracking = load_defense_tracking(season)
        trk_def = load_tracking_defense(season)
        features = compute_features(
            versatility, difficulty, def_tracking, trk_def,
            box, profiles, speed_dist, reb_tracking, hustle, position_estimates, season,
        )
        if features.empty:
            continue

        results = classify_defenders(features)
        if results.empty or "defensive_archetype" not in results.columns:
            print("  No qualified defenders classified.")
            continue
        all_results.append(results)

        arch_dist = results[
            results["defensive_archetype"] != "Insufficient Minutes"
        ]["defensive_archetype"].value_counts()
        print("  Distribution:")
        for arch, count in arch_dist.items():
            print(f"    {arch}: {count}")

    if not all_results:
        print("\nNo results")
        return

    final_df = pd.concat(all_results, ignore_index=True)
    save_standardized(final_df, OUTPUT_DIR / "defensive_archetypes_v2.parquet")
    final_df.to_csv(OUTPUT_DIR / "defensive_archetypes_v2.csv", index=False)

    print(f"\nSaved {len(final_df)} records to defensive_archetypes_v2.parquet")

    # ---- Validation ----
    print("\n=== VALIDATION (Known Defenders 2024-25) ===")
    known = [
        ("Jrue Holiday", "POA Defender"),
        ("Draymond Green", "Versatile Defender or Mobile Big"),
        ("Rudy Gobert", "Rim Protector"),
        ("Anthony Davis", "Rim Protector"),
        ("Herbert Jones", "Wing Stopper or Versatile"),
        ("Dyson Daniels", "POA Defender or Off-Ball Chaser"),
        ("Bam Adebayo", "Mobile Big or Versatile"),
        ("Victor Wembanyama", "Rim Protector"),
        ("Trae Young", "Low-Activity"),
        ("Chet Holmgren", "Rim Protector"),
        ("OG Anunoby", "Wing Stopper"),
        ("Derrick White", "POA Defender"),
        ("Cam Thomas", "Low-Activity or Rotational Defender"),
        ("Luka Don", "Low-Activity or Rotational Defender"),
    ]

    for name, expected in known:
        player = final_df[
            (final_df["player_name"].str.contains(name, case=False, na=False))
            & (final_df["season"] == "2024-25")
        ]
        if len(player) == 0:
            player = final_df[
                final_df["player_name"].str.contains(name, case=False, na=False)
            ]
        if len(player) > 0:
            r = player.iloc[0]
            sec = f" ({r['defensive_secondary']})" if r["defensive_secondary"] else ""
            scores = (
                f"POA={r.get('poa_score',0):.2f} Wing={r.get('wing_score',0):.2f} "
                f"Chase={r.get('chaser_score',0):.2f} Vers={r.get('versatile_score',0):.2f} "
                f"Rim={r.get('rim_score',0):.2f} Drop={r.get('drop_big_score',0):.2f} "
                f"Mob={r.get('mobile_big_score',0):.2f}"
            )
            print(
                f"  {r['player_name'][:22]:22} "
                f"| {r['defensive_archetype']:22}{sec:15}"
                f" | BLK%={r['blk_pct']:.3f}"
                f" Eff={r['defensive_effectiveness']:.2f}"
                f" Fit={r['defensive_fit']}"
                f" | Conf={r['defensive_confidence']:.2f}"
            )
            print(f"    {scores}")
            print(f"    Expected: {expected}")

    # Distribution summary
    q = final_df[final_df["defensive_archetype"] != "Insufficient Minutes"]
    print(f"\n=== OVERALL DISTRIBUTION ===")
    print(q["defensive_archetype"].value_counts().to_string())
    print(f"\n=== BLK_PCT STATS (should be 0-1 range) ===")
    blk = q["blk_pct"]
    print(f"  min={blk.min():.4f} max={blk.max():.4f} mean={blk.mean():.4f}")
    print(f"\n=== CONFIDENCE STATS ===")
    conf = q["defensive_confidence"]
    print(f"  min={conf.min():.2f} max={conf.max():.2f} mean={conf.mean():.2f}")
    print(f"\n=== EFFECTIVENESS STATS ===")
    eff = q["defensive_effectiveness"]
    print(f"  min={eff.min():.3f} max={eff.max():.3f} mean={eff.mean():.3f}")


if __name__ == "__main__":
    main()
