"""
src/data_compute/compute_defensive_archetypes_v2.py
Score-based defensive archetype classification (v3.0).

v3.0 Key Changes from v2.1:
  - Positional feasibility masks (guards can't be bigs, bigs can't be POA)
  - Versatile Defender added as 8th primary archetype
  - Role repulsion terms (incompatible archetypes suppress each other)
  - Fixed BLK_PCT (tracking DEF_RIM_FGA is per-game, box BLK must be per-game too)
  - Integrated hustle stats (deflections, contested shots, loose balls, charges)
  - Defensive fit (scheme suitability) + defensive effectiveness (results quality)
  - POA Defender tightened with on-ball exclusivity signal
  - Mobile Big tightened to require actual big evidence
  - Confidence formula improved via masks + repulsion score separation

Defensive Archetype Definitions (v3.0 — Score-Based + Masks):
=============================================================
Primary Roles (7 scored + 1 filter):
  1. POA Defender       — Guards primary ball handlers exclusively, high elite %
  2. Wing Stopper       — Forward/wing assignments, strong results, specialist
  3. Off-Ball Chaser    — High steals/deflections, roaming, lower difficulty
  4. Versatile Defender — Switches across all positions effectively (no dominant share)
  5. Rim Protector      — Elite blocks + rim FG suppression, anchors interior
  6. Dropping Big       — Interior-only, low versatility, stays in paint
  7. Mobile Big         — Versatile big who switches, must be an actual big
  8. Low-Activity Defender — Engagement filter (bottom ~18% + poor results)
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

SEASONS = ["2022-23", "2023-24", "2024-25"]

# Minimum requirements
MIN_MINUTES = 400
MIN_GP = 15


# =============================================================================
# DATA LOADING
# =============================================================================

def load_matchup_versatility() -> pd.DataFrame:
    path = MATCHUP_DIR / "matchup_versatility.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def load_matchup_difficulty() -> pd.DataFrame:
    path = MATCHUP_DIR / "matchup_difficulty.parquet"
    if path.exists():
        return pd.read_parquet(path)
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
        ov = pd.read_parquet(path_ov)
        ov = ov.rename(columns={
            "D_FG_PCT": "D_FG_PCT_Overall",
            "PCT_PLUSMINUS": "PCT_PLUSMINUS_Overall",
            "D_FGM": "D_FGM_Overall",
            "D_FGA": "D_FGA_Overall",
            "FREQ": "FREQ_Overall",
        })
        keep = ["CLOSE_DEF_PERSON_ID", "PLAYER_NAME"] + [c for c in ov.columns if "Overall" in c]
        merged = ov[keep].copy()

    # --- LessThan6Ft ---
    path_lt = season_dir / "defense_LessThan6Ft.parquet"
    if path_lt.exists():
        lt = pd.read_parquet(path_lt)
        lt = lt.rename(columns={
            "LT_06_PCT": "D_FG_PCT_LessThan6Ft",
            "PLUSMINUS": "PCT_PLUSMINUS_LessThan6Ft",
            "FGM_LT_06": "D_FGM_LessThan6Ft",
            "FGA_LT_06": "D_FGA_LessThan6Ft",
            "FREQ": "FREQ_LessThan6Ft",
        })
        keep = ["CLOSE_DEF_PERSON_ID", "PLAYER_NAME"] + [c for c in lt.columns if "LessThan6Ft" in c]
        lt = lt[keep]
        if len(merged) > 0:
            merged = merged.merge(lt, on=["CLOSE_DEF_PERSON_ID", "PLAYER_NAME"], how="outer")
        else:
            merged = lt.copy()

    # --- 3Pointers ---
    path_3p = season_dir / "defense_3Pointers.parquet"
    if path_3p.exists():
        tp = pd.read_parquet(path_3p)
        tp = tp.rename(columns={
            "FG3_PCT": "D_FG_PCT_3Pointers",
            "PLUSMINUS": "PCT_PLUSMINUS_3Pointers",
            "FG3M": "D_FGM_3Pointers",
            "FG3A": "D_FGA_3Pointers",
            "FREQ": "FREQ_3Pointers",
        })
        keep = ["CLOSE_DEF_PERSON_ID", "PLAYER_NAME"] + [c for c in tp.columns if "3Pointers" in c]
        tp = tp[keep]
        if len(merged) > 0:
            merged = merged.merge(tp, on=["CLOSE_DEF_PERSON_ID", "PLAYER_NAME"], how="outer")
        else:
            merged = tp.copy()

    if len(merged) == 0:
        return pd.DataFrame()

    merged["SEASON"] = season
    merged = merged.rename(columns={"CLOSE_DEF_PERSON_ID": "PLAYER_ID"})
    return merged


def load_tracking_defense(season: str) -> pd.DataFrame:
    """Load tracking_Defense.parquet -- NOTE: all stats here are PER GAME."""
    path = TRACKING_DIR / season / "tracking_Defense.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        df["SEASON"] = season
        return df
    return pd.DataFrame()


def load_box_score_data() -> pd.DataFrame:
    path = HISTORICAL_DIR / "complete_player_season_stats.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    cols = ["PLAYER_ID", "PLAYER_NAME", "SEASON", "GP", "MIN", "STL", "BLK",
            "DREB", "TOV", "PF", "REB_PCT"]
    available = [c for c in cols if c in df.columns]
    return df[available]


def load_player_profiles() -> pd.DataFrame:
    path = OUTPUT_DIR / "player_profiles_advanced.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df = df.rename(columns={
        "player_id": "PLAYER_ID",
        "season": "SEASON",
        "player_name": "PLAYER_NAME",
    })
    return df


def load_speed_distance() -> pd.DataFrame:
    all_data = []
    for season in SEASONS:
        path = TRACKING_DIR / season / "tracking_SpeedDistance.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            df["SEASON"] = season
            all_data.append(df)
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


def load_tracking_rebounding() -> pd.DataFrame:
    all_data = []
    for season in SEASONS:
        path = TRACKING_DIR / season / "tracking_Rebounding.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            df["SEASON"] = season
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
            df = pd.read_parquet(path)
            if "SEASON" not in df.columns:
                df["SEASON"] = season
            all_data.append(df)
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


def load_player_bios() -> pd.DataFrame:
    """Load player bio data for own-position and height.
    
    players.parquet columns: player_id, full_name, primary_position,
    height_inches, weight_lbs, ...
    primary_position values: Guard, Guard-Forward, Forward, Forward-Guard,
    Forward-Center, Center-Forward, Center
    """
    path = HISTORICAL_DIR / "players.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df = df.rename(columns={
        "player_id": "PLAYER_ID",
        "full_name": "PLAYER_NAME",
    })
    return df


# =============================================================================
# FEATURE COMPUTATION
# =============================================================================

def safe_col(df, col, default=0):
    if col in df.columns:
        return df[col].fillna(default)
    return pd.Series([default] * len(df), index=df.index)


def normalize_player_id(df: pd.DataFrame) -> pd.DataFrame:
    if "PLAYER_ID" in df.columns:
        df["PLAYER_ID"] = pd.to_numeric(df["PLAYER_ID"], errors="coerce")
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
    bios: pd.DataFrame,
    season: str,
) -> pd.DataFrame:
    vers = versatility[versatility["SEASON"] == season].copy() if len(versatility) > 0 else pd.DataFrame()
    diff = difficulty[difficulty["SEASON"] == season].copy() if len(difficulty) > 0 else pd.DataFrame()
    def_trk = defense_tracking.copy() if len(defense_tracking) > 0 else pd.DataFrame()
    trk_def = tracking_defense.copy() if len(tracking_defense) > 0 else pd.DataFrame()
    bx = box[box["SEASON"] == season].copy() if len(box) > 0 else pd.DataFrame()
    prof = profiles[profiles["SEASON"] == season].copy() if len(profiles) > 0 else pd.DataFrame()
    spd = speed_dist[speed_dist["SEASON"] == season].copy() if len(speed_dist) > 0 else pd.DataFrame()
    reb = reb_tracking[reb_tracking["SEASON"] == season].copy() if len(reb_tracking) > 0 else pd.DataFrame()
    hst = hustle[hustle["SEASON"] == season].copy() if len(hustle) > 0 else pd.DataFrame()

    for frame in [vers, diff, def_trk, trk_def, bx, prof, spd, reb, hst]:
        normalize_player_id(frame)

    if bx.empty:
        return pd.DataFrame()

    features = bx.copy()

    # Merge matchup versatility
    if not vers.empty:
        vers_r = vers.rename(columns={"DEF_PLAYER_ID": "PLAYER_ID"})
        normalize_player_id(vers_r)
        features = features.merge(
            vers_r.drop(columns=["DEF_PLAYER_NAME", "SEASON"], errors="ignore"),
            on="PLAYER_ID", how="left",
        )

    # Merge matchup difficulty
    if not diff.empty:
        diff_r = diff.rename(columns={"DEF_PLAYER_ID": "PLAYER_ID"})
        normalize_player_id(diff_r)
        features = features.merge(
            diff_r.drop(columns=["DEF_PLAYER_NAME", "SEASON"], errors="ignore"),
            on="PLAYER_ID", how="left",
        )

    # Merge defense tracking (leaguedashptdefend — D_FG_PCT_Overall, etc.)
    if not def_trk.empty:
        features = features.merge(
            def_trk.drop(columns=["PLAYER_NAME", "SEASON"], errors="ignore"),
            on="PLAYER_ID", how="left",
        )

    # Merge tracking defense (per-game: STL, BLK, DREB, DEF_RIM_FGM/FGA/FG_PCT)
    if not trk_def.empty:
        trk_def_r = trk_def.rename(columns={
            "STL": "STL_PG_TRK", "BLK": "BLK_PG_TRK", "DREB": "DREB_PG_TRK",
        })
        drop_cols = ["PLAYER_NAME", "SEASON", "GP", "W", "L", "MIN",
                     "TEAM_ID", "TEAM_ABBREVIATION"]
        features = features.merge(
            trk_def_r.drop(columns=drop_cols, errors="ignore"),
            on="PLAYER_ID", how="left",
        )

    # Merge profiles (defensive workload)
    if not prof.empty:
        features = features.merge(
            prof.drop(columns=["PLAYER_NAME", "SEASON"], errors="ignore"),
            on="PLAYER_ID", how="left", suffixes=("", "_prof"),
        )

    # Merge speed/distance
    if not spd.empty and "PLAYER_ID" in spd.columns:
        spd_cols = ["PLAYER_ID", "DIST_MILES", "DIST_MILES_DEF", "AVG_SPEED",
                    "AVG_SPEED_DEF"]
        spd_cols = [c for c in spd_cols if c in spd.columns]
        features = features.merge(spd[spd_cols], on="PLAYER_ID", how="left")

    # Merge rebounding tracking
    if not reb.empty and "PLAYER_ID" in reb.columns:
        reb_cols = ["PLAYER_ID", "DREB_CHANCES", "DREB_CHANCE_PCT",
                    "DREB_CONTEST", "DREB_CONTEST_PCT"]
        reb_cols = [c for c in reb_cols if c in reb.columns]
        features = features.merge(reb[reb_cols], on="PLAYER_ID", how="left")

    # Merge hustle stats
    if not hst.empty and "PLAYER_ID" in hst.columns:
        hustle_cols = ["PLAYER_ID", "DEFLECTIONS", "CONTESTED_SHOTS",
                       "CONTESTED_SHOTS_2PT", "CONTESTED_SHOTS_3PT",
                       "CHARGES_DRAWN", "SCREEN_ASSISTS",
                       "DEF_LOOSE_BALLS_RECOVERED", "LOOSE_BALLS_RECOVERED",
                       "DEF_BOXOUTS", "BOX_OUTS"]
        hustle_cols = [c for c in hustle_cols if c in hst.columns]
        features = features.merge(hst[hustle_cols], on="PLAYER_ID", how="left")

    # Merge player bios (own position, height)
    if not bios.empty:
        bios_c = bios.copy()
        normalize_player_id(bios_c)
        bio_cols = ["PLAYER_ID", "primary_position", "height_inches"]
        bio_cols = [c for c in bio_cols if c in bios_c.columns]
        if bio_cols:
            features = features.merge(
                bios_c[bio_cols].drop_duplicates(subset=["PLAYER_ID"]),
                on="PLAYER_ID", how="left",
            )

    # ---- Compute derived features ----
    poss_def = safe_col(features, "POSS_DEF", np.nan).fillna(features["MIN"] * 2.0)
    seconds_def = safe_col(features, "SECONDS_DEF", np.nan).fillna(features["MIN"] * 60.0)
    features["POSS_DEF"] = poss_def
    features["SECONDS_DEF"] = seconds_def

    features["STL_PER100_DEF_POSS"] = (
        safe_col(features, "STL", 0)
        / poss_def.replace(0, np.nan)
        * 100
    )
    features["DREB_PER100_DEF_POSS"] = (
        safe_col(features, "DREB", 0)
        / poss_def.replace(0, np.nan)
        * 100
    )

    # ---- FIX BLK_PCT: Use per-game BLK from tracking_Defense / per-game DEF_RIM_FGA ----
    blk_pg = safe_col(features, "BLK_PG_TRK", np.nan)
    rim_fga_pg = safe_col(features, "DEF_RIM_FGA", np.nan)

    # Fallback: compute per-game BLK from box totals
    blk_pg = blk_pg.fillna(
        safe_col(features, "BLK", 0) / features["GP"].replace(0, np.nan)
    )
    # Fallback rim FGA: use LessThan6Ft D_FGA if available
    rim_fga_pg = rim_fga_pg.fillna(
        safe_col(features, "D_FGA_LessThan6Ft", np.nan)
    )
    rim_fga_pg = rim_fga_pg.fillna(3.0)  # league average fallback

    features["BLK_PG"] = blk_pg
    features["DEF_RIM_FGA_PG"] = rim_fga_pg
    features["BLK_PCT"] = blk_pg / rim_fga_pg.replace(0, np.nan)
    features["BLK_PCT"] = features["BLK_PCT"].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0)
    features["BLK_PCT"] = features["BLK_PCT"].clip(0, 1.0)

    features["RIM_FGA_RATE"] = rim_fga_pg
    features["DEF_SECONDS_PER_GAME"] = (
        seconds_def / features["GP"].replace(0, np.nan)
    )
    features["DEF_RIM_FG_PCT"] = safe_col(features, "DEF_RIM_FG_PCT", 0.62)
    features["D_FG_DIFF"] = safe_col(features, "PCT_PLUSMINUS_Overall", 0)

    features["SEASON"] = season
    return features


# =============================================================================
# CLASSIFICATION (v3.0 — Score-Based + Masks + Repulsion)
# =============================================================================

SCORE_ROLES = [
    "POA Defender", "Wing Stopper", "Off-Ball Chaser", "Versatile Defender",
    "Rim Protector", "Dropping Big", "Mobile Big",
]
SCORE_COLS = [
    "poa_score", "wing_score", "chaser_score", "versatile_score",
    "rim_score", "drop_big_score", "mobile_big_score",
]

LOW_ACTIVITY_ENGAGEMENT_GATE = 0.18
LOW_ACTIVITY_RESULTS_GATE = 0.40


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


def _apply_positional_masks(scores, own_pos, pct_guards, pct_centers, height):
    """Soft positional feasibility masks (multipliers, not hard gates)."""
    masked = scores.copy()

    is_short = height < 77   # < 6\'5"
    is_tall = height >= 80    # >= 6\'8"

    # Guards cannot be bigs
    if own_pos in ("guard", "guard-wing"):
        masked["rim_score"] *= 0.10
        masked["drop_big_score"] *= 0.05
        masked["mobile_big_score"] *= 0.15
        if is_short:
            masked["rim_score"] *= 0.05
            masked["drop_big_score"] *= 0.02
            masked["mobile_big_score"] *= 0.05

    # Pure bigs cannot be POA
    if own_pos == "big":
        masked["poa_score"] *= 0.15
        masked["chaser_score"] *= 0.30
        if pct_guards < 0.20:
            masked["poa_score"] *= 0.10

    # Wing-bigs get moderate POA suppression
    if own_pos == "wing-big":
        masked["poa_score"] *= 0.40
        masked["chaser_score"] *= 0.60

    # Low guard share -> suppress POA
    if pct_guards < 0.30:
        masked["poa_score"] *= 0.50

    # Low center share -> suppress big roles
    if pct_centers < 0.15:
        masked["rim_score"] *= 0.30
        masked["drop_big_score"] *= 0.20
        masked["mobile_big_score"] *= 0.25

    # Tall players suppress POA
    if is_tall:
        masked["poa_score"] *= 0.50
        masked["chaser_score"] *= 0.70

    # Wing Stopper: suppress for pure guards unless tall
    if own_pos == "guard" and not is_tall:
        masked["wing_score"] *= 0.50

    return masked


def _apply_role_repulsion(scores):
    """High affinity for one role suppresses incompatible roles."""
    r = scores.copy()

    if scores["rim_score"] > 0.60:
        f = 1 - (scores["rim_score"] - 0.60) * 0.5
        r["poa_score"] *= max(f, 0.3)
        r["chaser_score"] *= max(f, 0.4)

    if scores["poa_score"] > 0.55:
        f = 1 - (scores["poa_score"] - 0.55) * 0.5
        r["rim_score"] *= max(f, 0.3)
        r["drop_big_score"] *= max(f, 0.2)
        r["mobile_big_score"] *= max(f, 0.3)

    if scores["chaser_score"] > 0.55:
        f = 1 - (scores["chaser_score"] - 0.55) * 0.4
        r["wing_score"] *= max(f, 0.5)

    if scores["drop_big_score"] > 0.55:
        f = 1 - (scores["drop_big_score"] - 0.55) * 0.5
        r["versatile_score"] *= max(f, 0.3)
        r["mobile_big_score"] *= max(f, 0.4)

    if scores["versatile_score"] > 0.60:
        f = 1 - (scores["versatile_score"] - 0.60) * 0.3
        r["drop_big_score"] *= max(f, 0.4)
        r["wing_score"] *= max(f, 0.6)

    return r


def _pick_secondary(archetype, row):
    """Pick secondary tag for the given primary."""
    vpctl = row["versatility_pctl"]
    stl_pctl = row["stl_pctl"]
    blk_pctl = row["blk_pctl"]
    d_res = row["d_results_pctl"]
    hustle_pctl = row["hustle_pctl"]
    defl_pctl = row.get("deflections_pctl", 0.5)

    if archetype == "POA Defender":
        if vpctl >= 0.75:
            return "Switchable"
        if stl_pctl >= 0.75 or defl_pctl >= 0.80:
            return "Ball Hawk"
        if hustle_pctl >= 0.75:
            return "Hustler"
        return "Primary"

    if archetype == "Wing Stopper":
        if d_res >= 0.75:
            return "Lockdown"
        if hustle_pctl >= 0.75:
            return "Hustler"
        return "Primary"

    if archetype == "Off-Ball Chaser":
        if stl_pctl >= 0.85 or defl_pctl >= 0.85:
            return "Ball Hawk"
        if hustle_pctl >= 0.75:
            return "Hustler"
        return "Active Hands"

    if archetype == "Versatile Defender":
        if d_res >= 0.75:
            return "Lockdown"
        if stl_pctl >= 0.70:
            return "Ball Hawk"
        if hustle_pctl >= 0.75:
            return "Hustler"
        return "Switchable"

    if archetype == "Rim Protector":
        if vpctl >= 0.75:
            return "Switchable"
        if blk_pctl >= 0.90:
            return "Shot Blocker"
        return "Interior"

    if archetype == "Dropping Big":
        if blk_pctl >= 0.70:
            return "Interior"
        return "Help"

    if archetype == "Mobile Big":
        if d_res >= 0.70:
            return "Lockdown"
        if hustle_pctl >= 0.75:
            return "Hustler"
        return "Switchable"

    return "Help"


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
    """Score-based defensive classification (v3.0)."""
    df = features.copy()
    qualified = df[(df["MIN"] >= MIN_MINUTES) & (df["GP"] >= MIN_GP)].copy()
    unqualified = df[(df["MIN"] < MIN_MINUTES) | (df["GP"] < MIN_GP)].copy()

    if len(qualified) == 0:
        return pd.DataFrame()

    # -- Column defaults --
    qualified["switch_score"] = safe_col(qualified, "switch_score", 0.5)
    qualified["avg_opponent_ppg"] = safe_col(qualified, "avg_opponent_ppg", 11.5)
    qualified["elite_matchup_pct"] = safe_col(qualified, "elite_matchup_pct", 0.15)
    qualified["STL_PER100_DEF_POSS"] = safe_col(qualified, "STL_PER100_DEF_POSS", 0.8)
    qualified["BLK_PCT"] = safe_col(qualified, "BLK_PCT", 0.03)
    qualified["DEF_RIM_FG_PCT"] = safe_col(qualified, "DEF_RIM_FG_PCT", 0.62)
    qualified["RIM_FGA_RATE"] = safe_col(qualified, "RIM_FGA_RATE", 3.0)
    qualified["D_FG_DIFF"] = safe_col(qualified, "D_FG_DIFF", 0)
    qualified["POSS_DEF"] = safe_col(qualified, "POSS_DEF", qualified["MIN"] * 2.0)
    qualified["SECONDS_DEF"] = safe_col(qualified, "SECONDS_DEF", qualified["MIN"] * 60.0)
    qualified["DIST_MILES_DEF"] = safe_col(qualified, "DIST_MILES_DEF", 0)
    qualified["AVG_SPEED_DEF"] = safe_col(qualified, "AVG_SPEED_DEF", 0)
    qualified["DREB_CHANCES"] = safe_col(qualified, "DREB_CHANCES", 0)
    qualified["DREB_CHANCE_PCT"] = safe_col(qualified, "DREB_CHANCE_PCT", 0)
    qualified["REB_PCT"] = safe_col(qualified, "REB_PCT", 0)
    qualified["pct_guards"] = safe_col(qualified, "pct_guards", 0.33)
    qualified["pct_forwards"] = safe_col(qualified, "pct_forwards", 0.33)
    qualified["pct_centers"] = safe_col(qualified, "pct_centers", 0.33)
    if "primary_position" not in qualified.columns:
        qualified["primary_position"] = "Unknown"
    else:
        qualified["primary_position"] = qualified["primary_position"].fillna("Unknown")
    qualified["height_inches"] = safe_col(qualified, "height_inches", 78)

    # Hustle stat defaults
    for hcol in ["DEFLECTIONS", "CONTESTED_SHOTS", "CONTESTED_SHOTS_3PT",
                 "CHARGES_DRAWN", "DEF_LOOSE_BALLS_RECOVERED", "DEF_BOXOUTS"]:
        qualified[hcol] = safe_col(qualified, hcol, 0)

    # -- Percentiles --
    qualified["versatility_pctl"] = qualified["switch_score"].rank(pct=True)
    qualified["difficulty_pctl"] = qualified["avg_opponent_ppg"].rank(pct=True)
    qualified["elite_matchup_pctl"] = qualified["elite_matchup_pct"].rank(pct=True)
    qualified["stl_pctl"] = qualified["STL_PER100_DEF_POSS"].rank(pct=True)
    qualified["blk_pctl"] = qualified["BLK_PCT"].rank(pct=True)
    qualified["rim_fg_pctl"] = qualified["DEF_RIM_FG_PCT"].rank(pct=True)
    qualified["rim_fga_pctl"] = qualified["RIM_FGA_RATE"].rank(pct=True)
    qualified["d_results_pctl"] = 1 - qualified["D_FG_DIFF"].rank(pct=True)
    qualified["reb_pctl"] = qualified["REB_PCT"].rank(pct=True)
    qualified["guard_pctl"] = qualified["pct_guards"].rank(pct=True)
    qualified["forward_pctl"] = qualified["pct_forwards"].rank(pct=True)
    qualified["center_pctl"] = qualified["pct_centers"].rank(pct=True)
    qualified["speed_pctl"] = qualified["AVG_SPEED_DEF"].rank(pct=True)
    qualified["height_pctl"] = qualified["height_inches"].rank(pct=True)

    # Hustle percentiles
    qualified["deflections_pctl"] = qualified["DEFLECTIONS"].rank(pct=True)
    qualified["contested_shots_pctl"] = qualified["CONTESTED_SHOTS"].rank(pct=True)
    qualified["contested_3pt_pctl"] = qualified["CONTESTED_SHOTS_3PT"].rank(pct=True)
    qualified["charges_pctl"] = qualified["CHARGES_DRAWN"].rank(pct=True)
    qualified["loose_balls_pctl"] = qualified["DEF_LOOSE_BALLS_RECOVERED"].rank(pct=True)
    qualified["boxouts_pctl"] = qualified["DEF_BOXOUTS"].rank(pct=True)

    # -- Engagement & Hustle composite --
    poss_def_pg = qualified["POSS_DEF"] / qualified["GP"].replace(0, np.nan)
    seconds_def_pg = qualified["SECONDS_DEF"] / qualified["GP"].replace(0, np.nan)
    dist_def_pg = qualified["DIST_MILES_DEF"] / qualified["GP"].replace(0, np.nan)

    def zscore(s):
        m, sd = s.mean(skipna=True), s.std(skipna=True)
        if sd == 0 or np.isnan(sd):
            return s * 0
        return (s - m) / sd

    engagement_score = (
        zscore(poss_def_pg.fillna(0))
        + zscore(seconds_def_pg.fillna(0))
        + zscore(dist_def_pg.fillna(0))
        + zscore(qualified["AVG_SPEED_DEF"].fillna(0))
    )
    hustle_score = (
        zscore(qualified["DREB_CHANCES"].fillna(0))
        + zscore(qualified["DREB_CHANCE_PCT"].fillna(0))
        + zscore(qualified["DEFLECTIONS"].fillna(0)) * 0.5
        + zscore(qualified["CONTESTED_SHOTS"].fillna(0)) * 0.5
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

    # ==== Role Affinity Scores (0-1, weighted percentile sums) ====

    # POA Defender
    qualified["poa_score"] = (
        qualified["elite_matchup_pctl"] * 0.25
        + qualified["difficulty_pctl"] * 0.20
        + qualified["onball_pctl"] * 0.20
        + qualified["guard_pctl"] * 0.15
        + qualified["deflections_pctl"] * 0.10
        + qualified["d_results_pctl"] * 0.10
    )

    # Wing Stopper
    qualified["wing_score"] = (
        qualified["forward_pctl"] * 0.25
        + qualified["difficulty_pctl"] * 0.20
        + qualified["d_results_pctl"] * 0.25
        + qualified["contested_shots_pctl"] * 0.10
        + (1 - qualified["guard_pctl"]) * 0.10
        + (1 - qualified["center_pctl"]) * 0.10
    )

    # Off-Ball Chaser
    qualified["chaser_score"] = (
        qualified["stl_pctl"] * 0.25
        + qualified["deflections_pctl"] * 0.20
        + qualified["speed_pctl"] * 0.10
        + (1 - qualified["difficulty_pctl"]) * 0.20
        + qualified["loose_balls_pctl"] * 0.10
        + qualified["d_results_pctl"] * 0.15
    )

    # Versatile Defender
    max_pos_share = qualified[["pct_guards", "pct_forwards", "pct_centers"]].max(axis=1)
    qualified["pos_balance_pctl"] = (1 - max_pos_share).rank(pct=True)
    qualified["versatile_score"] = (
        qualified["versatility_pctl"] * 0.30
        + qualified["pos_balance_pctl"] * 0.20
        + qualified["difficulty_pctl"] * 0.15
        + qualified["d_results_pctl"] * 0.20
        + qualified["contested_shots_pctl"] * 0.15
    )

    # Rim Protector
    qualified["rim_score"] = (
        qualified["blk_pctl"] * 0.30
        + (1 - qualified["rim_fg_pctl"]) * 0.25
        + qualified["rim_fga_pctl"] * 0.15
        + qualified["center_pctl"] * 0.15
        + qualified["contested_shots_pctl"] * 0.10
        + qualified["height_pctl"] * 0.05
    )

    # Dropping Big
    qualified["drop_big_score"] = (
        (1 - qualified["versatility_pctl"]) * 0.25
        + qualified["center_pctl"] * 0.25
        + qualified["reb_pctl"] * 0.20
        + qualified["rim_fga_pctl"] * 0.15
        + qualified["boxouts_pctl"] * 0.10
        + (1 - qualified["guard_pctl"]) * 0.05
    )

    # Mobile Big
    qualified["mobile_big_score"] = (
        qualified["versatility_pctl"] * 0.20
        + qualified["center_pctl"] * 0.20
        + qualified["reb_pctl"] * 0.15
        + qualified["d_results_pctl"] * 0.15
        + qualified["height_pctl"] * 0.10
        + qualified["forward_pctl"] * 0.10
        + qualified["contested_shots_pctl"] * 0.10
    )

    # ==== Assign archetype via argmax + masks + repulsion ====
    results = []

    for idx, row in qualified.iterrows():
        eng_pctl = row["engagement_pctl"]
        d_res = row["d_results_pctl"]

        # Extended Low-Activity gate: sliding threshold based on results
        low_act_eng_gate = LOW_ACTIVITY_ENGAGEMENT_GATE
        if d_res < 0.10:
            low_act_eng_gate = 0.95
        elif d_res < 0.15:
            low_act_eng_gate = 0.50
        elif d_res < 0.25:
            low_act_eng_gate = 0.30
        elif d_res <= LOW_ACTIVITY_RESULTS_GATE:
            low_act_eng_gate = LOW_ACTIVITY_ENGAGEMENT_GATE

        is_low_activity = (
            (eng_pctl <= low_act_eng_gate and d_res <= LOW_ACTIVITY_RESULTS_GATE)
            or (d_res <= 0.08)  # bottom ~8% results always Low-Activity
        )

        if is_low_activity:
            archetype = "Low-Activity Defender"
            secondary = "Liability" if d_res <= 0.20 else "Hidden"
            confidence = 0.30 + (1 - eng_pctl) * 0.40
            effectiveness = compute_defensive_effectiveness(row, archetype)
            fit = compute_defensive_fit(row, archetype)
        else:
            raw_scores = {c: row[c] for c in SCORE_COLS}

            # Results damping: bad defenders get all scores suppressed
            results_factor = max(0.35, 0.40 + 0.60 * d_res)
            raw_scores = {k: v * results_factor for k, v in raw_scores.items()}

            own_pos = _own_position_group(row.get("primary_position", "Unknown"))
            pct_g = row["pct_guards"]
            pct_c = row["pct_centers"]
            height = row.get("height_inches", 78)

            masked_scores = _apply_positional_masks(
                raw_scores, own_pos, pct_g, pct_c, height
            )
            final_scores = _apply_role_repulsion(masked_scores)

            col_to_role = dict(zip(SCORE_COLS, SCORE_ROLES))
            sorted_roles = sorted(
                final_scores.items(), key=lambda x: x[1], reverse=True
            )
            archetype = col_to_role[sorted_roles[0][0]]
            top_score = sorted_roles[0][1]
            second_score = sorted_roles[1][1]
            separation = (top_score - second_score) / max(top_score, 0.01)
            confidence = 0.40 + separation * 0.55

            # Post-classification override: extreme liability
            effectiveness = compute_defensive_effectiveness(row, archetype)
            if effectiveness < 0.15:
                archetype = "Low-Activity Defender"
                secondary = "Liability"
                confidence = 0.35

            secondary = _pick_secondary(archetype, row)
            effectiveness = compute_defensive_effectiveness(row, archetype)
            fit = compute_defensive_fit(row, archetype)

        diff_pctl = row["difficulty_pctl"]
        difficulty_level = (
            "High" if diff_pctl >= 0.75
            else ("Low" if diff_pctl <= 0.25 else "Medium")
        )

        results.append({
            "PLAYER_ID": row["PLAYER_ID"],
            "PLAYER_NAME": row["PLAYER_NAME"],
            "SEASON": row["SEASON"],
            "GP": row["GP"],
            "MIN": row["MIN"],
            "defensive_archetype": archetype,
            "defensive_secondary": secondary,
            "defensive_confidence": min(0.95, confidence),
            "defensive_effectiveness": round(effectiveness, 3),
            "defensive_fit": fit,
            "assignment_difficulty": difficulty_level,
            "switch_score": row["switch_score"],
            "versatility_pctl": row["versatility_pctl"],
            "avg_opponent_ppg": row["avg_opponent_ppg"],
            "elite_matchup_pct": row["elite_matchup_pct"],
            "difficulty_pctl": diff_pctl,
            "STL_PER100_DEF_POSS": row["STL_PER100_DEF_POSS"],
            "stl_pctl": row["stl_pctl"],
            "BLK_PCT": row["BLK_PCT"],
            "blk_pctl": row["blk_pctl"],
            "DEF_RIM_FG_PCT": row["DEF_RIM_FG_PCT"],
            "RIM_FGA_RATE": row["RIM_FGA_RATE"],
            "engagement_score": row["engagement_score"],
            "engagement_pctl": eng_pctl,
            "hustle_score": row["hustle_score"],
            "hustle_pctl": row["hustle_pctl"],
            "D_FG_DIFF": row["D_FG_DIFF"],
            "d_results_pctl": d_res,
            "DEFLECTIONS": row.get("DEFLECTIONS", 0),
            "deflections_pctl": row.get("deflections_pctl", 0),
            "CONTESTED_SHOTS": row.get("CONTESTED_SHOTS", 0),
            "contested_shots_pctl": row.get("contested_shots_pctl", 0),
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
            "PLAYER_ID": row["PLAYER_ID"],
            "PLAYER_NAME": row["PLAYER_NAME"],
            "SEASON": row["SEASON"],
            "GP": row["GP"],
            "MIN": row["MIN"],
            "defensive_archetype": "Insufficient Minutes",
            "defensive_secondary": None,
            "defensive_confidence": 0.0,
            "defensive_effectiveness": 0.0,
            "defensive_fit": "N/A",
            "assignment_difficulty": "Unknown",
            "switch_score": 0, "versatility_pctl": 0,
            "avg_opponent_ppg": 0, "elite_matchup_pct": 0,
            "difficulty_pctl": 0,
            "STL_PER100_DEF_POSS": 0, "stl_pctl": 0,
            "BLK_PCT": 0, "blk_pctl": 0,
            "DEF_RIM_FG_PCT": 0, "RIM_FGA_RATE": 0,
            "engagement_score": 0, "engagement_pctl": 0,
            "hustle_score": 0, "hustle_pctl": 0,
            "D_FG_DIFF": 0, "d_results_pctl": 0,
            "DEFLECTIONS": 0, "deflections_pctl": 0,
            "CONTESTED_SHOTS": 0, "contested_shots_pctl": 0,
            **_zero,
        })

    return pd.DataFrame(results)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("DEFENSIVE ARCHETYPE CLASSIFICATION v3.0")
    print("  Score-Based + Positional Masks + Role Repulsion")
    print("=" * 70)

    print("\nLoading data...")
    versatility = load_matchup_versatility()
    difficulty = load_matchup_difficulty()
    box = load_box_score_data()
    profiles = load_player_profiles()
    speed_dist = load_speed_distance()
    reb_tracking = load_tracking_rebounding()
    hustle = load_hustle_stats()
    bios = load_player_bios()

    print(f"  Versatility: {len(versatility)} records")
    print(f"  Difficulty:  {len(difficulty)} records")
    print(f"  Box scores:  {len(box)} records")
    print(f"  Profiles:    {len(profiles)} records")
    print(f"  Hustle:      {len(hustle)} records")
    print(f"  Bios:        {len(bios)} records")

    all_results = []

    for season in SEASONS:
        print(f"\nProcessing {season}...")
        def_tracking = load_defense_tracking(season)
        trk_def = load_tracking_defense(season)
        features = compute_features(
            versatility, difficulty, def_tracking, trk_def,
            box, profiles, speed_dist, reb_tracking, hustle, bios, season,
        )
        if features.empty:
            continue

        results = classify_defenders(features)
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
    final_df.to_parquet(OUTPUT_DIR / "defensive_archetypes_v2.parquet", index=False)
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
        ("Cam Thomas", "Low-Activity or Off-Ball Chaser"),
        ("Luka Don", "Low-Activity or Off-Ball Chaser"),
    ]

    for name, expected in known:
        player = final_df[
            (final_df["PLAYER_NAME"].str.contains(name, case=False, na=False))
            & (final_df["SEASON"] == "2024-25")
        ]
        if len(player) == 0:
            player = final_df[
                final_df["PLAYER_NAME"].str.contains(name, case=False, na=False)
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
                f"  {r['PLAYER_NAME'][:22]:22} "
                f"| {r['defensive_archetype']:22}{sec:15}"
                f" | BLK%={r['BLK_PCT']:.3f}"
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
    blk = q["BLK_PCT"]
    print(f"  min={blk.min():.4f} max={blk.max():.4f} mean={blk.mean():.4f}")
    print(f"\n=== CONFIDENCE STATS ===")
    conf = q["defensive_confidence"]
    print(f"  min={conf.min():.2f} max={conf.max():.2f} mean={conf.mean():.2f}")
    print(f"\n=== EFFECTIVENESS STATS ===")
    eff = q["defensive_effectiveness"]
    print(f"  min={eff.min():.3f} max={eff.max():.3f} mean={eff.mean():.3f}")


if __name__ == "__main__":
    main()
