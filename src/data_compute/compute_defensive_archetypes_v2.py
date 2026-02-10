"""
src/data_compute/compute_defensive_archetypes_v2.py
Score-based defensive archetype classification (v2.1).
Defensive workload normalization (no per-36 for defense).

Defensive Archetype Definitions (v2.1 — Score-Based):
=====================================================
Primary Roles (6 scored + 1 filter):
  1. POA Defender       — Toughest perimeter assignments, high elite matchup share
  2. Wing Stopper       — Forward/wing assignments, strong results, specialist
  3. Off-Ball Chaser    — High steals, fast, lower matchup difficulty (roaming)
  4. Rim Protector      — Elite blocks + rim FG suppression
  5. Dropping Big       — Interior-only, low versatility, rim presence
  6. Mobile Big         — Versatile big, switches across positions
  7. Low-Activity Defender — Engagement filter (bottom ~18% + poor results)

Classification Method:
  - Compute 6 role affinity scores (weighted percentile sums, 0–1)
  - Primary archetype = argmax(scores)
  - Engagement used only as filter for Low-Activity
  - Secondary tags are modifiers (Hustler, Switchable, etc.), never primary
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
    season_dir = TRACKING_DIR / season
    all_defense = []
    categories = ["Overall", "3Pointers", "LessThan6Ft"]

    for cat in categories:
        path = season_dir / f"defense_{cat}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            df = df.rename(
                columns={
                    "D_FG_PCT": f"D_FG_PCT_{cat}",
                    "PCT_PLUSMINUS": f"PCT_PLUSMINUS_{cat}",
                    "D_FGM": f"D_FGM_{cat}",
                    "D_FGA": f"D_FGA_{cat}",
                    "FREQ": f"FREQ_{cat}",
                }
            )
            keep_cols = ["CLOSE_DEF_PERSON_ID", "PLAYER_NAME"] + [c for c in df.columns if cat in c]
            df = df[keep_cols]
            all_defense.append(df)

    if not all_defense:
        return pd.DataFrame()

    merged = all_defense[0]
    for df in all_defense[1:]:
        merged = merged.merge(df, on=["CLOSE_DEF_PERSON_ID", "PLAYER_NAME"], how="outer")

    merged["SEASON"] = season
    merged = merged.rename(columns={"CLOSE_DEF_PERSON_ID": "PLAYER_ID"})
    return merged


def load_tracking_defense(season: str) -> pd.DataFrame:
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
    cols = ["PLAYER_ID", "PLAYER_NAME", "SEASON", "GP", "MIN", "STL", "BLK", "DREB", "TOV", "PF"]
    available = [c for c in cols if c in df.columns]
    return df[available]


def load_player_profiles() -> pd.DataFrame:
    path = OUTPUT_DIR / "player_profiles_advanced.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df = df.rename(columns={"player_id": "PLAYER_ID", "season": "SEASON", "player_name": "PLAYER_NAME"})
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

    vers = normalize_player_id(vers)
    diff = normalize_player_id(diff)
    def_trk = normalize_player_id(def_trk)
    trk_def = normalize_player_id(trk_def)
    bx = normalize_player_id(bx)
    prof = normalize_player_id(prof)
    spd = normalize_player_id(spd)
    reb = normalize_player_id(reb)

    if bx.empty:
        return pd.DataFrame()

    features = bx.copy()

    if not vers.empty:
        vers_renamed = vers.rename(columns={"DEF_PLAYER_ID": "PLAYER_ID"})
        vers_renamed = normalize_player_id(vers_renamed)
        features = features.merge(
            vers_renamed.drop(columns=["DEF_PLAYER_NAME", "SEASON"], errors="ignore"),
            on="PLAYER_ID",
            how="left",
        )

    if not diff.empty:
        diff_renamed = diff.rename(columns={"DEF_PLAYER_ID": "PLAYER_ID"})
        diff_renamed = normalize_player_id(diff_renamed)
        features = features.merge(
            diff_renamed.drop(columns=["DEF_PLAYER_NAME", "SEASON"], errors="ignore"),
            on="PLAYER_ID",
            how="left",
        )

    if not def_trk.empty:
        features = features.merge(
            def_trk.drop(columns=["PLAYER_NAME", "SEASON"], errors="ignore"),
            on="PLAYER_ID",
            how="left",
        )

    if not trk_def.empty:
        features = features.merge(
            trk_def.drop(columns=["PLAYER_NAME", "SEASON"], errors="ignore"),
            on="PLAYER_ID",
            how="left",
        )

    if not prof.empty:
        features = features.merge(
            prof.drop(columns=["PLAYER_NAME", "SEASON"], errors="ignore"),
            on="PLAYER_ID",
            how="left",
            suffixes=("", "_prof"),
        )

    if not spd.empty and "PLAYER_ID" in spd.columns:
        spd_cols = ["PLAYER_ID", "DIST_MILES", "DIST_MILES_DEF", "AVG_SPEED", "AVG_SPEED_DEF"]
        spd_cols = [c for c in spd_cols if c in spd.columns]
        features = features.merge(spd[spd_cols], on="PLAYER_ID", how="left")

    if not reb.empty and "PLAYER_ID" in reb.columns:
        reb_cols = ["PLAYER_ID", "DREB_CHANCES", "DREB_CHANCE_PCT", "DREB_CONTEST", "DREB_CONTEST_PCT"]
        reb_cols = [c for c in reb_cols if c in reb.columns]
        features = features.merge(reb[reb_cols], on="PLAYER_ID", how="left")

    poss_def = safe_col(features, "POSS_DEF", np.nan)
    seconds_def = safe_col(features, "SECONDS_DEF", np.nan)

    poss_def = poss_def.fillna(features["MIN"] * 2.0)
    seconds_def = seconds_def.fillna(features["MIN"] * 60.0)

    features["POSS_DEF"] = poss_def
    features["SECONDS_DEF"] = seconds_def

    features["STL_PER100_DEF_POSS"] = safe_col(features, "STL", 0) / poss_def.replace(0, np.nan) * 100
    features["DREB_PER100_DEF_POSS"] = safe_col(features, "DREB", 0) / poss_def.replace(0, np.nan) * 100

    rim_fga = safe_col(features, "DEF_RIM_FGA", 0)
    features["BLK_PCT"] = safe_col(features, "BLK", 0) / rim_fga.replace(0, np.nan)
    features["RIM_FGA_RATE"] = rim_fga / poss_def.replace(0, np.nan) * 100

    features["BLK_PCT"] = features["BLK_PCT"].replace([np.inf, -np.inf], np.nan).fillna(0)
    features["RIM_FGA_RATE"] = features["RIM_FGA_RATE"].replace([np.inf, -np.inf], np.nan).fillna(0)

    features["DEF_SECONDS_PER_GAME"] = seconds_def / features["GP"].replace(0, np.nan)
    features["DEF_RIM_FG_PCT"] = safe_col(features, "DEF_RIM_FG_PCT", 0.62)

    features["SEASON"] = season
    return features


# =============================================================================
# CLASSIFICATION (v2.1 — Score-Based, 7 Archetypes)
# =============================================================================

SCORE_ROLES = ["POA Defender", "Wing Stopper", "Off-Ball Chaser",
               "Rim Protector", "Dropping Big", "Mobile Big"]
SCORE_COLS  = ["poa_score", "wing_score", "chaser_score",
               "rim_score", "drop_big_score", "mobile_big_score"]

LOW_ACTIVITY_ENGAGEMENT_GATE = 0.18
LOW_ACTIVITY_RESULTS_GATE = 0.40


def _pick_secondary(archetype: str, row) -> str:
    """Pick the most relevant secondary tag for the given primary."""
    vpctl = row["versatility_pctl"]
    stl_pctl = row["stl_pctl"]
    blk_pctl = row["blk_pctl"]
    d_res = row["d_results_pctl"]
    hustle_pctl = row["hustle_pctl"]

    if archetype == "POA Defender":
        if vpctl >= 0.75:
            return "Switchable"
        if stl_pctl >= 0.75:
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
        if stl_pctl >= 0.85:
            return "Ball Hawk"
        if hustle_pctl >= 0.75:
            return "Hustler"
        return "Active Hands"

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


def classify_defenders(features: pd.DataFrame) -> pd.DataFrame:
    """
    Score-based defensive classification (v2.1).
    Compute 6 role affinity scores (0-1) for each player, then assign the
    primary archetype via argmax.  Engagement is used only as a filter to
    identify Low-Activity Defenders before scoring.

    Primary Archetypes:
        POA Defender, Wing Stopper, Off-Ball Chaser,
        Rim Protector, Dropping Big, Mobile Big,
        Low-Activity Defender (engagement filter)
    """
    df = features.copy()
    qualified = df[(df["MIN"] >= MIN_MINUTES) & (df["GP"] >= MIN_GP)].copy()
    unqualified = df[(df["MIN"] < MIN_MINUTES) | (df["GP"] < MIN_GP)].copy()

    if len(qualified) == 0:
        return pd.DataFrame()

    # ---- Ensure columns exist with sensible defaults ----
    qualified["switch_score"] = safe_col(qualified, "switch_score", 0.5)
    qualified["avg_opponent_ppg"] = safe_col(qualified, "avg_opponent_ppg", 11.5)
    qualified["elite_matchup_pct"] = safe_col(qualified, "elite_matchup_pct", 0.15)
    qualified["STL_PER100_DEF_POSS"] = safe_col(qualified, "STL_PER100_DEF_POSS", 0.8)
    qualified["BLK_PCT"] = safe_col(qualified, "BLK_PCT", 0.03)
    qualified["DEF_RIM_FG_PCT"] = safe_col(qualified, "DEF_RIM_FG_PCT", 0.62)
    qualified["RIM_FGA_RATE"] = safe_col(qualified, "RIM_FGA_RATE", 6.0)
    qualified["D_FG_DIFF"] = safe_col(qualified, "PCT_PLUSMINUS_Overall", 0)
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

    # ---- Percentiles (adaptive across seasons) ----
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

    # ---- Engagement & Hustle (filter + modifier only) ----
    poss_def_pg = qualified["POSS_DEF"] / qualified["GP"].replace(0, np.nan)
    seconds_def_pg = qualified["SECONDS_DEF"] / qualified["GP"].replace(0, np.nan)
    dist_def_pg = qualified["DIST_MILES_DEF"] / qualified["GP"].replace(0, np.nan)

    def zscore(series):
        mean = series.mean(skipna=True)
        std = series.std(skipna=True)
        if std == 0 or np.isnan(std):
            return series * 0
        return (series - mean) / std

    engagement_score = (
        zscore(poss_def_pg.fillna(0))
        + zscore(seconds_def_pg.fillna(0))
        + zscore(dist_def_pg.fillna(0))
        + zscore(qualified["AVG_SPEED_DEF"].fillna(0))
    )
    hustle_score = zscore(qualified["DREB_CHANCES"].fillna(0)) + zscore(qualified["DREB_CHANCE_PCT"].fillna(0))

    qualified["engagement_score"] = engagement_score
    qualified["hustle_score"] = hustle_score
    qualified["engagement_pctl"] = qualified["engagement_score"].rank(pct=True)
    qualified["hustle_pctl"] = qualified["hustle_score"].rank(pct=True)

    # ---- Role Affinity Scores (0-1, weighted percentile sums) ----
    # POA Defender — guards primary ball handlers
    qualified["poa_score"] = (
        qualified["elite_matchup_pctl"] * 0.30
        + qualified["difficulty_pctl"] * 0.25
        + qualified["guard_pctl"] * 0.25
        + qualified["d_results_pctl"] * 0.10
        + qualified["stl_pctl"] * 0.10
    )
    # Wing Stopper — defends forwards/wings, results-focused
    qualified["wing_score"] = (
        qualified["forward_pctl"] * 0.25
        + qualified["difficulty_pctl"] * 0.25
        + qualified["d_results_pctl"] * 0.30
        + (1 - qualified["versatility_pctl"]) * 0.10
        + (1 - qualified["center_pctl"]) * 0.10
    )
    # Off-Ball Chaser — high steals, fast, lower difficulty
    qualified["chaser_score"] = (
        qualified["stl_pctl"] * 0.35
        + qualified["speed_pctl"] * 0.15
        + (1 - qualified["difficulty_pctl"]) * 0.25
        + (1 - qualified["center_pctl"]) * 0.10
        + qualified["d_results_pctl"] * 0.15
    )
    # Rim Protector — elite blocks + rim FG suppression
    qualified["rim_score"] = (
        qualified["blk_pctl"] * 0.35
        + (1 - qualified["rim_fg_pctl"]) * 0.30
        + qualified["rim_fga_pctl"] * 0.15
        + qualified["center_pctl"] * 0.20
    )
    # Dropping Big — interior-only, low versatility
    qualified["drop_big_score"] = (
        (1 - qualified["versatility_pctl"]) * 0.25
        + qualified["center_pctl"] * 0.25
        + qualified["reb_pctl"] * 0.25
        + qualified["rim_fga_pctl"] * 0.15
        + (1 - qualified["guard_pctl"]) * 0.10
    )
    # Mobile Big — versatile big with decent results
    qualified["mobile_big_score"] = (
        qualified["versatility_pctl"] * 0.30
        + qualified["reb_pctl"] * 0.20
        + qualified["center_pctl"] * 0.15
        + qualified["d_results_pctl"] * 0.20
        + qualified["forward_pctl"] * 0.15
    )

    # ---- Assign primary archetype via argmax ----
    results = []

    for _, row in qualified.iterrows():
        eng_pctl = row["engagement_pctl"]
        d_res = row["d_results_pctl"]

        # Filter: Low-Activity
        if eng_pctl <= LOW_ACTIVITY_ENGAGEMENT_GATE and d_res <= LOW_ACTIVITY_RESULTS_GATE:
            archetype = "Low-Activity Defender"
            secondary = "Liability" if d_res <= 0.25 else "Hidden"
            confidence = 0.30 + (1 - eng_pctl) * 0.40
        else:
            scores = {name: row[col] for name, col in zip(SCORE_ROLES, SCORE_COLS)}
            sorted_roles = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            archetype = sorted_roles[0][0]
            top_score = sorted_roles[0][1]
            second_score = sorted_roles[1][1]
            separation = (top_score - second_score) / max(top_score, 0.01)
            confidence = 0.40 + separation * 0.55
            secondary = _pick_secondary(archetype, row)

        diff_pctl = row["difficulty_pctl"]
        difficulty_level = "High" if diff_pctl >= 0.75 else ("Low" if diff_pctl <= 0.25 else "Medium")

        results.append({
            "PLAYER_ID": row["PLAYER_ID"],
            "PLAYER_NAME": row["PLAYER_NAME"],
            "SEASON": row["SEASON"],
            "GP": row["GP"],
            "MIN": row["MIN"],
            "defensive_archetype": archetype,
            "defensive_secondary": secondary,
            "defensive_confidence": min(0.95, confidence),
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
            "poa_score": row["poa_score"],
            "wing_score": row["wing_score"],
            "chaser_score": row["chaser_score"],
            "rim_score": row["rim_score"],
            "drop_big_score": row["drop_big_score"],
            "mobile_big_score": row["mobile_big_score"],
        })

    _zero_scores = {c: 0 for c in SCORE_COLS}
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
            **_zero_scores,
        })

    return pd.DataFrame(results)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("DEFENSIVE ARCHETYPE CLASSIFICATION v2.1 (Score-Based)")
    print("=" * 70)

    print("\nLoading data...")
    versatility = load_matchup_versatility()
    difficulty = load_matchup_difficulty()
    box = load_box_score_data()
    profiles = load_player_profiles()
    speed_dist = load_speed_distance()
    reb_tracking = load_tracking_rebounding()

    print(f"  Versatility: {len(versatility)} records")
    print(f"  Difficulty: {len(difficulty)} records")
    print(f"  Box scores: {len(box)} records")
    print(f"  Profiles: {len(profiles)} records")

    all_results = []

    for season in SEASONS:
        print(f"\nProcessing {season}...")

        def_tracking = load_defense_tracking(season)
        trk_def = load_tracking_defense(season)
        features = compute_features(
            versatility,
            difficulty,
            def_tracking,
            trk_def,
            box,
            profiles,
            speed_dist,
            reb_tracking,
            season,
        )

        if features.empty:
            continue

        results = classify_defenders(features)
        all_results.append(results)

        arch_dist = results[results["defensive_archetype"] != "Insufficient Minutes"][
            "defensive_archetype"
        ].value_counts()
        print("  Distribution:")
        for arch, count in arch_dist.items():
            print(f"    {arch}: {count}")

    if not all_results:
        print("\nNo results")
        return

    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_parquet(OUTPUT_DIR / "defensive_archetypes_v2.parquet", index=False)
    final_df.to_csv(OUTPUT_DIR / "defensive_archetypes_v2.csv", index=False)

    print(f"\nSaved {len(final_df)} records")

    print("\n=== VALIDATION (Known Defenders 2024-25) ===")
    known = [
        ("Jrue Holiday", "POA Defender"),
        ("Draymond Green", "Mobile Big"),
        ("Rudy Gobert", "Rim Protector"),
        ("Anthony Davis", "Rim Protector"),
        ("Herb Jones", "Wing Stopper or POA"),
        ("Dyson Daniels", "POA Defender"),
        ("Bam Adebayo", "Mobile Big"),
        ("Victor Wembanyama", "Rim Protector"),
        ("Trae Young", "Low-Activity"),
        ("Chet Holmgren", "Rim Protector"),
        ("OG Anunoby", "Wing Stopper"),
        ("Derrick White", "POA Defender"),
    ]

    for name, expected in known:
        player = final_df[
            (final_df["PLAYER_NAME"].str.contains(name, case=False, na=False))
            & (final_df["SEASON"] == "2024-25")
        ]
        if len(player) == 0:
            player = final_df[final_df["PLAYER_NAME"].str.contains(name, case=False, na=False)]

        if len(player) > 0:
            r = player.iloc[0]
            sec = f" ({r['defensive_secondary']})" if r["defensive_secondary"] else ""
            scores = (
                f"POA={r.get('poa_score',0):.2f} Wing={r.get('wing_score',0):.2f} "
                f"Chase={r.get('chaser_score',0):.2f} Rim={r.get('rim_score',0):.2f} "
                f"Drop={r.get('drop_big_score',0):.2f} Mob={r.get('mobile_big_score',0):.2f}"
            )
            print(
                f"  {r['PLAYER_NAME'][:22]:22} | {r['defensive_archetype']:22}{sec:15}"
                f" | {scores} | Expected: {expected}"
            )


if __name__ == "__main__":
    main()
