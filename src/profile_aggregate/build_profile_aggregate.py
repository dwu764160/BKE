"""
src/profile_aggregate/build_profile_aggregate.py
=============================================================================
Build the comprehensive Player Profile Aggregate — one row per player-season
containing ALL available pipeline data merged from every source.

This file is the foundational data layer for all downstream products.
Future pipelines, viewers, and models should read from the aggregate rather
than merging raw sources themselves.

Sources merged:
  1. BKE v28 decomposition          (impact dimensions, playtypes, tracking, archetypes)
  2. BKE Scores v27 JSON            (raw OBKE/DBKE, percentiles, dimension scores)
  3. Complete player season stats    (box stats, advanced stats from NBA API)
  4. Player archetypes               (offensive roles, embeddings, playtype breakdown)
  5. Defensive archetypes v2         (defensive roles, hustle, matchup data)
  6. Archetype embeddings            (standalone embedding vectors)
  7. Position estimates              (position distributions)
  8. Player profiles advanced        (advanced per-possession ratings)
  9. Metrics linear                  (WS, BPM, VORP)
  10. Player RAPM                    (multi-season pooled RAPM)
  11. Player xRAPM v1/v2            (expected RAPM with priors)
  12. DARKO                         (external DPM model data)
  13. Players metadata              (bio: height, weight, wingspan, experience)
  14. Salary data                   (per-season contract info)
    15. Game logs aggregates          (MPG variance, DNP estimation)
    16. Player clutch stats           (clutch minutes/production by season)
    17. Step 1 impact profiles        (curated impact/behavioral features)
    18. BKE diagnostic reports        (stability, shrinkage)
    19. Player draft history          (draft class year, round, pick, tier)

Output:
  aggregate/player_profile_aggregate.parquet

Usage:
  python3 src/profile_aggregate/build_profile_aggregate.py
=============================================================================
"""

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.player_eval.constants import (
    AGGREGATE_DIR,
    AGGREGATE_VALIDATION_REPORT,
    BKE_DECOMP_PATH,
    BKE_SCORES_PATH,
    BKE_V29_PLAYER_DIAGNOSTIC_PATH,
    CLUTCH_STATS_ALL_PATH,
    CLUTCH_STATS_GLOB,
    COMPLETE_STATS_PATH,
    DARKO_DIR,
    DBKE_V30_SHRINKAGE_PATH,
    DEF_ARCHETYPES_PATH,
    GAME_LOGS_PATH,
    HISTORICAL_DIR,
    METRICS_LINEAR_PATH,
    PLAYER_ARCHETYPES_PATH,
    PLAYER_DRAFT_HISTORY_PATH,
    PLAYER_PROFILES_PARQUET,
    PLAYERS_META_PATH,
    POSITION_ESTIMATES_PATH,
    PROFILE_AGGREGATE_PATH,
    PROCESSED_DIR,
    REPORTS_DIR,
    XRAPM_V1_PATH,
    XRAPM_V2_PATH,
)
from src.utils.player_name_normalizer import (
    apply_player_name_normalization,
    build_player_name_maps,
)


def _norm_id(series: pd.Series) -> pd.Series:
    """Normalize player_id to clean string (no trailing .0)."""
    return series.astype(str).str.replace(r"\.0$", "", regex=True)


def _safe_load(path: Path, **kwargs) -> pd.DataFrame:
    """Load parquet, return empty DataFrame on failure."""
    try:
        return load_standardized(path, **kwargs)
    except Exception:
        return pd.DataFrame()


def _dedup(df: pd.DataFrame, key_cols=("player_id", "season")) -> pd.DataFrame:
    """Drop duplicate rows on key columns, keeping first."""
    return df.drop_duplicates(subset=list(key_cols), keep="first")


# ═════════════════════════════════════════════════════════════════════
# Loader helpers — each returns a DataFrame keyed on (player_id, season)
# ═════════════════════════════════════════════════════════════════════

def _load_bke_decomp() -> pd.DataFrame:
    """BKE v28 decomposition — the densest single source (379 cols)."""
    df = _safe_load(BKE_DECOMP_PATH)
    if df.empty:
        return df
    df["player_id"] = _norm_id(df["player_id"])
    df["season"] = df["season"].astype(str)
    return _dedup(df)


def _load_complete_stats() -> pd.DataFrame:
    """NBA API box-score stats. Deduplicate traded players (keep max-minute row)."""
    df = _safe_load(COMPLETE_STATS_PATH)
    if df.empty:
        return df
    df = df.rename(columns={"player_id": "player_id", "season": "season"})
    df["player_id"] = _norm_id(df["player_id"])
    df["season"] = df["season"].astype(str)
    df["min"] = pd.to_numeric(df.get("min"), errors="coerce")
    df = df.sort_values("min", ascending=False).drop_duplicates(
        subset=["player_id", "season"], keep="first"
    )
    # Prefix rank columns to avoid collisions
    rank_cols = [c for c in df.columns if c.endswith("_RANK")]
    df = df.rename(columns={c: f"box_{c}" for c in rank_cols})
    return df


def _load_archetypes() -> pd.DataFrame:
    """Offensive archetype data + tracking metrics."""
    df = _safe_load(PLAYER_ARCHETYPES_PATH)
    if df.empty:
        return df
    df = df.rename(columns={"player_id": "player_id", "season": "season"})
    df["player_id"] = _norm_id(df["player_id"])
    df["season"] = df["season"].astype(str)
    return _dedup(df)


def _load_def_archetypes() -> pd.DataFrame:
    """Defensive archetypes v2 with role scores and percentiles."""
    df = _safe_load(DEF_ARCHETYPES_PATH)
    if df.empty:
        return df
    df = df.rename(columns={"player_id": "player_id", "season": "season"})
    df["player_id"] = _norm_id(df["player_id"])
    df["season"] = df["season"].astype(str)
    return _dedup(df)


def _load_positions() -> pd.DataFrame:
    df = _safe_load(POSITION_ESTIMATES_PATH)
    if df.empty:
        return df
    df = df.rename(columns={"player_id": "player_id", "season": "season"})
    df["player_id"] = _norm_id(df["player_id"])
    df["season"] = df["season"].astype(str)
    return _dedup(df)


def _load_metrics_linear() -> pd.DataFrame:
    df = _safe_load(METRICS_LINEAR_PATH)
    if df.empty:
        return df
    df["player_id"] = _norm_id(df["player_id"])
    df["season"] = df["season"].astype(str)
    return _dedup(df)


def _load_rapm() -> pd.DataFrame:
    path = PROCESSED_DIR / "player_rapm.parquet"
    df = _safe_load(path)
    if df.empty:
        return df
    df["player_id"] = _norm_id(df["player_id"])
    df["season"] = df["season"].astype(str)
    # Keep single-season or best RAPM row per (player, season)
    df = df.sort_values("possessions_played", ascending=False)
    return _dedup(df)


def _load_xrapm() -> pd.DataFrame:
    """Load xRAPM v2 (preferred) falling back to v1."""
    df = _safe_load(XRAPM_V2_PATH)
    if df.empty:
        df = _safe_load(XRAPM_V1_PATH)
    if df.empty:
        return df
    df["player_id"] = _norm_id(df["player_id"])
    df["season"] = df["season"].astype(str)
    return _dedup(df)


def _load_darko() -> pd.DataFrame:
    """Combine per-season DARKO parquet files."""
    frames = []
    for f in sorted(glob.glob(str(DARKO_DIR / "darko_*.parquet"))):
        d = _safe_load(Path(f))
        if d.empty:
            continue
        d = d.rename(columns={"nba_id": "player_id"})
        d["player_id"] = _norm_id(d["player_id"])
        d["season"] = d["season"].astype(str)
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return _dedup(df)


def _load_players_meta() -> pd.DataFrame:
    """Player bio data (height, weight, wingspan, experience)."""
    df = _safe_load(PLAYERS_META_PATH)
    if df.empty:
        return df
    df = df.rename(columns={"player_id": "player_id"})
    df["player_id"] = _norm_id(df["player_id"])
    # Bio is not season-specific — drop season-like columns if present
    keep = ["player_id", "height_inches", "weight_lbs", "wingspan_inches",
            "experience_years", "primary_position"]
    keep = [c for c in keep if c in df.columns]
    return df[keep].drop_duplicates(subset=["player_id"], keep="first")


def _load_player_draft_history() -> pd.DataFrame:
    """Load player draft history (player-level metadata)."""
    df = _safe_load(PLAYER_DRAFT_HISTORY_PATH)
    if df.empty or "player_id" not in df.columns:
        return pd.DataFrame()

    df["player_id"] = _norm_id(df["player_id"])
    keep = [
        "player_id",
        "draft_class_year",
        "draft_round",
        "draft_pick_in_round",
        "draft_pick_overall",
        "draft_tier",
        "is_drafted",
        "is_undrafted",
        "draft_team_abbreviation",
        "draft_source",
    ]
    keep = [c for c in keep if c in df.columns]
    if not keep:
        return pd.DataFrame()
    return df[keep].drop_duplicates(subset=["player_id"], keep="first")


def _load_salaries() -> pd.DataFrame:
    """Combine per-season salary parquet files.
    
    Handles both ID-matched rows (player_id is set) and name-only rows
    (player_id is NaN but player_name is set, from fetcher fallbacks).
    """
    files = sorted(glob.glob(str(HISTORICAL_DIR / "player_salaries_*.parquet")))
    frames = []
    for f in files:
        d = _safe_load(Path(f))
        if d.empty:
            continue
        # Keep rows with valid player_id
        has_id = d["player_id"].notna()
        if has_id.any():
            matched = d.loc[has_id].copy()
            matched["player_id"] = _norm_id(matched["player_id"])
            matched["season"] = matched["season"].astype(str)
            frames.append(matched[["player_id", "season", "salary"]])
    if not frames:
        return pd.DataFrame(columns=["player_id", "season", "salary"])
    return _dedup(pd.concat(frames, ignore_index=True))


def _load_game_log_aggregates() -> pd.DataFrame:
    """Compute per-player-season aggregates from individual game logs."""
    df = _safe_load(GAME_LOGS_PATH)
    if df.empty:
        return pd.DataFrame()
    # Normalize columns — handle both Player_ID and PLAYER_ID
    # Drop duplicate player id column first
    if "Player_ID" in df.columns and "player_id" in df.columns:
        df = df.drop(columns=["Player_ID"])
        df = df.rename(columns={"player_id": "player_id"})
    elif "Player_ID" in df.columns:
        df = df.rename(columns={"Player_ID": "player_id"})
    elif "player_id" in df.columns:
        df = df.rename(columns={"player_id": "player_id"})
    df = df.rename(columns={"season": "season"})
    if "player_id" not in df.columns:
        return pd.DataFrame()
    df["player_id"] = _norm_id(df["player_id"])
    df["season"] = df["season"].astype(str)
    df["min"] = pd.to_numeric(df["min"], errors="coerce")

    agg = df.groupby(["player_id", "season"]).agg(
        gl_games_total=("min", "count"),
        gl_mpg_mean=("min", "mean"),
        gl_mpg_std=("min", "std"),
        gl_mpg_median=("min", "median"),
        gl_mpg_max=("min", "max"),
        gl_mpg_min=("min", "min"),
        gl_dnp_count=("min", lambda x: (x.fillna(0) == 0).sum()),
        gl_low_min_count=("min", lambda x: ((x.fillna(0) > 0) & (x.fillna(0) < 5)).sum()),
    ).reset_index()
    agg["gl_mpg_std"] = agg["gl_mpg_std"].fillna(0.0)
    return agg


def _load_clutch_stats() -> pd.DataFrame:
    """Load player clutch stats fetched from LeagueDashPlayerClutch."""
    frames = []

    if CLUTCH_STATS_ALL_PATH.exists():
        all_df = _safe_load(CLUTCH_STATS_ALL_PATH)
        if not all_df.empty:
            frames.append(all_df)
    else:
        for file in sorted(HISTORICAL_DIR.glob(CLUTCH_STATS_GLOB)):
            df = _safe_load(file)
            if not df.empty:
                frames.append(df)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    rename_map = {}
    if "player_id" in df.columns:
        rename_map["player_id"] = "player_id"
    if "season" in df.columns:
        rename_map["season"] = "season"
    df = df.rename(columns=rename_map)

    if "player_id" not in df.columns or "season" not in df.columns:
        return pd.DataFrame()

    df["player_id"] = _norm_id(df["player_id"])
    df["season"] = df["season"].astype(str)

    # Ensure clutch columns are explicitly prefixed to avoid merge collisions.
    candidates = [
        "player_name",
        "team_id",
        "team_abbreviation",
        "clutch_gp",
        "clutch_minutes",
        "clutch_pts",
        "clutch_ast",
        "clutch_reb",
        "clutch_plus_minus",
    ]
    for col in candidates:
        if col in df.columns and not col.startswith("clutch_") and col not in {"player_name"}:
            df = df.rename(columns={col: f"clutch_{col}"})

    # Keep one row per player-season (top clutch-minute sample for traded players).
    min_col = "clutch_minutes" if "clutch_minutes" in df.columns else None
    if min_col is not None:
        df[min_col] = pd.to_numeric(df[min_col], errors="coerce").fillna(0.0)
        df = df.sort_values(min_col, ascending=False)
    df = _dedup(df)

    keep = ["player_id", "season"]
    for col in [
        "clutch_team_id",
        "clutch_team_abbreviation",
        "clutch_gp",
        "clutch_minutes",
        "clutch_pts",
        "clutch_ast",
        "clutch_reb",
        "clutch_plus_minus",
    ]:
        if col in df.columns:
            keep.append(col)

    # Backward compatibility for older fetch outputs without explicit clutch_ prefixes.
    if "team_id" in df.columns and "clutch_team_id" not in keep:
        df = df.rename(columns={"team_id": "clutch_team_id"})
        keep.append("clutch_team_id")
    if "team_abbreviation" in df.columns and "clutch_team_abbreviation" not in keep:
        df = df.rename(columns={"team_abbreviation": "clutch_team_abbreviation"})
        keep.append("clutch_team_abbreviation")
    if "gp" in df.columns and "clutch_gp" not in keep:
        df = df.rename(columns={"gp": "clutch_gp"})
        keep.append("clutch_gp")
    if "min" in df.columns and "clutch_minutes" not in keep:
        df = df.rename(columns={"min": "clutch_minutes"})
        keep.append("clutch_minutes")

    keep = [c for c in keep if c in df.columns]
    return df[keep]


def _load_bke_json_scores() -> pd.DataFrame:
    """Extract per-player BKE scores from JSON (OBKE, DBKE, percentiles, dimensions)."""
    if not BKE_SCORES_PATH.exists():
        return pd.DataFrame()
    payload = json.loads(BKE_SCORES_PATH.read_text(encoding="utf-8"))
    rows = []
    for _, info in payload.get("players", {}).items():
        row = {
            "player_id": str(info.get("player_id", "")),
            "season": str(info.get("season", "")),
            "bke_raw_obke": info.get("raw_OBKE"),
            "bke_raw_dbke": info.get("raw_DBKE"),
            "bke_raw_bke": info.get("raw_BKE"),
            "bke_transformed_bke": info.get("transformed_BKE"),
            "bke_final_pctl": info.get("final_BKE_percentile"),
            "bke_obke_pctl": info.get("final_OBKE_percentile"),
            "bke_dbke_pctl": info.get("final_DBKE_percentile"),
            "bke_rank": info.get("rank"),
        }
        # Dimension scores
        dims = info.get("dimension_scores", {})
        for dk, dv in dims.items():
            row[f"bke_dim_{dk}"] = dv
        rows.append(row)
    df = pd.DataFrame(rows)
    df["player_id"] = _norm_id(df["player_id"])
    return _dedup(df)


def _load_diagnostics() -> pd.DataFrame:
    """Load v29 diagnostic and v30 shrinkage data."""
    frames = []
    # v29
    if BKE_V29_PLAYER_DIAGNOSTIC_PATH.exists():
        payload = json.loads(BKE_V29_PLAYER_DIAGNOSTIC_PATH.read_text(encoding="utf-8"))
        rows = payload.get("player_level_metrics", [])
        if rows:
            v29 = pd.DataFrame(rows)
            v29 = v29.rename(columns={
                "d_stability": "diag_d_stability",
                "o_stability": "diag_o_stability",
                "vol_ratio_dbke_to_drapm": "diag_vol_ratio",
            })
            v29["player_id"] = _norm_id(v29["player_id"])
            v29["season"] = v29["season"].astype(str)
            frames.append(v29)
    # v30
    if DBKE_V30_SHRINKAGE_PATH.exists():
        payload = json.loads(DBKE_V30_SHRINKAGE_PATH.read_text(encoding="utf-8"))
        rows = payload.get("player_outputs", [])
        if rows:
            v30 = pd.DataFrame(rows)
            v30["player_id"] = _norm_id(v30["player_id"])
            v30["season"] = v30["season"].astype(str)
            frames.append(v30)
    if not frames:
        return pd.DataFrame()
    result = frames[0]
    for f in frames[1:]:
        result = result.merge(f, on=["player_id", "season"], how="outer", suffixes=("", "_v30"))
    return _dedup(result)


def _load_pts_v40() -> pd.DataFrame:
    """Load PTS v4.0 composite scores (production metric)."""
    from src.modeling.model_config import PTS_V40_PARQUET
from src.data.schema_contract import load_standardized, save_standardized
    df = _safe_load(Path(PTS_V40_PARQUET))
    if df.empty:
        return df
    df["player_id"] = _norm_id(df["player_id"])
    df["season"] = df["season"].astype(str)
    keep = ["player_id", "season", "pts_o_v40", "pts_d_v40"]
    return _dedup(df[[c for c in keep if c in df.columns]])


def _load_step1_profiles() -> pd.DataFrame:
    """Load Step 1 curated impact profiles."""
    df = _safe_load(PLAYER_PROFILES_PARQUET)
    if df.empty:
        return df
    df["player_id"] = _norm_id(df["player_id"])
    df["season"] = df["season"].astype(str)
    # Prefix to avoid column collisions
    skip_cols = {"player_id", "season", "player_name", "team_abbreviation", "team_id"}
    rename = {c: f"pec_{c}" for c in df.columns if c not in skip_cols}
    df = df.rename(columns=rename)
    return _dedup(df)


# ═════════════════════════════════════════════════════════════════════
# Smart merge — handles column conflicts by suffixing the incoming source
# ═════════════════════════════════════════════════════════════════════

def _smart_merge(base: pd.DataFrame, incoming: pd.DataFrame,
                 on: list, source_tag: str, how: str = "left") -> pd.DataFrame:
    """Merge two DataFrames, handling overlapping columns by keeping base version."""
    if incoming.empty:
        return base
    overlap = set(base.columns) & set(incoming.columns) - set(on)
    if overlap:
        incoming = incoming.rename(columns={c: f"{source_tag}_{c}" for c in overlap})
    return base.merge(incoming, on=on, how=how)


def _to_snake(name: str) -> str:
    text = str(name).strip()
    text = text.replace("%", " pct ").replace("+", " plus ")
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    text = re.sub(r"[^0-9a-zA-Z]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_").lower()
    return text or "col"


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize all column names to portable snake_case and resolve collisions."""
    used = {}
    renamed = {}
    for col in df.columns:
        base = _to_snake(col)
        count = used.get(base, 0)
        if count == 0:
            final = base
        else:
            final = f"{base}_{count + 1}"
        used[base] = count + 1
        renamed[col] = final
    return df.rename(columns=renamed)


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════

def main() -> None:
    print("Building Player Profile Aggregate ...")
    merge_key = ["player_id", "season"]

    # 1. Spine: BKE decomposition (most comprehensive single source)
    spine = _load_bke_decomp()
    if spine.empty:
        raise FileNotFoundError(f"Missing BKE decomposition: {BKE_DECOMP_PATH}")
    print(f"  [1] BKE decomposition: {len(spine)} rows, {len(spine.columns)} cols")

    # 2. Complete box stats
    stats = _load_complete_stats()
    spine = _smart_merge(spine, stats, merge_key, "box", how="left")
    print(f"  [2] + Box stats: {len(spine.columns)} cols")

    # 3. Offensive archetypes (adds tracking detail not in decomp)
    arche = _load_archetypes()
    spine = _smart_merge(spine, arche, merge_key, "arche", how="left")
    # BKE decomp spine carries stale archetype labels (None for pre-2022, and
    # "Insufficient Minutes" from the old 500-min threshold for 2022+). After
    # _smart_merge the fresh values from player_archetypes.parquet land as
    # arche_primary_archetype / arche_secondary_archetype. Prefer the fresh file
    # value whenever it is non-null — this overwrites both None and stale labels.
    for _col, _src in [("primary_archetype", "arche_primary_archetype"),
                       ("secondary_archetype", "arche_secondary_archetype"),
                       ("defensive_archetype", "arche_defensive_archetype")]:
        if _src in spine.columns and _col in spine.columns:
            spine[_col] = spine[_src].where(spine[_src].notna(), spine[_col])
    # Secondary archetype: if fresh primary is non-null, the fresh secondary (even if None)
    # should overwrite the stale spine value — otherwise old tags persist when new
    # classification correctly assigns no secondary.
    if "arche_primary_archetype" in spine.columns and "secondary_archetype" in spine.columns:
        fresh_has_primary = spine["arche_primary_archetype"].notna()
        fresh_secondary = spine.get("arche_secondary_archetype")
        if fresh_secondary is not None:
            spine.loc[fresh_has_primary, "secondary_archetype"] = spine.loc[
                fresh_has_primary, "arche_secondary_archetype"
            ]
    print(f"  [3] + Archetypes: {len(spine.columns)} cols")

    # 4. Defensive archetypes
    def_arche = _load_def_archetypes()
    spine = _smart_merge(spine, def_arche, merge_key, "defarche", how="left")
    # Same coalesce for defensive_archetype
    if "defarche_defensive_archetype" in spine.columns and "defensive_archetype" in spine.columns:
        spine["defensive_archetype"] = spine["defensive_archetype"].where(
            spine["defensive_archetype"].notna(), spine["defarche_defensive_archetype"]
        )
    print(f"  [4] + Def archetypes: {len(spine.columns)} cols")

    # 5. Position estimates
    pos = _load_positions()
    spine = _smart_merge(spine, pos, merge_key, "pos", how="left")
    print(f"  [5] + Positions: {len(spine.columns)} cols")

    # 6. Metrics linear (WS, BPM, VORP)
    ml = _load_metrics_linear()
    spine = _smart_merge(spine, ml, merge_key, "ml", how="left")
    print(f"  [6] + Linear metrics: {len(spine.columns)} cols")

    # 7. RAPM
    rapm = _load_rapm()
    spine = _smart_merge(spine, rapm, merge_key, "rapm", how="left")
    print(f"  [7] + RAPM: {len(spine.columns)} cols")

    # 8. xRAPM
    xrapm = _load_xrapm()
    spine = _smart_merge(spine, xrapm, merge_key, "xrapm", how="left")
    print(f"  [8] + xRAPM: {len(spine.columns)} cols")

    # 9. DARKO
    darko = _load_darko()
    spine = _smart_merge(spine, darko, merge_key, "darko", how="left")
    print(f"  [9] + DARKO: {len(spine.columns)} cols")

    # 10. BKE JSON scores
    bke_json = _load_bke_json_scores()
    spine = _smart_merge(spine, bke_json, merge_key, "bkejson", how="left")
    print(f"  [10] + BKE JSON: {len(spine.columns)} cols")

    # 11. Diagnostics (v29 + v30)
    diag = _load_diagnostics()
    spine = _smart_merge(spine, diag, merge_key, "diag", how="left")
    print(f"  [11] + Diagnostics: {len(spine.columns)} cols")

    # 12. Salary
    salary = _load_salaries()
    spine = _smart_merge(spine, salary, merge_key, "sal", how="left")
    print(f"  [12] + Salary: {len(spine.columns)} cols")

    # 13. Game log aggregates
    gl = _load_game_log_aggregates()
    spine = _smart_merge(spine, gl, merge_key, "gl", how="left")
    print(f"  [13] + Game logs: {len(spine.columns)} cols")

    # 14. Player bio metadata (merge on player_id only)
    bio = _load_players_meta()
    if not bio.empty:
        bio_overlap = set(spine.columns) & set(bio.columns) - {"player_id"}
        if bio_overlap:
            bio = bio.rename(columns={c: f"bio_{c}" for c in bio_overlap})
        spine = spine.merge(bio, on="player_id", how="left")
    print(f"  [14] + Bio metadata: {len(spine.columns)} cols")

    # 15. Player draft metadata (merge on player_id only)
    draft = _load_player_draft_history()
    if not draft.empty:
        draft_overlap = set(spine.columns) & set(draft.columns) - {"player_id"}
        if draft_overlap:
            draft = draft.rename(columns={c: f"draft_{c}" for c in draft_overlap})
        spine = spine.merge(draft, on="player_id", how="left")
    print(f"  [15] + Draft history: {len(spine.columns)} cols")

    # 16. Clutch stats
    clutch = _load_clutch_stats()
    spine = _smart_merge(spine, clutch, merge_key, "clutch", how="left")
    print(f"  [16] + Clutch stats: {len(spine.columns)} cols")

    # 17. Step 1 curated profiles
    step1 = _load_step1_profiles()
    spine = _smart_merge(spine, step1, merge_key, "s1", how="left")
    print(f"  [17] + Step 1 profiles: {len(spine.columns)} cols")

    # 18. PTS v4.0 scores (production metric — must appear after BKE decomp spine)
    pts_v40 = _load_pts_v40()
    spine = _smart_merge(spine, pts_v40, merge_key, "pts40", how="left")
    print(f"  [18] + PTS v4.0: {len(spine.columns)} cols")

    # 19. Name normalization (ID-first and alias-aware)
    name_sources = [
        (PLAYERS_META_PATH, ["id", "player_id"], ["full_name", "player_name"], 1),
        (PLAYER_ARCHETYPES_PATH, ["player_id", "player_id"], ["player_name", "player_name"], 2),
        (COMPLETE_STATS_PATH, ["player_id", "player_id"], ["player_name", "player_name"], 2),
        (BKE_DECOMP_PATH, ["player_id", "player_id"], ["player_name", "player_name"], 3),
    ]
    id_to_name, key_to_name = build_player_name_maps(name_sources)
    if "player_name" in spine.columns:
        spine = apply_player_name_normalization(
            df=spine,
            player_id_col="player_id",
            player_name_col="player_name",
            id_to_name=id_to_name,
            key_to_name=key_to_name,
        )

    # ── Computed fields ────────────────────────────────────────────────
    # MPG from game logs or box stats
    mins = pd.to_numeric(spine.get("min", spine.get("min")), errors="coerce")
    gp = pd.to_numeric(spine.get("gp", spine.get("gp")), errors="coerce")
    spine["agg_mpg"] = mins / gp.replace(0, np.nan)

    # Minute share within team
    team_mins = spine.groupby(["season", spine.columns[spine.columns.str.contains("team_abbreviation", case=False)].tolist()[0] if any(spine.columns.str.contains("team_abbreviation", case=False)) else "season"])["min"].transform("sum")
    # Safer team abbreviation lookup
    team_col = None
    for c in spine.columns:
        if "team_abbreviation" in c.lower() and "rank" not in c.lower():
            team_col = c
            break
    if team_col:
        min_col = "min" if "min" in spine.columns else "min"
        team_min_sum = spine.groupby(["season", team_col])[min_col].transform("sum")
        spine["agg_minute_share"] = mins / team_min_sum.replace(0, np.nan)

    # ── Player tier (BKE-anchored, within-season) ──────────────────────
    # Tiers: Superstar / All-Star / Starter / Rotation Player / Reserve / Fringe
    # Percentiles computed within each season among BKE-scored players only.
    _score_col = 'total_impact_score' if 'total_impact_score' in spine.columns else None
    _min_col = 'min' if 'min' in spine.columns else ('min' if 'min' in spine.columns else None)
    if _score_col and _min_col:
        _mins = pd.to_numeric(spine[_min_col], errors='coerce').fillna(0)
        _scores = pd.to_numeric(spine[_score_col], errors='coerce')
        _pctl = spine.groupby('season')[_score_col].rank(pct=True, na_option='keep')
        has_score = _scores.notna()
        _tier = pd.Series('Fringe', index=spine.index, dtype=object)
        _tier.loc[~has_score & (_mins >= 200)] = 'Reserve'
        _tier.loc[has_score & (_mins >= 200)] = 'Reserve'
        _tier.loc[has_score & (_mins >= 300) & (_pctl >= 0.30)] = 'Rotation Player'
        _tier.loc[has_score & (_mins >= 600) & (_pctl >= 0.55)] = 'Starter'
        _tier.loc[has_score & (_mins >= 600) & (_pctl >= 0.82)] = 'All-Star'
        _tier.loc[has_score & (_mins >= 800) & (_pctl >= 0.95)] = 'Superstar'
        spine['player_tier'] = _tier
        print(f"  player_tier distribution:\n{spine['player_tier'].value_counts().to_string()}")

    # ── Final dedup & sort ─────────────────────────────────────────────
    spine = _dedup(spine)
    spine = spine.sort_values(["season", "player_id"]).reset_index(drop=True)
    spine = _normalize_column_names(spine)

    # ── Save ───────────────────────────────────────────────────────────
    AGGREGATE_DIR.mkdir(parents=True, exist_ok=True)
    save_standardized(spine, PROFILE_AGGREGATE_PATH)
    print(f"\nSaved aggregate: {PROFILE_AGGREGATE_PATH}")
    print(f"  rows={len(spine)}, cols={len(spine.columns)}")

    # ── Validation report ──────────────────────────────────────────────
    report = {
        "rows": int(len(spine)),
        "columns": int(len(spine.columns)),
        "seasons": sorted(spine["season"].unique().tolist()),
        "unique_players": int(spine["player_id"].nunique()),
        "column_groups": {
            "bke_decomp": int(len([c for c in spine.columns if c.startswith("dim_") or c.startswith("portable_") or c.startswith("elevation_")])),
            "box_stats": int(len([c for c in spine.columns if c.startswith("box_")])),
            "archetypes": int(len([c for c in spine.columns if "archetype" in c.lower() or c.startswith("emb_")])),
            "defensive": int(len([c for c in spine.columns if c.startswith("defarche_") or "defensive" in c.lower()])),
            "bke_json": int(len([c for c in spine.columns if c.startswith("bke_")])),
            "game_logs": int(len([c for c in spine.columns if c.startswith("gl_")])),
            "clutch": int(len([c for c in spine.columns if c.startswith("clutch_")])),
            "draft": int(len([c for c in spine.columns if c.startswith("draft_") or c in {
                "draft_class_year", "draft_round", "draft_pick_in_round", "draft_pick_overall",
                "draft_tier", "is_drafted", "is_undrafted"
            }])),
            "step1_pec": int(len([c for c in spine.columns if c.startswith("pec_")])),
        },
        "coverage": {
            "salary": float(spine.get("salary", pd.Series(dtype=float)).notna().mean()),
            "game_logs": float(spine.get("gl_games_total", pd.Series(dtype=float)).notna().mean()),
            "clutch_minutes": float(spine.get("clutch_minutes", pd.Series(dtype=float)).notna().mean()),
            "bio_height": float(spine.get("height_inches", pd.Series(dtype=float)).notna().mean()),
            "draft_class_year": float(spine.get("draft_class_year", pd.Series(dtype=float)).notna().mean()),
            "mpg": float(spine.get("agg_mpg", pd.Series(dtype=float)).notna().mean()),
        },
        "sample_columns": sorted(spine.columns.tolist())[:50],
    }
    AGGREGATE_VALIDATION_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved validation: {AGGREGATE_VALIDATION_REPORT}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
