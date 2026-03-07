"""
app/player_data_viewer.py
=============================================================================
Optimised player data browser with full-stats modal popups.
Lightweight cards for fast rendering; click any card for all details.
Generates app/player_data.html.
=============================================================================
"""

import pandas as pd
import numpy as np
import json
import os
import re

PROCESSED_DIR = "data/processed"
HISTORICAL_DIR = "data/historical"
OUTPUT_FILE = "app/player_data.html"

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_archetypes():
    off = pd.read_parquet(f"{PROCESSED_DIR}/player_archetypes.parquet")
    defense = pd.read_parquet(f"{PROCESSED_DIR}/defensive_archetypes_v2.parquet")
    off['player_id'] = off['PLAYER_ID'].astype(str)
    defense['player_id'] = defense['PLAYER_ID'].astype(str)
    return off, defense

def _load_parquet(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if df.empty:
        return df
    df = df.copy()
    if "PLAYER_ID" in df.columns and "player_id" not in df.columns:
        df["player_id"] = df["PLAYER_ID"].astype(str)
    if "SEASON" in df.columns:
        df["SEASON"] = df["SEASON"].astype(str)
    if "player_id" in df.columns:
        df["player_id"] = df["player_id"].astype(str)
    if "season" in df.columns:
        df["SEASON"] = df["season"].astype(str)
    return df

def load_player_profiles():
    return _load_parquet(f"{PROCESSED_DIR}/player_profiles_advanced.parquet")

def load_player_bios():
    return _load_parquet(f"{HISTORICAL_DIR}/players.parquet")

def load_position_estimates():
    files = []
    try:
        files = sorted([
            os.path.join(PROCESSED_DIR, f)
            for f in os.listdir(PROCESSED_DIR)
            if f.startswith("player_position_estimates_") and f.endswith(".parquet")
        ])
    except Exception:
        files = []

    if files:
        frames = []
        for path in files:
            df = _load_parquet(path)
            if df.empty:
                continue
            if "SEASON" not in df.columns:
                season = os.path.basename(path).replace("player_position_estimates_", "").replace(".parquet", "")
                df["SEASON"] = season
            frames.append(df)
        if frames:
            return pd.concat(frames, ignore_index=True)

    # Backward compatibility with legacy single-file output
    return _load_parquet(f"{PROCESSED_DIR}/player_position_estimates.parquet")

def load_rapm():
    return _load_parquet(f"{PROCESSED_DIR}/player_rapm.parquet")

def load_xrapm():
    return _load_parquet(f"{PROCESSED_DIR}/player_xrapm.parquet")

def load_xrapm_v2():
    return _load_parquet(f"{PROCESSED_DIR}/player_xrapm_v2.parquet")

def load_linear_metrics():
    return _load_parquet(f"{PROCESSED_DIR}/metrics_linear.parquet")

def load_bke_scores():
    path = f"{PROCESSED_DIR}/bke/BKE_Scores_v27.json"
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {}

    players = payload.get("players", {}) if isinstance(payload, dict) else {}
    index = {}
    for _, rec in players.items():
        if not isinstance(rec, dict):
            continue
        pid = normalize_player_id(rec.get("player_id"))
        season = str(rec.get("season", "")).strip()
        if not pid or not season:
            continue
        index[(pid, season)] = rec
    return index

def load_bke_v28_scores():
    """Load v28 OBKE/DBKE scores, layer scores, and dimension scores.

    Returns dict keyed by (player_id, season) with composite BKE,
    transformed OBKE/DBKE, rank, and all layer/dimension z-scores.
    """
    from collections import defaultdict

    def _load_json(path):
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    obke_dbke = _load_json(f"{PROCESSED_DIR}/bke/obke_dbke_scores_v28.json")
    layers    = _load_json(f"{PROCESSED_DIR}/bke/layer_scores_v28.json")
    dims      = _load_json(f"{PROCESSED_DIR}/bke/dimension_scores_v28.json")

    records = []
    for _, rec in obke_dbke.get("players", {}).items():
        if not isinstance(rec, dict):
            continue
        pid = normalize_player_id(rec.get("player_id"))
        season = str(rec.get("season", "")).strip()
        if not pid or not season:
            continue
        obke_t = rec.get("obke_transformed")
        dbke_t = rec.get("dbke_transformed")
        bke = round(0.60 * obke_t + 0.40 * dbke_t, 2) if obke_t is not None and dbke_t is not None else None
        records.append({
            "player_id": pid, "season": season,
            "obke_transformed": obke_t, "dbke_transformed": dbke_t,
            "bke_composite": bke,
            "portable_talent_z_adj": rec.get("portable_talent_z_adj"),
            "offensive_portable_z": rec.get("offensive_portable_z"),
            "defensive_portable_z": rec.get("defensive_portable_z"),
        })

    # Rank per season
    by_season = defaultdict(list)
    for r in records:
        if r["bke_composite"] is not None:
            by_season[r["season"]].append(r)
    for _, players in by_season.items():
        n = len(players)
        players.sort(key=lambda x: x["bke_composite"], reverse=True)
        for i, p in enumerate(players):
            p["rank"] = i + 1
            p["pct"] = p["bke_composite"]  # already on 0-100 percentile-like scale
        # OBKE percentile
        players.sort(key=lambda x: (x["obke_transformed"] or 0), reverse=True)
        for i, p in enumerate(players):
            p["obke_pct"] = round((1 - i / max(n - 1, 1)) * 100, 1)
        # DBKE percentile
        players.sort(key=lambda x: (x["dbke_transformed"] or 0), reverse=True)
        for i, p in enumerate(players):
            p["dbke_pct"] = round((1 - i / max(n - 1, 1)) * 100, 1)

    # Merge layer scores
    layer_idx = {}
    for _, rec in layers.get("players", {}).items():
        if not isinstance(rec, dict):
            continue
        pid = normalize_player_id(rec.get("player_id"))
        season = str(rec.get("season", "")).strip()
        if pid and season:
            layer_idx[(pid, season)] = {k: v for k, v in rec.items() if k not in ("player_id", "season")}

    # Merge dimension scores
    dim_idx = {}
    for _, rec in dims.get("players", {}).items():
        if not isinstance(rec, dict):
            continue
        pid = normalize_player_id(rec.get("player_id"))
        season = str(rec.get("season", "")).strip()
        if pid and season:
            dim_idx[(pid, season)] = {k: v for k, v in rec.items() if k not in ("player_id", "season")}

    index = {}
    for r in records:
        key = (r["player_id"], r["season"])
        r["layer_scores"] = layer_idx.get(key, {})
        r["dimension_scores"] = dim_idx.get(key, {})
        index[key] = r
    return index

def load_bke_v30_scores():
    """Load v3.0 defense shrinkage scores and compute percentiles.

    Returns dict keyed by (player_id, season) with BKE_v30, DBKE_scaled_v30,
    OBKE_v30_reference, intermediate fields, and computed percentiles/ranks.
    """
    from collections import defaultdict

    path = "reports/dbke_v30_defense_shrinkage.json"
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {}

    player_outputs = payload.get("player_outputs", [])
    records = []
    for rec in player_outputs:
        pid = normalize_player_id(rec.get("player_id"))
        season = str(rec.get("season", "")).strip()
        if not pid or not season:
            continue
        records.append({
            "player_id": pid, "season": season,
            "player_name": rec.get("player_name"),
            "BKE_v30": rec.get("BKE_v30"),
            "DBKE_scaled_v30": rec.get("DBKE_scaled_v30"),
            "OBKE_v30_reference": rec.get("OBKE_v30_reference"),
            "DBKE_raw_v30": rec.get("DBKE_raw_v30"),
            "DBKE_final_v30": rec.get("DBKE_final_v30"),
            "DBKE_asym_v30": rec.get("DBKE_asym_v30"),
            "DBKE_norm_v30": rec.get("DBKE_norm_v30"),
            "D_port": rec.get("D_port"),
            "D_rapm_shrunk": rec.get("D_rapm_shrunk"),
            "D_stabilized": rec.get("D_stabilized"),
            "lambda_shrink": rec.get("lambda_shrink"),
        })

    # Compute percentile ranks per season
    by_season = defaultdict(list)
    for r in records:
        if r["BKE_v30"] is not None:
            by_season[r["season"]].append(r)
    for _, players in by_season.items():
        n = len(players)
        # BKE rank + percentile
        players.sort(key=lambda x: x["BKE_v30"], reverse=True)
        for i, p in enumerate(players):
            p["rank"] = i + 1
            p["bke_pct"] = round((1 - i / max(n - 1, 1)) * 100, 1)
        # DBKE percentile
        players.sort(key=lambda x: (x["DBKE_scaled_v30"] or 0), reverse=True)
        for i, p in enumerate(players):
            p["dbke_pct"] = round((1 - i / max(n - 1, 1)) * 100, 1)
        # OBKE percentile
        players.sort(key=lambda x: (x["OBKE_v30_reference"] or 0), reverse=True)
        for i, p in enumerate(players):
            p["obke_pct"] = round((1 - i / max(n - 1, 1)) * 100, 1)

    index = {}
    for r in records:
        index[(r["player_id"], r["season"])] = r
    return index

def load_player_season_stats():
    path = f"{HISTORICAL_DIR}/complete_player_season_stats.parquet"
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if df.empty:
        return df
    df = df.copy()
    if "PLAYER_ID" in df.columns:
        df["player_id"] = df["PLAYER_ID"].astype(str)
    if "SEASON" in df.columns:
        df["SEASON"] = df["SEASON"].astype(str)
    return df

def load_player_salaries():
    files = []
    try:
        files = sorted([
            os.path.join(HISTORICAL_DIR, f)
            for f in os.listdir(HISTORICAL_DIR)
            if re.match(r"player_salaries_\d{4}-\d{2}\.parquet$", f)
        ])
    except Exception:
        files = []

    if files:
        frames = []
        for path in files:
            df = _load_parquet(path)
            if df.empty:
                continue
            if "SEASON" not in df.columns:
                season = os.path.basename(path).replace("player_salaries_", "").replace(".parquet", "")
                df["SEASON"] = season
            frames.append(df)
        if frames:
            return pd.concat(frames, ignore_index=True)

    return _load_parquet(f"{HISTORICAL_DIR}/player_salaries.parquet")

def load_team_map():
    path = f"{HISTORICAL_DIR}/teams.parquet"
    if not os.path.exists(path):
        return {}
    teams = pd.read_parquet(path)
    if teams.empty:
        return {}
    cols = {c.lower(): c for c in teams.columns}
    tid_col = cols.get("team_id") or cols.get("id")
    abbr_col = cols.get("abbreviation")
    if not tid_col or not abbr_col:
        return {}
    mapping = {}
    for _, row in teams[[tid_col, abbr_col]].dropna().iterrows():
        try:
            tid = str(int(float(row[tid_col])))
        except (ValueError, TypeError):
            tid = str(row[tid_col]).strip()
        abbr = str(row[abbr_col]).strip().upper()
        if tid and abbr:
            mapping[tid] = abbr
    return mapping

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_height(inches):
    if inches is None or (isinstance(inches, float) and pd.isna(inches)):
        return None
    try:
        total = int(round(float(inches)))
    except (ValueError, TypeError):
        return None
    return f"{total // 12}'{total % 12}\""

def clean_value(val):
    if val is None:
        return None
    if isinstance(val, float) and pd.isna(val):
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    if isinstance(val, (pd.Timestamp,)):
        return val.isoformat()
    return val

def normalize_player_id(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        return str(int(float(val)))
    except (ValueError, TypeError):
        s = str(val).strip()
        if s.endswith('.0'):
            s = s[:-2]
        return s or None


def _safe_int(val, default=1):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return default
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return default


def _safe_bool(val, default=False):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return default
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, str):
        text = val.strip().lower()
        if text in {"1", "true", "t", "yes", "y"}:
            return True
        if text in {"0", "false", "f", "no", "n"}:
            return False
    try:
        return bool(int(float(val)))
    except (TypeError, ValueError):
        return default


def _clean_team_abbreviation(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    text = str(val).strip().upper()
    if not text:
        return ""
    if text in {"NAN", "NONE", "NULL"}:
        return ""
    return text


def _build_player_stint_key(player_id, season, team_abbr, stint_number):
    team = team_abbr or "UNK"
    stint = max(1, _safe_int(stint_number, default=1))
    return f"{player_id}::{season}::{team}::{stint}"

def row_to_dict(row, exclude=None):
    exclude = set(exclude or [])
    return {k: clean_value(row.get(k)) for k in row.index if k not in exclude}

def build_record_map(df, key_cols, drop_cols=None):
    if df is None or df.empty:
        return {}
    drop_cols = set(drop_cols or [])
    cols = [c for c in df.columns if c not in drop_cols]
    temp = df[cols].copy()
    for k in key_cols:
        if k not in temp.columns:
            return {}
    temp = temp.drop_duplicates(subset=key_cols).set_index(key_cols)
    records = temp.to_dict(orient="index")
    return {k: {rk: clean_value(rv) for rk, rv in rec.items()} for k, rec in records.items()}

def _format_pct(val, scale=100):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    result = val * scale if abs(val) < 1.5 else val
    return round(result, 1)


def _safe_float(val):
    if val is None:
        return None
    try:
        out = float(val)
    except (TypeError, ValueError):
        return None
    if pd.isna(out):
        return None
    return out


def apply_bke_split_variants(cards, details):
    """Attach 60/40 + 55/45 split variants and per-season rank/pct to each card/version.

    This keeps historical behavior with 60/40 as default while making 55/45
    available instantly in the UI without requiring additional backend score files.
    """
    split_defs = {
        "split_60_40": {"off_weight": 0.60, "def_weight": 0.40},
        "split_55_45": {"off_weight": 0.55, "def_weight": 0.45},
    }

    buckets = {}
    for card in cards:
        season = str(card.get("season", "")).strip()
        versions = card.get("bke_versions") or {}
        for version_name, version_payload in versions.items():
            obke = _safe_float(version_payload.get("obke"))
            dbke = _safe_float(version_payload.get("dbke"))
            splits = {}
            for split_name, coeffs in split_defs.items():
                bke_val = None
                if obke is not None and dbke is not None:
                    bke_val = round(coeffs["off_weight"] * obke + coeffs["def_weight"] * dbke, 3)
                splits[split_name] = {
                    "off_weight": coeffs["off_weight"],
                    "def_weight": coeffs["def_weight"],
                    "bke": bke_val,
                    "rank": None,
                    "pct": None,
                }
                key = (version_name, split_name, season)
                buckets.setdefault(key, [])
                if bke_val is not None:
                    buckets[key].append((card, splits[split_name]))

            version_payload["splits"] = splits
            baseline = splits["split_60_40"]
            version_payload["bke"] = baseline.get("bke")
            version_payload["rank"] = baseline.get("rank")
            version_payload["pct"] = baseline.get("pct")

    for _, entries in buckets.items():
        if not entries:
            continue
        entries.sort(key=lambda pair: pair[1]["bke"], reverse=True)
        n = len(entries)
        for i, (_, split_payload) in enumerate(entries):
            split_payload["rank"] = i + 1
            split_payload["pct"] = round((1 - i / max(n - 1, 1)) * 100, 1)

    for card in cards:
        key = card.get("key")
        detail = details.get(key)
        if not isinstance(detail, dict):
            continue

        version_splits = card.get("bke_versions") or {}
        v27 = (version_splits.get("v27") or {}).get("splits")
        v28 = (version_splits.get("v28") or {}).get("splits")
        v30 = (version_splits.get("v30") or {}).get("splits")

        for section_name in ("bke", "bke_details"):
            section = detail.get(section_name)
            if isinstance(section, dict) and v27:
                section["split_variants"] = v27
                section["active_split_default"] = "split_60_40"

        section = detail.get("bke_v28")
        if isinstance(section, dict) and v28:
            section["split_variants"] = v28
            section["active_split_default"] = "split_60_40"

        section = detail.get("bke_v30")
        if isinstance(section, dict) and v30:
            section["split_variants"] = v30
            section["active_split_default"] = "split_60_40"

# ---------------------------------------------------------------------------
# Archetype reason generators (same logic as player_archetype_viewer.py)
# ---------------------------------------------------------------------------

def get_offensive_reasons(row):
    reasons = []
    arch = row.get('primary_archetype', 'Unknown')
    sub = row.get('secondary_archetype', '')
    ball_dom = row.get('BALL_DOMINANT_PCT', 0)
    playmaking = row.get('PLAYMAKING_SCORE', 0)
    pts36 = row.get('PTS_PER36', 0)
    fg2a_rate = row.get('FG2A_RATE', 0)
    ast36 = row.get('AST_PER36', 0)
    eff_tier = row.get('efficiency_tier', '')

    if arch == 'Ball Dominant Creator':
        iso_pct = row.get('ISOLATION_POSS_PCT', 0)
        pnr_pct = row.get('PRBALLHANDLER_POSS_PCT', 0)
        post_pct = row.get('POSTUP_POSS_PCT', 0)
        reasons.append(f"Ball dominance: {ball_dom*100:.0f}% (ISO {iso_pct*100:.0f}% + PnR BH {pnr_pct*100:.0f}%)")
        if post_pct > 0.05:
            reasons.append(f"Post-up: {post_pct*100:.0f}%")
        reasons.append(f"Playmaking: {playmaking:.1f} | AST/36: {ast36:.1f}")
        reasons.append(f"Scoring: {pts36:.1f} PTS/36 | Efficiency: {eff_tier}")
        subs = {'Offensive Hub': "Offensive hub — high playmaking", 'Primary Scorer': "Primary scorer — high volume on-ball creation",
                'Post Hub': f"Post hub — post-up {row.get('POSTUP_POSS_PCT',0)*100:.0f}%",
                'Midrange Scorer': f"Midrange-heavy: {row.get('MIDRANGE_FREQ',0)*100:.1f}%",
                'Inside-the-Arc': f"Inside-the-arc: {row.get('MIDRANGE_FREQ',0)*100:.1f}%"}
        if sub in subs:
            reasons.append(subs[sub])
    elif arch == 'Ballhandler':
        reasons += [f"High playmaking ({playmaking:.1f}) with AST/36: {ast36:.1f}", f"Ball dominance: {ball_dom*100:.0f}%", f"Not primary scorer ({pts36:.1f} PTS/36)", "Facilitator role"]
    elif arch == 'All-Around Scorer':
        reasons += [f"Balanced 2PT/3PT (FG2A rate: {fg2a_rate*100:.0f}%)", f"Scoring: {pts36:.1f} PTS/36 | Efficiency: {eff_tier}", f"Ball dominance: {ball_dom*100:.0f}%"]
        if sub == 'High Volume': reasons.append("High volume scoring")
        elif sub in ('Midrange Scorer', 'Inside-the-Arc'): reasons.append(f"Midrange: {row.get('MIDRANGE_FREQ',0)*100:.1f}%")
    elif arch == 'Perimeter Scorer':
        reasons += [f"Perimeter-heavy: {row.get('FG3A_PER36',0):.1f} 3PA/36", f"3P%: {(row.get('FG3_PCT',0)*100 if row.get('FG3_PCT',0)<1 else row.get('FG3_PCT',0)):.1f}%", f"Scoring: {pts36:.1f} PTS/36"]
    elif arch == 'Interior Scorer':
        reasons += [f"Interior-heavy: FG2A rate {fg2a_rate*100:.0f}%", f"At-rim {row.get('AT_RIM_FREQ',0)*100:.1f}% | Midrange {row.get('MIDRANGE_FREQ',0)*100:.1f}%", f"Scoring: {pts36:.1f} PTS/36 | Efficiency: {eff_tier}"]
        if sub == 'Midrange Scorer': reasons.append("Midrange-heavy profile")
        elif sub == 'Inside-the-Arc': reasons.append("Inside-the-arc scorer")
        elif sub == 'Rim Finisher': reasons.append("Rim-dominant")
    elif arch == 'Connector':
        reasons += [f"Playmaker without volume scoring (AST/36: {ast36:.1f})", f"Scoring: {pts36:.1f} PTS/36", "Distributes and facilitates"]
    elif arch == 'Off-Ball Finisher':
        reasons += [f"Cuts: {row.get('CUT_POSS_PCT',0)*100:.1f}%, PnR Roll: {row.get('PRROLLMAN_POSS_PCT',0)*100:.1f}%", f"Low ball dominance ({ball_dom*100:.0f}%)", f"Efficiency: {eff_tier}"]
    elif arch == 'Off-Ball Movement Shooter':
        reasons += [f"Handoff: {row.get('HANDOFF_POSS_PCT',0)*100:.1f}%, Off-Screen: {row.get('OFFSCREEN_POSS_PCT',0)*100:.1f}%", "Movement-based shooting", f"Efficiency: {eff_tier}"]
    elif arch == 'Off-Ball Stationary Shooter':
        reasons += [f"Spot-up: {row.get('SPOTUP_POSS_PCT',0)*100:.1f}%", f"C&S 3P%: {row.get('CATCH_SHOOT_FG3_PCT',0)*100:.1f}%", f"Efficiency: {eff_tier}"]
    elif arch == 'PnR Rolling Big':
        reasons += [f"PnR Roll Man: {row.get('PRROLLMAN_POSS_PCT',0)*100:.1f}%", f"At-rim: {row.get('AT_RIM_FREQ',0)*100:.1f}%"]
    elif arch == 'PnR Popping Big':
        reasons += ["PnR with perimeter shooting", f"3PA/36: {row.get('FG3A_PER36',0):.1f}"]
    elif arch == 'Insufficient Minutes':
        reasons.append(f"Only {row.get('MIN',0):.0f} minutes")
    return reasons


def get_defensive_reasons(row):
    reasons = []
    arch = row.get('defensive_archetype', 'Unknown')
    sec = row.get('defensive_secondary', '')
    vpctl = row.get('versatility_pctl', 0) or 0
    dpctl = row.get('difficulty_pctl', 0) or 0
    opp = row.get('avg_opponent_ppg', 0) or 0
    elite = row.get('elite_matchup_pct', 0) or 0
    blk_pct = row.get('BLK_PCT', 0) or 0
    stl_p100 = row.get('STL_PER100_DEF_POSS', 0) or 0
    dres = row.get('d_results_pctl', 0) or 0
    eng = row.get('engagement_pctl', 0) or 0
    rim_fg = row.get('DEF_RIM_FG_PCT', 0) or 0
    hustle = row.get('hustle_pctl', 0) or 0

    if arch == 'POA Defender':
        reasons.append(f"Tough assignments ({dpctl*100:.0f}th pctl difficulty)")
        reasons.append(f"Elite matchup share: {elite*100:.1f}%")
        reasons.append(f"Avg opponent PPG: {opp:.1f}")
        if sec == 'Ball Hawk': reasons.append(f"High steals ({stl_p100:.1f} per 100 def poss)")
        elif sec == 'Switchable': reasons.append("Versatile — switches across positions")
        elif sec == 'Hustler': reasons.append("High hustle / effort metrics")
    elif arch == 'Wing Stopper':
        reasons.append("Forward/wing assignment specialist")
        reasons.append(f"Matchup difficulty: {dpctl*100:.0f}th pctl")
        reasons.append(f"Defensive results: {dres*100:.0f}th pctl")
        if sec == 'Lockdown': reasons.append("Elite defensive results")
        elif sec == 'Hustler': reasons.append("High hustle metrics")
    elif arch == 'Off-Ball Chaser':
        reasons.append(f"High steals: {stl_p100:.1f} per 100 def poss")
        reasons.append("Lower matchup difficulty (roaming/off-ball)")
        if sec == 'Ball Hawk': reasons.append("Elite steal rate")
        elif sec == 'Hustler': reasons.append("High hustle metrics")
    elif arch == 'Rim Protector':
        reasons.append(f"Block rate: {blk_pct:.1%}")
        reasons.append(f"Opponent rim FG%: {rim_fg:.1%}")
        if sec == 'Switchable': reasons.append("Can also switch on perimeter")
        elif sec == 'Shot Blocker': reasons.append("Elite shot blocking volume")
        else: reasons.append("Interior anchor")
    elif arch == 'Dropping Big':
        reasons.append("Interior-focused, stays in paint")
        reasons.append(f"Low versatility ({vpctl*100:.0f}th pctl)")
        reasons.append("Strong rebounding / rim presence")
    elif arch == 'Mobile Big':
        reasons.append(f"High versatility ({vpctl*100:.0f}th pctl)")
        reasons.append("Switches across positions as a big")
        reasons.append(f"Defensive results: {dres*100:.0f}th pctl")
        if sec == 'Lockdown': reasons.append("Strong results across matchup types")
        elif sec == 'Hustler': reasons.append("High hustle metrics")
    elif arch == 'Low-Activity Defender':
        reasons.append(f"Low engagement ({eng*100:.0f}th pctl)")
        if sec == 'Liability': reasons.append("Poor defensive results")
        else: reasons.append("Hidden on defense / low involvement")
        reasons.append(f"Defensive results: {dres*100:.0f}th pctl")
    elif arch == 'Insufficient Minutes':
        reasons.append(f"Only {row.get('MIN', 0):.0f} minutes")
        reasons.append("Not enough data")
    return reasons


def get_playtype_rankings(row):
    playtype_cols = [
        'ISOLATION_POSS_PCT', 'PRBALLHANDLER_POSS_PCT', 'POSTUP_POSS_PCT',
        'CUT_POSS_PCT', 'PRROLLMAN_POSS_PCT', 'HANDOFF_POSS_PCT',
        'OFFSCREEN_POSS_PCT', 'SPOTUP_POSS_PCT', 'TRANSITION_POSS_PCT',
        'OFFREBOUND_POSS_PCT', 'MISC_POSS_PCT'
    ]
    pretty = {
        'ISOLATION_POSS_PCT': 'Isolation', 'PRBALLHANDLER_POSS_PCT': 'PnR Ball Handler',
        'POSTUP_POSS_PCT': 'Post Up', 'CUT_POSS_PCT': 'Cut',
        'PRROLLMAN_POSS_PCT': 'PnR Roll Man', 'HANDOFF_POSS_PCT': 'Handoff',
        'OFFSCREEN_POSS_PCT': 'Off Screen', 'SPOTUP_POSS_PCT': 'Spot Up',
        'TRANSITION_POSS_PCT': 'Transition', 'OFFREBOUND_POSS_PCT': 'Putback',
        'MISC_POSS_PCT': 'Misc',
    }
    pairs = []
    for c in playtype_cols:
        if c in row.index:
            val = row.get(c, 0) or 0
            pct = val * 100 if val < 1 else val
            if pct > 0:
                pairs.append((pretty.get(c, c), pct))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return [{'type': p[0], 'pct': round(p[1], 2)} for p in pairs]

# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def generate_html(off_df, def_df, profiles_df, bios_df, pos_est_df, rapm_df, xrapm_df, xrapm_v2_df, linear_df, season_stats_df, salaries_df, team_map, bke_map, bke_v28_map=None, bke_v30_map=None):
    """Generate optimised HTML viewer with lazy-loaded detail data."""
    bke_v28_map = bke_v28_map or {}
    bke_v30_map = bke_v30_map or {}

    off_work = off_df.copy()
    def_work = def_df.copy()

    # Carry canonical team/stint join fields when available so traded seasons
    # do not cross-join offense and defense rows.
    if "TEAM_ABBREVIATION" not in off_work.columns:
        if "team_abbreviation" in off_work.columns:
            off_work["TEAM_ABBREVIATION"] = off_work["team_abbreviation"]
        elif "team" in off_work.columns:
            off_work["TEAM_ABBREVIATION"] = off_work["team"]
    if "TEAM_ABBREVIATION" in off_work.columns:
        off_work["TEAM_ABBREVIATION"] = off_work["TEAM_ABBREVIATION"].map(_clean_team_abbreviation)

    if "TEAM_ABBREVIATION" not in def_work.columns:
        if "team_abbreviation" in def_work.columns:
            def_work["TEAM_ABBREVIATION"] = def_work["team_abbreviation"]
        elif "team" in def_work.columns:
            def_work["TEAM_ABBREVIATION"] = def_work["team"]
    if "TEAM_ABBREVIATION" in def_work.columns:
        def_work["TEAM_ABBREVIATION"] = def_work["TEAM_ABBREVIATION"].map(_clean_team_abbreviation)

    off_stint_col = "stint_number" if "stint_number" in off_work.columns else ("team_stint_number" if "team_stint_number" in off_work.columns else None)
    def_stint_col = "stint_number" if "stint_number" in def_work.columns else ("team_stint_number" if "team_stint_number" in def_work.columns else None)
    if off_stint_col:
        off_work["_stint_number_join"] = pd.to_numeric(off_work[off_stint_col], errors="coerce").fillna(1).astype(int)
    if def_stint_col:
        def_work["_stint_number_join"] = pd.to_numeric(def_work[def_stint_col], errors="coerce").fillna(1).astype(int)

    join_keys = ["player_id", "SEASON"]
    if "TEAM_ABBREVIATION" in off_work.columns and "TEAM_ABBREVIATION" in def_work.columns:
        join_keys.append("TEAM_ABBREVIATION")
    if "_stint_number_join" in off_work.columns and "_stint_number_join" in def_work.columns:
        join_keys.append("_stint_number_join")

    def_cols = ['player_id', 'SEASON', 'defensive_archetype', 'defensive_secondary',
                'defensive_confidence', 'assignment_difficulty',
                'switch_score', 'versatility_pctl', 'avg_opponent_ppg',
                'elite_matchup_pct', 'difficulty_pctl',
                'STL_PER100_DEF_POSS', 'stl_pctl', 'BLK_PCT', 'blk_pctl',
                'DEF_RIM_FG_PCT', 'RIM_FGA_RATE',
                'engagement_score', 'engagement_pctl',
                'hustle_score', 'hustle_pctl',
                'D_FG_DIFF', 'd_results_pctl',
                'poa_score', 'wing_score', 'chaser_score',
                'rim_score', 'drop_big_score', 'mobile_big_score']
    def_cols = [c for c in def_cols if c in def_work.columns]
    for key_col in join_keys:
        if key_col in def_work.columns and key_col not in def_cols:
            def_cols.append(key_col)

    def_for_merge = def_work[def_cols].drop_duplicates(subset=join_keys, keep="first")
    merged = off_work.merge(def_for_merge, on=join_keys, how='left')

    profiles_map = build_record_map(profiles_df, ["player_id", "SEASON"], ["player_name", "season"])
    rapm_map     = build_record_map(rapm_df,     ["player_id", "SEASON"], ["player_name", "season"])
    xrapm_map    = build_record_map(xrapm_df,    ["player_id", "SEASON"], ["player_name", "season"])
    xrapm_v2_map = build_record_map(xrapm_v2_df, ["player_id", "SEASON"], ["season"])
    linear_map   = build_record_map(linear_df,   ["player_id", "SEASON"], ["player_name", "season"])
    season_stats_map = build_record_map(season_stats_df, ["player_id", "SEASON"], ["PLAYER_NAME", "TEAM_ABBREVIATION", "season"])
    pos_est_map = build_record_map(pos_est_df, ["player_id", "SEASON"], [])

    salary_map = {}
    if salaries_df is not None and not salaries_df.empty:
        temp = salaries_df.copy()
        if "player_id" not in temp.columns and "PERSON_ID" in temp.columns:
            temp["player_id"] = temp["PERSON_ID"]
        if "SEASON" not in temp.columns and "season" in temp.columns:
            temp["SEASON"] = temp["season"].astype(str)
        if "player_id" in temp.columns:
            temp["player_id"] = temp["player_id"].apply(normalize_player_id)
        if "SEASON" in temp.columns:
            temp["SEASON"] = temp["SEASON"].astype(str)
        if all(c in temp.columns for c in ["player_id", "SEASON", "salary"]):
            for _, r in temp[["player_id", "SEASON", "salary"]].dropna(subset=["player_id", "SEASON"]).iterrows():
                salary_map[(str(r["player_id"]), str(r["SEASON"]))] = clean_value(r.get("salary"))

    bios_map = {}
    if bios_df is not None and not bios_df.empty:
        bios_df = bios_df.copy()
        bios_df["player_id"] = bios_df["player_id"].astype(str)
        for _, r in bios_df.iterrows():
            pid = str(r.get("player_id"))
            tid = str(int(float(r.get("team_id")))) if pd.notna(r.get("team_id")) else None
            bios_map[pid] = {
                "full_name": r.get("full_name"), "position": r.get("primary_position"),
                "age": clean_value(r.get("age")), "height": format_height(r.get("height_inches")),
                "height_inches": clean_value(r.get("height_inches")),
                "weight_lbs": clean_value(r.get("weight_lbs")),
                "wingspan_inches": clean_value(r.get("wingspan_inches")),
                "experience_years": clean_value(r.get("experience_years")),
                "team_id": tid, "team_abbrev": team_map.get(tid) if tid else None,
            }

    # ---- Build two data structures: cards (small) + details (large, lazy) ----
    cards = []       # lightweight – rendered in grid
    details = {}     # keyed by player key – parsed on first modal open

    for _, row in merged.iterrows():
        if row.get('primary_archetype') == 'Insufficient Minutes':
            continue
        pid, season = str(row['player_id']), row['SEASON']
        team_abbr = _clean_team_abbreviation(
            row.get("TEAM_ABBREVIATION")
            or row.get("team_abbreviation")
            or row.get("team")
        )
        stint_number = max(
            1,
            _safe_int(
                row.get("stint_number", row.get("team_stint_number", row.get("_stint_number_join", 1))),
                default=1,
            ),
        )
        stint_count = max(1, _safe_int(row.get("stint_count"), default=1))
        stint_team_count = max(1, _safe_int(row.get("stint_team_count"), default=1))
        is_primary_stint = _safe_bool(row.get("is_primary_stint"), default=(stint_number == 1))
        is_final_stint = _safe_bool(row.get("is_final_stint"), default=(stint_number == stint_count))
        team_assignment_source = str(row.get("team_assignment_source", "") or "").strip() or "season"
        key = _build_player_stint_key(pid, season, team_abbr, stint_number)
        pos_est = pos_est_map.get((pid, season), {})
        bke_rec = bke_map.get((pid, season), {}) or {}

        card = {
            'key': key,
            'id': pid,
            'name': row['PLAYER_NAME'],
            'season': season,
            'team': team_abbr,
            'stint_number': stint_number,
            'stint_count': stint_count,
            'stint_team_count': stint_team_count,
            'is_primary_stint': is_primary_stint,
            'is_final_stint': is_final_stint,
            'team_assignment_source': team_assignment_source,
            'off_archetype': row.get('primary_archetype', 'Unknown'),
            'off_secondary': row.get('secondary_archetype', ''),
            'off_confidence': round(row.get('role_confidence', row.get('archetype_confidence', 0)) * 100),
            'off_effectiveness': round(row.get('role_effectiveness', 0) * 100),
            'def_archetype': row.get('defensive_archetype', 'Unknown'),
            'def_confidence': round(row.get('defensive_confidence', 0) * 100) if pd.notna(row.get('defensive_confidence')) else 0,
            'ppg': round(row.get('PPG', row.get('PTS', 0) / max(row.get('GP', 1), 1)), 1),
            'apg': round(row.get('AST', 0) / max(row.get('GP', 1), 1), 1),
            'rpg': round(row.get('REB', 0) / max(row.get('GP', 1), 1), 1),
            'usg': _format_pct(row.get('USG_PCT'), scale=100),
            'ts': _format_pct(row.get('TS_PCT'), scale=100),
            'mpg': round(row.get('MPG', 0), 1),
            'eff_tier': row.get('efficiency_tier', ''),
            'at_rim': round(row.get('AT_RIM_FREQ', 0) * 100, 1) if pd.notna(row.get('AT_RIM_FREQ')) else None,
            'position_primary': pos_est.get('primary_position_estimate') or pos_est.get('primary_position') or '',
            'bke_rank': clean_value(bke_rec.get('rank')),
            'bke_pct': clean_value(bke_rec.get('final_BKE_percentile')),
            'transformed_bke': clean_value(bke_rec.get('transformed_BKE')),
            'transformed_obke': clean_value(bke_rec.get('transformed_OBKE')),
            'transformed_dbke': clean_value(bke_rec.get('transformed_DBKE')),
        }

        # ---- Version-specific BKE card data for toggle ----
        v28_rec = bke_v28_map.get((pid, season), {})
        v30_rec = bke_v30_map.get((pid, season), {})

        card['bke_versions'] = {
            'v27': {
                'pct': card['bke_pct'],
                'rank': card['bke_rank'],
                'bke': card['transformed_bke'],
                'obke': card['transformed_obke'],
                'dbke': card['transformed_dbke'],
                'obke_pct': clean_value(bke_rec.get('final_OBKE_percentile')),
                'dbke_pct': clean_value(bke_rec.get('final_DBKE_percentile')),
            },
        }
        if v28_rec:
            card['bke_versions']['v28'] = {
                'pct': clean_value(v28_rec.get('pct')),
                'rank': clean_value(v28_rec.get('rank')),
                'bke': clean_value(v28_rec.get('bke_composite')),
                'obke': clean_value(v28_rec.get('obke_transformed')),
                'dbke': clean_value(v28_rec.get('dbke_transformed')),
                'obke_pct': clean_value(v28_rec.get('obke_pct')),
                'dbke_pct': clean_value(v28_rec.get('dbke_pct')),
            }
        if v30_rec:
            card['bke_versions']['v30'] = {
                'pct': clean_value(v30_rec.get('bke_pct')),
                'rank': clean_value(v30_rec.get('rank')),
                'bke': clean_value(v30_rec.get('BKE_v30')),
                'obke': clean_value(v30_rec.get('OBKE_v30_reference')),
                'dbke': clean_value(v30_rec.get('DBKE_scaled_v30')),
                'obke_pct': clean_value(v30_rec.get('obke_pct')),
                'dbke_pct': clean_value(v30_rec.get('dbke_pct')),
            }

        cards.append(card)
        profile = dict(bios_map.get(pid, {}))
        profile["position"] = (
            pos_est.get("primary_position_estimate")
            or pos_est.get("primary_position")
            or profile.get("position")
        )
        for col in [
            "position_estimate_method",
            "pct_pg", "pct_sg", "pct_sf", "pct_pf", "pct_c",
            "pct_guards_own", "pct_forwards_own", "pct_centers_own",
        ]:
            if col in pos_est:
                profile[col] = pos_est.get(col)

        detail = {
            'profile': profile,
            'salary': salary_map.get((pid, season)),
            'stint_context': {
                'team': team_abbr,
                'stint_number': stint_number,
                'stint_count': stint_count,
                'stint_team_count': stint_team_count,
                'is_primary_stint': is_primary_stint,
                'is_final_stint': is_final_stint,
                'team_assignment_source': team_assignment_source,
            },
            'off_reasons': get_offensive_reasons(row),
            'def_reasons': get_defensive_reasons(row),
            'playtypes': get_playtype_rankings(row),
            'archetype_model': row_to_dict(row, exclude={'PLAYER_ID','player_id','PLAYER_NAME','SEASON','TEAM_ABBREVIATION'}),
            'profile_stats': profiles_map.get((pid, season), {}),
            'rapm': rapm_map.get((pid, season), {}),
            'xrapm': xrapm_map.get((pid, season), {}),
            'xrapm_v2': xrapm_v2_map.get((pid, season), {}),
            'linear': linear_map.get((pid, season), {}),
            'season_stats': season_stats_map.get((pid, season), {}),
            'bke': {
                'rank': clean_value(bke_rec.get('rank')),
                'final_BKE_percentile': clean_value(bke_rec.get('final_BKE_percentile')),
                'final_OBKE_percentile': clean_value(bke_rec.get('final_OBKE_percentile')),
                'final_DBKE_percentile': clean_value(bke_rec.get('final_DBKE_percentile')),
                'transformed_BKE': clean_value(bke_rec.get('transformed_BKE')),
                'transformed_OBKE': clean_value(bke_rec.get('transformed_OBKE')),
                'transformed_DBKE': clean_value(bke_rec.get('transformed_DBKE')),
                'position_bucket': clean_value(bke_rec.get('position_bucket')),
                'primary_archetype': clean_value(bke_rec.get('primary_archetype')),
                'defensive_archetype': clean_value(bke_rec.get('defensive_archetype')),
                'position_band_BKE_percentile': clean_value(bke_rec.get('position_band_BKE_percentile')),
                'position_band_OBKE_percentile': clean_value(bke_rec.get('position_band_OBKE_percentile')),
                'position_band_DBKE_percentile': clean_value(bke_rec.get('position_band_DBKE_percentile')),
                'off_archetype_BKE_percentile': clean_value(bke_rec.get('off_archetype_BKE_percentile')),
                'off_archetype_OBKE_percentile': clean_value(bke_rec.get('off_archetype_OBKE_percentile')),
                'off_archetype_DBKE_percentile': clean_value(bke_rec.get('off_archetype_DBKE_percentile')),
                'def_archetype_BKE_percentile': clean_value(bke_rec.get('def_archetype_BKE_percentile')),
                'def_archetype_OBKE_percentile': clean_value(bke_rec.get('def_archetype_OBKE_percentile')),
                'def_archetype_DBKE_percentile': clean_value(bke_rec.get('def_archetype_DBKE_percentile')),
            },
            'bke_details': {
                'rank': clean_value(bke_rec.get('rank')),
                'raw_OBKE': clean_value(bke_rec.get('raw_OBKE')),
                'raw_DBKE': clean_value(bke_rec.get('raw_DBKE')),
                'raw_BKE': clean_value(bke_rec.get('raw_BKE')),
                'transformed_OBKE': clean_value(bke_rec.get('transformed_OBKE')),
                'transformed_DBKE': clean_value(bke_rec.get('transformed_DBKE')),
                'transformed_BKE': clean_value(bke_rec.get('transformed_BKE')),
                'final_OBKE_percentile': clean_value(bke_rec.get('final_OBKE_percentile')),
                'final_DBKE_percentile': clean_value(bke_rec.get('final_DBKE_percentile')),
                'final_BKE_percentile': clean_value(bke_rec.get('final_BKE_percentile')),
                'position_bucket': clean_value(bke_rec.get('position_bucket')),
                'primary_archetype': clean_value(bke_rec.get('primary_archetype')),
                'defensive_archetype': clean_value(bke_rec.get('defensive_archetype')),
                'position_band_BKE_percentile': clean_value(bke_rec.get('position_band_BKE_percentile')),
                'position_band_OBKE_percentile': clean_value(bke_rec.get('position_band_OBKE_percentile')),
                'position_band_DBKE_percentile': clean_value(bke_rec.get('position_band_DBKE_percentile')),
                'off_archetype_BKE_percentile': clean_value(bke_rec.get('off_archetype_BKE_percentile')),
                'off_archetype_OBKE_percentile': clean_value(bke_rec.get('off_archetype_OBKE_percentile')),
                'off_archetype_DBKE_percentile': clean_value(bke_rec.get('off_archetype_DBKE_percentile')),
                'def_archetype_BKE_percentile': clean_value(bke_rec.get('def_archetype_BKE_percentile')),
                'def_archetype_OBKE_percentile': clean_value(bke_rec.get('def_archetype_OBKE_percentile')),
                'def_archetype_DBKE_percentile': clean_value(bke_rec.get('def_archetype_DBKE_percentile')),
                'layer1_offensive_raw': clean_value((bke_rec.get('layer_scores') or {}).get('layer1_offensive_raw')),
                'layer1_defensive_raw': clean_value((bke_rec.get('layer_scores') or {}).get('layer1_defensive_raw')),
                'layer2_rue_raw': clean_value((bke_rec.get('layer_scores') or {}).get('layer2_rue_raw')),
                'layer3_off_elevation_raw': clean_value((bke_rec.get('layer_scores') or {}).get('layer3_off_elevation_raw')),
                'layer3_def_elevation_raw': clean_value((bke_rec.get('layer_scores') or {}).get('layer3_def_elevation_raw')),
                'layer4_scheme_bonus_raw': clean_value((bke_rec.get('layer_scores') or {}).get('layer4_scheme_bonus_raw')),
            },
        }

        # ---- v28 detail data ----
        if v28_rec:
            v28_detail = {
                'obke_transformed': clean_value(v28_rec.get('obke_transformed')),
                'dbke_transformed': clean_value(v28_rec.get('dbke_transformed')),
                'bke_composite': clean_value(v28_rec.get('bke_composite')),
                'rank': clean_value(v28_rec.get('rank')),
                'portable_talent_z_adj': clean_value(v28_rec.get('portable_talent_z_adj')),
                'offensive_portable_z': clean_value(v28_rec.get('offensive_portable_z')),
                'defensive_portable_z': clean_value(v28_rec.get('defensive_portable_z')),
            }
            for k, v in v28_rec.get('layer_scores', {}).items():
                v28_detail[f'layer_{k}'] = clean_value(v)
            for k, v in v28_rec.get('dimension_scores', {}).items():
                v28_detail[f'dim_{k}'] = clean_value(v)
            detail['bke_v28'] = v28_detail
        else:
            detail['bke_v28'] = {}

        # ---- v30 detail data ----
        if v30_rec:
            detail['bke_v30'] = {
                'BKE_v30': clean_value(v30_rec.get('BKE_v30')),
                'DBKE_scaled_v30': clean_value(v30_rec.get('DBKE_scaled_v30')),
                'OBKE_v30_reference': clean_value(v30_rec.get('OBKE_v30_reference')),
                'DBKE_raw_v30': clean_value(v30_rec.get('DBKE_raw_v30')),
                'DBKE_final_v30': clean_value(v30_rec.get('DBKE_final_v30')),
                'DBKE_asym_v30': clean_value(v30_rec.get('DBKE_asym_v30')),
                'DBKE_norm_v30': clean_value(v30_rec.get('DBKE_norm_v30')),
                'D_port': clean_value(v30_rec.get('D_port')),
                'D_rapm_shrunk': clean_value(v30_rec.get('D_rapm_shrunk')),
                'D_stabilized': clean_value(v30_rec.get('D_stabilized')),
                'lambda_shrink': clean_value(v30_rec.get('lambda_shrink')),
                'rank': clean_value(v30_rec.get('rank')),
                'bke_pct': clean_value(v30_rec.get('bke_pct')),
                'dbke_pct': clean_value(v30_rec.get('dbke_pct')),
                'obke_pct': clean_value(v30_rec.get('obke_pct')),
            }
        else:
            detail['bke_v30'] = {}

        details[key] = detail

    # Deduplicate cards by key (traded players can appear twice)
    seen_keys = set()
    unique_cards = []
    for c in cards:
        if c['key'] not in seen_keys:
            seen_keys.add(c['key'])
            unique_cards.append(c)
    cards = unique_cards

    apply_bke_split_variants(cards, details)

    cards.sort(key=lambda x: x['ppg'], reverse=True)

    cards_json = json.dumps(cards, separators=(',', ':'))
    details_json = json.dumps(details, separators=(',', ':'))

    print(f"  Card index: {len(cards_json):,} bytes ({len(cards_json)/1024:.0f} KB)")
    print(f"  Detail store: {len(details_json):,} bytes ({len(details_json)/1024:.0f} KB)")

    # ---- HTML template ----
    # IMPORTANT: {{/}} are used for CSS/JS literal braces.
    # Template replacement order: first convert {{ -> { and }} -> },
    # THEN inject JSON via __CARDS_JSON__ / __DETAILS_JSON__ placeholders.
    # This prevents JSON corruption from brace replacement.
    html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>NBA Player Data Browser</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');
*{{box-sizing:border-box;font-family:'Space Grotesk',sans-serif}}
body{{margin:0;padding:20px;background:#1a1a2e;color:#eee}}
h1{{color:#00d9ff;margin-bottom:5px}}
.sub{{color:#888;margin-bottom:20px}}
.controls{{display:flex;gap:12px;margin-bottom:20px;flex-wrap:wrap;align-items:center}}
input,select{{padding:10px 15px;border:1px solid #333;border-radius:8px;background:#16213e;color:#fff;font-size:14px}}
input{{width:300px}}select{{min-width:180px}}
.sort-btn{{padding:8px 14px;border:1px solid #444;border-radius:8px;background:#16213e;color:#aaa;cursor:pointer;font-size:13px}}
.sort-btn.active{{border-color:#00d9ff;color:#00d9ff}}
.sort-hint{{color:#6174a8;font-size:12px}}
.chip-row{{display:flex;gap:8px;flex-wrap:wrap;margin:-6px 0 16px 0}}
.chip-btn{{padding:6px 12px;border:1px solid #2d3e70;border-radius:999px;background:#16213e;color:#8aa0d6;cursor:pointer;font-size:12px}}
.chip-btn.active{{border-color:#00d9ff;color:#00d9ff}}
.reset-btn{{padding:8px 14px;border:1px solid #e63946;border-radius:8px;background:#16213e;color:#e63946;cursor:pointer;font-size:13px;font-weight:600;transition:background .2s,color .2s}}
.reset-btn:hover{{background:#e63946;color:#fff}}
.stats-bar{{display:flex;gap:20px;margin-bottom:20px;color:#888}}
.stats-bar span{{background:#16213e;padding:8px 15px;border-radius:6px}}
.ver-group{{display:flex;gap:4px;align-items:center}}
.ver-btn{{padding:6px 12px;border:1px solid #3d5a80;border-radius:6px;background:#16213e;color:#7c8bb5;cursor:pointer;font-size:12px;font-weight:600;transition:all .2s}}
.ver-btn.active{{border-color:#f4a261;color:#f4a261;background:#1e2a4a}}
.ver-btn:hover{{border-color:#f4a261;color:#f4a261}}
.ver-label{{color:#666;font-size:12px;margin-right:2px}}
.split-label{{color:#666;font-size:12px;margin-right:2px;margin-left:8px}}

.load-wrap{{background:#101a33;border:1px solid #1f2a4f;border-radius:8px;padding:10px 12px;margin-bottom:16px;display:none}}
.load-wrap.active{{display:block}}
.load-label{{color:#7c8bb5;font-size:12px;margin-bottom:6px}}
.load-track{{height:8px;background:#0b132b;border-radius:999px;overflow:hidden}}
.load-fill{{height:100%;width:0%;background:linear-gradient(90deg,#00d9ff,#4ecdc4);border-radius:999px;transition:width .12s ease}}
.load-pct{{color:#7c8bb5;font-size:12px;margin-top:4px;text-align:right}}

.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(380px,1fr));gap:14px}}
.card{{background:#16213e;border-radius:12px;padding:16px;border:1px solid #333;cursor:pointer;transition:transform .2s,border-color .2s;content-visibility:auto;contain-intrinsic-size:auto 140px}}
.card:hover{{transform:translateY(-3px);border-color:#00d9ff}}
.card-hdr{{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}}
.pname{{font-size:17px;font-weight:600;color:#fff}}
.pmeta{{color:#888;font-size:12px;text-align:right}}
.pteam{{color:#aaa;font-weight:500}}
.srow{{display:flex;gap:10px;margin-bottom:10px;flex-wrap:wrap}}
.st{{text-align:center;min-width:38px}}
.sv{{font-size:14px;font-weight:600;color:#00d9ff}}
.sl{{color:#555;font-size:9px;text-transform:uppercase}}
.sv.na{{color:#555}}
.arow{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}
.abox{{padding:8px;border-radius:8px;background:#0d1b3e}}
.albl{{font-size:9px;color:#555;text-transform:uppercase;letter-spacing:1px;margin-bottom:2px}}
.aname{{font-size:13px;font-weight:600;margin-bottom:1px}}
.aname.off{{color:#ff6b6b}}.aname.def{{color:#4ecdc4}}
.conf{{font-size:10px;color:#666}}
.eff-badge{{display:inline-block;padding:1px 5px;border-radius:4px;font-size:9px;font-weight:600;margin-left:4px}}
.eff-badge.elite{{background:#2d6a4f;color:#95d5b2}}
.eff-badge.high{{background:#1b4332;color:#74c69d}}
.eff-badge.avg{{background:#333;color:#888}}
.eff-badge.low{{background:#6b2c2c;color:#e09898}}
.no-results{{text-align:center;padding:40px;color:#666}}

/* modal */
.mo{{position:fixed;inset:0;background:rgba(4,8,20,.85);display:none;align-items:center;justify-content:center;z-index:999}}
.mo.open{{display:flex}}
.mc{{width:min(1600px,80vw);max-height:90vh;overflow:auto;background:#0f1730;border:1px solid #1e2a50;border-radius:18px;padding:36px;box-shadow:0 40px 120px rgba(0,0,0,.55)}}
.mh{{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;margin-bottom:16px}}
.mt{{font-size:32px;font-weight:700;color:#fff}}
.ms{{color:#7c8bb5;font-size:16px;margin-top:6px}}
.mx{{background:transparent;border:1px solid #2a3b70;color:#8aa0d6;width:34px;height:34px;border-radius:8px;cursor:pointer;font-size:18px}}
.msec{{margin-top:16px;padding:14px 16px 12px;border:1px solid #243763;border-radius:12px;background:#101a36;box-shadow:inset 0 1px 0 rgba(255,255,255,.03)}}
.msec h3{{margin:0 0 14px 0;font-size:15px;text-transform:uppercase;letter-spacing:1.5px;color:#8ea7db;cursor:pointer;user-select:none;display:flex;align-items:center;gap:8px;padding-bottom:8px;border-bottom:1px solid #263a66}}
.msec h4.msec-sub{{margin:18px 0 10px 0;font-size:13px;text-transform:uppercase;letter-spacing:1.2px;color:#71dacd;font-weight:600;border-bottom:1px solid #2a416f;padding-bottom:6px}}
.msec.msec-bke{{border-color:#2b5f78;background:#112637}}
.collapse-toggle{{transition:transform .2s;display:inline-block;font-size:13px;color:#7c8bb5}}
.msec.collapsed>:not(h3){{display:none}}
.msec.collapsed .collapse-toggle{{transform:rotate(-90deg)}}
.sg{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px}}
.sc{{background:#142246;border:1px solid #2b4475;border-radius:10px;padding:10px}}
.sc .lb{{font-size:13px;color:#6f7ea7;text-transform:uppercase;letter-spacing:1px}}
.sc .vl{{font-size:20px;font-weight:600;color:#e6f0ff;margin-top:6px}}
.kvt{{display:grid;grid-template-columns:1fr 1fr;gap:10px 24px;font-size:15px;color:#a5b3d6}}
.kvr{{display:flex;justify-content:space-between;gap:10px;border-bottom:1px dashed #2d4575;padding-bottom:4px}}
.kvk{{color:#7f8db4}}.kvv{{color:#d7e3ff;text-align:right}}
.reasons{{margin-top:6px}}
.reason{{font-size:15px;color:#bac7e6;padding:3px 0 3px 16px;position:relative}}
.reason::before{{content:"\\2022";position:absolute;left:0;color:#555}}
.pt-row{{display:flex;align-items:center;gap:10px;font-size:15px;margin-bottom:6px}}
.pt-name{{width:140px;color:#aaa;text-align:right;white-space:nowrap}}
.pt-bar{{flex:1;height:18px;background:#0a1428;border-radius:4px;overflow:hidden}}
.pt-fill{{height:100%;border-radius:3px;min-width:2px}}
.pt-val{{width:50px;color:#888;font-size:14px}}

/* lean bar */
.lean-wrap{{display:flex;align-items:center;justify-content:center;margin:2px 0 4px}}
.lean-bar{{width:44px;height:4px;border-radius:2px;overflow:hidden;display:flex}}
.lean-o{{height:100%;background:#ff6b6b}}
.lean-d{{height:100%;background:#4ecdc4}}

/* modal nav arrows */
.modal-nav{{position:absolute;top:50%;transform:translateY(-50%);background:rgba(15,23,48,.9);border:1px solid #2a3b70;color:#8aa0d6;width:44px;height:44px;border-radius:50%;cursor:pointer;font-size:22px;display:flex;align-items:center;justify-content:center;z-index:1001;transition:background .2s}}
.modal-nav:hover{{background:rgba(30,42,80,.95);color:#fff}}
.modal-nav.prev{{left:max(1vw,8px)}}
.modal-nav.next{{right:max(1vw,8px)}}
.modal-nav:disabled{{opacity:.25;cursor:default}}

/* comparison overlay */
.cmp-overlay{{position:fixed;inset:0;background:rgba(4,8,20,.9);display:none;align-items:center;justify-content:center;z-index:1000}}
.cmp-overlay.open{{display:flex}}
.cmp-panel{{width:min(920px,92vw);max-height:90vh;overflow:auto;background:#0f1730;border:1px solid #1e2a50;border-radius:18px;padding:28px;box-shadow:0 40px 120px rgba(0,0,0,.55)}}
.cmp-hdr{{display:flex;justify-content:space-between;align-items:center;margin-bottom:16px}}
.cmp-title{{font-size:22px;font-weight:700;color:#fff}}
.cmp-search-row{{display:flex;gap:12px;margin-bottom:18px;align-items:center}}
.cmp-input{{flex:1;position:relative}}
.cmp-input input{{width:100%;padding:8px 12px;background:#142246;border:1px solid #2b4475;border-radius:8px;color:#e6f0ff;font-size:14px;outline:none;box-sizing:border-box}}
.cmp-input input:focus{{border-color:#4e7cc9}}
.cmp-dd{{position:absolute;top:100%;left:0;right:0;background:#142246;border:1px solid #2b4475;border-radius:0 0 8px 8px;max-height:200px;overflow:auto;z-index:10;display:none}}
.cmp-dd.open{{display:block}}
.cmp-dd-item{{padding:6px 12px;cursor:pointer;font-size:13px;color:#aaa}}
.cmp-dd-item:hover{{background:#1e3260;color:#fff}}
.cmp-dd-season{{color:#666;font-size:11px;margin-left:6px}}
.cmp-hdr-row{{display:grid;grid-template-columns:1fr 100px 1fr;gap:0;margin-bottom:10px;padding-bottom:6px;border-bottom:1px solid #263a66}}
.cmp-player-hdr{{font-size:16px;font-weight:600;color:#fff;text-align:center;padding:8px 0}}
.cmp-stats-row{{display:grid;grid-template-columns:1fr 100px 1fr;gap:0;margin-bottom:2px;font-size:14px;color:#bac7e6;align-items:center}}
.cmp-val{{padding:4px 8px;text-align:center}}
.cmp-val.left{{text-align:right}}
.cmp-val.right{{text-align:left}}
.cmp-val.better{{color:#4ecdc4;font-weight:600}}
.cmp-label{{padding:4px 8px;text-align:center;color:#7c8bb5;font-size:12px;text-transform:uppercase;min-width:80px}}
.radar-wrap{{display:flex;justify-content:center;margin:20px 0}}
.radar-legend{{display:flex;gap:16px;justify-content:center;margin-bottom:12px;font-size:13px;color:#bac7e6}}
.radar-legend span{{display:flex;align-items:center;gap:4px}}
.radar-dot{{width:10px;height:10px;border-radius:50%;display:inline-block}}

/* modal tabs */
.m-tabs{{display:flex;gap:0;margin:8px 0 0}}
.m-tab{{padding:8px 18px;background:transparent;border:1px solid #2a3b70;color:#7c8bb5;font-size:14px;cursor:pointer;border-radius:8px 8px 0 0;border-bottom:none;transition:all .2s}}
.m-tab.active{{background:#101a36;color:#e6f0ff;border-color:#4e7cc9;font-weight:600}}
.m-tab:hover:not(.active){{background:rgba(30,42,80,.5)}}
/* graphs sections */
.gfx-sec{{margin-bottom:20px;padding:16px;border:1px solid #243763;border-radius:12px;background:#101a36}}
.gfx-title{{font-size:15px;font-weight:600;color:#8ea7db;text-transform:uppercase;letter-spacing:1.2px;margin-bottom:12px}}
/* YoY chart */
.yoy-filters{{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:12px}}
.yoy-filter{{display:flex;align-items:center;gap:4px;cursor:pointer;font-size:12px;color:#bac7e6;padding:3px 8px;border-radius:4px;background:#142246;border:1px solid #2b4475;user-select:none}}
.yoy-filter.off{{opacity:.35;border-color:transparent}}
.yoy-filter .yoy-dot{{width:8px;height:8px;border-radius:50%;display:inline-block}}
/* layer bars */
.layer-row{{display:flex;align-items:center;gap:8px;margin-bottom:8px}}
.layer-label{{width:140px;font-size:13px;color:#8ea7db;text-align:right;white-space:nowrap}}
.layer-bar-wrap{{flex:1;height:24px;position:relative;background:#0a1428;border-radius:4px;overflow:hidden}}
.layer-center{{position:absolute;left:50%;top:0;bottom:0;width:1px;background:#3a5180;z-index:1}}
.layer-bar{{height:100%;position:absolute;top:0;border-radius:3px;min-width:2px}}
.layer-val{{width:55px;font-size:13px;color:#bac7e6;text-align:left;font-weight:600}}
/* position stacked */
.pos-bar-wrap{{display:flex;height:28px;border-radius:6px;overflow:hidden;margin-top:8px}}
.pos-seg{{height:100%;display:flex;align-items:center;justify-content:center;color:#fff;font-size:10px;font-weight:600;min-width:0}}
.pos-legend{{display:flex;gap:12px;margin-top:8px;flex-wrap:wrap}}
.pos-legend-item{{display:flex;align-items:center;gap:4px;font-size:12px;color:#bac7e6}}
.pos-legend-dot{{width:10px;height:10px;border-radius:2px;display:inline-block}}
</style>
</head>
<body>
<h1>&#127936; NBA Player Data Browser</h1>
<p class="sub">Click any card to view full stats, playtypes &amp; archetype reasoning</p>

<div class="controls">
  <input type="text" id="search" placeholder="Search player name..." oninput="filterPlayers()">
  <select id="offFilter" onchange="filterPlayers()"><option value="">All Offensive</option>
    <option>Ball Dominant Creator</option><option>Ballhandler</option><option>All-Around Scorer</option>
    <option>Perimeter Scorer</option><option>Interior Scorer</option><option>Connector</option>
    <option>Off-Ball Finisher</option><option>Off-Ball Movement Shooter</option>
    <option>Off-Ball Stationary Shooter</option><option>PnR Rolling Big</option><option>PnR Popping Big</option>
  </select>
  <select id="defFilter" onchange="filterPlayers()"><option value="">All Defensive</option></select>
  <select id="seasonFilter" onchange="filterPlayers()"><option value="">All Seasons</option>
    <option>2024-25</option><option>2023-24</option><option>2022-23</option>
  </select>
  <span style="color:#555">Sort:</span>
  <span class="sort-btn active" data-sort="ppg" onclick="setSort(this)">PPG</span>
    <span class="sort-btn" data-sort="off_confidence" onclick="setSort(this)">Off Fit</span>
    <span class="sort-btn" data-sort="off_effectiveness" onclick="setSort(this)">Off Effectiveness</span>
    <span class="sort-btn" data-sort="transformed_bke" onclick="setSort(this)">tBKE</span>
    <span class="sort-btn" data-sort="transformed_obke" onclick="setSort(this)">tOBKE</span>
    <span class="sort-btn" data-sort="transformed_dbke" onclick="setSort(this)">tDBKE</span>
  <span class="sort-btn" data-sort="ts" onclick="setSort(this)">TS%</span>
  <span class="sort-btn" data-sort="usg" onclick="setSort(this)">USG%</span>
    <span class="sort-hint">Click active sort to toggle direction</span>
  <div class="ver-group">
    <span class="ver-label">BKE:</span>
    <span class="ver-btn active" data-ver="v27" onclick="switchBkeVersion(this)">v27</span>
    <span class="ver-btn" data-ver="v28" onclick="switchBkeVersion(this)">v28</span>
    <span class="ver-btn" data-ver="v30" onclick="switchBkeVersion(this)">v3.0</span>
        <span class="split-label">Split:</span>
        <span class="ver-btn active split-btn" data-split="split_60_40" onclick="switchBkeSplit(this)">60/40</span>
        <span class="ver-btn split-btn" data-split="split_55_45" onclick="switchBkeSplit(this)">55/45</span>
  </div>
  <span class="sort-btn" onclick="openCmp()" style="background:#1e3260;margin-left:8px">&#128200; Compare</span>
  <span class="reset-btn" onclick="resetFilters()">Reset All</span>
</div>

<div class="chip-row" id="positionChips">
    <button class="chip-btn active" data-pos="" onclick="setPositionChip(this)">All Positions</button>
    <button class="chip-btn" data-pos="Guard" onclick="setPositionChip(this)">Guard</button>
    <button class="chip-btn" data-pos="Guard-Forward" onclick="setPositionChip(this)">Guard-Forward</button>
    <button class="chip-btn" data-pos="Forward" onclick="setPositionChip(this)">Forward</button>
    <button class="chip-btn" data-pos="Forward-Center" onclick="setPositionChip(this)">Forward-Center</button>
    <button class="chip-btn" data-pos="Center" onclick="setPositionChip(this)">Center</button>
</div>

<div class="stats-bar"><span id="totalCount">Loading...</span></div>
<div id="loadBar" class="load-wrap active" aria-live="polite">
  <div class="load-label" id="loadLabel">Preparing...</div>
  <div class="load-track"><div class="load-fill" id="loadFill"></div></div>
  <div class="load-pct" id="loadPct">0 / 0 (0%)</div>
</div>

<div id="grid" class="grid"></div>

<div id="modal" class="mo" onclick="closeModal(event)">
  <button class="modal-nav prev" id="modalPrev" onclick="event.stopPropagation();navigateModal(-1)" title="Previous season">&#9664;</button>
  <div class="mc" role="dialog" aria-modal="true">
    <div class="mh">
      <div>
        <div id="mTitle" class="mt"></div>
        <div id="mSub" class="ms"></div>
        <div class="m-tabs">
          <button class="m-tab active" id="tabStats" onclick="switchModalTab('stats')">&#128202; Stats</button>
          <button class="m-tab" id="tabGraphs" onclick="switchModalTab('graphs')">&#128200; Graphs</button>
        </div>
      </div>
      <button class="mx" onclick="closeModal(event,true)">&times;</button>
    </div>
    <div id="mBody"></div>
    <div id="mGraphs" style="display:none"></div>
  </div>
  <button class="modal-nav next" id="modalNext" onclick="event.stopPropagation();navigateModal(1)" title="Next season">&#9654;</button>
</div>

<div id="cmpModal" class="cmp-overlay" onclick="if(event.target===this)closeCmp()">
  <div class="cmp-panel">
    <div class="cmp-hdr">
      <span class="cmp-title">&#128200; Player Comparison</span>
      <button class="mx" onclick="closeCmp()">&times;</button>
    </div>
    <div class="cmp-search-row">
      <div class="cmp-input"><input id="cmpSearch1" placeholder="Search player 1..." oninput="cmpAutocomplete(1)" onfocus="cmpAutocomplete(1)"><div class="cmp-dd" id="cmpDd1"></div></div>
      <div style="color:#555;align-self:center;font-size:22px;font-weight:700">vs</div>
      <div class="cmp-input"><input id="cmpSearch2" placeholder="Search player 2..." oninput="cmpAutocomplete(2)" onfocus="cmpAutocomplete(2)"><div class="cmp-dd" id="cmpDd2"></div></div>
    </div>
    <div id="cmpBody"></div>
  </div>
</div>

<!-- Detail data stored as inert text; parsed lazily on first modal open -->
<script id="detailStore" type="application/json">__DETAILS_JSON__</script>

<script>
/* ---- card index (small, parsed immediately) ---- */
var cards=__CARDS_JSON__;
var curSort='ppg';
var curSortDir='desc';
var curPos='';
var CHUNK=80;
var fData=[],rendered=0,gen=0;
var _detailCache=null;  /* lazy-parsed on first modal open */
var _sortBaseLabels={};
var curBkeVersion='v27';
var curBkeSplit='split_60_40';
var curModalKey=null;
var cmpSel={1:null,2:null};

var cardByKey={};
for(var _ci=0;_ci<cards.length;_ci++){
    cardByKey[cards[_ci].key]=cards[_ci];
}

function stintSuffix(card){
    if(!card)return '';
    var n=Number(card.stint_number);
    var c=Number(card.stint_count);
    var t=Number(card.stint_team_count);
    if(!isFinite(n)||n<1)n=1;
    if(!isFinite(c)||c<1)c=1;
    if(!isFinite(t)||t<1)t=c;
    if(c<=1&&t<=1)return '';
    var tags=['S'+n+'/'+c];
    if(card.is_final_stint)tags.push('Final');
    else if(card.is_primary_stint)tags.push('Primary');
    return '('+tags.join(' · ')+')';
}

function stintMetaText(card){
    if(!card)return '';
    var bits=[];
    var suffix=stintSuffix(card);
    if(suffix)bits.push(suffix.replace(/^\(|\)$/g,''));
    if(card.team_assignment_source==='stint')bits.push('stint-assigned');
    return bits.join(' · ');
}

function _preferSeasonCard(candidate,current){
    if(!current)return true;
    if(!!candidate.is_final_stint!==!!current.is_final_stint)return !!candidate.is_final_stint;
    if(!!candidate.is_primary_stint!==!!current.is_primary_stint)return !!candidate.is_primary_stint;
    var cMpg=Number(candidate.mpg),pMpg=Number(current.mpg);
    if(isFinite(cMpg)&&isFinite(pMpg)&&cMpg!==pMpg)return cMpg>pMpg;
    var cStint=Number(candidate.stint_number),pStint=Number(current.stint_number);
    if(isFinite(cStint)&&isFinite(pStint)&&cStint!==pStint)return cStint>pStint;
    return false;
}

var playerSeasonPreferredKey={};
for(var _si=0;_si<cards.length;_si++){
    var c=cards[_si];
    var pid=(c.id||((c.key||'').split('::')[0])||'');
    var season=(c.season||'');
    if(!pid||!season)continue;
    var seasonKey=pid+'::'+season;
    var existingKey=playerSeasonPreferredKey[seasonKey];
    var existingCard=existingKey?cardByKey[existingKey]:null;
    if(_preferSeasonCard(c,existingCard))playerSeasonPreferredKey[seasonKey]=c.key;
}

function seasonLookupKey(pid,season){
    var seasonKey=pid+'::'+season;
    if(playerSeasonPreferredKey[seasonKey])return playerSeasonPreferredKey[seasonKey];
    for(var i=0;i<cards.length;i++){
        var c=cards[i];
        var cpid=(c.id||((c.key||'').split('::')[0])||'');
        if(cpid===pid&&c.season===season)return c.key;
    }
    return seasonKey;
}

/* ---- BKE version helpers ---- */
function splitDisplayLabel(){
    return curBkeSplit==='split_55_45'?'55/45':'60/40';
}
function getSplitData(rec){
    var splits=(rec&&rec.split_variants)?rec.split_variants:{};
    return splits[curBkeSplit]||splits['split_60_40']||null;
}
function getBkeField(card, field) {
    var v = (card.bke_versions || {})[curBkeVersion];
    if (!v) v = (card.bke_versions || {})['v27'] || {};
    if(field==='bke' || field==='rank' || field==='pct'){
        var splitRec=(v.splits||{})[curBkeSplit] || (v.splits||{})['split_60_40'];
        if(splitRec && splitRec[field] !== undefined && splitRec[field] !== null) return splitRec[field];
    }
    return v[field] !== undefined ? v[field] : null;
}
function switchBkeVersion(el) {
    var ver = el.dataset.ver;
    if (ver === curBkeVersion) return;
    curBkeVersion = ver;
    document.querySelectorAll('.ver-btn[data-ver]').forEach(function(b) {
        b.classList.toggle('active', b.dataset.ver === ver);
    });
    filterPlayers();
}
function switchBkeSplit(el) {
    var split = el.dataset.split;
    if (split === curBkeSplit) return;
    curBkeSplit = split;
    document.querySelectorAll('.split-btn').forEach(function(b) {
        b.classList.toggle('active', b.dataset.split === split);
    });
    filterPlayers();
}

/* populate defensive filter from card data */
var dSel=document.getElementById('defFilter');
var _dSet={};
for(var i=0;i<cards.length;i++){var a=cards[i].def_archetype;if(a&&a!=='Unknown')_dSet[a]=1;}
Object.keys(_dSet).sort().forEach(function(t){
  var o=document.createElement('option');o.value=t;o.textContent=t;dSel.appendChild(o);
});

var ptCols={
  'Isolation':'#e63946','PnR Ball Handler':'#f4a261','Post Up':'#e76f51',
  'Cut':'#2a9d8f','PnR Roll Man':'#264653','Handoff':'#a8dadc',
  'Off Screen':'#457b9d','Spot Up':'#1d3557','Transition':'#f1faee',
  'Putback':'#606c38','Misc':'#555'
};

function eBadge(t){if(!t)return '';var c=t==='Elite'?'elite':t==='High'?'high':t==='Low'?'low':'avg';return '<span class="eff-badge '+c+'">'+t+'</span>';}
function fs(v,s){if(v===null||v===undefined)return '<span class="na">&mdash;</span>';return v+(s||'');}
function fp(v){
    if(v===null||v===undefined||v==='')return null;
    var num=Number(v);
    if(!isFinite(num))return String(v);
    if(Math.abs(num)<1.5)num=num*100;
    return num.toFixed(1)+'%';
}
function perGame(val,gp){
    if(val===null||val===undefined||gp===null||gp===undefined||gp===0)return null;
    var num=Number(val)/Number(gp);
    if(!isFinite(num))return null;
    return Number(num.toFixed(2));
}
function fv(v){
  if(v===null||v===undefined||v==='')return '&mdash;';
  if(typeof v==='number'){if(Number.isInteger(v))return v.toString();var a=Math.abs(v);if(a>=100)return v.toFixed(1);if(a>=10)return v.toFixed(2);return v.toFixed(3);}
  return String(v);
}
function fMoney(v){
    if(v===null||v===undefined||v==='')return '';
    var n=Number(v);
    if(!isFinite(n))return String(v);
    return '$'+Math.round(n).toLocaleString('en-US');
}
function fbke(v){
    if(v===null||v===undefined||v==='')return '<span class="na">&mdash;</span>';
    var n=Number(v);
    if(!isFinite(n))return '<span class="na">&mdash;</span>';
    return n.toFixed(1)+'%';
}
function ordSuffix(n){
    if(n===null||n===undefined)return '';
    var v=Number(n);if(!isFinite(v)||v<1)return '';
    var s=['th','st','nd','rd'],m=v%100;
    return v+(s[(m-20)%10]||s[m]||s[0]);
}
function fbkeCard(pct,rank){
    if(pct===null||pct===undefined||pct==='')return '<span class="na">&mdash;</span>';
    var n=Number(pct);
    if(!isFinite(n))return '<span class="na">&mdash;</span>';
    var txt=n.toFixed(1)+'%';
    if(rank!==null&&rank!==undefined){var o=ordSuffix(rank);if(o)txt+=' <span style="color:#7c8bb5;font-size:11px">('+o+')</span>';}
    return txt;
}

function initSortUI(){
    document.querySelectorAll('.sort-btn[data-sort]').forEach(function(btn){
        var key=btn.dataset.sort;
        if(!_sortBaseLabels[key])_sortBaseLabels[key]=btn.textContent.trim();
    });
    refreshSortUI();
}

function refreshSortUI(){
    document.querySelectorAll('.sort-btn[data-sort]').forEach(function(btn){
        var key=btn.dataset.sort;
        var base=_sortBaseLabels[key]||btn.textContent.trim();
        if(key===curSort){
            btn.classList.add('active');
            btn.textContent=base+(curSortDir==='desc'?' ↓':' ↑');
        }else{
            btn.classList.remove('active');
            btn.textContent=base;
        }
    });
}

/* ---- progress ---- */
function updProg(cur,tot){
  var pct=tot>0?Math.round(cur/tot*100):0;
  var f=document.getElementById('loadFill'),t=document.getElementById('loadPct'),l=document.getElementById('loadLabel');
  if(f)f.style.width=pct+'%';
  if(t)t.textContent=cur+' / '+tot+' ('+pct+'%)';
  if(l)l.textContent=cur<tot?'Rendering cards...':'Done!';
}
function showLoad(){document.getElementById('loadBar').classList.add('active');updProg(0,0);}
function hideLoad(){setTimeout(function(){document.getElementById('loadBar').classList.remove('active');},350);}

/* ---- card ---- */
function cardHTML(p){
  var op=getBkeField(p,'obke_pct'),dp=getBkeField(p,'dbke_pct');var lean='';
  if(op!=null&&dp!=null&&(op+dp)>0){var ow=op/(op+dp)*100;lean='<div class="lean-wrap" title="O/D Lean: Off '+op.toFixed(0)+'% | Def '+dp.toFixed(0)+'%"><div class="lean-bar"><div class="lean-o" style="width:'+ow.toFixed(1)+'%"></div><div class="lean-d" style="width:'+(100-ow).toFixed(1)+'%"></div></div></div>';}
    var sfx=stintSuffix(p);
    var teamLbl='';
    if(p.team)teamLbl='<span class="pteam"> &middot; '+p.team+(sfx?' '+sfx:'')+'</span>';
    else if(sfx)teamLbl='<span class="pteam"> &middot; '+sfx+'</span>';
    var seasonLbl=p.season+(sfx?' '+sfx:'');
  return '<div class="card" onclick="openModal(\\''+p.key+'\\')">'
                +'<div class="card-hdr"><div><span class="pname">'+p.name+'</span>'+teamLbl+'</div><div class="pmeta">'+seasonLbl+'<br>'+p.mpg+' MPG</div></div>'
    +'<div class="srow">'
    +'<div class="st"><div class="sv">'+p.ppg+'</div><div class="sl">PPG</div></div>'
    +'<div class="st"><div class="sv">'+p.apg+'</div><div class="sl">APG</div></div>'
    +'<div class="st"><div class="sv">'+p.rpg+'</div><div class="sl">RPG</div></div>'
    +'<div class="st"><div class="sv">'+fs(p.usg,'%')+'</div><div class="sl">USG</div></div>'
    +'<div class="st"><div class="sv">'+fs(p.ts,'%')+'</div><div class="sl">TS%</div></div>'
    +'<div class="st"><div class="sv">'+fbkeCard(getBkeField(p,"pct"),getBkeField(p,"rank"))+'</div><div class="sl">BKE'+(curBkeVersion!=='v27'?' <span style="color:#f4a261;font-size:8px">'+curBkeVersion.toUpperCase()+'</span>':'')+' <span style="color:#8aa0d6;font-size:8px">'+splitDisplayLabel()+'</span></div></div>'
    +'</div>'
    +lean
    +'<div class="arow">'
    +'<div class="abox"><div class="albl">Offense</div><div class="aname off">'+p.off_archetype+eBadge(p.eff_tier)+'</div><div class="conf">Fit: '+p.off_confidence+'%'+(p.off_secondary?' &middot; '+p.off_secondary:'')+'</div></div>'
    +'<div class="abox"><div class="albl">Defense</div><div class="aname def">'+(p.def_archetype||'Unknown')+'</div><div class="conf">'+p.def_confidence+'%</div></div>'
    +'</div></div>';
}

/* ---- chunked render ---- */
function renderPlayers(arr){
  gen++;var g=gen;fData=arr;rendered=0;
  var grid=document.getElementById('grid');grid.innerHTML='';
  document.getElementById('totalCount').textContent='Showing '+arr.length+' of '+cards.length+' players';
  if(!arr.length){grid.innerHTML='<div class="no-results">No players match your filters</div>';hideLoad();return;}
  showLoad();doChunk(g);
}
function doChunk(g){
  if(g!==gen)return;
  var grid=document.getElementById('grid');
  var end=Math.min(rendered+CHUNK,fData.length);
  var parts=[];
  for(var i=rendered;i<end;i++)parts.push(cardHTML(fData[i]));
  grid.insertAdjacentHTML('beforeend',parts.join(''));
  rendered=end;updProg(rendered,fData.length);
  if(rendered<fData.length)requestAnimationFrame(function(){doChunk(g);});
  else hideLoad();
}

/* ---- lazy detail loading ---- */
function getDetail(key){
  if(!_detailCache){
    var el=document.getElementById('detailStore');
    if(!el){return {};}
    _detailCache=JSON.parse(el.textContent);
    el.remove();  /* free memory - we no longer need the DOM node */
  }
  return _detailCache[key]||{};
}

/* ---- modal helpers ---- */
function renderSG(stats){
  var h='<div class="sg">';
  for(var i=0;i<stats.length;i++)h+='<div class="sc"><div class="lb">'+stats[i].label+'</div><div class="vl">'+fv(stats[i].value)+'</div></div>';
  return h+'</div>';
}
function renderKV(title,data){
  if(!data)return '';
  var keys=Object.keys(data).filter(function(k){return data[k]!==null&&data[k]!==undefined&&data[k]!=='';}).sort();
  if(!keys.length)return '';
  var rows='';
  for(var i=0;i<keys.length;i++)rows+='<div class="kvr"><span class="kvk">'+keys[i]+'</span><span class="kvv">'+fv(data[keys[i]])+'</span></div>';
  return '<div class="msec"><h3>'+title+'</h3><div class="kvt">'+rows+'</div></div>';
}
function renderPT(playtypes){
  if(!playtypes||!playtypes.length)return '';
  var mx=0;for(var i=0;i<playtypes.length;i++)if(playtypes[i].pct>mx)mx=playtypes[i].pct;
  var bars='';
  for(var i=0;i<playtypes.length;i++){
    var pt=playtypes[i],w=mx>0?(pt.pct/mx*100):0,c=ptCols[pt.type]||'#555';
    bars+='<div class="pt-row"><span class="pt-name">'+pt.type+'</span><div class="pt-bar"><div class="pt-fill" style="width:'+w+'%;background:'+c+'"></div></div><span class="pt-val">'+pt.pct+'%</span></div>';
  }
  return '<div class="msec"><h3>Playtype Breakdown</h3>'+bars+'</div>';
}
function renderReasons(off,def){
  var h='';
  if(off&&off.length){h+='<div class="msec"><h3>Offensive Reasoning</h3><div class="reasons">';for(var i=0;i<off.length;i++)h+='<div class="reason">'+off[i]+'</div>';h+='</div></div>';}
  if(def&&def.length){h+='<div class="msec"><h3>Defensive Reasoning</h3><div class="reasons">';for(var i=0;i<def.length;i++)h+='<div class="reason">'+def[i]+'</div>';h+='</div></div>';}
  return h;
}

function renderBKEHighlights(bke, version){
    if(!bke)return '';
    version = version || 'v27';
    var cards=[];
    function push(label,val){if(val!==null&&val!==undefined&&val!=='')cards.push({label:label,value:val});}
    function pct(v){return v!=null?Number(v).toFixed(1)+'%':null;}
    var splitData=getSplitData(bke);
    var splitBke=splitData?splitData.bke:null;
    var splitRank=splitData?splitData.rank:null;
    var splitPct=splitData?splitData.pct:null;
    var splitLabel=splitDisplayLabel();

    if(version==='v27'){
        push('BKE Rank ('+splitLabel+')',splitRank!=null?splitRank:bke.rank);
        push('BKE Percentile ('+splitLabel+')',pct(splitPct!=null?splitPct:bke.final_BKE_percentile));
        push('OBKE Percentile',pct(bke.final_OBKE_percentile));
        push('DBKE Percentile',pct(bke.final_DBKE_percentile));
        push('Pos-Band BKE %ile',pct(bke.position_band_BKE_percentile));
        push('Pos-Band OBKE %ile',pct(bke.position_band_OBKE_percentile));
        push('Pos-Band DBKE %ile',pct(bke.position_band_DBKE_percentile));
        push('Off-Arch BKE %ile',pct(bke.off_archetype_BKE_percentile));
        push('Off-Arch OBKE %ile',pct(bke.off_archetype_OBKE_percentile));
        push('Off-Arch DBKE %ile',pct(bke.off_archetype_DBKE_percentile));
        push('Def-Arch BKE %ile',pct(bke.def_archetype_BKE_percentile));
        push('Def-Arch OBKE %ile',pct(bke.def_archetype_OBKE_percentile));
        push('Def-Arch DBKE %ile',pct(bke.def_archetype_DBKE_percentile));
        push('Split BKE ('+splitLabel+')',splitBke!=null?splitBke:bke.transformed_BKE);
        push('Transformed OBKE',bke.transformed_OBKE);
        push('Transformed DBKE',bke.transformed_DBKE);
        push('Position Band',bke.position_bucket);
        push('Off Archetype',bke.primary_archetype);
        push('Def Archetype',bke.defensive_archetype);
    } else if(version==='v28'){
        push('BKE Composite ('+splitLabel+')',splitBke!=null?splitBke:bke.bke_composite);
        push('OBKE Transformed',bke.obke_transformed);
        push('DBKE Transformed',bke.dbke_transformed);
        push('Rank ('+splitLabel+')',splitRank!=null?splitRank:bke.rank);
        push('BKE Percentile ('+splitLabel+')',pct(splitPct));
        push('Portable Talent Z',bke.portable_talent_z_adj);
        push('Off Portable Z',bke.offensive_portable_z);
        push('Def Portable Z',bke.defensive_portable_z);
        for(var k in bke){if(k.indexOf('layer_')===0)push(k.replace('layer_','L: '),bke[k]);}
        for(var k in bke){if(k.indexOf('dim_')===0)push(k.replace('dim_','D: '),bke[k]);}
    } else if(version==='v30'){
        push('BKE v3.0 ('+splitLabel+')',splitBke!=null?splitBke:bke.BKE_v30);
        push('DBKE Scaled v3.0',bke.DBKE_scaled_v30);
        push('OBKE v3.0 Ref',bke.OBKE_v30_reference);
        push('Rank ('+splitLabel+')',splitRank!=null?splitRank:bke.rank);
        push('BKE Percentile ('+splitLabel+')',pct(splitPct!=null?splitPct:bke.bke_pct));
        push('DBKE Percentile',pct(bke.dbke_pct));
        push('OBKE Percentile',pct(bke.obke_pct));
        push('DBKE Raw v3.0',bke.DBKE_raw_v30);
        push('DBKE Final v3.0',bke.DBKE_final_v30);
        push('D Portable',bke.D_port);
        push('D RAPM Shrunk',bke.D_rapm_shrunk);
        push('\u03BB Shrink',bke.lambda_shrink);
    }

    if(!cards.length)return '<div class="msec msec-bke"><h3>BKE Highlights ('+version.toUpperCase()+' · '+splitLabel+')</h3><p style="color:#666;font-size:14px">No data for this version</p></div>';
    return '<div class="msec msec-bke"><h3>BKE Highlights ('+version.toUpperCase()+' · '+splitLabel+')</h3>'+renderSG(cards)+'</div>';
}

function renderPositionEstimate(pr){
    if(!pr)return '';
    var hasShares = pr.pct_pg!=null || pr.pct_sg!=null || pr.pct_sf!=null || pr.pct_pf!=null || pr.pct_c!=null;
    if(!hasShares && !pr.position)return '';

    function pct(v){
        if(v===null||v===undefined||v==='')return '&mdash;';
        var n=Number(v);
        if(!isFinite(n))return '&mdash;';
        if(Math.abs(n)<=1.5)n=n*100;
        return n.toFixed(1)+'%';
    }

    var method = pr.position_estimate_method ? '<div class="reason">Method: '+pr.position_estimate_method+'</div>' : '';
    var shares = '<div class="sg">'
        +'<div class="sc"><div class="lb">PG</div><div class="vl">'+pct(pr.pct_pg)+'</div></div>'
        +'<div class="sc"><div class="lb">SG</div><div class="vl">'+pct(pr.pct_sg)+'</div></div>'
        +'<div class="sc"><div class="lb">SF</div><div class="vl">'+pct(pr.pct_sf)+'</div></div>'
        +'<div class="sc"><div class="lb">PF</div><div class="vl">'+pct(pr.pct_pf)+'</div></div>'
        +'<div class="sc"><div class="lb">C</div><div class="vl">'+pct(pr.pct_c)+'</div></div>'
        +'</div>';

    return '<div class="msec"><h3>Position Estimate</h3>'
        +'<div class="reasons"><div class="reason">Primary: '+(pr.position||'Unknown')+'</div>'+method+'</div>'
        +shares
        +'</div>';
}

function openModal(key){
  curModalKey=key;
  /* find the card record for basic info */
  var p=null;for(var i=0;i<cards.length;i++){if(cards[i].key===key){p=cards[i];break;}}
  if(!p)return;
  /* lazy-load detail record */
  var d=getDetail(key);
  var pr=d.profile||{};
    var salary=d.salary;
  var team=p.team||pr.team_abbrev||'';
  document.getElementById('mTitle').textContent=p.name+' ('+p.season+')';
  var sub=team;
    var stintInfo=stintMetaText(p);
    if(stintInfo)sub+=(sub?' \u00b7 ':'')+stintInfo;
  if(pr.position)sub+=' \u00b7 '+pr.position;
  if(pr.age)sub+=' \u00b7 Age '+pr.age;
  if(pr.height)sub+=' \u00b7 '+pr.height;
    if(salary!==null&&salary!==undefined&&salary!=='')sub+=' \u00b7 '+fMoney(salary);
  document.getElementById('mSub').textContent=sub;

  var hl=[];
  var ps=d.profile_stats||{};
  var ra=d.rapm||{};
  var xr=d.xrapm||{};
  var ln=d.linear||{};
    var ss=d.season_stats||{};
    var am=d.archetype_model||{};
    var bk;
    var bkDetails;
    if(curBkeVersion==='v28'){bk=d.bke_v28||{};bkDetails=d.bke_v28||{};}
    else if(curBkeVersion==='v30'){bk=d.bke_v30||{};bkDetails=d.bke_v30||{};}
    else{bk=d.bke||{};bkDetails=d.bke_details||{};}
    var splitRec=getSplitData(bkDetails)||getSplitData(bk);
    var bkDetailsDisplay=Object.assign({},bkDetails);
    bkDetailsDisplay.active_split=splitDisplayLabel();
    if(splitRec){
        bkDetailsDisplay.split_bke=splitRec.bke;
        bkDetailsDisplay.split_rank=splitRec.rank;
        bkDetailsDisplay.split_pct=splitRec.pct;
    }
    var gp=ps.GP||ss.GP||null;
  function add(l,v){if(v!==null&&v!==undefined&&v!=='')hl.push({label:l,value:v});}
    add('MPG',p.mpg);add('PPG',p.ppg);add('APG',p.apg);add('RPG',p.rpg);
    add('TOV/G',perGame(ps.TOV,gp));add('STL/G',perGame(ps.STL,gp));add('BLK/G',perGame(ps.BLK,gp));
    add('FG%',fp(ss.FG_PCT!=null?ss.FG_PCT:am.FG_PCT));
    add('3P%',fp(ss.FG3_PCT!=null?ss.FG3_PCT:am.FG3_PCT));
    add('FT%',fp(ss.FT_PCT!=null?ss.FT_PCT:am.FT_PCT));
    add('PF/G',perGame(ps.PF,gp));
    add('DD',ss.DD2);add('TD',ss.TD3);
    add('+/-/G',perGame(ss.PLUS_MINUS,gp));

  var adv=[];
  function addAdv(l,v){if(v!==null&&v!==undefined&&v!=='')adv.push({label:l,value:v});}
  addAdv('USG%',p.usg);addAdv('TS%',p.ts);
  addAdv('ORTG',ps.ORTG);addAdv('DRTG',ps.DRTG);addAdv('NET RTG',ps.NET_RTG);
  addAdv('WS',ln.WS);addAdv('OWS',ln.OWS);addAdv('DWS',ln.DWS);
  addAdv('BPM',ln.BPM);addAdv('OBPM',ln.OBPM);addAdv('DBPM',ln.DBPM);
  addAdv('VORP',ln.VORP);addAdv('GmSc',ln.GMSC_AVG);
  addAdv('RAPM',ra.RAPM);addAdv('ORAPM',ra.ORAPM);addAdv('DRAPM',ra.DRAPM);
  addAdv('xRAPM',xr.xRAPM);addAdv('O_xRAPM',xr.O_xRAPM);addAdv('D_xRAPM',xr.D_xRAPM);

  var arch='<div class="msec"><h3>Archetypes</h3><div class="sg">'
    +'<div class="sc"><div class="lb">Offense</div><div class="vl">'+p.off_archetype+'</div></div>'
    +'<div class="sc"><div class="lb">Off Fit</div><div class="vl">'+fv(p.off_confidence)+'%</div></div>'
    +'<div class="sc"><div class="lb">Off Eff</div><div class="vl">'+fv(p.off_effectiveness)+'%</div></div>'
    +'<div class="sc"><div class="lb">Defense</div><div class="vl">'+(p.def_archetype||'Unknown')+'</div></div>'
    +'</div></div>';

  document.getElementById('mBody').innerHTML=
    (hl.length?'<div class="msec"><h3>Season Highlights</h3><h4 class="msec-sub">General Box Score</h4>'+renderSG(hl)+(adv.length?'<h4 class="msec-sub">Advanced Statistics</h4>'+renderSG(adv):'')+'</div>':'')
        +renderBKEHighlights(bk, curBkeVersion)
    +arch
        +renderPositionEstimate(pr)
    +renderReasons(d.off_reasons,d.def_reasons)
    +renderKV('Profile',pr)
    +renderKV('Archetype Model Stats',d.archetype_model||{})
    +renderKV('Profile Stats',d.profile_stats||{})
    +renderKV('RAPM',d.rapm||{})
    +renderKV('xRAPM',d.xrapm||{})
    +renderKV('xRAPM v2',d.xrapm_v2||{})
    +renderKV('Linear Metrics (BKE-computed)',d.linear||{})
    +renderKV('BKE Details ('+curBkeVersion.toUpperCase()+' · '+splitDisplayLabel()+')',bkDetailsDisplay);

  /* add collapsible toggles to each section */
  document.querySelectorAll('#mBody .msec').forEach(function(sec){
    var h3=sec.querySelector('h3');
    if(!h3)return;
    var span=document.createElement('span');
    span.className='collapse-toggle';
    span.textContent='\u25BC';
    h3.insertBefore(span,h3.firstChild);
    h3.addEventListener('click',function(){sec.classList.toggle('collapsed');});
  });

  /* reset tabs */
  document.getElementById('tabStats').classList.add('active');
  document.getElementById('tabGraphs').classList.remove('active');
  document.getElementById('mBody').style.display='';
  document.getElementById('mGraphs').style.display='none';
  document.getElementById('mGraphs').innerHTML='';
  window._graphsRendered=false;

  updateNavArrows();
  document.getElementById('modal').classList.add('open');
  document.body.style.overflow='hidden';
}
function closeModal(e,force){
  if(e)e.stopPropagation();
  var m=document.getElementById('modal');
  if(!force&&e&&!e.target.classList.contains('mo'))return;
  m.classList.remove('open');document.body.style.overflow='';
}
document.addEventListener('keydown',function(e){if(e.key==='Escape')closeModal(null,true);});

/* ---- filter / sort ---- */
function setSort(el){
    var nextSort=el.dataset.sort;
    if(nextSort===curSort){
        curSortDir=(curSortDir==='desc')?'asc':'desc';
    }else{
        curSort=nextSort;
        curSortDir='desc';
    }
    refreshSortUI();
    filterPlayers();
}
function setPositionChip(el){
    document.querySelectorAll('.chip-btn').forEach(function(b){b.classList.remove('active');});
    el.classList.add('active');
    curPos=el.dataset.pos||'';
    filterPlayers();
}
function resetFilters(){
  document.getElementById('search').value='';
  document.getElementById('offFilter').value='';
  document.getElementById('defFilter').value='';
  document.getElementById('seasonFilter').value='';
    curPos='';
    document.querySelectorAll('.chip-btn').forEach(function(b){b.classList.remove('active');});
    var defaultChip=document.querySelector('.chip-btn[data-pos=""]');
    if(defaultChip)defaultChip.classList.add('active');
    curSort='ppg';
    curSortDir='desc';
    curBkeVersion='v27';
    curBkeSplit='split_60_40';
    document.querySelectorAll('.ver-btn[data-ver]').forEach(function(b){b.classList.toggle('active',b.dataset.ver==='v27');});
    document.querySelectorAll('.split-btn').forEach(function(b){b.classList.toggle('active',b.dataset.split==='split_60_40');});
    refreshSortUI();
    filterPlayers();
}
function filterPlayers(){
  showLoad();
  var s=document.getElementById('search').value.toLowerCase();
  var of=document.getElementById('offFilter').value;
  var df=document.getElementById('defFilter').value;
  var sn=document.getElementById('seasonFilter').value;
  setTimeout(function(){
    var f=cards.filter(function(p){
      if(s&&p.name.toLowerCase().indexOf(s)===-1)return false;
      if(of&&p.off_archetype!==of)return false;
      if(df&&p.def_archetype!==df)return false;
      if(sn&&p.season!==sn)return false;
            if(curPos&&p.position_primary!==curPos)return false;
      return true;
    });
        f.sort(function(a,b){
            var av,bv;
            if(curSort==='transformed_bke'||curSort==='transformed_obke'||curSort==='transformed_dbke'){
                var fld=curSort==='transformed_bke'?'bke':curSort==='transformed_obke'?'obke':'dbke';
                av=getBkeField(a,fld);bv=getBkeField(b,fld);
                av=(av!==null&&av!==undefined)?Number(av):null;
                bv=(bv!==null&&bv!==undefined)?Number(bv):null;
            }else{
                av=(a[curSort]!==null&&a[curSort]!==undefined)?Number(a[curSort]):null;
                bv=(b[curSort]!==null&&b[curSort]!==undefined)?Number(b[curSort]):null;
            }
            av=isFinite(av)?av:null;
            bv=isFinite(bv)?bv:null;
            if(av===null&&bv===null)return 0;
            if(av===null)return 1;
            if(bv===null)return -1;
            return curSortDir==='desc' ? (bv-av) : (av-bv);
        });
    renderPlayers(f);
  },0);
}

/* ---- season navigation ---- */
function getPlayerSeasonKeys(key){
    var pid=key.split('::')[0];
    var keys=[];
    for(var i=0;i<cards.length;i++){if(cards[i].key.split('::')[0]===pid)keys.push(cards[i].key);}
    keys.sort(function(a,b){
        var sa=(a.split('::')[1]||'');
        var sb=(b.split('::')[1]||'');
        if(sa!==sb)return sa.localeCompare(sb);
        var ca=cardByKey[a]||{};
        var cb=cardByKey[b]||{};
        var na=Number(ca.stint_number),nb=Number(cb.stint_number);
        if(!isFinite(na)||na<1)na=1;
        if(!isFinite(nb)||nb<1)nb=1;
        return na-nb;
    });
    return keys;
}
function updateNavArrows(){
    if(!curModalKey)return;
    var keys=getPlayerSeasonKeys(curModalKey);
    var idx=keys.indexOf(curModalKey);
    var prev=document.getElementById('modalPrev'),next=document.getElementById('modalNext');
    if(prev)prev.disabled=(idx<=0);
    if(next)next.disabled=(idx>=keys.length-1);
}
function navigateModal(dir){
    if(!curModalKey)return;
    var keys=getPlayerSeasonKeys(curModalKey);
    var idx=keys.indexOf(curModalKey);
    if(idx===-1)return;
    var ni=idx+dir;
    if(ni<0||ni>=keys.length)return;
    openModal(keys[ni]);
}

/* ---- comparison feature ---- */
/* precompute league max for radar chart normalization */
var leagueMax={ppg:1,apg:1,rpg:1};
for(var _i=0;_i<cards.length;_i++){
    if(cards[_i].ppg>leagueMax.ppg)leagueMax.ppg=cards[_i].ppg;
    if(cards[_i].apg>leagueMax.apg)leagueMax.apg=cards[_i].apg;
    if(cards[_i].rpg>leagueMax.rpg)leagueMax.rpg=cards[_i].rpg;
}
var radarDims=[
    {label:'PPG',fn:function(c){return c.ppg/leagueMax.ppg*100;}},
    {label:'APG',fn:function(c){return c.apg/leagueMax.apg*100;}},
    {label:'RPG',fn:function(c){return c.rpg/leagueMax.rpg*100;}},
    {label:'USG%',fn:function(c){var v=Number(c.usg);return isFinite(v)?v/45*100:0;}},
    {label:'TS%',fn:function(c){var v=Number(c.ts);return isFinite(v)?v/75*100:0;}},
    {label:'BKE',fn:function(c){var v=getBkeField(c,'pct');return v!=null?Number(v):0;}},
    {label:'Off Fit',fn:function(c){return c.off_confidence||0;}},
    {label:'Def Fit',fn:function(c){return c.def_confidence||0;}}
];

function openCmp(){
    cmpSel={1:null,2:null};
    document.getElementById('cmpSearch1').value='';
    document.getElementById('cmpSearch2').value='';
    document.getElementById('cmpBody').innerHTML='<p style="color:#666;text-align:center">Select two players to compare</p>';
    document.querySelectorAll('.cmp-dd').forEach(function(d){d.classList.remove('open');});
    document.getElementById('cmpModal').classList.add('open');
    document.body.style.overflow='hidden';
}
function closeCmp(){
    document.getElementById('cmpModal').classList.remove('open');
    document.body.style.overflow='';
}
function cmpAutocomplete(slot){
    var inp=document.getElementById('cmpSearch'+slot);
    var dd=document.getElementById('cmpDd'+slot);
    var q=inp.value.toLowerCase().trim();
    if(q.length<2){dd.classList.remove('open');return;}
    var matches=[];
    for(var i=0;i<cards.length&&matches.length<20;i++){
        if(cards[i].name.toLowerCase().indexOf(q)!==-1)matches.push(cards[i]);
    }
    if(!matches.length){dd.innerHTML='<div class="cmp-dd-item" style="color:#555">No results</div>';dd.classList.add('open');return;}
    var html='';
    for(var i=0;i<matches.length;i++){
        var msfx=stintSuffix(matches[i]);
        html+='<div class="cmp-dd-item" onclick="cmpSelect('+slot+',\\''+matches[i].key+'\\')">'+matches[i].name+'<span class="cmp-dd-season">'+matches[i].season+(msfx?' '+msfx:'')+(matches[i].team?' \u00b7 '+matches[i].team:'')+'</span></div>';
    }
    dd.innerHTML=html;dd.classList.add('open');
}
function cmpSelect(slot,key){
    var card=null;for(var i=0;i<cards.length;i++){if(cards[i].key===key){card=cards[i];break;}}
    if(!card)return;
    cmpSel[slot]=card;
    var sfx=stintSuffix(card);
    document.getElementById('cmpSearch'+slot).value=card.name+' ('+card.season+(sfx?' '+sfx:'')+')';
    document.getElementById('cmpDd'+slot).classList.remove('open');
    if(cmpSel[1]&&cmpSel[2])runComparison();
}
document.addEventListener('click',function(e){if(!e.target.closest('.cmp-input'))document.querySelectorAll('.cmp-dd').forEach(function(d){d.classList.remove('open');});});

function runComparison(){
    var p1=cmpSel[1],p2=cmpSel[2];
    if(!p1||!p2)return;
    var stats=[
        {label:'PPG',v1:p1.ppg,v2:p2.ppg,higher:true},
        {label:'APG',v1:p1.apg,v2:p2.apg,higher:true},
        {label:'RPG',v1:p1.rpg,v2:p2.rpg,higher:true},
        {label:'MPG',v1:p1.mpg,v2:p2.mpg,higher:true},
        {label:'USG%',v1:p1.usg,v2:p2.usg,higher:true},
        {label:'TS%',v1:p1.ts,v2:p2.ts,higher:true},
        {label:'BKE',v1:getBkeField(p1,'pct'),v2:getBkeField(p2,'pct'),higher:true},
        {label:'OBKE',v1:getBkeField(p1,'obke_pct'),v2:getBkeField(p2,'obke_pct'),higher:true},
        {label:'DBKE',v1:getBkeField(p1,'dbke_pct'),v2:getBkeField(p2,'dbke_pct'),higher:true},
        {label:'Off Fit',v1:p1.off_confidence,v2:p2.off_confidence,higher:true},
        {label:'Off Eff',v1:p1.off_effectiveness,v2:p2.off_effectiveness,higher:true},
        {label:'Def Fit',v1:p1.def_confidence,v2:p2.def_confidence,higher:true}
    ];

    var p1sfx=stintSuffix(p1),p2sfx=stintSuffix(p2);
    var hdr='<div class="cmp-hdr-row"><div class="cmp-player-hdr" style="color:#ff6b6b">'+p1.name+'<br><span style="font-size:12px;color:#888">'+p1.season+(p1sfx?' '+p1sfx:'')+(p1.team?' \u00b7 '+p1.team:'')+'</span></div><div class="cmp-label" style="font-size:10px;color:#555">STAT</div><div class="cmp-player-hdr" style="color:#4ecdc4">'+p2.name+'<br><span style="font-size:12px;color:#888">'+p2.season+(p2sfx?' '+p2sfx:'')+(p2.team?' \u00b7 '+p2.team:'')+'</span></div></div>';

    var rows='';
    for(var i=0;i<stats.length;i++){
        var s=stats[i];
        var n1=(s.v1!==null&&s.v1!==undefined)?Number(s.v1):null;
        var n2=(s.v2!==null&&s.v2!==undefined)?Number(s.v2):null;
        var b1='',b2='';
        if(n1!==null&&n2!==null&&isFinite(n1)&&isFinite(n2)){
            if(s.higher){if(n1>n2)b1='better';else if(n2>n1)b2='better';}
            else{if(n1<n2)b1='better';else if(n2<n1)b2='better';}
        }
        rows+='<div class="cmp-stats-row"><div class="cmp-val left '+b1+'">'+fv(n1)+'</div><div class="cmp-label">'+s.label+'</div><div class="cmp-val right '+b2+'">'+fv(n2)+'</div></div>';
    }

    /* archetype rows (non-numeric) */
    rows+='<div class="cmp-stats-row"><div class="cmp-val left">'+p1.off_archetype+'</div><div class="cmp-label">Off Arch</div><div class="cmp-val right">'+p2.off_archetype+'</div></div>';
    rows+='<div class="cmp-stats-row"><div class="cmp-val left">'+(p1.def_archetype||'Unknown')+'</div><div class="cmp-label">Def Arch</div><div class="cmp-val right">'+(p2.def_archetype||'Unknown')+'</div></div>';

    document.getElementById('cmpBody').innerHTML=hdr+rows+'<div class="radar-legend"><span><span class="radar-dot" style="background:#ff6b6b"></span>'+p1.name+'</span><span><span class="radar-dot" style="background:#4ecdc4"></span>'+p2.name+'</span></div><div class="radar-wrap"><div id="radarSvg"></div></div>';
    drawRadar(p1,p2);
}

function drawRadar(p1,p2){
    var cx=160,cy=160,r=120,n=radarDims.length;
    var as=2*Math.PI/n;
    var svg='<svg width="320" height="320" viewBox="0 0 320 320">';
    /* reference circles */
    [0.25,0.5,0.75,1.0].forEach(function(pct){
        var cr=r*pct;
        svg+='<circle cx="'+cx+'" cy="'+cy+'" r="'+cr+'" fill="none" stroke="#2b4475" stroke-width="0.5" />';
    });
    /* axis lines + labels */
    for(var i=0;i<n;i++){
        var a=-Math.PI/2+i*as;
        var ex=cx+r*Math.cos(a),ey=cy+r*Math.sin(a);
        svg+='<line x1="'+cx+'" y1="'+cy+'" x2="'+ex.toFixed(1)+'" y2="'+ey.toFixed(1)+'" stroke="#1e2a50" stroke-width="1" />';
        var lx=cx+(r+22)*Math.cos(a),ly=cy+(r+22)*Math.sin(a);
        svg+='<text x="'+lx.toFixed(1)+'" y="'+ly.toFixed(1)+'" fill="#7c8bb5" font-size="10" text-anchor="middle" dominant-baseline="central">'+radarDims[i].label+'</text>';
    }
    /* polygon helper */
    function poly(card,color,op){
        var pts=[];
        for(var i=0;i<n;i++){
            var a=-Math.PI/2+i*as;
            var v=Math.min(Math.max(radarDims[i].fn(card),0),100)/100;
            pts.push((cx+r*v*Math.cos(a)).toFixed(1)+','+(cy+r*v*Math.sin(a)).toFixed(1));
        }
        return '<polygon points="'+pts.join(' ')+'" fill="'+color+'" fill-opacity="'+op+'" stroke="'+color+'" stroke-width="2" />';
    }
    svg+=poly(p2,'#4ecdc4',0.15);
    svg+=poly(p1,'#ff6b6b',0.15);
    /* dots */
    for(var i=0;i<n;i++){
        var a=-Math.PI/2+i*as;
        var v1=Math.min(Math.max(radarDims[i].fn(p1),0),100)/100;
        var v2=Math.min(Math.max(radarDims[i].fn(p2),0),100)/100;
        svg+='<circle cx="'+(cx+r*v1*Math.cos(a)).toFixed(1)+'" cy="'+(cy+r*v1*Math.sin(a)).toFixed(1)+'" r="3" fill="#ff6b6b" />';
        svg+='<circle cx="'+(cx+r*v2*Math.cos(a)).toFixed(1)+'" cy="'+(cy+r*v2*Math.sin(a)).toFixed(1)+'" r="3" fill="#4ecdc4" />';
    }
    svg+='</svg>';
    document.getElementById('radarSvg').innerHTML=svg;
}

/* ---- modal tab + graphs ---- */
function switchModalTab(tab){
    var isStats=(tab==='stats');
    document.getElementById('tabStats').classList.toggle('active',isStats);
    document.getElementById('tabGraphs').classList.toggle('active',!isStats);
    document.getElementById('mBody').style.display=isStats?'':'none';
    document.getElementById('mGraphs').style.display=isStats?'none':'';
    if(!isStats&&!window._graphsRendered){
        window._graphsRendered=true;
        renderGraphsTab();
    }
}

function renderGraphsTab(){
    if(!curModalKey)return;
    var p=null;for(var i=0;i<cards.length;i++){if(cards[i].key===curModalKey){p=cards[i];break;}}
    if(!p)return;
    var d=getDetail(curModalKey);
    var pid=p.id||curModalKey.split('::')[0];
    var curSeason=p.season||curModalKey.split('::')[1];
    var html='';
    /* 1. YoY progression */
    html+=renderYoYSection(pid,curSeason);
    /* 2. Offensive BKE layers */
    var bkd=d.bke_details||{};
    html+=renderLayerSection('Offensive BKE Layer Breakdown',[
        {label:'Portable Talent',value:bkd.layer1_offensive_raw},
        {label:'Role Utilization',value:bkd.layer2_rue_raw},
        {label:'Archetype Elevation',value:bkd.layer3_off_elevation_raw},
        {label:'Scheme Bonus',value:bkd.layer4_scheme_bonus_raw}
    ],'off');
    /* 3. Defensive BKE layers */
    html+=renderLayerSection('Defensive BKE Layer Breakdown',[
        {label:'Defensive Portable',value:bkd.layer1_defensive_raw},
        {label:'Def Elevation',value:bkd.layer3_def_elevation_raw},
        {label:'Scheme Bonus',value:bkd.layer4_scheme_bonus_raw}
    ],'def');
    /* 4. Position breakdown */
    html+=renderPositionSection(d.profile||{});
    /* 5. Defensive archetype radar */
    html+=renderDefArchSection(p,d);
    /* 6. Playtype breakdown (moved from Stats) */
    html+=renderPT(d.playtypes);
    document.getElementById('mGraphs').innerHTML=html;
    drawYoYChart();
}

/* ---- YoY progression chart ---- */
function renderYoYSection(pid,curSeason){
    var allSeasons=['2022-23','2023-24','2024-25'];
    var seriesDefs=[
        {key:'BKE%',color:'#f4a261',fn:function(det){return (det.bke||{}).final_BKE_percentile;}},
        {key:'OBKE%',color:'#ff6b6b',fn:function(det){return (det.bke||{}).final_OBKE_percentile;}},
        {key:'DBKE%',color:'#4ecdc4',fn:function(det){return (det.bke||{}).final_DBKE_percentile;}},
        {key:'BPM',color:'#a78bfa',fn:function(det){return (det.linear||{}).BPM;}},
        {key:'WS',color:'#34d399',fn:function(det){return (det.linear||{}).WS;}},
        {key:'RAPM',color:'#fb923c',fn:function(det){return (det.rapm||{}).RAPM;}}
    ];
    var seriesData={};
    for(var si=0;si<seriesDefs.length;si++){
        var sd=seriesDefs[si];var pts=[];
        for(var j=0;j<allSeasons.length;j++){
            var k=seasonLookupKey(pid,allSeasons[j]);
            var det=getDetail(k);
            if(!det||!Object.keys(det).length){pts.push(null);continue;}
            var val=sd.fn(det);
            if(val===null||val===undefined||val===''){pts.push(null);continue;}
            var n=Number(val);pts.push(isFinite(n)?n:null);
        }
        seriesData[sd.key]=pts;
    }
    window._yoyData={seasons:allSeasons,series:seriesDefs,data:seriesData,curSeason:curSeason};
    window._yoyActive={};
    for(var si=0;si<seriesDefs.length;si++)window._yoyActive[seriesDefs[si].key]=true;
    var filters='';
    for(var si=0;si<seriesDefs.length;si++){
        var sd=seriesDefs[si];var hasData=false;
        for(var j=0;j<seriesData[sd.key].length;j++){if(seriesData[sd.key][j]!==null){hasData=true;break;}}
        if(!hasData)continue;
        filters+='<div class="yoy-filter" data-key="'+sd.key+'" onclick="toggleYoY(this)"><span class="yoy-dot" style="background:'+sd.color+'"></span>'+sd.key+'</div>';
    }
    return '<div class="gfx-sec"><div class="gfx-title">Year-over-Year Progression</div>'
        +'<div class="yoy-filters" id="yoyFilters">'+filters+'</div>'
        +'<div id="yoyChartWrap" style="display:flex;justify-content:center"></div></div>';
}

function toggleYoY(el){
    var key=el.dataset.key;
    window._yoyActive[key]=!window._yoyActive[key];
    el.classList.toggle('off',!window._yoyActive[key]);
    drawYoYChart();
}

function drawYoYChart(){
    var yd=window._yoyData;if(!yd)return;
    var W=560,H=220,pad={t:20,r:30,b:40,l:55};
    var pw=W-pad.l-pad.r,ph=H-pad.t-pad.b;
    var seasons=yd.seasons,nS=seasons.length;
    var allVals=[];
    for(var si=0;si<yd.series.length;si++){
        if(!window._yoyActive[yd.series[si].key])continue;
        var pts=yd.data[yd.series[si].key];
        for(var j=0;j<pts.length;j++){if(pts[j]!==null)allVals.push(pts[j]);}
    }
    if(!allVals.length){document.getElementById('yoyChartWrap').innerHTML='<p style="color:#666;font-size:14px">No data available</p>';return;}
    var yMin=Math.min.apply(null,allVals),yMax=Math.max.apply(null,allVals);
    var yRange=yMax-yMin;if(yRange<0.01){yMin-=1;yMax+=1;yRange=2;}
    yMin-=yRange*0.1;yMax+=yRange*0.1;yRange=yMax-yMin;
    function xP(idx){return pad.l+idx*(pw/(nS-1));}
    function yP(val){return pad.t+ph-(val-yMin)/yRange*ph;}
    var svg='<svg width="'+W+'" height="'+H+'" viewBox="0 0 '+W+' '+H+'">';
    /* grid */
    for(var gi=0;gi<=4;gi++){
        var gy=pad.t+ph*gi/4;var gv=yMax-(yMax-yMin)*gi/4;
        svg+='<line x1="'+pad.l+'" y1="'+gy.toFixed(1)+'" x2="'+(W-pad.r)+'" y2="'+gy.toFixed(1)+'" stroke="#1e2a50" stroke-width="0.5"/>';
        svg+='<text x="'+(pad.l-6)+'" y="'+(gy+4).toFixed(1)+'" fill="#7c8bb5" font-size="10" text-anchor="end">'+gv.toFixed(1)+'</text>';
    }
    /* season labels */
    for(var j=0;j<nS;j++){
        var sx=xP(j);
        svg+='<text x="'+sx.toFixed(1)+'" y="'+(H-8)+'" fill="#7c8bb5" font-size="11" text-anchor="middle">'+seasons[j]+'</text>';
        svg+='<line x1="'+sx.toFixed(1)+'" y1="'+pad.t+'" x2="'+sx.toFixed(1)+'" y2="'+(pad.t+ph)+'" stroke="#1e2a50" stroke-width="0.5" stroke-dasharray="3,3"/>';
    }
    /* lines + dots */
    for(var si=0;si<yd.series.length;si++){
        var sd=yd.series[si];if(!window._yoyActive[sd.key])continue;
        var pts=yd.data[sd.key];var pp=[];
        for(var j=0;j<pts.length;j++){if(pts[j]!==null)pp.push({x:xP(j),y:yP(pts[j]),s:seasons[j],v:pts[j]});}
        if(pp.length>=2){
            var d2='M'+pp[0].x.toFixed(1)+','+pp[0].y.toFixed(1);
            for(var k=1;k<pp.length;k++)d2+='L'+pp[k].x.toFixed(1)+','+pp[k].y.toFixed(1);
            svg+='<path d="'+d2+'" fill="none" stroke="'+sd.color+'" stroke-width="2" stroke-linecap="round"/>';
        }
        for(var k=0;k<pp.length;k++){
            var isCur=(pp[k].s===yd.curSeason);
            svg+='<circle cx="'+pp[k].x.toFixed(1)+'" cy="'+pp[k].y.toFixed(1)+'" r="'+(isCur?6:4)+'" fill="'+sd.color+'"'+(isCur?' stroke="#fff" stroke-width="2"':'')+'/>';
            if(isCur)svg+='<text x="'+pp[k].x.toFixed(1)+'" y="'+(pp[k].y-10).toFixed(1)+'" fill="'+sd.color+'" font-size="11" font-weight="700" text-anchor="middle">'+pp[k].v.toFixed(1)+'</text>';
        }
    }
    svg+='</svg>';
    document.getElementById('yoyChartWrap').innerHTML=svg;
}

/* ---- BKE layer breakdown (diverging bars) ---- */
function renderLayerSection(title,layers,type){
    var valid=[];
    for(var i=0;i<layers.length;i++){if(layers[i].value!==null&&layers[i].value!==undefined)valid.push(layers[i]);}
    if(!valid.length)return '<div class="gfx-sec"><div class="gfx-title">'+title+'</div><p style="color:#666;font-size:13px">No layer data available</p></div>';
    var maxAbs=0;
    for(var i=0;i<valid.length;i++){var a=Math.abs(Number(valid[i].value));if(a>maxAbs)maxAbs=a;}
    if(maxAbs<0.001)maxAbs=1;
    var bars='';
    for(var i=0;i<valid.length;i++){
        var v=Number(valid[i].value);
        var pct=Math.abs(v)/maxAbs*45;
        var isPos=v>=0;
        var color=type==='off'?(isPos?'#ff6b6b':'#5a2d2d'):(isPos?'#4ecdc4':'#2d5a57');
        var left=isPos?50:(50-pct);
        bars+='<div class="layer-row"><span class="layer-label">'+valid[i].label+'</span>'
            +'<div class="layer-bar-wrap"><div class="layer-center"></div>'
            +'<div class="layer-bar" style="left:'+left.toFixed(1)+'%;width:'+pct.toFixed(1)+'%;background:'+color+'"></div></div>'
            +'<span class="layer-val" style="color:'+color+'">'+(isPos?'+':'')+v.toFixed(3)+'</span></div>';
    }
    return '<div class="gfx-sec"><div class="gfx-title">'+title+'</div>'+bars+'</div>';
}

/* ---- position breakdown (stacked bar) ---- */
function renderPositionSection(pr){
    var positions=[
        {key:'pct_pg',label:'PG',color:'#e63946'},
        {key:'pct_sg',label:'SG',color:'#f4a261'},
        {key:'pct_sf',label:'SF',color:'#2a9d8f'},
        {key:'pct_pf',label:'PF',color:'#264653'},
        {key:'pct_c',label:'C',color:'#457b9d'}
    ];
    var total=0;var hasData=false;
    for(var i=0;i<positions.length;i++){
        var v=pr[positions[i].key];
        if(v!==null&&v!==undefined&&v!==''){
            v=Number(v);if(Math.abs(v)<=1.5)v=v*100;
            positions[i].pct=v;total+=v;hasData=true;
        }else{positions[i].pct=0;}
    }
    if(!hasData)return '<div class="gfx-sec"><div class="gfx-title">Position Breakdown</div><p style="color:#666;font-size:13px">No position data</p></div>';
    if(total<1)total=100;
    var bar='<div class="pos-bar-wrap">';
    var legend='<div class="pos-legend">';
    for(var i=0;i<positions.length;i++){
        var wp=positions[i].pct/total*100;
        if(wp<0.5)continue;
        bar+='<div class="pos-seg" style="width:'+wp.toFixed(1)+'%;background:'+positions[i].color+'">'+(wp>8?positions[i].label:'')+'</div>';
        legend+='<div class="pos-legend-item"><span class="pos-legend-dot" style="background:'+positions[i].color+'"></span>'+positions[i].label+': '+positions[i].pct.toFixed(1)+'%</div>';
    }
    bar+='</div>';legend+='</div>';
    return '<div class="gfx-sec"><div class="gfx-title">Position Breakdown</div>'
        +'<div style="color:#bac7e6;font-size:14px;margin-bottom:6px">Primary: '+(pr.position||'Unknown')+'</div>'
        +bar+legend+'</div>';
}

/* ---- defensive archetype radar ---- */
function renderDefArchSection(p,d){
    var bv=d.bke_v28||{};var bk=d.bke||{};var ln=d.linear||{};var ra=d.rapm||{};
    var ps=d.profile_stats||{};var ss=d.season_stats||{};var gp=ps.GP||ss.GP||1;
    function zTo100(z){if(z===null||z===undefined)return null;var v=Number(z);if(!isFinite(v))return null;return Math.min(100,Math.max(0,v*20+50));}
    function pctVal(v){if(v===null||v===undefined)return null;var n=Number(v);if(!isFinite(n))return null;return Math.min(100,Math.max(0,n));}
    function rateNorm(val,maxV){if(val===null||val===undefined)return null;var n=Number(val);if(!isFinite(n))return null;return Math.min(100,Math.max(0,n/maxV*100));}
    var dims=[
        {label:'DBKE',value:pctVal(bk.final_DBKE_percentile)},
        {label:'Def Confidence',value:pctVal(p.def_confidence)},
        {label:'Def Playmaking',value:zTo100(bv.dim_defensive_playmaking_z)},
        {label:'Def Impact',value:zTo100(bv.dim_defensive_impact_z)},
        {label:'Versatility',value:zTo100(bv.dim_defensive_versatility_z)}
    ];
    if(dims[2].value===null){var blk=perGame(ps.BLK||ss.BLK,gp);dims[2]={label:'BLK/G',value:rateNorm(blk,3.5)};}
    if(dims[3].value===null){var drapm=(ra.DRAPM!==undefined)?ra.DRAPM:null;dims[3]={label:'DRAPM',value:drapm!==null?zTo100(drapm/3):null};}
    if(dims[4].value===null){var stl=perGame(ps.STL||ss.STL,gp);dims[4]={label:'STL/G',value:rateNorm(stl,2.5)};}
    var validDims=[];for(var i=0;i<dims.length;i++){if(dims[i].value!==null)validDims.push(dims[i]);}
    if(validDims.length<3)return '<div class="gfx-sec"><div class="gfx-title">Defensive Archetype: '+(p.def_archetype||'Unknown')+'</div><p style="color:#666;font-size:13px">Insufficient defensive data</p></div>';
    var cx=130,cy=130,radius=95,n=validDims.length;
    var as=2*Math.PI/n;
    var svg='<svg width="260" height="260" viewBox="0 0 260 260">';
    [0.25,0.5,0.75,1.0].forEach(function(pct){svg+='<circle cx="'+cx+'" cy="'+cy+'" r="'+(radius*pct).toFixed(1)+'" fill="none" stroke="#2b4475" stroke-width="0.5"/>';});
    for(var i=0;i<n;i++){
        var a=-Math.PI/2+i*as;
        var ex=cx+radius*Math.cos(a),ey=cy+radius*Math.sin(a);
        svg+='<line x1="'+cx+'" y1="'+cy+'" x2="'+ex.toFixed(1)+'" y2="'+ey.toFixed(1)+'" stroke="#1e2a50" stroke-width="1"/>';
        var lx=cx+(radius+18)*Math.cos(a),ly=cy+(radius+18)*Math.sin(a);
        svg+='<text x="'+lx.toFixed(1)+'" y="'+ly.toFixed(1)+'" fill="#7c8bb5" font-size="9" text-anchor="middle" dominant-baseline="central">'+validDims[i].label+'</text>';
    }
    var pts=[];
    for(var i=0;i<n;i++){
        var a=-Math.PI/2+i*as;var v=Math.min(Math.max(validDims[i].value,0),100)/100;
        pts.push((cx+radius*v*Math.cos(a)).toFixed(1)+','+(cy+radius*v*Math.sin(a)).toFixed(1));
    }
    svg+='<polygon points="'+pts.join(' ')+'" fill="#4ecdc4" fill-opacity="0.2" stroke="#4ecdc4" stroke-width="2"/>';
    for(var i=0;i<n;i++){
        var a=-Math.PI/2+i*as;var v=Math.min(Math.max(validDims[i].value,0),100)/100;
        svg+='<circle cx="'+(cx+radius*v*Math.cos(a)).toFixed(1)+'" cy="'+(cy+radius*v*Math.sin(a)).toFixed(1)+'" r="4" fill="#4ecdc4"/>';
    }
    svg+='</svg>';
    return '<div class="gfx-sec"><div class="gfx-title">Defensive Archetype: '+(p.def_archetype||'Unknown')+'</div>'
        +'<div style="display:flex;justify-content:center">'+svg+'</div></div>';
}

initSortUI();
filterPlayers();
</script>
</body>
</html>'''

    # Step 1: Convert template braces in CSS ONLY (not JS, which has
    # legitimate }} sequences for nested blocks like  for(){if(){...}} )
    import re
    html = re.sub(
        r'(<style>)(.*?)(</style>)',
        lambda m: m.group(1) + m.group(2).replace('{{', '{').replace('}}', '}') + m.group(3),
        html_template,
        flags=re.DOTALL,
    )
    # Step 2: Inject JSON AFTER brace conversion (prevents corruption)
    html = html.replace("__CARDS_JSON__", cards_json)
    html = html.replace("__DETAILS_JSON__", details_json)
    return html


def main():
    print("Loading data for player data browser...")
    off_df, def_df = load_archetypes()
    profiles_df = load_player_profiles()
    bios_df     = load_player_bios()
    pos_est_df  = load_position_estimates()
    team_map    = load_team_map()
    rapm_df     = load_rapm()
    xrapm_df    = load_xrapm()
    xrapm_v2_df = load_xrapm_v2()
    linear_df   = load_linear_metrics()
    bke_map     = load_bke_scores()
    bke_v28_map = load_bke_v28_scores()
    bke_v30_map = load_bke_v30_scores()
    season_stats_df = load_player_season_stats()
    salaries_df = load_player_salaries()

    print(f"  Offensive archetypes: {len(off_df)}")
    print(f"  Defensive archetypes: {len(def_df)}")
    print(f"  Position estimates: {len(pos_est_df)}")
    print(f"  Linear metrics: {len(linear_df)}")
    print(f"  BKE v27 scores: {len(bke_map)}")
    print(f"  BKE v28 scores: {len(bke_v28_map)}")
    print(f"  BKE v30 scores: {len(bke_v30_map)}")
    print(f"  Season stats: {len(season_stats_df)}")
    print(f"  Salaries: {len(salaries_df)}")

    html = generate_html(
        off_df, def_df, profiles_df, bios_df, pos_est_df,
        rapm_df, xrapm_df, xrapm_v2_df, linear_df, season_stats_df, salaries_df, team_map, bke_map,
        bke_v28_map=bke_v28_map, bke_v30_map=bke_v30_map,
    )

    os.makedirs("app", exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\u2705 Saved to {OUTPUT_FILE}")
    print(f"Open in browser: file://{os.path.abspath(OUTPUT_FILE)}")


if __name__ == "__main__":
    main()
