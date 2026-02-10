"""
app/player_data_viewer.py
Optimised player data browser with full-stats modal popups.
Lightweight cards for fast rendering; click any card for all details.
Generates app/player_data.html.
"""

import pandas as pd
import numpy as np
import json
import os

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
    if "player_id" in df.columns:
        df["player_id"] = df["player_id"].astype(str)
    if "season" in df.columns:
        df["SEASON"] = df["season"].astype(str)
    return df

def load_player_profiles():
    return _load_parquet(f"{PROCESSED_DIR}/player_profiles_advanced.parquet")

def load_player_bios():
    return _load_parquet(f"{HISTORICAL_DIR}/players.parquet")

def load_rapm():
    return _load_parquet(f"{PROCESSED_DIR}/player_rapm.parquet")

def load_xrapm():
    return _load_parquet(f"{PROCESSED_DIR}/player_xrapm.parquet")

def load_xrapm_v2():
    return _load_parquet(f"{PROCESSED_DIR}/player_xrapm_v2.parquet")

def load_linear_metrics():
    return _load_parquet(f"{PROCESSED_DIR}/metrics_linear.parquet")

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
    vpctl = row.get('versatility_pctl', 0)
    dpctl = row.get('difficulty_pctl', 0)
    opp = row.get('avg_opponent_ppg', 0)
    elite = row.get('elite_matchup_pct', 0)
    blk = row.get('BLK_PER36', 0)
    stl = row.get('STL_PER36', 0)
    dres = row.get('d_results_pctl', 0)

    if arch == 'POA Defender':
        reasons += [f"Tough assignments (top {100*(1-dpctl):.0f}% difficulty)", f"Elite matchup %: {elite*100:.1f}%", f"Avg opponent PPG: {opp:.1f}"]
        if sec == 'Ball Hawk': reasons.append(f"High steals ({stl:.1f}/36)")
        elif sec == 'Switchable': reasons.append("Also versatile defender")
    elif arch == 'Switchable Defender':
        reasons += [f"High versatility (top {100*(1-vpctl):.0f}%)", "Guards multiple positions"]
        if dres >= 0.7: reasons.append("With strong results")
        if sec == 'Rim Protector': reasons.append("Can also protect the rim")
        elif sec == 'Lockdown': reasons.append("Elite defensive results")
    elif arch == 'Rim Protector':
        reasons += [f"High blocks ({blk:.1f} per 36)", "Interior defense anchor"]
        if sec == 'Switchable': reasons.append("Can also switch on perimeter")
        elif sec == 'Shot Blocker': reasons.append("Elite shot blocking")
    elif arch == 'Rotation Defender':
        reasons += ["Help/rotation defense", "Medium assignment difficulty"]
        if sec == 'Active Hands': reasons.append(f"Good steals ({stl:.1f}/36)")
        elif sec == 'Solid': reasons.append("Solid defensive results")
        else: reasons.append("Team defense contributor")
    elif arch == 'Off-Ball Defender':
        reasons.append("Gets easier assignments")
        if sec == 'Liability': reasons += ["Poor defensive results", "Hidden on defense"]
        else: reasons += ["Low versatility", f"Avg opponent PPG: {opp:.1f}"]
    elif arch == 'Insufficient Minutes':
        reasons += [f"Only {row.get('MIN',0):.0f} minutes", "Not enough data"]
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

def generate_html(off_df, def_df, profiles_df, bios_df, rapm_df, xrapm_df, xrapm_v2_df, linear_df, season_stats_df, team_map):
    """Generate optimised HTML viewer with lazy-loaded detail data."""

    def_cols = ['player_id', 'SEASON', 'defensive_archetype', 'defensive_secondary',
                'defensive_confidence', 'assignment_difficulty',
                'switch_score', 'versatility_pctl', 'avg_opponent_ppg',
                'elite_matchup_pct', 'difficulty_pctl', 'BLK_PER36', 'STL_PER36',
                'd_results_pctl']
    def_cols = [c for c in def_cols if c in def_df.columns]

    merged = off_df.merge(def_df[def_cols], on=['player_id', 'SEASON'], how='left')

    profiles_map = build_record_map(profiles_df, ["player_id", "SEASON"], ["player_name", "season"])
    rapm_map     = build_record_map(rapm_df,     ["player_id", "SEASON"], ["player_name", "season"])
    xrapm_map    = build_record_map(xrapm_df,    ["player_id", "SEASON"], ["player_name", "season"])
    xrapm_v2_map = build_record_map(xrapm_v2_df, ["player_id", "SEASON"], ["season"])
    linear_map   = build_record_map(linear_df,   ["player_id", "SEASON"], ["player_name", "season"])
    season_stats_map = build_record_map(season_stats_df, ["player_id", "SEASON"], ["PLAYER_NAME", "TEAM_ABBREVIATION", "season"])

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
        key = f"{pid}::{season}"

        card = {
            'key': key,
            'name': row['PLAYER_NAME'],
            'season': season,
            'team': row.get('TEAM_ABBREVIATION', ''),
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
        }
        cards.append(card)

        detail = {
            'profile': bios_map.get(pid, {}),
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
        }
        details[key] = detail

    # Deduplicate cards by key (traded players can appear twice)
    seen_keys = set()
    unique_cards = []
    for c in cards:
        if c['key'] not in seen_keys:
            seen_keys.add(c['key'])
            unique_cards.append(c)
    cards = unique_cards

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
.reset-btn{{padding:8px 14px;border:1px solid #e63946;border-radius:8px;background:#16213e;color:#e63946;cursor:pointer;font-size:13px;font-weight:600;transition:background .2s,color .2s}}
.reset-btn:hover{{background:#e63946;color:#fff}}
.stats-bar{{display:flex;gap:20px;margin-bottom:20px;color:#888}}
.stats-bar span{{background:#16213e;padding:8px 15px;border-radius:6px}}

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
.msec{{margin-top:16px;padding-top:12px;border-top:1px solid #1f2a4f}}
.msec h3{{margin:0 0 14px 0;font-size:15px;text-transform:uppercase;letter-spacing:1.5px;color:#6a789f;cursor:pointer;user-select:none;display:flex;align-items:center;gap:8px}}
.msec h4.msec-sub{{margin:18px 0 10px 0;font-size:13px;text-transform:uppercase;letter-spacing:1.2px;color:#4ecdc4;font-weight:600;border-bottom:1px solid #1f2a4f;padding-bottom:6px}}
.collapse-toggle{{transition:transform .2s;display:inline-block;font-size:13px;color:#7c8bb5}}
.msec.collapsed>:not(h3){{display:none}}
.msec.collapsed .collapse-toggle{{transform:rotate(-90deg)}}
.sg{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px}}
.sc{{background:#131f3e;border:1px solid #1f2a4f;border-radius:10px;padding:10px}}
.sc .lb{{font-size:13px;color:#6f7ea7;text-transform:uppercase;letter-spacing:1px}}
.sc .vl{{font-size:20px;font-weight:600;color:#e6f0ff;margin-top:6px}}
.kvt{{display:grid;grid-template-columns:1fr 1fr;gap:10px 24px;font-size:15px;color:#a5b3d6}}
.kvr{{display:flex;justify-content:space-between;gap:10px;border-bottom:1px dashed #1f2a4f;padding-bottom:4px}}
.kvk{{color:#7f8db4}}.kvv{{color:#d7e3ff;text-align:right}}
.reasons{{margin-top:6px}}
.reason{{font-size:15px;color:#aaa;padding:3px 0 3px 16px;position:relative}}
.reason::before{{content:"\\2022";position:absolute;left:0;color:#555}}
.pt-row{{display:flex;align-items:center;gap:10px;font-size:15px;margin-bottom:6px}}
.pt-name{{width:140px;color:#aaa;text-align:right;white-space:nowrap}}
.pt-bar{{flex:1;height:18px;background:#0a1428;border-radius:4px;overflow:hidden}}
.pt-fill{{height:100%;border-radius:3px;min-width:2px}}
.pt-val{{width:50px;color:#888;font-size:14px}}
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
  <span class="sort-btn" data-sort="off_confidence" onclick="setSort(this)">Fit</span>
  <span class="sort-btn" data-sort="off_effectiveness" onclick="setSort(this)">Effectiveness</span>
  <span class="sort-btn" data-sort="ts" onclick="setSort(this)">TS%</span>
  <span class="sort-btn" data-sort="usg" onclick="setSort(this)">USG%</span>
  <span class="reset-btn" onclick="resetFilters()">Reset All</span>
</div>

<div class="stats-bar"><span id="totalCount">Loading...</span></div>
<div id="loadBar" class="load-wrap active" aria-live="polite">
  <div class="load-label" id="loadLabel">Preparing...</div>
  <div class="load-track"><div class="load-fill" id="loadFill"></div></div>
  <div class="load-pct" id="loadPct">0 / 0 (0%)</div>
</div>

<div id="grid" class="grid"></div>

<div id="modal" class="mo" onclick="closeModal(event)">
  <div class="mc" role="dialog" aria-modal="true">
    <div class="mh">
      <div><div id="mTitle" class="mt"></div><div id="mSub" class="ms"></div></div>
      <button class="mx" onclick="closeModal(event,true)">&times;</button>
    </div>
    <div id="mBody"></div>
  </div>
</div>

<!-- Detail data stored as inert text; parsed lazily on first modal open -->
<script id="detailStore" type="application/json">__DETAILS_JSON__</script>

<script>
/* ---- card index (small, parsed immediately) ---- */
var cards=__CARDS_JSON__;
var curSort='ppg';
var CHUNK=80;
var fData=[],rendered=0,gen=0;
var _detailCache=null;  /* lazy-parsed on first modal open */

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
  return '<div class="card" onclick="openModal(\\''+p.key+'\\')">'
    +'<div class="card-hdr"><div><span class="pname">'+p.name+'</span>'+(p.team?'<span class="pteam"> &middot; '+p.team+'</span>':'')+'</div><div class="pmeta">'+p.season+'<br>'+p.mpg+' MPG</div></div>'
    +'<div class="srow">'
    +'<div class="st"><div class="sv">'+p.ppg+'</div><div class="sl">PPG</div></div>'
    +'<div class="st"><div class="sv">'+p.apg+'</div><div class="sl">APG</div></div>'
    +'<div class="st"><div class="sv">'+p.rpg+'</div><div class="sl">RPG</div></div>'
    +'<div class="st"><div class="sv">'+fs(p.usg,'%')+'</div><div class="sl">USG</div></div>'
    +'<div class="st"><div class="sv">'+fs(p.ts,'%')+'</div><div class="sl">TS%</div></div>'
    +'<div class="st"><div class="sv">'+fs(p.at_rim,'%')+'</div><div class="sl">RIM</div></div>'
    +'</div>'
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

function openModal(key){
  /* find the card record for basic info */
  var p=null;for(var i=0;i<cards.length;i++){if(cards[i].key===key){p=cards[i];break;}}
  if(!p)return;
  /* lazy-load detail record */
  var d=getDetail(key);
  var pr=d.profile||{};
  var team=p.team||pr.team_abbrev||'';
  document.getElementById('mTitle').textContent=p.name+' ('+p.season+')';
  var sub=team;
  if(pr.position)sub+=' \u00b7 '+pr.position;
  if(pr.age)sub+=' \u00b7 Age '+pr.age;
  if(pr.height)sub+=' \u00b7 '+pr.height;
  document.getElementById('mSub').textContent=sub;

  var hl=[];
  var ps=d.profile_stats||{};
  var ra=d.rapm||{};
  var xr=d.xrapm||{};
  var ln=d.linear||{};
    var ss=d.season_stats||{};
    var am=d.archetype_model||{};
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
    +arch
    +renderPT(d.playtypes)
    +renderReasons(d.off_reasons,d.def_reasons)
    +renderKV('Profile',pr)
    +renderKV('Archetype Model Stats',d.archetype_model||{})
    +renderKV('Profile Stats',d.profile_stats||{})
    +renderKV('RAPM',d.rapm||{})
    +renderKV('xRAPM',d.xrapm||{})
    +renderKV('xRAPM v2',d.xrapm_v2||{})
    +renderKV('Linear Metrics (BKE-computed)',d.linear||{});

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
  document.querySelectorAll('.sort-btn').forEach(function(b){b.classList.remove('active');});
  el.classList.add('active');curSort=el.dataset.sort;filterPlayers();
}
function resetFilters(){
  document.getElementById('search').value='';
  document.getElementById('offFilter').value='';
  document.getElementById('defFilter').value='';
  document.getElementById('seasonFilter').value='';
  document.querySelectorAll('.sort-btn').forEach(function(b){b.classList.remove('active');});
  document.querySelector('.sort-btn[data-sort="ppg"]').classList.add('active');
  curSort='ppg';filterPlayers();
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
      return true;
    });
    f.sort(function(a,b){return(b[curSort]!=null?b[curSort]:-999)-(a[curSort]!=null?a[curSort]:-999);});
    renderPlayers(f);
  },0);
}

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
    team_map    = load_team_map()
    rapm_df     = load_rapm()
    xrapm_df    = load_xrapm()
    xrapm_v2_df = load_xrapm_v2()
    linear_df   = load_linear_metrics()
    season_stats_df = load_player_season_stats()

    print(f"  Offensive archetypes: {len(off_df)}")
    print(f"  Defensive archetypes: {len(def_df)}")
    print(f"  Linear metrics: {len(linear_df)}")
    print(f"  Season stats: {len(season_stats_df)}")

    html = generate_html(off_df, def_df, profiles_df, bios_df, rapm_df, xrapm_df, xrapm_v2_df, linear_df, season_stats_df, team_map)

    os.makedirs("app", exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\u2705 Saved to {OUTPUT_FILE}")
    print(f"Open in browser: file://{os.path.abspath(OUTPUT_FILE)}")


if __name__ == "__main__":
    main()
