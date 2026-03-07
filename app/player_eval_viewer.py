"""
app/player_eval_viewer.py
=============================================================================
Generate a standalone interactive HTML viewer for Player Evaluation Core (PEC).
Displays Step 1 impact profiles and Step 2 MPG predictions.

Features:
  - Player cards with BKE, RAPM, predicted MPG, actual MPG
  - Offensive/defensive archetype probability strips
  - Season and team filters, search by name
  - Sortable by various metrics
  - Player detail modal with full profile
  - Team minute distribution view
  - Residual analysis (predicted vs actual MPG)

Input:
  data/processed/player_eval/player_impact_profiles.parquet
  data/processed/player_eval/minute_model_predictions_v2.parquet

Output:
  app/player_eval.html

Usage:
  python3 app/player_eval_viewer.py
=============================================================================
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.player_eval.constants import BKE_DECOMP_PATH, COMPLETE_STATS_PATH, PLAYER_ARCHETYPES_PATH, PLAYERS_META_PATH
from src.utils.player_name_normalizer import (
  apply_player_name_normalization,
  build_player_name_maps,
)

PROFILES_PATH = "data/processed/player_eval/player_impact_profiles.parquet"
PREDICTIONS_PATH = "data/processed/player_eval/minute_model_predictions_v2.parquet"
TEAM_AGG_PATH = "data/processed/player_eval/team_feature_aggregation.parquet"
OUTPUT_HTML = "app/player_eval.html"


def _safe_json_val(v):
    """Make value JSON-safe."""
    if isinstance(v, (float, np.floating)):
        if np.isnan(v) or np.isinf(v):
            return None
        return round(float(v), 4)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, dict):
        return {k: _safe_json_val(vv) for k, vv in v.items()}
    if isinstance(v, list):
        return [_safe_json_val(vv) for vv in v]
    return str(v) if v is not None else None


def _safe_int(v, default: int = 0) -> int:
    try:
        if v is None:
            return default
        if isinstance(v, (float, np.floating)) and np.isnan(v):
            return default
        return int(float(v))
    except Exception:
        return default


def _clean_team_abbreviation(v):
    if v is None:
        return ""
    team = str(v).strip().upper()
    if team in {"", "NAN", "NONE", "TOT"}:
        return ""
    return team


def build_payload():
    """Build JSON payload from Step 1 profiles and Step 2 predictions."""
    profiles = pd.read_parquet(PROFILES_PATH)
    predictions = pd.read_parquet(PREDICTIONS_PATH)

    # Merge predictions into profiles
    predictions = predictions.rename(
        columns={
            "team_abbreviation": "pred_team_abbreviation",
            "team_id": "pred_team_id",
        }
    )
    pred_cols = ["player_id", "season"]
    for c in predictions.columns:
        if c not in ["player_id", "season", "player_name"]:
            pred_cols.append(c)
    merged = profiles.merge(predictions[pred_cols], on=["player_id", "season"], how="left")

    # Normalize names in payload (ID-first, alias-aware)
    name_sources = [
      (PLAYERS_META_PATH, ["id", "player_id"], ["full_name", "player_name"], 1),
      (PLAYER_ARCHETYPES_PATH, ["PLAYER_ID", "player_id"], ["PLAYER_NAME", "player_name"], 2),
      (COMPLETE_STATS_PATH, ["PLAYER_ID", "player_id"], ["PLAYER_NAME", "player_name"], 2),
      (BKE_DECOMP_PATH, ["player_id", "PLAYER_ID"], ["player_name", "PLAYER_NAME"], 3),
    ]
    id_to_name, key_to_name = build_player_name_maps(name_sources)
    merged = apply_player_name_normalization(
      df=merged,
      player_id_col="player_id",
      player_name_col="player_name",
      id_to_name=id_to_name,
      key_to_name=key_to_name,
    )

    off_prob_cols = sorted([c for c in merged.columns if c.startswith("off_prob_")])
    def_prob_cols = sorted([c for c in merged.columns if c.startswith("def_prob_")])
    playtype_cols = sorted(
      [c for c in merged.columns if c.startswith("playtype_") and c[9:].isdigit()],
      key=lambda c: int(c.split("_")[1]),
    )

    players = []
    for _, row in merged.iterrows():
        # Extract archetype probs
        off_probs = {}
        def_probs = {}
        playtypes = []
        for c in off_prob_cols:
            off_probs[c.replace("off_prob_", "")] = _safe_json_val(row[c])
        for c in def_prob_cols:
            def_probs[c.replace("def_prob_", "")] = _safe_json_val(row[c])
        for c in playtype_cols:
            playtypes.append(_safe_json_val(row[c]))

        team_abbr = (
          _clean_team_abbreviation(row.get("team_abbreviation"))
          or _clean_team_abbreviation(row.get("TEAM_ABBREVIATION"))
          or _clean_team_abbreviation(row.get("pred_team_abbreviation"))
          or _clean_team_abbreviation(row.get("team"))
        )
        stint_number = max(1, _safe_int(row.get("stint_number"), default=1))
        stint_count = max(1, _safe_int(row.get("stint_count"), default=1))
        stint_team_count = max(1, _safe_int(row.get("stint_team_count"), default=1))
        player_key = f"{row.get('player_id', '')}::{row.get('season', '')}::{team_abbr}::{stint_number}"

        player = {
          "key": player_key,
            "id": str(row.get("player_id", "")),
            "name": str(row.get("player_name", "Unknown")),
            "season": str(row.get("season", "")),
          "team": team_abbr,
          "stint_number": stint_number,
          "stint_count": stint_count,
          "stint_team_count": stint_team_count,
          "is_primary_stint": bool(row.get("is_primary_stint", False)),
          "is_final_stint": bool(row.get("is_final_stint", False)),
          "stint_first_game_date": str(row.get("stint_first_game_date", "") or ""),
          "stint_last_game_date": str(row.get("stint_last_game_date", "") or ""),
            # Impact
            "bke": _safe_json_val(row.get("impact_bke")),
            "obke": _safe_json_val(row.get("impact_obke")),
            "dbke": _safe_json_val(row.get("impact_dbke")),
            "orapm": _safe_json_val(row.get("impact_orapm")),
            "drapm": _safe_json_val(row.get("impact_drapm")),
            "stability": _safe_json_val(row.get("impact_stability")),
            "ws": _safe_json_val(row.get("impact_ws")),
            "bpm": _safe_json_val(row.get("impact_bpm")),
            "vorp": _safe_json_val(row.get("impact_vorp")),
            "portable_talent": _safe_json_val(row.get("impact_portable_talent")),
            "total_impact": _safe_json_val(row.get("impact_total_impact")),
            # Behavioral
            "usage": _safe_json_val(row.get("behavioral_usage")),
            "ast_rate": _safe_json_val(row.get("behavioral_assist_rate")),
            "tov_rate": _safe_json_val(row.get("behavioral_turnover_rate")),
            "three_rate": _safe_json_val(row.get("behavioral_three_point_rate")),
            "rim_rate": _safe_json_val(row.get("behavioral_rim_rate")),
            "efg": _safe_json_val(row.get("behavioral_efg")),
            "orb_rate": _safe_json_val(row.get("behavioral_orb_rate")),
            "drb_rate": _safe_json_val(row.get("behavioral_drb_rate")),
            "ft_rate": _safe_json_val(row.get("behavioral_free_throw_rate")),
            "hustle": _safe_json_val(row.get("behavioral_hustle_pctl")),
            "foul_rate": _safe_json_val(row.get("behavioral_foul_rate")),
            # Volume
            "mpg": _safe_json_val(row.get("mpg")),
            "minutes": _safe_json_val(row.get("minutes")),
            "games": _safe_json_val(row.get("games")),
            "possessions": _safe_json_val(row.get("possessions")),
            # Context
            "age": _safe_json_val(row.get("age")),
            "height": _safe_json_val(row.get("height_inches")),
            "weight": _safe_json_val(row.get("weight_lbs")),
            "experience": _safe_json_val(row.get("experience_years")),
            "position": str(row.get("position_proxy", "")),
            "salary": _safe_json_val(row.get("salary")),
            # Scheme
            "on_off_diff": _safe_json_val(row.get("on_off_diff")),
            "scheme_stability": _safe_json_val(row.get("scheme_stability_index")),
            # Predictions
            "pred_mpg": _safe_json_val(row.get("pred_mpg_raw")),
            "pred_mpg_norm": _safe_json_val(row.get("pred_mpg_team_norm")),
            "target_mpg": _safe_json_val(row.get("target_mpg")),
            # Archetypes
            "off_probs": off_probs,
            "def_probs": def_probs,
            "off_archetype": str(row.get("off_primary_archetype", "") or "Unknown"),
            "off_secondary": str(row.get("off_secondary_archetype", "") or ""),
            "off_confidence": _safe_json_val(row.get("off_role_confidence")),
            "off_effectiveness": _safe_json_val(row.get("off_role_effectiveness")),
            "def_archetype": str(row.get("def_primary_archetype", "") or "Unknown"),
            "def_secondary": str(row.get("def_secondary_archetype", "") or ""),
            "def_confidence": _safe_json_val(row.get("def_role_confidence")),
            "playtypes": playtypes,
        }
        players.append(player)

    return players


def build_team_payload():
    """Build JSON payload from Step 3 team feature aggregation."""
    if not os.path.exists(TEAM_AGG_PATH):
        print(f"  Team aggregation data not found: {TEAM_AGG_PATH}")
        return []

    df = pd.read_parquet(TEAM_AGG_PATH)
    teams = []
    for _, row in df.iterrows():
        team_abbrev = _clean_team_abbreviation(row.get("team_abbreviation"))
        if not team_abbrev:
            continue

        off_arch = {}
        def_arch = {}
        interactions = []
        player_summaries = []
        try:
            off_arch = json.loads(row.get("off_archetype_distribution", "{}"))
        except Exception:
            pass
        try:
            def_arch = json.loads(row.get("def_archetype_distribution", "{}"))
        except Exception:
            pass
        try:
            interactions = json.loads(row.get("interaction_details", "[]"))
        except Exception:
            pass
        try:
            player_summaries = json.loads(row.get("player_summaries", "[]"))
        except Exception:
            pass

        team = {
            "team": team_abbrev,
            "season": str(row.get("season", "")),
            "n_players": int(row.get("n_players", 0)),
            "off_talent": _safe_json_val(row.get("off_talent_base")),
            "off_interaction": _safe_json_val(row.get("off_interaction_term")),
            "off_structure": _safe_json_val(row.get("off_structure_term")),
            "off_mean": _safe_json_val(row.get("off_mean")),
            "tov_penalty": _safe_json_val(row.get("off_tov_penalty")),
            "ftr_bonus": _safe_json_val(row.get("off_ftr_bonus")),
            "playmaking_adj": _safe_json_val(row.get("off_playmaking_adj")),
            "spacing_adj": _safe_json_val(row.get("off_spacing_adj")),
            "transition_bonus": _safe_json_val(row.get("off_transition_bonus")),
            "n_playmakers": int(row.get("off_n_playmakers", 0)),
            "n_shooters": int(row.get("off_n_shooters", 0)),
            "transition_freq": _safe_json_val(row.get("off_team_transition_freq")),
            "transition_success": _safe_json_val(row.get("off_team_transition_success")),
            "def_talent": _safe_json_val(row.get("def_talent_base")),
            "def_adj": _safe_json_val(row.get("def_adjustments")),
            "def_mean": _safe_json_val(row.get("def_mean")),
            "rp_penalty": _safe_json_val(row.get("def_rim_protector_penalty")),
            "poa_penalty": _safe_json_val(row.get("def_poa_defender_penalty")),
            "anchor_bonus": _safe_json_val(row.get("def_anchor_bonus")),
            "diversity_bonus": _safe_json_val(row.get("def_diversity_bonus")),
            "liability_penalty": _safe_json_val(row.get("def_liability_penalty")),
            "n_liabilities": int(row.get("def_n_liabilities", 0)),
            "unique_def_archetypes": int(row.get("def_unique_archetypes", 0)),
            "net_projected": _safe_json_val(row.get("team_net_rating_projected")),
            "vol_base": _safe_json_val(row.get("vol_base")),
            "vol_3pa": _safe_json_val(row.get("vol_3pa")),
            "vol_creation": _safe_json_val(row.get("vol_creation")),
            "vol_transition": _safe_json_val(row.get("vol_transition")),
            "vol_total": _safe_json_val(row.get("vol_total")),
            "off_arch_dist": off_arch,
            "def_arch_dist": def_arch,
            "interactions": interactions,
            "players": player_summaries,
        }
        teams.append(team)

    return teams


def generate_html():
    """Generate the standalone HTML viewer."""
    players = build_payload()
    team_agg = build_team_payload()
    data_json = json.dumps(players, separators=(",", ":"))
    team_json = json.dumps(team_agg, separators=(",", ":"))

    # Get unique seasons and teams (union of player and team-aggregation sources)
    seasons = sorted(set(p["season"] for p in players) | set(tm.get("season") for tm in team_agg))
    teams = sorted((set(p["team"] for p in players if p["team"]) | set(tm.get("team") for tm in team_agg if tm.get("team"))) )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Player Evaluation Core — PEC v1</title>
<style>
:root {{
  --bg: #0d1117; --surface: #161b22; --surface2: #1c2128; --border: #30363d;
  --text: #c9d1d9; --text-muted: #8b949e; --accent: #58a6ff;
  --green: #3fb950; --red: #f85149; --orange: #d29922; --purple: #bc8cff;
  --cyan: #39d2c0;
}}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:var(--bg); color:var(--text); font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif; font-size:14px; }}
.container {{ max-width:1600px; margin:0 auto; padding:16px; }}
h1 {{ font-size:22px; font-weight:600; color:var(--accent); margin-bottom:2px; }}
.subtitle {{ color:var(--text-muted); font-size:12px; margin-bottom:16px; }}

/* Controls */
.controls {{ display:flex; flex-wrap:wrap; gap:12px; align-items:center; background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:12px 16px; margin-bottom:16px; }}
.controls label {{ font-size:11px; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.5px; }}
.controls select, .controls input[type=text] {{ background:var(--bg); color:var(--text); border:1px solid var(--border); border-radius:4px; padding:6px 10px; font-size:13px; }}
.controls input[type=text] {{ width:200px; }}
.sort-btns {{ display:flex; gap:4px; }}
.sort-btns button {{ background:var(--bg); color:var(--text-muted); border:1px solid var(--border); border-radius:4px; padding:4px 10px; font-size:12px; cursor:pointer; }}
.sort-btns button.active {{ background:var(--accent); color:#fff; border-color:var(--accent); }}
.tab-btns {{ display:flex; gap:0; margin-bottom:16px; }}
.tab-btns button {{ background:var(--surface); color:var(--text-muted); border:1px solid var(--border); padding:8px 20px; font-size:13px; cursor:pointer; }}
.tab-btns button:first-child {{ border-radius:6px 0 0 6px; }}
.tab-btns button:last-child {{ border-radius:0 6px 6px 0; }}
.tab-btns button.active {{ background:var(--accent); color:#fff; border-color:var(--accent); font-weight:600; }}
.stats-bar {{ display:flex; gap:12px; flex-wrap:wrap; margin-bottom:16px; }}
.stat-card {{ background:var(--surface); border:1px solid var(--border); border-radius:6px; padding:10px 14px; min-width:120px; }}
.stat-card .sl {{ font-size:10px; color:var(--text-muted); text-transform:uppercase; }}
.stat-card .sv {{ font-size:18px; font-weight:700; margin-top:2px; }}

/* Cards */
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(320px,1fr)); gap:12px; }}
.card {{ background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:14px; cursor:pointer; transition:border-color 0.15s; }}
.card:hover {{ border-color:var(--accent); }}
.card-header {{ display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:8px; }}
.card-name {{ font-size:15px; font-weight:600; }}
.card-team {{ font-size:11px; color:var(--text-muted); }}
.card-badge {{ display:inline-block; padding:2px 8px; border-radius:10px; font-size:11px; font-weight:600; }}
.card-metrics {{ display:grid; grid-template-columns:repeat(4,1fr); gap:6px; margin-bottom:8px; }}
.cm {{ text-align:center; }}
.cm-val {{ font-size:14px; font-weight:600; }}
.cm-label {{ font-size:10px; color:var(--text-muted); }}
.prob-strip {{ display:flex; height:6px; border-radius:3px; overflow:hidden; margin:4px 0; }}
.prob-strip div {{ transition:width 0.2s; }}
.prob-label {{ font-size:10px; color:var(--text-muted); }}
.mpg-bar {{ display:flex; align-items:center; gap:8px; margin-top:6px; }}
.mpg-bar-bg {{ flex:1; height:8px; background:var(--bg); border-radius:4px; overflow:hidden; position:relative; }}
.mpg-bar-actual {{ height:100%; border-radius:4px; }}
.mpg-bar-pred {{ position:absolute; top:0; height:100%; border-right:2px solid var(--accent); }}
.mpg-vals {{ font-size:11px; color:var(--text-muted); white-space:nowrap; }}

/* Modal */
.modal-overlay {{ display:none; position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.7); z-index:1000; justify-content:center; align-items:flex-start; padding-top:40px; }}
.modal-overlay.open {{ display:flex; }}
.modal {{ background:var(--surface); border:1px solid var(--border); border-radius:12px; width:90%; max-width:800px; max-height:85vh; overflow-y:auto; padding:24px; }}
.modal-close {{ float:right; background:none; border:none; color:var(--text-muted); font-size:20px; cursor:pointer; }}
.modal h2 {{ font-size:20px; margin-bottom:4px; }}
.modal .meta {{ color:var(--text-muted); font-size:13px; margin-bottom:16px; }}
.section {{ margin-bottom:16px; }}
.section h3 {{ font-size:13px; color:var(--accent); text-transform:uppercase; letter-spacing:0.5px; margin-bottom:8px; border-bottom:1px solid var(--border); padding-bottom:4px; }}
.metric-grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(130px,1fr)); gap:8px; }}
.mg-item {{ background:var(--bg); border-radius:4px; padding:8px; }}
.mg-item .mg-val {{ font-size:16px; font-weight:600; }}
.mg-item .mg-lbl {{ font-size:10px; color:var(--text-muted); }}
.arch-bar {{ display:flex; align-items:center; gap:8px; margin:3px 0; }}
.arch-bar-label {{ width:140px; font-size:11px; text-align:right; color:var(--text-muted); }}
.arch-bar-fill {{ height:14px; border-radius:3px; min-width:2px; }}
.arch-bar-val {{ font-size:11px; width:40px; }}

/* Team view */
.team-table {{ width:100%; border-collapse:collapse; }}
.team-table th, .team-table td {{ padding:6px 10px; border-bottom:1px solid var(--border); text-align:left; font-size:13px; }}
.team-table th {{ color:var(--text-muted); font-size:11px; text-transform:uppercase; background:var(--surface2); position:sticky; top:0; }}
.team-table tr:hover {{ background:var(--surface2); }}
.positive {{ color:var(--green); }}
.negative {{ color:var(--red); }}
</style>
</head>
<body>
<div class="container">
<h1>Player Evaluation Core — PEC v1</h1>
<div class="subtitle">Impact profiles & predicted minutes per game | {len(players)} player-seasons across {len(seasons)} seasons</div>

<div class="tab-btns">
  <button class="active" onclick="switchTab('cards')">Player Cards</button>
  <button onclick="switchTab('team')">Team View</button>
  <button onclick="switchTab('teamagg')">Team Aggregation</button>
</div>

<div class="controls">
  <div>
    <label>Season</label><br>
    <select id="seasonFilter" onchange="applyFilters()">
      <option value="all">All Seasons</option>
      {''.join(f'<option value="{s}">{s}</option>' for s in seasons)}
    </select>
  </div>
  <div>
    <label>Team</label><br>
    <select id="teamFilter" onchange="applyFilters()">
      <option value="all">All Teams</option>
      {''.join(f'<option value="{t}">{t}</option>' for t in teams)}
    </select>
  </div>
  <div>
    <label>Search</label><br>
    <input type="text" id="searchBox" placeholder="Player name..." oninput="applyFilters()">
  </div>
  <div>
    <label>Sort by</label><br>
    <div class="sort-btns">
      <button class="active" data-sort="bke" onclick="setSort(this)">BKE</button>
      <button data-sort="mpg" onclick="setSort(this)">MPG</button>
      <button data-sort="pred_mpg" onclick="setSort(this)">Pred MPG</button>
      <button data-sort="orapm" onclick="setSort(this)">ORAPM</button>
      <button data-sort="ws" onclick="setSort(this)">WS</button>
      <button data-sort="usage" onclick="setSort(this)">USG%</button>
    </div>
  </div>
</div>

<div class="stats-bar" id="statsBar"></div>
<div id="cardsView"><div class="grid" id="cardGrid"></div></div>
<div id="teamView" style="display:none"></div>
<div id="teamAggView" style="display:none"></div>
</div>

<div class="modal-overlay" id="modal" onclick="if(event.target===this)closeModal()">
  <div class="modal" id="modalContent"></div>
</div>

<div class="modal-overlay" id="teamModal" onclick="if(event.target===this)closeTeamModal()">
  <div class="modal" id="teamModalContent" style="max-width:1000px"></div>
</div>

<script>
const DATA = {data_json};
const TEAM_AGG = {team_json};
let currentSort = 'bke';
let sortAsc = false;
const OFF_COLORS = ['#f85149','#ff7b72','#ffa657','#d29922','#e3b341','#3fb950','#56d364','#39d2c0','#58a6ff','#bc8cff','#d2a8ff'];
const DEF_COLORS = ['#f85149','#ff7b72','#ffa657','#d29922','#3fb950','#58a6ff','#bc8cff'];
const OFF_LABELS = {{
  'emb_ball_dominant_creator':'Ball Dom Creator','emb_all_around_scorer':'All-Around Scorer',
  'emb_ballhandler':'Ballhandler','emb_interior_scorer':'Interior Scorer',
  'emb_perimeter_scorer':'Perimeter Scorer','emb_connector':'Connector',
  'emb_pnr_rolling_big':'PnR Roll Big','emb_pnr_popping_big':'PnR Pop Big',
  'emb_off_ball_finisher':'Off-Ball Finisher','emb_off_ball_movement_shooter':'Off-Ball Move Shooter',
  'emb_off_ball_stationary_shooter':'Off-Ball Stat Shooter'
}};
const DEF_LABELS = {{
  'poa_score':'POA Defender','wing_score':'Wing Defender','chaser_score':'Chaser',
  'versatile_score':'Versatile','rim_score':'Rim Protector',
  'drop_big_score':'Drop Big','mobile_big_score':'Mobile Big'
}};
const PLAYTYPE_LABELS = ['ISO','PnR BH','Post','Cut','PnR RM','Handoff','OffScreen','SpotUp','Trans','OffReb','Misc'];

function fmt(v,d=1){{ return v==null?'—':(typeof v==='number'?v.toFixed(d):'—'); }}
function fmtPct(v){{ return v==null?'—':(v*100).toFixed(1)+'%'; }}
function fmtSalary(v){{ if(v==null||isNaN(v))return'Salary not found'; if(Number(v)<=0)return'Salary not found'; if(v>=1e6)return'$'+((v/1e6).toFixed(1))+'M'; return'$'+(v/1e3).toFixed(0)+'K'; }}
function bkeColor(v){{ if(v==null)return'var(--text-muted)'; if(v>2)return'var(--green)'; if(v>0)return'var(--cyan)'; if(v>-2)return'var(--orange)'; return'var(--red)'; }}
function residColor(r){{ if(r==null)return'var(--text-muted)'; if(Math.abs(r)<2)return'var(--text)'; return r>0?'var(--green)':'var(--red)'; }}
function stripDiacritics(s){{ return s.normalize('NFD').replace(/[\u0300-\u036f]/g,''); }}

function getFiltered() {{
  let s = document.getElementById('seasonFilter').value;
  let t = document.getElementById('teamFilter').value;
  let q = stripDiacritics(document.getElementById('searchBox').value.toLowerCase());
  let list = DATA.filter(p => {{
    if(s!=='all' && p.season!==s) return false;
    if(t!=='all' && p.team!==t) return false;
    if(q && !stripDiacritics(p.name.toLowerCase()).includes(q)) return false;
    return true;
  }});
  list.sort((a,b) => {{
    let va=a[currentSort], vb=b[currentSort];
    if(va==null && vb==null) return 0;
    if(va==null) return 1;
    if(vb==null) return -1;
    const av = (typeof va === 'number') ? va : String(va).toLowerCase();
    const bv = (typeof vb === 'number') ? vb : String(vb).toLowerCase();
    if(av < bv) return sortAsc ? -1 : 1;
    if(av > bv) return sortAsc ? 1 : -1;
    return 0;
  }});
  return list;
}}

function updateStats(list) {{
  const bar = document.getElementById('statsBar');
  const n = list.length;
  const avgBke = n?list.reduce((s,p)=>s+(p.bke||0),0)/n:0;
  const avgMpg = n?list.reduce((s,p)=>s+(p.mpg||0),0)/n:0;
  const avgPred = list.filter(p=>p.pred_mpg!=null);
  const ap = avgPred.length?avgPred.reduce((s,p)=>s+(p.pred_mpg||0),0)/avgPred.length:0;
  bar.innerHTML = `
    <div class="stat-card"><div class="sl">Players</div><div class="sv">${{n}}</div></div>
    <div class="stat-card"><div class="sl">Avg BKE</div><div class="sv" style="color:${{bkeColor(avgBke)}}">${{fmt(avgBke,2)}}</div></div>
    <div class="stat-card"><div class="sl">Avg MPG</div><div class="sv">${{fmt(avgMpg,1)}}</div></div>
    <div class="stat-card"><div class="sl">Avg Pred MPG</div><div class="sv" style="color:var(--accent)">${{fmt(ap,1)}}</div></div>
  `;
}}

function makeProbStrip(probs, colors, labels) {{
  if(!probs) return '';
  const entries = Object.entries(probs).filter(([k,v])=>v!=null);
  const total = entries.reduce((s,[k,v])=>s+v,0)||1;
  let html = '<div class="prob-strip">';
  entries.forEach(([k,v],i) => {{
    const pct = (v/total*100).toFixed(1);
    html += `<div style="width:${{pct}}%;background:${{colors[i%colors.length]}}" title="${{labels[k]||k}}: ${{pct}}%"></div>`;
  }});
  html += '</div>';
  return html;
}}

function stintSuffix(p) {{
  if (!p || !p.stint_count || p.stint_count <= 1) return '';
  const role = p.is_final_stint ? ' Final' : (p.is_primary_stint ? ' Primary' : '');
  return ` · Stint ${{p.stint_number}}/${{p.stint_count}}${{role}}`;
}}

function renderCards(list) {{
  const grid = document.getElementById('cardGrid');
  const maxMpg = 40;
  grid.innerHTML = list.map((p,i) => {{
    const bkeStyle = `color:${{bkeColor(p.bke)}}`;
    const mpgPct = Math.min((p.mpg||0)/maxMpg*100,100);
    const predPct = p.pred_mpg!=null?Math.min(p.pred_mpg/maxMpg*100,100):0;
    const resid = (p.mpg!=null&&p.pred_mpg!=null)?(p.mpg-p.pred_mpg):null;
    return `<div class="card" onclick="openModal('${{p.key}}')">
      <div class="card-header">
        <div><div class="card-name">${{p.name}}</div>
          <div class="card-team">${{p.team}} · ${{p.season}}${{stintSuffix(p)}} · ${{p.position||'—'}} · Age ${{p.age||'—'}}</div>
        </div>
        <span class="card-badge" style="background:${{bkeColor(p.bke)}}33;${{bkeStyle}}">${{fmt(p.bke,2)}} BKE</span>
      </div>
      <div class="card-metrics">
        <div class="cm"><div class="cm-val" style="${{bkeStyle}}">${{fmt(p.orapm,1)}}</div><div class="cm-label">ORAPM</div></div>
        <div class="cm"><div class="cm-val" style="${{bkeStyle}}">${{fmt(p.drapm,1)}}</div><div class="cm-label">DRAPM</div></div>
        <div class="cm"><div class="cm-val">${{fmt(p.ws,1)}}</div><div class="cm-label">WS</div></div>
        <div class="cm"><div class="cm-val">${{fmtPct(p.usage)}}</div><div class="cm-label">USG%</div></div>
      </div>
      <div class="prob-label">Offense: ${{p.off_archetype||'—'}}${{p.off_secondary?` · ${{p.off_secondary}}`:''}}${{p.off_confidence!=null?` (${{fmtPct(p.off_confidence)}} fit)`:''}}</div>
      ${{makeProbStrip(p.off_probs, OFF_COLORS, OFF_LABELS)}}
      <div class="prob-label">Defense: ${{p.def_archetype||'—'}}${{p.def_secondary?` · ${{p.def_secondary}}`:''}}${{p.def_confidence!=null?` (${{fmtPct(p.def_confidence)}} fit)`:''}}</div>
      ${{makeProbStrip(p.def_probs, DEF_COLORS, DEF_LABELS)}}
      <div class="mpg-bar">
        <div class="mpg-vals">${{fmt(p.mpg,1)}} MPG</div>
        <div class="mpg-bar-bg">
          <div class="mpg-bar-actual" style="width:${{mpgPct}}%;background:var(--green)"></div>
          ${{p.pred_mpg!=null?`<div class="mpg-bar-pred" style="left:${{predPct}}%"></div>`:''}}
        </div>
        <div class="mpg-vals" style="color:var(--accent)">${{p.pred_mpg!=null?fmt(p.pred_mpg,1)+'p':'—'}}</div>
        ${{resid!=null?`<span style="font-size:11px;color:${{residColor(resid)}}">${{resid>0?'+':''}}${{fmt(resid,1)}}</span>`:''}}
      </div>
    </div>`;
  }}).join('');
}}

function renderTeamView(list) {{
  const tv = document.getElementById('teamView');
  const teams = {{}};
  list.forEach(p => {{
    if(!teams[p.team]) teams[p.team] = [];
    teams[p.team].push(p);
  }});
  let html = '';
  Object.keys(teams).sort().forEach((team, teamIdx) => {{
    const players = teams[team].sort((a,b) => (b.mpg||0)-(a.mpg||0));
    const totalMpg = players.reduce((s,p)=>s+(p.mpg||0),0);
    const sectionId = `team_section_${{teamIdx}}`;
    html += `<div style="margin-bottom:12px;border:1px solid var(--border);border-radius:8px;background:var(--surface)">
      <button onclick="toggleTeam('${{sectionId}}')" style="width:100%;text-align:left;background:transparent;border:none;color:var(--accent);padding:10px 12px;cursor:pointer;font-size:14px;font-weight:600">
        ▶ ${{team}} <span style="color:var(--text-muted);font-size:12px">(${{players.length}} players, total MPG: ${{fmt(totalMpg,0)}})</span>
      </button>
      <div id="${{sectionId}}" style="display:none;padding:0 10px 10px 10px">
      <table class="team-table"><thead><tr>
        <th>#</th><th>Player</th><th>Pos</th><th>Age</th><th>BKE</th><th>Actual MPG</th><th>Pred MPG</th><th>Residual</th><th>USG%</th><th>WS</th>
      </tr></thead><tbody>`;
    players.forEach((p,i) => {{
      const resid = (p.mpg!=null&&p.pred_mpg!=null)?(p.mpg-p.pred_mpg):null;
      html += `<tr onclick="openModal('${{p.key}}')" style="cursor:pointer">
        <td>${{i+1}}</td>
        <td style="font-weight:600">${{p.name}} (${{p.season}}${{p.stint_count>1 ? ` · S${{p.stint_number}}/${{p.stint_count}}` : ''}})</td>
        <td>${{p.position||'—'}}</td>
        <td>${{p.age||'—'}}</td>
        <td style="color:${{bkeColor(p.bke)}}">${{fmt(p.bke,2)}}</td>
        <td>${{fmt(p.mpg,1)}}</td>
        <td style="color:var(--accent)">${{p.pred_mpg!=null?fmt(p.pred_mpg,1):'—'}}</td>
        <td style="color:${{residColor(resid)}}">${{resid!=null?(resid>0?'+':'')+fmt(resid,1):'—'}}</td>
        <td>${{fmtPct(p.usage)}}</td>
        <td>${{fmt(p.ws,1)}}</td>
      </tr>`;
    }});
    html += '</tbody></table></div></div>';
  }});
  tv.innerHTML = html;
}}

function openModal(playerKey) {{
  const p = DATA.find(d => d.key===playerKey);
  if(!p) return;
  const m = document.getElementById('modalContent');
  const resid = (p.mpg!=null&&p.pred_mpg!=null)?(p.mpg-p.pred_mpg):null;

  // Archetype bars
  function archBars(probs, colors, labels) {{
    if(!probs) return '<div style="color:var(--text-muted)">No data</div>';
    const entries = Object.entries(probs).sort((a,b)=>(b[1]||0)-(a[1]||0));
    return entries.map(([k,v],i) => {{
      const pct = ((v||0)*100).toFixed(1);
      return `<div class="arch-bar">
        <div class="arch-bar-label">${{labels[k]||k}}</div>
        <div style="flex:1;background:var(--bg);border-radius:3px;overflow:hidden">
          <div class="arch-bar-fill" style="width:${{pct}}%;background:${{colors[i%colors.length]}}"></div>
        </div>
        <div class="arch-bar-val">${{pct}}%</div>
      </div>`;
    }}).join('');
  }}

  // Playtype bars
  let playtypeHtml = '';
  if(p.playtypes && p.playtypes.length) {{
    playtypeHtml = p.playtypes.map((v,i) => {{
      const pct = ((v||0)*100).toFixed(1);
      return `<div class="arch-bar">
        <div class="arch-bar-label">${{PLAYTYPE_LABELS[i]||'Play '+i}}</div>
        <div style="flex:1;background:var(--bg);border-radius:3px;overflow:hidden">
          <div class="arch-bar-fill" style="width:${{Math.min(pct,100)}}%;background:var(--accent)"></div>
        </div>
        <div class="arch-bar-val">${{pct}}%</div>
      </div>`;
    }}).join('');
  }}

  const stintMeta = p.stint_count>1
    ? ` · Stint ${{p.stint_number}}/${{p.stint_count}}${{p.is_primary_stint?' (Primary)':''}}${{p.is_final_stint?' (Final)':''}}`
    : '';

  m.innerHTML = `
    <button class="modal-close" onclick="closeModal()">&times;</button>
    <h2>${{p.name}}</h2>
    <div class="meta">${{p.team}} · ${{p.season}}${{stintMeta}} · ${{p.position||'—'}} · Age ${{p.age||'—'}} · ${{p.height?Math.floor(p.height/12)+"'"+Math.round(p.height%12)+'"':'—'}} · ${{p.weight?p.weight+' lbs':'—'}} · Exp ${{p.experience||'—'}}y · ${{fmtSalary(p.salary)}}</div>

    <div class="section">
      <h3>Impact Metrics</h3>
      <div class="metric-grid">
        <div class="mg-item"><div class="mg-val" style="color:${{bkeColor(p.bke)}}">${{fmt(p.bke,2)}}</div><div class="mg-lbl">BKE</div></div>
        <div class="mg-item"><div class="mg-val">${{fmt(p.obke,2)}}</div><div class="mg-lbl">OBKE</div></div>
        <div class="mg-item"><div class="mg-val">${{fmt(p.dbke,2)}}</div><div class="mg-lbl">DBKE</div></div>
        <div class="mg-item"><div class="mg-val">${{fmt(p.orapm,2)}}</div><div class="mg-lbl">ORAPM</div></div>
        <div class="mg-item"><div class="mg-val">${{fmt(p.drapm,2)}}</div><div class="mg-lbl">DRAPM</div></div>
        <div class="mg-item"><div class="mg-val">${{fmt(p.ws,1)}}</div><div class="mg-lbl">Win Shares</div></div>
        <div class="mg-item"><div class="mg-val">${{fmt(p.bpm,1)}}</div><div class="mg-lbl">BPM</div></div>
        <div class="mg-item"><div class="mg-val">${{fmt(p.vorp,1)}}</div><div class="mg-lbl">VORP</div></div>
        <div class="mg-item"><div class="mg-val">${{fmt(p.portable_talent,2)}}</div><div class="mg-lbl">Portable Talent</div></div>
        <div class="mg-item"><div class="mg-val">${{fmt(p.total_impact,2)}}</div><div class="mg-lbl">Total Impact</div></div>
        <div class="mg-item"><div class="mg-val">${{fmtPct(p.stability)}}</div><div class="mg-lbl">Stability</div></div>
        <div class="mg-item"><div class="mg-val">${{fmt(p.on_off_diff,1)}}</div><div class="mg-lbl">On/Off Diff</div></div>
      </div>
    </div>

    <div class="section">
      <h3>Minutes Prediction</h3>
      <div class="metric-grid">
        <div class="mg-item"><div class="mg-val">${{fmt(p.mpg,1)}}</div><div class="mg-lbl">Actual MPG</div></div>
        <div class="mg-item"><div class="mg-val" style="color:var(--accent)">${{p.pred_mpg!=null?fmt(p.pred_mpg,1):'—'}}</div><div class="mg-lbl">Predicted MPG</div></div>
        <div class="mg-item"><div class="mg-val" style="color:${{residColor(resid)}}">${{resid!=null?(resid>0?'+':'')+fmt(resid,1):'—'}}</div><div class="mg-lbl">Residual</div></div>
        <div class="mg-item"><div class="mg-val">${{fmt(p.games,0)}}</div><div class="mg-lbl">Games</div></div>
        <div class="mg-item"><div class="mg-val">${{fmt(p.minutes,0)}}</div><div class="mg-lbl">Total Min</div></div>
        <div class="mg-item"><div class="mg-val">${{fmt(p.possessions,0)}}</div><div class="mg-lbl">Possessions</div></div>
      </div>
    </div>

    <div class="section">
      <h3>Behavioral Profile</h3>
      <div class="metric-grid">
        <div class="mg-item"><div class="mg-val">${{fmtPct(p.usage)}}</div><div class="mg-lbl">Usage Rate</div></div>
        <div class="mg-item"><div class="mg-val">${{fmtPct(p.efg)}}</div><div class="mg-lbl">eFG%</div></div>
        <div class="mg-item"><div class="mg-val">${{fmtPct(p.ast_rate)}}</div><div class="mg-lbl">AST%</div></div>
        <div class="mg-item"><div class="mg-val">${{fmtPct(p.tov_rate)}}</div><div class="mg-lbl">TOV%</div></div>
        <div class="mg-item"><div class="mg-val">${{fmtPct(p.three_rate)}}</div><div class="mg-lbl">3PA Rate</div></div>
        <div class="mg-item"><div class="mg-val">${{fmtPct(p.rim_rate)}}</div><div class="mg-lbl">Rim Freq</div></div>
        <div class="mg-item"><div class="mg-val">${{fmtPct(p.orb_rate)}}</div><div class="mg-lbl">ORB%</div></div>
        <div class="mg-item"><div class="mg-val">${{fmtPct(p.drb_rate)}}</div><div class="mg-lbl">DRB%</div></div>
        <div class="mg-item"><div class="mg-val">${{fmt(p.ft_rate,2)}}</div><div class="mg-lbl">FT Rate</div></div>
        <div class="mg-item"><div class="mg-val">${{fmtPct(p.hustle)}}</div><div class="mg-lbl">Hustle Pctl</div></div>
        <div class="mg-item"><div class="mg-val">${{fmt(p.foul_rate,1)}}</div><div class="mg-lbl">Fouls/36</div></div>
      </div>
    </div>

    <div class="section">
      <h3>Offensive Archetype Mix</h3>
      <div style="font-size:12px;color:var(--text-muted);margin-bottom:6px">${{p.off_archetype||'—'}}${{p.off_secondary?` · ${{p.off_secondary}}`:''}}${{p.off_confidence!=null?` · Fit ${{fmtPct(p.off_confidence)}}`:''}}${{p.off_effectiveness!=null?` · Effectiveness ${{fmtPct(p.off_effectiveness)}}`:''}}</div>
      ${{archBars(p.off_probs, OFF_COLORS, OFF_LABELS)}}
    </div>

    <div class="section">
      <h3>Defensive Archetype Mix</h3>
      <div style="font-size:12px;color:var(--text-muted);margin-bottom:6px">${{p.def_archetype||'—'}}${{p.def_secondary?` · ${{p.def_secondary}}`:''}}${{p.def_confidence!=null?` · Fit ${{fmtPct(p.def_confidence)}}`:''}}</div>
      ${{archBars(p.def_probs, DEF_COLORS, DEF_LABELS)}}
    </div>

    <div class="section">
      <h3>Playtype Distribution</h3>
      ${{playtypeHtml||'<div style="color:var(--text-muted)">No data</div>'}}
    </div>
  `;
  document.getElementById('modal').classList.add('open');
}}

function closeModal() {{ document.getElementById('modal').classList.remove('open'); }}
function closeTeamModal() {{ document.getElementById('teamModal').classList.remove('open'); }}

// ─── Team Aggregation View ──────────────────────────────────────
function getFilteredTeams() {{
  let s = document.getElementById('seasonFilter').value;
  let t = document.getElementById('teamFilter').value;
  return TEAM_AGG.filter(tm => {{
    if(s!=='all' && tm.season!==s) return false;
    if(t!=='all' && tm.team!==t) return false;
    return true;
  }}).sort((a,b) => (b.net_projected||0)-(a.net_projected||0));
}}

function renderTeamAggView() {{
  const view = document.getElementById('teamAggView');
  const teams = getFilteredTeams();
  if(!teams.length) {{
    view.innerHTML = '<div style="color:var(--text-muted);text-align:center;padding:40px">No team aggregation data. Run Step 3 first.</div>';
    return;
  }}
  let html = '<div class="grid" style="grid-template-columns:repeat(auto-fill,minmax(360px,1fr))">';
  teams.forEach(tm => {{
    const netColor = (tm.net_projected||0)>0?'var(--green)':'var(--red)';
    const offColor = (tm.off_mean||0)>0?'var(--green)':'var(--red)';
    const defColor = (tm.def_mean||0)>0?'var(--green)':'var(--red)';
    // Top archetype
    const topOff = Object.entries(tm.off_arch_dist||{{}}).sort((a,b)=>b[1]-a[1]);
    const topDef = Object.entries(tm.def_arch_dist||{{}}).sort((a,b)=>b[1]-a[1]);
    html += `<div class="card" onclick="openTeamModal('${{tm.team}}','${{tm.season}}')" style="padding:16px">
      <div class="card-header">
        <div>
          <div class="card-name" style="font-size:17px">${{tm.team}}</div>
          <div class="card-team">${{tm.season}} · ${{tm.n_players}} players</div>
        </div>
        <span class="card-badge" style="background:${{netColor}}22;color:${{netColor}};font-size:13px;padding:4px 12px">${{fmt(tm.net_projected,2)}} Net</span>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin:10px 0">
        <div style="text-align:center"><div style="font-size:16px;font-weight:600;color:${{offColor}}">${{fmt(tm.off_mean,2)}}</div><div style="font-size:10px;color:var(--text-muted)">OFF Mean</div></div>
        <div style="text-align:center"><div style="font-size:16px;font-weight:600;color:${{defColor}}">${{fmt(tm.def_mean,2)}}</div><div style="font-size:10px;color:var(--text-muted)">DEF Mean</div></div>
        <div style="text-align:center"><div style="font-size:16px;font-weight:600;color:var(--purple)">${{fmt(tm.vol_total,1)}}</div><div style="font-size:10px;color:var(--text-muted)">Volatility</div></div>
      </div>
      <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:4px;font-size:11px;text-align:center;margin-bottom:8px">
        <div><div style="font-weight:600">${{fmt(tm.off_talent,2)}}</div><div style="color:var(--text-muted)">Talent</div></div>
        <div><div style="font-weight:600;color:${{(tm.off_interaction||0)>=0?'var(--green)':'var(--red)'}}">${{fmt(tm.off_interaction,2)}}</div><div style="color:var(--text-muted)">Interact</div></div>
        <div><div style="font-weight:600;color:${{(tm.off_structure||0)>=0?'var(--green)':'var(--red)'}}">${{fmt(tm.off_structure,2)}}</div><div style="color:var(--text-muted)">Structure</div></div>
        <div><div style="font-weight:600">${{tm.n_playmakers}}</div><div style="color:var(--text-muted)">Playmakrs</div></div>
        <div><div style="font-weight:600">${{tm.n_shooters}}</div><div style="color:var(--text-muted)">Shooters</div></div>
      </div>
      <div style="font-size:11px;color:var(--text-muted)">
        Off: ${{topOff.slice(0,3).map(([k,v])=>k.replace(/_/g,' ')+' ('+v+')').join(', ')||'—'}}<br>
        Def: ${{topDef.slice(0,3).map(([k,v])=>k.replace(/_/g,' ')+' ('+v+')').join(', ')||'—'}}
      </div>
    </div>`;
  }});
  html += '</div>';
  view.innerHTML = html;
}}

function openTeamModal(team, season) {{
  const tm = TEAM_AGG.find(t => t.team===team && t.season===season);
  if(!tm) return;
  const m = document.getElementById('teamModalContent');

  // Players table
  let playersHtml = '';
  if(tm.players && tm.players.length) {{
    const sorted = [...tm.players].sort((a,b)=>(b.mpg||0)-(a.mpg||0));
    playersHtml = `<table class="team-table"><thead><tr>
      <th>#</th><th>Player</th><th>MPG</th><th>Min Share</th><th>OBKE</th><th>DBKE</th>
      <th>Off Archetype</th><th>Def Archetype</th><th>USG</th><th>AST</th><th>3P Rate</th><th>Trans</th>
    </tr></thead><tbody>`;
    sorted.forEach((p,i) => {{
      const obkeColor = (p.impact_obke||0)>0?'var(--green)':'var(--red)';
      const dbkeColor = (p.impact_dbke||0)>0?'var(--green)':'var(--red)';
      playersHtml += `<tr>
        <td>${{i+1}}</td>
        <td style="font-weight:600">${{p.player_name}}</td>
        <td>${{fmt(p.mpg,1)}}</td>
        <td>${{fmtPct(p.minute_share)}}</td>
        <td style="color:${{obkeColor}}">${{fmt(p.impact_obke,2)}}</td>
        <td style="color:${{dbkeColor}}">${{fmt(p.impact_dbke,2)}}</td>
        <td style="font-size:11px">${{(p.off_archetype||'').replace(/_/g,' ')}}</td>
        <td style="font-size:11px">${{(p.def_archetype||'').replace(/_/g,' ')}}</td>
        <td>${{fmtPct(p.usage)}}</td>
        <td>${{fmtPct(p.ast_rate)}}</td>
        <td>${{fmtPct(p.three_rate)}}</td>
        <td>${{fmtPct(p.transition_freq)}}</td>
      </tr>`;
    }});
    playersHtml += '</tbody></table>';
  }}

  // Interaction details
  let interHtml = '';
  if(tm.interactions && tm.interactions.length) {{
    const sorted = [...tm.interactions].sort((a,b)=>Math.abs(b.contribution)-Math.abs(a.contribution));
    interHtml = sorted.slice(0,15).map(ix => {{
      const color = ix.contribution>0?'var(--green)':'var(--red)';
      const sign = ix.contribution>0?'+':'';
      return `<div style="display:flex;justify-content:space-between;align-items:center;padding:4px 8px;border-bottom:1px solid var(--border);font-size:12px">
        <div>${{ix.player_i}} <span style="color:var(--text-muted)">${{(ix.arch_i||'').replace(/_/g,' ')}}</span>
        × ${{ix.player_j}} <span style="color:var(--text-muted)">${{(ix.arch_j||'').replace(/_/g,' ')}}</span></div>
        <div style="color:${{color}};font-weight:600">${{sign}}${{fmt(ix.contribution,4)}}</div>
      </div>`;
    }}).join('');
  }}

  // Offense breakdown visual
  const offParts = [
    {{label:'Talent Base', val:tm.off_talent, color:'var(--accent)'}},
    {{label:'Interaction', val:tm.off_interaction, color:tm.off_interaction>=0?'var(--green)':'var(--red)'}},
    {{label:'Structure', val:tm.off_structure, color:tm.off_structure>=0?'var(--green)':'var(--red)'}},
  ];
  const defParts = [
    {{label:'Talent Base', val:tm.def_talent, color:'var(--accent)'}},
    {{label:'Adjustments', val:tm.def_adj, color:tm.def_adj>=0?'var(--green)':'var(--red)'}},
  ];

  m.innerHTML = `
    <button class="modal-close" onclick="closeTeamModal()">&times;</button>
    <h2>${{tm.team}} — ${{tm.season}}</h2>
    <div class="meta">${{tm.n_players}} players · Projected Net Rating: <span style="color:${{(tm.net_projected||0)>0?'var(--green)':'var(--red)'}};font-weight:700">${{fmt(tm.net_projected,2)}}</span> · Volatility: <span style="color:var(--purple)">${{fmt(tm.vol_total,1)}}</span></div>

    <div class="section">
      <h3>Offensive Model Breakdown</h3>
      <div style="display:flex;gap:16px;margin-bottom:12px">
        ${{offParts.map(p=>`<div style="flex:1;background:var(--bg);border-radius:4px;padding:10px;text-align:center">
          <div style="font-size:18px;font-weight:700;color:${{p.color}}">${{fmt(p.val,3)}}</div>
          <div style="font-size:10px;color:var(--text-muted)">${{p.label}}</div>
        </div>`).join('')}}
        <div style="flex:1;background:var(--bg);border-radius:4px;padding:10px;text-align:center">
          <div style="font-size:18px;font-weight:700;color:${{(tm.off_mean||0)>0?'var(--green)':'var(--red)'}}">${{fmt(tm.off_mean,3)}}</div>
          <div style="font-size:10px;color:var(--text-muted)">OFF Total</div>
        </div>
      </div>
      <div class="metric-grid">
        <div class="mg-item"><div class="mg-val">${{fmt(tm.tov_penalty,3)}}</div><div class="mg-lbl">TOV Penalty</div></div>
        <div class="mg-item"><div class="mg-val">${{fmt(tm.ftr_bonus,3)}}</div><div class="mg-lbl">FTR Bonus</div></div>
        <div class="mg-item"><div class="mg-val">${{fmt(tm.playmaking_adj,2)}}</div><div class="mg-lbl">Playmaking (${{tm.n_playmakers}})</div></div>
        <div class="mg-item"><div class="mg-val">${{fmt(tm.spacing_adj,2)}}</div><div class="mg-lbl">Spacing (${{tm.n_shooters}})</div></div>
        <div class="mg-item"><div class="mg-val">${{fmt(tm.transition_bonus,3)}}</div><div class="mg-lbl">Transition</div></div>
        <div class="mg-item"><div class="mg-val">${{fmtPct(tm.transition_freq)}}</div><div class="mg-lbl">Trans Freq</div></div>
      </div>
    </div>

    <div class="section">
      <h3>Defensive Model Breakdown</h3>
      <div style="display:flex;gap:16px;margin-bottom:12px">
        ${{defParts.map(p=>`<div style="flex:1;background:var(--bg);border-radius:4px;padding:10px;text-align:center">
          <div style="font-size:18px;font-weight:700;color:${{p.color}}">${{fmt(p.val,3)}}</div>
          <div style="font-size:10px;color:var(--text-muted)">${{p.label}}</div>
        </div>`).join('')}}
        <div style="flex:1;background:var(--bg);border-radius:4px;padding:10px;text-align:center">
          <div style="font-size:18px;font-weight:700;color:${{(tm.def_mean||0)>0?'var(--green)':'var(--red)'}}">${{fmt(tm.def_mean,3)}}</div>
          <div style="font-size:10px;color:var(--text-muted)">DEF Total</div>
        </div>
      </div>
      <div class="metric-grid">
        <div class="mg-item"><div class="mg-val" style="color:${{tm.rp_penalty<0?'var(--red)':'var(--green)'}}">${{fmt(tm.rp_penalty,2)}}</div><div class="mg-lbl">Rim Protector</div></div>
        <div class="mg-item"><div class="mg-val" style="color:${{tm.poa_penalty<0?'var(--red)':'var(--green)'}}">${{fmt(tm.poa_penalty,2)}}</div><div class="mg-lbl">POA Defender</div></div>
        <div class="mg-item"><div class="mg-val">${{fmt(tm.anchor_bonus,2)}}</div><div class="mg-lbl">Anchor Quality</div></div>
        <div class="mg-item"><div class="mg-val">${{fmt(tm.diversity_bonus,2)}}</div><div class="mg-lbl">Diversity (${{tm.unique_def_archetypes}} types)</div></div>
        <div class="mg-item"><div class="mg-val" style="color:var(--red)">${{fmt(tm.liability_penalty,2)}}</div><div class="mg-lbl">Liability (${{tm.n_liabilities}})</div></div>
      </div>
    </div>

    <div class="section">
      <h3>Volatility Components</h3>
      <div class="metric-grid">
        <div class="mg-item"><div class="mg-val">${{fmt(tm.vol_base,1)}}</div><div class="mg-lbl">Base (League σ)</div></div>
        <div class="mg-item"><div class="mg-val">${{fmt(tm.vol_3pa,2)}}</div><div class="mg-lbl">3PA Impact</div></div>
        <div class="mg-item"><div class="mg-val">${{fmt(tm.vol_creation,2)}}</div><div class="mg-lbl">Creation Conc.</div></div>
        <div class="mg-item"><div class="mg-val">${{fmt(tm.vol_transition,2)}}</div><div class="mg-lbl">Transition</div></div>
        <div class="mg-item"><div class="mg-val" style="color:var(--purple)">${{fmt(tm.vol_total,1)}}</div><div class="mg-lbl">Total σ</div></div>
      </div>
    </div>

    ${{tm.interactions&&tm.interactions.length ? `<div class="section">
      <h3>Top Archetype Interactions</h3>
      <div style="max-height:300px;overflow-y:auto;border:1px solid var(--border);border-radius:4px">${{interHtml}}</div>
    </div>` : ''}}

    <div class="section">
      <h3>Roster (${{tm.n_players}} players)</h3>
      <div style="overflow-x:auto">${{playersHtml||'<div style="color:var(--text-muted)">No player data</div>'}}</div>
    </div>
  `;
  document.getElementById('teamModal').classList.add('open');
}}

function setSort(btn) {{
  const newSort = btn.dataset.sort;
  if(newSort === currentSort) {{
    sortAsc = !sortAsc;
  }} else {{
    currentSort = newSort;
    sortAsc = false;
  }}

  document.querySelectorAll('.sort-btns button').forEach(b=>{{
    b.classList.remove('active');
    const label = b.textContent.replace(' ↑','').replace(' ↓','');
    b.textContent = label;
  }});
  btn.classList.add('active');
  btn.textContent = btn.textContent.replace(' ↑','').replace(' ↓','') + (sortAsc ? ' ↑' : ' ↓');
  applyFilters();
}}

function toggleTeam(sectionId) {{
  const el = document.getElementById(sectionId);
  if(!el) return;
  const btn = el.previousElementSibling;
  const isHidden = el.style.display === 'none';
  el.style.display = isHidden ? 'block' : 'none';
  if(btn) {{
    btn.textContent = (isHidden ? '▼ ' : '▶ ') + btn.textContent.slice(2);
  }}
}}

function switchTab(tab) {{
  document.querySelectorAll('.tab-btns button').forEach(b=>b.classList.remove('active'));
  document.getElementById('cardsView').style.display='none';
  document.getElementById('teamView').style.display='none';
  document.getElementById('teamAggView').style.display='none';
  if(tab==='cards') {{
    document.querySelector('.tab-btns button:nth-child(1)').classList.add('active');
    document.getElementById('cardsView').style.display='';
  }} else if(tab==='team') {{
    document.querySelector('.tab-btns button:nth-child(2)').classList.add('active');
    document.getElementById('teamView').style.display='';
  }} else if(tab==='teamagg') {{
    document.querySelector('.tab-btns button:nth-child(3)').classList.add('active');
    document.getElementById('teamAggView').style.display='';
    renderTeamAggView();
  }}
  applyFilters();
}}

function applyFilters() {{
  const list = getFiltered();
  updateStats(list);
  renderCards(list);
  renderTeamView(list);
  if(document.getElementById('teamAggView').style.display !== 'none') {{
    renderTeamAggView();
  }}
}}

// Init
document.addEventListener('DOMContentLoaded', () => {{
  // Default to latest season
  const sel = document.getElementById('seasonFilter');
  if(sel.options.length > 1) sel.value = sel.options[sel.options.length-1].value;
  const activeSortBtn = document.querySelector('.sort-btns button.active');
  if(activeSortBtn) activeSortBtn.textContent = activeSortBtn.textContent + ' ↓';
  applyFilters();
}});

// Keyboard shortcuts
document.addEventListener('keydown', e => {{
  if(e.key==='Escape') {{ closeModal(); closeTeamModal(); }}
}});
</script>
</body>
</html>"""

    os.makedirs(os.path.dirname(OUTPUT_HTML), exist_ok=True)
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Generated: {OUTPUT_HTML} ({len(html):,} bytes, {len(players)} player-seasons)")
    return OUTPUT_HTML


if __name__ == "__main__":
    generate_html()
