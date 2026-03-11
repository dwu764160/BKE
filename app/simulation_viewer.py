"""
app/simulation_viewer.py
=============================================================================
Generate a standalone interactive HTML viewer for Simulation Core outputs.

View 1 (Step 1):
  - Season simulation stats, charts, sortable table
  - Single-game simulator

View 2 (Step 2):
  - Lineup projection summary cards
  - Clickable team cards (per season)
  - Team modal with starter/rotation/clutch details and validation signals

Inputs:
  reports/simulation_step1_season_results.json
  reports/simulation_step1_validation.json
  reports/simulation_step2_lineup_profiles.json

Output:
  app/simulation.html

Usage:
  python3 app/simulation_viewer.py
=============================================================================
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

SEASON_RESULTS = "reports/simulation_step1_season_results.json"
VALIDATION_RESULTS = "reports/simulation_step1_validation.json"
LINEUP_RESULTS = "reports/simulation_step2_lineup_profiles.json"
FORECAST_RESULTS = "reports/forecast_season_results.json"
FORECAST_VALIDATION = "reports/forecast_validation.json"
FORECAST_LINEUP = "reports/forecast_lineup_profiles.json"
PLAYER_STAT_VALIDATION = "reports/player_stat_sim_validation.json"
PLAYER_STAT_VALIDATION_FORECAST = "reports/player_stat_sim_validation_forecast.json"
OUTPUT_HTML = "app/simulation.html"


def _load_json_if_exists(path: str) -> dict:
  if not os.path.exists(path):
    return {}
  with open(path, "r", encoding="utf-8") as f:
    return json.load(f)


def _scenario_key_from_path(path: Path, stem_prefix: str) -> str:
  stem = path.stem
  if not stem.startswith(stem_prefix):
    return ""
  key = stem[len(stem_prefix):]
  return key.strip("_")


def _discover_scenario_files(base_path: str) -> dict:
  base = Path(base_path)
  parent = base.parent
  stem_prefix = f"{base.stem}_"
  out = {}
  for p in sorted(parent.glob(f"{stem_prefix}*.json")):
    key = _scenario_key_from_path(p, stem_prefix)
    if key:
      out[key] = str(p)
  return out


def _safe(v):
    if isinstance(v, (float, np.floating)):
        if np.isnan(v) or np.isinf(v):
            return None
        return round(float(v), 4)
    if isinstance(v, (int, np.integer)):
        return int(v)
    return v


def generate_html(
    season_path: str = SEASON_RESULTS,
    validation_path: str = VALIDATION_RESULTS,
    lineup_path: str = LINEUP_RESULTS,
    output_path: str = OUTPUT_HTML,
) -> str:
    if not os.path.exists(season_path):
        raise FileNotFoundError(
            f"Missing season results: {season_path}\n"
            "Run: python3 src/simulation/season_sim.py"
        )

    with open(season_path, "r", encoding="utf-8") as f:
        season_data = json.load(f)

    validation_data = {}
    if os.path.exists(validation_path):
        with open(validation_path, "r", encoding="utf-8") as f:
            validation_data = json.load(f)

    lineup_data = {}
    if os.path.exists(lineup_path):
        with open(lineup_path, "r", encoding="utf-8") as f:
            lineup_data = json.load(f)

    data_blob = {
        "config": season_data.get("config", {}),
        "seasons": season_data.get("seasons", {}),
        "validation": validation_data,
        "lineup_step2": lineup_data,
    }

    # Load forecast data if available
    forecast_data = _load_json_if_exists(FORECAST_RESULTS)
    forecast_validation = _load_json_if_exists(FORECAST_VALIDATION)
    forecast_lineup = _load_json_if_exists(FORECAST_LINEUP)

    scenario_result_files = _discover_scenario_files(FORECAST_RESULTS)
    scenario_validation_files = _discover_scenario_files(FORECAST_VALIDATION)
    scenario_lineup_files = _discover_scenario_files(FORECAST_LINEUP)

    # Forecast payload supports both legacy single-scenario and new multi-scenario formats.
    forecast_blob = {
      "config": forecast_data.get("config", {}),
      "seasons": forecast_data.get("seasons", {}),
      "validation": forecast_validation,
      "lineup": forecast_lineup,
      "default_scenario": "default",
      "scenario_labels": {},
      "scenarios": {},
    }

    labels_map = {}
    for source in (forecast_data, forecast_validation, forecast_lineup):
        if isinstance(source, dict) and isinstance(source.get("scenario_labels"), dict):
            labels_map.update(source.get("scenario_labels", {}))

    scenario_keys = set()
    if isinstance(forecast_data.get("scenarios"), dict):
        scenario_keys.update(forecast_data["scenarios"].keys())
    if isinstance(forecast_validation.get("scenarios"), dict):
        scenario_keys.update(forecast_validation["scenarios"].keys())
    if isinstance(forecast_lineup.get("scenarios"), dict):
        scenario_keys.update(forecast_lineup["scenarios"].keys())

    # Only fall back to per-scenario files when combined scenario payloads are absent.
    # This avoids mixing stale files with fresh combined outputs.
    if not scenario_keys:
      scenario_keys.update(scenario_result_files.keys())
      scenario_keys.update(scenario_validation_files.keys())
      scenario_keys.update(scenario_lineup_files.keys())

    if scenario_keys:
        scenarios_blob = {}
        for scenario_key in sorted(scenario_keys):
            scenario_payload = {}
            if isinstance(forecast_data.get("scenarios"), dict):
                scenario_payload = forecast_data.get("scenarios", {}).get(scenario_key, {}) or {}

            # Legacy fallback: read scenario-suffixed season results when combined file is single-scenario.
            if not scenario_payload and scenario_key in scenario_result_files:
                scenario_payload = _load_json_if_exists(scenario_result_files[scenario_key])

            scenario_validation = {}
            if isinstance(forecast_validation.get("scenarios"), dict):
                scenario_validation = forecast_validation.get("scenarios", {}).get(scenario_key, {}) or {}
            if not scenario_validation and scenario_key in scenario_validation_files:
                scenario_validation = _load_json_if_exists(scenario_validation_files[scenario_key])

            scenario_lineup = {}
            if isinstance(forecast_lineup.get("scenarios"), dict):
                scenario_lineup = forecast_lineup.get("scenarios", {}).get(scenario_key, {}) or {}
            if not scenario_lineup and scenario_key in scenario_lineup_files:
              scenario_lineup = _load_json_if_exists(scenario_lineup_files[scenario_key])

            label = (
                scenario_payload.get("label")
                or labels_map.get(scenario_key)
                or scenario_key
            )

            scenarios_blob[scenario_key] = {
                "label": label,
                "config": scenario_payload.get("config", forecast_data.get("config", {})),
                "seasons": scenario_payload.get("seasons", {}),
                "validation": scenario_validation,
                "lineup": scenario_lineup,
            }

        default_scenario = (
            forecast_data.get("default_scenario")
            or forecast_validation.get("default_scenario")
            or forecast_lineup.get("default_scenario")
            or next(iter(scenarios_blob.keys()))
        )
        fallback_key = default_scenario if default_scenario in scenarios_blob else next(iter(scenarios_blob.keys()))
        fallback_payload = scenarios_blob[fallback_key]

        forecast_blob = {
            "config": fallback_payload.get("config", {}),
            "seasons": fallback_payload.get("seasons", {}),
            "validation": fallback_payload.get("validation", {}),
            "lineup": fallback_payload.get("lineup", {}),
            "default_scenario": fallback_key,
            "scenario_labels": {k: v.get("label", k) for k, v in scenarios_blob.items()},
            "scenarios": scenarios_blob,
        }

    data_blob["forecast"] = forecast_blob

    # Load player stat sim validation data
    player_stat_val = _load_json_if_exists(PLAYER_STAT_VALIDATION)
    player_stat_val_forecast = _load_json_if_exists(PLAYER_STAT_VALIDATION_FORECAST)
    data_blob["player_stat_sim"] = {
        "backtest": player_stat_val,
        "forecast": player_stat_val_forecast,
    }

    data_json = json.dumps(data_blob, separators=(",", ":"))
    html = _build_html(data_json)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Generated: {output_path}")
    return output_path


def _build_html(data_json: str) -> str:
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BKE Simulation Core</title>
<style>
:root {
  --bg: #0d1117; --surface: #161b22; --border: #30363d;
  --text: #c9d1d9; --text-muted: #8b949e; --accent: #58a6ff;
  --green: #3fb950; --red: #f85149; --orange: #d29922; --purple: #bc8cff;
  --yellow: #e3b341;
}
* { margin:0; padding:0; box-sizing:border-box; }
body {
  background:var(--bg); color:var(--text);
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;
  font-size:14px;
}
.container { max-width:1440px; margin:0 auto; padding:20px; }
h1 { font-size:26px; font-weight:700; color:var(--accent); margin-bottom:2px; }
.subtitle { color:var(--text-muted); font-size:13px; margin-bottom:16px; }

/* View Tabs */
.view-tabs { display:flex; gap:8px; margin-bottom:16px; flex-wrap:wrap; }
.view-tab {
  border:1px solid var(--border); background:var(--surface); color:var(--text-muted);
  border-radius:8px; padding:8px 12px; font-size:12px; font-weight:700; cursor:pointer;
  text-transform:uppercase; letter-spacing:0.4px;
}
.view-tab.active { background:var(--accent); border-color:var(--accent); color:#fff; }
.view-panel.hidden { display:none; }

/* Season Tabs */
.season-tabs { display:flex; gap:0; margin-bottom:20px; flex-wrap:wrap; }
.season-tab {
  padding:10px 24px; background:var(--surface); border:1px solid var(--border);
  color:var(--text-muted); cursor:pointer; font-size:14px; font-weight:600;
  transition:all 0.15s; user-select:none;
}
.season-tab:first-child { border-radius:8px 0 0 8px; }
.season-tab:last-child { border-radius:0 8px 8px 0; }
.season-tab.active { background:var(--accent); color:#fff; border-color:var(--accent); }
.season-tab:hover:not(.active) { color:var(--text); background:rgba(88,166,255,0.1); }

/* Model Tabs */
.model-tabs { display:flex; gap:8px; margin:-8px 0 16px 0; flex-wrap:wrap; }
.model-tab {
  border:1px solid var(--border); background:var(--surface); color:var(--text-muted);
  border-radius:8px; padding:6px 10px; font-size:12px; font-weight:700; cursor:pointer;
  text-transform:uppercase; letter-spacing:0.35px;
}
.model-tab.active { background:var(--accent); border-color:var(--accent); color:#fff; }

/* Conference Filter Tabs */
.filter-tabs { display:flex; gap:8px; margin:-8px 0 16px 0; flex-wrap:wrap; }
.filter-tab {
  border:1px solid var(--border); background:var(--surface); color:var(--text-muted);
  border-radius:8px; padding:6px 10px; font-size:12px; font-weight:700; cursor:pointer;
  letter-spacing:0.3px;
}
.filter-tab.active { background:var(--green); border-color:var(--green); color:#fff; }

/* Stats Cards */
.stats-row { display:flex; gap:12px; flex-wrap:wrap; margin-bottom:20px; }
.stat-card {
  background:var(--surface); border:1px solid var(--border); border-radius:8px;
  padding:14px 18px; min-width:130px; flex:1;
}
.stat-card .label { font-size:11px; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.5px; }
.stat-card .value { font-size:22px; font-weight:700; margin-top:4px; }
.stat-card .detail { font-size:11px; color:var(--text-muted); margin-top:2px; }
.val-green { color:var(--green); }
.val-accent { color:var(--accent); }
.val-orange { color:var(--orange); }
.val-purple { color:var(--purple); }

/* Charts Row */
.charts-row { display:flex; gap:16px; margin-bottom:20px; flex-wrap:wrap; }
.chart-box {
  background:var(--surface); border:1px solid var(--border); border-radius:8px;
  padding:16px; flex:1; min-width:320px;
}
.chart-box h3 {
  font-size:14px; color:var(--text-muted); margin-bottom:12px; font-weight:600;
  text-transform:uppercase; letter-spacing:0.3px;
}
svg text { fill:var(--text); font-family:inherit; }

/* Single Game Sim */
.single-game-box {
  background:var(--surface); border:1px solid var(--border); border-radius:8px;
  padding:16px; margin-bottom:20px;
}
.single-game-box h3 {
  font-size:14px; color:var(--text-muted); margin-bottom:12px; font-weight:600;
  text-transform:uppercase; letter-spacing:0.3px;
}
.sg-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:10px; margin-bottom:10px; }
.sg-group label {
  display:block; font-size:11px; color:var(--text-muted); margin-bottom:4px;
  text-transform:uppercase; letter-spacing:0.3px;
}
.sg-group select {
  width:100%; background:var(--bg); color:var(--text); border:1px solid var(--border);
  border-radius:6px; padding:8px; font-size:13px;
}
.sg-actions { display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin-bottom:10px; }
.sg-btn {
  background:var(--accent); color:#fff; border:0; border-radius:6px;
  padding:8px 12px; font-weight:700; cursor:pointer;
}
.sg-btn:hover { filter:brightness(1.05); }
.sg-status { font-size:12px; color:var(--text-muted); }
.sg-result {
  background:linear-gradient(180deg, rgba(13,17,23,1) 0%, rgba(15,23,34,1) 100%);
  border:1px solid var(--border);
  border-radius:10px;
  padding:12px;
  font-size:13px;
}
.sg-header {
  display:flex; justify-content:space-between; align-items:center; gap:10px;
  margin-bottom:10px; flex-wrap:wrap;
}
.sg-matchup {
  font-size:15px; font-weight:700;
}
.sg-badge-home { color:var(--accent); }
.sg-badge-away { color:var(--orange); }
.sg-time { color:var(--text-muted); font-size:11px; }
.sg-kpi-grid {
  display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr));
  gap:8px; margin-bottom:10px;
}
.sg-kpi {
  border:1px solid var(--border);
  background:rgba(88,166,255,0.05);
  border-radius:8px;
  padding:8px;
}
.sg-kpi .k { color:var(--text-muted); font-size:11px; text-transform:uppercase; letter-spacing:0.4px; }
.sg-kpi .v { font-size:16px; font-weight:700; margin-top:3px; }
.sg-detail-grid {
  display:grid; grid-template-columns:repeat(auto-fit, minmax(260px, 1fr));
  gap:10px;
}
.sg-card {
  border:1px solid var(--border);
  border-radius:8px;
  padding:10px;
  background:var(--bg);
}
.sg-card h4 {
  font-size:11px;
  color:var(--text-muted);
  text-transform:uppercase;
  letter-spacing:0.4px;
  margin-bottom:8px;
}
.sg-list { display:grid; gap:4px; }
.sg-list .row {
  display:flex; justify-content:space-between; gap:10px;
  font-size:12px;
}
.sg-list .row .k { color:var(--text-muted); }
.sg-list .row .v { font-variant-numeric:tabular-nums; }
.sg-edge { margin-bottom:7px; }
.sg-edge:last-child { margin-bottom:0; }
.sg-edge-top {
  display:flex; justify-content:space-between; gap:8px;
  font-size:11px; margin-bottom:3px;
}
.sg-edge-top .label { color:var(--text-muted); }
.sg-edge-track {
  position:relative; height:8px; border-radius:99px;
  background:#1f2730; overflow:hidden;
}
.sg-edge-home {
  position:absolute; left:0; top:0; bottom:0;
  background:linear-gradient(90deg, rgba(88,166,255,0.75), rgba(88,166,255,0.45));
}
.sg-edge-away {
  position:absolute; right:0; top:0; bottom:0;
  background:linear-gradient(270deg, rgba(210,153,34,0.8), rgba(210,153,34,0.45));
}
.sg-contrib-grid {
  display:grid; grid-template-columns:1fr 1fr; gap:8px;
}
.sg-contrib-col { border:1px solid var(--border); border-radius:6px; padding:8px; }
.sg-contrib-col .title { font-size:11px; font-weight:700; margin-bottom:6px; }
.sg-contrib-col .title.home { color:var(--accent); }
.sg-contrib-col .title.away { color:var(--orange); }
.sg-contrib-item {
  display:flex; justify-content:space-between; gap:8px;
  font-size:12px; margin-bottom:4px;
}
.sg-contrib-item:last-child { margin-bottom:0; }
.sg-contrib-item .name { white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.sg-contrib-item .val { font-variant-numeric:tabular-nums; }

/* Chart Tooltip (hover popup for scatter plot) */
.chart-tooltip {
  position:fixed; pointer-events:none; z-index:9999;
  background:var(--surface); border:1px solid var(--accent); border-radius:8px;
  padding:8px 12px; font-size:12px; color:var(--text);
  box-shadow:0 4px 16px rgba(0,0,0,0.5);
  max-width:280px; display:none;
  line-height:1.5;
}
.chart-tooltip .tt-name { font-weight:700; color:var(--accent); font-size:13px; margin-bottom:2px; }
.chart-tooltip .tt-team { color:var(--text-muted); font-size:11px; }
.chart-tooltip .tt-row { display:flex; justify-content:space-between; gap:12px; }
.chart-tooltip .tt-label { color:var(--text-muted); }
.chart-tooltip .tt-val { font-weight:600; font-variant-numeric:tabular-nums; }

/* Table */
.table-wrap { border:1px solid var(--border); border-radius:8px; overflow:hidden; }
.table-scroll { max-height:72vh; overflow-y:auto; }
table { width:100%; border-collapse:collapse; font-size:13px; }
thead { position:sticky; top:0; z-index:10; }
th {
  background:var(--surface); color:var(--text-muted); text-align:left;
  padding:10px 12px; border-bottom:2px solid var(--border); cursor:pointer;
  user-select:none; white-space:nowrap; font-size:11px; text-transform:uppercase;
  letter-spacing:0.4px;
}
th:hover { color:var(--accent); }
th.sorted-asc::after { content:" ▲"; color:var(--accent); }
th.sorted-desc::after { content:" ▼"; color:var(--accent); }
td { padding:8px 12px; border-bottom:1px solid var(--border); white-space:nowrap; }
tr:hover { background:rgba(88,166,255,0.06); }
.rank { font-weight:700; color:var(--accent); text-align:right; width:36px; }
.team { font-weight:700; font-size:14px; }
.num { text-align:right; font-variant-numeric:tabular-nums; }
.err-good { color:var(--green); }
.err-ok { color:var(--orange); }
.err-bad { color:var(--red); }
.conf-band { color:var(--text-muted); font-size:12px; }

/* Step 2 Grid */
.lineup-grid {
  display:grid;
  grid-template-columns:repeat(auto-fill, minmax(260px, 1fr));
  gap:12px;
  margin-bottom:20px;
}
.lineup-card {
  background:var(--surface);
  border:1px solid var(--border);
  border-radius:10px;
  padding:12px;
  cursor:pointer;
  transition:transform 0.12s ease, border-color 0.12s ease;
}
.lineup-card:hover { transform:translateY(-2px); border-color:var(--accent); }
.lineup-card .title-row {
  display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;
}
.lineup-card .title-row .team-name { font-size:16px; font-weight:700; color:var(--accent); }
.lineup-card .title-row .conf { font-size:11px; color:var(--text-muted); text-transform:uppercase; }
.lineup-card .mini {
  display:grid;
  grid-template-columns:1fr 1fr;
  gap:6px;
  font-size:12px;
  margin-bottom:8px;
}
.lineup-card .mini .k { color:var(--text-muted); }
.lineup-card .mini .v { text-align:right; font-variant-numeric:tabular-nums; }
.lineup-card .validation {
  font-size:12px;
  border-top:1px solid var(--border);
  padding-top:8px;
  color:var(--text-muted);
  display:flex;
  justify-content:space-between;
  gap:8px;
}

/* Step 2 Modal */
.lineup-modal-overlay {
  position:fixed; inset:0; background:rgba(0,0,0,0.62); z-index:1000;
  display:none; align-items:center; justify-content:center; padding:16px;
}
.lineup-modal-overlay.open { display:flex; }
.lineup-modal {
  width:min(1100px, 96vw); max-height:92vh; overflow:auto;
  background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:14px;
}
.lineup-modal .modal-head {
  display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;
}
.lineup-modal .modal-title { font-size:22px; color:var(--accent); font-weight:700; }
.lineup-close {
  border:1px solid var(--border); background:var(--bg); color:var(--text);
  border-radius:8px; padding:6px 10px; cursor:pointer; font-size:12px;
}
.lineup-close:hover { border-color:var(--accent); color:var(--accent); }
.lineup-modal-grid {
  display:grid; grid-template-columns:1fr 1fr; gap:12px;
}
.lineup-panel {
  border:1px solid var(--border); border-radius:8px; padding:10px; background:var(--bg);
}
.lineup-panel h4 {
  font-size:12px; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.4px;
  margin-bottom:8px;
}
.lineup-panel .kv {
  display:grid; grid-template-columns:1fr auto; gap:6px 10px; font-size:13px;
}
.lineup-panel .kv .k { color:var(--text-muted); }
.lineup-panel .kv .v { font-variant-numeric:tabular-nums; }
.player-list {
  display:grid; gap:6px;
}
.player-item {
  border:1px solid var(--border); border-radius:6px; padding:6px;
  display:grid; grid-template-columns:1fr auto auto; gap:8px;
  align-items:center;
  font-size:12px;
}
.player-item .name { font-weight:600; }
.player-item .role { color:var(--text-muted); }
.player-item .meta { display:grid; gap:2px; }
.player-item .arche { color:var(--text-muted); font-size:11px; }
.player-item .num { font-variant-numeric:tabular-nums; text-align:right; }

/* Footer */
.footer {
  margin-top:24px; padding:16px; color:var(--text-muted); font-size:11px;
  text-align:center; border-top:1px solid var(--border);
}

@media (max-width: 900px) {
  .lineup-modal-grid { grid-template-columns:1fr; }
}
</style>
</head>
<body>
<div class="chart-tooltip" id="chartTooltip"></div>
<div class="container">
  <h1>BKE Simulation Core</h1>
  <div class="subtitle">Step 1: Season Simulation + Step 2: Lineup Projection + Player Stat Sim + Single Game + Forecast</div>

  <div class="view-tabs" id="viewTabs">
    <button class="view-tab active" data-view="step1View">Step 1: Season Sim</button>
    <button class="view-tab" data-view="step2View">Step 2: Lineup</button>
    <button class="view-tab" data-view="playerStatSimView">Player Stat Sim</button>
    <button class="view-tab" data-view="singleGameView">Single Game Sim</button>
    <button class="view-tab" data-view="step1ForecastView">Step 1 Forecast</button>
    <button class="view-tab" data-view="step2ForecastView">Step 2 Forecast</button>
  </div>

  <div id="step1View" class="view-panel">
    <div class="season-tabs" id="seasonTabs"></div>
    <div class="model-tabs" id="step1ModelTabs"></div>
    <div class="filter-tabs" id="step1ConferenceTabs"></div>
    <div class="stats-row" id="statsRow"></div>
    <div class="charts-row">
      <div class="chart-box" style="flex:2"><h3>Projected vs Actual Wins</h3><div id="winsChart"></div></div>
      <div class="chart-box" style="flex:1"><h3>Calibration</h3><div id="calChart"></div></div>
    </div>

    <div class="table-wrap"><div class="table-scroll" id="tableWrap"></div></div>
  </div>

  <div id="step2View" class="view-panel hidden">
    <div class="season-tabs" id="lineupSeasonTabs"></div>
    <div class="filter-tabs" id="step2ConferenceTabs"></div>
    <div class="stats-row" id="lineupStatsRow"></div>
    <div class="lineup-grid" id="lineupCardGrid"></div>
  </div>

  <div id="playerStatSimView" class="view-panel hidden">
    <div class="model-tabs" id="pssModeTabs"></div>
    <div class="season-tabs" id="pssSeasonTabs"></div>
    <div class="filter-tabs" id="pssTeamTabs"></div>
    <div class="stats-row" id="pssStatsRow"></div>
    <div class="charts-row">
      <div class="chart-box" style="flex:2"><h3>Simulated vs Actual PPG</h3><div id="pssPtsChart"></div></div>
      <div class="chart-box" style="flex:1"><h3>Stat Accuracy Summary</h3><div id="pssAccChart"></div></div>
    </div>
    <div style="margin-bottom:10px">
      <input type="text" id="pssSearch" placeholder="Search player name..." style="background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:6px;padding:8px 12px;width:300px;font-size:13px;">
    </div>
    <div class="table-wrap"><div class="table-scroll" id="pssTableWrap" style="max-height:80vh"></div></div>
  </div>

  <div id="singleGameView" class="view-panel hidden">
    <div class="single-game-box">
      <h3>Single Game Simulator</h3>
      <div class="sg-grid">
        <div class="sg-group"><label for="sgHomeSeason">Home Season</label><select id="sgHomeSeason"></select></div>
        <div class="sg-group"><label for="sgHomeTeam">Home Team</label><select id="sgHomeTeam"></select></div>
        <div class="sg-group"><label for="sgAwaySeason">Away Season</label><select id="sgAwaySeason"></select></div>
        <div class="sg-group"><label for="sgAwayTeam">Away Team</label><select id="sgAwayTeam"></select></div>
      </div>
      <div class="sg-actions">
        <button class="sg-btn" id="sgRunBtn">Simulate 1 Game</button>
        <span class="sg-status" id="sgStatus">Uses one random draw from the margin distribution (not 10,000 sims).</span>
      </div>
      <div class="sg-result" id="sgResult">No game simulated yet.</div>
    </div>
  </div>

  <div id="step1ForecastView" class="view-panel hidden">
    <div class="model-tabs" id="forecastScenarioTabs"></div>
    <div class="season-tabs" id="forecastSeasonTabs"></div>
    <div class="model-tabs" id="forecastModelTabs"></div>
    <div class="filter-tabs" id="forecastConferenceTabs"></div>
    <div class="stats-row" id="forecastStatsRow"></div>
    <div class="charts-row">
      <div class="chart-box" style="flex:2"><h3>Projected Wins (Forecast)</h3><div id="forecastWinsChart"></div></div>
      <div class="chart-box" style="flex:1"><h3>Projection Validation</h3><div id="forecastValChart"></div></div>
    </div>
    <div class="table-wrap"><div class="table-scroll" id="forecastTableWrap"></div></div>
  </div>

  <div id="step2ForecastView" class="view-panel hidden">
    <div class="model-tabs" id="forecastScenarioTabsStep2"></div>
    <div class="season-tabs" id="forecastSeasonTabsStep2"></div>
    <div class="stats-row" id="forecastLineupStatsRow"></div>
    <div class="filter-tabs" id="forecastLineupConferenceTabs"></div>
    <div class="lineup-grid" id="forecastLineupCardGrid"></div>
  </div>

  <div class="lineup-modal-overlay" id="lineupModalOverlay">
    <div class="lineup-modal">
      <div class="modal-head">
        <div class="modal-title" id="lineupModalTitle">Lineup Details</div>
        <button class="lineup-close" id="lineupModalClose">Close</button>
      </div>
      <div id="lineupModalContent"></div>
    </div>
  </div>

  <div class="footer">
    BKE Simulation Core &mdash; Step 1 (season sim), Step 2 (lineup), Player Stat Sim, Single Game Sim, Forecast.
    &sigma;<sub>league</sub> = <span id="footSigma"></span>,
    HCA = <span id="footHCA"></span>,
    N = <span id="footN"></span>
  </div>
</div>

<script>
const DATA = __DATA_JSON__;
const STEP2 = DATA.lineup_step2 || {};
const FORECAST = DATA.forecast || {};
const PSS_DATA = DATA.player_stat_sim || {};

let currentSeason = null;
let lineupSeason = null;
let forecastSeason = null;
let forecastScenario = null;
let step1Model = "margin";
let forecastModel = "margin";
let sortCol = "projected_rank";
let sortDir = "asc";
let forecastSortCol = "projected_rank";
let forecastSortDir = "asc";
let step1ConferenceFilter = "all";
let step2ConferenceFilter = "all";
let forecastConferenceFilter = "all";
let forecastLineupConferenceFilter = "all";
let currentSimulatedGame = null;
let activeView = "step1View";
let pssMode = "backtest";
let pssSeason = null;
let pssTeamFilter = "all";
let pssSortCol = "sim_pts";
let pssSortDir = "desc";
let pssSearchText = "";
const STEP2_SEASON_PLAYER_MAP = {};
const FORECAST_STEP2_SEASON_PLAYER_MAP = {};

function normalizeConference(conf) {
  const text = String(conf || "").trim().toLowerCase();
  if (text === "east" || text === "eastern") return "east";
  if (text === "west" || text === "western") return "west";
  return "unknown";
}

function conferenceFilterLabel(key) {
  if (key === "east") return "East";
  if (key === "west") return "West";
  return "All Teams";
}

function filterRowsByConference(rows, filterKey) {
  if (!Array.isArray(rows)) return [];
  if (!filterKey || filterKey === "all") return rows;
  return rows.filter(r => normalizeConference(r && r.conference) === filterKey);
}

function getStep1Seasons() {
  return Object.keys(DATA.seasons || {}).sort();
}

function getForecastScenarioKeys() {
  const scenarioMap = (FORECAST && FORECAST.scenarios) || {};
  const keys = Object.keys(scenarioMap || {});
  if (keys.length) {
    const preferred = ["end_of_season", "preseason_snapshot"];
    const ordered = [];
    preferred.forEach(k => {
      if (keys.includes(k)) ordered.push(k);
    });
    keys.sort().forEach(k => {
      if (!ordered.includes(k)) ordered.push(k);
    });
    return ordered;
  }
  return ["default"];
}

function getForecastScenarioLabel(key) {
  const labels = (FORECAST && FORECAST.scenario_labels) || {};
  if (labels[key]) return labels[key];
  if (key === "end_of_season") return "End-of-Season Roster (Peek)";
  if (key === "preseason_snapshot") return "Preseason Roster Snapshot";
  if (key === "default") return "Forecast";
  return String(key || "").replaceAll("_", " ");
}

function getForecastScenarioPayload(scenarioKey) {
  const map = (FORECAST && FORECAST.scenarios) || {};
  if (map[scenarioKey]) return map[scenarioKey];
  return {
    config: (FORECAST && FORECAST.config) || {},
    seasons: (FORECAST && FORECAST.seasons) || {},
    validation: (FORECAST && FORECAST.validation) || {},
    lineup: (FORECAST && FORECAST.lineup) || {},
  };
}

function getForecastSeasons() {
  const payload = getForecastScenarioPayload(forecastScenario);
  return Object.keys((payload && payload.seasons) || {}).sort();
}

function _renderForecastSeasonTabs(containerId) {
  const seasons = getForecastSeasons();
  const tabsEl = document.getElementById(containerId);
  if (!tabsEl) return false;

  tabsEl.innerHTML = "";
  if (seasons.length === 0) {
    tabsEl.innerHTML = '<div style="color:var(--text-muted);padding:10px">No forecast data available for this scenario. Run: python3 src/simulation/run_forecast.py</div>';
    return false;
  }

  if (!seasons.includes(forecastSeason)) {
    forecastSeason = seasons[seasons.length - 1];
  }

  seasons.forEach(s => {
    const tab = document.createElement("div");
    tab.className = "season-tab" + (s === forecastSeason ? " active" : "");
    tab.textContent = s + " (Forecast)";
    tab.onclick = () => setForecastSeason(s);
    tabsEl.appendChild(tab);
  });
  return true;
}

function setForecastScenario(scenarioKey) {
  forecastScenario = scenarioKey;
  const seasons = getForecastSeasons();
  forecastSeason = seasons.length ? seasons[seasons.length - 1] : null;
  forecastModel = "margin";
  forecastSortCol = "projected_rank";
  forecastSortDir = "asc";
  initForecastView();
  initForecastStep2View();
  renderForecast();
}

function setForecastSeason(season) {
  forecastSeason = season;
  syncForecastModel();
  initForecastView();
  initForecastStep2View();
  renderForecast();
}

function getStep2Seasons() {
  return Object.keys((STEP2 && STEP2.seasons) || {}).sort();
}

function modelLabel(modelKey) {
  if (modelKey === "margin") return "Margin";
  if (modelKey === "ppp") return "PPP";
  return String(modelKey || "").toUpperCase();
}

function orderedModels(modelMap, configModels) {
  const keys = Object.keys(modelMap || {});
  const out = [];
  (configModels || []).forEach(m => {
    if (keys.includes(m) && !out.includes(m)) out.push(m);
  });
  keys.sort().forEach(m => {
    if (!out.includes(m)) out.push(m);
  });
  return out;
}

function getSeasonModelMap(sdata, defaultModel) {
  if (sdata && sdata.team_results_by_model && Object.keys(sdata.team_results_by_model).length) {
    return sdata.team_results_by_model;
  }
  const map = {};
  const fallbackModel = (sdata && sdata.default_model) || defaultModel || "margin";
  map[fallbackModel] = (sdata && sdata.team_results) || [];
  return map;
}

function getSeasonStatsMap(sdata, defaultModel) {
  if (sdata && sdata.season_stats_by_model && Object.keys(sdata.season_stats_by_model).length) {
    return sdata.season_stats_by_model;
  }
  const map = {};
  const fallbackModel = (sdata && sdata.default_model) || defaultModel || "margin";
  map[fallbackModel] = (sdata && sdata.season_stats) || {};
  return map;
}

function getStep1RowsForSeason(season, modelKey) {
  const sdata = (DATA.seasons || {})[season] || {};
  const modelMap = getSeasonModelMap(sdata, "margin");
  if (modelMap[modelKey]) return modelMap[modelKey];
  const fallback = sdata.default_model && modelMap[sdata.default_model] ? sdata.default_model : Object.keys(modelMap)[0];
  return fallback ? modelMap[fallback] : [];
}

function getForecastRowsForSeason(season, modelKey) {
  const payload = getForecastScenarioPayload(forecastScenario);
  const sdata = ((payload && payload.seasons) || {})[season] || {};
  const modelMap = getSeasonModelMap(sdata, "margin");
  if (modelMap[modelKey]) return modelMap[modelKey];
  const fallback = sdata.default_model && modelMap[sdata.default_model] ? sdata.default_model : Object.keys(modelMap)[0];
  return fallback ? modelMap[fallback] : [];
}

function getForecastStatsForSeason(season, modelKey) {
  const payload = getForecastScenarioPayload(forecastScenario);
  const sdata = ((payload && payload.seasons) || {})[season] || {};
  const statsMap = getSeasonStatsMap(sdata, "margin");
  if (statsMap[modelKey]) return statsMap[modelKey];
  const fallback = sdata.default_model && statsMap[sdata.default_model] ? sdata.default_model : Object.keys(statsMap)[0];
  return fallback ? statsMap[fallback] : {};
}

function syncStep1Model() {
  const sdata = (DATA.seasons || {})[currentSeason] || {};
  const modelMap = getSeasonModelMap(sdata, "margin");
  const models = orderedModels(modelMap, (DATA.config || {}).models || []);
  if (!models.length) {
    step1Model = "margin";
    return;
  }
  if (!models.includes(step1Model)) {
    step1Model = models.includes(sdata.default_model) ? sdata.default_model : models[0];
  }
}

function syncForecastModel() {
  const payload = getForecastScenarioPayload(forecastScenario);
  const sdata = ((payload && payload.seasons) || {})[forecastSeason] || {};
  const modelMap = getSeasonModelMap(sdata, "margin");
  const models = orderedModels(modelMap, ((payload && payload.config) || {}).models || []);
  if (!models.length) {
    forecastModel = "margin";
    return;
  }
  if (!models.includes(forecastModel)) {
    forecastModel = models.includes(sdata.default_model) ? sdata.default_model : models[0];
  }
}

function initStep1ModelTabs() {
  const el = document.getElementById("step1ModelTabs");
  if (!el) return;
  const sdata = (DATA.seasons || {})[currentSeason] || {};
  const modelMap = getSeasonModelMap(sdata, "margin");
  const models = orderedModels(modelMap, (DATA.config || {}).models || []);
  if (!models.length) {
    el.innerHTML = "";
    return;
  }
  syncStep1Model();
  el.innerHTML = models.map(m => `
    <button class="model-tab${m === step1Model ? " active" : ""}" data-model="${m}">${modelLabel(m)}</button>
  `).join("");
  el.querySelectorAll(".model-tab").forEach(btn => {
    btn.addEventListener("click", () => {
      step1Model = btn.dataset.model;
      sortCol = "projected_rank";
      sortDir = "asc";
      initStep1ModelTabs();
      renderStep1();
    });
  });
}

function initForecastModelTabs() {
  const el = document.getElementById("forecastModelTabs");
  if (!el) return;
  const payload = getForecastScenarioPayload(forecastScenario);
  const sdata = ((payload && payload.seasons) || {})[forecastSeason] || {};
  const modelMap = getSeasonModelMap(sdata, "margin");
  const models = orderedModels(modelMap, ((payload && payload.config) || {}).models || []);
  if (!models.length) {
    el.innerHTML = "";
    return;
  }
  syncForecastModel();
  el.innerHTML = models.map(m => `
    <button class="model-tab${m === forecastModel ? " active" : ""}" data-model="${m}">${modelLabel(m)}</button>
  `).join("");
  el.querySelectorAll(".model-tab").forEach(btn => {
    btn.addEventListener("click", () => {
      forecastModel = btn.dataset.model;
      forecastSortCol = "projected_rank";
      forecastSortDir = "asc";
      initForecastModelTabs();
      renderForecast();
    });
  });
}

function _renderForecastScenarioTabs(containerId) {
  const el = document.getElementById(containerId);
  if (!el) return;
  const keys = getForecastScenarioKeys();
  if (!keys.length) {
    el.innerHTML = "";
    return;
  }
  if (!keys.includes(forecastScenario)) {
    forecastScenario = keys[0];
  }
  if (keys.length === 1 && keys[0] === "default") {
    el.innerHTML = "";
    return;
  }
  el.innerHTML = keys.map(k => `
    <button class="model-tab${k === forecastScenario ? " active" : ""}" data-scenario="${k}">${getForecastScenarioLabel(k)}</button>
  `).join("");
  el.querySelectorAll(".model-tab").forEach(btn => {
    btn.addEventListener("click", () => {
      setForecastScenario(btn.dataset.scenario);
    });
  });
}

function initForecastScenarioTabs() {
  _renderForecastScenarioTabs("forecastScenarioTabs");
}

function initForecastScenarioTabsStep2() {
  _renderForecastScenarioTabs("forecastScenarioTabsStep2");
}

(function init() {
  const seasons = getStep1Seasons();
  currentSeason = seasons.length ? seasons[seasons.length - 1] : null;

  const scenarioKeys = getForecastScenarioKeys();
  const preferredScenario = (FORECAST && FORECAST.default_scenario) || null;
  forecastScenario = scenarioKeys.includes(preferredScenario) ? preferredScenario : scenarioKeys[0];
  const fSeasons = getForecastSeasons();
  forecastSeason = fSeasons.length ? fSeasons[fSeasons.length - 1] : null;
  syncStep1Model();
  syncForecastModel();

  const cfg = DATA.config || {};
  document.getElementById("footSigma").textContent = cfg.sigma_league || "?";
  document.getElementById("footHCA").textContent = cfg.home_court_advantage || "?";
  document.getElementById("footN").textContent = (cfg.n_simulations || 0).toLocaleString();

  initStep1SeasonTabs();
  initStep1ModelTabs();
  initStep1ConferenceTabs();
  initSingleGameSimulator(seasons);
  initViewTabs();
  initStep2View();
  initForecastView();
  initForecastStep2View();
  bindModalEvents();
  renderStep1();
})();

function initViewTabs() {
  document.querySelectorAll("#viewTabs .view-tab").forEach(btn => {
    btn.addEventListener("click", () => {
      activeView = btn.dataset.view;
      document.querySelectorAll("#viewTabs .view-tab").forEach(b => b.classList.toggle("active", b === btn));
      document.querySelectorAll(".view-panel").forEach(panel => {
        panel.classList.toggle("hidden", panel.id !== activeView);
      });
      if (activeView === "step1View") {
        renderStep1();
      } else if (activeView === "step2View") {
        renderStep2();
      } else if (activeView === "playerStatSimView") {
        initPlayerStatSimView();
        renderPlayerStatSim();
      } else if (activeView === "singleGameView") {
        // already initialized
      } else if (activeView === "step1ForecastView") {
        renderForecast();
      } else if (activeView === "step2ForecastView") {
        renderForecastLineup();
      }
    });
  });
}

function initStep1SeasonTabs() {
  const seasons = getStep1Seasons();
  const tabsEl = document.getElementById("seasonTabs");
  tabsEl.innerHTML = "";
  seasons.forEach(s => {
    const tab = document.createElement("div");
    tab.className = "season-tab" + (s === currentSeason ? " active" : "");
    tab.textContent = s;
    tab.onclick = () => switchSeason(s);
    tabsEl.appendChild(tab);
  });
}

function switchSeason(s) {
  currentSeason = s;
  document.querySelectorAll("#seasonTabs .season-tab").forEach(t => {
    t.classList.toggle("active", t.textContent === s);
  });
  syncStep1Model();
  initStep1ModelTabs();
  sortCol = "projected_rank";
  sortDir = "asc";
  renderStep1();
}

function initStep1ConferenceTabs() {
  const el = document.getElementById("step1ConferenceTabs");
  if (!el) return;
  const options = [
    { key: "all", label: "All Teams" },
    { key: "east", label: "East" },
    { key: "west", label: "West" },
  ];
  el.innerHTML = options.map(opt => `
    <button class="filter-tab${opt.key === step1ConferenceFilter ? " active" : ""}" data-filter="${opt.key}">${opt.label}</button>
  `).join("");
  el.querySelectorAll(".filter-tab").forEach(btn => {
    btn.addEventListener("click", () => {
      step1ConferenceFilter = btn.dataset.filter || "all";
      initStep1ConferenceTabs();
      renderStep1();
    });
  });
}

function renderStep1() {
  if (!currentSeason) return;
  const sdata = DATA.seasons[currentSeason];
  if (!sdata) return;
  syncStep1Model();
  const rows = getStep1RowsForSeason(currentSeason, step1Model);
  const filteredRows = filterRowsByConference(rows, step1ConferenceFilter);
  renderStats(sdata, step1Model, filteredRows);
  renderTable(sdata, step1Model, filteredRows);
  renderWinsChart(sdata, step1Model, filteredRows);
  renderCalibration();
}

function renderStats(sdata, modelKey, filteredRows) {
  const ssMap = getSeasonStatsMap(sdata, "margin");
  const ss = ssMap[modelKey] || sdata.season_stats || {};
  const vg = (DATA.validation && DATA.validation.game_level && DATA.validation.game_level[currentSeason]) || {};
  const teams = Array.isArray(filteredRows) ? filteredRows : getStep1RowsForSeason(currentSeason, modelKey);
  const isMarginModel = modelKey === "margin";

  const cards = [
    { label:"Model", value: modelLabel(modelKey), cls:"val-purple", detail:"Projection view" },
    { label:"Correlation", value: fmt(ss.correlation, 3), cls:"val-green", detail:"Proj vs Actual wins" },
    { label:"MAE", value: fmt(ss.mae, 2), cls:"val-accent", detail:"Avg win error" },
    { label:"RMSE", value: fmt(ss.rmse, 2), cls:"val-accent", detail:"Root mean sq error" },
    { label:"Brier Score", value: isMarginModel ? fmt(vg.brier_score, 4) : "-", cls:"val-orange", detail:isMarginModel?"Game-level (0.25=naive)":"Margin-model only" },
    { label:"Accuracy", value: isMarginModel && vg.accuracy ? (vg.accuracy * 100).toFixed(1) + "%" : "-", cls:"val-green", detail:isMarginModel?"Game pick accuracy":"Margin-model only" },
    { label:"Log Loss", value: isMarginModel ? fmt(vg.log_loss, 4) : "-", cls:"val-purple", detail:isMarginModel?"(0.693=coin flip)":"Margin-model only" },
    { label:"Margin RMSE", value: isMarginModel ? fmt(vg.margin_rmse, 1) : "-", cls:"val-accent", detail:isMarginModel?"Predicted vs actual margin":"Margin-model only" },
    { label:"Teams", value: teams.length, cls:"val-accent", detail: `${currentSeason} · ${conferenceFilterLabel(step1ConferenceFilter)}` },
  ];

  const el = document.getElementById("statsRow");
  el.innerHTML = cards.map(c => `
    <div class="stat-card">
      <div class="label">${c.label}</div>
      <div class="value ${c.cls}">${c.value || "-"}</div>
      <div class="detail">${c.detail}</div>
    </div>
  `).join("");
}

function renderTable(sdata, modelKey, filteredRows) {
  let teams = [...(Array.isArray(filteredRows) ? filteredRows : filterRowsByConference(getStep1RowsForSeason(currentSeason, modelKey), step1ConferenceFilter))];
  teams.sort((a, b) => {
    let va = a[sortCol], vb = b[sortCol];
    if (va == null) va = 0;
    if (vb == null) vb = 0;
    return sortDir === "asc" ? (va > vb ? 1 : va < vb ? -1 : 0) : (va < vb ? 1 : va > vb ? -1 : 0);
  });

  const cols = [
    { key:"projected_rank", label:"Rank", cls:"rank" },
    { key:"team", label:"Team", cls:"team" },
    { key:"conference", label:"Conf", cls:"num" },
    { key:"projected_conf_rank", label:"Conf Rk", cls:"num", fmt:v=>v!=null?Number(v).toFixed(1):"-" },
    { key:"mu", label:modelKey === "ppp" ? "Net PPP" : "Net Rtg", cls:"num", fmt:v=>v!=null?Number(v).toFixed(2):"-" },
    { key:"sigma", label:"Vol", cls:"num", fmt:v=>v!=null?Number(v).toFixed(2):"-" },
    { key:"predicted_pace", label:"Pace", cls:"num", fmt:v=>v!=null?Number(v).toFixed(2):"-" },
    { key:"ppp_offense", label:"Off PPP", cls:"num", fmt:v=>v!=null?Number(v).toFixed(3):"-" },
    { key:"ppp_defense", label:"Def PPP", cls:"num", fmt:v=>v!=null?Number(v).toFixed(3):"-" },
    { key:"projected_wins", label:"Proj W", cls:"num", fmt:v=>v!=null?Number(v).toFixed(1):"-" },
    { key:"win_p5", label:"90% CI", cls:"conf-band", fmt:(v,r)=>`${r.win_p5}-${r.win_p95}` },
    { key:"actual_wins", label:"Actual W", cls:"num", fmt:v=>v!=null?v:"-" },
    { key:"win_error", label:"Error", cls:"num", fmt:(v) => {
      if (v == null) return "-";
      const abs = Math.abs(v);
      const sign = v > 0 ? "+" : "";
      const cls = abs <= 3 ? "err-good" : abs <= 7 ? "err-ok" : "err-bad";
      return `<span class="${cls}">${sign}${Number(v).toFixed(1)}</span>`;
    } },
    { key:"direct_playoff_probability", label:"Top 6 %", cls:"num", fmt:v=>v!=null?(v*100).toFixed(0)+"%":"-" },
    { key:"top_10_probability", label:"Top 10 %", cls:"num", fmt:v=>v!=null?(v*100).toFixed(0)+"%":"-" },
    { key:"playin_only_probability", label:"7-10 %", cls:"num", fmt:v=>v!=null?(v*100).toFixed(0)+"%":"-" },
    { key:"win_std", label:"Win SD", cls:"num", fmt:v=>v!=null?Number(v).toFixed(2):"-" },
  ];

  let html = "<table><thead><tr>";
  cols.forEach(c => {
    const cls = c.key === sortCol ? (sortDir === "asc" ? "sorted-asc" : "sorted-desc") : "";
    html += `<th class="${cls}" data-col="${c.key}">${c.label}</th>`;
  });
  html += "</tr></thead><tbody>";

  teams.forEach(t => {
    html += "<tr>";
    cols.forEach(c => {
      const raw = c.key === "win_p5" && c.label === "90% CI" ? t["win_p5"] : t[c.key];
      const display = c.fmt ? c.fmt(raw, t) : (raw != null ? raw : "-");
      html += `<td class="${c.cls || ""}">${display}</td>`;
    });
    html += "</tr>";
  });

  html += "</tbody></table>";
  document.getElementById("tableWrap").innerHTML = html;

  document.querySelectorAll("#tableWrap th").forEach(th => {
    th.addEventListener("click", () => {
      const col = th.dataset.col;
      if (sortCol === col) {
        sortDir = sortDir === "asc" ? "desc" : "asc";
      } else {
        sortCol = col;
        sortDir = col === "team" ? "asc" : "desc";
      }
      renderStep1();
    });
  });
}

function renderWinsChart(sdata, modelKey, filteredRows) {
  const teams = [...(Array.isArray(filteredRows) ? filteredRows : filterRowsByConference(getStep1RowsForSeason(currentSeason, modelKey), step1ConferenceFilter))].sort((a,b) => (b.projected_wins||0) - (a.projected_wins||0));
  if (!teams.length) {
    document.getElementById("winsChart").innerHTML = '<div style="color:var(--text-muted);padding:20px">No teams for this conference filter.</div>';
    return;
  }
  const n = teams.length;
  const barH = 18, gap = 4, leftMargin = 50, rightMargin = 80;
  const h = n * (barH + gap) + 40;
  const chartW = 700;
  const maxWins = Math.max(82, ...teams.map(t => Math.max(t.projected_wins || 0, t.actual_wins || 0, t.win_p95 || 0)));
  const xScale = (chartW - leftMargin - rightMargin) / maxWins;

  let svg = `<svg width="100%" viewBox="0 0 ${chartW} ${h}" style="max-height:${Math.min(h, 900)}px">`;

  const x41 = leftMargin + 41 * xScale;
  svg += `<line x1="${x41}" y1="0" x2="${x41}" y2="${h}" stroke="var(--text-muted)" stroke-dasharray="4,3" opacity="0.4"/>`;
  svg += `<text x="${x41}" y="12" text-anchor="middle" font-size="10" fill="var(--text-muted)">41 Wins</text>`;

  teams.forEach((t, i) => {
    const y = 20 + i * (barH + gap);
    const projW = leftMargin + (t.projected_wins || 0) * xScale;
    const actW = t.actual_wins != null ? leftMargin + t.actual_wins * xScale : null;
    const p5x = leftMargin + (t.win_p5 || 0) * xScale;
    const p95x = leftMargin + (t.win_p95 || 0) * xScale;

    svg += `<rect x="${p5x}" y="${y+2}" width="${p95x - p5x}" height="${barH-4}" rx="2" fill="var(--accent)" opacity="0.12"/>`;
    svg += `<rect x="${leftMargin}" y="${y+4}" width="${projW - leftMargin}" height="${barH-8}" rx="2" fill="var(--accent)" opacity="0.7"/>`;
    if (actW != null) {
      svg += `<line x1="${actW}" y1="${y+1}" x2="${actW}" y2="${y+barH-1}" stroke="var(--green)" stroke-width="2.5"/>`;
    }

    svg += `<text x="${leftMargin - 4}" y="${y + barH/2 + 4}" text-anchor="end" font-size="11" font-weight="600">${t.team}</text>`;
    const errStr = t.win_error != null ? ` (${t.win_error > 0 ? "+" : ""}${Number(t.win_error).toFixed(1)})` : "";
    svg += `<text x="${Math.max(projW, actW || 0) + 6}" y="${y + barH/2 + 4}" font-size="10" fill="var(--text-muted)">${(t.projected_wins||0).toFixed(0)}P / ${t.actual_wins != null ? t.actual_wins : "?"}A${errStr}</text>`;
  });

  svg += `<rect x="${chartW - 200}" y="${h - 18}" width="12" height="6" rx="1" fill="var(--accent)" opacity="0.7"/>`;
  svg += `<text x="${chartW - 184}" y="${h - 12}" font-size="10" fill="var(--text-muted)">Projected</text>`;
  svg += `<line x1="${chartW - 120}" y1="${h - 18}" x2="${chartW - 120}" y2="${h - 12}" stroke="var(--green)" stroke-width="2.5"/>`;
  svg += `<text x="${chartW - 114}" y="${h - 12}" font-size="10" fill="var(--text-muted)">Actual</text>`;
  svg += `<rect x="${chartW - 80}" y="${h - 18}" width="20" height="6" rx="1" fill="var(--accent)" opacity="0.12"/>`;
  svg += `<text x="${chartW - 56}" y="${h - 12}" font-size="10" fill="var(--text-muted)">90% CI</text>`;

  svg += "</svg>";
  document.getElementById("winsChart").innerHTML = svg;
}

function renderCalibration() {
  const calData = DATA.validation && DATA.validation.calibration && DATA.validation.calibration[currentSeason];
  if (!calData || calData.length === 0) {
    document.getElementById("calChart").innerHTML = '<div style="color:var(--text-muted);padding:20px">No calibration data</div>';
    return;
  }

  const w = 320, h = 280, pad = 40;
  const plotW = w - 2 * pad, plotH = h - 2 * pad;
  let svg = `<svg width="100%" viewBox="0 0 ${w} ${h}">`;

  svg += `<line x1="${pad}" y1="${pad + plotH}" x2="${pad + plotW}" y2="${pad}" stroke="var(--text-muted)" stroke-dasharray="4,3" opacity="0.5"/>`;
  svg += `<line x1="${pad}" y1="${pad + plotH}" x2="${pad + plotW}" y2="${pad + plotH}" stroke="var(--border)" stroke-width="1"/>`;
  svg += `<line x1="${pad}" y1="${pad}" x2="${pad}" y2="${pad + plotH}" stroke="var(--border)" stroke-width="1"/>`;

  svg += `<text x="${w/2}" y="${h - 4}" text-anchor="middle" font-size="10" fill="var(--text-muted)">Predicted Win Prob</text>`;
  svg += `<text x="10" y="${h/2}" text-anchor="middle" font-size="10" fill="var(--text-muted)" transform="rotate(-90,10,${h/2})">Actual Win Rate</text>`;

  for (let i = 0; i <= 10; i += 2) {
    const v = i / 10;
    const x = pad + v * plotW;
    const y = pad + plotH - v * plotH;
    svg += `<text x="${x}" y="${pad + plotH + 14}" text-anchor="middle" font-size="9" fill="var(--text-muted)">${v.toFixed(1)}</text>`;
    svg += `<text x="${pad - 6}" y="${y + 3}" text-anchor="end" font-size="9" fill="var(--text-muted)">${v.toFixed(1)}</text>`;
  }

  calData.forEach(bin => {
    const px = pad + bin.predicted_rate * plotW;
    const py = pad + plotH - bin.actual_rate * plotH;
    const r = Math.max(3, Math.min(8, Math.sqrt(bin.count) * 0.8));
    svg += `<circle cx="${px}" cy="${py}" r="${r}" fill="var(--accent)" opacity="0.85"/>`;
  });

  let path = "";
  calData.forEach((bin, i) => {
    const px = pad + bin.predicted_rate * plotW;
    const py = pad + plotH - bin.actual_rate * plotH;
    path += (i === 0 ? "M" : "L") + `${px},${py}`;
  });
  svg += `<path d="${path}" fill="none" stroke="var(--accent)" stroke-width="1.5" opacity="0.6"/>`;

  svg += "</svg>";
  document.getElementById("calChart").innerHTML = svg;
}

function initSingleGameSimulator(seasons) {
  const homeSeasonEl = document.getElementById("sgHomeSeason");
  const awaySeasonEl = document.getElementById("sgAwaySeason");
  homeSeasonEl.innerHTML = "";
  awaySeasonEl.innerHTML = "";

  seasons.forEach(s => {
    const o1 = document.createElement("option");
    o1.value = s;
    o1.textContent = s;
    homeSeasonEl.appendChild(o1);

    const o2 = document.createElement("option");
    o2.value = s;
    o2.textContent = s;
    awaySeasonEl.appendChild(o2);
  });

  if (currentSeason) {
    homeSeasonEl.value = currentSeason;
    awaySeasonEl.value = currentSeason;
  }

  populateTeamOptions(homeSeasonEl.value, "sgHomeTeam");
  populateTeamOptions(awaySeasonEl.value, "sgAwayTeam");

  homeSeasonEl.addEventListener("change", () => populateTeamOptions(homeSeasonEl.value, "sgHomeTeam"));
  awaySeasonEl.addEventListener("change", () => populateTeamOptions(awaySeasonEl.value, "sgAwayTeam"));
  document.getElementById("sgRunBtn").addEventListener("click", runSingleGameSimulation);
}

function populateTeamOptions(season, teamSelectId) {
  const el = document.getElementById(teamSelectId);
  const teams = getStep1RowsForSeason(season, step1Model)
    .slice()
    .sort((a, b) => (a.team || "").localeCompare(b.team || ""));
  el.innerHTML = teams.map(t => `<option value="${t.team}">${t.team}</option>`).join("");
}

function getTeamEntry(season, teamAbbr) {
  const rows = getStep1RowsForSeason(season, step1Model);
  for (const row of rows) {
    if (row.team === teamAbbr) return row;
  }
  return null;
}

function numOrZero(v) {
  const n = Number(v);
  return Number.isFinite(n) ? n : 0;
}

function summarizeTeamStyle(step2Team) {
  const players = (step2Team && step2Team.starter_players) || [];
  const out = {
    creation: 0,
    spacing: 0,
    rimPressure: 0,
    poa: 0,
    rimDefense: 0,
    switchability: 0,
  };

  players.forEach(p => {
    const off = (p.off_archetype || "").toLowerCase();
    const def = (p.def_archetype || "").toLowerCase();

    if (/(ball dominant|ballhandler|all-around|hub|playmaking|creator)/.test(off)) out.creation += 1;
    if (/(shooter|perimeter|popping|gravity)/.test(off)) out.spacing += 1;
    if (/(interior|finisher|rolling|rim)/.test(off)) out.rimPressure += 1;

    if (/(poa|wing stopper|off-ball chaser)/.test(def)) out.poa += 1;
    if (/(rim protector|dropping big|mobile big)/.test(def)) out.rimDefense += 1;
    if (/(versatile|mobile big|switch)/.test(def)) out.switchability += 1;
  });
  return out;
}

function computeTopContributors(step2Team, limit = 3) {
  if (!step2Team) return [];
  const store = new Map();

  const addPlayers = (players, wImpact, wScore) => {
    (players || []).forEach(p => {
      const name = cleanNameLabel(p.player_name) || `ID ${cleanText(p.player_id) || "?"}`;
      const impact = numOrZero(p.impact);
      const score = numOrZero(p.score);
      const contrib = (wImpact * impact) + (wScore * score);

      const cur = store.get(name) || { name, value: 0, off: cleanText(p.off_archetype), def: cleanText(p.def_archetype) };
      cur.value += contrib;
      if (!cur.off && cleanText(p.off_archetype)) cur.off = cleanText(p.off_archetype);
      if (!cur.def && cleanText(p.def_archetype)) cur.def = cleanText(p.def_archetype);
      store.set(name, cur);
    });
  };

  addPlayers(step2Team.starter_players, 0.70, 0.30);
  addPlayers(step2Team.clutch_players, 0.60, 0.40);

  return Array.from(store.values())
    .sort((a, b) => b.value - a.value)
    .slice(0, limit);
}

function buildMatchupContext(homeSeason, homeTeam, awaySeason, awayTeam, home, away, hca) {
  const homeStep2 = findStep2Team(homeSeason, homeTeam);
  const awayStep2 = findStep2Team(awaySeason, awayTeam);

  const starterEdge = homeStep2 && awayStep2 ? (numOrZero(homeStep2.mu_start) - numOrZero(awayStep2.mu_start)) : null;
  const rotationEdge = homeStep2 && awayStep2 ? (numOrZero(homeStep2.mu_rotation) - numOrZero(awayStep2.mu_rotation)) : null;
  const clutchEdge = homeStep2 && awayStep2 ? (numOrZero(homeStep2.mu_clutch) - numOrZero(awayStep2.mu_clutch)) : null;

  let phaseBlendEdge = null;
  if (starterEdge != null && rotationEdge != null && clutchEdge != null) {
    phaseBlendEdge = 0.45 * starterEdge + 0.35 * rotationEdge + 0.20 * clutchEdge;
  }

  const homeStyle = summarizeTeamStyle(homeStep2);
  const awayStyle = summarizeTeamStyle(awayStep2);
  const headToHead = [
    { label: "On-Ball Creation", home: homeStyle.creation, away: awayStyle.creation },
    { label: "Spacing Gravity", home: homeStyle.spacing, away: awayStyle.spacing },
    { label: "Rim Pressure", home: homeStyle.rimPressure, away: awayStyle.rimPressure },
    { label: "POA Defense", home: homeStyle.poa, away: awayStyle.poa },
    { label: "Rim Protection", home: homeStyle.rimDefense, away: awayStyle.rimDefense },
    { label: "Switchability", home: homeStyle.switchability, away: awayStyle.switchability },
  ];

  return {
    homeStep2,
    awayStep2,
    phaseEdges: {
      starter: starterEdge,
      rotation: rotationEdge,
      clutch: clutchEdge,
      blend: phaseBlendEdge,
    },
    drivers: {
      talentEdge: numOrZero(home.mu) - numOrZero(away.mu),
      homeCourtEdge: numOrZero(hca),
      volatilityGap: numOrZero(home.sigma) - numOrZero(away.sigma),
    },
    headToHead,
    contributors: {
      home: computeTopContributors(homeStep2, 3),
      away: computeTopContributors(awayStep2, 3),
    },
  };
}

function renderHeadToHeadBars(homeTeam, awayTeam, rows) {
  if (!rows || !rows.length) {
    return `<div style="font-size:12px;color:var(--text-muted)">No Step 2 lineup data available for this matchup.</div>`;
  }

  return rows.map(r => {
    const h = numOrZero(r.home);
    const a = numOrZero(r.away);
    const total = Math.max(1, h + a);
    const homePct = Math.max(5, (h / total) * 100);
    const awayPct = Math.max(5, (a / total) * 100);
    return `
      <div class="sg-edge">
        <div class="sg-edge-top">
          <span class="label">${r.label}</span>
          <span>${homeTeam} ${h} - ${a} ${awayTeam}</span>
        </div>
        <div class="sg-edge-track">
          <div class="sg-edge-home" style="width:${homePct}%"></div>
          <div class="sg-edge-away" style="width:${awayPct}%"></div>
        </div>
      </div>
    `;
  }).join("");
}

function renderContributors(teamName, sideClass, items) {
  if (!items || !items.length) {
    return `<div class="sg-contrib-col"><div class="title ${sideClass}">${teamName}</div><div style="font-size:12px;color:var(--text-muted)">No Step 2 contributors available.</div></div>`;
  }
  return `
    <div class="sg-contrib-col">
      <div class="title ${sideClass}">${teamName}</div>
      ${items.map(p => `
        <div class="sg-contrib-item">
          <span class="name" title="OFF: ${cleanText(p.off) || "-"} | DEF: ${cleanText(p.def) || "-"}">${p.name}</span>
          <span class="val">${numOrZero(p.value).toFixed(3)}</span>
        </div>
      `).join("")}
    </div>
  `;
}

function erf(x) {
  const sign = x >= 0 ? 1 : -1;
  const ax = Math.abs(x);
  const p = 0.3275911;
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const t = 1.0 / (1.0 + p * ax);
  const y = 1.0 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-ax * ax));
  return sign * y;
}

function normalCdf(x) {
  return 0.5 * (1 + erf(x / Math.sqrt(2)));
}

function randn() {
  let u = 0;
  let v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function runSingleGameSimulation() {
  const homeSeason = document.getElementById("sgHomeSeason").value;
  const awaySeason = document.getElementById("sgAwaySeason").value;
  const homeTeam = document.getElementById("sgHomeTeam").value;
  const awayTeam = document.getElementById("sgAwayTeam").value;

  const home = getTeamEntry(homeSeason, homeTeam);
  const away = getTeamEntry(awaySeason, awayTeam);

  if (!home || !away) {
    document.getElementById("sgResult").innerHTML = "Unable to find one or both teams in the selected seasons.";
    return;
  }

  const cfg = DATA.config || {};
  const sigmaLeague = Number(cfg.sigma_league || 3.0);
  const hca = Number(cfg.home_court_advantage || 2.0);
  const homeMu = Number(home.mu || 0);
  const awayMu = Number(away.mu || 0);
  const homeSigma = Number(home.sigma || 0);
  const awaySigma = Number(away.sigma || 0);

  const deltaMu = (homeMu + hca) - awayMu;
  const sigmaGame = Math.sqrt(homeSigma * homeSigma + awaySigma * awaySigma + sigmaLeague * sigmaLeague);
  const z = sigmaGame > 0 ? deltaMu / sigmaGame : 0;
  const homeWinProb = normalCdf(z);
  const sampledMargin = deltaMu + sigmaGame * randn();
  const context = buildMatchupContext(homeSeason, homeTeam, awaySeason, awayTeam, home, away, hca);

  currentSimulatedGame = {
    homeSeason,
    awaySeason,
    homeTeam,
    awayTeam,
    deltaMu,
    sigmaGame,
    homeWinProb,
    sampledMargin,
    winner: sampledMargin >= 0 ? `${homeTeam} (${homeSeason})` : `${awayTeam} (${awaySeason})`,
    timestamp: new Date().toLocaleString(),
    context,
  };

  renderEnhancedSingleGameResult();
}

function renderSingleGameResult() {
  const el = document.getElementById("sgResult");
  if (!currentSimulatedGame) {
    el.innerHTML = "No game simulated yet.";
    return;
  }
  const g = currentSimulatedGame;
  const ctx = g.context || {};
  const drivers = ctx.drivers || {};
  const phase = (ctx.phaseEdges || {});

  const h2hHtml = renderHeadToHeadBars(g.homeTeam, g.awayTeam, ctx.headToHead || []);
  const contribHome = renderContributors(`${g.homeTeam} Impact Drivers`, "home", (ctx.contributors || {}).home || []);
  const contribAway = renderContributors(`${g.awayTeam} Impact Drivers`, "away", (ctx.contributors || {}).away || []);

  const sampledWinnerClass = g.sampledMargin >= 0 ? "sg-badge-home" : "sg-badge-away";

  el.innerHTML = `
    <div class="sg-header">
      <div class="sg-matchup">
        <span class="sg-badge-home">${g.homeTeam} (${g.homeSeason})</span> vs
        <span class="sg-badge-away">${g.awayTeam} (${g.awaySeason})</span>
      </div>
      <div class="sg-time">Simulated at ${g.timestamp}</div>
    </div>

    <div class="sg-kpi-grid">
      <div class="sg-kpi"><div class="k">Home Win Probability</div><div class="v">${(g.homeWinProb * 100).toFixed(1)}%</div></div>
      <div class="sg-kpi"><div class="k">Expected Margin (Home)</div><div class="v">${g.deltaMu.toFixed(2)}</div></div>
      <div class="sg-kpi"><div class="k">Game Volatility (Sigma)</div><div class="v">${g.sigmaGame.toFixed(2)}</div></div>
      <div class="sg-kpi"><div class="k">Single-Game Draw Winner</div><div class="v ${sampledWinnerClass}">${g.winner}</div></div>
    </div>

    <div class="sg-detail-grid">
      <div class="sg-card">
        <h4>Margin Drivers</h4>
        <div class="sg-list">
          <div class="row"><span class="k">Team Strength Edge (mu)</span><span class="v">${numOrZero(drivers.talentEdge).toFixed(2)}</span></div>
          <div class="row"><span class="k">Home Court Edge</span><span class="v">+${numOrZero(drivers.homeCourtEdge).toFixed(2)}</span></div>
          <div class="row"><span class="k">Phase Blend Edge (Step 2)</span><span class="v">${phase.blend == null ? "-" : numOrZero(phase.blend).toFixed(2)}</span></div>
          <div class="row"><span class="k">Starter / Rotation / Clutch</span><span class="v">${phase.starter == null ? "-" : numOrZero(phase.starter).toFixed(2)} / ${phase.rotation == null ? "-" : numOrZero(phase.rotation).toFixed(2)} / ${phase.clutch == null ? "-" : numOrZero(phase.clutch).toFixed(2)}</span></div>
          <div class="row"><span class="k">Volatility Gap (Home-Away Sigma)</span><span class="v">${numOrZero(drivers.volatilityGap).toFixed(2)}</span></div>
          <div class="row"><span class="k">Single Draw Margin (Home)</span><span class="v">${g.sampledMargin.toFixed(2)}</span></div>
        </div>
      </div>

      <div class="sg-card">
        <h4>Head-to-Head Profile</h4>
        ${h2hHtml}
      </div>

      <div class="sg-card">
        <h4>Top Player Contributors</h4>
        <div class="sg-contrib-grid">
          ${contribHome}
          ${contribAway}
        </div>
      </div>
    </div>
  `;
}

function initStep2View() {
  const seasons = getStep2Seasons();
  const tabs = document.getElementById("lineupSeasonTabs");
  tabs.innerHTML = "";

  if (!seasons.length) {
    document.getElementById("lineupStatsRow").innerHTML = `
      <div class="stat-card"><div class="label">Step 2</div><div class="value val-orange">No Data</div><div class="detail">Run: python3 src/simulation/lineup_projection.py</div></div>
    `;
    document.getElementById("lineupCardGrid").innerHTML = "";
    return;
  }

  lineupSeason = seasons[seasons.length - 1];
  seasons.forEach(s => {
    const tab = document.createElement("div");
    tab.className = "season-tab" + (s === lineupSeason ? " active" : "");
    tab.textContent = s;
    tab.onclick = () => switchLineupSeason(s);
    tabs.appendChild(tab);
  });

  initStep2ConferenceTabs();
  renderStep2();
}

function switchLineupSeason(season) {
  lineupSeason = season;
  document.querySelectorAll("#lineupSeasonTabs .season-tab").forEach(t => {
    t.classList.toggle("active", t.textContent === season);
  });
  initStep2ConferenceTabs();
  renderStep2();
}

function initStep2ConferenceTabs() {
  const el = document.getElementById("step2ConferenceTabs");
  if (!el) return;
  const options = [
    { key: "all", label: "All Teams" },
    { key: "east", label: "East" },
    { key: "west", label: "West" },
  ];
  el.innerHTML = options.map(opt => `
    <button class="filter-tab${opt.key === step2ConferenceFilter ? " active" : ""}" data-filter="${opt.key}">${opt.label}</button>
  `).join("");
  el.querySelectorAll(".filter-tab").forEach(btn => {
    btn.addEventListener("click", () => {
      step2ConferenceFilter = btn.dataset.filter || "all";
      initStep2ConferenceTabs();
      renderStep2();
    });
  });
}

function renderStep2() {
  const seasons = (STEP2 && STEP2.seasons) || {};
  const seasonData = seasons[lineupSeason];
  if (!seasonData) return;
  renderStep2Stats(seasonData);
  renderStep2Cards(seasonData);
}

function renderStep2Stats(seasonData) {
  const s = seasonData.summary || {};
  const cards = [
    {
      label: "Starter Overlap",
      value: s.starter_overlap_count_mean != null ? Number(s.starter_overlap_count_mean).toFixed(2) : "-",
      cls: (s.starter_target_met ? "val-green" : "val-orange"),
      detail: "Predicted vs observed first-quarter starters from PBP"
    },
    {
      label: "Clutch Overlap",
      value: s.clutch_overlap_count_mean != null ? Number(s.clutch_overlap_count_mean).toFixed(2) : "-",
      cls: (s.clutch_target_met ? "val-green" : "val-orange"),
      detail: "Predicted vs top-5 clutch-minute players"
    },
    {
      label: "Rotation Corr",
      value: s.rotation_corr != null ? Number(s.rotation_corr).toFixed(3) : "-",
      cls: (s.rotation_target_met ? "val-green" : "val-purple"),
      detail: "Predicted rotation vs actual bench-player impact from game logs"
    },
    {
      label: "Teams",
      value: s.n_teams != null ? String(s.n_teams) : "-",
      cls: "val-accent",
      detail: lineupSeason
    },
  ];

  document.getElementById("lineupStatsRow").innerHTML = cards.map(c => `
    <div class="stat-card">
      <div class="label">${c.label}</div>
      <div class="value ${c.cls}">${c.value}</div>
      <div class="detail">${c.detail}</div>
    </div>
  `).join("");
}

function renderStep2Cards(seasonData) {
  let teams = (seasonData.teams || []).slice().sort((a, b) => (a.team_abbreviation || "").localeCompare(b.team_abbreviation || ""));
  teams = filterRowsByConference(teams, step2ConferenceFilter);
  const grid = document.getElementById("lineupCardGrid");

  if (!teams.length) {
    grid.innerHTML = `<div class="stat-card"><div class="label">Lineup Cards</div><div class="value val-orange">No Teams</div><div class="detail">No Step 2 rows for ${lineupSeason}</div></div>`;
    return;
  }

  grid.innerHTML = teams.map(team => {
    const v = team.validation || {};
    return `
      <div class="lineup-card" data-team="${team.team_abbreviation}">
        <div class="title-row">
          <div class="team-name">${team.team_abbreviation}</div>
          <div class="conf">${team.conference || "Unknown"}</div>
        </div>
        <div class="mini">
          <div class="k">Starter</div><div class="v">${fmt(team.mu_start, 2)} / ${fmt(team.sigma_start, 2)}</div>
          <div class="k">Rotation</div><div class="v">${fmt(team.mu_rotation, 2)} / ${fmt(team.sigma_rotation, 2)}</div>
          <div class="k">Clutch</div><div class="v">${fmt(team.mu_clutch, 2)} / ${fmt(team.sigma_clutch, 2)}</div>
          <div class="k">Pool Size</div><div class="v">${team.n_players_pool || "-"}</div>
        </div>
        <div class="validation">
          <span>S: ${v.starter_overlap != null ? v.starter_overlap : "-"}/5</span>
          <span>C: ${v.clutch_overlap != null ? v.clutch_overlap : "-"}/5</span>
          <span>R: ${v.rotation_actual_net != null ? fmt(v.rotation_actual_net, 2) : "-"}</span>
        </div>
      </div>
    `;
  }).join("");

  document.querySelectorAll("#lineupCardGrid .lineup-card").forEach(card => {
    card.addEventListener("click", () => {
      const team = card.dataset.team;
      openLineupModal(lineupSeason, team);
    });
  });
}

function findStep2Team(season, teamAbbr) {
  const seasonData = (((STEP2 || {}).seasons || {})[season] || {});
  const teams = seasonData.teams || [];
  return teams.find(t => t.team_abbreviation === teamAbbr) || null;
}

function isIdPlaceholder(name) {
  const text = cleanText(name);
  if (!text) return false;
  return /^id\s+\d+$/i.test(text);
}

function getSeasonPlayerNameMap(season) {
  if (STEP2_SEASON_PLAYER_MAP[season]) {
    return STEP2_SEASON_PLAYER_MAP[season];
  }

  const seasonData = (((STEP2 || {}).seasons || {})[season] || {});
  const teams = seasonData.teams || [];
  const map = {};

  const addPlayerRows = (rows) => {
    (rows || []).forEach((p) => {
      const id = cleanText(p && p.player_id);
      const name = cleanNameLabel(p && p.player_name);
      if (!id || !name || isIdPlaceholder(name)) return;
      if (!map[id]) map[id] = name;
    });
  };

  teams.forEach((team) => {
    addPlayerRows(team.pool_players);
    addPlayerRows(team.starter_players);
    addPlayerRows(team.clutch_players);

    const v = team.validation || {};
    const starterIds = v.actual_starter_ids || [];
    const starterNames = v.actual_starter_names || [];
    starterIds.forEach((idRaw, idx) => {
      const id = cleanText(idRaw);
      const name = cleanNameLabel(starterNames[idx]);
      if (!id || !name || isIdPlaceholder(name)) return;
      if (!map[id]) map[id] = name;
    });

    const clutchIds = v.actual_clutch_ids || [];
    const clutchNames = v.actual_clutch_names || [];
    clutchIds.forEach((idRaw, idx) => {
      const id = cleanText(idRaw);
      const name = cleanNameLabel(clutchNames[idx]);
      if (!id || !name || isIdPlaceholder(name)) return;
      if (!map[id]) map[id] = name;
    });
  });

  STEP2_SEASON_PLAYER_MAP[season] = map;
  return map;
}

function resolveObservedNames(season, ids, names) {
  const idList = (ids || []).map(v => cleanText(v));
  const nameList = names || [];
  const idToName = getSeasonPlayerNameMap(season);

  const seen = new Set();
  const out = [];
  const n = Math.max(idList.length, nameList.length);
  for (let i = 0; i < n; i += 1) {
    const id = idList[i];
    let name = cleanNameLabel(nameList[i]);

    if (!name || isIdPlaceholder(name)) {
      if (id && idToName[id]) {
        name = idToName[id];
      }
    }

    if (!name || isIdPlaceholder(name) || seen.has(name)) {
      continue;
    }
    seen.add(name);
    out.push(name);
  }
  return out;
}

function getForecastSeasonPlayerNameMap(season) {
  const cacheKey = `${forecastScenario || "default"}::${season}`;
  if (FORECAST_STEP2_SEASON_PLAYER_MAP[cacheKey]) {
    return FORECAST_STEP2_SEASON_PLAYER_MAP[cacheKey];
  }

  const lineupData = getForecastLineupData();
  const seasonData = ((lineupData && lineupData.seasons) || {})[season] || {};
  const teams = seasonData.teams || [];
  const map = {};

  const addPlayerRows = (rows) => {
    (rows || []).forEach((p) => {
      const id = cleanText(p && p.player_id);
      const name = cleanNameLabel(p && p.player_name);
      if (!id || !name || isIdPlaceholder(name)) return;
      if (!map[id]) map[id] = name;
    });
  };

  teams.forEach((team) => {
    addPlayerRows(team.pool_players);
    addPlayerRows(team.starter_players);
    addPlayerRows(team.clutch_players);

    const v = team.validation || {};
    const starterIds = v.actual_starter_ids || [];
    const starterNames = v.actual_starter_names || [];
    starterIds.forEach((idRaw, idx) => {
      const id = cleanText(idRaw);
      const name = cleanNameLabel(starterNames[idx]);
      if (!id || !name || isIdPlaceholder(name)) return;
      if (!map[id]) map[id] = name;
    });

    const clutchIds = v.actual_clutch_ids || [];
    const clutchNames = v.actual_clutch_names || [];
    clutchIds.forEach((idRaw, idx) => {
      const id = cleanText(idRaw);
      const name = cleanNameLabel(clutchNames[idx]);
      if (!id || !name || isIdPlaceholder(name)) return;
      if (!map[id]) map[id] = name;
    });
  });

  FORECAST_STEP2_SEASON_PLAYER_MAP[cacheKey] = map;
  return map;
}

function resolveForecastObservedNames(season, ids, names) {
  const idList = (ids || []).map(v => cleanText(v));
  const nameList = names || [];
  const idToName = getForecastSeasonPlayerNameMap(season);

  const seen = new Set();
  const out = [];
  const n = Math.max(idList.length, nameList.length);
  for (let i = 0; i < n; i += 1) {
    const id = idList[i];
    let name = cleanNameLabel(nameList[i]);

    if (!name || isIdPlaceholder(name)) {
      if (id && idToName[id]) {
        name = idToName[id];
      }
    }

    if (!name || isIdPlaceholder(name) || seen.has(name)) {
      continue;
    }
    seen.add(name);
    out.push(name);
  }
  return out;
}

function renderPlayerList(players) {
  const cleanPlayers = (players || []).filter(p => !!cleanNameLabel(p.player_name));
  if (!cleanPlayers.length) {
    return `<div style="color:var(--text-muted);font-size:12px">No players</div>`;
  }
  return `<div class="player-list">` + cleanPlayers.map(p => `
    <div class="player-item">
      <div class="meta">
        <div class="name">${cleanNameLabel(p.player_name)}</div>
        <div class="role">${cleanText(p.role) || "-"}${cleanText(p.position_band) ? ` · ${cleanText(p.position_band)}` : ""}</div>
        <div class="arche">OFF: ${cleanText(p.off_archetype) || "-"} | DEF: ${cleanText(p.def_archetype) || "-"}</div>
      </div>
      <div class="num">MIN ${fmt(p.minutes, 1)}</div>
      <div class="num">S ${fmt(p.score, 3)}</div>
    </div>
  `).join("") + `</div>`;
}

function openLineupModal(season, teamAbbr) {
  const team = findStep2Team(season, teamAbbr);
  if (!team) return;

  const v = team.validation || {};
  const actualStarterNames = resolveObservedNames(
    season,
    v.actual_starter_ids || [],
    v.actual_starter_names || []
  );
  const actualClutchNames = resolveObservedNames(
    season,
    v.actual_clutch_ids || [],
    v.actual_clutch_names || []
  );

  document.getElementById("lineupModalTitle").textContent = `${team.team_abbreviation} (${season})`;

  const html = `
    <div class="lineup-modal-grid">
      <div class="lineup-panel">
        <h4>Phase Metrics</h4>
        <div class="kv">
          <div class="k">Starter Mu / Sigma</div><div class="v">${fmt(team.mu_start,2)} / ${fmt(team.sigma_start,2)}</div>
          <div class="k">Rotation Mu / Sigma</div><div class="v">${fmt(team.mu_rotation,2)} / ${fmt(team.sigma_rotation,2)}</div>
          <div class="k">Clutch Mu / Sigma</div><div class="v">${fmt(team.mu_clutch,2)} / ${fmt(team.sigma_clutch,2)}</div>
          <div class="k">Team Clutch Minutes</div><div class="v">${fmt(team.team_clutch_minutes_total,1)}</div>
          <div class="k">Pool Size</div><div class="v">${team.n_players_pool || "-"}</div>
        </div>
      </div>

      <div class="lineup-panel">
        <h4>Validation</h4>
        <div class="kv">
          <div class="k">Starter Overlap</div><div class="v">${v.starter_overlap != null ? v.starter_overlap + "/5" : "-"}</div>
          <div class="k">Clutch Overlap</div><div class="v">${v.clutch_overlap != null ? v.clutch_overlap + "/5" : "-"}</div>
          <div class="k">Rotation Predicted</div><div class="v">${fmt(v.rotation_predicted,2)}</div>
          <div class="k">Rotation Actual (bench-player impact)</div><div class="v">${fmt(v.rotation_actual_net,2)}</div>
          <div class="k">Observed Starter Players</div><div class="v">${actualStarterNames.length || 0}</div>
          <div class="k">Observed Clutch Players</div><div class="v">${actualClutchNames.length || 0}</div>
        </div>
      </div>

      <div class="lineup-panel">
        <h4>Predicted Starters</h4>
        ${renderPlayerList(team.starter_players || [])}
      </div>

      <div class="lineup-panel">
        <h4>Predicted Clutch Lineup</h4>
        ${renderPlayerList(team.clutch_players || [])}
      </div>

      <div class="lineup-panel">
        <h4>Observed Starter Players (Proxy)</h4>
        <div style="font-size:12px;color:var(--text-muted)">${actualStarterNames.length ? actualStarterNames.join(", ") : "No observed starter proxy"}</div>
      </div>

      <div class="lineup-panel">
        <h4>Observed Clutch Players</h4>
        <div style="font-size:12px;color:var(--text-muted)">${actualClutchNames.length ? actualClutchNames.join(", ") : "No observed clutch lineup"}</div>
      </div>
    </div>
  `;

  document.getElementById("lineupModalContent").innerHTML = html;
  document.getElementById("lineupModalOverlay").classList.add("open");
}

function bindModalEvents() {
  const overlay = document.getElementById("lineupModalOverlay");
  const closeBtn = document.getElementById("lineupModalClose");
  closeBtn.addEventListener("click", () => overlay.classList.remove("open"));
  overlay.addEventListener("click", (ev) => {
    if (ev.target === overlay) {
      overlay.classList.remove("open");
    }
  });
}

function fmt(v, dec) {
  if (v == null || v === undefined || Number.isNaN(Number(v))) return "-";
  return Number(v).toFixed(dec);
}

function cleanText(value) {
  if (value == null || value === undefined) return null;
  const text = String(value).trim();
  if (!text) return null;
  const low = text.toLowerCase();
  if (["nan", "none", "null", "unknown", "n/a", "na"].includes(low)) return null;
  return text;
}

function cleanNameLabel(value) {
  return cleanText(value);
}

function uniqueNamedList(values) {
  const seen = new Set();
  const out = [];
  (values || []).forEach(v => {
    const name = cleanNameLabel(v);
    if (!name || isIdPlaceholder(name) || seen.has(name)) return;
    seen.add(name);
    out.push(name);
  });
  return out;
}

// ═══════════════════════════════════════════════════════════════
// Forecast View
// ═══════════════════════════════════════════════════════════════

function initForecastView() {
  initForecastScenarioTabs();
  const hasSeasons = _renderForecastSeasonTabs("forecastSeasonTabs");
  const modelTabsEl = document.getElementById("forecastModelTabs");
  if (modelTabsEl) modelTabsEl.innerHTML = "";
  if (!hasSeasons) {
    return;
  }
  syncForecastModel();
  initForecastModelTabs();
  initForecastConferenceTabs();
}

function initForecastStep2View() {
  initForecastScenarioTabsStep2();
  _renderForecastSeasonTabs("forecastSeasonTabsStep2");
  initForecastLineupConferenceTabs();
  renderForecastLineup();
}

function switchForecastSeason(s) {
  setForecastSeason(s);
}

function initForecastConferenceTabs() {
  const el = document.getElementById("forecastConferenceTabs");
  if (!el) return;
  const options = [
    { key: "all", label: "All Teams" },
    { key: "east", label: "East" },
    { key: "west", label: "West" },
  ];
  el.innerHTML = options.map(opt => `
    <button class="filter-tab${opt.key === forecastConferenceFilter ? " active" : ""}" data-filter="${opt.key}">${opt.label}</button>
  `).join("");
  el.querySelectorAll(".filter-tab").forEach(btn => {
    btn.addEventListener("click", () => {
      forecastConferenceFilter = btn.dataset.filter || "all";
      initForecastConferenceTabs();
      renderForecast();
    });
  });
}

function renderForecast() {
  const payload = getForecastScenarioPayload(forecastScenario);
  if (!forecastSeason || !payload.seasons) return;
  const sdata = payload.seasons[forecastSeason];
  if (!sdata) return;
  syncForecastModel();
  const rows = getForecastRowsForSeason(forecastSeason, forecastModel);
  const filteredRows = filterRowsByConference(rows, forecastConferenceFilter);
  renderForecastStats(sdata, forecastModel, filteredRows);
  renderForecastTable(sdata, forecastModel, filteredRows);
  renderForecastWinsChart(sdata, forecastModel, filteredRows);
  renderForecastValidation();
  renderForecastLineup();
}

function renderForecastStats(sdata, modelKey, filteredRows) {
  const payload = getForecastScenarioPayload(forecastScenario);
  const teams = Array.isArray(filteredRows) ? filteredRows : getForecastRowsForSeason(forecastSeason, modelKey);
  const seasonStats = getForecastStatsForSeason(forecastSeason, modelKey) || {};
  const wins = teams.map(t => t.projected_wins || 0);
  const avgWins = wins.length ? (wins.reduce((a,b) => a+b, 0) / wins.length) : 0;
  const maxW = wins.length ? Math.max(...wins) : 0;
  const minW = wins.length ? Math.min(...wins) : 0;
  const spread = maxW - minW;

  // Get validation data
  const valData = payload.validation || {};
  const projections = valData.projections || {};

  // Find the transition that targets this season
  let bkeCorr = null, bkeMae = null, mpgCorr = null;
  for (const [key, val] of Object.entries(projections)) {
    if (key.includes(forecastSeason)) {
      bkeCorr = val.bke_correlation;
      bkeMae = val.bke_mae;
      mpgCorr = val.mpg_correlation;
      break;
    }
  }

  const fcfg = payload.config || {};
  const scenarioLabel = getForecastScenarioLabel(forecastScenario);

  const cards = [
    { label: "Scenario", value: scenarioLabel, cls: "val-purple", detail: forecastScenario || "default" },
    { label: "Model", value: modelLabel(modelKey), cls: "val-purple", detail: "Projection view" },
    { label: "Win Corr", value: fmt(seasonStats.correlation, 3), cls: "val-green", detail: "Proj vs Actual wins" },
    { label: "Win MAE", value: fmt(seasonStats.mae, 2), cls: "val-accent", detail: "Average win error" },
    { label: "Win RMSE", value: fmt(seasonStats.rmse, 2), cls: "val-accent", detail: "Root mean sq error" },
    { label: "Teams", value: teams.length, cls: "val-accent", detail: `${forecastSeason} · ${conferenceFilterLabel(forecastConferenceFilter)}` },
    { label: "Avg Wins", value: avgWins.toFixed(1), cls: "val-accent", detail: "Mean projected" },
    { label: "Win Spread", value: spread.toFixed(1), cls: "val-orange", detail: `${minW.toFixed(0)}-${maxW.toFixed(0)}` },
    { label: "BKE r", value: bkeCorr != null ? bkeCorr.toFixed(3) : "-", cls: "val-green", detail: "Yr-to-yr carry" },
    { label: "BKE MAE", value: bkeMae != null ? bkeMae.toFixed(3) : "-", cls: "val-accent", detail: "Impact predict err" },
    { label: "MPG r", value: mpgCorr != null ? mpgCorr.toFixed(3) : "-", cls: "val-green", detail: "Minutes carry" },
    { label: "Mode", value: (fcfg.mode || "FORECAST").toUpperCase(), cls: "val-purple", detail: "Pipeline mode" },
    { label: "Sims", value: fcfg.n_simulations ? fcfg.n_simulations.toLocaleString() : "-", cls: "val-accent", detail: "Monte Carlo runs" },
  ];

  const el = document.getElementById("forecastStatsRow");
  el.innerHTML = cards.map(c => `
    <div class="stat-card">
      <div class="label">${c.label}</div>
      <div class="value ${c.cls}">${c.value || "-"}</div>
      <div class="detail">${c.detail}</div>
    </div>
  `).join("");
}

function renderForecastTable(sdata, modelKey, filteredRows) {
  let teams = [...(Array.isArray(filteredRows) ? filteredRows : filterRowsByConference(getForecastRowsForSeason(forecastSeason, modelKey), forecastConferenceFilter))];
  teams.sort((a, b) => {
    let va = a[forecastSortCol], vb = b[forecastSortCol];
    if (va == null) va = 0;
    if (vb == null) vb = 0;
    return forecastSortDir === "asc" ? (va > vb ? 1 : va < vb ? -1 : 0) : (va < vb ? 1 : va > vb ? -1 : 0);
  });

  const cols = [
    { key: "projected_rank", label: "Rank", cls: "rank" },
    { key: "team", label: "Team", cls: "team" },
    { key: "conference", label: "Conf", cls: "num" },
    { key: "projected_conf_rank", label: "Conf Rk", cls: "num", fmt: v => v != null ? Number(v).toFixed(1) : "-" },
    { key: "mu", label: modelKey === "ppp" ? "Net PPP" : "Net Rtg", cls: "num", fmt: v => v != null ? Number(v).toFixed(2) : "-" },
    { key: "sigma", label: "Vol", cls: "num", fmt: v => v != null ? Number(v).toFixed(2) : "-" },
    { key: "predicted_pace", label: "Pace", cls: "num", fmt: v => v != null ? Number(v).toFixed(2) : "-" },
    { key: "ppp_offense", label: "Off PPP", cls: "num", fmt: v => v != null ? Number(v).toFixed(3) : "-" },
    { key: "ppp_defense", label: "Def PPP", cls: "num", fmt: v => v != null ? Number(v).toFixed(3) : "-" },
    { key: "projected_wins", label: "Proj W", cls: "num val-accent", fmt: v => v != null ? Number(v).toFixed(1) : "-" },
    { key: "win_p5", label: "90% CI", cls: "conf-band", fmt: (v, r) => `${r.win_p5}-${r.win_p95}` },
    { key: "actual_wins", label: "Actual W", cls: "num", fmt: v => v != null ? v : "-" },
    { key: "actual_losses", label: "Actual L", cls: "num", fmt: v => v != null ? v : "-" },
    { key: "win_error", label: "Error", cls: "num", fmt: (v) => {
      if (v == null) return "-";
      const abs = Math.abs(v);
      const sign = v > 0 ? "+" : "";
      const cls = abs <= 3 ? "err-good" : abs <= 7 ? "err-ok" : "err-bad";
      return `<span class="${cls}">${sign}${Number(v).toFixed(1)}</span>`;
    } },
    { key: "direct_playoff_probability", label: "Top 6 %", cls: "num", fmt: v => v != null ? (v * 100).toFixed(0) + "%" : "-" },
    { key: "top_10_probability", label: "Top 10 %", cls: "num", fmt: v => v != null ? (v * 100).toFixed(0) + "%" : "-" },
    { key: "playin_only_probability", label: "7-10 %", cls: "num", fmt: v => v != null ? (v * 100).toFixed(0) + "%" : "-" },
    { key: "win_std", label: "Win SD", cls: "num", fmt: v => v != null ? Number(v).toFixed(2) : "-" },
  ];

  let html = "<table><thead><tr>";
  cols.forEach(c => {
    const cls = c.key === forecastSortCol ? (forecastSortDir === "asc" ? "sorted-asc" : "sorted-desc") : "";
    html += `<th class="${cls}" data-col="${c.key}">${c.label}</th>`;
  });
  html += "</tr></thead><tbody>";

  teams.forEach(t => {
    html += "<tr>";
    cols.forEach(c => {
      const raw = c.key === "win_p5" && c.label === "90% CI" ? t["win_p5"] : t[c.key];
      const display = c.fmt ? c.fmt(raw, t) : (raw != null ? raw : "-");
      html += `<td class="${c.cls || ""}">${display}</td>`;
    });
    html += "</tr>";
  });

  html += "</tbody></table>";
  document.getElementById("forecastTableWrap").innerHTML = html;

  document.querySelectorAll("#forecastTableWrap th").forEach(th => {
    th.addEventListener("click", () => {
      const col = th.dataset.col;
      if (forecastSortCol === col) {
        forecastSortDir = forecastSortDir === "asc" ? "desc" : "asc";
      } else {
        forecastSortCol = col;
        forecastSortDir = col === "team" ? "asc" : "desc";
      }
      renderForecast();
    });
  });
}

function renderForecastWinsChart(sdata, modelKey, filteredRows) {
  const teams = [...(Array.isArray(filteredRows) ? filteredRows : filterRowsByConference(getForecastRowsForSeason(forecastSeason, modelKey), forecastConferenceFilter))].sort((a, b) => (b.projected_wins || 0) - (a.projected_wins || 0));
  if (!teams.length) {
    document.getElementById("forecastWinsChart").innerHTML = '<div style="color:var(--text-muted);padding:20px">No teams for this conference filter.</div>';
    return;
  }
  const n = teams.length;
  const barH = 18, gap = 4, leftMargin = 50, rightMargin = 80;
  const h = n * (barH + gap) + 40;
  const chartW = 700;
  const maxWins = Math.max(82, ...teams.map(t => Math.max(t.projected_wins || 0, t.win_p95 || 0, t.actual_wins || 0)));
  const xScale = (chartW - leftMargin - rightMargin) / maxWins;

  let svg = `<svg width="100%" viewBox="0 0 ${chartW} ${h}" style="max-height:${Math.min(h, 900)}px">`;

  const x41 = leftMargin + 41 * xScale;
  svg += `<line x1="${x41}" y1="0" x2="${x41}" y2="${h}" stroke="var(--text-muted)" stroke-dasharray="4,3" opacity="0.4"/>`;
  svg += `<text x="${x41}" y="12" text-anchor="middle" font-size="10" fill="var(--text-muted)">41 Wins</text>`;

  teams.forEach((t, i) => {
    const y = 20 + i * (barH + gap);
    const projW = leftMargin + (t.projected_wins || 0) * xScale;
    const p5x = leftMargin + (t.win_p5 || 0) * xScale;
    const p95x = leftMargin + (t.win_p95 || 0) * xScale;
    const actW = t.actual_wins != null ? leftMargin + t.actual_wins * xScale : null;

    // 90% CI band
    svg += `<rect x="${p5x}" y="${y+2}" width="${p95x - p5x}" height="${barH-4}" rx="2" fill="var(--purple)" opacity="0.15"/>`;
    // Projected bar
    svg += `<rect x="${leftMargin}" y="${y+4}" width="${projW - leftMargin}" height="${barH-8}" rx="2" fill="var(--purple)" opacity="0.7"/>`;
    if (actW != null) {
      svg += `<line x1="${actW}" y1="${y+1}" x2="${actW}" y2="${y+barH-1}" stroke="var(--green)" stroke-width="2.5"/>`;
    }

    svg += `<text x="${leftMargin - 4}" y="${y + barH/2 + 4}" text-anchor="end" font-size="11" font-weight="600">${t.team}</text>`;
    const errStr = t.win_error != null ? ` (${t.win_error > 0 ? "+" : ""}${Number(t.win_error).toFixed(1)})` : "";
    svg += `<text x="${Math.max(projW, p95x, actW || 0) + 6}" y="${y + barH/2 + 4}" font-size="10" fill="var(--text-muted)">${(t.projected_wins || 0).toFixed(0)}P / ${t.actual_wins != null ? t.actual_wins : "?"}A ${errStr}</text>`;
  });

  svg += `<rect x="${chartW - 160}" y="${h - 18}" width="12" height="6" rx="1" fill="var(--purple)" opacity="0.7"/>`;
  svg += `<text x="${chartW - 144}" y="${h - 12}" font-size="10" fill="var(--text-muted)">Projected</text>`;
  svg += `<line x1="${chartW - 104}" y1="${h - 18}" x2="${chartW - 104}" y2="${h - 12}" stroke="var(--green)" stroke-width="2.5"/>`;
  svg += `<text x="${chartW - 98}" y="${h - 12}" font-size="10" fill="var(--text-muted)">Actual</text>`;
  svg += `<rect x="${chartW - 80}" y="${h - 18}" width="20" height="6" rx="1" fill="var(--purple)" opacity="0.15"/>`;
  svg += `<text x="${chartW - 56}" y="${h - 12}" font-size="10" fill="var(--text-muted)">90% CI</text>`;

  svg += "</svg>";
  document.getElementById("forecastWinsChart").innerHTML = svg;
}

function renderForecastValidation() {
  const payload = getForecastScenarioPayload(forecastScenario);
  const valData = payload.validation || {};
  const projections = valData.projections || {};
  const el = document.getElementById("forecastValChart");

  if (Object.keys(projections).length === 0) {
    el.innerHTML = '<div style="color:var(--text-muted);padding:20px">No validation data for this scenario (backtest mode only)</div>';
    return;
  }

  let html = '<div style="padding:8px">';
  html += '<div style="font-size:12px;color:var(--text-muted);margin-bottom:8px">Player Projection Accuracy (Backtest)</div>';

  for (const [key, val] of Object.entries(projections)) {
    html += `<div style="margin-bottom:12px;padding:8px;background:var(--surface);border:1px solid var(--border);border-radius:6px">`;
    html += `<div style="font-weight:600;color:var(--accent);margin-bottom:4px">${key}</div>`;
    html += `<div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;font-size:12px">`;
    html += `<div>Returning: <span style="color:var(--green)">${val.n_returning}</span></div>`;
    html += `<div>Rookies: <span style="color:var(--orange)">${val.n_rookies}</span></div>`;
    html += `<div>BKE r: <span style="color:var(--green)">${val.bke_correlation?.toFixed(3) || "-"}</span></div>`;
    html += `<div>BKE MAE: <span style="color:var(--accent)">${val.bke_mae?.toFixed(3) || "-"}</span></div>`;
    html += `<div>MPG r: <span style="color:var(--green)">${val.mpg_correlation?.toFixed(3) || "-"}</span></div>`;
    html += `<div>MPG MAE: <span style="color:var(--accent)">${val.mpg_mae?.toFixed(1) || "-"}</span></div>`;
    html += `<div>Team Match: <span style="color:var(--green)">${val.team_match_rate ? (val.team_match_rate * 100).toFixed(0) + "%" : "-"}</span></div>`;
    html += `</div></div>`;
  }

  html += '</div>';
  el.innerHTML = html;
}

function getForecastLineupData() {
  const payload = getForecastScenarioPayload(forecastScenario);
  return (payload && payload.lineup) || {};
}

function getForecastLineupSeasons() {
  const lineupData = getForecastLineupData();
  return Object.keys((lineupData && lineupData.seasons) || {}).sort();
}

function initForecastLineupConferenceTabs() {
  const el = document.getElementById("forecastLineupConferenceTabs");
  if (!el) return;
  const options = [
    { key: "all", label: "All Teams" },
    { key: "east", label: "East" },
    { key: "west", label: "West" },
  ];
  el.innerHTML = options.map(opt => `
    <button class="filter-tab${opt.key === forecastLineupConferenceFilter ? " active" : ""}" data-filter="${opt.key}">${opt.label}</button>
  `).join("");
  el.querySelectorAll(".filter-tab").forEach(btn => {
    btn.addEventListener("click", () => {
      forecastLineupConferenceFilter = btn.dataset.filter || "all";
      initForecastLineupConferenceTabs();
      renderForecastLineup();
    });
  });
}

function renderForecastLineup() {
  const lineupData = getForecastLineupData();
  const lineupSeasons = getForecastLineupSeasons();

  if (!lineupSeasons.length || !lineupData.seasons) {
    document.getElementById("forecastLineupStatsRow").innerHTML = `
      <div class="stat-card"><div class="label">Forecast Lineup</div><div class="value val-orange">No Data</div><div class="detail">Run: python3 src/simulation/run_forecast.py (without --skip-lineup)</div></div>
    `;
    document.getElementById("forecastLineupCardGrid").innerHTML = "";
    return;
  }

  // Use the same season as the forecast season tab if available, else latest
  const targetSeason = lineupSeasons.includes(forecastSeason) ? forecastSeason : lineupSeasons[lineupSeasons.length - 1];
  const seasonData = lineupData.seasons[targetSeason];
  if (!seasonData) {
    document.getElementById("forecastLineupStatsRow").innerHTML = "";
    document.getElementById("forecastLineupCardGrid").innerHTML = "";
    return;
  }

  renderForecastLineupStats(seasonData, targetSeason);
  renderForecastLineupCards(seasonData, targetSeason);
}

function renderForecastLineupStats(seasonData, season) {
  const s = seasonData.summary || {};
  const scenarioLabel = getForecastScenarioLabel(forecastScenario);
  const cards = [
    {
      label: "Scenario",
      value: scenarioLabel,
      cls: "val-purple",
      detail: forecastScenario || "default"
    },
    {
      label: "Season",
      value: season || "-",
      cls: "val-purple",
      detail: "Forecast lineup projection"
    },
    {
      label: "Teams",
      value: s.n_teams != null ? String(s.n_teams) : "-",
      cls: "val-accent",
      detail: "Teams with projected lineups"
    },
    {
      label: "Starter Overlap",
      value: s.starter_overlap_count_mean != null ? Number(s.starter_overlap_count_mean).toFixed(2) : "-",
      cls: (s.starter_target_met ? "val-green" : "val-orange"),
      detail: "Predicted vs observed first-quarter starters from PBP"
    },
    {
      label: "Clutch Overlap",
      value: s.clutch_overlap_count_mean != null ? Number(s.clutch_overlap_count_mean).toFixed(2) : "-",
      cls: (s.clutch_target_met ? "val-green" : "val-orange"),
      detail: "Predicted vs top-5 clutch-minute players"
    },
    {
      label: "Rotation Corr",
      value: s.rotation_corr != null ? Number(s.rotation_corr).toFixed(3) : "-",
      cls: (s.rotation_target_met ? "val-green" : "val-purple"),
      detail: "Predicted rotation vs actual bench-player impact from game logs"
    },
    {
      label: "Starter Exact-5",
      value: (s.observed_starter_exact5_count != null && s.n_teams != null)
        ? `${s.observed_starter_exact5_count}/${s.n_teams}`
        : "-",
      cls: (s.observed_starter_exact5_rate != null && Number(s.observed_starter_exact5_rate) >= 0.999)
        ? "val-green"
        : "val-orange",
      detail: "Observed starter-set size integrity"
    },
    {
      label: "Starter Set Size",
      value: s.observed_starter_size_mean != null ? Number(s.observed_starter_size_mean).toFixed(2) : "-",
      cls: "val-accent",
      detail: "Mean observed starter pool size"
    },
  ];

  document.getElementById("forecastLineupStatsRow").innerHTML = cards.map(c => `
    <div class="stat-card">
      <div class="label">${c.label}</div>
      <div class="value ${c.cls}">${c.value}</div>
      <div class="detail">${c.detail}</div>
    </div>
  `).join("");
}

function renderForecastLineupCards(seasonData, season) {
  let teams = (seasonData.teams || []).slice().sort((a, b) => (a.team_abbreviation || "").localeCompare(b.team_abbreviation || ""));
  teams = filterRowsByConference(teams, forecastLineupConferenceFilter);
  const grid = document.getElementById("forecastLineupCardGrid");

  if (!teams.length) {
    grid.innerHTML = `<div class="stat-card"><div class="label">Lineup Cards</div><div class="value val-orange">No Teams</div><div class="detail">No forecast lineup data for ${conferenceFilterLabel(forecastLineupConferenceFilter)}</div></div>`;
    return;
  }

  grid.innerHTML = teams.map(team => {
    const v = team.validation || {};
    return `
      <div class="lineup-card" data-team="${team.team_abbreviation}" data-source="forecast" data-season="${season}">
        <div class="title-row">
          <div class="team-name">${team.team_abbreviation}</div>
          <div class="conf">${team.conference || "Unknown"}</div>
        </div>
        <div class="mini">
          <div class="k">Starter</div><div class="v">${fmt(team.mu_start, 2)} / ${fmt(team.sigma_start, 2)}</div>
          <div class="k">Rotation</div><div class="v">${fmt(team.mu_rotation, 2)} / ${fmt(team.sigma_rotation, 2)}</div>
          <div class="k">Clutch</div><div class="v">${fmt(team.mu_clutch, 2)} / ${fmt(team.sigma_clutch, 2)}</div>
          <div class="k">Pool Size</div><div class="v">${team.n_players_pool || "-"}</div>
        </div>
        <div class="validation">
          <span>S: ${v.starter_overlap != null ? v.starter_overlap : "-"}/5</span>
          <span>C: ${v.clutch_overlap != null ? v.clutch_overlap : "-"}/5</span>
          <span>R: ${v.rotation_actual_net != null ? fmt(v.rotation_actual_net, 2) : "-"}</span>
        </div>
      </div>
    `;
  }).join("");

  document.querySelectorAll("#forecastLineupCardGrid .lineup-card").forEach(card => {
    card.addEventListener("click", () => {
      const teamAbbr = card.dataset.team;
      const cardSeason = card.dataset.season;
      openForecastLineupModal(cardSeason, teamAbbr);
    });
  });
}

function findForecastLineupTeam(season, teamAbbr) {
  const lineupData = getForecastLineupData();
  const seasonData = ((lineupData && lineupData.seasons) || {})[season] || {};
  const teams = seasonData.teams || [];
  return teams.find(t => t.team_abbreviation === teamAbbr) || null;
}

function openForecastLineupModal(season, teamAbbr) {
  const team = findForecastLineupTeam(season, teamAbbr);
  if (!team) return;

  const v = team.validation || {};
  const actualStarterNames = resolveForecastObservedNames(
    season,
    v.actual_starter_ids || [],
    v.actual_starter_names || []
  );
  const actualClutchNames = resolveForecastObservedNames(
    season,
    v.actual_clutch_ids || [],
    v.actual_clutch_names || []
  );
  const starterSource = cleanText(v.starter_target_source) || "n/a";
  const starterGamesTotal = v.starter_games_total != null ? v.starter_games_total : "-";
  const starterGamesVerified = v.starter_games_verified != null ? v.starter_games_verified : "-";

  document.getElementById("lineupModalTitle").textContent = `${team.team_abbreviation} (${season}) — Forecast`;

  const html = `
    <div class="lineup-modal-grid">
      <div class="lineup-panel">
        <h4>Phase Metrics</h4>
        <div class="kv">
          <div class="k">Starter Mu / Sigma</div><div class="v">${fmt(team.mu_start,2)} / ${fmt(team.sigma_start,2)}</div>
          <div class="k">Rotation Mu / Sigma</div><div class="v">${fmt(team.mu_rotation,2)} / ${fmt(team.sigma_rotation,2)}</div>
          <div class="k">Clutch Mu / Sigma</div><div class="v">${fmt(team.mu_clutch,2)} / ${fmt(team.sigma_clutch,2)}</div>
          <div class="k">Team Clutch Minutes</div><div class="v">${fmt(team.team_clutch_minutes_total,1)}</div>
          <div class="k">Pool Size</div><div class="v">${team.n_players_pool || "-"}</div>
        </div>
      </div>

      <div class="lineup-panel">
        <h4>Validation</h4>
        <div class="kv">
          <div class="k">Starter Overlap</div><div class="v">${v.starter_overlap != null ? v.starter_overlap + "/5" : "-"}</div>
          <div class="k">Clutch Overlap</div><div class="v">${v.clutch_overlap != null ? v.clutch_overlap + "/5" : "-"}</div>
          <div class="k">Rotation Predicted</div><div class="v">${fmt(v.rotation_predicted,2)}</div>
          <div class="k">Rotation Actual (bench-player impact)</div><div class="v">${fmt(v.rotation_actual_net,2)}</div>
          <div class="k">Observed Starter Players</div><div class="v">${actualStarterNames.length || 0}</div>
          <div class="k">Observed Clutch Players</div><div class="v">${actualClutchNames.length || 0}</div>
          <div class="k">Starter Target Source</div><div class="v">${starterSource}</div>
          <div class="k">Starter Games (verified/total)</div><div class="v">${starterGamesVerified}/${starterGamesTotal}</div>
        </div>
      </div>

      <div class="lineup-panel">
        <h4>Rotation Breakdown</h4>
        <div class="kv">
          <div class="k">Bench Impact</div><div class="v">${fmt((team.rotation_breakdown||{}).bench_impact, 2)}</div>
          <div class="k">Stagger Impact</div><div class="v">${fmt((team.rotation_breakdown||{}).stagger_impact, 2)}</div>
          <div class="k">Alpha</div><div class="v">${fmt((team.rotation_breakdown||{}).alpha, 2)}</div>
        </div>
      </div>

      <div class="lineup-panel">
        <h4>Predicted Starters</h4>
        ${renderPlayerList(team.starter_players || [])}
      </div>

      <div class="lineup-panel">
        <h4>Predicted Clutch Lineup</h4>
        ${renderPlayerList(team.clutch_players || [])}
      </div>

      <div class="lineup-panel">
        <h4>Observed Starter Players (Proxy)</h4>
        <div style="font-size:12px;color:var(--text-muted)">${actualStarterNames.length ? actualStarterNames.join(", ") : "No observed starter proxy"}</div>
      </div>

      <div class="lineup-panel">
        <h4>Observed Clutch Players</h4>
        <div style="font-size:12px;color:var(--text-muted)">${actualClutchNames.length ? actualClutchNames.join(", ") : "No observed clutch lineup"}</div>
      </div>
    </div>
  `;

  document.getElementById("lineupModalContent").innerHTML = html;
  document.getElementById("lineupModalOverlay").classList.add("open");
}

// ═══════════════════════════════════════════════════════════════
// Player Stat Sim View
// ═══════════════════════════════════════════════════════════════

function getPssPayload() {
  return (PSS_DATA && PSS_DATA[pssMode]) || {};
}

function getPssSeasons() {
  const payload = getPssPayload();
  return (payload && payload.seasons) || [];
}

function getPssPlayers() {
  const payload = getPssPayload();
  return (payload && payload.player_comparisons) || [];
}

function getPssTeams() {
  const players = getPssPlayers();
  const teams = new Set();
  players.forEach(p => { if (p.team) teams.add(p.team); });
  return Array.from(teams).sort();
}

function initPlayerStatSimView() {
  // Mode tabs
  const modeEl = document.getElementById("pssModeTabs");
  const modes = [
    { key: "backtest", label: "Backtest (Actual Seasons)" },
    { key: "forecast", label: "Forecast" },
  ];
  modeEl.innerHTML = modes.map(m => `
    <button class="model-tab${m.key === pssMode ? " active" : ""}" data-mode="${m.key}">${m.label}</button>
  `).join("");
  modeEl.querySelectorAll(".model-tab").forEach(btn => {
    btn.addEventListener("click", () => {
      pssMode = btn.dataset.mode;
      pssSeason = null;
      pssTeamFilter = "all";
      initPlayerStatSimView();
      renderPlayerStatSim();
    });
  });

  // Season tabs
  const seasons = getPssSeasons();
  const seasonEl = document.getElementById("pssSeasonTabs");
  if (!pssSeason && seasons.length) pssSeason = seasons[seasons.length - 1];
  if (pssSeason && !seasons.includes(pssSeason) && seasons.length) pssSeason = seasons[seasons.length - 1];
  const allSeasons = ["all", ...seasons];
  seasonEl.innerHTML = allSeasons.map(s => {
    const label = s === "all" ? "All Seasons" : s;
    return `<div class="season-tab${(pssSeason || "all") === s ? " active" : ""}" data-season="${s}">${label}</div>`;
  }).join("");
  seasonEl.querySelectorAll(".season-tab").forEach(tab => {
    tab.addEventListener("click", () => {
      pssSeason = tab.dataset.season === "all" ? null : tab.dataset.season;
      initPlayerStatSimView();
      renderPlayerStatSim();
    });
  });

  // Team filter
  const teams = getPssTeams();
  const teamEl = document.getElementById("pssTeamTabs");
  const teamOptions = [{ key: "all", label: "All Teams" }, ...teams.map(t => ({ key: t, label: t }))];
  teamEl.innerHTML = teamOptions.slice(0, 32).map(opt => `
    <button class="filter-tab${opt.key === pssTeamFilter ? " active" : ""}" data-filter="${opt.key}">${opt.label}</button>
  `).join("");
  teamEl.querySelectorAll(".filter-tab").forEach(btn => {
    btn.addEventListener("click", () => {
      pssTeamFilter = btn.dataset.filter || "all";
      renderPlayerStatSim();
      // re-highlight
      teamEl.querySelectorAll(".filter-tab").forEach(b => b.classList.toggle("active", b.dataset.filter === pssTeamFilter));
    });
  });

  // Search
  const searchEl = document.getElementById("pssSearch");
  searchEl.value = pssSearchText;
  searchEl.oninput = () => { pssSearchText = searchEl.value; renderPlayerStatSim(); };
}

function getFilteredPssPlayers() {
  let players = getPssPlayers();
  if (pssSeason) {
    players = players.filter(p => p.season === pssSeason);
  }
  if (pssTeamFilter && pssTeamFilter !== "all") {
    players = players.filter(p => p.team === pssTeamFilter);
  }
  if (pssSearchText) {
    const q = pssSearchText.toLowerCase();
    players = players.filter(p => (p.player_name || "").toLowerCase().includes(q));
  }
  return players;
}

function renderPlayerStatSim() {
  const payload = getPssPayload();
  if (!payload || !payload.stat_accuracy) {
    document.getElementById("pssStatsRow").innerHTML = `
      <div class="stat-card"><div class="label">Player Stat Sim</div><div class="value val-orange">No Data</div>
      <div class="detail">Run: python3 scripts/validate_player_stat_sim.py</div></div>
    `;
    document.getElementById("pssPtsChart").innerHTML = "";
    document.getElementById("pssAccChart").innerHTML = "";
    document.getElementById("pssTableWrap").innerHTML = "";
    return;
  }

  renderPssStats(payload);
  renderPssCharts(payload);
  renderPssTable();
}

function renderPssStats(payload) {
  const acc = payload.stat_accuracy || {};
  const pts = acc.pts || {};
  const ast = acc.ast || {};
  const reb = acc.reb || {};
  const stl = acc.stl || {};
  const blk = acc.blk || {};

  const cards = [
    { label: "Mode", value: (payload.mode || pssMode).toUpperCase(), cls: "val-purple", detail: payload.scenario || "default" },
    { label: "Players", value: payload.n_compared || 0, cls: "val-accent", detail: `Matched comparisons` },
    { label: "PTS r", value: pts.correlation != null ? pts.correlation.toFixed(3) : "-", cls: "val-green", detail: `MAE: ${pts.mae != null ? pts.mae.toFixed(2) : "-"}` },
    { label: "PTS Bias", value: pts.bias != null ? (pts.bias >= 0 ? "+" : "") + pts.bias.toFixed(2) : "-", cls: pts.bias > 0.5 ? "val-orange" : "val-green", detail: `RMSE: ${pts.rmse != null ? pts.rmse.toFixed(2) : "-"}` },
    { label: "AST r", value: ast.correlation != null ? ast.correlation.toFixed(3) : "-", cls: "val-green", detail: `MAE: ${ast.mae != null ? ast.mae.toFixed(2) : "-"}` },
    { label: "REB r", value: reb.correlation != null ? reb.correlation.toFixed(3) : "-", cls: "val-green", detail: `MAE: ${reb.mae != null ? reb.mae.toFixed(2) : "-"}` },
    { label: "STL r", value: stl.correlation != null ? stl.correlation.toFixed(3) : "-", cls: "val-green", detail: `MAE: ${stl.mae != null ? stl.mae.toFixed(2) : "-"}` },
    { label: "BLK r", value: blk.correlation != null ? blk.correlation.toFixed(3) : "-", cls: "val-green", detail: `MAE: ${blk.mae != null ? blk.mae.toFixed(2) : "-"}` },
    { label: "Seasons", value: (payload.seasons || []).join(", ") || "-", cls: "val-accent", detail: "Validated" },
  ];

  document.getElementById("pssStatsRow").innerHTML = cards.map(c => `
    <div class="stat-card">
      <div class="label">${c.label}</div>
      <div class="value ${c.cls}">${c.value}</div>
      <div class="detail">${c.detail}</div>
    </div>
  `).join("");
}

function renderPssCharts(payload) {
  // Scatter plot: sim vs actual PPG
  const players = getFilteredPssPlayers();
  const chartEl = document.getElementById("pssPtsChart");

  if (!players.length) {
    chartEl.innerHTML = '<div style="color:var(--text-muted);padding:20px">No players to display</div>';
    document.getElementById("pssAccChart").innerHTML = "";
    return;
  }

  const w = 550, h = 400, pad = 50;
  const plotW = w - 2 * pad, plotH = h - 2 * pad;
  const maxPts = Math.max(35, ...players.map(p => Math.max(p.sim_pts || 0, p.actual_pts || 0)));

  let svg = `<svg width="100%" viewBox="0 0 ${w} ${h}">`;
  // Diagonal
  svg += `<line x1="${pad}" y1="${pad + plotH}" x2="${pad + plotW}" y2="${pad}" stroke="var(--text-muted)" stroke-dasharray="4,3" opacity="0.4"/>`;
  // Axes
  svg += `<line x1="${pad}" y1="${pad + plotH}" x2="${pad + plotW}" y2="${pad + plotH}" stroke="var(--border)"/>`;
  svg += `<line x1="${pad}" y1="${pad}" x2="${pad}" y2="${pad + plotH}" stroke="var(--border)"/>`;
  svg += `<text x="${w/2}" y="${h - 6}" text-anchor="middle" font-size="11" fill="var(--text-muted)">Actual PPG</text>`;
  svg += `<text x="12" y="${h/2}" text-anchor="middle" font-size="11" fill="var(--text-muted)" transform="rotate(-90,12,${h/2})">Simulated PPG</text>`;

  for (let v = 0; v <= maxPts; v += 5) {
    const x = pad + (v / maxPts) * plotW;
    const y = pad + plotH - (v / maxPts) * plotH;
    svg += `<text x="${x}" y="${pad + plotH + 14}" text-anchor="middle" font-size="9" fill="var(--text-muted)">${v}</text>`;
    svg += `<text x="${pad - 6}" y="${y + 3}" text-anchor="end" font-size="9" fill="var(--text-muted)">${v}</text>`;
  }

  players.forEach(p => {
    if (p.actual_pts == null || p.sim_pts == null) return;
    const x = pad + (p.actual_pts / maxPts) * plotW;
    const y = pad + plotH - (p.sim_pts / maxPts) * plotH;
    const delta = Math.abs(p.sim_pts - p.actual_pts);
    const color = delta <= 2 ? "var(--green)" : delta <= 5 ? "var(--orange)" : "var(--red)";
    const safeName = (p.player_name || "").replace(/"/g, '&quot;');
    const safeTeam = (p.team || "").replace(/"/g, '&quot;');
    const safeSeason = (p.season || "").replace(/"/g, '&quot;');
    svg += `<circle cx="${x}" cy="${y}" r="4.5" fill="${color}" opacity="0.75" style="cursor:pointer" data-name="${safeName}" data-team="${safeTeam}" data-season="${safeSeason}" data-sim="${(p.sim_pts || 0).toFixed(1)}" data-actual="${(p.actual_pts || 0).toFixed(1)}" data-delta="${delta.toFixed(1)}"/>`;
  });

  svg += "</svg>";
  chartEl.innerHTML = svg;

  // Attach hover tooltip to scatter circles
  const tooltip = document.getElementById("chartTooltip");
  chartEl.querySelectorAll("circle[data-name]").forEach(circle => {
    circle.addEventListener("mouseenter", (e) => {
      const d = circle.dataset;
      tooltip.innerHTML = `<div class="tt-name">${d.name}</div>`
        + `<div class="tt-team">${d.team} | ${d.season}</div>`
        + `<div class="tt-row"><span class="tt-label">Simulated PPG</span><span class="tt-val">${d.sim}</span></div>`
        + `<div class="tt-row"><span class="tt-label">Actual PPG</span><span class="tt-val">${d.actual}</span></div>`
        + `<div class="tt-row"><span class="tt-label">Delta</span><span class="tt-val" style="color:${parseFloat(d.delta) <= 2 ? 'var(--green)' : parseFloat(d.delta) <= 5 ? 'var(--orange)' : 'var(--red)'}">${d.delta > 0 ? '+' : ''}${d.delta}</span></div>`;
      tooltip.style.display = "block";
      tooltip.style.left = (e.clientX + 14) + "px";
      tooltip.style.top  = (e.clientY - 10) + "px";
    });
    circle.addEventListener("mousemove", (e) => {
      tooltip.style.left = (e.clientX + 14) + "px";
      tooltip.style.top  = (e.clientY - 10) + "px";
    });
    circle.addEventListener("mouseleave", () => {
      tooltip.style.display = "none";
    });
  });

  // Accuracy bar chart
  const accEl = document.getElementById("pssAccChart");
  const acc = payload.stat_accuracy || {};
  const statKeys = ["pts", "ast", "reb", "stl", "blk", "tov"];
  const barW = 280, barH = statKeys.length * 30 + 30;
  let bSvg = `<svg width="100%" viewBox="0 0 ${barW} ${barH}">`;
  statKeys.forEach((k, i) => {
    const a = acc[k] || {};
    const corr = a.correlation || 0;
    const y = 15 + i * 30;
    const barLen = Math.max(0, corr) * 180;
    const color = corr >= 0.95 ? "var(--green)" : corr >= 0.85 ? "var(--accent)" : "var(--orange)";
    bSvg += `<text x="35" y="${y + 14}" text-anchor="end" font-size="12" fill="var(--text-muted)">${k.toUpperCase()}</text>`;
    bSvg += `<rect x="40" y="${y + 2}" width="${barLen}" height="16" rx="3" fill="${color}" opacity="0.7"/>`;
    bSvg += `<text x="${45 + barLen}" y="${y + 14}" font-size="11" fill="var(--text)">${corr.toFixed(3)}</text>`;
  });
  bSvg += "</svg>";
  accEl.innerHTML = bSvg;
}

function renderPssTable() {
  let players = getFilteredPssPlayers();
  // Sort
  players.sort((a, b) => {
    let va = a[pssSortCol] ?? 0, vb = b[pssSortCol] ?? 0;
    return pssSortDir === "asc" ? (va > vb ? 1 : -1) : (va < vb ? 1 : -1);
  });

  const cols = [
    { key: "player_name", label: "Player", cls: "team" },
    { key: "season", label: "Season", cls: "num" },
    { key: "team", label: "Team", cls: "num" },
    { key: "archetype", label: "Archetype", cls: "" },
    { key: "sim_gp", label: "Sim GP", cls: "num", fmt: v => v != null ? Math.round(v) : "-" },
    { key: "actual_gp", label: "Act GP", cls: "num", fmt: v => v != null ? Math.round(v) : "-" },
    { key: "sim_mpg", label: "Sim MPG", cls: "num", fmt: v => v != null ? v.toFixed(1) : "-" },
    { key: "actual_mpg", label: "Act MPG", cls: "num", fmt: v => v != null ? v.toFixed(1) : "-" },
    { key: "sim_pts", label: "Sim PPG", cls: "num", fmt: v => v != null ? v.toFixed(1) : "-" },
    { key: "actual_pts", label: "Act PPG", cls: "num", fmt: v => v != null ? v.toFixed(1) : "-" },
    { key: "delta_pts", label: "Δ PTS", cls: "num", fmt: v => { if (v == null) return "-"; const c = Math.abs(v) <= 2 ? "err-good" : Math.abs(v) <= 5 ? "err-ok" : "err-bad"; return `<span class="${c}">${v >= 0 ? "+" : ""}${v.toFixed(1)}</span>`; } },
    { key: "sim_ast", label: "Sim APG", cls: "num", fmt: v => v != null ? v.toFixed(1) : "-" },
    { key: "actual_ast", label: "Act APG", cls: "num", fmt: v => v != null ? v.toFixed(1) : "-" },
    { key: "delta_ast", label: "Δ AST", cls: "num", fmt: v => { if (v == null) return "-"; const c = Math.abs(v) <= 1 ? "err-good" : Math.abs(v) <= 2 ? "err-ok" : "err-bad"; return `<span class="${c}">${v >= 0 ? "+" : ""}${v.toFixed(1)}</span>`; } },
    { key: "sim_reb", label: "Sim RPG", cls: "num", fmt: v => v != null ? v.toFixed(1) : "-" },
    { key: "actual_reb", label: "Act RPG", cls: "num", fmt: v => v != null ? v.toFixed(1) : "-" },
    { key: "delta_reb", label: "Δ REB", cls: "num", fmt: v => { if (v == null) return "-"; const c = Math.abs(v) <= 1 ? "err-good" : Math.abs(v) <= 2 ? "err-ok" : "err-bad"; return `<span class="${c}">${v >= 0 ? "+" : ""}${v.toFixed(1)}</span>`; } },
    { key: "sim_stl", label: "Sim SPG", cls: "num", fmt: v => v != null ? v.toFixed(1) : "-" },
    { key: "actual_stl", label: "Act SPG", cls: "num", fmt: v => v != null ? v.toFixed(1) : "-" },
    { key: "sim_blk", label: "Sim BPG", cls: "num", fmt: v => v != null ? v.toFixed(1) : "-" },
    { key: "actual_blk", label: "Act BPG", cls: "num", fmt: v => v != null ? v.toFixed(1) : "-" },
    { key: "sim_tov", label: "Sim TOV", cls: "num", fmt: v => v != null ? v.toFixed(1) : "-" },
    { key: "actual_tov", label: "Act TOV", cls: "num", fmt: v => v != null ? v.toFixed(1) : "-" },
    { key: "sim_fga", label: "Sim FGA", cls: "num", fmt: v => v != null ? v.toFixed(1) : "-" },
    { key: "actual_fga", label: "Act FGA", cls: "num", fmt: v => v != null ? v.toFixed(1) : "-" },
    { key: "sim_fg3m", label: "Sim 3PM", cls: "num", fmt: v => v != null ? v.toFixed(1) : "-" },
    { key: "actual_fg3m", label: "Act 3PM", cls: "num", fmt: v => v != null ? v.toFixed(1) : "-" },
    { key: "sim_ftm", label: "Sim FTM", cls: "num", fmt: v => v != null ? v.toFixed(1) : "-" },
    { key: "actual_ftm", label: "Act FTM", cls: "num", fmt: v => v != null ? v.toFixed(1) : "-" },
  ];

  let html = "<table><thead><tr>";
  cols.forEach(c => {
    const cls = c.key === pssSortCol ? (pssSortDir === "asc" ? "sorted-asc" : "sorted-desc") : "";
    html += `<th class="${cls}" data-col="${c.key}">${c.label}</th>`;
  });
  html += "</tr></thead><tbody>";

  players.forEach(p => {
    html += "<tr>";
    cols.forEach(c => {
      const raw = p[c.key];
      const display = c.fmt ? c.fmt(raw) : (raw != null ? raw : "-");
      html += `<td class="${c.cls || ""}">${display}</td>`;
    });
    html += "</tr>";
  });
  html += "</tbody></table>";

  document.getElementById("pssTableWrap").innerHTML = html;
  document.querySelectorAll("#pssTableWrap th").forEach(th => {
    th.addEventListener("click", () => {
      const col = th.dataset.col;
      if (pssSortCol === col) {
        pssSortDir = pssSortDir === "asc" ? "desc" : "asc";
      } else {
        pssSortCol = col;
        pssSortDir = col === "player_name" || col === "team" || col === "season" ? "asc" : "desc";
      }
      renderPssTable();
    });
  });
}

// ═══════════════════════════════════════════════════════════════
// Enhanced Single Game Simulator (own tab)
// ═══════════════════════════════════════════════════════════════

function simulatePlayerGameStats(teamEntry, teamStep2, opponentStep2, teamScore, oppScore, possessions, isHome, closeGame) {
  const players = [];
  if (!teamStep2) return players;

  // ── 1. Build full roster pool (deduped by player_id) ──
  const starterIds = new Set((teamStep2.starter_players || []).map(p => String(p.player_id)));
  const clutchIds  = new Set((teamStep2.clutch_players  || []).map(p => String(p.player_id)));
  const seen = new Set();
  const fullPool = [];
  const addToPool = (arr) => (arr || []).forEach(p => {
    const pid = String(p.player_id);
    if (!seen.has(pid)) { seen.add(pid); fullPool.push(p); }
  });
  addToPool(teamStep2.starter_players);
  addToPool(teamStep2.clutch_players);
  addToPool(teamStep2.pool_players);

  if (!fullPool.length) return players;

  // ── 2. Rotation selection: limit to 10 players ──
  const ROTATION_SIZE = 10;
  const rotSize = Math.min(ROTATION_SIZE, fullPool.length);

  // Compute importance for selection (starters always included)
  const importance = fullPool.map(p => {
    const base = Math.max(numOrZero(p.minutes) || numOrZero(p.mpg) || 10, 1);
    const pid = String(p.player_id);
    let imp = base;
    if (starterIds.has(pid)) imp *= 1.20;
    if (clutchIds.has(pid)) imp *= 1.10;
    return imp;
  });

  // Select rotation: starters first, then fill by importance
  const selectedIdxs = new Set();
  fullPool.forEach((p, i) => { if (starterIds.has(String(p.player_id))) selectedIdxs.add(i); });
  const sortedByImp = [...fullPool.keys()].sort((a, b) => importance[b] - importance[a]);
  for (const idx of sortedByImp) {
    if (selectedIdxs.size >= rotSize) break;
    selectedIdxs.add(idx);
  }
  const rotation = [...selectedIdxs].map(i => fullPool[i]);

  // ── 3. Anchored minute allocation (matches backend logic) ──
  const TOTAL_MINUTES = 240;
  const baseMpgs = rotation.map(p => Math.max(numOrZero(p.minutes) || numOrZero(p.mpg) || 10, 6));
  const allocBase = baseMpgs.map((bm, i) => {
    const pid = String(rotation[i].player_id);
    let m = bm;
    if (starterIds.has(pid))       m += 1.4;   // starter bonus (+0.14 * 10)
    if (clutchIds.has(pid))        m += 0.5;   // clutch bonus (+0.05 * 10)
    if (!starterIds.has(pid))      m -= 0.8;   // bench penalty (-0.08 * 10)
    return Math.max(m, 4);
  });
  const allocSum = allocBase.reduce((a, b) => a + b, 0);
  let mins = allocBase.map(b => TOTAL_MINUTES * b / Math.max(allocSum, 1));

  // Cap: no player plays more than base_mpg + 4
  const caps = baseMpgs.map(bm => Math.min(bm + 4, 40));
  let excess = 0;
  mins = mins.map((m, i) => {
    if (m > caps[i]) { excess += m - caps[i]; return caps[i]; }
    return m;
  });
  if (excess > 0.5) {
    const headroom = mins.map((m, i) => caps[i] - m);
    const hrTotal = headroom.reduce((a, b) => a + b, 0);
    if (hrTotal > 0.5) {
      mins = mins.map((m, i) => Math.min(m + headroom[i] * (excess / hrTotal), caps[i]));
    }
  }
  mins = mins.map(m => Math.max(4, Math.min(40, m)));
  const mTotal = mins.reduce((a, b) => a + b, 0);
  mins = mins.map(m => TOTAL_MINUTES * m / Math.max(mTotal, 1));

  // ── 4. Opponent style for matchup adjustments ──
  const oppStyle = summarizeTeamStyle(opponentStep2);

  // ── 5. Simulate physically-consistent stats for each player ──
  // Helper: binomial-ish draw (clamp a percentage to make/attempt)
  const binomDraw = (n, p) => {
    if (n <= 0) return 0;
    let made = 0;
    for (let i = 0; i < n; i++) { if (Math.random() < p) made++; }
    return made;
  };

  let rawTotalPts = 0;
  const rawPlayers = [];

  rotation.forEach((p, i) => {
    const min = mins[i];
    const pid = String(p.player_id);
    const off = (cleanText(p.off_archetype) || "Unknown").toLowerCase();
    const def = (cleanText(p.def_archetype) || "Unknown").toLowerCase();
    const impact = numOrZero(p.impact) || numOrZero(p.score) || 0;

    // Archetype matchup adjustments
    let scoringEff = 1.0, threeEff = 1.0, tovMult = 1.0, astMult = 1.0, rebMult = 1.0, stocksMult = 1.0;
    const oppPoa = oppStyle.poa / 5;
    const oppRimDef = oppStyle.rimDefense / 5;
    const oppVersatile = oppStyle.switchability / 5;
    const oppCreation = oppStyle.creation / 5;
    const oppRimPressure = oppStyle.rimPressure / 5;

    if (/(ball dominant|ballhandler|perimeter|all-around|creator)/.test(off)) {
      scoringEff -= 0.08 * oppPoa;
      tovMult += 0.16 * (oppPoa + oppVersatile);
      astMult -= 0.07 * oppPoa;
    }
    if (/(shooter|popping)/.test(off)) {
      threeEff -= 0.09 * oppVersatile;
      threeEff += 0.06 * (1 - oppVersatile);
    }
    if (/(interior|finisher|rolling|rim)/.test(off)) {
      scoringEff -= 0.10 * oppRimDef;
      rebMult += 0.05 * (1 - oppRimDef);
    }
    if (/(poa|wing stopper|chaser)/.test(def)) {
      stocksMult += 0.12 * oppCreation;
    }
    if (/(rim protector|dropping|mobile big)/.test(def)) {
      stocksMult += 0.10 * oppRimPressure;
      rebMult += 0.05;
    }

    // Base rates scaled by minutes and archetype
    const baseUsage = 0.15 + 0.25 * Math.min(1, Math.max(0, impact / 60));
    const baseFga = Math.max(2, 12 * baseUsage * 2.5) * (min / 36);
    const baseFta = Math.max(0.5, 3 * baseUsage * 2.5) * (min / 36);

    // Shot distribution and efficiency by archetype
    const fg3Share = /(shooter|perimeter|popping)/.test(off) ? 0.45 : /(interior|rolling|finisher)/.test(off) ? 0.05 : 0.30;
    const fg2Pct = Math.min(0.70, (/(interior|rolling|finisher)/.test(off) ? 0.58 : 0.48) * scoringEff);
    const fg3Pct = Math.min(0.50, (/(shooter)/.test(off) ? 0.38 : 0.34) * threeEff);
    const ftPct = 0.77;

    // Shot attempts (with noise)
    const fga = Math.max(1, Math.round(baseFga + (Math.random() - 0.5) * 3));
    const fg3a = Math.max(0, Math.min(fga, Math.round(fga * fg3Share + (Math.random() - 0.5) * 1)));
    const fg2a = fga - fg3a;

    // Field goals made via binomial draws (physically consistent)
    const fg2m = binomDraw(fg2a, fg2Pct);
    const fg3m = binomDraw(fg3a, fg3Pct);

    // Free throws
    const fta = Math.max(0, Math.round(baseFta + (Math.random() - 0.5) * 2));
    const ftm = binomDraw(fta, ftPct);

    // PTS = 2*FG2M + 3*FG3M + FTM (always physically consistent)
    const pts = 2 * fg2m + 3 * fg3m + ftm;

    // Other stats
    const ast = Math.max(0, Math.round((/(ball dominant|ballhandler|creator|playmaking)/.test(off) ? 6 : 2) * (min / 36) * astMult + (Math.random() - 0.5) * 2));
    const reb = Math.max(0, Math.round((/(interior|rolling|big|center)/.test(off) ? 8 : 4) * (min / 36) * rebMult + (Math.random() - 0.5) * 2));
    const stl = Math.max(0, Math.round(1.0 * (min / 36) * stocksMult + (Math.random() - 0.5) * 0.8));
    const blk = Math.max(0, Math.round((/(rim protector|interior|big)/.test(off + " " + def) ? 1.5 : 0.4) * (min / 36) * stocksMult + (Math.random() - 0.5) * 0.8));
    const tov = Math.max(0, Math.round(1.5 * baseUsage * 5 * (min / 36) * tovMult + (Math.random() - 0.5) * 1));
    const pf = Math.max(0, Math.round(2.5 * (min / 36) + (Math.random() - 0.5) * 1));

    rawTotalPts += pts;

    rawPlayers.push({
      name: cleanNameLabel(p.player_name) || `ID ${cleanText(p.player_id) || "?"}`,
      player_id: p.player_id,
      phase: starterIds.has(pid) ? "Starter" : clutchIds.has(pid) ? "Clutch" : "Bench",
      off_archetype: cleanText(p.off_archetype) || "Unknown",
      def_archetype: cleanText(p.def_archetype) || "Unknown",
      role: cleanText(p.role) || cleanText(p.position_band) || "-",
      minutes: Math.round(min * 10) / 10,
      pts, ast, reb, stl, blk, tov, pf,
      fga, fgm: fg2m + fg3m, fg2a, fg2m, fg3a, fg3m, fta, ftm,
      impact: numOrZero(impact).toFixed(2),
      adjustments: {
        scoring_eff: Math.round(scoringEff * 1000) / 1000,
        three_eff: Math.round(threeEff * 1000) / 1000,
        tov_mult: Math.round(tovMult * 1000) / 1000,
        ast_mult: Math.round(astMult * 1000) / 1000,
        reb_mult: Math.round(rebMult * 1000) / 1000,
        stocks_mult: Math.round(stocksMult * 1000) / 1000,
      },
    });
  });

  // ── 6. Reconcile points to team score (add/remove FTM one at a time) ──
  if (rawTotalPts > 0 && teamScore > 0) {
    // Sort by usage (descending) for reconciliation order
    const sortedIdxs = [...rawPlayers.keys()].sort((a, b) => {
      const ua = rawPlayers[a].fga + rawPlayers[a].fta;
      const ub = rawPlayers[b].fga + rawPlayers[b].fta;
      return ub - ua;
    });
    let current = rawTotalPts;
    let cursor = 0;
    // Add points via FTM/FTA if under
    while (current < teamScore && sortedIdxs.length) {
      const idx = sortedIdxs[cursor % sortedIdxs.length];
      rawPlayers[idx].ftm += 1;
      rawPlayers[idx].fta += 1;
      rawPlayers[idx].pts += 1;
      current += 1;
      cursor += 1;
    }
    // Remove points via FTM if over
    cursor = 0;
    let safety = 0;
    while (current > teamScore && safety < 500) {
      const idx = sortedIdxs[cursor % sortedIdxs.length];
      if (rawPlayers[idx].ftm > 0) {
        rawPlayers[idx].ftm -= 1;
        rawPlayers[idx].fta = Math.max(rawPlayers[idx].fta, rawPlayers[idx].ftm);
        rawPlayers[idx].pts -= 1;
        current -= 1;
      }
      cursor += 1;
      safety += 1;
    }
  }

  return rawPlayers;
}

function renderEnhancedSingleGameResult() {
  const el = document.getElementById("sgResult");
  if (!currentSimulatedGame) {
    el.innerHTML = "No game simulated yet.";
    return;
  }
  const g = currentSimulatedGame;
  const ctx = g.context || {};
  const drivers = ctx.drivers || {};
  const phase = ctx.phaseEdges || {};
  const h2h = ctx.headToHead || [];

  // Simulate player stats for both teams
  const homePoss = 100;
  const homeScore = Math.round(Math.max(85, 110 + g.sampledMargin / 2 + (Math.random() - 0.5) * 8));
  const awayScore = Math.round(Math.max(85, 110 - g.sampledMargin / 2 + (Math.random() - 0.5) * 8));
  const closeGame = Math.abs(g.sampledMargin) <= 7.5;

  const homePlayerStats = simulatePlayerGameStats(
    getTeamEntry(g.homeSeason, g.homeTeam),
    ctx.homeStep2, ctx.awayStep2, homeScore, awayScore, homePoss, true, closeGame
  );
  const awayPlayerStats = simulatePlayerGameStats(
    getTeamEntry(g.awaySeason, g.awayTeam),
    ctx.awayStep2, ctx.homeStep2, awayScore, homeScore, homePoss, false, closeGame
  );

  const h2hHtml = renderHeadToHeadBars(g.homeTeam, g.awayTeam, h2h);
  const contribHome = renderContributors(`${g.homeTeam} Impact Drivers`, "home", (ctx.contributors || {}).home || []);
  const contribAway = renderContributors(`${g.awayTeam} Impact Drivers`, "away", (ctx.contributors || {}).away || []);

  const sampledWinnerClass = g.sampledMargin >= 0 ? "sg-badge-home" : "sg-badge-away";

  // Build lineup phase bonus section
  const homeStep2 = ctx.homeStep2 || {};
  const awayStep2 = ctx.awayStep2 || {};
  const lineupPhasesHtml = `
    <div class="sg-card">
      <h4>Lineup Phase Bonuses & Penalties</h4>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
        <div>
          <div style="font-weight:700;color:var(--accent);margin-bottom:6px">${g.homeTeam} (Home)</div>
          <div class="sg-list">
            <div class="row"><span class="k">Starter Mu</span><span class="v">${fmt(homeStep2.mu_start, 2)}</span></div>
            <div class="row"><span class="k">Rotation Mu</span><span class="v">${fmt(homeStep2.mu_rotation, 2)}</span></div>
            <div class="row"><span class="k">Clutch Mu</span><span class="v">${fmt(homeStep2.mu_clutch, 2)}</span></div>
            <div class="row"><span class="k">Continuity</span><span class="v">${fmt(homeStep2.continuity_score, 2)}</span></div>
            <div class="row"><span class="k">Starter Fit</span><span class="v">${fmt(homeStep2.starter_fit_score, 2)}</span></div>
          </div>
        </div>
        <div>
          <div style="font-weight:700;color:var(--orange);margin-bottom:6px">${g.awayTeam} (Away)</div>
          <div class="sg-list">
            <div class="row"><span class="k">Starter Mu</span><span class="v">${fmt(awayStep2.mu_start, 2)}</span></div>
            <div class="row"><span class="k">Rotation Mu</span><span class="v">${fmt(awayStep2.mu_rotation, 2)}</span></div>
            <div class="row"><span class="k">Clutch Mu</span><span class="v">${fmt(awayStep2.mu_clutch, 2)}</span></div>
            <div class="row"><span class="k">Continuity</span><span class="v">${fmt(awayStep2.continuity_score, 2)}</span></div>
            <div class="row"><span class="k">Starter Fit</span><span class="v">${fmt(awayStep2.starter_fit_score, 2)}</span></div>
          </div>
        </div>
      </div>
    </div>
  `;

  // Build player stats table for both teams
  function renderPlayerStatsTable(teamName, color, playerStats) {
    if (!playerStats.length) return `<div style="color:var(--text-muted)">No Step 2 data for ${teamName}</div>`;
    return `
      <div style="margin-top:10px">
        <div style="font-weight:700;color:${color};margin-bottom:6px;font-size:13px">${teamName} Player Stats</div>
        <div style="overflow-x:auto">
          <table style="width:100%;font-size:11px">
            <thead><tr>
              <th style="text-align:left">Player</th>
              <th>Phase</th>
              <th>OFF Arch</th>
              <th>DEF Arch</th>
              <th>MIN</th>
              <th>PTS</th>
              <th>AST</th>
              <th>REB</th>
              <th>STL</th>
              <th>BLK</th>
              <th>TOV</th>
              <th>FGM-A</th>
              <th>3PM-A</th>
              <th>FTM-A</th>
              <th>Score Eff</th>
              <th>3pt Eff</th>
              <th>TOV Mult</th>
              <th>STK Mult</th>
            </tr></thead>
            <tbody>
              ${playerStats.map(p => {
                const adj = p.adjustments || {};
                const phaseColor = p.phase === "Starter" ? "var(--green)" : p.phase === "Clutch" ? "var(--yellow)" : "var(--text-muted)";
                const effColor = (v) => v < 0.95 ? "var(--red)" : v > 1.05 ? "var(--green)" : "var(--text)";
                return `<tr>
                  <td style="text-align:left;font-weight:600;white-space:nowrap">${p.name}</td>
                  <td style="color:${phaseColor}">${p.phase}</td>
                  <td style="font-size:10px">${p.off_archetype}</td>
                  <td style="font-size:10px">${p.def_archetype}</td>
                  <td class="num">${p.minutes.toFixed(1)}</td>
                  <td class="num" style="font-weight:700">${p.pts}</td>
                  <td class="num">${p.ast}</td>
                  <td class="num">${p.reb}</td>
                  <td class="num">${p.stl}</td>
                  <td class="num">${p.blk}</td>
                  <td class="num">${p.tov}</td>
                  <td class="num">${p.fgm}-${p.fga}</td>
                  <td class="num">${p.fg3m}-${p.fg3a}</td>
                  <td class="num">${p.ftm}-${p.fta}</td>
                  <td class="num" style="color:${effColor(adj.scoring_eff)}">${adj.scoring_eff}</td>
                  <td class="num" style="color:${effColor(adj.three_eff)}">${adj.three_eff}</td>
                  <td class="num" style="color:${effColor(1/adj.tov_mult)}">${adj.tov_mult}</td>
                  <td class="num" style="color:${effColor(adj.stocks_mult)}">${adj.stocks_mult}</td>
                </tr>`;
              }).join("")}
            </tbody>
          </table>
        </div>
      </div>
    `;
  }

  // Archetype interaction summary — now includes player names
  // Build maps: archetype → [player names] for starters
  const homeStarters = homePlayerStats.filter(p => p.phase === "Starter");
  const awayStarters = awayPlayerStats.filter(p => p.phase === "Starter");

  const buildArchMap = (players, key) => {
    const m = new Map();
    players.forEach(p => {
      const arch = p[key] || "Unknown";
      if (!m.has(arch)) m.set(arch, []);
      m.get(arch).push(p.name);
    });
    return m;
  };
  const homeOffMap = buildArchMap(homeStarters, "off_archetype");
  const awayOffMap = buildArchMap(awayStarters, "off_archetype");
  const homeDefMap = buildArchMap(homeStarters, "def_archetype");
  const awayDefMap = buildArchMap(awayStarters, "def_archetype");

  function archInteraction(offMap, defMap) {
    const interactions = [];
    const creators = ["Ball Dominant Creator", "Ballhandler", "All-Around Scorer", "Perimeter Scorer"];
    const shooters = ["Off-Ball Movement Shooter", "Off-Ball Stationary Shooter", "PnR Popping Big"];
    const rimOff = ["Interior Scorer", "Off-Ball Finisher", "PnR Rolling Big"];
    const poaDef = ["POA Defender", "Wing Stopper", "Off-Ball Chaser"];
    const rimDef = ["Rim Protector", "Dropping Big", "Mobile Big"];

    creators.forEach(a => { if (offMap.has(a)) {
      poaDef.forEach(d => { if (defMap.has(d)) interactions.push({off: a, offPlayers: offMap.get(a), def: d, defPlayers: defMap.get(d), effect: "Scoring/AST penalized", type: "penalty"}); });
    }});
    shooters.forEach(a => { if (offMap.has(a)) {
      ["Versatile Defender", "Wing Stopper"].forEach(d => { if (defMap.has(d)) interactions.push({off: a, offPlayers: offMap.get(a), def: d, defPlayers: defMap.get(d), effect: "3PT efficiency reduced", type: "penalty"}); });
      ["Dropping Big"].forEach(d => { if (defMap.has(d)) interactions.push({off: a, offPlayers: offMap.get(a), def: d, defPlayers: defMap.get(d), effect: "3PT efficiency boosted", type: "bonus"}); });
    }});
    rimOff.forEach(a => { if (offMap.has(a)) {
      rimDef.forEach(d => { if (defMap.has(d)) interactions.push({off: a, offPlayers: offMap.get(a), def: d, defPlayers: defMap.get(d), effect: "Scoring efficiency reduced", type: "penalty"}); });
    }});
    return interactions;
  }

  const homeVsAwayInteractions = archInteraction(homeOffMap, awayDefMap);
  const awayVsHomeInteractions = archInteraction(awayOffMap, homeDefMap);

  function renderInteractions(interactions, offTeam, defTeam) {
    if (!interactions.length) return `<div style="color:var(--text-muted);font-size:12px">No specific archetype interactions detected</div>`;
    return interactions.map(i => {
      const color = i.type === "penalty" ? "var(--red)" : "var(--green)";
      const icon = i.type === "penalty" ? "▼" : "▲";
      const offNames = (i.offPlayers || []).join(", ");
      const defNames = (i.defPlayers || []).join(", ");
      return `<div style="font-size:12px;margin-bottom:6px;display:flex;gap:8px;align-items:flex-start">
        <span style="color:${color};font-weight:700;margin-top:1px">${icon}</span>
        <span>
          ${offTeam} <span style="color:var(--accent)">${i.off}</span>
          <span style="color:var(--text);font-weight:600"> (${offNames})</span>
          vs ${defTeam} <span style="color:var(--orange)">${i.def}</span>
          <span style="color:var(--text);font-weight:600"> (${defNames})</span>:
          <span style="color:${color}">${i.effect}</span>
        </span>
      </div>`;
    }).join("");
  }

  el.innerHTML = `
    <div class="sg-header">
      <div class="sg-matchup">
        <span class="sg-badge-home">${g.homeTeam} (${g.homeSeason})</span> vs
        <span class="sg-badge-away">${g.awayTeam} (${g.awaySeason})</span>
      </div>
      <div class="sg-time">Simulated at ${g.timestamp} | Score: ${homeScore}-${awayScore}</div>
    </div>

    <div class="sg-kpi-grid">
      <div class="sg-kpi"><div class="k">Home Win Probability</div><div class="v">${(g.homeWinProb * 100).toFixed(1)}%</div></div>
      <div class="sg-kpi"><div class="k">Expected Margin (Home)</div><div class="v">${g.deltaMu.toFixed(2)}</div></div>
      <div class="sg-kpi"><div class="k">Game Volatility</div><div class="v">${g.sigmaGame.toFixed(2)}</div></div>
      <div class="sg-kpi"><div class="k">Single-Draw Winner</div><div class="v ${sampledWinnerClass}">${g.winner}</div></div>
      <div class="sg-kpi"><div class="k">Sampled Margin</div><div class="v">${g.sampledMargin.toFixed(1)}</div></div>
      <div class="sg-kpi"><div class="k">Close Game?</div><div class="v">${closeGame ? '<span style="color:var(--orange)">YES</span>' : "No"}</div></div>
    </div>

    <div class="sg-detail-grid">
      <div class="sg-card">
        <h4>Margin Drivers</h4>
        <div class="sg-list">
          <div class="row"><span class="k">Team Strength Edge (mu)</span><span class="v">${numOrZero(drivers.talentEdge).toFixed(2)}</span></div>
          <div class="row"><span class="k">Home Court Edge</span><span class="v">+${numOrZero(drivers.homeCourtEdge).toFixed(2)}</span></div>
          <div class="row"><span class="k">Phase Blend Edge (Step 2)</span><span class="v">${phase.blend == null ? "-" : numOrZero(phase.blend).toFixed(2)}</span></div>
          <div class="row"><span class="k">Starter / Rotation / Clutch</span><span class="v">${phase.starter == null ? "-" : numOrZero(phase.starter).toFixed(2)} / ${phase.rotation == null ? "-" : numOrZero(phase.rotation).toFixed(2)} / ${phase.clutch == null ? "-" : numOrZero(phase.clutch).toFixed(2)}</span></div>
          <div class="row"><span class="k">Volatility Gap</span><span class="v">${numOrZero(drivers.volatilityGap).toFixed(2)}</span></div>
          <div class="row"><span class="k">Sampled Margin</span><span class="v">${g.sampledMargin.toFixed(2)}</span></div>
        </div>
      </div>

      <div class="sg-card">
        <h4>Head-to-Head Profile</h4>
        ${h2hHtml}
      </div>

      ${lineupPhasesHtml}

      <div class="sg-card">
        <h4>Archetype Interactions: ${g.homeTeam} OFF vs ${g.awayTeam} DEF</h4>
        ${renderInteractions(homeVsAwayInteractions, g.homeTeam, g.awayTeam)}
      </div>

      <div class="sg-card">
        <h4>Archetype Interactions: ${g.awayTeam} OFF vs ${g.homeTeam} DEF</h4>
        ${renderInteractions(awayVsHomeInteractions, g.awayTeam, g.homeTeam)}
      </div>

      <div class="sg-card">
        <h4>Top Player Contributors</h4>
        <div class="sg-contrib-grid">
          ${contribHome}
          ${contribAway}
        </div>
      </div>
    </div>

    ${renderPlayerStatsTable(g.homeTeam, "var(--accent)", homePlayerStats)}
    ${renderPlayerStatsTable(g.awayTeam, "var(--orange)", awayPlayerStats)}
  `;
}
</script>
</body>
</html>"""
    return html.replace("__DATA_JSON__", data_json)


if __name__ == "__main__":
    generate_html()
