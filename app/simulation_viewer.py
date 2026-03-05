"""
app/simulation_viewer.py
=============================================================================
Generate a standalone interactive HTML viewer for Simulation Core results.

Features:
  - Season selector tabs
  - Team rankings table sorted by projected wins with confidence bands
  - Projected vs actual wins horizontal bar chart
  - Season-level stats cards (MAE, RMSE, correlation, Brier, accuracy)
  - Team detail expandable rows (distributions, probabilities)
  - Calibration chart (predicted vs actual win rates)
  - Color-coded prediction errors
  - Sortable columns
  - Dark theme matching existing BKE viewers

Input:
  reports/simulation_step1_season_results.json
  reports/simulation_step1_validation.json

Output:
  app/simulation.html

Usage:
  python3 app/simulation_viewer.py
=============================================================================
"""

import json
import os
import sys

import numpy as np

SEASON_RESULTS = "reports/simulation_step1_season_results.json"
VALIDATION_RESULTS = "reports/simulation_step1_validation.json"
OUTPUT_HTML = "app/simulation.html"


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

    # Build combined data blob
    data_blob = {
        "config": season_data.get("config", {}),
        "seasons": season_data.get("seasons", {}),
        "validation": validation_data,
    }

    data_json = json.dumps(data_blob, separators=(",", ":"))

    html = _build_html(data_json)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Generated: {output_path}")
    return output_path


def _build_html(data_json: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BKE Simulation Core — Season Projections</title>
<style>
:root {{
  --bg: #0d1117; --surface: #161b22; --border: #30363d;
  --text: #c9d1d9; --text-muted: #8b949e; --accent: #58a6ff;
  --green: #3fb950; --red: #f85149; --orange: #d29922; --purple: #bc8cff;
  --yellow: #e3b341;
}}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:var(--bg); color:var(--text); font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif; font-size:14px; }}
.container {{ max-width:1440px; margin:0 auto; padding:20px; }}
h1 {{ font-size:26px; font-weight:700; color:var(--accent); margin-bottom:2px; }}
.subtitle {{ color:var(--text-muted); font-size:13px; margin-bottom:20px; }}

/* Season Tabs */
.season-tabs {{ display:flex; gap:0; margin-bottom:20px; }}
.season-tab {{ padding:10px 24px; background:var(--surface); border:1px solid var(--border); color:var(--text-muted); cursor:pointer; font-size:14px; font-weight:600; transition:all 0.15s; user-select:none; }}
.season-tab:first-child {{ border-radius:8px 0 0 8px; }}
.season-tab:last-child {{ border-radius:0 8px 8px 0; }}
.season-tab.active {{ background:var(--accent); color:#fff; border-color:var(--accent); }}
.season-tab:hover:not(.active) {{ color:var(--text); background:rgba(88,166,255,0.1); }}

/* Stats Cards */
.stats-row {{ display:flex; gap:12px; flex-wrap:wrap; margin-bottom:20px; }}
.stat-card {{ background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:14px 18px; min-width:130px; flex:1; }}
.stat-card .label {{ font-size:11px; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.5px; }}
.stat-card .value {{ font-size:22px; font-weight:700; margin-top:4px; }}
.stat-card .detail {{ font-size:11px; color:var(--text-muted); margin-top:2px; }}
.val-green {{ color:var(--green); }}
.val-accent {{ color:var(--accent); }}
.val-orange {{ color:var(--orange); }}
.val-purple {{ color:var(--purple); }}

/* Charts Row */
.charts-row {{ display:flex; gap:16px; margin-bottom:20px; flex-wrap:wrap; }}
.chart-box {{ background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:16px; flex:1; min-width:320px; }}
.chart-box h3 {{ font-size:14px; color:var(--text-muted); margin-bottom:12px; font-weight:600; text-transform:uppercase; letter-spacing:0.3px; }}
svg text {{ fill:var(--text); font-family:inherit; }}

/* Single Game Sim */
.single-game-box {{ background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:16px; margin-bottom:20px; }}
.single-game-box h3 {{ font-size:14px; color:var(--text-muted); margin-bottom:12px; font-weight:600; text-transform:uppercase; letter-spacing:0.3px; }}
.sg-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:10px; margin-bottom:10px; }}
.sg-group label {{ display:block; font-size:11px; color:var(--text-muted); margin-bottom:4px; text-transform:uppercase; letter-spacing:0.3px; }}
.sg-group select {{ width:100%; background:var(--bg); color:var(--text); border:1px solid var(--border); border-radius:6px; padding:8px; font-size:13px; }}
.sg-actions {{ display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin-bottom:10px; }}
.sg-btn {{ background:var(--accent); color:#fff; border:0; border-radius:6px; padding:8px 12px; font-weight:700; cursor:pointer; }}
.sg-btn:hover {{ filter:brightness(1.05); }}
.sg-status {{ font-size:12px; color:var(--text-muted); }}
.sg-result {{ background:var(--bg); border:1px solid var(--border); border-radius:6px; padding:10px; font-size:13px; }}
.sg-result .row {{ display:flex; justify-content:space-between; gap:10px; margin-bottom:4px; }}
.sg-result .row:last-child {{ margin-bottom:0; }}

/* Table */
.table-wrap {{ border:1px solid var(--border); border-radius:8px; overflow:hidden; }}
.table-scroll {{ max-height:72vh; overflow-y:auto; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; }}
thead {{ position:sticky; top:0; z-index:10; }}
th {{ background:var(--surface); color:var(--text-muted); text-align:left; padding:10px 12px; border-bottom:2px solid var(--border); cursor:pointer; user-select:none; white-space:nowrap; font-size:11px; text-transform:uppercase; letter-spacing:0.4px; }}
th:hover {{ color:var(--accent); }}
th.sorted-asc::after {{ content:" ▲"; color:var(--accent); }}
th.sorted-desc::after {{ content:" ▼"; color:var(--accent); }}
td {{ padding:8px 12px; border-bottom:1px solid var(--border); white-space:nowrap; }}
tr:hover {{ background:rgba(88,166,255,0.06); }}
.rank {{ font-weight:700; color:var(--accent); text-align:right; width:36px; }}
.team {{ font-weight:700; font-size:14px; }}
.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
.err-good {{ color:var(--green); }}
.err-ok {{ color:var(--orange); }}
.err-bad {{ color:var(--red); }}
.conf-band {{ color:var(--text-muted); font-size:12px; }}
.prob-bar-wrap {{ width:80px; height:14px; background:var(--bg); border-radius:3px; display:inline-block; vertical-align:middle; overflow:hidden; }}
.prob-bar {{ height:100%; border-radius:3px; }}
.prob-playoff {{ background:var(--green); }}
.prob-50 {{ background:var(--accent); }}
.prob-60 {{ background:var(--purple); }}

/* Tooltip */
.tooltip {{ position:relative; cursor:help; }}
.tooltip .tip-text {{ visibility:hidden; background:var(--surface); border:1px solid var(--border); color:var(--text); padding:10px 14px; border-radius:6px; position:absolute; z-index:100; bottom:125%; left:50%; transform:translateX(-50%); width:260px; font-size:12px; line-height:1.5; box-shadow:0 4px 12px rgba(0,0,0,0.4); pointer-events:none; }}
.tooltip:hover .tip-text {{ visibility:visible; }}

/* Footer */
.footer {{ margin-top:24px; padding:16px; color:var(--text-muted); font-size:11px; text-align:center; border-top:1px solid var(--border); }}
</style>
</head>
<body>
<div class="container">
  <h1>BKE Simulation Core</h1>
  <div class="subtitle">Margin-Based Season Projections — 10,000 Monte Carlo Simulations per Season</div>

  <div class="season-tabs" id="seasonTabs"></div>
  <div class="stats-row" id="statsRow"></div>
  <div class="charts-row">
    <div class="chart-box" style="flex:2"><h3>Projected vs Actual Wins</h3><div id="winsChart"></div></div>
    <div class="chart-box" style="flex:1"><h3>Calibration</h3><div id="calChart"></div></div>
  </div>

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

  <div class="table-wrap"><div class="table-scroll" id="tableWrap"></div></div>
  <div class="footer">
    BKE Simulation Core v1 &mdash; Margin-based team-level simulation.
    &sigma;<sub>league</sub> = <span id="footSigma"></span>,
    HCA = <span id="footHCA"></span>,
    N = <span id="footN"></span>
  </div>
</div>

<script>
const DATA = {data_json};

let currentSeason = null;
let sortCol = "projected_rank";
let sortDir = "asc";
let currentSimulatedGame = null;

// ── Init ──
(function init() {{
  const seasons = Object.keys(DATA.seasons).sort();
  currentSeason = seasons[seasons.length - 1];

  // Footer
  const cfg = DATA.config || {{}};
  document.getElementById("footSigma").textContent = cfg.sigma_league || "?";
  document.getElementById("footHCA").textContent = cfg.home_court_advantage || "?";
  document.getElementById("footN").textContent = (cfg.n_simulations || 0).toLocaleString();

  // Season tabs
  const tabsEl = document.getElementById("seasonTabs");
  seasons.forEach(s => {{
    const tab = document.createElement("div");
    tab.className = "season-tab" + (s === currentSeason ? " active" : "");
    tab.textContent = s;
    tab.onclick = () => switchSeason(s);
    tabsEl.appendChild(tab);
  }});

  initSingleGameSimulator(seasons);

  render();
}})();

function switchSeason(s) {{
  currentSeason = s;
  document.querySelectorAll(".season-tab").forEach(t => {{
    t.classList.toggle("active", t.textContent === s);
  }});
  sortCol = "projected_rank";
  sortDir = "asc";
  render();
}}

function render() {{
  const sdata = DATA.seasons[currentSeason];
  if (!sdata) return;
  renderStats(sdata);
  renderTable(sdata);
  renderWinsChart(sdata);
  renderCalibration();
}}

// ── Stats Cards ──
function renderStats(sdata) {{
  const ss = sdata.season_stats || {{}};
  const vg = (DATA.validation && DATA.validation.game_level && DATA.validation.game_level[currentSeason]) || {{}};
  const vs = (DATA.validation && DATA.validation.season_level && DATA.validation.season_level.per_season && DATA.validation.season_level.per_season[currentSeason]) || {{}};

  const cards = [
    {{ label:"Correlation", value: fmt(ss.correlation, 3), cls:"val-green", detail:"Proj vs Actual wins" }},
    {{ label:"MAE", value: fmt(ss.mae, 2), cls:"val-accent", detail:"Avg win error" }},
    {{ label:"RMSE", value: fmt(ss.rmse, 2), cls:"val-accent", detail:"Root mean sq error" }},
    {{ label:"Brier Score", value: fmt(vg.brier_score, 4), cls:"val-orange", detail:"Game-level (0.25=naive)" }},
    {{ label:"Accuracy", value: vg.accuracy ? (vg.accuracy * 100).toFixed(1) + "%" : "—", cls:"val-green", detail:"Game pick accuracy" }},
    {{ label:"Log Loss", value: fmt(vg.log_loss, 4), cls:"val-purple", detail:"(0.693=coin flip)" }},
    {{ label:"Margin RMSE", value: fmt(vg.margin_rmse, 1), cls:"val-accent", detail:"Predicted vs actual margin" }},
    {{ label:"Teams", value: (sdata.team_results || []).length, cls:"val-accent", detail: currentSeason }},
  ];

  const el = document.getElementById("statsRow");
  el.innerHTML = cards.map(c => `
    <div class="stat-card">
      <div class="label">${{c.label}}</div>
      <div class="value ${{c.cls}}">${{c.value || "—"}}</div>
      <div class="detail">${{c.detail}}</div>
    </div>
  `).join("");
}}

// ── Table ──
function renderTable(sdata) {{
  let teams = [...(sdata.team_results || [])];

  // Sort
  teams.sort((a, b) => {{
    let va = a[sortCol], vb = b[sortCol];
    if (va == null) va = 0;
    if (vb == null) vb = 0;
    return sortDir === "asc" ? (va > vb ? 1 : va < vb ? -1 : 0) : (va < vb ? 1 : va > vb ? -1 : 0);
  }});

  const cols = [
    {{ key:"projected_rank", label:"Rank", cls:"rank" }},
    {{ key:"team", label:"Team", cls:"team" }},
    {{ key:"conference", label:"Conf", cls:"num" }},
    {{ key:"projected_conf_rank", label:"Conf Rk", cls:"num", fmt:v=>v!=null?Number(v).toFixed(1):"—" }},
    {{ key:"mu", label:"Net Rtg", cls:"num", fmt:v=>v!=null?v.toFixed(2):"—" }},
    {{ key:"sigma", label:"Vol", cls:"num", fmt:v=>v!=null?v.toFixed(2):"—" }},
    {{ key:"projected_wins", label:"Proj W", cls:"num", fmt:v=>v!=null?v.toFixed(1):"—" }},
    {{ key:"win_p5", label:"90% CI", cls:"conf-band", fmt:(v,r)=>`${{r.win_p5}}-${{r.win_p95}}` }},
    {{ key:"actual_wins", label:"Actual W", cls:"num", fmt:v=>v!=null?v:"—" }},
    {{ key:"win_error", label:"Error", cls:"num", fmt:(v,r) => {{
      if (v == null) return "—";
      const abs = Math.abs(v);
      const sign = v > 0 ? "+" : "";
      const cls = abs <= 3 ? "err-good" : abs <= 7 ? "err-ok" : "err-bad";
      return `<span class="${{cls}}">${{sign}}${{v.toFixed(1)}}</span>`;
    }} }},
    {{ key:"direct_playoff_probability", label:"Top 6 %", cls:"num", fmt:v=>v!=null?(v*100).toFixed(0)+"%":"—" }},
    {{ key:"top_10_probability", label:"Top 10 %", cls:"num", fmt:v=>v!=null?(v*100).toFixed(0)+"%":"—" }},
    {{ key:"playin_only_probability", label:"7-10 %", cls:"num", fmt:v=>v!=null?(v*100).toFixed(0)+"%":"—" }},
    {{ key:"win_std", label:"Win SD", cls:"num", fmt:v=>v!=null?v.toFixed(2):"—" }},
  ];

  let html = "<table><thead><tr>";
  cols.forEach(c => {{
    const cls = c.key === sortCol ? (sortDir === "asc" ? "sorted-asc" : "sorted-desc") : "";
    html += `<th class="${{cls}}" data-col="${{c.key}}">${{c.label}}</th>`;
  }});
  html += "</tr></thead><tbody>";

  teams.forEach(t => {{
    html += "<tr>";
    cols.forEach(c => {{
      const raw = c.key === "win_p5" && c.label === "90% CI" ? t["win_p5"] : t[c.key];
      const display = c.fmt ? c.fmt(raw, t) : (raw != null ? raw : "—");
      html += `<td class="${{c.cls || ""}}">${{display}}</td>`;
    }});
    html += "</tr>";
  }});

  html += "</tbody></table>";
  document.getElementById("tableWrap").innerHTML = html;

  // Header click → sort
  document.querySelectorAll("#tableWrap th").forEach(th => {{
    th.addEventListener("click", () => {{
      const col = th.dataset.col;
      if (sortCol === col) {{
        sortDir = sortDir === "asc" ? "desc" : "asc";
      }} else {{
        sortCol = col;
        sortDir = col === "team" ? "asc" : "desc";
      }}
      renderTable(sdata);
    }});
  }});
}}

// ── Wins Chart (Horizontal Bars) ──
function renderWinsChart(sdata) {{
  const teams = [...(sdata.team_results || [])].sort((a,b) => (b.projected_wins||0) - (a.projected_wins||0));
  const n = teams.length;
  const barH = 18, gap = 4, leftMargin = 50, rightMargin = 80;
  const h = n * (barH + gap) + 40;
  const chartW = 700;
  const maxWins = Math.max(82, ...teams.map(t => Math.max(t.projected_wins || 0, t.actual_wins || 0, t.win_p95 || 0)));

  const xScale = (chartW - leftMargin - rightMargin) / maxWins;

  let svg = `<svg width="100%" viewBox="0 0 ${{chartW}} ${{h}}" style="max-height:${{Math.min(h, 900)}}px">`;

  // 41-win reference line (.500 record)
  const x41 = leftMargin + 41 * xScale;
  svg += `<line x1="${{x41}}" y1="0" x2="${{x41}}" y2="${{h}}" stroke="var(--text-muted)" stroke-dasharray="4,3" opacity="0.4"/>`;
  svg += `<text x="${{x41}}" y="12" text-anchor="middle" font-size="10" fill="var(--text-muted)">41 Wins</text>`;

  teams.forEach((t, i) => {{
    const y = 20 + i * (barH + gap);
    const projW = leftMargin + (t.projected_wins || 0) * xScale;
    const actW = t.actual_wins != null ? leftMargin + t.actual_wins * xScale : null;
    const p5x = leftMargin + (t.win_p5 || 0) * xScale;
    const p95x = leftMargin + (t.win_p95 || 0) * xScale;

    // Confidence band (p5-p95) — light background
    svg += `<rect x="${{p5x}}" y="${{y+2}}" width="${{p95x - p5x}}" height="${{barH-4}}" rx="2" fill="var(--accent)" opacity="0.12"/>`;

    // Projected wins bar
    svg += `<rect x="${{leftMargin}}" y="${{y+4}}" width="${{projW - leftMargin}}" height="${{barH-8}}" rx="2" fill="var(--accent)" opacity="0.7"/>`;

    // Actual wins marker
    if (actW != null) {{
      svg += `<line x1="${{actW}}" y1="${{y+1}}" x2="${{actW}}" y2="${{y+barH-1}}" stroke="var(--green)" stroke-width="2.5"/>`;
    }}

    // Team label
    svg += `<text x="${{leftMargin - 4}}" y="${{y + barH/2 + 4}}" text-anchor="end" font-size="11" font-weight="600">${{t.team}}</text>`;

    // Values
    const errStr = t.win_error != null ? ` (${{t.win_error > 0 ? "+" : ""}}${{t.win_error.toFixed(1)}})` : "";
    svg += `<text x="${{Math.max(projW, actW || 0) + 6}}" y="${{y + barH/2 + 4}}" font-size="10" fill="var(--text-muted)">${{(t.projected_wins||0).toFixed(0)}}P / ${{t.actual_wins != null ? t.actual_wins : "?"}}A${{errStr}}</text>`;
  }});

  // Legend
  svg += `<rect x="${{chartW - 200}}" y="${{h - 18}}" width="12" height="6" rx="1" fill="var(--accent)" opacity="0.7"/>`;
  svg += `<text x="${{chartW - 184}}" y="${{h - 12}}" font-size="10" fill="var(--text-muted)">Projected</text>`;
  svg += `<line x1="${{chartW - 120}}" y1="${{h - 18}}" x2="${{chartW - 120}}" y2="${{h - 12}}" stroke="var(--green)" stroke-width="2.5"/>`;
  svg += `<text x="${{chartW - 114}}" y="${{h - 12}}" font-size="10" fill="var(--text-muted)">Actual</text>`;
  svg += `<rect x="${{chartW - 80}}" y="${{h - 18}}" width="20" height="6" rx="1" fill="var(--accent)" opacity="0.12"/>`;
  svg += `<text x="${{chartW - 56}}" y="${{h - 12}}" font-size="10" fill="var(--text-muted)">90% CI</text>`;

  svg += "</svg>";
  document.getElementById("winsChart").innerHTML = svg;
}}

// ── Calibration Chart ──
function renderCalibration() {{
  const calData = DATA.validation && DATA.validation.calibration && DATA.validation.calibration[currentSeason];
  if (!calData || calData.length === 0) {{
    document.getElementById("calChart").innerHTML = '<div style="color:var(--text-muted);padding:20px">No calibration data</div>';
    return;
  }}

  const w = 320, h = 280, pad = 40;
  const plotW = w - 2 * pad, plotH = h - 2 * pad;

  let svg = `<svg width="100%" viewBox="0 0 ${{w}} ${{h}}">`;

  // Perfect calibration line
  svg += `<line x1="${{pad}}" y1="${{pad + plotH}}" x2="${{pad + plotW}}" y2="${{pad}}" stroke="var(--text-muted)" stroke-dasharray="4,3" opacity="0.5"/>`;

  // Axes
  svg += `<line x1="${{pad}}" y1="${{pad + plotH}}" x2="${{pad + plotW}}" y2="${{pad + plotH}}" stroke="var(--border)" stroke-width="1"/>`;
  svg += `<line x1="${{pad}}" y1="${{pad}}" x2="${{pad}}" y2="${{pad + plotH}}" stroke="var(--border)" stroke-width="1"/>`;

  // Labels
  svg += `<text x="${{w/2}}" y="${{h - 4}}" text-anchor="middle" font-size="10" fill="var(--text-muted)">Predicted Win Prob</text>`;
  svg += `<text x="10" y="${{h/2}}" text-anchor="middle" font-size="10" fill="var(--text-muted)" transform="rotate(-90,10,${{h/2}})">Actual Win Rate</text>`;

  // Tick labels
  for (let i = 0; i <= 10; i += 2) {{
    const v = i / 10;
    const x = pad + v * plotW;
    const y = pad + plotH - v * plotH;
    svg += `<text x="${{x}}" y="${{pad + plotH + 14}}" text-anchor="middle" font-size="9" fill="var(--text-muted)">${{v.toFixed(1)}}</text>`;
    svg += `<text x="${{pad - 6}}" y="${{y + 3}}" text-anchor="end" font-size="9" fill="var(--text-muted)">${{v.toFixed(1)}}</text>`;
  }}

  // Data points
  calData.forEach(bin => {{
    const px = pad + bin.predicted_rate * plotW;
    const py = pad + plotH - bin.actual_rate * plotH;
    const r = Math.max(3, Math.min(8, Math.sqrt(bin.count) * 0.8));
    svg += `<circle cx="${{px}}" cy="${{py}}" r="${{r}}" fill="var(--accent)" opacity="0.85"/>`;
  }});

  // Connect points with line
  let path = "";
  calData.forEach((bin, i) => {{
    const px = pad + bin.predicted_rate * plotW;
    const py = pad + plotH - bin.actual_rate * plotH;
    path += (i === 0 ? "M" : "L") + `${{px}},${{py}}`;
  }});
  svg += `<path d="${{path}}" fill="none" stroke="var(--accent)" stroke-width="1.5" opacity="0.6"/>`;

  svg += "</svg>";
  document.getElementById("calChart").innerHTML = svg;
}}

// ── Single-Game Simulator ──
function initSingleGameSimulator(seasons) {{
  const homeSeasonEl = document.getElementById("sgHomeSeason");
  const awaySeasonEl = document.getElementById("sgAwaySeason");
  homeSeasonEl.innerHTML = "";
  awaySeasonEl.innerHTML = "";

  seasons.forEach(s => {{
    const o1 = document.createElement("option");
    o1.value = s;
    o1.textContent = s;
    homeSeasonEl.appendChild(o1);

    const o2 = document.createElement("option");
    o2.value = s;
    o2.textContent = s;
    awaySeasonEl.appendChild(o2);
  }});

  homeSeasonEl.value = currentSeason;
  awaySeasonEl.value = currentSeason;

  populateTeamOptions(homeSeasonEl.value, "sgHomeTeam");
  populateTeamOptions(awaySeasonEl.value, "sgAwayTeam");

  homeSeasonEl.addEventListener("change", () => {{
    populateTeamOptions(homeSeasonEl.value, "sgHomeTeam");
  }});
  awaySeasonEl.addEventListener("change", () => {{
    populateTeamOptions(awaySeasonEl.value, "sgAwayTeam");
  }});

  document.getElementById("sgRunBtn").addEventListener("click", runSingleGameSimulation);
}}

function populateTeamOptions(season, teamSelectId) {{
  const el = document.getElementById(teamSelectId);
  const teams = ((DATA.seasons[season] || {{}}).team_results || [])
    .slice()
    .sort((a, b) => (a.team || "").localeCompare(b.team || ""));
  el.innerHTML = teams.map(t => `<option value="${{t.team}}">${{t.team}}</option>`).join("");
}}

function getTeamEntry(season, teamAbbr) {{
  const rows = ((DATA.seasons[season] || {{}}).team_results || []);
  for (const row of rows) {{
    if (row.team === teamAbbr) return row;
  }}
  return null;
}}

function erf(x) {{
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
}}

function normalCdf(x) {{
  return 0.5 * (1 + erf(x / Math.sqrt(2)));
}}

function randn() {{
  let u = 0;
  let v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}}

function runSingleGameSimulation() {{
  const homeSeason = document.getElementById("sgHomeSeason").value;
  const awaySeason = document.getElementById("sgAwaySeason").value;
  const homeTeam = document.getElementById("sgHomeTeam").value;
  const awayTeam = document.getElementById("sgAwayTeam").value;

  const home = getTeamEntry(homeSeason, homeTeam);
  const away = getTeamEntry(awaySeason, awayTeam);

  if (!home || !away) {{
    document.getElementById("sgResult").innerHTML = "Unable to find one or both teams in the selected seasons.";
    return;
  }}

  const cfg = DATA.config || {{}};
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

  currentSimulatedGame = {{
    homeSeason,
    awaySeason,
    homeTeam,
    awayTeam,
    homeMu,
    awayMu,
    homeSigma,
    awaySigma,
    deltaMu,
    sigmaGame,
    homeWinProb,
    awayWinProb: 1 - homeWinProb,
    sampledMargin,
    winner: sampledMargin >= 0 ? `${{homeTeam}} (${{homeSeason}})` : `${{awayTeam}} (${{awaySeason}})`,
    timestamp: new Date().toLocaleString(),
  }};

  renderSingleGameResult();
}}

function renderSingleGameResult() {{
  const el = document.getElementById("sgResult");
  if (!currentSimulatedGame) {{
    el.innerHTML = "No game simulated yet.";
    return;
  }}

  const g = currentSimulatedGame;
  el.innerHTML = `
    <div class="row"><span>Matchup</span><strong>${{g.homeTeam}} (${{g.homeSeason}}) vs ${{g.awayTeam}} (${{g.awaySeason}})</strong></div>
    <div class="row"><span>Expected Margin (home)</span><strong>${{g.deltaMu.toFixed(2)}}</strong></div>
    <div class="row"><span>Game Sigma</span><strong>${{g.sigmaGame.toFixed(2)}}</strong></div>
    <div class="row"><span>Home Win Probability</span><strong>${{(g.homeWinProb * 100).toFixed(1)}}%</strong></div>
    <div class="row"><span>Single Draw Margin (home)</span><strong>${{g.sampledMargin.toFixed(2)}}</strong></div>
    <div class="row"><span>Winner (this run)</span><strong>${{g.winner}}</strong></div>
    <div class="row"><span>Simulated At</span><span>${{g.timestamp}}</span></div>
  `;
}}

// ── Util ──
function fmt(v, dec) {{
  if (v == null || v === undefined) return "—";
  return Number(v).toFixed(dec);
}}

</script>
</body>
</html>"""


if __name__ == "__main__":
    generate_html()
