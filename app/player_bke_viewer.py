"""
app/player_bke_viewer.py
=============================================================================
Generate a lightweight standalone HTML viewer for interactive BKE exploration.

Features:
  - Lambda slider (0.00–0.15) to adjust production tilt in real-time
  - O/D weight split toggle (60/40 vs 55/45)
  - Sortable table with rank, BKE, OBKE, DBKE, production proxy, RAPM
  - Season filter
  - Search / filter by player name
  - Top-N bar chart and rank-shift distribution chart
  - All computation happens client-side from precomputed components JSON

Input:
  data/processed/bke/bke_v31_components.json

Output:
  app/player_bke_viewer.html

Usage:
  python3 app/player_bke_viewer.py
=============================================================================
"""

import json
import os
import sys

COMPONENTS_JSON = "data/processed/bke/bke_v31_components.json"
OUTPUT_HTML = "app/player_bke_viewer.html"


def generate_html(components_path: str = COMPONENTS_JSON, output_path: str = OUTPUT_HTML) -> str:
    if not os.path.exists(components_path):
        raise FileNotFoundError(f"Missing components JSON: {components_path}\nRun: python3 scripts/export_bke_components.py")

    with open(components_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Embed the data directly as a JS variable
    data_json = json.dumps(data, separators=(",", ":"))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BKE Interactive Explorer — v3.1</title>
<style>
:root {{
  --bg: #0d1117; --surface: #161b22; --border: #30363d;
  --text: #c9d1d9; --text-muted: #8b949e; --accent: #58a6ff;
  --green: #3fb950; --red: #f85149; --orange: #d29922; --purple: #bc8cff;
}}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:var(--bg); color:var(--text); font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif; }}
.container {{ max-width:1400px; margin:0 auto; padding:16px; }}
h1 {{ font-size:24px; font-weight:600; margin-bottom:4px; color:var(--accent); }}
.subtitle {{ color:var(--text-muted); font-size:13px; margin-bottom:16px; }}
.controls {{ display:flex; flex-wrap:wrap; gap:16px; align-items:center; background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:16px; margin-bottom:16px; }}
.control-group {{ display:flex; flex-direction:column; gap:4px; }}
.control-group label {{ font-size:12px; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.5px; }}
.control-group input[type=range] {{ width:240px; accent-color:var(--accent); }}
.control-group select, .control-group input[type=text] {{ background:var(--bg); color:var(--text); border:1px solid var(--border); border-radius:4px; padding:6px 10px; font-size:14px; }}
.lambda-display {{ font-size:22px; font-weight:700; color:var(--accent); min-width:60px; text-align:center; }}
.split-toggle {{ display:flex; gap:0; }}
.split-toggle button {{ background:var(--bg); color:var(--text-muted); border:1px solid var(--border); padding:6px 14px; font-size:13px; cursor:pointer; transition:all 0.15s; }}
.split-toggle button:first-child {{ border-radius:4px 0 0 4px; }}
.split-toggle button:last-child {{ border-radius:0 4px 4px 0; }}
.split-toggle button.active {{ background:var(--accent); color:#fff; border-color:var(--accent); font-weight:600; }}
.stats-bar {{ display:flex; gap:16px; flex-wrap:wrap; margin-bottom:16px; }}
.stat-card {{ background:var(--surface); border:1px solid var(--border); border-radius:6px; padding:12px 16px; min-width:140px; }}
.stat-card .stat-label {{ font-size:11px; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.5px; }}
.stat-card .stat-value {{ font-size:20px; font-weight:700; margin-top:2px; }}
.stat-card .stat-delta {{ font-size:12px; margin-top:2px; }}
.stat-delta.positive {{ color:var(--green); }}
.stat-delta.negative {{ color:var(--red); }}
.charts-row {{ display:flex; gap:16px; margin-bottom:16px; flex-wrap:wrap; }}
.chart-box {{ background:var(--surface); border:1px solid var(--border); border-radius:6px; padding:16px; flex:1; min-width:300px; }}
.chart-box h3 {{ font-size:14px; color:var(--text-muted); margin-bottom:8px; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; }}
thead {{ position:sticky; top:0; z-index:10; }}
th {{ background:var(--surface); color:var(--text-muted); text-align:left; padding:8px 10px; border-bottom:2px solid var(--border); cursor:pointer; user-select:none; white-space:nowrap; font-size:12px; text-transform:uppercase; letter-spacing:0.3px; }}
th:hover {{ color:var(--accent); }}
th.sorted-asc::after {{ content:" ▲"; color:var(--accent); }}
th.sorted-desc::after {{ content:" ▼"; color:var(--accent); }}
td {{ padding:6px 10px; border-bottom:1px solid var(--border); white-space:nowrap; }}
tr:hover {{ background:rgba(88,166,255,0.06); }}
.table-wrap {{ max-height:70vh; overflow-y:auto; border:1px solid var(--border); border-radius:6px; }}
.rank-col {{ font-weight:700; color:var(--accent); min-width:40px; text-align:right; }}
.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
.name-col {{ font-weight:600; max-width:180px; overflow:hidden; text-overflow:ellipsis; }}
.pos-col {{ color:var(--text-muted); }}
.bke-col {{ font-weight:700; }}
.bke-positive {{ color:var(--green); }}
.bke-negative {{ color:var(--red); }}
.prod-positive {{ color:var(--green); }}
.prod-negative {{ color:var(--orange); }}
.bar {{ height:14px; border-radius:2px; min-width:1px; }}
.bar-pos {{ background:var(--green); }}
.bar-neg {{ background:var(--red); }}
svg text {{ fill:var(--text); font-family:inherit; }}
.footer {{ margin-top:24px; padding:12px 0; border-top:1px solid var(--border); color:var(--text-muted); font-size:12px; text-align:center; }}
.highlight {{ background:rgba(88,166,255,0.12) !important; }}
</style>
</head>
<body>
<div class="container">
<h1>BKE Interactive Explorer</h1>
<p class="subtitle">v3.1 — Layer 3+6 baseline with production-tilt slider &middot; Generated: {data.get("generated_at", "")}</p>

<div class="controls">
  <div class="control-group">
    <label>Production Tilt (λ)</label>
    <div style="display:flex;align-items:center;gap:10px">
      <input type="range" id="lambdaSlider" min="0" max="0.15" step="0.005" value="0">
      <span class="lambda-display" id="lambdaVal">0.000</span>
    </div>
  </div>
  <div class="control-group">
    <label>O/D Split</label>
    <div class="split-toggle" id="splitToggle">
      <button class="active" data-split="60_40">60/40</button>
      <button data-split="55_45">55/45</button>
    </div>
  </div>
  <div class="control-group">
    <label>Season</label>
    <select id="seasonFilter"><option value="all">All Seasons</option></select>
  </div>
  <div class="control-group">
    <label>Search</label>
    <input type="text" id="searchBox" placeholder="Player name..." style="width:180px">
  </div>
  <div class="control-group">
    <label>Top N Chart</label>
    <select id="topNSelect">
      <option value="15">15</option>
      <option value="25" selected>25</option>
      <option value="50">50</option>
      <option value="100">100</option>
    </select>
  </div>
</div>

<div class="stats-bar" id="statsBar"></div>
<div class="charts-row">
  <div class="chart-box" style="flex:2"><h3>Top N Players — BKE</h3><div id="topChart"></div></div>
  <div class="chart-box" style="flex:1"><h3>Rank Shift Distribution (vs λ=0)</h3><div id="shiftChart"></div></div>
</div>
<div class="table-wrap"><table><thead id="thead"></thead><tbody id="tbody"></tbody></table></div>
<div class="footer">BKE v3.1 Interactive Explorer &middot; Formula: BKE = base + λ × prod_proxy_z &middot; Internal use only</div>
</div>

<script>
// === Embedded data ===
const DATA = {data_json};
const PLAYERS_SOURCE = Array.isArray(DATA.players) ? DATA.players : [];
const CONFIG = DATA.config;
const PROXY = DATA.production_proxy;

function buildRowKey(p, idx) {{
  const pid = String(p && p.player_id != null ? p.player_id : "").trim() || "UNK";
  const season = String(p && p.season != null ? p.season : "").trim() || "UNK";
  const team = String((p && (p.team_abbreviation || p.team)) || "UNK").trim() || "UNK";
  const stintRaw = p && (p.stint_number != null ? p.stint_number : p.team_stint_number);
  const stint = Number.isFinite(Number(stintRaw)) ? Math.max(1, Math.trunc(Number(stintRaw))) : 1;
  return `${{pid}}::${{season}}::${{team}}::${{stint}}::${{idx}}`;
}}

function buildTeamDisplay(p) {{
  const team = String((p && (p.team_abbreviation || p.team)) || "").trim();
  const stintRaw = p && (p.stint_number != null ? p.stint_number : p.team_stint_number);
  const stintCountRaw = p && (p.stint_count != null ? p.stint_count : p.stint_team_count);
  const stint = Number.isFinite(Number(stintRaw)) ? Math.max(1, Math.trunc(Number(stintRaw))) : 1;
  const stintCount = Number.isFinite(Number(stintCountRaw)) ? Math.max(1, Math.trunc(Number(stintCountRaw))) : 1;
  if (stintCount > 1 || stint > 1) {{
    return `${{team || 'UNK'}} (S${{stint}}/${{stintCount}})`;
  }}
  return team;
}}

const PLAYERS_RAW = PLAYERS_SOURCE.map((p, idx) => ({{
  ...p,
  __row_key: buildRowKey(p, idx),
  team_display: buildTeamDisplay(p),
}}));

// State
let currentSplit = "60_40";
let currentLambda = 0.0;
let currentSeason = "all";
let currentSearch = "";
let sortCol = "rank";
let sortAsc = true;
let topN = 25;

// Precompute base ranks (lambda=0) per season for shift calculation
let baseRanksMap = {{}};

function getBaseCol() {{ return "bke_base_" + currentSplit; }}

function computeBKE(p, lam) {{
  return p[getBaseCol()] + lam * p.prod_proxy_z;
}}

function buildBaseRanks() {{
  baseRanksMap = {{}};
  const groups = {{}};
  PLAYERS_RAW.forEach(p => {{
    const s = p.season;
    if (!groups[s]) groups[s] = [];
    groups[s].push({{ id: p.__row_key, bke: computeBKE(p, 0) }});
  }});
  for (const s in groups) {{
    groups[s].sort((a,b) => b.bke - a.bke);
    groups[s].forEach((x, i) => {{ baseRanksMap[x.id] = i + 1; }});
  }}
}}

function getFilteredPlayers() {{
  let list = PLAYERS_RAW;
  if (currentSeason !== "all") list = list.filter(p => p.season === currentSeason);
  if (currentSearch) {{
    const q = currentSearch.toLowerCase();
    list = list.filter(p => (p.player_name || "").toLowerCase().includes(q));
  }}
  // Compute BKE and rank per season
  const groups = {{}};
  list.forEach(p => {{
    const s = p.season;
    if (!groups[s]) groups[s] = [];
    const bke = computeBKE(p, currentLambda);
    groups[s].push({{ ...p, bke_final: bke }});
  }});
  let result = [];
  for (const s in groups) {{
    groups[s].sort((a,b) => b.bke_final - a.bke_final);
    groups[s].forEach((x, i) => {{
      x.rank = i + 1;
      const key = x.__row_key;
      x.base_rank = baseRanksMap[key] || null;
      x.rank_shift = x.base_rank ? (x.base_rank - x.rank) : 0;
      result.push(x);
    }});
  }}
  return result;
}}

function sortPlayers(list) {{
  const col = sortCol;
  const asc = sortAsc;
  list.sort((a, b) => {{
    let va = a[col], vb = b[col];
    if (va == null) va = asc ? Infinity : -Infinity;
    if (vb == null) vb = asc ? Infinity : -Infinity;
    if (typeof va === 'string') return asc ? va.localeCompare(vb) : vb.localeCompare(va);
    return asc ? va - vb : vb - va;
  }});
  return list;
}}

const COLUMNS = [
  {{ key:"rank", label:"#", cls:"rank-col num", fmt: v => v }},
  {{ key:"player_name", label:"Player", cls:"name-col", fmt: v => v || "?" }},
  {{ key:"season", label:"Season", cls:"pos-col", fmt: v => v }},
  {{ key:"position", label:"Pos", cls:"pos-col", fmt: v => v || "" }},
  {{ key:"team_display", label:"Team", cls:"pos-col", fmt: v => v || "" }},
  {{ key:"bke_final", label:"BKE", cls:"bke-col num", fmt: v => v != null ? v.toFixed(3) : "" }},
  {{ key:"obke", label:"OBKE", cls:"num", fmt: v => v != null ? v.toFixed(3) : "" }},
  {{ key:"dbke_l36", label:"DBKE", cls:"num", fmt: v => v != null ? v.toFixed(3) : "" }},
  {{ key:"prod_proxy_z", label:"Prod Proxy Z", cls:"num", fmt: v => v != null ? v.toFixed(3) : "" }},
  {{ key:"rank_shift", label:"Rank Δ", cls:"num", fmt: v => {{
    if (v == null || v === 0) return "—";
    return (v > 0 ? "+" : "") + v;
  }} }},
  {{ key:"rapm", label:"RAPM", cls:"num", fmt: v => v != null ? v.toFixed(2) : "" }},
  {{ key:"orapm", label:"ORAPM", cls:"num", fmt: v => v != null ? v.toFixed(2) : "" }},
  {{ key:"drapm", label:"DRAPM", cls:"num", fmt: v => v != null ? v.toFixed(2) : "" }},
  {{ key:"MIN", label:"MIN", cls:"num", fmt: v => v != null ? Math.round(v) : "" }},
  {{ key:"USG_RATE", label:"USG%", cls:"num", fmt: v => v != null ? (v * 100).toFixed(1) : "" }},
];

function renderHeader() {{
  const tr = document.createElement("tr");
  COLUMNS.forEach(col => {{
    const th = document.createElement("th");
    th.textContent = col.label;
    th.dataset.col = col.key;
    if (sortCol === col.key) th.className = sortAsc ? "sorted-asc" : "sorted-desc";
    th.onclick = () => {{
      if (sortCol === col.key) sortAsc = !sortAsc;
      else {{ sortCol = col.key; sortAsc = col.key === "rank"; }}
      refresh();
    }};
    tr.appendChild(th);
  }});
  document.getElementById("thead").innerHTML = "";
  document.getElementById("thead").appendChild(tr);
}}

function renderTable(players) {{
  const tbody = document.getElementById("tbody");
  const frag = document.createDocumentFragment();
  players.forEach(p => {{
    const tr = document.createElement("tr");
    COLUMNS.forEach(col => {{
      const td = document.createElement("td");
      td.className = col.cls || "";
      const val = p[col.key];
      td.textContent = col.fmt(val);
      // Color coding
      if (col.key === "bke_final") {{
        td.classList.add(val >= 0 ? "bke-positive" : "bke-negative");
      }}
      if (col.key === "prod_proxy_z") {{
        td.classList.add(val >= 0 ? "prod-positive" : "prod-negative");
      }}
      if (col.key === "rank_shift" && val !== 0 && val != null) {{
        td.style.color = val > 0 ? "var(--green)" : "var(--red)";
        td.style.fontWeight = "600";
      }}
      tr.appendChild(td);
    }});
    frag.appendChild(tr);
  }});
  tbody.innerHTML = "";
  tbody.appendChild(frag);
}}

function renderStats(players) {{
  const bar = document.getElementById("statsBar");
  const n = players.length;
  if (!n) {{ bar.innerHTML = ""; return; }}
  const bkeVals = players.map(p => p.bke_final);
  const mean = bkeVals.reduce((a,b) => a+b, 0) / n;
  const std = Math.sqrt(bkeVals.map(v => (v-mean)**2).reduce((a,b)=>a+b,0)/n);
  const shifts = players.map(p => Math.abs(p.rank_shift || 0));
  const meanShift = shifts.reduce((a,b)=>a+b,0)/n;
  const top100 = players.filter(p => p.rank <= 100);
  const lowProd = top100.filter(p => p.prod_proxy_z < -0.5).length;
  const highProd = top100.filter(p => p.prod_proxy_z > 0.5).length;

  const cards = [
    {{ label:"Players", value:n, delta:null }},
    {{ label:"Mean BKE", value:mean.toFixed(3), delta:null }},
    {{ label:"Std BKE", value:std.toFixed(3), delta:null }},
    {{ label:"Mean |Rank Shift|", value:meanShift.toFixed(1), delta:null }},
    {{ label:"Low-Prod Top-100", value:lowProd, delta:null }},
    {{ label:"High-Prod Top-100", value:highProd, delta:null }},
    {{ label:"Lambda", value:currentLambda.toFixed(3), delta:null }},
  ];
  bar.innerHTML = cards.map(c => `<div class="stat-card"><div class="stat-label">${{c.label}}</div><div class="stat-value">${{c.value}}</div></div>`).join("");
}}

function renderTopChart(players) {{
  const wrap = document.getElementById("topChart");
  const top = players.slice(0, topN);
  if (!top.length) {{ wrap.innerHTML = "<p style='color:var(--text-muted)'>No data</p>"; return; }}
  const maxVal = Math.max(...top.map(p => Math.abs(p.bke_final)), 0.001);
  const h = top.length * 22 + 10;
  const barW = 400;
  let svg = `<svg width="100%" viewBox="0 0 700 ${{h}}" style="max-height:${{Math.min(h, 600)}}px">`;
  top.forEach((p, i) => {{
    const y = i * 22 + 5;
    const w = Math.abs(p.bke_final) / maxVal * barW;
    const cls = p.bke_final >= 0 ? "bar-pos" : "bar-neg";
    const name = (p.player_name || "?").substring(0, 22);
    const shift = p.rank_shift;
    const shiftStr = shift === 0 ? "" : (shift > 0 ? ` (+${{shift}})` : ` (${{shift}})`);
    const shiftColor = shift > 0 ? "var(--green)" : shift < 0 ? "var(--red)" : "var(--text-muted)";
    svg += `<rect x="200" y="${{y}}" width="${{w}}" height="16" rx="2" class="${{cls}}" opacity="0.85"/>`;
    svg += `<text x="195" y="${{y+12}}" text-anchor="end" font-size="11">${{p.rank}}. ${{name}}</text>`;
    svg += `<text x="${{205+w}}" y="${{y+12}}" font-size="11" fill="var(--text-muted)">${{p.bke_final.toFixed(3)}}<tspan fill="${{shiftColor}}">${{shiftStr}}</tspan></text>`;
  }});
  svg += "</svg>";
  wrap.innerHTML = svg;
}}

function renderShiftChart(players) {{
  const wrap = document.getElementById("shiftChart");
  if (currentLambda === 0) {{ wrap.innerHTML = "<p style='color:var(--text-muted)'>Set λ > 0 to see rank shifts</p>"; return; }}
  // Histogram of rank shifts
  const shifts = players.map(p => p.rank_shift || 0);
  const bins = {{}};
  const binSize = 3;
  shifts.forEach(s => {{
    const b = Math.floor(s / binSize) * binSize;
    bins[b] = (bins[b] || 0) + 1;
  }});
  const keys = Object.keys(bins).map(Number).sort((a,b) => a - b);
  if (!keys.length) {{ wrap.innerHTML = ""; return; }}
  const maxCount = Math.max(...Object.values(bins));
  const barH = 160;
  const barW = Math.max(20, Math.min(40, 300 / keys.length));
  const w = keys.length * (barW + 4) + 60;
  let svg = `<svg width="100%" viewBox="0 0 ${{w}} ${{barH + 40}}">`;
  keys.forEach((k, i) => {{
    const x = 30 + i * (barW + 4);
    const h = (bins[k] / maxCount) * barH;
    const cls = k >= 0 ? "bar-pos" : "bar-neg";
    svg += `<rect x="${{x}}" y="${{barH - h}}" width="${{barW}}" height="${{h}}" rx="2" class="${{cls}}" opacity="0.75"/>`;
    svg += `<text x="${{x + barW/2}}" y="${{barH + 14}}" text-anchor="middle" font-size="10">${{k >= 0 ? "+":"" }}${{k}}</text>`;
    svg += `<text x="${{x + barW/2}}" y="${{barH - h - 3}}" text-anchor="middle" font-size="9" fill="var(--text-muted)">${{bins[k]}}</text>`;
  }});
  svg += "</svg>";
  wrap.innerHTML = svg;
}}

function refresh() {{
  buildBaseRanks();
  let players = getFilteredPlayers();
  players = sortPlayers(players);
  renderHeader();
  renderTable(players);
  renderStats(players);
  // For charts, use rank-sorted list
  const ranked = [...players].sort((a,b) => a.rank - b.rank);
  renderTopChart(ranked);
  renderShiftChart(ranked);
}}

// === Init ===
function init() {{
  // Populate season filter
  const seasons = [...new Set(PLAYERS_RAW.map(p => p.season))].sort();
  const sel = document.getElementById("seasonFilter");
  seasons.forEach(s => {{
    const opt = document.createElement("option");
    opt.value = s; opt.textContent = s;
    sel.appendChild(opt);
  }});

  // Lambda slider
  const slider = document.getElementById("lambdaSlider");
  const lambdaVal = document.getElementById("lambdaVal");
  let debounceTimer = null;
  slider.addEventListener("input", () => {{
    currentLambda = parseFloat(slider.value);
    lambdaVal.textContent = currentLambda.toFixed(3);
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(refresh, 80);
  }});

  // Split toggle
  document.querySelectorAll("#splitToggle button").forEach(btn => {{
    btn.addEventListener("click", () => {{
      document.querySelectorAll("#splitToggle button").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      currentSplit = btn.dataset.split;
      refresh();
    }});
  }});

  // Season filter
  sel.addEventListener("change", () => {{
    currentSeason = sel.value;
    refresh();
  }});

  // Search
  document.getElementById("searchBox").addEventListener("input", (e) => {{
    currentSearch = e.target.value;
    refresh();
  }});

  // Top N
  document.getElementById("topNSelect").addEventListener("change", (e) => {{
    topN = parseInt(e.target.value);
    refresh();
  }});

  refresh();
}}

document.addEventListener("DOMContentLoaded", init);
</script>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✅ BKE viewer written: {output_path}")
    print(f"   Open in browser: file://{os.path.abspath(output_path)}")
    return html


if __name__ == "__main__":
    generate_html()
