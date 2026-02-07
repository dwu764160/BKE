"""
app/player_archetype_viewer.py
Interactive player archetype viewer with search and filtering.
Generates an HTML file that can be opened in any browser.
"""

import pandas as pd
import json
import os

DATA_DIR = "data/processed"
OUTPUT_FILE = "app/player_archetypes.html"


def load_archetypes():
    """Load offensive and defensive archetypes."""
    off = pd.read_parquet(f"{DATA_DIR}/player_archetypes.parquet")
    # Use v2 defensive archetypes (5 archetypes instead of 9)
    defense = pd.read_parquet(f"{DATA_DIR}/defensive_archetypes_v2.parquet")
    
    # Normalize IDs
    off['player_id'] = off['PLAYER_ID'].astype(str)
    defense['player_id'] = defense['PLAYER_ID'].astype(str)
    
    return off, defense


def get_offensive_reasons(row):
    """Generate human-readable reasons for offensive archetype."""
    reasons = []
    arch = row.get('primary_archetype', 'Unknown')
    sub = row.get('secondary_archetype', '')
    
    # Common metrics
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
        on_ball = row.get('ON_BALL_CREATION', 0)
        reasons.append(f"Ball dominance: {ball_dom*100:.0f}% (ISO {iso_pct*100:.0f}% + PnR BH {pnr_pct*100:.0f}%)")
        if post_pct > 0.05:
            reasons.append(f"Post-up: {post_pct*100:.0f}%")
        reasons.append(f"Playmaking: {playmaking:.1f} | AST/36: {ast36:.1f}")
        reasons.append(f"Scoring: {pts36:.1f} PTS/36 | Efficiency: {eff_tier}")
        if sub == 'Offensive Hub':
            reasons.append("Offensive hub — high playmaking, non-ISO/PnR creator")
        elif sub == 'Primary Scorer':
            reasons.append("Primary scorer — high volume on-ball creation")
        
    elif arch == 'Ballhandler':
        reasons.append(f"High playmaking ({playmaking:.1f}) with AST/36: {ast36:.1f}")
        reasons.append(f"Ball dominance: {ball_dom*100:.0f}%")
        reasons.append(f"Not a primary scorer ({pts36:.1f} PTS/36)")
        reasons.append("Facilitator role — distributes more than scores")
        
    elif arch == 'All-Around Scorer':
        reasons.append(f"Balanced 2PT/3PT mix (FG2A rate: {fg2a_rate*100:.0f}%)")
        reasons.append(f"Scoring: {pts36:.1f} PTS/36 | Efficiency: {eff_tier}")
        reasons.append(f"Ball dominance: {ball_dom*100:.0f}%")
        if sub == 'High Volume':
            reasons.append("High volume — top-tier scoring output")
            
    elif arch == 'Perimeter Scorer':
        fg3a = row.get('FG3A_PER36', 0)
        fg3_pct = row.get('FG3_PCT', 0) * 100 if row.get('FG3_PCT', 0) < 1 else row.get('FG3_PCT', 0)
        reasons.append(f"Perimeter-heavy: {fg3a:.1f} 3PA/36 (FG2A rate: {fg2a_rate*100:.0f}%)")
        reasons.append(f"3P%: {fg3_pct:.1f}% | Efficiency: {eff_tier}")
        reasons.append(f"Scoring: {pts36:.1f} PTS/36")
        
    elif arch == 'Interior Scorer':
        fg3a = row.get('FG3A_PER36', 0)
        reasons.append(f"Interior-heavy: FG2A rate {fg2a_rate*100:.0f}% (3PA/36: {fg3a:.1f})")
        reasons.append(f"Scoring: {pts36:.1f} PTS/36 | Efficiency: {eff_tier}")
        reasons.append(f"Ball dominance: {ball_dom*100:.0f}%")
        
    elif arch == 'Connector':
        reasons.append(f"Playmaker without volume scoring (AST/36: {ast36:.1f})")
        reasons.append(f"Scoring: {pts36:.1f} PTS/36")
        reasons.append(f"Ball dominance: {ball_dom*100:.0f}%")
        reasons.append("Connects teammates — distributes and facilitates")
        
    elif arch == 'Off-Ball Finisher':
        cut_pct = row.get('CUT_POSS_PCT', 0)
        pnrm_pct = row.get('PRROLLMAN_POSS_PCT', 0)
        trans_pct = row.get('TRANSITION_POSS_PCT', 0)
        reasons.append(f"Cuts: {cut_pct*100:.1f}%, PnR Roll: {pnrm_pct*100:.1f}%, Trans: {trans_pct*100:.1f}%")
        reasons.append(f"Low ball dominance ({ball_dom*100:.0f}%) — finisher role")
        reasons.append(f"Efficiency: {eff_tier}")
        
    elif arch == 'Off-Ball Movement Shooter':
        handoff = row.get('HANDOFF_POSS_PCT', 0)
        offscreen = row.get('OFFSCREEN_POSS_PCT', 0)
        reasons.append(f"Handoff: {handoff*100:.1f}%, Off-Screen: {offscreen*100:.1f}%")
        reasons.append("Movement-based shooting (more movement than spot-up)")
        reasons.append(f"Efficiency: {eff_tier}")
        
    elif arch == 'Off-Ball Stationary Shooter':
        spotup = row.get('SPOTUP_POSS_PCT', 0)
        catch_shoot = row.get('CATCH_SHOOT_FG3_PCT', 0) * 100
        reasons.append(f"Spot-up: {spotup*100:.1f}%")
        reasons.append(f"Catch & Shoot 3P%: {catch_shoot:.1f}%")
        reasons.append(f"Efficiency: {eff_tier}")
        
    elif arch == 'Rotation Piece':
        reasons.append("No dominant offensive role")
        reasons.append(f"Scoring: {pts36:.1f} PTS/36 | Ball dom: {ball_dom*100:.0f}%")
        reasons.append("Low usage, versatile role player")
        
    elif arch == 'Insufficient Minutes':
        mins = row.get('MIN', 0)
        reasons.append(f"Only {mins:.0f} minutes — not enough data")
    
    return reasons


def get_defensive_reasons(row):
    """Generate human-readable reasons for defensive archetype (5-archetype system)."""
    reasons = []
    arch = row.get('defensive_archetype', 'Unknown')
    secondary = row.get('defensive_secondary', '')
    
    # Key metrics from v2
    versatility = row.get('switch_score', 0)
    versatility_pctl = row.get('versatility_pctl', 0)
    difficulty_pctl = row.get('difficulty_pctl', 0)
    avg_opp_ppg = row.get('avg_opponent_ppg', 0)
    elite_pct = row.get('elite_matchup_pct', 0)
    blk_p36 = row.get('BLK_PER36', 0)
    stl_p36 = row.get('STL_PER36', 0)
    d_results = row.get('d_results_pctl', 0)
    
    if arch == 'POA Defender':
        reasons.append(f"Tough assignments (top {100*(1-difficulty_pctl):.0f}% difficulty)")
        reasons.append(f"Elite matchup %: {elite_pct*100:.1f}%")
        reasons.append(f"Avg opponent PPG: {avg_opp_ppg:.1f}")
        if secondary == 'Ball Hawk':
            reasons.append(f"High steals ({stl_p36:.1f}/36)")
        elif secondary == 'Switchable':
            reasons.append("Also versatile defender")
            
    elif arch == 'Switchable Defender':
        reasons.append(f"High versatility (top {100*(1-versatility_pctl):.0f}%)")
        reasons.append("Guards multiple positions effectively")
        if d_results >= 0.7:
            reasons.append("With strong results")
        if secondary == 'Rim Protector':
            reasons.append("Can also protect the rim")
        elif secondary == 'Lockdown':
            reasons.append("Elite defensive results")
            
    elif arch == 'Rim Protector':
        reasons.append(f"High blocks ({blk_p36:.1f} per 36)")
        reasons.append("Interior defense anchor")
        if secondary == 'Switchable':
            reasons.append("Can also switch on perimeter")
        elif secondary == 'Shot Blocker':
            reasons.append("Elite shot blocking")
            
    elif arch == 'Rotation Defender':
        reasons.append("Help/rotation defense focus")
        reasons.append(f"Medium assignment difficulty")
        if secondary == 'Active Hands':
            reasons.append(f"Good steals ({stl_p36:.1f}/36)")
        elif secondary == 'Solid':
            reasons.append("Solid defensive results")
        else:
            reasons.append("Team defense contributor")
            
    elif arch == 'Off-Ball Defender':
        reasons.append("Gets easier assignments")
        if secondary == 'Liability':
            reasons.append("Poor defensive results")
            reasons.append("Hidden on defense")
        else:
            reasons.append("Low versatility")
            reasons.append(f"Avg opponent PPG: {avg_opp_ppg:.1f}")
            
    elif arch == 'Insufficient Minutes':
        mins = row.get('MIN', 0)
        reasons.append(f"Only {mins:.0f} minutes")
        reasons.append("Not enough data for classification")
    
    return reasons


def get_playtype_rankings(row):
    """Return playtypes sorted by percentage (most -> least).
    Looks for common playtype columns and returns list of {name, pct}.
    """
    # Only the 11 real Synergy playtypes (not derived composites like BALL_DOMINANT_PCT)
    playtype_cols = [
        'ISOLATION_POSS_PCT', 'PRBALLHANDLER_POSS_PCT', 'POSTUP_POSS_PCT',
        'CUT_POSS_PCT', 'PRROLLMAN_POSS_PCT', 'HANDOFF_POSS_PCT',
        'OFFSCREEN_POSS_PCT', 'SPOTUP_POSS_PCT', 'TRANSITION_POSS_PCT',
        'OFFREBOUND_POSS_PCT', 'MISC_POSS_PCT'
    ]

    pairs = []
    for c in playtype_cols:
        if c in row.index:
            val = row.get(c, 0) or 0
            # Interpret <1 as fraction -> percent
            if val < 1:
                pct = val * 100
            else:
                pct = val
            if pct > 0:
                # Pretty name mapping
                pretty_names = {
                    'ISOLATION_POSS_PCT': 'Isolation',
                    'PRBALLHANDLER_POSS_PCT': 'PnR Ball Handler',
                    'POSTUP_POSS_PCT': 'Post Up',
                    'CUT_POSS_PCT': 'Cut',
                    'PRROLLMAN_POSS_PCT': 'PnR Roll Man',
                    'HANDOFF_POSS_PCT': 'Handoff',
                    'OFFSCREEN_POSS_PCT': 'Off Screen',
                    'SPOTUP_POSS_PCT': 'Spot Up',
                    'TRANSITION_POSS_PCT': 'Transition',
                    'OFFREBOUND_POSS_PCT': 'Putback',
                    'MISC_POSS_PCT': 'Misc',
                }
                name = pretty_names.get(c, c.replace('_POSS_PCT', '').replace('_', ' ').title())
                pairs.append((name, pct))

    pairs.sort(key=lambda x: x[1], reverse=True)
    return [{'type': p[0], 'pct': round(p[1], 2)} for p in pairs]


def _format_pct(val, scale=100):
    """Safely format a percentage value, handling NaN and None.
    If val is a 0-1 fraction (e.g. 0.22 = 22%), multiply by scale.
    Returns a number or None for missing data.
    """
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    result = val * scale if abs(val) < 1.5 else val  # already scaled if > 1.5
    return round(result, 1)


def generate_html(off_df, def_df):
    """Generate interactive HTML viewer."""
    
    # Merge offensive and defensive - use v2 columns
    def_cols = ['player_id', 'SEASON', 'defensive_archetype', 'defensive_secondary',
                'defensive_confidence', 'assignment_difficulty',
                'switch_score', 'versatility_pctl', 'avg_opponent_ppg', 
                'elite_matchup_pct', 'difficulty_pctl', 'BLK_PER36', 'STL_PER36',
                'd_results_pctl']
    
    # Filter to columns that exist
    def_cols = [c for c in def_cols if c in def_df.columns]
    
    merged = off_df.merge(
        def_df[def_cols],
        left_on=['player_id', 'SEASON'],
        right_on=['player_id', 'SEASON'],
        how='left'
    )
    
    # Build player data for JSON
    players = []
    for _, row in merged.iterrows():
        if row.get('primary_archetype') == 'Insufficient Minutes':
            continue  # Skip insufficient minutes players
            
        player = {
            'id': str(row['player_id']),
            'name': row['PLAYER_NAME'],
            'season': row['SEASON'],
            'team': row.get('TEAM_ABBREVIATION', ''),
            
            # Offensive
            'off_archetype': row.get('primary_archetype', 'Unknown'),
            'off_secondary': row.get('secondary_archetype', ''),
            'off_confidence': round(row.get('archetype_confidence', 0) * 100),
            'off_reasons': get_offensive_reasons(row),

            # Playtype breakdown (most -> least)
            'playtypes': get_playtype_rankings(row),
            
            # Defensive
            'def_archetype': row.get('defensive_archetype', 'Unknown'),
            'def_confidence': round(row.get('defensive_confidence', 0) * 100) if pd.notna(row.get('defensive_confidence')) else 0,
            'def_reasons': get_defensive_reasons(row),
            
            # Key stats
            'ppg': round(row.get('PPG', row.get('PTS', 0) / max(row.get('GP', 1), 1)), 1),
            'apg': round(row.get('AST', 0) / max(row.get('GP', 1), 1), 1),
            'rpg': round(row.get('REB', 0) / max(row.get('GP', 1), 1), 1),
            'usg': _format_pct(row.get('USG_PCT'), scale=100),  # stored as 0-1 fraction
            'ts': _format_pct(row.get('TS_PCT'), scale=100),    # stored as 0-1 fraction
            'mins': round(row.get('MIN', 0)),
            'mpg': round(row.get('MPG', 0), 1),
            'pts36': round(row.get('PTS_PER36', 0), 1),
            'eff_tier': row.get('efficiency_tier', ''),
        }
        players.append(player)
    
    # Sort by PPG
    players.sort(key=lambda x: x['ppg'], reverse=True)
    
    html = f'''<!DOCTYPE html>
<html>
<head>
    <title>NBA Player Archetype Viewer</title>
    <style>
        * {{ box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
        body {{ margin: 0; padding: 20px; background: #1a1a2e; color: #eee; }}
        h1 {{ color: #00d9ff; margin-bottom: 5px; }}
        .subtitle {{ color: #888; margin-bottom: 20px; }}
        
        .controls {{ display: flex; gap: 12px; margin-bottom: 20px; flex-wrap: wrap; align-items: center; }}
        input, select {{ padding: 10px 15px; border: 1px solid #333; border-radius: 8px; 
                        background: #16213e; color: #fff; font-size: 14px; }}
        input {{ width: 300px; }}
        select {{ min-width: 180px; }}
        
        .sort-btn {{ padding: 8px 14px; border: 1px solid #444; border-radius: 8px;
                    background: #16213e; color: #aaa; cursor: pointer; font-size: 13px; }}
        .sort-btn.active {{ border-color: #00d9ff; color: #00d9ff; }}
        
        .stats-bar {{ display: flex; gap: 20px; margin-bottom: 20px; color: #888; }}
        .stats-bar span {{ background: #16213e; padding: 8px 15px; border-radius: 6px; }}
        
        .player-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 15px; }}
        
        .player-card {{ background: #16213e; border-radius: 12px; padding: 18px; 
                       border: 1px solid #333; transition: transform 0.2s, border-color 0.2s; }}
        .player-card:hover {{ transform: translateY(-3px); border-color: #00d9ff; }}
        
        .player-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
        .player-name {{ font-size: 18px; font-weight: 600; color: #fff; }}
        .player-meta {{ color: #888; font-size: 13px; text-align: right; }}
        .player-team {{ color: #aaa; font-weight: 500; }}
        
        .stats-row {{ display: flex; gap: 12px; margin-bottom: 14px; flex-wrap: wrap; }}
        .stat {{ text-align: center; min-width: 42px; }}
        .stat-value {{ font-size: 15px; font-weight: 600; color: #00d9ff; }}
        .stat-label {{ color: #555; font-size: 10px; text-transform: uppercase; }}
        .stat-value.na {{ color: #555; }}
        
        .archetype-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px; }}
        .archetype-box {{ padding: 10px; border-radius: 8px; background: #0d1b3e; }}
        .archetype-label {{ font-size: 10px; color: #555; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 3px; }}
        .archetype-name {{ font-size: 14px; font-weight: 600; margin-bottom: 2px; }}
        .archetype-name.offense {{ color: #ff6b6b; }}
        .archetype-name.defense {{ color: #4ecdc4; }}
        .confidence {{ font-size: 11px; color: #666; }}
        .eff-badge {{ display: inline-block; padding: 1px 6px; border-radius: 4px; font-size: 10px; font-weight: 600; margin-left: 6px; }}
        .eff-badge.elite {{ background: #2d6a4f; color: #95d5b2; }}
        .eff-badge.high {{ background: #1b4332; color: #74c69d; }}
        .eff-badge.avg {{ background: #333; color: #888; }}
        .eff-badge.low {{ background: #6b2c2c; color: #e09898; }}
        
        .detail-toggle {{ font-size: 12px; color: #00a8e8; cursor: pointer; margin-top: 8px; user-select: none; }}
        .detail-toggle:hover {{ text-decoration: underline; }}
        .detail-section {{ display: none; margin-top: 10px; padding-top: 10px; border-top: 1px solid #222; }}
        .detail-section.open {{ display: block; }}
        
        .reasons {{ margin-top: 6px; }}
        .reason {{ font-size: 12px; color: #aaa; padding: 2px 0 2px 14px; position: relative; }}
        .reason::before {{ content: "•"; position: absolute; left: 0; color: #555; }}
        
        .playtype-section {{ margin-top: 10px; padding-top: 10px; border-top: 1px solid #222; }}
        .playtype-section h4 {{ margin: 0 0 6px 0; font-size: 11px; color: #555; text-transform: uppercase; letter-spacing: 1px; }}
        .playtype-bar-container {{ margin-bottom: 4px; }}
        .playtype-row {{ display: flex; align-items: center; gap: 8px; font-size: 12px; }}
        .playtype-name {{ width: 110px; color: #aaa; text-align: right; white-space: nowrap; }}
        .playtype-bar {{ flex: 1; height: 14px; background: #0a1428; border-radius: 3px; overflow: hidden; }}
        .playtype-fill {{ height: 100%; border-radius: 3px; min-width: 2px; }}
        .playtype-val {{ width: 42px; color: #888; font-size: 11px; }}
        
        .no-results {{ text-align: center; padding: 40px; color: #666; }}
    </style>
</head>
<body>
    <h1>🏀 NBA Player Archetype Viewer</h1>
    <p class="subtitle">Offensive & defensive archetypes with playtype breakdowns</p>
    
    <div class="controls">
        <input type="text" id="search" placeholder="Search player name..." oninput="filterPlayers()">
        <select id="offFilter" onchange="filterPlayers()">
            <option value="">All Offensive</option>
            <option value="Ball Dominant Creator">Ball Dominant Creator</option>
            <option value="Ballhandler">Ballhandler</option>
            <option value="All-Around Scorer">All-Around Scorer</option>
            <option value="Perimeter Scorer">Perimeter Scorer</option>
            <option value="Interior Scorer">Interior Scorer</option>
            <option value="Connector">Connector</option>
            <option value="Off-Ball Finisher">Off-Ball Finisher</option>
            <option value="Off-Ball Movement Shooter">Off-Ball Movement Shooter</option>
            <option value="Off-Ball Stationary Shooter">Off-Ball Stationary Shooter</option>
            <option value="Rotation Piece">Rotation Piece</option>
        </select>
        <select id="defFilter" onchange="filterPlayers()">
            <option value="">All Defensive</option>
        </select>
        <select id="seasonFilter" onchange="filterPlayers()">
            <option value="">All Seasons</option>
            <option value="2024-25">2024-25</option>
            <option value="2023-24">2023-24</option>
            <option value="2022-23">2022-23</option>
        </select>
        <span style="color:#555;">Sort:</span>
        <span class="sort-btn active" data-sort="ppg" onclick="setSort(this)">PPG</span>
        <span class="sort-btn" data-sort="off_confidence" onclick="setSort(this)">Confidence</span>
        <span class="sort-btn" data-sort="ts" onclick="setSort(this)">TS%</span>
        <span class="sort-btn" data-sort="usg" onclick="setSort(this)">USG%</span>
    </div>
    
    <div class="stats-bar">
        <span id="totalCount">Loading...</span>
    </div>
    
    <div id="playerGrid" class="player-grid"></div>
    
    <script>
        const players = {json.dumps(players)};
        let currentSort = 'ppg';
        
        // Populate defensive filter
        const defTypes = [...new Set(players.map(p => p.def_archetype).filter(a => a && a !== 'Unknown'))];
        const defSel = document.getElementById('defFilter');
        defTypes.sort().forEach(t => {{
            const opt = document.createElement('option');
            opt.value = t;
            opt.textContent = t;
            defSel.appendChild(opt);
        }});
        
        // Playtype color palette
        const ptColors = {{
            'Isolation': '#e63946', 'PnR Ball Handler': '#f4a261', 'Post Up': '#e76f51',
            'Cut': '#2a9d8f', 'PnR Roll Man': '#264653', 'Handoff': '#a8dadc',
            'Off Screen': '#457b9d', 'Spot Up': '#1d3557', 'Transition': '#f1faee',
            'Putback': '#606c38', 'Misc': '#555'
        }};
        
        function effBadge(tier) {{
            if (!tier) return '';
            const cls = tier === 'Elite' ? 'elite' : tier === 'High' ? 'high' : tier === 'Low' ? 'low' : 'avg';
            return `<span class="eff-badge ${{cls}}">${{tier}}</span>`;
        }}
        
        function fmtStat(v, suffix) {{
            if (v === null || v === undefined) return '<span class="na">—</span>';
            return v + (suffix || '');
        }}
        
        function renderPlayers(filtered) {{
            const grid = document.getElementById('playerGrid');
            document.getElementById('totalCount').textContent = `Showing ${{filtered.length}} of ${{players.length}} players`;
            
            if (filtered.length === 0) {{
                grid.innerHTML = '<div class="no-results">No players match your filters</div>';
                return;
            }}
            
            grid.innerHTML = filtered.map((p, idx) => {{
                // Playtype bars
                let ptHtml = '';
                if (p.playtypes && p.playtypes.length > 0) {{
                    const maxPct = Math.max(...p.playtypes.map(pt => pt.pct));
                    ptHtml = p.playtypes.map(pt => {{
                        const w = maxPct > 0 ? (pt.pct / maxPct * 100) : 0;
                        const col = ptColors[pt.type] || '#555';
                        return `<div class="playtype-bar-container"><div class="playtype-row">
                            <span class="playtype-name">${{pt.type}}</span>
                            <div class="playtype-bar"><div class="playtype-fill" style="width:${{w}}%;background:${{col}}"></div></div>
                            <span class="playtype-val">${{pt.pct}}%</span>
                        </div></div>`;
                    }}).join('');
                }}
                
                return `
                <div class="player-card">
                    <div class="player-header">
                        <div>
                            <span class="player-name">${{p.name}}</span>
                            ${{p.team ? '<span class="player-team"> · ' + p.team + '</span>' : ''}}
                        </div>
                        <div class="player-meta">${{p.season}}<br>${{p.mpg}} MPG</div>
                    </div>
                    
                    <div class="stats-row">
                        <div class="stat"><div class="stat-value">${{p.ppg}}</div><div class="stat-label">PPG</div></div>
                        <div class="stat"><div class="stat-value">${{p.apg}}</div><div class="stat-label">APG</div></div>
                        <div class="stat"><div class="stat-value">${{p.rpg}}</div><div class="stat-label">RPG</div></div>
                        <div class="stat"><div class="stat-value">${{fmtStat(p.usg, '%')}}</div><div class="stat-label">USG</div></div>
                        <div class="stat"><div class="stat-value">${{fmtStat(p.ts, '%')}}</div><div class="stat-label">TS%</div></div>
                        <div class="stat"><div class="stat-value">${{p.pts36}}</div><div class="stat-label">PTS/36</div></div>
                    </div>
                    
                    <div class="archetype-row">
                        <div class="archetype-box">
                            <div class="archetype-label">Offense</div>
                            <div class="archetype-name offense">${{p.off_archetype}}${{effBadge(p.eff_tier)}}</div>
                            <div class="confidence">${{p.off_confidence}}%${{p.off_secondary ? ' · ' + p.off_secondary : ''}}</div>
                        </div>
                        <div class="archetype-box">
                            <div class="archetype-label">Defense</div>
                            <div class="archetype-name defense">${{p.def_archetype || 'Unknown'}}</div>
                            <div class="confidence">${{p.def_confidence}}%</div>
                        </div>
                    </div>
                    
                    <div class="playtype-section">
                        <h4>Playtype Breakdown</h4>
                        ${{ptHtml || '<div style="color:#555;font-size:12px">No playtype data</div>'}}
                    </div>
                    
                    <div class="detail-toggle" onclick="this.nextElementSibling.classList.toggle('open'); this.textContent = this.nextElementSibling.classList.contains('open') ? '▾ Hide details' : '▸ Show details'">▸ Show details</div>
                    <div class="detail-section">
                        <div class="archetype-label" style="margin-bottom:4px;">Offensive Reasoning</div>
                        <div class="reasons">
                            ${{p.off_reasons.map(r => `<div class="reason">${{r}}</div>`).join('')}}
                        </div>
                        <div class="archetype-label" style="margin-top:10px;margin-bottom:4px;">Defensive Reasoning</div>
                        <div class="reasons">
                            ${{p.def_reasons.map(r => `<div class="reason">${{r}}</div>`).join('')}}
                        </div>
                    </div>
                </div>
            `;
            }}).join('');
        }}
        
        function setSort(el) {{
            document.querySelectorAll('.sort-btn').forEach(b => b.classList.remove('active'));
            el.classList.add('active');
            currentSort = el.dataset.sort;
            filterPlayers();
        }}
        
        function filterPlayers() {{
            const search = document.getElementById('search').value.toLowerCase();
            const offType = document.getElementById('offFilter').value;
            const defType = document.getElementById('defFilter').value;
            const season = document.getElementById('seasonFilter').value;
            
            let filtered = players.filter(p => {{
                if (search && !p.name.toLowerCase().includes(search)) return false;
                if (offType && p.off_archetype !== offType) return false;
                if (defType && p.def_archetype !== defType) return false;
                if (season && p.season !== season) return false;
                return true;
            }});
            
            // Sort
            filtered.sort((a, b) => {{
                const av = a[currentSort] ?? -999;
                const bv = b[currentSort] ?? -999;
                return bv - av;
            }});
            
            renderPlayers(filtered);
        }}
        
        // Initial render
        filterPlayers();
    </script>
</body>
</html>'''
    
    return html


def main():
    print("Loading archetype data...")
    off_df, def_df = load_archetypes()
    
    print(f"Offensive: {len(off_df)} players")
    print(f"Defensive: {len(def_df)} players")
    
    print("\nGenerating HTML viewer...")
    html = generate_html(off_df, def_df)
    
    os.makedirs("app", exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ Saved to {OUTPUT_FILE}")
    print(f"Open in browser: file://{os.path.abspath(OUTPUT_FILE)}")


if __name__ == "__main__":
    main()
