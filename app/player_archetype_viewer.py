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
    
    # Ball dominance info
    ball_dom = row.get('BALL_DOMINANT_PCT', 0)
    playmaking = row.get('PLAYMAKING_SCORE', 0)
    
    if arch == 'Ball Dominant Creator':
        iso_pct = row.get('ISOLATION_POSS_PCT', 0)
        pnr_pct = row.get('PRBALLHANDLER_POSS_PCT', 0)
        reasons.append(f"High ball dominance ({ball_dom:.0f}%)")
        reasons.append(f"ISO: {iso_pct:.1f}%, PnR BH: {pnr_pct:.1f}%")
        reasons.append(f"High playmaking score ({playmaking:.1f})")
        
    elif arch == 'Perimeter Scorer':
        fg3a = row.get('FG3A_PER36', 0)
        fg3_pct = row.get('FG3_PCT', 0) * 100 if row.get('FG3_PCT', 0) < 1 else row.get('FG3_PCT', 0)
        reasons.append(f"High 3PA ({fg3a:.1f} per 36)")
        reasons.append(f"3P%: {fg3_pct:.1f}%")
        reasons.append(f"Ball dominance: {ball_dom:.0f}%")
        
    elif arch == 'Interior Scorer':
        drives = row.get('DRIVES_PER36', 0)
        interior = row.get('INTERIOR_RATIO', 0)
        reasons.append(f"High interior ratio ({interior:.2f})")
        reasons.append(f"Drives: {drives:.1f} per 36")
        reasons.append(f"Low 3PT volume")
        
    elif arch == 'Off-Ball Finisher':
        cut_pct = row.get('CUT_POSS_PCT', 0)
        pnrm_pct = row.get('PRROLLMAN_POSS_PCT', 0)
        trans_pct = row.get('TRANSITION_POSS_PCT', 0)
        reasons.append(f"Cuts: {cut_pct:.1f}%, PnR Roll: {pnrm_pct:.1f}%")
        reasons.append(f"Transition: {trans_pct:.1f}%")
        reasons.append("Low ball dominance - finisher role")
        
    elif arch == 'Off-Ball Movement Shooter':
        handoff = row.get('HANDOFF_POSS_PCT', 0)
        offscreen = row.get('OFFSCREEN_POSS_PCT', 0)
        reasons.append(f"Handoff: {handoff:.1f}%, Off-Screen: {offscreen:.1f}%")
        reasons.append("Movement-based shooting")
        
    elif arch == 'Off-Ball Stationary Shooter':
        spotup = row.get('SPOTUP_POSS_PCT', 0)
        catch_shoot = row.get('CATCH_SHOOT_FG3_PCT', 0) * 100
        reasons.append(f"Spot-up: {spotup:.1f}%")
        reasons.append(f"Catch & Shoot 3P%: {catch_shoot:.1f}%")
        
    elif arch == 'Rotation Piece':
        reasons.append("Balanced skill set")
        reasons.append("Low usage rate")
        reasons.append("Versatile offensive role")
        
    elif arch == 'Insufficient Minutes':
        mins = row.get('MIN', 0)
        reasons.append(f"Only {mins:.0f} minutes")
        reasons.append("Not enough data for classification")
    
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
            
            # Defensive
            'def_archetype': row.get('defensive_archetype', 'Unknown'),
            'def_confidence': round(row.get('defensive_confidence', 0) * 100) if pd.notna(row.get('defensive_confidence')) else 0,
            'def_reasons': get_defensive_reasons(row),
            
            # Key stats
            'ppg': round(row.get('PPG', row.get('PTS', 0) / max(row.get('GP', 1), 1)), 1),
            'apg': round(row.get('AST', 0) / max(row.get('GP', 1), 1), 1),
            'rpg': round(row.get('REB', 0) / max(row.get('GP', 1), 1), 1),
            'usg': round(row.get('USG_PCT', 0), 1),
            'ts': round(row.get('TS_PCT', 0) * 100 if row.get('TS_PCT', 0) < 1 else row.get('TS_PCT', 0), 1),
            'mins': round(row.get('MIN', 0)),
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
        
        .controls {{ display: flex; gap: 15px; margin-bottom: 20px; flex-wrap: wrap; }}
        input, select {{ padding: 10px 15px; border: 1px solid #333; border-radius: 8px; 
                        background: #16213e; color: #fff; font-size: 14px; }}
        input {{ width: 300px; }}
        select {{ min-width: 200px; }}
        
        .stats {{ display: flex; gap: 20px; margin-bottom: 20px; color: #888; }}
        .stats span {{ background: #16213e; padding: 8px 15px; border-radius: 6px; }}
        
        .player-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(380px, 1fr)); gap: 15px; }}
        
        .player-card {{ background: #16213e; border-radius: 12px; padding: 18px; 
                       border: 1px solid #333; transition: transform 0.2s, border-color 0.2s; }}
        .player-card:hover {{ transform: translateY(-3px); border-color: #00d9ff; }}
        
        .player-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }}
        .player-name {{ font-size: 18px; font-weight: 600; color: #fff; }}
        .player-season {{ color: #888; font-size: 13px; }}
        
        .stats-row {{ display: flex; gap: 15px; margin-bottom: 15px; font-size: 13px; }}
        .stat {{ text-align: center; }}
        .stat-value {{ font-size: 16px; font-weight: 600; color: #00d9ff; }}
        .stat-label {{ color: #666; font-size: 11px; }}
        
        .archetype-section {{ margin-top: 12px; padding-top: 12px; border-top: 1px solid #333; }}
        .archetype-label {{ font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }}
        .archetype-name {{ font-size: 15px; font-weight: 600; margin-bottom: 4px; }}
        .archetype-name.offense {{ color: #ff6b6b; }}
        .archetype-name.defense {{ color: #4ecdc4; }}
        .confidence {{ font-size: 11px; color: #888; }}
        
        .reasons {{ margin-top: 8px; }}
        .reason {{ font-size: 12px; color: #aaa; padding: 3px 0; padding-left: 15px; 
                  position: relative; }}
        .reason::before {{ content: "•"; position: absolute; left: 0; color: #666; }}
        
        .no-results {{ text-align: center; padding: 40px; color: #666; }}
    </style>
</head>
<body>
    <h1>🏀 NBA Player Archetype Viewer</h1>
    <p class="subtitle">Search and explore player offensive & defensive archetypes</p>
    
    <div class="controls">
        <input type="text" id="search" placeholder="Search player name..." oninput="filterPlayers()">
        <select id="offFilter" onchange="filterPlayers()">
            <option value="">All Offensive Archetypes</option>
            <option value="Ball Dominant Creator">Ball Dominant Creator</option>
            <option value="Perimeter Scorer">Perimeter Scorer</option>
            <option value="Interior Scorer">Interior Scorer</option>
            <option value="Off-Ball Finisher">Off-Ball Finisher</option>
            <option value="Off-Ball Movement Shooter">Off-Ball Movement Shooter</option>
            <option value="Off-Ball Stationary Shooter">Off-Ball Stationary Shooter</option>
            <option value="Rotation Piece">Rotation Piece</option>
        </select>
        <select id="defFilter" onchange="filterPlayers()">
            <option value="">All Defensive Archetypes</option>
        </select>
        <select id="seasonFilter" onchange="filterPlayers()">
            <option value="">All Seasons</option>
            <option value="2024-25">2024-25</option>
            <option value="2023-24">2023-24</option>
            <option value="2022-23">2022-23</option>
        </select>
    </div>
    
    <div class="stats">
        <span id="totalCount">Loading...</span>
    </div>
    
    <div id="playerGrid" class="player-grid"></div>
    
    <script>
        const players = {json.dumps(players)};
        
        // Populate defensive filter
        const defTypes = [...new Set(players.map(p => p.def_archetype).filter(a => a && a !== 'Unknown'))];
        const defFilter = document.getElementById('defFilter');
        defTypes.sort().forEach(t => {{
            const opt = document.createElement('option');
            opt.value = t;
            opt.textContent = t;
            defFilter.appendChild(opt);
        }});
        
        function renderPlayers(filtered) {{
            const grid = document.getElementById('playerGrid');
            document.getElementById('totalCount').textContent = `Showing ${{filtered.length}} of ${{players.length}} players`;
            
            if (filtered.length === 0) {{
                grid.innerHTML = '<div class="no-results">No players match your filters</div>';
                return;
            }}
            
            grid.innerHTML = filtered.map(p => `
                <div class="player-card">
                    <div class="player-header">
                        <span class="player-name">${{p.name}}</span>
                        <span class="player-season">${{p.season}}</span>
                    </div>
                    
                    <div class="stats-row">
                        <div class="stat"><div class="stat-value">${{p.ppg}}</div><div class="stat-label">PPG</div></div>
                        <div class="stat"><div class="stat-value">${{p.apg}}</div><div class="stat-label">APG</div></div>
                        <div class="stat"><div class="stat-value">${{p.rpg}}</div><div class="stat-label">RPG</div></div>
                        <div class="stat"><div class="stat-value">${{p.usg}}%</div><div class="stat-label">USG</div></div>
                        <div class="stat"><div class="stat-value">${{p.ts}}%</div><div class="stat-label">TS%</div></div>
                    </div>
                    
                    <div class="archetype-section">
                        <div class="archetype-label">Offensive Role</div>
                        <div class="archetype-name offense">${{p.off_archetype}}</div>
                        <div class="confidence">${{p.off_confidence}}% confidence${{p.off_secondary ? ' • Secondary: ' + p.off_secondary : ''}}</div>
                        <div class="reasons">
                            ${{p.off_reasons.map(r => `<div class="reason">${{r}}</div>`).join('')}}
                        </div>
                    </div>
                    
                    <div class="archetype-section">
                        <div class="archetype-label">Defensive Role</div>
                        <div class="archetype-name defense">${{p.def_archetype || 'Unknown'}}</div>
                        <div class="confidence">${{p.def_confidence}}% confidence</div>
                        <div class="reasons">
                            ${{p.def_reasons.map(r => `<div class="reason">${{r}}</div>`).join('')}}
                        </div>
                    </div>
                </div>
            `).join('');
        }}
        
        function filterPlayers() {{
            const search = document.getElementById('search').value.toLowerCase();
            const offType = document.getElementById('offFilter').value;
            const defType = document.getElementById('defFilter').value;
            const season = document.getElementById('seasonFilter').value;
            
            const filtered = players.filter(p => {{
                if (search && !p.name.toLowerCase().includes(search)) return false;
                if (offType && p.off_archetype !== offType) return false;
                if (defType && p.def_archetype !== defType) return false;
                if (season && p.season !== season) return false;
                return true;
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
