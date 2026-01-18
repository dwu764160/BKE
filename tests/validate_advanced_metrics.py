"""
tests/validate_advanced_metrics.py
Validates BPM, VORP, and Win Shares against confirmed Basketball-Reference data.

Reference players chosen to represent diverse player types:
- Superstars: Giannis Antetokounmpo, Nikola Jokić
- Role players: Tobias Harris, Daniel Gafford
- Young players: Brandin Podziemski
"""

import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

DATA_PATH = "data/processed/metrics_linear.parquet"

# Ground Truth Data (Verified from B-REF screenshots January 2026)
# Format: Season -> Player -> [WS, OWS, DWS, BPM, VORP]
TRUTH_DATA = {
    "2022-23": {
        "Giannis Antetokounmpo": [8.6, 4.9, 3.7, 8.5, 5.4],
        "Nikola Jokić":          [14.9, 11.2, 3.8, 13.0, 10.6],
        "Tobias Harris":         [5.9, 2.8, 3.1, 0.7, 1.7],
        "Daniel Gafford":        [6.0, 4.2, 1.9, 1.6, 1.3],
        # New validation players
        "Jonathan Kuminga":      [1.4, 1.0, 0.5, -1.7, 0.1],
        "Trae Young":            [6.7, 5.3, 1.4, 3.3, 3.4],
        "Michael Porter Jr.":    [4.3, 2.6, 1.8, 0.2, 1.0],
        "Julius Randle":         [8.1, 5.0, 3.1, 3.9, 3.9],
    },
    "2023-24": {
        "Giannis Antetokounmpo": [13.2, 9.5, 3.7, 9.0, 7.2],
        "Nikola Jokić":          [17.0, 12.0, 5.1, 13.2, 10.6],
        "Brandin Podziemski":    [4.1, 2.1, 2.0, -0.1, 0.9],
        "Tobias Harris":         [5.9, 3.2, 2.7, 0.9, 1.2],
        "Daniel Gafford":        [7.8, 5.2, 2.5, 2.6, 2.1],
        # New validation players
        "Jonathan Kuminga":      [2.7, 1.9, 0.8, -0.4, 0.8],
        "Trae Young":            [4.6, 4.0, 0.6, 2.6, 2.2],
        "Eric Gordon":           [2.5, 1.5, 1.0, -1.7, 0.1],
        "Michael Porter Jr.":    [6.2, 3.1, 3.1, 0.1, 1.3],
        "Julius Randle":         [3.8, 1.9, 1.9, 1.9, 1.6],
    },
    "2024-25": {
        "Giannis Antetokounmpo": [11.5, 7.8, 3.7, 9.5, 6.6],
        "Nikola Jokić":          [16.4, 12.7, 3.8, 13.3, 9.8],
        "Brandin Podziemski":    [4.2, 1.9, 2.3, 0.7, 1.2],
        "Tobias Harris":         [5.2, 2.9, 2.3, 0.1, 1.2],
        "Daniel Gafford":        [5.9, 4.4, 1.5, 3.8, 1.8],
        # New validation players
        "Jonathan Kuminga":      [1.9, 1.5, 0.5, -0.6, 0.4],
        "Trae Young":            [5.7, 4.4, 1.3, 0.5, 1.7],
        "Eric Gordon":           [0.9, 0.7, 0.2, -1.6, 0.1],
        "Michael Porter Jr.":    [6.4, 4.6, 1.8, -0.1, 1.2],
        "Julius Randle":         [6.2, 3.6, 2.6, 1.3, 1.8],
    }
}

def validate():
    if not os.path.exists(DATA_PATH):
        print(f"❌ File not found: {DATA_PATH}")
        return

    print(f"Loading metrics from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    
    ws_errors, ows_errors, dws_errors, bpm_errors, vorp_errors = [], [], [], [], []

    for season, players in TRUTH_DATA.items():
        print(f"\n{'='*120}")
        print(f"  {season} VALIDATION")
        print(f"{'='*120}")
        print(f"{'Player':<26} | {'WS':<14} {'Err':<6} | {'OWS':<14} {'Err':<6} | {'DWS':<14} {'Err':<6} | {'BPM':<14} {'Err':<6} | {'VORP':<14} {'Err':<6}")
        print("-" * 120)
        
        season_df = df[df['season'] == season]
        
        for player, truth in players.items():
            match = season_df[season_df['player_name'].str.contains(player, case=False, na=False)]
            if len(match) == 0:
                print(f"{player:<26} | NOT FOUND IN DATA")
                continue
                
            row = match.iloc[0]
            
            # Get Calculated Values
            c_ws = row.get('WS', 0.0)
            c_ows = row.get('OWS', 0.0)
            c_dws = row.get('DWS', 0.0)
            c_bpm = row.get('BPM', 0.0)
            c_vorp = row.get('VORP', 0.0)
            
            # Truth: [WS, OWS, DWS, BPM, VORP]
            t_ws, t_ows, t_dws, t_bpm, t_vorp = truth
            
            # Diffs
            d_ws = c_ws - t_ws
            d_ows = c_ows - t_ows
            d_dws = c_dws - t_dws
            d_bpm = c_bpm - t_bpm
            d_vorp = c_vorp - t_vorp
            
            ws_errors.append(abs(d_ws))
            ows_errors.append(abs(d_ows))
            dws_errors.append(abs(d_dws))
            bpm_errors.append(abs(d_bpm))
            vorp_errors.append(abs(d_vorp))
            
            # Format: "Calc (Ref)"
            f_ws = f"{c_ws:>5.1f} ({t_ws:>4.1f})"
            f_ows = f"{c_ows:>5.1f} ({t_ows:>4.1f})"
            f_dws = f"{c_dws:>5.1f} ({t_dws:>4.1f})"
            f_bpm = f"{c_bpm:>5.1f} ({t_bpm:>4.1f})"
            f_vorp = f"{c_vorp:>5.1f} ({t_vorp:>4.1f})"
            
            print(f"{player:<26} | {f_ws:<14} {d_ws:+.1f}   | {f_ows:<14} {d_ows:+.1f}   | {f_dws:<14} {d_dws:+.1f}   | {f_bpm:<14} {d_bpm:+.1f}   | {f_vorp:<14} {d_vorp:+.1f}")

    print("\n" + "="*80)
    print("SUMMARY: Mean Absolute Errors")
    print("="*80)
    print(f"  WS   MAE: {np.mean(ws_errors):>5.2f}  (Target < 1.5)")
    print(f"  OWS  MAE: {np.mean(ows_errors):>5.2f}  (Target < 1.0)")
    print(f"  DWS  MAE: {np.mean(dws_errors):>5.2f}  (Target < 1.0)")
    print(f"  BPM  MAE: {np.mean(bpm_errors):>5.2f}  (Target < 1.0)")
    print(f"  VORP MAE: {np.mean(vorp_errors):>5.2f}  (Target < 1.0)")
    
    # Pass/Fail
    ws_ok = np.mean(ws_errors) < 1.5
    ows_ok = np.mean(ows_errors) < 1.0
    dws_ok = np.mean(dws_errors) < 1.0
    bpm_ok = np.mean(bpm_errors) < 1.0
    vorp_ok = np.mean(vorp_errors) < 1.0
    
    print("\n" + "="*80)
    print("PASS/FAIL STATUS")
    print("="*80)
    print(f"  WS:   {'✅ PASS' if ws_ok else '❌ FAIL'}")
    print(f"  OWS:  {'✅ PASS' if ows_ok else '❌ FAIL'}")
    print(f"  DWS:  {'✅ PASS' if dws_ok else '❌ FAIL'}")
    print(f"  BPM:  {'✅ PASS' if bpm_ok else '❌ FAIL'}")
    print(f"  VORP: {'✅ PASS' if vorp_ok else '❌ FAIL'}")

if __name__ == "__main__":
    validate()