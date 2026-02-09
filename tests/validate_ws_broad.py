"""
tests/validate_ws_broad.py
Validates calculated Win Shares against Basketball-Reference (2023-24)
for a broad set of players to ensure the formula holds across different archetypes.
"""

import pandas as pd
import numpy as np
import os
import sys

# Adjust path to find src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

DATA_PATH = "data/processed/metrics_linear.parquet"

# Ground Truth Data (Source: Basketball-Reference 2023-24 Advanced Stats)
# Format: Player: [WS, OWS, DWS]
TRUTH_DATA = {
    "Nikola Jokić":       [17.0, 12.0, 5.1], # MVP Center
    "Domantas Sabonis":   [12.6, 8.6, 4.0],  # Rebounding Big
    "Anthony Davis":      [11.8, 7.2, 4.7],  # Two-Way Big
    "Kevin Durant":       [8.3, 5.1, 3.2],   # Scoring Wing
    "DeMar DeRozan":      [9.2, 7.0, 2.2],   # High Volume Scorer
    "Fred VanVleet":      [8.4, 5.3, 3.1],   # Playmaking Guard
    "Tyrese Haliburton":  [9.1, 7.6, 1.5],   # High Assist Guard
}

# Thresholds for per-player pass/warn/fail
WS_THRESHOLD_PASS = 10.0    # WS% error < 10% = pass
WS_THRESHOLD_WARN = 25.0    # WS% error < 25% = warn, else fail

def validate():
    if not os.path.exists(DATA_PATH):
        print(f"❌ File not found: {DATA_PATH}")
        print("   Run 'python src/data_compute/compute_linear_metrics.py' first.")
        return

    print(f"Loading metrics from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    
    # Filter for 2023-24 season
    df = df[df['season'] == '2023-24'].copy()
    
    if len(df) == 0:
        print("❌ No data for 2023-24 season")
        return

    print(f"✅ Loaded {len(df)} players for 2023-24")

    results = []
    
    print("\n--- Win Shares Validation (2023-24) ---")
    print(f"{'Player':<20} | {'WS (Calc)':<9} {'WS (Ref)':<9} {'Diff%':<7} | {'OWS (Calc)':<9} {'OWS (Ref)':<9} {'Diff%':<7} | {'DWS (Calc)':<9} {'DWS (Ref)':<9} {'Diff%':<7}")
    print("-" * 115)
    
    pd.options.mode.chained_assignment = None  # Silence warnings
    
    for player, truth in TRUTH_DATA.items():
        # Fuzzy match player name
        match = df[df['player_name'].str.contains(player, case=False, na=False)]
        
        if len(match) == 0:
            print(f"⚠️  {player:<19} | Not found in dataset")
            continue
            
        row = match.iloc[0]
        
        # Extract calculated values
        calc_ws = row.get('WS', 0.0)
        calc_ows = row.get('OWS', 0.0)
        calc_dws = row.get('DWS', 0.0)
        
        # Extract truth values
        ref_ws, ref_ows, ref_dws = truth
        
        # Calculate Errors
        ws_err = (calc_ws - ref_ws) / ref_ws * 100
        ows_err = (calc_ows - ref_ows) / ref_ows * 100
        # Handle small denominators for DWS
        dws_err = (calc_dws - ref_dws) / ref_dws * 100 if ref_dws > 0.1 else 0.0
        
        # Determine status based on WS error (primary metric)
        abs_ws_err = abs(ws_err)
        if abs_ws_err < WS_THRESHOLD_PASS:
            status = "✅"
        elif abs_ws_err < WS_THRESHOLD_WARN:
            status = "⚠️"
        else:
            status = "❌"
        
        print(f"{status} {player:<18} | {calc_ws:<9.2f} {ref_ws:<9.1f} {ws_err:+.1f}%  | {calc_ows:<9.2f} {ref_ows:<9.1f} {ows_err:+.1f}%  | {calc_dws:<9.2f} {ref_dws:<9.1f} {dws_err:+.1f}%")
        
        results.append({
            'Player': player,
            'WS_Error': abs(ws_err),
            'OWS_Error': abs(ows_err),
            'DWS_Error': abs(dws_err)
        })

    # Summary Statistics
    if results:
        res_df = pd.DataFrame(results)
        print("-" * 115)
        
        ws_mae = res_df['WS_Error'].mean()
        ows_mae = res_df['OWS_Error'].mean()
        dws_mae = res_df['DWS_Error'].mean()
        
        print(f"   {'MAE %':<18} | {'':<9} {'':<9} {ws_mae:.1f}%   | {'':<9} {'':<9} {ows_mae:.1f}%   | {'':<9} {'':<9} {dws_mae:.1f}%")
        
        # Overall verdict
        print()
        if ws_mae < 5.0:
            print(f"✅ WS accuracy excellent (MAE {ws_mae:.1f}%, target <5%)")
        elif ws_mae < 10.0:
            print(f"✅ WS accuracy good (MAE {ws_mae:.1f}%, target <10%)")
        elif ws_mae < 20.0:
            print(f"⚠️ WS accuracy moderate (MAE {ws_mae:.1f}%)")
        else:
            print(f"❌ WS accuracy poor (MAE {ws_mae:.1f}%)")
            
        if ows_mae < 15.0:
            print(f"✅ OWS accuracy acceptable (MAE {ows_mae:.1f}%)")
        else:
            print(f"⚠️ OWS accuracy needs improvement (MAE {ows_mae:.1f}%)")
            
        if dws_mae < 30.0:
            print(f"⚠️ DWS has expected divergence (MAE {dws_mae:.1f}%, DWS is notoriously hard)")
        else:
            print(f"❌ DWS accuracy poor (MAE {dws_mae:.1f}%)")

if __name__ == "__main__":
    validate()