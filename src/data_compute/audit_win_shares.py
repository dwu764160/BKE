"""
src/data_compute/audit_win_shares.py
Deep Debugger for Win Shares - Validates output from compute_linear_metrics.py

Validates against B-REF targets:
- Jokić 2023-24: OWS ~12.0, DWS ~5.1, WS ~17.0
- League Total WS: ~1230 (30 teams * 41 avg wins)
"""

import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

DATA_DIR = "data/processed"
HISTORICAL_DIR = "data/historical"


def load_metrics():
    """Load the computed metrics from compute_linear_metrics.py output."""
    path = os.path.join(DATA_DIR, "metrics_linear.parquet")
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)


def load_player_data():
    """Load full player data for additional context."""
    path = os.path.join(DATA_DIR, "player_profiles_advanced.parquet")
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)


def audit_win_shares():
    """
    Audit Win Shares by loading the output from compute_linear_metrics.py
    and comparing against B-REF targets.
    """
    metrics = load_metrics()
    if metrics is None:
        print("ERROR: metrics_linear.parquet not found. Run compute_linear_metrics.py first.")
        return
    
    player_data = load_player_data()
    
    print("\n" + "=" * 60)
    print("WIN SHARES AUDIT - Validating compute_linear_metrics.py output")
    print("=" * 60)
    
    for season in sorted(metrics['season'].unique()):
        season_df = metrics[metrics['season'] == season].copy()
        
        print(f"\n--- {season} ---")
        print(f"Total OWS: {season_df['OWS'].sum():.1f}")
        print(f"Total DWS: {season_df['DWS'].sum():.1f}")
        print(f"Total WS:  {season_df['WS'].sum():.1f} (Target: ~1230)")
        
        # Jokić validation
        jokic = season_df[season_df['player_name'].str.contains("Jok", case=False, na=False)]
        if len(jokic) > 0:
            j = jokic.iloc[0]
            
            # Get B-REF targets based on season
            if season == '2023-24':
                target_ows, target_dws, target_ws = 12.0, 5.1, 17.0
            elif season == '2022-23':
                target_ows, target_dws, target_ws = 11.2, 3.8, 14.9
            elif season == '2024-25':
                target_ows, target_dws, target_ws = 9.9, 3.3, 13.3  # Partial season estimate
            else:
                target_ows, target_dws, target_ws = 0, 0, 0
            
            print(f"\n  Jokić Validation:")
            print(f"  {'Metric':<12} {'Calculated':>10} {'B-REF Target':>12} {'Error':>10}")
            print(f"  {'-'*46}")
            print(f"  {'OWS':<12} {j['OWS']:>10.2f} {target_ows:>12.1f} {j['OWS']-target_ows:>+10.2f}")
            print(f"  {'DWS':<12} {j['DWS']:>10.2f} {target_dws:>12.1f} {j['DWS']-target_dws:>+10.2f}")
            print(f"  {'WS':<12} {j['WS']:>10.2f} {target_ws:>12.1f} {j['WS']-target_ws:>+10.2f}")
            
            if 'PProd' in j.index and 'TotPoss' in j.index:
                print(f"\n  Additional metrics:")
                print(f"  PProd: {j['PProd']:.1f}")
                print(f"  TotPoss: {j['TotPoss']:.1f}")
                if 'qAST' in j.index:
                    print(f"  qAST: {j['qAST']:.3f}")
    
    # Top 10 overall for latest season
    latest = metrics['season'].max()
    latest_df = metrics[metrics['season'] == latest].nlargest(10, 'WS')
    
    print(f"\n{'='*60}")
    print(f"TOP 10 BY WIN SHARES ({latest})")
    print(f"{'='*60}")
    cols = ['player_name', 'GP', 'WS', 'OWS', 'DWS']
    print(latest_df[cols].to_string(index=False))


if __name__ == "__main__":
    audit_win_shares()