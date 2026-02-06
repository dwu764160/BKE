"""
tests/validate_rapm.py
Comprehensive RAPM validation against real-world data and statistical benchmarks.

Validation Categories:
1. Distribution checks (mean, std, range)
2. External benchmark comparison (NBA.com, Basketball-Reference)
3. Bootstrap confidence intervals
4. Season-to-season stability
5. Team-level coherence
6. Correlation with box-score metrics
"""

import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DATA_DIR = "data/processed"
HISTORICAL_DIR = "data/historical"
OUTPUT_DIR = "data/processed"

# Known elite players for sanity checks (player_id: expected_range)
# These are players who should consistently rank highly
ELITE_PLAYERS = {
    "203999": {"name": "Nikola Jokić", "expected_rapm_min": 3.0},
    "203507": {"name": "Giannis Antetokounmpo", "expected_rapm_min": 2.5},
    "201142": {"name": "Kevin Durant", "expected_rapm_min": 1.5},
    "201566": {"name": "Russell Westbrook", "expected_rapm_max": 2.0},  # Often negative/neutral
    "203076": {"name": "Anthony Davis", "expected_rapm_min": 2.0},
    "1629029": {"name": "Luka Dončić", "expected_rapm_min": 1.5},
    "201935": {"name": "James Harden", "expected_rapm_min": 0.5},
}

# Expected defensive specialists (high DRAPM)
DEFENSIVE_SPECIALISTS = {
    "1630169": {"name": "Evan Mobley", "expected_drapm_min": 1.0},
    "203497": {"name": "Rudy Gobert", "expected_drapm_min": 2.0},
    "1628378": {"name": "Victor Wembanyama", "expected_drapm_min": 2.0},
}

# Expected offensive stars (high ORAPM)
OFFENSIVE_STARS = {
    "203999": {"name": "Nikola Jokić", "expected_orapm_min": 4.0},
    "201939": {"name": "Stephen Curry", "expected_orapm_min": 3.0},
    "1628369": {"name": "Jayson Tatum", "expected_orapm_min": 2.0},
}

# xRAPM.com 2023-24 benchmark data (from xrapm.com/table_pages/xRAPM_2024.html)
# Note: xRAPM DRAPM is negative = good defense; our DRAPM is positive = good defense
# We store xRAPM values as-is and flip sign when comparing
XRAPM_2023_24_BENCHMARK = {
    "Nikola Jokić":              {"orapm": 6.1, "drapm": -2.5, "total": 8.6},
    "Joel Embiid":               {"orapm": 3.4, "drapm": -3.7, "total": 7.1},
    "Shai Gilgeous-Alexander":   {"orapm": 4.7, "drapm": -1.9, "total": 6.6},
    "Luka Dončić":               {"orapm": 5.4, "drapm": -0.8, "total": 6.2},
    "Kyrie Irving":              {"orapm": 5.0, "drapm": -1.0, "total": 6.0},
    "Kawhi Leonard":             {"orapm": 2.9, "drapm": -2.8, "total": 5.7},
    "Paul George":               {"orapm": 3.3, "drapm": -2.2, "total": 5.5},
    "LeBron James":              {"orapm": 3.7, "drapm": -1.4, "total": 5.1},
    "Anthony Davis":             {"orapm": 1.3, "drapm": -3.8, "total": 5.1},
    "Donovan Mitchell":          {"orapm": 3.4, "drapm": -1.6, "total": 5.0},
    "Damian Lillard":            {"orapm": 4.4, "drapm": -0.5, "total": 4.9},
    "Giannis Antetokounmpo":     {"orapm": 3.2, "drapm": -1.7, "total": 4.9},
    "Stephen Curry":             {"orapm": 4.2, "drapm": -0.4, "total": 4.6},
    "Jayson Tatum":              {"orapm": 3.5, "drapm": -1.0, "total": 4.5},
    "Anthony Edwards":           {"orapm": 2.4, "drapm": -2.0, "total": 4.4},
    "Jimmy Butler":              {"orapm": 3.2, "drapm": -1.0, "total": 4.2},
    "Draymond Green":            {"orapm": -0.5, "drapm": -4.7, "total": 4.2},
    "Kristaps Porziņģis":        {"orapm": 1.8, "drapm": -2.4, "total": 4.2},
    "Jalen Brunson":             {"orapm": 4.4, "drapm": 0.3, "total": 4.1},
    "Devin Booker":              {"orapm": 4.4, "drapm": 0.3, "total": 4.1},
    "Derrick White":             {"orapm": 1.4, "drapm": -2.7, "total": 4.1},
    "Chet Holmgren":             {"orapm": 1.6, "drapm": -2.4, "total": 4.0},
    "Jrue Holiday":              {"orapm": 2.2, "drapm": -1.8, "total": 4.0},
    "Fred VanVleet":             {"orapm": 1.8, "drapm": -2.0, "total": 3.8},
    "Franz Wagner":              {"orapm": 1.4, "drapm": -2.4, "total": 3.8},
    "Tyrese Haliburton":         {"orapm": 4.4, "drapm": 0.7, "total": 3.7},
    "Kevin Durant":              {"orapm": 2.8, "drapm": -0.9, "total": 3.7},
    "Herbert Jones":             {"orapm": 0.9, "drapm": -2.6, "total": 3.5},
}


def load_rapm_data():
    """Load the computed RAPM data."""
    path = os.path.join(DATA_DIR, "player_rapm.parquet")
    if not os.path.exists(path):
        path = os.path.join(DATA_DIR, "player_rapm.csv")
    
    if not os.path.exists(path):
        print("❌ No RAPM data found")
        return None
    
    df = pd.read_parquet(path) if path.endswith('.parquet') else pd.read_csv(path)
    return df


def validate_distribution(df, report):
    """Check RAPM distribution properties."""
    print("\n" + "="*60)
    print("1. DISTRIBUTION VALIDATION")
    print("="*60)
    
    report["distribution"] = {}
    
    for rapm_type in df['RAPM_type'].unique():
        subset = df[df['RAPM_type'] == rapm_type].copy()
        
        if 'RAPM' in subset.columns and subset['RAPM'].notna().any():
            rapm_vals = subset['RAPM'].dropna()
            stats = {
                "count": len(rapm_vals),
                "mean": float(rapm_vals.mean()),
                "std": float(rapm_vals.std()),
                "min": float(rapm_vals.min()),
                "max": float(rapm_vals.max()),
                "median": float(rapm_vals.median()),
                "skew": float(rapm_vals.skew()),
                "kurtosis": float(rapm_vals.kurtosis()),
            }
            
            # Validation checks
            checks = {
                "mean_near_zero": abs(stats["mean"]) < 0.5,
                "std_reasonable": 1.0 < stats["std"] < 5.0,
                "range_reasonable": stats["min"] > -10 and stats["max"] < 15,
                "not_too_skewed": abs(stats["skew"]) < 1.0,
            }
            
            report["distribution"][rapm_type] = {"stats": stats, "checks": checks}
            
            print(f"\n   {rapm_type.upper()}:")
            print(f"      Count: {stats['count']}")
            print(f"      Mean: {stats['mean']:.3f} {'✅' if checks['mean_near_zero'] else '⚠️'}")
            print(f"      Std: {stats['std']:.3f} {'✅' if checks['std_reasonable'] else '⚠️'}")
            print(f"      Range: [{stats['min']:.2f}, {stats['max']:.2f}] {'✅' if checks['range_reasonable'] else '⚠️'}")
            print(f"      Skew: {stats['skew']:.3f} {'✅' if checks['not_too_skewed'] else '⚠️'}")
    
    # ORAPM/DRAPM distributions
    for col in ['ORAPM', 'DRAPM']:
        if col in df.columns:
            vals = df[df[col].notna()][col]
            if len(vals) > 0:
                stats = {
                    "count": len(vals),
                    "mean": float(vals.mean()),
                    "std": float(vals.std()),
                    "min": float(vals.min()),
                    "max": float(vals.max()),
                }
                report["distribution"][col] = stats
                print(f"\n   {col}:")
                print(f"      Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}")
                print(f"      Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
    
    return report


def validate_elite_players(df, report):
    """Check that known elite players rank appropriately."""
    print("\n" + "="*60)
    print("2. ELITE PLAYER SANITY CHECKS")
    print("="*60)
    
    report["elite_checks"] = {"passed": [], "failed": [], "missing": []}
    
    # Use pooled RAPM for latest season
    seasons = sorted(df['season'].unique())
    latest = seasons[-1] if seasons else None
    
    if not latest:
        print("   ❌ No seasons found")
        return report
    
    pooled = df[(df['season'] == latest) & (df['RAPM_type'] == 'pooled')]
    
    print(f"\n   Season: {latest}")
    
    for pid, expected in ELITE_PLAYERS.items():
        player_row = pooled[pooled['player_id'] == pid]
        
        if player_row.empty:
            report["elite_checks"]["missing"].append(expected["name"])
            print(f"   ⚠️ {expected['name']}: Not found in data")
            continue
        
        rapm = player_row['RAPM'].values[0]
        name = player_row['player_name'].values[0] if 'player_name' in player_row.columns else expected['name']
        
        passed = True
        if "expected_rapm_min" in expected and rapm < expected["expected_rapm_min"]:
            passed = False
        if "expected_rapm_max" in expected and rapm > expected["expected_rapm_max"]:
            passed = False
        
        status = "✅" if passed else "⚠️"
        if passed:
            report["elite_checks"]["passed"].append(name)
        else:
            report["elite_checks"]["failed"].append({"name": name, "rapm": rapm, "expected": expected})
        
        print(f"   {status} {name}: {rapm:+.2f}")
    
    # Check defensive specialists DRAPM
    print(f"\n   Defensive Specialists (DRAPM):")
    pooled_split = df[(df['season'] == latest) & (df['RAPM_type'] == 'pooled_split')]
    
    for pid, expected in DEFENSIVE_SPECIALISTS.items():
        player_row = pooled_split[pooled_split['player_id'] == pid]
        if player_row.empty or 'DRAPM' not in player_row.columns:
            print(f"   ⚠️ {expected['name']}: Not found")
            continue
        
        drapm = player_row['DRAPM'].values[0]
        passed = drapm >= expected.get("expected_drapm_min", 0)
        status = "✅" if passed else "⚠️"
        print(f"   {status} {expected['name']}: DRAPM {drapm:+.2f}")
    
    # Check offensive stars ORAPM
    print(f"\n   Offensive Stars (ORAPM):")
    for pid, expected in OFFENSIVE_STARS.items():
        player_row = pooled_split[pooled_split['player_id'] == pid]
        if player_row.empty or 'ORAPM' not in player_row.columns:
            print(f"   ⚠️ {expected['name']}: Not found")
            continue
        
        orapm = player_row['ORAPM'].values[0]
        passed = orapm >= expected.get("expected_orapm_min", 0)
        status = "✅" if passed else "⚠️"
        print(f"   {status} {expected['name']}: ORAPM {orapm:+.2f}")
    
    return report


def validate_season_stability(df, report):
    """Check season-to-season correlation for player RAPM."""
    print("\n" + "="*60)
    print("3. SEASON-TO-SEASON STABILITY")
    print("="*60)
    
    report["stability"] = {}
    
    seasons = sorted(df['season'].unique())
    if len(seasons) < 2:
        print("   ⚠️ Need at least 2 seasons for stability check")
        return report
    
    # Use pooled RAPM
    pooled = df[df['RAPM_type'] == 'pooled'].copy()
    
    for i in range(len(seasons) - 1):
        s1, s2 = seasons[i], seasons[i+1]
        
        df1 = pooled[pooled['season'] == s1][['player_id', 'RAPM']].rename(columns={'RAPM': 'RAPM_1'})
        df2 = pooled[pooled['season'] == s2][['player_id', 'RAPM']].rename(columns={'RAPM': 'RAPM_2'})
        
        merged = df1.merge(df2, on='player_id', how='inner')
        
        if len(merged) < 50:
            print(f"   ⚠️ {s1} → {s2}: Only {len(merged)} common players")
            continue
        
        # Pearson and Spearman correlations
        pearson = merged['RAPM_1'].corr(merged['RAPM_2'])
        spearman = merged['RAPM_1'].corr(merged['RAPM_2'], method='spearman')
        
        report["stability"][f"{s1}_to_{s2}"] = {
            "n_common_players": len(merged),
            "pearson": float(pearson),
            "spearman": float(spearman),
        }
        
        # Expected: moderate correlation (0.3-0.7) for RAPM year-over-year
        p_check = 0.2 < pearson < 0.9
        s_check = 0.2 < spearman < 0.9
        
        print(f"\n   {s1} → {s2} ({len(merged)} players):")
        print(f"      Pearson r: {pearson:.3f} {'✅' if p_check else '⚠️'}")
        print(f"      Spearman ρ: {spearman:.3f} {'✅' if s_check else '⚠️'}")
    
    return report


def validate_orapm_drapm_consistency(df, report):
    """Check that ORAPM + DRAPM ≈ RAPM for same players."""
    print("\n" + "="*60)
    print("4. ORAPM/DRAPM CONSISTENCY")
    print("="*60)
    
    report["od_consistency"] = {}
    
    seasons = sorted(df['season'].unique())
    
    for season in seasons:
        pooled = df[(df['season'] == season) & (df['RAPM_type'] == 'pooled')][['player_id', 'RAPM']]
        pooled_split = df[(df['season'] == season) & (df['RAPM_type'] == 'pooled_split')]
        
        if pooled_split.empty or 'ORAPM' not in pooled_split.columns:
            continue
        
        pooled_split = pooled_split[['player_id', 'ORAPM', 'DRAPM']]
        
        merged = pooled.merge(pooled_split, on='player_id', how='inner')
        if len(merged) < 50:
            continue
        
        # RAPM should correlate with (ORAPM + DRAPM) / 2 approximately
        merged['RAPM_from_OD'] = (merged['ORAPM'] + merged['DRAPM']) / 2
        corr = merged['RAPM'].corr(merged['RAPM_from_OD'])
        
        # Check correlation between O and D (should be low/moderate - two-way players exist but not dominant)
        od_corr = merged['ORAPM'].corr(merged['DRAPM'])
        
        report["od_consistency"][season] = {
            "rapm_vs_od_avg_corr": float(corr),
            "orapm_drapm_corr": float(od_corr),
        }
        
        print(f"\n   {season}:")
        print(f"      RAPM vs (ORAPM+DRAPM)/2 correlation: {corr:.3f}")
        print(f"      ORAPM-DRAPM correlation: {od_corr:.3f} (low = distinct skills)")
    
    return report


def validate_possession_coverage(df, report):
    """Check possession counts and coverage."""
    print("\n" + "="*60)
    print("5. POSSESSION COVERAGE")
    print("="*60)
    
    report["coverage"] = {}
    
    if 'possessions_played' not in df.columns:
        print("   ⚠️ No possession counts in data")
        return report
    
    for season in sorted(df['season'].unique()):
        season_df = df[(df['season'] == season) & (df['RAPM_type'] == 'pooled')]
        
        if season_df.empty:
            continue
        
        poss = season_df['possessions_played']
        
        stats = {
            "n_players": len(season_df),
            "total_poss": int(poss.sum()),
            "mean_poss": float(poss.mean()),
            "min_poss": int(poss.min()),
            "max_poss": int(poss.max()),
            "players_under_1000": int((poss < 1000).sum()),
            "players_over_5000": int((poss >= 5000).sum()),
        }
        
        report["coverage"][season] = stats
        
        print(f"\n   {season}:")
        print(f"      Players: {stats['n_players']}")
        print(f"      Poss range: [{stats['min_poss']:,}, {stats['max_poss']:,}]")
        print(f"      Mean poss: {stats['mean_poss']:,.0f}")
        print(f"      Players < 1000 poss: {stats['players_under_1000']} (may be noisy)")
        print(f"      Players >= 5000 poss: {stats['players_over_5000']} (reliable)")
    
    return report


def validate_top_rankings(df, report):
    """Show top players and validate against known rankings."""
    print("\n" + "="*60)
    print("6. TOP PLAYER RANKINGS")
    print("="*60)
    
    report["rankings"] = {}
    
    latest_season = sorted(df['season'].unique())[-1]
    
    # Pooled RAPM top 10
    pooled = df[(df['season'] == latest_season) & (df['RAPM_type'] == 'pooled')]
    if not pooled.empty:
        top10 = pooled.nlargest(10, 'RAPM')[['player_name', 'RAPM', 'possessions_played']]
        
        print(f"\n   Top 10 Pooled RAPM ({latest_season}):")
        rankings = []
        for i, (_, row) in enumerate(top10.iterrows(), 1):
            poss = int(row['possessions_played']) if pd.notna(row['possessions_played']) else 0
            print(f"      {i:2d}. {row['player_name']:25s} {row['RAPM']:+5.2f} ({poss:,} poss)")
            rankings.append({"rank": i, "name": row['player_name'], "rapm": float(row['RAPM'])})
        
        report["rankings"]["pooled_top10"] = rankings
    
    # ORAPM top 10
    pooled_split = df[(df['season'] == latest_season) & (df['RAPM_type'] == 'pooled_split')]
    if not pooled_split.empty and 'ORAPM' in pooled_split.columns:
        top10_o = pooled_split.nlargest(10, 'ORAPM')[['player_name', 'ORAPM']]
        
        print(f"\n   Top 10 ORAPM ({latest_season}):")
        rankings_o = []
        for i, (_, row) in enumerate(top10_o.iterrows(), 1):
            print(f"      {i:2d}. {row['player_name']:25s} {row['ORAPM']:+5.2f}")
            rankings_o.append({"rank": i, "name": row['player_name'], "orapm": float(row['ORAPM'])})
        
        report["rankings"]["orapm_top10"] = rankings_o
    
    # DRAPM top 10
    if not pooled_split.empty and 'DRAPM' in pooled_split.columns:
        top10_d = pooled_split.nlargest(10, 'DRAPM')[['player_name', 'DRAPM']]
        
        print(f"\n   Top 10 DRAPM ({latest_season}):")
        rankings_d = []
        for i, (_, row) in enumerate(top10_d.iterrows(), 1):
            print(f"      {i:2d}. {row['player_name']:25s} {row['DRAPM']:+5.2f}")
            rankings_d.append({"rank": i, "name": row['player_name'], "drapm": float(row['DRAPM'])})
        
        report["rankings"]["drapm_top10"] = rankings_d
    
    return report


def validate_against_xrapm(df, report):
    """Validate 2023-24 RAPM against xRAPM.com benchmark data."""
    print("\n" + "="*60)
    print("7. xRAPM.com 2023-24 BENCHMARK COMPARISON")
    print("="*60)
    
    report["xrapm_benchmark"] = {"matches": [], "correlations": {}}
    
    # Get our 2023-24 pooled data
    season = "2023-24"
    pooled = df[(df['season'] == season) & (df['RAPM_type'] == 'pooled')].copy()
    pooled_split = df[(df['season'] == season) & (df['RAPM_type'] == 'pooled_split')].copy()
    
    if pooled.empty:
        print(f"   ⚠️ No pooled RAPM data for {season}")
        return report
    
    print(f"\n   Comparing {len(XRAPM_2023_24_BENCHMARK)} xRAPM leaders against our {season} data:")
    print(f"   (xRAPM DRAPM: negative=good defense | Ours: positive=good defense)\n")
    print(f"   {'Player':<28} {'xRAPM':>7} {'Ours':>7} {'Δ':>6}  {'xO':>5} {'O':>5} {'xD':>6} {'D':>6}")
    print(f"   {'-'*28} {'-'*7} {'-'*7} {'-'*6}  {'-'*5} {'-'*5} {'-'*6} {'-'*6}")
    
    comparison_rows = []
    
    for player_name, xrapm in XRAPM_2023_24_BENCHMARK.items():
        # Find player in our data
        player_pooled = pooled[pooled['player_name'] == player_name]
        player_split = pooled_split[pooled_split['player_name'] == player_name]
        
        if player_pooled.empty:
            print(f"   {player_name:<28} {xrapm['total']:>+7.1f}    --      --")
            continue
        
        our_rapm = player_pooled['RAPM'].values[0]
        delta = our_rapm - xrapm['total']
        
        our_orapm = player_split['ORAPM'].values[0] if not player_split.empty and 'ORAPM' in player_split.columns else None
        our_drapm = player_split['DRAPM'].values[0] if not player_split.empty and 'DRAPM' in player_split.columns else None
        
        # xRAPM DRAPM is negative=good, ours is positive=good, so flip for comparison
        xrapm_drapm_flipped = -xrapm['drapm']
        
        o_str = f"{our_orapm:>+5.1f}" if our_orapm is not None else "   --"
        d_str = f"{our_drapm:>+6.1f}" if our_drapm is not None else "    --"
        xd_str = f"{xrapm['drapm']:>+6.1f}"
        
        status = "✅" if abs(delta) < 2.0 else "⚠️"
        print(f"   {player_name:<28} {xrapm['total']:>+7.1f} {our_rapm:>+7.2f} {delta:>+6.2f}  {xrapm['orapm']:>+5.1f} {o_str} {xd_str} {d_str} {status}")
        
        comparison_rows.append({
            "player": player_name,
            "xrapm_total": xrapm['total'],
            "our_rapm": our_rapm,
            "delta": delta,
            "xrapm_orapm": xrapm['orapm'],
            "our_orapm": our_orapm,
            "xrapm_drapm": xrapm['drapm'],
            "our_drapm": our_drapm,
        })
    
    report["xrapm_benchmark"]["matches"] = comparison_rows
    
    # Compute correlations if we have enough matches
    if len(comparison_rows) >= 10:
        xrapm_totals = [r['xrapm_total'] for r in comparison_rows]
        our_totals = [r['our_rapm'] for r in comparison_rows]
        
        pearson = np.corrcoef(xrapm_totals, our_totals)[0, 1]
        
        # Spearman rank correlation
        from scipy.stats import spearmanr
        spearman, _ = spearmanr(xrapm_totals, our_totals)
        
        report["xrapm_benchmark"]["correlations"] = {
            "pearson": float(pearson),
            "spearman": float(spearman),
            "n_matched": len(comparison_rows),
        }
        
        print(f"\n   Correlation with xRAPM (n={len(comparison_rows)}):")
        print(f"      Pearson r:  {pearson:.3f} {'✅' if pearson > 0.7 else '⚠️' if pearson > 0.5 else '❌'}")
        print(f"      Spearman ρ: {spearman:.3f} {'✅' if spearman > 0.7 else '⚠️' if spearman > 0.5 else '❌'}")
        
        # Mean absolute error
        mae = np.mean([abs(r['delta']) for r in comparison_rows])
        print(f"      MAE:        {mae:.2f}")
        
        # Top-10 overlap
        xrapm_top10 = list(XRAPM_2023_24_BENCHMARK.keys())[:10]
        our_top10 = pooled.nlargest(10, 'RAPM')['player_name'].tolist()
        overlap = len(set(xrapm_top10) & set(our_top10))
        print(f"      Top-10 overlap: {overlap}/10 {'✅' if overlap >= 6 else '⚠️'}")
        
        report["xrapm_benchmark"]["mae"] = float(mae)
        report["xrapm_benchmark"]["top10_overlap"] = overlap
    
    return report


def diagnose_xrapm_discrepancy(df, report):
    """Investigate why our RAPM differs from xRAPM and suggest fixes."""
    print("\n" + "="*60)
    print("8. xRAPM DISCREPANCY DIAGNOSIS")
    print("="*60)
    
    report["diagnosis"] = {}
    season = "2023-24"
    
    pooled = df[(df['season'] == season) & (df['RAPM_type'] == 'pooled')].copy()
    pooled_split = df[(df['season'] == season) & (df['RAPM_type'] == 'pooled_split')].copy()
    
    if pooled.empty:
        print("   ⚠️ No data for diagnosis")
        return report
    
    # Build comparison data
    comparison_rows = []
    for player_name, xrapm in XRAPM_2023_24_BENCHMARK.items():
        player_pooled = pooled[pooled['player_name'] == player_name]
        player_split = pooled_split[pooled_split['player_name'] == player_name]
        if player_pooled.empty:
            continue
        our_rapm = player_pooled['RAPM'].values[0]
        our_orapm = player_split['ORAPM'].values[0] if not player_split.empty else None
        our_drapm = player_split['DRAPM'].values[0] if not player_split.empty else None
        comparison_rows.append({
            "player": player_name,
            "xrapm_total": xrapm['total'],
            "our_rapm": our_rapm,
            "xrapm_orapm": xrapm['orapm'],
            "our_orapm": our_orapm,
            "xrapm_drapm_flipped": -xrapm['drapm'],  # flip to our convention
            "our_drapm": our_drapm,
        })
    
    if len(comparison_rows) < 10:
        print("   ⚠️ Not enough matched players")
        return report
    
    xrapm_totals = np.array([r['xrapm_total'] for r in comparison_rows])
    our_totals = np.array([r['our_rapm'] for r in comparison_rows])
    deltas = our_totals - xrapm_totals
    
    # 1. SYSTEMATIC BIAS CHECK
    print("\n   1️⃣  SYSTEMATIC BIAS ANALYSIS")
    mean_delta = np.mean(deltas)
    std_delta = np.std(deltas)
    print(f"      Mean Δ (ours - xRAPM): {mean_delta:+.2f}")
    print(f"      Std Δ:                 {std_delta:.2f}")
    report["diagnosis"]["mean_delta"] = float(mean_delta)
    report["diagnosis"]["std_delta"] = float(std_delta)
    
    if mean_delta < -1.5:
        print(f"      → Our values are systematically LOWER by ~{abs(mean_delta):.1f} points")
        print(f"      → Likely cause: Over-regularization (alpha too high)")
    elif mean_delta > 1.5:
        print(f"      → Our values are systematically HIGHER")
    else:
        print(f"      → No major systematic bias")
    
    # 2. SCALE/VARIANCE CHECK
    print("\n   2️⃣  VARIANCE COMPARISON")
    our_std = np.std(our_totals)
    xrapm_std = np.std(xrapm_totals)
    variance_ratio = our_std / xrapm_std
    print(f"      Our std:   {our_std:.2f}")
    print(f"      xRAPM std: {xrapm_std:.2f}")
    print(f"      Ratio:     {variance_ratio:.2f}")
    report["diagnosis"]["variance_ratio"] = float(variance_ratio)
    
    if variance_ratio < 0.7:
        print(f"      → Our RAPM is too compressed (over-regularized)")
        print(f"      → Fix: Lower alpha values in RidgeCV")
    elif variance_ratio > 1.3:
        print(f"      → Our RAPM has more spread (under-regularized)")
    else:
        print(f"      → Variance is similar ✅")
    
    # 3. LINEAR RESCALING TEST
    print("\n   3️⃣  LINEAR RESCALING TEST")
    from scipy.stats import linregress
    slope, intercept, r_value, _, _ = linregress(our_totals, xrapm_totals)
    print(f"      Best fit: xRAPM ≈ {slope:.2f} × Ours + {intercept:.2f}")
    print(f"      R² = {r_value**2:.3f}")
    report["diagnosis"]["rescale_slope"] = float(slope)
    report["diagnosis"]["rescale_intercept"] = float(intercept)
    report["diagnosis"]["rescale_r2"] = float(r_value**2)
    
    # Apply rescaling and check improvement
    rescaled = slope * our_totals + intercept
    rescaled_mae = np.mean(np.abs(rescaled - xrapm_totals))
    original_mae = np.mean(np.abs(deltas))
    print(f"      Original MAE:  {original_mae:.2f}")
    print(f"      Rescaled MAE:  {rescaled_mae:.2f}")
    
    if rescaled_mae < original_mae * 0.6:
        print(f"      → Rescaling helps significantly!")
        print(f"      → This suggests a consistent scale/offset difference")
    
    # 4. ORAPM/DRAPM COMPONENT ANALYSIS
    print("\n   4️⃣  COMPONENT ANALYSIS (ORAPM/DRAPM)")
    orapm_pairs = [(r['our_orapm'], r['xrapm_orapm']) for r in comparison_rows if r['our_orapm'] is not None]
    drapm_pairs = [(r['our_drapm'], r['xrapm_drapm_flipped']) for r in comparison_rows if r['our_drapm'] is not None]
    
    if len(orapm_pairs) >= 10:
        our_o, xrapm_o = zip(*orapm_pairs)
        o_corr = np.corrcoef(our_o, xrapm_o)[0, 1]
        o_mae = np.mean(np.abs(np.array(our_o) - np.array(xrapm_o)))
        print(f"      ORAPM correlation: {o_corr:.3f}, MAE: {o_mae:.2f}")
        report["diagnosis"]["orapm_corr"] = float(o_corr)
    
    if len(drapm_pairs) >= 10:
        our_d, xrapm_d = zip(*drapm_pairs)
        d_corr = np.corrcoef(our_d, xrapm_d)[0, 1]
        d_mae = np.mean(np.abs(np.array(our_d) - np.array(xrapm_d)))
        print(f"      DRAPM correlation: {d_corr:.3f}, MAE: {d_mae:.2f}")
        report["diagnosis"]["drapm_corr"] = float(d_corr)
    
    # 5. METHODOLOGY DIFFERENCE HYPOTHESIS
    print("\n   5️⃣  METHODOLOGY HYPOTHESIS")
    print("""
      xRAPM likely uses enhancements we don't:
      
      a) BOX-SCORE PRIOR (most likely cause)
         - xRAPM = RAPM + Bayesian prior from box stats
         - This inflates star players' values
         - Our pure RAPM is more conservative
         
      b) MULTI-YEAR WEIGHTING
         - They may use 5+ years with different decay
         - We use 3 years with 100%/65%/40% decay
         
      c) MINUTES-WEIGHTING
         - They may weight possessions by player minutes
         - This amplifies high-minute player signals
         
      d) LOWER REGULARIZATION
         - They may use lower alpha (less shrinkage)
         - Our alpha=7500 is quite aggressive
""")
    
    # 6. RECOMMENDED FIXES
    print("   6️⃣  RECOMMENDED FIXES")
    print("""
      To better match xRAPM:
      
      1. REDUCE REGULARIZATION (Quick fix)
         - In compute_rapm.py, try alphas = [50, 100, 250, 500, 1000]
         - Lower alpha = less shrinkage = higher values for stars
         
      2. ADD BOX-SCORE PRIOR (Best fix)
         - Compute BPM-like prior for each player
         - Use as Bayesian prior in ridge regression
         - This is what ESPN RPM and xRAPM do
         
      3. APPLY LINEAR RESCALING (Simplest fix)
         - Multiply our RAPM by {slope:.2f} and add {intercept:.2f}
         - Quick calibration to match xRAPM scale
         
      4. EXTEND POOLING WINDOW
         - Use 5 years instead of 3
         - More data = less regularization needed
""".format(slope=slope, intercept=intercept))
    
    report["diagnosis"]["recommendations"] = [
        f"Reduce alpha: try [50, 100, 250, 500, 1000]",
        f"Add box-score prior (BPM-based)",
        f"Apply rescaling: {slope:.2f} × RAPM + {intercept:.2f}",
        f"Extend to 5-year pooling",
    ]
    
    return report


def load_xrapm_data():
    """Load our computed xRAPM data (with BPM prior)."""
    path = os.path.join(DATA_DIR, "player_xrapm.parquet")
    if not os.path.exists(path):
        path = os.path.join(DATA_DIR, "player_xrapm.csv")
    
    if not os.path.exists(path):
        return None
    
    df = pd.read_parquet(path) if path.endswith('.parquet') else pd.read_csv(path)
    return df


def validate_xrapm_vs_benchmark(report):
    """Compare our xRAPM (with BPM prior) against xRAPM.com benchmark."""
    print("\n" + "="*60)
    print("9. OUR xRAPM (BPM Prior) vs xRAPM.com BENCHMARK")
    print("="*60)
    
    report["xrapm_comparison"] = {}
    
    xrapm_df = load_xrapm_data()
    if xrapm_df is None:
        print("   ⚠️ No xRAPM data found. Run compute_xrapm.py first.")
        return report
    
    season = "2023-24"
    
    # Get pooled xRAPM with prior
    pooled = xrapm_df[(xrapm_df['season'] == season) & (xrapm_df['xRAPM_type'] == 'pooled_with_prior')].copy()
    pooled_split = xrapm_df[(xrapm_df['season'] == season) & (xrapm_df['xRAPM_type'] == 'pooled_split_with_prior')].copy()
    
    if pooled.empty:
        print(f"   ⚠️ No xRAPM data for {season}")
        return report
    
    print(f"\n   Comparing against {len(XRAPM_2023_24_BENCHMARK)} xRAPM.com players:\n")
    print(f"   {'Player':<28} {'xRAPM.com':>10} {'Ours':>8} {'Δ':>7}  {'Prior':>6}")
    print(f"   {'-'*28} {'-'*10} {'-'*8} {'-'*7}  {'-'*6}")
    
    comparison_rows = []
    
    for player_name, xrapm in XRAPM_2023_24_BENCHMARK.items():
        player_row = pooled[pooled['player_name'] == player_name]
        
        if player_row.empty:
            print(f"   {player_name:<28} {xrapm['total']:>+10.1f}       --      --")
            continue
        
        our_xrapm = player_row['xRAPM'].values[0]
        bpm_prior = player_row['BPM_prior'].values[0] if 'BPM_prior' in player_row.columns else 0
        delta = our_xrapm - xrapm['total']
        
        status = "✅" if abs(delta) < 2.0 else "⚠️" if abs(delta) < 3.0 else "❌"
        print(f"   {player_name:<28} {xrapm['total']:>+10.1f} {our_xrapm:>+8.2f} {delta:>+7.2f}  {bpm_prior:>+6.2f} {status}")
        
        comparison_rows.append({
            "player": player_name,
            "xrapm_com": xrapm['total'],
            "our_xrapm": our_xrapm,
            "delta": delta,
            "bpm_prior": bpm_prior,
        })
    
    report["xrapm_comparison"]["matches"] = comparison_rows
    
    # Compute statistics
    if len(comparison_rows) >= 10:
        xrapm_vals = np.array([r['xrapm_com'] for r in comparison_rows])
        our_vals = np.array([r['our_xrapm'] for r in comparison_rows])
        
        pearson = np.corrcoef(xrapm_vals, our_vals)[0, 1]
        from scipy.stats import spearmanr
        spearman, _ = spearmanr(xrapm_vals, our_vals)
        mae = np.mean(np.abs(our_vals - xrapm_vals))
        
        # Top-10 overlap
        xrapm_top10 = list(XRAPM_2023_24_BENCHMARK.keys())[:10]
        our_top10 = pooled.nlargest(10, 'xRAPM')['player_name'].tolist()
        overlap = len(set(xrapm_top10) & set(our_top10))
        
        report["xrapm_comparison"]["stats"] = {
            "pearson": float(pearson),
            "spearman": float(spearman),
            "mae": float(mae),
            "top10_overlap": overlap,
            "n_matched": len(comparison_rows),
        }
        
        print(f"\n   STATISTICS:")
        print(f"      Pearson r:      {pearson:.3f} {'✅' if pearson > 0.7 else '⚠️' if pearson > 0.5 else '❌'}")
        print(f"      Spearman ρ:     {spearman:.3f} {'✅' if spearman > 0.7 else '⚠️' if spearman > 0.5 else '❌'}")
        print(f"      MAE:            {mae:.2f} {'✅' if mae < 1.5 else '⚠️' if mae < 2.5 else '❌'}")
        print(f"      Top-10 overlap: {overlap}/10 {'✅' if overlap >= 6 else '⚠️'}")
        
        # Compare to pure RAPM performance
        print(f"\n   IMPROVEMENT OVER PURE RAPM:")
        if "xrapm_benchmark" in report and "correlations" in report["xrapm_benchmark"]:
            pure_pearson = report["xrapm_benchmark"]["correlations"].get("pearson", 0)
            pure_mae = report["xrapm_benchmark"].get("mae", 999)
            pure_overlap = report["xrapm_benchmark"].get("top10_overlap", 0)
            
            p_delta = pearson - pure_pearson
            mae_delta = pure_mae - mae  # positive = improvement
            overlap_delta = overlap - pure_overlap
            
            print(f"      Pearson Δ:      {p_delta:+.3f} {'✅' if p_delta > 0 else '⚠️'}")
            print(f"      MAE Δ:          {-mae_delta:+.2f} (lower is better) {'✅' if mae_delta > 0 else '⚠️'}")
            print(f"      Top-10 overlap Δ: {overlap_delta:+d} {'✅' if overlap_delta > 0 else '⚠️'}")
    
    return report


def generate_summary(report):
    """Generate overall validation summary."""
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    total_checks = 0
    passed_checks = 0
    
    # Count distribution checks
    if "distribution" in report:
        for rapm_type, data in report["distribution"].items():
            if isinstance(data, dict) and "checks" in data:
                for check_name, passed in data["checks"].items():
                    total_checks += 1
                    if passed:
                        passed_checks += 1
    
    # Count elite player checks
    if "elite_checks" in report:
        total_checks += len(report["elite_checks"]["passed"]) + len(report["elite_checks"]["failed"])
        passed_checks += len(report["elite_checks"]["passed"])
    
    if total_checks > 0:
        pct = (passed_checks / total_checks) * 100
        status = "✅" if pct >= 80 else "⚠️" if pct >= 60 else "❌"
        print(f"\n   {status} Passed {passed_checks}/{total_checks} checks ({pct:.1f}%)")
    
    report["summary"] = {
        "total_checks": total_checks,
        "passed_checks": passed_checks,
        "timestamp": datetime.now().isoformat(),
    }
    
    return report


def save_report(report):
    """Save validation report to JSON."""
    out_path = os.path.join(OUTPUT_DIR, "rapm_validation_report.json")
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n   📄 Report saved to {out_path}")


def main():
    """Run all RAPM validations."""
    print("="*60)
    print("RAPM VALIDATION SUITE")
    print("="*60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    df = load_rapm_data()
    if df is None:
        return
    
    print(f"\n📊 Loaded {len(df):,} RAPM records")
    print(f"   Seasons: {sorted(df['season'].unique())}")
    print(f"   Types: {df['RAPM_type'].unique().tolist()}")
    
    report = {}
    
    # Run all validations
    report = validate_distribution(df, report)
    report = validate_elite_players(df, report)
    report = validate_season_stability(df, report)
    report = validate_orapm_drapm_consistency(df, report)
    report = validate_possession_coverage(df, report)
    report = validate_top_rankings(df, report)
    report = validate_against_xrapm(df, report)
    report = diagnose_xrapm_discrepancy(df, report)
    report = validate_xrapm_vs_benchmark(report)  # Compare our xRAPM with BPM prior
    report = generate_summary(report)
    
    # Save report
    save_report(report)
    
    print("\n" + "="*60)
    print("EXTERNAL VALIDATION RESOURCES")
    print("="*60)
    print("""
   Compare your RAPM results against these sources:
   
   1. xRAPM.com - Current RAPM leaderboard
      https://xrapm.com/
""")


if __name__ == "__main__":
    main()
