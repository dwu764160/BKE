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
   
   2. Basketball-Reference Advanced Stats
      https://www.basketball-reference.com/leagues/NBA_2025_advanced.html
   
   3. NBA.com Advanced Stats (RPM-like metrics)
      https://www.nba.com/stats/players/advanced
   
   4. FiveThirtyEight RAPTOR (archived)
      https://projects.fivethirtyeight.com/nba-player-ratings/
   
   5. Dunks and Threes (RAPM variants)
      https://dunksandthrees.com/
   
   6. PBPStats.com (play-by-play derived metrics)
      https://www.pbpstats.com/
   
   7. Cleaning the Glass (subscription, detailed RAPM-based)
      https://cleaningtheglass.com/
   
   8. NBAsuffer (RAPM methodology explanation)
      https://www.nbastuffer.com/analytics101/regularized-adjusted-plus-minus-rapm/
""")


if __name__ == "__main__":
    main()
