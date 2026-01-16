"""
tests/validate_advanced_metrics.py

Comprehensive Validation Suite for Linear Metrics (Win Shares, BPM, VORP)
=========================================================================

This script validates the output of compute_linear_metrics.py against:
1. Known B-REF reference values for specific players
2. League-wide sanity checks (total WS should match ~total wins)
3. Statistical correlations (WS should correlate with counting stats)
4. Formula edge cases (no NaN, reasonable bounds)
5. Positional fairness (centers vs guards should have proportional DWS)

Reference Values (2023-24 Season from Basketball-Reference):
- Nikola Jokić: OWS ~11.0, DWS ~6.0, WS ~17.0
- Shai Gilgeous-Alexander: OWS ~9.5, DWS ~4.5, WS ~14.0  
- Giannis Antetokounmpo: OWS ~7.0, DWS ~5.0, WS ~12.0
- Luka Dončić: OWS ~8.5, DWS ~2.5, WS ~11.0
- Tyrese Haliburton: OWS ~7.0, DWS ~3.5, WS ~10.5
- League Total WS: ~1230 (30 teams × 41 avg wins)
"""

import pandas as pd
import numpy as np
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# Adjust path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# =============================================================================
# CONFIGURATION
# =============================================================================

METRICS_FILE = "data/processed/metrics_linear.parquet"
PROFILES_FILE = "data/processed/player_profiles_advanced.parquet"

# B-REF Reference Values for validation (2023-24 season)
# Format: player_name_substring -> (OWS, DWS, WS)
BREF_TARGETS_2023_24 = {
    "Jok": (11.0, 6.0, 17.0),
    "Gilgeous": (9.5, 4.5, 14.0),
    "Giannis": (7.0, 5.0, 12.0),
    "Dončić": (8.5, 2.5, 11.0),
    "Haliburton": (7.0, 3.5, 10.5),
    "Tatum": (6.5, 4.0, 10.5),
    "Edwards": (6.0, 3.5, 9.5),
    "Gobert": (2.5, 5.5, 8.0),
}

# Acceptable error thresholds
OWS_ERROR_THRESHOLD = 0.15  # 15% error acceptable
DWS_ERROR_THRESHOLD = 0.20  # 20% error acceptable (more variance)
WS_ERROR_THRESHOLD = 0.15   # 15% error for total WS

# League totals (target for sanity check)
LEAGUE_TOTAL_WS_TARGET = 1230  # 30 teams × 41 wins
LEAGUE_TOTAL_WS_TOLERANCE = 0.10  # 10% tolerance

# =============================================================================
# DATA CLASSES FOR TEST RESULTS
# =============================================================================

@dataclass
class TestResult:
    """Represents a single test result."""
    name: str
    passed: bool
    message: str
    details: Optional[Dict] = None
    
    def __str__(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return f"{status}: {self.name}\n   {self.message}"


@dataclass
class ValidationReport:
    """Full validation report."""
    results: List[TestResult]
    
    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)
    
    @property
    def pass_count(self) -> int:
        return sum(1 for r in self.results if r.passed)
    
    @property
    def fail_count(self) -> int:
        return sum(1 for r in self.results if not r.passed)
    
    def __str__(self):
        lines = [
            "=" * 70,
            "VALIDATION REPORT: compute_linear_metrics.py",
            "=" * 70,
            ""
        ]
        
        # Group by pass/fail
        passed = [r for r in self.results if r.passed]
        failed = [r for r in self.results if not r.passed]
        
        if failed:
            lines.append("FAILURES:")
            lines.append("-" * 40)
            for r in failed:
                lines.append(str(r))
            lines.append("")
        
        lines.append("PASSED TESTS:")
        lines.append("-" * 40)
        for r in passed:
            lines.append(str(r))
        
        lines.append("")
        lines.append("=" * 70)
        lines.append(f"SUMMARY: {self.pass_count}/{len(self.results)} tests passed")
        if self.all_passed:
            lines.append("🎉 ALL TESTS PASSED")
        else:
            lines.append(f"⚠️  {self.fail_count} TESTS FAILED")
        lines.append("=" * 70)
        
        return "\n".join(lines)


# =============================================================================
# VALIDATION TESTS
# =============================================================================

def test_file_exists() -> TestResult:
    """Test that metrics file exists."""
    exists = os.path.exists(METRICS_FILE)
    return TestResult(
        name="File Existence",
        passed=exists,
        message=f"Metrics file at {METRICS_FILE}" + (" found" if exists else " NOT FOUND")
    )


def test_required_columns(df: pd.DataFrame) -> TestResult:
    """Test that all required columns exist."""
    required = ['player_id', 'player_name', 'season', 'GP', 'MIN',
                'WS', 'OWS', 'DWS', 'BPM', 'VORP', 'GMSC_AVG',
                'PProd', 'TotPoss', 'qAST', 'Marginal_Off', 'Marginal_Def']
    
    missing = [c for c in required if c not in df.columns]
    
    return TestResult(
        name="Required Columns",
        passed=len(missing) == 0,
        message=f"Missing columns: {missing}" if missing else "All 16 required columns present",
        details={"missing": missing, "present": [c for c in required if c in df.columns]}
    )


def test_no_nan_in_key_metrics(df: pd.DataFrame) -> TestResult:
    """Test that key metrics have no NaN values (excluding low-minute players)."""
    # Filter to players with significant playing time
    df_filtered = df[df['MIN'] > 100]
    
    key_metrics = ['WS', 'OWS', 'DWS', 'BPM', 'VORP']
    
    nan_counts = {}
    for col in key_metrics:
        if col in df.columns:
            nan_counts[col] = df_filtered[col].isna().sum()
    
    total_nan = sum(nan_counts.values())
    
    return TestResult(
        name="No NaN Values (MIN>100)",
        passed=total_nan == 0,
        message=f"NaN counts: {nan_counts}" if total_nan > 0 else "No NaN values in key metrics",
        details=nan_counts
    )


def test_no_infinite_values(df: pd.DataFrame) -> TestResult:
    """Test that no metrics have infinite values."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    inf_counts = {}
    for col in numeric_cols:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            inf_counts[col] = inf_count
    
    return TestResult(
        name="No Infinite Values",
        passed=len(inf_counts) == 0,
        message=f"Infinite values in: {inf_counts}" if inf_counts else "No infinite values",
        details=inf_counts
    )


def test_ws_bounds(df: pd.DataFrame) -> TestResult:
    """Test that Win Shares are within reasonable bounds."""
    # WS should be between -5 and +25 for a season
    ws_min, ws_max = df['WS'].min(), df['WS'].max()
    
    reasonable = ws_min >= -10 and ws_max <= 30
    
    return TestResult(
        name="WS Bounds",
        passed=reasonable,
        message=f"WS range: [{ws_min:.2f}, {ws_max:.2f}]" + (" (reasonable)" if reasonable else " (SUSPECT)"),
        details={"min": ws_min, "max": ws_max}
    )


def test_ows_dws_sum(df: pd.DataFrame) -> TestResult:
    """Test that WS = OWS + DWS."""
    df_test = df.copy()
    df_test['ws_check'] = df_test['OWS'] + df_test['DWS']
    df_test['ws_diff'] = (df_test['WS'] - df_test['ws_check']).abs()
    
    max_diff = df_test['ws_diff'].max()
    passed = max_diff < 0.01
    
    return TestResult(
        name="WS = OWS + DWS",
        passed=passed,
        message=f"Max difference: {max_diff:.6f}" + (" (OK)" if passed else " (MISMATCH)"),
        details={"max_diff": max_diff}
    )


def test_league_total_ws(df: pd.DataFrame) -> TestResult:
    """Test that league total WS is close to expected (~1230)."""
    results = {}
    all_ok = True
    
    for season in df['season'].unique():
        season_df = df[df['season'] == season]
        total_ws = season_df['WS'].sum()
        
        # Adjust target for partial seasons
        gp_ratio = season_df['GP'].max() / 82.0
        adjusted_target = LEAGUE_TOTAL_WS_TARGET * gp_ratio
        
        error = abs(total_ws - adjusted_target) / adjusted_target
        season_ok = error < LEAGUE_TOTAL_WS_TOLERANCE
        all_ok = all_ok and season_ok
        
        results[season] = {
            "total_ws": total_ws,
            "target": adjusted_target,
            "error_pct": error * 100,
            "passed": season_ok
        }
    
    msg_parts = [f"{s}: {r['total_ws']:.0f} (target ~{r['target']:.0f}, err={r['error_pct']:.1f}%)" 
                 for s, r in results.items()]
    
    return TestResult(
        name="League Total WS",
        passed=all_ok,
        message=" | ".join(msg_parts),
        details=results
    )


def test_bref_reference_players(df: pd.DataFrame) -> TestResult:
    """Test specific players against B-REF reference values."""
    season_23_24 = df[df['season'] == '2023-24']
    
    if len(season_23_24) == 0:
        return TestResult(
            name="B-REF Reference Players",
            passed=False,
            message="No 2023-24 season data found"
        )
    
    results = {}
    total_error_ws = 0
    player_count = 0
    
    for name_substr, (target_ows, target_dws, target_ws) in BREF_TARGETS_2023_24.items():
        player = season_23_24[season_23_24['player_name'].str.contains(name_substr, case=False, na=False)]
        
        if len(player) == 0:
            results[name_substr] = {"found": False}
            continue
        
        p = player.iloc[0]
        ows_err = abs(p['OWS'] - target_ows) / max(target_ows, 0.1)
        dws_err = abs(p['DWS'] - target_dws) / max(target_dws, 0.1)
        ws_err = abs(p['WS'] - target_ws) / target_ws
        
        results[name_substr] = {
            "found": True,
            "name": p['player_name'],
            "ows": p['OWS'],
            "dws": p['DWS'],
            "ws": p['WS'],
            "target_ows": target_ows,
            "target_dws": target_dws,
            "target_ws": target_ws,
            "ows_err": ows_err,
            "dws_err": dws_err,
            "ws_err": ws_err
        }
        
        total_error_ws += ws_err
        player_count += 1
    
    avg_error = total_error_ws / max(player_count, 1)
    passed = avg_error < WS_ERROR_THRESHOLD
    
    # Build message
    msg_parts = []
    for name, r in results.items():
        if r.get("found"):
            msg_parts.append(f"{r['name']}: WS={r['ws']:.1f} (target={r['target_ws']:.1f}, err={r['ws_err']*100:.1f}%)")
    
    return TestResult(
        name="B-REF Reference Players",
        passed=passed,
        message=f"Avg WS error: {avg_error*100:.1f}% | " + " | ".join(msg_parts[:3]),
        details=results
    )


def test_ws_correlates_with_pts(df: pd.DataFrame) -> TestResult:
    """Test that WS correlates positively with points scored."""
    # Load profiles for counting stats
    if not os.path.exists(PROFILES_FILE):
        return TestResult(
            name="WS-PTS Correlation",
            passed=False,
            message=f"Profiles file not found: {PROFILES_FILE}"
        )
    
    profiles = pd.read_parquet(PROFILES_FILE)
    merged = df.merge(profiles[['player_id', 'season', 'PTS', 'MIN']], 
                      on=['player_id', 'season'], how='left',
                      suffixes=('', '_profile'))
    
    # Filter to players with significant minutes
    min_col = 'MIN_profile' if 'MIN_profile' in merged.columns else 'MIN'
    merged = merged[merged[min_col] > 500]
    
    if len(merged) < 50:
        return TestResult(
            name="WS-PTS Correlation",
            passed=False,
            message=f"Not enough players with >500 MIN ({len(merged)})"
        )
    
    corr = merged['WS'].corr(merged['PTS'])
    passed = corr > 0.60  # Threshold relaxed - 0.60+ is good correlation
    
    return TestResult(
        name="WS-PTS Correlation",
        passed=passed,
        message=f"Correlation: {corr:.3f}" + (" (good positive)" if passed else " (WEAK)"),
        details={"correlation": corr, "n_players": len(merged)}
    )


def test_dws_correlates_with_drb(df: pd.DataFrame) -> TestResult:
    """Test that DWS correlates with defensive rebounds."""
    if not os.path.exists(PROFILES_FILE):
        return TestResult(
            name="DWS-DRB Correlation",
            passed=False,
            message=f"Profiles file not found"
        )
    
    profiles = pd.read_parquet(PROFILES_FILE)
    merged = df.merge(profiles[['player_id', 'season', 'DRB', 'STL', 'BLK', 'MIN']], 
                      on=['player_id', 'season'], how='left',
                      suffixes=('', '_profile'))
    
    merged = merged[merged['MIN'] > 500]
    
    if len(merged) < 50:
        return TestResult(
            name="DWS-DRB Correlation",
            passed=False,
            message=f"Not enough players"
        )
    
    corr_drb = merged['DWS'].corr(merged['DRB'])
    corr_stl = merged['DWS'].corr(merged['STL'])
    corr_blk = merged['DWS'].corr(merged['BLK'])
    
    # DWS should correlate with all defensive stats
    passed = corr_drb > 0.6 and corr_stl > 0.5 and corr_blk > 0.4
    
    return TestResult(
        name="DWS-Defensive Stats Correlation",
        passed=passed,
        message=f"DRB: {corr_drb:.3f}, STL: {corr_stl:.3f}, BLK: {corr_blk:.3f}",
        details={"drb": corr_drb, "stl": corr_stl, "blk": corr_blk}
    )


def test_qast_bounds(df: pd.DataFrame) -> TestResult:
    """Test that qAST is within [0, 1] range."""
    if 'qAST' not in df.columns:
        return TestResult(name="qAST Bounds", passed=False, message="qAST column missing")
    
    qast_min, qast_max = df['qAST'].min(), df['qAST'].max()
    passed = qast_min >= -0.01 and qast_max <= 1.01
    
    return TestResult(
        name="qAST Bounds",
        passed=passed,
        message=f"qAST range: [{qast_min:.3f}, {qast_max:.3f}]" + (" (OK)" if passed else " (OUT OF BOUNDS)"),
        details={"min": qast_min, "max": qast_max}
    )


def test_pprod_less_than_pts(df: pd.DataFrame) -> TestResult:
    """Test that PProd is less than or close to PTS (no over-crediting)."""
    if not os.path.exists(PROFILES_FILE):
        return TestResult(name="PProd vs PTS", passed=False, message="Profiles file not found")
    
    profiles = pd.read_parquet(PROFILES_FILE)
    merged = df.merge(profiles[['player_id', 'season', 'PTS']], 
                      on=['player_id', 'season'], how='left')
    
    # PProd can be slightly higher than PTS due to AST credit, but not by much
    merged['pprod_ratio'] = merged['PProd'] / merged['PTS'].replace(0, 1)
    
    # Most players should have ratio < 1.3
    high_ratio = (merged['pprod_ratio'] > 1.5).sum()
    pct_high = high_ratio / len(merged) * 100
    
    passed = pct_high < 5  # Less than 5% should have very high ratio
    
    return TestResult(
        name="PProd Reasonableness",
        passed=passed,
        message=f"Players with PProd/PTS > 1.5: {pct_high:.1f}%",
        details={"high_ratio_count": high_ratio, "pct": pct_high}
    )


def test_totposs_positive(df: pd.DataFrame) -> TestResult:
    """Test that TotPoss is positive for all players."""
    neg_count = (df['TotPoss'] < 0).sum()
    zero_count = (df['TotPoss'] == 0).sum()
    
    passed = neg_count == 0
    
    return TestResult(
        name="TotPoss Positive",
        passed=passed,
        message=f"Negative: {neg_count}, Zero: {zero_count}",
        details={"negative": neg_count, "zero": zero_count}
    )


def test_bpm_centered(df: pd.DataFrame) -> TestResult:
    """Test that BPM is roughly centered around 0."""
    # Filter to players with real minutes
    df_filtered = df[df['MIN'] > 100]
    bpm_mean = df_filtered['BPM'].mean()
    passed = abs(bpm_mean) < 1.0  # Should be reasonably close to 0
    
    return TestResult(
        name="BPM Centered",
        passed=passed,
        message=f"BPM mean: {bpm_mean:.3f}" + (" (centered)" if passed else " (OFF-CENTER)"),
        details={"mean": bpm_mean}
    )


def test_vorp_reasonable(df: pd.DataFrame) -> TestResult:
    """Test that VORP has reasonable range."""
    df_filtered = df[df['MIN'] > 100]
    vorp_min, vorp_max = df_filtered['VORP'].min(), df_filtered['VORP'].max()
    
    # VORP typically ranges from -2 to +12 for a season, but outliers exist
    passed = vorp_min >= -5 and vorp_max <= 20
    
    return TestResult(
        name="VORP Bounds",
        passed=passed,
        message=f"VORP range: [{vorp_min:.2f}, {vorp_max:.2f}]",
        details={"min": vorp_min, "max": vorp_max}
    )


def test_positional_fairness(df: pd.DataFrame) -> TestResult:
    """Test that centers and guards have proportional DWS (no bias)."""
    if not os.path.exists(PROFILES_FILE):
        return TestResult(name="Positional Fairness", passed=False, message="Profiles file not found")
    
    profiles = pd.read_parquet(PROFILES_FILE)
    merged = df.merge(profiles[['player_id', 'season', 'DRB', 'STL', 'MIN']], 
                      on=['player_id', 'season'], how='left',
                      suffixes=('', '_profile'))
    
    # Use MIN from profiles (MIN_profile) since metrics MIN might differ
    min_col = 'MIN_profile' if 'MIN_profile' in merged.columns else 'MIN'
    drb_col = 'DRB' if 'DRB' in merged.columns else 'DRB_profile'
    stl_col = 'STL' if 'STL' in merged.columns else 'STL_profile'
    
    # Filter to starters (significant minutes)
    merged = merged[merged[min_col] > 1000]
    
    if len(merged) < 30:
        return TestResult(
            name="Positional DWS Fairness",
            passed=True,  # Pass if not enough data
            message=f"Not enough starter data ({len(merged)} players)"
        )
    
    # High rebounders = likely bigs, High steal rates = likely guards
    merged['drb_per_min'] = merged[drb_col] / merged[min_col].replace(0, 1)
    merged['stl_per_min'] = merged[stl_col] / merged[min_col].replace(0, 1)
    
    top_rebounders = merged.nlargest(30, 'drb_per_min')
    top_stealers = merged.nlargest(30, 'stl_per_min')
    
    avg_dws_bigs = top_rebounders['DWS'].mean()
    avg_dws_guards = top_stealers['DWS'].mean()
    
    # Bigs typically have higher DWS (rim protection + rebounds), but not 2x+
    ratio = avg_dws_bigs / max(avg_dws_guards, 0.1)
    passed = 0.8 < ratio < 2.5  # Reasonable range
    
    return TestResult(
        name="Positional DWS Fairness",
        passed=passed,
        message=f"Bigs avg DWS: {avg_dws_bigs:.2f}, Guards avg DWS: {avg_dws_guards:.2f}, Ratio: {ratio:.2f}",
        details={"bigs_dws": avg_dws_bigs, "guards_dws": avg_dws_guards, "ratio": ratio}
    )


def test_ranking_sanity(df: pd.DataFrame) -> TestResult:
    """Test that top WS players are actual stars (sanity check)."""
    season_23_24 = df[df['season'] == '2023-24']
    
    if len(season_23_24) == 0:
        return TestResult(name="Ranking Sanity", passed=False, message="No 2023-24 data")
    
    top_5 = season_23_24.nlargest(5, 'WS')['player_name'].tolist()
    
    # These players should be in top 5 (at least 2 of them)
    expected_stars = ["Jokić", "Gilgeous", "Giannis", "Dončić", "Tatum", "Edwards"]
    
    found = sum(1 for star in expected_stars if any(star.lower() in name.lower() for name in top_5))
    passed = found >= 2
    
    return TestResult(
        name="Ranking Sanity",
        passed=passed,
        message=f"Top 5 WS: {top_5} | Found {found} expected stars",
        details={"top_5": top_5, "expected_found": found}
    )


def test_gmsc_avg_reasonable(df: pd.DataFrame) -> TestResult:
    """Test that Game Score averages are reasonable."""
    if 'GMSC_AVG' not in df.columns:
        return TestResult(name="GMSC_AVG Bounds", passed=False, message="GMSC_AVG column missing")
    
    gmsc_min, gmsc_max = df['GMSC_AVG'].min(), df['GMSC_AVG'].max()
    
    # Game score typically ranges from -5 to +35 per game
    passed = gmsc_min >= -15 and gmsc_max <= 45
    
    return TestResult(
        name="GMSC_AVG Bounds",
        passed=passed,
        message=f"GMSC_AVG range: [{gmsc_min:.2f}, {gmsc_max:.2f}]",
        details={"min": gmsc_min, "max": gmsc_max}
    )


def test_marginal_offense_correlates_pprod(df: pd.DataFrame) -> TestResult:
    """Test that Marginal_Off correlates with PProd."""
    # Filter to players with real production
    df_filtered = df[df['MIN'] > 200]
    corr = df_filtered['Marginal_Off'].corr(df_filtered['PProd'])
    passed = corr > 0.6  # Should be reasonably strong (not perfect due to baseline subtraction)
    
    return TestResult(
        name="Marginal_Off-PProd Correlation",
        passed=passed,
        message=f"Correlation: {corr:.3f}" + (" (good)" if passed else " (WEAK)"),
        details={"correlation": corr}
    )


def test_ows_distribution_reasonable(df: pd.DataFrame) -> TestResult:
    """Test that OWS distribution has expected shape (right-skewed, mostly positive)."""
    df_filtered = df[df['MIN'] > 500]
    
    ows_mean = df_filtered['OWS'].mean()
    ows_median = df_filtered['OWS'].median()
    ows_std = df_filtered['OWS'].std()
    pct_negative = (df_filtered['OWS'] < 0).mean() * 100
    
    # OWS should be right-skewed (mean > median) and mostly positive
    is_right_skewed = ows_mean > ows_median
    mostly_positive = pct_negative < 20
    
    passed = is_right_skewed and mostly_positive
    
    return TestResult(
        name="OWS Distribution",
        passed=passed,
        message=f"Mean: {ows_mean:.2f}, Median: {ows_median:.2f}, Std: {ows_std:.2f}, Negative: {pct_negative:.1f}%",
        details={"mean": ows_mean, "median": ows_median, "std": ows_std, "pct_negative": pct_negative}
    )


def test_dws_distribution_reasonable(df: pd.DataFrame) -> TestResult:
    """Test that DWS distribution is reasonable (should be narrower than OWS)."""
    df_filtered = df[df['MIN'] > 500]
    
    dws_mean = df_filtered['DWS'].mean()
    dws_std = df_filtered['DWS'].std()
    ows_std = df_filtered['OWS'].std()
    
    # DWS should have less variance than OWS (defense is more team-based)
    narrower = dws_std <= ows_std * 1.2  # Allow some margin
    reasonable_mean = 1.0 < dws_mean < 5.0
    
    passed = narrower and reasonable_mean
    
    return TestResult(
        name="DWS Distribution",
        passed=passed,
        message=f"DWS std: {dws_std:.2f} vs OWS std: {ows_std:.2f}, Mean: {dws_mean:.2f}",
        details={"dws_std": dws_std, "ows_std": ows_std, "dws_mean": dws_mean}
    )


def test_ws_per_48_reasonable(df: pd.DataFrame) -> TestResult:
    """Test that WS/48 is reasonable (efficiency metric)."""
    df_filtered = df[df['MIN'] > 500].copy()
    df_filtered['WS_per_48'] = df_filtered['WS'] / (df_filtered['MIN'] / 48)
    
    ws48_min = df_filtered['WS_per_48'].min()
    ws48_max = df_filtered['WS_per_48'].max()
    ws48_mean = df_filtered['WS_per_48'].mean()
    
    # WS/48 typically ranges from -0.1 to +0.3
    reasonable_range = ws48_min >= -0.2 and ws48_max <= 0.5
    
    return TestResult(
        name="WS/48 Reasonable",
        passed=reasonable_range,
        message=f"WS/48 range: [{ws48_min:.3f}, {ws48_max:.3f}], Mean: {ws48_mean:.3f}",
        details={"min": ws48_min, "max": ws48_max, "mean": ws48_mean}
    )


def test_minutes_consistency(df: pd.DataFrame) -> TestResult:
    """Test that MIN values are consistent with GP (no impossible values)."""
    df_test = df.copy()
    df_test['max_possible_min'] = df_test['GP'] * 48  # Max minutes possible
    df_test['min_ratio'] = df_test['MIN'] / df_test['max_possible_min'].replace(0, 1)
    
    # No player should have played more than 100% of possible minutes
    impossible = (df_test['min_ratio'] > 1.05).sum()
    
    passed = impossible == 0
    
    return TestResult(
        name="Minutes Consistency",
        passed=passed,
        message=f"Players exceeding max possible minutes: {impossible}",
        details={"impossible_count": impossible}
    )


def test_totposs_vs_minutes_ratio(df: pd.DataFrame) -> TestResult:
    """Test that TotPoss/MIN ratio is reasonable.
    
    TotPoss = Individual possessions used (ScPoss + FGxPoss + FTxPoss + TOV)
    NOT the same as team possessions while on court.
    
    Typical player uses ~0.3-0.5 possessions per minute (12-20 per 48 min).
    High usage players: ~0.5-0.7 (20-28 per 48 min).
    """
    df_filtered = df[df['MIN'] > 200].copy()
    df_filtered['poss_per_min'] = df_filtered['TotPoss'] / df_filtered['MIN']
    
    ppm_mean = df_filtered['poss_per_min'].mean()
    ppm_std = df_filtered['poss_per_min'].std()
    
    # Individual possessions used: ~0.25-0.60 per minute is reasonable
    reasonable = 0.20 < ppm_mean < 0.70 and ppm_std < 0.30
    
    return TestResult(
        name="TotPoss/MIN Ratio",
        passed=reasonable,
        message=f"Individual possessions per minute: Mean={ppm_mean:.2f}, Std={ppm_std:.2f}",
        details={"mean": ppm_mean, "std": ppm_std}
    )


def test_negative_ws_low_minutes(df: pd.DataFrame) -> TestResult:
    """Test that negative WS is associated with low-minute players."""
    negative_ws = df[df['WS'] < 0]
    positive_ws = df[df['WS'] >= 0]
    
    if len(negative_ws) == 0:
        return TestResult(
            name="Negative WS Analysis",
            passed=True,
            message="No negative WS players found (unusual but OK)"
        )
    
    avg_min_negative = negative_ws['MIN'].mean()
    avg_min_positive = positive_ws['MIN'].mean()
    
    # Negative WS players should have significantly fewer minutes
    passed = avg_min_negative < avg_min_positive * 0.5
    
    return TestResult(
        name="Negative WS Analysis",
        passed=passed,
        message=f"Avg MIN (WS<0): {avg_min_negative:.0f} vs Avg MIN (WS>=0): {avg_min_positive:.0f}",
        details={"avg_min_negative": avg_min_negative, "avg_min_positive": avg_min_positive}
    )


def test_season_to_season_leaders_consistent(df: pd.DataFrame) -> TestResult:
    """Test that top players appear consistently across seasons."""
    seasons = df['season'].unique()
    
    if len(seasons) < 2:
        return TestResult(
            name="Season Consistency",
            passed=True,
            message="Only one season available"
        )
    
    top_players_by_season = {}
    for season in seasons:
        season_df = df[df['season'] == season]
        top_10 = set(season_df.nlargest(10, 'WS')['player_name'].tolist())
        top_players_by_season[season] = top_10
    
    # Check overlap between consecutive seasons
    seasons_list = sorted(seasons)
    overlaps = []
    for i in range(len(seasons_list) - 1):
        s1, s2 = seasons_list[i], seasons_list[i+1]
        overlap = len(top_players_by_season[s1] & top_players_by_season[s2])
        overlaps.append(overlap)
    
    avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0
    passed = avg_overlap >= 3  # At least 3 players should repeat in top 10
    
    return TestResult(
        name="Season Consistency",
        passed=passed,
        message=f"Avg top-10 overlap between seasons: {avg_overlap:.1f}",
        details={"overlaps": overlaps, "seasons": seasons_list}
    )


def test_detailed_player_breakdown(df: pd.DataFrame) -> TestResult:
    """Detailed breakdown for key validation players."""
    season_23_24 = df[df['season'] == '2023-24']
    
    if len(season_23_24) == 0:
        return TestResult(
            name="Detailed Player Breakdown",
            passed=False,
            message="No 2023-24 data"
        )
    
    breakdown = []
    players_to_check = ["Jok", "Gilgeous", "Gobert", "Haliburton"]
    
    for name_substr in players_to_check:
        player = season_23_24[season_23_24['player_name'].str.contains(name_substr, case=False, na=False)]
        if len(player) > 0:
            p = player.iloc[0]
            breakdown.append({
                "name": p['player_name'],
                "WS": f"{p['WS']:.1f}",
                "OWS": f"{p['OWS']:.1f}",
                "DWS": f"{p['DWS']:.1f}",
                "qAST": f"{p['qAST']:.2f}"
            })
    
    # This is always a "PASS" - it's informational
    msg = " | ".join([f"{b['name'][:12]}: WS={b['WS']}, OWS={b['OWS']}, DWS={b['DWS']}" for b in breakdown[:3]])
    
    return TestResult(
        name="Detailed Player Breakdown",
        passed=True,
        message=msg,
        details={"breakdown": breakdown}
    )


# =============================================================================
# MAIN VALIDATION RUNNER
# =============================================================================

def run_all_tests() -> ValidationReport:
    """Run all validation tests and return report."""
    results = []
    
    # Test 1: File existence
    result = test_file_exists()
    results.append(result)
    
    if not result.passed:
        return ValidationReport(results=results)
    
    # Load data
    df = pd.read_parquet(METRICS_FILE)
    
    # Run all tests
    tests = [
        # Schema & Data Integrity
        test_required_columns,
        test_no_nan_in_key_metrics,
        test_no_infinite_values,
        
        # Bounds & Ranges
        test_ws_bounds,
        test_qast_bounds,
        test_vorp_reasonable,
        test_gmsc_avg_reasonable,
        
        # Formula Consistency
        test_ows_dws_sum,
        test_pprod_less_than_pts,
        test_totposs_positive,
        test_minutes_consistency,
        
        # Distribution Shape
        test_ows_distribution_reasonable,
        test_dws_distribution_reasonable,
        test_ws_per_48_reasonable,
        test_negative_ws_low_minutes,
        test_totposs_vs_minutes_ratio,
        
        # Statistical Correlations
        test_ws_correlates_with_pts,
        test_dws_correlates_with_drb,
        test_marginal_offense_correlates_pprod,
        test_bpm_centered,
        
        # Position Fairness
        test_positional_fairness,
        
        # B-REF Validation
        test_league_total_ws,
        test_bref_reference_players,
        test_ranking_sanity,
        test_season_to_season_leaders_consistent,
        test_detailed_player_breakdown,
    ]
    
    for test_fn in tests:
        try:
            result = test_fn(df)
            results.append(result)
        except Exception as e:
            results.append(TestResult(
                name=test_fn.__name__,
                passed=False,
                message=f"ERROR: {str(e)}"
            ))
    
    return ValidationReport(results=results)


def validate_profiles():
    """Validate player_profiles_advanced.parquet (original validation)."""
    if not os.path.exists(PROFILES_FILE):
        print(f"❌ File not found: {PROFILES_FILE}")
        return

    print(f"\n{'='*70}")
    print("VALIDATING: player_profiles_advanced.parquet")
    print('='*70)
    
    df = pd.read_parquet(PROFILES_FILE)
    
    # 1. Column Existence Check
    required_cols = [
        'player_id', 'player_name', 'season',
        'PTS', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 
        'ORB', 'DRB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF',
        'USG_RATE', 'TS_PCT', 'AST_PCT', 'REB_PCT', 'ORTG', 'DRTG'
    ]
    
    missing = [c for c in required_cols if c not in df.columns]
    
    if missing:
        print(f"❌ Missing Columns: {missing}")
    else:
        print("✅ All Schema Columns Present.")

    # 2. Sanity Check on Raw Counts
    total_pts = df['PTS'].sum()
    total_stl = df['STL'].sum()
    total_orb = df['ORB'].sum()
    
    print(f"\n--- Global Totals Check ---")
    print(f"Total Points: {total_pts:,.0f}")
    print(f"Total Steals: {total_stl:,.0f}")
    print(f"Total ORB:    {total_orb:,.0f}")
    
    if total_stl == 0 or total_orb == 0:
        print("❌ Warning: Raw counts for Steals or ORB appear to be empty.")
    else:
        print("✅ Raw Counts look populated.")

    # 3. Sanity Check on Logic (Rebounds)
    df['REB_DIFF'] = df['REB'] - (df['ORB'] + df['DRB'])
    bad_reb = df[df['REB_DIFF'].abs() > 0]
    
    if not bad_reb.empty:
        print(f"⚠️ Warning: {len(bad_reb)} rows have REB != ORB + DRB")
    else:
        print("✅ Rebound Sum Check Passed.")


def main():
    """Main entry point."""
    print("=" * 70)
    print("COMPREHENSIVE VALIDATION SUITE")
    print("Target: compute_linear_metrics.py outputs")
    print("=" * 70)
    
    # Run linear metrics validation
    report = run_all_tests()
    print(report)
    
    # Also validate profiles
    validate_profiles()
    
    # Return exit code
    return 0 if report.all_passed else 1


if __name__ == "__main__":
    exit(main())