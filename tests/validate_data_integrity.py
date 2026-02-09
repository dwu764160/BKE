"""
tests/validate_data_integrity.py
Comprehensive data integrity validation against real-world sources.

Compares our PBP-derived data against:
  - NBA.com official advanced stats (already fetched in data/official_stats/)
  - Internal consistency checks (recompute rates from raw totals)
  - Possession count validation
  - USG% reconciliation

Player sample: 60-75 diverse players across archetypes, positions, minutes tiers.

Usage:
    python tests/validate_data_integrity.py [--season 2024-25] [--bref PATH]
"""

import pandas as pd
import numpy as np
import os
import sys
import json
import argparse
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

PROCESSED_DIR = "data/processed"
HISTORICAL_DIR = "data/historical"
OFFICIAL_DIR = "data/official_stats"
OUTPUT_DIR = "data/processed"

# ─────────────────────────────────────────────────────────────────────────────
# 1. DIVERSE PLAYER SAMPLE SELECTION
# ─────────────────────────────────────────────────────────────────────────────
# Manually curated to cover archetypes, positions, minutes tiers, team diversity
#
# Tiers:
#   Star: >2000 min,  Starter: 1500-2000,  Rotation: 800-1500,  Bench: <800
#
# We use player names for matching (both bref + official use names).

VALIDATION_SAMPLE = {
    # ── Ball Dominant Creators (star-level guards/wings) ──
    "Giannis Antetokounmpo":   {"pos": "F",  "tier": "star",   "archetype": "Ball Dominant Creator"},
    "Luka Dončić":             {"pos": "G",  "tier": "star",   "archetype": "Ball Dominant Creator"},
    "Shai Gilgeous-Alexander": {"pos": "G",  "tier": "star",   "archetype": "Ball Dominant Creator"},
    "Jayson Tatum":            {"pos": "F",  "tier": "star",   "archetype": "Ball Dominant Creator"},
    "LeBron James":            {"pos": "F",  "tier": "star",   "archetype": "Ball Dominant Creator"},
    "Anthony Edwards":         {"pos": "G",  "tier": "star",   "archetype": "Ball Dominant Creator"},
    "Donovan Mitchell":        {"pos": "G",  "tier": "star",   "archetype": "Ball Dominant Creator"},
    "De'Aaron Fox":            {"pos": "G",  "tier": "star",   "archetype": "Ball Dominant Creator"},

    # ── All-Around Scorers ──
    "Kevin Durant":            {"pos": "F",  "tier": "star",   "archetype": "All-Around Scorer"},
    "DeMar DeRozan":           {"pos": "F",  "tier": "star",   "archetype": "All-Around Scorer"},
    "Zach LaVine":             {"pos": "G",  "tier": "star",   "archetype": "All-Around Scorer"},
    "Paolo Banchero":          {"pos": "F",  "tier": "star",   "archetype": "All-Around Scorer"},
    "Brandon Ingram":          {"pos": "F",  "tier": "starter", "archetype": "All-Around Scorer"},

    # ── Perimeter Scorers ──
    "Stephen Curry":           {"pos": "G",  "tier": "star",   "archetype": "Perimeter Scorer"},
    "Damian Lillard":          {"pos": "G",  "tier": "star",   "archetype": "Perimeter Scorer"},
    "Devin Booker":            {"pos": "G",  "tier": "star",   "archetype": "Perimeter Scorer"},
    "CJ McCollum":             {"pos": "G",  "tier": "starter", "archetype": "Perimeter Scorer"},

    # ── Interior Scorers ──
    "Nikola Jokić":            {"pos": "C",  "tier": "star",   "archetype": "Interior Scorer"},
    "Joel Embiid":             {"pos": "C",  "tier": "star",   "archetype": "Interior Scorer"},
    "Zion Williamson":         {"pos": "F",  "tier": "starter", "archetype": "Interior Scorer"},
    "Julius Randle":           {"pos": "F",  "tier": "starter", "archetype": "Interior Scorer"},

    # ── Off-Ball Finishers ──
    "Jarrett Allen":           {"pos": "C",  "tier": "starter", "archetype": "Off-Ball Finisher"},
    "Daniel Gafford":          {"pos": "C",  "tier": "starter", "archetype": "Off-Ball Finisher"},
    "Clint Capela":            {"pos": "C",  "tier": "starter", "archetype": "Off-Ball Finisher"},

    # ── PnR Rolling Bigs ──
    "Anthony Davis":           {"pos": "C",  "tier": "star",   "archetype": "PnR Rolling Big"},
    "Domantas Sabonis":        {"pos": "C",  "tier": "star",   "archetype": "PnR Rolling Big"},
    "Bam Adebayo":             {"pos": "C",  "tier": "star",   "archetype": "PnR Rolling Big"},
    "Alperen Şengün":          {"pos": "C",  "tier": "star",   "archetype": "PnR Rolling Big"},

    # ── Off-Ball Stationary Shooters ──
    "Klay Thompson":           {"pos": "G",  "tier": "starter", "archetype": "Off-Ball Stationary Shooter"},
    "Buddy Hield":             {"pos": "G",  "tier": "starter", "archetype": "Off-Ball Stationary Shooter"},
    "Corey Kispert":           {"pos": "F",  "tier": "rotation", "archetype": "Off-Ball Stationary Shooter"},
    "Michael Porter Jr.":      {"pos": "F",  "tier": "starter", "archetype": "Off-Ball Stationary Shooter"},

    # ── Off-Ball Movement Shooters ──
    "Desmond Bane":            {"pos": "G",  "tier": "starter", "archetype": "Off-Ball Movement Shooter"},
    "Bogdan Bogdanović":       {"pos": "G",  "tier": "rotation", "archetype": "Off-Ball Movement Shooter"},

    # ── Ballhandlers / Connectors ──
    "Tyrese Haliburton":       {"pos": "G",  "tier": "star",   "archetype": "Ballhandler"},
    "Jalen Brunson":           {"pos": "G",  "tier": "star",   "archetype": "Ballhandler"},
    "Chris Paul":              {"pos": "G",  "tier": "rotation", "archetype": "Connector"},
    "Draymond Green":          {"pos": "F",  "tier": "starter", "archetype": "Connector"},
    "Fred VanVleet":           {"pos": "G",  "tier": "starter", "archetype": "Connector"},

    # ── PnR Popping Bigs ──
    "Karl-Anthony Towns":      {"pos": "C",  "tier": "star",   "archetype": "PnR Popping Big"},
    "Brook Lopez":             {"pos": "C",  "tier": "starter", "archetype": "PnR Popping Big"},
    "Kristaps Porziņģis":      {"pos": "C",  "tier": "starter", "archetype": "PnR Popping Big"},

    # ── Defensive Specialists / Low-usage guys ──
    "Rudy Gobert":             {"pos": "C",  "tier": "starter", "archetype": "Off-Ball Finisher"},
    "Herb Jones":              {"pos": "F",  "tier": "starter", "archetype": "Connector"},
    "Jrue Holiday":            {"pos": "G",  "tier": "starter", "archetype": "Connector"},
    "Derrick White":           {"pos": "G",  "tier": "starter", "archetype": "Off-Ball Stationary Shooter"},

    # ── Low-minutes / Bench players ──
    "Gary Payton II":          {"pos": "G",  "tier": "bench",    "archetype": "Connector"},
    "Luka Garza":              {"pos": "C",  "tier": "bench",    "archetype": "Off-Ball Finisher"},
    "Jordan Clarkson":         {"pos": "G",  "tier": "rotation", "archetype": "All-Around Scorer"},
    "Aaron Holiday":           {"pos": "G",  "tier": "bench",    "archetype": "Ballhandler"},
    "Dominick Barlow":         {"pos": "F",  "tier": "bench",    "archetype": "Off-Ball Finisher"},

    # ── Rising Stars / Young Players ──
    "Victor Wembanyama":       {"pos": "C",  "tier": "star",   "archetype": "Interior Scorer"},
    "Chet Holmgren":           {"pos": "C",  "tier": "starter", "archetype": "PnR Popping Big"},
    "Jalen Williams":          {"pos": "F",  "tier": "star",   "archetype": "All-Around Scorer"},
    "Evan Mobley":             {"pos": "C",  "tier": "star",   "archetype": "PnR Rolling Big"},
    "Franz Wagner":            {"pos": "F",  "tier": "star",   "archetype": "All-Around Scorer"},
    "Scottie Barnes":          {"pos": "F",  "tier": "star",   "archetype": "Ball Dominant Creator"},
    "Trae Young":              {"pos": "G",  "tier": "star",   "archetype": "Ball Dominant Creator"},
    "Ja Morant":               {"pos": "G",  "tier": "star",   "archetype": "Ball Dominant Creator"},

    # ── Veteran role players (mid-tier minutes) ──
    "Tobias Harris":           {"pos": "F",  "tier": "starter", "archetype": "All-Around Scorer"},
    "Marcus Smart":            {"pos": "G",  "tier": "starter", "archetype": "Connector"},
    "Bobby Portis":            {"pos": "F",  "tier": "rotation", "archetype": "Off-Ball Finisher"},
    "Aaron Gordon":            {"pos": "F",  "tier": "starter", "archetype": "Off-Ball Finisher"},
    "Brandin Podziemski":      {"pos": "G",  "tier": "rotation", "archetype": "Connector"},
}

# ─────────────────────────────────────────────────────────────────────────────
# Data Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_our_profiles(season):
    """Load our PBP-computed player profiles."""
    df = pd.read_parquet(f"{PROCESSED_DIR}/player_profiles_advanced.parquet")
    df = df[df['season'] == season].copy()
    df['player_id'] = df['player_id'].astype(str)
    return df


def load_our_archetypes(season):
    """Load archetype classifications (has the USG_PCT with simplified formula)."""
    df = pd.read_parquet(f"{PROCESSED_DIR}/player_archetypes.parquet")
    df = df[df['SEASON'] == season].copy()
    df['player_id'] = df['PLAYER_ID'].astype(str)
    return df


def load_official_stats(season):
    """Load NBA.com official advanced stats."""
    path = f"{OFFICIAL_DIR}/official_advanced_{season}.parquet"
    if not os.path.exists(path):
        print(f"❌ Official stats not found: {path}")
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df['PLAYER_ID'] = df['PLAYER_ID'].astype(str)
    return df


def load_bios():
    """Load player bio data."""
    path = f"{HISTORICAL_DIR}/players.parquet"
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df['player_id'] = df['id'].astype(str) if 'id' in df.columns else df['player_id'].astype(str)
    return df


def fuzzy_match(name, name_series):
    """Find best match for a player name in a series."""
    # Try exact
    exact = name_series[name_series == name]
    if len(exact) > 0:
        return exact.index[0]
    # Try contains (handles accented chars)
    import unicodedata
    name_ascii = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('utf-8')
    for idx, n in name_series.items():
        n_ascii = unicodedata.normalize('NFKD', str(n)).encode('ASCII', 'ignore').decode('utf-8')
        if n_ascii == name_ascii:
            return idx
    # Try partial
    for idx, n in name_series.items():
        if name.lower() in str(n).lower() or str(n).lower() in name.lower():
            return idx
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 2. TIER 1: BOX SCORE TOTALS VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_box_score_totals(profiles, official, bref_df=None):
    """Compare box score totals (should be exact or near-exact)."""
    print("\n" + "=" * 100)
    print("TIER 1: BOX SCORE TOTALS VALIDATION")
    print("=" * 100)
    print("Comparing our PBP-aggregated totals vs NBA.com official stats")
    print("Expected: EXACT match for counting stats (FGM, FGA, PTS, etc.)")
    print()

    # Stats to compare (our column -> official column)
    box_cols = {
        'GP':  'GP',
        'FGM': 'FGM',
        'FGA': 'FGA',
    }

    results = []
    mismatches = 0

    for player_name, meta in VALIDATION_SAMPLE.items():
        # Find in our data
        our_idx = fuzzy_match(player_name, profiles['player_name'])
        if our_idx is None:
            results.append({'player': player_name, 'status': 'NOT_FOUND_OURS'})
            continue
        our = profiles.loc[our_idx]

        # Find in official
        off_idx = fuzzy_match(player_name, official['PLAYER_NAME'])
        if off_idx is None:
            results.append({'player': player_name, 'status': 'NOT_FOUND_OFFICIAL'})
            continue
        off = official.loc[off_idx]

        row = {'player': player_name, 'status': 'OK'}
        any_mismatch = False

        for our_col, off_col in box_cols.items():
            our_val = our.get(our_col, np.nan)
            off_val = off.get(off_col, np.nan)
            if pd.notna(our_val) and pd.notna(off_val):
                diff = abs(float(our_val) - float(off_val))
                row[f'{our_col}_ours'] = our_val
                row[f'{our_col}_official'] = off_val
                row[f'{our_col}_diff'] = diff
                if diff > 0:
                    any_mismatch = True

        # Minutes comparison (we have total, official has per-game)
        our_min = our.get('MIN', 0)
        off_mpg = off.get('MIN', 0)
        off_gp = off.get('GP', 1)
        off_total_min = off_mpg * off_gp
        min_diff = abs(our_min - off_total_min)
        min_pct_diff = (min_diff / off_total_min * 100) if off_total_min > 0 else 0
        row['MIN_ours'] = round(our_min, 1)
        row['MIN_official'] = round(off_total_min, 1)
        row['MIN_diff'] = round(min_diff, 1)
        row['MIN_pct_diff'] = round(min_pct_diff, 2)

        if any_mismatch:
            row['status'] = 'MISMATCH'
            mismatches += 1

        results.append(row)

    # Print results
    found = [r for r in results if r['status'] != 'NOT_FOUND_OURS' and r['status'] != 'NOT_FOUND_OFFICIAL']
    not_found = [r for r in results if r['status'] in ('NOT_FOUND_OURS', 'NOT_FOUND_OFFICIAL')]

    print(f"{'Player':<30} {'GP':>5} {'FGM':>7} {'FGA':>7} {'MIN Diff':>10} {'%':>6}  Status")
    print("-" * 80)

    for r in found:
        gp_flag = "✅" if r.get('GP_diff', 0) == 0 else "❌"
        fgm_flag = "✅" if r.get('FGM_diff', 0) == 0 else "❌"
        fga_flag = "✅" if r.get('FGA_diff', 0) == 0 else "❌"
        min_flag = "✅" if r.get('MIN_pct_diff', 0) < 1.0 else "⚠️" if r.get('MIN_pct_diff', 0) < 3.0 else "❌"

        print(f"{r['player']:<30} {gp_flag}    {fgm_flag}     {fga_flag}     "
              f"{r.get('MIN_diff', 0):>8.1f} {r.get('MIN_pct_diff', 0):>5.1f}%  "
              f"{'✅' if r.get('status') == 'OK' else '⚠️'}")

    if not_found:
        print(f"\n  ⚠️ {len(not_found)} players not found:")
        for r in not_found:
            print(f"     {r['player']} — {r['status']}")

    # Summary
    all_gp_match = all(r.get('GP_diff', 0) == 0 for r in found)
    all_fgm_match = all(r.get('FGM_diff', 0) == 0 for r in found)
    all_fga_match = all(r.get('FGA_diff', 0) == 0 for r in found)
    min_diffs = [r.get('MIN_pct_diff', 0) for r in found]

    print(f"\n  SUMMARY ({len(found)} players matched):")
    print(f"    GP match:  {'✅ ALL EXACT' if all_gp_match else '❌ MISMATCHES'}")
    print(f"    FGM match: {'✅ ALL EXACT' if all_fgm_match else '❌ MISMATCHES'}")
    print(f"    FGA match: {'✅ ALL EXACT' if all_fga_match else '❌ MISMATCHES'}")
    if min_diffs:
        print(f"    MIN drift: mean {np.mean(min_diffs):.2f}%, max {np.max(min_diffs):.2f}%")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3. TIER 2: INTERNAL RATE STAT CONSISTENCY
# ─────────────────────────────────────────────────────────────────────────────

def validate_internal_consistency(profiles):
    """Recompute rate stats from raw totals and check they match stored values."""
    print("\n" + "=" * 100)
    print("TIER 2: INTERNAL RATE STAT CONSISTENCY")
    print("=" * 100)
    print("Recomputing rate stats from raw totals in our own data")
    print("Expected: EXACT match (same formula, same inputs)")
    print()

    results = []

    for player_name, meta in VALIDATION_SAMPLE.items():
        idx = fuzzy_match(player_name, profiles['player_name'])
        if idx is None:
            continue
        row = profiles.loc[idx]

        checks = {}

        # 1. PTS identity: PTS = (FGM - FG3M)*2 + FG3M*3 + FTM
        fgm = row.get('FGM', 0) or 0
        fg3m = row.get('FG3M', 0) or 0
        ftm = row.get('FTM', 0) or 0
        pts_computed = int((fgm - fg3m) * 2 + fg3m * 3 + ftm)
        pts_stored = int(row.get('PTS', 0))
        checks['PTS_identity'] = abs(pts_computed - pts_stored)

        # 2. REB identity: REB = ORB + DRB
        orb = row.get('ORB', 0) or 0
        drb = row.get('DRB', 0) or 0
        reb_computed = orb + drb
        reb_stored = row.get('REB', 0) or 0
        checks['REB_identity'] = abs(reb_computed - reb_stored)

        # 3. TS% recompute
        fga = row.get('FGA', 0) or 0
        fta = row.get('FTA', 0) or 0
        if fga + 0.44 * fta > 0:
            ts_recomputed = pts_stored / (2 * (fga + 0.44 * fta))
            ts_stored = row.get('TS_PCT', 0) or 0
            checks['TS_PCT_diff'] = abs(ts_recomputed - ts_stored)
        else:
            checks['TS_PCT_diff'] = 0

        # 4. eFG% recompute
        if fga > 0:
            efg_recomputed = (fgm + 0.5 * fg3m) / fga
            efg_stored = row.get('EFG_PCT', 0) or 0
            checks['EFG_PCT_diff'] = abs(efg_recomputed - efg_stored)
        else:
            checks['EFG_PCT_diff'] = 0

        # 5. FT_RATE recompute
        if fga > 0:
            ft_rate_recomputed = fta / fga
            ft_rate_stored = row.get('FT_RATE', 0) or 0
            checks['FT_RATE_diff'] = abs(ft_rate_recomputed - ft_rate_stored)
        else:
            checks['FT_RATE_diff'] = 0

        # 6. Seconds accounting: SECONDS_OFF + SECONDS_DEF ≈ MIN * 60
        sec_off = row.get('SECONDS_OFF', 0) or 0
        sec_def = row.get('SECONDS_DEF', 0) or 0
        min_val = row.get('MIN', 0) or 0
        expected_sec = min_val * 60
        actual_sec = sec_off + sec_def
        checks['seconds_accounting_diff'] = abs(actual_sec - expected_sec)
        checks['seconds_accounting_pct'] = (abs(actual_sec - expected_sec) / expected_sec * 100) if expected_sec > 0 else 0

        results.append({
            'player': player_name,
            **checks
        })

    # Print
    print(f"{'Player':<30} {'PTS':>5} {'REB':>5} {'TS%':>8} {'eFG%':>8} {'FT_R':>8} {'Sec%':>8}")
    print("-" * 80)

    for r in results:
        pts_ok = "✅" if r['PTS_identity'] == 0 else f"❌ Δ{r['PTS_identity']}"
        reb_ok = "✅" if r['REB_identity'] == 0 else f"❌ Δ{r['REB_identity']}"
        ts_ok = "✅" if r['TS_PCT_diff'] < 0.001 else f"⚠️ {r['TS_PCT_diff']:.4f}"
        efg_ok = "✅" if r['EFG_PCT_diff'] < 0.001 else f"⚠️ {r['EFG_PCT_diff']:.4f}"
        ftr_ok = "✅" if r['FT_RATE_diff'] < 0.001 else f"⚠️ {r['FT_RATE_diff']:.4f}"
        sec_ok = "✅" if r['seconds_accounting_pct'] < 0.5 else f"⚠️ {r['seconds_accounting_pct']:.2f}%"

        print(f"{r['player']:<30} {pts_ok:>5} {reb_ok:>5} {ts_ok:>8} {efg_ok:>8} {ftr_ok:>8} {sec_ok:>8}")

    # Summary
    pts_all_ok = all(r['PTS_identity'] == 0 for r in results)
    reb_all_ok = all(r['REB_identity'] == 0 for r in results)
    ts_max_err = max(r['TS_PCT_diff'] for r in results) if results else 0
    sec_max_err = max(r['seconds_accounting_pct'] for r in results) if results else 0

    print(f"\n  SUMMARY ({len(results)} players):")
    print(f"    PTS identity:     {'✅ ALL PASS' if pts_all_ok else '❌ FAILURES'}")
    print(f"    REB identity:     {'✅ ALL PASS' if reb_all_ok else '❌ FAILURES'}")
    print(f"    TS% max error:    {ts_max_err:.6f} {'✅' if ts_max_err < 0.001 else '⚠️'}")
    print(f"    Seconds max err:  {sec_max_err:.2f}% {'✅' if sec_max_err < 1.0 else '⚠️'}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 4. TIER 3: RATE STATS VS NBA.COM OFFICIAL
# ─────────────────────────────────────────────────────────────────────────────

def validate_rate_stats_vs_official(profiles, official):
    """Compare our computed rate stats vs NBA.com official values."""
    print("\n" + "=" * 100)
    print("TIER 3: RATE STATS VS NBA.COM OFFICIAL")
    print("=" * 100)
    print("Our PBP-derived rate stats vs NBA.com advanced stats")
    print()

    results = []

    for player_name, meta in VALIDATION_SAMPLE.items():
        our_idx = fuzzy_match(player_name, profiles['player_name'])
        off_idx = fuzzy_match(player_name, official['PLAYER_NAME'])
        if our_idx is None or off_idx is None:
            continue
        our = profiles.loc[our_idx]
        off = official.loc[off_idx]

        row = {'player': player_name, 'tier': meta['tier']}

        # TS% (our is decimal, official is decimal)
        our_ts = our.get('TS_PCT', np.nan)
        off_ts = off.get('TS_PCT', np.nan)
        if pd.notna(our_ts) and pd.notna(off_ts):
            row['TS_ours'] = round(our_ts, 4)
            row['TS_official'] = round(off_ts, 4)
            row['TS_diff'] = round(abs(our_ts - off_ts), 4)

        # eFG% 
        our_efg = our.get('EFG_PCT', np.nan)
        off_efg = off.get('EFG_PCT', np.nan)
        if pd.notna(our_efg) and pd.notna(off_efg):
            row['EFG_ours'] = round(our_efg, 4)
            row['EFG_official'] = round(off_efg, 4)
            row['EFG_diff'] = round(abs(our_efg - off_efg), 4)

        # USG% — our USG_RATE vs official USG_PCT
        our_usg = our.get('USG_RATE', np.nan)  # This is 0-100 scale
        off_usg = off.get('USG_PCT', np.nan)   # This is 0-1 scale
        if pd.notna(our_usg) and pd.notna(off_usg):
            off_usg_100 = off_usg * 100 if off_usg < 1.0 else off_usg
            our_usg_100 = our_usg if our_usg > 1.0 else our_usg * 100
            row['USG_ours'] = round(our_usg_100, 2)
            row['USG_official'] = round(off_usg_100, 2)
            row['USG_diff'] = round(abs(our_usg_100 - off_usg_100), 2)

        # ORTG
        our_ortg = our.get('ORTG', np.nan)
        off_ortg = off.get('OFF_RATING', np.nan)
        if pd.notna(our_ortg) and pd.notna(off_ortg):
            row['ORTG_ours'] = round(our_ortg, 1)
            row['ORTG_official'] = round(off_ortg, 1)
            row['ORTG_diff'] = round(our_ortg - off_ortg, 1)

        # DRTG
        our_drtg = our.get('DRTG', np.nan)
        off_drtg = off.get('DEF_RATING', np.nan)
        if pd.notna(our_drtg) and pd.notna(off_drtg):
            row['DRTG_ours'] = round(our_drtg, 1)
            row['DRTG_official'] = round(off_drtg, 1)
            row['DRTG_diff'] = round(our_drtg - off_drtg, 1)

        # NET_RTG
        our_net = our.get('NET_RTG', np.nan)
        off_net = off.get('NET_RATING', np.nan)
        if pd.notna(our_net) and pd.notna(off_net):
            row['NET_ours'] = round(our_net, 1)
            row['NET_official'] = round(off_net, 1)
            row['NET_diff'] = round(our_net - off_net, 1)

        # REB_PCT
        our_reb_pct = our.get('REB_PCT', np.nan)
        off_reb_pct = off.get('REB_PCT', np.nan)
        if pd.notna(our_reb_pct) and pd.notna(off_reb_pct):
            off_reb_100 = off_reb_pct * 100 if off_reb_pct < 1.0 else off_reb_pct
            our_reb_100 = our_reb_pct if our_reb_pct > 1.0 else our_reb_pct * 100
            row['REB_PCT_ours'] = round(our_reb_100, 1)
            row['REB_PCT_official'] = round(off_reb_100, 1)
            row['REB_PCT_diff'] = round(abs(our_reb_100 - off_reb_100), 1)

        # AST_PCT
        our_ast_pct = our.get('AST_PCT', np.nan)
        off_ast_pct = off.get('AST_PCT', np.nan)
        if pd.notna(our_ast_pct) and pd.notna(off_ast_pct):
            off_ast_100 = off_ast_pct * 100 if off_ast_pct < 1.0 else off_ast_pct
            our_ast_100 = our_ast_pct if our_ast_pct > 1.0 else our_ast_pct * 100
            row['AST_PCT_ours'] = round(our_ast_100, 1)
            row['AST_PCT_official'] = round(off_ast_100, 1)
            row['AST_PCT_diff'] = round(abs(our_ast_100 - off_ast_100), 1)

        results.append(row)

    # Print Rate Stats Table
    print(f"\n{'Player':<28} {'TS%':>12} {'eFG%':>12} {'USG%':>14} {'ORTG':>14} {'DRTG':>14} {'NET':>12}")
    print("-" * 110)
    for r in results:
        ts_flag = "=" if r.get('TS_diff', 0) < 0.003 else f"+{r.get('TS_diff',0):.3f}"
        efg_flag = "=" if r.get('EFG_diff', 0) < 0.003 else f"+{r.get('EFG_diff',0):.3f}"
        usg_str = f"{r.get('USG_ours', '?')}/{r.get('USG_official', '?')}" if 'USG_ours' in r else "N/A"
        ortg_str = f"{r.get('ORTG_diff', 0):+.1f}" if 'ORTG_diff' in r else "N/A"
        drtg_str = f"{r.get('DRTG_diff', 0):+.1f}" if 'DRTG_diff' in r else "N/A"
        net_str = f"{r.get('NET_diff', 0):+.1f}" if 'NET_diff' in r else "N/A"

        print(f"{r['player']:<28} {ts_flag:>12} {efg_flag:>12} {usg_str:>14} {ortg_str:>14} {drtg_str:>14} {net_str:>12}")

    # Aggregate stats
    usg_diffs = [r['USG_diff'] for r in results if 'USG_diff' in r]
    ortg_diffs = [r['ORTG_diff'] for r in results if 'ORTG_diff' in r]
    drtg_diffs = [r['DRTG_diff'] for r in results if 'DRTG_diff' in r]
    net_diffs = [r['NET_diff'] for r in results if 'NET_diff' in r]
    reb_diffs = [r['REB_PCT_diff'] for r in results if 'REB_PCT_diff' in r]

    print(f"\n  SUMMARY ({len(results)} players):")
    if usg_diffs:
        print(f"    USG%:     mean Δ{np.mean(usg_diffs):.2f}  max Δ{np.max(usg_diffs):.2f}  {'⚠️ SYSTEMATIC BIAS' if np.mean(usg_diffs) > 1.0 else '✅'}")
    if ortg_diffs:
        print(f"    ORTG:     mean Δ{np.mean(ortg_diffs):+.1f}  max Δ{np.max(np.abs(ortg_diffs)):+.1f}  {'⚠️ SYSTEMATIC BIAS' if abs(np.mean(ortg_diffs)) > 1.0 else '✅'}")
    if drtg_diffs:
        print(f"    DRTG:     mean Δ{np.mean(drtg_diffs):+.1f}  max Δ{np.max(np.abs(drtg_diffs)):+.1f}  {'⚠️ SYSTEMATIC BIAS' if abs(np.mean(drtg_diffs)) > 1.0 else '✅'}")
    if net_diffs:
        print(f"    NET_RTG:  mean Δ{np.mean(net_diffs):+.1f}  max Δ{np.max(np.abs(net_diffs)):+.1f}")
    if reb_diffs:
        print(f"    REB_PCT:  mean Δ{np.mean(reb_diffs):.1f}  max Δ{np.max(reb_diffs):.1f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 5. TIER 4: POSSESSION COUNT VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_possessions(profiles, official):
    """Compare our PBP-derived possession counts vs NBA.com possessions."""
    print("\n" + "=" * 100)
    print("TIER 4: POSSESSION COUNT VALIDATION")
    print("=" * 100)
    print("Our PBP-derived POSS_OFF vs NBA.com POSS")
    print("This is the most likely source of ORTG/DRTG/USG drift")
    print()

    results = []

    for player_name, meta in VALIDATION_SAMPLE.items():
        our_idx = fuzzy_match(player_name, profiles['player_name'])
        off_idx = fuzzy_match(player_name, official['PLAYER_NAME'])
        if our_idx is None or off_idx is None:
            continue
        our = profiles.loc[our_idx]
        off = official.loc[off_idx]

        our_poss = our.get('POSS_OFF', np.nan)
        off_poss = off.get('POSS', np.nan)

        if pd.notna(our_poss) and pd.notna(off_poss) and off_poss > 0:
            diff = our_poss - off_poss
            pct_diff = (diff / off_poss) * 100
            results.append({
                'player': player_name,
                'tier': meta['tier'],
                'our_poss': int(our_poss),
                'official_poss': int(off_poss),
                'diff': int(diff),
                'pct_diff': round(pct_diff, 2),
                'min': round(our.get('MIN', 0), 0),
            })

    # Sort by pct_diff for easy scanning
    results.sort(key=lambda x: x['pct_diff'])

    print(f"{'Player':<30} {'Tier':<10} {'MIN':>6} {'Our POSS':>10} {'NBA POSS':>10} {'Diff':>8} {'%':>8}")
    print("-" * 95)

    for r in results:
        flag = "✅" if abs(r['pct_diff']) < 2.0 else "⚠️" if abs(r['pct_diff']) < 5.0 else "❌"
        print(f"{flag} {r['player']:<28} {r['tier']:<10} {r['min']:>6.0f} {r['our_poss']:>10} {r['official_poss']:>10} {r['diff']:>8} {r['pct_diff']:>7.1f}%")

    if results:
        pct_diffs = [r['pct_diff'] for r in results]
        print(f"\n  SUMMARY ({len(results)} players):")
        print(f"    Mean possession drift: {np.mean(pct_diffs):+.2f}%")
        print(f"    Median:                {np.median(pct_diffs):+.2f}%")
        print(f"    Std:                   {np.std(pct_diffs):.2f}%")
        print(f"    Range:                 [{np.min(pct_diffs):.1f}%, {np.max(pct_diffs):.1f}%]")
        
        # Detect systematic bias
        if abs(np.mean(pct_diffs)) > 1.0:
            direction = "UNDER-counting" if np.mean(pct_diffs) < 0 else "OVER-counting"
            print(f"\n  ⚠️ SYSTEMATIC BIAS DETECTED: We are {direction} possessions by ~{abs(np.mean(pct_diffs)):.1f}%")
            print(f"     This directly explains ORTG/DRTG inflation/deflation")
            
            # Explain the math
            avg_poss_diff = np.mean([r['diff'] for r in results])
            print(f"     Average possession difference: {avg_poss_diff:+.0f}")
            
            if np.mean(pct_diffs) < 0:
                print(f"     Lower possessions → higher per-100-possession rates → inflated ORTG/DRTG")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 6. USG% RECONCILIATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_usg_reconciliation(profiles, archetypes, official):
    """Reconcile all three USG% versions."""
    print("\n" + "=" * 100)
    print("TIER 5: USG% RECONCILIATION")
    print("=" * 100)
    print("Three USG% sources:")
    print("  1. USG_RATE (profiles):    (FGA + 0.44*FTA + TOV) / TEAM_PLAYS_ON_COURT * 100")
    print("  2. USG_PCT  (archetypes):  (FGA + 0.44*FTA + TOV) * 2.4 / (MIN * 5)")
    print("  3. USG_PCT  (NBA.com):     Official formula with actual team stats")
    print()

    results = []

    for player_name, meta in VALIDATION_SAMPLE.items():
        prof_idx = fuzzy_match(player_name, profiles['player_name'])
        arch_idx = fuzzy_match(player_name, archetypes['PLAYER_NAME']) if 'PLAYER_NAME' in archetypes.columns else None
        off_idx = fuzzy_match(player_name, official['PLAYER_NAME'])

        if prof_idx is None or off_idx is None:
            continue

        our_prof = profiles.loc[prof_idx]
        off_row = official.loc[off_idx]

        prof_usg = our_prof.get('USG_RATE', np.nan)
        off_usg = off_row.get('USG_PCT', np.nan)
        off_usg_100 = off_usg * 100 if pd.notna(off_usg) and off_usg < 1.0 else off_usg

        arch_usg = np.nan
        if arch_idx is not None:
            arch_row = archetypes.loc[arch_idx]
            arch_usg_raw = arch_row.get('USG_PCT', np.nan)
            arch_usg = arch_usg_raw * 100 if pd.notna(arch_usg_raw) and arch_usg_raw < 1.0 else arch_usg_raw

        if pd.notna(prof_usg) and pd.notna(off_usg_100):
            results.append({
                'player': player_name,
                'profile_usg': round(prof_usg, 2),
                'archetype_usg': round(arch_usg, 2) if pd.notna(arch_usg) else None,
                'official_usg': round(off_usg_100, 2),
                'prof_vs_official': round(prof_usg - off_usg_100, 2),
                'arch_vs_official': round(arch_usg - off_usg_100, 2) if pd.notna(arch_usg) else None,
            })

    print(f"{'Player':<28} {'Profile':>9} {'Archetype':>10} {'NBA.com':>9} {'Prof-Off':>10} {'Arch-Off':>10}")
    print("-" * 85)
    for r in results:
        prof_flag = "✅" if abs(r['prof_vs_official']) < 1.0 else "⚠️"
        arch_flag = "✅" if r.get('arch_vs_official') is not None and abs(r['arch_vs_official']) < 1.0 else "⚠️"
        arch_str = f"{r['archetype_usg']:.1f}" if r['archetype_usg'] is not None else "N/A"
        arch_diff_str = f"{r['arch_vs_official']:+.1f}" if r.get('arch_vs_official') is not None else "N/A"

        print(f"{r['player']:<28} {r['profile_usg']:>8.1f} {arch_str:>10} {r['official_usg']:>8.1f}  {prof_flag}{r['prof_vs_official']:>+8.1f}  {arch_flag}{arch_diff_str:>8}")

    if results:
        prof_diffs = [r['prof_vs_official'] for r in results]
        arch_diffs = [r['arch_vs_official'] for r in results if r.get('arch_vs_official') is not None]
        print(f"\n  SUMMARY:")
        print(f"    Profile USG vs Official:   mean Δ{np.mean(prof_diffs):+.2f}, std {np.std(prof_diffs):.2f}")
        print(f"    Archetype USG vs Official: mean Δ{np.mean(arch_diffs):+.2f}, std {np.std(arch_diffs):.2f}")
        print()
        print(f"    ROOT CAUSE ANALYSIS:")
        print(f"    Profile formula uses TEAM_PLAYS_ON_COURT (our PBP-derived → possession count issue)")
        print(f"    Archetype formula uses constant 2.4 multiplier (simplified, always biased upward)")
        print(f"    NBA.com uses actual team game-by-game totals")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 7. LEAGUE-WIDE POSSESSION SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

def validate_league_possessions(profiles, official):
    """League-wide possession sanity: total player possessions ≈ league_poss * 5."""
    print("\n" + "=" * 100)
    print("TIER 6: LEAGUE-WIDE POSSESSION SANITY")
    print("=" * 100)

    # Our data: sum all POSS_OFF weighted by minutes
    our_total_poss = profiles['POSS_OFF'].sum()
    off_total_poss = official['POSS'].sum() if 'POSS' in official.columns else 0

    print(f"  Our total POSS_OFF (all players):     {our_total_poss:,.0f}")
    print(f"  NBA.com total POSS (all players):     {off_total_poss:,.0f}")

    if off_total_poss > 0:
        pct_diff = (our_total_poss - off_total_poss) / off_total_poss * 100
        print(f"  Difference:                           {our_total_poss - off_total_poss:+,.0f} ({pct_diff:+.1f}%)")

    # Per-game pace check
    our_avg_pace = profiles.loc[profiles['GP'] >= 50, 'POSS_OFF'].sum() / profiles.loc[profiles['GP'] >= 50, 'GP'].sum()
    print(f"\n  Our avg POSS_OFF / game (for GP≥50 players): {our_avg_pace:.1f}")
    
    if 'PACE' in official.columns:
        nba_avg_pace = official.loc[official['GP'] >= 50, 'PACE'].mean()
        print(f"  NBA.com avg PACE (GP≥50):                    {nba_avg_pace:.1f}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. BREF GROUND TRUTH CSV TEMPLATE
# ─────────────────────────────────────────────────────────────────────────────

def generate_bref_template(season):
    """Generate a CSV template for manual bref data collection."""
    rows = []
    for player_name, meta in VALIDATION_SAMPLE.items():
        rows.append({
            'player_name': player_name,
            'season': season,
            'position': meta['pos'],
            'tier': meta['tier'],
            'archetype': meta['archetype'],
            # Box score totals (from bref)
            'bref_GP': '',
            'bref_MP': '',     # Total minutes (not MPG)
            'bref_FGM': '',
            'bref_FGA': '',
            'bref_FG3M': '',
            'bref_FG3A': '',
            'bref_FTM': '',
            'bref_FTA': '',
            'bref_ORB': '',
            'bref_DRB': '',
            'bref_TRB': '',
            'bref_AST': '',
            'bref_STL': '',
            'bref_BLK': '',
            'bref_TOV': '',
            'bref_PF': '',
            'bref_PTS': '',
            # Advanced stats (from bref advanced page)
            'bref_TS_PCT': '',     # e.g. .625
            'bref_USG_PCT': '',    # e.g. .346
            'bref_ORTG': '',       # Offensive Rating
            'bref_DRTG': '',       # Defensive Rating
            'bref_BPM': '',
            'bref_VORP': '',
            'bref_WS': '',
            'bref_OWS': '',
            'bref_DWS': '',
        })

    df = pd.DataFrame(rows)
    path = f"tests/bref_ground_truth_{season}.csv"
    df.to_csv(path, index=False)
    print(f"\n  📝 Generated bref template: {path}")
    print(f"     {len(rows)} players to fill in from Basketball-Reference")
    print(f"     Columns: {list(df.columns)}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 9. BREF COMPARISON (if data provided)
# ─────────────────────────────────────────────────────────────────────────────

def validate_vs_bref(profiles, bref_path):
    """Compare our data vs manually collected Basketball-Reference data."""
    if not os.path.exists(bref_path):
        print(f"\n  ⚠️ Bref ground truth file not found: {bref_path}")
        print(f"     Generate template with --generate-template flag")
        return []

    bref = pd.read_csv(bref_path)
    # Skip empty rows
    bref = bref.dropna(subset=['bref_GP'])

    if bref.empty:
        print(f"\n  ⚠️ Bref template is empty — please fill in data from Basketball-Reference")
        return []

    print("\n" + "=" * 100)
    print("TIER 7: VS BASKETBALL-REFERENCE GROUND TRUTH")
    print("=" * 100)

    results = []

    for _, bref_row in bref.iterrows():
        player_name = bref_row['player_name']
        our_idx = fuzzy_match(player_name, profiles['player_name'])
        if our_idx is None:
            continue
        our = profiles.loc[our_idx]

        checks = {}
        # Box score totals
        for stat in ['GP', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'ORB', 'DRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']:
            our_val = our.get(stat, np.nan)
            bref_val = bref_row.get(f'bref_{stat}', np.nan)
            if pd.notna(our_val) and pd.notna(bref_val) and str(bref_val).strip():
                try:
                    diff = abs(float(our_val) - float(bref_val))
                    checks[f'{stat}_diff'] = diff
                except (ValueError, TypeError):
                    pass

        # Minutes
        our_min = our.get('MIN', np.nan)
        bref_min = bref_row.get('bref_MP', np.nan)
        if pd.notna(our_min) and pd.notna(bref_min) and str(bref_min).strip():
            try:
                min_diff = abs(float(our_min) - float(bref_min))
                checks['MIN_diff'] = min_diff
                checks['MIN_pct_diff'] = (min_diff / float(bref_min) * 100) if float(bref_min) > 0 else 0
            except (ValueError, TypeError):
                pass

        # Advanced stats
        for stat_pair in [('TS_PCT', 'bref_TS_PCT'), ('ORTG', 'bref_ORTG'), ('DRTG', 'bref_DRTG')]:
            our_val = our.get(stat_pair[0], np.nan)
            bref_val = bref_row.get(stat_pair[1], np.nan)
            if pd.notna(our_val) and pd.notna(bref_val) and str(bref_val).strip():
                try:
                    checks[f'{stat_pair[0]}_vs_bref'] = round(float(our_val) - float(bref_val), 3)
                except (ValueError, TypeError):
                    pass

        results.append({'player': player_name, **checks})

    # Print summary
    if results:
        # Count exact-match box stat cols
        box_stats = ['GP_diff', 'FGM_diff', 'FGA_diff', 'PTS_diff', 'AST_diff', 'REB_diff']
        for stat in box_stats:
            vals = [r.get(stat, np.nan) for r in results if stat in r]
            if vals:
                exact = sum(1 for v in vals if v == 0)
                print(f"    {stat[:-5]:>5}: {exact}/{len(vals)} exact matches")

        min_diffs = [r.get('MIN_pct_diff', 0) for r in results if 'MIN_pct_diff' in r]
        if min_diffs:
            print(f"    MIN:  mean Δ{np.mean(min_diffs):.2f}%, max Δ{np.max(min_diffs):.2f}%")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BKE Data Integrity Validation")
    parser.add_argument('--season', default='2024-25', help='Season to validate (default: 2024-25)')
    parser.add_argument('--bref', default=None, help='Path to bref ground truth CSV')
    parser.add_argument('--generate-template', action='store_true', help='Generate bref template CSV')
    parser.add_argument('--output', default=None, help='Save JSON report to this path')
    args = parser.parse_args()

    season = args.season
    print(f"{'='*100}")
    print(f"  BKE DATA INTEGRITY VALIDATION — {season}")
    print(f"  Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Sample: {len(VALIDATION_SAMPLE)} players")
    print(f"{'='*100}")

    # Generate template if requested
    if args.generate_template:
        generate_bref_template(season)
        return

    # Load data
    profiles = load_our_profiles(season)
    archetypes = load_our_archetypes(season)
    official = load_official_stats(season)

    if profiles.empty:
        print("❌ No profile data found")
        return
    if official.empty:
        print("❌ No official stats found")
        return

    print(f"\n  Loaded: {len(profiles)} player profiles, {len(official)} official records")

    report = {}

    # Run validations
    report['tier1_box_scores'] = validate_box_score_totals(profiles, official)
    report['tier2_internal'] = validate_internal_consistency(profiles)
    report['tier3_rate_stats'] = validate_rate_stats_vs_official(profiles, official)
    report['tier4_possessions'] = validate_possessions(profiles, official)
    report['tier5_usg'] = validate_usg_reconciliation(profiles, archetypes, official)
    validate_league_possessions(profiles, official)

    # Bref comparison if provided
    bref_path = args.bref or f"tests/bref_ground_truth_{season}.csv"
    if os.path.exists(bref_path):
        report['tier7_bref'] = validate_vs_bref(profiles, bref_path)

    # Generate template for later
    generate_bref_template(season)

    # Save JSON report
    if args.output:
        # Convert to serializable
        report_clean = {}
        for k, v in report.items():
            if isinstance(v, list):
                report_clean[k] = [{kk: (vv if not isinstance(vv, (np.floating, np.integer)) else float(vv)) for kk, vv in item.items()} for item in v]
        with open(args.output, 'w') as f:
            json.dump(report_clean, f, indent=2, default=str)
        print(f"\n  📊 Report saved to {args.output}")

    # Final verdict
    print("\n" + "=" * 100)
    print("  FINAL ASSESSMENT")
    print("=" * 100)

    poss_diffs = [r['pct_diff'] for r in report.get('tier4_possessions', []) if 'pct_diff' in r]
    usg_diffs = [r['prof_vs_official'] for r in report.get('tier5_usg', []) if 'prof_vs_official' in r]
    ortg_diffs = [r['ORTG_diff'] for r in report.get('tier3_rate_stats', []) if 'ORTG_diff' in r]

    print("\n  KNOWN ISSUES:")
    if poss_diffs and abs(np.mean(poss_diffs)) > 1.0:
        print(f"  1. POSSESSION COUNT DRIFT: Our PBP possessions are {np.mean(poss_diffs):+.1f}% vs NBA.com")
        print(f"     → Directly causes ORTG/DRTG inflation/deflation")
    if usg_diffs and abs(np.mean(usg_diffs)) > 0.5:
        print(f"  2. USG% DRIFT: Profile USG is Δ{np.mean(usg_diffs):+.1f}pp vs NBA.com")
        print(f"     → Partially caused by possession count issue, partially by formula difference")
    if ortg_diffs and abs(np.mean(ortg_diffs)) > 1.0:
        print(f"  3. ORTG/DRTG INFLATION: Our ratings are {np.mean(ortg_diffs):+.1f} pts vs NBA.com")
        print(f"     → Direct consequence of undercounted possessions")

    print("\n  RECOMMENDATIONS:")
    print("  • Investigate PBP possession derivation (likely missing some possession-ending events)")
    print("  • Consider using NBA.com POSS as calibration target")
    print("  • Archetype USG_PCT formula needs team-level denominators, not constant 2.4")
    print("  • Fill in bref_ground_truth CSV for authoritative box-score cross-check")
    print()


if __name__ == "__main__":
    main()
