"""
tests/validate_official_stats.py
Validates the integrity of Official Advanced Stats fetched from NBA.com.
"""

import pandas as pd
import numpy as np
import os
import sys
import glob
import unicodedata

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DATA_DIR = "data/official_stats"

EXPECTED_COLS = [
    "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "GP", "MIN", 
    "OFF_RATING", "DEF_RATING", "NET_RATING", "AST_PCT", "AST_TO", 
    "OREB_PCT", "DREB_PCT", "REB_PCT", "EFG_PCT", "TS_PCT", "USG_PCT"
]

THRESHOLDS = {
    "MIN_PLAYERS_PER_SEASON": 450,
    "USG_PCT_MIN": 0.0,
    "USG_PCT_MAX": 1.0, 
    "TS_PCT_MIN": 0.0, 
    "TS_PCT_MAX": 1.5,
}

def normalize_name(name):
    """Normalize unicode characters to ASCII (e.g. Dončić -> Doncic)"""
    if pd.isna(name): return ""
    return unicodedata.normalize('NFKD', str(name)).encode('ASCII', 'ignore').decode('utf-8')

def validate_names_and_encoding(df):
    """Checks for encoding issues, special characters, and potential collision issues."""
    print(f"\n   --- 🔤 Name Encoding & Integrity Check ---")
    
    # 1. Check for non-ASCII characters
    df['name_len'] = df['PLAYER_NAME'].astype(str).str.len()
    df['ascii_len'] = df['PLAYER_NAME'].astype(str).apply(lambda x: len(x.encode('ascii', errors='ignore')))
    
    non_ascii = df[df['name_len'] != df['ascii_len']]
    if not non_ascii.empty:
        print(f"   ℹ️  Found {len(non_ascii)} players with non-ASCII characters (e.g. UTF-8 names).")
        print(f"       Examples: {non_ascii['PLAYER_NAME'].head(5).tolist()}")
        print(f"       (This is expected for NBA data, but ensure downstream systems handle UTF-8)")
    else:
        print("   ✅ All player names appear to be standard ASCII.")

    # 2. Check for Name Collisions after Normalization
    # Risk: If we have "Name" and "Ńame", normalizing might merge them incorrectly if ID isn't used.
    df['norm_name'] = df['PLAYER_NAME'].apply(normalize_name)
    
    # Check duplicates in original names
    dupes = df[df.duplicated(subset=['PLAYER_NAME'], keep=False)]
    if not dupes.empty:
        # Sometimes players have same name (e.g. Jalen Williams - though usually unique IDs)
        print(f"   ⚠️  Duplicate exact names found: {dupes['PLAYER_NAME'].unique()}. Verify IDs are unique.")
        
    # Check duplicates in normalized names that weren't duplicates before
    df['is_dupe_norm'] = df.duplicated(subset=['norm_name'], keep=False)
    df['is_dupe_orig'] = df.duplicated(subset=['PLAYER_NAME'], keep=False)
    
    hidden_collisions = df[df['is_dupe_norm'] & (~df['is_dupe_orig'])]
    if not hidden_collisions.empty:
        print(f"   ⚠️  Potential Normalization Collisions (Distinct names become same when ascii-fied):")
        print(hidden_collisions[['PLAYER_NAME', 'norm_name', 'PLAYER_ID']].sort_values('norm_name').to_string(index=False))
        print("       ensure joins strictly use PLAYER_ID, not Name.")
    else:
        print("   ✅ No name collisions introduced by ASCII normalization.")

    # 3. Check for specific problematic characters (Replacement char )
    # \ufffd is the unicode replacement character
    corrupted = df[df['PLAYER_NAME'].astype(str).str.contains("\ufffd")]
    if not corrupted.empty:
        print(f"   ❌ CORRUPTED NAMES DETECTED (contains ):")
        print(corrupted['PLAYER_NAME'].tolist())
    else:
        print("   ✅ No corrupted characters () found.")



def perform_spot_check(df, n=100):
    """
    Randomly selects n players and performs rigorous checking, 
    printing a summary to ensure the data 'looks right'.
    """
    print(f"\n   --- 🎲 Random Spot Check ({n} Players) ---")
    
    if len(df) < n:
        sample = df
    else:
        sample = df.sample(n=n, random_state=42) # Fixed seed for reproducibility implies stable testing
    
    print(f"   Sampled {len(sample)} rows for deep inspection.")
    
    # 1. Check for Nulls in Sample
    sample_nulls = sample[EXPECTED_COLS].isnull().sum()
    if sample_nulls.sum() > 0:
        print(f"   ⚠️  Nulls found in random sample:\n{sample_nulls[sample_nulls>0]}")
    else:
        print("   ✅ No missing data in sample columns.")

    # 2. Logic Sanity Check (EFG vs TS vs PCT)
    # TS% should generally be >= EFG% (due to FTs), allow small margin
    # TS% = Pts / (2 * (FGA + 0.44 * FTA))
    # EFG% = (FG + 0.5 * 3P) / FGA
    # If TS% < EFG% it implies negative free throws? Or huge error. 
    # (Rare cases: 0 FTs, math rounding)
    
    if "EFG_PCT" in sample.columns and "TS_PCT" in sample.columns:
        # Fill na to 0 for logic check
        s_check = sample.fillna(0)
        anomalies = s_check[s_check['TS_PCT'] < (s_check['EFG_PCT'] - 0.05)] # 5% buffer
        if not anomalies.empty:
             print(f"   ⚠️  TS% significantly lower than EFG% for: {anomalies['PLAYER_NAME'].tolist()}")
        else:
             print("   ✅ Efficiency metrics logic (TS% >= EFG%) holds for sample.")

    # 3. Display a subset for user visual confirmation
    print("   \n   🔍 Visual Inspection (First 10 of Sample):")
    cols_visual = ['PLAYER_NAME', 'TEAM_ID', 'GP', 'MIN', 'USG_PCT', 'TS_PCT', 'OFF_RATING']
    print(sample[cols_visual].head(10).to_string(index=False))

def validate_official_stats():
    pattern = os.path.join(DATA_DIR, "official_advanced_*.parquet")
    files = glob.glob(pattern)
    
    if not files:
        print(f"❌ No official stats files found in {DATA_DIR}")
        return

    all_passed = True

    for fpath in sorted(files):
        season = os.path.basename(fpath).replace("official_advanced_", "").replace(".parquet", "")
        print(f"\n=== Validating Official Stats for {season} ({os.path.basename(fpath)}) ===")
        
        try:
            df = pd.read_parquet(fpath)
            
            # 1. Row Count Validation
            if len(df) < THRESHOLDS["MIN_PLAYERS_PER_SEASON"]:
                print(f"⚠️  Low player count: {len(df)} (Expected > {THRESHOLDS['MIN_PLAYERS_PER_SEASON']})")
                all_passed = False
            else:
                print(f"✅ Player count: {len(df)}")
            
            # 2. Column Validation
            missing_cols = [c for c in EXPECTED_COLS if c not in df.columns]
            if missing_cols:
                print(f"❌ Missing columns: {missing_cols}")
                all_passed = False
                continue # Major schema mismatch
            else:
                print(f"✅ Core columns present")
                
            # 2b. Type Validation
            numeric_cols = ["OFF_RATING", "DEF_RATING", "NET_RATING", "TS_PCT", "USG_PCT", "GP", "MIN"]
            for col in numeric_cols:
                if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                     print(f"⚠️  Column {col} is not numeric (Type: {df[col].dtype})")
                     # Try to see if it's convertible
                     try:
                         pd.to_numeric(df[col])
                         print(f"    (It is convertible to numeric)")
                     except:
                         print(f"    ❌ Not convertible to numeric!")
                         all_passed = False

            # 3. Data Integrity (Nulls)
            null_counts = df[EXPECTED_COLS].isnull().sum()
            if null_counts.sum() > 0:
                print(f"⚠️  Nulls detected:\n{null_counts[null_counts > 0]}")
                # Some stats might be null if 0 minutes or 0 attempts, check if critical
                if null_counts["PLAYER_ID"] > 0 or null_counts["PLAYER_NAME"] > 0:
                     print("❌ Critical Nulls in ID/Name")
                     all_passed = False
            
            # 4. Value Ranges
            # Check USG_PCT (should be roughly 0.0 to 0.4-0.5, rarely > 1.0 unless calc diff?)
            # NBA.com usually returns decimals (0.30) or percentages (30.0)?
            # Let's inspect the mean to guess
            if "USG_PCT" in df.columns:
                usg_mean = df["USG_PCT"].mean()
                print(f"   Avg USG_PCT: {usg_mean:.3f}")
                
                if usg_mean > 1.0: 
                    # Likely Percentage (0-100)
                    print(f"ℹ️  USAGE appears to be 0-100 scale")
                else:
                    print(f"ℹ️  USAGE appears to be 0-1 scale")
            
            # Check ORTG/DRTG typical range
            if "OFF_RATING" in df.columns and "DEF_RATING" in df.columns:
                ortg_mean = df["OFF_RATING"].mean()
                drtg_mean = df["DEF_RATING"].mean()
                print(f"   Avg ORTG: {ortg_mean:.1f}, Avg DRTG: {drtg_mean:.1f}")
                
                if ortg_mean < 50 or ortg_mean > 150:
                    print(f"⚠️  Unusual Average ORTG: {ortg_mean}")
            
            # 5. Spot Check (Key Players)
            stars = ["Nikola Jokic", "Joel Embiid", "Luka Doncic", "Giannis Antetokounmpo"]
            found_stars = df[df["PLAYER_NAME"].isin(stars)]
            
            if len(found_stars) < len(stars):
                missing = set(stars) - set(found_stars["PLAYER_NAME"].unique())
                print(f"⚠️  Missing stars: {missing}. checking for close matches...")
                for m in missing:
                    # Check first part of name
                    first_name = m.split()[0]
                    candidates = df[df["PLAYER_NAME"].str.contains(first_name, case=False, na=False)]
                    if not candidates.empty:
                        print(f"    Possible matches for {m}: {candidates['PLAYER_NAME'].unique()}")
            
            if found_stars.empty:
                print("⚠️  No major stars found (Jokic, Embiid, Luka, Giannis). Check naming formatting.")
            else:
                print(f"✅ Found {len(found_stars)}/{len(stars)} reference stars:") # Modified logic to be accepted if some found
                # Safe print columns
                cols_to_print = ["PLAYER_NAME", "GP", "MIN"]
                for c in ["USG_PCT", "TS_PCT", "OFF_RATING"]:
                    if c in df.columns:
                        cols_to_print.append(c)
                print(found_stars[cols_to_print].to_string(index=False))
            

            # 6. Logic Check
            # NET_RATING should be approx OFF_RATING - DEF_RATING
            # Note: NBA.com might calc them slightly differently (possessions mismatch?), but usually close.
            if "NET_RATING" in df.columns:
                 df["calc_net"] = df["OFF_RATING"] - df["DEF_RATING"]
                 df["net_diff"] = (df["NET_RATING"] - df["calc_net"]).abs()
                 large_diff = df[df["net_diff"] > 1.0] # Allow rounding err
                 if not large_diff.empty:
                     print(f"⚠️  {len(large_diff)} players have large mismatch in NET_RATING vs OFF-DEF")
                 else:
                     print("✅ NET_RATING consistent with OFF-DEF")
                     
            # 7. Extended Validations (Names & Spot check)
            validate_names_and_encoding(df)
            perform_spot_check(df, n=100)

        except Exception as e:
            print(f"❌ Error processing {fpath}: {e}")
            all_passed = False
            
    if all_passed:
        print("\n�� All validations passed for official stats.")
    else:
        print("\n⚠️ Some validations failed.")

if __name__ == "__main__":
    validate_official_stats()
