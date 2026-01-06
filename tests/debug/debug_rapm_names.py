"""
tests/debug/debug_rapm_names.py
Diagnoses why player names are showing as 'Unknown' in RAPM output.
"""
import pandas as pd
import os
import sys

# Adjust path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

DATA_DIR = "data/historical"
OUTPUT_DIR = "data/processed"

def clean_id(val):
    """The same cleaning logic used in RAPM."""
    if pd.isna(val): return "0"
    return str(val).replace(".0", "")

def check_mismatch():
    print("=== RAPM ID Debugger ===\n")

    # 1. Load RAPM Output
    rapm_path = os.path.join(OUTPUT_DIR, "player_rapm.parquet")
    if not os.path.exists(rapm_path):
        print(f"❌ {rapm_path} not found. Run compute_rapm.py first.")
        return
    
    rapm_df = pd.read_parquet(rapm_path)
    print(f"Loaded RAPM results: {len(rapm_df)} rows")
    
    # Get top 5 Unknown IDs
    unknowns = rapm_df[rapm_df['player_name'] == 'Unknown'].head(5)
    if unknowns.empty:
        print("✅ No 'Unknown' names found! Problem solved?")
        return
        
    print(f"\nSample 'Unknown' IDs from RAPM (Column: player_id):")
    print(unknowns[['player_id', 'RAPM']])
    sample_rapm_id = unknowns.iloc[0]['player_id']
    print(f"Type of RAPM ID: {type(sample_rapm_id)} | Value: '{sample_rapm_id}'")

    # 2. Load Players Parquet
    p_path = os.path.join(DATA_DIR, "players.parquet")
    if not os.path.exists(p_path):
        print(f"❌ {p_path} not found.")
        return
        
    players_df = pd.read_parquet(p_path)
    print(f"\nLoaded players.parquet: {len(players_df)} rows")
    print(f"Columns: {players_df.columns.tolist()}")
    
    # Check if we can find the sample ID
    # Try direct match
    if 'id' in players_df.columns:
        col = 'id'
    elif 'player_id' in players_df.columns:
        col = 'player_id'
    else:
        print("❌ No 'id' or 'player_id' column in players.parquet")
        return

    # Debug format in players.parquet
    sample_ref_val = players_df[col].iloc[0]
    print(f"Sample Reference ID in parquet: '{sample_ref_val}' (Type: {type(sample_ref_val)})")

    # Try to find the missing ID
    match = players_df[players_df[col].astype(str) == str(sample_rapm_id)]
    
    if not match.empty:
        print(f"\n✅ ID '{sample_rapm_id}' FOUND in players.parquet!")
        print(f"Data for this ID:\n{match.iloc[0].to_dict()}")
        name = match.iloc[0].get('full_name')
        print(f"Name value: '{name}' (Type: {type(name)})")
        if pd.isna(name):
            print("⚠️ Name is None/NaN. This explains the 'Unknown'.")
    else:
        print(f"\n❌ ID '{sample_rapm_id}' NOT FOUND in players.parquet.")
        print("Trying with .0 suffix check...")
        match_float = players_df[players_df[col].astype(str) == str(sample_rapm_id) + ".0"]
        if not match_float.empty:
            print(f"⚠️ Found match via float string ('{sample_rapm_id}.0'). CLEANING LOGIC NEEDED.")
        else:
            print("ID truly missing from reference file.")

if __name__ == "__main__":
    check_mismatch()