"""
src/data_compute/compute_defensive_archetypes_v2.py
Revised defensive archetype classification with 5 archetypes.

Defensive Archetype Definitions (REVISED):
==========================================
1. POA Defender - Guards/wings with toughest assignments, medium results
2. Rotation Defender - Medium matchups, low versatility, help defense
3. Off-Ball Defender - Easiest matchups, low versatility, poor results
4. Switchable Defender - High defensive versatility, guards all positions
5. Rim Protector - Interior defense, high blocks, wings/bigs

Data-Driven Thresholds (based on actual distributions):
- switch_score: min=0.60, mean=0.81, max=0.94 (very narrow range)
- avg_opponent_ppg: mean=11.6, 75th=13.4, 90th=14.7
- elite_matchup_pct: mean=0.15, 75th=0.20, 90th=0.28
- BLK per 36: mean=0.75, 90th=1.57, 95th=2.07
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = Path("data")
MATCHUP_DIR = DATA_DIR / "matchup"
TRACKING_DIR = DATA_DIR / "tracking"
HISTORICAL_DIR = DATA_DIR / "historical"
OUTPUT_DIR = DATA_DIR / "processed"
OUTPUT_DIR.mkdir(exist_ok=True)

SEASONS = ['2022-23', '2023-24', '2024-25']

# Minimum requirements
MIN_MINUTES = 400
MIN_GP = 15


# =============================================================================
# DATA LOADING
# =============================================================================
def load_matchup_versatility() -> pd.DataFrame:
    path = MATCHUP_DIR / "matchup_versatility.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def load_matchup_difficulty() -> pd.DataFrame:
    path = MATCHUP_DIR / "matchup_difficulty.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def load_defense_tracking(season: str) -> pd.DataFrame:
    season_dir = TRACKING_DIR / season
    all_defense = []
    categories = ['Overall', '3Pointers', 'LessThan6Ft']
    
    for cat in categories:
        path = season_dir / f"defense_{cat}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            df = df.rename(columns={
                'D_FG_PCT': f'D_FG_PCT_{cat}',
                'PCT_PLUSMINUS': f'PCT_PLUSMINUS_{cat}',
                'D_FGM': f'D_FGM_{cat}',
                'D_FGA': f'D_FGA_{cat}',
                'FREQ': f'FREQ_{cat}'
            })
            keep_cols = ['CLOSE_DEF_PERSON_ID', 'PLAYER_NAME'] + [c for c in df.columns if cat in c]
            df = df[keep_cols]
            all_defense.append(df)
    
    if not all_defense:
        return pd.DataFrame()
    
    merged = all_defense[0]
    for df in all_defense[1:]:
        merged = merged.merge(df, on=['CLOSE_DEF_PERSON_ID', 'PLAYER_NAME'], how='outer')
    
    merged['SEASON'] = season
    merged = merged.rename(columns={'CLOSE_DEF_PERSON_ID': 'PLAYER_ID'})
    return merged


def load_box_score_data() -> pd.DataFrame:
    path = HISTORICAL_DIR / "complete_player_season_stats.parquet"
    if not path.exists():
        return pd.DataFrame()
    
    df = pd.read_parquet(path)
    cols = ['PLAYER_ID', 'PLAYER_NAME', 'SEASON', 'GP', 'MIN', 
            'STL', 'BLK', 'DREB', 'TOV', 'PF']
    available = [c for c in cols if c in df.columns]
    return df[available]


def load_speed_distance() -> pd.DataFrame:
    all_data = []
    for season in SEASONS:
        path = TRACKING_DIR / season / "tracking_SpeedDistance.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            df['SEASON'] = season
            all_data.append(df)
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


# =============================================================================
# FEATURE COMPUTATION
# =============================================================================
def safe_col(df, col, default=0):
    if col in df.columns:
        return df[col].fillna(default)
    return pd.Series([default] * len(df), index=df.index)


def compute_features(versatility: pd.DataFrame, difficulty: pd.DataFrame,
                     defense_tracking: pd.DataFrame, box: pd.DataFrame,
                     speed_dist: pd.DataFrame, season: str) -> pd.DataFrame:
    
    vers = versatility[versatility['SEASON'] == season].copy() if len(versatility) > 0 else pd.DataFrame()
    diff = difficulty[difficulty['SEASON'] == season].copy() if len(difficulty) > 0 else pd.DataFrame()
    def_trk = defense_tracking.copy() if len(defense_tracking) > 0 else pd.DataFrame()
    bx = box[box['SEASON'] == season].copy() if len(box) > 0 else pd.DataFrame()
    spd = speed_dist[speed_dist['SEASON'] == season].copy() if len(speed_dist) > 0 else pd.DataFrame()
    
    if bx.empty:
        return pd.DataFrame()
    
    features = bx.copy()
    
    # Merge versatility
    if not vers.empty:
        vers_renamed = vers.rename(columns={'DEF_PLAYER_ID': 'PLAYER_ID'})
        features = features.merge(
            vers_renamed.drop(columns=['DEF_PLAYER_NAME', 'SEASON'], errors='ignore'),
            on='PLAYER_ID', how='left'
        )
    
    # Merge difficulty
    if not diff.empty:
        diff_renamed = diff.rename(columns={'DEF_PLAYER_ID': 'PLAYER_ID'})
        features = features.merge(
            diff_renamed.drop(columns=['DEF_PLAYER_NAME', 'SEASON'], errors='ignore'),
            on='PLAYER_ID', how='left'
        )
    
    # Merge defense tracking
    if not def_trk.empty:
        features = features.merge(
            def_trk.drop(columns=['PLAYER_NAME', 'SEASON'], errors='ignore'),
            on='PLAYER_ID', how='left'
        )
    
    # Merge speed/distance
    if not spd.empty and 'PLAYER_ID' in spd.columns:
        spd_cols = ['PLAYER_ID', 'DIST_MILES', 'DIST_MILES_DEF', 'AVG_SPEED', 'AVG_SPEED_DEF']
        spd_cols = [c for c in spd_cols if c in spd.columns]
        features = features.merge(
            spd[spd_cols],
            on='PLAYER_ID', how='left'
        )
    
    # Compute per-36 stats
    mpg = (features['MIN'] / features['GP']).replace(0, np.nan)
    minutes_factor = 36 / mpg
    
    features['STL_PER36'] = (features['STL'] / features['GP']) * minutes_factor
    features['BLK_PER36'] = (features['BLK'] / features['GP']) * minutes_factor
    features['DREB_PER36'] = (features['DREB'] / features['GP']) * minutes_factor
    
    features['SEASON'] = season
    return features


# =============================================================================
# CLASSIFICATION (5 ARCHETYPES)
# =============================================================================
def classify_defenders(features: pd.DataFrame) -> pd.DataFrame:
    """
    Classify players using data-driven percentile thresholds.
    
    Classification priority:
    1. Rim Protector - Top 10% in blocks
    2. POA Defender - Top 25% in difficulty/elite matchups  
    3. Switchable Defender - Top 25% in versatility
    4. Off-Ball Defender - Bottom 25% in difficulty, bad results
    5. Rotation Defender - Everyone else (middle ground)
    """
    
    df = features.copy()
    
    # Filter to minimum requirements
    qualified = df[(df['MIN'] >= MIN_MINUTES) & (df['GP'] >= MIN_GP)].copy()
    unqualified = df[(df['MIN'] < MIN_MINUTES) | (df['GP'] < MIN_GP)].copy()
    
    if len(qualified) == 0:
        return pd.DataFrame()
    
    # Fill missing values with defaults
    qualified['switch_score'] = safe_col(qualified, 'switch_score', 0.5)
    qualified['avg_opponent_ppg'] = safe_col(qualified, 'avg_opponent_ppg', 11.5)
    qualified['elite_matchup_pct'] = safe_col(qualified, 'elite_matchup_pct', 0.15)
    qualified['BLK_PER36'] = safe_col(qualified, 'BLK_PER36', 0.5)
    qualified['D_FG_DIFF'] = safe_col(qualified, 'PCT_PLUSMINUS_Overall', 0)
    
    # ==========================================================================
    # COMPUTE PERCENTILE RANKS (within this batch)
    # ==========================================================================
    qualified['versatility_pctl'] = qualified['switch_score'].rank(pct=True)
    qualified['difficulty_pctl'] = qualified['avg_opponent_ppg'].rank(pct=True)
    qualified['elite_matchup_pctl'] = qualified['elite_matchup_pct'].rank(pct=True)
    qualified['blocks_pctl'] = qualified['BLK_PER36'].rank(pct=True)
    qualified['d_results_pctl'] = (1 - qualified['D_FG_DIFF'].rank(pct=True))  # Lower is better
    
    # Combined scores
    qualified['rim_score'] = qualified['blocks_pctl'] * 0.7 + qualified['d_results_pctl'] * 0.3
    qualified['poa_score'] = qualified['elite_matchup_pctl'] * 0.5 + qualified['difficulty_pctl'] * 0.5
    
    # ==========================================================================
    # CLASSIFICATION
    # ==========================================================================
    results = []
    
    for idx, row in qualified.iterrows():
        versatility_pctl = row['versatility_pctl']
        difficulty_pctl = row['difficulty_pctl']
        elite_matchup_pctl = row['elite_matchup_pctl']
        blocks_pctl = row['blocks_pctl']
        d_results_pctl = row['d_results_pctl']
        rim_score = row['rim_score']
        poa_score = row['poa_score']
        
        archetype = 'Rotation Defender'  # Default
        secondary = None
        confidence = 0.5
        difficulty_level = 'Medium'
        
        # 1. RIM PROTECTOR - Top 10% in blocks
        # Check raw blocks first (more interpretable)
        blk_p36 = row['BLK_PER36'] or 0
        
        # POA threshold - more forgiving for toughest assignments
        poa_d_threshold = 0.30 if difficulty_pctl >= 0.90 else 0.40
        
        if blk_p36 >= 1.5 or (blocks_pctl >= 0.90):
            archetype = 'Rim Protector'
            confidence = 0.6 + blocks_pctl * 0.3
            
            if versatility_pctl >= 0.75:
                secondary = 'Switchable'
            elif blk_p36 >= 2.0:
                secondary = 'Shot Blocker'
            else:
                secondary = 'Interior'
        
        # 2. POA DEFENDER - Top 25% in difficulty/elite matchups AND not terrible
        # More forgiving for players with absolute toughest assignments (top 10% difficulty)
        elif poa_score >= 0.75 and d_results_pctl >= poa_d_threshold:
            archetype = 'POA Defender'
            confidence = 0.5 + poa_score * 0.4
            difficulty_level = 'High'
            
            if versatility_pctl >= 0.75:
                secondary = 'Switchable'
            elif row.get('STL_PER36', 0) >= 1.0:
                secondary = 'Ball Hawk'
            else:
                secondary = 'Primary'
        
        # 3. SWITCHABLE DEFENDER - Top 25% versatility with decent results
        elif versatility_pctl >= 0.75 and d_results_pctl >= 0.35:
            archetype = 'Switchable Defender'
            confidence = 0.5 + versatility_pctl * 0.4
            
            if blocks_pctl >= 0.75:
                secondary = 'Rim Protector'
            elif d_results_pctl >= 0.70:
                secondary = 'Lockdown'
            else:
                secondary = 'Versatile'
        
        # 4. OFF-BALL DEFENDER - Bottom 25% difficulty OR bad defender with low difficulty
        elif (difficulty_pctl <= 0.25 and d_results_pctl <= 0.40) or \
             (d_results_pctl <= 0.25 and difficulty_pctl <= 0.50):
            archetype = 'Off-Ball Defender'
            confidence = 0.4 + (1 - d_results_pctl) * 0.3
            difficulty_level = 'Low'
            
            if d_results_pctl <= 0.25:
                secondary = 'Liability'
            else:
                secondary = 'Hidden'
        
        # 5. ROTATION DEFENDER - Everyone else (the middle)
        else:
            archetype = 'Rotation Defender'
            confidence = 0.45 + d_results_pctl * 0.3
            
            if row.get('STL_PER36', 0) >= 1.0:
                secondary = 'Active Hands'
            elif d_results_pctl >= 0.60:
                secondary = 'Solid'
            else:
                secondary = 'Help'
        
        results.append({
            'PLAYER_ID': row['PLAYER_ID'],
            'PLAYER_NAME': row['PLAYER_NAME'],
            'SEASON': row['SEASON'],
            'GP': row['GP'],
            'MIN': row['MIN'],
            'defensive_archetype': archetype,
            'defensive_secondary': secondary,
            'defensive_confidence': min(0.95, confidence),
            'assignment_difficulty': difficulty_level,
            # Key metrics for viewer
            'switch_score': row['switch_score'],
            'versatility_pctl': versatility_pctl,
            'avg_opponent_ppg': row['avg_opponent_ppg'],
            'elite_matchup_pct': row['elite_matchup_pct'],
            'difficulty_pctl': difficulty_pctl,
            'BLK_PER36': row['BLK_PER36'],
            'blocks_pctl': blocks_pctl,
            'STL_PER36': row.get('STL_PER36', 0),
            'D_FG_DIFF': row['D_FG_DIFF'],
            'd_results_pctl': d_results_pctl,
        })
    
    # Add unqualified players
    for idx, row in unqualified.iterrows():
        results.append({
            'PLAYER_ID': row['PLAYER_ID'],
            'PLAYER_NAME': row['PLAYER_NAME'],
            'SEASON': row['SEASON'],
            'GP': row['GP'],
            'MIN': row['MIN'],
            'defensive_archetype': 'Insufficient Minutes',
            'defensive_secondary': None,
            'defensive_confidence': 0.0,
            'assignment_difficulty': 'Unknown',
            'switch_score': 0,
            'versatility_pctl': 0,
            'avg_opponent_ppg': 0,
            'elite_matchup_pct': 0,
            'difficulty_pctl': 0,
            'BLK_PER36': 0,
            'blocks_pctl': 0,
            'STL_PER36': 0,
            'D_FG_DIFF': 0,
            'd_results_pctl': 0,
        })
    
    return pd.DataFrame(results)


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("DEFENSIVE ARCHETYPE CLASSIFICATION v2 (5 Archetypes)")
    print("=" * 70)
    
    # Load data
    print("\n📊 Loading data...")
    versatility = load_matchup_versatility()
    difficulty = load_matchup_difficulty()
    box = load_box_score_data()
    speed_dist = load_speed_distance()
    
    print(f"  Versatility: {len(versatility)} records")
    print(f"  Difficulty: {len(difficulty)} records")
    print(f"  Box scores: {len(box)} records")
    
    all_results = []
    
    for season in SEASONS:
        print(f"\n🏀 Processing {season}...")
        
        def_tracking = load_defense_tracking(season)
        features = compute_features(versatility, difficulty, def_tracking, box, speed_dist, season)
        
        if features.empty:
            continue
        
        results = classify_defenders(features)
        all_results.append(results)
        
        # Show distribution
        arch_dist = results[results['defensive_archetype'] != 'Insufficient Minutes']['defensive_archetype'].value_counts()
        print(f"  Distribution:")
        for arch, count in arch_dist.items():
            print(f"    {arch}: {count}")
    
    if not all_results:
        print("\n❌ No results")
        return
    
    final_df = pd.concat(all_results, ignore_index=True)
    
    # Save
    final_df.to_parquet(OUTPUT_DIR / "defensive_archetypes_v2.parquet", index=False)
    final_df.to_csv(OUTPUT_DIR / "defensive_archetypes_v2.csv", index=False)
    
    print(f"\n✅ Saved {len(final_df)} records")
    
    # Validate known defenders
    print("\n=== VALIDATION (Known Defenders 2024-25) ===")
    known = [
        ('Jrue Holiday', 'POA or Switchable'),
        ('Draymond Green', 'Switchable'),
        ('Rudy Gobert', 'Rim Protector'),
        ('Anthony Davis', 'Rim Protector'),
        ('Herb Jones', 'POA'),
        ('Dyson Daniels', 'POA'),
        ('Bam Adebayo', 'Switchable or Rim'),
        ('Victor Wembanyama', 'Rim Protector'),
        ('Trae Young', 'Off-Ball (weak defender)'),
        ('Chet Holmgren', 'Rim Protector'),
    ]
    
    for name, expected in known:
        player = final_df[(final_df['PLAYER_NAME'].str.contains(name, case=False, na=False)) & 
                          (final_df['SEASON'] == '2024-25')]
        if len(player) == 0:
            player = final_df[final_df['PLAYER_NAME'].str.contains(name, case=False, na=False)]
        
        if len(player) > 0:
            r = player.iloc[0]
            sec = f" ({r['defensive_secondary']})" if r['defensive_secondary'] else ""
            blk = r.get('BLK_PER36', 0)
            print(f"  {r['PLAYER_NAME'][:22]:22} | {r['defensive_archetype']:20}{sec:15} | BLK/36: {blk:.1f} | Expected: {expected}")


if __name__ == "__main__":
    main()
