"""
src/data_compute/compute_defensive_archetypes.py
Classifies NBA players into defensive archetypes based on matchup and tracking data.

Defensive Archetype Definitions:
================================
1. Lockdown Perimeter Defender - High versatility, guards best perimeter scorers, low FG% allowed
2. Switchable Defender - Very high position versatility (guards 1-5), good overall defense
3. Rim Protector - Primary interior defender, high blocks, contests shots at rim
4. Help/Rotation Defender - Good rim defense + perimeter, helps off ball
5. Point-of-Attack Defender - Guards primary ballhandlers, good at forcing turnovers
6. Wing Stopper - Specializes in guarding wings (SF/PF), versatile with size
7. Post Defender - Guards traditional post players (C/PF), physical presence
8. Limited/Scheme Defender - Low versatility, needs scheme help, role player defense
9. Defensive Liability - Poor metrics across the board

Key Metrics Used:
- Matchup Versatility (switch_score, positions_guarded)
- Matchup Difficulty (avg_opponent_ppg, elite_matchup_pct)
- Defense Tracking (D_FG_PCT, PCT_PLUSMINUS by distance)
- Defensive Synergy (PPP allowed by playtype)
- Hustle Stats (contests, deflections)
- Box Score (BLK, STL, DREB)
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

# Thresholds for defensive archetype classification
THRESHOLDS = {
    # Versatility thresholds
    'HIGH_SWITCH_SCORE': 0.65,      # High position versatility
    'VERY_HIGH_SWITCH_SCORE': 0.80, # Elite switcher (guards all positions)
    
    # Difficulty thresholds
    'HIGH_DIFFICULTY': 15.0,        # Avg opponent PPG
    'ELITE_MATCHUP_PCT': 0.30,      # 30%+ time vs 20+ PPG scorers
    
    # Defense tracking thresholds
    'GOOD_D_FG_PCT_DIFF': -0.02,    # 2% below normal
    'ELITE_D_FG_PCT_DIFF': -0.04,   # 4% below normal
    
    # Box score thresholds (per 36)
    'HIGH_BLOCKS': 1.5,
    'ELITE_BLOCKS': 2.0,
    'HIGH_STEALS': 1.2,
    'ELITE_STEALS': 1.8,
    
    # Minutes threshold
    'MIN_MINUTES': 500,
    'MIN_GP': 20,
}

# =============================================================================
# DATA LOADING
# =============================================================================
def load_matchup_versatility() -> pd.DataFrame:
    """Load matchup versatility data (switch_score, positions_guarded)."""
    path = MATCHUP_DIR / "matchup_versatility.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def load_matchup_difficulty() -> pd.DataFrame:
    """Load matchup difficulty data (avg_opponent_ppg, elite_matchup_pct)."""
    path = MATCHUP_DIR / "matchup_difficulty.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def load_defense_tracking(season: str) -> pd.DataFrame:
    """Load defense tracking data (Overall, 3PT, <6ft)."""
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
            # Keep only essential columns
            keep_cols = ['CLOSE_DEF_PERSON_ID', 'PLAYER_NAME'] + [c for c in df.columns if cat in c]
            df = df[keep_cols]
            all_defense.append(df)
    
    if not all_defense:
        return pd.DataFrame()
    
    # Merge all categories
    merged = all_defense[0]
    for df in all_defense[1:]:
        merged = merged.merge(df, on=['CLOSE_DEF_PERSON_ID', 'PLAYER_NAME'], how='outer')
    
    merged['SEASON'] = season
    merged = merged.rename(columns={'CLOSE_DEF_PERSON_ID': 'PLAYER_ID'})
    return merged


def load_defensive_synergy(season: str) -> pd.DataFrame:
    """Load defensive synergy playtype data."""
    season_dir = TRACKING_DIR / season
    
    playtypes = ['Isolation', 'PRBallHandler', 'Postup', 'Spotup', 'Handoff', 'OffScreen', 'PRRollman']
    all_data = []
    
    for playtype in playtypes:
        path = season_dir / f"synergy_Defensive_{playtype}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            # Aggregate duplicates
            agg_df = df.groupby('PLAYER_ID').agg({
                'PLAYER_NAME': 'first',
                'PPP': 'mean',
                'POSS': 'sum',
                'POSS_PCT': 'mean',
            }).reset_index()
            
            agg_df = agg_df.rename(columns={
                'PPP': f'DEF_{playtype.upper()}_PPP',
                'POSS': f'DEF_{playtype.upper()}_POSS',
                'POSS_PCT': f'DEF_{playtype.upper()}_PCT'
            })
            all_data.append(agg_df)
    
    if not all_data:
        return pd.DataFrame()
    
    # Merge all playtypes
    merged = all_data[0]
    for df in all_data[1:]:
        merged = merged.merge(
            df.drop(columns=['PLAYER_NAME'], errors='ignore'),
            on='PLAYER_ID', how='outer'
        )
    
    merged['SEASON'] = season
    return merged


def load_box_score_data() -> pd.DataFrame:
    """Load box score data for defensive stats."""
    path = HISTORICAL_DIR / "complete_player_season_stats.parquet"
    if not path.exists():
        return pd.DataFrame()
    
    df = pd.read_parquet(path)
    cols = ['PLAYER_ID', 'PLAYER_NAME', 'SEASON', 'GP', 'MIN', 
            'STL', 'BLK', 'DREB', 'TOV', 'PF']
    available = [c for c in cols if c in df.columns]
    return df[available]


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
def compute_defensive_features(versatility: pd.DataFrame, difficulty: pd.DataFrame,
                                 defense_tracking: pd.DataFrame, defensive_synergy: pd.DataFrame,
                                 box: pd.DataFrame, season: str) -> pd.DataFrame:
    """Compute all features needed for defensive archetype classification."""
    
    # Filter to season
    vers = versatility[versatility['SEASON'] == season].copy() if not versatility.empty else pd.DataFrame()
    diff = difficulty[difficulty['SEASON'] == season].copy() if not difficulty.empty else pd.DataFrame()
    def_trk = defense_tracking.copy() if not defense_tracking.empty else pd.DataFrame()
    def_syn = defensive_synergy.copy() if not defensive_synergy.empty else pd.DataFrame()
    bx = box[box['SEASON'] == season].copy() if not box.empty else pd.DataFrame()
    
    # Start with box score as base
    if bx.empty:
        return pd.DataFrame()
    
    features = bx.copy()
    
    # Merge versatility (using DEF_PLAYER_ID)
    if not vers.empty:
        vers_renamed = vers.rename(columns={'DEF_PLAYER_ID': 'PLAYER_ID', 'DEF_PLAYER_NAME': 'PLAYER_NAME_vers'})
        features = features.merge(
            vers_renamed.drop(columns=['PLAYER_NAME_vers', 'SEASON'], errors='ignore'),
            on='PLAYER_ID', how='left'
        )
    
    # Merge difficulty
    if not diff.empty:
        diff_renamed = diff.rename(columns={'DEF_PLAYER_ID': 'PLAYER_ID', 'DEF_PLAYER_NAME': 'PLAYER_NAME_diff'})
        features = features.merge(
            diff_renamed.drop(columns=['PLAYER_NAME_diff', 'SEASON'], errors='ignore'),
            on='PLAYER_ID', how='left'
        )
    
    # Merge defense tracking
    if not def_trk.empty:
        features = features.merge(
            def_trk.drop(columns=['PLAYER_NAME', 'SEASON'], errors='ignore'),
            on='PLAYER_ID', how='left'
        )
    
    # Merge defensive synergy
    if not def_syn.empty:
        features = features.merge(
            def_syn.drop(columns=['PLAYER_NAME', 'SEASON'], errors='ignore'),
            on='PLAYER_ID', how='left'
        )
    
    # ==========================================================================
    # COMPUTE DERIVED FEATURES
    # ==========================================================================
    
    # Per-36 stats
    minutes_factor = 36 / (features['MIN'] / features['GP']).replace(0, np.nan)
    features['STL_PER36'] = (features['STL'] / features['GP']) * minutes_factor
    features['BLK_PER36'] = (features['BLK'] / features['GP']) * minutes_factor
    features['DREB_PER36'] = (features['DREB'] / features['GP']) * minutes_factor
    
    # Safe column access helper
    def safe_col(df, col, default=0):
        if col in df.columns:
            return df[col].fillna(default)
        return pd.Series([default] * len(df), index=df.index)
    
    # Defensive composite scores
    features['RIM_PROTECTION_SCORE'] = (
        features['BLK_PER36'].fillna(0) * 2 +
        safe_col(features, 'PCT_PLUSMINUS_LessThan6Ft').abs() * 50
    )
    
    features['PERIMETER_D_SCORE'] = (
        features['STL_PER36'].fillna(0) * 2 +
        safe_col(features, 'PCT_PLUSMINUS_3Pointers').abs() * 50 +
        safe_col(features, 'PCT_PLUSMINUS_Overall').abs() * 30
    )
    
    # Defensive playtype composite (lower PPP = better)
    ppp_cols = [c for c in features.columns if 'DEF_' in c and '_PPP' in c]
    if ppp_cols:
        features['AVG_DEF_PPP'] = features[ppp_cols].mean(axis=1)
    else:
        features['AVG_DEF_PPP'] = 1.0  # League average if no data
    
    # Fill NaN values using safe access
    features['switch_score'] = safe_col(features, 'switch_score')
    features['positions_guarded'] = safe_col(features, 'positions_guarded', 1)
    features['avg_opponent_ppg'] = safe_col(features, 'avg_opponent_ppg', 10)
    features['elite_matchup_pct'] = safe_col(features, 'elite_matchup_pct')
    features['pct_guards'] = safe_col(features, 'pct_guards', 0.33)
    features['pct_forwards'] = safe_col(features, 'pct_forwards', 0.33)
    features['pct_centers'] = safe_col(features, 'pct_centers', 0.33)
    
    features['SEASON'] = season
    return features


# =============================================================================
# ARCHETYPE CLASSIFICATION
# =============================================================================
def classify_defensive_archetype(row: pd.Series) -> dict:
    """
    Classify a single player into a defensive archetype.
    
    Classification hierarchy:
    1. Check minimum requirements
    2. Check rim protection ability
    3. Check perimeter defense ability
    4. Check versatility/switching
    5. Assign based on best fit
    """
    
    result = {
        'defensive_archetype': 'Unknown',
        'defensive_secondary': None,
        'defensive_confidence': 0.0,
        'versatility_tier': 'Low',
        'rim_protection_tier': 'Low',
        'perimeter_tier': 'Low',
    }
    
    # Check minimum requirements
    if row.get('MIN', 0) < THRESHOLDS['MIN_MINUTES'] or row.get('GP', 0) < THRESHOLDS['MIN_GP']:
        result['defensive_archetype'] = 'Insufficient Minutes'
        return result
    
    # ==========================================================================
    # TIER ASSIGNMENTS
    # ==========================================================================
    
    # Versatility tier
    switch_score = row.get('switch_score', 0)
    if switch_score >= THRESHOLDS['VERY_HIGH_SWITCH_SCORE']:
        result['versatility_tier'] = 'Elite'
    elif switch_score >= THRESHOLDS['HIGH_SWITCH_SCORE']:
        result['versatility_tier'] = 'High'
    else:
        result['versatility_tier'] = 'Low'
    
    # Rim protection tier
    blk_per36 = row.get('BLK_PER36', 0)
    rim_pct_diff = row.get('PCT_PLUSMINUS_LessThan6Ft', 0) or 0
    if blk_per36 >= THRESHOLDS['ELITE_BLOCKS'] and rim_pct_diff < THRESHOLDS['ELITE_D_FG_PCT_DIFF']:
        result['rim_protection_tier'] = 'Elite'
    elif blk_per36 >= THRESHOLDS['HIGH_BLOCKS'] or rim_pct_diff < THRESHOLDS['GOOD_D_FG_PCT_DIFF']:
        result['rim_protection_tier'] = 'High'
    else:
        result['rim_protection_tier'] = 'Low'
    
    # Perimeter defense tier
    stl_per36 = row.get('STL_PER36', 0)
    peri_pct_diff = row.get('PCT_PLUSMINUS_3Pointers', 0) or 0
    elite_matchup_pct = row.get('elite_matchup_pct', 0)
    
    if stl_per36 >= THRESHOLDS['ELITE_STEALS'] or (elite_matchup_pct >= THRESHOLDS['ELITE_MATCHUP_PCT'] and peri_pct_diff < 0):
        result['perimeter_tier'] = 'Elite'
    elif stl_per36 >= THRESHOLDS['HIGH_STEALS'] or elite_matchup_pct >= 0.20:
        result['perimeter_tier'] = 'High'
    else:
        result['perimeter_tier'] = 'Low'
    
    # ==========================================================================
    # CLASSIFICATION LOGIC
    # ==========================================================================
    
    is_elite_rim = result['rim_protection_tier'] == 'Elite'
    is_high_rim = result['rim_protection_tier'] in ['Elite', 'High']
    is_elite_perimeter = result['perimeter_tier'] == 'Elite'
    is_high_perimeter = result['perimeter_tier'] in ['Elite', 'High']
    is_versatile = result['versatility_tier'] in ['Elite', 'High']
    is_elite_versatile = result['versatility_tier'] == 'Elite'
    
    avg_opp_ppg = row.get('avg_opponent_ppg', 10)
    guards_best = avg_opp_ppg >= THRESHOLDS['HIGH_DIFFICULTY']
    
    # Primary position info
    pct_guards = row.get('pct_guards', 0.33)
    pct_forwards = row.get('pct_forwards', 0.33)
    pct_centers = row.get('pct_centers', 0.33)
    primary_pos = row.get('primary_position', 'Unknown')
    
    # ---------------------------------------------------------------------------
    # CLASSIFICATION HIERARCHY
    # ---------------------------------------------------------------------------
    
    # 1. LOCKDOWN PERIMETER DEFENDER
    # Elite perimeter D, guards best scorers, high difficulty
    if is_elite_perimeter and guards_best and elite_matchup_pct >= THRESHOLDS['ELITE_MATCHUP_PCT']:
        result['defensive_archetype'] = 'Lockdown Perimeter Defender'
        result['defensive_confidence'] = min(1.0, elite_matchup_pct + stl_per36 / 3)
        if is_versatile:
            result['defensive_secondary'] = 'Versatile'
        return result
    
    # 2. SWITCHABLE DEFENDER
    # Elite versatility, guards all positions effectively
    if is_elite_versatile and (is_high_perimeter or is_high_rim):
        result['defensive_archetype'] = 'Switchable Defender'
        result['defensive_confidence'] = min(1.0, switch_score)
        if is_high_rim:
            result['defensive_secondary'] = 'Rim Protector'
        elif is_high_perimeter:
            result['defensive_secondary'] = 'Perimeter'
        return result
    
    # 3. RIM PROTECTOR
    # Elite shot blocking, interior defense
    if is_elite_rim:
        result['defensive_archetype'] = 'Rim Protector'
        result['defensive_confidence'] = min(1.0, blk_per36 / 2.5)
        if is_versatile:
            result['defensive_secondary'] = 'Switchable'
        return result
    
    # 4. HELP/ROTATION DEFENDER
    # High rim protection + good perimeter, rotates well
    if is_high_rim and is_high_perimeter:
        result['defensive_archetype'] = 'Help/Rotation Defender'
        result['defensive_confidence'] = min(1.0, (blk_per36 + stl_per36) / 3)
        return result
    
    # 5. POINT-OF-ATTACK DEFENDER
    # Primarily guards ballhandlers, high steals
    if pct_guards > 0.5 and is_high_perimeter:
        result['defensive_archetype'] = 'Point-of-Attack Defender'
        result['defensive_confidence'] = min(1.0, stl_per36 / 1.5)
        if guards_best:
            result['defensive_secondary'] = 'Primary Stopper'
        return result
    
    # 6. WING STOPPER
    # Guards wings (SF/PF), size with perimeter skills
    if pct_forwards > 0.4 and is_high_perimeter:
        result['defensive_archetype'] = 'Wing Stopper'
        result['defensive_confidence'] = min(1.0, pct_forwards + stl_per36 / 2)
        return result
    
    # 7. POST DEFENDER
    # Guards centers/PFs, physical presence
    if pct_centers > 0.4 and is_high_rim:
        result['defensive_archetype'] = 'Post Defender'
        result['defensive_confidence'] = min(1.0, pct_centers + blk_per36 / 2)
        return result
    
    # 8. LIMITED/SCHEME DEFENDER
    # Not terrible, but needs help
    if is_high_perimeter or is_high_rim:
        result['defensive_archetype'] = 'Scheme Defender'
        result['defensive_confidence'] = 0.5
        if is_high_rim:
            result['defensive_secondary'] = 'Rim Help'
        else:
            result['defensive_secondary'] = 'Perimeter Help'
        return result
    
    # 9. DEFENSIVE LIABILITY (or just neutral)
    # Poor metrics, needs to be hidden
    overall_pct_diff = row.get('PCT_PLUSMINUS_Overall', 0) or 0
    if overall_pct_diff > 0.02:  # Opponents shoot 2%+ better
        result['defensive_archetype'] = 'Defensive Liability'
        result['defensive_confidence'] = 0.3
    else:
        result['defensive_archetype'] = 'Neutral Defender'
        result['defensive_confidence'] = 0.4
    
    return result


def classify_all_defenders(features: pd.DataFrame) -> pd.DataFrame:
    """Classify all players in the features dataframe."""
    
    classifications = []
    for idx, row in features.iterrows():
        result = classify_defensive_archetype(row)
        result['PLAYER_ID'] = row['PLAYER_ID']
        result['PLAYER_NAME'] = row['PLAYER_NAME']
        result['SEASON'] = row['SEASON']
        classifications.append(result)
    
    class_df = pd.DataFrame(classifications)
    output = features.merge(class_df, on=['PLAYER_ID', 'PLAYER_NAME', 'SEASON'])
    
    return output


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    print("=" * 70)
    print("DEFENSIVE ARCHETYPE CLASSIFICATION")
    print("=" * 70)
    
    # Load all data
    print("\n📊 Loading data...")
    
    versatility = load_matchup_versatility()
    difficulty = load_matchup_difficulty()
    box_df = load_box_score_data()
    
    print(f"   Versatility data: {len(versatility)} rows")
    print(f"   Difficulty data: {len(difficulty)} rows")
    print(f"   Box score data: {len(box_df)} rows")
    
    all_results = []
    
    for season in SEASONS:
        print(f"\n🔄 Processing {season}...")
        
        # Load season-specific tracking data
        defense_tracking = load_defense_tracking(season)
        defensive_synergy = load_defensive_synergy(season)
        
        print(f"   Defense tracking: {len(defense_tracking)} rows")
        print(f"   Defensive synergy: {len(defensive_synergy)} rows")
        
        # Compute features
        features = compute_defensive_features(
            versatility, difficulty, defense_tracking, defensive_synergy, box_df, season
        )
        
        if features.empty:
            print(f"   ⚠️ No data for {season}")
            continue
        
        # Classify
        classified = classify_all_defenders(features)
        all_results.append(classified)
        
        # Print summary
        archetype_counts = classified['defensive_archetype'].value_counts()
        print(f"   Classified {len(classified)} players:")
        for arch, count in archetype_counts.items():
            print(f"      {arch}: {count}")
    
    # Combine all seasons
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        
        # Save results
        final_df.to_parquet(OUTPUT_DIR / "defensive_archetypes.parquet", index=False)
        final_df.to_csv(OUTPUT_DIR / "defensive_archetypes.csv", index=False)
        
        print(f"\n✅ Saved {len(final_df)} player-seasons to data/processed/defensive_archetypes.parquet")
        
        # Print example classifications
        print("\n" + "=" * 70)
        print("SAMPLE DEFENSIVE CLASSIFICATIONS (2024-25 Known Defenders)")
        print("=" * 70)
        
        s24 = final_df[final_df['SEASON'] == '2024-25'].copy()
        defenders = [
            'Rudy Gobert', 'Bam Adebayo', 'Draymond Green', 'Mikal Bridges',
            'Jrue Holiday', 'Alex Caruso', 'Marcus Smart', 'OG Anunoby',
            'Herb Jones', 'Evan Mobley', 'Anthony Davis', 'Jaren Jackson',
            'Victor Wembanyama', 'Chet Holmgren', 'Dyson Daniels'
        ]
        
        for name in defenders:
            player = s24[s24['PLAYER_NAME'].str.contains(name.split()[0], case=False, na=False)]
            if len(player) > 0:
                p = player.iloc[0]
                secondary = f" / {p['defensive_secondary']}" if p['defensive_secondary'] else ""
                print(f"  {p['PLAYER_NAME']:<25} → {p['defensive_archetype']}{secondary} ({p['defensive_confidence']:.0%})")
    
    return final_df


if __name__ == "__main__":
    result = main()
