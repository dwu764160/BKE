#!/usr/bin/env python3
"""
scripts/phase4_apply_schema_contract.py
======================================
Phase 4: Apply schema_contract to all data_compute scripts.
- Add load_standardized / save_standardized imports
- Replace pd.read_parquet → load_standardized
- Replace df.to_parquet → save_standardized
- Lowercase all column name string literals
"""

import re
import sys


# ─── Column-name mapping (uppercase → lowercase) ───────────────────────────
# Order matters: longer names first to avoid partial-match bugs.
COLMAP = {
    # Identifiers
    "PLAYER_ID": "player_id",
    "PLAYER_NAME": "player_name",
    "TEAM_ABBREVIATION": "team_abbreviation",
    "TEAM_ID": "team_id",
    "GAME_ID": "game_id",
    "GAME_DATE": "game_date",
    "SEASON_ID": "season_id",
    "SEASON": "season",
    # Box score basics
    "GP": "gp",
    "MIN": "min",
    "PTS": "pts",
    "AST": "ast",
    "REB": "reb",
    "OREB": "oreb",
    "DREB": "dreb",
    "STL": "stl",
    "BLK": "blk",
    "TOV": "tov",
    "PF": "pf",
    "FGA": "fga",
    "FGM": "fgm",
    "FG3A": "fg3a",
    "FG3M": "fg3m",
    "FTA": "fta",
    "FTM": "ftm",
    "FG_PCT": "fg_pct",
    "FG3_PCT": "fg3_pct",
    "FT_PCT": "ft_pct",
    "USG_PCT": "usg_pct",
    "TS_PCT": "ts_pct",
    "EFG_PCT": "efg_pct",
    "MATCHUP": "matchup",
    "WL": "wl",
    "PLUS_MINUS": "plus_minus",
    "PLUS_MINUS_PER_GAME": "plus_minus_per_game",
    # Team metrics
    "NET_RTG": "net_rtg",
    "ORTG": "ortg",
    "DRTG": "drtg",
    "DRTG": "drtg",
    "PACE": "pace",
    # Tracking / synergy source columns
    "POSS_PCT": "poss_pct",
    "PPP": "ppp",
    "DRIVES": "drives",
    "DRIVE_PTS": "drive_pts",
    "DRIVE_FG_PCT": "drive_fg_pct",
    "DRIVE_AST": "drive_ast",
    "DRIVE_TOV": "drive_tov",
    "PASSES_MADE": "passes_made",
    "SECONDARY_AST": "secondary_ast",
    "POTENTIAL_AST": "potential_ast",
    "TOUCHES": "touches",
    "TIME_OF_POSS": "time_of_poss",
    "AVG_SEC_PER_TOUCH": "avg_sec_per_touch",
    "AVG_DRIB_PER_TOUCH": "avg_drib_per_touch",
    "FRONT_CT_TOUCHES": "front_ct_touches",
    "DIST_MILES": "dist_miles",
    "DIST_MILES_DEF": "dist_miles_def",
    "AVG_SPEED": "avg_speed",
    "AVG_SPEED_DEF": "avg_speed_def",
    "OREB_CONTEST": "oreb_contest",
    "DREB_CONTEST": "dreb_contest",
    "REB_CONTEST": "reb_contest",
    "DREB_CHANCES": "dreb_chances",
    "DREB_CHANCE_PCT": "dreb_chance_pct",
    # Catch-shoot
    "CATCH_SHOOT_FGM": "catch_shoot_fgm",
    "CATCH_SHOOT_FGA": "catch_shoot_fga",
    "CATCH_SHOOT_FG_PCT": "catch_shoot_fg_pct",
    "CATCH_SHOOT_PTS": "catch_shoot_pts",
    "CATCH_SHOOT_FG3M": "catch_shoot_fg3m",
    "CATCH_SHOOT_FG3A": "catch_shoot_fg3a",
    "CATCH_SHOOT_FG3_PCT": "catch_shoot_fg3_pct",
    # Shot zones
    "AT_RIM_PLUS_PAINT_FREQ": "at_rim_plus_paint_freq",
    "AT_RIM_FG_PCT": "at_rim_fg_pct",
    "MIDRANGE_FG_PCT": "midrange_fg_pct",
    "AT_RIM_FREQ": "at_rim_freq",
    "PAINT_FREQ": "paint_freq",
    "MIDRANGE_FREQ": "midrange_freq",
    "CORNER3_FREQ": "corner3_freq",
    "AB3_FREQ": "ab3_freq",
    "PAINT_FGA": "paint_fga",
    "TOTAL_FGA": "total_fga",
    "RA_FGA": "ra_fga",
    "MR_FGA": "mr_fga",
    # Defense tracking columns
    "CLOSE_DEF_PERSON_ID": "close_def_person_id",
    "D_FG_PCT": "d_fg_pct",
    "PCT_PLUSMINUS": "pct_plusminus",
    "D_FGM": "d_fgm",
    "D_FGA": "d_fga",
    "FREQ": "freq",
    "LT_06_PCT": "lt_06_pct",
    "PLUSMINUS": "plusminus",
    "FGM_LT_06": "fgm_lt_06",
    "FGA_LT_06": "fga_lt_06",
    "DEF_RIM_FGM": "def_rim_fgm",
    "DEF_RIM_FGA": "def_rim_fga",
    "DEF_RIM_FG_PCT": "def_rim_fg_pct",
    "DEF_RIM_FG_PCT": "def_rim_fg_pct",
    # Hustle
    "DEFLECTIONS": "deflections",
    "CONTESTED_SHOTS": "contested_shots",
    "CONTESTED_SHOTS_2PT": "contested_shots_2pt",
    "CONTESTED_SHOTS_3PT": "contested_shots_3pt",
    "CHARGES_DRAWN": "charges_drawn",
    "SCREEN_ASSISTS": "screen_assists",
    "DEF_LOOSE_BALLS_RECOVERED": "def_loose_balls_recovered",
    "LOOSE_BALLS_RECOVERED": "loose_balls_recovered",
    "DEF_BOXOUTS": "def_boxouts",
    "BOX_OUTS": "box_outs",
    # Computed per-36 / derived (archetypes)
    "BALL_DOMINANT_PCT": "ball_dominant_pct",
    "ON_BALL_CREATION": "on_ball_creation",
    "POST_CREATION": "post_creation",
    "AST_PER36": "ast_per36",
    "PTS_PER36": "pts_per36",
    "REB_PER36": "reb_per36",
    "TOV_PER36": "tov_per36",
    "STL_PER36": "stl_per36",
    "BLK_PER36": "blk_per36",
    "SECONDARY_AST_PER36": "secondary_ast_per36",
    "POTENTIAL_AST_PER36": "potential_ast_per36",
    "PLAYMAKING_SCORE": "playmaking_score",
    "FG2A_RATE": "fg2a_rate",
    "FG3A_PER36": "fg3a_per36",
    "DRIVES_PER36": "drives_per36",
    "INTERIOR_RATIO": "interior_ratio",
    "CUT_PNRRM_PCT": "cut_pnrrm_pct",
    "MOVEMENT_SHOOTER_PCT": "movement_shooter_pct",
    "SPOTUP_PCT": "spotup_pct",
    "TRANSITION_PCT": "transition_pct",
    "PUTBACK_PCT": "putback_pct",
    "TOV_PCT": "tov_pct",
    "FT_RATE": "ft_rate",
    "TIME_OF_POSS_PER36": "time_of_poss_per36",
    "DRIBBLES_PER_TOUCH": "dribbles_per_touch",
    "PPG": "ppg",
    "TS_ZSCORE": "ts_zscore",
    "LEAGUE_AVG_TS": "league_avg_ts",
    "MPG": "mpg",
    # Defensive computed
    "STL_PER100_DEF_POSS": "stl_per100_def_poss",
    "DREB_PER100_DEF_POSS": "dreb_per100_def_poss",
    "BLK_PCT": "blk_pct",
    "RIM_FGA_RATE": "rim_fga_rate",
    "BLK_PG": "blk_pg",
    "DEF_RIM_FGA_PG": "def_rim_fga_pg",
    "POSS_DEF": "poss_def",
    "SECONDS_DEF": "seconds_def",
    "D_FG_DIFF": "d_fg_diff",
    # compute_linear_metrics columns
    "ORB": "orb",
    "DRB": "drb",
    "TRB": "trb",
    "NET_RTG": "net_rtg",
    "TEAM_NET_RTG": "team_net_rtg",
    "USG_RATE": "usg_rate",
    "REB_PCT": "reb_pct",
    # fit_rest_hca / fit_team_pace
    "TEAM_ABBREVIATION": "team_abbreviation",
    # compute_local_metrics
    "TEAM": "team",
    # team_game_logs
    "HOME_OR_AWAY": "home_or_away",
}


def replace_quoted_column(text: str, old: str, new: str) -> str:
    """Replace 'OLD' and "OLD" with 'new' and "new" respectively."""
    text = text.replace(f"'{old}'", f"'{new}'")
    text = text.replace(f'"{old}"', f'"{new}"')
    return text


def apply_colmap(text: str) -> str:
    # Sort by length desc to avoid partial-match bugs (e.g. FG3_PCT before FG_PCT)
    for old, new in sorted(COLMAP.items(), key=lambda x: -len(x[0])):
        text = replace_quoted_column(text, old, new)
    return text


# Special format-string patterns in compute_player_archetypes.py
def fix_archetype_format_strings(text: str) -> str:
    """Convert '{playtype.upper()}_POSS_PCT' etc. to '{playtype.lower()}_poss_pct'."""
    text = text.replace(
        "f'{playtype.upper()}_POSS_PCT'",
        "f'{playtype.lower()}_poss_pct'"
    )
    text = text.replace(
        "f'{playtype.upper()}_PPP'",
        "f'{playtype.lower()}_ppp'"
    )
    text = text.replace(
        "f'{playtype.upper()}_POSS'",
        "f'{playtype.lower()}_poss'"
    )
    # agg dict keys after load_standardized returns lowercase
    # 'POSS_PCT': mean → 'poss_pct': mean  (already handled by COLMAP)
    return text


# Playtype-specific column names generated by format strings
# e.g. 'ISOLATION_POSS_PCT' → 'isolation_poss_pct'
PLAYTYPES = [
    'ISOLATION', 'PRBALLHANDLER', 'POSTUP', 'CUT', 'PRROLLMAN',
    'HANDOFF', 'OFFSCREEN', 'SPOTUP', 'TRANSITION', 'OFFREBOUND', 'MISC'
]

def fix_playtype_columns(text: str) -> str:
    for pt in PLAYTYPES:
        for suffix in ['_POSS_PCT', '_PPP', '_POSS']:
            old = f'{pt}{suffix}'
            new = old.lower()
            text = replace_quoted_column(text, old, new)
    return text


IMPORT_LINE = "from src.data.schema_contract import load_standardized, save_standardized\n"


def add_import(text: str) -> str:
    """Add import after the last existing import block."""
    if "schema_contract" in text:
        return text  # already imported
    # Find first non-import line after imports
    import re
    # Insert after 'import pandas as pd' or 'import ...' block
    # Find a good insertion point: after the last top-level import
    lines = text.split('\n')
    last_import_idx = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('import ') or stripped.startswith('from '):
            last_import_idx = i
    if last_import_idx >= 0:
        lines.insert(last_import_idx + 1, IMPORT_LINE.rstrip())
        return '\n'.join(lines)
    return IMPORT_LINE + text


def replace_reads(text: str) -> str:
    """Replace pd.read_parquet(...) with load_standardized(...)."""
    return text.replace('pd.read_parquet(', 'load_standardized(')


def replace_writes(text: str) -> str:
    """Replace .to_parquet(path, index=False) with save_standardized(df, path)."""
    # Pattern: .to_parquet(OUTPUT_FILE, index=False) etc.
    # We use a regex to handle various forms
    import re
    # .to_parquet(path_var, index=False)
    text = re.sub(
        r'(\w+)\.to_parquet\(([^,\)]+),\s*index=False\)',
        lambda m: f'save_standardized({m.group(1)}, {m.group(2)})',
        text
    )
    # .to_parquet(path_var)  (no keyword args)
    text = re.sub(
        r'(\w+)\.to_parquet\(([^,\)]+)\)',
        lambda m: f'save_standardized({m.group(1)}, {m.group(2)})',
        text
    )
    return text


def transform(path: str) -> str:
    with open(path) as f:
        text = f.read()

    text = add_import(text)
    text = replace_reads(text)
    text = replace_writes(text)
    text = apply_colmap(text)
    text = fix_archetype_format_strings(text)
    text = fix_playtype_columns(text)
    return text


if __name__ == "__main__":
    scripts = [
        "src/data_compute/compute_player_archetypes.py",
        "src/data_compute/compute_defensive_archetypes_v2.py",
        "src/data_compute/compute_linear_metrics.py",
        "src/data_compute/compute_player_profiles.py",
        "src/data_compute/compute_position_estimate.py",
        "src/data_compute/compute_local_metrics.py",
        "src/data_compute/fit_rest_hca_coefficients.py",
        "src/data_compute/fit_team_pace.py",
    ]
    for path in scripts:
        result = transform(path)
        with open(path, 'w') as f:
            f.write(result)
        print(f"  Updated: {path}")

    print("\nDone. Verify with: pytest -q tests/ && python3 scripts/validate_backup_integrity.py --post-rewrite --quiet --dirs processed")
