import time
import re
import pandas as pd
from nba_api.stats.endpoints import teamgamelog, playergamelog, playerprofilev2, leaguedashplayerstats
from nba_api.stats.static import teams, players as static_players


def flatten_json_schema(schema, parent_key='', sep='.'):
        items = []
        for k, v in schema.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                        items.extend(flatten_json_schema(v, new_key, sep=sep))
                else:
                        items.append(new_key)
        return items


# Best-effort alias map for common stat names → NBA stat column names
COMMON_ALIASES = {
        'usage_rate': ['USG_PCT', 'USG%'],
        'ts_percent': ['TS_PCT', 'TS%'],
        'efg_percent': ['EFG_PCT', 'eFG%'],
        'turnover_rate': ['TOV_PCT', 'TOV%'],
        'assist_percent': ['AST_PCT'],
        'oreb_percent': ['OREB_PCT'],
        'dreb_percent': ['DREB_PCT'],
        'three_point_accuracy': ['FG3_PCT', '3P%'],
        'minutes_per_game': ['MIN'],
        'points_per_shot': ['PTS_PER_FG_ATT', 'PTS_PER_SHOT'],
        'free_throw_rate': ['FTA_RATE', 'FT_RATE'],
        'block_percent': ['BLK_PCT'],
        'steal_percent': ['STL_PCT'],
        'defensive_rating': ['D_RATING', 'DEF_RATING', 'DRtg'],
        'pace': ['PACE']
}


def normalize(s: str) -> str:
        return re.sub(r'[^a-z0-9]', '', str(s).lower())


if __name__ == "__main__":
        # Inline JSON schema (as provided) — flattened for comparison
        JSON_SCHEMA = {
            "player_id": "integer",
            "season": "string",
            "team_id": "integer",

            "context": {
                "usage_and_role": {
                    "usage_rate": "float",
                    "minutes_per_game": "float",
                    "production_points_responsible": {
                        "points_scored": "float",
                        "points_assisted": "float",
                        "total_points_responsible": "float"
                    },
                    "drives_per_game": "float",
                    "drives_per_100_possessions": "float",
                    "primary_playtypes": {
                        "isolation_frequency": "float",
                        "pick_and_roll_ball_handler_frequency": "float",
                        "pick_and_roll_rollman_frequency": "float",
                        "spot_up_frequency": "float",
                        "post_up_frequency": "float",
                        "handoff_frequency": "float",
                        "cut_frequency": "float",
                        "off_screen_frequency": "float",
                        "transition_frequency": "float",
                        "putbacks_frequency": "float",
                        "misc_frequency": "float"
                    }
                },
                "team_context": {
                    "team_net_rating": "float",
                    "team_net_rating_difference_from_avg": "float",
                    "team_pace": "float",
                    "team_possession_factor": "float"
                }
            },

            "offensive_impact": {
                "efficiency": {
                    "ts_percent": "float",
                    "efg_percent": "float",
                    "points_per_shot": "float",
                    "turnover_rate": "float"
                },
                "creation_and_playmaking": {
                    "assist_percent": "float",
                    "ast_to_tov_ratio": "float",
                    "potential_assists": "float",
                    "secondary_assists": "float",
                    "hockey_assists": "float"
                },
                "scoring_profile": {
                    "points_per_75_possessions": "float",
                    "rim_freq": "float",
                    "mid_range_freq": "float",
                    "three_point_freq": "float",
                    "finishing_fg_percent": "float",
                    "three_point_accuracy": "float",
                    "free_throw_rate": "float"
                },
                "offensive_rebounding": {
                    "oreb_percent": "float",
                    "putbacks_per_game": "float"
                },
                "spacing_and_gravity": {
                    "three_point_attempt_rate": "float",
                    "catch_and_shoot_efficiency": "float",
                    "pull_up_efficiency": "float"
                }
            },

            "defensive_impact": {
                "perimeter_defense": {
                    "opponent_fg_percent_defended": "float",
                    "deflections_per_game": "float",
                    "steal_percent": "float"
                },
                "interior_defense": {
                    "block_percent": "float",
                    "rim_defended_fg_percent": "float",
                    "opponent_rim_attempts_faced": "float"
                },
                "rebounding": {
                    "dreb_percent": "float"
                },
                "defensive_box_plus_minus": "float",
                "defensive_rating": "float"
            },

            "impact_and_value": {
                "overall_offensive_impact": "float",
                "overall_defensive_impact": "float",
                "net_impact": "float",
                "on_off_net_rating_diff": "float",
                "box_plus_minus": "float",
                "wins_above_replacement": "float",
                "player_impact_estimate": "float"
            },

            "adjustable_weights": {
                "usage_and_role_weight": "float",
                "offensive_impact_weight": "float",
                "defensive_impact_weight": "float",
                "team_context_weight": "float",
                "efficiency_weight": "float",
                "creation_weight": "float",
                "rebounding_weight": "float"
            }
        }

        flat_keys = flatten_json_schema(JSON_SCHEMA)
        print(f"Flattened JSON schema keys ({len(flat_keys)}):")
        for k in flat_keys:
                print(" -", k)

        # Choose sample team and player
        all_teams = teams.get_teams()
        lakers = [t for t in all_teams if 'Los Angeles Lakers' in t['full_name']]
        if lakers:
                team_id = lakers[0]['id']
        else:
                team_id = all_teams[0]['id']

        # Known sample player (LeBron James = 2544) as a stable test
        sample_player_id = 2544

        endpoints_columns = {}

        # 1) teamgamelog
        try:
                print("\nFetching TeamGameLog columns from NBA API (sample team)...")
                tdf = teamgamelog.TeamGameLog(team_id=team_id, season='2023-24').get_data_frames()[0]
                endpoints_columns['teamgamelog'] = list(tdf.columns)
                print(' teamgamelog columns:', endpoints_columns['teamgamelog'])
                time.sleep(0.6)
        except Exception as e:
                print(' teamgamelog fetch failed:', e)

        # 2) playergamelog
        try:
                print("\nFetching PlayerGameLog columns from NBA API (sample player)...")
                pg = playergamelog.PlayerGameLog(player_id=sample_player_id, season='2023-24')
                pdf = pg.get_data_frames()[0]
                endpoints_columns['playergamelog'] = list(pdf.columns)
                print(' playergamelog columns:', endpoints_columns['playergamelog'])
                time.sleep(0.6)
        except Exception as e:
                print(' playergamelog fetch failed:', e)

        # 3) playerprofilev2 (has many advanced metrics)
        try:
                print("\nFetching PlayerProfileV2 columns (sample player)...")
                prof = playerprofilev2.PlayerProfileV2(player_id=sample_player_id, season_type_all_star='Regular Season')
                cols = []
                for df in prof.get_data_frames():
                        cols.extend(list(df.columns))
                # dedupe while preserving order
                seen = set()
                cols_unique = [c for c in cols if not (c in seen or seen.add(c))]
                endpoints_columns['playerprofilev2'] = cols_unique
                print(' playerprofilev2 columns (sample across tables):', endpoints_columns['playerprofilev2'])
                time.sleep(0.6)
        except Exception as e:
                print(' playerprofilev2 fetch failed:', e)

        # 4) leaguedashplayerstats (season-level aggregated stats)
        try:
                print("\nFetching LeagueDashPlayerStats columns (season-level aggregated)...")
                ldp = leaguedashplayerstats.LeagueDashPlayerStats(season='2023-24', per_mode_detailed='PerGame')
                ldp_df = ldp.get_data_frames()[0]
                endpoints_columns['leaguedashplayerstats'] = list(ldp_df.columns)
                print(' leaguedashplayerstats columns:', endpoints_columns['leaguedashplayerstats'])
                time.sleep(0.6)
        except Exception as e:
                print(' leaguedashplayerstats fetch failed:', e)

        # Build union of all columns
        all_api_cols = set()
        for name, cols in endpoints_columns.items():
                all_api_cols.update(cols)

        print(f"\nTotal unique API columns discovered across endpoints: {len(all_api_cols)}")

        # Try to match each flattened JSON key to API columns
        print("\nBest-effort mapping from JSON keys → API columns:")
        norm_api_map = {normalize(c): c for c in all_api_cols}

        for key in flat_keys:
                short = key.split('.')[-1]
                normalized = normalize(short)
                found = []

                # 1) exact normalized match
                if normalized in norm_api_map:
                        found.append(norm_api_map[normalized])

                # 2) check common alias map
                if not found and short in COMMON_ALIASES:
                        for alias in COMMON_ALIASES[short]:
                                if alias in all_api_cols:
                                        found.append(alias)
                                elif normalize(alias) in norm_api_map:
                                        found.append(norm_api_map[normalize(alias)])

                # 3) substring fuzzy match
                if not found:
                        for cand_norm, cand in norm_api_map.items():
                                if normalized and (normalized in cand_norm or cand_norm in normalized):
                                        found.append(cand)
                # dedupe
                found = list(dict.fromkeys(found))

                if found:
                        print(f" - {key}  -> {found}")
                else:
                        print(f" - {key}  -> [NOT FOUND in sampled API endpoints]")

        print('\nDone. If many keys show [NOT FOUND], we can (A) query additional endpoints, (B) perform online name lookups for each metric, or (C) try different sample players/seasons.')

