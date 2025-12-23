"""Clean cascade fetcher replacement (safe to run).

This file is a sanitized replacement. It avoids network calls by default and
derives team-game logs from local player-game parquet files. Run this instead
of the corrupted module until we fully clean the original file.
"""

from __future__ import annotations
import os
import traceback
from typing import List, Optional

import pandas as pd


def try_data_nba_net(*args, **kwargs) -> pd.DataFrame:
    print('DEBUG: data.nba.net fetch disabled in this clean stub')
    return pd.DataFrame()


def try_balldontlie(*args, **kwargs) -> pd.DataFrame:
    print('DEBUG: balldontlie fetch disabled in this clean stub')
    return pd.DataFrame()


def derive_from_player_logs(seasons: List[str]) -> pd.DataFrame:
    candidates = [
        'data/player_game_logs.parquet',
        'data/historical/player_game_logs.parquet',
        'data/historical/final_player_game_logs.parquet',
    ]
    path = next((c for c in candidates if os.path.exists(c)), None)
    if not path:
        print('WARN: no player_game_logs parquet found for deriving team logs')
        return pd.DataFrame()

    print(f'DEBUG: deriving team logs from player logs at {path}')
    df = pd.read_parquet(path)

    cols = list(df.columns)
    # case-insensitive map
    colmap = {c.lower(): c for c in cols}

    game_id_candidates = ['game_id', 'gameid', 'game id', 'gid']
    date_candidates = ['game_date', 'game_date_est', 'date', 'gamedate']
    team_candidates = ['team_id', 'team', 'team_abbreviation', 'team_abbrev', 'team_abbr', 'team_code', 'team_name', 'teamname']

    game_id_col = next((colmap[c] for c in game_id_candidates if c in colmap), None)
    date_col = next((colmap[c] for c in date_candidates if c in colmap), None)
    team_col = next((colmap[c] for c in team_candidates if c in colmap), None)

    # If no explicit team column, try to extract from MATCHUP
    if team_col is None and 'matchup' in colmap:
        matchup_col = colmap['matchup']
        def _extract_team(matchup):
            try:
                return str(matchup).split()[0]
            except Exception:
                return None
        df['TEAM_ABBREVIATION'] = df[matchup_col].apply(_extract_team)
        team_col = 'TEAM_ABBREVIATION'

    if not game_id_col:
        # if we have a date and a team-like column, synthesize GAME_ID
        if date_col and team_col:
            df['GAME_ID'] = df[date_col].astype(str) + '_' + df[team_col].astype(str)
            game_id_col = 'GAME_ID'

    if not game_id_col:
        print('WARN: cannot synthesize GAME_ID; aborting derive')
        return pd.DataFrame()

    # coerce GAME_ID to string
    if game_id_col and game_id_col != 'GAME_ID':
        try:
            df['GAME_ID'] = df[game_id_col].astype(str)
        except Exception:
            df['GAME_ID'] = df[game_id_col].apply(lambda x: str(x))

    group_keys = ['GAME_ID']
    if team_col:
        group_keys.append(team_col)

    # pick numeric columns but exclude identifier-like fields
    id_like = {'player_id', 'person_id', 'video_available', 'id'}
    numeric = [c for c in df.select_dtypes(include=['number']).columns if c.lower() not in ('season',) and c.lower() not in id_like]
    if not numeric:
        print('WARN: no numeric columns to aggregate')
        return pd.DataFrame()

    agg = df.groupby(group_keys)[numeric].sum().reset_index()
    if team_col and team_col != 'TEAM_ID':
        agg = agg.rename(columns={team_col: 'TEAM_ID'})

    return agg


def main():
    seasons = os.environ.get('SEASONS', '2022-23,2023-24,2024-25').split(',')
    seasons = [s.strip() for s in seasons if s.strip()]
    os.makedirs('data/historical', exist_ok=True)

    df = try_data_nba_net(seasons, headers={})
    if df.empty:
        df = try_balldontlie(seasons, headers={})
    if df.empty:
        df = derive_from_player_logs(seasons)

    out_path = 'data/historical/team_game_logs.parquet'
    if not df.empty:
        df.to_parquet(out_path, index=False)
        print(f'Saved derived team logs to {out_path} ({len(df)} rows)')
    else:
        pd.DataFrame().to_parquet(out_path, index=False)
        print('No team logs collected; wrote empty parquet')


if __name__ == '__main__':
    main()
