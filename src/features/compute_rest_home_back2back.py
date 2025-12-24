"""Compute and persist simple situational features:

- team-level: `rest_days`, `home_flag`, `back_to_back` (based on previous game for that team)
- player-level: attach team-level context to player-game rows and persist

Writes:
- `data/features/team_game_context.parquet`
- `data/features/player_game_context.parquet`

This script is defensive about column name casing and common MATCHUP formats.
"""
from __future__ import annotations
import os
from pathlib import Path
import pandas as pd


def _upper_map(df: pd.DataFrame) -> dict:
    return {c.upper(): c for c in df.columns}


def choose_team_token_col(df: pd.DataFrame) -> str:
    cmap = _upper_map(df)
    for cand in ('TEAM_ID', 'TEAM_ABBREVIATION', 'TEAM'):
        if cand in cmap:
            return cmap[cand]
    # fallback: we'll derive from MATCHUP
    return None


def extract_team_from_matchup(matchup: str) -> str:
    if not isinstance(matchup, str):
        return None
    # Examples: 'SAS @ IND', 'IND vs. SAS', 'GSW vs LAC'
    if ' @ ' in matchup:
        return matchup.split(' @ ')[0].strip()
    if ' vs ' in matchup:
        return matchup.split(' vs ')[0].strip()
    # generic token
    return matchup.split()[0].strip()


def compute_team_context(player_parquet: str, out_dir: str = 'data/features') -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(player_parquet)
    cmap = _upper_map(df)

    # detect GAME_DATE column
    date_col = cmap.get('GAME_DATE') or cmap.get('GAME_DATE_EST') or cmap.get('DATE')
    if date_col is None:
        raise KeyError('GAME_DATE column not found in player logs')
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    # determine team token
    team_col = choose_team_token_col(df)
    if team_col is None:
        # derive from MATCHUP
        matchup_col = cmap.get('MATCHUP')
        if matchup_col is None:
            raise KeyError('No TEAM or MATCHUP column found in player logs')
        df = df.copy()
        df['DERIVED_TEAM'] = df[matchup_col].astype(str).map(extract_team_from_matchup)
        team_col = 'DERIVED_TEAM'

    # build team-game unique rows
    # prefer SEASON when available to avoid cross-season ordering issues
    season_col = cmap.get('SEASON') or cmap.get('SEASON_ID')
    keys = [k for k in ([season_col] if season_col else [])]
    # ensure GAME_ID exists
    game_id_col = cmap.get('GAME_ID') or cmap.get('GAMEID') or cmap.get('Game_ID')
    if game_id_col is None:
        raise KeyError('GAME_ID not found in player logs')
    keys += [game_id_col, team_col]

    team_games = df.drop_duplicates(subset=keys).copy()
    # keep relevant columns
    team_games = team_games[[c for c in keys if c is not None] + [date_col, cmap.get('MATCHUP')] if cmap.get('MATCHUP') in df.columns else [date_col]]

    # normalize column names
    team_games = team_games.rename(columns={date_col: 'GAME_DATE', game_id_col: 'GAME_ID', team_col: 'TEAM_TOKEN'})

    # sort per team to compute previous game
    team_games = team_games.sort_values(['TEAM_TOKEN', 'GAME_DATE'])
    team_games['PREV_GAME_DATE'] = team_games.groupby('TEAM_TOKEN')['GAME_DATE'].shift(1)
    team_games['REST_DAYS'] = (team_games['GAME_DATE'] - team_games['PREV_GAME_DATE']).dt.days
    # define back_to_back when previous game was 1 day earlier
    team_games['BACK_TO_BACK'] = team_games['REST_DAYS'] == 1

    # compute home_flag from MATCHUP where possible
    def is_home(row):
        m = row.get('MATCHUP')
        if not isinstance(m, str):
            return None
        if ' @ ' in m:
            # first token is away
            return False
        # assume vs means home
        if ' vs ' in m or ' vs.' in m or ' vs,' in m:
            return True
        # fallback: if first token equals TEAM_TOKEN assume home
        try:
            tok = str(m).split()[0]
            return tok == row.get('TEAM_TOKEN')
        except Exception:
            return None

    if cmap.get('MATCHUP') in df.columns:
        # ensure MATCHUP column available in team_games
        if 'MATCHUP' not in team_games.columns and cmap.get('MATCHUP') in df.columns:
            team_games['MATCHUP'] = team_games.apply(lambda r: df[(df['GAME_ID'] == r['GAME_ID']) & (df.get(team_col) == r['TEAM_TOKEN'])][cmap.get('MATCHUP')].iat[0] if not df[(df['GAME_ID'] == r['GAME_ID']) & (df.get(team_col) == r['TEAM_TOKEN'])].empty else None, axis=1)
        team_games['HOME_FLAG'] = team_games.apply(is_home, axis=1)
    else:
        team_games['HOME_FLAG'] = None

    # fill REST_DAYS NaN with large number for season openers
    team_games['REST_DAYS'] = team_games['REST_DAYS'].fillna(99).astype(int)

    # persist team-level context
    team_out = os.path.join(out_dir, 'team_game_context.parquet')
    team_games.to_parquet(team_out, index=False)
    print(f'Wrote team context to {team_out} rows={len(team_games)}')

    # Now attach to player rows by GAME_ID and team token
    player_ctx = df.copy()
    # ensure GAME_ID string
    player_ctx[game_id_col] = player_ctx[game_id_col].astype(str)
    team_games['GAME_ID'] = team_games['GAME_ID'].astype(str)

    # create player token column name used for merge
    player_token_col = None
    if 'DERIVED_TEAM' in player_ctx.columns:
        player_token_col = 'DERIVED_TEAM'
    else:
        # find original team col name (case-insensitive)
        for cand in ('TEAM_ID', 'TEAM_ABBREVIATION', 'TEAM'):
            if cand in player_ctx.columns:
                player_token_col = cand
                break

    if player_token_col is None:
        # fallback to extracting from MATCHUP
        match_col = cmap.get('MATCHUP')
        player_ctx['DERIVED_TEAM'] = player_ctx[match_col].astype(str).map(extract_team_from_matchup)
        player_token_col = 'DERIVED_TEAM'

    # merge on GAME_ID and token
    merged = player_ctx.merge(team_games[['GAME_ID', 'TEAM_TOKEN', 'REST_DAYS', 'BACK_TO_BACK', 'HOME_FLAG']], left_on=[game_id_col, player_token_col], right_on=['GAME_ID', 'TEAM_TOKEN'], how='left')
    # write player-level context
    player_out = os.path.join(out_dir, 'player_game_context.parquet')
    merged.to_parquet(player_out, index=False)
    print(f'Wrote player context to {player_out} rows={len(merged)}')


if __name__ == '__main__':
    # try common locations for player logs
    candidates = [
        'data/player_game_logs.parquet',
        'data/historical/player_game_logs.parquet',
        'data/historical/final_player_game_logs.parquet'
    ]
    p = next((c for c in candidates if os.path.exists(c)), None)
    if not p:
        print('No player_game_logs parquet found; please run ingestion first')
        raise SystemExit(2)
    compute_team_context(p)
