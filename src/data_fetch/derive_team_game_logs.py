"""Derive team-level game logs from player-level game logs.

Reads: data/historical/player_game_logs.parquet
Writes: data/historical/team_game_logs.parquet

The script is defensive: it looks for common team and season column names,
uses CORE_STATS intersection to aggregate totals, and computes WIN by
comparing team PTS to opponent PTS per GAME_ID.
"""
from __future__ import annotations
import os
import sys
from typing import List

import pandas as pd

CORE_STATS = [
    'PTS', 'REB', 'OREB', 'DREB', 'AST', 'STL', 'BLK', 'TOV', 'PF',
    'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'MIN', 'PLUS_MINUS'
]


def _upper_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.upper() for c in df.columns]
    return df


def find_column(df: pd.DataFrame, candidates: List[str]) -> str | None:
    for c in candidates:
        if c.upper() in df.columns:
            return c.upper()
    return None


def infer_season(df: pd.DataFrame, season_col: str | None) -> pd.Series:
    if season_col and season_col in df.columns:
        return df[season_col].astype(str)
    # fallback: try to infer from GAME_DATE
    if 'GAME_DATE' in df.columns:
        try:
            dates = pd.to_datetime(df['GAME_DATE'], errors='coerce')
            # season convention: season starts in Oct -> Oct-Dec use year-year+1
            def to_season(dt):
                if pd.isna(dt):
                    return None
                y = dt.year
                if dt.month >= 10:
                    return f"{y}-{str(y+1)[-2:]}"
                else:
                    return f"{y-1}-{str(y)[-2:]}"
            return dates.map(to_season)
        except Exception:
            pass
    # last resort: use env or default
    seasons = os.environ.get('SEASONS', '2022-23').split(',')
    return pd.Series([seasons[0].strip()] * len(df))


def main():
    p_in = os.path.join('data', 'historical', 'player_game_logs.parquet')
    p_out = os.path.join('data', 'historical', 'team_game_logs.parquet')
    if not os.path.exists(p_in):
        print(f'ERROR: {p_in} not found', file=sys.stderr)
        sys.exit(2)

    df = pd.read_parquet(p_in)
    if df.empty:
        print('ERROR: player game logs empty', file=sys.stderr)
        sys.exit(2)

    df = _upper_cols(df)

    # Find team identifier column (allow common names). If missing, try to derive from MATCHUP.
    team_col = find_column(df, ['TEAM_ID', 'TEAM_ABBREVIATION', 'TEAM_ABBR', 'TEAM', 'TEAM_NAME'])
    if team_col is None:
        if 'MATCHUP' in df.columns:
            # derive team abbreviation from MATCHUP field (token before space, e.g. 'TOR vs MIL')
            def _extract_team(matchup):
                try:
                    return str(matchup).split()[0]
                except Exception:
                    return None
            df['TEAM_ABBREVIATION'] = df['MATCHUP'].apply(_extract_team)
            team_col = 'TEAM_ABBREVIATION'
        else:
            print('ERROR: could not find a team column in player logs', file=sys.stderr)
            sys.exit(2)

    # Ensure GAME_ID exists
    if 'GAME_ID' not in df.columns:
        print('ERROR: GAME_ID missing from player logs', file=sys.stderr)
        sys.exit(2)
    df['GAME_ID'] = df['GAME_ID'].astype(str)

    # Season handling
    season_col = find_column(df, ['SEASON', 'SEASON_ID', 'SEASON_YEAR'])
    df['SEASON'] = infer_season(df, season_col)

    # Decide which stats to aggregate
    stats = [s for s in CORE_STATS if s in df.columns]
    if not stats:
        # fallback: numeric columns excluding some obvious identifiers
        num = df.select_dtypes(include='number').columns.tolist()
        # exclude columns that are identifiers
        exclude = {'GAME_ID'}
        stats = [c for c in num if c not in exclude]
        if not stats:
            print('WARN: no numeric stats found to aggregate; output will be limited', file=sys.stderr)

    group_cols = ['SEASON', 'GAME_ID', team_col]

    agg_map = {s: 'sum' for s in stats}
    # Aggregate totals per team-game
    team_aggs = df.groupby(group_cols).agg(agg_map).reset_index()

    # Normalize TEAM column name to TEAM_ID in output (string)
    team_aggs = team_aggs.rename(columns={team_col: 'TEAM_ID'})
    team_aggs['TEAM_ID'] = team_aggs['TEAM_ID'].astype(str)

    # Compute WINS by comparing team PTS to opponent PTS per GAME_ID
    if 'PTS' in team_aggs.columns:
        opp = team_aggs[['GAME_ID', 'TEAM_ID', 'PTS']].rename(columns={'TEAM_ID': 'OPP_TEAM_ID', 'PTS': 'OPP_PTS'})
        merged = team_aggs.merge(opp, on='GAME_ID', how='left')
        merged = merged[merged['TEAM_ID'] != merged['OPP_TEAM_ID']]
        merged = merged.drop_duplicates(subset=['SEASON', 'GAME_ID', 'TEAM_ID'])
        merged['WIN'] = merged['PTS'] > merged['OPP_PTS']
        # Resolve ties where PTS == OPP_PTS using player-level WL if available
        try:
            # identify game ids where both teams have identical PTS
            pts_per_game_nunique = merged.groupby('GAME_ID')['PTS'].nunique()
            tie_gids = pts_per_game_nunique[pts_per_game_nunique == 1].index.tolist()
            resolved = 0
            unresolved = []
            if tie_gids and not df.empty:
                # try to resolve each tie using player-level WL majority per team token
                for gid in tie_gids:
                    # player-level rows for this game
                    ply = df[df['GAME_ID'].astype(str) == str(gid)]
                    if ply.empty:
                        unresolved.append(gid)
                        continue
                    # determine a team token column on player rows
                    if 'TEAM_ID' in ply.columns:
                        ply_team_col = 'TEAM_ID'
                        ply['TEAM_TOKEN'] = ply[ply_team_col].astype(str)
                    elif 'TEAM_ABBREVIATION' in ply.columns:
                        ply_team_col = 'TEAM_ABBREVIATION'
                        ply['TEAM_TOKEN'] = ply[ply_team_col].astype(str)
                    elif 'MATCHUP' in ply.columns:
                        ply['TEAM_TOKEN'] = ply['MATCHUP'].astype(str).map(lambda m: str(m).split()[0])
                    else:
                        unresolved.append(gid)
                        continue

                    # count player-level WL wins per token
                    if 'WL' in ply.columns:
                        wins_by_token = ply[ply['WL'] == 'W'].groupby('TEAM_TOKEN').size()
                        if not wins_by_token.empty:
                            winner_token = wins_by_token.idxmax()
                            # map winner_token to merged TEAM_ID values (they should match tokens)
                            if winner_token is not None:
                                merged.loc[(merged['GAME_ID'] == gid) & (merged['TEAM_ID'].astype(str) == str(winner_token)), 'WIN'] = True
                                merged.loc[(merged['GAME_ID'] == gid) & (merged['TEAM_ID'].astype(str) != str(winner_token)), 'WIN'] = False
                                resolved += 1
                                continue

                    # fallback: use PLUS_MINUS if available on merged rows
                    grows = merged[merged['GAME_ID'] == gid]
                    if 'PLUS_MINUS' in grows.columns:
                        # choose the team with higher PLUS_MINUS as winner
                        try:
                            best_idx = grows['PLUS_MINUS'].astype(float).idxmax()
                            winner_team = merged.loc[best_idx, 'TEAM_ID']
                            merged.loc[(merged['GAME_ID'] == gid) & (merged['TEAM_ID'].astype(str) == str(winner_team)), 'WIN'] = True
                            merged.loc[(merged['GAME_ID'] == gid) & (merged['TEAM_ID'].astype(str) != str(winner_team)), 'WIN'] = False
                            resolved += 1
                            continue
                        except Exception:
                            pass

                    unresolved.append(gid)

            # optional: log resolution counts to stdout for visibility
            if len(tie_gids) > 0:
                print(f"Resolved {resolved} of {len(tie_gids)} tied games using player WL/PLUS_MINUS; unresolved={len(unresolved)}")
        except Exception:
            # don't fail derive on diagnostic errors
            pass
    else:
        merged = team_aggs.copy()
        merged['WIN'] = False

    # final per-team-per-game totals (keep base stat names like PTS, AST, etc.)
    final = merged.copy()

    # compute GAMES and WINS per season/team
    games = final.groupby(['SEASON', 'TEAM_ID'])['GAME_ID'].nunique().rename('GAMES')
    wins = final.groupby(['SEASON', 'TEAM_ID'])['WIN'].sum().rename('WINS')
    summary = pd.concat([games, wins], axis=1).reset_index()

    # Merge totals into summary (not necessary for writing team_game_logs but keep for debug)
    stat_cols = [c for c in final.columns if c in stats]
    if stat_cols:
        stat_aggs = final.groupby(['SEASON', 'TEAM_ID'])[stat_cols].sum().reset_index()
        # rename stat totals to {STAT}_sum for compatibility with other pieces that may expect that
        stat_aggs = stat_aggs.rename(columns={c: f"{c}_sum" for c in stat_cols})
        summary = summary.merge(stat_aggs, on=['SEASON', 'TEAM_ID'], how='left')

    # Write team_game_logs.parquet in team-game-per-row format
    # But also write the per-game rows (final) for downstream compatibility
    os.makedirs(os.path.dirname(p_out), exist_ok=True)
    # final contains per-team-per-game rows
    final_out = final.copy()
    final_out.to_parquet(p_out, index=False)
    print(f'Wrote derived team game logs to {p_out}; rows={len(final_out)}')


if __name__ == '__main__':
    main()
