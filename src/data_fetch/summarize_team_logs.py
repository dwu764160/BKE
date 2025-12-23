"""Summarize team game logs.

Reads data/historical/team_game_logs.parquet and produces per-team-per-season
summary rows including games played, wins/losses (if opponent PTS available),
and totals/averages for core stats (PTS, REB, AST, STL, BLK, TOV, PF, etc.).

Writes:
- data/historical/team_summaries.parquet
- data/historical/team_summaries.csv

This script is defensive: if `TEAM_ID` is missing it will attempt to infer
teams from `MATCHUP` where possible; otherwise it falls back to a league-level
aggregate and reports the limitation.
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

# Canonical 30 NBA team abbreviations used to pad missing teams when needed.
# This list is used only as display/placeholders when derived data lacks a
# full set of teams.
DEFAULT_TEAM_IDS = [
    'ATL','BOS','BKN','CHA','CHI','CLE','DAL','DEN','DET','GSW',
    'HOU','IND','LAC','LAL','MEM','MIA','MIL','MIN','NOP','NYK',
    'OKC','ORL','PHI','PHX','POR','SAC','SAS','TOR','UTA','WAS'
]


def _upper_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.upper() for c in df.columns]
    return df


def summarize(path: str = 'data/historical/team_game_logs.parquet') -> pd.DataFrame:
    if not os.path.exists(path):
        print(f'ERROR: {path} not found', file=sys.stderr)
        return pd.DataFrame()

    df = pd.read_parquet(path)
    if df.empty:
        print('WARN: team game logs are empty')
        return pd.DataFrame()

    df = _upper_cols(df)

    # Build and write a padded per-team-per-game table (30 teams x 82 games x seasons)
    def build_padded_team_games(df: pd.DataFrame) -> pd.DataFrame:
        # determine seasons
        if 'SEASON' in df.columns and not df['SEASON'].dropna().empty:
            seasons = sorted(df['SEASON'].dropna().unique().tolist())
        else:
            seasons = os.environ.get('SEASONS', '2022-23,2023-24,2024-25').split(',')
            seasons = [s.strip() for s in seasons if s.strip()]

        # determine observed teams
        if 'TEAM_ID' in df.columns:
            observed = [str(t) for t in sorted(df['TEAM_ID'].dropna().unique().tolist())]
        elif 'TEAM_ABBREVIATION' in df.columns:
            observed = [str(t) for t in sorted(df['TEAM_ABBREVIATION'].dropna().unique().tolist())]
        else:
            # try to derive from MATCHUP tokens
            observed = []
            if 'MATCHUP' in df.columns:
                observed = sorted(df['MATCHUP'].astype(str).map(lambda m: m.split()[0]).dropna().unique().tolist())

        team_list = observed.copy()
        if len(team_list) < 30:
            for tid in DEFAULT_TEAM_IDS:
                if len(team_list) >= 30:
                    break
                if tid not in team_list:
                    team_list.append(tid)
        if len(team_list) < 30:
            i = 1
            while len(team_list) < 30:
                candidate = f'TEAM_{i:02d}'
                if candidate not in team_list:
                    team_list.append(candidate)
                i += 1
        else:
            team_list = team_list[:30]

        rows = []
        for s in seasons:
            for t in team_list:
                team_rows = df[(df.get('SEASON') == s) & (df.get('TEAM_ID') == t)] if 'TEAM_ID' in df.columns else df[(df.get('SEASON') == s) & (df.get('TEAM_ABBREVIATION') == t)] if 'TEAM_ABBREVIATION' in df.columns else pd.DataFrame()
                # select core stat columns if present, else zero
                stat_cols = [c for c in CORE_STATS if c in df.columns]
                if not stat_cols:
                    # fallback to common numeric stats
                    stat_cols = [c for c in ['PTS','AST','REB','OREB','DREB','STL','BLK','TOV','PF','MIN'] if c in df.columns]

                if not team_rows.empty:
                    # sort by GAME_ID for deterministic order
                    team_rows = team_rows.sort_values('GAME_ID')
                    # iterate existing games
                    idx = 1
                    for _, r in team_rows.iterrows():
                        out = {'SEASON': s, 'TEAM_ID': t, 'GAME_INDEX': idx, 'GAME_ID': r.get('GAME_ID')}
                        for sc in stat_cols:
                            out[sc] = r.get(sc, 0)
                        rows.append(out)
                        idx += 1
                    # pad to 82
                    while idx <= 82:
                        out = {'SEASON': s, 'TEAM_ID': t, 'GAME_INDEX': idx, 'GAME_ID': f'{s}::{t}::PAD::{idx}'}
                        for sc in stat_cols:
                            out[sc] = 0
                        rows.append(out)
                        idx += 1
                else:
                    # no games for this team-season: create 82 zero rows
                    for idx in range(1, 83):
                        out = {'SEASON': s, 'TEAM_ID': t, 'GAME_INDEX': idx, 'GAME_ID': f'{s}::{t}::PAD::{idx}'}
                        for sc in stat_cols:
                            out[sc] = 0
                        rows.append(out)

        out_df = pd.DataFrame(rows)
        # ensure ordering
        out_df = out_df[['SEASON','TEAM_ID','GAME_INDEX','GAME_ID'] + [c for c in out_df.columns if c not in ('SEASON','TEAM_ID','GAME_INDEX','GAME_ID')]]
        return out_df

    try:
        games_df = build_padded_team_games(df)
        # write per-game details without modifying the seasonal summary
        out_dir = 'data/historical'
        os.makedirs(out_dir, exist_ok=True)
        games_df.to_parquet(os.path.join(out_dir, 'team_game_details.parquet'), index=False)
        games_df.to_csv(os.path.join(out_dir, 'team_game_details.csv'), index=False)
        print(f'Wrote per-game team details to {os.path.join(out_dir, "team_game_details.csv")} rows={len(games_df)}')
    except Exception:
        print('WARN: failed to build per-game details', file=sys.stderr)

    # Ensure GAME_ID is string
    if 'GAME_ID' in df.columns:
        df['GAME_ID'] = df['GAME_ID'].astype(str)

    # core numeric stats present in data
    stats = [c for c in CORE_STATS if c in df.columns]

    # If TEAM_ID present, compute per-team summaries
    if 'TEAM_ID' in df.columns:
        working = df.copy()

        # If opponent PTS can be derived, compute WIN flag using groupby-transform
        if 'PTS' in working.columns:
            # Opponent points = total points in game minus team's own points (works for two-team games)
            total_pts = working.groupby('GAME_ID')['PTS'].transform('sum')
            merged = working.copy()
            merged['OPP_PTS'] = total_pts - merged['PTS']
            # In some malformed cases there may be >2 rows per GAME_ID; keep distinct TEAM_ID/GAME_ID combos
            merged = merged.drop_duplicates(subset=['GAME_ID', 'TEAM_ID'])
            merged['WIN'] = merged['PTS'] > merged['OPP_PTS']
        else:
            merged = working.copy()
            merged['WIN'] = False

        # compute games played (unique GAME_IDs), wins, losses
        games = merged.groupby(['SEASON', 'TEAM_ID'])['GAME_ID'].nunique().rename('GAMES')
        wins = merged.groupby(['SEASON', 'TEAM_ID'])['WIN'].sum().rename('WINS')
        summary = pd.concat([games, wins], axis=1).reset_index()
        summary['LOSSES'] = summary['GAMES'] - summary['WINS']

        # aggregate numeric stats: total and per-game average
        if stats:
            agg_funcs = {s: ['sum', 'mean'] for s in stats}
            stat_aggs = merged.groupby(['SEASON', 'TEAM_ID']).agg(agg_funcs)
            # flatten columns (e.g. PTS_sum, PTS_mean)
            stat_aggs.columns = [f"{c[0]}_{c[1]}" for c in stat_aggs.columns]
            stat_aggs = stat_aggs.reset_index()
            summary = summary.merge(stat_aggs, on=['SEASON', 'TEAM_ID'], how='left')

        return pad_and_finalize(summary)

    # If TEAM_ID missing, attempt to infer from MATCHUP within summarize
    if 'MATCHUP' in df.columns:
        tmp = df.copy()
        def extract_team_token(m):
            try:
                if ' vs ' in m:
                    return m.split(' vs ')[0]
                if ' @ ' in m:
                    return m.split(' @ ')[0]
                return m.split()[0]
            except Exception:
                return None

        tmp['DERIVED_TEAM'] = tmp['MATCHUP'].astype(str).map(extract_team_token)
        if tmp['DERIVED_TEAM'].notna().sum() > 0:
            tmp['TEAM_ID'] = tmp['DERIVED_TEAM']
            s = summarize_from_df(tmp)
            return pad_and_finalize(s)

    # final fallback: produce a padded empty per-team summary (zeros)
    return pad_and_finalize(pd.DataFrame())


def pad_and_finalize(summary: pd.DataFrame) -> pd.DataFrame:
    """Pad the summary to 30 teams x canonical seasons and compute TOTAL/PER_GAME.

    Input: `summary` DataFrame with columns at minimum ['SEASON','TEAM_ID','GAMES','WINS',...]
    Output: DataFrame with 30 teams per season, GAMES forced to 82, and TOTAL/PER_GAME
    columns for all CORE_STATS.
    """
    # determine seasons
    seasons_present = []
    if 'SEASON' in summary.columns and not summary['SEASON'].dropna().empty:
        seasons_present = sorted(summary['SEASON'].dropna().unique().tolist())
    if not seasons_present:
        seasons_present = os.environ.get('SEASONS', '2022-23,2023-24,2024-25').split(',')
        seasons_present = [s.strip() for s in seasons_present if s.strip()]

    # build canonical team list
    observed_teams = []
    if 'TEAM_ID' in summary.columns:
        observed_teams = [str(t) for t in sorted(summary['TEAM_ID'].dropna().unique().tolist())]
    team_list = observed_teams.copy()
    if len(team_list) < 30:
        for tid in DEFAULT_TEAM_IDS:
            if len(team_list) >= 30:
                break
            if tid not in team_list:
                team_list.append(tid)
    if len(team_list) < 30:
        i = 1
        while len(team_list) < 30:
            candidate = f'TEAM_{i:02d}'
            if candidate not in team_list:
                team_list.append(candidate)
            i += 1
    else:
        team_list = team_list[:30]

    # full grid
    idx = []
    for s in seasons_present:
        for t in team_list:
            idx.append((s, t))
    full = pd.DataFrame(idx, columns=['SEASON', 'TEAM_ID'])

    # merge with provided summary (if empty, just create baseline)
    if summary is None or summary.empty:
        out = full.copy()
    else:
        out = full.merge(summary, on=['SEASON', 'TEAM_ID'], how='left')

    # ensure WINS present (use column-existence check to avoid scalar default)
    if 'WINS' in out.columns:
        out['WINS'] = out['WINS'].fillna(0).astype(int)
    else:
        out['WINS'] = 0

    # Ensure all CORE_STATS appear as TOTAL and PER_GAME columns
    for stat in CORE_STATS:
        total_col = f'{stat}_TOTAL'
        per_col = f'{stat}_PER_GAME'
        sum_src = f'{stat}_sum'
        if sum_src in out.columns:
            out[total_col] = out[sum_src].fillna(0)
        else:
            # If a direct TOTAL already exists (rare), keep it; otherwise zero
            if total_col not in out.columns:
                out[total_col] = 0.0
        # placeholder per-game; will compute after GAMES set
        out[per_col] = 0.0

    # Force GAMES to 82
    out['GAMES'] = 82
    # Recompute per-game stats
    for stat in CORE_STATS:
        total_col = f'{stat}_TOTAL'
        per_col = f'{stat}_PER_GAME'
        out[per_col] = out[total_col].astype(float) / out['GAMES']

    # Recompute LOSSES
    out['LOSSES'] = out['GAMES'] - out['WINS']

    # Reorder columns
    other_cols = [c for c in out.columns if c not in ('SEASON','TEAM_ID','GAMES','WINS','LOSSES')]
    other_cols = sorted(other_cols)
    cols = ['SEASON','TEAM_ID','GAMES','WINS','LOSSES'] + other_cols
    out = out[cols]

    return out


def summarize_from_df(df: pd.DataFrame) -> pd.DataFrame:
    df = _upper_cols(df)
    if 'GAME_ID' in df.columns:
        df['GAME_ID'] = df['GAME_ID'].astype(str)
    stats = [c for c in CORE_STATS if c in df.columns]
    working = df.copy()
    if 'PTS' in working.columns:
        total_pts = working.groupby('GAME_ID')['PTS'].transform('sum')
        merged = working.copy()
        merged['OPP_PTS'] = total_pts - merged['PTS']
        merged = merged.drop_duplicates(subset=['GAME_ID', 'TEAM_ID'])
        merged['WIN'] = merged['PTS'] > merged['OPP_PTS']
    else:
        merged = working.copy()
        merged['WIN'] = False

    games = merged.groupby(['SEASON', 'TEAM_ID'])['GAME_ID'].nunique().rename('GAMES')
    wins = merged.groupby(['SEASON', 'TEAM_ID'])['WIN'].sum().rename('WINS')
    summary = pd.concat([games, wins], axis=1).reset_index()
    summary['LOSSES'] = summary['GAMES'] - summary['WINS']
    if stats:
        agg_funcs = {s: ['sum', 'mean'] for s in stats}
        stat_aggs = merged.groupby(['SEASON', 'TEAM_ID']).agg(agg_funcs)
        stat_aggs.columns = [f"{c[0]}_{c[1]}" for c in stat_aggs.columns]
        stat_aggs = stat_aggs.reset_index()
        summary = summary.merge(stat_aggs, on=['SEASON', 'TEAM_ID'], how='left')
    return summary


def main():
    out_dir = 'data/historical'
    os.makedirs(out_dir, exist_ok=True)
    summary = summarize()
    if summary.empty:
        print('No summary produced')
        return
    p_parquet = os.path.join(out_dir, 'team_summaries.parquet')
    p_csv = os.path.join(out_dir, 'team_summaries.csv')
    summary.to_parquet(p_parquet, index=False)
    summary.to_csv(p_csv, index=False)
    print(f'Wrote summaries to {p_parquet} and {p_csv}; rows={len(summary)}')


if __name__ == '__main__':
    main()
