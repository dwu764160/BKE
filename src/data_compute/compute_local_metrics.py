"""
compute_local_metrics.py

Reads:
 - data/player_game_logs.parquet  (required)  -- player-level box scores, one row per player-game
 - data/team_game_logs.parquet    (optional)  -- team-level game stats (one row per team-game). If not present,
                                                team aggregates will be computed from player_game_logs.

Outputs:
 - data/advanced_local_metrics.parquet  (player-season aggregated metrics)
 - (optional) prints summary stats for sanity checks

Notes:
 - The script computes only metrics derivable from box scores and team aggregates.
 - Some formulas use approximations (documented inline).
"""

import os
import math
import pandas as pd
import numpy as np

# Try common locations for player/team game logs (prefer project-level, then historical/)
PLAYER_LOGS_CANDIDATES = [
    "data/player_game_logs.parquet",
    "data/historical/player_game_logs.parquet",
    "data/historical/final_player_game_logs.parquet",
    "data/historical/final_player_game_logs.parquet",
]
TEAM_LOGS_CANDIDATES = [
    "data/team_game_logs.parquet",
    "data/historical/team_game_logs.parquet",
    "data/historical/final_team_game_logs.parquet",
]

OUTPUT_PATH = "data/advanced_local_metrics.parquet"

def pick_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

PLAYER_LOGS_PATH = pick_existing(PLAYER_LOGS_CANDIDATES)
TEAM_LOGS_PATH = pick_existing(TEAM_LOGS_CANDIDATES)

# ---------------------------
# Helper functions / formulas
# ---------------------------

def safe_div(a, b, fill=np.nan):
    """Safe division: returns fill if denominator is zero or NaN."""
    try:
        a = np.array(a, dtype=float)
        b = np.array(b, dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            res = np.where((b == 0) | np.isnan(b), fill, a / b)
        return res
    except Exception:
        return fill

def compute_team_aggregates_from_player_logs(player_df):
    """
    If team-level game logs are not available, compute team aggregates per game by summing
    player-level stats grouped by GAME_ID and TEAM_ID (or TEAM_ABBREVIATION).
    Returns a DataFrame with columns: GAME_ID, TEAM_ID, TEAM_FGA, TEAM_FGM, TEAM_FG3A, TEAM_FG3M,
    TEAM_FTA, TEAM_FTM, TEAM_TOV, TEAM_OREB, TEAM_DREB, TEAM_MIN (expected 240 per full game).
    """
    # determine team id column if present; if missing try to derive from MATCHUP
    team_col = None
    for candidate in ["TEAM_ID", "TEAM_ABBREVIATION", "TEAM"]:
        if candidate in player_df.columns:
            team_col = candidate
            break
    if team_col is None:
        # attempt to extract team abbreviation from MATCHUP (format 'TOR vs. MIL' or 'TOR @ MIL')
        if "MATCHUP" in player_df.columns:
            player_df = player_df.copy()
            def _extract_team(matchup):
                try:
                    return str(matchup).split()[0]
                except Exception:
                    return None
            player_df["TEAM_ABBREVIATION"] = player_df["MATCHUP"].apply(_extract_team)
            team_col = "TEAM_ABBREVIATION"
        else:
            raise ValueError("No TEAM column found in player logs and MATCHUP not available to derive team. Provide team_game_logs.parquet or include TEAM_ID/TEAM_ABBREVIATION in player logs.")

    agg = player_df.groupby(["GAME_ID", team_col]).agg(
        TEAM_FGA=("FGA", "sum"),
        TEAM_FGM=("FGM", "sum"),
        TEAM_FG3A=("FG3A", "sum"),
        TEAM_FG3M=("FG3M", "sum"),
        TEAM_FTA=("FTA", "sum"),
        TEAM_FTM=("FTM", "sum"),
        TEAM_TOV=("TOV", "sum"),
        TEAM_OREB=("OREB", "sum"),
        TEAM_DREB=("DREB", "sum"),
        TEAM_MIN=("MIN", "sum")
    ).reset_index().rename(columns={team_col: "TEAM_ID"})
    # Note: TEAM_MIN should be about 240 minutes for full team (5 players * 48), but substitution can change sums.
    return agg

def estimate_player_possessions(row):
    """
    Basic approximation of a player's possessions used in a game:
    player_poss = FGA + 0.44 * FTA + TOV
    (This ignores offensive rebounds as possessions preserved; it's a common approximation.)
    """
    return row.get("FGA", 0) + 0.44 * row.get("FTA", 0) + row.get("TOV", 0)

def team_possessions_formula(team_row):
    """
    Estimate team possessions for a game. One commonly used formula:
      Poss = FGA + 0.44*FTA + TOV - OREB
    This is an approximation of possessions.
    """
    return team_row.get("TEAM_FGA", 0) + 0.44 * team_row.get("TEAM_FTA", 0) + team_row.get("TEAM_TOV", 0) - team_row.get("TEAM_OREB", 0)

# ---------------------------
# Core computation functions
# ---------------------------

def aggregate_player_season(player_df, team_game_df=None, min_games_threshold=1):
    """
    Given a player-level game log DataFrame, compute season-level aggregated metrics.
    Returns a DataFrame indexed by PLAYER_ID and SEASON with computed metrics.
    """

    df = player_df.copy()

    # Normalize / ensure we have a GAME_ID column on player logs.
    # Many sources use different names (game_id, Game_ID, GAMEID, etc.). Try to detect and rename.
    if "GAME_ID" not in df.columns:
        found_game_col = None
        for c in df.columns:
            cl = c.lower().replace(' ', '').replace('-', '')
            if 'game' in cl and 'id' in cl:
                found_game_col = c
                break
        if found_game_col:
            df = df.rename(columns={found_game_col: 'GAME_ID'})
        else:
            # If no explicit game id column, try to synthesize one from GAME_DATE + TEAM/MATCHUP + SEASON
            if 'GAME_DATE' in df.columns and any(k in df.columns for k in ['TEAM_ABBREVIATION', 'TEAM', 'TEAM_ID', 'MATCHUP']):
                print("Info: no GAME_ID column found — synthesizing GAME_ID from GAME_DATE + TEAM/MATCHUP + SEASON")
                def _mk_game_id(row):
                    parts = []
                    if 'SEASON' in df.columns:
                        parts.append(str(row.get('SEASON')))
                    parts.append(str(row.get('GAME_DATE')))
                    # prefer TEAM_ABBREVIATION, fallback to TEAM_ID or MATCHUP
                    if 'TEAM_ABBREVIATION' in df.columns:
                        parts.append(str(row.get('TEAM_ABBREVIATION')))
                    elif 'TEAM' in df.columns:
                        parts.append(str(row.get('TEAM')))
                    elif 'TEAM_ID' in df.columns:
                        parts.append(str(row.get('TEAM_ID')))
                    elif 'MATCHUP' in df.columns:
                        parts.append(str(row.get('MATCHUP')))
                    return "::".join(parts)
                df['GAME_ID'] = df.apply(_mk_game_id, axis=1)
            else:
                # helpful error listing available columns
                raise KeyError(f"No GAME_ID-like column found in player logs and cannot synthesize one. Available columns: {list(df.columns)}")

    # Ensure numeric columns exist and fillna with 0 where appropriate
    numeric_cols = ["PTS","AST","REB","OREB","DREB","MIN","FGM","FGA","FG3M","FG3A","FTA","FTM","TOV"]
    for c in numeric_cols:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # If team_game_df isn't provided, compute team aggregates per game from player logs
    # Ensure we have a team identifier on player logs. If TEAM_ID/TEAM_ABBREVIATION missing, try to derive from MATCHUP
    team_col_present = any(c in df.columns for c in ["TEAM_ID", "TEAM_ABBREVIATION", "TEAM"])
    if not team_col_present and "MATCHUP" in df.columns:
        def _extract_team(matchup):
            try:
                return str(matchup).split()[0]
            except Exception:
                return None
        df["TEAM_ABBREVIATION"] = df["MATCHUP"].apply(_extract_team)

    if team_game_df is None:
        team_agg = compute_team_aggregates_from_player_logs(df)
    else:
        team_agg = team_game_df.copy()
        # If the provided team_game_df is empty, fall back to computing team aggregates from player logs
        if getattr(team_agg, 'empty', False):
            print("Warning: provided team_game_df is empty — computing team aggregates from player logs instead.")
            team_agg = None
        
    # If after copying/inspection we don't have a valid team_agg, compute it from player logs
    if team_agg is None:
        team_agg = compute_team_aggregates_from_player_logs(df)
    else:
        # Normalize common GAME_ID / TEAM_ID variations (case-insensitive) so downstream merges work
        # Detect a GAME_ID-like column (e.g., 'Game_ID', 'game_id', 'GAMEID') and rename to 'GAME_ID'
        game_col = None
        for c in team_agg.columns:
            cl = c.lower().replace(' ', '').replace('-', '')
            if 'game' in cl and 'id' in cl:
                game_col = c
                break
        if game_col and game_col != 'GAME_ID':
            team_agg = team_agg.rename(columns={game_col: 'GAME_ID'})

        # Detect a TEAM identifier column and normalize to TEAM_ID or TEAM_ABBREVIATION
        teamid_col = None
        for c in team_agg.columns:
            cl = c.lower()
            if 'team' in cl and ('id' in cl or 'teamid' in cl):
                teamid_col = c
                team_agg = team_agg.rename(columns={c: 'TEAM_ID'})
                break
        if teamid_col is None:
            for c in team_agg.columns:
                cl = c.lower()
                if 'team' in cl and ('abbrev' in cl or 'abbreviation' in cl):
                    teamid_col = c
                    team_agg = team_agg.rename(columns={c: 'TEAM_ABBREVIATION'})
                    break
        # normalize column names expected by later merges
        # ensure columns TEAM_FGA, TEAM_FGM, TEAM_FTA, TEAM_TOV, TEAM_OREB, TEAM_DREB, TEAM_MIN exist
        mapping = {}
        # handle likely column names
        if "FGA" in team_agg.columns and "TEAM_FGA" not in team_agg.columns:
            mapping["FGA"] = "TEAM_FGA"
        if "FGM" in team_agg.columns and "TEAM_FGM" not in team_agg.columns:
            mapping["FGM"] = "TEAM_FGM"
        if "FTA" in team_agg.columns and "TEAM_FTA" not in team_agg.columns:
            mapping["FTA"] = "TEAM_FTA"
        if "TOV" in team_agg.columns and "TEAM_TOV" not in team_agg.columns:
            mapping["TOV"] = "TEAM_TOV"
        if "OREB" in team_agg.columns and "TEAM_OREB" not in team_agg.columns:
            mapping["OREB"] = "TEAM_OREB"
        if "DREB" in team_agg.columns and "TEAM_DREB" not in team_agg.columns:
            mapping["DREB"] = "TEAM_DREB"
        if "MIN" in team_agg.columns and "TEAM_MIN" not in team_agg.columns:
            mapping["MIN"] = "TEAM_MIN"
        if mapping:
            team_agg = team_agg.rename(columns=mapping)

    # Merge team aggregate info into player logs by GAME_ID and TEAM_ID (or TEAM_ABBREVIATION)
    # Determine team key column on player logs
    team_col = None
    for candidate in ["TEAM_ID", "TEAM_ABBREVIATION", "TEAM"]:
        if candidate in df.columns:
            team_col = candidate
            break
    if team_col is None:
        raise ValueError("No team identifier found in player logs (TEAM_ID or TEAM_ABBREVIATION).")

    # Determine team key in team_agg
    if "TEAM_ID" in team_agg.columns:
        merge_team_col = "TEAM_ID"
    elif "TEAM_ABBREVIATION" in team_agg.columns:
        merge_team_col = "TEAM_ABBREVIATION"
    else:
        other_cols = [c for c in team_agg.columns if c != "GAME_ID"]
        if len(other_cols) == 0:
            # No explicit team column available in team_agg: fall back to merging only on GAME_ID
            merge_team_col = None
        else:
            # fall back to the first non-GAME_ID column
            merge_team_col = other_cols[0]

    # Perform merge: if we don't have a team column in team_agg, merge on GAME_ID only
    if merge_team_col is None:
        df = df.merge(team_agg, on=["GAME_ID"], how="left")
    else:
        df = df.merge(team_agg, left_on=["GAME_ID", team_col], right_on=["GAME_ID", merge_team_col], how="left")

    # Compute per-game derived columns (player-level)
    df["PLAYER_POSSESSIONS_EST"] = df.apply(estimate_player_possessions, axis=1)

    # Group by player & season
    group_cols = ["PLAYER_ID", "SEASON"]
    agg_funcs = {
        "PTS": ["sum", "mean"],
        "AST": ["sum", "mean"],
        "REB": ["sum", "mean"],
        "OREB": ["sum", "mean"],
        "DREB": ["sum", "mean"],
        "MIN": ["sum", "mean"],
        "FGA": ["sum", "mean"],
        "FGM": ["sum", "mean"],
        "FG3A": ["sum", "mean"],
        "FG3M": ["sum", "mean"],
        "FTA": ["sum", "mean"],
        "FTM": ["sum", "mean"],
        "TOV": ["sum", "mean"],
        "PLAYER_POSSESSIONS_EST": ["sum", "mean"]
    }

    grouped = df.groupby(group_cols).agg(agg_funcs)
    # flatten multiindex columns
    grouped.columns = ["_".join(col).strip() for col in grouped.columns.values]
    grouped = grouped.reset_index()

    # Add derived metrics
    def compute_row_metrics(row):
        out = {}
        # counts
        games_played = row.get("MIN_mean", 0)
        # For games played we can approximate using sum MIN / mean MIN? Better to compute directly
        # We'll compute games_played by counting rows in df
        return out

    # compute games played explicitly:
    gp = df.groupby(["PLAYER_ID","SEASON"]).size().reset_index(name="GAMES")
    grouped = grouped.merge(gp, on=["PLAYER_ID","SEASON"], how="left")

    # compute MPG
    grouped["MPG"] = grouped["MIN_sum"] / grouped["GAMES"].replace(0, np.nan)

    # True Shooting % (TS%) = PTS / (2*(FGA + 0.44*FTA))
    grouped["TS_pct"] = safe_div(grouped["PTS_sum"], 2*(grouped["FGA_sum"] + 0.44*grouped["FTA_sum"]))

    # Effective FG% = (FGM + 0.5*FG3M) / FGA
    grouped["eFG_pct"] = safe_div((grouped["FGM_sum"] + 0.5*grouped["FG3M_sum"]), grouped["FGA_sum"])

    # Turnover % (TOV%) approximation per game-level possessions
    grouped["TOV_pct"] = 100 * safe_div(grouped["TOV_sum"], (grouped["FGA_sum"] + 0.44*grouped["FTA_sum"] + grouped["TOV_sum"]))

    # Assist % (AST%) approximation:
    # AST% = 100 * AST * (TEAM_MIN/5) / (MIN * (TEAM_FGM - FGM))
    # We need average TEAM_FGM & TEAM_MIN for games where player played. We'll compute team averages per player's games.
    # Start by computing per-game team averages for each (PLAYER_ID, SEASON)
    team_stats_for_player = df.groupby(["PLAYER_ID","SEASON"]).agg({
        "TEAM_FGM": "mean",
        "TEAM_MIN": "mean",
        "TEAM_FGA": "mean",
        "TEAM_FTA": "mean",
        "TEAM_TOV": "mean",
        "TEAM_OREB": "mean",
        "TEAM_DREB": "mean"
    }).reset_index()
    # Prefix aggregated team columns with TEAM_ to match downstream names (e.g., TEAM_TEAM_MIN)
    agg_cols = ["TEAM_FGM","TEAM_MIN","TEAM_FGA","TEAM_FTA","TEAM_TOV","TEAM_OREB","TEAM_DREB"]
    rename_map = {c: f"TEAM_{c}" for c in agg_cols}
    team_stats_for_player = team_stats_for_player.rename(columns=rename_map)

    # merge team averages into grouped
    grouped = grouped.merge(team_stats_for_player, on=["PLAYER_ID","SEASON"], how="left")

    # AST% compute (using season sums and team means)
    # AST% = 100 * AST * (TEAM_MIN/5) / (MIN * (TEAM_FGM - FGM))
    grouped["AST_pct"] = 100 * safe_div(
        grouped["AST_sum"] * (grouped["TEAM_TEAM_MIN"] / 5),
        grouped["MIN_sum"] * (grouped["TEAM_TEAM_FGM"] - grouped["FGM_sum"])
    )

    # OREB% and DREB% (using season sums & team means)
    grouped["OREB_pct"] = 100 * safe_div(grouped["OREB_sum"] * (grouped["TEAM_TEAM_MIN"] / 5), grouped["MIN_sum"] * grouped["TEAM_TEAM_OREB"])
    grouped["DREB_pct"] = 100 * safe_div(grouped["DREB_sum"] * (grouped["TEAM_TEAM_MIN"] / 5), grouped["MIN_sum"] * grouped["TEAM_TEAM_DREB"])

    # Usage Rate (USG%): approximate seasonal usage
    # USG% = 100 * ((FGA + 0.44*FTA + TOV) * (Team Minutes / 5)) / (Minutes * (Team FGA + 0.44*Team FTA + Team TOV))
    grouped["USG_pct"] = 100 * safe_div(
        (grouped["FGA_sum"] + 0.44*grouped["FTA_sum"]) + grouped["TOV_sum"],
        grouped["MIN_sum"]
    ) * safe_div(grouped["TEAM_TEAM_MIN"]/5, (grouped["TEAM_TEAM_FGA"] + 0.44*grouped["TEAM_TEAM_FTA"] + grouped["TEAM_TEAM_TOV"]))

    # Points responsible (simple): PTS + 0.7 * AST
    grouped["PTS_from_assists_est"] = 0.7 * grouped["AST_sum"]
    grouped["points_responsible"] = grouped["PTS_sum"] + grouped["PTS_from_assists_est"]

    # AST/TOV ratio
    grouped["AST_TOV_ratio"] = safe_div(grouped["AST_sum"], grouped["TOV_sum"].replace(0, np.nan))

    # Per36 metrics
    for stat in ["PTS_sum","AST_sum","REB_sum","OREB_sum","DREB_sum","TOV_sum","FGM_sum","FGA_sum"]:
        short = stat.replace("_sum","")
        grouped[f"{short}_per36"] = safe_div(grouped[stat], grouped["MIN_sum"]) * 36

    # Player possessions estimate (season)
    grouped["PLAYER_POSS"] = grouped["PLAYER_POSSESSIONS_EST_sum"]
    # Points per 75 possessions
    grouped["PTS_per75poss"] = safe_div(grouped["PTS_sum"], grouped["PLAYER_POSS"]) * 75

    # Three-point frequency and accuracy
    grouped["three_point_freq"] = safe_div(grouped["FG3A_sum"], grouped["FGA_sum"])
    grouped["three_point_pct"] = safe_div(grouped["FG3M_sum"], grouped["FG3A_sum"])

    # Points per shot (approx): PTS / (FGA + 0.44*FTA)
    grouped["points_per_shot"] = safe_div(grouped["PTS_sum"], (grouped["FGA_sum"] + 0.44*grouped["FTA_sum"]))

    # Free throw rate = FTA / (FGA + 0.44*FTA)
    grouped["FT_rate"] = safe_div(grouped["FTA_sum"], (grouped["FGA_sum"] + 0.44*grouped["FTA_sum"]))

    # PROD composite example (PTS + 0.7*AST + 0.3*REB)
    grouped["PROD"] = grouped["PTS_sum"] + 0.7*grouped["AST_sum"] + 0.3*grouped["REB_sum"]

    # Clean up columns to return useful fields
    out_cols = [
        # identity
        "PLAYER_ID","SEASON","GAMES","MPG",
        # raw season sums
        "PTS_sum","AST_sum","REB_sum","MIN_sum","FGM_sum","FGA_sum","FG3M_sum","FG3A_sum","FTA_sum","FTM_sum","TOV_sum",
        # rate metrics
        "TS_pct","eFG_pct","TOV_pct","AST_pct","AST_TOV_ratio",
        "OREB_pct","DREB_pct","USG_pct",
        # intermediate / diagnostics
        "PTS_from_assists_est","PLAYER_POSS",
        # per-usage metrics
        "points_responsible","PTS_per75poss","points_per_shot","FT_rate",
        "three_point_freq","three_point_pct",
        # per-36
        "PTS_per36","AST_per36","REB_per36","OREB_per36","DREB_per36","TOV_per36","FGM_per36","FGA_per36",
        # team averaged columns used in calculations (prefixed TEAM_TEAM_...)
        "TEAM_TEAM_MIN","TEAM_TEAM_FGM","TEAM_TEAM_FGA","TEAM_TEAM_FTA","TEAM_TEAM_TOV","TEAM_TEAM_OREB","TEAM_TEAM_DREB",
        # composite
        "PROD"
    ]
    # Ensure all out_cols exist
    for c in out_cols:
        if c not in grouped.columns:
            grouped[c] = np.nan

    result = grouped[out_cols].copy()
    # Rename some columns to clearer names
    result = result.rename(columns={
        "PTS_sum": "PTS_total",
        "AST_sum": "AST_total",
        "REB_sum": "REB_total",
        "MIN_sum": "MIN_total"
    })
    return result

# ---------------------------
# Main entry
# ---------------------------

def main():
    if not PLAYER_LOGS_PATH or not os.path.exists(PLAYER_LOGS_PATH):
        raise FileNotFoundError(
            f"Player game logs not found. Looked at: {PLAYER_LOGS_CANDIDATES} - please place your parquet at one of these paths or set PLAYER_LOGS_PATH."
        )

    print("Loading player game logs...")
    player_df = pd.read_parquet(PLAYER_LOGS_PATH)

    team_df = None
    if TEAM_LOGS_PATH and os.path.exists(TEAM_LOGS_PATH):
        print(f"Loading team game logs from {TEAM_LOGS_PATH}...")
        team_df = pd.read_parquet(TEAM_LOGS_PATH)

    print("Aggregating and computing local advanced metrics (this may take a bit)...")
    adv = aggregate_player_season(player_df, team_game_df=team_df)

    print(f"Saving {len(adv)} player-season rows to {OUTPUT_PATH}")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    adv.to_parquet(OUTPUT_PATH, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
