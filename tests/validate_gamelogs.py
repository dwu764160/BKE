"""
tests/validate_gamelogs.py
Validates player and team game logs for completeness and basic sanity checks.

Input files:
  - data/historical/final_player_game_logs.parquet
  - data/historical/team_game_logs.parquet
  - data/historical/team_game_details.parquet

Output: printed diagnostics with ✅/⚠️/❌ markers.
"""

import os
import sys
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DATA_DIR = "data/historical"

# Expected seasons in the dataset
EXPECTED_SEASONS = ["2022-23", "2023-24", "2024-25"]



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_team_from_matchup(matchup):
    """Extract home team abbreviation from MATCHUP string ('BOS vs. LAL')."""
    if not isinstance(matchup, str):
        return None
    return matchup.split(" ")[0]


def safe_load(path):
    """Load a parquet file, return None if missing."""
    if not os.path.exists(path):
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def load_team_abbrev_map(teams_path):
    """Load team abbreviation -> TEAM_ID map from teams.parquet."""
    if not os.path.exists(teams_path):
        return {}
    try:
        teams_df = pd.read_parquet(teams_path)
    except Exception:
        return {}
    if teams_df.empty:
        return {}

    cols = {c.lower(): c for c in teams_df.columns}
    abbr_col = cols.get("abbreviation")
    id_col = cols.get("team_id") or cols.get("id")
    if not abbr_col or not id_col:
        return {}

    mapping = {}
    for _, row in teams_df[[abbr_col, id_col]].dropna().iterrows():
        abbr = str(row[abbr_col]).strip().upper()
        try:
            tid = str(int(float(row[id_col])))
        except (ValueError, TypeError):
            tid = str(row[id_col]).strip()
        if abbr and tid:
            mapping[abbr] = tid
    return mapping


def normalize_team_id_series(series):
    """Normalize TEAM_ID values to integer strings."""
    s = pd.to_numeric(series, errors="coerce").astype("Int64")
    s = s.astype(str)
    return s.where(s != "<NA>", other=None)


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------

def validate_file_existence():
    """Check that all expected historical data files exist."""
    print("\n" + "=" * 60)
    print("1. FILE EXISTENCE CHECKS")
    print("=" * 60)

    files = {
        "final_player_game_logs.parquet": os.path.join(DATA_DIR, "final_player_game_logs.parquet"),
        "team_game_logs.parquet":         os.path.join(DATA_DIR, "team_game_logs.parquet"),
        "team_game_details.parquet":      os.path.join(DATA_DIR, "team_game_details.parquet"),
    }

    loaded = {}
    for label, path in files.items():
        if os.path.exists(path):
            try:
                df = pd.read_parquet(path)
                loaded[label] = df
                print(f"   ✅ {label}: {df.shape[0]:,} rows × {df.shape[1]} cols")
            except Exception as e:
                loaded[label] = None
                print(f"   ❌ {label}: exists but unreadable ({e})")
        else:
            loaded[label] = None
            print(f"   ❌ {label}: not found")

    return loaded


def validate_player_logs(pdf):
    """Validate the player game logs DataFrame."""
    print("\n" + "=" * 60)
    print("2. PLAYER GAME LOG VALIDATION")
    print("=" * 60)

    if pdf is None:
        print("   ❌ No player game log data available")
        return

    # --- 2a. Critical columns ---
    # Accept both casing variants
    pid_col = "PLAYER_ID" if "PLAYER_ID" in pdf.columns else ("Player_ID" if "Player_ID" in pdf.columns else None)
    gid_col = "Game_ID" if "Game_ID" in pdf.columns else ("GAME_ID" if "GAME_ID" in pdf.columns else None)

    critical = {
        "player_id_col": pid_col,
        "game_id_col":   gid_col,
        "SEASON":        "SEASON" in pdf.columns,
        "MATCHUP":       "MATCHUP" in pdf.columns,
        "PTS":           "PTS" in pdf.columns,
    }
    missing = [k for k, v in critical.items() if not v]
    if missing:
        print(f"   ❌ Missing critical columns: {missing}")
    else:
        print(f"   ✅ All critical columns present")

    # --- 2b. Row counts per season ---
    if "SEASON" in pdf.columns:
        for season in EXPECTED_SEASONS:
            n = int((pdf["SEASON"] == season).sum())
            # Typical: ~500 players × 40-82 games ≈ 15k-40k rows per season
            if n == 0:
                print(f"   ❌ Season {season}: 0 rows")
            elif n < 10000:
                print(f"   ⚠️ Season {season}: {n:,} rows (low, expected ~20k+)")
            else:
                print(f"   ✅ Season {season}: {n:,} rows")
    else:
        print("   ⚠️ No SEASON column — cannot check per-season counts")

    # --- 2c. Unique teams ---
    if "MATCHUP" in pdf.columns:
        pdf = pdf.copy()
        pdf["_TEAM"] = pdf["MATCHUP"].apply(extract_team_from_matchup)
        n_teams = pdf["_TEAM"].nunique()
        if n_teams == 30:
            print(f"   ✅ Unique teams: {n_teams}")
        elif 28 <= n_teams <= 32:
            print(f"   ⚠️ Unique teams: {n_teams} (expected 30)")
        else:
            print(f"   ❌ Unique teams: {n_teams} (expected 30)")

    # --- 2d. Games per season ---
    if gid_col and "SEASON" in pdf.columns:
        for season in EXPECTED_SEASONS:
            sub = pdf[pdf["SEASON"] == season]
            n_games = sub[gid_col].nunique()
            if 1200 <= n_games <= 1240:
                print(f"   ✅ Games in {season}: {n_games}")
            elif n_games >= 1100:
                print(f"   ⚠️ Games in {season}: {n_games} (expected ~1230)")
            elif n_games > 0:
                print(f"   ⚠️ Games in {season}: {n_games} (partial season?)")
            else:
                print(f"   ❌ Games in {season}: 0")

    # --- 2e. Duplicate rows ---
    dup_count = int(pdf.duplicated().sum())
    if dup_count == 0:
        print(f"   ✅ No duplicate rows")
    else:
        print(f"   ⚠️ {dup_count:,} duplicate rows found")

    # --- 2f. Nulls in critical columns ---
    for col in ["PTS", "MATCHUP"]:
        if col in pdf.columns:
            n_null = int(pdf[col].isnull().sum())
            if n_null == 0:
                print(f"   ✅ No nulls in {col}")
            else:
                print(f"   ⚠️ {n_null} nulls in {col}")

    # --- 2g. Negative stats ---
    stat_cols = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FGM", "FGA"]
    neg_found = []
    for c in stat_cols:
        if c in pdf.columns:
            neg = int((pdf[c] < 0).sum())
            if neg > 0:
                neg_found.append(f"{c}({neg})")
    if neg_found:
        print(f"   ⚠️ Negative values: {', '.join(neg_found)}")
    else:
        print(f"   ✅ No negative stat values")

    # --- 2h. Win/Loss column ---
    if "WL" in pdf.columns:
        wl_vals = set(pdf["WL"].dropna().unique())
        expected_wl = {"W", "L"}
        if wl_vals == expected_wl:
            print(f"   ✅ WL column has expected values (W, L)")
        elif wl_vals.issubset(expected_wl | {None}):
            print(f"   ✅ WL column values OK: {wl_vals}")
        else:
            print(f"   ⚠️ Unexpected WL values: {wl_vals}")

        # Check each game has both W and L among its players
        if gid_col:
            wl_per_game = pdf.groupby(gid_col)["WL"].nunique()
            bad_wl = int((wl_per_game < 2).sum())
            if bad_wl == 0:
                print(f"   ✅ All games have both W and L players")
            else:
                print(f"   ⚠️ {bad_wl} games missing W or L designation")
    else:
        print(f"   ⚠️ No WL column in player logs")


def validate_team_logs(tdf):
    """Validate team_game_logs.parquet."""
    print("\n" + "=" * 60)
    print("3. TEAM GAME LOG VALIDATION")
    print("=" * 60)

    if tdf is None:
        print("   ❌ No team game log data available")
        return

    # --- 3a. Rows per season ---
    if "SEASON" in tdf.columns:
        for season in EXPECTED_SEASONS:
            n = int((tdf["SEASON"] == season).sum())
            # 30 teams × 82 games = 2460 rows per season
            if 2400 <= n <= 2520:
                print(f"   ✅ Season {season}: {n:,} team-game rows (expected ~2460)")
            elif n >= 2000:
                print(f"   ⚠️ Season {season}: {n:,} rows (expected ~2460)")
            elif n > 0:
                print(f"   ⚠️ Season {season}: {n:,} rows (partial)")
            else:
                print(f"   ❌ Season {season}: 0 rows")

    # --- 3b. Two rows per game (home + away) ---
    if "GAME_ID" in tdf.columns:
        rows_per_game = tdf.groupby("GAME_ID").size()
        bad = int((rows_per_game != 2).sum())
        if bad == 0:
            print(f"   ✅ All {rows_per_game.size:,} games have exactly 2 team rows")
        else:
            print(f"   ⚠️ {bad} games do not have exactly 2 rows")

    # --- 3c. PTS vs OPP_PTS consistency ---
    if "PTS" in tdf.columns and "OPP_PTS" in tdf.columns and "GAME_ID" in tdf.columns:
        # For each game, team A's PTS should equal team B's OPP_PTS
        games = tdf.groupby("GAME_ID")
        mismatches = 0
        sample_games = 0
        for gid, grp in games:
            if len(grp) != 2:
                continue
            sample_games += 1
            r1, r2 = grp.iloc[0], grp.iloc[1]
            if r1["PTS"] != r2["OPP_PTS"] or r2["PTS"] != r1["OPP_PTS"]:
                mismatches += 1

        if mismatches == 0:
            print(f"   ✅ PTS/OPP_PTS cross-check passed ({sample_games:,} games)")
        else:
            pct = mismatches / max(sample_games, 1) * 100
            print(f"   ⚠️ {mismatches}/{sample_games} games have PTS↔OPP_PTS mismatch ({pct:.1f}%)")

    # --- 3d. No ties (PTS != OPP_PTS) ---
    if "PTS" in tdf.columns and "OPP_PTS" in tdf.columns:
        ties = int((tdf["PTS"] == tdf["OPP_PTS"]).sum())
        if ties == 0:
            print(f"   ✅ No tied games (PTS == OPP_PTS)")
        else:
            print(f"   ⚠️ {ties} rows have PTS == OPP_PTS (ties not expected in NBA)")


def validate_team_details(tdd):
    """Validate team_game_details.parquet."""
    print("\n" + "=" * 60)
    print("4. TEAM GAME DETAIL VALIDATION")
    print("=" * 60)

    if tdd is None:
        print("   ❌ No team game detail data available")
        return

    # --- 4a. Critical stat columns ---
    expected_stats = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FGM", "FGA", "FTM", "FTA"]
    present = [c for c in expected_stats if c in tdd.columns]
    missing = [c for c in expected_stats if c not in tdd.columns]
    if not missing:
        print(f"   ✅ All {len(expected_stats)} stat columns present")
    else:
        print(f"   ⚠️ Missing stat columns: {missing}")

    # --- 4b. Rows per season ---
    if "SEASON" in tdd.columns:
        for season in EXPECTED_SEASONS:
            n = int((tdd["SEASON"] == season).sum())
            if 2400 <= n <= 2520:
                print(f"   ✅ Season {season}: {n:,} detail rows")
            elif n >= 2000:
                print(f"   ⚠️ Season {season}: {n:,} detail rows (expected ~2460)")
            elif n > 0:
                print(f"   ⚠️ Season {season}: {n:,} detail rows (partial)")
            else:
                print(f"   ❌ Season {season}: 0 detail rows")

    # --- 4c. Negative stats ---
    neg_found = []
    for c in expected_stats:
        if c in tdd.columns:
            neg = int((tdd[c] < 0).sum())
            if neg > 0:
                neg_found.append(f"{c}({neg})")
    if neg_found:
        print(f"   ⚠️ Negative values in team details: {', '.join(neg_found)}")
    else:
        print(f"   ✅ No negative stat values in team details")

    # --- 4d. FGM <= FGA, FTM <= FTA ---
    for made, att in [("FGM", "FGA"), ("FTM", "FTA"), ("FG3M", "FG3A")]:
        if made in tdd.columns and att in tdd.columns:
            bad = int((tdd[made] > tdd[att]).sum())
            if bad == 0:
                print(f"   ✅ {made} <= {att} for all rows")
            else:
                print(f"   ⚠️ {bad} rows have {made} > {att}")


def validate_cross_consistency(player_df, team_df, team_details_df):
    """Cross-validate between player and team data."""
    print("\n" + "=" * 60)
    print("5. CROSS-TABLE CONSISTENCY")
    print("=" * 60)

    if player_df is None:
        print("   ❌ Cannot cross-validate — missing player data")
        return

    # Use the correct game_id column from player logs
    gid_col = "Game_ID" if "Game_ID" in player_df.columns else "GAME_ID"
    if gid_col not in player_df.columns:
        print("   ⚠️ No game ID column in player logs")
        return

    # --- 5a. Player PTS vs team_game_logs (official final scores) ---
    if team_df is not None and "PTS" in player_df.columns and "PTS" in team_df.columns and "GAME_ID" in team_df.columns:
        player_pts_game = player_df.groupby(gid_col)["PTS"].sum().rename("player_pts")
        # team_game_logs: each row is one team, sum both teams = game total
        team_pts_game = team_df.groupby("GAME_ID")["PTS"].sum().rename("team_pts")

        merged_tl = pd.concat([player_pts_game, team_pts_game], axis=1).dropna()
        if len(merged_tl) > 0:
            merged_tl["diff"] = (merged_tl["team_pts"] - merged_tl["player_pts"]).abs()
            exact = int((merged_tl["diff"] == 0).sum())
            close = int((merged_tl["diff"] <= 2).sum())
            total = len(merged_tl)

            if exact == total:
                print(f"   ✅ Player PTS vs Team Logs: all {total:,} games match exactly")
            elif exact / total >= 0.90:
                print(f"   ⚠️ Player PTS vs Team Logs: {exact}/{total} exact ({exact/total*100:.0f}%), {close} within ±2")
            else:
                pct = exact / total * 100
                # Check if the non-matching games are at least close
                big_diff = int((merged_tl["diff"] > 10).sum())
                print(f"   ⚠️ Player PTS vs Team Logs: {exact}/{total} exact ({pct:.0f}%), {big_diff} with >10pt gap")
                # This is expected: some players missing from game logs → lower player totals
                avg_diff = merged_tl[merged_tl["diff"] > 0]["diff"].mean()
                print(f"      Mean gap when mismatched: {avg_diff:.1f} pts (missing player entries expected)")

    # --- 5b. Team details PTS vs team logs PTS ---
    if team_details_df is not None and team_df is not None:
        if "PTS" in team_details_df.columns and "PTS" in team_df.columns:
            td_pts = team_details_df.groupby("GAME_ID")["PTS"].sum().rename("detail_pts")
            tl_pts = team_df.groupby("GAME_ID")["PTS"].sum().rename("log_pts")
            merged_tt = pd.concat([td_pts, tl_pts], axis=1).dropna()
            if len(merged_tt) > 0:
                exact_tt = int((merged_tt["detail_pts"] == merged_tt["log_pts"]).sum())
                total_tt = len(merged_tt)
                if exact_tt == total_tt:
                    print(f"   ✅ Team Details vs Team Logs PTS: all {total_tt:,} games match")
                elif exact_tt / total_tt >= 0.90:
                    print(f"   ⚠️ Team Details vs Logs PTS: {exact_tt}/{total_tt} match ({exact_tt/total_tt*100:.0f}%)")
                else:
                    print(f"   ⚠️ Team Details vs Logs PTS: {exact_tt}/{total_tt} match ({exact_tt/total_tt*100:.0f}%)")
                    avg_gap = (merged_tt["log_pts"] - merged_tt["detail_pts"]).mean()
                    print(f"      Avg gap: {avg_gap:.1f} pts (detail data may be truncated for later games)")

    # --- 5c. Unique game overlap ---
    player_games = set(player_df[gid_col].unique())
    if team_details_df is not None and "GAME_ID" in team_details_df.columns:
        team_games = set(team_details_df["GAME_ID"].unique())
        overlap = len(player_games & team_games)
        only_player = len(player_games - team_games)
        only_team = len(team_games - player_games)

        if only_player == 0 and only_team == 0:
            print(f"   ✅ Perfect game overlap: {overlap:,} games in both tables")
        else:
            print(f"   ⚠️ Game overlap: {overlap:,} shared, {only_player} player-only, {only_team} team-only")

    # --- 5d. Players per team per game ---
    if "MATCHUP" in player_df.columns:
        pdf2 = player_df.copy()
        pdf2["_TEAM"] = pdf2["MATCHUP"].apply(extract_team_from_matchup)
        pid_col = "PLAYER_ID" if "PLAYER_ID" in pdf2.columns else "Player_ID"
        if pid_col in pdf2.columns:
            pcounts = pdf2.groupby([gid_col, "_TEAM"])[pid_col].nunique()
            too_few = int((pcounts < 5).sum())
            too_many = int((pcounts > 15).sum())
            avg_ppg = pcounts.mean()

            if too_few == 0 and too_many == 0:
                print(f"   ✅ Players/team/game: avg {avg_ppg:.1f}, all in [5,15]")
            elif too_few <= 20:
                print(f"   ⚠️ Players/team/game: avg {avg_ppg:.1f}, {too_few} entries with <5 (minor gaps)")
            else:
                print(f"   ❌ Players/team/game: avg {avg_ppg:.1f}, {too_few} entries with <5 players")

    # --- 5e. Player-derived team stats vs team details ---
    if team_details_df is not None and "MATCHUP" in player_df.columns:
        teams_path = os.path.join(DATA_DIR, "teams.parquet")
        team_map = load_team_abbrev_map(teams_path)

        pdf = player_df.copy()
        pdf["_TEAM_ABBR"] = pdf["MATCHUP"].apply(extract_team_from_matchup)

        stat_cols = [
            c for c in [
                "PTS", "REB", "AST", "STL", "BLK", "TOV", "FGM", "FGA",
                "FTM", "FTA", "OREB", "DREB", "FG3M", "FG3A", "PF", "MIN", "PLUS_MINUS"
            ] if c in pdf.columns and c in team_details_df.columns
        ]

        if not stat_cols:
            print("   INFO No shared stat columns for player-vs-team comparison")
            return

        tdd = team_details_df.copy()
        if "TEAM_ID" not in tdd.columns:
            print("   INFO Team Details missing TEAM_ID; skipping player-vs-team stat comparison")
            return

        tdd_team_ids = tdd["TEAM_ID"].dropna().astype(str)
        looks_like_abbr = (
            tdd_team_ids.str.fullmatch(r"[A-Za-z]{2,4}").mean() if len(tdd_team_ids) else 0.0
        ) >= 0.8

        if looks_like_abbr:
            pdf["TEAM_KEY"] = pdf["_TEAM_ABBR"].astype(str).str.upper()
            tdd["TEAM_KEY"] = tdd["TEAM_ID"].astype(str).str.upper()
        else:
            if not team_map:
                print("   INFO Team mapping unavailable; skipping player-vs-team stat comparison")
                return
            pdf["TEAM_ID"] = pdf["_TEAM_ABBR"].map(team_map)
            pdf["TEAM_KEY"] = normalize_team_id_series(pdf["TEAM_ID"])
            tdd["TEAM_KEY"] = normalize_team_id_series(tdd["TEAM_ID"])

        pdf = pdf.dropna(subset=["TEAM_KEY", gid_col, "SEASON"])
        pdf[gid_col] = pdf[gid_col].astype(str)
        pdf["TEAM_KEY"] = pdf["TEAM_KEY"].astype(str)

        player_team = (
            pdf.groupby(["SEASON", "TEAM_KEY", gid_col], as_index=False)[stat_cols]
            .sum()
        )

        if "GAME_ID" in tdd.columns:
            tdd["GAME_ID"] = tdd["GAME_ID"].astype(str)

        merged = player_team.merge(
            tdd[["SEASON", "TEAM_KEY", "GAME_ID"] + stat_cols],
            left_on=["SEASON", "TEAM_KEY", gid_col],
            right_on=["SEASON", "TEAM_KEY", "GAME_ID"],
            how="inner",
            suffixes=("_player", "_team"),
        )

        if merged.empty:
            print("   INFO Player-vs-team stat comparison skipped (no overlap)")
            return

        print(f"   INFO Player-derived vs Team Details: {len(merged):,} team-games compared")
        for stat in stat_cols:
            diff = (merged[f"{stat}_player"] - merged[f"{stat}_team"]).abs()
            mae = diff.mean() if len(diff) else 0.0
            exact = (diff == 0).mean() * 100 if len(diff) else 0.0
            print(f"      {stat}: MAE {mae:.2f}, exact {exact:.1f}%")


def generate_summary(loaded):
    """Print final summary."""
    print("\n" + "=" * 60)
    print("GAME LOG VALIDATION SUMMARY")
    print("=" * 60)

    files_present = sum(1 for v in loaded.values() if v is not None)
    files_total = len(loaded)

    if files_present == files_total:
        print(f"   ✅ All {files_total} data files present and readable")
    elif files_present > 0:
        print(f"   ⚠️ {files_present}/{files_total} data files available")
    else:
        print(f"   ❌ No data files found")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("GAME LOG VALIDATION SUITE")
    print("=" * 60)

    loaded = validate_file_existence()

    player_df     = loaded.get("final_player_game_logs.parquet")
    team_logs_df  = loaded.get("team_game_logs.parquet")
    team_det_df   = loaded.get("team_game_details.parquet")

    validate_player_logs(player_df)
    validate_team_logs(team_logs_df)
    validate_team_details(team_det_df)
    validate_cross_consistency(player_df, team_logs_df, team_det_df)
    generate_summary(loaded)


if __name__ == "__main__":
    main()

