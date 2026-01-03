import os

import pandas as pd

# Load the combined player game logs parquet file
# Update the path if your file is named differently
parquet_path = "data/historical/player_game_logs.parquet"
df = pd.read_parquet(parquet_path)

# 1. Check the DataFrame shape
# 1. Check the DataFrame shape
print("DataFrame shape (rows, columns):", df.shape)

# --- TEAM GAME LOGS CHECK ---
team_parquet_path = "data/historical/team_game_logs.parquet"
if not os.path.exists(team_parquet_path):
	print("\n[WARNING] Team game logs parquet file not found:", team_parquet_path)
else:
	try:
		team_df = pd.read_parquet(team_parquet_path)
		if team_df.empty:
			print("\n[WARNING] Team game logs parquet file is empty:", team_parquet_path)
		else:
			print("\nTeam game logs shape (rows, columns):", team_df.shape)
			print("Team game logs columns:", list(team_df.columns))
	except Exception as e:
		print(f"\n[ERROR] Could not read team game logs parquet: {e}")

#check column names (debugging)
#print(df.columns)

# 2. Inspect for missing or duplicate entries
print("\nMissing values per column:")
print(df.isnull().sum())
print("\nNumber of duplicate rows:", df.duplicated().sum())


# 3. Validate against known NBA facts
# Use MATCHUP column to estimate number of unique teams
def extract_team_from_matchup(matchup):
	# Assumes format 'TEAM vs. OPP' or 'TEAM @ OPP'
	return matchup.split(' ')[0] if isinstance(matchup, str) else None

df['TEAM'] = df['MATCHUP'].apply(extract_team_from_matchup)
print("\nNumber of unique teams (from MATCHUP):", df['TEAM'].nunique())

# Example: total games in 2022-23 should be around 1,230 (regular season)
games_per_season = df[df['SEASON_ID'] == '22022']['Game_ID'].nunique() #22022 is the format used in data for 2022-23 season
print("Number of unique games in 2022-23:", games_per_season)


import csv

# 4. Spot check player data using player_id_name_map CSV
player_map_path = "data/historical/player_id_name_map_['2022-23', '2023-24', '2024-25'].csv"
player_name_to_id = {}
with open(player_map_path, newline='', encoding='utf-8') as csvfile:
	reader = csv.DictReader(csvfile)
	for row in reader:
		player_name_to_id[row['DISPLAY_FIRST_LAST']] = int(row['PERSON_ID'])

# Example spot check: LeBron James
player_name = "Brandin Podziemski"
player_id = player_name_to_id.get(player_name)
if player_id is not None:
	print(f"\nSample games for {player_name} (ID: {player_id}):")
	# Try to find games for this player by PLAYER_ID column
	if 'PLAYER_ID' in df.columns:
		print(df[df['PLAYER_ID'] == player_id].head())
	else:
		print("PLAYER_ID column not found in DataFrame.")
else:
	print(f"Player '{player_name}' not found in mapping file.")


# -------------------------
# Additional thorough data quality validations
# -------------------------
import json
import numpy as np
from pathlib import Path


def write_report(report_obj, txt_path, json_path=None):
	Path(txt_path).parent.mkdir(parents=True, exist_ok=True)
	with open(txt_path, 'w', encoding='utf-8') as f:
		f.write(report_obj['summary'] + '\n\n')
		for line in report_obj.get('details', []):
			f.write(line + '\n')
	if json_path:
		with open(json_path, 'w', encoding='utf-8') as fj:
			json.dump(report_obj, fj, indent=2)


report = {'summary': '', 'details': []}

# helper
def safe_read_parquet(p):
	try:
		return pd.read_parquet(p)
	except Exception as e:
		report['details'].append(f"[ERROR] Could not read parquet {p}: {e}")
		return None


# Paths
team_parquet_path = "data/historical/team_game_logs.parquet"
player_parquet_path = parquet_path  # already defined above
report_txt = "data/historical/data_quality_report.txt"
report_json = "data/historical/data_quality_report.json"

# Read team and player tables if available
team_df = None
if os.path.exists(team_parquet_path):
	team_df = safe_read_parquet(team_parquet_path)

player_df = None
if os.path.exists(player_parquet_path):
	player_df = safe_read_parquet(player_parquet_path)

# 1) Basic presence checks
report['details'].append(f"player_game_logs present: {player_df is not None}")
report['details'].append(f"team_game_logs present: {team_df is not None}")

# 2) Critical missing values in player table
critical_cols_player = ['GAME_ID', 'SEASON', 'MATCHUP', 'PLAYER_ID', 'PTS']
missing_player = {}
if player_df is not None:
	for c in critical_cols_player:
		missing_player[c] = int(player_df[c].isnull().sum()) if c in player_df.columns else None
	report['details'].append("Missing values in player table: " + str(missing_player))

# 3) Critical missing values in team table
critical_cols_team = ['GAME_ID', 'SEASON', 'TEAM', 'PTS']
missing_team = {}
if team_df is not None:
	for c in critical_cols_team:
		missing_team[c] = int(team_df[c].isnull().sum()) if c in team_df.columns else None
	report['details'].append("Missing values in team table: " + str(missing_team))

# 4) GAME_ID -> number of team rows sanity (should be 2 per game)
def game_team_counts(df, team_col):
	# count unique team values per GAME_ID using provided team column
	return df.groupby('GAME_ID')[team_col].nunique()


# choose a sensible team column name if present
def choose_team_col(df):
	if df is None:
		return None
	candidates = ['TEAM', 'TEAM_ID', 'TEAM_ABBREVIATION', 'TEAM_ABBR', 'TEAM_NAME', 'TEAMNAME']
	cols = [c.upper() for c in df.columns]
	for cand in candidates:
		if cand.upper() in cols:
			# return actual case-sensitive column name from df
			for c in df.columns:
				if c.upper() == cand.upper():
					return c
	return None


team_col_used = choose_team_col(team_df)
if team_df is not None and 'GAME_ID' in team_df.columns and team_col_used:
	gt = game_team_counts(team_df, team_col_used)
	bad_games = gt[gt != 2]
	report['details'].append(f"Total unique GAME_IDs in team table: {gt.size}")
	report['details'].append(f"Games with team-count !=2: {len(bad_games)} (using team column '{team_col_used}')")
	if len(bad_games) > 0:
		report['details'].append("Examples (GAME_ID -> team_count):")
		for gid, cnt in bad_games.head(10).items():
			report['details'].append(f"  {gid} -> {int(cnt)}")
else:
	report['details'].append(f"Skipping GAME_ID team-count check: required columns not present in team table (tried TEAM variants). Found team col: {team_col_used}")

# 5) For games present in both tables: check team PTS sums == player PTS sums (within tolerance)
if team_df is not None and player_df is not None and 'GAME_ID' in player_df.columns and 'GAME_ID' in team_df.columns:
	# Sum player PTS per game
	if 'PTS' in player_df.columns and 'PTS' in team_df.columns:
		player_pts = player_df.groupby('GAME_ID', dropna=False)['PTS'].sum().rename('player_pts')
		team_pts = team_df.groupby('GAME_ID', dropna=False)['PTS'].sum().rename('team_pts')
		pts_cmp = pd.concat([player_pts, team_pts], axis=1)
		pts_cmp['diff'] = pts_cmp['team_pts'] - pts_cmp['player_pts']
		# allow small negative/positive diffs (e.g., rounding/data issues), but flag large
		tol = 1e-6
		large_mismatch = pts_cmp[np.abs(pts_cmp['diff']) > tol]
		report['details'].append(f"Games compared for pts mismatch: {len(pts_cmp)}")
		report['details'].append(f"Games with non-zero PTS diff: {len(large_mismatch)}")
		if len(large_mismatch) > 0:
			report['details'].append("Top mismatches (GAME_ID, team_pts, player_pts, diff):")
			for idx, row in large_mismatch.sort_values('diff', key=lambda s: np.abs(s)).head(10).iterrows():
				report['details'].append(f"  {idx}: {row['team_pts']} vs {row['player_pts']} -> diff={row['diff']}")
	else:
		report['details'].append("Skipping PTS consistency check: PTS missing in one of the tables.")

# 6) Negative stat checks (player and team)
stat_cols = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TO']
neg_issues = []
for df_name, df_obj in [('player', player_df), ('team', team_df)]:
	if df_obj is None:
		continue
	for c in stat_cols:
		if c in df_obj.columns:
			neg_count = int((df_obj[c] < 0).sum())
			if neg_count > 0:
				neg_issues.append(f"{neg_count} negative values in {df_name}.{c}")
report['details'].append("Negative stat issues: " + (", ".join(neg_issues) if neg_issues else "none"))

# 7) Duplicate team rows for same GAME_ID and TEAM
if team_df is not None:
	if set(['GAME_ID', 'TEAM']).issubset(team_df.columns):
		dup_team_rows = team_df.duplicated(subset=['GAME_ID', 'TEAM']).sum()
		report['details'].append(f"Duplicate (GAME_ID,TEAM) rows in team table: {int(dup_team_rows)}")

# 8) Win consistency: exactly one winner per GAME_ID
if team_df is not None and 'GAME_ID' in team_df.columns and 'PTS' in team_df.columns:
	# compute winner per game from PTS
	winners = team_df.loc[team_df.groupby('GAME_ID')['PTS'].idxmax()]
	winner_counts = winners.groupby('GAME_ID').size()
	# should be 1 per game
	report['details'].append(f"Winners computed for {winner_counts.size} games (expected one per game)")
	# if there's a WIN column, cross-check
	if 'WIN' in team_df.columns:
		win_by_game = team_df.groupby('GAME_ID')['WIN'].sum()
		bad_win = win_by_game[win_by_game != 1]
		report['details'].append(f"Games where WIN column sum != 1: {len(bad_win)}")
		if len(bad_win) > 0:
			report['details'].append("Examples of WIN sums != 1:")
			for gid, val in bad_win.head(10).items():
				report['details'].append(f"  {gid} -> WIN_sum={int(val)}")

			# Additional diagnostics: check for games where both teams have identical PTS
			try:
				pts_unique = team_df.groupby('GAME_ID')['PTS'].nunique()
				tie_gids = pts_unique[pts_unique == 1].index.tolist()
				report['details'].append(f"Games where both teams have identical PTS (ties): {len(tie_gids)}")
				# Attempt to resolve ties using player-level WL field (if available)
				resolved = 0
				unresolved_examples = []
				if player_df is not None:
					cmap = {c.upper(): c for c in player_df.columns}
					gcol = cmap.get('GAME_ID')
					mcol = cmap.get('MATCHUP')
					wlcol = cmap.get('WL')
					if gcol and mcol and wlcol:
						for gid in tie_gids[:50]:
							sub = player_df[player_df[gcol].astype(str) == gid]
							if sub.empty:
								unresolved_examples.append(gid)
								continue
							sub = sub.copy()
							sub['TEAM_TOKEN'] = sub[mcol].astype(str).map(lambda m: str(m).split()[0])
							wins_by_token = sub[sub[wlcol] == 'W'].groupby('TEAM_TOKEN').size()
							if wins_by_token.empty:
								unresolved_examples.append(gid)
								continue
							# if we can identify a token with W counts, consider resolved
							resolved += 1
					report['details'].append(f"Ties resolved via player WL (approx): {resolved}")
					if unresolved_examples:
						report['details'].append(f"Tie examples unresolved via player WL: {unresolved_examples[:10]}")
			except Exception:
				# diagnostic should not crash the QC script
				pass

# 9) Players-per-team-per-game reasonable bounds
if player_df is not None and set(['GAME_ID', 'TEAM', 'PLAYER_ID']).issubset(player_df.columns):
	pcounts = player_df.groupby(['GAME_ID', 'TEAM'])['PLAYER_ID'].nunique()
	too_few = (pcounts < 5).sum()
	too_many = (pcounts > 15).sum()
	report['details'].append(f"Player-counts per (GAME_ID,TEAM): <5: {int(too_few)}, >15: {int(too_many)}")

# 10) Expected overall team-game row count (30 teams * 82 games * seasons)
expected_calc = None
if team_df is not None and 'SEASON' in team_df.columns:
	seasons = team_df['SEASON'].nunique()
	expected_calc = 30 * 82 * int(seasons)
	report['details'].append(f"Observed team-game rows: {len(team_df)}, seasons observed: {int(seasons)}, expected ~{expected_calc}")

# 11) Report summary
summary_lines = []
if team_df is None and player_df is None:
	report['summary'] = "No player or team parquet files readable; nothing to validate."
else:
	problems = [ln for ln in report['details'] if 'ERROR' in ln or 'non-zero PTS diff' in ln or 'negative' in ln or '!=2' in ln or 'Duplicate' in ln or '<5' in ln or '>15' in ln]
	if len(problems) == 0:
		report['summary'] = "Basic data quality checks passed. No glaring issues detected."
	else:
		report['summary'] = f"Data quality checks found potential issues (count={len(problems)}). See details below."

# Write report files
write_report(report, report_txt, report_json)
print(f"\nData quality report written to: {report_txt} and {report_json}")

