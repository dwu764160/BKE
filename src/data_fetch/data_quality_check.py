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
