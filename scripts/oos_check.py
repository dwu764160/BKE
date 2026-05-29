import pandas as pd
import numpy as np
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.validate_lineup_pts_v2 import build_player_on_court_nrtg, predict_lineup, compute_split_metrics
from scripts.validate_lineup_pts import aggregate_lineup_possessions

PTS_PATH = REPO / "data/processed/bke/pts_v40.parquet"
PBP_DIR = REPO / "data/historical"

seasons = ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]

pts_df = pd.read_parquet(PTS_PATH)

naive_wrs = []
pts_wrs = []

for season in seasons:
    prev_season_year = int(season[:4]) - 1
    prev_season = f"{prev_season_year}-{str(prev_season_year+1)[2:]}"
    
    # Get N-1 per100
    per100_prev = build_player_on_court_nrtg(prev_season)
    if per100_prev.empty:
        continue
        
    pbp_path = PBP_DIR / f"pbp_with_lineups_{season}.parquet"
    if not pbp_path.exists():
        continue
    pbp = pd.read_parquet(pbp_path)
    lineups = aggregate_lineup_possessions(pbp)
    
    # Naive model: use prev season's per100
    pred_naive = predict_lineup(lineups, per100_prev, "per100_off_z", "per100_def_z", scale=20.0)
    naive_res = compute_split_metrics(pred_naive, 50)
    
    # PTS model: use current season's PTS
    pts_season = pts_df[pts_df['season'] == season]
    pred_pts = predict_lineup(lineups, pts_season, "pts_o_v40", "pts_d_v40", scale=20.0)
    pts_res = compute_split_metrics(pred_pts, 50)
    
    if "error" not in naive_res and "error" not in pts_res:
        naive_wrs.append(naive_res["joint"]["weighted_pearson_r"])
        pts_wrs.append(pts_res["joint"]["weighted_pearson_r"])

print(f"Mean Naive OOS wr: {np.mean(naive_wrs):.4f}")
print(f"Mean PTS v4.0 wr: {np.mean(pts_wrs):.4f}")
print(f"Difference: {np.mean(pts_wrs) - np.mean(naive_wrs):+.4f}")
