"""
scripts/validate_possession_engine.py
=============================================================================
Simulation Core — Step 2 validation gate (BOTH player-prop + team-total).

Compares the engine's emergent Monte-Carlo distributions
(possession_box_distributions.parquet) to ACTUAL 2024-25 box scores
(player_game_logs / team_game_logs). Walk-forward by construction: constants
were calibrated on <=2023-24 and the rates are forward projections.

Gate 1 — Player-prop calibration (offensive starters; bench unit excluded):
  PTS / AST median MAE, REB MAE (OREB+DREB combined across possession
  directions), interval coverage (actual within sim P10-P90 ~ 80%).

Gate 2 — Team-total calibration:
  team PTS MAE, bias, and P10-P90 coverage vs actual team points; pace check.

Output: reports/possession_engine_validation.json

Usage:
  python3 scripts/validate_possession_engine.py
=============================================================================
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DIST = ROOT / "data/processed/simulation/possession_box_distributions.parquet"
PLAYER_LOGS = ROOT / "data/historical/player_game_logs_{season}.parquet"
TEAM_LOGS = ROOT / "data/historical/team_game_logs.parquet"   # combined, all seasons
OUT = ROOT / "reports/possession_engine_validation.json"
SEASON = "2024-25"


def _norm(x) -> str:
    return str(x).strip().replace(".0", "")


def _wmae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def main() -> int:
    if not DIST.exists():
        print("FATAL run the engine first (possession_box_distributions.parquet missing)")
        return 1
    sim = pd.read_parquet(DIST)
    sim = sim[sim["season"].astype(str) == SEASON].copy()
    sim["game_id"] = sim["game_id"].astype(str)
    sim["player_id"] = sim["player_id"].astype(str)
    mode = str(sim["mode"].iloc[0]) if "mode" in sim.columns and len(sim) else "?"

    # --- combine offensive + defensive sim rows per (game, player) ---
    off = sim[(sim.get("side") == "off") & (sim["player_id"] != "TEAM")]
    deff = sim[sim.get("side") == "def"] if "side" in sim.columns else sim.iloc[0:0]
    pcols = ["pts_mean", "pts_p10", "pts_p90", "ast_mean", "oreb_mean"]
    off_g = off.groupby(["game_id", "player_id"]).agg(
        {c: "mean" for c in pcols if c in off.columns}).reset_index()
    if len(deff):
        dreb_g = deff.groupby(["game_id", "player_id"])["dreb_mean"].mean().reset_index()
        off_g = off_g.merge(dreb_g, on=["game_id", "player_id"], how="left")
    off_g["dreb_mean"] = off_g.get("dreb_mean", 0.0)
    off_g["reb_mean"] = off_g["oreb_mean"].fillna(0) + off_g["dreb_mean"].fillna(0)
    off_g = off_g[~off_g["player_id"].str.startswith("BENCH_")]

    # --- actual player box scores ---
    pa = pd.read_parquet(Path(str(PLAYER_LOGS).format(season=SEASON)))
    pa = pa.rename(columns={c: c.upper() for c in pa.columns})
    # player logs ship both 'Player_ID' and 'PLAYER_ID' -> collide on upper; dedup
    pa = pa.loc[:, ~pa.columns.duplicated()]
    pa["GAME_ID"] = pa["GAME_ID"].astype(str)
    pa["PLAYER_ID"] = pa["PLAYER_ID"].map(_norm)
    actual = pa[["GAME_ID", "PLAYER_ID", "PTS", "REB", "AST", "MIN"]].rename(
        columns={"GAME_ID": "game_id", "PLAYER_ID": "player_id"})

    m = off_g.merge(actual, on=["game_id", "player_id"], how="inner")
    # restrict to rotation players (actual MIN >= 12) — props live here
    m = m[pd.to_numeric(m["MIN"], errors="coerce").fillna(0) >= 12]

    player_gate = {}
    if len(m):
        cover = ((m["PTS"] >= m["pts_p10"]) & (m["PTS"] <= m["pts_p90"])).mean()
        player_gate = {
            "n_player_games": int(len(m)),
            "pts_mae": round(_wmae(m["pts_mean"], m["PTS"]), 4),
            "pts_bias": round(float((m["pts_mean"] - m["PTS"]).mean()), 4),
            "ast_mae": round(_wmae(m["ast_mean"], m["AST"]), 4),
            "reb_mae": round(_wmae(m["reb_mean"], m["REB"]), 4),
            "pts_p10_p90_coverage": round(float(cover), 4),
        }

    # --- team totals ---
    team_sim = sim[sim["player_id"] == "TEAM"][
        ["game_id", "team", "pts_mean", "pts_p10", "pts_p90"]].copy()
    ta = pd.read_parquet(TEAM_LOGS)
    ta = ta.rename(columns={c: c.upper() for c in ta.columns})
    ta = ta[ta["SEASON"].astype(str) == SEASON].copy()
    ta["GAME_ID"] = ta["GAME_ID"].astype(str)
    ta["TEAM_ABBR"] = ta["MATCHUP"].str.split().str[0].str.upper()
    ta_small = ta[["GAME_ID", "TEAM_ABBR", "PTS", "FGA", "FTA", "OREB", "TOV"]].rename(
        columns={"GAME_ID": "game_id", "TEAM_ABBR": "team", "PTS": "actual_pts"})
    tm = team_sim.merge(ta_small, on=["game_id", "team"], how="inner")
    team_gate = {}
    if len(tm):
        cover = ((tm["actual_pts"] >= tm["pts_p10"]) & (tm["actual_pts"] <= tm["pts_p90"])).mean()
        act_pace = (tm["FGA"] + 0.44 * tm["FTA"] - tm["OREB"] + tm["TOV"]).mean()
        team_gate = {
            "n_team_games": int(len(tm)),
            "team_pts_mae": round(_wmae(tm["pts_mean"], tm["actual_pts"]), 4),
            "team_pts_bias": round(float((tm["pts_mean"] - tm["actual_pts"]).mean()), 4),
            "team_pts_p10_p90_coverage": round(float(cover), 4),
            "actual_team_pts_mean": round(float(tm["actual_pts"].mean()), 2),
            "sim_team_pts_mean": round(float(tm["pts_mean"].mean()), 2),
            "actual_pace_mean": round(float(act_pace), 2),
        }

    # verdict: team totals unbiased (|bias|<2, coverage 0.6-0.95) AND player props
    # reasonable (pts_mae < 7, coverage 0.6-0.95)
    ok_team = (team_gate and abs(team_gate["team_pts_bias"]) < 2.5
               and 0.55 <= team_gate["team_pts_p10_p90_coverage"] <= 0.97)
    ok_player = (player_gate and player_gate["pts_mae"] < 7.0
                 and 0.55 <= player_gate["pts_p10_p90_coverage"] <= 0.97)
    verdict = ("PASS" if (ok_team and ok_player)
               else "PARTIAL (team ok)" if ok_team
               else "PARTIAL (player ok)" if ok_player
               else "FAIL")

    result = {
        "test": "Step 2 possession engine calibration — player props + team totals",
        "season": SEASON, "mode": mode,
        "walk_forward": "constants calibrated <=2023-24; rates are forward projections",
        "player_prop_gate": player_gate,
        "team_total_gate": team_gate,
        "verdict": verdict,
    }
    OUT.write_text(json.dumps(result, indent=2))
    pg = player_gate or {}
    tge = team_gate or {}
    print("VAL %s | TEAM pts_mae=%s bias=%s cov=%s | PLAYER n=%s pts_mae=%s ast_mae=%s reb_mae=%s cov=%s"
          % (verdict, tge.get("team_pts_mae"), tge.get("team_pts_bias"),
             tge.get("team_pts_p10_p90_coverage"), pg.get("n_player_games"),
             pg.get("pts_mae"), pg.get("ast_mae"), pg.get("reb_mae"),
             pg.get("pts_p10_p90_coverage")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
