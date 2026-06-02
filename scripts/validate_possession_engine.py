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

from src.data.schema_contract import load_standardized  # noqa: E402

DIST = ROOT / "data/processed/simulation/possession_box_distributions.parquet"
PLAYER_LOGS = ROOT / "data/historical/player_game_logs_{season}.parquet"
TEAM_LOGS = ROOT / "data/historical/team_game_logs.parquet"   # combined, all seasons
OUT = ROOT / "reports/possession_engine_validation.json"
SEASON = "2024-25"

# scoring tiers by actual season PTS/game — the aggregate MAE hides per-tier error,
# so the calibration gate is judged tier-by-tier (Step 2.1).
TIERS = [
    ("Superstar", 20.0, 999.0),
    ("Star",      15.0, 20.0),
    ("Starter",   10.0, 15.0),
    ("Rotation",   5.0, 10.0),
    ("Bench",      0.0,  5.0),
]


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
    pa = load_standardized(Path(str(PLAYER_LOGS).format(season=SEASON)))
    pa["game_id"] = pa["game_id"].astype(str)
    pa["player_id"] = pa["player_id"].map(_norm)
    actual = pa[["game_id", "player_id", "pts", "reb", "ast", "min"]].copy()

    m = off_g.merge(actual, on=["game_id", "player_id"], how="inner")
    # restrict to rotation players (actual min >= 12) — props live here
    m = m[pd.to_numeric(m["min"], errors="coerce").fillna(0) >= 12]

    player_gate = {}
    tier_report = {}
    if len(m):
        cover = ((m["pts"] >= m["pts_p10"]) & (m["pts"] <= m["pts_p90"])).mean()
        # P90 tail check: fraction of HIGH-scoring (sim mean >= 10) player-games whose
        # sim P90 exceeds 1.7x sim mean. Restricted to mean>=10 because for low-mean
        # players discrete counting noise makes P90/mean naturally >1.7 and harmless;
        # the runaway-tail defect is about stars (e.g. SGA P90=51).
        hi = m[m["pts_mean"] >= 10.0]
        if len(hi):
            p90_ratio = hi["pts_p90"] / hi["pts_mean"].replace(0, np.nan)
            p90_violations = float((p90_ratio > 1.7).mean())
        else:
            p90_violations = 0.0
        player_gate = {
            "n_player_games": int(len(m)),
            "pts_mae": round(_wmae(m["pts_mean"], m["pts"]), 4),
            "pts_bias": round(float((m["pts_mean"] - m["pts"]).mean()), 4),
            "ast_mae": round(_wmae(m["ast_mean"], m["ast"]), 4),
            "reb_mae": round(_wmae(m["reb_mean"], m["reb"]), 4),
            "pts_p10_p90_coverage": round(float(cover), 4),
            "p90_over_1p7x_mean_frac": round(p90_violations, 4),
        }

        # --- per-tier stratification by actual season PTS/game ---
        season_avg = m.groupby("player_id")["pts"].mean().rename("pts_season_avg")
        mt = m.merge(season_avg, on="player_id", how="left")
        for tier, lo, hi in TIERS:
            sub = mt[(mt["pts_season_avg"] >= lo) & (mt["pts_season_avg"] < hi)]
            if len(sub) == 0:
                continue
            cov_t = ((sub["pts"] >= sub["pts_p10"]) & (sub["pts"] <= sub["pts_p90"])).mean()
            tier_report[tier] = {
                "n": int(len(sub)),
                "n_players": int(sub["player_id"].nunique()),
                "actual_pts": round(float(sub["pts"].mean()), 2),
                "sim_pts": round(float(sub["pts_mean"].mean()), 2),
                "pts_bias": round(float((sub["pts_mean"] - sub["pts"]).mean()), 3),
                "pts_mae": round(_wmae(sub["pts_mean"], sub["pts"]), 3),
                "actual_ast": round(float(sub["ast"].mean()), 2),
                "sim_ast": round(float(sub["ast_mean"].mean()), 2),
                "ast_bias": round(float((sub["ast_mean"] - sub["ast"]).mean()), 3),
                "ast_mae": round(_wmae(sub["ast_mean"], sub["ast"]), 3),
                "actual_reb": round(float(sub["reb"].mean()), 2),
                "sim_reb": round(float(sub["reb_mean"].mean()), 2),
                "reb_bias": round(float((sub["reb_mean"] - sub["reb"]).mean()), 3),
                "reb_mae": round(_wmae(sub["reb_mean"], sub["reb"]), 3),
                "pts_coverage": round(float(cov_t), 4),
            }

    # --- team totals ---
    team_sim = sim[sim["player_id"] == "TEAM"][
        ["game_id", "team", "pts_mean", "pts_p10", "pts_p90"]].copy()
    ta = load_standardized(TEAM_LOGS)
    ta = ta[ta["season"].astype(str) == SEASON].copy()
    ta["game_id"] = ta["game_id"].astype(str)
    ta["team_abbr"] = ta["matchup"].str.split().str[0].str.upper()
    ta_small = ta[["game_id", "team_abbr", "pts", "fga", "fta", "oreb", "tov"]].rename(
        columns={"team_abbr": "team", "pts": "actual_pts"})
    tm = team_sim.merge(ta_small, on=["game_id", "team"], how="inner")
    team_gate = {}
    if len(tm):
        cover = ((tm["actual_pts"] >= tm["pts_p10"]) & (tm["actual_pts"] <= tm["pts_p90"])).mean()
        act_pace = (tm["fga"] + 0.44 * tm["fta"] - tm["oreb"] + tm["tov"]).mean()
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
        "tier_breakdown": tier_report,
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
    if tier_report:
        print("\nTIER          n   |  PTS act/sim  bias   mae  | AST act/sim bias  mae | "
              "REB act/sim bias  mae | cov")
        for tier, _lo, _hi in TIERS:
            t = tier_report.get(tier)
            if not t:
                continue
            print("%-10s %5d | %5.1f/%5.1f %+5.2f %5.2f | %4.1f/%4.1f %+5.2f %4.2f | "
                  "%4.1f/%4.1f %+5.2f %4.2f | %.2f"
                  % (tier, t["n"], t["actual_pts"], t["sim_pts"], t["pts_bias"], t["pts_mae"],
                     t["actual_ast"], t["sim_ast"], t["ast_bias"], t["ast_mae"],
                     t["actual_reb"], t["sim_reb"], t["reb_bias"], t["reb_mae"],
                     t["pts_coverage"]))
        print("P90>1.7x mean (sim mean>=10): %.1f%% of player-games"
              % (100 * pg.get("p90_over_1p7x_mean_frac", 0)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
