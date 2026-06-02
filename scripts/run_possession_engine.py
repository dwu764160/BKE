"""
scripts/run_possession_engine.py
=============================================================================
Simulation Core — Step 2 runner.

Loads projected lineups (Step 0), behavioral rates (projected_player_profiles),
pts_v40 talent, and the Step-1 matchup adjustments, calibrates the engine's
global constants on TRAIN seasons (<=2023-24), then Monte-Carlo simulates a
target season's schedule and writes per-(game, team) box-score distributions.

  v0  : 5 starters (projected MPG) + 1 aggregate bench unit (covers 240 min).
  v1  : top 9-11 players by projected MPG, individually (no bench aggregate).

Calibration (structure fixed, scale tuned, leakage-free):
  pace      = mean team possessions/game on train
  oreb      = mean OREB% on train
  league_ft = mean FT% on train
  league_3p = mean 3P% on train
  efg_scale = single global multiplier so simulated mean team PTS == actual
              mean team PTS on a train sample.

Output:
  data/processed/simulation/possession_box_distributions.parquet
  data/processed/simulation/possession_engine_constants.json

Usage:
  python3 scripts/run_possession_engine.py --mode v0 --season 2024-25 --sims 120
  python3 scripts/run_possession_engine.py --mode v1 --season 2024-25 --sims 120
=============================================================================
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.schema_contract import load_standardized  # noqa: E402
from src.simulation.possession_engine import (  # noqa: E402
    GlobalConstants, Player, Lineup, RateModel, OutcomeSamplerResolver, simulate_game,
)

LINEUPS = ROOT / "data/processed/simulation/simulation_step2_lineup_profiles.parquet"
PROFILES = ROOT / "data/processed/forecast/projected_player_profiles.parquet"
IMPACT_PROFILES = ROOT / "data/processed/player_eval/player_impact_profiles.parquet"
PTS_V40 = ROOT / "data/processed/bke/pts_v40.parquet"
MATCHUP = ROOT / "data/processed/bke/cross_team_interactions.parquet"
TEAM_LOGS = ROOT / "data/historical/team_game_logs.parquet"   # combined, all seasons
TEAMS = ROOT / "data/historical/teams.parquet"
_TEAM_LOGS_CACHE: Optional[pd.DataFrame] = None
OUT = ROOT / "data/processed/simulation/possession_box_distributions.parquet"
CONST_OUT = ROOT / "data/processed/simulation/possession_engine_constants.json"

TRAIN_SEASONS = ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24"]


def _norm(x) -> str:
    return str(x).strip().replace(".0", "")


def _ids(v) -> List[str]:
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if isinstance(v, (list, tuple)):
        return [_norm(x) for x in v if str(x).strip()]
    if isinstance(v, str) and v.strip().startswith("["):
        try:
            return [_norm(x) for x in ast.literal_eval(v)]
        except Exception:
            return [_norm(x) for x in v.strip("[]").replace("'", "").replace('"', "").split(",") if x.strip()]
    return []


# ---------------------------------------------------------------------------
def load_rates() -> Dict[Tuple[str, str], dict]:
    """(player_id, season) -> behavioral rate dict + archetype/band/mpg."""
    rates: Dict[Tuple[str, str], dict] = {}
    cols = ["player_id", "season", "off_primary_archetype", "position_band_3",
            "behavioral_usage", "behavioral_efg", "behavioral_three_point_rate",
            "behavioral_turnover_rate", "behavioral_assist_rate",
            "behavioral_free_throw_rate", "behavioral_orb_rate",
            "behavioral_drb_rate", "behavioral_foul_rate", "mpg"]
    df = pd.read_parquet(PROFILES)
    have = [c for c in cols if c in df.columns]
    df = df[have].copy()
    df["player_id"] = df["player_id"].map(_norm)
    df["season"] = df["season"].astype(str)
    for r in df.itertuples(index=False):
        d = r._asdict()
        rates[(d["player_id"], d["season"])] = d
    return rates


def load_pts() -> Tuple[dict, dict]:
    pts = pd.read_parquet(PTS_V40)
    pts["player_id"] = pts["player_id"].map(_norm)
    pts["season"] = pts["season"].astype(str)
    po = {(r.player_id, r.season): float(r.pts_o_v40) for r in pts.itertuples()}
    pd_ = {(r.player_id, r.season): float(r.pts_d_v40) for r in pts.itertuples()}
    return po, pd_


def load_matchup() -> Tuple[dict, dict]:
    """(season, team, opp) -> (player_matchup_adjs dict, center_overloaded)."""
    if not MATCHUP.exists():
        return {}, {}
    m = pd.read_parquet(MATCHUP)
    adjs, overload = {}, {}
    for r in m.itertuples():
        key = (str(r.season), str(r.team), str(r.opponent))
        try:
            adjs[key] = {(_norm(k)): float(v) for k, v in json.loads(r.player_matchup_adjs).items()}
        except Exception:
            adjs[key] = {}
        overload[key] = bool(getattr(r, "center_overloaded", False))
    return adjs, overload


def _player_from_rate(pid: str, season: str, rd: Optional[dict], pts_o: dict, pts_d: dict,
                      matchup_adj: float) -> Player:
    rd = rd or {}
    def g(k, d):
        v = rd.get(k)
        try:
            v = float(v)
            return v if np.isfinite(v) else d
        except Exception:
            return d
    return Player(
        player_id=pid, mpg=g("mpg", 0.0),
        off_arch=rd.get("off_primary_archetype"), band=rd.get("position_band_3"),
        usage=g("behavioral_usage", 0.18), efg=g("behavioral_efg", 0.52),
        three_rate=g("behavioral_three_point_rate", 0.38),
        tov_rate=g("behavioral_turnover_rate", 0.124),
        ast_rate=g("behavioral_assist_rate", 0.137),
        ftr=g("behavioral_free_throw_rate", 0.22),
        orb_rate=g("behavioral_orb_rate", 0.026),
        drb_rate=g("behavioral_drb_rate", 0.118),
        foul_rate=g("behavioral_foul_rate", 0.124),
        pts_o=pts_o.get((pid, season), 0.0), pts_d=pts_d.get((pid, season), 0.0),
        matchup_adj=matchup_adj,
    )


def _bench_unit(season: str, used_min: float) -> Player:
    bench_min = max(240.0 - used_min, 0.0)
    return Player(player_id=f"BENCH_{season}", mpg=bench_min, off_arch=None, band="wings",
                  usage=0.18, efg=0.52, three_rate=0.36, tov_rate=0.13, ast_rate=0.13,
                  ftr=0.20, orb_rate=0.03, drb_rate=0.12, foul_rate=0.14, is_bench_unit=True)


def build_lineup(season: str, team: str, opp: str, starter_ids: List[str], rates,
                 pts_o, pts_d, adjs, overload, mode: str,
                 team_roster: Optional[List[str]] = None) -> Lineup:
    key = (season, team, opp)
    padj = adjs.get(key, {})
    co = overload.get(key, False)
    if mode == "v0":
        players = []
        used = 0.0
        for pid in starter_ids[:5]:
            p = _player_from_rate(pid, season, rates.get((pid, season)), pts_o, pts_d, padj.get(pid, 0.0))
            used += p.mpg
            players.append(p)
        players.append(_bench_unit(season, used))
    else:  # v1: tight 9-man rotation by mpg from the roster
        roster = team_roster or starter_ids
        cand = []
        for pid in roster:
            p = _player_from_rate(pid, season, rates.get((pid, season)), pts_o, pts_d, padj.get(pid, 0.0))
            cand.append(p)
        cand.sort(key=lambda x: x.mpg, reverse=True)
        # Step 2.1 Defect 3 — tighten to 9. Season-average projected minutes spread
        # across 11+ bodies, but any single game uses ~8-9 players at 12+ min. Simulating
        # an 11-man rotation leaked shot volume to a phantom 10th/11th man (excluded from
        # the >=12min validation), systematically under-crediting the real rotation. A
        # 9-man cap redistributes that share proportionally back to ranks 1-8.
        players = cand[:9]
        players = [p for p in players if p.mpg > 0] or cand[:5]
        # normalize minutes to 240
        tot = sum(p.mpg for p in players)
        if tot > 0:
            for p in players:
                p.mpg = p.mpg * 240.0 / tot
    return Lineup(team=team, players=players, center_overloaded=co)


# ---------------------------------------------------------------------------
def _all_team_logs() -> pd.DataFrame:
    global _TEAM_LOGS_CACHE
    if _TEAM_LOGS_CACHE is None:
        tg = load_standardized(TEAM_LOGS)
        tg["season"] = tg["season"].astype(str)
        _TEAM_LOGS_CACHE = tg
    return _TEAM_LOGS_CACHE


def load_team_logs(season: str) -> pd.DataFrame:
    tg = _all_team_logs()
    tg = tg[tg["season"] == str(season)].copy()
    tg["game_id"] = tg["game_id"].astype(str)
    # parse team + opp + home from matchup "ABC vs. XYZ" / "ABC @ XYZ"
    tg["TEAM_ABBR"] = tg["matchup"].str.split().str[0]
    tg["HOME"] = tg["matchup"].str.contains("vs.", regex=False)
    tg["OPP_ABBR"] = tg["matchup"].str.split().str[-1]
    poss = tg["fga"] + 0.44 * tg["fta"] - tg["oreb"] + tg["tov"]
    tg["POSS_EST"] = poss
    return tg


def team_roster_map(season: str, rates) -> Dict[str, List[str]]:
    """team_abbr -> [player_ids] for the season, from profiles (for v1)."""
    df = pd.read_parquet(PROFILES, columns=["player_id", "season", "team_abbreviation", "mpg"])
    df = df[df["season"].astype(str) == season].copy()
    df["player_id"] = df["player_id"].map(_norm)
    df["team_abbreviation"] = df["team_abbreviation"].astype(str).str.upper()
    out: Dict[str, List[str]] = {}
    for team, g in df.sort_values("mpg", ascending=False).groupby("team_abbreviation"):
        out[team] = g["player_id"].tolist()
    return out


def calibrate(rates, pts_o, pts_d, adjs, overload, lineup_map, mode: str,
              rng: np.random.Generator) -> GlobalConstants:
    """Set global constants from train actuals; tune efg_scale to match team PTS."""
    # league rate anchors from train team logs (combined file, filtered)
    paces, orebs, fts, threes, ptss = [], [], [], [], []
    all_tg = _all_team_logs()
    for s in TRAIN_SEASONS:
        tg = all_tg[all_tg["season"] == str(s)]
        if tg.empty:
            continue
        poss = tg["fga"] + 0.44 * tg["fta"] - tg["oreb"] + tg["tov"]
        paces.append(poss.mean())
        orebs.append((tg["oreb"] / (tg["oreb"] + tg["dreb"]).replace(0, np.nan)).mean())
        fts.append((tg["ftm"] / tg["fta"].replace(0, np.nan)).mean())
        threes.append((tg["fg3m"] / tg["fg3a"].replace(0, np.nan)).mean())
        ptss.append(tg["pts"].mean())
    g = GlobalConstants(
        pace=float(np.nanmean(paces)) if paces else 99.0,
        oreb=float(np.nanmean(orebs)) if orebs else 0.26,
        league_ft=float(np.nanmean(fts)) if fts else 0.78,
        league_3p=float(np.nanmean(threes)) if threes else 0.36,
    )
    target_pts = float(np.nanmean(ptss)) if ptss else 113.0

    # tune efg_scale: simulate a sample of train matchups, match mean team PTS
    resolver = OutcomeSamplerResolver(RateModel(g))
    sample = []
    for s in TRAIN_SEASONS:
        teams = lineup_map.get(s, {})
        names = list(teams.keys())
        for i in range(0, min(len(names), 16), 2):
            if i + 1 < len(names):
                sample.append((s, names[i], names[i + 1]))
    if not sample:
        g.efg_scale = 1.0
        return g

    def sim_mean(scale: float) -> float:
        g.efg_scale = scale
        vals = []
        for (s, a, b) in sample:
            la = teams_lineup(s, a, b, rates, pts_o, pts_d, adjs, overload, lineup_map, mode)
            lb = teams_lineup(s, b, a, rates, pts_o, pts_d, adjs, overload, lineup_map, mode)
            if la is None or lb is None:
                continue
            ra = simulate_game(la, lb, resolver, 30, rng)
            vals.append(ra["team_pts_mean"])
        return float(np.mean(vals)) if vals else target_pts

    base = sim_mean(1.0)
    scale = float(np.clip(target_pts / base, 0.7, 1.4)) if base > 0 else 1.0
    # one refinement pass
    refined = sim_mean(scale)
    if refined > 0:
        scale *= float(np.clip(target_pts / refined, 0.85, 1.15))
    g.efg_scale = float(np.clip(scale, 0.7, 1.4))
    return g


_LINEUP_CACHE: Dict[tuple, dict] = {}


def teams_lineup(season, team, opp, rates, pts_o, pts_d, adjs, overload, lineup_map, mode):
    starters = _LINEUP_CACHE.get((season, team))
    roster = lineup_map.get(season, {}).get(team)
    if starters is None and roster is None:
        return None
    sids = starters or []
    return build_lineup(season, team, opp, sids, rates, pts_o, pts_d, adjs, overload, mode,
                        team_roster=roster)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["v0", "v1"], default="v0")
    ap.add_argument("--season", default="2024-25")
    ap.add_argument("--sims", type=int, default=120)
    args = ap.parse_args()

    rng = np.random.default_rng(20260531)
    rates = load_rates()
    pts_o, pts_d = load_pts()
    adjs, overload = load_matchup()

    # starter map from lineup profiles
    lp = load_standardized(LINEUPS)
    lp["season"] = lp["season"].astype(str)
    lp["team_abbreviation"] = lp["team_abbreviation"].astype(str).str.upper()
    for r in lp.itertuples():
        _LINEUP_CACHE[(str(r.season), str(r.team_abbreviation))] = _ids(getattr(r, "starter_player_ids", None))

    # roster map per train+target season for v1 + calibration sampling
    lineup_map: Dict[str, Dict[str, List[str]]] = {}
    for s in TRAIN_SEASONS + [args.season]:
        try:
            lineup_map[s] = team_roster_map(s, rates)
        except Exception:
            lineup_map[s] = {}

    g = calibrate(rates, pts_o, pts_d, adjs, overload, lineup_map, args.mode, rng)
    CONST_OUT.parent.mkdir(parents=True, exist_ok=True)
    CONST_OUT.write_text(json.dumps({"mode": args.mode, **g.to_dict()}, indent=2))
    resolver = OutcomeSamplerResolver(RateModel(g))

    # simulate target season schedule
    tg = load_team_logs(args.season)
    games = tg[["game_id", "TEAM_ABBR", "OPP_ABBR", "HOME"]].drop_duplicates()
    rows = []
    n_done = 0
    for r in games.itertuples():
        team, opp = str(r.TEAM_ABBR).upper(), str(r.OPP_ABBR).upper()
        off = teams_lineup(args.season, team, opp, rates, pts_o, pts_d, adjs, overload, lineup_map, args.mode)
        deff = teams_lineup(args.season, opp, team, rates, pts_o, pts_d, adjs, overload, lineup_map, args.mode)
        if off is None or deff is None or not off.players or not deff.players:
            continue
        res = simulate_game(off, deff, resolver, args.sims, rng)
        for pr in res["player_rows"]:
            rows.append({"game_id": r.game_id, "season": args.season, "team": team,
                         "opponent": opp, "home": bool(r.HOME), "mode": args.mode,
                         "side": "off", **pr})
        # defensive contributions accrue to the DEFENDING players (team=opp)
        for dr in res["def_rows"]:
            rows.append({"game_id": r.game_id, "season": args.season, "team": opp,
                         "opponent": team, "home": not bool(r.HOME), "mode": args.mode,
                         "side": "def", **dr})
        rows.append({"game_id": r.game_id, "season": args.season, "team": team,
                     "opponent": opp, "home": bool(r.HOME), "mode": args.mode,
                     "player_id": "TEAM", "is_bench_unit": False,
                     "pts_mean": res["team_pts_mean"], "pts_p10": res["team_pts_p10"],
                     "pts_p50": res["team_pts_p50"], "pts_p90": res["team_pts_p90"],
                     "pts_std": res["team_pts_std"]})
        n_done += 1

    out_df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(OUT, index=False)
    team_rows = out_df[out_df["player_id"] == "TEAM"]
    print("RUN_OK mode=%s season=%s games=%d rows=%d pace=%.1f efg_scale=%.3f "
          "sim_team_pts_mean=%.1f" % (
              args.mode, args.season, n_done, len(out_df), g.pace, g.efg_scale,
              float(team_rows["pts_mean"].mean()) if len(team_rows) else -1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
