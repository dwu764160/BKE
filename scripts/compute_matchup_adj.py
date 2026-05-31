"""
scripts/compute_matchup_adj.py
=============================================================================
Simulation Core — Step 1 (Matchup Engine), part 2 of 3.

Given projected lineups (Step 0 projector output,
simulation_step2_lineup_profiles.parquet -> starter_player_ids) and the shrunk
cross-team interaction matrix (fit_archetype_interactions_v2.py), assign 5
offensive vs 5 defensive players with the locked 3-pair engine and emit a
per-(season, off_team, def_team) interaction artifact that BOTH downstream
tracks consume:

  - Markets track: re-run after overriding projected starters with live actuals.
  - Game track (Step 2 possession engine): reads the per-matchup breakdown.

Granularity note: the source matchup data (league_season_matchups.parquet) is
SEASON-aggregated per off x def player pair — it has NO game_id. The Step-0
projector likewise produces ONE projected starting five per (season, team).
A projected matchup between team A and team B is therefore identical for every
game they play in a season, so the artifact is emitted at (season, team,
opponent) granularity with game_id=NULL. The schema's game_id column is
preserved for the Markets track, which will later fan this out per game with
live (not projected) lineups.

3-pair engine (Resolved Design, docs/plans/cross_team_interaction_model.md):
  Pair 1  Primary perimeter threat (smalls/wings creator, max pts_o_v40)
          -> best POA Defender / Wing Stopper (max pts_d_v40). argmax cell.
  Pair 2  Interior threat (bigs/interior-arch, max pts_o_v40) -> Sub-case A
          (big-bodied Forward/Forward-Center defender, pts_d_v40>0 & top-3 on
          team) guards; else Sub-case B center overloaded. Rim-anchor paint
          term modeled as a lineup-composition adjustment.
  Pairs 3-5  Remaining players ranked by pts_o_v40 within position band,
          matched rank-to-rank same band. Cross-band forced -> discounted
          mismatch adj (0.5x empirical: diff0 -0.010, diff1 +0.005, diff2 +0.020).

Player attributes (archetype, position_band_3, position_proxy, mpg) are joined
from projected_player_profiles.parquet (the projection-time source), falling
back to player_impact_profiles.parquet. Talent rank = pts_o_v40 / pts_d_v40
(pts_v40.parquet).

Output: data/processed/bke/cross_team_interactions.parquet

Usage:
  python3 scripts/compute_matchup_adj.py
=============================================================================
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROFILES_LINEUP = ROOT / "data" / "processed" / "simulation" / "simulation_step2_lineup_profiles.parquet"
PROJ_PROFILES = ROOT / "data" / "processed" / "forecast" / "projected_player_profiles.parquet"
IMPACT_PROFILES = ROOT / "data" / "processed" / "player_eval" / "player_impact_profiles.parquet"
MATRIX = ROOT / "data" / "processed" / "bke" / "cross_team_interactions_matrix.parquet"
PTS_V40 = ROOT / "data" / "processed" / "bke" / "pts_v40.parquet"
OUT = ROOT / "data" / "processed" / "bke" / "cross_team_interactions.parquet"

BAND_INDEX = {"smalls": 0, "wings": 1, "bigs": 2}
PROXY_OF_BAND = {"smalls": "Guard", "wings": "Forward", "bigs": "Center"}

CREATOR_ARCHS = {"Ball Dominant Creator", "All-Around Scorer", "Ballhandler"}
POA_ARCHS = {"POA Defender", "Wing Stopper"}
INTERIOR_OFF_ARCHS = {"PnR Rolling Big", "Interior Scorer", "PnR Popping Big"}
RIM_ANCHOR_ARCHS = {"Rim Protector", "Dropping Big"}
BIG_BODIED_PROXIES = {"Forward", "Forward-Center"}

OFF_ARCH_BAND = {
    "Ball Dominant Creator": "smalls", "Ballhandler": "smalls",
    "Perimeter Scorer": "smalls", "Off-Ball Movement Shooter": "smalls",
    "Off-Ball Stationary Shooter": "smalls",
    "All-Around Scorer": "wings", "Connector": "wings", "Off-Ball Finisher": "wings",
    "PnR Rolling Big": "bigs", "PnR Popping Big": "bigs", "Interior Scorer": "bigs",
}

MISMATCH_DISCOUNT = {0: -0.010, 1: 0.005, 2: 0.020}
UNKNOWN_ARCH = {None, "", "Unknown", "Insufficient Minutes", "nan", "None", "NaN"}


def _norm_id(x) -> str:
    return str(x).strip().replace(".0", "")


def _parse_id_list(v) -> List[str]:
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if isinstance(v, (list, tuple)):
        return [_norm_id(x) for x in v if str(x).strip()]
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                return [_norm_id(x) for x in parsed if str(x).strip()]
            except Exception:
                return [_norm_id(x) for x in s.strip("[]").replace("'", "").replace('"', "").split(",") if x.strip()]
        return [_norm_id(s)] if s else []
    return []


# ---------------------------------------------------------------------------
class Matrix:
    """Interaction-cell lookup: argmax + Unknown-off-archetype band fallback."""

    def __init__(self, df: pd.DataFrame):
        self.cell: Dict[Tuple[str, str], float] = {
            (str(r.off_arch), str(r.def_arch)): float(r.interaction_ppp) for r in df.itertuples()
        }
        d = df.copy()
        d["_band"] = d["off_arch"].map(OFF_ARCH_BAND)
        self.band_avg: Dict[Tuple[str, str], float] = {
            (str(b), str(da)): float(g["interaction_ppp"].mean())
            for (b, da), g in d.dropna(subset=["_band"]).groupby(["_band", "def_arch"])
        }
        self.def_avg: Dict[str, float] = {
            str(da): float(g["interaction_ppp"].mean()) for da, g in d.groupby("def_arch")
        }

    def lookup(self, off_arch, def_arch, off_band) -> Tuple[float, str]:
        da = str(def_arch) if def_arch not in UNKNOWN_ARCH else None
        oa = str(off_arch) if off_arch not in UNKNOWN_ARCH else None
        if da is None:
            return 0.0, "no_def_arch"
        if oa is not None and (oa, da) in self.cell:
            return self.cell[(oa, da)], "argmax"
        if off_band and (off_band, da) in self.band_avg:
            return self.band_avg[(off_band, da)], "band_average_fallback"
        if da in self.def_avg:
            return self.def_avg[da], "def_average_fallback"
        return 0.0, "miss"


# ---------------------------------------------------------------------------
def load_player_attrs() -> Dict[Tuple[str, str], dict]:
    """Per (player_id, season): off_arch, def_arch, band(smalls/wings/bigs),
    proxy, mpg. Prefer projected_player_profiles; fall back to impact profiles."""
    attrs: Dict[Tuple[str, str], dict] = {}

    def ingest(path: Path, prefer: bool):
        if not path.exists():
            return
        df = pd.read_parquet(path)
        cols = {c.lower(): c for c in df.columns}
        need = ("player_id", "season", "off_primary_archetype", "def_primary_archetype")
        if not all(k in cols for k in ("player_id", "season")):
            return
        pid_c, seas_c = cols["player_id"], cols["season"]
        off_c = cols.get("off_primary_archetype")
        def_c = cols.get("def_primary_archetype")
        band_c = cols.get("position_band_3")
        proxy_c = cols.get("position_proxy")
        mpg_c = cols.get("mpg") or cols.get("minutes")
        for r in df.itertuples(index=False):
            d = r._asdict()
            key = (_norm_id(d[pid_c]), str(d[seas_c]))
            if not prefer and key in attrs:
                continue
            band = str(d.get(band_c)) if band_c else None
            if band not in BAND_INDEX:
                band = None
            attrs[key] = {
                "off_arch": d.get(off_c) if off_c else None,
                "def_arch": d.get(def_c) if def_c else None,
                "band": band,
                "proxy": str(d.get(proxy_c)) if proxy_c else None,
                "mpg": float(d.get(mpg_c)) if (mpg_c and d.get(mpg_c) is not None and np.isfinite(pd.to_numeric(d.get(mpg_c), errors="coerce"))) else 0.0,
            }

    ingest(IMPACT_PROFILES, prefer=False)   # base coverage (all seasons, actual)
    ingest(PROJ_PROFILES, prefer=True)      # projection-time override where present
    return attrs


def load_pts() -> Tuple[Dict, Dict]:
    pts = pd.read_parquet(PTS_V40)
    pts["player_id"] = pts["player_id"].map(_norm_id)
    pts["season"] = pts["season"].astype(str)
    po = {(r.player_id, r.season): float(r.pts_o_v40) for r in pts.itertuples()}
    pd_ = {(r.player_id, r.season): float(r.pts_d_v40) for r in pts.itertuples()}
    return po, pd_


def build_team_players(lineups: pd.DataFrame, attrs: Dict, pts_o: Dict, pts_d: Dict) -> Dict[Tuple[str, str], List[dict]]:
    out: Dict[Tuple[str, str], List[dict]] = {}
    for r in lineups.itertuples():
        season = str(r.season)
        team = str(r.team_abbreviation).upper()
        sids = _parse_id_list(getattr(r, "starter_player_ids", None))
        players = []
        for pid in sids:
            a = attrs.get((pid, season), {})
            band = a.get("band")
            if band not in BAND_INDEX:
                band = "wings"
            proxy = a.get("proxy")
            if not proxy or proxy in ("nan", "None"):
                proxy = PROXY_OF_BAND[band]
            players.append({
                "player_id": pid,
                "band": band,
                "proxy": str(proxy),
                "off_arch": a.get("off_arch"),
                "def_arch": a.get("def_arch"),
                "minutes": a.get("mpg", 0.0) or 0.0,
                "pts_o": pts_o.get((pid, season), 0.0),
                "pts_d": pts_d.get((pid, season), 0.0),
            })
        if players:
            out[(season, team)] = players
    return out


def _pick(players, pred, key, default_pool=None):
    cands = [p for p in players if pred(p)]
    if not cands and default_pool is not None:
        cands = [p for p in players if default_pool(p)]
    if not cands:
        return None
    return max(cands, key=key)


def assign_matchups(off_players: List[dict], def_players: List[dict], mtx: Matrix) -> dict:
    off = list(off_players)
    deff = list(def_players)
    used_off, used_def = set(), set()
    pairs = []
    player_adjs: Dict[str, float] = {}
    mismatch_flags: Dict[str, int] = {}

    # ---- Pair 1: primary perimeter threat ----
    o1 = _pick(off, lambda p: p["band"] in ("smalls", "wings") and str(p["off_arch"]) in CREATOR_ARCHS,
               lambda p: p["pts_o"], default_pool=lambda p: p["band"] in ("smalls", "wings")) \
        or _pick(off, lambda p: True, lambda p: p["pts_o"])
    d1 = _pick(deff, lambda p: p["band"] in ("smalls", "wings") and str(p["def_arch"]) in POA_ARCHS,
               lambda p: p["pts_d"], default_pool=lambda p: p["band"] in ("smalls", "wings")) \
        or _pick(deff, lambda p: True, lambda p: p["pts_d"])
    ppp1, src1 = mtx.lookup(o1["off_arch"], d1["def_arch"], o1["band"]) if (o1 and d1) else (0.0, "miss")
    if o1:
        used_off.add(o1["player_id"]); player_adjs[o1["player_id"]] = ppp1
    if d1:
        used_def.add(d1["player_id"])
    pairs.append({"pair": 1, "off_id": o1["player_id"] if o1 else None,
                  "def_arch": d1["def_arch"] if d1 else None, "ppp": ppp1, "source": src1})

    # ---- Pair 2: interior threat + big behavior ----
    rem_off = [p for p in off if p["player_id"] not in used_off]
    rem_def = [p for p in deff if p["player_id"] not in used_def]
    o2 = _pick(rem_off, lambda p: str(p["off_arch"]) in INTERIOR_OFF_ARCHS,
               lambda p: p["pts_o"], default_pool=lambda p: p["band"] == "bigs") \
        or _pick(rem_off, lambda p: True, lambda p: p["pts_o"])

    rim_anchor = _pick(rem_def, lambda p: p["band"] == "bigs" and str(p["def_arch"]) in RIM_ANCHOR_ARCHS,
                       lambda p: p["pts_d"]) or _pick(rem_def, lambda p: p["band"] == "bigs", lambda p: p["pts_d"])
    top3_def_ids = {p["player_id"] for p in sorted(deff, key=lambda p: p["pts_d"], reverse=True)[:3]}
    big_bodied = _pick(rem_def, lambda p: p["band"] == "wings" and p["proxy"] in BIG_BODIED_PROXIES
                       and p["pts_d"] > 0.0 and p["player_id"] in top3_def_ids, lambda p: p["pts_d"])

    if big_bodied is not None:
        sub_case, center_overloaded, d2 = "A", False, big_bodied
    elif rim_anchor is not None:
        sub_case, center_overloaded, d2 = "B", True, rim_anchor
    else:
        sub_case, center_overloaded = "B", True
        d2 = _pick(rem_def, lambda p: True, lambda p: p["pts_d"])

    ppp2, src2 = mtx.lookup(o2["off_arch"], d2["def_arch"], o2["band"]) if (o2 and d2) else (0.0, "miss")
    rim_anchor_term = 0.0
    if rim_anchor is not None and o2 is not None and sub_case == "A":
        rt, _ = mtx.lookup(o2["off_arch"], rim_anchor["def_arch"], o2["band"])
        rim_anchor_term = 0.5 * rt  # paint-presence weight, not a 1-on-1 assignment
    if o2:
        used_off.add(o2["player_id"]); player_adjs[o2["player_id"]] = ppp2 + rim_anchor_term
    if d2:
        used_def.add(d2["player_id"])
    pairs.append({"pair": 2, "off_id": o2["player_id"] if o2 else None,
                  "def_arch": d2["def_arch"] if d2 else None, "ppp": ppp2,
                  "rim_anchor_term": rim_anchor_term, "sub_case": sub_case,
                  "center_overloaded": center_overloaded, "source": src2})

    # ---- Pairs 3-5: band + talent residual ----
    rem_off = [p for p in off if p["player_id"] not in used_off]
    rem_def = [p for p in deff if p["player_id"] not in used_def]
    def_by: Dict[str, List[dict]] = {}
    for p in sorted(rem_def, key=lambda x: x["pts_d"], reverse=True):
        def_by.setdefault(p["band"], []).append(p)
    p35_sum = 0.0
    pair_n = 3
    for op in sorted(rem_off, key=lambda x: x["pts_o"], reverse=True):
        band = op["band"]
        dp, band_diff, forced = None, 0, False
        if def_by.get(band):
            dp = def_by[band].pop(0)
        else:
            best = None
            for b, lst in def_by.items():
                if lst:
                    diff = abs(BAND_INDEX[band] - BAND_INDEX[b])
                    if best is None or diff < best[0]:
                        best = (diff, b)
            if best is not None:
                dp = def_by[best[1]].pop(0); band_diff = best[0]; forced = True
        if dp is None:
            continue
        ppp, src = mtx.lookup(op["off_arch"], dp["def_arch"], band)
        if forced:
            mismatch = MISMATCH_DISCOUNT.get(band_diff, 0.0)
            mismatch_flags[f"pair{pair_n}"] = band_diff
            src += "+mismatch"
        else:
            mismatch = MISMATCH_DISCOUNT.get(0, 0.0)  # same-band calibration correction
        total = ppp + mismatch
        p35_sum += total
        player_adjs[op["player_id"]] = total
        pairs.append({"pair": pair_n, "off_id": op["player_id"], "def_arch": dp["def_arch"],
                      "ppp": ppp, "mismatch": mismatch, "band_diff": band_diff, "forced": forced, "source": src})
        pair_n += 1

    # ---- lineup net rating adj (minute-weighted, ppp -> pts/100) ----
    tot_min = sum(p["minutes"] for p in off)
    wsum = 0.0
    for p in off:
        adj = player_adjs.get(p["player_id"], 0.0)
        w = (p["minutes"] / tot_min) if tot_min > 0 else (1.0 / max(len(off), 1))
        wsum += adj * w
    lineup_net_rating_adj = wsum * 100.0

    return {
        "pair1_off_player_id": pairs[0]["off_id"],
        "pair1_def_arch": pairs[0]["def_arch"],
        "pair1_interaction_ppp": round(pairs[0]["ppp"], 6),
        "pair1_source": pairs[0]["source"],
        "pair2_off_player_id": pairs[1]["off_id"],
        "pair2_sub_case": pairs[1]["sub_case"],
        "pair2_interaction_ppp": round(pairs[1]["ppp"] + pairs[1]["rim_anchor_term"], 6),
        "pair3_5_interaction_ppp": round(p35_sum, 6),
        "band_mismatch_flags": json.dumps(mismatch_flags),
        "lineup_net_rating_adj": round(lineup_net_rating_adj, 5),
        "player_matchup_adjs": json.dumps({k: round(v, 6) for k, v in player_adjs.items()}),
        "center_overloaded": bool(pairs[1]["center_overloaded"]),
        "pairs_detail": json.dumps(pairs),
        "source": "emergent_matchup;assignment_proxy",
    }


def main() -> int:
    for pth in (PROFILES_LINEUP, MATRIX, PTS_V40):
        if not pth.exists():
            print(f"FATAL missing {pth}")
            return 1
    lineups = pd.read_parquet(PROFILES_LINEUP)
    mtx = Matrix(pd.read_parquet(MATRIX))
    attrs = load_player_attrs()
    pts_o, pts_d = load_pts()
    team_players = build_team_players(lineups, attrs, pts_o, pts_d)

    by_season: Dict[str, List[str]] = {}
    for (season, team) in team_players:
        by_season.setdefault(season, []).append(team)

    rows = []
    for season, teams in by_season.items():
        for ot in teams:
            for dt in teams:
                if ot == dt:
                    continue
                res = assign_matchups(team_players[(season, ot)], team_players[(season, dt)], mtx)
                res.update({"game_id": None, "season": season, "team": ot, "opponent": dt,
                            "granularity": "season_pair"})
                rows.append(res)

    out_df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(OUT, index=False)

    n = len(out_df)
    n_overload = int(out_df["center_overloaded"].sum()) if n else 0
    # coverage diagnostic: share of pair1 cells resolved via argmax (not fallback/miss)
    argmax_rate = float((out_df["pair1_source"] == "argmax").mean()) if n else 0.0
    summary = {
        "artifact": str(OUT.relative_to(ROOT)),
        "granularity": "season_pair (game_id=NULL; matchup source has no game_id)",
        "n_rows": n, "n_seasons": len(by_season),
        "center_overloaded_rate": round(n_overload / max(n, 1), 4),
        "pair1_argmax_rate": round(argmax_rate, 4),
        "mean_lineup_net_rating_adj": round(float(out_df["lineup_net_rating_adj"].mean()), 4) if n else None,
        "std_lineup_net_rating_adj": round(float(out_df["lineup_net_rating_adj"].std()), 4) if n else None,
        "min_nr": round(float(out_df["lineup_net_rating_adj"].min()), 3) if n else None,
        "max_nr": round(float(out_df["lineup_net_rating_adj"].max()), 3) if n else None,
    }
    (ROOT / "reports" / "cross_team_interactions.json").write_text(json.dumps(summary, indent=2))
    print("ADJ_OK rows=%d seas=%d overload=%.3f argmax=%.3f mean_nr=%s std_nr=%s range=[%s,%s]"
          % (n, len(by_season), summary["center_overloaded_rate"], summary["pair1_argmax_rate"],
             summary["mean_lineup_net_rating_adj"], summary["std_lineup_net_rating_adj"],
             summary["min_nr"], summary["max_nr"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
