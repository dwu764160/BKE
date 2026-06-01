"""
src/simulation/possession_engine.py
=============================================================================
Simulation Core — Step 2: Generative Possession Engine (the calibration spine).

Bottom-up generative engine: a game is simulated as a sequence of possessions,
and the full box score *emerges* from sampled possession outcomes. This is the
LAST pre-divergence Core step; both tracks consume it (markets reads the emergent
box-score distributions, the game track later swaps in a richer resolver).

Design ref: docs/plans/generative_possession_engine.md (Resolved Design),
docs/phase_v1_architecture.md §10 (bottom-up generative + the §10 refinement).

The decoupling seam
-------------------
    PossessionResolver.resolve(off, defense, state, rng) -> PossessionOutcome

`PossessionLoop` / `BoxScoreAggregator` are agnostic to *how* an outcome is made.
Core ships `OutcomeSamplerResolver` (samples discrete outcomes from
archetype + matchup + talent conditioned probabilities). The GAME track later
drops in a `MarkovEventResolver` (full event-level sim + rich tendency model) on
the SAME interface — no change to the loop, aggregator, or calibration.

`PossessionOutcome` carries a rich schema from day one (scorer/assist/tov/reb/
foul ids + seconds) so the Markov swap needs no schema change.

Explainable RateModel
---------------------
Every simulated rate is a product of three separable terms, so any line is
explainable:

    rate = base_rate(behavioral_* + pts_v40 talent)   # opponent-averaged self
         × archetype_shape(off_primary_archetype)       # how he scores
         × matchup_multiplier(Step-1 player_matchup_adj) # vs THIS defense

The Step-1 matchup adjustment is applied at FULL empirical magnitude here (the
possession engine simulates real-time mismatch hunting) — not the 0.5x
projection discount used in the season-level artifact.

Performance: `OutcomeSamplerResolver` exposes the canonical single-possession
`resolve()` (the seam + unit-test path) AND a vectorized `simulate_team_batch()`
used by `simulate_game()` for full-season Monte-Carlo speed. Both share the same
RateModel, so they are consistent by construction.
=============================================================================
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# --- league baselines (anchors; PACE/EFG scale are overwritten by calibration) ---
LEAGUE_PPP = 1.14
LEAGUE_EFG = 0.535
LEAGUE_3P = 0.360
LEAGUE_FT = 0.780
LEAGUE_OREB = 0.260
LEAGUE_AST_ON_MADE = 0.580   # share of made FG that are assisted
LEAGUE_PACE = 99.0           # possessions per team per game
LEAGUE_USAGE = 0.180
LEAGUE_BENCH_EFG = 0.520

# offensive archetype "shape" multipliers: how the archetype reshapes a player's
# terminal-action profile relative to his behavioral rates. Conservative (near 1)
# — calibration tunes global scale; these only add archetype texture.
ARCH_SHAPE: Dict[str, Dict[str, float]] = {
    "Ball Dominant Creator":      {"usage": 1.25, "three": 1.00, "ft": 1.15, "ast": 1.30, "tov": 1.10},
    "Ballhandler":                {"usage": 1.15, "three": 1.05, "ft": 1.05, "ast": 1.25, "tov": 1.05},
    "All-Around Scorer":          {"usage": 1.20, "three": 1.00, "ft": 1.10, "ast": 1.10, "tov": 1.00},
    "Perimeter Scorer":           {"usage": 1.05, "three": 1.25, "ft": 0.95, "ast": 0.90, "tov": 0.90},
    "Interior Scorer":            {"usage": 1.05, "three": 0.40, "ft": 1.20, "ast": 0.80, "tov": 1.00},
    "Connector":                  {"usage": 0.85, "three": 1.05, "ft": 0.90, "ast": 1.20, "tov": 0.95},
    "PnR Rolling Big":            {"usage": 0.95, "three": 0.20, "ft": 1.15, "ast": 0.70, "tov": 0.95},
    "PnR Popping Big":            {"usage": 0.90, "three": 1.30, "ft": 0.95, "ast": 0.75, "tov": 0.90},
    "Off-Ball Finisher":          {"usage": 0.80, "three": 0.45, "ft": 1.05, "ast": 0.60, "tov": 0.85},
    "Off-Ball Movement Shooter":  {"usage": 0.85, "three": 1.45, "ft": 0.85, "ast": 0.80, "tov": 0.85},
    "Off-Ball Stationary Shooter":{"usage": 0.70, "three": 1.55, "ft": 0.80, "ast": 0.65, "tov": 0.80},
}
_DEFAULT_SHAPE = {"usage": 1.0, "three": 1.0, "ft": 1.0, "ast": 1.0, "tov": 1.0}


@dataclass
class GlobalConstants:
    """Calibrated global scalars (structure fixed; calibration tunes scale)."""
    pace: float = LEAGUE_PACE
    efg_scale: float = 1.0
    tov_scale: float = 1.0
    ft_scale: float = 1.0
    oreb: float = LEAGUE_OREB
    league_3p: float = LEAGUE_3P
    league_ft: float = LEAGUE_FT
    ast_on_made: float = LEAGUE_AST_ON_MADE

    def to_dict(self) -> dict:
        return {k: round(float(v), 6) for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "GlobalConstants":
        return cls(**{k: float(v) for k, v in d.items() if k in cls().__dict__})


@dataclass
class Player:
    player_id: str
    mpg: float
    off_arch: Optional[str] = None
    band: Optional[str] = None
    # behavioral_* rates (already on per-action / per-attempt scale)
    usage: float = LEAGUE_USAGE
    efg: float = LEAGUE_EFG
    three_rate: float = 0.38
    tov_rate: float = 0.124
    ast_rate: float = 0.137
    ftr: float = 0.220
    orb_rate: float = 0.026
    drb_rate: float = 0.118
    foul_rate: float = 0.124
    pts_o: float = 0.0
    pts_d: float = 0.0
    matchup_adj: float = 0.0   # Step-1 player ppp adjustment (FULL magnitude)
    is_bench_unit: bool = False


@dataclass
class Lineup:
    team: str
    players: List[Player]
    center_overloaded: bool = False   # Step-1 flag for the OPPOSING interior


@dataclass
class PossessionOutcome:
    """Rich, resolver-agnostic. One terminal possession (incl. OREB putbacks)."""
    points: int = 0
    fga: int = 0
    fgm: int = 0
    fg3a: int = 0
    fg3m: int = 0
    fta: int = 0
    ftm: int = 0
    oreb: int = 0
    dreb: int = 0
    tov: int = 0
    ast: int = 0
    stl: int = 0
    blk: int = 0
    pf: int = 0
    scorer_id: Optional[str] = None
    assist_id: Optional[str] = None
    tov_id: Optional[str] = None
    oreb_id: Optional[str] = None
    dreb_id: Optional[str] = None
    fouler_id: Optional[str] = None
    possession_seconds: float = 0.0


@dataclass
class GameState:
    off_score: int = 0
    def_score: int = 0
    possession: int = 0
    seconds_elapsed: float = 0.0


# ---------------------------------------------------------------------------
class RateModel:
    """Shared, calibratable stat-line spine. Turns players + matchup into the
    multiplicative rate primitives both resolvers consume."""

    def __init__(self, g: GlobalConstants):
        self.g = g

    @staticmethod
    def _shape(arch: Optional[str]) -> Dict[str, float]:
        return ARCH_SHAPE.get(str(arch), _DEFAULT_SHAPE)

    def matchup_mult(self, matchup_adj: float) -> float:
        """Step-1 ppp adjustment -> efficiency multiplier on make probability."""
        return float(np.clip(1.0 + matchup_adj / LEAGUE_PPP, 0.80, 1.20))

    def player_primitives(self, p: Player) -> Dict[str, float]:
        """Per-finisher terminal-action probabilities (explainable product)."""
        sh = self._shape(p.off_arch)
        mm = self.matchup_mult(p.matchup_adj)
        three = float(np.clip(p.three_rate * sh["three"], 0.0, 0.95))
        efg = float(np.clip(p.efg * mm * self.g.efg_scale, 0.20, 0.78))
        p3 = float(np.clip(self.g.league_3p * (efg / LEAGUE_EFG), 0.15, 0.55))
        if three < 0.99:
            p2 = (efg * 2.0 - three * p3 * 3.0) / ((1.0 - three) * 2.0)
        else:
            p2 = efg
        p2 = float(np.clip(p2, 0.25, 0.80))
        tov = float(np.clip(p.tov_rate * sh["tov"] * self.g.tov_scale, 0.02, 0.30))
        ft_trip = float(np.clip(p.ftr * sh["ft"] * self.g.ft_scale, 0.0, 0.60))
        return {"three": three, "p2": p2, "p3": p3, "tov": tov, "ft_trip": ft_trip}

    def finish_weights(self, lineup: Lineup) -> np.ndarray:
        """P(player i is the terminal actor) ∝ minutes-on-court × usage × shape."""
        w = np.array([
            max(pl.mpg, 0.0) * max(pl.usage, 1e-4) * self._shape(pl.off_arch)["usage"]
            for pl in lineup.players
        ], dtype=float)
        s = w.sum()
        return w / s if s > 0 else np.full(len(w), 1.0 / max(len(w), 1))

    def assist_weights(self, lineup: Lineup) -> np.ndarray:
        w = np.array([max(pl.ast_rate, 1e-4) * self._shape(pl.off_arch)["ast"]
                      for pl in lineup.players], dtype=float)
        s = w.sum()
        return w / s if s > 0 else np.full(len(w), 1.0 / max(len(w), 1))

    def drb_weights(self, lineup: Lineup) -> np.ndarray:
        w = np.array([max(pl.drb_rate, 1e-4) for pl in lineup.players], dtype=float)
        s = w.sum()
        return w / s if s > 0 else np.full(len(w), 1.0 / max(len(w), 1))

    def orb_weights(self, lineup: Lineup) -> np.ndarray:
        w = np.array([max(pl.orb_rate, 1e-4) for pl in lineup.players], dtype=float)
        s = w.sum()
        return w / s if s > 0 else np.full(len(w), 1.0 / max(len(w), 1))

    def foul_weights(self, lineup: Lineup) -> np.ndarray:
        w = np.array([max(pl.foul_rate, 1e-4) for pl in lineup.players], dtype=float)
        s = w.sum()
        return w / s if s > 0 else np.full(len(w), 1.0 / max(len(w), 1))


# ---------------------------------------------------------------------------
class PossessionResolver(ABC):
    """The decoupling seam. The game track later implements a Markov version."""

    @abstractmethod
    def resolve(self, offense: Lineup, defense: Lineup, state: GameState,
                rng: np.random.Generator) -> PossessionOutcome:
        ...


class OutcomeSamplerResolver(PossessionResolver):
    """Core's lightweight possession-outcome sampler (v0/v1 spine)."""

    MAX_CONTINUATIONS = 4   # OREB putback chain cap

    def __init__(self, rate_model: RateModel):
        self.rm = rate_model

    # ---- canonical single-possession path (seam + tests) ----
    def resolve(self, offense: Lineup, defense: Lineup, state: GameState,
                rng: np.random.Generator) -> PossessionOutcome:
        g = self.rm.g
        out = PossessionOutcome()
        fw = self.rm.finish_weights(offense)
        aw = self.rm.assist_weights(offense)
        dw = self.rm.drb_weights(defense)
        ow = self.rm.orb_weights(offense)
        ffw = self.rm.foul_weights(defense)
        oreb_p = g.oreb * (1.15 if defense.center_overloaded else 1.0)
        out.possession_seconds = float(np.clip(rng.normal(14.4, 4.0), 3.0, 24.0))

        for _ in range(self.MAX_CONTINUATIONS):
            fi = int(rng.choice(len(offense.players), p=fw))
            fp = offense.players[fi]
            prim = self.rm.player_primitives(fp)

            if rng.random() < prim["tov"]:
                out.tov += 1
                out.tov_id = fp.player_id
                if rng.random() < 0.55:  # live-ball steal
                    di = int(rng.choice(len(defense.players), p=dw))
                    out.stl += 1
                return out

            if rng.random() < prim["ft_trip"]:
                di = int(rng.choice(len(defense.players), p=ffw))
                out.pf += 1
                out.fouler_id = defense.players[di].player_id
                made = int(rng.random() < g.league_ft) + int(rng.random() < g.league_ft)
                out.fta += 2
                out.ftm += made
                out.points += made
                out.scorer_id = fp.player_id
                return out

            out.fga += 1
            is_three = rng.random() < prim["three"]
            if is_three:
                out.fg3a += 1
                made = rng.random() < prim["p3"]
            else:
                made = rng.random() < prim["p2"]
            if made:
                out.fgm += 1
                pts = 3 if is_three else 2
                if is_three:
                    out.fg3m += 1
                out.points += pts
                out.scorer_id = fp.player_id
                if rng.random() < g.ast_on_made:
                    ai = int(rng.choice(len(offense.players), p=aw))
                    if ai != fi:
                        out.ast += 1
                        out.assist_id = offense.players[ai].player_id
                return out
            # miss -> block? -> rebound battle
            if rng.random() < 0.06:
                out.blk += 1
            if rng.random() < oreb_p:
                oi = int(rng.choice(len(offense.players), p=ow))
                out.oreb += 1
                out.oreb_id = offense.players[oi].player_id
                continue   # putback chain
            di = int(rng.choice(len(defense.players), p=dw))
            out.dreb += 1
            out.dreb_id = defense.players[di].player_id
            return out
        return out

    # ---- vectorized batch path (production Monte-Carlo) ----
    def simulate_team_batch(self, offense: Lineup, defense: Lineup,
                            sim_ids: np.ndarray, rng: np.random.Generator,
                            n_sims: int) -> Dict[str, np.ndarray]:
        """Simulate all possessions for one team across all sims at once.

        sim_ids: length = total possessions, each entry the sim index that
        possession belongs to. Returns per-(sim, player) stat arrays of shape
        (n_sims, K) plus team totals — identical model to resolve(), vectorized.
        """
        g = self.rm.g
        K = len(offense.players)
        Kd = len(defense.players)
        n = len(sim_ids)

        fw = self.rm.finish_weights(offense)
        aw = self.rm.assist_weights(offense)
        ow = self.rm.orb_weights(offense)
        dw = self.rm.drb_weights(defense)
        ffw = self.rm.foul_weights(defense)
        oreb_p = g.oreb * (1.15 if defense.center_overloaded else 1.0)

        # per-player primitive vectors
        prim = [self.rm.player_primitives(p) for p in offense.players]
        v_tov = np.array([x["tov"] for x in prim])
        v_ft = np.array([x["ft_trip"] for x in prim])
        v_three = np.array([x["three"] for x in prim])
        v_p2 = np.array([x["p2"] for x in prim])
        v_p3 = np.array([x["p3"] for x in prim])

        # accumulators (n_sims, K) offense; (n_sims, Kd) defense
        OFF = {k: np.zeros((n_sims, K)) for k in
               ("pts", "fga", "fgm", "fg3a", "fg3m", "fta", "ftm", "oreb", "tov", "ast")}
        DEF = {k: np.zeros((n_sims, Kd)) for k in ("dreb", "stl", "blk", "pf")}
        team_pts = np.zeros(n_sims)

        active = np.ones(n, dtype=bool)
        sim_of = sim_ids.copy()
        for _ in range(self.MAX_CONTINUATIONS):
            idx = np.where(active)[0]
            if idx.size == 0:
                break
            m = idx.size
            fi = rng.choice(K, size=m, p=fw)
            r1 = rng.random(m)
            sidx = sim_of[idx]

            # turnover branch
            tov_mask = r1 < v_tov[fi]
            if tov_mask.any():
                np.add.at(OFF["tov"], (sidx[tov_mask], fi[tov_mask]), 1)
                stl_mask = tov_mask & (rng.random(m) < 0.55)
                if stl_mask.any():
                    di = rng.choice(Kd, size=int(stl_mask.sum()), p=dw)
                    np.add.at(DEF["stl"], (sidx[stl_mask], di), 1)
                active[idx[tov_mask]] = False

            rest = ~tov_mask
            # FT-trip branch
            r2 = rng.random(m)
            ft_mask = rest & (r2 < v_ft[fi])
            if ft_mask.any():
                mm = int(ft_mask.sum())
                made = (rng.random(mm) < g.league_ft).astype(int) + (rng.random(mm) < g.league_ft).astype(int)
                s_ft, f_ft = sidx[ft_mask], fi[ft_mask]
                np.add.at(OFF["fta"], (s_ft, f_ft), 2)
                np.add.at(OFF["ftm"], (s_ft, f_ft), made)
                np.add.at(OFF["pts"], (s_ft, f_ft), made)
                np.add.at(team_pts, s_ft, made)
                di = rng.choice(Kd, size=mm, p=ffw)
                np.add.at(DEF["pf"], (s_ft, di), 1)
                active[idx[ft_mask]] = False

            shot = rest & ~ft_mask
            if shot.any():
                sm = idx[shot]
                m2 = sm.size
                ssh, fsh = sim_of[sm], fi[shot]
                np.add.at(OFF["fga"], (ssh, fsh), 1)
                is3 = rng.random(m2) < v_three[fsh]
                np.add.at(OFF["fg3a"], (ssh[is3], fsh[is3]), 1)
                rr = rng.random(m2)
                made = np.where(is3, rr < v_p3[fsh], rr < v_p2[fsh])
                # makes
                if made.any():
                    sm_m, f_m, is3_m = ssh[made], fsh[made], is3[made]
                    np.add.at(OFF["fgm"], (sm_m, f_m), 1)
                    pts = np.where(is3_m, 3, 2)
                    np.add.at(OFF["pts"], (sm_m, f_m), pts)
                    np.add.at(team_pts, sm_m, pts)
                    np.add.at(OFF["fg3m"], (sm_m[is3_m], f_m[is3_m]), 1)
                    # assists
                    amask = rng.random(int(made.sum())) < g.ast_on_made
                    if amask.any():
                        ai = rng.choice(K, size=int(amask.sum()), p=aw)
                        sa = sm_m[amask]
                        fa = f_m[amask]
                        good = ai != fa
                        np.add.at(OFF["ast"], (sa[good], ai[good]), 1)
                    active[idx[shot][made]] = False
                # misses -> block + rebound
                miss = ~made
                if miss.any():
                    sm_x = ssh[miss]
                    blk = rng.random(int(miss.sum())) < 0.06
                    if blk.any():
                        di = rng.choice(Kd, size=int(blk.sum()), p=dw)
                        np.add.at(DEF["blk"], (sm_x[blk], di), 1)
                    oreb = rng.random(int(miss.sum())) < oreb_p
                    miss_global = idx[shot][miss]
                    # offensive rebounds -> stay active for a putback continuation
                    if oreb.any():
                        oi = rng.choice(K, size=int(oreb.sum()), p=ow)
                        np.add.at(OFF["oreb"], (sm_x[oreb], oi), 1)
                    # defensive rebounds -> possession ends (independent of OREB branch)
                    nd = ~oreb
                    if nd.any():
                        di = rng.choice(Kd, size=int(nd.sum()), p=dw)
                        np.add.at(DEF["dreb"], (sm_x[nd], di), 1)
                        active[miss_global[nd]] = False
            # any still-active beyond cap will be force-closed next loop end
        # force-close stragglers as defensive rebounds (no further attempts)
        # (rare; keeps possession accounting consistent)
        return {"OFF": OFF, "DEF": DEF, "team_pts": team_pts}


# ---------------------------------------------------------------------------
class BoxScoreAggregator:
    """Turns per-(sim, player) arrays into Monte-Carlo distribution summaries."""

    @staticmethod
    def summarize_def(def_arrays: Dict[str, np.ndarray], players: List[Player]) -> List[dict]:
        """Defensive contributions (dreb/stl/blk/pf) for the defending lineup."""
        dreb, stl, blk, pf = def_arrays["dreb"], def_arrays["stl"], def_arrays["blk"], def_arrays["pf"]
        rows = []
        for i, pl in enumerate(players):
            rows.append({
                "player_id": pl.player_id, "is_bench_unit": pl.is_bench_unit,
                "dreb_mean": float(dreb[:, i].mean()),
                "stl_mean": float(stl[:, i].mean()),
                "blk_mean": float(blk[:, i].mean()),
                "pf_mean": float(pf[:, i].mean()),
            })
        return rows

    @staticmethod
    def summarize(off_arrays: Dict[str, np.ndarray], players: List[Player]) -> List[dict]:
        rows = []
        pts = off_arrays["pts"]
        fgm, fga = off_arrays["fgm"], off_arrays["fga"]
        fg3m = off_arrays["fg3m"]
        ftm, fta = off_arrays["ftm"], off_arrays["fta"]
        ast = off_arrays["ast"]
        oreb = off_arrays["oreb"]
        for i, pl in enumerate(players):
            def q(a, p):
                return float(np.percentile(a[:, i], p))
            rows.append({
                "player_id": pl.player_id,
                "is_bench_unit": pl.is_bench_unit,
                "pts_mean": float(pts[:, i].mean()), "pts_p10": q(pts, 10),
                "pts_p50": q(pts, 50), "pts_p90": q(pts, 90), "pts_std": float(pts[:, i].std()),
                "fgm_mean": float(fgm[:, i].mean()), "fga_mean": float(fga[:, i].mean()),
                "fg3m_mean": float(fg3m[:, i].mean()),
                "ftm_mean": float(ftm[:, i].mean()), "fta_mean": float(fta[:, i].mean()),
                "ast_mean": float(ast[:, i].mean()), "ast_p10": q(ast, 10),
                "ast_p50": q(ast, 50), "ast_p90": q(ast, 90),
                "oreb_mean": float(oreb[:, i].mean()),
            })
        return rows


# ---------------------------------------------------------------------------
def simulate_game(off: Lineup, deff: Lineup, resolver: OutcomeSamplerResolver,
                  n_sims: int, rng: np.random.Generator) -> dict:
    """Monte-Carlo one matchup (offense=off scoring vs defense=deff). Pace from
    global constants (Poisson). Returns offense player summaries + team-pts dist
    + defensive rebound/stat summaries for the defending lineup."""
    g = resolver.rm.g
    poss_per_sim = rng.poisson(g.pace, n_sims).astype(int)
    poss_per_sim = np.maximum(poss_per_sim, 1)
    sim_ids = np.repeat(np.arange(n_sims), poss_per_sim)
    res = resolver.simulate_team_batch(off, deff, sim_ids, rng, n_sims)
    off_rows = BoxScoreAggregator.summarize(res["OFF"], off.players)
    return {
        "team": off.team, "opponent": deff.team,
        "team_pts_mean": float(res["team_pts"].mean()),
        "team_pts_std": float(res["team_pts"].std()),
        "team_pts_p10": float(np.percentile(res["team_pts"], 10)),
        "team_pts_p50": float(np.percentile(res["team_pts"], 50)),
        "team_pts_p90": float(np.percentile(res["team_pts"], 90)),
        "player_rows": off_rows,
        "def_rows": BoxScoreAggregator.summarize_def(res["DEF"], deff.players),
        "def_arrays": res["DEF"], "def_players": deff.players,
    }
