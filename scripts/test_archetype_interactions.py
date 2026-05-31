"""
scripts/test_archetype_interactions.py
=============================================================================
Step 1 PRE-FLIGHT — Does cross-team attacker-vs-defender archetype matchup carry
REAL signal, beyond player talent? Measure it before building any model.

Source of truth: data/matchup/league_season_matchups.parquet — NBA defensive
matchup data (LeagueSeasonMatchups): per (offensive player x defensive player)
PARTIAL_POSS, PLAYER_PTS, FG, etc. for 2022-23..2024-25.

The trap (docs/plans/cross_team_interaction_model.md): a creator guarded by a POA
defender being efficient/inefficient could just be two good/bad players. PTS/RAPM
already carry each player's OPPONENT-AVERAGED talent. The interaction model must
add ONLY the matchup-dependent delta. So we identify the effect with TWO-WAY
PLAYER FIXED EFFECTS:

    ppp_ij = a_i (off-player FE) + b_j (def-player FE) + e_ij

We remove a_i and b_j by weighted iterative demeaning, then ask whether the
RESIDUAL e_ij still varies by (off_archetype x def_archetype). Any structure that
survives is pure matchup interaction, purged of both players' average levels —
exactly the mean-zero-over-defenders delta the design calls for.

Significance: cluster bootstrap over offensive players (the dominant correlation
unit), 2000 resamples -> 95% CI per cell. A cell is REAL if its CI excludes 0.
We also report the NAIVE (talent-confounded) cell deltas so the talent washout is
visible, and a global test (how many cells survive vs ~5% expected by chance, +
Benjamini-Hochberg FDR).

Output:
  reports/archetype_interaction_signal_test.json
Usage:
  python3 scripts/test_archetype_interactions.py
=============================================================================
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MATCHUPS = ROOT / "data/matchup/league_season_matchups.parquet"
PROFILES = ROOT / "data/processed/player_eval/player_impact_profiles.parquet"
REPORT = ROOT / "reports/archetype_interaction_signal_test.json"

MIN_POSS = 2.0          # drop ultra-thin matchup rows (ppp undefined/noisy)
DEMEAN_ITERS = 12       # alternating-projection passes for two-way FE
N_BOOT = 2000
DROP_ARCH = {"Insufficient Minutes", "Unknown", None}


def _norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()


def load() -> pd.DataFrame:
    m = pd.read_parquet(
        MATCHUPS,
        columns=["OFF_PLAYER_ID", "DEF_PLAYER_ID", "PARTIAL_POSS", "PLAYER_PTS",
                 "MATCHUP_FGM", "MATCHUP_FGA", "MATCHUP_FG3M", "SEASON"],
    )
    m["OFF_PLAYER_ID"] = _norm(m["OFF_PLAYER_ID"])
    m["DEF_PLAYER_ID"] = _norm(m["DEF_PLAYER_ID"])
    m["SEASON"] = m["SEASON"].astype(str)
    m = m[m["PARTIAL_POSS"] >= MIN_POSS].copy()
    m["ppp"] = m["PLAYER_PTS"] / m["PARTIAL_POSS"]

    prof = pd.read_parquet(
        PROFILES, columns=["player_id", "season", "off_primary_archetype", "def_primary_archetype"]
    )
    prof["player_id"] = _norm(prof["player_id"])
    prof["season"] = prof["season"].astype(str)

    off = prof[["player_id", "season", "off_primary_archetype"]].rename(
        columns={"player_id": "OFF_PLAYER_ID", "season": "SEASON", "off_primary_archetype": "off_arch"}
    )
    deff = prof[["player_id", "season", "def_primary_archetype"]].rename(
        columns={"player_id": "DEF_PLAYER_ID", "season": "SEASON", "def_primary_archetype": "def_arch"}
    )
    m = m.merge(off, on=["OFF_PLAYER_ID", "SEASON"], how="left")
    m = m.merge(deff, on=["DEF_PLAYER_ID", "SEASON"], how="left")
    m = m[~m["off_arch"].isin(DROP_ARCH) & ~m["def_arch"].isin(DROP_ARCH)]
    m = m.dropna(subset=["off_arch", "def_arch"])
    return m.reset_index(drop=True)


def two_way_demean(df: pd.DataFrame, val: str, w: str, g1: str, g2: str, iters: int) -> np.ndarray:
    """Weighted alternating-projection removal of g1 and g2 fixed effects."""
    r = df[val].to_numpy(dtype=float).copy()
    weight = df[w].to_numpy(dtype=float)
    a = df[g1].to_numpy()
    b = df[g2].to_numpy()
    # precompute group index arrays
    for grp in (a, b):
        pass
    ia = pd.factorize(a)[0]
    ib = pd.factorize(b)[0]
    na, nb = ia.max() + 1, ib.max() + 1
    for _ in range(iters):
        # remove weighted mean within g1
        sw = np.bincount(ia, weights=weight, minlength=na)
        swr = np.bincount(ia, weights=weight * r, minlength=na)
        r = r - (swr / np.maximum(sw, 1e-12))[ia]
        # remove weighted mean within g2
        sw = np.bincount(ib, weights=weight, minlength=nb)
        swr = np.bincount(ib, weights=weight * r, minlength=nb)
        r = r - (swr / np.maximum(sw, 1e-12))[ib]
    return r


def cell_means(df: pd.DataFrame, val_col: str) -> pd.DataFrame:
    g = df.groupby(["off_arch", "def_arch"])
    num = g.apply(lambda x: np.average(x[val_col], weights=x["PARTIAL_POSS"]))
    poss = g["PARTIAL_POSS"].sum()
    n = g.size()
    out = pd.DataFrame({"value": num, "poss": poss, "n_pairs": n}).reset_index()
    return out


def cluster_bootstrap(df: pd.DataFrame, val_col: str, n_boot: int) -> pd.DataFrame:
    """Resample offensive players -> 95% CI per (off_arch,def_arch) cell."""
    # Pre-aggregate each off_player's contribution per cell: sum(w*resid), sum(w)
    df = df.copy()
    df["wv"] = df["PARTIAL_POSS"] * df[val_col]
    agg = (
        df.groupby(["OFF_PLAYER_ID", "off_arch", "def_arch"])
        .agg(wv=("wv", "sum"), w=("PARTIAL_POSS", "sum"))
        .reset_index()
    )
    cells = sorted(set(zip(agg["off_arch"], agg["def_arch"])))
    cell_idx = {c: i for i, c in enumerate(cells)}
    agg["cidx"] = list(zip(agg["off_arch"], agg["def_arch"]))
    agg["cidx"] = agg["cidx"].map(cell_idx)

    players = agg["OFF_PLAYER_ID"].unique()
    p_to_rows = {p: g for p, g in agg.groupby("OFF_PLAYER_ID")}
    # Stack per-player contributions for fast resampled summation
    pl_wv = {}
    pl_w = {}
    for p, g in p_to_rows.items():
        wv = np.zeros(len(cells)); w = np.zeros(len(cells))
        wv[g["cidx"].to_numpy()] = g["wv"].to_numpy()
        w[g["cidx"].to_numpy()] = g["w"].to_numpy()
        pl_wv[p] = wv; pl_w[p] = w
    P = np.array(list(players))
    WV = np.array([pl_wv[p] for p in P])   # (n_players, n_cells)
    W = np.array([pl_w[p] for p in P])

    rng = np.random.default_rng(42)
    n_pl = len(P)
    boot = np.empty((n_boot, len(cells)))
    for b in range(n_boot):
        idx = rng.integers(0, n_pl, n_pl)
        swv = WV[idx].sum(axis=0)
        sw = W[idx].sum(axis=0)
        boot[b] = swv / np.maximum(sw, 1e-12)
    lo = np.percentile(boot, 2.5, axis=0)
    hi = np.percentile(boot, 97.5, axis=0)
    point = (WV.sum(axis=0) / np.maximum(W.sum(axis=0), 1e-12))
    # two-sided bootstrap p (fraction crossing 0)
    pval = 2 * np.minimum((boot > 0).mean(axis=0), (boot < 0).mean(axis=0))
    rows = []
    for c, i in cell_idx.items():
        rows.append({"off_arch": c[0], "def_arch": c[1], "interaction_ppp": float(point[i]),
                     "ci_lo": float(lo[i]), "ci_hi": float(hi[i]), "p": float(pval[i])})
    return pd.DataFrame(rows)


def bh_fdr(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    n = len(pvals)
    order = np.argsort(pvals)
    thresh = alpha * (np.arange(1, n + 1) / n)
    passed = pvals[order] <= thresh
    if not passed.any():
        return np.zeros(n, dtype=bool)
    kmax = np.where(passed)[0].max()
    cut = pvals[order][kmax]
    return pvals <= cut


def main() -> None:
    df = load()
    print(f"Matchup rows after filter/join: {len(df)} | seasons {sorted(df['SEASON'].unique())}")
    print(f"Offensive players: {df['OFF_PLAYER_ID'].nunique()} | defenders: {df['DEF_PLAYER_ID'].nunique()}")
    print(f"Total partial possessions: {df['PARTIAL_POSS'].sum():,.0f}")

    # ---- NAIVE (talent-confounded): cell ppp minus offensive-archetype mean ----
    naive_cells = cell_means(df, "ppp")
    off_mean = (
        df.groupby("off_arch").apply(lambda x: np.average(x["ppp"], weights=x["PARTIAL_POSS"]))
    ).to_dict()
    naive_cells["naive_delta"] = naive_cells.apply(
        lambda r: r["value"] - off_mean[r["off_arch"]], axis=1
    )

    # ---- FE-ADJUSTED: two-way demean ppp by off & def player, then residual cells ----
    df["resid"] = two_way_demean(df, "ppp", "PARTIAL_POSS", "OFF_PLAYER_ID", "DEF_PLAYER_ID", DEMEAN_ITERS)
    print(f"Residual weighted-mean (should be ~0): {np.average(df['resid'], weights=df['PARTIAL_POSS']):.4e}")
    print(f"ppp weighted std: {np.sqrt(np.average((df['ppp']-np.average(df['ppp'],weights=df['PARTIAL_POSS']))**2, weights=df['PARTIAL_POSS'])):.3f} "
          f"| residual weighted std: {np.sqrt(np.average(df['resid']**2, weights=df['PARTIAL_POSS'])):.3f}")

    boot = cluster_bootstrap(df, "resid", N_BOOT)
    boot = boot.merge(naive_cells[["off_arch", "def_arch", "naive_delta", "poss", "n_pairs"]],
                      on=["off_arch", "def_arch"], how="left")
    boot["sig_95"] = (boot["ci_lo"] > 0) | (boot["ci_hi"] < 0)
    boot["fdr_pass"] = bh_fdr(boot["p"].to_numpy(), 0.05)
    boot = boot.sort_values("interaction_ppp")

    n_cells = len(boot)
    n_sig = int(boot["sig_95"].sum())
    n_fdr = int(boot["fdr_pass"].sum())
    expected_fp = round(0.05 * n_cells, 1)

    # variance of FE-adjusted interaction across cells, possession-weighted
    interaction_std = float(np.sqrt(np.average(boot["interaction_ppp"]**2, weights=boot["poss"].fillna(0)+1)))
    naive_std = float(np.sqrt(np.average(boot["naive_delta"].fillna(0)**2, weights=boot["poss"].fillna(0)+1)))

    print(f"\nCells: {n_cells} | significant @95% CI: {n_sig} (expected by chance ~{expected_fp}) | BH-FDR pass: {n_fdr}")
    print(f"Interaction spread (poss-wtd std): FE-adjusted {interaction_std:.4f} ppp vs naive {naive_std:.4f} ppp")
    print(f"\nStrongest SUPPRESSION (def archetype lowers off efficiency), FE-adjusted:")
    for _, r in boot[boot["sig_95"]].head(8).iterrows():
        print(f"  {r['off_arch']:28s} vs {r['def_arch']:20s} {r['interaction_ppp']:+.3f} "
              f"[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] naive {r['naive_delta']:+.3f} (poss {r['poss']:,.0f})")
    print(f"\nStrongest BOOST (off thrives vs def archetype), FE-adjusted:")
    for _, r in boot[boot["sig_95"]].tail(8).iloc[::-1].iterrows():
        print(f"  {r['off_arch']:28s} vs {r['def_arch']:20s} {r['interaction_ppp']:+.3f} "
              f"[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] naive {r['naive_delta']:+.3f} (poss {r['poss']:,.0f})")

    verdict = (
        "SIGNAL" if (n_fdr >= 3 and interaction_std >= 0.02 and n_sig > 2 * expected_fp)
        else "WEAK" if (n_sig > expected_fp and n_fdr >= 1)
        else "NULL"
    )
    report = {
        "test": "cross-team attacker-vs-defender archetype interaction (two-way player FE)",
        "source": "data/matchup/league_season_matchups.parquet (2017-18..2024-25, 8 seasons)",
        "method": {
            "metric": "PLAYER_PTS / PARTIAL_POSS (points per partial possession)",
            "fixed_effects": "weighted two-way demean by OFF_PLAYER_ID and DEF_PLAYER_ID",
            "demean_iters": DEMEAN_ITERS,
            "min_partial_poss": MIN_POSS,
            "significance": f"cluster bootstrap over offensive players, n={N_BOOT}, 95% CI + BH-FDR",
        },
        "n_matchup_rows": int(len(df)),
        "n_off_players": int(df["OFF_PLAYER_ID"].nunique()),
        "n_def_players": int(df["DEF_PLAYER_ID"].nunique()),
        "n_cells": n_cells,
        "n_significant_95": n_sig,
        "expected_false_positives": expected_fp,
        "n_fdr_pass": n_fdr,
        "interaction_std_fe_adjusted_ppp": round(interaction_std, 4),
        "interaction_std_naive_ppp": round(naive_std, 4),
        "talent_washout_ratio": round(interaction_std / naive_std, 3) if naive_std else None,
        "verdict": verdict,
        "cells": boot.round(4).to_dict(orient="records"),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nVERDICT: {verdict}  (FE-adjusted spread {interaction_std:.4f} vs naive {naive_std:.4f} ppp)")
    print(f"Saved: {REPORT}")


if __name__ == "__main__":
    main()
