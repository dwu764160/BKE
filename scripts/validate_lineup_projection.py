"""
scripts/validate_lineup_projection.py
=============================================================================
Step 0a — Lineup Projection Tuning (offline, game-by-game).

Validates and tunes the season-level lineup projector
(`src/simulation/lineup_projection.py`) GAME-BY-GAME against ACTUAL lineups
derived offline from `data/historical/pbp_with_lineups_*.parquet` (all seasons;
on-court 5 per team per event via `lineup_<team_id>` columns).

Ground truth (per game, per team), derived from pbp only:
  - starters : the 5 players on court at the first valid 5-man event of period 1
               (opening-tip lineup).
  - clutch-5 : the most-used 5-man lineup during the clutch window
               (period >= 4, |margin| <= 5, game clock <= 5:00).
  - minutes  : per-player seconds on court, from lineup-stint durations
               (clock deltas between consecutive events).

Per-game metrics (projected season lineup vs each actual game):
  - starter_hit   : |proj_starter_5 ∩ actual_game_starter_5| / 5
  - clutch_overlap: |proj_clutch_5  ∩ actual_game_clutch_5| (0-5)
  - minutes_corr  : Spearman corr(proj MPG, actual game minutes) over the union
  - rotation top-9 set overlap (secondary)

Tuning is WALK-FORWARD: for each eval season T, pick the config that maximizes
the objective on seasons < T, then score it on T. The tuned config must beat the
baseline (current `simulation_config` defaults) on starter hit-rate AND minutes
corr without regressing clutch overlap.

Output: reports/lineup_projection_validation.json
Cache  : data/processed/simulation/lineup_ground_truth.pkl (gitignored)

Usage:
  python3 scripts/validate_lineup_projection.py [--rebuild-truth] [--seasons 2022-23,2023-24]
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.simulation.lineup_projection import (  # noqa: E402
    _lineup_from_value,
    _load_team_map,
    build_projected_lineup_rows,
)
from src.simulation.simulation_config import HISTORICAL_DIR, LINEUP_SIZE  # noqa: E402

TRUTH_CACHE = ROOT / "data/processed/simulation/lineup_ground_truth.pkl"
REPORT_PATH = ROOT / "reports/lineup_projection_validation.json"

# Clutch window (standard NBA definition).
CLUTCH_PERIOD_MIN = 4
CLUTCH_MARGIN_MAX = 5
CLUTCH_CLOCK_MAX_SEC = 300.0


# ---------------------------------------------------------------------------
# Ground truth from pbp
# ---------------------------------------------------------------------------
def _parse_clock(series: pd.Series) -> pd.Series:
    """'MM:SS' -> seconds remaining in the period."""
    parts = series.astype(str).str.split(":", expand=True)
    mins = pd.to_numeric(parts[0], errors="coerce")
    secs = pd.to_numeric(parts[1], errors="coerce") if parts.shape[1] > 1 else 0.0
    return mins * 60.0 + secs


def _season_from_path(path: Path) -> Optional[str]:
    import re

    m = re.search(r"pbp_with_lineups_(\d{4}-\d{2})\.parquet$", path.name)
    return m.group(1) if m else None


def build_ground_truth(seasons: Optional[Set[str]] = None) -> pd.DataFrame:
    """One pass per season file: starters, clutch-5, per-player minutes per game/team."""
    _, _, team_id_to_abbr = _load_team_map()
    files = sorted(Path(HISTORICAL_DIR).glob("pbp_with_lineups_*.parquet"))

    records: List[Dict] = []
    for path in files:
        season = _season_from_path(path)
        if not season or (seasons and season not in seasons):
            continue

        lineup_cols = [f"lineup_{tid}" for tid in team_id_to_abbr]
        df = pd.read_parquet(path)
        keep = [c for c in ["game_id", "period", "clock", "home_score", "away_score"] if c in df.columns]
        present_lineup_cols = [c for c in lineup_cols if c in df.columns]
        df = df[keep + present_lineup_cols].copy()

        df["period"] = pd.to_numeric(df["period"], errors="coerce")
        df["csec"] = _parse_clock(df["clock"])
        margin = (
            pd.to_numeric(df["home_score"], errors="coerce")
            - pd.to_numeric(df["away_score"], errors="coerce")
        ).abs()
        df["margin"] = margin
        # Stint duration credited to the lineup at the current row (clock counts down).
        df["csec_next"] = df.groupby(["game_id", "period"], sort=False)["csec"].shift(-1)
        df["dt"] = (df["csec"] - df["csec_next"]).clip(lower=0).fillna(0.0)
        df["_row"] = np.arange(len(df))

        for col in present_lineup_cols:
            team_id = col.replace("lineup_", "")
            abbr = team_id_to_abbr.get(team_id)
            if not abbr:
                continue
            parsed = df[col].apply(_lineup_from_value)
            valid = parsed.apply(lambda ids: len(ids) == LINEUP_SIZE)
            if not valid.any():
                continue
            sub = df.loc[valid, ["game_id", "period", "csec", "margin", "dt", "_row"]].copy()
            sub["lineup"] = parsed[valid]

            # ---- starters: first valid 5-man of period 1 (earliest row) ----
            p1 = sub[sub["period"] == 1]
            starters_by_game: Dict[str, List[str]] = {}
            if not p1.empty:
                first_idx = p1.sort_values("_row").groupby("game_id", sort=False).head(1)
                for _, r in first_idx.iterrows():
                    starters_by_game[str(r["game_id"])] = list(r["lineup"])

            # ---- clutch-5: most-used 5-man lineup in the clutch window ----
            clutch_rows = sub[
                (sub["period"] >= CLUTCH_PERIOD_MIN)
                & (sub["margin"] <= CLUTCH_MARGIN_MAX)
                & (sub["csec"] <= CLUTCH_CLOCK_MAX_SEC)
            ]
            clutch_by_game: Dict[str, List[str]] = {}
            if not clutch_rows.empty:
                for gid, gdf in clutch_rows.groupby("game_id", sort=False):
                    # weight each lineup by time on court in the window
                    weights: Counter = Counter()
                    for ids, dt in zip(gdf["lineup"], gdf["dt"]):
                        key = tuple(sorted(ids))
                        weights[key] += float(dt) + 1e-3  # +eps so zero-dt events still count
                    if weights:
                        clutch_by_game[str(gid)] = list(max(weights.items(), key=lambda kv: kv[1])[0])

            # ---- minutes: per-player seconds on court ----
            mins_by_game: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
            expl = sub[["game_id", "dt", "lineup"]].explode("lineup")
            expl = expl[expl["dt"] > 0]
            for gid, pid, dt in zip(expl["game_id"], expl["lineup"], expl["dt"]):
                mins_by_game[str(gid)][str(pid)] += float(dt)

            game_ids = set(starters_by_game) | set(clutch_by_game) | set(mins_by_game)
            for gid in game_ids:
                records.append(
                    {
                        "season": season,
                        "game_id": gid,
                        "team_abbreviation": abbr,
                        "starters": starters_by_game.get(gid),
                        "clutch5": clutch_by_game.get(gid),
                        "minutes": dict(mins_by_game.get(gid, {})),
                    }
                )
        print(f"  [{season}] ground-truth rows so far: {len(records)}")

    return pd.DataFrame(records)


def load_or_build_truth(rebuild: bool, seasons: Optional[Set[str]]) -> pd.DataFrame:
    if TRUTH_CACHE.exists() and not rebuild:
        with open(TRUTH_CACHE, "rb") as fh:
            gt = pickle.load(fh)
        if seasons:
            gt = gt[gt["season"].isin(seasons)]
        return gt
    print("Building ground truth from pbp (one pass per season)...")
    gt = build_ground_truth(seasons)
    TRUTH_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(TRUTH_CACHE, "wb") as fh:
        pickle.dump(gt, fh)
    print(f"Cached ground truth: {TRUTH_CACHE} ({len(gt)} game-team rows)")
    return gt


# ---------------------------------------------------------------------------
# Projection (per config)
# ---------------------------------------------------------------------------
def project(config_overrides: Dict) -> Dict[Tuple[str, str], Dict]:
    rows, *_ = build_projected_lineup_rows(forecast_mode=False, config_overrides=config_overrides)
    out: Dict[Tuple[str, str], Dict] = {}
    for r in rows:
        key = (str(r["season"]), str(r["team_abbreviation"]).upper())
        proj_minutes = {
            str(p["player_id"]): float(p.get("minutes") or 0.0) for p in r.get("pool_players", [])
        }
        out[key] = {
            "starters": set(str(x) for x in r.get("starter_player_ids", [])),
            "clutch5": set(str(x) for x in r.get("clutch_player_ids", [])),
            "minutes": proj_minutes,
        }
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _spearman(proj: Dict[str, float], actual: Dict[str, float]) -> Optional[float]:
    pids = sorted(set(proj) | set(actual))
    if len(pids) < 3:
        return None
    a = np.array([proj.get(p, 0.0) for p in pids], dtype=float)
    b = np.array([actual.get(p, 0.0) for p in pids], dtype=float)
    if np.all(a == a[0]) or np.all(b == b[0]):
        return None
    ar = pd.Series(a).rank().to_numpy()
    br = pd.Series(b).rank().to_numpy()
    if np.std(ar) == 0 or np.std(br) == 0:
        return None
    return float(np.corrcoef(ar, br)[0, 1])


def score(gt: pd.DataFrame, proj: Dict[Tuple[str, str], Dict]) -> Dict:
    """Per-game scoring -> per-season + overall aggregates."""
    per_game: List[Dict] = []
    for row in gt.itertuples(index=False):
        key = (str(row.season), str(row.team_abbreviation).upper())
        p = proj.get(key)
        if p is None:
            continue

        rec: Dict = {"season": row.season, "team_abbreviation": row.team_abbreviation}

        if row.starters and len(row.starters) == LINEUP_SIZE and p["starters"]:
            rec["starter_hit"] = len(p["starters"] & set(map(str, row.starters))) / float(LINEUP_SIZE)
        if row.clutch5 and len(row.clutch5) == LINEUP_SIZE and p["clutch5"]:
            rec["clutch_overlap"] = float(len(p["clutch5"] & set(map(str, row.clutch5))))
        if row.minutes:
            actual_min = {pid: sec / 60.0 for pid, sec in row.minutes.items()}
            sp = _spearman(p["minutes"], actual_min)
            if sp is not None:
                rec["minutes_corr"] = sp
            # top-9 rotation set overlap
            proj_top9 = set(
                pid for pid, _ in sorted(p["minutes"].items(), key=lambda kv: kv[1], reverse=True)[:9]
            )
            act_top9 = set(
                pid for pid, _ in sorted(actual_min.items(), key=lambda kv: kv[1], reverse=True)[:9]
            )
            if proj_top9 and act_top9:
                rec["rotation_top9_overlap"] = len(proj_top9 & act_top9) / 9.0
        per_game.append(rec)

    pg = pd.DataFrame(per_game)

    def agg(df: pd.DataFrame) -> Dict:
        return {
            "n_games": int(len(df)),
            "starter_hit_rate": _m(df, "starter_hit"),
            "clutch_overlap": _m(df, "clutch_overlap"),
            "minutes_corr": _m(df, "minutes_corr"),
            "rotation_top9_overlap": _m(df, "rotation_top9_overlap"),
        }

    per_season = {
        str(s): agg(sdf) for s, sdf in pg.groupby("season", sort=True)
    } if not pg.empty else {}
    overall = agg(pg) if not pg.empty else {}
    return {"overall": overall, "per_season": per_season}


def _m(df: pd.DataFrame, col: str) -> Optional[float]:
    if col not in df.columns:
        return None
    v = pd.to_numeric(df[col], errors="coerce")
    if v.notna().sum() == 0:
        return None
    return round(float(v.mean()), 4)


# ---------------------------------------------------------------------------
# Tuning grid + walk-forward
# ---------------------------------------------------------------------------
def tuning_grid() -> List[Tuple[str, Dict]]:
    """Small, principled grid over the levers that most drive starter/minutes
    accuracy. Baseline = current simulation_config defaults (empty override)."""
    grid: List[Tuple[str, Dict]] = [("baseline", {})]
    # Starter score = w_m*minutes + w_i*impact + w_pos*scarcity + w_c*clutch_share.
    # Actual starters are overwhelmingly the high-minute players -> push minutes weight.
    starter_weights = [
        ("starter_m060", {"starter_weight_m": 0.60, "starter_weight_i": 0.20, "starter_weight_pos": 0.08, "starter_weight_c": 0.12}),
        ("starter_m070", {"starter_weight_m": 0.70, "starter_weight_i": 0.15, "starter_weight_pos": 0.05, "starter_weight_c": 0.10}),
        ("starter_m080", {"starter_weight_m": 0.80, "starter_weight_i": 0.12, "starter_weight_pos": 0.04, "starter_weight_c": 0.04}),
    ]
    grid.extend(starter_weights)
    # Low-minutes penalty: discourage starting sub-threshold-minute players.
    grid.append(("lowmin_125", {"low_minutes_penalty": 1.25}))
    grid.append(("lowmin_150", {"low_minutes_penalty": 1.50}))
    # Continuity prior strength (returning-player carry).
    grid.append(("kappa_050", {"continuity_kappa": 0.50, "continuity_starter_bonus": 0.12}))
    # Drop the true-big positional constraint (lets pure minutes/impact pick starters).
    grid.append(("no_true_big", {"require_true_big": False}))
    # Best-guess combo: heavy minutes + low-min penalty, no positional constraint.
    grid.append(
        (
            "combo_min_heavy",
            {
                "starter_weight_m": 0.70,
                "starter_weight_i": 0.15,
                "starter_weight_pos": 0.05,
                "starter_weight_c": 0.10,
                "low_minutes_penalty": 1.25,
            },
        )
    )
    return grid


def objective(season_metrics: Dict) -> Optional[float]:
    """Combined starter + minutes objective used for walk-forward selection."""
    sh = season_metrics.get("starter_hit_rate")
    mc = season_metrics.get("minutes_corr")
    if sh is None or mc is None:
        return None
    return sh + mc


def walk_forward(all_scores: Dict[str, Dict], eval_seasons: List[str]) -> Dict:
    """For each eval season T, pick the config maximizing the train objective on
    seasons < T (clutch must not regress vs baseline-train), score it on T."""
    baseline = all_scores["baseline"]["per_season"]
    chosen: Dict[str, str] = {}
    rows: List[Dict] = []

    for i, T in enumerate(eval_seasons):
        train_seasons = eval_seasons[:i]
        if not train_seasons:
            chosen[T] = "baseline"  # no history -> default
        else:
            base_clutch_train = np.mean(
                [baseline[s]["clutch_overlap"] for s in train_seasons if baseline.get(s, {}).get("clutch_overlap") is not None]
            )
            best_name, best_obj = "baseline", -np.inf
            for name, sc in all_scores.items():
                ps = sc["per_season"]
                objs = [objective(ps[s]) for s in train_seasons if s in ps]
                objs = [o for o in objs if o is not None]
                clutches = [ps[s]["clutch_overlap"] for s in train_seasons if ps.get(s, {}).get("clutch_overlap") is not None]
                if not objs or not clutches:
                    continue
                # no-clutch-regression gate (0.10 tolerance on overlap count)
                if np.mean(clutches) < base_clutch_train - 0.10:
                    continue
                mobj = float(np.mean(objs))
                if mobj > best_obj:
                    best_obj, best_name = mobj, name
            chosen[T] = best_name

        tuned = all_scores[chosen[T]]["per_season"].get(T, {})
        base = baseline.get(T, {})
        rows.append(
            {
                "season": T,
                "chosen_config": chosen[T],
                "baseline_starter_hit": base.get("starter_hit_rate"),
                "tuned_starter_hit": tuned.get("starter_hit_rate"),
                "baseline_minutes_corr": base.get("minutes_corr"),
                "tuned_minutes_corr": tuned.get("minutes_corr"),
                "baseline_clutch_overlap": base.get("clutch_overlap"),
                "tuned_clutch_overlap": tuned.get("clutch_overlap"),
            }
        )

    def _avg(rows_, key):
        vals = [r[key] for r in rows_ if r.get(key) is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    summary = {
        "eval_seasons": eval_seasons,
        "baseline_starter_hit": _avg(rows, "baseline_starter_hit"),
        "tuned_starter_hit": _avg(rows, "tuned_starter_hit"),
        "baseline_minutes_corr": _avg(rows, "baseline_minutes_corr"),
        "tuned_minutes_corr": _avg(rows, "tuned_minutes_corr"),
        "baseline_clutch_overlap": _avg(rows, "baseline_clutch_overlap"),
        "tuned_clutch_overlap": _avg(rows, "tuned_clutch_overlap"),
    }
    # Performance gate
    gate = {
        "starter_hit_improved": (
            summary["tuned_starter_hit"] is not None
            and summary["baseline_starter_hit"] is not None
            and summary["tuned_starter_hit"] >= summary["baseline_starter_hit"]
        ),
        "minutes_corr_improved": (
            summary["tuned_minutes_corr"] is not None
            and summary["baseline_minutes_corr"] is not None
            and summary["tuned_minutes_corr"] >= summary["baseline_minutes_corr"]
        ),
        "clutch_not_regressed": (
            summary["tuned_clutch_overlap"] is not None
            and summary["baseline_clutch_overlap"] is not None
            and summary["tuned_clutch_overlap"] >= summary["baseline_clutch_overlap"] - 0.10
        ),
    }
    gate["PASS"] = bool(gate["starter_hit_improved"] and gate["minutes_corr_improved"] and gate["clutch_not_regressed"])
    return {"per_season": rows, "summary": summary, "gate": gate}


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild-truth", action="store_true")
    ap.add_argument("--seasons", type=str, default=None, help="comma-separated subset")
    args = ap.parse_args()

    seasons = set(args.seasons.split(",")) if args.seasons else None
    gt = load_or_build_truth(args.rebuild_truth, seasons)
    gt_seasons = sorted(gt["season"].unique())
    print(f"Ground truth: {len(gt)} game-team rows over seasons {gt_seasons}")

    grid = tuning_grid()
    all_scores: Dict[str, Dict] = {}
    for name, override in grid:
        print(f"\n=== Projecting config '{name}' {override or '(defaults)'} ===")
        proj = project(override)
        all_scores[name] = score(gt, proj)
        ov = all_scores[name]["overall"]
        print(
            f"  overall: starter_hit={ov.get('starter_hit_rate')} "
            f"clutch={ov.get('clutch_overlap')} minutes_corr={ov.get('minutes_corr')} "
            f"top9={ov.get('rotation_top9_overlap')} (n={ov.get('n_games')})"
        )

    # eval seasons = those present in both projector and ground truth
    proj_seasons = set()
    for sc in all_scores.values():
        proj_seasons |= set(sc["per_season"].keys())
    eval_seasons = sorted(s for s in gt_seasons if s in proj_seasons)

    wf = walk_forward(all_scores, eval_seasons)

    report = {
        "generated_for": "Step 0a — lineup projection tuning (game-by-game, offline)",
        "ground_truth": {
            "source": "data/historical/pbp_with_lineups_*.parquet",
            "n_game_team_rows": int(len(gt)),
            "seasons": gt_seasons,
            "clutch_window": {
                "period_min": CLUTCH_PERIOD_MIN,
                "margin_max": CLUTCH_MARGIN_MAX,
                "clock_max_sec": CLUTCH_CLOCK_MAX_SEC,
            },
        },
        "config_grid": {name: override for name, override in grid},
        "baseline": all_scores["baseline"],
        "all_configs_overall": {
            name: sc["overall"] for name, sc in all_scores.items()
        },
        "walk_forward": wf,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved report: {REPORT_PATH}")
    print("\nWalk-forward summary:")
    print(json.dumps(wf["summary"], indent=2))
    print("\nGate:")
    print(json.dumps(wf["gate"], indent=2))


if __name__ == "__main__":
    main()
