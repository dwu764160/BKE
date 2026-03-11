"""
scripts/validate_player_stat_sim.py
=============================================================================
Validation phase for the Player Stat Sim module.

Compares simulated player season stats against actual stats from the
player profile aggregate. Produces a validation report with:
  - Per-player stat deltas (simulated minus actual)
  - Aggregate accuracy metrics (MAE, correlation, RMSE) for major stats
  - Archetype-level and team-level accuracy breakdowns
  - Outlier detection (largest misses)
  - Summary JSON written to reports/

Usage:
  python3 scripts/validate_player_stat_sim.py
  python3 scripts/validate_player_stat_sim.py --mode forecast
=============================================================================
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

AGGREGATE_PATH = ROOT / "aggregate" / "player_profile_aggregate.parquet"
SIM_SEASON_STATS_PATH = ROOT / "data" / "processed" / "simulation" / "simulation_step1_player_season_stats.parquet"
FORECAST_SEASON_STATS_PATTERN = ROOT / "data" / "processed" / "simulation" / "forecast_step1_player_season_stats_*.parquet"
REPORT_OUTPUT = ROOT / "reports" / "player_stat_sim_validation.json"

STAT_COLS_MAP = {
    "pts": ("pts", "pts"),
    "ast": ("ast", "ast"),
    "reb": ("reb", "reb"),
    "stl": ("stl", "stl"),
    "blk": ("blk", "blk"),
    "tov": ("tov", "tov"),
    "fgm": ("fgm", "fgm"),
    "fga": ("fga", "fga"),
    "fg3m": ("fg3m", "fg3_m"),
    "fg3a": ("fg3a", "fg3_a"),
    "ftm": ("ftm", "ftm"),
    "fta": ("fta", "fta"),
}

PER_GAME_STATS = ["pts", "ast", "reb", "stl", "blk", "tov", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta"]


def _safe(v):
    if isinstance(v, (float, np.floating)):
        if np.isnan(v) or np.isinf(v):
            return None
        return round(float(v), 4)
    if isinstance(v, (int, np.integer)):
        return int(v)
    return v


def _norm_id(s):
    return s.astype(str).str.replace(r"\.0$", "", regex=True)


def load_actual_stats(aggregate_path: Path) -> pd.DataFrame:
    if not aggregate_path.exists():
        raise FileNotFoundError(f"Profile aggregate not found: {aggregate_path}")
    df = pd.read_parquet(aggregate_path)
    df["player_id"] = _norm_id(df["player_id"].astype(str))
    df["season"] = df["season"].astype(str)

    games_col = None
    for c in ["gp", "games", "g"]:
        if c in df.columns:
            games_col = c
            break
    if games_col is None:
        df["games_actual"] = 82
    else:
        df["games_actual"] = pd.to_numeric(df[games_col], errors="coerce").fillna(82).clip(lower=1)

    minutes_col = None
    for c in ["min", "minutes", "total_minutes"]:
        if c in df.columns:
            minutes_col = c
            break
    if minutes_col:
        df["minutes_actual"] = pd.to_numeric(df[minutes_col], errors="coerce").fillna(0)
    else:
        mpg = pd.to_numeric(df.get("mpg", 0), errors="coerce").fillna(0)
        df["minutes_actual"] = mpg * df["games_actual"]

    df["mpg_actual"] = df["minutes_actual"] / df["games_actual"].replace(0, np.nan)

    keep = ["player_id", "season", "player_name", "games_actual", "minutes_actual", "mpg_actual"]
    for stat_key, (_, agg_col) in STAT_COLS_MAP.items():
        if agg_col in df.columns:
            df[f"actual_{stat_key}"] = pd.to_numeric(df[agg_col], errors="coerce").fillna(0)
            keep.append(f"actual_{stat_key}")

    # TS%
    if "ts_pct" in df.columns:
        df["actual_ts_pct"] = pd.to_numeric(df["ts_pct"], errors="coerce")
        keep.append("actual_ts_pct")

    # Archetypes
    for c in ["primary_archetype", "off_primary_archetype"]:
        if c in df.columns:
            df["actual_archetype"] = df[c].fillna("Unknown").astype(str)
            keep.append("actual_archetype")
            break

    # Team
    for c in ["team_abbreviation", "team"]:
        if c in df.columns:
            df["actual_team"] = df[c].astype(str).str.upper()
            keep.append("actual_team")
            break

    return df[keep].copy()


def load_sim_stats(sim_path: Path) -> pd.DataFrame:
    if not sim_path.exists():
        raise FileNotFoundError(f"Sim stats not found: {sim_path}")
    df = pd.read_parquet(sim_path)
    df["player_id"] = _norm_id(df["player_id"].astype(str))
    df["season"] = df["season"].astype(str)
    return df


def merge_and_compare(sim_df: pd.DataFrame, actual_df: pd.DataFrame, min_minutes: float = 200.0) -> pd.DataFrame:
    merged = sim_df.merge(actual_df, on=["player_id", "season"], how="inner", suffixes=("_sim", "_actual"))

    # Filter to players with enough minutes in both
    merged = merged[merged["minutes"] >= min_minutes].copy()
    merged = merged[merged["minutes_actual"] >= min_minutes].copy()

    if merged.empty:
        return merged

    # Compute per-game stats
    sim_gp = merged["games_played"].replace(0, np.nan)
    actual_gp = merged["games_actual"].replace(0, np.nan)

    for stat_key in PER_GAME_STATS:
        sim_col = stat_key
        actual_col = f"actual_{stat_key}"
        if sim_col in merged.columns and actual_col in merged.columns:
            merged[f"sim_pg_{stat_key}"] = merged[sim_col] / sim_gp
            merged[f"actual_pg_{stat_key}"] = merged[actual_col] / actual_gp
            merged[f"delta_{stat_key}"] = merged[f"sim_pg_{stat_key}"] - merged[f"actual_pg_{stat_key}"]
            merged[f"abs_delta_{stat_key}"] = merged[f"delta_{stat_key}"].abs()

    # MPG comparison
    merged["sim_mpg"] = merged["mpg"] if "mpg" in merged.columns else merged["minutes"] / sim_gp
    merged["delta_mpg"] = merged["sim_mpg"] - merged["mpg_actual"]

    # TS% comparison
    if "ts_pct" in merged.columns and "actual_ts_pct" in merged.columns:
        merged["delta_ts_pct"] = merged["ts_pct"] - merged["actual_ts_pct"]

    return merged


def compute_stat_accuracy(merged: pd.DataFrame) -> dict:
    results = {}
    for stat_key in PER_GAME_STATS:
        sim_col = f"sim_pg_{stat_key}"
        actual_col = f"actual_pg_{stat_key}"
        if sim_col not in merged.columns or actual_col not in merged.columns:
            continue
        sim_vals = merged[sim_col].dropna()
        actual_vals = merged[actual_col].dropna()
        common = sim_vals.index.intersection(actual_vals.index)
        if len(common) < 5:
            continue
        s = sim_vals.loc[common]
        a = actual_vals.loc[common]
        delta = s - a
        results[stat_key] = {
            "n": int(len(common)),
            "mae": _safe(delta.abs().mean()),
            "rmse": _safe(np.sqrt((delta ** 2).mean())),
            "correlation": _safe(np.corrcoef(s, a)[0, 1]) if s.std() > 1e-9 and a.std() > 1e-9 else None,
            "mean_sim": _safe(s.mean()),
            "mean_actual": _safe(a.mean()),
            "median_abs_error": _safe(delta.abs().median()),
            "bias": _safe(delta.mean()),
        }
    return results


def compute_team_accuracy(merged: pd.DataFrame) -> dict:
    results = {}
    team_col = "team_abbreviation" if "team_abbreviation" in merged.columns else None
    if not team_col:
        return results

    for team, group in merged.groupby(team_col):
        team_stats = {}
        for stat_key in ["pts", "ast", "reb"]:
            sim_col = f"sim_pg_{stat_key}"
            actual_col = f"actual_pg_{stat_key}"
            if sim_col in group.columns and actual_col in group.columns:
                s = group[sim_col].dropna()
                a = group[actual_col].dropna()
                common = s.index.intersection(a.index)
                if len(common) >= 3:
                    delta = s.loc[common] - a.loc[common]
                    team_stats[stat_key] = {
                        "mae": _safe(delta.abs().mean()),
                        "bias": _safe(delta.mean()),
                        "n": int(len(common)),
                    }
        if team_stats:
            results[str(team)] = team_stats
    return results


def compute_archetype_accuracy(merged: pd.DataFrame) -> dict:
    results = {}
    arch_col = "actual_archetype" if "actual_archetype" in merged.columns else None
    if not arch_col:
        return results

    for arch, group in merged.groupby(arch_col):
        if str(arch).lower() in ("unknown", "nan", "none", ""):
            continue
        arch_stats = {}
        for stat_key in ["pts", "ast", "reb", "stl", "blk"]:
            sim_col = f"sim_pg_{stat_key}"
            actual_col = f"actual_pg_{stat_key}"
            if sim_col in group.columns and actual_col in group.columns:
                s = group[sim_col].dropna()
                a = group[actual_col].dropna()
                common = s.index.intersection(a.index)
                if len(common) >= 3:
                    delta = s.loc[common] - a.loc[common]
                    arch_stats[stat_key] = {
                        "mae": _safe(delta.abs().mean()),
                        "bias": _safe(delta.mean()),
                        "n": int(len(common)),
                    }
        if arch_stats:
            results[str(arch)] = arch_stats
    return results


def find_outliers(merged: pd.DataFrame, stat_key: str = "pts", top_n: int = 10) -> list:
    delta_col = f"abs_delta_{stat_key}"
    if delta_col not in merged.columns:
        return []
    sorted_df = merged.nlargest(top_n, delta_col)
    out = []
    for _, row in sorted_df.iterrows():
        out.append({
            "player_name": str(row.get("player_name_sim", row.get("player_name", "?"))),
            "player_id": str(row["player_id"]),
            "season": str(row["season"]),
            "team": str(row.get("team_abbreviation", "?")),
            f"sim_{stat_key}_pg": _safe(row.get(f"sim_pg_{stat_key}")),
            f"actual_{stat_key}_pg": _safe(row.get(f"actual_pg_{stat_key}")),
            f"delta_{stat_key}": _safe(row.get(f"delta_{stat_key}")),
        })
    return out


def build_player_comparison_rows(merged: pd.DataFrame) -> list:
    """Build per-player comparison rows for frontend display."""
    rows = []
    for _, row in merged.iterrows():
        entry = {
            "player_name": str(row.get("player_name_sim", row.get("player_name", "?"))),
            "player_id": str(row["player_id"]),
            "season": str(row["season"]),
            "team": str(row.get("team_abbreviation", "?")),
            "sim_gp": _safe(row.get("games_played")),
            "actual_gp": _safe(row.get("games_actual")),
            "sim_mpg": _safe(row.get("sim_mpg")),
            "actual_mpg": _safe(row.get("mpg_actual")),
        }
        for stat_key in PER_GAME_STATS:
            entry[f"sim_{stat_key}"] = _safe(row.get(f"sim_pg_{stat_key}"))
            entry[f"actual_{stat_key}"] = _safe(row.get(f"actual_pg_{stat_key}"))
            entry[f"delta_{stat_key}"] = _safe(row.get(f"delta_{stat_key}"))
        if "ts_pct" in row.index:
            entry["sim_ts_pct"] = _safe(row.get("ts_pct"))
        if "actual_ts_pct" in row.index:
            entry["actual_ts_pct"] = _safe(row.get("actual_ts_pct"))
        if "delta_ts_pct" in row.index:
            entry["delta_ts_pct"] = _safe(row.get("delta_ts_pct"))
        if "actual_archetype" in row.index:
            entry["archetype"] = str(row.get("actual_archetype", ""))
        rows.append(entry)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Validate player stat sim against profile aggregate")
    parser.add_argument("--mode", choices=["backtest", "forecast"], default="backtest",
                        help="Which sim output to validate")
    parser.add_argument("--scenario", type=str, default=None,
                        help="Forecast scenario key (e.g., end_of_season)")
    parser.add_argument("--min-minutes", type=float, default=200.0,
                        help="Minimum total minutes for inclusion")
    parser.add_argument("--output", type=str, default=None, help="Output path override")
    args = parser.parse_args()

    print("=" * 70)
    print(f"  PLAYER STAT SIM VALIDATION — mode={args.mode}")
    print("=" * 70)

    # Load actual stats
    print("\n[1] Loading profile aggregate...")
    actual = load_actual_stats(AGGREGATE_PATH)
    print(f"  Loaded {len(actual)} player-season rows from aggregate")

    # Load sim stats
    print("\n[2] Loading simulated stats...")
    if args.mode == "forecast":
        if args.scenario:
            sim_path = ROOT / "data" / "processed" / "simulation" / f"forecast_step1_player_season_stats_{args.scenario}.parquet"
        else:
            candidates = sorted(FORECAST_SEASON_STATS_PATTERN.parent.glob("forecast_step1_player_season_stats_*.parquet"))
            if not candidates:
                print("  No forecast stats found. Run: python3 src/simulation/run_forecast.py")
                return
            sim_path = candidates[0]
            print(f"  Using first scenario: {sim_path.stem}")
    else:
        sim_path = SIM_SEASON_STATS_PATH

    sim = load_sim_stats(sim_path)
    print(f"  Loaded {len(sim)} simulated player-season rows")
    print(f"  Seasons: {sorted(sim['season'].unique())}")

    # Merge
    print(f"\n[3] Merging and comparing (min_minutes={args.min_minutes})...")  # noqa: F541
    merged = merge_and_compare(sim, actual, min_minutes=args.min_minutes)
    print(f"  Matched {len(merged)} player-season comparisons")

    if merged.empty:
        print("\n  No matched players found — check season/player_id alignment")
        return

    # Compute accuracy
    print("\n[4] Computing accuracy metrics...")
    stat_accuracy = compute_stat_accuracy(merged)
    for stat_key, metrics in stat_accuracy.items():
        print(f"  {stat_key:>5s}: MAE={metrics['mae']:.2f}  RMSE={metrics['rmse']:.2f}  "
              f"r={metrics['correlation']:.3f}  bias={metrics['bias']:+.2f}  n={metrics['n']}")

    print("\n[5] Computing team-level accuracy...")
    team_accuracy = compute_team_accuracy(merged)

    print("\n[6] Computing archetype-level accuracy...")
    archetype_accuracy = compute_archetype_accuracy(merged)
    for arch, stats in sorted(archetype_accuracy.items()):
        pts_info = stats.get("pts", {})
        print(f"  {arch:>30s}: pts_mae={pts_info.get('mae', '-')}"
              f"  bias={pts_info.get('bias', '-')}"
              f"  n={pts_info.get('n', 0)}")

    print("\n[7] Finding outliers...")
    pts_outliers = find_outliers(merged, "pts", top_n=10)
    ast_outliers = find_outliers(merged, "ast", top_n=5)
    reb_outliers = find_outliers(merged, "reb", top_n=5)

    print("\n[8] Building player comparison rows...")
    player_rows = build_player_comparison_rows(merged)

    # Seasons covered
    seasons = sorted(merged["season"].unique().tolist())

    report = {
        "mode": args.mode,
        "scenario": args.scenario,
        "source": str(sim_path),
        "aggregate_source": str(AGGREGATE_PATH),
        "min_minutes": args.min_minutes,
        "n_compared": len(merged),
        "seasons": seasons,
        "stat_accuracy": stat_accuracy,
        "team_accuracy": team_accuracy,
        "archetype_accuracy": archetype_accuracy,
        "outliers": {
            "pts": pts_outliers,
            "ast": ast_outliers,
            "reb": reb_outliers,
        },
        "player_comparisons": player_rows,
    }

    default_output = REPORT_OUTPUT if args.mode == "backtest" else ROOT / "reports" / "player_stat_sim_validation_forecast.json"
    output_path = Path(args.output) if args.output else default_output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n  Saved validation report: {output_path}")
    print("=" * 70)
    print("  DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
