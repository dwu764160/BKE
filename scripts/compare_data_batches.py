#!/usr/bin/env python3
"""
compare_data_batches.py — Compare old vs new BKE data outputs.

Usage:
    python scripts/compare_data_batches.py <old_dir> <new_dir>
    python scripts/compare_data_batches.py data data_temp_reprod

Performs:
    A. Completeness checks (file existence, row counts, schema, coverage)
    B. Correctness checks (numeric drift, categorical stability, spot checks, KS-test)
    
Outputs:
    - Console summary with pass/fail per file
    - <new_dir>/logs/comparison_report.json
    - <new_dir>/logs/comparison_report.txt
"""

import sys
import os
import json
import datetime
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# COMPARISON MANIFEST — derived from reference/pipeline/data_reproduction.md §4
# ---------------------------------------------------------------------------

SEASONS = ["2022-23", "2023-24", "2024-25"]

COMPARISON_MANIFEST = {
    # --- Layer 1 outputs ---
    "historical/pbp_normalized_{season}.parquet": {
        "seasons": SEASONS,
        "key_cols": ["game_id", "event_type", "player_id"],
        "check": "row_count_and_schema",
    },
    "historical/pbp_with_lineups_{season}.parquet": {
        "seasons": SEASONS,
        "key_cols": ["game_id", "off_lineup", "def_lineup"],
        "check": "row_count_and_schema",
    },
    "historical/possessions_clean_{season}.parquet": {
        "seasons": SEASONS,
        "key_cols": ["game_id", "off_team_id", "points"],
        "check": "row_count_and_numeric_drift",
    },
    "historical/team_game_logs.parquet": {
        "key_cols": ["GAME_ID", "TEAM_ID", "PTS"],
        "check": "exact_match",
    },
    "historical/team_summaries.parquet": {
        "join_on": ["TEAM_ID", "SEASON"],
        "numeric_cols": ["GAMES", "WINS", "LOSSES"],
        "check": "numeric_drift",
    },
    "historical/feature_schedule_context.parquet": {
        "check": "row_count_and_schema",
    },
    # --- Layer 2 outputs ---
    "processed/player_profiles_advanced.parquet": {
        "join_on": ["player_id", "season"],
        "numeric_cols": ["PTS", "AST", "REB", "STL", "BLK", "TOV", "MIN", "GP",
                         "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
                         "TS_PCT", "USG_PCT"],
        "tolerance": 0.01,
        "check": "full_numeric",
    },
    "processed/metrics_linear.parquet": {
        "join_on": ["player_id", "season"],
        "numeric_cols": ["WS", "OWS", "DWS", "BPM", "VORP"],
        "tolerance": 0.05,
        "check": "full_numeric",
    },
    "processed/metrics_teams.parquet": {
        "join_on": ["team_id", "season"],
        "numeric_cols": ["ORTG", "DRTG", "NET_RTG"],
        "tolerance": 2.0,
        "check": "full_numeric",
        # Non-deterministic ridge regression means team ratings can drift ~1-2 pts
        "corr_threshold": 0.95,
    },
    "processed/player_rapm.parquet": {
        "join_on": ["player_id", "season", "RAPM_type"],
        "numeric_cols": ["RAPM", "ORAPM", "DRAPM"],
        "tolerance": 0.5,
        "check": "full_numeric",
        # RidgeCV alpha selection is non-deterministic
        "corr_threshold": 0.90,
    },
    "processed/player_xrapm.parquet": {
        "join_on": ["player_id", "season", "xRAPM_type"],
        "numeric_cols": ["xRAPM", "O_xRAPM", "D_xRAPM"],
        "tolerance": 0.5,
        "check": "full_numeric",
        "corr_threshold": 0.90,
    },
    "processed/player_xrapm_v2.parquet": {
        "join_on": ["player_id", "season"],
        "numeric_cols": ["xRAPM"],
        "tolerance": 0.5,
        "check": "full_numeric",
        "corr_threshold": 0.90,
    },
    "advanced_local_metrics.parquet": {
        "join_on": ["PLAYER_ID", "SEASON"],
        "numeric_cols": ["GAMES", "MPG", "PTS_total", "AST_total", "REB_total",
                         "MIN_total", "TS_pct", "eFG_pct", "TOV_pct",
                         "AST_TOV_ratio",
                         "points_responsible", "PTS_per75poss", "points_per_shot",
                         "FT_rate", "three_point_freq", "three_point_pct", "PROD"],
        "check": "full_numeric",
        "tolerance": 0.05,
        "corr_threshold": 0.95,
        # AST_pct, USG_pct, OREB_pct, DREB_pct excluded: they depend on
        # TEAM_FGM/TEAM_OREB/TEAM_DREB which differ between official team_game_logs
        # vs player-log-aggregate computation paths
        "row_count_tolerance": 10.0,  # allow up to 10% row count diff
    },
    "official_stats/official_advanced_{season}.parquet": {
        "seasons": SEASONS,
        "check": "exact_match",
    },
}

# ---------------------------------------------------------------------------
# CORE COMPARISON FUNCTIONS
# ---------------------------------------------------------------------------


class ComparisonResult:
    """Holds results for one file comparison."""
    def __init__(self, rel_path: str):
        self.rel_path = rel_path
        self.status = "SKIP"  # PASS, WARN, FAIL, SKIP, MISSING
        self.rows_old = 0
        self.rows_new = 0
        self.drift = None
        self.notes = []
        self.details = {}

    def to_dict(self):
        return {
            "file": self.rel_path,
            "status": self.status,
            "rows_old": self.rows_old,
            "rows_new": self.rows_new,
            "drift": self.drift,
            "notes": self.notes,
            "details": self.details,
        }


def load_parquet_safe(path: Path) -> pd.DataFrame | None:
    """Load parquet, return None if missing or corrupt."""
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception as e:
        return None


def check_file_existence(old_path: Path, new_path: Path, result: ComparisonResult) -> tuple:
    """Check that both files exist. Returns (old_df, new_df) or (None, None)."""
    old_df = load_parquet_safe(old_path)
    new_df = load_parquet_safe(new_path)

    if old_df is None and new_df is None:
        result.status = "SKIP"
        result.notes.append("Both files missing — skipping")
        return None, None
    if old_df is None:
        result.status = "WARN"
        result.notes.append("Old file missing, new exists")
        result.rows_new = len(new_df)
        return None, new_df
    if new_df is None:
        result.status = "FAIL"
        result.notes.append("New file MISSING")
        result.rows_old = len(old_df)
        return old_df, None

    result.rows_old = len(old_df)
    result.rows_new = len(new_df)
    return old_df, new_df


def check_schema(old_df: pd.DataFrame, new_df: pd.DataFrame, result: ComparisonResult):
    """Check columns match."""
    old_cols = set(old_df.columns)
    new_cols = set(new_df.columns)
    missing = old_cols - new_cols
    extra = new_cols - old_cols
    if missing:
        result.notes.append(f"Missing cols in new: {sorted(missing)}")
        result.status = "WARN" if result.status != "FAIL" else "FAIL"
    if extra:
        result.notes.append(f"Extra cols in new: {sorted(extra)}")


def check_row_count(old_df: pd.DataFrame, new_df: pd.DataFrame, result: ComparisonResult,
                    tolerance_pct: float = 5.0):
    """Check row counts are within tolerance."""
    diff_pct = abs(len(old_df) - len(new_df)) / max(len(old_df), 1) * 100
    if diff_pct > tolerance_pct:
        result.notes.append(f"Row count diff: {len(old_df)} → {len(new_df)} ({diff_pct:.1f}%)")
        result.status = "FAIL"
    elif diff_pct > 0.5:
        result.notes.append(f"Row count diff: {len(old_df)} → {len(new_df)} ({diff_pct:.1f}%)")
        if result.status == "PASS":
            result.status = "WARN"


def compute_numeric_drift(old_df: pd.DataFrame, new_df: pd.DataFrame, 
                          join_on: list[str] | None = None,
                          numeric_cols: list[str] | None = None,
                          tolerance: float = 0.01) -> dict:
    """Compute per-column numeric drift stats between old and new."""
    drift_info = {}

    if join_on:
        # Normalize join columns to string for safe merge
        for col in join_on:
            if col in old_df.columns and col in new_df.columns:
                old_df[col] = old_df[col].astype(str)
                new_df[col] = new_df[col].astype(str)
        merged = old_df.merge(new_df, on=join_on, suffixes=("_old", "_new"), how="inner")
    else:
        # Positional comparison — just align by index
        min_len = min(len(old_df), len(new_df))
        merged = pd.concat([
            old_df.head(min_len).reset_index(drop=True).add_suffix("_old"),
            new_df.head(min_len).reset_index(drop=True).add_suffix("_new"),
        ], axis=1)

    if numeric_cols is None:
        # Auto-detect numeric columns from old_df
        numeric_cols = old_df.select_dtypes(include=[np.number]).columns.tolist()
        # Filter to cols that exist in both
        numeric_cols = [c for c in numeric_cols 
                       if f"{c}_old" in merged.columns and f"{c}_new" in merged.columns]

    for col in numeric_cols:
        old_col = f"{col}_old" if f"{col}_old" in merged.columns else col
        new_col = f"{col}_new" if f"{col}_new" in merged.columns else col
        if old_col not in merged.columns or new_col not in merged.columns:
            continue

        old_vals = pd.to_numeric(merged[old_col], errors="coerce")
        new_vals = pd.to_numeric(merged[new_col], errors="coerce")
        
        # Drop NaN pairs
        mask = old_vals.notna() & new_vals.notna()
        old_v = old_vals[mask].values
        new_v = new_vals[mask].values

        if len(old_v) == 0:
            continue

        abs_diff = np.abs(old_v - new_v)
        mad = np.mean(abs_diff)
        max_diff = np.max(abs_diff)

        # Pearson correlation
        if np.std(old_v) > 0 and np.std(new_v) > 0:
            corr = np.corrcoef(old_v, new_v)[0, 1]
        else:
            corr = 1.0 if np.allclose(old_v, new_v) else 0.0

        # Percent changed by > tolerance
        denom = np.maximum(np.abs(old_v), 1e-8)
        pct_changed = np.mean(abs_diff / denom > tolerance) * 100

        # KS test
        try:
            ks_stat, ks_p = stats.ks_2samp(old_v, new_v)
        except Exception:
            ks_stat, ks_p = 0.0, 1.0

        drift_info[col] = {
            "MAD": round(float(mad), 6),
            "max_diff": round(float(max_diff), 6),
            "correlation": round(float(corr), 6),
            "pct_changed_above_tol": round(float(pct_changed), 2),
            "KS_stat": round(float(ks_stat), 4),
            "KS_p": round(float(ks_p), 4),
            "n_compared": int(len(old_v)),
        }

    return drift_info


def check_labels(old_df: pd.DataFrame, new_df: pd.DataFrame,
                 join_on: list[str], label_cols: list[str]) -> dict:
    """Check categorical label stability between old and new."""
    label_info = {}
    for col in join_on:
        if col in old_df.columns and col in new_df.columns:
            old_df[col] = old_df[col].astype(str)
            new_df[col] = new_df[col].astype(str)
    merged = old_df.merge(new_df, on=join_on, suffixes=("_old", "_new"), how="inner")

    for col in label_cols:
        old_col = f"{col}_old"
        new_col = f"{col}_new"
        if old_col not in merged.columns or new_col not in merged.columns:
            continue

        changed_mask = merged[old_col] != merged[new_col]
        n_changed = int(changed_mask.sum())
        n_total = len(merged)
        pct_changed = n_changed / max(n_total, 1) * 100

        # List changed players if any
        changed_rows = merged[changed_mask]
        changes = []
        for _, row in changed_rows.head(20).iterrows():
            player_key = " / ".join(str(row[c]) for c in join_on)
            changes.append(f"{player_key}: {row[old_col]} → {row[new_col]}")

        label_info[col] = {
            "n_changed": n_changed,
            "n_total": n_total,
            "pct_changed": round(pct_changed, 2),
            "sample_changes": changes,
        }

    return label_info


# ---------------------------------------------------------------------------
# HIGH-LEVEL CHECK DISPATCHERS
# ---------------------------------------------------------------------------


def compare_row_count_and_schema(old_path, new_path, spec, result):
    """Check row count and schema only."""
    old_df, new_df = check_file_existence(old_path, new_path, result)
    if old_df is None or new_df is None:
        return
    check_schema(old_df, new_df, result)
    check_row_count(old_df, new_df, result)
    if result.status not in ("FAIL", "WARN"):
        result.status = "PASS"


def compare_exact_match(old_path, new_path, spec, result):
    """Check for exact byte-level or row-level match."""
    old_df, new_df = check_file_existence(old_path, new_path, result)
    if old_df is None or new_df is None:
        return
    check_schema(old_df, new_df, result)
    check_row_count(old_df, new_df, result, tolerance_pct=0.0)

    # Compare content
    common_cols = sorted(set(old_df.columns) & set(new_df.columns))
    if len(old_df) == len(new_df) and len(common_cols) > 0:
        try:
            match = (old_df[common_cols].reset_index(drop=True)
                     .equals(new_df[common_cols].reset_index(drop=True)))
            if match:
                result.status = "PASS"
                result.notes.append("Exact match ✓")
            else:
                result.status = "WARN"
                result.notes.append("Row/col counts match but values differ")
        except Exception as e:
            result.status = "WARN"
            result.notes.append(f"Could not compare: {e}")
    elif result.status not in ("FAIL",):
        result.status = "WARN"


def compare_numeric_drift(old_path, new_path, spec, result):
    """Check numeric drift only (no join needed, simpler)."""
    old_df, new_df = check_file_existence(old_path, new_path, result)
    if old_df is None or new_df is None:
        return
    check_schema(old_df, new_df, result)
    check_row_count(old_df, new_df, result)

    join_on = spec.get("join_on")
    numeric_cols = spec.get("numeric_cols")
    tolerance = spec.get("tolerance", 0.01)

    drift = compute_numeric_drift(old_df, new_df, join_on, numeric_cols, tolerance)
    result.details["drift"] = drift

    # Compute aggregate drift
    if drift:
        max_mad = max(d["MAD"] for d in drift.values())
        min_corr = min(d["correlation"] for d in drift.values())
        result.drift = round(max_mad, 4)

        worst_cols = [c for c, d in drift.items() if d["correlation"] < 0.99]
        if worst_cols:
            result.notes.append(f"Low corr cols: {worst_cols}")
        if max_mad > tolerance * 10:
            result.status = "WARN"
        else:
            result.status = "PASS" if result.status not in ("FAIL",) else result.status


def compare_row_count_and_numeric_drift(old_path, new_path, spec, result):
    """Row count + schema + numeric drift."""
    compare_numeric_drift(old_path, new_path, spec, result)


def compare_full_numeric(old_path, new_path, spec, result):
    """Full numeric comparison with per-column stats."""
    old_df, new_df = check_file_existence(old_path, new_path, result)
    if old_df is None or new_df is None:
        return
    check_schema(old_df, new_df, result)
    row_tol = spec.get("row_count_tolerance", 5.0)
    check_row_count(old_df, new_df, result, tolerance_pct=row_tol)

    join_on = spec.get("join_on")
    numeric_cols = spec.get("numeric_cols")
    tolerance = spec.get("tolerance", 0.01)
    corr_threshold = spec.get("corr_threshold", 0.98)

    drift = compute_numeric_drift(old_df, new_df, join_on, numeric_cols, tolerance)
    result.details["drift"] = drift

    if drift:
        max_mad = max(d["MAD"] for d in drift.values())
        min_corr = min(d["correlation"] for d in drift.values())
        max_pct = max(d["pct_changed_above_tol"] for d in drift.values())
        result.drift = round(max_mad, 4)

        # Grade
        if min_corr < corr_threshold:
            result.status = "FAIL"
            result.notes.append(f"Min corr={min_corr:.4f} < {corr_threshold}")
        elif min_corr < 0.995 or max_mad > tolerance * 5:
            result.status = "WARN"
            result.notes.append(f"Min corr={min_corr:.4f}, Max MAD={max_mad:.4f}")
        else:
            result.status = "PASS"
            result.notes.append(f"Max MAD={max_mad:.4f}, Min corr={min_corr:.4f}")
    else:
        if result.status not in ("FAIL", "WARN"):
            result.status = "PASS"
            result.notes.append("No numeric columns to compare")


def compare_labels_and_numeric(old_path, new_path, spec, result):
    """Compare both labels and numeric columns."""
    old_df, new_df = check_file_existence(old_path, new_path, result)
    if old_df is None or new_df is None:
        return
    check_schema(old_df, new_df, result)
    check_row_count(old_df, new_df, result)

    join_on = spec.get("join_on", [])
    label_cols = spec.get("label_cols", [])
    numeric_cols = spec.get("numeric_cols")
    tolerance = spec.get("tolerance", 0.02)

    # Labels
    if label_cols and join_on:
        label_info = check_labels(old_df, new_df, join_on, label_cols)
        result.details["labels"] = label_info
        for col, info in label_info.items():
            if info["pct_changed"] > 10:
                result.status = "FAIL"
                result.notes.append(f"{col}: {info['pct_changed']:.1f}% labels changed (>{10}% threshold)")
            elif info["pct_changed"] > 2:
                result.status = "WARN" if result.status != "FAIL" else "FAIL"
                result.notes.append(f"{col}: {info['pct_changed']:.1f}% labels changed")
            else:
                result.notes.append(f"{col}: {info['n_changed']} label changes ({info['pct_changed']:.1f}%)")

    # Numeric
    if numeric_cols or not label_cols:
        drift = compute_numeric_drift(old_df, new_df, join_on, numeric_cols, tolerance)
        result.details["drift"] = drift
        if drift:
            max_mad = max(d["MAD"] for d in drift.values())
            result.drift = round(max_mad, 4)

    if result.status not in ("FAIL", "WARN"):
        result.status = "PASS"


# Map check types to functions
CHECK_DISPATCH = {
    "row_count_and_schema": compare_row_count_and_schema,
    "exact_match": compare_exact_match,
    "numeric_drift": compare_numeric_drift,
    "row_count_and_numeric_drift": compare_row_count_and_numeric_drift,
    "full_numeric": compare_full_numeric,
    "labels_and_numeric": compare_labels_and_numeric,
}


# ---------------------------------------------------------------------------
# MAIN — RUN ALL COMPARISONS
# ---------------------------------------------------------------------------


def expand_manifest(manifest: dict) -> list[tuple[str, dict]]:
    """Expand {season} templates into concrete file paths."""
    expanded = []
    for pattern, spec in manifest.items():
        if "{season}" in pattern:
            for season in spec.get("seasons", SEASONS):
                concrete = pattern.replace("{season}", season)
                expanded.append((concrete, spec))
        else:
            expanded.append((pattern, spec))
    return expanded


def format_console_table(results: list[ComparisonResult]) -> str:
    """Format results as a console table."""
    lines = []
    header = f"{'FILE':<55} {'STATUS':<10} {'ROWS_OLD':>10} {'ROWS_NEW':>10} {'DRIFT':>8}  NOTES"
    sep = "─" * 120
    lines.append(header)
    lines.append(sep)

    status_icons = {"PASS": "✅", "WARN": "⚠️ ", "FAIL": "❌", "SKIP": "⏭️ ", "MISSING": "❓"}
    for r in results:
        icon = status_icons.get(r.status, "  ")
        drift_str = f"{r.drift:.4f}" if r.drift is not None else "—"
        notes_str = "; ".join(r.notes[:2]) if r.notes else ""
        # Truncate if needed
        if len(notes_str) > 60:
            notes_str = notes_str[:57] + "..."
        line = f"{r.rel_path:<55} {icon} {r.status:<6} {r.rows_old:>10,} {r.rows_new:>10,} {drift_str:>8}  {notes_str}"
        lines.append(line)

    return "\n".join(lines)


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <old_dir> <new_dir>")
        sys.exit(1)

    old_dir = Path(sys.argv[1])
    new_dir = Path(sys.argv[2])

    print(f"\n=== DATA COMPARISON REPORT ===")
    print(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Old: {old_dir}/    New: {new_dir}/\n")

    # Expand manifest
    file_specs = expand_manifest(COMPARISON_MANIFEST)

    results: list[ComparisonResult] = []
    for rel_path, spec in file_specs:
        old_path = old_dir / rel_path
        new_path = new_dir / rel_path
        result = ComparisonResult(rel_path)

        check_type = spec.get("check", "row_count_and_schema")
        check_fn = CHECK_DISPATCH.get(check_type, compare_row_count_and_schema)

        try:
            check_fn(old_path, new_path, spec, result)
        except Exception as e:
            result.status = "FAIL"
            result.notes.append(f"Exception: {e}")

        results.append(result)

    # Also check for files in old that aren't in manifest
    extra_check_dirs = ["processed", "historical"]
    for d in extra_check_dirs:
        old_d = old_dir / d
        new_d = new_dir / d
        if not old_d.exists():
            continue
        manifest_files = {r for r, _ in file_specs}
        for f in old_d.glob("*.parquet"):
            rel = f"{d}/{f.name}"
            if rel not in manifest_files:
                # Check if it exists in new
                r = ComparisonResult(rel)
                new_f = new_d / f.name
                if new_f.exists():
                    r.status = "PASS"
                    r.rows_old = len(pd.read_parquet(f))
                    r.rows_new = len(pd.read_parquet(new_f))
                    r.notes.append("Not in manifest but exists in both")
                else:
                    r.status = "SKIP"
                    r.rows_old = len(pd.read_parquet(f))
                    r.notes.append("Not in manifest, missing from new (may be expected)")
                results.append(r)

    # Print console table
    table = format_console_table(results)
    print(table)

    # Summary
    counts = defaultdict(int)
    for r in results:
        counts[r.status] += 1
    print(f"\n--- SUMMARY ---")
    print(f"  ✅ PASS: {counts['PASS']}   ⚠️  WARN: {counts['WARN']}   ❌ FAIL: {counts['FAIL']}   ⏭️  SKIP: {counts['SKIP']}")

    total_checked = counts["PASS"] + counts["WARN"] + counts["FAIL"]
    if total_checked > 0:
        pass_rate = counts["PASS"] / total_checked * 100
        print(f"  Pass rate: {pass_rate:.0f}% ({counts['PASS']}/{total_checked})")

    # Save JSON report
    log_dir = new_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    report_data = {
        "date": datetime.datetime.now().isoformat(),
        "old_dir": str(old_dir),
        "new_dir": str(new_dir),
        "summary": dict(counts),
        "results": [r.to_dict() for r in results],
    }
    json_path = log_dir / "comparison_report.json"
    with open(json_path, "w") as f:
        json.dump(report_data, f, indent=2, default=str)
    print(f"\n  JSON report: {json_path}")

    # Save TXT report
    txt_path = log_dir / "comparison_report.txt"
    with open(txt_path, "w") as f:
        f.write(f"=== DATA COMPARISON REPORT ===\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Old: {old_dir}/    New: {new_dir}/\n\n")
        f.write(table + "\n\n")

        # Detail section — per-file drift info
        for r in results:
            if r.details:
                f.write(f"\n{'='*80}\n")
                f.write(f"FILE: {r.rel_path}\n")
                f.write(f"Status: {r.status}\n")
                if "drift" in r.details:
                    f.write(f"  Numeric Drift:\n")
                    for col, d in r.details["drift"].items():
                        f.write(f"    {col}: MAD={d['MAD']:.6f}, max={d['max_diff']:.6f}, "
                                f"corr={d['correlation']:.6f}, Δ>{100*0.01:.0f}%={d['pct_changed_above_tol']:.1f}%, "
                                f"KS={d['KS_stat']:.4f} (p={d['KS_p']:.4f})\n")
                if "labels" in r.details:
                    f.write(f"  Label Changes:\n")
                    for col, info in r.details["labels"].items():
                        f.write(f"    {col}: {info['n_changed']}/{info['n_total']} changed ({info['pct_changed']:.1f}%)\n")
                        for change in info.get("sample_changes", []):
                            f.write(f"      - {change}\n")
    print(f"  TXT report: {txt_path}")

    # Exit code
    if counts["FAIL"] > 0:
        print(f"\n⛔ {counts['FAIL']} files FAILED — investigation needed")
        sys.exit(1)
    elif counts["WARN"] > 0:
        print(f"\n⚠️  {counts['WARN']} files with warnings — review recommended")
        sys.exit(0)
    else:
        print(f"\n✅ All files passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
