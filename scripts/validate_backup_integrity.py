#!/usr/bin/env python3
"""
scripts/validate_backup_integrity.py
=============================================================================
Validate backup integrity and post-rewrite schema correctness.

Default (pre-flight / integrity): for every parquet in data/, verify:
  - Corresponding file exists in data_backup_20260601/
  - SHA-256 checksum match
  - Row count match

--post-rewrite: for every parquet in data/, verify:
  - Row count matches backup (no rows dropped)
  - All column names are lowercase
  - No null values introduced in key join columns (player_id, game_id, season)

Exits 1 on any FAIL so CI/shell scripts can gate on it.
=============================================================================
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys

import pandas as pd

DATA_DIR = "data"
BACKUP_DIR = "data_backup_20260601"
KEY_COLS = {"player_id", "game_id", "season"}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def find_parquets(root: str, scope_dirs: list[str] | None = None) -> list[str]:
    """Return all .parquet file paths relative to root.

    scope_dirs: if provided, only include files whose path starts with one of
    these directory prefixes (relative to root). Useful for per-phase checks.
    """
    result = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".parquet"):
                rel = os.path.relpath(os.path.join(dirpath, fn), root)
                if scope_dirs:
                    if not any(rel.startswith(d) for d in scope_dirs):
                        continue
                result.append(rel)
    return sorted(result)


def preflight(verbose: bool = True, scope_dirs: list[str] | None = None) -> bool:
    files = find_parquets(DATA_DIR, scope_dirs)
    if not files:
        print(f"FAIL: no parquet files found in {DATA_DIR}/")
        return False

    if not os.path.isdir(BACKUP_DIR):
        print(f"FAIL: backup directory {BACKUP_DIR}/ does not exist")
        return False

    fails: list[str] = []
    for rel in files:
        src = os.path.join(DATA_DIR, rel)
        bak = os.path.join(BACKUP_DIR, rel)

        if not os.path.exists(bak):
            fails.append(f"  MISSING backup: {rel}")
            continue

        try:
            src_rows = len(pd.read_parquet(src))
            bak_rows = len(pd.read_parquet(bak))
        except Exception as e:
            fails.append(f"  UNREADABLE: {rel}: {e}")
            continue

        if src_rows != bak_rows:
            fails.append(f"  ROW COUNT MISMATCH: {rel}  data={src_rows}  backup={bak_rows}")
            continue

        src_sum = sha256(src)
        bak_sum = sha256(bak)
        if src_sum != bak_sum:
            fails.append(f"  CHECKSUM MISMATCH: {rel}")
            continue

        if verbose:
            print(f"  OK  {rel}  rows={src_rows}")

    if fails:
        print(f"\nFAIL — {len(fails)} issue(s):")
        for f in fails:
            print(f)
        return False

    print(f"\nPASS — {len(files)} parquet(s) verified (checksums + row counts match)")
    return True


def post_rewrite(verbose: bool = True, scope_dirs: list[str] | None = None) -> bool:
    if not os.path.isdir(BACKUP_DIR):
        print(f"FAIL: backup directory {BACKUP_DIR}/ does not exist")
        return False

    files = find_parquets(DATA_DIR, scope_dirs)
    if not files:
        print(f"FAIL: no parquet files found in {DATA_DIR}/")
        return False

    fails: list[str] = []
    for rel in files:
        src = os.path.join(DATA_DIR, rel)
        bak = os.path.join(BACKUP_DIR, rel)

        # Readability
        try:
            df = pd.read_parquet(src)
        except Exception as e:
            fails.append(f"  UNREADABLE: {rel}: {e}")
            continue

        # Row count vs backup
        if os.path.exists(bak):
            try:
                bak_rows = len(pd.read_parquet(bak))
                if len(df) != bak_rows:
                    fails.append(
                        f"  ROW COUNT CHANGED: {rel}  now={len(df)}  backup={bak_rows}"
                    )
            except Exception as e:
                fails.append(f"  BACKUP UNREADABLE: {rel}: {e}")
        else:
            fails.append(f"  MISSING backup: {rel}")

        # Lowercase columns
        bad_cols = [c for c in df.columns if c != c.lower()]
        if bad_cols:
            fails.append(f"  UPPERCASE COLS: {rel}  {bad_cols[:5]}")

        # No new nulls in key join columns
        for kc in KEY_COLS:
            if kc in df.columns:
                null_count = df[kc].isna().sum()
                if null_count > 0:
                    fails.append(f"  NULLS in {kc}: {rel}  count={null_count}")

        if verbose and rel not in [e.split()[2] for e in fails if len(e.split()) >= 3]:
            print(f"  OK  {rel}  rows={len(df)}")

    if fails:
        print(f"\nFAIL — {len(fails)} issue(s):")
        for f in fails:
            print(f)
        return False

    print(f"\nPASS — {len(files)} parquet(s) verified (rows, lowercase schema, no new nulls)")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="BKE backup integrity validator")
    parser.add_argument(
        "--post-rewrite",
        action="store_true",
        help="Post-rewrite check: row counts + schema validity (not checksums)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress per-file OK lines")
    parser.add_argument(
        "--dirs",
        nargs="*",
        metavar="DIR",
        help="Limit check to these subdirectory prefixes (relative to data/). "
             "E.g. --dirs historical matchup official_stats tracking features",
    )
    args = parser.parse_args()

    scope_dirs = args.dirs if args.dirs else None
    verbose = not args.quiet
    if args.post_rewrite:
        print("=== POST-REWRITE VALIDATION ===")
        if scope_dirs:
            print(f"    Scope: {scope_dirs}")
        ok = post_rewrite(verbose=verbose, scope_dirs=scope_dirs)
    else:
        print("=== PRE-FLIGHT INTEGRITY CHECK ===")
        if scope_dirs:
            print(f"    Scope: {scope_dirs}")
        ok = preflight(verbose=verbose, scope_dirs=scope_dirs)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
