#!/usr/bin/env python3
"""Log and print workspace files (lightweight read_file wrapper).

Usage:
    python3 scripts/log_read_file.py path/to/file --agent AGENT_ID

This script emulates a simple `read_file` tool: it prints the file
contents to stdout and appends a JSON-line entry to `loop/skill_usage.log`
with a timestamp, agent id, relative path, size and a SHA256 of the
content. It intentionally does NOT log file contents to the log file.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys


def write_log(log_path: str, entry: dict) -> None:
    try:
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        sys.stderr.write(f"WARNING: failed to write log {log_path}: {e}\n")


def read_and_log(repo_root: str, target: str, agent: str | None, log_path: str, no_print: bool) -> int:
    if not os.path.isabs(target):
        path = os.path.normpath(os.path.join(repo_root, target))
    else:
        path = target

    rel = os.path.relpath(path, repo_root)
    if not os.path.exists(path):
        sys.stderr.write(f"ERROR: file not found: {rel}\n")
        return 2

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception as e:
        sys.stderr.write(f"ERROR reading {rel}: {e}\n")
        return 3

    size = len(content)
    sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()

    entry = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "agent": agent,
        "path": rel,
        "size": size,
        "sha256": sha256,
    }

    write_log(log_path, entry)

    if not no_print:
        sys.stdout.write(content)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Log and print a workspace file (read_file wrapper)")
    parser.add_argument("paths", nargs="+", help="Workspace-relative or absolute path(s) to read")
    parser.add_argument("--agent", help="Agent id or name invoking the read", default=None)
    parser.add_argument("--log", help="Custom log path (workspace-relative)", default=None)
    parser.add_argument("--no-print", help="Do not print file contents to stdout", action="store_true")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    loop_dir = os.path.join(repo_root, "loop")
    os.makedirs(loop_dir, exist_ok=True)

    log_path = os.path.join(repo_root, args.log) if args.log else os.path.join(loop_dir, "skill_usage.log")

    exit_code = 0
    for p in args.paths:
        rc = read_and_log(repo_root, p, args.agent, log_path, args.no_print)
        if rc != 0:
            exit_code = rc

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
