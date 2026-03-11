#!/usr/bin/env python3
"""Append a JSON-line entry to loop/skill_usage.log for a SKILL.md file.

Example:
  python scripts/log_skill_usage.py .github/skills/audit-compute/SKILL.md --agent copilot-agent-1

Fields written: timestamp, agent, path, size, sha256, trigger, context
"""
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path", help="Path to SKILL.md to log")
    p.add_argument("--agent", required=True, help="Agent identifier")
    p.add_argument("--trigger", default="", help="Trigger text or matched query")
    p.add_argument("--context", default="", help="Short context summary")
    p.add_argument("--out", default="loop/skill_usage.log", help="Log file path")
    p.add_argument("--no-print", action="store_true", help="Don't print the appended line")
    args = p.parse_args()

    path = args.path
    if not os.path.exists(path):
        raise SystemExit(f"Path not found: {path}")

    st = os.stat(path)
    size = st.st_size
    sha = sha256_file(path)
    ts = datetime.now(timezone.utc).isoformat()

    entry = {
        "timestamp": ts,
        "agent": args.agent,
        "path": path.replace('\\\\', '/'),
        "size": size,
        "sha256": sha,
        "trigger": args.trigger,
        "context": args.context,
    }

    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    if not args.no_print:
        print(json.dumps(entry, ensure_ascii=False))


if __name__ == "__main__":
    main()
