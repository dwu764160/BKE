#!/usr/bin/env python3
"""Build a skill index from SKILL.md frontmatter.

Writes a YAML index to `.github/skills/skill_index.yaml` by default.

Usage:
  python .github/skills/build_skill_index.py --out .github/skills/skill_index.yaml
"""
import argparse
import glob
import os
import re
import yaml


FRONT_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def parse_frontmatter(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    m = FRONT_RE.match(txt)
    if not m:
        return {}
    fm = m.group(1)
    try:
        data = yaml.safe_load(fm) or {}
    except Exception:
        data = {}
    return data


def build_index(skills_dir: str) -> dict:
    entries = []
    # Match both direct children (skill-name/SKILL.md) and one level of subdirectory
    # (context-engineering/skill-name/SKILL.md) so nested skill groups are indexed.
    pattern_direct = os.path.join(skills_dir, "*/SKILL.md")
    pattern_nested = os.path.join(skills_dir, "*/*/SKILL.md")
    files = sorted(glob.glob(pattern_direct) + glob.glob(pattern_nested))
    for path in files:
        rel = os.path.relpath(path)
        fm = parse_frontmatter(path)
        name = fm.get("name") or os.path.basename(os.path.dirname(path))
        desc = fm.get("description", "")
        tags = fm.get("tags", []) or []
        triggers = fm.get("triggers") or tags or []
        preferred_tools = fm.get("preferred_tools", []) or []
        avoid_tools = fm.get("avoid_tools", []) or []
        job_scope = fm.get("job_scope", []) or []
        when_to_use = fm.get("when_to_use", []) or []
        log_usage = fm.get("log_usage", False)
        usage_log = fm.get("usage_log", "loop/skill_usage.log")

        entries.append({
            "name": name,
            "path": rel.replace('\\\\', '/'),
            "description": desc if isinstance(desc, str) else "",
            "tags": tags,
            "triggers": triggers,
            "preferred_tools": preferred_tools,
            "avoid_tools": avoid_tools,
            "job_scope": job_scope,
            "when_to_use": when_to_use,
            "log_usage": log_usage,
            "usage_log": usage_log,
        })

    return {"skills": entries}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skills-dir", default=".github/skills", help="Skills folder")
    p.add_argument("--out", default=".github/skills/skill_index.yaml", help="Output index path")
    args = p.parse_args()

    idx = build_index(args.skills_dir)

    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as f:
        yaml.safe_dump(idx, f, sort_keys=False)

    print(f"Wrote {len(idx.get('skills', []))} skills to {args.out}")


if __name__ == "__main__":
    main()
