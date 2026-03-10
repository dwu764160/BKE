#!/usr/bin/env python3
"""Validator for agent skills.

Supports both legacy layout (`.github/skills/*.skill.md`) and the
AgentSkills layout (`.github/skills/<skill-name>/SKILL.md`).

Checks that required YAML frontmatter keys exist and, for the new layout,
that the `name` frontmatter matches the directory name (lowercase-hyphen form).
"""
import os
import sys
import glob
import re

REQUIRED = [
    'name:',
    'description:',
    'persona:',
    'preferred_tools:',
    'avoid_tools:',
    'job_scope:',
    'when_to_use:',
    'example_prompts:'
]


def find_skill_files(skills_dir: str):
    legacy = sorted(glob.glob(os.path.join(skills_dir, '*.skill.md')))
    new = sorted(glob.glob(os.path.join(skills_dir, '*', 'SKILL.md')))
    return legacy + new


def extract_frontmatter(text: str):
    start = text.find('---')
    if start == -1:
        return None
    end = text.find('---', start + 3)
    if end == -1:
        return None
    return text[start + 3:end]


def frontmatter_name_value(fm: str):
    m = re.search(r'^\s*name:\s*(?:["\']?)([^"\']+)(?:["\']?)\s*$', fm, flags=re.M)
    if m:
        return m.group(1).strip()
    return None


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    skills_dir = os.path.join(repo_root, '.github', 'skills')
    files = find_skill_files(skills_dir)
    if not files:
        print('No skill files found in', skills_dir)
        return 1

    failed = False
    for path in files:
        rel = os.path.relpath(path, repo_root)
        txt = open(path, 'r', encoding='utf-8').read()
        fm = extract_frontmatter(txt)
        if fm is None:
            print(f'{rel}: missing frontmatter start/end (---)')
            failed = True
            continue

        missing = []
        for key in REQUIRED:
            if re.search(r'^\s*' + re.escape(key), fm, flags=re.M) is None:
                missing.append(key.rstrip(':'))

        nameval = frontmatter_name_value(fm)
        if os.path.basename(path) == 'SKILL.md':
            expected = os.path.basename(os.path.dirname(path))
            if nameval is None:
                if 'name' not in missing:
                    missing.append('name')
            else:
                if nameval != expected:
                    print(f'{rel}: NAME MISMATCH expected {expected!r} but frontmatter has {nameval!r}')
                    failed = True
                    continue

        if missing:
            print(f'{rel}: MISSING {", ".join(missing)}')
            failed = True
        else:
            print(f'{rel}: OK')

    if failed:
        print('\nOne or more skill files are missing required frontmatter keys or have issues.')
        return 1
    print('\nAll skill files look valid.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
