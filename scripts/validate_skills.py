#!/usr/bin/env python3
"""Simple validator for .github/skills/*.skill.md frontmatter presence.

Checks each skill file for required frontmatter keys and returns exit code 1
if any are missing. This avoids yaml dependencies and focuses on presence checks.
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

def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    skills_dir = os.path.join(repo_root, '.github', 'skills')
    pattern = os.path.join(skills_dir, '*.skill.md')
    files = sorted(glob.glob(pattern))
    if not files:
        print('No skill files found in', skills_dir)
        return 1

    failed = False
    for path in files:
        rel = os.path.relpath(path, repo_root)
        txt = open(path, 'r', encoding='utf-8').read()
        start = txt.find('---')
        if start == -1:
            print(f'{rel}: missing frontmatter start (---)')
            failed = True
            continue
        end = txt.find('---', start+3)
        if end == -1:
            print(f'{rel}: missing frontmatter end (second ---)')
            failed = True
            continue
        fm = txt[start+3:end]
        missing = []
        for key in REQUIRED:
            if re.search(r'^\s*' + re.escape(key), fm, flags=re.M) is None:
                missing.append(key.rstrip(':'))
        if missing:
            print(f'{rel}: MISSING {", ".join(missing)}')
            failed = True
        else:
            print(f'{rel}: OK')

    if failed:
        print('\nOne or more skill files are missing required frontmatter keys.')
        return 1
    print('\nAll skill files have the required frontmatter keys.')
    return 0

if __name__ == '__main__':
    sys.exit(main())
