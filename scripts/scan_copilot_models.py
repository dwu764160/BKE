#!/usr/bin/env python3
"""
Scan VS Code user/workspace/globalStorage for model settings and print findings.

Usage:
  python3 scripts/scan_copilot_models.py
"""
import json
import re
from pathlib import Path


def load_jsonc(p: Path):
    try:
        s = p.read_text(encoding='utf-8')
    except Exception:
        return None
    s = re.sub(r'(?m)^\s*//.*\n?', '', s)
    s = re.sub(r'/\*.*?\*/', '', s, flags=re.S)
    s = re.sub(r',\s*([\]}])', r'\1', s)
    try:
        return json.loads(s)
    except Exception:
        return None


def get_nested(d, key):
    parts = key.split('.')
    v = d
    for p in parts:
        if isinstance(v, dict) and p in v:
            v = v[p]
        else:
            return None
    return v


keys_to_check = [
    'inlineChat.defaultModel',
    'github.copilot.chat.selectedModel',
    'github.copilot.chat.defaultModel',
    'github.copilot.defaultModel',
    'claudeCode.selectedModel',
]


def report(src, key, val):
    print(f"{src} :: {key} = {val!r}")


def scan_user_settings(path: Path):
    d = load_jsonc(path)
    if not d:
        return
    for k in keys_to_check:
        v = get_nested(d, k)
        if v is not None:
            report(str(path), k, v)
    # common direct keys
    for k in ('github.copilot.chat.responsesApiReasoningEffort', 'inlineChat.defaultModel'):
        if k in d:
            report(str(path), k, d[k])


def scan_workspace(root: Path):
    for p in root.rglob('.vscode/settings.json'):
        d = load_jsonc(p)
        if not d:
            continue
        for k in keys_to_check:
            v = get_nested(d, k)
            if v is not None:
                report(str(p), k, v)
        for k in ('github.copilot.chat.responsesApiReasoningEffort', 'inlineChat.defaultModel'):
            if k in d:
                report(str(p), k, d[k])


def scan_global_storage(gs_root: Path):
    if not gs_root.exists():
        return
    for f in gs_root.rglob('*.json'):
        d = load_jsonc(f)
        if not d:
            continue
        for k in ('model', 'selectedModel', 'defaultModel', 'preferredModel'):
            if k in d:
                report(str(f), k, d[k])
        for k in keys_to_check:
            v = get_nested(d, k)
            if v is not None:
                report(str(f), k, v)
        # search any string containing 'gpt' recursively
        def walk(o, path=''):
            if isinstance(o, dict):
                for kk, vv in o.items():
                    walk(vv, path + ('.' + kk if path else kk))
            elif isinstance(o, list):
                for i, vv in enumerate(o):
                    walk(vv, f"{path}[{i}]")
            elif isinstance(o, str) and 'gpt' in o.lower():
                report(str(f), path, o)
        walk(d)


def main():
    user_settings = Path('/mnt/c/Users/danie/AppData/Roaming/Code/User/settings.json')
    print('Scanning user settings...')
    if user_settings.exists():
        scan_user_settings(user_settings)
    else:
        print(f'user settings not found: {user_settings}')

    print('\nScanning workspace .vscode settings...')
    scan_workspace(Path('.').resolve())

    print('\nScanning VS Code globalStorage...')
    gs = Path('/mnt/c/Users/danie/AppData/Roaming/Code/User/globalStorage')
    scan_global_storage(gs)

    print('\nScan complete')


if __name__ == '__main__':
    main()
