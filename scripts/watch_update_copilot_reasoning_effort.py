#!/usr/bin/env python3
"""
Watch VS Code user/workspace `settings.json` and set
`github.copilot.chat.responsesApiReasoningEffort` to:
 - "high" when `inlineChat.defaultModel` contains "gpt-5 mini"
 - "xhigh" for any other model

Run as a background process (WSL/Linux) or a scheduled task on Windows.

Usage:
  python3 scripts/watch_update_copilot_reasoning_effort.py [--settings PATH] [--interval N] [--once]

Notes:
 - The script is safe to run continuously; it only writes when a change is required.
 - Default user settings path is `/mnt/c/Users/danie/AppData/Roaming/Code/User/settings.json`.
 - If a workspace `.vscode/settings.json` exists, it is updated too to avoid override conflicts.
"""
import argparse
import json
import os
import re
import sqlite3
import shutil
import sys
import time
from datetime import datetime


def load_jsonc(path):
    text = open(path, 'r', encoding='utf-8').read()
    # strip // line comments
    text = re.sub(r'(?m)^\s*//.*\n?', '', text)
    # strip /* ... */ block comments
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.S)
    # remove trailing commas before } or ]
    text = re.sub(r',\s*([\]}])', r'\1', text)
    return json.loads(text)


def write_json_atomic(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)
    bak = None
    if os.path.exists(path):
        bak = path + '.bak.' + datetime.utcnow().strftime('%Y%m%d%H%M%S')
        shutil.copy2(path, bak)
    os.replace(tmp, path)
    return bak


def normalize_model_id(model_val: str) -> str:
    if not model_val:
        return ''
    return model_val.strip().lower().replace(' ', '').replace('(copilot)', '')


def desired_value_for_model(model_val: str) -> str:
    if not model_val:
        return 'xhigh'
    m = normalize_model_id(model_val)
    # Treat only GPT-5 mini as high. Everything else uses xhigh.
    if 'gpt-5-mini' in m or m == 'gpt5mini':
        return 'high'
    return 'xhigh'


def detect_panel_chat_model(state_db_path: str) -> str:
    """Read active chat model from VS Code global state DB.

    Key discovered by live scan:
      ItemTable.key = 'chat.currentLanguageModel.panel'
    """
    if not state_db_path or not os.path.exists(state_db_path):
        return ''
    try:
        con = sqlite3.connect(state_db_path)
        cur = con.cursor()
        cur.execute(
            "SELECT value FROM ItemTable WHERE key = ? LIMIT 1",
            ('chat.currentLanguageModel.panel',),
        )
        row = cur.fetchone()
        con.close()
        if row and row[0]:
            return str(row[0]).strip()
    except Exception as e:
        print(f'[{datetime.utcnow().isoformat()}] state db read error: {e}')
    return ''


def detect_effective_model(settings_path: str, state_db_path: str):
    # 1) Prefer chat panel model selection from VS Code state DB (real chat dropdown)
    panel_model = detect_panel_chat_model(state_db_path)
    if panel_model:
        return panel_model, f'state.vscdb:chat.currentLanguageModel.panel ({state_db_path})'

    # 2) Fallback to settings inline chat model
    try:
        data = load_jsonc(settings_path)
    except Exception:
        return '', 'none'
    model_val = data.get('inlineChat.defaultModel', '')
    if model_val:
        return model_val, f'settings:inlineChat.defaultModel ({settings_path})'
    return '', 'none'


def update_reasoning_key_if_needed(target_path: str, desired: str) -> bool:
    key = 'github.copilot.chat.responsesApiReasoningEffort'
    try:
        data = load_jsonc(target_path) if os.path.exists(target_path) else {}
    except Exception as e:
        print(f'[{datetime.utcnow().isoformat()}] parse error in {target_path}: {e}')
        return False

    prev = data.get(key)
    if prev == desired:
        print(f'[{datetime.utcnow().isoformat()}] no change in {target_path}: {key} already {prev}')
        return False

    data[key] = desired
    try:
        bak = write_json_atomic(target_path, data)
    except Exception as e:
        print(f'[{datetime.utcnow().isoformat()}] write error in {target_path}: {e}')
        return False

    print(f'[{datetime.utcnow().isoformat()}] updated {target_path}: {key} {prev} -> {desired}' + (f' (bak: {bak})' if bak else ''))
    return True


def update_settings_if_needed(settings_path: str, state_db_path: str, workspace_settings_path: str = '') -> bool:
    try:
        data = load_jsonc(settings_path)
    except Exception as e:
        print(f'[{datetime.utcnow().isoformat()}] parse error: {e}')
        return False

    model_val, model_source = detect_effective_model(settings_path, state_db_path)
    print(f'[{datetime.utcnow().isoformat()}] Detected model: {model_val!r} from {model_source}')
    desired = desired_value_for_model(model_val)
    changed = False
    changed = update_reasoning_key_if_needed(settings_path, desired) or changed
    if workspace_settings_path:
        changed = update_reasoning_key_if_needed(workspace_settings_path, desired) or changed
    return changed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings', default='/mnt/c/Users/danie/AppData/Roaming/Code/User/settings.json')
    parser.add_argument('--state-db', default='/mnt/c/Users/danie/AppData/Roaming/Code/User/globalStorage/state.vscdb')
    parser.add_argument('--workspace-settings', default='.vscode/settings.json')
    parser.add_argument('--interval', type=float, default=2.0, help='poll interval in seconds')
    parser.add_argument('--once', action='store_true', help='run once and exit')
    args = parser.parse_args()

    settings_path = args.settings
    state_db_path = args.state_db
    workspace_settings_path = os.path.abspath(args.workspace_settings) if args.workspace_settings else ''
    if workspace_settings_path and not os.path.exists(workspace_settings_path):
        workspace_settings_path = ''
    if not os.path.exists(settings_path):
        print(f'settings not found: {settings_path}')
        sys.exit(2)

    # initial attempt
    update_settings_if_needed(settings_path, state_db_path, workspace_settings_path)
    if args.once:
        return

    try:
        last_settings_mtime = os.path.getmtime(settings_path)
    except Exception:
        last_settings_mtime = None

    try:
        last_db_mtime = os.path.getmtime(state_db_path) if os.path.exists(state_db_path) else None
    except Exception:
        last_db_mtime = None

    try:
        while True:
            time.sleep(args.interval)
            settings_changed = False
            db_changed = False

            try:
                settings_mtime = os.path.getmtime(settings_path)
            except Exception:
                settings_mtime = None

            if last_settings_mtime is None or settings_mtime != last_settings_mtime:
                settings_changed = True
                last_settings_mtime = settings_mtime

            try:
                db_mtime = os.path.getmtime(state_db_path) if os.path.exists(state_db_path) else None
            except Exception:
                db_mtime = None

            if last_db_mtime is None or db_mtime != last_db_mtime:
                db_changed = True
                last_db_mtime = db_mtime

            if settings_changed or db_changed:
                update_settings_if_needed(settings_path, state_db_path, workspace_settings_path)
    except KeyboardInterrupt:
        print('\nexiting')


if __name__ == '__main__':
    main()
